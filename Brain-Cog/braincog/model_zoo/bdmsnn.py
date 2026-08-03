
import os

import torch
from torch import nn
import torch.nn.functional as F

from braincog.base.node.node import IFNode, SimHHNode
from braincog.base.learningrule.STDP import STDP, MutliInputSTDP
from braincog.base.connection.CustomLinear import CustomLinear
from braincog.base.brainarea.basalganglia import basalganglia
from braincog.model_zoo.communication_subspace import (
    CrossCoreTrafficMonitor,
    CommonModeCountCommunication,
    ReducedRankCommunication,
    cross_core_link_dimensions,
)

import pygame
from pygame.locals import *
from collections import deque
from random import randint
#os.environ["SDL_VIDEODRIVER"] = "dummy"


class BDMSNN(nn.Module):
    def __init__(self, num_state, num_action, weight_exc, weight_inh, node_type,
                 communication_rank=0, communication_window=128,
                 communication_warmup=0, communication_refit_interval=1,
                 communication_mode="striatum_to_output", communication_links=None,
                 communication_lossless_only=False, pm_threshold=None,
                 pm_lateral_gain=None, compact_striatum=False,
                 communication_link_ranks=None, common_mode_count_links=None,
                 striatum_latent_rank=0, striatum_latent_delta_transport=False,
                 striatum_latent_quant_bits=0, striatum_latent_quant_scale=5.0,
                 striatum_latent_spike_transport=False,
                 striatum_latent_spike_threshold=0.5,
                 striatum_latent_spike_gain=1.0,
                 striatum_latent_spike_decoder_gain=1.0,
                 striatum_latent_spike_soft_reset=False,
                 striatum_latent_spike_normalize_factors=False,
                 striatum_latent_spike_contrast_encode=False):
        """
        定义BDM-SNN网络
        :param num_state: 状态个数
        :param num_action: 动作个数
        :param weight_exc: 兴奋性连接权重
        :param weight_inh: 抑制性连接权重
        """
        super().__init__()
        # parameters
        self.num_state = num_state
        self.num_action = num_action
        BG = basalganglia(num_state, num_action, weight_exc, weight_inh, node_type,
                          compact_striatum=compact_striatum)
        dm_connection = BG.getweight()
        dm_mask = BG.getmask()
        # input-dlpfc
        con_matrix9 = torch.eye((num_state), dtype=torch.float)
        dm_connection.append(CustomLinear(weight_exc * con_matrix9, con_matrix9))
        dm_mask.append(con_matrix9)
        # gpi-th
        con_matrix10 = torch.eye((num_action), dtype=torch.float)
        dm_mask.append(con_matrix10)
        dm_connection.append(CustomLinear(weight_inh * con_matrix10, con_matrix10))
        # th-pm
        dm_mask.append(con_matrix10)
        dm_connection.append(CustomLinear(weight_exc * con_matrix10, con_matrix10))
        # dlpfc-th
        con_matrix11 = torch.ones((num_state, num_action), dtype=torch.float)
        dm_mask.append(con_matrix11)
        dm_connection.append(CustomLinear(0.2 * weight_exc * con_matrix11, con_matrix11))
        # pm-pm
        con_matrix3 = torch.ones((num_action, num_action), dtype=torch.float)
        con_matrix4 = torch.eye((num_action), dtype=torch.float)
        con_matrix5 = con_matrix3 - con_matrix4
        con_matrix5 = con_matrix5
        dm_mask.append(con_matrix5)
        dm_connection.append(CustomLinear(5 * weight_inh * con_matrix5, con_matrix5))
        # dlpfc thalamus pm +bg
        self.weight_exc = weight_exc
        self.compact_striatum = compact_striatum
        self.num_subDM = 8
        # The original implementation used plain lists, which leaves these
        # modules unregistered and prevents ``model.to(device)`` from moving
        # their parameters.  ModuleList preserves the network topology while
        # enabling the standard PyTorch device transfer path.
        self.connection = nn.ModuleList(dm_connection)
        self.mask = dm_mask
        self.node = nn.ModuleList(BG.node)
        self.node_type = node_type
        self.communication_rank = communication_rank
        self.communication_warmup = communication_warmup
        self.communication_refit_interval = communication_refit_interval
        self.communication_mode = communication_mode
        self.communication_link_names = (set(communication_links)
                                         if communication_links is not None else None)
        self.communication_link_ranks = (dict(communication_link_ranks)
                                         if communication_link_ranks is not None else None)
        self.common_mode_count_link_names = (set(common_mode_count_links)
                                              if common_mode_count_links is not None else set())
        self.striatum_latent_rank = int(striatum_latent_rank)
        if self.striatum_latent_rank < 0 or self.striatum_latent_rank > num_action:
            raise ValueError("striatum latent rank must be in [0, num_action]")
        self.striatum_latent_delta_transport = bool(striatum_latent_delta_transport)
        self.striatum_latent_quant_bits = int(striatum_latent_quant_bits)
        self.striatum_latent_quant_scale = float(striatum_latent_quant_scale)
        self.striatum_latent_spike_transport = bool(striatum_latent_spike_transport)
        self.striatum_latent_spike_threshold = float(striatum_latent_spike_threshold)
        self.striatum_latent_spike_gain = float(striatum_latent_spike_gain)
        self.striatum_latent_spike_decoder_gain = float(striatum_latent_spike_decoder_gain)
        self.striatum_latent_spike_soft_reset = bool(striatum_latent_spike_soft_reset)
        self.striatum_latent_spike_normalize_factors = bool(
            striatum_latent_spike_normalize_factors)
        self.striatum_latent_spike_contrast_encode = bool(
            striatum_latent_spike_contrast_encode)
        if self.striatum_latent_quant_bits < 0 or self.striatum_latent_quant_bits > 16:
            raise ValueError("striatum latent quantization must use 0..16 bits")
        if self.striatum_latent_quant_bits and self.striatum_latent_quant_scale <= 0:
            raise ValueError("striatum latent quantization scale must be positive")
        if self.striatum_latent_spike_transport and self.striatum_latent_rank == 0:
            raise ValueError("latent spike transport requires a nonzero latent rank")
        if (self.striatum_latent_spike_contrast_encode and
                not self.striatum_latent_spike_normalize_factors):
            raise ValueError("contrast latent encoding requires normalized factors")
        if (self.striatum_latent_spike_threshold <= 0 or self.striatum_latent_spike_gain <= 0 or
                self.striatum_latent_spike_decoder_gain <= 0):
            raise ValueError("latent spike threshold and gains must be positive")
        self.communication_lossless_only = communication_lossless_only
        self.communication_updates = 0
        self.communication_refit_history = []
        self.direct_link = None
        self.indirect_link = None
        self.cross_core_links = nn.ModuleDict()
        self.common_mode_links = nn.ModuleDict()
        self.full_traffic_monitor = None
        # Warm-start factors for the optional online nonnegative low-rank
        # constraint. Empty buffers keep the default/full model unchanged.
        for pathway in ("d1", "d2"):
            self.register_buffer(f"_striatum_nmf_{pathway}_left", torch.empty(0))
            self.register_buffer(f"_striatum_nmf_{pathway}_right", torch.empty(0))
        self.striatum_latent_stats = {
            pathway: {
                "samples": 0,
                "full_fallback_samples": 0,
                "source_logical_events": 0,
                "latent_scalar_values": 0,
                "latent_nonzero_values": 0,
                "latent_absolute_sum": 0.0,
                "latent_peak_absolute_value": 0.0,
                "source_projection_array_activations": 0,
                "target_decoder_array_activations": 0,
                "continuous_vector_records": 0,
                "suppressed_repeat_samples": 0,
                "squared_error": 0.0,
                "target_energy": 0.0,
                "peak_absolute_error": 0.0,
                "quantized_vector_records": 0,
                "quantization_clipped_values": 0,
                "quantization_total_values": 0,
                "quantization_squared_current_error": 0.0,
                "quantization_peak_current_error": 0.0,
                "latent_spike_events": 0,
                "latent_spike_active_samples": 0,
                "latent_spike_peak_events_per_sample": 0,
            }
            for pathway in ("d1", "d2")
        }
        self._striatum_latent_cache = {pathway: None for pathway in ("d1", "d2")}
        self._striatum_latent_decision_events = {
            pathway: None for pathway in ("d1", "d2")
        }
        self._striatum_latent_last_events = {
            pathway: None for pathway in ("d1", "d2")
        }
        self._striatum_latent_decision_pattern = {
            pathway: [] for pathway in ("d1", "d2")
        }
        self._striatum_latent_decision_target_current = {
            pathway: None for pathway in ("d1", "d2")
        }
        self._striatum_latent_decoder_teacher_updates = 0
        self._striatum_latent_decoder_teacher_event_mass = 0.0
        self._striatum_latent_decoder_teacher_ranking_updates = 0
        self._striatum_latent_shadow_calibration_updates = 0
        self._striatum_latent_shadow_calibration_squared_error = 0.0
        # These source-core neurons receive RRAM projection column currents.
        # Their binary outputs are the only values that cross cores in the
        # spiking transport branch; there is deliberately no latent ADC.
        self.striatum_latent_nodes = nn.ModuleDict({
            pathway: IFNode(threshold=self.striatum_latent_spike_threshold)
            for pathway in ("d1", "d2")
        })
        self._striatum_latent_residual_mem = {pathway: None for pathway in ("d1", "d2")}
        if communication_rank or self.communication_link_ranks:
            if communication_mode == "state_to_striatum":
                # In the compact-Str implementation each striatal population
                # has one output channel per action, not S * A output neurons.
                # The communication-link dimensions must describe the actual
                # target current, otherwise a DLPFC--Str RRR fit cannot be
                # evaluated after the population compression.
                striatum_dim = num_action if compact_striatum else num_state * num_action
                self.direct_link = ReducedRankCommunication(
                    num_state, striatum_dim, communication_rank,
                    communication_window,
                    fit_lossless_tolerance=(float("inf") if not communication_lossless_only else 1e-6))
                self.indirect_link = ReducedRankCommunication(
                    num_state, striatum_dim, communication_rank,
                    communication_window,
                    fit_lossless_tolerance=(float("inf") if not communication_lossless_only else 1e-6))
            elif communication_mode == "striatum_to_output":
                self.direct_link = ReducedRankCommunication(
                    num_state * num_action, num_action, communication_rank,
                    communication_window,
                    fit_lossless_tolerance=(float("inf") if not communication_lossless_only else 1e-6))
                self.indirect_link = ReducedRankCommunication(
                    num_state * num_action, num_action, communication_rank,
                    communication_window,
                    fit_lossless_tolerance=(float("inf") if not communication_lossless_only else 1e-6))
            elif communication_mode == "all_cross_core":
                # Logical four-core partition used by the extended prototype:
                # (DLPFC, StrD1, StrD2), STN, (GPe, GPi), and (thalamus, PM).
                # These are all eight connections crossing that partition;
                # local plastic DLPFC-to-striatum weights stay untouched.
                link_dimensions = cross_core_link_dimensions(
                    num_state, num_action, compact_striatum=compact_striatum)
                if self.communication_link_names is not None:
                    unknown = self.communication_link_names.difference(link_dimensions)
                    if unknown:
                        raise ValueError("unknown all_cross_core links: " + ", ".join(sorted(unknown)))
                    link_dimensions = {
                        name: dimensions for name, dimensions in link_dimensions.items()
                        if name in self.communication_link_names
                    }
                if self.communication_link_ranks is not None:
                    unknown = set(self.communication_link_ranks).difference(link_dimensions)
                    if unknown:
                        raise ValueError("unknown or unselected cross-core rank links: " +
                                         ", ".join(sorted(unknown)))
                    # An explicit rank map selects only its named links.
                    link_dimensions = {
                        name: dimensions for name, dimensions in link_dimensions.items()
                        if name in self.communication_link_ranks
                    }
                self.cross_core_links = nn.ModuleDict({
                    name: ReducedRankCommunication(
                        source_dim, target_dim,
                        (self.communication_link_ranks[name]
                         if self.communication_link_ranks is not None else communication_rank),
                        communication_window,
                        fit_lossless_tolerance=(float("inf") if not communication_lossless_only else 1e-6))
                    for name, (source_dim, target_dim) in link_dimensions.items()
                })
            else:
                raise ValueError(
                    "communication_mode must be state_to_striatum, "
                    "striatum_to_output, or all_cross_core")
        elif communication_mode == "all_cross_core":
            self.full_traffic_monitor = CrossCoreTrafficMonitor(
                cross_core_link_dimensions(num_state, num_action,
                                           compact_striatum=compact_striatum))
        if self.common_mode_count_link_names:
            if communication_mode != "all_cross_core":
                raise ValueError("common-mode count links require all_cross_core communication mode")
            fixed_connections = {
                "stn_gpe": self.connection[7],
                "stn_gpi": self.connection[6],
                "gpe_stn": self.connection[8],
            }
            unknown = self.common_mode_count_link_names.difference(fixed_connections)
            if unknown:
                raise ValueError("unknown common-mode links: " + ", ".join(sorted(unknown)))
            self.common_mode_links = nn.ModuleDict({
                name: CommonModeCountCommunication(fixed_connections[name])
                for name in self.common_mode_count_link_names
            })
        if self.node_type == "hh":
            self.node.extend([SimHHNode() for i in range(self.num_subDM - BG.num_subBG)])
            self.node[6].g_Na = torch.tensor(12)
            self.node[6].g_K = torch.tensor(3.6)
            self.node[6].g_L = torch.tensor(0.03)
        if self.node_type == "lif":
            self.node.extend([IFNode() for i in range(self.num_subDM - BG.num_subBG)])
        # Keep the original PM dynamics unless an explicit ablation changes
        # them.  PM is node 6 and connection 13 is its off-diagonal lateral
        # inhibition; the thalamic feed-forward path remains untouched.
        self.pm_threshold = (float(self.node[6].threshold.detach().cpu())
                             if pm_threshold is None else float(pm_threshold))
        self.pm_lateral_gain = (5 * weight_inh if pm_lateral_gain is None
                                else float(pm_lateral_gain))
        with torch.no_grad():
            self.node[6].threshold.fill_(self.pm_threshold)
            self.connection[13].weight.copy_(self.pm_lateral_gain * self.connection[13].mask)
        self.learning_rule = BG.learning_rule
        self.learning_rule.append(MutliInputSTDP(self.node[5], [self.connection[10], self.connection[12]]))  # gpi-丘脑
        self.learning_rule.append(MutliInputSTDP(self.node[6], [self.connection[11], self.connection[13]]))  # pm
        self.learning_rule.append(STDP(self.node[7], self.connection[9]))

        out_shape=[self.connection[0].weight.shape[1],self.connection[1].weight.shape[1],self.connection[2].weight.shape[1],self.connection[4].weight.shape[1],self.connection[3].weight.shape[1],self.connection[10].weight.shape[1],self.connection[11].weight.shape[1],self.connection[9].weight.shape[1]]
        self.out = []
        self.dw = []
        for i in range(self.num_subDM):
            self.out.append(torch.zeros((out_shape[i]), dtype=torch.float))
            self.dw.append(torch.zeros((out_shape[i]), dtype=torch.float))

    def forward(self, input):
        """
        根据输入得到网络的输出
        :param input: 输入
        :return: 网络的输出
        """
        # Region activities are stateful tensors rather than buffers.  Move
        # their initial zero states on the first call after a device transfer.
        self.out = [out.to(input.device) if isinstance(out, torch.Tensor) else out
                    for out in self.out]
        self.dw = [dw.to(input.device) if isinstance(dw, torch.Tensor) else dw
                   for dw in self.dw]
        self.out[7] = self.node[7](self.connection[9](input))
        if self.communication_mode == "state_to_striatum":
            self.out[0], self.dw[0] = self._stdp_forward(
                self.learning_rule[0], self.out[7], self.direct_link)
            self.out[1], self.dw[1] = self._stdp_forward(
                self.learning_rule[1], self.out[7], self.indirect_link)
        else:
            self.out[0], self.dw[0] = self._stdp_forward(
                self.learning_rule[0], self.out[7], nmf_pathway="d1")
            self.out[1], self.dw[1] = self._stdp_forward(
                self.learning_rule[1], self.out[7], nmf_pathway="d2")
        if self.communication_mode == "all_cross_core" and (self.cross_core_links or self.common_mode_links):
            stn_links = [self._communication_link("dlpfc_stn"),
                         self._communication_link("gpe_stn")]
        elif self.full_traffic_monitor is not None:
            self.full_traffic_monitor.observe("dlpfc_stn", self.out[7])
            self.full_traffic_monitor.observe("gpe_stn", self.out[3])
            stn_links = [None, None]
        else:
            stn_links = [None, None]
        self.out[2], self.dw[2] = self._multi_input_stdp_forward(
            self.learning_rule[2], [self.out[7], self.out[3]], stn_links)
        if self.communication_mode == "all_cross_core" and (self.cross_core_links or self.common_mode_links):
            gpe_links = [self._communication_link("strd2_gpe"),
                         self._communication_link("stn_gpe")]
            gpi_links = [self._communication_link("strd1_gpi"), None,
                         self._communication_link("stn_gpi")]
            thalamus_links = [self._communication_link("gpi_thalamus"),
                              self._communication_link("dlpfc_thalamus")]
        elif self.full_traffic_monitor is not None:
            self.full_traffic_monitor.observe("strd2_gpe", self.out[1])
            self.full_traffic_monitor.observe("stn_gpe", self.out[2])
            self.full_traffic_monitor.observe("strd1_gpi", self.out[0])
            self.full_traffic_monitor.observe("stn_gpi", self.out[2])
            self.full_traffic_monitor.observe("gpi_thalamus", self.out[4])
            self.full_traffic_monitor.observe("dlpfc_thalamus", self.out[7])
            gpe_links = [None, None]
            gpi_links = [None, None, None]
            thalamus_links = [None, None]
        else:
            gpe_link = self.indirect_link if self.communication_mode == "striatum_to_output" else None
            gpi_link = self.direct_link if self.communication_mode == "striatum_to_output" else None
            gpe_links = [gpe_link, None]
            gpi_links = [gpi_link, None, None]
            thalamus_links = [None, None]
        self.out[3], self.dw[3] = self._multi_input_stdp_forward(
            self.learning_rule[3], [self.out[1], self.out[2]], gpe_links)
        self.out[4], self.dw[4] = self._multi_input_stdp_forward(
            self.learning_rule[4], [self.out[0], self.out[3], self.out[2]],
            gpi_links)
        self.out[5], self.dw[5] = self._multi_input_stdp_forward(
            self.learning_rule[5], [self.out[4], self.out[7]], thalamus_links)
        self.out[6], self.dw[6] = self.learning_rule[6](self.out[5], self.out[6])
        br = ["StrD1", "StrD2", "STN", "Gpe", "Gpi", "thalamus", "PM", "DLPFC"]
        for i in range(self.num_subDM):
            if torch.max(self.out[i]) > 0 and self.node_type == "hh":
                self.node[i].n_reset()
            if os.environ.get("BDMSNN_VERBOSE") == "1":
                print("every areas:", br[i], self.out[i])
        return self.out[6], self.dw

    def _cross_core_link(self, name):
        """Return a selected RRR link, or retain full communication."""
        if name in self.cross_core_links:
            return self.cross_core_links[name]
        return None

    def _communication_link(self, name):
        """Return a selected lossy RRR or lossless common-mode link."""
        if name in self.common_mode_links:
            return self.common_mode_links[name]
        return self._cross_core_link(name)

    def _nmf_latent_current(self, pathway, source, fallback_current):
        """Model a DLPFC-source / Str-target NMF link.

        ``left`` is the source-side nonnegative projection U and ``right`` is
        the target-side nonnegative decoder V. The default branch transports
        continuous values for the earlier algorithmic control. The optional
        spike branch sends only IF output events across the core boundary.
        """
        stats = self.striatum_latent_stats[pathway]
        left = getattr(self, f"_striatum_nmf_{pathway}_left")
        right = getattr(self, f"_striatum_nmf_{pathway}_right")
        expected_left = (self.num_state, self.striatum_latent_rank)
        expected_right = (self.striatum_latent_rank, self.num_action)
        if (self.striatum_latent_rank == 0 or left.shape != expected_left or
                right.shape != expected_right):
            stats["full_fallback_samples"] += 1
            return fallback_current
        cached = self._striatum_latent_cache[pathway]
        if (not self.striatum_latent_spike_transport and
                self.striatum_latent_delta_transport and cached is not None and
                torch.equal(source, cached["source"])):
            # An unchanged source spike vector reuses the previously decoded
            # current. This models a target-side sample-and-hold between latent
            # updates; it is exact only while both source and factors are fixed.
            latent = cached["latent"]
            current = cached["current"]
            stats["suppressed_repeat_samples"] += 1
        else:
            spike_left, spike_right = left, right
            if self.striatum_latent_spike_transport and self.striatum_latent_spike_normalize_factors:
                # Equivalent local reparameterization: (U / c)(diag(c)V) = UV.
                # It bounds a one-hot source's latent current without sending
                # a dynamic scale across the link.
                column_scale = left.amax(dim=0).clamp_min(torch.finfo(left.dtype).eps)
                spike_left = left / column_scale
                spike_right = right * column_scale[:, None]
                if self.striatum_latent_spike_contrast_encode:
                    # Remove each latent column's state-independent conductance
                    # before its IF neuron. This is a source-local calibrated
                    # inhibitory reference current, not a transmitted value.
                    # It preserves nonnegative row conductances while exposing
                    # state-dependent variation that max-only normalization hid.
                    column_floor = spike_left.amin(dim=0)
                    column_range = (spike_left.amax(dim=0) - column_floor).clamp_min(
                        torch.finfo(left.dtype).eps)
                    spike_left = (spike_left - column_floor) / column_range
            reference_latent = source @ left
            latent = source @ (spike_left if self.striatum_latent_spike_transport else left)
            if self.striatum_latent_spike_transport:
                # U is a source-core RRAM projection. Its column currents enter
                # local IF neurons; their binary output pulses, not `latent`,
                # are delivered to the decoder crossbar in the Str core.
                projection_current = latent * self.striatum_latent_spike_gain
                if self.striatum_latent_spike_soft_reset:
                    # A first-order sigma-delta encoder preserves the leftover
                    # subthreshold current rather than discarding it after a
                    # pulse. It still emits at most one binary event/neuron/slot.
                    with torch.no_grad():
                        membrane = self._striatum_latent_residual_mem[pathway]
                        if membrane is None or membrane.shape != projection_current.shape:
                            membrane = torch.zeros_like(projection_current)
                        membrane = membrane + projection_current
                        transmitted_latent = (membrane >= self.striatum_latent_spike_threshold).to(
                            projection_current.dtype)
                        membrane = membrane - (transmitted_latent *
                                               self.striatum_latent_spike_threshold)
                        self._striatum_latent_residual_mem[pathway] = membrane.detach()
                else:
                    transmitted_latent = self.striatum_latent_nodes[pathway](projection_current)
                # Binary AER events drive a fixed-amplitude target synapse;
                # this is a conductance/pulse-width calibration, not a payload.
                current = self.striatum_latent_spike_decoder_gain * (transmitted_latent @ spike_right)
                # Expose the physical binary payload for decision-trace audits.
                self._striatum_latent_last_events[pathway] = transmitted_latent.detach().clone()
                self._striatum_latent_decision_pattern[pathway].append(
                    transmitted_latent.detach().clone())
                event_count = int(torch.count_nonzero(transmitted_latent).item())
                decision_events = self._striatum_latent_decision_events[pathway]
                if decision_events is None:
                    decision_events = torch.zeros_like(transmitted_latent)
                self._striatum_latent_decision_events[pathway] = (
                    decision_events + transmitted_latent.detach())
                target_sum = self._striatum_latent_decision_target_current[pathway]
                if target_sum is None:
                    target_sum = torch.zeros_like(fallback_current)
                self._striatum_latent_decision_target_current[pathway] = (
                    target_sum + fallback_current.detach())
                stats["latent_spike_events"] += event_count
                stats["latent_spike_active_samples"] += int(event_count > 0)
                stats["latent_spike_peak_events_per_sample"] = max(
                    stats["latent_spike_peak_events_per_sample"], event_count)
            else:
                transmitted_latent = latent
            if not self.striatum_latent_spike_transport and self.striatum_latent_quant_bits:
                # Fixed full scale is deliberately simple: it maps directly to
                # a DAC/code range and does not hide per-record scale metadata.
                levels = (1 << self.striatum_latent_quant_bits) - 1
                scale = self.striatum_latent_quant_scale
                clipped = latent.clamp(0.0, scale)
                transmitted_latent = torch.round(clipped * (levels / scale)) * (scale / levels)
                stats["quantized_vector_records"] += 1
                stats["quantization_clipped_values"] += int(
                    torch.count_nonzero((latent < 0.0) | (latent > scale)).item())
                stats["quantization_total_values"] += int(latent.numel())
            if not self.striatum_latent_spike_transport:
                current = transmitted_latent @ right
            quantization_error = current - (reference_latent @ right)
            stats["quantization_squared_current_error"] += float(
                quantization_error.square().sum().item())
            stats["quantization_peak_current_error"] = max(
                stats["quantization_peak_current_error"],
                float(quantization_error.abs().max().item()))
            if not self.striatum_latent_spike_transport:
                self._striatum_latent_cache[pathway] = {
                    "source": source.detach().clone(),
                    "latent": transmitted_latent.detach().clone(),
                    "current": current.detach().clone(),
                }
                stats["continuous_vector_records"] += 1
            # The source U MVM runs each internal step in spike mode. The
            # decoder is driven only if at least one binary event arrives.
            stats["source_projection_array_activations"] += 1
            stats["target_decoder_array_activations"] += int(
                (not self.striatum_latent_spike_transport) or
                torch.count_nonzero(transmitted_latent).item() > 0)
        error = current - fallback_current
        stats["samples"] += 1
        stats["source_logical_events"] += int(torch.count_nonzero(source).item())
        stats["latent_scalar_values"] += int(latent.numel())
        stats["latent_nonzero_values"] += int(torch.count_nonzero(latent).item())
        stats["latent_absolute_sum"] += float(latent.abs().sum().item())
        stats["latent_peak_absolute_value"] = max(
            stats["latent_peak_absolute_value"], float(latent.abs().max().item()))
        stats["squared_error"] += float(error.square().sum().item())
        stats["target_energy"] += float(fallback_current.square().sum().item())
        stats["peak_absolute_error"] = max(
            stats["peak_absolute_error"], float(error.abs().max().item()))
        return current

    def begin_striatum_latent_decision(self, reset_residual=False, residual_decay=1.0):
        """Prepare local pulse accounting before one high-level SNN decision."""
        if not 0.0 <= residual_decay <= 1.0:
            raise ValueError("latent residual boundary decay must be in [0, 1]")
        for pathway in self._striatum_latent_decision_events:
            self._striatum_latent_decision_events[pathway] = None
            self._striatum_latent_decision_target_current[pathway] = None
            self._striatum_latent_last_events[pathway] = None
            self._striatum_latent_decision_pattern[pathway] = []
            if reset_residual:
                # This only resets the source sigma-delta phase. BG/PM and
                # striatal states keep their episode history unchanged.
                self._striatum_latent_residual_mem[pathway] = None
            elif residual_decay < 1.0 and self._striatum_latent_residual_mem[pathway] is not None:
                # This is source-local neuron leakage at a control boundary;
                # it does not add a cross-core payload or alter BG/PM state.
                self._striatum_latent_residual_mem[pathway].mul_(residual_decay)

    @torch.no_grad()
    def striatum_latent_residuals(self):
        """Return source-local residuals for trace diagnostics, never payloads."""
        return {
            pathway: (None if self._striatum_latent_residual_mem[pathway] is None else
                      self._striatum_latent_residual_mem[pathway].detach().clone())
            for pathway in ("d1", "d2")
        }

    @torch.no_grad()
    def calibrate_striatum_spike_decoder_to_shadow(self, learning_rate):
        """Calibrate V from actual pulse counts to the local continuous shadow.

        During training, the compact DLPFC--Str table already exists locally
        for STDP and supplies a current target.  This rule uses no action
        label: it fits the *accumulated* binary AER decoder current to that
        local target over one decision window.  At deployment the shadow is
        not consulted; the source U, binary payload, and target V are exactly
        the same spiking path as before.
        """
        if not self.striatum_latent_spike_transport:
            raise ValueError("shadow decoder calibration requires latent spike transport")
        if learning_rate <= 0:
            raise ValueError("shadow decoder calibration rate must be positive")
        changes = {}
        epsilon = torch.finfo(torch.float32).eps
        for pathway in ("d1", "d2"):
            events = self._striatum_latent_decision_events[pathway]
            target = self._striatum_latent_decision_target_current[pathway]
            left = getattr(self, f"_striatum_nmf_{pathway}_left")
            right = getattr(self, f"_striatum_nmf_{pathway}_right")
            if events is None or target is None or left.numel() == 0 or right.numel() == 0:
                changes[pathway] = 0.0
                continue
            # The source-side normalization is an equivalent local change of
            # variables.  Calibrate in decoder-crossbar coordinates, then
            # write the result back into the stored NMF V factor.
            column_scale = (left.amax(dim=0).clamp_min(epsilon)
                            if self.striatum_latent_spike_normalize_factors
                            else torch.ones(right.shape[0], device=right.device,
                                            dtype=right.dtype))
            decoder = right * column_scale[:, None]
            decoded_sum = self.striatum_latent_spike_decoder_gain * (events @ decoder)
            error = target - decoded_sum
            denominator = events.square().sum().clamp_min(1.0)
            before = right.clone()
            decoder.add_(learning_rate * torch.outer(events, error) / denominator).clamp_min_(epsilon)
            right.copy_(decoder / column_scale[:, None]).clamp_min_(epsilon)
            self.connection[0 if pathway == "d1" else 1].weight.copy_(left @ right)
            changes[pathway] = float((right - before).abs().sum().item())
            self._striatum_latent_shadow_calibration_squared_error += float(error.square().sum().item())
        self._striatum_latent_shadow_calibration_updates += 1
        return changes

    @torch.no_grad()
    def teacher_adapt_striatum_spike_decoder(self, action, learning_rate,
                                             competing_action=None):
        """Use locally received latent spikes to imprint a teacher action on V.

        This is a target-core three-factor rule: presynaptic eligibility is the
        actual binary event count accumulated during the decision, while the
        teacher supplies the action-specific modulatory signal.  Optionally,
        a competing PM winner receives the opposite local modulation.  This
        makes the update a ranking signal on the actual spiking readout, not
        merely a positive update to one decoder column.  No source factor or
        cross-core payload is changed by this update.
        """
        if not self.striatum_latent_spike_transport:
            raise ValueError("decoder teacher adaptation requires latent spike transport")
        if action < 0 or action >= self.num_action or learning_rate <= 0:
            raise ValueError("invalid decoder teacher adaptation action or rate")
        if competing_action is not None and (competing_action < 0 or
                                             competing_action >= self.num_action):
            raise ValueError("invalid competing action for decoder teacher adaptation")
        use_ranking = competing_action is not None and competing_action != action
        changes = {}
        epsilon = torch.finfo(torch.float32).eps
        for pathway, sign in (("d1", 1.0), ("d2", -1.0)):
            events = self._striatum_latent_decision_events[pathway]
            left = getattr(self, f"_striatum_nmf_{pathway}_left")
            right = getattr(self, f"_striatum_nmf_{pathway}_right")
            if events is None or left.numel() == 0 or right.numel() == 0:
                changes[pathway] = 0.0
                continue
            eligibility = events / events.sum().clamp_min(1.0)
            before = right[:, action].clone()
            right[:, action].add_(sign * learning_rate * eligibility).clamp_min_(epsilon)
            if use_ranking:
                # D1 should favour the teacher action while D2 suppresses it.
                # For the current wrong PM winner the signs reverse.  Both
                # changes are conditioned on pulses that truly reached Str.
                competitor_before = right[:, competing_action].clone()
                right[:, competing_action].add_(-sign * learning_rate * eligibility).clamp_min_(epsilon)
                changes[pathway] = float(
                    (right[:, action] - before).abs().sum().item() +
                    (right[:, competing_action] - competitor_before).abs().sum().item())
            else:
                changes[pathway] = float((right[:, action] - before).abs().sum().item())
            # Keep the local full shadow table synchronized with the actual
            # target decoder used by the spiking forward path.
            self.connection[0 if pathway == "d1" else 1].weight.copy_(left @ right)
            self._striatum_latent_decoder_teacher_event_mass += float(events.sum().item())
        self._striatum_latent_decoder_teacher_updates += 1
        self._striatum_latent_decoder_teacher_ranking_updates += int(use_ranking)
        return changes

    @torch.no_grad()
    def adapt_striatum_spike_decoder_three_factor(self, action, td_error,
                                                   learning_rate, maximum_update=0.0):
        """Locally adapt target V from binary latent-event eligibility.

        U and the source IF encoder remain fixed.  Each target-core decoder
        row is updated only from pulses received in the just-finished decision
        window, so this path never refactors the global DLPFC--Str table.
        """
        if not self.striatum_latent_spike_transport:
            raise ValueError("decoder three-factor adaptation requires latent spike transport")
        if action < 0 or action >= self.num_action or learning_rate < 0 or maximum_update < 0:
            raise ValueError("invalid decoder three-factor adaptation parameters")
        changes = {}
        epsilon = torch.finfo(torch.float32).eps
        for pathway, dopamine_sign in (("d1", 1.0), ("d2", -1.0)):
            events = self._striatum_latent_decision_events[pathway]
            left = getattr(self, f"_striatum_nmf_{pathway}_left")
            right = getattr(self, f"_striatum_nmf_{pathway}_right")
            if events is None or left.numel() == 0 or right.numel() == 0:
                changes[pathway] = 0.0
                continue
            eligibility = events / events.sum().clamp_min(1.0)
            delta = dopamine_sign * learning_rate * float(td_error) * eligibility
            if maximum_update:
                delta = delta.clamp(-maximum_update, maximum_update)
            before = right[:, action].clone()
            right[:, action].add_(delta).clamp_min_(epsilon)
            self.connection[0 if pathway == "d1" else 1].weight.copy_(left @ right)
            changes[pathway] = float((right[:, action] - before).abs().sum().item())
        return changes

    def _stdp_forward(self, learning_rule, source, communication_link=None, nmf_pathway=None):
        """Preserve STDP on a full shadow weight while transporting RRR current."""
        source = source.clone().detach()
        full_current = learning_rule.connection(source)
        current = full_current
        if communication_link is not None:
            communication_link.observe(source, full_current)
            current = communication_link(source, full_current)
        elif nmf_pathway is not None and self.striatum_latent_rank:
            current = self._nmf_latent_current(nmf_pathway, source, full_current)
        with torch.no_grad():
            spike = learning_rule.node(current)
            # Keep the original local STDP gradient: trace(source) outer spike.
            full_current.data += spike - full_current.data
            trace = learning_rule.cal_trace(source)
            source.data += trace - source.data
        dw = torch.autograd.grad(
            outputs=full_current, inputs=learning_rule.connection.weight,
            grad_outputs=full_current)
        return spike, dw

    def _multi_input_stdp_forward(self, learning_rule, sources, communication_links):
        """Apply optional RRR transport to selected inputs of a multi-input rule."""
        sources = [source.clone().detach() for source in sources]
        full_currents = [connection(source) for source, connection in
                         zip(sources, learning_rule.connection)]
        currents = []
        for source, full_current, link in zip(sources, full_currents, communication_links):
            if link is not None:
                link.observe(source, full_current)
                currents.append(link(source, full_current))
            else:
                currents.append(full_current)
        current = sum(currents)
        full_current = sum(full_currents)
        with torch.no_grad():
            spike = learning_rule.node(current)
            full_current.data += spike - full_current.data
            traces = learning_rule.cal_trace(sources)
            for source, trace in zip(sources, traces):
                source.data += trace - source.data
        dw = torch.autograd.grad(
            outputs=full_current,
            inputs=[connection.weight for connection in learning_rule.connection],
            grad_outputs=full_current)
        return spike, dw

    def UpdateWeight(self, i, s, num_action, dw):
        """
        更新网络中第i组连接的权重
        :param i:要更新的连接组索引
        :param s:传入状态
        :param dw:更新权重的量
        :return:
        """
        action_slice = (slice(0, num_action) if self.compact_striatum else
                        slice(s * num_action, (s + 1) * num_action))
        if s < 0 or action_slice.stop > self.connection[i].weight.shape[1]:
            raise IndexError("state/action slice is outside the DLPFC-to-striatum weight matrix")

        # Only the state-action synapses used for this decision receive the
        # reward-modulated update.  The former two-index form was tied to the
        # Flappy Bird action space and produced NaNs for equal eligibility values.
        delta = torch.zeros_like(dw)
        selected_dw = torch.nan_to_num(dw[s, action_slice], nan=0.0,
                                       posinf=0.0, neginf=0.0)
        if self.node_type == "hh":
            delta[s, action_slice] = 0.2 * self.weight_exc * selected_dw
        elif self.node_type == "lif":
            centered_dw = selected_dw - selected_dw.mean()
            dw_std = centered_dw.std(unbiased=False)
            if torch.isfinite(dw_std) and dw_std > 1e-12:
                delta[s, action_slice] = centered_dw / dw_std
        else:
            raise ValueError("node_type must be 'lif' or 'hh'")

        self.connection[i].update(delta)
        selected_weights = self.connection[i].weight.data[s, action_slice]
        selected_max = selected_weights.max()
        if torch.isfinite(selected_max) and selected_max > 1e-12:
            selected_weights /= selected_max
        else:
            # Preserve a valid exploratory path if inhibition has clipped every
            # action synapse for a state.
            selected_weights.fill_(self.weight_exc)
        if i in [0, 1, 2, 6, 7, 11, 12]:
            self.connection[i].weight.data = torch.clamp(self.connection[i].weight.data, 0, None)
        if i in [3, 4, 5, 8, 10]:
            self.connection[i].weight.data = torch.clamp(self.connection[i].weight.data, None, 0)
        if i == 1 and (self.direct_link is not None or self.cross_core_links):
            # Legacy reward-modulated callers still obtain one refit after
            # their D2 update; three-factor callers invoke the same method
            # explicitly after the decision-level weight update.
            self.refit_communication()

    def refit_communication(self):
        """Causally refit communication links once after a control decision.

        Three-factor updates write the DLPFC--Str weights directly and bypass
        UpdateWeight.  Scheduling here keeps RRR calibration independent of
        which local plasticity rule performed the weight update.
        """
        if self.direct_link is None and not self.cross_core_links:
            return False
        self.communication_updates += 1
        if (self.communication_updates < self.communication_warmup or
                self.communication_updates % self.communication_refit_interval != 0):
            return False
        if self.direct_link is not None:
            self.direct_link.refit()
            self.indirect_link.refit()
        for link in self.cross_core_links.values():
            link.refit()
        self.communication_refit_history.append({
            "decision_update": self.communication_updates,
            "metrics": self.communication_metrics(),
        })
        return True

    def project_striatum_weights_rank(self, rank, method="svd", nmf_iterations=12,
                                      binary_aware=False, binary_window_steps=30,
                                      spike_surrogate=False,
                                      surrogate_steps=16, surrogate_learning_rate=0.05,
                                      surrogate_slope=8.0, surrogate_event_penalty=0.0):
        """Constrain the two DLPFC--Str state-action tables to rank ``rank``.

        This is deliberately separate from the forward RRR replacement.  It
        gives online STDP / teacher updates a low-rank *weight* constraint,
        then a later experiment can transport the resulting factors across a
        core boundary.  The compact-Str masks are dense; the expanded-Str
        topology has a structural action mask and is therefore not a valid
        target for this dense matrix projection.
        """
        if not self.compact_striatum:
            raise ValueError("low-rank DLPFC-to-Str projection requires compact_striatum")
        if rank < 1 or rank > self.num_action:
            raise ValueError("striatum rank must be in [1, num_action]")
        if method not in ("svd", "nmf"):
            raise ValueError("striatum factorization must be 'svd' or 'nmf'")
        if nmf_iterations < 1:
            raise ValueError("nmf_iterations must be positive")
        if binary_aware and method != "nmf":
            raise ValueError("binary-aware projection requires NMF factors")
        if binary_aware and binary_window_steps < 1:
            raise ValueError("binary-aware projection needs a positive window")
        if spike_surrogate and method != "nmf":
            raise ValueError("spike surrogate projection requires NMF factors")
        if spike_surrogate and binary_aware:
            raise ValueError("select either static binary-aware or surrogate spike projection")
        if spike_surrogate and (binary_window_steps < 1 or surrogate_steps < 1 or
                                surrogate_learning_rate <= 0 or surrogate_slope <= 0 or
                                surrogate_event_penalty < 0):
            raise ValueError("invalid surrogate spike projection parameters")
        diagnostics = {}
        for name, connection in (("d1", self.connection[0]), ("d2", self.connection[1])):
            before = connection.weight.detach().clone()
            if method == "svd":
                left, singular_values, right = torch.linalg.svd(before, full_matrices=False)
                left = left[:, :rank] * singular_values[:rank]
                right = right[:rank]
            else:
                # The original DLPFC--Str pathways are excitatory. Multiplicative
                # NMF updates preserve U,V >= 0, unlike truncated SVD. Warm starts
                # allow one local TD/STDP weight change to be absorbed without
                # solving a new factorization from scratch at every decision.
                target = before.clamp_min(0)
                left_name = f"_striatum_nmf_{name}_left"
                right_name = f"_striatum_nmf_{name}_right"
                left = getattr(self, left_name)
                right = getattr(self, right_name)
                if left.shape != (target.shape[0], rank) or right.shape != (rank, target.shape[1]):
                    u, singular_values, vh = torch.linalg.svd(target, full_matrices=False)
                    root = singular_values[:rank].sqrt()
                    left = u[:, :rank].abs() * root
                    right = root[:, None] * vh[:rank].abs()
                    scale = max(float(target.mean().item()), 1e-4) ** 0.5 * 1e-3
                    left.clamp_min_(scale)
                    right.clamp_min_(scale)
                epsilon = torch.finfo(target.dtype).eps
                for _ in range(nmf_iterations):
                    right.mul_((left.transpose(0, 1) @ target) /
                               ((left.transpose(0, 1) @ left) @ right + epsilon))
                    left.mul_((target @ right.transpose(0, 1)) /
                              (left @ (right @ right.transpose(0, 1)) + epsilon))
                    left.clamp_min_(epsilon)
                    right.clamp_min_(epsilon)
                if binary_aware:
                    # Train the stored low-rank table for the *window-average*
                    # output of the normalized sigma-delta latent encoder,
                    # rather than for its unobserved continuous current.
                    column_scale = left.amax(dim=0).clamp_min_(epsilon)
                    left.div_(column_scale)
                    right.mul_(column_scale[:, None])
                    encoded_left = torch.round(left * binary_window_steps).clamp_(
                        0, binary_window_steps) / binary_window_steps
                    # With the encoder basis fixed, locally adapt V to the
                    # actual pulse-count basis. No count is sent cross-core.
                    for _ in range(nmf_iterations):
                        right.mul_((encoded_left.transpose(0, 1) @ target) /
                                   ((encoded_left.transpose(0, 1) @ encoded_left) @ right + epsilon))
                        right.clamp_min_(epsilon)
                    left = encoded_left
                elif spike_surrogate:
                    # Optimize the factors against the *actual discrete
                    # soft-reset encoder used by the hardware branch.  The
                    # forward value is a hard binary spike; sigmoid supplies
                    # only the backward surrogate derivative.  This remains a
                    # local offline factor fit, not differentiating through
                    # the rest of the basal-ganglia control loop.
                    column_scale = left.amax(dim=0).clamp_min(epsilon)
                    train_left = (left / column_scale).clamp(0.0, 1.0).detach().requires_grad_(True)
                    train_decoder = (right * column_scale[:, None]).detach().requires_grad_(True)
                    target_scale = target.square().mean().clamp_min(epsilon)
                    for _ in range(surrogate_steps):
                        with torch.enable_grad():
                            membrane = torch.zeros_like(train_left)
                            count = torch.zeros_like(train_left)
                            for _ in range(binary_window_steps):
                                membrane = membrane + train_left
                                hard = (membrane >= self.striatum_latent_spike_threshold).to(
                                    train_left.dtype)
                                soft = torch.sigmoid(surrogate_slope * (
                                    membrane - self.striatum_latent_spike_threshold))
                                # Straight-through binary event: exact hard
                                # AER semantics forward, smooth derivative back.
                                spike = hard + soft - soft.detach()
                                membrane = membrane - self.striatum_latent_spike_threshold * spike
                                count = count + spike
                            reconstructed_spike = (
                                self.striatum_latent_spike_decoder_gain *
                                (count / binary_window_steps) @ train_decoder)
                            loss = ((reconstructed_spike - target).square().mean() / target_scale)
                            if surrogate_event_penalty:
                                loss = loss + surrogate_event_penalty * (
                                    count.mean() / binary_window_steps)
                            grad_left, grad_decoder = torch.autograd.grad(
                                loss, (train_left, train_decoder), create_graph=False)
                        with torch.no_grad():
                            train_left.sub_(surrogate_learning_rate * grad_left).clamp_(0.0, 1.0)
                            train_decoder.sub_(surrogate_learning_rate * grad_decoder).clamp_min_(epsilon)
                    with torch.no_grad():
                        membrane = torch.zeros_like(train_left)
                        count = torch.zeros_like(train_left)
                        for _ in range(binary_window_steps):
                            membrane = membrane + train_left
                            spike = (membrane >= self.striatum_latent_spike_threshold).to(train_left.dtype)
                            membrane = membrane - self.striatum_latent_spike_threshold * spike
                            count = count + spike
                        left = train_left.detach()
                        right = train_decoder.detach()
                        reconstructed = (self.striatum_latent_spike_decoder_gain *
                                         (count / binary_window_steps) @ right)
                        diagnostics[name] = {
                            "surrogate_spike_nrmse": float(
                                ((reconstructed - target).square().mean() / target_scale).sqrt().item()),
                            "surrogate_mean_events_per_source_step": float(
                                (count.mean() / binary_window_steps).item()),
                        }
                setattr(self, left_name, left)
                setattr(self, right_name, right)
            reconstructed = left @ right
            with torch.no_grad():
                connection.weight.copy_(reconstructed)
            # A new online factorization invalidates the held target current.
            self._striatum_latent_cache[name] = None
            self.striatum_latent_nodes[name].n_reset()
            self._striatum_latent_residual_mem[name] = None
            denominator = before.square().sum()
            diagnostics[name] = {
                **diagnostics.get(name, {}),
                "rank": rank,
                "factorization": method,
                "retained_weight_energy": float(
                    (reconstructed.square().sum() / denominator).item()
                    if denominator > 1e-12 else 1.0),
                "projection_nrmse": float(
                    (((before - reconstructed).square().sum() / denominator).sqrt().item()
                     if denominator > 1e-12 else 0.0)),
                "minimum_weight": float(reconstructed.min().item()),
                "maximum_weight": float(reconstructed.max().item()),
                "binary_aware_projection": bool(binary_aware),
                "binary_window_steps": (int(binary_window_steps) if binary_aware else None),
                "spike_surrogate_projection": bool(spike_surrogate),
                "spike_surrogate_steps": (int(surrogate_steps) if spike_surrogate else None),
            }
        return diagnostics

    def train_striatum_spike_pm_teacher(self, source, action, steps=16,
                                         learning_rate=0.01, surrogate_slope=8.0,
                                         unroll_steps=30, event_penalty=0.0,
                                         weight_penalty=1e-4, class_weights=None,
                                         decoder_maximum=16.0, gradient_clip=10.0,
                                         warmup_decisions=0, decision_margin=0.0,
                                         margin_penalty=0.0,
                                         latent_action_penalty=0.0,
                                         reset_latent_each_decision=False,
                                         trainable_left_rows=None,
                                         freeze_decoder=False):
        """Train NMF factors using a teacher action through a BG--PM surrogate.

        The source latent encoder still emits hard binary soft-reset events in
        the forward pass.  A compact differentiable unrolling of the existing
        StrD1/StrD2--GPe/GPi--thalamus--PM topology supplies a PM action loss.
        By default both U and V are changed, and all recurrent BG weights
        remain fixed. ``trainable_left_rows`` instead restricts programming to
        selected U source rows and ``freeze_decoder`` keeps V bit-identical;
        together they support local context-neuron adaptation without
        perturbing an established sensory communication pathway.
        """
        if (not self.compact_striatum or not self.striatum_latent_spike_transport or
                not self.striatum_latent_spike_soft_reset or
                not self.striatum_latent_spike_normalize_factors):
            raise ValueError("PM teacher surrogate requires normalized soft-reset compact latent spikes")
        if source.ndim == 1:
            source = source.unsqueeze(0).unsqueeze(1)
        elif source.ndim == 2:
            source = source.unsqueeze(1)
        if source.ndim != 3 or source.shape[2] != self.num_state:
            raise ValueError("invalid PM teacher source")
        batch_size, decision_steps, _ = source.shape
        if isinstance(action, torch.Tensor):
            actions = action.to(device=source.device, dtype=torch.long).reshape(-1)
        else:
            actions = torch.as_tensor(action, device=source.device, dtype=torch.long).reshape(-1)
        if actions.numel() == 1 and batch_size * decision_steps != 1:
            actions = actions.expand(batch_size * decision_steps)
        if actions.numel() != batch_size * decision_steps or torch.any(actions < 0) or torch.any(actions >= self.num_action):
            raise ValueError("invalid PM teacher action")
        actions = actions.view(batch_size, decision_steps)
        if class_weights is not None:
            class_weights = class_weights.to(device=source.device, dtype=source.dtype)
            if class_weights.shape != (self.num_action,) or torch.any(class_weights < 0):
                raise ValueError("invalid PM teacher class weights")
        if (steps < 1 or unroll_steps < 1 or learning_rate < 0 or surrogate_slope <= 0 or
                warmup_decisions < 0 or warmup_decisions >= decision_steps):
            raise ValueError("invalid PM teacher surrogate parameters")
        if (event_penalty < 0 or weight_penalty < 0 or decision_margin < 0 or
                margin_penalty < 0 or latent_action_penalty < 0 or decoder_maximum <= 0 or
                gradient_clip <= 0):
            raise ValueError("PM teacher regularizers must be nonnegative")
        if trainable_left_rows is not None:
            trainable_left_rows = torch.as_tensor(
                trainable_left_rows, device=source.device, dtype=torch.long).reshape(-1)
            if (trainable_left_rows.numel() == 0 or torch.any(trainable_left_rows < 0) or
                    torch.any(trainable_left_rows >= self.num_state)):
                raise ValueError("invalid trainable latent-U rows")
            trainable_left_rows = torch.unique(trainable_left_rows)
        epsilon = torch.finfo(source.dtype).eps
        diagnostics = {}

        # The non-plastic BG/PM matrices are constants in this offline BPTT
        # surrogate.  Their signs and action-channel topology are identical
        # to the deployed BDMSNN; only the spike derivative is approximated.
        fixed = [connection.weight.detach() for connection in self.connection]
        thresholds = [node.threshold.detach().to(source) for node in self.node]

        def ste_spike(membrane, current, threshold):
            """Match IFNode: hard threshold followed by a full reset."""
            membrane = membrane + current
            hard = (membrane >= threshold).to(source.dtype)
            soft = torch.sigmoid(surrogate_slope * (membrane - threshold))
            spike = hard + soft - soft.detach()
            # Base IFNode resets its full membrane after a spike.  The detach
            # preserves that exact hard forward reset while retaining the STE
            # derivative through the emitted event.
            return membrane * (1.0 - spike.detach()), spike

        for name in ("d1", "d2"):
            left = getattr(self, f"_striatum_nmf_{name}_left")
            right = getattr(self, f"_striatum_nmf_{name}_right")
            expected_left = (self.num_state, self.striatum_latent_rank)
            expected_right = (self.striatum_latent_rank, self.num_action)
            if left.shape != expected_left or right.shape != expected_right:
                raise ValueError("PM teacher surrogate requires initialized NMF factors")

        left_d1 = self._striatum_nmf_d1_left
        left_d2 = self._striatum_nmf_d2_left
        right_d1 = self._striatum_nmf_d1_right
        right_d2 = self._striatum_nmf_d2_right
        local_left_only = trainable_left_rows is not None
        if local_left_only:
            # Do not normalize a factorization that is meant to stay frozen:
            # contrast coding depends on the actual U rows, not just U @ V.
            train_left_d1 = left_d1.detach().clone().requires_grad_(True)
            train_left_d2 = left_d2.detach().clone().requires_grad_(True)
            train_decoder_d1 = right_d1.detach().clone().requires_grad_(True)
            train_decoder_d2 = right_d2.detach().clone().requires_grad_(True)
            frozen_left_d1 = train_left_d1.detach().clone()
            frozen_left_d2 = train_left_d2.detach().clone()
            left_floor_d1 = frozen_left_d1.amin(dim=0)
            left_floor_d2 = frozen_left_d2.amin(dim=0)
        else:
            scale_d1 = left_d1.amax(dim=0).clamp_min(epsilon)
            scale_d2 = left_d2.amax(dim=0).clamp_min(epsilon)
            train_left_d1 = (left_d1 / scale_d1).clamp(0.0, 1.0).detach().requires_grad_(True)
            train_left_d2 = (left_d2 / scale_d2).clamp(0.0, 1.0).detach().requires_grad_(True)
            train_decoder_d1 = (right_d1 * scale_d1[:, None]).detach().requires_grad_(True)
            train_decoder_d2 = (right_d2 * scale_d2[:, None]).detach().requires_grad_(True)

        def encoded_left(left):
            if not self.striatum_latent_spike_contrast_encode:
                return left
            floor = left.amin(dim=0)
            dynamic_range = (left.amax(dim=0) - floor).clamp_min(epsilon)
            return (left - floor) / dynamic_range

        for _ in range(steps):
            # Separate latent encoders are physically located at the source
            # side of their respective D1/D2 links.
            latent_mem_d1 = torch.zeros(batch_size, self.striatum_latent_rank, device=source.device,
                                        dtype=source.dtype)
            latent_mem_d2 = torch.zeros_like(latent_mem_d1)
            dlpfc_mem = torch.zeros(batch_size, self.num_state, device=source.device,
                                    dtype=source.dtype)
            str_d1_mem = torch.zeros(batch_size, self.num_action, device=source.device, dtype=source.dtype)
            str_d2_mem = torch.zeros_like(str_d1_mem)
            stn_mem = torch.zeros(batch_size, fixed[2].shape[1], device=source.device,
                                  dtype=source.dtype)
            gpe_mem = torch.zeros(batch_size, self.num_action, device=source.device, dtype=source.dtype)
            gpi_mem = torch.zeros_like(gpe_mem)
            thalamus_mem = torch.zeros_like(gpe_mem)
            pm_mem = torch.zeros_like(gpe_mem)
            gpe_spike = torch.zeros_like(gpe_mem)
            pm_spike = torch.zeros_like(gpe_mem)
            event_total = source.new_zeros(())
            task_loss = source.new_zeros(())
            margin_total = source.new_zeros(())
            margin_violation_total = source.new_zeros(())
            latent_action_total = source.new_zeros(())
            for decision_index in range(decision_steps):
                # State persists within a trajectory, whereas the PM vote is
                # read over the current high-level decision window only.
                if reset_latent_each_decision:
                    # Match the deployment boundary-reset experiment without
                    # resetting Str/BG/PM membrane history.
                    latent_mem_d1.zero_()
                    latent_mem_d2.zero_()
                pm_count = torch.zeros_like(gpe_mem)
                d1_current_count = torch.zeros_like(gpe_mem)
                decision_source = source[:, decision_index]
                for _ in range(unroll_steps):
                    # Match BDMSNN.forward: sensory current first creates a
                    # DLPFC IF spike.  That binary DLPFC output drives U, STN,
                    # and thalamus, rather than the unthresholded encoder current.
                    dlpfc_mem, dlpfc = ste_spike(dlpfc_mem, decision_source @ fixed[9], thresholds[7])
                    latent_mem_d1 = latent_mem_d1 + (
                        dlpfc @ encoded_left(train_left_d1) * self.striatum_latent_spike_gain)
                    hard_d1 = (latent_mem_d1 >= self.striatum_latent_spike_threshold).to(source.dtype)
                    soft_d1 = torch.sigmoid(surrogate_slope * (
                        latent_mem_d1 - self.striatum_latent_spike_threshold))
                    latent_d1 = hard_d1 + soft_d1 - soft_d1.detach()
                    latent_mem_d1 = latent_mem_d1 - self.striatum_latent_spike_threshold * latent_d1.detach()
                    latent_mem_d2 = latent_mem_d2 + (
                        dlpfc @ encoded_left(train_left_d2) * self.striatum_latent_spike_gain)
                    hard_d2 = (latent_mem_d2 >= self.striatum_latent_spike_threshold).to(source.dtype)
                    soft_d2 = torch.sigmoid(surrogate_slope * (
                        latent_mem_d2 - self.striatum_latent_spike_threshold))
                    latent_d2 = hard_d2 + soft_d2 - soft_d2.detach()
                    latent_mem_d2 = latent_mem_d2 - self.striatum_latent_spike_threshold * latent_d2.detach()
                    str_d1_mem, str_d1 = ste_spike(
                        str_d1_mem, self.striatum_latent_spike_decoder_gain * latent_d1 @ train_decoder_d1,
                        thresholds[0])
                    d1_current_count = d1_current_count + latent_d1 @ train_decoder_d1
                    str_d2_mem, str_d2 = ste_spike(
                        str_d2_mem, self.striatum_latent_spike_decoder_gain * latent_d2 @ train_decoder_d2,
                        thresholds[1])
                    stn_mem, stn = ste_spike(stn_mem, dlpfc @ fixed[2] + gpe_spike @ fixed[8], thresholds[2])
                    gpe_mem, gpe_spike = ste_spike(gpe_mem, str_d2 @ fixed[4] + stn @ fixed[7], thresholds[3])
                    gpi_mem, gpi = ste_spike(
                        gpi_mem, str_d1 @ fixed[3] + gpe_spike @ fixed[5] + stn @ fixed[6], thresholds[4])
                    thalamus_mem, thalamus = ste_spike(
                        thalamus_mem, gpi @ fixed[10] + dlpfc @ fixed[12], thresholds[5])
                    pm_mem, pm_spike = ste_spike(
                        pm_mem, thalamus @ fixed[11] + pm_spike @ fixed[13], thresholds[6])
                    pm_count = pm_count + pm_spike
                    event_total = event_total + latent_d1.sum() + latent_d2.sum()
                current_actions = actions[:, decision_index]
                if decision_index >= warmup_decisions:
                    task_loss = task_loss + F.cross_entropy(
                        pm_count, current_actions, weight=class_weights)
                    if latent_action_penalty:
                        # This uses the same hard AER pulses that drive StrD1;
                        # it is a training constraint, not a deployment bypass.
                        latent_action_total = latent_action_total + F.cross_entropy(
                            d1_current_count, current_actions, weight=class_weights)
                    target_score = pm_count.gather(1, current_actions[:, None]).squeeze(1)
                    other_scores = pm_count.clone()
                    other_scores.scatter_(1, current_actions[:, None], float("-inf"))
                    competitor_score = other_scores.max(dim=1).values
                    realized_margin = target_score - competitor_score
                    margin_total = margin_total + realized_margin.mean()
                    if decision_margin and margin_penalty:
                        # CE rewards the correct class on average, but a tied
                        # integer spike vote still causes random deployment
                        # selection.  Demand a physical count lead explicitly.
                        margin_violation_total = margin_violation_total + F.relu(
                            decision_margin - realized_margin).mean()
            supervised_decisions = decision_steps - warmup_decisions
            loss = (task_loss + latent_action_penalty * latent_action_total) / supervised_decisions
            if decision_margin and margin_penalty:
                loss = loss + margin_penalty * margin_violation_total / supervised_decisions
            if event_penalty:
                loss = loss + event_penalty * event_total / (
                    2 * batch_size * decision_steps * self.striatum_latent_rank * unroll_steps)
            if weight_penalty:
                loss = loss + weight_penalty * (
                    train_decoder_d1.square().mean() + train_decoder_d2.square().mean())
            gradients = torch.autograd.grad(
                loss, (train_left_d1, train_decoder_d1, train_left_d2, train_decoder_d2),
                create_graph=False)
            with torch.no_grad():
                if learning_rate:
                    # Long warm-up windows can amplify a surrogate derivative
                    # through recurrent IF resets.  Preserve the hard forward
                    # path, but never program a non-finite conductance.
                    clipped = [torch.nan_to_num(gradient, nan=0.0,
                                                posinf=gradient_clip,
                                                neginf=-gradient_clip).clamp(
                                                -gradient_clip, gradient_clip)
                               for gradient in gradients]
                    if local_left_only:
                        # Keep old sensory rows and the contrast-code floor
                        # fixed. Only designated context rows can be written.
                        train_left_d1.index_add_(0, trainable_left_rows,
                                                 -learning_rate * clipped[0].index_select(
                                                     0, trainable_left_rows))
                        train_left_d2.index_add_(0, trainable_left_rows,
                                                 -learning_rate * clipped[2].index_select(
                                                     0, trainable_left_rows))
                        train_left_d1.clamp_(min=0.0)
                        train_left_d2.clamp_(min=0.0)
                        train_left_d1.index_copy_(0, trainable_left_rows,
                                                  train_left_d1.index_select(
                                                      0, trainable_left_rows).maximum(left_floor_d1))
                        train_left_d2.index_copy_(0, trainable_left_rows,
                                                  train_left_d2.index_select(
                                                      0, trainable_left_rows).maximum(left_floor_d2))
                        train_left_d1.copy_(frozen_left_d1.scatter(
                            0, trainable_left_rows[:, None].expand(-1, train_left_d1.shape[1]),
                            train_left_d1.index_select(0, trainable_left_rows)))
                        train_left_d2.copy_(frozen_left_d2.scatter(
                            0, trainable_left_rows[:, None].expand(-1, train_left_d2.shape[1]),
                            train_left_d2.index_select(0, trainable_left_rows)))
                        if not freeze_decoder:
                            train_decoder_d1.sub_(learning_rate * clipped[1]).clamp_(epsilon, decoder_maximum)
                            train_decoder_d2.sub_(learning_rate * clipped[3]).clamp_(epsilon, decoder_maximum)
                    else:
                        train_left_d1.sub_(learning_rate * clipped[0]).clamp_(0.0, 1.0)
                        train_decoder_d1.sub_(learning_rate * clipped[1]).clamp_(epsilon, decoder_maximum)
                        train_left_d2.sub_(learning_rate * clipped[2]).clamp_(0.0, 1.0)
                        train_decoder_d2.sub_(learning_rate * clipped[3]).clamp_(epsilon, decoder_maximum)

        with torch.no_grad():
            if local_left_only:
                self._striatum_nmf_d1_left.copy_(train_left_d1)
                self._striatum_nmf_d1_right.copy_(train_decoder_d1)
                self._striatum_nmf_d2_left.copy_(train_left_d2)
                self._striatum_nmf_d2_right.copy_(train_decoder_d2)
            else:
                deployed_scale_d1 = train_left_d1.amax(dim=0).clamp_min(epsilon)
                deployed_scale_d2 = train_left_d2.amax(dim=0).clamp_min(epsilon)
                self._striatum_nmf_d1_left.copy_(train_left_d1)
                self._striatum_nmf_d1_right.copy_(
                    train_decoder_d1 / deployed_scale_d1[:, None]).clamp_min_(epsilon)
                self._striatum_nmf_d2_left.copy_(train_left_d2)
                self._striatum_nmf_d2_right.copy_(
                    train_decoder_d2 / deployed_scale_d2[:, None]).clamp_min_(epsilon)
            self.connection[0].weight.copy_(self._striatum_nmf_d1_left @ self._striatum_nmf_d1_right)
            self.connection[1].weight.copy_(self._striatum_nmf_d2_left @ self._striatum_nmf_d2_right)
            diagnostics["loss"] = float(loss.item())
            diagnostics["latent_action_loss"] = float(
                (latent_action_total / supervised_decisions).item())
            diagnostics["pm_teacher_margin"] = float((margin_total / supervised_decisions).item())
            diagnostics["pm_teacher_margin_violation"] = float(
                (margin_violation_total / supervised_decisions).item())
            diagnostics["mean_latent_events_per_step"] = float(
                (event_total / (2 * batch_size * decision_steps * self.striatum_latent_rank * unroll_steps)).item())
            diagnostics["local_left_only"] = bool(local_left_only)
            diagnostics["freeze_decoder"] = bool(freeze_decoder)
        return diagnostics

    def striatum_latent_metrics(self):
        """Report continuous latent traffic without assigning it an AER format."""
        if self.striatum_latent_rank == 0:
            return None
        links = {}
        for pathway, stats in self.striatum_latent_stats.items():
            target_energy = stats["target_energy"]
            links[f"dlpfc_str{pathway}"] = {
                "source_dim": self.num_state,
                "latent_dim": self.striatum_latent_rank,
                "target_dim": self.num_action,
                "transport": ("binary_if_spike_nmf_latent"
                              if self.striatum_latent_spike_transport
                              else "continuous_nonnegative_nmf_latent"),
                "spike_transport": {
                    "enabled": self.striatum_latent_spike_transport,
                    "source_neuron": "IF",
                    "encoder": ("soft_reset_sigma_delta"
                                if self.striatum_latent_spike_soft_reset else "hard_reset_if"),
            "factor_normalization": self.striatum_latent_spike_normalize_factors,
                    "contrast_encoding": self.striatum_latent_spike_contrast_encode,
                    "threshold": self.striatum_latent_spike_threshold,
                    "projection_current_gain": self.striatum_latent_spike_gain,
                    "decoder_pulse_gain": self.striatum_latent_spike_decoder_gain,
                    "logical_spike_events": stats["latent_spike_events"],
                    "active_event_samples": stats["latent_spike_active_samples"],
                    "peak_events_per_sample": stats["latent_spike_peak_events_per_sample"],
                    "mean_events_per_sample": (
                        stats["latent_spike_events"] / stats["samples"]
                        if stats["samples"] else None),
                    "decoder_teacher_updates": self._striatum_latent_decoder_teacher_updates,
                    "decoder_teacher_ranking_updates": (
                        self._striatum_latent_decoder_teacher_ranking_updates),
                    "shadow_calibration_updates": self._striatum_latent_shadow_calibration_updates,
                    "shadow_calibration_squared_error": (
                        self._striatum_latent_shadow_calibration_squared_error),
                    "decoder_teacher_event_mass": self._striatum_latent_decoder_teacher_event_mass,
                },
                "quantization": {
                    "enabled": bool(self.striatum_latent_quant_bits),
                    "bits_per_scalar": self.striatum_latent_quant_bits,
                    "fixed_full_scale": self.striatum_latent_quant_scale,
                    "quantized_vector_records": stats["quantized_vector_records"],
                    "clipped_values": stats["quantization_clipped_values"],
                    "total_quantized_values": stats["quantization_total_values"],
                    "clip_rate": (
                        stats["quantization_clipped_values"] / stats["quantization_total_values"]
                        if stats["quantization_total_values"] else None),
                    "current_nrmse_vs_unquantized_latent": (
                        (stats["quantization_squared_current_error"] / target_energy) ** 0.5
                        if target_energy > 1e-12 else None),
                    "peak_current_error_vs_unquantized_latent": (
                        stats["quantization_peak_current_error"]),
                },
                "active_samples": stats["samples"],
                "full_warmup_fallback_samples": stats["full_fallback_samples"],
                "source_logical_events": stats["source_logical_events"],
                "continuous_latent_scalar_values": stats["latent_scalar_values"],
                "continuous_latent_nonzero_values": stats["latent_nonzero_values"],
                "transmitted_continuous_vector_records": stats["continuous_vector_records"],
                "transmitted_continuous_latent_scalar_values": (
                    stats["continuous_vector_records"] * self.striatum_latent_rank),
                "mean_absolute_latent_value": (
                    stats["latent_absolute_sum"] / stats["latent_scalar_values"]
                    if stats["latent_scalar_values"] else None),
                "peak_absolute_latent_value": stats["latent_peak_absolute_value"],
                "source_projection_array_activations": stats["source_projection_array_activations"],
                "target_decoder_array_activations": stats["target_decoder_array_activations"],
                "continuous_vector_records": stats["continuous_vector_records"],
                "suppressed_repeat_samples": stats["suppressed_repeat_samples"],
                "repeat_suppression_rate": (
                    stats["suppressed_repeat_samples"] / stats["samples"]
                    if stats["samples"] else None),
                "online_current_ev": (
                    1.0 - stats["squared_error"] / target_energy
                    if target_energy > 1e-12 else None),
                "online_current_nrmse": (
                    (stats["squared_error"] / target_energy) ** 0.5
                    if target_energy > 1e-12 else None),
                "online_current_peak_absolute_error": stats["peak_absolute_error"],
            }
        return links

    def communication_metrics(self):
        """Return the current RRR fit diagnostics without claiming event savings."""
        latent_links = self.striatum_latent_metrics()
        if (self.direct_link is None and not self.cross_core_links and not self.common_mode_links and
                latent_links is None and
                self.full_traffic_monitor is None):
            return None
        if self.cross_core_links or self.common_mode_links:
            links = {
                name: {
                    "source_dim": link.source_dim,
                    "target_dim": link.target_dim,
                    "rank": link.rank,
                    "retained_variance": link.last_retained_variance,
                    "fitted": link.fitted,
                    "transport": link.transport_summary(),
                }
                for name, link in self.cross_core_links.items()
            }
            links.update({
                name: {
                    "source_dim": link.source_dim,
                    "target_dim": link.target_dim,
                    "rank": 1,
                    "retained_variance": 1.0,
                    "fitted": True,
                    "transport": link.transport_summary(),
                }
                for name, link in self.common_mode_links.items()
            })
            return {
                "rank": (self.communication_link_ranks
                         if self.communication_link_ranks is not None else self.communication_rank),
                "mode": self.communication_mode,
                "fitted": all(link["fitted"] for link in links.values()),
                "links": links,
            }
        if self.full_traffic_monitor is not None:
            if latent_links is not None:
                return {
                    "rank": self.striatum_latent_rank,
                    "mode": "nmf_dlpfc_to_striatum_latent",
                    "fitted": all(link["active_samples"] > 0 for link in latent_links.values()),
                    "links": latent_links,
                    "other_full_cross_core_links": self.full_traffic_monitor.summary(),
                }
            return {
                "rank": 0,
                "mode": self.communication_mode,
                "fitted": False,
                "links": self.full_traffic_monitor.summary(),
            }
        return {
            "rank": self.communication_rank,
            "mode": self.communication_mode,
            "direct_retained_variance": self.direct_link.last_retained_variance,
            "indirect_retained_variance": self.indirect_link.last_retained_variance,
            "fitted": self.direct_link.fitted and self.indirect_link.fitted,
        }

    def communication_history(self):
        """Return immutable per-refit diagnostics for offline experiment logs."""
        return list(self.communication_refit_history)

    def reset(self):
        """
        reset神经元或学习法则的中间量
        :return: None
        """
        for i in range(self.num_subDM):
            self.node[i].n_reset()
        for i in range(len(self.learning_rule)):
            self.learning_rule[i].reset()
        for pathway in self._striatum_latent_cache:
            self._striatum_latent_cache[pathway] = None
            self._striatum_latent_decision_events[pathway] = None
            self._striatum_latent_last_events[pathway] = None
            self._striatum_latent_decision_pattern[pathway] = []
            self._striatum_latent_decision_target_current[pathway] = None
            self.striatum_latent_nodes[pathway].n_reset()
            self._striatum_latent_residual_mem[pathway] = None

    def getweight(self):
        """
        获取网络的连接(包括权值等)
        :return: 网络的连接
        """
        return self.connection
