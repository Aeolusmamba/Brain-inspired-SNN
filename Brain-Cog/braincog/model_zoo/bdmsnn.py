
import os

import torch
from torch import nn

from braincog.base.node.node import IFNode, SimHHNode
from braincog.base.learningrule.STDP import STDP, MutliInputSTDP
from braincog.base.connection.CustomLinear import CustomLinear
from braincog.base.brainarea.basalganglia import basalganglia
from braincog.model_zoo.communication_subspace import (
    CrossCoreTrafficMonitor,
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
                 pm_lateral_gain=None, compact_striatum=False):
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
        self.communication_lossless_only = communication_lossless_only
        self.communication_updates = 0
        self.communication_refit_history = []
        self.direct_link = None
        self.indirect_link = None
        self.cross_core_links = nn.ModuleDict()
        self.full_traffic_monitor = None
        if communication_rank:
            if communication_mode == "state_to_striatum":
                self.direct_link = ReducedRankCommunication(
                    num_state, num_state * num_action, communication_rank,
                    communication_window,
                    fit_lossless_tolerance=(float("inf") if not communication_lossless_only else 1e-6))
                self.indirect_link = ReducedRankCommunication(
                    num_state, num_state * num_action, communication_rank,
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
                self.cross_core_links = nn.ModuleDict({
                    name: ReducedRankCommunication(
                        source_dim, target_dim, communication_rank,
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
            self.out[0], self.dw[0] = self._stdp_forward(self.learning_rule[0], self.out[7])
            self.out[1], self.dw[1] = self._stdp_forward(self.learning_rule[1], self.out[7])
        if self.communication_mode == "all_cross_core" and self.cross_core_links:
            stn_links = [self._cross_core_link("dlpfc_stn"),
                         self._cross_core_link("gpe_stn")]
        elif self.full_traffic_monitor is not None:
            self.full_traffic_monitor.observe("dlpfc_stn", self.out[7])
            self.full_traffic_monitor.observe("gpe_stn", self.out[3])
            stn_links = [None, None]
        else:
            stn_links = [None, None]
        self.out[2], self.dw[2] = self._multi_input_stdp_forward(
            self.learning_rule[2], [self.out[7], self.out[3]], stn_links)
        if self.communication_mode == "all_cross_core" and self.cross_core_links:
            gpe_links = [self._cross_core_link("strd2_gpe"),
                         self._cross_core_link("stn_gpe")]
            gpi_links = [self._cross_core_link("strd1_gpi"), None,
                         self._cross_core_link("stn_gpi")]
            thalamus_links = [self._cross_core_link("gpi_thalamus"),
                              self._cross_core_link("dlpfc_thalamus")]
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

    def _stdp_forward(self, learning_rule, source, communication_link=None):
        """Preserve STDP on a full shadow weight while transporting RRR current."""
        source = source.clone().detach()
        full_current = learning_rule.connection(source)
        current = full_current
        if communication_link is not None:
            communication_link.observe(source, full_current)
            current = communication_link(source, full_current)
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
            self.communication_updates += 1
            if (self.communication_updates >= self.communication_warmup and
                    self.communication_updates % self.communication_refit_interval == 0):
                if self.direct_link is not None:
                    self.direct_link.refit()
                    self.indirect_link.refit()
                for link in self.cross_core_links.values():
                    link.refit()
                self.communication_refit_history.append({
                    "update": self.communication_updates,
                    "metrics": self.communication_metrics(),
                })

    def communication_metrics(self):
        """Return the current RRR fit diagnostics without claiming event savings."""
        if (self.direct_link is None and not self.cross_core_links and
                self.full_traffic_monitor is None):
            return None
        if self.cross_core_links:
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
            return {
                "rank": self.communication_rank,
                "mode": self.communication_mode,
                "fitted": all(link["fitted"] for link in links.values()),
                "links": links,
            }
        if self.full_traffic_monitor is not None:
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

    def getweight(self):
        """
        获取网络的连接(包括权值等)
        :return: 网络的连接
        """
        return self.connection
