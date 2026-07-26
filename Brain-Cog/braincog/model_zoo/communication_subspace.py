"""Online reduced-rank communication links for the BDM-SNN prototype."""

from collections import deque

import torch
from torch import nn


def cross_core_link_dimensions(num_state=9, num_action=2):
    """Return the BDM-SNN core-boundary dimensions for a state/action space.

    The original constant described only Flappy Bird (9 states, 2 actions).
    Keeping the topology parameterized lets a full-traffic monitor and a later
    RRR ablation describe arbitrary discrete control tasks consistently.
    """
    return {
        "dlpfc_stn": (num_state, 2),
        "strd2_gpe": (num_state * num_action, num_action),
        "stn_gpe": (2, num_action),
        "strd1_gpi": (num_state * num_action, num_action),
        "stn_gpi": (2, num_action),
        "gpe_stn": (num_action, 2),
        "gpi_thalamus": (num_action, num_action),
        "dlpfc_thalamus": (num_state, num_action),
    }


# Backward-compatible Flappy Bird dimensions for external callers.
ALL_CROSS_CORE_LINK_DIMENSIONS = cross_core_link_dimensions()


class CrossCoreTrafficMonitor(nn.Module):
    """Count source-side logical spike events for a fixed core partition."""

    def __init__(self, link_dimensions):
        super().__init__()
        self.link_dimensions = dict(link_dimensions)
        self.sample_count = {name: 0 for name in link_dimensions}
        self.source_events = {name: 0 for name in link_dimensions}

    @torch.no_grad()
    def observe(self, name, source):
        self.sample_count[name] += 1
        self.source_events[name] += int(torch.count_nonzero(source).item())

    def summary(self):
        return {
            name: {
                "source_dim": self.link_dimensions[name][0],
                "target_dim": self.link_dimensions[name][1],
                "samples": self.sample_count[name],
                "source_logical_events": self.source_events[name],
            }
            for name in self.link_dimensions
        }


class ReducedRankCommunication(nn.Module):
    """Fit an affine RRR map from source activity to full target current.

    The full connection remains the learning shadow used by STDP.  This link
    only replaces the forward cross-core current after a causal calibration
    window has been observed, so low-rank transport can be ablated without
    changing the local plasticity rule.
    """

    def __init__(self, source_dim, target_dim, rank, window=128,
                 fit_lossless_tolerance=1e-6):
        super().__init__()
        if rank < 1 or rank > min(source_dim, target_dim):
            raise ValueError("rank must be between 1 and min(source_dim, target_dim)")
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.rank = rank
        self.window = window
        self.fit_lossless_tolerance = fit_lossless_tolerance
        self.samples_x = deque(maxlen=window)
        self.samples_y = deque(maxlen=window)
        self.fitted = False
        self.last_retained_variance = float("nan")
        self.last_refit_status = "not_fitted"
        self.active_rank = 0
        self.observed_samples = 0
        self.observed_source_events = 0
        self.full_fallback_samples = 0
        self.full_fallback_source_events = 0
        self.latent_samples = 0
        self.latent_scalar_values = 0
        self.register_buffer("source_mean", torch.zeros(source_dim))
        self.register_buffer("target_mean", torch.zeros(target_dim))
        self.register_buffer("projection", torch.zeros(source_dim, rank))
        self.register_buffer("decoder", torch.zeros(rank, target_dim))

    def observe(self, source, target_current):
        """Store detached full-link samples for the next causal RRR refit."""
        # STDP later mutates its working tensors in-place to obtain a local
        # weight update.  Keep immutable samples of the analog full-link
        # current rather than accidentally fitting to those later spikes.
        self.samples_x.append(source.detach().clone())
        self.samples_y.append(target_current.detach().clone())
        self.observed_samples += 1
        self.observed_source_events += int(torch.count_nonzero(source).item())

    def forward(self, source, fallback_current):
        if not self.fitted:
            self.full_fallback_samples += 1
            self.full_fallback_source_events += int(torch.count_nonzero(source).item())
            return fallback_current
        self.latent_samples += 1
        self.latent_scalar_values += self.active_rank
        latent = (source - self.source_mean) @ self.projection
        return latent @ self.decoder + self.target_mean

    @torch.no_grad()
    def refit(self):
        """Fit the rank-k RRR predictor to the most recent full-link samples."""
        if len(self.samples_x) < 2:
            return False

        x = torch.stack(tuple(self.samples_x))
        y = torch.stack(tuple(self.samples_y))
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            return False
        source_mean = x.mean(dim=0)
        target_mean = y.mean(dim=0)
        centered_x = x - source_mean
        centered_y = y - target_mean
        total_variance = centered_y.square().sum()
        if total_variance <= 1e-12:
            # A short internal-step window can contain only a single state.
            # Keeping full communication is safer than replacing a link that
            # may later become dynamic with its momentary mean current.
            self.last_refit_status = "constant_target_fallback"
            self.fitted = False
            self.active_rank = 0
            return False
        source_rank = int(torch.linalg.matrix_rank(centered_x).item())
        if source_rank == 0:
            self.last_refit_status = "constant_source_fallback"
            self.fitted = False
            self.active_rank = 0
            return False

        # RRR is OLS followed by projection onto target-predictive directions.
        ols = torch.linalg.pinv(centered_x) @ centered_y
        predicted_y = centered_x @ ols
        _, singular_values, right = torch.linalg.svd(predicted_y, full_matrices=False)
        active_rank = min(self.rank, source_rank, singular_values.numel())
        target_vectors = right[:active_rank].transpose(0, 1)

        self.source_mean.copy_(source_mean)
        self.target_mean.copy_(target_mean)
        self.projection.zero_()
        self.decoder.zero_()
        self.active_rank = active_rank
        self.projection[:, :active_rank].copy_(ols @ target_vectors)
        self.decoder[:active_rank].copy_(target_vectors.transpose(0, 1))

        residual = centered_y - predicted_y @ (target_vectors @ target_vectors.transpose(0, 1))
        retained_variance = 1 - residual.square().sum() / total_variance
        if not torch.isfinite(retained_variance):
            return False
        if retained_variance < 1.0 - self.fit_lossless_tolerance:
            self.fitted = False
            self.active_rank = 0
            self.last_retained_variance = float(retained_variance.item())
            self.last_refit_status = "lossy_window_fallback"
            return False
        self.last_retained_variance = float(retained_variance.item())
        self.fitted = True
        self.last_refit_status = "fitted"
        return True

    def transport_summary(self):
        """Expose activity and continuous-value counts without calling them AER."""
        return {
            "observed_samples": self.observed_samples,
            "observed_source_logical_events": self.observed_source_events,
            "full_fallback_samples": self.full_fallback_samples,
            "full_fallback_source_logical_events": self.full_fallback_source_events,
            "latent_samples": self.latent_samples,
            "latent_active_rank": self.active_rank,
            "continuous_latent_scalar_values": self.latent_scalar_values,
            "last_refit_status": self.last_refit_status,
        }
