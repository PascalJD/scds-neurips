from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# JAX / inference‑gym imports are only needed inside this file
# (keeps rest of repo Torch‑only).
import jax
import jax.numpy as jnp
import inference_gym.using_jax as gym  # pip install inference-gym

from .base import Distribution


class BrownianMotion(Distribution):
    def __init__(
        self,
        dim=32,
        log_norm_const=None,
        domain=None,
        n_reference_samples=None,
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            domain=domain,
            n_reference_samples=n_reference_samples,
        )

        # Build the JAX target (exactly like baseline).
        base = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations()
        self._jax_target = gym.targets.VectorModel(
            base,
            flatten_sample_transformations=True,
        )
        self._bij = self._jax_target.default_event_space_bijector
        def _jax_logprob(z):
            x = self._bij(z)
            return (
                self._jax_target.unnormalized_log_prob(x)
                + self._bij.forward_log_det_jacobian(z, event_ndims=1)
            )
        self._jax_logprob = jax.jit(_jax_logprob)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z_np = x.detach().cpu().numpy().astype(np.float32)  # shape (B, D)
        logp_np = np.asarray(self._jax_logprob(z_np))  # (B,)
        return torch.from_numpy(logp_np).to(x.device).unsqueeze(1)  # (B,1)