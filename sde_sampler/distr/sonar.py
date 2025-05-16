from __future__ import annotations
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as pydist
from jax._src.flatten_util import ravel_pytree
import pickle

from .base import Distribution
from inference_gym.using_jax import project_path  # or your own path_utils


def _standardize_and_pad(X: np.ndarray) -> np.ndarray:
    """Center–scale each column to unit std, then prepend a constant-1 column."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    # pad a column of ones
    return np.hstack([np.ones((X.shape[0], 1)), X])


def _load_sonar_model():
    """Recreate the JAX/Numpyro Bernoulli‐logistic model for Sonar."""
    # load data
    p = project_path("targets/data/sonar_full.pkl")
    with open(p, "rb") as f:
        X, Y = pickle.load(f)
    # map from ±1→{0,1}
    Y = ((np.array(Y) + 1) // 2).astype(np.float32)
    X = _standardize_and_pad(np.array(X, np.float32))

    n_data, dim = X.shape

    def model(Y_obs):
        # global weight prior
        w = numpyro.sample("weights", pydist.Normal(jnp.zeros(dim), jnp.ones(dim)))
        logits = jnp.dot(X, w)
        with numpyro.plate("data", n_data):
            numpyro.sample("obs", pydist.BernoulliLogits(logits), obs=Y_obs)

    return model, (jnp.array(Y),), dim  # return dim so wrapper can override default


class Sonar(Distribution):
    def __init__(
        self,
        dim=61,                    
        log_norm_const=None,
        domain=None,
        n_reference_samples=None,
    ):
        # temporarily pass dim; we'll reset it below
        super().__init__(dim=dim,
                         log_norm_const=log_norm_const,
                         domain=domain,
                         n_reference_samples=n_reference_samples)

        # build the JAX model & get true dim
        rng = jax.random.PRNGKey(0)
        model, model_args, true_dim = _load_sonar_model()
        # initialize with numpyro
        init_info = numpyro.infer.util.initialize_model(rng, model, model_args=model_args)
        params_flat, potential_fn, constrain_fn, _ = init_info
        # flatten the params
        flat_params, unflatten = ravel_pytree(params_flat[0])
        # define a pure‐JAX log‐prob
        def _jax_logprob(z):
            # unconstrained→constrained
            θ = constrain_fn(z)
            # negative potential = unnorm log p
            return -1.0 * potential_fn(θ)
        # JIT for speed
        self._jax_logprob = jax.jit(_jax_logprob)
        # update our dim to match data
        self.dim = true_dim

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().numpy().astype(np.float32)
        logp = np.asarray(self._jax_logprob(arr))  # (B,)
        return torch.from_numpy(logp).to(x.device).unsqueeze(1)