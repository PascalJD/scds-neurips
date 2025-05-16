import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import Distribution


class Ionosphere(Distribution):
    def __init__(
        self,
        dim=35, 
        log_norm_const=None,
        domain=None,
        n_reference_samples=None,
    ):
        data_dir = Path(__file__).parents[2] / "data"
        X_np, Y_np = pickle.load(open(data_dir / "ionosphere_full.pkl", "rb"))

        # labels ∈ {‑1,+1}  →  {0,1}
        Y_np = ((Y_np + 1) // 2).astype(np.int64)

        # standardise columns and add bias
        std = X_np.std(axis=0, keepdims=True)
        std[std == 0.0] = 1.0 # avoid divide‑by‑zero
        X_np = (X_np - X_np.mean(axis=0, keepdims=True)) / std
        X_np = np.concatenate([np.ones((len(X_np), 1)), X_np], axis=1)  # bias term

        if dim is None:
            dim = X_np.shape[1]

        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            domain=domain,
            n_reference_samples=n_reference_samples,
        )

        # register tensors
        self.register_buffer("data", torch.from_numpy(X_np).float())        # (N, dim)
        self.register_buffer("labels", torch.from_numpy(Y_np).unsqueeze(1)) # (N, 1)


    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        flat = x.reshape(B, self.dim)                # (B, dim)

        # logistic regression log‑likelihood
        feats = -(flat @ self.data.t())              # (B, N)
        ll_pt = torch.where(
            self.labels.t() == 1,
            F.logsigmoid(feats),
            F.logsigmoid(feats) - feats
        )                                            # (B, N)
        ll = ll_pt.sum(dim=1, keepdim=True)          # (B, 1)

        return ll                                    # (B, 1)