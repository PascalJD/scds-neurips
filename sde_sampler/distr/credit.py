# sde_sampler/credit.py

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import Distribution 

class GermanCredit(Distribution):
    def __init__(
        self,
        dim=25,
        log_norm_const=None,
        domain=None,
        n_reference_samples=None,
    ):
        super().__init__(
            dim=25,
            log_norm_const=log_norm_const,
            domain=domain,
            n_reference_samples=n_reference_samples,
        )
        data_dir = Path(__file__).parents[2] / "data"
        # --- load and preprocess features/labels ---
        arr = np.loadtxt(data_dir / "german.data-numeric")
        X = arr[:, :-1]
        # standardize each feature to unit std
        X = X / (X.std(axis=0, keepdims=True) + 1e-8)
        # add bias column
        X = np.concatenate([np.ones((len(X), 1)), X], axis=1)
        self.register_buffer("data", torch.from_numpy(X).float())    # (N, dim)
        labels = (arr[:, -1].astype(int) - 1).astype(np.int64)      # from {1,2}→{0,1}
        self.register_buffer("labels", torch.from_numpy(labels).unsqueeze(1))  # (N,1)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim)
        B = x.shape[0]
        flat = x.reshape(B, self.dim)  # (B, dim)
        feats = -(flat @ self.data.t())  # (B, N)
        ll_pt = torch.where(
            self.labels.t() == 1,
            F.logsigmoid(feats),
            F.logsigmoid(feats) - feats
        )  # (B, N)
        ll = ll_pt.sum(dim=1, keepdim=True)  # (B, 1)
        return ll

    def sample(self, shape: tuple[int, ...]) -> torch.Tensor:
        data_dir = Path(__file__).parents[2] / "data"
        arr = np.load(data_dir / "german_credit10k.npy") 
        tensor = torch.from_numpy(arr).float().to(self.data.device)
        n = tensor.shape[0]
        num = int(math.prod(shape))
        idx = torch.randint(0, n, (num,), device=tensor.device)
        out = tensor[idx]                                 # (num, dim)
        return out.view(*shape, self.dim)
