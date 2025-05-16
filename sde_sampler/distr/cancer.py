# sde_sampler/cancer.py

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .base import Distribution 

class BreastCancer(Distribution):
    def __init__(
        self,
        dim=31,
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
        data_dir = Path(__file__).parents[2] / "data"
        arr = np.loadtxt(data_dir / "breast_cancer.data")
        labels = arr[:, 1].astype(np.int64)
        X = arr[:, 2:]
        # standardize to unit std (add small eps for safety)
        eps = 1e-8
        X = X / (X.std(axis=0, keepdims=True) + eps)
        # prepend bias column
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X = np.concatenate([ones, X], axis=1)  # now shape (N, 31)

        # register as buffers so they move with .to(device)
        self.register_buffer("data", torch.from_numpy(X).float())
        self.register_buffer("labels", torch.from_numpy(labels).long().unsqueeze(1))

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
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
        arr = np.load(data_dir / "breastcancer_gt_with_lns_10k.npz")
        gt = torch.from_numpy(arr["groundtruth"]).float().to(self.data.device)
        M = gt.shape[0]

        num = int(math.prod(shape))
        idx = torch.randint(0, M, (num,), device=gt.device)
        draws = gt[idx]                          # (num, 31)
        return draws.view(*shape, self.dim)