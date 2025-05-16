from typing import Callable

import torch
import torch.nn.functional as F
from sde_sampler.eq.sdes import OU, TorchSDE
from sde_sampler.solver.base import Solver
from sde_sampler.utils.common import Results, pad_dims_like

class ConsistencyTraining:
    def __init__(
        self,
        consistency_model: Callable,
    ) -> None:
        self.consistency_model = consistency_model

    def compute_loss(
        self, 
        t1: torch.Tensor, 
        t2: torch.Tensor, 
        x1: torch.Tensor, 
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # More noise in inference process (from data)
        denoise_student = self.consistency_model(t2, x2)  
        with torch.no_grad():
            denoise_target = self.consistency_model(t1, x1)  # Less noise
        loss = F.mse_loss(denoise_student, denoise_target, reduction='mean')
        return loss, {}, denoise_student

    def __call__(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,  # Initial sample in data space [batch_size, dim]
    ) -> tuple[torch.Tensor, dict]:
        # Sample t
        batch_size = x.shape[0]
        i = torch.randint(0, len(ts)-1, (batch_size,))
        t1 = ts[i]  # [batch_size]
        t2 = ts[i + 1]  # More noise 
        # [batch_size, 1]
        t1 = t1.unsqueeze(-1)
        t2 = t2.unsqueeze(-1)

        # Compute x1, x2 according to (hardcoded) Karras inference SDE
        z = torch.randn_like(x)
        x1 = x + t1 * z
        x2 = x + t2 * z

        return self.compute_loss(t1, t2, x1, x2)

    @torch.no_grad()
    def singlestep_sampling(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return self.consistency_model(ts[-1], x)

    @torch.no_grad()
    def multistep_sampling(
        self, 
        ts: torch.Tensor,  # t_0 < t_1 < ... < t_n
        xT: torch.Tensor  # Initial sample in prior space [batch_size, dim]
    ) -> torch.Tensor:
        xt = xT
        for i in reversed(range(1, len(ts))):
            x = self.consistency_model(ts[i], xt)
            xt = x + ts[i-1] * torch.randn_like(x)  # Karras SDE 
        return x

    def eval(
        self,
        ts: torch.Tensor,
        xT: torch.Tensor,
    ) -> Results:
        x = self.multistep_sampling(ts, xT)
        return Results(samples=x, weights=None)


class ConsistencyDistillation:
    def __init__(
        self,
        generative_ctrl: Callable,
        consistency_model: Callable,
        sde: OU,
        method: str = "l2",
    ) -> None:
        self.generative_ctrl = generative_ctrl
        self.consistency_model = consistency_model
        self.sde = sde
        self.method = method

    @torch.no_grad()
    def simulate_ode(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        return_traj: bool = False,
    ): # -> tuple[torch.Tensor, torch.Tensor | None]:   
        xs = [x] if return_traj else None
        for s, t in zip(ts[:-1], ts[1:]):
            dt = t - s
            # Euler-Maruyama PF ODE
            x = x + ((self.sde.drift(s, x)
                      + 0.5 * self.sde.diff(s, x) 
                      * self.generative_ctrl(s, x)) * dt)
            if return_traj:
                xs.append(x)

        if return_traj:
            xs = torch.stack(xs)
        return x, xs
    
    @torch.no_grad()
    def singlestep_sampling(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return self.consistency_model(ts[-1], x)

    @torch.no_grad()
    def multistep_sampling(
        self, ts: torch.Tensor, xt: torch.Tensor, inference_sde: TorchSDE
    ) -> torch.Tensor:
        for i in reversed(range(1, len(ts))):
            x = self.consistency_model(ts[i], xt)
            loc, var = inference_sde.marginal_params(ts[i-1], x)
            z = torch.randn_like(x)
            xt = loc + var.sqrt() * z
        return x

    def compute_loss(
        self, 
        t1: torch.Tensor, 
        t2: torch.Tensor, 
        x1: torch.Tensor, 
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # More noise in backward process
        denoise_student = self.consistency_model(t1, x1)  
        with torch.no_grad():
            denoise_target = self.consistency_model(t2, x2)  # Less noise
        loss = F.mse_loss(denoise_student, denoise_target, reduction='mean')
        return loss, {}

    def __call__(
        self, ts: torch.Tensor, x: torch.Tensor, cm_indices: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        _, xs = self.simulate_ode(ts, x, True)

        batch_size = x.shape[0]
        i = torch.randint(0, len(cm_indices)-1, (batch_size,))
        i1 = cm_indices[i]
        i2 = cm_indices[i + 1]
        t_max = ts[-1]

        xs_transposed = xs.transpose(0, 1)  
        # Now shape is [batch_size, time_steps, feature_dim]
        x1 = xs_transposed[torch.arange(batch_size), i1]
        x2 = xs_transposed[torch.arange(batch_size), i2]

        ts1 = pad_dims_like(ts[i1], x)
        ts2 = pad_dims_like(ts[i2], x)
        t_max = pad_dims_like(t_max, x)

        return self.compute_loss(t_max-ts1, t_max-ts2, x1, x2)  # Reverse time
    
    def eval(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        inference_sde: TorchSDE,
    ) -> Results:
        print(f"Evaluating consistency model with timesteps {ts}")
        samples = self.multistep_sampling(ts, x, inference_sde)
        return Results(samples=samples, weights=None)
