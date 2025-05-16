from typing import Callable
import math
import numpy as np

import torch
from torch.nn import Module
from hydra.utils import instantiate
from omegaconf import DictConfig

from sde_sampler.distr.base import Distribution
from sde_sampler.eq.sdes import OU, ControlledSDE
from sde_sampler.solver.base import Trainable
from sde_sampler.utils.common import Results, pad_dims_like
from sde_sampler.losses.consistency import ConsistencyDistillation
from sde_sampler.eq.integrator import EulerIntegrator

import pdb

class ConsistencySolver(Trainable):
    save_attrs = Trainable.save_attrs + [
        "generative_ctrl", "consistency_model", "loss"
    ] 

    def __init__(self, cfg: DictConfig):
        # Train
        self.train_timesteps: Callable = instantiate(cfg.train_timesteps)
        self.train_batch_size: int = cfg.train_batch_size
        self.clip_target: float | None = cfg.get("clip_target")
        self.cm_train_timesteps: int = cfg.get("cm_train_timesteps", 18)
        self.cm_eval_timesteps: int = cfg.get("cm_eval_timesteps", 18)
        # Eval
        self.eval_timesteps: Callable = instantiate(cfg.eval_timesteps)
        self.eval_batch_size: int = cfg.eval_batch_size
        self.eval_integrator = EulerIntegrator()

        self.batch_t: bool = cfg.get("batch_t", False)

        super().__init__(cfg=cfg)

    def setup_models(self):
        self.prior: Distribution = instantiate(self.cfg.prior)
        # Generative and inference SDEs
        self.sde: OU = instantiate(self.cfg.get("sde"))
        self.inference_ctrl = self.cfg.get("inference_ctrl")
        self.inference_sde: OU = instantiate(
            self.cfg.sde,
            generative=False,
        )
        if self.inference_ctrl is not None:
            self.inference_ctrl: Module = instantiate(
                self.cfg.inference_ctrl,
                sde=self.sde,
                prior_score=self.prior.score,
                target_score=self.target.score,
            )
            self.inference_sde = ControlledSDE(
                sde=self.inference_sde, ctrl=self.inference_ctrl
            )

        # Load pre-trained generative control model
        self.generative_ctrl: Module = instantiate(
            self.cfg.generative_ctrl,
            sde=self.sde,
            prior_score=self.prior.score,
            target_score=self.target.score,
        )
        self._load_pretrained_generative_ctrl()
        self.generative_ctrl.eval()
        self.generative_ctrl.requires_grad_(False)

        # Consistency model
        self.consistency_model: Module = instantiate(
            self.cfg.consistency_model,
            sigma_min=self.cfg.train_timesteps.start,
        )
        self._init_consistency_weights()
        self.consistency_model.train()
        self.consistency_model.requires_grad_(True)

        # Loss
        self.loss: ConsistencyDistillation = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            consistency_model=self.consistency_model,
            sde=self.sde,
        ) 

        # Steps
        self.ts = self.train_timesteps()
        self.train_ts_idx = np.linspace(
            0, len(self.ts)-1, self.cm_train_timesteps, endpoint=True
        ).round().astype(int)
        self.train_ts = self.ts[self.train_ts_idx]
        self.eval_ts_idx = np.linspace(
            0, len(self.train_ts)-1, self.cm_eval_timesteps, endpoint=True
        ).round().astype(int)
        self.eval_ts = self.train_ts[self.eval_ts_idx]

    def _load_pretrained_generative_ctrl(self):
        pretrained_path = self.cfg.generative_ctrl.get("pretrained_path", None)
        checkpoint = torch.load(pretrained_path)
        self.generative_ctrl.load_state_dict(checkpoint['generative_ctrl'])
            
    def _init_consistency_weights(self):
        self.consistency_model.base_model.load_state_dict(
            self.generative_ctrl.base_model.state_dict()
        )

    def get_numsteps(  # Not used
            self, 
            curriculum: str = "CT+", 
            s0: int = 10,
            s1: int = 1280
        ) -> int:
        k = self.n_steps
        K = self.train_steps
        if curriculum == "CT+":
            K_prime = math.floor(
                K / (math.log2(math.floor(s1 / s0)) + 1)
            )
            N = s0 + math.pow(2, math.floor(k / K_prime))
            return int(min(N, s1) + 1)
        elif curriculum == "square":
            return min(k * k, s1) + 1
        elif curriculum == "constant":
            return s0 + 1
        else:
            raise ValueError(f"Unknown curriculum {curriculum}")

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

    def compute_loss(self) -> tuple[torch.Tensor, dict]:
        batch_size = self.train_batch_size
        x = self.prior.sample((batch_size,))
        cm_indices = self.train_ts_idx
        
        if self.batch_t:
            # pdb.set_trace()
            # Fix t for the whole batch; efficient parallelization
            i = torch.randint(0, len(cm_indices)-1, (1,)).item()
            i1 = cm_indices[i]
            i2 = cm_indices[i + 1]
            ts = self.ts[:i2 + 1].to(x.device)  # Stop before the end
            _, xs = self.simulate_ode(ts, x, True)
        else:
            ts = self.ts.to(x.device)
            _, xs = self.simulate_ode(ts, x, True)
            i = torch.randint(0, len(cm_indices)-1, (batch_size,))
            i1 = cm_indices[i]
            i2 = cm_indices[i + 1]

        t_max = self.ts[-1].to(x.device)

        xs_transposed = xs.transpose(0, 1)  
        # Now shape is [batch_size, time_steps, feature_dim]
        x1 = xs_transposed[torch.arange(batch_size), i1]
        x2 = xs_transposed[torch.arange(batch_size), i2]

        ts1 = pad_dims_like(ts[i1], x)
        ts2 = pad_dims_like(ts[i2], x)
        t_max = pad_dims_like(t_max, x)

        return self.loss.compute_loss(t_max-ts1, t_max-ts2, x1, x2)

        # ts = self.ts.to(x.device)
        # train_ts_idx = self.train_ts_idx
        # return self.loss(ts, x, train_ts_idx)

    def compute_results(self, steps: int = None) -> Results:
        x = self.prior.sample((self.eval_batch_size,))
        if steps is None:
            eval_ts = self.eval_ts.to(x.device)
        else: 
            cm_indices = np.linspace(
                0, len(self.train_ts)-1, steps, endpoint=True
            ).round().astype(int)
            eval_ts = self.train_ts[cm_indices].to(x.device)
        
        return self._compute_results(eval_ts, x)
    
    def _compute_results(self, ts: torch.Tensor, x: torch.Tensor) -> Results:
        return self.loss.eval(ts, x, self.inference_sde)
