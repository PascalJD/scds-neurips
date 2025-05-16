from __future__ import annotations

import time
from typing import Callable

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module

from sde_sampler.distr.base import Distribution, sample_uniform
from sde_sampler.distr.delta import Delta
from sde_sampler.distr.gauss import Gauss
from sde_sampler.eq.integrator import EulerIntegrator
from sde_sampler.eq.sdes import OU, ControlledSDE
from sde_sampler.eval.plots import get_plots
from sde_sampler.losses.oc import BaseOCLoss, TimeReversalLoss
from sde_sampler.solver.base import Trainable
from sde_sampler.utils.common import Results, clip_and_log


class TrainableDiff(Trainable):
    save_attrs = Trainable.save_attrs + ["generative_ctrl"]

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)

        # Train
        self.train_batch_size: int = self.cfg.train_batch_size
        self.train_timesteps: Callable = instantiate(self.cfg.train_timesteps)
        self.clip_target: float | None = self.cfg.get("clip_target")

        # Eval
        self.eval_timesteps: Callable = instantiate(self.cfg.eval_timesteps)
        self.eval_batch_size: int = self.cfg.eval_batch_size
        self.eval_integrator = EulerIntegrator()

    def setup_models(self):
        self.prior: Distribution = instantiate(self.cfg.prior)
        self.sde: OU = instantiate(self.cfg.get("sde"))
        self.generative_ctrl: Module = instantiate(
            self.cfg.generative_ctrl,
            sde=self.sde,
            prior_score=self.prior.score,
            target_score=self.target.score,
        )

    def clipped_target_unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        output = clip_and_log(
            self.target.unnorm_log_prob(x),
            max_norm=self.clip_target,
            name="target",
        )
        return output

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        raise NotImplementedError

    def compute_loss(self) -> tuple[torch.Tensor, dict]:
        x = self.prior.sample((self.train_batch_size,))
        ts = self.train_timesteps(device=x.device)
        return self._compute_loss(ts, x)

    def compute_results(self) -> Results:
        # Sample trajectories
        x = self.prior.sample((self.eval_batch_size,))
        ts = self.eval_timesteps(device=x.device)

        results = self._compute_results(
            ts,
            x,
            compute_weights=True,
        )
        assert results.xs.shape == (len(ts), *results.samples.shape)

        # Sample w/o ito integral
        start_time = time.time()
        add_results = self._compute_results(
            ts,
            x,
            compute_weights=False,
            return_traj=False,
        )

        # Update results
        results.metrics["eval/sample_time"] = time.time() - start_time
        results.metrics.update(add_results.metrics)
        results.log_norm_const_preds.update(add_results.log_norm_const_preds)

        # Sample trajectories of inference proc
        if (
            self.plot_results
            and hasattr(self, "inference_sde")
            and hasattr(self.target, "sample")
        ):
            x_target = self.target.sample((self.eval_batch_size,))
            xs = self.eval_integrator.integrate(
                sde=self.inference_sde, ts=ts, x_init=x_target, timesteps=ts
            )
            plots = get_plots(
                distr=self.prior,
                samples=xs[-1],
                ts=ts,
                xs=xs,
                marginal_dims=self.eval_marginal_dims,
                domain=self.target.domain,
            )
            results.plots.update({f"{k}_inference": v for k, v in plots.items()})

        return results


class Bridge(TrainableDiff):
    save_attrs = TrainableDiff.save_attrs + ["inference_ctrl", "loss"]

    def setup_models(self):
        super().setup_models()
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
        elif not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")

        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            inference_ctrl=self.inference_ctrl,
            filter_samples=getattr(self.target, "filter", None),
        )

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            initial_log_prob=self.prior.log_prob,
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            initial_log_prob=self.prior.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class PIS(TrainableDiff):
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def setup_models(self):
        super().setup_models()
        if not isinstance(self.prior, Delta):
            raise ValueError("Can only be used with dirac delta prior.")
        self.reference_distr = self.sde.marginal_distr(
            t=self.sde.terminal_t, x_init=self.prior.loc
        )
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            filter_samples=getattr(self.target, "filter", None),
        )

        # Inference SDE
        inference_sde: OU = instantiate(
            self.cfg.sde,
            generative=False,
        )
        self.inference_sde = ControlledSDE(sde=inference_sde, ctrl=self.inference_ctrl)

    def inference_ctrl(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        reference_distr = self.sde.marginal_distr(t=t, x_init=self.prior.loc)
        return self.sde.diff(t, x) * reference_distr.score(x).clip(max=1e5)

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class DDS(TrainableDiff):
    # This implements the basic DDS algorithm
    # with the intended exponential integrator
    # https://arxiv.org/abs/2302.13834
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def setup_models(self):
        super().setup_models()
        if not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")

        # prior = reference_distr for terminal loss
        self.reference_distr = self.prior
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            filter_samples=getattr(self.target, "filter", None),
        )

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class EulerDDS(TrainableDiff):
    # This implementation induces the same objectives in the DDS paper (https://arxiv.org/abs/2302.13834).
    # However, we do not use the exponential integrator and the same parametrization.
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def setup_models(self):
        super().setup_models()
        if not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")
        self.inference_sde = instantiate(self.cfg.sde, generative=False)
        self.reference_distr = self.sde.marginal_distr(
            self.sde.terminal_t, x_init=self.prior.loc, var_init=self.prior.scale**2
        )
        if not torch.allclose(
            self.reference_distr.loc, self.prior.loc
        ) and torch.allclose(self.reference_distr.scale, self.prior.scale):
            raise ValueError(
                "Make sure that the Gaussian is the invariant distribution of the SDE."
            )
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            reference_ctrl=self.reference_ctrl,
            filter_samples=getattr(self.target, "filter", None),
        )

    def reference_ctrl(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.sde.diff(t, x) * self.prior.score(x)

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class SubtrajBridge(Bridge):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg=cfg)
        if not hasattr(self.generative_ctrl, "unnorm_log_prob"):
            raise ValueError("Needs an unnormalized log density.")
        if not self.loss.method in ["lv", "lv_traj"]:
            raise ValueError("Can only be used with log-variance loss.")
        if self.target.domain is None:
            raise ValueError("Need a domain for sampling.")
        self.subtraj_prob = self.cfg.get("subtraj_prob", 0.5)
        self.fix_terminal = self.cfg.get("fix_terminal", False)
        self.subtraj_steps = self.cfg.get("subtraj_steps")
        if self.fix_terminal and self.subtraj_steps is not None:
            raise ValueError("Cannot fix subtrajectory steps with fixed terminal time.")
        self.lerp_domain = self.cfg.get("lerp_domain", True)

    def get_log_prob(self, t: torch.Tensor, detach=False) -> Callable:
        if torch.isclose(t, self.sde.terminal_t):
            return self.clipped_target_unnorm_log_prob
        if torch.isclose(t, torch.zeros_like(t)):
            return self.prior.log_prob

        def log_prob(x: torch.Tensor) -> torch.Tensor:
            with torch.set_grad_enabled(detach):
                output = self.generative_ctrl.unnorm_log_prob(t=t, x=x)
                if self.inference_ctrl is not None:
                    output += self.inference_ctrl.unnorm_log_prob(t=t, x=x)
                return output

        return log_prob

    def compute_loss(
        self,
    ) -> tuple[torch.Tensor, dict]:
        if torch.rand(1) > self.subtraj_prob:
            return super().compute_loss()

        # Timesteps
        ts = self.train_timesteps(device=self.target.domain.device)
        idx_init = torch.randint(0, len(ts) - 1, tuple())

        if self.fix_terminal:
            idx_end = len(ts) - 1
        elif self.subtraj_steps is not None:
            idx_end = torch.minimum(
                idx_init + self.subtraj_steps, torch.tensor(len(ts)) - 1
            )
        else:
            idx_end = torch.randint(idx_init + 1, len(ts), tuple())

        # Get initial points
        domain = self.target.domain
        if self.lerp_domain:
            domain = torch.lerp(
                self.prior.domain, domain, ts[idx_init] / self.sde.terminal_t
            )

        x = sample_uniform(domain=domain, batchsize=self.train_batch_size)

        # Simulate loss
        subts = ts[idx_init : idx_end + 1]
        initial_log_prob = self.get_log_prob(t=ts[idx_init], detach=True)
        target_unnorm_log_prob = self.get_log_prob(t=ts[idx_end], detach=False)
        loss, metrics = self.loss(
            ts, x, target_unnorm_log_prob, initial_log_prob=initial_log_prob
        )
        loss *= len(subts) / len(ts)
        return loss, metrics


class SCDS(TrainableDiff):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.curriculum_N = self.cfg.get("method_N", None)
        self.curriculum_d = self.cfg.get("method_d", None)
        self.train_steps = self.cfg.train_steps
        self.N = self.cfg.train_timesteps.steps
        self.min_N = self.cfg.get("min_steps", None)
        self.max_pow = int(np.log2(self.N))
        self.min_pow = None
        if self.min_N is not None:
            self.min_pow = int(np.log2())

    def setup_models(self):
        super().setup_models()
        self.inference_sde: OU = instantiate(
            self.cfg.sde,
            generative=False,
        )
        self.method = "lv" 
        self.sampling_loss = TimeReversalLoss(
            generative_ctrl=self.generative_ctrl,
            sde=self.sde,
            # method="lv",
            method=self.method,
            max_rnd= 1e8,
            filter_samples=getattr(self.target, "filter", None),
            is_step_conditioned=True
        )
    
    def consistency_loss(
        self, d: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor,
    ) -> torch.Tensor:
        d = d.unsqueeze(-1)  # Broadcast with x_t
        t = t.unsqueeze(-1)

        d_half = 0.5 * d
        t_half = t + d_half

        with torch.no_grad():
            x_half = x_t + (
                self.sde.drift(t, x_t) 
                + 0.5 * self.sde.diff(t, x_t) 
                * self.generative_ctrl(t, x_t, d_half)
            ) * d_half
            x_d = x_half + (
                self.sde.drift(t_half, x_half) 
                + 0.5 * self.sde.diff(t_half, x_half) 
                * self.generative_ctrl(t_half, x_half, d_half)
            ) * d_half
        
        x_d_prime= x_t + (
            self.sde.drift(t, x_t)
            + 0.5 * self.sde.diff(t, x_t)
            * self.generative_ctrl(t, x_t, d)
        ) * d

        return torch.mean((x_d - x_d_prime).pow(2)) 
    
    def get_step_sizes(self, device: torch.device) -> torch.Tensor:
        powers_of_2 = [
            2**i for i in range(int(np.log2(self.N)) + 1)
        ]
        step_sizes = powers_of_2
        step_sizes = torch.tensor(
            step_sizes, 
            device=device
        ).long()
        return step_sizes
    
    def step_curriculum(self, loss_history=None, patience=500, threshold=0.01):
        # Schedule for number of diffusion steps
        if self.curriculum_N is None:
            return self.N
        elif self.curriculum_N == "linear":
            frac = min(self.n_steps / self.train_steps, 1.0)
            cur_pow = self.max_pow - int(
                round(frac * (self.max_pow - self.min_pow))
            )
            return 2**cur_pow
        elif self.curriculum_N == "root":
            frac = min(self.n_steps / self.train_steps, 1.0)
            cur_pow = self.max_pow - int(
                round((frac**0.5) * (self.max_pow - self.min_pow))
            )
            return 2**cur_pow
        elif self.curriculum_N == "consistency":
            if loss_history is not None and len(loss_history) >= patience:
                recent_loss = np.mean(loss_history[-patience:])
                if recent_loss < threshold and self.N > 2**self.min_pow:
                    loss_history.clear()
                    return max(self.N // 2, 2**self.min_pow)
            return self.N
        elif self.curriculum_N == "random":
            rand_pow = torch.randint(
                low=self.min_pow, high=self.max_pow + 1, size=()
            ).item()
            return 2**rand_pow
        else:
            raise ValueError(f"Unknown N‑curriculum '{self.curriculum_N}'")

    def sample_d_and_t(
        self,
        step_sizes: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = self.train_batch_size
        num_sizes = len(step_sizes)

        if self.curriculum_d is None or self.curriculum_d == "uniform":
            d_indices = torch.randint(low=1, high=num_sizes, size=(B,), device=device)
        elif self.curriculum_d == "biaised":
            β = 1.5
            powers = torch.arange(1, num_sizes, device=device)
            log_p = -β * powers.float()
            probs = torch.softmax(log_p, dim=0)
            d_indices = torch.multinomial(probs, num_samples=B, replacement=True)
            d_indices += 1  # shift back to 1-indexed (since range starts at 1)
        else:
            raise ValueError(f"Unknown d curriculum: '{self.curriculum_d}'")

        d = step_sizes[d_indices]

        # Uniformly sample k such that t_idx = k * d < N
        k_max = (self.N // d).long()
        k = (torch.rand_like(d, dtype=torch.float) * k_max).long()
        t_idx = k * d

        return d, t_idx
    
    def take_step(self, loss, optim, metrics):
        loss_ok = (
            loss.isfinite() \
                if self.max_loss is None else loss.abs() <= self.max_loss
        )
        if self.max_grad is None:
            grad_ok = all(
                p.grad.isfinite().all()
                for p in self.trainable_parameters()
                if p.grad is not None
            )
        else:
            max_grad = self.grad_norm(norm_type=torch.inf)
            grad_ok = max_grad <= self.max_grad
            metrics["train/max_grad"] = max_grad.item()

        # Step optimizer, scheduler, and ema
        if loss_ok and grad_ok:
            # Clip grads
            if self.grad_clip is not None:
                metrics["train/grad_clip_norm"] = self.grad_clip(
                    self.trainable_parameters()
                ).item()

            optim.step()
            self.scheduler.step()
            if self.ema:
                self.ema.update()
                metrics["train/ema_decay"] = self.ema.get_current_decay()
            return True
        else:
            self.n_steps_skip += 1
            return False

    def step(self) -> dict[str, float]:
        start_t = time.time()

        self.N = self.step_curriculum()
        x = self.prior.sample((self.train_batch_size,))
        ts = self.train_timesteps(device=x.device, steps=self.N)

        # Sampling loss (TimeReversalLoss)
        self.optim.zero_grad()
        loss_tuple, xs = self.sampling_loss(
            ts, 
            x, 
            terminal_unnorm_log_prob=self.clipped_target_unnorm_log_prob, 
            initial_log_prob=self.prior.log_prob,
            return_traj=True
        )
        sampling_loss, metrics = loss_tuple
        if self.scale_loss is not None:
            sampling_loss = self.scale_loss * sampling_loss
        sampling_loss.backward()
        _ = self.take_step(sampling_loss, self.optim, metrics)

        # Curriculum and d,t ~ p(d, t | N)
        step_sizes = self.get_step_sizes(device=x.device)
        d, t_idx = self.sample_d_and_t(
            step_sizes, 
            x.device
        )
        dt = d / self.N  # This assumes terminal time is 1 
        batch_indices = torch.arange(self.train_batch_size, device=xs.device)
        x_t = xs[t_idx, batch_indices]
        t = ts[t_idx]
        
        # Self-consistency loss
        self.optim.zero_grad()
        consistency_loss = self.consistency_loss(dt, t, x_t)
        if self.scale_loss is not None:
            consistency_loss = self.scale_loss * consistency_loss
        consistency_loss.backward()
        _ = self.take_step(consistency_loss, self.optim, metrics)

        time_step = time.time() - start_t
        metrics.update(
            {
                "train/diffusion_steps": self.N,
                "train/unique_step_sizes": len(step_sizes),
                "train/avg_step_size": d.float().mean().item(),
                "train/max_step_size": d.max().item(),
                "train/min_step_size": d.min().item(),
                "train/time_per_step": time_step,
                "train/sampling_loss": sampling_loss.item(),
                "train/sc_loss": consistency_loss.item(),
                "train/skipped_steps": self.n_steps_skip,
                "train/no_grad": sum(
                    p.grad is None for p in self.trainable_parameters()
                ),
            }
        )
        self.n_steps += 1
        return metrics

    def compute_results(self) -> Results:
        # Sample trajectories
        x = self.prior.sample((self.eval_batch_size,))
        ts = self.eval_timesteps(device=x.device)

        # Sample w/o ito integral
        start_time = time.time()
        results = self.sampling_loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.prior.log_prob,
            compute_weights=False,
            return_traj=False,
            ode=True,
        )

        results.metrics["eval/sample_time"] = time.time() - start_time

        # Sample trajectories of inference proc
        if (
            self.plot_results
            and hasattr(self, "inference_sde")
            and hasattr(self.target, "sample")
        ):
            x_target = self.target.sample((self.eval_batch_size,))
            xs = self.eval_integrator.integrate(
                sde=self.inference_sde, ts=ts, x_init=x_target, timesteps=ts
            )
            plots = get_plots(
                distr=self.prior,
                samples=xs[-1],
                ts=ts,
                xs=xs,
                marginal_dims=self.eval_marginal_dims,
                domain=self.target.domain,
            )
            results.plots.update({f"{k}_inference": v for k, v in plots.items()})

        return results