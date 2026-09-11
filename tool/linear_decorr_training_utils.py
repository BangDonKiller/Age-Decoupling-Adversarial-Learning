import math

import numpy as np
import torch


class WarmupExpDecayLR(torch.optim.lr_scheduler._LRScheduler):
    """Step-based LR: lr(t) = g(t) * h(t), where g is linear warmup and h is exponential decay."""

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int,
        eta_0: float,
        eta_T: float,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps 必須 > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps 必須 >= 0")
        if eta_0 <= 0 or eta_T <= 0:
            raise ValueError("eta_0 與 eta_T 必須 > 0")

        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.eta_0 = float(eta_0)
        self.eta_T = float(eta_T)
        super().__init__(optimizer, last_epoch=last_epoch)

    def _lr_at(self, step: int) -> float:
        t = max(0, min(int(step), self.total_steps))

        if self.warmup_steps > 0 and t < self.warmup_steps:
            g_t = float(t) / float(self.warmup_steps)
        else:
            g_t = 1.0

        h_t = self.eta_0 * math.exp((float(t) / float(self.total_steps)) * math.log(self.eta_T / self.eta_0))
        return g_t * h_t

    def get_lr(self):
        lr = self._lr_at(self.last_epoch)
        return [lr for _ in self.optimizer.param_groups]


class MarginScheduler:
    """Step-based ArcFace margin scheduler with 3 stages."""

    def __init__(self, t1: int, t2: int, target_margin: float):
        if t1 < 0 or t2 < 0:
            raise ValueError("t1 與 t2 必須 >= 0")
        if t2 < t1:
            raise ValueError("t2 必須 >= t1")
        if target_margin < 0:
            raise ValueError("target_margin 必須 >= 0")

        self.t1 = int(t1)
        self.t2 = int(t2)
        self.target_margin = float(target_margin)
        self.current_step = -1

    def get_margin(self, step: int) -> float:
        t = int(step)
        if t < self.t1:
            return 0.0
        if t >= self.t2:
            return self.target_margin
        if self.t2 == self.t1:
            return self.target_margin

        return self.target_margin * (float(t - self.t1) / float(self.t2 - self.t1))

    def step(self) -> float:
        self.current_step += 1
        return self.get_margin(self.current_step)


class RunningAgeCorrelation:
    """用串流統計計算前 age_dim 個 age neuron 與其他 neuron 的 Pearson 相關係數。"""

    def __init__(self, latent_dim: int, age_dim: int = 1):
        if age_dim < 1:
            raise ValueError("age_dim 必須 >= 1")
        if latent_dim <= age_dim:
            raise ValueError("latent_dim 必須 > age_dim")

        self.age_dim = age_dim
        other_dim = latent_dim - age_dim
        self.n = 0
        self.sum_x = np.zeros(age_dim, dtype=np.float64)
        self.sum_x2 = np.zeros(age_dim, dtype=np.float64)
        self.sum_y = np.zeros(other_dim, dtype=np.float64)
        self.sum_y2 = np.zeros(other_dim, dtype=np.float64)
        self.sum_xy = np.zeros((age_dim, other_dim), dtype=np.float64)

    def update(self, z: torch.Tensor):
        if z is None or z.numel() == 0:
            return

        z_np = z.detach().cpu().numpy().astype(np.float64, copy=False)
        x = z_np[:, : self.age_dim]
        y = z_np[:, self.age_dim :]

        self.n += z_np.shape[0]
        self.sum_x += x.sum(axis=0)
        self.sum_x2 += (x * x).sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.sum_y2 += (y * y).sum(axis=0)
        self.sum_xy += x.T @ y

    def correlations(self, eps: float = 1e-12):
        if self.n < 2:
            return None

        n = float(self.n)
        ex = self.sum_x / n
        ey = self.sum_y / n
        ex2 = self.sum_x2 / n
        ey2 = self.sum_y2 / n
        exy = self.sum_xy / n

        cov = exy - ex[:, None] * ey[None, :]
        var_x = np.maximum(ex2 - ex * ex, eps)
        var_y = np.maximum(ey2 - ey * ey, eps)
        corr_other = cov / np.sqrt(var_x[:, None] * var_y[None, :])
        corr_other = np.clip(corr_other, -1.0, 1.0)

        return np.concatenate([np.eye(self.age_dim, dtype=np.float64), corr_other], axis=1)
