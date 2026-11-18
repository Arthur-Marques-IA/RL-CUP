"""Actor-critic MLP shared by every controlled player, and its numpy export."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cup.features import FEATURE_DIM

N_ACTIONS = 19


def _layer(i, o, gain=math.sqrt(2)):
    lin = nn.Linear(i, o)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.zeros_(lin.bias)
    return lin


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (tanh). Observation normalisation stats are buffers,
    so they travel with the weights (the classic "works in training, garbage in the
    tournament" bug otherwise)."""

    def __init__(self, obs_dim: int = FEATURE_DIM, hidden: int = 256, n_actions: int = N_ACTIONS):
        super().__init__()
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.actor = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                   _layer(hidden, n_actions, gain=0.01))
        self.critic = nn.Sequential(_layer(obs_dim, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                    _layer(hidden, 1, gain=1.0))

    def norm(self, x):
        return (x - self.obs_mean) / self.obs_std

    def logits(self, x):
        return self.actor(self.norm(x))

    def value(self, x):
        return self.critic(self.norm(x)).squeeze(-1)

    @torch.no_grad()
    def set_obs_stats(self, x: torch.Tensor, min_std: float = 0.1):
        """Freeze normalisation stats from data. The std floor matters: rare one-hots (penalty mode)
        or features that barely move in the sample (steps_left near kick-off) would otherwise be
        blown up to huge values later in a match."""
        self.obs_mean.copy_(x.mean(0))
        self.obs_std.copy_(x.std(0).clamp_min(min_std))

    def export_npz(self, path: str | Path) -> None:
        """Actor weights + normalisation stats in the format submission_template/agent.py reads."""
        linears = [m for m in self.actor if isinstance(m, nn.Linear)]
        arrays = {"obs_mean": self.obs_mean.cpu().numpy(), "obs_std": self.obs_std.cpu().numpy()}
        for i, lin in enumerate(linears):
            arrays[f"w{i}"] = lin.weight.detach().cpu().numpy().astype(np.float32)
            arrays[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float32)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
