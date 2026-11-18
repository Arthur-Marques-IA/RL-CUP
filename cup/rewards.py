"""Potential-based shaping (Ng, Harada & Russell 1999), batched on the device.

    F(s, s') = gamma * Phi(s') - Phi(s),   Phi(s) = alpha * ball_x + beta * possession(s)

- ball_x is in the left team's frame (we always learn from the left side), in [-1, 1].
- possession is +1 / -1 by the LAST team that owned the ball (a loose ball during a pass or a
  shot keeps the previous owner), 0 before anyone touched it.
- The match end is terminal: Phi(terminal) = 0. With auto-reset, the observation after `done`
  belongs to a new match, so Phi and the last-touch memory restart there.
- gamma here MUST equal the PPO gamma, or the invariance guarantee does not hold.
"""
from __future__ import annotations

import torch


class PotentialShaping:
    def __init__(self, num_envs: int, alpha: float, beta: float, gamma: float, device):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.last_touch = torch.zeros(num_envs, device=device)   # +1 ours, -1 theirs, 0 none
        self.phi = torch.zeros(num_envs, device=device)

    def _update_touch(self, raw_left):
        owned = raw_left["ball_owned_team"][:, 0].long()           # left view of slot 0
        self.last_touch = torch.where(owned == 0, torch.ones_like(self.last_touch),
                                      torch.where(owned == 1, -torch.ones_like(self.last_touch),
                                                  self.last_touch))

    def _phi(self, raw_left):
        ball_x = raw_left["ball"][:, 0, 0].float()
        return self.alpha * ball_x + self.beta * self.last_touch

    def reset(self, raw_left):
        self.last_touch.zero_()
        self._update_touch(raw_left)
        self.phi = self._phi(raw_left)

    def step(self, raw_left_next, done: torch.Tensor) -> torch.Tensor:
        """Shaping reward for the transition into raw_left_next. done[e] = the match ended on this
        step (raw_left_next[e] then already shows the next match)."""
        d = done.bool()
        # terminal: F = gamma * 0 - Phi(s)
        self._update_touch(raw_left_next)
        phi_next = self._phi(raw_left_next)
        shaped = torch.where(d, -self.phi, self.gamma * phi_next - self.phi)
        # new match: restart last touch and Phi from its first observation (no host sync)
        self._update_touch_masked(raw_left_next, d)
        phi_next = self._phi(raw_left_next)
        self.phi = phi_next
        return shaped

    def _update_touch_masked(self, raw_left, mask):
        owned = raw_left["ball_owned_team"][:, 0].long()
        new = torch.where(owned == 0, 1.0, torch.where(owned == 1, -1.0, 0.0)).to(self.last_touch)
        self.last_touch = torch.where(mask, new, self.last_touch)
