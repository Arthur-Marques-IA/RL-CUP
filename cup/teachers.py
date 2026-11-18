"""Teachers for imitation learning, batched over cuju's raw tensors.

Each teacher maps the raw observation dict of tensors ([E, A, ...]) to actions [E, A].

`ScriptedTemplate` is a torch port of t-zero's arena/template/agent.py (run at the ball,
carry it towards goal, shoot when close). Its numpy original is the fallback submission,
so student-vs-teacher behaviour can also be checked in the arena.
"""
from __future__ import annotations

import math

import torch

IDLE, RIGHT, SHOT = 0, 5, 12
# Eight movement actions, counter-clockwise from "right" (y grows downwards).
_DIRECTIONS = torch.tensor([5, 4, 3, 2, 1, 8, 7, 6])  # RIGHT, TOP_RIGHT, TOP, TOP_LEFT, LEFT, BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT


def move_towards(pos: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """pos, target [..., 2] -> direction action [...] (same rounding as the template)."""
    d = target - pos
    angle = torch.atan2(-d[..., 1], d[..., 0])
    idx = torch.round(angle / (math.pi / 4)).long() % 8
    return _DIRECTIONS.to(pos.device)[idx]


def _active_positions(raw):
    lt = raw["left_team"].float()                      # [E, A, P, 2]
    active = raw["active"].long().clamp_min(0)         # [E, A]; -1 = controls nobody (red card)
    idx = active[..., None, None].expand(*active.shape, 1, 2)
    return torch.gather(lt, 2, idx).squeeze(2), active


class ScriptedTemplate:
    name = "template"

    def __call__(self, raw) -> torch.Tensor:
        me, active = _active_positions(raw)
        ball = raw["ball"].float()[..., :2]
        have = (raw["ball_owned_team"].long() == 0) & (raw["ball_owned_player"].long() == active)
        goal = torch.tensor([1.0, 0.0], device=me.device).expand_as(me)
        near_goal = torch.linalg.vector_norm(goal - me, dim=-1) < 0.3
        to_ball = move_towards(me, ball)
        to_goal = move_towards(me, goal)
        attack = torch.where(near_goal, torch.full_like(to_goal, SHOT), to_goal)
        return torch.where(have, attack, to_ball)


TEACHERS = {"template": ScriptedTemplate}


def make_teacher(spec: str):
    """`template` or `grf:<GRF_MARL checkpoint>` (e.g. grf:GKBug_v1)."""
    if spec.startswith("grf:"):
        from cup.grf_marl import GRFMarlTeacher
        return GRFMarlTeacher(spec.split(":", 1)[1])
    return TEACHERS[spec]()
