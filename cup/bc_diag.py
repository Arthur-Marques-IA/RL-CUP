"""Diagnostics of a BC student against its GRF_MARL teacher (results feed IMITACAO.md / the report).

    python -m cup.bc_diag onpolicy --student runs/bc_gkbug_v1_cw0/model.pt --teacher GKBug_v1 --matches 16
    python -m cup.bc_diag masked   --student runs/bc_gkbug_v1_cw0/model.pt --teacher GKBug_v1 --matches 32

onpolicy: the student plays (unmasked, argmax) vs builtin; at every step the teacher labels the
          same states. Reports on-policy agreement and both action histograms, plus how often the
          student picks an action the teacher's mask marks illegal.
masked:   the student plays with the teacher's legal-action mask on its logits; W/D/L vs builtin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import cuju
from cup import features
from cup.evaluate import play
from cup.grf_marl import GRFMarlTeacher, MaskedStudent
from cup.model import N_ACTIONS, ActorCritic


def load_student(path: str) -> ActorCritic:
    st = torch.load(path, map_location="cpu", weights_only=False)
    sd = st.get("model", st)
    m = ActorCritic(hidden=sd["actor.0.weight"].shape[0])
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def onpolicy(student: ActorCritic, teacher: GRFMarlTeacher, matches: int, seed: int) -> dict:
    hs, ht = torch.zeros(N_ACTIONS), torch.zeros(N_ACTIONS)
    agree = illegal = total = 0
    with cuju.BatchEnvironment("5_vs_5_l4_r0", environment_count=matches, device="cpu",
                               representation="raw", rewards="scoring", seed=seed, auto_reset=False) as b:
        raw = b.reset()
        fin = torch.zeros(matches, dtype=torch.bool)
        while not fin.all():
            feats = teacher.encode(raw)
            t = torch.as_tensor(teacher.net.act(feats))
            mask = teacher.legal_mask(feats)
            s = student.logits(features.encode(raw)).argmax(-1)
            live = (~fin)[:, None].expand_as(s)
            hs += torch.bincount(s[live], minlength=N_ACTIONS)
            ht += torch.bincount(t[live], minlength=N_ACTIONS)
            agree += int((s == t)[live].sum())
            illegal += int((~mask.gather(-1, s[..., None]).squeeze(-1))[live].sum())
            total += int(live.sum())
            teacher.set_executed(s)
            raw, _, done, _ = b.step(s.to(torch.int32))
            fin |= done.bool()
    return {"agreement": agree / total, "illegal_share": illegal / total,
            "student_hist": (hs / hs.sum()).round(decimals=4).tolist(),
            "teacher_hist": (ht / ht.sum()).round(decimals=4).tolist(), "steps": total}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["onpolicy", "masked"])
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", default="GKBug_v1")
    p.add_argument("--matches", type=int, default=16)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    student, teacher = load_student(a.student), GRFMarlTeacher(a.teacher)
    if a.mode == "onpolicy":
        res = onpolicy(student, teacher, a.matches, a.seed)
    else:
        res = play(MaskedStudent(student, teacher), None, a.matches, a.seed)
    res.update(mode=a.mode, student=a.student, teacher=a.teacher, matches=a.matches, seed=a.seed)
    print(json.dumps(res))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
