# Guia de treino (para quem vai rodar na GPU)

Tudo roda a partir de um `git clone` deste repositório. O estado de cada treino fica em
`checkpoints/<linha>/` no git: qualquer membro continua de onde outro parou.

## 1. Setup (uma vez por máquina)

**Recomendado: Linux ou WSL2 (Ubuntu 22.04/24.04).** Funciona no Windows nativo também, só o build CUDA do cuju é mais chato.

```bash
git clone https://github.com/Arthur-Marques-IA/RL-CUP
cd RL-CUP
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# torch com CUDA primeiro (o cuju mantém o torch que já estiver instalado)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m pytest -q                                      # paridade de features (2–3 testes)
python -m cup.doctor                                     # versões, cuju check, vazão CPU/GPU, num_envs sugerido
```

### GPU: build do cuju (RTX 50xx = compute capability 12.0)
A wheel do PyPI só tem kernels para 7.5/8.0/8.6/8.9 (RTX 20/30/40, T4, A100, L4). **Para RTX 50 é preciso compilar**:

```bash
# Linux/WSL2: CUDA toolkit >= 12.8, CMake >= 3.24, GCC >= 12
nvcc --version
git clone https://github.com/AndreKoraleski/cuju /tmp/cuju && cd /tmp/cuju
git checkout v0.2.1 2>/dev/null || true
CMAKE_ARGS="-DCUJU_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120" pip install --force-reinstall --no-deps .
cuju check --device cuda        # tem que terminar com PASSED
cd - && python -m cup.doctor
```
Windows nativo: use o "x64 Native Tools Command Prompt for VS 2022" com o CUDA toolkit instalado,
`pip install ninja`, `set CMAKE_GENERATOR=Ninja`, `set CMAKE_ARGS=-DCUJU_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`, `pip install .`
(o CUDA 12.9 recusa o Visual Studio 2026).

**Se o build falhar, treine na CPU** (`--set env.device=cpu env.num_envs=64`). O cuju na CPU é ~10× mais rápido que o GRF.
Segundo o próprio cuju, uma RTX 4090 empata com ~31 threads de CPU. O `cup.doctor` mostra o que compensa na sua máquina.

> ⚠ O caminho de GPU **não foi testado** na máquina de desenvolvimento (que não tem GPU NVIDIA). Rode o `doctor` e o smoke test antes de deixar um treino longo.

## 2. Quem roda o quê (uma linha, um dono por vez)

Binários não fazem merge. **Duas pessoas não podem treinar a mesma linha ao mesmo tempo.** Combinem no grupo e anotem aqui:

| Linha (`line`) | Config | Dono atual | Máquina | Status |
|---|---|---|---|---|
| `B0_bc_template` | `configs/ppo_bc_template.yaml` | | | pronta (init: clone do template, `checkpoints/bc_template`) |
| `A_scratch` | `configs/ppo_scratch.yaml` | | | pronta |
| `B_bc_gkbug` | `configs/ppo_bc_gkbug.yaml` | | | **pendente**: o clone do GKBug_v1 ainda não passou no gate (ver IMITACAO.md) |

Para experimentar uma variação a partir de uma linha existente, **ramifique com outro nome**:
`python -m cup.train --resume B_bc_gkbug --line B2_sem_shaping --set reward.shaping_final=0`

## 3. Treinar

```bash
git pull
python -m cup.train --config configs/smoke.yaml                 # 2 min, confere que tudo funciona
python -m cup.train --config configs/ppo_bc_template.yaml       # linha nova
python -m cup.train --resume B_bc_gkbug                         # continuar (pega o mais novo: local ou publicado)
python -m cup.train --resume B_bc_gkbug --set env.num_envs=4096 # pode mudar device/num_envs ao retomar
```

- Ajuste `env.num_envs` pelo `cup.doctor`, por exemplo `--set env.num_envs=4096`. Todos os cronogramas são em **passos de ambiente**, então máquinas diferentes seguem o mesmo currículo.
- `log.publish_every_minutes: 60` nas configs: o treino **publica sozinho** a cada hora (commit + push de `checkpoints/<linha>/`). Para publicar na mão:
  `python -m cup.checkpoint publish B_bc_gkbug`
- A publicação nunca usa force push. Ela faz `pull --rebase`, e **recusa sobrescrever** se a versão publicada estiver à frente (alguém treinou depois de você).
- Para ver o que está publicado: `python -m cup.checkpoint status`.
- No GPU, um `cuju.UndefinedPathError` derruba o contexto CUDA. **Basta retomar** (`--resume`); o checkpoint local é salvo a cada 2M passos.

### O que é publicado em `checkpoints/<linha>/`
| Arquivo | Conteúdo |
|---|---|
| `latest.pt` | modelo + otimizador + contadores + config + lista do pool (o necessário para continuar) |
| `pool/snap_*.pt` | snapshots da liga (adversários do self-play) |
| `metrics.jsonl`, `evals.jsonl` | curvas de treino e avaliações contra o builtin (viram figuras do relatório) |
| `latest_match.cjr` | última partida de avaliação, para assistir em 3D |
| `meta.json` | quem, máquina, commit do código, passos, linha-pai, última avaliação |

Cada publicação adiciona ~2 MB (latest.pt) + 0.4 MB por snapshot novo + 1.8 MB (partida) ao histórico do git (sem LFS, por escolha do grupo). Publicar de hora em hora é suficiente.

## 4. Acompanhar visualmente

```bash
# curvas ao vivo do SEU treino
tensorboard --logdir runs/B_bc_gkbug/tb
# curvas de TODAS as linhas do time (depois de git pull)
python -m cup.dashboard && tensorboard --logdir runs/_dashboard
# assistir à última partida de avaliação de uma linha (3D no navegador)
python -m cup.watch --line B_bc_gkbug
# assistir ao vivo um checkpoint contra o builtin, contra um snapshot ou contra uma política do GRF_MARL
python -m cup.watch --checkpoint checkpoints/B_bc_gkbug/latest.pt
python -m cup.watch --checkpoint checkpoints/B_bc_gkbug/latest.pt --opponent grf:GKBug_v1
```
Métricas principais: `eval/goal_diff_per_match` e `eval/points_per_match` (contra o builtin, partidas completas), `builtin/*` e `league/*` (jogos de treino), `entropy`, `kl_init`, `top_action_share` (o verificador avisa acima de 0.9), `shot_share`, `possession_ours` e `reward_shaping_per_step`.

## 5. Gerar a submissão e verificar
```bash
python -m cup.export checkpoints/B_bc_gkbug/latest.pt --out submissions/B_bc_gkbug
python -m cup.local_check submissions/B_bc_gkbug                 # probe + tempo de act(), sem Docker
# verificador oficial (Docker; dentro de ../ref/t-zero-learning, com a imagem gfootball construída)
docker compose run --rm -v "<caminho>/RL-CUP/submissions:/subs:ro" gfootball python -m arena.check /subs/B_bc_gkbug
```
O `cup.export` lê tanto `latest.pt` (checkpoint de PPO) quanto `model.pt` (BC).

## 6. Clone BC do GKBug_v1 (em andamento; ainda não publicado)
```bash
git clone https://github.com/jidiai/GRF_MARL /tmp/GRF_MARL
python -m cup.grf_marl extract /tmp/GRF_MARL         # pesos -> third_party/grf_marl/weights (já versionados)
for s in 1 2 3 4 5 6 7; do python -m cup.bc collect --teacher grf:GKBug_v1 --envs 14 --seed $s --out data/gkbug_v1/shard$s.pt & done; wait
python -m cup.bc train --data data/gkbug_v1 --epochs 30 --out runs/bc_gkbug_v1          # avaliar antes de publicar
# só publique (--publish bc_gkbug_v1) se superar o clone do template contra o builtin
```
A coleta é CPU-bound (o encoder do GRF_MARL é Python puro, ~200 passos/s por processo). Não vale rodar na máquina da GPU.
