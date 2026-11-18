# RL-CUP: Google Research Football 5v5

Agente de aprendizado por reforço para o torneio 5_vs_5 da disciplina (ambiente de referência: [t-zero-learning](https://github.com/BrunoBSM/t-zero-learning)).
O treino roda no [cuju](https://github.com/AndreKoraleski/cuju), uma reimplementação bit-exata e em lote do GRF (CPU e GPU). A entrega roda na arena oficial do GRF.

## Começo rápido
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
python -m cup.doctor --quick
python -m cup.train --config configs/smoke.yaml       # 2 min, testa todo o pipeline
```
**Quem vai treinar na GPU: leia [TREINO.md](TREINO.md)** (build do cuju para RTX 50, linhas, publicar e retomar, acompanhar).

## Estrutura
| Caminho | O quê |
|---|---|
| `cup/features.py` | features egocêntricas (87) em torch, para o treino |
| `submission_template/agent.py` | as mesmas features em numpy + MLP: o agente da arena |
| `cup/train.py` | PPO, política compartilhada pelos 4 jogadores, shaping de potencial, liga de snapshots, KL ao BC |
| `cup/bc.py`, `cup/teachers.py`, `cup/grf_marl.py` | imitation learning: template scriptado e políticas GRF_MARL como professores |
| `cup/checkpoint.py` | salvar, publicar no git e retomar o trabalho de outro membro |
| `cup/evaluate.py`, `cup/watch.py`, `cup/dashboard.py` | avaliação, partidas em 3D no navegador, curvas de todas as linhas |
| `cup/export.py`, `cup/local_check.py` | gerar a pasta de submissão com probe e checar sem Docker |
| `configs/` | configs das linhas de treino |
| `checkpoints/<linha>/` | estado publicado de cada linha (modelo, liga, métricas, última partida) |
| `results/` | resultados pequenos (JSON/CSV) que viram tabelas e figuras do relatório |
| `third_party/grf_marl/` | encoders e pesos do [GRF_MARL](https://github.com/jidiai/GRF_MARL) (Apache-2.0) |

## Planejamento e decisões
[PLANO.md](PLANO.md) · [RECOMPENSA.md](RECOMPENSA.md) · [LIGAS.md](LIGAS.md) · [ESTRATEGIA.md](ESTRATEGIA.md) · [LITERATURA.md](LITERATURA.md) · [IMITACAO.md](IMITACAO.md)

## Uso de IA generativa
Partes do código e do planejamento foram feitas com assistência do Claude (Anthropic). Isso deve ser declarado no relatório, com os prompts em anexo.
