# Imitation learning: começar o agente sem partir do zero

## Por que imitar
- No 5v5, um PPO do zero passa muito tempo sem gols, com um sinal esparso (GRF_MARL: "não conseguimos treinar fácil com recompensa esparsa").
- Clonar um professor dá ao agente, desde o primeiro passo de RL, uma política que já conduz a bola, chuta e marca. Depois o RL refina.
- Na literatura, o 3º colocado do Kaggle (Raw Beast) clonou os melhores agentes e depois fez RL com KL para o clone, e a força do clone se correlacionou com a força final. O TiKick mostrou que um clone acelera o MAPPO depois, e o AlphaStar seguiu a mesma receita.

## Pipeline
```
1. professor joga no cuju (5_vs_5_l4_r0, dificuldade 0.05/0.05)
   └─ com prob. ε o agente executa uma ação aleatória, mas o rótulo gravado é o do professor
      (noise injection: o aluno vê estados fora da trajetória do professor e aprende a voltar)
2. BC: cross-entropy (features egocêntricas → ação do professor)
3. (opcional) DAgger: o aluno joga, o professor rotula, repete. Só se a distância aluno-professor for grande
4. Passagem BC → PPO:
   a. aquecer o crítico com o ator congelado (senão as primeiras vantagens, ruins, destroem o clone)
   b. PPO com penalidade KL(π || π_BC) que decai, lr menor e bônus de entropia ≈ 0
```

## Professores
| Professor | Status | Notas |
|---|---|---|
| **Template scriptado** (corre até a bola, conduz, chuta) | **implementado** (`cup/teachers.py`) | Valida o encanamento: features, export, probe, `arena.check`. Também é a linha D (submissão de reserva). Fraco: os 4 jogadores perseguem a bola |
| Scriptado melhorado (posicionamento, passe) | a fazer | Barato, e nosso. Bom professor e bom adversário |
| **Políticas 5v5 do GRF_MARL** (Apache-2.0) | **depende de medir** | Treinadas a dificuldade 1.0 em outro arquivo de cenário. Encoder de 192 dims com estado (histórico de ações, impedimento) e rede recorrente. Usar o código delas **sem portar**, num ambiente compatível com o GRF |

### Gate para o GRF_MARL (antes de qualquer porte)
1. Medir cada checkpoint em 20–50 partidas contra o builtin no `5_vs_5` a 0.05/0.05. Se nenhum vencer com folga, descartar.
2. No `desc.pkl` e no encoder, verificar:
   - se a camada recorrente está ligada;
   - se o histórico de ações é usado quando `use_action_gramma` é False;
   - se a política pode emitir uma ação fora de 0–18 (`builtin_ai` = 19 no action set v2).
3. Carregar os pickles **uma vez**, num venv isolado, e salvar um `state_dict` simples.
4. **Um professor por aluno.** Misturar estilos num mesmo dataset dá uma média sem estilo nenhum.
5. Um checkpoint usado como professor **não pode** ser hold-out. Escolher o mais forte como professor e deixar outra família de estilo intocada para hold-out.
6. Testar o estilo "GKBug" a 0.05. Os goleiros da arena são sempre da IA, então a exploração pode funcionar no torneio.

### ⚠ Pesos de terceiros
Destilar a política treinada de outro grupo para uma submissão avaliada **precisa do OK do professor da disciplina** e de declaração no relatório. A linha "do zero" continua viva para o agente final não depender dessa resposta.

## Como avaliar o clone (o relatório pede dado, não acurácia)
- Saldo de gols e V/E/D contra o builtin e **contra o próprio professor** (o clone deveria empatar com ele).
- Histograma dos rótulos: direções "sticky" dominam e podem pedir pesos de classe.
- Experimento principal: **IL-init × do zero com o mesmo número de passos de RL**, 2–3 seeds. Responde "por que não PPO padrão?".

## Resultados até agora (cuju, 5_vs_5 contra o builtin a 0.05, 32 partidas a partir do kick-off, seed 1000)

| Run | Dados | Acurácia val. | V/E/D | Saldo/partida |
|---|---|---|---|---|
| Professor: template scriptado (4 jogadores) | — | — | 22/7/3 | **+1.03** |
| Professor com 10% de ações aleatórias | — | — | (5–65 em gols) | −1.88 |
| `bc_template_s1`: BC, ruído 10% | 384k | 98.1% | 4/6/22 | **−1.09** |
| `bc_template_noise0`: BC, sem ruído | 384k | 97.9% | 19/5/8 | **+0.78** |

**Observações:**
- O professor é frágil: até 10% de ruído só nas direções derruba o saldo para −0.56. As demonstrações com ruído vêm de partidas perdidas, e a distribuição de estados muda.
- O clone com ruído concorda 96% com o professor jogando, mas quase nunca **chuta** (o professor chuta em ~1% dos passos). Uma ação rara e decisiva some no BC. → Para professores melhores: pesos de classe ou sobreamostragem de chutes/passes, e ruído só leve e só em direções.
- Submissão `submissions/bc_template`: **`arena.check` oficial = PASS** (Docker, GRF real). Probe gravado no cuju reproduzido 2400/2400 (100%), 6–0 contra random, 2–1 contra builtin, `act()` 0.89 ms em média (limite 20 ms), carga em 0.30 s.

## Professores GRF_MARL medidos (cuju, 32 partidas contra o builtin a 0.05)
| Checkpoint | V/E/D | Saldo/partida | Pontos/partida |
|---|---|---|---|
| GKBug_v1 | 24/7/1 | +3.34 | 2.47 |
| GKBug_v2 (= v3, pesos idênticos) | 23/8/1 | +3.09 | 2.41 |
| GKBug_v0 | 20/12/0 | +3.00 | 2.25 |
| defense_v3 | 29/2/1 | +2.22 | 2.78 |
| 3-1_LongPass | 30/2/0 | +1.94 | 2.88 |
| PassingMain_v2 | 24/8/0 | +1.28 | 2.50 |
| PassingMain_v1 | 9/12/11 | −0.12 | 1.22 |
| 3-1_formation | 0/11/21 | −1.03 | 0.34 |

Nenhum usa RNN (`use_rnn: False`); o histórico de ações não entra (`use_action_gramma: False`). A rede reconstruída dá saída idêntica à original (diferença 0.0).

### BC do GKBug_v1 (1.18M amostras, 98 partidas, professor 256–8 em gols)
| Aluno | Acurácia val. | Recall do chute | V/E/D | Saldo |
|---|---|---|---|---|
| MLP 256, 20 épocas | 81.3% | 63% | 1/27/4 | −0.03 |
| MLP 512, 40 épocas | ~82% (parou por falta de RAM na época 12) | 65% | — | — |

**Diagnóstico** (`cup/bc_diag.py`, resultados em `results/bc_gkbug_v1_cw0/`):
- **A máscara de ações do professor não ajuda**: o aluno mascarado faz exatamente 1/27/4, e o aluno sem máscara escolhe ação ilegal em 0% dos passos.
- **Erro composto**: o aluno concorda 81% com o professor nos estados do professor, mas só **49% nos estados que ele mesmo visita**. Nesses estados, ele escolhe "release direction" (parar) em 56% dos passos, contra 24% do professor, que ali quer se mover (cima 17%, baixo-direita 13%).
- → **DAgger** (`scripts/dagger.sh`): o aluno joga, o professor rotula, e o aluno é retreinado na base agregada, com β = 0.5 → 0.25 → 0.

40% dos rótulos são "release direction" (14). O aluno **não reproduz** o professor: rede maior quase não ajuda, o que sugere falta de informação nas features (máscara de ações e impedimento do professor) e erro composto. Próximos passos: (1) aplicar a máscara de ações do professor ao aluno e medir; (2) se ajudar, portar as regras da máscara como features; (3) 2–3 rodadas de DAgger. **Gate**: só vira init quando superar o clone do template.

## Comandos
```bash
# testes de paridade (features torch × numpy, professor torch × template numpy)
.venv/Scripts/python -m pytest -q
# BC a partir do template + submissão com probe
.venv/Scripts/python -m cup.bc --teacher template --envs 64 --steps 3001 --out runs/bc_template --submission submissions/bc_template
# verificação oficial (Docker, dentro de ref/t-zero-learning)
docker compose run --rm -v "<caminho>/RL-CUP/submissions:/subs" gfootball python -m arena.check /subs/bc_template
```
