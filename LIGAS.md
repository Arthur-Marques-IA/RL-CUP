# Ligas e adversários de treino

## 1. O que a literatura faz

| Trabalho | Esquema de adversários | O que aproveitar |
|---|---|---|
| **Self-play ingênuo** (sempre contra a cópia mais recente) | 100% contra si mesmo | Base mais simples, mas sofre de *ciclos* (A vence B, B vence C, C vence A) e *esquecimento*: o agente deixa de saber vencer estratégias antigas |
| **Fictitious Self-Play (FSP)** | Amostra uniforme de todos os snapshots passados | Resolve o esquecimento, mas gasta muitos jogos contra adversários fracos |
| **OpenAI Five** (Dota 2) | 80% contra os pesos atuais, 20% contra versões passadas. Cada snapshot tem um *quality score* q_i, amostrado por softmax(q). Quando o agente vence aquele snapshot, q diminui (lr 0.01). A cada 10 iterações entra um snapshot novo com q = max(q) | Regra simples e barata para escolher o adversário em proporção à dificuldade |
| **AlphaStar** (StarCraft II) | Três papéis: **main agent** (PFSP contra a liga toda), **main exploiter** (treina só contra o main atual para achar falhas) e **league exploiter** (PFSP contra a liga para achar falhas sistêmicas). Os exploiters são reiniciados de tempos em tempos | **PFSP**: probabilidade de enfrentar o adversário j ∝ f(P[vencer j]), com f(p) = (1−p)². Exploiters como gerador de diversidade |
| **TiZero** (GRF 11v11, AAMAS 2023) | Fase 1: currículo de 10 níveis (força do adversário 0→1 e posição inicial perto do gol adversário), adversário = cópia do nível anterior, avança ao bater um limiar. Fase 2 "Challenge": 80% contra os mais recentes, 20% contra antigos. "Generalise": amostra o pool todo com p ∝ (1−p_vitória)². Snapshot entra no pool quando atinge um critério de desempenho | É o caso mais próximo do nosso. Recompensa: ±1 por gol, +0.0001/passo com posse, +0.05 por passe antes de gol, −0.001 por jogadores agrupados e por bola fora. γ = 0.999, LSTM, política compartilhada condicionada no ID do jogador |
| **WeKick** (1º lugar no Kaggle GRF 2020, 1 jogador controlado) | PPO distribuído com self-play. O pool de adversários é mantido com base no leaderboard e com agentes de **estilos diferentes** (inclusive GAIL), enfrentando o modelo central | Diversidade de estilo importa mais que quantidade de snapshots |
| **GRF-MARL / Light-MALib** (Song et al., AAMAS 2024) | Framework com PSRO e League Training, com configs prontas de 5v5 | Referência de hiperparâmetros e baselines para o 5v5 |

**Lição comum:** o que melhora um agente de RL competitivo é a *distribuição* de adversários. Ela precisa ser **difícil** (PFSP), **diversa** (estilos e exploiters) e **sem esquecimento** (snapshots antigos e âncoras fixas).

## 2. O que muda no nosso caso

1. **Os adversários reais são agentes de outras equipes, não o builtin.** O script `export_submission.py` do t-zero só exporta agentes com **1 jogador controlado** e `simple115v2`, então é *provável* (hipótese, não fato) que várias equipes entreguem agentes `controlled_players: 1` (o resto do time é a IA a 0.05). Outras vão controlar 4. → A liga precisa ter adversários com **1, 2 e 4 jogadores controlados** (bundles `5_vs_5_l4_r1`, `_r2`, `_r4` no cuju), não só `r4`.
2. **O builtin a 0.05 é fraco e explorável.** Um agente que só treina contra ele aprende truques que não funcionam contra agentes treinados. Ele continua no pool como âncora, mas com peso pequeno.
3. **Não manter os ambientes sincronizados.** No cuju, as partidas de 5_vs_5 duram 3001 passos e, resetadas juntas, terminariam juntas. Mas aí cada rollout de 128 passos veria um só pedaço do jogo (kick-off em ~1 de cada 23 updates, uma única faixa de `steps_left`). → **Escalonar os resets** com a máscara de reset do cuju e trocar o adversário **por ambiente**, no `done` de cada um.
4. **Cada partida de treino já é um jogo de avaliação.** O placar final de cada ambiente atualiza P[vencer j] de graça. Não é preciso rodar partidas extras para o PFSP.

## 3. Proposta de liga (em fases, cada uma com motivo observado)

### Fase L0 — Contra o builtin (aquecimento)
- `5_vs_5_l4_r0` (e `l1_r0` se 1 jogador ganhar a comparação), dificuldade 0.05/0.05.
- Opcional: currículo na dificuldade **só do adversário** (direita 0.0 → 0.05), à la TiZero.
- **Saída:** quando o saldo médio contra o builtin estabilizar (ex.: > +2 gols/jogo). Esse vira o snapshot `S0`.

### Fase L1 — Self-play com memória (OpenAI Five / TiZero "Challenge")
- Distribuição de adversários por update:
  - 60% → snapshot mais recente
  - 25% → pool de snapshots antigos (uniforme)
  - 15% → âncoras fixas: builtin (`r0`), random (`r1`), template scriptado (`r1`)
- Novo snapshot no pool a cada K updates **ou** quando a taxa de vitória contra o mais recente passar de 55%.
- **Motivo esperado para o relatório:** mostrar que o agente de L0 perde para estratégias simples (ex.: o template) ou que o self-play puro esquece como vencer `S0` (curva de vitória contra snapshots fixos).

### Fase L2 — PFSP (AlphaStar / TiZero "Generalise")
- Para cada membro j do pool, manter uma média móvel p_j = P[vencer j] alimentada pelos próprios jogos de treino.
- Amostrar j com probabilidade ∝ (1 − p_j)² + ε (com o ε, nenhum adversário fica com probabilidade zero e não é esquecido).
- Manter uma fração fixa de 10–15% nas âncoras.
- Pool limitado (ex.: 20–30 membros). Remover os que vencemos > 95% há muito tempo, mas manter um "hall da fama" de alguns marcos.

### Fase L3 — Diversidade (só se L2 mostrar um platô medido)
- **Main exploiter:** uma segunda rede, inicializada de um snapshot mais antigo, que treina **só contra o main atual**. Quando vence o main > 65%, é congelada e entra no pool, depois reinicia. Ela expõe falhas que o self-play não vê.
- **Agentes de estilo** (como no WeKick), treinados contra o builtin com recompensas diferentes, de pouco treino cada:
  - *posse*: + por passo com a bola, − por perder a bola
  - *chute de longe*: + por chute ao gol
  - *defensivo*: − por gol sofrido mais forte, + por recuperar a bola
  - *1 jogador*: imita o perfil esperado das outras equipes (`controlled_players: 1`)
- Agentes scriptados variados: o template, um que só faz passes longos e um que pressiona a bola com todos.

## 4. Implementação no cuju (rascunho)

```
BatchEnvironment([ "5_vs_5_l4_r0", "5_vs_5_l4_r1", "5_vs_5_l4_r2", "5_vs_5_l4_r4" ],
                 assignment = env → bundle,  difficulty 0.05/0.05 em todos)

no início: resets escalonados (cada ambiente começa em um ponto diferente da partida)
no done de cada ambiente:
    sortear adversário j via PFSP para aquele ambiente, respeitando o bundle (nº de jogadores de j)
    agrupar ambientes por adversário → um forward por rede congelada (no_grad, na GPU)
a cada passo:
    ações_esq = learner(obs[:, slots_esq])        # só esses entram no PPO
    ações_dir = opp_j(obs[envs de j, slots_dir])  # cada lado vê o campo do próprio ponto de vista
fim de episódio:
    resultado (V/E/D, saldo) por ambiente → atualiza p_j e o log da matriz de payoff
```

- Snapshots salvos como `state_dict` (pequenos, ~1 MB para MLP), carregados sob demanda. Pool de até ~30 redes cabe na GPU com folga.
- Âncora `builtin` = bundle `r0` (nenhum agente na direita). Âncora `random` = `r1` com ações uniformes.
- Adversários do pool podem jogar com argmax (como as outras equipes, que provavelmente usam `deterministic: true`) e às vezes com amostragem, para gerar diversidade.
- Recompensas de shaping **só para o learner**. O adversário congelado não treina, então o problema de shaping não ser soma-zero não aparece.

## 5. Como medir (vira figuras do relatório)

| Medida | Figura |
|---|---|
| Taxa de vitória e saldo contra um **conjunto fixo de avaliação** (random, builtin, template, S0, S_meio) ao longo do treino | Curvas por run: mostram esquecimento ou progresso de verdade |
| **Matriz de payoff** entre snapshots | Heatmap: evidencia ciclos (não-transitividade) e justifica o PFSP |
| Rating **Bradley–Terry/Elo** via `arena.tournament` | Tabela final do agente contra versões anteriores |
| Distribuição de amostragem do PFSP ao longo do tempo | Mostra quem eram os adversários "difíceis" em cada fase |
| Estatísticas de comportamento (posse, passes, chutes, posição média) por adversário | Ajuda a explicar *por que* uma mudança funcionou |

**Comparações mínimas para responder "por que não PPO padrão?":**
1. PPO contra o builtin (L0) × L0 + self-play (L1) × L1 + PFSP (L2). Mesmo orçamento de passos, avaliado no conjunto fixo e num mini-torneio entre si.
2. Liga só com `r4` × liga com `r1/r2/r4`, avaliadas contra adversários de 1 jogador.

## Fontes
- AlphaStar: [blog DeepMind](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/), [TStarBot-X](https://arxiv.org/pdf/2011.13729) (estudo aberto de league training)
- OpenAI Five: [Berner et al. 2019](https://arxiv.org/pdf/1912.06680)
- TiZero: [Lin et al. 2023](https://arxiv.org/abs/2302.07515v2)
- WeKick / TiKick: [TiKick](https://arxiv.org/abs/2110.04507v5), [survey de self-play](https://arxiv.org/pdf/2408.01072)
- GRF-MARL / Light-MALib: [Song et al. 2024](https://arxiv.org/pdf/2309.12951), [código](https://github.com/jidiai/GRF_MARL)
