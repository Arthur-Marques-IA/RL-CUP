# Literatura: o que funcionou no GRF

Todas as afirmações abaixo vêm de fontes primárias (write-ups originais, repositórios, PDFs dos artigos), exceto onde marcado. O Kaggle 2020 era 11v11 com **1 jogador controlado**. GRF_MARL é o trabalho mais próximo do nosso 5v5.

## 1. Tabela-resumo

| Fonte | Algoritmo | Recompensa | Adversários | Compute | Resultado |
|---|---|---|---|---|---|
| **WeKick** (1º Kaggle) | PPO assíncrono, MLP 256 + LSTM | Soma-zero: troca de posse ±0.2, carrinho certo ±0.2 (só no início), posse ±0.001/passo, +0.1 por passe numa sequência que acaba em gol. **Valor com várias cabeças, cada uma com seu desconto** | Liga estilo AlphaStar com modelos de estilo (contra-ataque, passe curto, posse) e GAIL sobre replays de outras equipes | 800 núcleos + 1 GPU, 2 dias + ~1 dia por refinamento | Modelo final contra a liga toda: +100 Elo interno |
| **SaltyFish** (2º) | IMPALA, MLP | Fase 1: scoring+checkpoints contra o bot fácil (0.05) e depois o difícil (1.0). **Self-play: só scoring** | Troca de adversário a 80% de vitória ou a cada 20 episódios. O mais novo em ~70% dos jogos, que rendeu igual a escolher por Elo | 1000–2500 núcleos | A fase 1 venceu o bot difícil por >10 gols, mas virou um **"sempre driblar" explorável** → motivou o self-play. Shaping de posse/passe mudou o estilo mas não deu força |
| **Raw Beast** (3º) | Clonagem de comportamento → IMPALA + KL para o clone, UPGO | **Só ±1 por gol, γ = 0.999.** Recompensas complexas falharam, e o RL "começou a funcionar" quando simplificaram | 50% self-play, 50% pool de ~10 adversários fixos diversos. Sem self-play e sem KL, sobreajuste | 20 núcleos + 1 GPU, ~4 dias | Espelhamento em y, máscara de no-op e transformers **não ajudaram** |
| **TamakEri** (5º) | HandyRL: UPGO + TD(λ), atenção + GRU | Alvo de valor de resultado = tanh(saldo) + cabeça de retorno ±1 | Self-play de 10% → 80%, bots scriptados, bot difícil e modelos passados, pesados para ~50% de vitória | até ~4000 CPUs + 20 GPUs | **30% dos jogos começavam depois de passos aleatórios de um bot scriptado** (início não-kick-off) |
| **liveinparis** (6º) | PPO + LSTM do zero, γ = 0.993, λ = 0.96 | `5·vitória + 5·gol + 0.003·zona_da_bola + cartão` (soma-zero) | 50% entre os últimos 10 snapshots, 50% uniforme sobre ~380 snapshots | 30 núcleos + 1 RTX 2080, 370 h | 92% contra o bot difícil sem nunca ter treinado contra ele. Conv 1D sobre os times > transformer |
| **s_toppo** (7º) | Imitação com GBDT | — | — | — | Espelhamento em y também não ajudou |
| **GRF_MARL** (Song et al. 2024, **5v5**) | IPPO/MAPPO/HAPPO/MAT, MLP 256-128-64, PopArt, γ = 0.995, λ = 0.95, lr 5e-4 | scoring+checkpoints, compartilhada pela equipe. Com recompensa só esparsa "não conseguimos treinar fácil" (política conservadora) | Builtin a **1.0**. Liga/PSRO: γ = 1.0, `5·vitória + 5·gol + cartão + 0.1·dist` | 1e8 passos, ~4–8 h em 128–200 CPUs + 4 GPUs, 5 seeds | **IPPO 99%, MAT 99%, MAPPO 96%** de vitória contra o bot 1.0. QMIX/QPLEX 0% |
| **Song et al. 2023** (11v11) | IPPO | ±1 por gol | Bot 1.0, depois PSRO | 250 workers, 4.9 h até 80% | **γ = 1 > 0.995 > 0.99.** Reduzir a penalidade por gol sofrido para 0.2 salvou seeds que falhavam. `simple115` foi "terrível" contra encoders com features relativas. A máscara de ações acelera contra o bot, mas foi removida no treino populacional. Incluir o bot na população reduziu a diversidade |
| **Kurach et al. 2020** (artigo do GRF) | PPO / IMPALA / DQN | scoring × checkpoints | Bot 0.05/0.6/0.95 | 50M–500M passos | Checkpoints ajudam muito o policy gradient (PPO@50M no fácil: 0.09 → 8.71 de saldo). **Sobreajuste a adversário fixo:** B treinado contra A vence A por +11.93, mas faz −0.27 contra o builtin. **Controlar 3 jogadores:** pior aos 5M passos e melhor aos 50M |
| **TiKick** | RL offline a partir de 22 mil jogos do WeKick | — | — | — | 94% contra o builtin. Serve de pré-treino para MAPPO |

## 2. Métodos de estado inicial (para o banco de estados)

| Fonte | Ideia | Lição para nós |
|---|---|---|
| **Pardo et al. 2018** | Tarefa com limite de tempo: o fim é terminal **e** o tempo restante (normalizado) vai na observação. Limite só por conveniência de treino: bootstrap no fim | Temos `steps_left` na observação → fim terminal. Confirma a correção do RECOMPENSA.md |
| **Tavakoli et al. 2018** (*Exploring Restart Distributions*) | Reinicia de estados já visitados pelo próprio agente (uniforme, priorizado por erro TD com \|δ\|^0.4, ou dos melhores episódios), ~10% das transições. O sub-episódio herda o tempo restante original | É quase exatamente o nosso banco. Ganho grande em tarefas esparsas. A avaliação é sempre do início real |
| **Kakade & Langford 2002** | A garantia do CPI no início real depende de quão bem a distribuição de reinício cobre os estados da política ótima | Misturar reinícios com kick-offs, nunca só reinícios |
| **Florensa 2017 / Backplay 2018 / Salimans & Chen 2018** | Currículo reverso: começa perto do objetivo e recua. Avançar rápido demais atrapalha | Pouco necessário aqui (já marcamos gols aleatoriamente no 5v5), mas reforça repetir estados antigos para não esquecer |
| **Prioritized Level Replay 2021** | Prioriza níveis pelo \|GAE\| médio (rank, temperatura 0.1) + staleness | Variante de priorização para o banco, se houver tempo |
| **TamakEri** | 30% dos jogos começam depois de passos aleatórios de um bot | Evidência prática no próprio GRF |

## 3. Implicações para o nosso projeto

1. **Recompensa simples vence no longo prazo.** Raw Beast (só ±1) e SaltyFish (shaping não deu força) concordam. O shaping serve de andaime no início (GRF_MARL e Kurach: o esparso puro trava). → Mantém o nosso plano: Φ no começo, decaindo até só placar no self-play.
2. **γ alto.** 0.993–0.999 nos melhores, e γ = 1 melhor que 0.995 no estudo do Song. → O sweep do R5 passa a ser **0.995 × 0.999 × 1.0**. Com γ alto, considerar penalidade por gol sofrido menor que 1 se seeds falharem (Song).
3. **Treinar contra o bot fácil gera política explorável** (SaltyFish: "sempre driblar"; Kurach: −0.27 contra o builtin após treinar contra um fixo). Isso motiva a liga com dados de outros, mas o relatório precisa mostrar o **nosso** dado.
4. **Mistura de adversários**: todos convergem para cerca de metade self-play recente e metade pool diverso/fixo. Escolher o mais novo ~70% rendeu igual ao Elo (SaltyFish). → O PFSP é um refinamento, não um requisito. Começar simples (L1) e só passar ao PFSP com motivo.
5. **Features relativas importam** (Song: `simple115` "terrível"; SaltyFish: absolutas "sem generalização"; WeKick: +30% de velocidade). → Confirma a observação egocêntrica.
6. **Espelhamento em y: dois times relatam que não ajudou** (Raw Beast, s_toppo). → Rebaixado de "dados de graça" para experimento opcional de baixa prioridade.
7. **Máscara de ações:** acelera contra o bot, mas bloqueia passes de primeira e não melhorou o resultado final (SaltyFish 1597.5 sem × 1597.0 com). → Só experimento.
8. **Valor com várias cabeças** (WeKick) e alvo tanh(saldo) (TamakEri) → apoia as duas cabeças de valor e um termo de resultado.
9. **Controlar mais jogadores: mais lento no início, melhor depois** (Kurach). → Cuidado no corte por successive halving: não cortar a linha de 4 jogadores cedo demais.
10. **Escala:** o GRF_MARL chegou a 99% contra o bot 1.0 no 5v5 com **1e8 passos**. No cuju com a 5070 Ti, isso é questão de horas, não de dias. O orçamento é viável.

## 4. Adversários hold-out prontos: políticas pré-treinadas do GRF_MARL
`light_malib/trained_models/gr_football/5_vs_5/`: 3-1_formation, 3-1_LongPass, defense_v3, GKBug_v0–v3, PassingMain_v1/v2.
- Estilos: passe curto 1-3, passe longo, defensivo, "GKBug" (passe longo ao gol enquanto um atacante distrai o goleiro).
- Formato: `torch.nn.Module` inteiro em pickle (`actor.pt`), encoder de 133 dimensões, action set v2. Exige portar o encoder e carregar com `light_malib` importável.
- Uso: **adversários de treino e de hold-out** (rodando dentro do nosso treino ou via adaptador `opponent=` no t-zero). Não vão na submissão.
- Pendências: treinados com dificuldade 1.0 (atenção ao rodar a 0.05), verificar a licença do repositório e medir a força deles contra o builtin a 0.05.

## Fontes
- WeKick: https://www.kaggle.com/c/google-football/discussion/202232
- SaltyFish: https://www.kaggle.com/competitions/google-football/writeups/saltyfish-saltyfish-current-2nd-place-solution
- Raw Beast: https://www.kaggle.com/c/google-football/discussion/200709
- TamakEri: https://www.kaggle.com/competitions/google-football/writeups/tamakeri-tamakeri-5th-place-solution
- liveinparis: https://www.kaggle.com/c/google-football/discussion/201376, https://github.com/seungeunrho/football-paris
- s_toppo: https://www.kaggle.com/c/google-football/discussion/200912
- Índice dos write-ups: https://www.kaggle.com/c/google-football/discussion/204645
- Kurach et al. 2020: https://arxiv.org/abs/1907.11180
- TiKick: https://arxiv.org/abs/2110.04507
- GRF_MARL: https://arxiv.org/abs/2309.12951, https://github.com/jidiai/GRF_MARL
- Song et al. 2023: https://arxiv.org/abs/2305.09458
- Pardo et al. 2018: https://arxiv.org/abs/1712.00378
- Tavakoli et al. 2018: https://arxiv.org/abs/1811.11298
- Florensa et al. 2017: https://arxiv.org/abs/1707.05300
- Backplay: https://arxiv.org/abs/1807.06919
- Salimans & Chen 2018: https://arxiv.org/abs/1812.03381
- Prioritized Level Replay: https://arxiv.org/abs/2010.03934
