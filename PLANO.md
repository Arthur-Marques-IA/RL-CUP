# Plano — Google Research Football (5_vs_5)

Referências clonadas em `../ref/`:
- `ref/t-zero-learning` — ambiente oficial, arena (`arena.check`, `arena.match`, `arena.tournament`), template de submissão, Docker.
- `ref/cuju` — reimplementação **bit-exata** do GRF 2.10.3, em lote, CPU (thread pool) e CUDA. Mesma seed + mesmas ações = mesmo jogo do GRF.

## Ideia central

**Treinar no cuju, avaliar e entregar no GRF/arena.**
O cuju resolve o gargalo (GRF ≈ 200–300 passos/s por processo; cuju CPU ≈ 2.000/s por thread, GPU ≈ dezenas de milhares/s).
Como é bit-exato, a política treinada joga igual no GRF — mas isso precisa ser **verificado** por nós (Fase 1), porque o que importa é a interface da arena (troca de slots com N<4, `reset()` no kick-off, visão espelhada do time da direita).

## Fatos que guiam as decisões

| Fato | Consequência |
|---|---|
| Torneio: `5_vs_5`, 3000 passos, gol **não** encerra o episódio, dificuldade 0.05 nos **dois** times | O episódio sempre termina por truncamento → bootstrap no truncamento (o `ppo.py` do cuju trata como terminal; decidir conscientemente) |
| Dificuldade escala os atributos de **todos** os jogadores do time, inclusive os nossos (`0.3 + 0.7·d`) | Nunca mudar a dificuldade do lado esquerdo; avaliar sempre em 0.05/0.05. Subir só a do adversário como currículo |
| Recompensa `checkpoints` paga cada zona **uma vez por episódio** | Em 3000 passos ela só ajuda no primeiro ataque → shaping próprio (via `with_raw=True`) ou currículo na academy |
| `act()`: 1 thread de CPU, média ≤ 20 ms para **todos** os jogadores | MLP pequena, um único forward com as N observações empilhadas; nada de render/pixels |
| Entrada no torneio: só o dict bruto | Qualquer feature extra (placar, relógio, sticky actions, distâncias) precisa ser recalculada **identicamente** em `act()` → uma função de features com teste de igualdade treino × submissão |
| Imagem da arena: `torch==2.9.1`, `numpy==2.2.6`, CPU | Salvar pesos como `state_dict`/npz e reconstruir a rede em `agent.py` (mais seguro que TorchScript entre versões) |
| RTX 5070 Ti = compute capability 12.0 | **A wheel do cuju não roda na GPU dela.** Precisa build do código-fonte com CUDA ≥ 12.8 (Linux/WSL2 é o caminho mais fácil; no Windows, VS 2022). Antes, medir o backend de CPU dessa máquina — numa 4090 a GPU empata com 31 threads de CPU |
| Verificador avisa se uma ação passa de 90% das ações | Logar histograma de ações em todo treino |

## Fases

### Fase 0 — Encanamento ponta a ponta (antes de qualquer treino sério)
1. Ambiente local: `pip install cuju` (CPU) + build da imagem Docker `gfootball` do t-zero.
2. Rodar `arena.check` no template → PASS (garante que o Docker funciona).
3. PPO minúsculo no cuju por alguns minutos → exportar para uma pasta `submission/` com `agent.py` próprio (rede reconstruída + features) + `probe.pkl` → `arena.check` → **PASS**.
   - O probe precisa ser gravado com observações no **formato da arena** (dicts brutos). Plano: gravar no GRF via shim do t-zero rodando a política exportada, ou converter `raw` do cuju para dicts numpy.

### Fase 1 — Paridade cuju × arena
- Igualdade numérica: `simple115v2` do cuju == `Simple115StateWrapper.convert_observation(raw)` e nossa função de features em torch == versão numpy em `agent.py`.
- Mesmo checkpoint, N partidas contra `builtin` no cuju e via `arena.match`: saldo de gols compatível dentro do intervalo de confiança.

### Fase 2 — Infra de experimentos (desde o dia 1, é o que alimenta o relatório)
- Cada run com ID, config salva, seed, snapshots periódicos.
- Métricas de treino: retorno, gols pró/contra por partida, entropia, value loss, histograma de ações, posse de bola, % de passes, passos/s.
- 2–3 seeds por comparação principal. Contra o builtin as variantes tendem a saturar → usar **cross-play entre variantes** e o pool fixo como métrica decisiva.
- Protocolo de avaliação fixo: X partidas contra `random`, `builtin` e snapshots anteriores (no cuju para curvas rápidas; `arena.tournament` para a tabela final com Bradley–Terry).

### Fase 3 — Baseline: PPO padrão
- PPO vanilla, `simple115v2`, `5_vs_5_l4_r0` (e `l1_r0`), recompensa `scoring,checkpoints`, contra `builtin`.
- É a resposta para "por que não um PPO padrão?": registrar onde ele falha (ex.: fica parado com a bola, chuta de longe, não defende, colapsa em uma ação) — cada falha observada motiva a próxima mudança.

### Fase 4 — Eixos de experimento (cada um: hipótese → métrica → comparação contra o run anterior)
| Eixo | Variações |
|---|---|
| Jogadores controlados | 1 vs 4 (política compartilhada, `active` na observação). Os companheiros da IA estão na dificuldade 0.05, então controlar 4 tende a ser melhor — mas tem que ser medido |
| Recompensa | `scoring` puro × `scoring,checkpoints` × shaping próprio (progresso da bola, posse, perda de bola), com decaimento ao longo do treino |
| Currículo | Cenários customizados de 5v5 (início avançado, `end_episode_on_score`) → `5_vs_5` com adversário mais fraco → 0.05. **Sempre com dificuldade 0.05 no nosso time**: as academies rodam em 1.0/0.6, com jogadores ~3× mais fortes, e a habilidade não transfere limpa. Se usar academy, sobrescrever a dificuldade |
| Adversários | `builtin` → self-play (`l4_r4`) → liga com pool de snapshots + builtin, misturando bundles `r0..r4` (os adversários do torneio controlam 1–4 jogadores). Desenho detalhado em [LIGAS.md](LIGAS.md) |
| Representação | `simple115v2` × **egocêntrica**: posições da bola, do gol, dos companheiros e dos adversários relativas ao jogador ativo, "tenho a bola?", sticky actions, saldo de gols, `steps_left`, sem o padding −1 (44 das 88 dimensões de jogadores do `simple115v2` são constantes no 5v5) |
| Aumento de dados | Espelhamento vertical (y → −y) com troca de ações cima↔baixo. **Baixa prioridade**: Raw Beast e s_toppo relataram que não ajudou ([LITERATURA.md](LITERATURA.md)) |
| Arquitetura | MLP 256×2 × maior; ator-crítico separado × compartilhado; duas cabeças de valor (gols / shaping). Adiados: MAPPO (o vetor já é quase o estado global), LSTM e frame stack (velocidades e sticky actions tornam o estado quase Markoviano, e uma rede sem memória evita a semântica de `reset()`) |
| Máscara de ações | Só como experimento: primeiro verificar o que passe/chute fazem sem a bola; a máscara precisa ser replicada igual em `act()` |
| Seleção de ação | argmax × amostragem (define `deterministic` no manifest) |

### Fase 5 — Agente final e avaliação
- Escolha pelo rating contra pool fixo (random, builtin, template, snapshots), não pela recompensa de treino.
- Tabela final: taxas de vitória/empate/derrota e saldo contra `random`, `builtin` e versões anteriores; `arena.check` PASS.

## Divisão no hardware
- **Local (8 threads, CPU)**: Fases 0–2, depuração, testes curtos do cuju na CPU, avaliações via Docker.
- **5070 Ti do colega**: runs longos (Fases 3–5). Primeiro medir passos/s no backend CPU dele; se a GPU compensar, build do cuju do fonte (CUDA ≥ 12.8).

## Relatório (acumular durante o projeto)
- Guardar figura/tabela de cada mudança com o comportamento que a motivou.
- Uso de IA: declarar por seção e anexar os prompts — guardar os prompts desta conversa.

## Em aberto
- Prazo de entrega e do torneio.
- SO, CPU e RAM da máquina com a 5070 Ti (Linux/WSL2 facilita muito o build do cuju).
- Tamanho do grupo e quem cuida de quê.
