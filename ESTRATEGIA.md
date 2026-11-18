# Estratégia: estados iniciais, adversários e seleção do agente

Complementa [PLANO.md](PLANO.md), [RECOMPENSA.md](RECOMPENSA.md) e [LIGAS.md](LIGAS.md). A literatura está em [LITERATURA.md](LITERATURA.md).

---

## 1. Banco de estados: começar partidas no meio do jogo

### O que isso resolve (e o que não resolve)
- **Não resolve "fim de jogo" por si só.** Uma partida de 3000 passos já passa por todos os momentos do relógio. Com recompensa ±1 por gol, o fim do jogo não é especial para o agente.
- O fim do jogo **só fica diferente** se três coisas valerem juntas:
  - o agente observa o saldo de gols e `steps_left`;
  - o fim é terminal ([RECOMPENSA.md](RECOMPENSA.md) §5);
  - a recompensa tem um termo de **resultado** (vitória/empate/derrota), porque o torneio conta 3/1/0.

  Aí "vencendo por 1 faltando 30 s" pede outro comportamento (segurar a bola) do que "perdendo por 1" (arriscar). → Experimento próprio: termo terminal `c · pontos(resultado)`.
- **O que o banco compra de verdade:**
  - situações raras com mais frequência: bolas paradas (escanteio, falta, lateral, pênalti), defesa dentro da própria área, placares desequilibrados, estar perdendo no fim;
  - inícios descorrelacionados, já que cada ambiente começa num ponto diferente (resolve o problema de sincronia do [LIGAS.md](LIGAS.md) §2).

### Como o cuju permite
- `batch.get_state(i)` salva o estado completo de uma partida (~110 KB, ~1 ms). `batch.set_state(i, state)` carrega e continua bit a bit (~4 ms na GPU para 5_vs_5).
- O estado inclui placar, `steps_left`, posições, animações, memória da IA e o modo de jogo.
- **Restrição: um estado só carrega na mesma configuração** (cenário, nº de agentes de cada lado, dificuldades, flags). → **Um banco por bundle** (`l4_r0`, `l4_r1`, `l4_r2`, `l4_r4`).
- Cenários customizados não servem para isso: as posições ficam fixas na construção do bundle (uma forma para episódios pares e outra para ímpares) e não definem placar nem relógio.

### Desenho
```
coleta: durante os jogos da liga, a cada ~200 passos, get_state de alguns ambientes
        → FIFO por bundle (ex.: 5.000 estados), cada estado com tags:
          steps_left (faixa), saldo de gols, game_mode, quem tem a bola
reset:  quando um ambiente termina (done):
          com prob. p (≈ 0.5) → set_state com um estado do banco do mesmo bundle
                                (sobreamostrar tags raras: bola parada, perdendo no fim, |saldo| ≥ 2)
          senão               → kick-off normal
```
Cuidados de implementação:
- Copiar a recompensa e o `done` do passo **antes** do `set_state`, porque ele sobrescreve os buffers.
- Reinicializar Φ (potencial do shaping) e o "último toque" no estado carregado. Senão aparece um shaping espúrio na emenda.
- O banco vem dos jogos do próprio learner (e da liga), então ele se atualiza com o nível do agente. Estados antigos saem pela FIFO.
- Variante mais avançada (Prioritized Level Replay): priorizar estados com erro de vantagem/valor alto.

### Motivo observado (exigido pelo relatório) e avaliação
1. **Antes de ligar o banco**, medir no baseline:
   - o saldo de gols por **segmento de 500 passos** da partida;
   - o saldo **por estado do placar** (vencendo, empatando, perdendo).

   Se o agente piora no fim ou quando está perdendo, isso motiva o banco. Se for plano, o banco é justificado como diversidade de inícios.
2. Comparação: mesmo orçamento, p = 0 × p = 0.5 (× PLR, se houver tempo). 2–3 seeds.
3. **A avaliação é sempre com partidas completas a partir do kick-off**, porque o torneio é assim. O banco é só uma distribuição de treino.

---

## 2. Contra quem treinamos e avaliamos

### Revisão do pool (atualiza o [LIGAS.md](LIGAS.md))
| Problema | Correção |
|---|---|
| O pool é de **uma linhagem só** (snapshots do mesmo run são parecidos entre si) | Treinar **2–3 linhagens independentes** (seed, recompensa, N jogadores diferentes). O jogo entre linhagens é o melhor substituto para as equipes desconhecidas |
| Selecionar o agente contra os mesmos adversários do treino superestima a força | **Conjunto hold-out**: adversários nunca usados no treino, só na seleção (ex.: uma linhagem inteira separada, agentes scriptados extras) |
| Pool pequeno de formatos | Cada snapshot pode jogar controlando 1, 2 ou 4 jogadores (`r1`, `r2`, `r4`) → o pool **triplica** de graça |
| PFSP com taxa de vitória: no 5v5 muitas partidas empatam (0–0, 1–1) e tudo fica perto de 0.5 | Usar **pontos esperados** (3/1/0) ou o saldo médio de gols comprimido (ex.: tanh(saldo/2)) como sinal do PFSP |
| O lado do campo pode importar | A arena alterna os lados a cada partida. Medir o mesmo checkpoint jogando à esquerda e à direita contra o builtin (`5_vs_5_l0_r4`). Se o saldo diferir, treinar dos dois lados |
| Adversários determinísticos demais | Misturar argmax e amostragem nas ações dos snapshots. Adicionar bots scriptados vetorizados em torch e verificar o `cuju.players.Bot` (o bot do GRF) |
| Supor que as outras equipes são fracas | Não supor. Um colega escreveu o cuju e outras equipes podem usá-lo |

**Adversários prontos de estilos diferentes:** as políticas 5v5 pré-treinadas do GRF_MARL (passe curto, passe longo, defensivo, "GKBug") são candidatas a **hold-out** e a adversários da liga. Detalhes e pendências (licença, força a 0.05, portar o encoder) em [LITERATURA.md](LITERATURA.md) §4.

**O que a literatura usa:** perto de metade self-play recente e metade pool diverso (Raw Beast 50/50; liveinparis 50% últimos 10, 50% uniforme). O SaltyFish viu que escolher o mais novo ~70% rendeu igual ao Elo. → Começar com essa regra simples e só adotar o PFSP se o dado pedir.

### Composição final dos adversários de treino
| Grupo | Peso inicial | Formato |
|---|---|---|
| Snapshots da linhagem atual (PFSP) | ~50% | r1 / r2 / r4 |
| Snapshots de outras linhagens | ~25% | r1 / r2 / r4 |
| Âncoras: builtin, random, template, scriptados | ~15% | r0 / r1 |
| Exploiter (se existir) | ~10% | r4 |

### Conjunto de avaliação fixo (não muda durante o projeto)
`random`, `builtin`, template, 1–2 scriptados próprios, snapshots marco (S0, S_meio) e a **hold-out**. Toda curva do relatório usa esse conjunto.

---

## 3. Várias estratégias, ficar com a melhor

### Método: *successive halving* (eliminação por rodadas)
1. Rodar **3–4 linhas** que diferem num eixo de alto nível, com orçamento igual.
2. Em checkpoints fixos (ex.: 25% e 50% do orçamento), avaliar todas contra o "campo substituto" e **cortar a metade de baixo**. Atenção: controlar mais jogadores é pior no início e melhor depois (Kurach et al.: 3 jogadores perdem aos 5M passos e ganham aos 50M). Não cortar a linha de 4 jogadores no primeiro corte só por isso. O orçamento liberado vai para as sobreviventes.
3. As linhas eliminadas **não são desperdício**: viram adversários da liga (diversidade) e experimentos do relatório.

### Linhas candidatas
| Linha | Descrição | Hipótese |
|---|---|---|
| **A** | 4 jogadores, política compartilhada, observação egocêntrica, shaping Φ → liga PFSP | Melhor controle coletivo |
| **B** | 1 jogador (o ativo), mesma pipeline | Mais simples; os companheiros da IA ajudam mais do que atrapalham? |
| **C** | 4 jogadores, **só placar**, self-play com currículo desde o início | Shaping atrapalha no longo prazo? |
| **D** | **Agente scriptado** ajustado à mão | Piso de desempenho, submissão de reserva (passa no `arena.check` já), adversário da liga |
| (E) | A + banco de estados | Entra como variante da linha que sobreviver |

### Critério de seleção
- **Pontos esperados por partida** (3·V + 1·E) contra o campo substituto: random, builtin, template, hold-out e as outras linhas. Tem intervalo de confiança.
- **Centenas de partidas por par no cuju** (barato). Depois confirmar os 2 melhores no `arena.tournament` (o ambiente real), cada um com argmax **e** com amostragem.
- O vencedor vira a submissão. O agente scriptado (D) fica como reserva.

### O que falta para dimensionar
O orçamento total (passos de ambiente) depende de duas coisas ainda em aberto:
- o **prazo** de entrega e do torneio;
- a **vazão** da máquina com a 5070 Ti: CPU, SO e se o build CUDA do cuju compila.

Hipótese de trabalho até a medição: ~15–30 mil passos/s → ~50–100 M passos por dia. Isso dá 4 linhas × ~100 M até o primeiro corte.
