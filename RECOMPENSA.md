# Modelagem da recompensa

## 1. O objetivo real

No torneio, cada partida vale 3/1/0 pontos e o rating Bradley–Terry conta vitória e empate. O que queremos maximizar é **vencer**, e o saldo de gols é a melhor aproximação contínua disso.

A recompensa "verdadeira" é então `scoring`: **+1 por gol feito e −1 por gol sofrido**. Tudo o que for somado a isso é *shaping*: um andaime para o aprendizado, que precisa ser justificado e, de preferência, removido no final.

**Por que só `scoring` não basta:** uma partida tem 3000 passos e, no começo do treino, quase não sai gol. O sinal é esparso e o crédito de um gol precisa voltar dezenas de segundos de jogo (centenas de passos). O PPO padrão aprende devagar ou nem aprende. Esta é a primeira coisa a **mostrar com dado** (run R0 abaixo).

**Por que `checkpoints` (padrão do GRF) não resolve no 5_vs_5:** cada zona paga só **uma vez por episódio**. Depois do primeiro ataque, ou do primeiro gol, não há mais shaping pelos ~2900 passos restantes. Além disso, ela paga por jogador controlado. Com 4 jogadores, isso incentiva cada um a carregar a bola em vez de passar.

## 2. Ferramenta principal: shaping baseado em potencial

Ng, Harada e Russell (1999) mostraram que, se o termo extra tem a forma

```
F(s, s') = γ·Φ(s') − Φ(s)
```

para qualquer função Φ do estado, **a política ótima não muda**. O shaping só acelera o aprendizado e não cria "atalhos" que o agente possa explorar. Por exemplo, ele não consegue ganhar recompensa andando em círculos, porque a soma de F ao longo de um ciclo é ≈ 0.

Isso é ótimo para o relatório: a escolha do shaping vira um argumento teórico, não "achamos que ficaria melhor".

Potenciais candidatos, todos calculáveis do raw (`with_raw=True` no cuju):

| Potencial Φ(s) | Significado | Comportamento que incentiva |
|---|---|---|
| `Φ_bola = α · x_bola` | quanto a bola está avançada no campo (x ∈ [−1, 1]) | levar a bola para frente, por passe ou condução, e afastá-la do próprio gol |
| `Φ_posse = β · [+1 se nossa, −1 se deles, 0 se livre]` | quem tem a bola | recuperar a bola (+2β) e não perdê-la (−2β) |
| `Φ_ameaça = δ · posse_nossa · exp(−d(bola, gol adversário))` | perigo de gol | entrar na área com a bola |

Valores iniciais típicos são α ≈ 0.1–0.5 e β ≈ 0.05: pequenos perto do ±1 de um gol. Atenção: *dentro de um ciclo* o shaping de potencial soma zero, mas o agente ainda recebe crédito imediato. É exatamente isso que queremos.

**Condições para a invariância valer (citar no relatório):**
- O γ dentro de F tem que ser **o mesmo γ do PPO**. No sweep de γ (R5), F muda junto.
- Φ depende **só do estado**, nunca da ação nem do tempo de treino. O peso w(t) da seção 7 quebra isso formalmente durante o decaimento; ver a seção 6.
- **O shaping adianta o gol, não soma por cima dele.** Depois de um gol a bola volta para x = 0 no kick-off e a posse muda. Φ devolve o progresso, e uma jogada que termina em gol rende ≈ +1 no total, não +1 + bônus.
- O termo (γ−1)·Φ por passo **se cancela no retorno descontado** que o PPO otimiza. Ele só aparece em logs não descontados (pode chegar a ~2 por partida). Logar o retorno de shaping **descontado**, senão a linha de base parece ≠ 0.

**Posse pelo último toque:** durante um passe ou chute a bola fica livre (`ball_owned_team = −1`). Com a posse instantânea, todo passe e todo chute pagariam −β na hora. Usar Φ_posse com o **último time que tocou na bola**, que só muda quando o adversário a domina.

## 3. Termos que **não** são de potencial (usar com cuidado)

| Termo | Risco | Uso |
|---|---|---|
| + por chute | o agente chuta de qualquer lugar | evitar, ou só se Φ_ameaça não bastar |
| + por passe completo | passes inúteis para trás | se usar, só passes que avançam a bola (já coberto por Φ_bola) |
| − por bola fora / cartão / falta (sliding) | baixo | penalidades pequenas (−0.01 a −0.05) se o comportamento aparecer |
| + por passo com posse (TiZero: +0.0001) | ficar parado com a bola no próprio campo | só se o agente perder a bola demais; Φ_posse já cobre isso |
| − por jogadores agrupados (TiZero: −0.001) | baixo | se os 4 controlados virarem um "enxame" atrás da bola, o que é comum em parameter sharing |

Regra: **cada termo só entra quando um comportamento observado motiva**. Exemplo: "aos 20M passos, 4 jogadores a menos de 0.1 da bola em 70% do tempo (figura X) → penalidade de agrupamento".

## 4. Recompensa individual × de equipe

Com 4 jogadores controlados por uma política compartilhada:
- **Recompensa de equipe** (todos recebem o mesmo r): é cooperativa e incentiva o passe. O crédito por jogador fica mais ruidoso.
- **Recompensa individual** (ex.: checkpoint por jogador): crédito mais claro, mas gera "fominha".

**Proposta:** gols e potenciais compartilhados pela equipe. Um crítico que vê o estado global (MAPPO) é um experimento à parte, caso o crédito vire problema.

## 5. Fator de desconto e horizonte

- Horizonte efetivo ≈ 1/(1−γ). Com γ = 0.99 são ~100 passos = **10 s de jogo**, curto para ligar uma recuperação de bola ao gol que vem depois.
- Proposta: **γ = 0.995** (~200 passos, 20 s), com GAE λ = 0.95. Testar 0.99 × 0.995 × 0.999 (TiZero usou 0.999 com LSTM).
- **Fim da partida: terminal ou bootstrap?** Depende de a política saber o tempo (Pardo et al. 2018, *Time Limits in RL*). As duas combinações coerentes:
  - **Com relógio** (`steps_left` na observação, nossa escolha): o limite de 3000 passos faz parte da tarefa → o fim é **terminal** (V = 0). O estado `steps_left = 0` nunca é treinado, então fazer bootstrap a partir dele seria ruído.
  - **Sem relógio** (`simple115v2` puro): o fim é invisível para o agente → bootstrap com V(`information['final_observation']`), porque no cuju, com auto-reset, a observação seguinte já é de outra partida.
  - Não misturar as duas. A combinação usada precisa ser declarada no relatório.
- Se a recompensa for só placar, o valor depende de **placar e tempo restante** (vencendo por 1 faltando 30 s, segurar a bola vale mais). Esses dois números **não estão** no `simple115v2` → adicionar `steps_left/3000` e o saldo de gols às features.

## 6. Normalização (detalhe de PPO que importa)

- Normalizar as vantagens por minibatch (já é padrão).
- Normalizar o retorno, ou usar normalização do valor (PopArt ou running std), porque a escala muda quando o shaping é desligado.
- Clipar a recompensa por passo não é necessário. Os termos já são pequenos e ±1 por gol é raro.
- **Duas cabeças de valor**: V_gols e V_shaping, com A = A_gols + w(t)·A_shaping. Assim, decair w não invalida o crítico, que continua prevendo cada parte separadamente.

## 7. Cronograma do shaping (andaime → objetivo real)

```
r_t = scoring_t + w(t) · [ γΦ(s_{t+1}) − Φ(s_t) ] + penalidades
w(t): 1.0 na fase L0 → decai linearmente → 0 (ou 0.1) nas fases de self-play/PFSP
```

No self-play final, a recompensa é praticamente só o placar. É soma-zero entre os times, então o que um ganha o outro perde, e o agente não consegue "inflar" recompensa contra um adversário congelado.

## 8. Experimentos de recompensa (todos contra o builtin, mesmo orçamento de passos, mesmas seeds)

| Run | Recompensa | Pergunta que responde |
|---|---|---|
| **R0** | `scoring` | O PPO padrão aprende só com o sinal esparso? (esperado: lento ou não) |
| **R1** | `scoring,checkpoints` (padrão GRF) | O shaping padrão ajuda no 5v5? (esperado: só no início do episódio, veja a recompensa por fase do jogo) |
| **R2** | `scoring` + Φ_bola | O potencial de progresso resolve a esparsidade? |
| **R3** | R2 + Φ_posse | Recuperar/manter a bola melhora a defesa? (gols sofridos) |
| **R4** | R3 com w(t) decaindo até 0 | O agente mantém o desempenho sem o andaime? |
| **R5** | γ 0.995 × 0.999 × 1.0 sobre o melhor | Qual horizonte de crédito funciona (no estudo de Song et al., γ = 1 > 0.995 > 0.99; Raw Beast usou 0.999) |
| **R6** | melhor + termo de resultado `c·pontos(V/E/D)` no fim (ou alvo tanh(saldo), como o TamakEri) | O agente aprende a administrar o placar (segurar vencendo, arriscar perdendo)? |

**Cuidados com a comparação:**
- Contra o builtin a 0.05, os runs devem saturar (todos vencem fácil). A métrica que separa as variantes é o **cross-play** entre elas (agente R2 × agente R3) e contra o pool fixo.
- **2–3 seeds** por comparação principal. Com 1 seed, a diferença pode ser ruído.

**O que medir (para detectar reward hacking):**
- saldo de gols e taxa de vitória contra o conjunto fixo (random, builtin, template)
- retorno de shaping × retorno de gols separados (se o shaping sobe e os gols não, há hacking)
- posse %, x médio da bola, chutes/jogo, passes/jogo, distância média entre os jogadores controlados
- histograma de ações (o verificador avisa se uma ação passa de 90%)

## Referências
- Ng, Harada, Russell (1999). *Policy invariance under reward transformations: theory and application to reward shaping.* ICML.
- Kurach et al. (2020). *Google Research Football: A Novel RL Environment* (scoring × checkpoints).
- Lin et al. (2023). *TiZero* (recompensas de posse, passe e agrupamento; γ = 0.999).
