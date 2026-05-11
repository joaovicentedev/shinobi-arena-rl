# Minimax Simples

Este projeto inclui um agente minimax simples para servir como baseline de busca.

## Ideia

O minimax tenta escolher uma ação olhando alguns passos à frente:

- no turno do jogador raiz, escolhe a ação com maior avaliação;
- no turno do oponente, assume que o oponente escolherá a pior resposta para o jogador raiz;
- quando chega na profundidade limite, usa uma função heurística para avaliar o estado.

## Função De Avaliação

A avaliação atual é simples e usa:

- vitória ou derrota;
- diferença de personagens vivos;
- diferença de HP total;
- diferença de chakra;
- bônus pequeno por invulnerabilidade;
- bônus pequeno por redução de dano.

Ela não entende profundamente todos os combos. O objetivo é ser um baseline fácil de debugar.

## Chakra Aleatório

O jogo tem chakra aleatório no início dos turnos.

Nesta primeira versão:

- o estado é copiado durante a busca;
- o RNG copiado sorteia chakra ao simular `EndTurnAction`;
- pagamentos de custo random são escolhidos de forma determinística;
- a busca não cria chance nodes explícitos.

Isso é mais simples que expectimax, mas já permite comparar jogadas.

## Ações Consideradas

Por padrão, o agente ignora `ReorderSkillsAction`, porque reorder não avança turno e aumenta muito o espaço de busca.

O agente considera:

- usar skill legal;
- encerrar turno.

## Limitações

- Não modela chance nodes formalmente.
- Usa profundidade baixa.
- A avaliação ainda é genérica.
- Pode perder combos longos.
- Pode subestimar skills defensivas ou de setup.

## Próximos Passos

- Trocar minimax por expectimax para modelar chakra aleatório.
- Melhorar a função de avaliação por tags e efeitos.
- Adicionar poda alpha-beta.
- Criar action ordering mais forte.
- Usar o minimax para gerar dados de imitation learning.
- Comparar win rate contra agentes random e heurísticos.

## Comandos

Simular uma partida:

```bash
make simulate-minimax ARGS="--game-seed 7 --depth 2"
```

Rodar torneio entre todas as composições de 3 personagens:

```bash
make tournament-minimax ARGS="--matches-per-pair 1 --depth 1"
```

Por padrão, o torneio também salva um relatório completo em:

```text
reports/minimax_tournament.json
```

Para escolher outro caminho:

```bash
make tournament-minimax ARGS="--matches-per-pair 1 --depth 1 --output reports/meu_torneio.json"
```

O jogo não tem empate. Quando uma simulação bate `--max-actions` sem vencedor, o relatório marca a partida como `unfinished`.

O JSON mostra duas taxas:

- `overall_win_rate`: vitórias divididas por todos os jogos, incluindo unfinished.
- `resolved_win_rate`: vitórias divididas apenas por jogos que terminaram com vencedor.
