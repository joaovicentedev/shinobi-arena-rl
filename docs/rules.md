# Regras Gerais Do Jogo

Este documento descreve apenas as regras gerais do motor. Ele não documenta personagens específicos.

## Partida

- O jogo é por turnos.
- Existem 2 jogadores.
- Cada jogador controla um time de 3 personagens.
- A ordem do time não importa.
- Um time não pode ter personagens duplicados.
- Cada personagem começa com 100 HP.
- Um personagem está morto quando chega a 0 HP.
- Um jogador vence quando todos os 3 personagens inimigos estão mortos.

## Turnos

- Apenas o jogador ativo pode agir.
- No início de cada turno, o jogador ativo ganha chakra igual ao número de seus personagens vivos.
- Exceção: no primeiro turno da partida, o jogador 0 ganha apenas 1 chakra.
- No primeiro turno do jogador 1, ele já ganha chakra normalmente pelo número de personagens vivos.
- Cada chakra ganho é sorteado independentemente entre os 4 tipos fixos, com mesma probabilidade.
- O jogador pode usar skills, reorganizar skills ou encerrar o turno.
- Cada personagem pode usar no máximo 1 skill nova por turno.
- Efeitos passivos e efeitos de skills usadas em turnos anteriores continuam funcionando normalmente.
- Ao encerrar o turno:
  - efeitos temporários do jogador ativo avançam;
  - cooldowns do jogador ativo diminuem;
  - o turno passa para o outro jogador.
- Uma skill com cooldown 1 não pode ser usada no próximo turno do mesmo personagem.

## Chakra

Existem 4 tipos fixos de chakra:

- `ninjutsu`
- `taijutsu`
- `genjutsu`
- `bloodline`

Também existe custo de `random chakra`.

Regras:

- Custo fixo exige o tipo exato de chakra.
- Custo random pode ser pago com qualquer tipo disponível.
- Skills podem custar:
  - nenhum chakra;
  - chakra fixo;
  - chakra random;
  - combinação de fixo e random.
- O pagamento de chakra é validado pelo motor.
- O chakra random é escolhido no momento da ação.

## Informação Oculta

- O estado real do jogo contém o chakra completo dos dois jogadores.
- Um jogador não observa diretamente o chakra do oponente.
- O jogador deve estimar o chakra inimigo com base em:
  - chakra ganho provável por turno;
  - número de personagens vivos do oponente;
  - custos das skills visíveis usadas pelo oponente;
  - efeitos visíveis de remoção, roubo ou gasto de chakra.
- Algumas ações ou efeitos podem ser invisíveis; nesses casos, a estimativa do chakra inimigo pode ficar incompleta ou incorreta.
- Agentes que simulam jogo com informação perfeita podem ser úteis para debug, mas não representam fielmente a informação disponível para um jogador real.
- Para agentes competitivos, o motor deve oferecer uma observação parcial do estado, escondendo o chakra inimigo real.

## Personagens

Cada personagem possui:

- identificador;
- nome;
- descrição;
- HP atual;
- HP máximo;
- lista de skills;
- ordem atual das skills;
- cooldowns;
- status temporários;
- passivas registradas;
- passivas já ativadas.

## Skills

Cada skill pode definir:

- nome;
- descrição;
- cooldown;
- custo de chakra;
- classes/tags;
- regra de alvo;
- requisitos;
- requisitos por alvo;
- efeitos;
- duração;
- marcador de status;
- substituição condicional;
- modificadores condicionais.

## Classes De Skill

Classes suportadas pelo motor incluem:

- `Physical`
- `Chakra`
- `Mental`
- `Melee`
- `Ranged`
- `Instant`
- `Action`
- `Affliction`
- `Stun`
- `Passive`
- `Unremovable`
- `Unique`

As classes podem ser usadas por efeitos, imunidades, stuns específicos, filtros de alvo e lógica condicional.

## Alvos

Uma skill pode mirar:

- ninguém;
- o próprio usuário;
- um inimigo;
- todos os inimigos;
- um aliado;
- todos os aliados.

O motor valida se os alvos escolhidos correspondem à regra da skill.

## Efeitos

O motor suporta efeitos como:

- dano direto;
- dano piercing;
- cura;
- stun total;
- stun por classe de skill;
- redução de dano fixa;
- redução de dano percentual;
- redução de dano unpierceable;
- dano ao longo do tempo;
- remoção ou roubo de chakra;
- cooldown;
- invulnerabilidade;
- marcadores de status;
- efeitos passivos;
- substituição condicional de skill;
- aumento condicional de dano.

## Dano

O dano é aplicado pelo motor.

Ordem geral:

- calcula dano base;
- aplica bônus ou penalidades condicionais;
- verifica invulnerabilidade;
- aplica reduções de dano;
- respeita piercing e unpierceable;
- reduz HP do alvo;
- verifica gatilhos passivos;
- verifica vencedor.

## Piercing E Unpierceable

- Dano piercing ignora reduções de dano normais.
- Dano piercing não ignora reduções unpierceable.
- Invulnerabilidade impede dano, exceto quando uma regra específica manda ignorar defesas.

## Passivas

- Passivas podem começar registradas e inativas.
- Passivas podem ativar por condições do jogo.
- Uma passiva pode ativar apenas uma vez se assim for definida.
- Passivas unremovable não devem ser removidas por efeitos comuns.

## Reorganização De Skills

O motor suporta uma ação genérica de reorganização:

- escolhe um personagem;
- escolhe uma skill;
- escolhe uma nova posição;
- a ordem da lista de skills do personagem é alterada.

Essa regra existe para permitir ajustes de timing e combos durante a partida.
Passivas também podem ser reorganizadas porque podem alterar timing de dano, buffs ou modificadores condicionais. Isso não torna passivas manualmente utilizáveis como ações.
- Cada jogador pode reorganizar skills no máximo 3 vezes por turno.

## Determinismo

- O jogo usa RNG para sortear chakra.
- O estado carrega um RNG inicializado por seed.
- Com a mesma seed e as mesmas ações, a partida é reprodutível.
