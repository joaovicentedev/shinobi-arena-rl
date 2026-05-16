# General Game Rules

This document describes the general game-engine rules. It does not document
specific characters.

## Match

- The game is turn-based.
- There are 2 players.
- Each player controls a team of 3 characters.
- Team order does not matter.
- A team cannot contain duplicate characters.
- Each character starts with 100 HP.
- A character is dead when they reach 0 HP.
- A player wins when all 3 enemy characters are dead.

## Turns

- Only the active player can act.
- At the start of each turn, the active player gains chakra equal to the number
  of their living characters.
- Exception: on the first turn of the match, player 0 gains only 1 chakra.
- On player 1's first turn, they gain chakra normally based on the number of
  living characters.
- Each chakra gained is rolled independently among the 4 fixed chakra types,
  with equal probability.
- The player can use skills, reorder skills, or end the turn.
- Each character can use at most 1 new skill per turn.
- Passive effects and effects from skills used on previous turns continue to
  operate normally.
- When the active player ends their turn:
  - queued skills are resolved in the current skill-stack order;
  - temporary effects owned by the active player advance;
  - cooldowns owned by the active player decrease;
  - the turn passes to the other player.
- A skill with cooldown 1 cannot be used on that same character's next turn.

## End of Turn and Skill Queue

After choosing which skills the team will use and their targets, the player ends
the turn. In the original Naruto-Arena interface, this is done with the
`Press When Ready` button, which opens the skill queue.

The skill queue is still part of the player's turn, so the turn timer continues
to run while the player is in that screen.

In the skill queue:

- The player chooses which specific chakra will be spent to pay random chakra
  costs.
- Skills are executed from left to right when the queue is confirmed.
- The player may reorder queued skills before confirming the turn.
- Confirming the queue completes the turn.
- Cancelling the queue returns the player to skill selection.
- If the timer expires before the turn is confirmed, no chosen skills are used
  that turn.

## Chakra

Chakra is used to perform skills. Every skill can require a different chakra
cost.

There are 4 fixed chakra types:

- `bloodline`
- `ninjutsu`
- `taijutsu`
- `genjutsu`

Each generated chakra has a 25% chance to be any one of these types. Chakra
generation is random, so a player can receive multiple chakra of the same type
in the same turn.

Some skills also require `random chakra`. Random chakra is not a fifth chakra
type. It is a flexible cost that can be paid with any available fixed chakra
type.

Rules:

- A fixed cost requires the exact chakra type.
- A random cost can be paid with any available chakra type.
- Skills can cost:
  - no chakra;
  - fixed chakra;
  - random chakra;
  - a combination of fixed and random chakra.
- Chakra payment is validated by the engine.
- The specific chakra used to pay random costs is chosen when the action is
  committed.
- A player may exchange 5 owned chakra of any types for 1 chakra of a chosen
  fixed type. The exchanged chakra are removed from the player's pool, then the
  chosen chakra is added.

## Hidden Information

- The real game state contains the full chakra pool of both players.
- A player does not directly observe the opponent's chakra.
- The player must estimate enemy chakra based on:
  - likely chakra gained each turn;
  - the number of living enemy characters;
  - visible skill costs paid by the opponent;
  - visible chakra removal, chakra steal, or chakra spending effects.
- Some actions or effects can be invisible. In those cases, the enemy chakra
  estimate can be incomplete or incorrect.
- Agents that simulate the game with perfect information can be useful for
  debugging, but they do not faithfully represent the information available to a
  real player.
- For competitive agents, the engine should expose a partial observation that
  hides the opponent's real chakra pool.

## Characters

Each character has:

- an identifier;
- a name;
- a description;
- current HP;
- maximum HP;
- a skill list;
- current skill order;
- cooldowns;
- temporary statuses;
- registered passives;
- passives that have already triggered.

## Skills

Each skill can define:

- a name;
- a description;
- a cooldown;
- a chakra cost;
- classes or tags;
- a target rule;
- requirements;
- per-target requirements;
- effects;
- duration;
- a status marker;
- conditional replacement;
- conditional modifiers.

## Common Terminology

These terms commonly appear in skill descriptions.

- Damage: Reduces health by either a fixed amount or a percentage.
- Piercing damage: Damage that ignores normal damage reduction.
- Affliction damage: Damage that ignores normal damage reduction and
  destructible defense.
- Increased damage: A skill may increase damage dealt or damage received,
  usually by a fixed amount or a percentage.
- Stun: Prevents a character from using affected skills for the stun duration.
- Damage reduction: Reduces incoming damage by a fixed amount or percentage
  after that turn's damage is calculated.
- Invulnerable: The character is not a valid target for enemy skills.
- Heal: Restores health by either a fixed amount or a percentage.
- Remove chakra or lose chakra: Removes a fixed or random amount of chakra from
  the opponent's chakra pool.
- Steal chakra: Removes chakra from the opponent and adds it to the user's
  chakra pool.
- Reflect: Returns the effects of a skill back onto the character that targeted
  the reflecting character.
- Counter: Negates an incoming skill.
- Remove an effect: Completely negates that effect on a character.
- Ignore an effect: Causes the ignored effect to have no impact. The ignored
  effect is not removed or negated globally.
- Destructible defense: Gives a character a temporary pool of protection that
  must be depleted before the character takes damage.
- Copy: Copies another skill, replacing an existing skill for the duration of
  the copy.
- Invisible: The opponent does not see icons for the skill while the invisible
  effect is active.
- Increase or decrease duration: Modifies the duration of a skill or effect.
  Related effects are modified accordingly.

## Cooldown

A skill's cooldown is the number of turns that skill cannot be used after it is
used.

For example, a skill with cooldown 4 is unavailable for the following 4 turns of
that same character after it is used.

## Skill Classes

Classes further describe a skill beyond its text description. They are mainly
important when determining whether one skill affects another skill during a
match.

Classes supported by the engine include:

- `Physical`
- `Chakra`
- `Mental`
- `Melee`
- `Ranged`
- `Instant`
- `Action`
- `Control`
- `Affliction`
- `Stun`
- `Passive`
- `Unremovable`
- `Unique`

Classes can be used by effects, immunities, class-specific stuns, target
filters, and conditional logic.

### Distance

- Melee: The skill is carried out at close range to the target and user.
- Ranged: The skill is carried out far away from the user.

### Skill Type

- Physical: The skill uses matter to create its result.
- Chakra: The skill uses energy to create its result.
- Affliction: The skill uses persisting matter or energy, such as poison or
  flames, to create its result.
- Mental: The skill uses thought or another metaphysical force to create its
  result.

### Special Classes

- Unique: The skill cannot be duplicated by another character, or is nearly
  impossible to duplicate.
- `*` suffix: If a class is followed by an asterisk, only part of the skill has
  that class.

### Persistence Type

- Instant: The skill resolves during queue confirmation, or is applied on the
  first resolution turn and has no further connection to the caster for the rest
  of its duration.
- Action: The skill lasts multiple turns and requires the caster's attention
  each turn. If the caster loses contact with the target because of a stun or
  invulnerability, the action has no effect for that turn. Since the skill is
  performed again each turn, it can continue once contact with the target is
  restored.
- Control: The skill requires constant contact between caster and target. If
  contact is broken, the skill ends. A control is only cast once, when it first
  makes contact with the target, similar to an instant skill.

## Targets

A skill can target:

- nobody;
- the user;
- one enemy;
- all enemies;
- one ally;
- all allies.

The engine validates whether the selected targets match the skill's target rule.

## Effects

The engine supports effects such as:

- direct damage;
- piercing damage;
- healing;
- total stun;
- skill-class stun;
- fixed damage reduction;
- percentage damage reduction;
- unpierceable damage reduction;
- damage over time;
- chakra removal or chakra steal;
- cooldown modification;
- invulnerability;
- status markers;
- passive effects;
- conditional skill replacement;
- conditional damage increase.

## Damage

Damage is applied by the engine.

General order:

- calculate base damage;
- apply conditional bonuses or penalties;
- check invulnerability;
- apply damage reduction;
- respect piercing and unpierceable rules;
- reduce the target's HP;
- check passive triggers;
- check for a winner.

## Piercing and Unpierceable

- Piercing damage ignores normal damage reduction.
- Piercing damage does not ignore unpierceable reduction.
- Invulnerability prevents damage unless a specific rule says to ignore
  defenses.

## Passives

- Passives can start registered and inactive.
- Passives can trigger based on game conditions.
- A passive can trigger only once if it is defined that way.
- Unremovable passives should not be removed by common removal effects.

## Skill Reordering

The engine supports a generic reorder action:

- choose a character;
- choose one of that player's used skills;
- choose a new position;
- the player's used-skill stack order is changed.

This rule exists to allow timing and combo adjustments during the match.
Character skill lists are fixed and are not changed by reorder actions.

Each player can reorder skills at most 3 times per turn. The same character
skill can be reordered at most once per turn.

## Determinism

- The game uses RNG to roll chakra.
- The state carries an RNG initialized by seed.
- With the same seed and the same actions, the match is reproducible.
