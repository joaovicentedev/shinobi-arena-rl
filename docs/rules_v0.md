# Rules V0: Idea 0 RL Lab

This document tracks the current simplified Naruto Arena RL setup. Idea 0 is a curriculum/debug environment for validating RL behavior before adding complex mechanics back.

## Scope

- 18 implemented hand-authored characters only.
- 3v3 battles.
- HP, cooldowns, target selection, random chakra, and hidden enemy chakra.
- Attack skills with direct damage.
- Simple defense effects that reduce incoming damage by a flat amount.
- DOT effects that deal deterministic damage over a fixed duration.
- Effects resolve deterministically at turn boundaries.

## Removed Mechanics

These mechanics are intentionally unavailable in V0:

- Stun.
- Buffs and debuffs.
- Chakra steal, chakra remove, and chakra gain manipulation.
- Invulnerability.
- Counters and reflection.
- Skill replacement.
- Conditional damage/modifier systems.
- Passive triggers.
- Stack reorder.
- Enemy chakra belief.
- JSON-loaded full roster characters.

Add removed mechanics back one at a time only after the V0 PPO setup is stable.

## Actions

Available action kinds:

- `END_TURN`
- `USE_SKILL`
- `GET_CHAKRA`

Removed action kinds:

- `REORDER_STACK`

`USE_SKILL` is selected through joint scoring:

```text
actor_slot x skill_slot x target_code x random_chakra_code
```

Independent actor/skill/target heads are not the main policy for the token model.

## Observation

Default observation version:

```text
attention_idea0_tokens_v1
```

Token layout:

- 1 global token.
- 3 allied character tokens.
- 3 enemy character tokens.
- 27 allied skill tokens.
- 27 enemy skill tokens.
- Allied visible effect/DOT stack tokens.
- Enemy visible effect/DOT stack tokens.

Enemy real chakra is hidden. There is no enemy chakra belief input.

Dead characters are not masked from attention. They remain in the token stream with:

- `alive=0`
- `hp=0`
- `can_use=0`

The token model uses semantic embeddings for:

- token type
- side
- character slot
- character id
- skill slot
- skill id
- stack position

## Reward

Reward remains simple:

- Terminal win: `+1`
- Terminal loss: `-1`
- Enemy HP damage: positive shaping
- Own HP damage: negative shaping
- Enemy KO: positive shaping
- Ally KO: negative shaping
- Small action cost for `USE_SKILL` and `GET_CHAKRA`

## Metrics

Tracked match metrics include:

- win rate
- model-as-player-0 win rate
- model-as-player-1 win rate
- average actions per turn
- average skills used per turn
- average unused chakra at end turn
- damage per chakra
- attacks into defense
- wasted overkill
- DOT value
- defense value

## Smoke Commands

```bash
uv run python scripts/simulate_random_battle.py --max-actions 80
uv run python scripts/train_rl_pytorch.py --episodes 1 --max-actions 40 --save-path /tmp/idea0.pt
uv run python scripts/evaluate_rl_benchmarks.py --model-path /tmp/idea0.pt --matches-per-benchmark 1 --random-mirror-matches 1 --paired-sides
uv run python scripts/simulate_rl_match.py --model-path /tmp/idea0.pt --max-actions 40 --output /tmp/idea0_replay.json
```
