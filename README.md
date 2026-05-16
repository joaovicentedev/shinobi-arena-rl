# Naruto Arena Simulator

A deterministic, turn-based Python battle engine inspired by Naruto Arena.

## Idea 0 RL Lab

The active RL environment is an intentionally simplified curriculum/debug lab. It keeps the 18 implemented characters, 3v3 battles, random chakra, hidden enemy chakra, target selection, cooldowns, direct attack skills, simple defense, and DOT effects.

Removed mechanics are deliberately unavailable in the Idea 0 engine: stun, buffs, debuffs, chakra steal/remove, invulnerability, counters, reflection, skill replacement, conditional modifiers, and stack reorder. Add these back one at a time only after the simplified PPO setup is stable.

The default RL observation is `attention_idea0_tokens_v1`: global token, 3 allied character tokens, 3 enemy character tokens, 27 allied skill tokens, 27 enemy skill tokens, and visible effect/DOT stack tokens for both sides. Enemy real chakra remains hidden and there is no enemy chakra belief input. Dead characters stay in the token stream with `alive=0`, `hp=0`, and unusable skill tokens.

Useful smoke commands:

```bash
uv run python scripts/train_rl_pytorch.py --episodes 1 --max-actions 40 --save-path /tmp/idea0.pt
uv run python scripts/evaluate_rl_benchmarks.py --model-path /tmp/idea0.pt --matches-per-benchmark 1 --random-mirror-matches 1 --paired-sides
uv run python scripts/simulate_rl_match.py --model-path /tmp/idea0.pt --max-actions 40 --output /tmp/idea0_replay.json
```

## Setup

```bash
uv sync --extra dev
```

## Run Tests

```bash
uv run pytest
```

## Simulate A Match

```bash
uv run python scripts/simulate_random_battle.py
```
