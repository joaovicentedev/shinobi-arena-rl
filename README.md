# Naruto Arena Simulator

A deterministic, turn-based Python battle engine inspired by Naruto Arena. This repo currently contains only the game engine, simple agents, data definitions, tests, and a random battle script. There is no UI and no neural network code yet.

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

## Notes

- Chakra gain is random with equal probability across the 4 chakra types, and reproducible when `GameState` and agents are seeded.
- Chakra is explicit and validated through `ChakraPool`.
- Character behavior lives in skill and effect definitions under `naruto_arena/data`.
- The current character data includes only complete definitions for Naruto, Sakura, and Sasuke.
- `ReorderSkillsAction` changes a character's in-match skill order for future combo timing.

## TODO For RL/MCTS

- Add immutable state snapshots or copy-on-write transitions for tree search.
- Add compact observation encoders and action masks.
- Add rollout policies and pluggable reward shaping.
- Add match serialization for self-play datasets.
- Add benchmark tests for simulator throughput.
