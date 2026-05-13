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