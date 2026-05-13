.PHONY: setup setup-rl lint format check test test-cov simulate simulate-minimax tournament-minimax
.PHONY: train-rl simulate-rl tournament-rl compare-rl clean

setup:
	uv sync --extra dev

setup-rl:
	uv sync --extra dev --extra rl

lint:
	uv run --extra dev ruff check naruto_arena scripts tests

format:
	uv run --extra dev ruff format naruto_arena scripts tests

check:
	uv run --extra dev ruff check naruto_arena scripts tests
	uv run --extra dev ruff format --check naruto_arena scripts tests
	uv run --extra dev pytest

test:
	uv run --extra dev pytest

test-cov:
	uv run --extra dev pytest --cov

simulate:
	uv run python scripts/simulate_random_battle.py $(ARGS)

simulate-minimax:
	uv run python scripts/simulate_minimax_battle.py $(ARGS)

tournament-minimax:
	uv run python scripts/tournament_minimax.py $(ARGS)

train-rl:
	uv run --extra rl python scripts/train_rl_pytorch.py $(ARGS)

simulate-rl:
	uv run --extra rl python scripts/simulate_rl_match.py $(ARGS)

tournament-rl:
	uv run --extra rl python scripts/tournament_rl.py $(ARGS)

compare-rl:
	uv run --extra rl python scripts/compare_rl_models.py $(ARGS)

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache
