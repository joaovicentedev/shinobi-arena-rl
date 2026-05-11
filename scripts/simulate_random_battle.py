from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naruto_arena.agents.heuristic_agent import SimpleHeuristicAgent
from naruto_arena.agents.random_agent import RandomAgent
from naruto_arena.data.characters import (
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate a seeded random Naruto Arena battle.")
    parser.add_argument("--game-seed", type=int, default=7, help="Seed for chakra generation.")
    parser.add_argument("--p0-seed", type=int, default=1, help="Seed for player 0 agent decisions.")
    parser.add_argument("--p1-seed", type=int, default=2, help="Seed for player 1 agent decisions.")
    parser.add_argument("--max-actions", type=int, default=500, help="Maximum actions before stopping.")
    parser.add_argument(
        "--allow-reorder",
        action="store_true",
        help="Allow baseline agents to choose ReorderSkillsAction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        rng_seed=args.game_seed,
    )
    agents = {
        0: SimpleHeuristicAgent(seed=args.p0_seed, allow_reorder=args.allow_reorder),
        1: RandomAgent(seed=args.p1_seed, allow_reorder=args.allow_reorder),
    }
    actions_taken = 0
    stop_reason = "max_actions"
    for actions_taken in range(1, args.max_actions + 1):
        if state.winner is not None:
            stop_reason = "winner"
            actions_taken -= 1
            break
        player_id = state.active_player
        action = agents[player_id].choose_action(state, player_id)
        apply_action(state, action)
    if state.winner is not None:
        stop_reason = "winner"
    print(
        f"game_seed={args.game_seed} p0_seed={args.p0_seed} "
        f"p1_seed={args.p1_seed} winner={state.winner} turn={state.turn_number} "
        f"actions={actions_taken} stop_reason={stop_reason}"
    )
    for player in state.players:
        hp = ", ".join(f"{c.definition.name}:{c.hp}" for c in player.characters)
        print(f"player {player.player_id}: {hp}")


if __name__ == "__main__":
    main()
