from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naruto_arena.agents.minimax_agent import MinimaxAgent, MinimaxConfig
from naruto_arena.data.characters import ALL_CHARACTERS
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate one minimax battle.")
    parser.add_argument("--game-seed", type=int, default=7)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--search-actions", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    characters = list(ALL_CHARACTERS.values())
    team_a = characters[:3]
    team_b = characters[3:6]
    state = create_initial_state(team_a, team_b, rng_seed=args.game_seed)
    agents = {
        0: MinimaxAgent(MinimaxConfig(depth=args.depth, max_actions=args.search_actions)),
        1: MinimaxAgent(MinimaxConfig(depth=args.depth, max_actions=args.search_actions)),
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
        f"game_seed={args.game_seed} depth={args.depth} winner={state.winner} "
        f"turn={state.turn_number} actions={actions_taken} stop_reason={stop_reason}"
    )
    for player in state.players:
        hp = ", ".join(f"{character.definition.name}:{character.hp}" for character in player.characters)
        print(f"player {player.player_id}: {hp}")


if __name__ == "__main__":
    main()

