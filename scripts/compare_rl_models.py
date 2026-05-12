from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.data.characters import ALL_CHARACTERS
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action


@dataclass
class ModelStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    unfinished: int = 0
    player_0_games: int = 0
    player_0_wins: int = 0
    player_1_games: int = 0
    player_1_wins: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def resolved_games(self) -> int:
        return self.wins + self.losses

    @property
    def resolved_win_rate(self) -> float:
        return self.wins / self.resolved_games if self.resolved_games else 0.0

    @property
    def player_0_win_rate(self) -> float:
        return self.player_0_wins / self.player_0_games if self.player_0_games else 0.0

    @property
    def player_1_win_rate(self) -> float:
        return self.player_1_wins / self.player_1_games if self.player_1_games else 0.0

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["win_rate"] = self.win_rate
        data["resolved_games"] = self.resolved_games
        data["resolved_win_rate"] = self.resolved_win_rate
        data["player_0_win_rate"] = self.player_0_win_rate
        data["player_1_win_rate"] = self.player_1_win_rate
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two RL checkpoints head-to-head.")
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--label-a", default="model_a")
    parser.add_argument("--label-b", default="model_b")
    parser.add_argument("--matches-per-pair", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax.")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rl_compare.json"),
        help="Path to write the JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    characters = sorted(ALL_CHARACTERS.values(), key=lambda character: character.id)
    teams = list(itertools.combinations(characters, 3))
    total_expected_games = len(teams) * len(teams) * args.matches_per_pair * 2
    stats = {args.label_a: ModelStats(), args.label_b: ModelStats()}
    agents = {
        args.label_a: RlAgent(args.model_a, deterministic=not args.sample, seed=args.seed * 2 + 1),
        args.label_b: RlAgent(args.model_b, deterministic=not args.sample, seed=args.seed * 2 + 2),
    }
    games: list[dict[str, Any]] = []
    total_games = 0
    started_at = time.monotonic()

    for team_0 in teams:
        for team_1 in teams:
            for match_index in range(args.matches_per_pair):
                base_seed = args.seed + total_games
                for player_0_label, player_1_label in (
                    (args.label_a, args.label_b),
                    (args.label_b, args.label_a),
                ):
                    winner, actions = simulate_match(
                        list(team_0),
                        list(team_1),
                        base_seed,
                        {0: agents[player_0_label], 1: agents[player_1_label]},
                        args.max_actions,
                    )
                    total_games += 1
                    update_stats(stats, player_0_label, player_1_label, winner)
                    games.append(
                        {
                            "game_index": total_games,
                            "match_index": match_index,
                            "seed": base_seed,
                            "player_0_model": player_0_label,
                            "player_1_model": player_1_label,
                            "team_0": team_key(team_0),
                            "team_1": team_key(team_1),
                            "winner_player": winner,
                            "winner_model": winner_model(player_0_label, player_1_label, winner),
                            "actions": actions,
                            "unfinished": winner is None,
                        }
                    )
                    if (
                        total_games == 1
                        or total_games % args.log_interval == 0
                        or total_games == total_expected_games
                    ):
                        log_progress(total_games, total_expected_games, started_at)

    write_report(args, characters, teams, stats, games, total_games)
    print_summary(args, stats, total_games, len(teams))


def simulate_match(
    team_0: list[CharacterDefinition],
    team_1: list[CharacterDefinition],
    seed: int,
    agents: dict[int, RlAgent],
    max_actions: int,
) -> tuple[int | None, int]:
    state = create_initial_state(team_0, team_1, rng_seed=seed)
    actions = 0
    for actions in range(1, max_actions + 1):
        if state.winner is not None:
            return state.winner, actions - 1
        player_id = state.active_player
        action = agents[player_id].choose_action(state, player_id)
        apply_action(state, action)
    return state.winner, actions


def update_stats(
    stats: dict[str, ModelStats],
    player_0_label: str,
    player_1_label: str,
    winner: int | None,
) -> None:
    stats[player_0_label].games += 1
    stats[player_0_label].player_0_games += 1
    stats[player_1_label].games += 1
    stats[player_1_label].player_1_games += 1
    if winner == 0:
        stats[player_0_label].wins += 1
        stats[player_0_label].player_0_wins += 1
        stats[player_1_label].losses += 1
    elif winner == 1:
        stats[player_1_label].wins += 1
        stats[player_1_label].player_1_wins += 1
        stats[player_0_label].losses += 1
    else:
        stats[player_0_label].unfinished += 1
        stats[player_1_label].unfinished += 1


def write_report(
    args: argparse.Namespace,
    characters: list[CharacterDefinition],
    teams: list[tuple[CharacterDefinition, ...]],
    stats: dict[str, ModelStats],
    games: list[dict[str, Any]],
    total_games: int,
) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),
            "label_a": args.label_a,
            "label_b": args.label_b,
            "characters": len(characters),
            "teams": len(teams),
            "games": total_games,
            "matches_per_pair": args.matches_per_pair,
            "seed": args.seed,
            "max_actions": args.max_actions,
            "deterministic": not args.sample,
        },
        "characters": [
            {"id": character.id, "name": character.name}
            for character in characters
        ],
        "models": {
            label: model_stats.to_json()
            for label, model_stats in stats.items()
        },
        "games": games,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def print_summary(
    args: argparse.Namespace,
    stats: dict[str, ModelStats],
    total_games: int,
    team_count: int,
) -> None:
    print(
        f"models={args.label_a},{args.label_b} teams={team_count} "
        f"games={total_games} matches_per_pair={args.matches_per_pair}"
    )
    print("model | resolved_wr | overall_wr | wins | losses | unfinished | p0_wr | p1_wr")
    for label in (args.label_a, args.label_b):
        model_stats = stats[label]
        print(
            f"{label} | {model_stats.resolved_win_rate:.3f} | {model_stats.win_rate:.3f} | "
            f"{model_stats.wins:>4} | {model_stats.losses:>6} | "
            f"{model_stats.unfinished:>10} | {model_stats.player_0_win_rate:.3f} | "
            f"{model_stats.player_1_win_rate:.3f}"
        )
    print(f"json_report={args.output}")


def log_progress(done: int, total: int, started_at: float) -> None:
    percent = 100 * done / total
    elapsed = time.monotonic() - started_at
    print(
        f"progress={percent:6.2f}% games={done}/{total} elapsed={elapsed:.1f}s",
        flush=True,
    )


def winner_model(player_0_label: str, player_1_label: str, winner: int | None) -> str | None:
    if winner == 0:
        return player_0_label
    if winner == 1:
        return player_1_label
    return None


def team_key(team: tuple[CharacterDefinition, ...]) -> str:
    return " / ".join(sorted(character.name for character in team))


if __name__ == "__main__":
    main()
