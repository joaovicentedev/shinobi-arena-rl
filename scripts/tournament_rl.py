from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.data.characters import ALL_CHARACTERS
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action


@dataclass
class TeamStats:
    games: int = 0
    wins: int = 0
    losses: int = 0
    unfinished: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def resolved_games(self) -> int:
        return self.wins + self.losses

    @property
    def resolved_win_rate(self) -> float:
        return self.wins / self.resolved_games if self.resolved_games else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an RL tournament across all teams.")
    parser.add_argument("--matches-per-pair", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--model-path", type=Path, default=Path("models/naruto_actor_critic.pt"))
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax.")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rl_tournament.json"),
        help="Path to write the full JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    characters = sorted(ALL_CHARACTERS.values(), key=lambda character: character.id)
    teams = list(itertools.combinations(characters, 3))
    total_expected_games = len(teams) * len(teams) * args.matches_per_pair
    stats: dict[str, TeamStats] = defaultdict(TeamStats)
    total_games = 0
    started_at = time.monotonic()
    agents = {
        0: RlAgent(args.model_path, deterministic=not args.sample, seed=args.seed * 2 + 1),
        1: RlAgent(args.model_path, deterministic=not args.sample, seed=args.seed * 2 + 2),
    }

    for team_a in teams:
        for team_b in teams:
            for _ in range(args.matches_per_pair):
                seed = args.seed + total_games
                winner = simulate_match(
                    list(team_a),
                    list(team_b),
                    seed,
                    agents,
                    args.max_actions,
                )
                total_games += 1
                update_stats(stats, team_key(team_a), team_key(team_b), winner)
                if (
                    total_games == 1
                    or total_games % args.log_interval == 0
                    or total_games == total_expected_games
                ):
                    log_progress(total_games, total_expected_games, started_at)

    ranked = sorted(
        stats.items(),
        key=lambda item: (item[1].resolved_win_rate, item[1].wins, -item[1].unfinished),
        reverse=True,
    )
    write_report(
        args.output,
        characters,
        teams,
        ranked,
        total_games,
        args.matches_per_pair,
        args.seed,
        args.max_actions,
        args.model_path,
        deterministic=not args.sample,
    )
    print(
        f"characters={len(characters)} teams={len(teams)} games={total_games} "
        f"matches_per_pair={args.matches_per_pair} model={args.model_path}"
    )
    print("rank | resolved_wr | overall_wr | wins | losses | unfinished | games | team")
    for rank, (team, team_stats) in enumerate(ranked[: args.top], start=1):
        print(
            f"{rank:>4} | {team_stats.resolved_win_rate:.3f} | {team_stats.win_rate:.3f} | "
            f"{team_stats.wins:>4} | {team_stats.losses:>6} | {team_stats.unfinished:>10} | "
            f"{team_stats.games:>5} | {team}"
        )
    print(f"json_report={args.output}")


def write_report(
    output: Path,
    characters: list[CharacterDefinition],
    teams: list[tuple[CharacterDefinition, ...]],
    ranked: list[tuple[str, TeamStats]],
    total_games: int,
    matches_per_pair: int,
    seed: int,
    max_actions: int,
    model_path: Path,
    *,
    deterministic: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    team_ids_by_key = {team_key(team): [character.id for character in team] for team in teams}
    report = {
        "metadata": {
            "characters": len(characters),
            "teams": len(teams),
            "games": total_games,
            "matches_per_pair": matches_per_pair,
            "seed": seed,
            "max_actions": max_actions,
            "model_path": str(model_path),
            "deterministic": deterministic,
        },
        "characters": [
            {"id": character.id, "name": character.name}
            for character in characters
        ],
        "teams": [
            {
                "rank": rank,
                "team": team,
                "character_ids": team_ids_by_key[team],
                "games": team_stats.games,
                "wins": team_stats.wins,
                "losses": team_stats.losses,
                "unfinished": team_stats.unfinished,
                "overall_win_rate": team_stats.win_rate,
                "resolved_games": team_stats.resolved_games,
                "resolved_win_rate": team_stats.resolved_win_rate,
            }
            for rank, (team, team_stats) in enumerate(ranked, start=1)
        ],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def simulate_match(
    team_a: list[CharacterDefinition],
    team_b: list[CharacterDefinition],
    seed: int,
    agents: dict[int, RlAgent],
    max_actions: int,
) -> int | None:
    state = create_initial_state(team_a, team_b, rng_seed=seed)

    for _ in range(max_actions):
        if state.winner is not None:
            return state.winner
        player_id = state.active_player
        action = agents[player_id].choose_action(state, player_id)
        apply_action(state, action)
    return state.winner


def update_stats(
    stats: dict[str, TeamStats],
    team_a: str,
    team_b: str,
    winner: int | None,
) -> None:
    stats[team_a].games += 1
    stats[team_b].games += 1
    if winner == 0:
        stats[team_a].wins += 1
        stats[team_b].losses += 1
    elif winner == 1:
        stats[team_b].wins += 1
        stats[team_a].losses += 1
    else:
        stats[team_a].unfinished += 1
        stats[team_b].unfinished += 1


def log_progress(done: int, total: int, started_at: float) -> None:
    percent = 100 * done / total
    elapsed = time.monotonic() - started_at
    print(
        f"progress={percent:6.2f}% games={done}/{total} elapsed={elapsed:.1f}s",
        flush=True,
    )


def team_key(team: tuple[CharacterDefinition, ...]) -> str:
    return " / ".join(sorted(character.name for character in team))


if __name__ == "__main__":
    main()
