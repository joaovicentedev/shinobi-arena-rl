from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naruto_arena.agents.heuristic_agent import SimpleHeuristicAgent
from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action
from naruto_arena.rl.teams import BENCHMARK_MATCHUPS, random_mirror_teams, team_from_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an RL model on fixed benchmark teams.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--matches-per-benchmark", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax.")
    parser.add_argument(
        "--opponent",
        choices=("self", "heuristic"),
        default="self",
        help="Benchmark against the same model or a heuristic player.",
    )
    parser.add_argument(
        "--random-mirror-matches",
        type=int,
        default=0,
        help="Also run this many random same-team matches with independently shuffled positions.",
    )
    parser.add_argument(
        "--paired-sides",
        action="store_true",
        help="For heuristic random mirrors, play each sampled matchup with model on both sides.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rl_benchmarks.json"),
        help="Path to write the benchmark JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agents = make_agents(args)
    results: list[dict[str, Any]] = []
    game_index = 0
    for matchup in BENCHMARK_MATCHUPS:
        wins = {0: 0, 1: 0, None: 0}
        actions_taken: list[int] = []
        for _ in range(args.matches_per_benchmark):
            seed = args.seed + game_index
            game_index += 1
            winner, actions = simulate_match(
                matchup.team_a,
                matchup.team_b,
                seed,
                agents,
                args.max_actions,
            )
            wins[winner] += 1
            actions_taken.append(actions)
        result = {
            "name": matchup.name,
            "team_a": list(matchup.team_a),
            "team_b": list(matchup.team_b),
            "matches": args.matches_per_benchmark,
            "player_0_wins": wins[0],
            "player_1_wins": wins[1],
            "unfinished": wins[None],
            "player_0_win_rate": wins[0] / args.matches_per_benchmark,
            "resolved_player_0_win_rate": (
                wins[0] / (wins[0] + wins[1]) if wins[0] + wins[1] else 0.0
            ),
            "avg_actions": sum(actions_taken) / len(actions_taken),
        }
        results.append(result)
        print(
            f"{matchup.name}: p0_wr={result['player_0_win_rate']:.3f} "
            f"resolved_p0_wr={result['resolved_player_0_win_rate']:.3f} "
            f"unfinished={wins[None]}"
        )

    report = {
        "metadata": {
            "model_path": str(args.model_path),
            "matches_per_benchmark": args.matches_per_benchmark,
            "random_mirror_matches": args.random_mirror_matches,
            "paired_sides": args.paired_sides,
            "seed": args.seed,
            "max_actions": args.max_actions,
            "deterministic": not args.sample,
            "opponent": args.opponent,
        },
        "benchmarks": results,
    }
    if args.random_mirror_matches:
        report["random_mirror"] = evaluate_random_mirrors(
            args,
            agents,
            start_seed=args.seed + 10_000,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"json_report={args.output}")


def make_agents(args: argparse.Namespace):
    player_0 = RlAgent(args.model_path, deterministic=not args.sample, seed=args.seed * 2 + 1)
    if args.opponent == "self":
        player_1 = RlAgent(args.model_path, deterministic=not args.sample, seed=args.seed * 2 + 2)
    else:
        player_1 = SimpleHeuristicAgent(seed=args.seed * 2 + 2, allow_reorder=False)
    return {0: player_0, 1: player_1}


def evaluate_random_mirrors(
    args: argparse.Namespace,
    agents,
    *,
    start_seed: int,
) -> dict[str, Any]:
    rng = random.Random(args.seed)
    if args.paired_sides and args.opponent != "heuristic":
        raise ValueError("--paired-sides is only meaningful with --opponent heuristic.")
    if args.paired_sides:
        return evaluate_paired_random_mirrors(args, rng, start_seed=start_seed)

    wins = {0: 0, 1: 0, None: 0}
    actions_taken: list[int] = []
    matchups: list[dict[str, Any]] = []
    for match_index in range(args.random_mirror_matches):
        team_a, team_b = random_mirror_teams(rng)
        winner, actions = simulate_match(
            team_a,
            team_b,
            start_seed + match_index,
            agents,
            args.max_actions,
        )
        wins[winner] += 1
        actions_taken.append(actions)
        matchups.append(
            {
                "team_a": [character.id for character in team_a],
                "team_b": [character.id for character in team_b],
                "winner": winner,
                "actions": actions,
            }
        )
    resolved = wins[0] + wins[1]
    summary = {
        "matches": args.random_mirror_matches,
        "player_0_wins": wins[0],
        "player_1_wins": wins[1],
        "unfinished": wins[None],
        "player_0_win_rate": wins[0] / args.random_mirror_matches,
        "resolved_player_0_win_rate": wins[0] / resolved if resolved else 0.0,
        "avg_actions": sum(actions_taken) / len(actions_taken),
        "matchups": matchups,
    }
    print(
        f"random_mirror: p0_wr={summary['player_0_win_rate']:.3f} "
        f"resolved_p0_wr={summary['resolved_player_0_win_rate']:.3f} "
        f"unfinished={wins[None]}"
    )
    return summary


def evaluate_paired_random_mirrors(
    args: argparse.Namespace,
    rng: random.Random,
    *,
    start_seed: int,
) -> dict[str, Any]:
    model_as_p0 = RlAgent(
        args.model_path,
        deterministic=not args.sample,
        seed=args.seed * 2 + 1,
    )
    model_as_p1 = RlAgent(
        args.model_path,
        deterministic=not args.sample,
        seed=args.seed * 2 + 2,
    )
    heuristic_as_p0 = SimpleHeuristicAgent(seed=args.seed * 2 + 3, allow_reorder=False)
    heuristic_as_p1 = SimpleHeuristicAgent(seed=args.seed * 2 + 4, allow_reorder=False)
    model_wins = 0
    heuristic_wins = 0
    unfinished = 0
    model_p0_wins = 0
    model_p1_wins = 0
    model_p0_games = 0
    model_p1_games = 0
    actions_taken: list[int] = []
    matchups: list[dict[str, Any]] = []
    for match_index in range(args.random_mirror_matches):
        team_a, team_b = random_mirror_teams(rng)
        base_seed = start_seed + (match_index * 2)
        p0_winner, p0_actions = simulate_match(
            team_a,
            team_b,
            base_seed,
            {0: model_as_p0, 1: heuristic_as_p1},
            args.max_actions,
        )
        model_p0_games += 1
        if p0_winner == 0:
            model_wins += 1
            model_p0_wins += 1
        elif p0_winner == 1:
            heuristic_wins += 1
        else:
            unfinished += 1
        actions_taken.append(p0_actions)

        p1_winner, p1_actions = simulate_match(
            team_b,
            team_a,
            base_seed + 1,
            {0: heuristic_as_p0, 1: model_as_p1},
            args.max_actions,
        )
        model_p1_games += 1
        if p1_winner == 1:
            model_wins += 1
            model_p1_wins += 1
        elif p1_winner == 0:
            heuristic_wins += 1
        else:
            unfinished += 1
        actions_taken.append(p1_actions)
        matchups.append(
            {
                "team_a": [character.id for character in team_a],
                "team_b": [character.id for character in team_b],
                "model_as_p0_winner": p0_winner,
                "model_as_p0_actions": p0_actions,
                "model_as_p1_winner": p1_winner,
                "model_as_p1_actions": p1_actions,
            }
        )

    total_games = args.random_mirror_matches * 2
    resolved = model_wins + heuristic_wins
    summary = {
        "matches": args.random_mirror_matches,
        "paired_games": total_games,
        "model_wins": model_wins,
        "heuristic_wins": heuristic_wins,
        "unfinished": unfinished,
        "model_win_rate": model_wins / total_games,
        "resolved_model_win_rate": model_wins / resolved if resolved else 0.0,
        "model_as_p0_win_rate": model_p0_wins / model_p0_games,
        "model_as_p1_win_rate": model_p1_wins / model_p1_games,
        "avg_actions": sum(actions_taken) / len(actions_taken),
        "matchups": matchups,
    }
    print(
        f"random_mirror_paired: model_wr={summary['model_win_rate']:.3f} "
        f"resolved_model_wr={summary['resolved_model_win_rate']:.3f} "
        f"model_p0_wr={summary['model_as_p0_win_rate']:.3f} "
        f"model_p1_wr={summary['model_as_p1_win_rate']:.3f} "
        f"unfinished={unfinished}"
    )
    return summary


def simulate_match(
    team_a: tuple[str, str, str] | list[CharacterDefinition],
    team_b: tuple[str, str, str] | list[CharacterDefinition],
    seed: int,
    agents,
    max_actions: int,
) -> tuple[int | None, int]:
    team_a_definitions = team_from_ids(team_a) if isinstance(team_a, tuple) else team_a
    team_b_definitions = team_from_ids(team_b) if isinstance(team_b, tuple) else team_b
    state = create_initial_state(team_a_definitions, team_b_definitions, rng_seed=seed)
    actions = 0
    for _ in range(max_actions):
        if state.winner is not None:
            break
        player_id = state.active_player
        action = agents[player_id].choose_action(state, player_id)
        apply_action(state, action)
        actions += 1
    return state.winner, actions


if __name__ == "__main__":
    main()
