from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.data.characters import ALL_CHARACTERS
from naruto_arena.engine.actions import (
    Action,
    EndTurnAction,
    GetChakraAction,
    ReorderSkillsAction,
    UseSkillAction,
)
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.effects import ActiveDamageOverTime, ActiveDamageReduction
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action, resolved_skill
from naruto_arena.engine.state import CharacterState, GameState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate one RL model-vs-itself match.")
    parser.add_argument("--model-path", type=Path, default=Path("models/naruto_actor_critic.pt"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument(
        "--team-a",
        default="uzumaki_naruto,sakura_haruno,sasuke_uchiha",
        help="Comma-separated character ids for player 0.",
    )
    parser.add_argument(
        "--team-b",
        default="uzumaki_naruto,sakura_haruno,sasuke_uchiha",
        help="Comma-separated character ids for player 1.",
    )
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rl_match.json"),
        help="Path to write the full JSON replay.",
    )
    parser.add_argument(
        "--list-characters",
        action="store_true",
        help="Print valid character ids and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_characters:
        for character in sorted(ALL_CHARACTERS.values(), key=lambda item: item.id):
            print(f"{character.id}: {character.name}")
        return

    team_a = parse_team(args.team_a)
    team_b = parse_team(args.team_b)
    state = create_initial_state(team_a, team_b, rng_seed=args.seed)
    agent = RlAgent(args.model_path, deterministic=not args.sample, seed=args.seed)
    timeline: list[dict[str, Any]] = []

    for index in range(1, args.max_actions + 1):
        if state.winner is not None:
            break
        player_id = state.active_player
        before = snapshot_state(state)
        before_state = deepcopy(state)
        action = agent.choose_action(state, player_id)
        action_json = action_to_json(state, action)
        apply_action(state, action)
        agent.observe_action(before_state, action, state)
        timeline.append(
            {
                "action_index": index,
                "player_id": player_id,
                "turn_number_before": before["turn_number"],
                "action": action_json,
                "before": before,
                "after": snapshot_state(state),
            }
        )

    report = {
        "metadata": {
            "model_path": str(args.model_path),
            "seed": args.seed,
            "max_actions": args.max_actions,
            "sample": args.sample,
            "actions_taken": len(timeline),
            "winner": state.winner,
            "stop_reason": "winner" if state.winner is not None else "max_actions",
        },
        "teams": {
            "player_0": team_to_json(team_a),
            "player_1": team_to_json(team_b),
        },
        "initial_state": timeline[0]["before"] if timeline else snapshot_state(state),
        "final_state": snapshot_state(state),
        "timeline": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"winner={state.winner} actions={len(timeline)} "
        f"stop_reason={report['metadata']['stop_reason']} json_report={args.output}"
    )


def parse_team(raw: str) -> list[CharacterDefinition]:
    ids = [item.strip() for item in raw.split(",") if item.strip()]
    if len(ids) != 3:
        raise ValueError("Team must contain exactly 3 comma-separated character ids.")
    if len(set(ids)) != len(ids):
        raise ValueError("Team cannot contain duplicate characters.")
    unknown = [character_id for character_id in ids if character_id not in ALL_CHARACTERS]
    if unknown:
        valid = ", ".join(sorted(ALL_CHARACTERS))
        raise ValueError(f"Unknown character ids: {unknown}. Valid ids: {valid}")
    return [ALL_CHARACTERS[character_id] for character_id in ids]


def snapshot_state(state: GameState) -> dict[str, Any]:
    return {
        "turn_number": state.turn_number,
        "active_player": state.active_player,
        "winner": state.winner,
        "players": [
            {
                "player_id": player.player_id,
                "chakra": {
                    chakra_type.value: amount
                    for chakra_type, amount in player.chakra.amounts.items()
                },
                "total_chakra": player.chakra.total(),
                "skill_stack": [
                    {
                        "actor_id": used_skill.actor_id,
                        "actor_name": state.get_character(used_skill.actor_id).definition.name,
                        "skill_id": used_skill.skill_id,
                        "remaining_turns": used_skill.remaining_turns,
                    }
                    for used_skill in player.skill_stack
                ],
                "characters": [character_to_json(character) for character in player.characters],
            }
            for player in state.players
        ],
    }


def character_to_json(character: CharacterState) -> dict[str, Any]:
    return {
        "instance_id": character.instance_id,
        "character_id": character.definition.id,
        "name": character.definition.name,
        "hp": character.hp,
        "max_hp": character.max_hp,
        "is_alive": character.is_alive,
        "used_skill_this_turn": character.used_skill_this_turn,
        "skill_order": list(character.skill_order),
        "cooldowns": dict(sorted(character.cooldowns.items())),
        "status": {
            "stunned_turns": character.status.stunned_turns,
            "class_stuns": dict(sorted(character.status.class_stuns.items())),
            "invulnerable_turns": character.status.invulnerable_turns,
            "damage_reductions": [
                damage_reduction_to_json(reduction)
                for reduction in character.status.damage_reductions
            ],
            "damage_over_time": [
                damage_over_time_to_json(dot) for dot in character.status.damage_over_time
            ],
            "active_markers": dict(sorted(character.status.active_markers.items())),
            "active_marker_stacks": dict(sorted(character.status.active_marker_stacks.items())),
        },
        "passives": dict(sorted(character.passives.items())),
        "passive_triggered": dict(sorted(character.passive_triggered.items())),
    }


def action_to_json(state: GameState, action: Action) -> dict[str, Any]:
    if isinstance(action, EndTurnAction):
        return {"type": "end_turn", "player_id": action.player_id}
    if isinstance(action, GetChakraAction):
        return {
            "type": "get_chakra",
            "player_id": action.player_id,
            "chakra_type": action.chakra_type.value,
        }
    if isinstance(action, ReorderSkillsAction):
        character = state.get_character(action.character_id)
        return {
            "type": "reorder_skills",
            "player_id": action.player_id,
            "character_id": action.character_id,
            "character_name": character.definition.name,
            "skill_id": action.skill_id,
            "new_index": action.new_index,
        }
    if isinstance(action, UseSkillAction):
        actor = state.get_character(action.actor_id)
        skill = resolved_skill(state, action.actor_id, action.skill_id)
        return {
            "type": "use_skill",
            "player_id": action.player_id,
            "actor_id": action.actor_id,
            "actor_name": actor.definition.name,
            "skill_id": action.skill_id,
            "skill_name": skill.name,
            "target_ids": list(action.target_ids),
            "target_names": [
                state.get_character(target_id).definition.name for target_id in action.target_ids
            ],
            "random_payment": {
                chakra_type.value: amount for chakra_type, amount in action.random_payment.items()
            },
            "classes": sorted(skill_class.value for skill_class in skill.classes),
            "target_rule": skill.target_rule.value,
        }
    return {"type": "unknown", "player_id": action.player_id}


def damage_reduction_to_json(reduction: ActiveDamageReduction) -> dict[str, Any]:
    return {
        "amount": reduction.amount,
        "remaining_turns": reduction.remaining_turns,
        "unpierceable": reduction.unpierceable,
        "percent": reduction.percent,
    }


def damage_over_time_to_json(dot: ActiveDamageOverTime) -> dict[str, Any]:
    return {
        "amount": dot.amount,
        "remaining_turns": dot.remaining_turns,
        "source_id": dot.source_id,
        "piercing": dot.piercing,
    }


def team_to_json(team: list[CharacterDefinition]) -> list[dict[str, str]]:
    return [{"id": character.id, "name": character.name} for character in team]


if __name__ == "__main__":
    main()
