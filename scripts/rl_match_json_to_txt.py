from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert an RL match JSON replay to readable TXT.")
    parser.add_argument("input", type=Path, help="Replay JSON from scripts/simulate_rl_match.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="TXT output path. Defaults to the input path with .txt suffix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text())
    output = args.output or args.input.with_suffix(".txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_match(report), encoding="utf-8")
    print(f"txt_report={output}")


def render_match(report: dict[str, Any]) -> str:
    metadata = report["metadata"]
    lines = [
        "RL Match Summary",
        f"model={metadata.get('model_path', 'unknown')}",
        (
            f"seed={metadata.get('seed')} actions={metadata.get('actions_taken')} "
            f"winner={_winner_text(metadata.get('winner'))} "
            f"stop={metadata.get('stop_reason')}"
        ),
        f"P0: {_team_text(report['teams']['player_0'])}",
        f"P1: {_team_text(report['teams']['player_1'])}",
        "",
        f"START | {_state_line(report['initial_state'])}",
    ]
    for item in report["timeline"]:
        lines.append(_timeline_line(item))
    lines.extend(
        [
            "",
            f"FINAL | winner={_winner_text(metadata.get('winner'))} | "
            f"{_state_line(report['final_state'])}",
        ]
    )
    return "\n".join(lines) + "\n"


def _timeline_line(item: dict[str, Any]) -> str:
    before = item["before"]
    after = item["after"]
    action = item["action"]
    player = item["player_id"]
    return (
        f"{item['action_index']:03d} | T{item['turn_number_before']:03d} P{player} | "
        f"{_action_text(action)} | {_hp_delta_text(before, after)} | "
        f"{_state_line(after)}"
    )


def _action_text(action: dict[str, Any]) -> str:
    action_type = action["type"]
    if action_type == "end_turn":
        return "END"
    if action_type == "get_chakra":
        return f"GET_CHAKRA {action['chakra_type']}"
    if action_type == "reorder_skills":
        return (
            f"REORDER {action['character_name']} {action['skill_id']} -> slot {action['new_index']}"
        )
    if action_type == "use_skill":
        targets = ", ".join(action["target_names"]) if action["target_names"] else "-"
        payment = _payment_text(action.get("random_payment", {}))
        return f"QUEUE {action['actor_name']} {action['skill_name']} -> {targets}{payment}"
    return action_type.upper()


def _payment_text(payment: dict[str, int]) -> str:
    spent = [f"{chakra}:{amount}" for chakra, amount in sorted(payment.items()) if amount]
    return "" if not spent else f" random[{', '.join(spent)}]"


def _hp_delta_text(before: dict[str, Any], after: dict[str, Any]) -> str:
    before_chars = _characters_by_id(before)
    deltas = []
    for character_id, after_character in _characters_by_id(after).items():
        before_hp = before_chars[character_id]["hp"]
        after_hp = after_character["hp"]
        if before_hp == after_hp:
            continue
        name = _short_name(after_character)
        sign = "+" if after_hp > before_hp else ""
        deltas.append(f"{name} {before_hp}->{after_hp} ({sign}{after_hp - before_hp})")
    return "hp: " + (", ".join(deltas) if deltas else "-")


def _state_line(state: dict[str, Any]) -> str:
    return " | ".join(_player_line(player) for player in state["players"])


def _player_line(player: dict[str, Any]) -> str:
    chars = " ".join(_character_state_text(character) for character in player["characters"])
    chakra = _chakra_text(player["chakra"])
    return f"P{player['player_id']} C{player['total_chakra']}[{chakra}] {chars}"


def _character_state_text(character: dict[str, Any]) -> str:
    name = _short_name(character)
    hp = character["hp"]
    tags = []
    status = character["status"]
    if not character["is_alive"]:
        tags.append("KO")
    if status["stunned_turns"]:
        tags.append(f"stun{status['stunned_turns']}")
    if status["class_stuns"]:
        tags.append("classStun")
    if status["invulnerable_turns"]:
        tags.append(f"inv{status['invulnerable_turns']}")
    if status["damage_over_time"]:
        dot = sum(item["amount"] for item in status["damage_over_time"])
        tags.append(f"dot{dot}")
    if status["damage_reductions"]:
        reductions = []
        for reduction in status["damage_reductions"]:
            if reduction["amount"]:
                reductions.append(str(reduction["amount"]))
            if reduction["percent"]:
                reductions.append(f"{reduction['percent']}%")
        if reductions:
            tags.append("dr" + "+".join(reductions))
    if status["active_markers"]:
        marker_count = len(status["active_markers"])
        tags.append(f"mk{marker_count}")
    suffix = "" if not tags else "{" + ",".join(tags) + "}"
    return f"{name}:{hp}{suffix}"


def _characters_by_id(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        character["instance_id"]: character
        for player in state["players"]
        for character in player["characters"]
    }


def _short_name(character: dict[str, Any]) -> str:
    return {
        "inuzuka_kiba": "Kiba",
        "sasuke_uchiha": "Sasuke",
        "uzumaki_naruto": "Naruto",
        "aburame_shino": "Shino",
        "sakura_haruno": "Sakura",
        "hyuuga_hinata": "Hinata",
    }.get(character["character_id"], character["name"].split()[-1])


def _chakra_text(chakra: dict[str, int]) -> str:
    order = ("ninjutsu", "taijutsu", "genjutsu", "bloodline")
    labels = {"ninjutsu": "N", "taijutsu": "T", "genjutsu": "G", "bloodline": "B"}
    return " ".join(f"{labels[name]}{chakra[name]}" for name in order)


def _team_text(team: list[dict[str, str]]) -> str:
    return " / ".join(character["name"] for character in team)


def _winner_text(winner: int | None) -> str:
    return "unfinished" if winner is None else f"P{winner}"


if __name__ == "__main__":
    main()
