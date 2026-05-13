from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.effects import (
    ChakraRemoval,
    DamageOverTime,
    DamageReduction,
    DirectDamage,
    Healing,
    Invulnerability,
    StatusMarker,
    Stun,
)
from naruto_arena.engine.skills import SkillClass, SkillDefinition, TargetRule

EXTRA_CHARACTER_COUNT = 25
_CHARACTER_DATA_DIR = Path(__file__).resolve().parents[1] / "characters"
_CHARACTER_INDEX_PATH = _CHARACTER_DATA_DIR / "characters_index.json"

_CHAKRA_TYPES_BY_NAME = {
    "ninjutsu": ChakraType.NINJUTSU,
    "taijutsu": ChakraType.TAIJUTSU,
    "bloodline": ChakraType.BLOODLINE,
    "genjutsu": ChakraType.GENJUTSU,
}
_SKILL_CLASSES_BY_NAME = {skill_class.value.lower(): skill_class for skill_class in SkillClass}


def load_extra_characters(
    existing_ids: set[str],
    *,
    limit: int = EXTRA_CHARACTER_COUNT,
) -> dict[str, CharacterDefinition]:
    """Load a small executable training roster from scraped character JSON.

    These definitions intentionally cover common mechanics only. Hand-authored
    characters remain the source of truth for exact rule behavior.
    """

    index = json.loads(_CHARACTER_INDEX_PATH.read_text())
    loaded: dict[str, CharacterDefinition] = {}
    for item in index:
        character_id = _id_from_slug(item["slug"])
        if character_id in existing_ids or character_id in loaded:
            continue
        definition = _load_character(item, character_id)
        if definition is None:
            continue
        loaded[definition.id] = definition
        if len(loaded) >= limit:
            break
    return loaded


def _load_character(item: dict[str, Any], character_id: str) -> CharacterDefinition | None:
    data_path = _CHARACTER_DATA_DIR / item["file"]
    data = json.loads(data_path.read_text())
    skills = tuple(
        skill
        for index, raw_skill in enumerate(data.get("skills", ()))
        if (skill := _load_skill(character_id, raw_skill, index)) is not None
    )
    if not skills:
        return None
    return CharacterDefinition(
        id=character_id,
        name=data["name"],
        description=data.get("description", ""),
        skills=skills,
    )


def _load_skill(
    character_id: str,
    raw_skill: dict[str, Any],
    index: int,
) -> SkillDefinition | None:
    name = raw_skill["name"]
    description = raw_skill.get("description", "")
    skill_id = f"{character_id}_{_id_from_slug(name)}"
    effects = _effects_from_description(character_id, skill_id, description)
    target_rule = _target_rule_from_description(description)
    if not effects and target_rule == TargetRule.ONE_ENEMY:
        effects = (DirectDamage(15),)
    return SkillDefinition(
        id=skill_id,
        name=name,
        description=description,
        cooldown=raw_skill["cooldown"] or 0,
        chakra_cost=_chakra_cost(raw_skill.get("chakra", ())),
        classes=_skill_classes(raw_skill.get("classes", ())),
        target_rule=target_rule,
        effects=effects,
    )


def _chakra_cost(raw_chakra: list[str]) -> ChakraCost:
    fixed: dict[ChakraType, int] = {}
    random = 0
    for chakra_name in raw_chakra:
        if chakra_name == "random":
            random += 1
            continue
        chakra_type = _CHAKRA_TYPES_BY_NAME.get(chakra_name)
        if chakra_type is not None:
            fixed[chakra_type] = fixed.get(chakra_type, 0) + 1
    return ChakraCost(fixed, random)


def _skill_classes(raw_classes: list[str]) -> frozenset[SkillClass]:
    return frozenset(
        skill_class
        for raw_class in raw_classes
        if (skill_class := _SKILL_CLASSES_BY_NAME.get(raw_class.lower())) is not None
    )


def _target_rule_from_description(description: str) -> TargetRule:
    text = description.lower()
    if "all enemies" in text:
        return TargetRule.ALL_ENEMIES
    if "all allies" in text or "your team" in text or "his team" in text or "her team" in text:
        return TargetRule.ALL_ALLIES
    if "one ally" in text or "an ally" in text or "target ally" in text:
        return TargetRule.ONE_ALLY
    if (
        "this skill makes" in text
        or "himself" in text
        or "herself" in text
        or "itself" in text
        or "the user" in text
    ):
        return TargetRule.SELF
    if "one enemy" in text or "an enemy" in text or "target enemy" in text:
        return TargetRule.ONE_ENEMY
    if "become invulnerable" in text or "becomes invulnerable" in text:
        return TargetRule.SELF
    return TargetRule.ONE_ENEMY


def _effects_from_description(
    character_id: str,
    skill_id: str,
    description: str,
) -> tuple:
    text = description.lower()
    effects: list = []
    duration = _duration(text)

    if damage := _first_number_before(text, "damage"):
        if duration > 1 and "for" in text:
            effects.append(DamageOverTime(damage, duration=duration, piercing="piercing" in text))
        else:
            effects.append(DirectDamage(damage, piercing="piercing" in text))
    if heal := _first_number_before(text, "health"):
        if "heal" in text:
            effects.append(Healing(heal))
    if "stun" in text:
        effects.append(Stun(max(1, duration)))
    if "invulnerable" in text:
        effects.append(Invulnerability(max(1, duration)))
    if reduction := _first_number_before(text, "damage reduction"):
        effects.append(
            DamageReduction(
                reduction,
                duration=max(1, duration),
                percent="%" in text,
            )
        )
    if "remove" in text and "chakra" in text:
        effects.append(ChakraRemoval(1))
    if not effects and any(word in text for word in ("gain", "increase", "protect", "counter")):
        effects.append(StatusMarker(f"json:{character_id}:{skill_id}", duration=max(1, duration)))
    return tuple(effects)


def _duration(text: str) -> int:
    match = re.search(r"for (\d+) turns?", text)
    if match is None:
        match = re.search(r"(\d+) turns?", text)
    return int(match.group(1)) if match is not None else 1


def _first_number_before(text: str, word: str) -> int | None:
    match = re.search(rf"(\d+)\s+(?:piercing\s+|affliction\s+)?{re.escape(word)}", text)
    return int(match.group(1)) if match is not None else None


def _id_from_slug(value: str) -> str:
    normalized = value.lower().replace("(s)", "s")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return normalized
