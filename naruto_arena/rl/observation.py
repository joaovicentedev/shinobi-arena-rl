from __future__ import annotations

from naruto_arena.data.characters import (
    ABURAME_SHINO,
    AKIMICHI_CHOUJI,
    HYUUGA_HINATA,
    INUZUKA_KIBA,
    NARA_SHIKAMARU,
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
    YAMANAKA_INO,
)
from naruto_arena.engine.chakra import ChakraType
from naruto_arena.engine.effects import (
    ChakraGainSteal,
    ChakraRemoval,
    ChakraSteal,
    DamageOverTime,
    DamageReduction,
    DirectDamage,
    Healing,
    Invulnerability,
    PassiveEffect,
    StatusMarker,
    Stun,
)
from naruto_arena.engine.skills import SkillClass, TargetRule
from naruto_arena.engine.state import CharacterState, GameState

ROSTER = (
    UZUMAKI_NARUTO,
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    INUZUKA_KIBA,
    ABURAME_SHINO,
    HYUUGA_HINATA,
    NARA_SHIKAMARU,
    AKIMICHI_CHOUJI,
    YAMANAKA_INO,
)
ROSTER_INDEX = {character.id: index for index, character in enumerate(ROSTER)}
MAX_TURN = 100
MAX_COOLDOWN = 5
MAX_DURATION = 5
MAX_CHAKRA = 12
MAX_SKILL_COST = 4
CHARACTER_SLOTS = 6
BASE_CHARACTER_FEATURE_SIZE = 40
BASE_OBSERVATION_VERSION = "base_v1"
MAX_SKILLS_PER_CHARACTER = 9
SKILL_FEATURE_SIZE = 51
CHARACTER_FEATURE_SIZE = BASE_CHARACTER_FEATURE_SIZE + (
    MAX_SKILLS_PER_CHARACTER * SKILL_FEATURE_SIZE
)
OBSERVATION_VERSION = "skill_features_v1"


def observation_size(
    perfect_info: bool = False,
    observation_version: str = OBSERVATION_VERSION,
) -> int:
    team = [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA]
    from naruto_arena.engine.rules import create_initial_state

    state = create_initial_state(team, team, rng_seed=0)
    return len(
        encode_observation(
            state,
            state.active_player,
            perfect_info=perfect_info,
            observation_version=observation_version,
        )
    )


def encode_observation(
    state: GameState,
    player_id: int,
    *,
    perfect_info: bool = False,
    observation_version: str = OBSERVATION_VERSION,
) -> list[float]:
    """Encode state from the acting player's perspective.

    Decision: by default, opponent chakra is hidden because docs/rules.md says a
    competitive player does not directly observe it. `perfect_info=True` is kept as
    a debug option for experiments and regression checks.
    """

    if observation_version not in {BASE_OBSERVATION_VERSION, OBSERVATION_VERSION}:
        raise ValueError(f"Unknown observation version: {observation_version}")
    include_skill_features = observation_version == OBSERVATION_VERSION
    enemy_id = 1 - player_id
    features: list[float] = [
        min(state.turn_number, MAX_TURN) / MAX_TURN,
        state.active_player == player_id,
        _living_ratio(state, player_id),
        _living_ratio(state, enemy_id),
    ]
    for character in state.players[player_id].characters:
        features.extend(
            _character_features(state, character, include_skill_features=include_skill_features)
        )
    for character in state.players[enemy_id].characters:
        features.extend(
            _character_features(state, character, include_skill_features=include_skill_features)
        )
    features.extend(_chakra_features(state, player_id))
    if perfect_info:
        features.extend(_chakra_features(state, enemy_id))
    else:
        features.extend([0.0] * 5)
    return [float(value) for value in features]


def _living_ratio(state: GameState, player_id: int) -> float:
    return len(state.players[player_id].living_characters()) / 3


def _character_features(
    state: GameState,
    character: CharacterState,
    *,
    include_skill_features: bool,
) -> list[float]:
    reduction_amount = sum(reduction.amount for reduction in character.status.damage_reductions)
    unpierceable_amount = sum(
        reduction.amount
        for reduction in character.status.damage_reductions
        if reduction.unpierceable
    )
    max_reduction_percent = max(
        [reduction.percent for reduction in character.status.damage_reductions] or [0]
    )
    dot_amount = sum(dot.amount for dot in character.status.damage_over_time)
    marker_duration = sum(character.status.active_markers.values())
    marker_stacks = sum(character.status.active_marker_stacks.values())
    features = [
        character.hp / character.max_hp,
        float(character.is_alive),
        min(character.status.stunned_turns, MAX_DURATION) / MAX_DURATION,
        min(character.status.invulnerable_turns, MAX_DURATION) / MAX_DURATION,
        min(reduction_amount, 100) / 100,
        min(unpierceable_amount, 100) / 100,
        min(max_reduction_percent, 100) / 100,
        min(dot_amount, 100) / 100,
        min(marker_duration, MAX_DURATION * 4) / (MAX_DURATION * 4),
        min(marker_stacks, 5) / 5,
        min(sum(character.passives.values()), 5) / 5,
        min(sum(character.passive_triggered.values()), 5) / 5,
        float(character.used_skill_this_turn),
    ]
    features.extend(_one_hot(ROSTER_INDEX.get(character.definition.id, -1), len(ROSTER)))
    for skill_id in character.skill_order[:5]:
        features.append(min(character.cooldowns.get(skill_id, 0), MAX_COOLDOWN) / MAX_COOLDOWN)
    while len(features) < 13 + len(ROSTER) + 5:
        features.append(0.0)
    for skill_class in SkillClass:
        features.append(
            min(character.status.class_stuns.get(skill_class.value, 0), MAX_DURATION) / MAX_DURATION
        )
    if include_skill_features:
        for skill_id in character.skill_order[:MAX_SKILLS_PER_CHARACTER]:
            features.extend(_skill_features(state, character, skill_id))
        while len(features) < CHARACTER_FEATURE_SIZE:
            features.append(0.0)
    return features


def _skill_features(
    state: GameState,
    character: CharacterState,
    skill_id: str,
) -> list[float]:
    from naruto_arena.engine.simulator import can_use_skill, resolved_skill

    base_skill = character.definition.skill(skill_id)
    skill = resolved_skill(state, character.instance_id, skill_id)
    effects = skill.all_effects(state, character.instance_id)
    direct_damage = sum(effect.amount for effect in effects if isinstance(effect, DirectDamage))
    piercing_direct_damage = sum(
        effect.amount for effect in effects if isinstance(effect, DirectDamage) and effect.piercing
    )
    conditional_damage_bonus = sum(
        effect.conditional_bonus for effect in effects if isinstance(effect, DirectDamage)
    )
    healing = sum(effect.amount for effect in effects if isinstance(effect, Healing))
    stuns = [effect for effect in effects if isinstance(effect, Stun)]
    stun_duration = max([effect.duration for effect in stuns] or [0])
    class_stun_count = sum(1 for effect in stuns if effect.classes is not None)
    damage_reductions = [effect for effect in effects if isinstance(effect, DamageReduction)]
    reduction_amount = sum(effect.amount for effect in damage_reductions)
    reduction_percent = max([effect.percent for effect in damage_reductions] or [0])
    unpierceable_reduction = any(effect.unpierceable for effect in damage_reductions)
    invulnerability_duration = max(
        [effect.duration for effect in effects if isinstance(effect, Invulnerability)] or [0]
    )
    dots = [effect for effect in effects if isinstance(effect, DamageOverTime)]
    dot_amount = sum(effect.amount for effect in dots)
    dot_duration = max([effect.duration for effect in dots] or [0])
    dot_piercing = any(effect.piercing for effect in dots)
    chakra_removal = sum(effect.amount for effect in effects if isinstance(effect, ChakraRemoval))
    chakra_steal = sum(
        effect.amount for effect in effects if isinstance(effect, (ChakraGainSteal, ChakraSteal))
    )
    status_marker_count = sum(1 for effect in effects if isinstance(effect, StatusMarker))
    passive_effect = any(isinstance(effect, PassiveEffect) for effect in effects)

    features = [
        1.0,
        float(skill.id != base_skill.id or skill.replacement_for is not None),
        float(can_use_skill(state, character.instance_id, skill_id)),
        float(skill.is_passive()),
        float(skill.chakra_cost.is_free()),
    ]
    for chakra_type in ChakraType:
        features.append(
            min(skill.chakra_cost.fixed.get(chakra_type, 0), MAX_SKILL_COST) / MAX_SKILL_COST
        )
    features.extend(
        [
            min(skill.chakra_cost.random, MAX_SKILL_COST) / MAX_SKILL_COST,
            min(skill.cooldown, MAX_COOLDOWN) / MAX_COOLDOWN,
            min(character.cooldowns.get(skill.id, 0), MAX_COOLDOWN) / MAX_COOLDOWN,
            min(skill.duration, MAX_DURATION) / MAX_DURATION,
        ]
    )
    features.extend(_one_hot(tuple(TargetRule).index(skill.target_rule), len(tuple(TargetRule))))
    for skill_class in SkillClass:
        features.append(float(skill_class in skill.classes))
    features.extend(
        [
            min(direct_damage, 100) / 100,
            min(piercing_direct_damage, 100) / 100,
            min(conditional_damage_bonus, 100) / 100,
            min(healing, 100) / 100,
            min(stun_duration, MAX_DURATION) / MAX_DURATION,
            min(class_stun_count, len(SkillClass)) / len(SkillClass),
            min(reduction_amount, 100) / 100,
            min(reduction_percent, 100) / 100,
            float(unpierceable_reduction),
            min(invulnerability_duration, MAX_DURATION) / MAX_DURATION,
            min(dot_amount, 100) / 100,
            min(dot_duration, MAX_DURATION) / MAX_DURATION,
            float(dot_piercing),
            min(chakra_removal, MAX_CHAKRA) / MAX_CHAKRA,
            min(chakra_steal, MAX_CHAKRA) / MAX_CHAKRA,
            min(status_marker_count, 3) / 3,
            float(passive_effect),
            float(bool(skill.requirements)),
            float(bool(skill.target_requirements)),
        ]
    )
    if len(features) != SKILL_FEATURE_SIZE:
        raise AssertionError(f"Skill feature size changed: {len(features)}")
    return features


def _chakra_features(state: GameState, player_id: int) -> list[float]:
    chakra = state.players[player_id].chakra
    values = [
        min(chakra.amounts[chakra_type], MAX_CHAKRA) / MAX_CHAKRA for chakra_type in ChakraType
    ]
    values.append(min(chakra.total(), MAX_CHAKRA) / MAX_CHAKRA)
    return values


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if index == item else 0.0 for item in range(size)]
