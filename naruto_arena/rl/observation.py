from __future__ import annotations

from naruto_arena.data.characters import ALL_CHARACTERS
from naruto_arena.engine.chakra import ChakraType
from naruto_arena.engine.effects import (
    DamageOverTime,
    DamageReduction,
    DirectDamage,
)
from naruto_arena.engine.skills import SkillClass, TargetRule
from naruto_arena.engine.state import CharacterState, GameState

ROSTER = tuple(sorted(ALL_CHARACTERS.values(), key=lambda character: character.id))
UNKNOWN_CHARACTER_INDEX = 0
ROSTER_INDEX = {character.id: index + 1 for index, character in enumerate(ROSTER)}
CHARACTER_ID_CODE_COUNT = 256
MAX_TURN = 100
MAX_COOLDOWN = 5
MAX_DURATION = 5
MAX_CHAKRA = 12
MAX_SKILL_COST = 4
CHARACTER_SLOTS = 6
GLOBAL_FEATURE_SIZE = 7
MAX_STACK_SIZE = 12
ATTENTION_MAX_STACK_SIZE = 16
ATTENTION_STACK_SIDES = 2
STACK_SKILL_FEATURE_SIZE = 34
ENEMY_CHAKRA_BELIEF_FEATURE_SIZE = 13
BASE_CHARACTER_FEATURE_SIZE = 13 + len(ROSTER) + 5 + len(tuple(SkillClass))
BASE_OBSERVATION_VERSION = "base_v1"
MAX_SKILLS_PER_CHARACTER = 9
SKILL_FEATURE_SIZE = 50
SKILL_FEATURES_OBSERVATION_VERSION = "skill_features_v1"
SKILL_FEATURES_CHARACTER_FEATURE_SIZE = BASE_CHARACTER_FEATURE_SIZE + (
    MAX_SKILLS_PER_CHARACTER * SKILL_FEATURE_SIZE
)
CHARACTER_ID_FEATURE_INDEX = 13
COMPACT_BASE_CHARACTER_FEATURE_SIZE = 13 + 1 + 5 + len(tuple(SkillClass))
COMPACT_CHARACTER_FEATURE_SIZE = COMPACT_BASE_CHARACTER_FEATURE_SIZE + (
    MAX_SKILLS_PER_CHARACTER * SKILL_FEATURE_SIZE
)
COMPACT_OBSERVATION_VERSION = "skill_features_compact_id_stack_v1"
ATTENTION_OBSERVATION_VERSION = "attention_idea0_tokens_v1"
CHARACTER_FEATURE_SIZE = COMPACT_CHARACTER_FEATURE_SIZE
OBSERVATION_VERSION = ATTENTION_OBSERVATION_VERSION
OBSERVATION_VERSIONS = (ATTENTION_OBSERVATION_VERSION,)
SKILL_ID_TO_INDEX = {
    skill_id: index + 1
    for index, skill_id in enumerate(
        sorted({skill.id for character in ALL_CHARACTERS.values() for skill in character.skills})
    )
}
SKILL_ID_CODE_COUNT = len(SKILL_ID_TO_INDEX) + 1
ATTENTION_GLOBAL_FEATURE_SIZE = 12
ATTENTION_CHAR_NUMERIC_SIZE = 12
ATTENTION_CHAR_TOKEN_SIZE = ATTENTION_CHAR_NUMERIC_SIZE + 1
ATTENTION_SKILL_NUMERIC_SIZE = SKILL_FEATURE_SIZE
ATTENTION_SKILL_TOKEN_SIZE = ATTENTION_SKILL_NUMERIC_SIZE + 4
ATTENTION_STACK_NUMERIC_SIZE = 14
ATTENTION_STACK_TOKEN_SIZE = ATTENTION_STACK_NUMERIC_SIZE + 6
ATTENTION_STACK_TOKEN_COUNT = ATTENTION_STACK_SIDES * ATTENTION_MAX_STACK_SIZE
ATTENTION_OBSERVATION_SIZE = (
    ATTENTION_GLOBAL_FEATURE_SIZE
    + CHARACTER_SLOTS * ATTENTION_CHAR_TOKEN_SIZE
    + (CHARACTER_SLOTS * MAX_SKILLS_PER_CHARACTER) * ATTENTION_SKILL_TOKEN_SIZE
    + ATTENTION_STACK_TOKEN_COUNT * ATTENTION_STACK_TOKEN_SIZE
)
_STATIC_SKILL_FEATURE_CACHE: dict[tuple[object, ...], list[float]] = {}


def observation_size(
    perfect_info: bool = False,
    observation_version: str = OBSERVATION_VERSION,
) -> int:
    team = [
        ALL_CHARACTERS["uzumaki_naruto"],
        ALL_CHARACTERS["sakura_haruno"],
        ALL_CHARACTERS["sasuke_uchiha"],
    ]
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
    """Encode the Idea 0 token observation from the acting player's perspective."""

    if observation_version not in OBSERVATION_VERSIONS:
        raise ValueError(f"Unknown observation version: {observation_version}")
    return encode_attention_observation(state, player_id)
    include_skill_features = observation_version != BASE_OBSERVATION_VERSION
    compact_character_id = observation_version == COMPACT_OBSERVATION_VERSION
    enemy_id = 1 - player_id
    features: list[float] = [
        min(state.turn_number, MAX_TURN) / MAX_TURN,
        float(state.active_player == player_id),
        _living_ratio(state, player_id),
        _living_ratio(state, enemy_id),
    ]
    include_stack_features = observation_version == COMPACT_OBSERVATION_VERSION
    if include_stack_features:
        features.extend(
            [
                min(_actions_this_turn(state, player_id), 3) / 3,
                0.0,
                min(_pending_stack_count(state, player_id), MAX_STACK_SIZE) / MAX_STACK_SIZE,
            ]
        )
    for character in state.players[player_id].characters:
        features.extend(
            _character_features(
                state,
                character,
                include_skill_features=include_skill_features,
                compact_character_id=compact_character_id,
            )
        )
    for character in state.players[enemy_id].characters:
        features.extend(
            _character_features(
                state,
                character,
                include_skill_features=include_skill_features,
                compact_character_id=compact_character_id,
            )
        )
    features.extend(_chakra_features(state, player_id))
    if include_stack_features:
        features.extend([0.0] * ENEMY_CHAKRA_BELIEF_FEATURE_SIZE)
    elif perfect_info:
        features.extend(_chakra_features(state, enemy_id))
    else:
        features.extend([0.0] * 5)
    if include_stack_features:
        features.extend(_stack_features(state, player_id))
    return [float(value) for value in features]


def encode_attention_observation(state: GameState, player_id: int) -> list[float]:
    """Token-parsable no-belief observation with separate own/enemy visible stacks."""

    enemy_id = 1 - player_id
    features: list[float] = [
        min(state.turn_number, MAX_TURN) / MAX_TURN,
        float(state.active_player == player_id),
        _living_ratio(state, player_id),
        _living_ratio(state, enemy_id),
        min(_actions_this_turn(state, player_id), 3) / 3,
        0.0,
        min(_pending_stack_count(state, player_id), ATTENTION_MAX_STACK_SIZE)
        / ATTENTION_MAX_STACK_SIZE,
    ]
    features.extend(_chakra_features(state, player_id))
    if len(features) != ATTENTION_GLOBAL_FEATURE_SIZE:
        raise AssertionError(f"Attention global feature size changed: {len(features)}")

    ordered_characters = list(state.players[player_id].characters) + list(
        state.players[enemy_id].characters
    )
    for character in ordered_characters:
        features.extend(_attention_character_token(state, character))
    for character in ordered_characters:
        for skill_slot in range(MAX_SKILLS_PER_CHARACTER):
            skill_id = (
                character.skill_order[skill_slot]
                if skill_slot < len(character.skill_order)
                else None
            )
            features.extend(_attention_skill_token(state, character, skill_slot, skill_id))
    for owner_id in (player_id, enemy_id):
        features.extend(_attention_stack_tokens(state, player_id, owner_id))
    if len(features) != ATTENTION_OBSERVATION_SIZE:
        raise AssertionError(f"Attention observation size changed: {len(features)}")
    return [float(value) for value in features]


def _attention_character_token(state: GameState, character: CharacterState) -> list[float]:
    base = _character_features(
        state,
        character,
        include_skill_features=False,
        compact_character_id=True,
    )
    token = [
        float(character.is_alive),
        character.hp / character.max_hp if character.is_alive else 0.0,
        base[2],
        max(base[18:]) if len(base) > 18 else 0.0,
        base[3],
        base[4],
        base[6],
        base[7],
        base[8] + base[9],
        base[10] + base[11],
        base[12],
        sum(base[14:19]) / 5,
        base[CHARACTER_ID_FEATURE_INDEX],
    ]
    return token


def _attention_skill_token(
    state: GameState,
    character: CharacterState,
    skill_slot: int,
    skill_id: str | None,
) -> list[float]:
    if skill_id is None:
        numeric = [0.0] * ATTENTION_SKILL_NUMERIC_SIZE
        skill_index = 0
    else:
        numeric = _skill_features(state, character, skill_id)
        resolved_skill = resolved_skill_for_observation(state, character, skill_id)
        skill_index = SKILL_ID_TO_INDEX.get(resolved_skill.id, 0)
        if not character.is_alive:
            numeric[2] = 0.0
    return numeric + [
        float(_character_slot(state, character)),
        float(ROSTER_INDEX.get(character.definition.id, UNKNOWN_CHARACTER_INDEX)),
        float(skill_slot),
        float(skill_index),
    ]


def _attention_stack_tokens(state: GameState, player_id: int, owner_id: int) -> list[float]:
    features: list[float] = []
    visible = list(state.players[owner_id].skill_stack)[:ATTENTION_MAX_STACK_SIZE]
    for index, used_skill in enumerate(visible):
        actor = state.get_character(used_skill.actor_id)
        try:
            skill_slot = actor.skill_order.index(used_skill.skill_id)
            skill = resolved_skill_for_observation(state, actor, used_skill.skill_id)
            effects = skill.all_effects(state, actor.instance_id)
            skill_index = SKILL_ID_TO_INDEX.get(skill.id, 0)
        except (KeyError, ValueError):
            skill_slot = 0
            skill = None
            effects = []
            skill_index = 0
        effect_features = _attention_effect_features(skill, effects)
        numeric = [
            1.0,
            min(used_skill.remaining_turns, MAX_DURATION * 2) / (MAX_DURATION * 2),
            index / ATTENTION_MAX_STACK_SIZE,
            float(used_skill.pending),
            float(not used_skill.pending),
        ] + effect_features
        features.extend(
            numeric
            + [
                float(actor.owner == player_id),
                float(_character_slot(state, actor)),
                float(ROSTER_INDEX.get(actor.definition.id, UNKNOWN_CHARACTER_INDEX)),
                float(skill_slot),
                float(skill_index),
                float(_target_code_for_used_skill(state, player_id, used_skill.target_ids)),
            ]
        )
    empty = [0.0] * ATTENTION_STACK_TOKEN_SIZE
    while len(features) < ATTENTION_MAX_STACK_SIZE * ATTENTION_STACK_TOKEN_SIZE:
        features.extend(empty)
    return features


def _attention_effect_features(skill, effects) -> list[float]:
    if skill is None:
        return [0.0] * 9
    direct_damage = sum(effect.amount for effect in effects if isinstance(effect, DirectDamage))
    reduction = sum(effect.amount for effect in effects if isinstance(effect, DamageReduction))
    dot = sum(effect.amount for effect in effects if isinstance(effect, DamageOverTime))
    dot_duration = max(
        [effect.duration for effect in effects if isinstance(effect, DamageOverTime)] or [0]
    )
    defense_duration = max(
        [effect.duration for effect in effects if isinstance(effect, DamageReduction)] or [0]
    )
    return [
        min(direct_damage, 100) / 100,
        min(reduction, 100) / 100,
        min(defense_duration, MAX_DURATION) / MAX_DURATION,
        min(dot, 100) / 100,
        min(dot_duration, MAX_DURATION) / MAX_DURATION,
        float(skill.target_rule == TargetRule.ONE_ENEMY),
        float(skill.target_rule == TargetRule.ALL_ENEMIES),
        float(skill.target_rule in {TargetRule.SELF, TargetRule.ONE_ALLY, TargetRule.ALL_ALLIES}),
        min(skill.chakra_cost.random + sum(skill.chakra_cost.fixed.values()), MAX_SKILL_COST)
        / MAX_SKILL_COST,
    ]


def _living_ratio(state: GameState, player_id: int) -> float:
    return len(state.players[player_id].living_characters()) / 3


def _actions_this_turn(state: GameState, player_id: int) -> int:
    return sum(character.used_skill_this_turn for character in state.players[player_id].characters)


def _pending_stack_count(state: GameState, player_id: int) -> int:
    return sum(used_skill.pending for used_skill in state.players[player_id].skill_stack)


def _character_features(
    state: GameState,
    character: CharacterState,
    *,
    include_skill_features: bool,
    compact_character_id: bool = False,
) -> list[float]:
    reduction_amount = sum(reduction.amount for reduction in character.status.defenses)
    max_reduction_turns = max(
        [reduction.remaining_turns for reduction in character.status.defenses] or [0]
    )
    dot_amount = sum(dot.amount for dot in character.status.dots)
    max_dot_turns = max([dot.remaining_turns for dot in character.status.dots] or [0])
    features = [
        character.hp / character.max_hp,
        float(character.is_alive),
        0.0,
        0.0,
        min(reduction_amount, 100) / 100,
        min(max_reduction_turns, MAX_DURATION) / MAX_DURATION,
        0.0,
        min(dot_amount, 100) / 100,
        min(max_dot_turns, MAX_DURATION) / MAX_DURATION,
        0.0,
        0.0,
        0.0,
        float(character.used_skill_this_turn),
    ]
    character_index = ROSTER_INDEX.get(character.definition.id, UNKNOWN_CHARACTER_INDEX)
    if compact_character_id:
        features.append(float(character_index))
    else:
        features.extend(_one_hot(character_index - 1, len(ROSTER)))
    for skill_id in character.skill_order[:5]:
        features.append(min(character.cooldowns.get(skill_id, 0), MAX_COOLDOWN) / MAX_COOLDOWN)
    identity_size = 1 if compact_character_id else len(ROSTER)
    expected_base_size = (
        COMPACT_BASE_CHARACTER_FEATURE_SIZE if compact_character_id else BASE_CHARACTER_FEATURE_SIZE
    )
    while len(features) < 13 + identity_size + 5:
        features.append(0.0)
    for _skill_class in SkillClass:
        features.append(0.0)
    if len(features) != expected_base_size:
        raise AssertionError(f"Character base feature size changed: {len(features)}")
    if include_skill_features:
        for skill_id in character.skill_order[:MAX_SKILLS_PER_CHARACTER]:
            features.extend(_skill_features(state, character, skill_id))
        expected_feature_size = (
            COMPACT_CHARACTER_FEATURE_SIZE
            if compact_character_id
            else SKILL_FEATURES_CHARACTER_FEATURE_SIZE
        )
        while len(features) < expected_feature_size:
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
    features = [
        1.0,
        float(skill.id != base_skill.id or skill.replacement_for is not None),
        float(can_use_skill(state, character.instance_id, skill_id)),
    ]
    static_prefix, static_suffix = _static_skill_features(state, character, skill)
    features.extend(static_prefix)
    features.append(min(character.cooldowns.get(skill.id, 0), MAX_COOLDOWN) / MAX_COOLDOWN)
    features.extend(static_suffix)
    if len(features) != SKILL_FEATURE_SIZE:
        raise AssertionError(f"Skill feature size changed: {len(features)}")
    return features


def _static_skill_features(
    state: GameState,
    character: CharacterState,
    skill,
) -> tuple[list[float], list[float]]:
    if skill.effect_factory is not None:
        return _build_static_skill_features(skill, skill.all_effects(state, character.instance_id))
    cache_key = _static_skill_cache_key(skill)
    cached = _STATIC_SKILL_FEATURE_CACHE.get(cache_key)
    if cached is None:
        prefix, suffix = _build_static_skill_features(
            skill,
            list(skill.effects),
        )
        cached = prefix + suffix
        _STATIC_SKILL_FEATURE_CACHE[cache_key] = cached
    prefix_size = 2 + len(tuple(ChakraType)) + 2
    return list(cached[:prefix_size]), list(cached[prefix_size:])


def _static_skill_cache_key(skill) -> tuple[object, ...]:
    return (
        skill.id,
        skill.cooldown,
        tuple(
            sorted(
                (chakra_type.value, amount)
                for chakra_type, amount in skill.chakra_cost.fixed.items()
            )
        ),
        skill.chakra_cost.random,
        tuple(sorted(skill_class.value for skill_class in skill.classes)),
        skill.target_rule.value,
        skill.effects,
        bool(skill.requirements),
        bool(skill.target_requirements),
        skill.duration,
        skill.status_marker,
        skill.replacement_for,
    )


def _build_static_skill_features(skill, effects) -> tuple[list[float], list[float]]:
    direct_damage = sum(effect.amount for effect in effects if isinstance(effect, DirectDamage))
    defense_effects = [effect for effect in effects if isinstance(effect, DamageReduction)]
    reduction_amount = sum(effect.amount for effect in defense_effects)
    reduction_duration = max([effect.duration for effect in defense_effects] or [0])
    dots = [effect for effect in effects if isinstance(effect, DamageOverTime)]
    dot_amount = sum(effect.amount for effect in dots)
    dot_duration = max([effect.duration for effect in dots] or [0])

    prefix = [
        float(skill.is_passive()),
        float(skill.chakra_cost.is_free()),
    ]
    for chakra_type in ChakraType:
        prefix.append(
            min(skill.chakra_cost.fixed.get(chakra_type, 0), MAX_SKILL_COST) / MAX_SKILL_COST
        )
    prefix.extend(
        [
            min(skill.chakra_cost.random, MAX_SKILL_COST) / MAX_SKILL_COST,
            min(skill.cooldown, MAX_COOLDOWN) / MAX_COOLDOWN,
        ]
    )
    suffix = [min(skill.duration, MAX_DURATION) / MAX_DURATION]
    suffix.extend(_one_hot(tuple(TargetRule).index(skill.target_rule), len(tuple(TargetRule))))
    for skill_class in SkillClass:
        suffix.append(float(skill_class in skill.classes))
    suffix.extend(
        [
            min(direct_damage, 100) / 100,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            min(reduction_amount, 100) / 100,
            min(reduction_duration, MAX_DURATION) / MAX_DURATION,
            0.0,
            0.0,
            min(dot_amount, 100) / 100,
            min(dot_duration, MAX_DURATION) / MAX_DURATION,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(bool(skill.requirements)),
            float(bool(skill.target_requirements)),
        ]
    )
    return prefix, suffix


def _chakra_features(state: GameState, player_id: int) -> list[float]:
    chakra = state.players[player_id].chakra
    values = [
        min(chakra.amounts[chakra_type], MAX_CHAKRA) / MAX_CHAKRA for chakra_type in ChakraType
    ]
    values.append(min(chakra.total(), MAX_CHAKRA) / MAX_CHAKRA)
    return values


def _stack_features(state: GameState, player_id: int) -> list[float]:
    features: list[float] = []
    stack = _visible_stack_items(state, player_id)[:MAX_STACK_SIZE]
    for index, used_skill in enumerate(stack):
        actor = state.get_character(used_skill.actor_id)
        try:
            skill_slot = actor.skill_order.index(used_skill.skill_id)
            skill = resolved_skill_for_observation(state, actor, used_skill.skill_id)
            effects = skill.all_effects(state, actor.instance_id)
        except (KeyError, ValueError):
            skill_slot = -1
            skill = None
            effects = []
        target_code = _target_code_for_used_skill(state, player_id, used_skill.target_ids)
        item = [
            1.0,
            float(actor.owner == player_id),
        ]
        item.extend(_one_hot(_character_slot(state, actor), 3))
        item.extend(_one_hot(skill_slot, MAX_SKILLS_PER_CHARACTER))
        item.extend(_one_hot(target_code, 10))
        item.extend(
            [
                min(used_skill.remaining_turns, MAX_DURATION * 2) / (MAX_DURATION * 2),
                index / MAX_STACK_SIZE,
                float(used_skill.pending),
                float(not used_skill.pending),
            ]
        )
        item.extend(_compact_effect_features(skill, effects))
        if len(item) != STACK_SKILL_FEATURE_SIZE:
            raise AssertionError(f"Stack skill feature size changed: {len(item)}")
        features.extend(item)
    while len(features) < MAX_STACK_SIZE * STACK_SKILL_FEATURE_SIZE:
        features.append(0.0)
    return features


def _visible_stack_items(state: GameState, player_id: int):
    items = []
    for owner_id in (player_id, 1 - player_id):
        for used_skill in state.players[owner_id].skill_stack:
            items.append(used_skill)
    return items


def resolved_skill_for_observation(state: GameState, character: CharacterState, skill_id: str):
    from naruto_arena.engine.simulator import resolved_skill

    return resolved_skill(state, character.instance_id, skill_id)


def _target_code_for_used_skill(
    state: GameState,
    player_id: int,
    target_ids: tuple[str, ...],
) -> int:
    if not target_ids:
        return 0
    if len(target_ids) == 1:
        target = state.get_character(target_ids[0])
        if target.instance_id in {
            character.instance_id for character in state.players[player_id].characters
        }:
            return 4 + _character_slot(state, target)
        return 4 + 3 + _character_slot(state, target)
    owners = {state.get_character(target_id).owner for target_id in target_ids}
    if owners == {1 - player_id}:
        return 2
    if owners == {player_id}:
        return 3
    return 0


def _character_slot(state: GameState, character: CharacterState) -> int:
    return state.players[character.owner].characters.index(character)


def _compact_effect_features(skill, effects) -> list[float]:
    if skill is None:
        return [0.0] * 6
    direct_damage = sum(effect.amount for effect in effects if isinstance(effect, DirectDamage))
    reduction = sum(effect.amount for effect in effects if isinstance(effect, DamageReduction))
    dot_amount = sum(effect.amount for effect in effects if isinstance(effect, DamageOverTime))
    dot_duration = max(
        [effect.duration for effect in effects if isinstance(effect, DamageOverTime)] or [0]
    )
    return [
        min(direct_damage, 100) / 100,
        min(reduction, 100) / 100,
        min(dot_amount, 100) / 100,
        min(dot_duration, MAX_DURATION) / MAX_DURATION,
        float(skill.target_rule in {TargetRule.SELF, TargetRule.ONE_ALLY, TargetRule.ALL_ALLIES}),
        0.0,
    ]


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if index == item else 0.0 for item in range(size)]
