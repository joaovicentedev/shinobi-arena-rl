from __future__ import annotations

from naruto_arena.data.characters import (
    ABURAME_SHINO,
    HYUUGA_HINATA,
    INUZUKA_KIBA,
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.chakra import ChakraType
from naruto_arena.engine.skills import SkillClass
from naruto_arena.engine.state import CharacterState, GameState

ROSTER = (
    UZUMAKI_NARUTO,
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    INUZUKA_KIBA,
    ABURAME_SHINO,
    HYUUGA_HINATA,
)
ROSTER_INDEX = {character.id: index for index, character in enumerate(ROSTER)}
MAX_TURN = 100
MAX_COOLDOWN = 5
MAX_DURATION = 5
MAX_CHAKRA = 12


def observation_size(perfect_info: bool = False) -> int:
    team = [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA]
    from naruto_arena.engine.rules import create_initial_state

    state = create_initial_state(team, team, rng_seed=0)
    return len(encode_observation(state, state.active_player, perfect_info=perfect_info))


def encode_observation(
    state: GameState,
    player_id: int,
    *,
    perfect_info: bool = False,
) -> list[float]:
    """Encode state from the acting player's perspective.

    Decision: by default, opponent chakra is hidden because docs/rules.md says a
    competitive player does not directly observe it. `perfect_info=True` is kept as
    a debug option for experiments and regression checks.
    """

    enemy_id = 1 - player_id
    features: list[float] = [
        min(state.turn_number, MAX_TURN) / MAX_TURN,
        state.active_player == player_id,
        _living_ratio(state, player_id),
        _living_ratio(state, enemy_id),
    ]
    for character in state.players[player_id].characters:
        features.extend(_character_features(character))
    for character in state.players[enemy_id].characters:
        features.extend(_character_features(character))
    features.extend(_chakra_features(state, player_id))
    if perfect_info:
        features.extend(_chakra_features(state, enemy_id))
    else:
        features.extend([0.0] * 5)
    return [float(value) for value in features]


def _living_ratio(state: GameState, player_id: int) -> float:
    return len(state.players[player_id].living_characters()) / 3


def _character_features(character: CharacterState) -> list[float]:
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
    ]
    features.extend(_one_hot(ROSTER_INDEX.get(character.definition.id, -1), len(ROSTER)))
    for skill_id in character.skill_order[:5]:
        features.append(min(character.cooldowns.get(skill_id, 0), MAX_COOLDOWN) / MAX_COOLDOWN)
    while len(features) < 12 + len(ROSTER) + 5:
        features.append(0.0)
    for skill_class in SkillClass:
        features.append(
            min(character.status.class_stuns.get(skill_class.value, 0), MAX_DURATION)
            / MAX_DURATION
        )
    return features


def _chakra_features(state: GameState, player_id: int) -> list[float]:
    chakra = state.players[player_id].chakra
    values = [
        min(chakra.amounts[chakra_type], MAX_CHAKRA) / MAX_CHAKRA
        for chakra_type in ChakraType
    ]
    values.append(min(chakra.total(), MAX_CHAKRA) / MAX_CHAKRA)
    return values


def _one_hot(index: int, size: int) -> list[float]:
    return [1.0 if index == item else 0.0 for item in range(size)]
