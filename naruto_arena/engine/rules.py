from __future__ import annotations

import random

from naruto_arena.engine.chakra import ChakraType
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.state import CharacterState, GameState, PlayerState


class RulesError(ValueError):
    pass


def create_initial_state(
    team_a: list[CharacterDefinition],
    team_b: list[CharacterDefinition],
    rng_seed: int = 0,
) -> GameState:
    """Create an Idea 0 3v3 state.

    Idea 0 is a curriculum/debug environment. Only HP, random chakra, direct
    damage, simple defense, DOT, cooldowns, costs, and target selection are active.
    Removed Naruto Arena mechanics should be reintroduced here one at a time.
    """

    if len(team_a) != 3 or len(team_b) != 3:
        raise RulesError("Each player must have exactly 3 characters.")
    _validate_no_duplicates(team_a)
    _validate_no_duplicates(team_b)
    players = (
        PlayerState(0, [_make_character(defn, 0, index) for index, defn in enumerate(team_a)]),
        PlayerState(1, [_make_character(defn, 1, index) for index, defn in enumerate(team_b)]),
    )
    state = GameState(players=players, rng_seed=rng_seed, rng=random.Random(rng_seed))
    state.metrics.update(
        {
            "actions": 0.0,
            "turns_ended": 0.0,
            "skills_used": 0.0,
            "get_chakra": 0.0,
            "unused_chakra_at_end_turn": 0.0,
            "damage_dealt": 0.0,
            "chakra_spent": 0.0,
            "attacks_into_defense": 0.0,
            "wasted_overkill": 0.0,
            "dot_value": 0.0,
            "defense_value": 0.0,
        }
    )
    start_turn(state)
    return state


def start_turn(state: GameState) -> None:
    player = state.players[state.active_player]
    for character in player.living_characters():
        character.used_skill_this_turn = False
    gain_chakra_for_living_characters(state, state.active_player)
    resolve_dots(state, state.active_player)
    check_winner(state)


def end_turn(state: GameState) -> None:
    state.metrics["turns_ended"] += 1
    state.metrics["unused_chakra_at_end_turn"] += state.players[state.active_player].chakra.total()
    tick_defenses(state, state.active_player)
    tick_skill_stack(state, state.active_player)
    decrement_cooldowns(state, state.active_player)
    state.active_player = 1 - state.active_player
    state.turn_number += 1
    start_turn(state)


def gain_chakra_for_living_characters(state: GameState, player_id: int) -> None:
    gain_count = 1 if state.turn_number == 1 and player_id == 0 else len(
        state.players[player_id].living_characters()
    )
    chakra_types = list(ChakraType)
    for _ in range(gain_count):
        state.players[player_id].chakra.add(state.rng.choice(chakra_types), 1)


def deal_damage(state: GameState, source_id: str, target_id: str, base_amount: int) -> int:
    source = state.get_character(source_id)
    target = state.get_character(target_id)
    if not source.is_alive or not target.is_alive:
        return 0
    amount = max(0, base_amount)
    blocked = 0
    for defense in target.status.defenses:
        before_reduction = amount
        amount = max(0, amount - defense.amount)
        blocked += before_reduction - amount
    before_hp = target.hp
    target.hp = max(0, target.hp - amount)
    actual = before_hp - target.hp
    state.metrics["damage_dealt"] += actual
    state.metrics["defense_value"] += blocked
    if blocked > 0:
        state.metrics["attacks_into_defense"] += 1
    state.metrics["wasted_overkill"] += max(0, amount - actual)
    check_winner(state)
    return actual


def resolve_dots(state: GameState, player_id: int) -> None:
    for character in state.players[player_id].living_characters():
        kept = []
        for dot in character.status.dots:
            state.metrics["dot_value"] += deal_damage(
                state, dot.source_id, character.instance_id, dot.amount
            )
            dot.remaining_turns -= 1
            if dot.remaining_turns > 0 and character.is_alive:
                kept.append(dot)
        character.status.dots = kept


def tick_defenses(state: GameState, player_id: int) -> None:
    for character in state.players[player_id].characters:
        kept = []
        for defense in character.status.defenses:
            defense.remaining_turns -= 1
            if defense.remaining_turns > 0 and character.is_alive:
                kept.append(defense)
        character.status.defenses = kept


def tick_skill_stack(state: GameState, player_id: int) -> None:
    kept = []
    for used_skill in state.players[player_id].skill_stack:
        used_skill.remaining_turns -= 1
        if used_skill.remaining_turns > 0:
            kept.append(used_skill)
    state.players[player_id].skill_stack = kept


def decrement_cooldowns(state: GameState, player_id: int) -> None:
    for character in state.players[player_id].characters:
        for skill_id, turns in list(character.cooldowns.items()):
            if turns > 0:
                character.cooldowns[skill_id] = turns - 1


def check_winner(state: GameState) -> None:
    for player_id in (0, 1):
        enemy = state.players[1 - player_id]
        if all(not character.is_alive for character in enemy.characters):
            state.winner = player_id


def _validate_no_duplicates(team: list[CharacterDefinition]) -> None:
    ids = [character.id for character in team]
    if len(ids) != len(set(ids)):
        raise RulesError("A team cannot have duplicate characters.")


def _make_character(definition: CharacterDefinition, owner: int, index: int) -> CharacterState:
    return CharacterState(
        definition=definition,
        owner=owner,
        instance_id=f"p{owner}:{definition.id}:{index}",
    )
