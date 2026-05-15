from __future__ import annotations

import random

from naruto_arena.engine.chakra import ChakraType
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.effects import ActiveDamageOverTime, ActiveDamageReduction
from naruto_arena.engine.state import CharacterState, GameState, PlayerState, UsedSkillState

PERMANENT_SKILL_STACK_DURATION = 1_000_000


class RulesError(ValueError):
    pass


def create_initial_state(
    team_a: list[CharacterDefinition],
    team_b: list[CharacterDefinition],
    rng_seed: int = 0,
) -> GameState:
    if len(team_a) != 3 or len(team_b) != 3:
        raise RulesError("Each player must have exactly 3 characters.")
    _validate_no_duplicates(team_a)
    _validate_no_duplicates(team_b)
    players = (
        PlayerState(0, [_make_character(defn, 0, index) for index, defn in enumerate(team_a)]),
        PlayerState(1, [_make_character(defn, 1, index) for index, defn in enumerate(team_b)]),
    )
    state = GameState(players=players, rng_seed=rng_seed, rng=random.Random(rng_seed))
    initialize_passive_skill_stack(state)
    start_turn(state)
    return state


def initialize_passive_skill_stack(state: GameState) -> None:
    for player in state.players:
        for character in player.characters:
            for skill in character.definition.skills:
                if skill.is_passive() and skill.replacement_for is None:
                    player.skill_stack.append(
                        UsedSkillState(
                            character.instance_id,
                            skill.id,
                            PERMANENT_SKILL_STACK_DURATION,
                        )
                    )


def start_turn(state: GameState) -> None:
    state.reorders_this_turn = 0
    state.reordered_skills_this_turn.clear()
    player = state.players[state.active_player]
    for character in player.living_characters():
        character.used_skill_this_turn = False
        character.chakra_gain_marker = True  # type: ignore[attr-defined]
    gain_chakra_for_living_characters(state, state.active_player)
    tick_start_of_turn_effects(state, state.active_player)
    check_winner(state)


def end_turn(state: GameState) -> None:
    tick_end_of_turn_effects(state, state.active_player)
    decrement_cooldowns(state, state.active_player)
    state.active_player = 1 - state.active_player
    state.turn_number += 1
    start_turn(state)


def gain_chakra_for_living_characters(state: GameState, player_id: int) -> None:
    player = state.players[player_id]
    gain_count = chakra_gain_count(state, player_id)
    gain_steals = chakra_gain_steals(state, player_id)
    chakra_types = list(ChakraType)
    for _ in range(gain_count):
        chakra_type = state.rng.choice(chakra_types)
        if gain_steals:
            target, marker_id, source_id = gain_steals.pop(0)
            target.status.active_marker_stacks[marker_id] -= 1
            if target.status.active_marker_stacks[marker_id] <= 0:
                target.status.active_marker_stacks.pop(marker_id, None)
                target.status.active_markers.pop(marker_id, None)
            source_owner = state.owner_of(source_id)
            if source_owner != player_id and state.get_character(source_id).is_alive:
                state.players[source_owner].chakra.add(chakra_type, 1)
                state.get_character(source_id).status.active_markers[
                    "chakra_leach_stolen_chakra"
                ] = 1
                continue
        player.chakra.add(chakra_type, 1)


def chakra_gain_count(state: GameState, player_id: int) -> int:
    if state.turn_number == 1 and player_id == 0:
        return 1
    return len(state.players[player_id].living_characters())


def chakra_gain_steals(state: GameState, player_id: int) -> list[tuple[CharacterState, str, str]]:
    steals: list[tuple[CharacterState, str, str]] = []
    for character in state.players[player_id].living_characters():
        for marker_id, stacks in character.status.active_marker_stacks.items():
            if not marker_id.startswith("chakra_gain_steal:"):
                continue
            source_id = marker_id.removeprefix("chakra_gain_steal:")
            steals.extend((character, marker_id, source_id) for _ in range(stacks))
    return steals


def deal_damage(
    state: GameState,
    source_id: str,
    target_id: str,
    base_amount: int,
    piercing: bool = False,
    ignore_defenses: bool = False,
) -> None:
    source = state.get_character(source_id)
    target = state.get_character(target_id)
    if not source.is_alive or not target.is_alive:
        return
    amount = max(
        0,
        base_amount
        + damage_bonus_for_skill(state, source_id)
        - damage_penalty_for_skill(state, source_id),
    )
    if not ignore_defenses and target.status.invulnerable_turns > 0:
        amount = 0
    elif not ignore_defenses:
        for reduction in target.status.damage_reductions:
            if piercing and not reduction.unpierceable:
                continue
            amount = max(0, amount - reduction.amount)
            if reduction.percent > 0:
                amount = max(0, amount - (amount * reduction.percent // 100))
    target.hp = max(0, target.hp - amount)
    if (
        amount > 0
        and target.is_alive
        and target.status.has_marker("damage_triggers_invulnerability")
    ):
        target.status.invulnerable_turns = max(target.status.invulnerable_turns, 1)
    check_passive_triggers(state, target)
    check_winner(state)


def damage_bonus_for_skill(state: GameState, source_id: str) -> int:
    source = state.get_character(source_id)
    skill_id = getattr(state, "_current_skill_id", None)
    if skill_id is None:
        return 0
    bonus = 0
    for skill in source.definition.skills:
        for modifier in skill.conditional_damage:
            if modifier.applies_to(source, skill_id):
                bonus += modifier.amount
    return bonus


def damage_penalty_for_skill(state: GameState, source_id: str) -> int:
    penalty = getattr(state, "_current_damage_penalty", 0)
    if state.get_character(source_id).instance_id != source_id:
        return 0
    return penalty


def check_passive_triggers(state: GameState, character: CharacterState) -> None:
    if character.definition.id == "uzumaki_naruto":
        passive_id = "kyuubi_chakra_awakening"
        if (
            character.hp <= 50
            and not character.passives.get(passive_id, False)
            and not character.passive_triggered.get(passive_id, False)
        ):
            character.passives[passive_id] = True
            character.passive_triggered[passive_id] = True
    if character.definition.id == "sasuke_uchiha":
        passive_id = "cursed_seal_awakening"
        if (
            character.hp <= 50
            and not character.passives.get(passive_id, False)
            and not character.passive_triggered.get(passive_id, False)
        ):
            character.passives[passive_id] = True
            character.passive_triggered[passive_id] = True
            character.status.damage_reductions.append(
                ActiveDamageReduction(0, 1_000_000, unpierceable=True, percent=25)
            )
    if character.definition.id == "akimichi_chouji":
        passive_id = "butterfly_mode"
        if (
            character.status.marker_stacks("akimichi_pills") >= 3
            and not character.passives.get(passive_id, False)
            and not character.passive_triggered.get(passive_id, False)
        ):
            character.passives[passive_id] = True
            character.passive_triggered[passive_id] = True
            character.status.damage_reductions.append(
                ActiveDamageReduction(0, 1_000_000, unpierceable=True, percent=75)
            )


def tick_start_of_turn_effects(state: GameState, player_id: int) -> None:
    player = state.players[player_id]
    for character in player.living_characters():
        if character.passives.get("kyuubi_chakra_awakening", False):
            character.hp = min(character.max_hp, character.hp + 5)
        if character.passives.get("butterfly_mode", False):
            player.chakra.add(state.rng.choice(list(ChakraType)), 1)
        dots: list[ActiveDamageOverTime] = []
        for dot in character.status.damage_over_time:
            deal_damage(
                state,
                dot.source_id,
                character.instance_id,
                dot.amount,
                piercing=dot.piercing,
            )
            dot.remaining_turns -= 1
            if dot.remaining_turns > 0:
                dots.append(dot)
        character.status.damage_over_time = dots


def tick_end_of_turn_effects(state: GameState, player_id: int) -> None:
    kept_skills = []
    for used_skill in state.players[player_id].skill_stack:
        used_skill.remaining_turns -= 1
        if used_skill.remaining_turns > 0:
            kept_skills.append(used_skill)
    state.players[player_id].skill_stack = kept_skills
    for character in state.players[player_id].characters:
        if character.status.stunned_turns > 0:
            character.status.stunned_turns -= 1
        for skill_class in list(character.status.class_stuns):
            character.status.class_stuns[skill_class] -= 1
            if character.status.class_stuns[skill_class] <= 0:
                del character.status.class_stuns[skill_class]
        if character.status.invulnerable_turns > 0:
            character.status.invulnerable_turns -= 1
        for marker_id in list(character.status.active_markers):
            character.status.active_markers[marker_id] -= 1
            if character.status.active_markers[marker_id] <= 0:
                del character.status.active_markers[marker_id]
                character.status.active_marker_stacks.pop(marker_id, None)
        kept = []
        for reduction in character.status.damage_reductions:
            reduction.remaining_turns -= 1
            if reduction.remaining_turns > 0:
                kept.append(reduction)
        character.status.damage_reductions = kept


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
