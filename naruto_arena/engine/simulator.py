from __future__ import annotations

from dataclasses import replace

from naruto_arena.engine.actions import Action, EndTurnAction, ReorderSkillsAction, UseSkillAction
from naruto_arena.engine.rules import RulesError, end_turn
from naruto_arena.engine.skills import SkillClass, TargetRule
from naruto_arena.engine.state import GameState, UsedSkillState

MAX_REORDERS_PER_TURN = 3


def legal_actions(state: GameState, player_id: int) -> list[Action]:
    if state.winner is not None or player_id != state.active_player:
        return []
    actions: list[Action] = [EndTurnAction(player_id)]
    player = state.players[player_id]
    enemy = state.players[1 - player_id]
    enemy_ids = tuple(character.instance_id for character in enemy.living_characters())
    ally_ids = tuple(character.instance_id for character in player.living_characters())
    for character in player.living_characters():
        for skill_id in character.skill_order:
            skill = resolved_skill(state, character.instance_id, skill_id)
            if not can_use_skill(state, character.instance_id, skill.id):
                continue
            targets: tuple[str, ...]
            if skill.target_rule == TargetRule.ONE_ENEMY:
                for target_id in enemy_ids:
                    if not targets_meet_requirements(
                        state, character.instance_id, skill.id, (target_id,)
                    ):
                        continue
                    actions.append(
                        UseSkillAction(player_id, character.instance_id, skill.id, (target_id,))
                    )
                continue
            if skill.target_rule == TargetRule.ALL_ENEMIES:
                if not targets_meet_requirements(state, character.instance_id, skill.id, enemy_ids):
                    continue
                actions.append(
                    UseSkillAction(player_id, character.instance_id, skill.id, enemy_ids)
                )
                continue
            if skill.target_rule == TargetRule.ONE_ALLY:
                for target_id in ally_ids:
                    if not targets_meet_requirements(
                        state, character.instance_id, skill.id, (target_id,)
                    ):
                        continue
                    actions.append(
                        UseSkillAction(player_id, character.instance_id, skill.id, (target_id,))
                    )
                continue
            if skill.target_rule == TargetRule.ALL_ALLIES:
                if not targets_meet_requirements(state, character.instance_id, skill.id, ally_ids):
                    continue
                actions.append(UseSkillAction(player_id, character.instance_id, skill.id, ally_ids))
                continue
            if skill.target_rule == TargetRule.SELF:
                targets = (character.instance_id,)
            else:
                targets = ()
            if not targets_meet_requirements(state, character.instance_id, skill.id, targets):
                continue
            actions.append(UseSkillAction(player_id, character.instance_id, skill.id, targets))
    if state.reorders_this_turn < MAX_REORDERS_PER_TURN:
        for used_skill in player.skill_stack:
            reorder_key = (used_skill.actor_id, used_skill.skill_id)
            if reorder_key in state.reordered_skills_this_turn:
                continue
            for index in range(len(player.skill_stack)):
                if player.skill_stack[index] != used_skill:
                    actions.append(
                        ReorderSkillsAction(
                            player_id,
                            used_skill.actor_id,
                            used_skill.skill_id,
                            index,
                        )
                    )
    return actions


def apply_action(state: GameState, action: Action) -> GameState:
    if state.winner is not None:
        raise RulesError("Game is over.")
    if action.player_id != state.active_player:
        raise RulesError("It is not this player's turn.")
    if isinstance(action, EndTurnAction):
        resolve_pending_skill_stack(state, action.player_id)
        end_turn(state)
        return state
    if isinstance(action, ReorderSkillsAction):
        apply_reorder(state, action)
        return state
    if isinstance(action, UseSkillAction):
        apply_skill(state, action)
        return state
    raise RulesError("Unknown action.")


def can_use_skill(state: GameState, actor_id: str, skill_id: str) -> bool:
    actor = state.get_character(actor_id)
    if not actor.is_alive or actor.status.stunned_turns > 0:
        return False
    if actor.used_skill_this_turn:
        return False
    try:
        skill = resolved_skill(state, actor_id, skill_id)
    except KeyError:
        return False
    if actor.status.has_marker("harmful_skills_stunned") and skill.target_rule in {
        TargetRule.ONE_ENEMY,
        TargetRule.ALL_ENEMIES,
    }:
        return False
    if any(skill_class.value in actor.status.class_stuns for skill_class in skill.classes):
        return False
    if skill.is_passive() or actor.cooldowns.get(skill.id, 0) > 0:
        return False
    if not all(requirement(state, actor_id) for requirement in skill.requirements):
        return False
    return state.players[actor.owner].chakra.can_afford(skill.chakra_cost)


def apply_skill(state: GameState, action: UseSkillAction) -> None:
    actor = state.get_character(action.actor_id)
    if actor.owner != action.player_id:
        raise RulesError("Actor does not belong to player.")
    skill = resolved_skill(state, action.actor_id, action.skill_id)
    if not can_use_skill(state, action.actor_id, skill.id):
        raise RulesError(f"Skill cannot be used: {skill.name}")
    validate_targets(state, action.player_id, skill.target_rule, action.target_ids)
    if not targets_meet_requirements(state, action.actor_id, skill.id, action.target_ids):
        raise RulesError(f"Skill cannot target selected characters: {skill.name}")
    state.players[action.player_id].chakra.pay(skill.chakra_cost, action.random_payment)
    if skill.cooldown > 0:
        actor.cooldowns[skill.id] = skill.cooldown + 1
    actor.used_skill_this_turn = True
    state.players[action.player_id].skill_stack.append(
        UsedSkillState(
            action.actor_id,
            skill.id,
            used_skill_duration(state, action.actor_id, skill),
            action.target_ids,
            dict(action.random_payment),
            pending=True,
        )
    )


def resolve_pending_skill_stack(state: GameState, player_id: int) -> None:
    for used_skill in list(state.players[player_id].skill_stack):
        if not used_skill.pending:
            continue
        used_skill.pending = False
        resolve_used_skill(state, player_id, used_skill)


def resolve_used_skill(state: GameState, player_id: int, used_skill: UsedSkillState) -> None:
    actor = state.get_character(used_skill.actor_id)
    if actor.owner != player_id or not actor.is_alive:
        return
    skill = resolved_skill(state, used_skill.actor_id, used_skill.skill_id)
    if skill.duration > 0 and skill.status_marker is not None:
        actor.status.active_markers[skill.status_marker] = skill.duration
    previous = getattr(state, "_current_skill_id", None)
    previous_penalty = getattr(state, "_current_damage_penalty", None)
    state._current_skill_id = skill.id  # type: ignore[attr-defined]
    state._current_damage_penalty = damage_penalty_from_status(  # type: ignore[attr-defined]
        actor,
        skill,
    )
    try:
        for effect in skill.all_effects(state, used_skill.actor_id):
            effect.apply(state, used_skill.actor_id, used_skill.target_ids)
    finally:
        if previous is None:
            delattr(state, "_current_skill_id")
        else:
            state._current_skill_id = previous  # type: ignore[attr-defined]
        if previous_penalty is None:
            delattr(state, "_current_damage_penalty")
        else:
            state._current_damage_penalty = previous_penalty  # type: ignore[attr-defined]
    from naruto_arena.engine.rules import check_passive_triggers

    check_passive_triggers(state, actor)


def apply_reorder(state: GameState, action: ReorderSkillsAction) -> None:
    if state.reorders_this_turn >= MAX_REORDERS_PER_TURN:
        raise RulesError("Reorder limit reached for this turn.")
    character = state.get_character(action.character_id)
    if character.owner != action.player_id:
        raise RulesError("Character does not belong to player.")
    reorder_key = (action.character_id, action.skill_id)
    if reorder_key in state.reordered_skills_this_turn:
        raise RulesError("Skill has already been reordered this turn.")
    player = state.players[action.player_id]
    stack_index = next(
        (
            index
            for index, used_skill in enumerate(player.skill_stack)
            if used_skill.actor_id == action.character_id and used_skill.skill_id == action.skill_id
        ),
        None,
    )
    if stack_index is None:
        raise RulesError("Cannot reorder skill that is not in the used skill stack.")
    if not 0 <= action.new_index < len(player.skill_stack):
        raise RulesError("Invalid skill index.")
    used_skill = player.skill_stack.pop(stack_index)
    player.skill_stack.insert(action.new_index, used_skill)
    state.reorders_this_turn += 1
    state.reordered_skills_this_turn.add(reorder_key)


def used_skill_duration(state: GameState, actor_id: str, skill) -> int:
    durations = [1, skill.duration]
    if skill.status_marker is not None:
        durations.append(skill.duration)
    for effect in skill.all_effects(state, actor_id):
        duration = getattr(effect, "duration", 0)
        if isinstance(duration, int):
            durations.append(duration)
    return max(durations)


def resolved_skill(state: GameState, actor_id: str, skill_id: str):
    actor = state.get_character(actor_id)
    skill = actor.definition.skill(skill_id)
    for candidate in actor.definition.skills:
        if candidate.replacement_for == skill.id:
            if all(requirement(state, actor_id) for requirement in candidate.requirements):
                return replace(candidate, id=skill.id)
    return skill


def targets_meet_requirements(
    state: GameState, actor_id: str, skill_id: str, target_ids: tuple[str, ...]
) -> bool:
    skill = resolved_skill(state, actor_id, skill_id)
    return all(
        requirement(state, actor_id, target_id)
        for target_id in target_ids
        for requirement in skill.target_requirements
    )


def damage_penalty_from_status(character, skill) -> int:
    is_harmful = skill.target_rule in {TargetRule.ONE_ENEMY, TargetRule.ALL_ENEMIES}
    if not is_harmful or SkillClass.AFFLICTION in skill.classes:
        return 0
    penalty = 0
    for marker_id in character.status.active_markers:
        if marker_id.startswith("female_bug:"):
            penalty += 5 * character.status.marker_stacks(marker_id)
    return penalty


def validate_targets(
    state: GameState, player_id: int, target_rule: TargetRule, target_ids: tuple[str, ...]
) -> None:
    if target_rule == TargetRule.NONE and target_ids:
        raise RulesError("Skill does not target.")
    if target_rule == TargetRule.SELF:
        if len(target_ids) != 1 or state.get_character(target_ids[0]).owner != player_id:
            raise RulesError("Invalid self target.")
    if target_rule == TargetRule.ONE_ALLY:
        if len(target_ids) != 1 or state.get_character(target_ids[0]).owner != player_id:
            raise RulesError("Invalid ally target.")
    if target_rule == TargetRule.ALL_ALLIES:
        ally_ids = {
            character.instance_id for character in state.players[player_id].living_characters()
        }
        if set(target_ids) != ally_ids:
            raise RulesError("Invalid ally targets.")
    if target_rule == TargetRule.ONE_ENEMY:
        if len(target_ids) != 1 or state.get_character(target_ids[0]).owner == player_id:
            raise RulesError("Invalid enemy target.")
    if target_rule == TargetRule.ALL_ENEMIES:
        enemy_ids = {
            character.instance_id for character in state.players[1 - player_id].living_characters()
        }
        if set(target_ids) != enemy_ids:
            raise RulesError("Invalid enemy targets.")
