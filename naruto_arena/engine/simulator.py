from __future__ import annotations

from naruto_arena.engine.actions import Action, EndTurnAction, GetChakraAction, UseSkillAction
from naruto_arena.engine.rules import RulesError, end_turn
from naruto_arena.engine.skills import TargetRule
from naruto_arena.engine.state import GameState, UsedSkillState


def legal_actions(state: GameState, player_id: int) -> list[Action]:
    if state.winner is not None or player_id != state.active_player:
        return []
    player = state.players[player_id]
    enemy = state.players[1 - player_id]
    actions: list[Action] = [EndTurnAction(player_id)]
    if player.chakra.can_exchange_for():
        actions.extend(
            GetChakraAction(player_id, chakra_type) for chakra_type in player.chakra.amounts
        )
    enemy_ids = tuple(character.instance_id for character in enemy.living_characters())
    ally_ids = tuple(character.instance_id for character in player.living_characters())
    for character in player.living_characters():
        if character.used_skill_this_turn:
            continue
        for skill_id in character.skill_order:
            skill = resolved_skill(state, character.instance_id, skill_id)
            if not can_use_skill(state, character.instance_id, skill.id):
                continue
            if skill.target_rule == TargetRule.ONE_ENEMY:
                actions.extend(
                    UseSkillAction(player_id, character.instance_id, skill.id, (target_id,))
                    for target_id in enemy_ids
                )
            elif skill.target_rule == TargetRule.ALL_ENEMIES and enemy_ids:
                actions.append(
                    UseSkillAction(player_id, character.instance_id, skill.id, enemy_ids)
                )
            elif skill.target_rule == TargetRule.ONE_ALLY:
                actions.extend(
                    UseSkillAction(player_id, character.instance_id, skill.id, (target_id,))
                    for target_id in ally_ids
                )
            elif skill.target_rule == TargetRule.ALL_ALLIES and ally_ids:
                actions.append(UseSkillAction(player_id, character.instance_id, skill.id, ally_ids))
            elif skill.target_rule == TargetRule.SELF:
                actions.append(
                    UseSkillAction(
                        player_id, character.instance_id, skill.id, (character.instance_id,)
                    )
                )
            elif skill.target_rule == TargetRule.NONE:
                actions.append(UseSkillAction(player_id, character.instance_id, skill.id, ()))
    return actions


def apply_action(state: GameState, action: Action) -> GameState:
    if state.winner is not None:
        raise RulesError("Game is over.")
    if action.player_id != state.active_player:
        raise RulesError("It is not this player's turn.")
    if isinstance(action, EndTurnAction):
        state.metrics["actions"] += 1
        resolve_pending_skill_stack(state, action.player_id)
        end_turn(state)
        return state
    if isinstance(action, GetChakraAction):
        state.metrics["actions"] += 1
        state.metrics["get_chakra"] += 1
        state.players[action.player_id].chakra.exchange_for(action.chakra_type)
        return state
    if isinstance(action, UseSkillAction):
        apply_skill(state, action)
        return state
    raise RulesError("Unknown action.")


def can_use_skill(state: GameState, actor_id: str, skill_id: str) -> bool:
    actor = state.get_character(actor_id)
    if not actor.is_alive or actor.used_skill_this_turn:
        return False
    try:
        skill = resolved_skill(state, actor_id, skill_id)
    except KeyError:
        return False
    if skill.is_passive() or actor.cooldowns.get(skill.id, 0) > 0:
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
    state.players[action.player_id].chakra.pay(skill.chakra_cost, action.random_payment)
    state.metrics["actions"] += 1
    state.metrics["skills_used"] += 1
    state.metrics["chakra_spent"] += (
        sum(skill.chakra_cost.fixed.values()) + skill.chakra_cost.random
    )
    if skill.cooldown > 0:
        actor.cooldowns[skill.id] = skill.cooldown + 1
    actor.used_skill_this_turn = True
    state.players[action.player_id].skill_stack.append(
        UsedSkillState(
            action.actor_id,
            skill.id,
            used_skill_duration(skill),
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
        actor = state.get_character(used_skill.actor_id)
        if actor.owner != player_id or not actor.is_alive:
            continue
        skill = resolved_skill(state, used_skill.actor_id, used_skill.skill_id)
        for effect in skill.all_effects(state, used_skill.actor_id):
            effect.apply(state, used_skill.actor_id, used_skill.target_ids)


def used_skill_duration(skill) -> int:
    durations = [1, skill.duration]
    for effect in skill.effects:
        duration = getattr(effect, "duration", 0)
        if isinstance(duration, int):
            durations.append(duration)
    return max(durations)


def resolved_skill(state: GameState, actor_id: str, skill_id: str):
    return state.get_character(actor_id).definition.skill(skill_id)


def validate_targets(
    state: GameState, player_id: int, target_rule: TargetRule, target_ids: tuple[str, ...]
) -> None:
    if target_rule == TargetRule.NONE and target_ids:
        raise RulesError("Skill does not target.")
    if target_rule == TargetRule.SELF:
        if len(target_ids) != 1 or target_ids[0] not in {
            character.instance_id for character in state.players[player_id].characters
        }:
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
