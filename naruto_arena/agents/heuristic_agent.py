from __future__ import annotations

import random

from naruto_arena.agents.random_agent import _simulation_actions
from naruto_arena.engine.actions import Action, UseSkillAction
from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.effects import (
    ChakraGainSteal,
    ChakraRemoval,
    ChakraSteal,
    DamageOverTime,
    DamageReduction,
    DirectDamage,
    Healing,
    Invulnerability,
    StatusMarker,
    Stun,
)
from naruto_arena.engine.simulator import resolved_skill
from naruto_arena.engine.skills import SkillDefinition, TargetRule
from naruto_arena.engine.state import GameState


class SimpleHeuristicAgent:
    def __init__(self, seed: int = 0, allow_reorder: bool = False) -> None:
        self.rng = random.Random(seed)
        self.allow_reorder = allow_reorder
        self.focus_targets: dict[int, str] = {}

    def choose_action(self, state: GameState, player_id: int) -> Action:
        actions = _simulation_actions(state, player_id, self.allow_reorder)
        use_skill_actions: list[UseSkillAction] = []
        for action in actions:
            if isinstance(action, UseSkillAction):
                use_skill_actions.append(action)
        if use_skill_actions:
            focus_target = self._focus_target(state, player_id)
            scored = [
                (self._action_score(state, action, focus_target), action)
                for action in use_skill_actions
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score = scored[0][0]
            tied = [action for score, action in scored if score == best_score]
            action = self.rng.choice(tied)
            skill = resolved_skill(state, action.actor_id, action.skill_id)
            return self._with_preserved_random_payment(state, action, skill.chakra_cost)
        return self.rng.choice(actions)

    def _focus_target(self, state: GameState, player_id: int) -> str | None:
        current = self.focus_targets.get(player_id)
        if current is not None:
            try:
                if state.get_character(current).is_alive:
                    return current
            except KeyError:
                self.focus_targets.pop(player_id, None)
        enemies = list(state.players[1 - player_id].living_characters())
        if not enemies:
            return None
        target = min(enemies, key=lambda character: (character.hp, character.instance_id))
        self.focus_targets[player_id] = target.instance_id
        return target.instance_id

    def _action_score(
        self,
        state: GameState,
        action: UseSkillAction,
        focus_target: str | None,
    ) -> tuple[int, int, int, int]:
        skill = resolved_skill(state, action.actor_id, action.skill_id)
        effects = skill.all_effects(state, action.actor_id)
        cost = _chakra_cost_value(skill.chakra_cost)
        damage = _damage_value(effects)
        is_attack = _is_attack(skill, damage)
        prep = _prep_value(state, action, skill)
        support = _support_value(state, action, skill)
        enemy_count = sum(
            1
            for target_id in action.target_ids
            if state.owner_of(target_id) != action.player_id
        )
        focused = focus_target in action.target_ids if focus_target is not None else False
        target_hp = min(
            (state.get_character(target_id).hp for target_id in action.target_ids),
            default=999,
        )

        score = 0
        score += prep * 130
        score += support * 85
        score += damage * max(1, enemy_count) * 4
        if skill.target_rule == TargetRule.ALL_ENEMIES and enemy_count >= 2:
            score += 70 + damage * enemy_count
        if focused and is_attack:
            score += 120
        elif is_attack:
            score -= 35
        if is_attack and target_hp <= damage:
            score += 180
        score += _disruption_value(effects) * 45
        score -= cost * (18 if prep or support else 9)
        if cost <= 1 and (prep or support):
            score += 80
        if skill.cooldown > 0 and is_attack and damage < 30:
            score -= 20
        category = 2 if prep or support else 1 if is_attack else 0
        return (score, category, -cost, -target_hp)

    def _with_preserved_random_payment(
        self,
        state: GameState,
        action: UseSkillAction,
        cost: ChakraCost,
    ) -> UseSkillAction:
        if cost.random == 0:
            return action
        pool = state.players[action.player_id].chakra.copy()
        for chakra_type, amount in cost.fixed.items():
            pool.amounts[chakra_type] -= amount
        payment = {chakra_type: 0 for chakra_type in ChakraType}
        for _ in range(cost.random):
            available = [chakra_type for chakra_type, amount in pool.amounts.items() if amount > 0]
            chosen = max(
                available,
                key=lambda chakra_type: (pool.amounts[chakra_type], chakra_type.value),
            )
            payment[chosen] += 1
            pool.amounts[chosen] -= 1
        return UseSkillAction(
            action.player_id,
            action.actor_id,
            action.skill_id,
            action.target_ids,
            {chakra_type: amount for chakra_type, amount in payment.items() if amount},
        )


def _chakra_cost_value(cost: ChakraCost) -> int:
    return cost.random + sum(cost.fixed.values())


def _damage_value(effects: list[object]) -> int:
    total = 0
    for effect in effects:
        if isinstance(effect, DirectDamage):
            total += effect.amount + effect.conditional_bonus
        elif isinstance(effect, DamageOverTime):
            total += effect.amount * max(1, effect.duration)
    return total


def _is_attack(skill: SkillDefinition, damage: int) -> bool:
    return skill.target_rule in {TargetRule.ONE_ENEMY, TargetRule.ALL_ENEMIES} and damage > 0


def _prep_value(state: GameState, action: UseSkillAction, skill: SkillDefinition) -> int:
    cost = _chakra_cost_value(skill.chakra_cost)
    if cost > 1:
        return 0
    effects = skill.all_effects(state, action.actor_id)
    if skill.target_rule in {TargetRule.SELF, TargetRule.NONE} and (
        skill.status_marker is not None
        or any(isinstance(effect, (StatusMarker, DamageReduction)) for effect in effects)
    ):
        actor = state.get_character(action.actor_id)
        if skill.status_marker is not None and actor.status.has_marker(skill.status_marker):
            return 0
        return 2 if cost == 0 else 1
    if skill.target_rule == TargetRule.ONE_ENEMY and any(
        isinstance(effect, StatusMarker) for effect in effects
    ):
        return 2 if cost == 0 else 1
    return 0


def _support_value(state: GameState, action: UseSkillAction, skill: SkillDefinition) -> int:
    cost = _chakra_cost_value(skill.chakra_cost)
    if cost > 1:
        return 0
    effects = skill.all_effects(state, action.actor_id)
    if not any(
        isinstance(effect, (Healing, Invulnerability, DamageReduction)) for effect in effects
    ):
        return 0
    target_hps = [state.get_character(target_id).hp for target_id in action.target_ids]
    actor_hp = state.get_character(action.actor_id).hp
    low_hp_bonus = 1 if min(target_hps or [actor_hp]) <= 45 else 0
    if skill.target_rule in {TargetRule.ALL_ALLIES, TargetRule.ONE_ALLY}:
        return 2 + low_hp_bonus
    if skill.target_rule == TargetRule.SELF and actor_hp <= 55:
        return 1 + low_hp_bonus
    return 0


def _disruption_value(effects: list[object]) -> int:
    value = 0
    for effect in effects:
        if isinstance(effect, Stun):
            value += 3
        elif isinstance(effect, (ChakraRemoval, ChakraSteal, ChakraGainSteal)):
            value += 2
    return value
