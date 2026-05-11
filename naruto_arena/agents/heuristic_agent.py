from __future__ import annotations

import random

from naruto_arena.agents.random_agent import _simulation_actions, _with_random_payment
from naruto_arena.engine.actions import Action, UseSkillAction
from naruto_arena.engine.effects import DirectDamage
from naruto_arena.engine.simulator import resolved_skill
from naruto_arena.engine.state import GameState


class SimpleHeuristicAgent:
    def __init__(self, seed: int = 0, allow_reorder: bool = False) -> None:
        self.rng = random.Random(seed)
        self.allow_reorder = allow_reorder

    def choose_action(self, state: GameState, player_id: int) -> Action:
        actions = _simulation_actions(state, player_id, self.allow_reorder)
        damaging: list[UseSkillAction] = []
        for action in actions:
            if isinstance(action, UseSkillAction):
                skill = resolved_skill(state, action.actor_id, action.skill_id)
                if any(isinstance(effect, DirectDamage) for effect in skill.all_effects(state, action.actor_id)):
                    damaging.append(action)
        if damaging:
            damaging.sort(key=lambda action: state.get_character(action.target_ids[0]).hp)
            action = damaging[0]
            skill = resolved_skill(state, action.actor_id, action.skill_id)
            return _with_random_payment(state, action, skill.chakra_cost, self.rng)
        action = self.rng.choice(actions)
        if isinstance(action, UseSkillAction):
            skill = resolved_skill(state, action.actor_id, action.skill_id)
            return _with_random_payment(state, action, skill.chakra_cost, self.rng)
        return action
