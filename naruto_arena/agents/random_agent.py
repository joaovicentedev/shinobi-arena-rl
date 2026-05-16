from __future__ import annotations

import random

from naruto_arena.engine.actions import Action, UseSkillAction
from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.simulator import legal_actions, resolved_skill
from naruto_arena.engine.state import GameState


class RandomAgent:
    def __init__(self, seed: int = 0, allow_reorder: bool = False) -> None:
        self.rng = random.Random(seed)
        self.allow_reorder = allow_reorder

    def choose_action(self, state: GameState, player_id: int) -> Action:
        actions = _simulation_actions(state, player_id, self.allow_reorder)
        action = self.rng.choice(actions)
        if isinstance(action, UseSkillAction):
            skill = resolved_skill(state, action.actor_id, action.skill_id)
            return _with_random_payment(state, action, skill.chakra_cost, self.rng)
        return action


def _simulation_actions(state: GameState, player_id: int, allow_reorder: bool) -> list[Action]:
    del allow_reorder
    return legal_actions(state, player_id)


def _with_random_payment(
    state: GameState, action: UseSkillAction, cost: ChakraCost, rng: random.Random
) -> UseSkillAction:
    if cost.random == 0:
        return action
    pool = state.players[action.player_id].chakra.copy()
    for chakra_type, amount in cost.fixed.items():
        pool.amounts[chakra_type] -= amount
    payment = {chakra_type: 0 for chakra_type in ChakraType}
    for _ in range(cost.random):
        available = [chakra_type for chakra_type, amount in pool.amounts.items() if amount > 0]
        chosen = rng.choice(available)
        payment[chosen] += 1
        pool.amounts[chosen] -= 1
    return UseSkillAction(
        action.player_id,
        action.actor_id,
        action.skill_id,
        action.target_ids,
        {chakra_type: amount for chakra_type, amount in payment.items() if amount},
    )
