from __future__ import annotations

import copy
from dataclasses import dataclass

from naruto_arena.agents.random_agent import _simulation_actions
from naruto_arena.engine.actions import Action, EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.effects import DirectDamage
from naruto_arena.engine.simulator import apply_action, resolved_skill
from naruto_arena.engine.state import GameState


@dataclass(frozen=True)
class MinimaxConfig:
    depth: int = 2
    allow_reorder: bool = False
    max_actions: int = 18


class MinimaxAgent:
    def __init__(self, config: MinimaxConfig | None = None) -> None:
        self.config = config or MinimaxConfig()

    def choose_action(self, state: GameState, player_id: int) -> Action:
        actions = candidate_actions(state, player_id, self.config)
        if not actions:
            return EndTurnAction(player_id)

        best_action = actions[0]
        best_score = float("-inf")
        for action in actions:
            child = copy.deepcopy(state)
            apply_action(child, action)
            score = minimax_score(child, player_id, self.config.depth - 1, self.config)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action


def minimax_score(
    state: GameState,
    root_player: int,
    depth: int,
    config: MinimaxConfig,
) -> float:
    if depth <= 0 or state.winner is not None:
        return evaluate_state(state, root_player)

    actions = candidate_actions(state, state.active_player, config)
    if not actions:
        return evaluate_state(state, root_player)

    scores = []
    for action in actions:
        child = copy.deepcopy(state)
        apply_action(child, action)
        scores.append(minimax_score(child, root_player, depth - 1, config))

    if state.active_player == root_player:
        return max(scores)
    return min(scores)


def candidate_actions(
    state: GameState,
    player_id: int,
    config: MinimaxConfig,
) -> list[Action]:
    actions = _simulation_actions(state, player_id, config.allow_reorder)
    paid_actions = [with_deterministic_payment(state, action) for action in actions]
    paid_actions.sort(key=lambda action: action_order_score(state, action), reverse=True)
    return paid_actions[: config.max_actions]


def with_deterministic_payment(state: GameState, action: Action) -> Action:
    if not isinstance(action, UseSkillAction):
        return action
    skill = resolved_skill(state, action.actor_id, action.skill_id)
    payment = deterministic_random_payment(
        state.players[action.player_id].chakra.amounts,
        skill.chakra_cost,
    )
    return UseSkillAction(
        action.player_id,
        action.actor_id,
        action.skill_id,
        action.target_ids,
        payment,
    )


def deterministic_random_payment(
    chakra_amounts: dict[ChakraType, int],
    cost: ChakraCost,
) -> dict[ChakraType, int]:
    if cost.random == 0:
        return {}

    remaining = dict(chakra_amounts)
    for chakra_type, amount in cost.fixed.items():
        remaining[chakra_type] -= amount

    payment: dict[ChakraType, int] = {}
    random_left = cost.random
    for chakra_type in ChakraType:
        if random_left == 0:
            break
        spend = min(remaining[chakra_type], random_left)
        if spend <= 0:
            continue
        payment[chakra_type] = spend
        random_left -= spend
    return payment


def action_order_score(state: GameState, action: Action) -> float:
    if isinstance(action, EndTurnAction):
        return -1_000
    if not isinstance(action, UseSkillAction):
        return -500

    score = 0.0
    skill = resolved_skill(state, action.actor_id, action.skill_id)
    effects = skill.all_effects(state, action.actor_id)
    if any(isinstance(effect, DirectDamage) for effect in effects):
        score += 100
        if action.target_ids:
            score += 100 - min(state.get_character(target_id).hp for target_id in action.target_ids)
    if skill.chakra_cost.is_free():
        score += 10
    return score


def evaluate_state(state: GameState, player_id: int) -> float:
    if state.winner == player_id:
        return 100_000
    if state.winner == 1 - player_id:
        return -100_000

    player = state.players[player_id]
    enemy = state.players[1 - player_id]

    player_hp = sum(character.hp for character in player.characters)
    enemy_hp = sum(character.hp for character in enemy.characters)
    player_alive = len(player.living_characters())
    enemy_alive = len(enemy.living_characters())
    player_chakra = player.chakra.total()
    enemy_chakra = enemy.chakra.total()

    score = 0.0
    score += (player_alive - enemy_alive) * 200
    score += player_hp - enemy_hp
    score += (player_chakra - enemy_chakra) * 5

    for character in player.characters:
        score += sum(defense.amount for defense in character.status.defenses) * 0.5
    for character in enemy.characters:
        score -= sum(defense.amount for defense in character.status.defenses) * 0.5

    return score
