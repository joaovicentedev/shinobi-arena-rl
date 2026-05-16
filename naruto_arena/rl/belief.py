from __future__ import annotations

from dataclasses import dataclass, field

from naruto_arena.engine.actions import Action, EndTurnAction, GetChakraAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraType
from naruto_arena.engine.simulator import resolved_skill
from naruto_arena.engine.state import GameState, UsedSkillState

MAX_BELIEF_CHAKRA = 12


@dataclass
class ChakraBelief:
    estimate: dict[ChakraType, float] = field(
        default_factory=lambda: {chakra_type: 0.0 for chakra_type in ChakraType}
    )
    minimum: dict[ChakraType, float] = field(
        default_factory=lambda: {chakra_type: 0.0 for chakra_type in ChakraType}
    )
    maximum: dict[ChakraType, float] = field(
        default_factory=lambda: {chakra_type: 0.0 for chakra_type in ChakraType}
    )

    def features(self) -> list[float]:
        values = [
            min(self.estimate[chakra_type], MAX_BELIEF_CHAKRA) / MAX_BELIEF_CHAKRA
            for chakra_type in ChakraType
        ]
        values.extend(
            min(self.minimum[chakra_type], MAX_BELIEF_CHAKRA) / MAX_BELIEF_CHAKRA
            for chakra_type in ChakraType
        )
        values.extend(
            min(self.maximum[chakra_type], MAX_BELIEF_CHAKRA) / MAX_BELIEF_CHAKRA
            for chakra_type in ChakraType
        )
        values.append(min(sum(self.estimate.values()), MAX_BELIEF_CHAKRA) / MAX_BELIEF_CHAKRA)
        return values


class ChakraBeliefTracker:
    def __init__(self, observer_id: int) -> None:
        self.observer_id = observer_id
        self.enemy_id = 1 - observer_id
        self.enemy = ChakraBelief()
        self.last_turn_number: int | None = None

    def reset(self, state: GameState) -> None:
        self.enemy = ChakraBelief()
        self.last_turn_number = state.turn_number
        if state.active_player == self.enemy_id:
            self._gain_for_enemy_turn_start(state)

    def features(self, state: GameState) -> list[float]:
        if self.last_turn_number is None:
            self.reset(state)
        return self.enemy.features()

    def observe_action(self, before: GameState, action: Action, after: GameState) -> None:
        if self.last_turn_number is None or after.turn_number < self.last_turn_number:
            self.reset(after)
        if action.player_id == self.enemy_id and isinstance(action, UseSkillAction):
            self._observe_enemy_skill(before, action)
        if action.player_id == self.enemy_id and isinstance(action, GetChakraAction):
            self._observe_enemy_chakra_exchange(action)
        if isinstance(action, EndTurnAction) and after.active_player == self.enemy_id:
            self._gain_for_enemy_turn_start(after)
        self.last_turn_number = after.turn_number

    def _gain_for_enemy_turn_start(self, state: GameState) -> None:
        gain_count = _visible_chakra_gain_count(state, self.enemy_id)
        if gain_count <= 0:
            return
        gain_per_type = gain_count / len(tuple(ChakraType))
        for chakra_type in ChakraType:
            self.enemy.estimate[chakra_type] += gain_per_type
            self.enemy.maximum[chakra_type] += gain_count

    def _observe_enemy_skill(self, state: GameState, action: UseSkillAction) -> None:
        used_skill = UsedSkillState(action.actor_id, action.skill_id, 1, action.target_ids)
        if _is_invisible_to_player(state, used_skill, self.observer_id):
            return
        actor = state.get_character(action.actor_id)
        skill = resolved_skill(state, action.actor_id, action.skill_id)
        if actor.owner != self.enemy_id:
            return
        for chakra_type, amount in skill.chakra_cost.fixed.items():
            self._spend_known(chakra_type, amount)
        self._spend_random(skill.chakra_cost.random)

    def _observe_enemy_chakra_exchange(self, action: GetChakraAction) -> None:
        self._spend_random(5)
        self.enemy.estimate[action.chakra_type] += 1.0
        self.enemy.minimum[action.chakra_type] += 1.0
        self.enemy.maximum[action.chakra_type] += 1.0

    def _spend_known(self, chakra_type: ChakraType, amount: int) -> None:
        self.enemy.estimate[chakra_type] = max(0.0, self.enemy.estimate[chakra_type] - amount)
        self.enemy.minimum[chakra_type] = max(0.0, self.enemy.minimum[chakra_type] - amount)
        self.enemy.maximum[chakra_type] = max(0.0, self.enemy.maximum[chakra_type] - amount)

    def _spend_random(self, amount: int) -> None:
        remaining = float(amount)
        while remaining > 0 and sum(self.enemy.estimate.values()) > 0:
            chakra_type = max(ChakraType, key=lambda item: self.enemy.estimate[item])
            spent = min(remaining, self.enemy.estimate[chakra_type])
            self.enemy.estimate[chakra_type] -= spent
            remaining -= spent
        for chakra_type in ChakraType:
            self.enemy.maximum[chakra_type] = max(0.0, self.enemy.maximum[chakra_type] - amount)


def fallback_enemy_chakra_belief_features(state: GameState, player_id: int) -> list[float]:
    enemy_id = 1 - player_id
    belief = ChakraBelief()
    estimate_total = _estimated_enemy_chakra_total(state, enemy_id, player_id)
    per_type_estimate = estimate_total / len(tuple(ChakraType))
    for chakra_type in ChakraType:
        belief.estimate[chakra_type] = per_type_estimate
        belief.maximum[chakra_type] = estimate_total
    return belief.features()


def _estimated_enemy_chakra_total(state: GameState, enemy_id: int, observer_id: int) -> int:
    starts = _estimated_turn_starts(state, enemy_id)
    living = len(state.players[enemy_id].living_characters())
    if starts <= 0:
        return 0
    first_gain = 1 if enemy_id == 0 else living
    later_gain = max(0, starts - 1) * living
    visible_spend = sum(
        _visible_skill_chakra_spend(state, used_skill.actor_id, used_skill.skill_id)
        for used_skill in state.players[enemy_id].skill_stack
        if not _is_invisible_to_player(state, used_skill, observer_id)
    )
    return max(0, min(MAX_BELIEF_CHAKRA, first_gain + later_gain - visible_spend))


def _estimated_turn_starts(state: GameState, player_id: int) -> int:
    completed_half_turns = state.turn_number - 1
    starts = completed_half_turns // 2
    if player_id == 0:
        starts += 1
    if state.active_player == player_id and player_id == 1:
        starts += 1
    return starts


def _visible_chakra_gain_count(state: GameState, player_id: int) -> int:
    if state.turn_number == 1 and player_id == 0:
        return 1
    return len(state.players[player_id].living_characters())


def _visible_skill_chakra_spend(state: GameState, actor_id: str, skill_id: str) -> int:
    try:
        skill = resolved_skill(state, actor_id, skill_id)
    except KeyError:
        return 0
    return sum(skill.chakra_cost.fixed.values()) + skill.chakra_cost.random


def _is_invisible_to_player(state: GameState, used_skill: UsedSkillState, player_id: int) -> bool:
    actor = state.get_character(used_skill.actor_id)
    if actor.owner == player_id:
        return False
    try:
        skill = resolved_skill(state, actor.instance_id, used_skill.skill_id)
    except KeyError:
        return True
    return "invisible" in skill.description.lower()
