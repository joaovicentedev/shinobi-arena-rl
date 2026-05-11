from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from naruto_arena.engine.actions import Action, EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.simulator import legal_actions, resolved_skill
from naruto_arena.engine.skills import TargetRule
from naruto_arena.engine.state import GameState

MAX_TEAM_SIZE = 3
MAX_SKILLS_PER_CHARACTER = 5

TARGET_NONE = 0
TARGET_SELF = 1
TARGET_ALL_ENEMIES = 2
TARGET_ALL_ALLIES = 3
TARGET_CHARACTER_OFFSET = 4
TARGET_CODE_COUNT = TARGET_CHARACTER_OFFSET + (MAX_TEAM_SIZE * 2)


class ActionKind(StrEnum):
    END_TURN = "end_turn"
    USE_SKILL = "use_skill"


@dataclass(frozen=True)
class ActionSpec:
    kind: ActionKind
    actor_slot: int = 0
    skill_slot: int = 0
    target_code: int = TARGET_NONE


def build_action_catalog() -> list[ActionSpec]:
    """Build a stable action catalog for a fixed-size policy head.

    Decision: reorder is intentionally excluded from the first RL action space. The
    rule exists in the engine, but adding it immediately makes exploration much harder.
    """

    catalog = [ActionSpec(ActionKind.END_TURN)]
    for actor_slot in range(MAX_TEAM_SIZE):
        for skill_slot in range(MAX_SKILLS_PER_CHARACTER):
            for target_code in range(TARGET_CODE_COUNT):
                catalog.append(
                    ActionSpec(
                        ActionKind.USE_SKILL,
                        actor_slot=actor_slot,
                        skill_slot=skill_slot,
                        target_code=target_code,
                    )
                )
    return catalog


ACTION_CATALOG = build_action_catalog()
NUM_ACTIONS = len(ACTION_CATALOG)


def action_id_to_engine_action(state: GameState, player_id: int, action_id: int) -> Action | None:
    if not 0 <= action_id < NUM_ACTIONS:
        return None
    spec = ACTION_CATALOG[action_id]
    if spec.kind == ActionKind.END_TURN:
        return EndTurnAction(player_id)
    player = state.players[player_id]
    if spec.actor_slot >= len(player.characters):
        return None
    actor = player.characters[spec.actor_slot]
    if spec.skill_slot >= len(actor.skill_order):
        return None
    skill_id = actor.skill_order[spec.skill_slot]
    try:
        skill = resolved_skill(state, actor.instance_id, skill_id)
    except KeyError:
        return None
    target_ids = _target_ids_for_code(
        state,
        player_id,
        actor.instance_id,
        skill.target_rule,
        spec.target_code,
    )
    if target_ids is None:
        return None
    payment = random_chakra_payment(state, player_id, skill.chakra_cost)
    return UseSkillAction(player_id, actor.instance_id, skill.id, target_ids, payment)


def legal_action_mask(state: GameState, player_id: int) -> list[bool]:
    if state.winner is not None or state.active_player != player_id:
        return [False] * NUM_ACTIONS
    legal = legal_actions(state, player_id)
    mask = []
    for action_id in range(NUM_ACTIONS):
        action = action_id_to_engine_action(state, player_id, action_id)
        mask.append(action is not None and _matches_any_legal_action(action, legal))
    return mask


def random_chakra_payment(
    state: GameState,
    player_id: int,
    cost: ChakraCost,
) -> dict[ChakraType, int]:
    """Choose random chakra payment deterministically for the prototype trainer.

    Decision: the policy does not learn random-chakra payment yet. The adapter pays
    from the most abundant remaining type after fixed costs, which keeps training
    focused on action selection while the engine still validates the final payment.
    """

    if cost.random == 0:
        return {}
    remaining = dict(state.players[player_id].chakra.amounts)
    for chakra_type, amount in cost.fixed.items():
        remaining[chakra_type] -= amount
    payment = {chakra_type: 0 for chakra_type in ChakraType}
    for _ in range(cost.random):
        available = [chakra_type for chakra_type, amount in remaining.items() if amount > 0]
        if not available:
            return {}
        chosen = max(available, key=lambda chakra_type: (remaining[chakra_type], chakra_type.value))
        payment[chosen] += 1
        remaining[chosen] -= 1
    return {chakra_type: amount for chakra_type, amount in payment.items() if amount}


def _target_ids_for_code(
    state: GameState,
    player_id: int,
    actor_id: str,
    target_rule: TargetRule,
    target_code: int,
) -> tuple[str, ...] | None:
    player = state.players[player_id]
    enemy = state.players[1 - player_id]
    if target_rule == TargetRule.NONE:
        return () if target_code == TARGET_NONE else None
    if target_rule == TargetRule.SELF:
        return (actor_id,) if target_code == TARGET_SELF else None
    if target_rule == TargetRule.ALL_ENEMIES:
        if target_code != TARGET_ALL_ENEMIES:
            return None
        return tuple(character.instance_id for character in enemy.living_characters())
    if target_rule == TargetRule.ALL_ALLIES:
        if target_code != TARGET_ALL_ALLIES:
            return None
        return tuple(character.instance_id for character in player.living_characters())
    target = _character_for_target_code(state, player_id, target_code)
    if target is None:
        return None
    if target_rule == TargetRule.ONE_ENEMY and target.owner != player_id and target.is_alive:
        return (target.instance_id,)
    if target_rule == TargetRule.ONE_ALLY and target.owner == player_id and target.is_alive:
        return (target.instance_id,)
    return None


def _character_for_target_code(state: GameState, player_id: int, target_code: int):
    slot = target_code - TARGET_CHARACTER_OFFSET
    if not 0 <= slot < MAX_TEAM_SIZE * 2:
        return None
    side = player_id if slot < MAX_TEAM_SIZE else 1 - player_id
    side_slot = slot if slot < MAX_TEAM_SIZE else slot - MAX_TEAM_SIZE
    if side_slot >= len(state.players[side].characters):
        return None
    return state.players[side].characters[side_slot]


def _matches_any_legal_action(action: Action, legal: list[Action]) -> bool:
    if isinstance(action, EndTurnAction):
        return any(isinstance(candidate, EndTurnAction) for candidate in legal)
    if isinstance(action, UseSkillAction):
        return any(
            isinstance(candidate, UseSkillAction)
            and candidate.actor_id == action.actor_id
            and candidate.skill_id == action.skill_id
            and candidate.target_ids == action.target_ids
            for candidate in legal
        )
    return False
