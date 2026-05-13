from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from naruto_arena.engine.actions import Action, EndTurnAction, ReorderSkillsAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.simulator import legal_actions, resolved_skill
from naruto_arena.engine.skills import TargetRule
from naruto_arena.engine.state import GameState

MAX_TEAM_SIZE = 3
MAX_SKILLS_PER_CHARACTER = 9

TARGET_NONE = 0
TARGET_SELF = 1
TARGET_ALL_ENEMIES = 2
TARGET_ALL_ALLIES = 3
TARGET_CHARACTER_OFFSET = 4
TARGET_CODE_COUNT = TARGET_CHARACTER_OFFSET + (MAX_TEAM_SIZE * 2)


class ActionKind(StrEnum):
    END_TURN = "end_turn"
    USE_SKILL = "use_skill"
    REORDER_SKILL = "reorder_skill"


ACTION_KIND_ORDER = (ActionKind.END_TURN, ActionKind.USE_SKILL, ActionKind.REORDER_SKILL)
ACTION_KIND_TO_INDEX = {kind: index for index, kind in enumerate(ACTION_KIND_ORDER)}
ACTION_KIND_COUNT = len(ACTION_KIND_ORDER)
REORDER_DESTINATION_COUNT = 2


@dataclass(frozen=True)
class ActionSpec:
    kind: ActionKind
    actor_slot: int = 0
    skill_slot: int = 0
    target_code: int = TARGET_NONE
    random_chakra_code: int = 0
    reorder_to_end: bool = False


@dataclass(frozen=True)
class FactoredAction:
    kind: ActionKind
    actor_slot: int = 0
    skill_slot: int = 0
    target_code: int = TARGET_NONE
    random_chakra_code: int = 0
    reorder_to_end: bool = False


RANDOM_CHAKRA_NONE = 0
RANDOM_CHAKRA_OFFSET = 1
RANDOM_CHAKRA_CODE_COUNT = RANDOM_CHAKRA_OFFSET + len(ChakraType)


def build_action_catalog() -> list[ActionSpec]:
    """Build a stable action catalog for a fixed-size policy head."""

    catalog = [ActionSpec(ActionKind.END_TURN)]
    for actor_slot in range(MAX_TEAM_SIZE):
        for skill_slot in range(MAX_SKILLS_PER_CHARACTER):
            for target_code in range(TARGET_CODE_COUNT):
                for random_chakra_code in range(RANDOM_CHAKRA_CODE_COUNT):
                    catalog.append(
                        ActionSpec(
                            ActionKind.USE_SKILL,
                            actor_slot=actor_slot,
                            skill_slot=skill_slot,
                            target_code=target_code,
                            random_chakra_code=random_chakra_code,
                        )
                    )
            for reorder_to_end in (False, True):
                catalog.append(
                    ActionSpec(
                        ActionKind.REORDER_SKILL,
                        actor_slot=actor_slot,
                        skill_slot=skill_slot,
                        reorder_to_end=reorder_to_end,
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
    if spec.kind == ActionKind.REORDER_SKILL:
        skill_id = actor.skill_order[spec.skill_slot]
        new_index = len(actor.skill_order) - 1 if spec.reorder_to_end else 0
        return ReorderSkillsAction(player_id, actor.instance_id, skill_id, new_index)
    if spec.kind != ActionKind.USE_SKILL:
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
    payment = _random_chakra_payment(
        state,
        player_id,
        skill.chakra_cost,
        spec.random_chakra_code,
    )
    if payment is None:
        return None
    return UseSkillAction(player_id, actor.instance_id, skill.id, target_ids, payment)


def factored_action_to_engine_action(
    state: GameState,
    player_id: int,
    action: FactoredAction,
) -> Action | None:
    if action.kind == ActionKind.END_TURN:
        return EndTurnAction(player_id)
    player = state.players[player_id]
    if action.actor_slot >= len(player.characters):
        return None
    actor = player.characters[action.actor_slot]
    if action.skill_slot >= len(actor.skill_order):
        return None
    if action.kind == ActionKind.REORDER_SKILL:
        skill_id = actor.skill_order[action.skill_slot]
        new_index = len(actor.skill_order) - 1 if action.reorder_to_end else 0
        return ReorderSkillsAction(player_id, actor.instance_id, skill_id, new_index)
    if action.kind != ActionKind.USE_SKILL:
        return None
    skill_id = actor.skill_order[action.skill_slot]
    try:
        skill = resolved_skill(state, actor.instance_id, skill_id)
    except KeyError:
        return None
    target_ids = _target_ids_for_code(
        state,
        player_id,
        actor.instance_id,
        skill.target_rule,
        action.target_code,
    )
    if target_ids is None:
        return None
    payment = _random_chakra_payment(
        state,
        player_id,
        skill.chakra_cost,
        action.random_chakra_code,
    )
    if payment is None:
        return None
    return UseSkillAction(player_id, actor.instance_id, skill.id, target_ids, payment)


def legal_factored_action_masks(
    state: GameState,
    player_id: int,
    partial: FactoredAction | None = None,
) -> dict[str, list[bool]]:
    if state.winner is not None or state.active_player != player_id:
        return _empty_factored_masks()
    partial = partial or FactoredAction(ActionKind.END_TURN)
    legal = legal_actions(state, player_id)
    return {
        "kind": _legal_kind_mask(state, player_id, legal),
        "actor": _legal_actor_mask(state, player_id, legal, partial.kind),
        "skill": _legal_skill_mask(state, player_id, legal, partial),
        "target": _legal_target_mask(state, player_id, legal, partial),
        "random_chakra": _legal_random_chakra_mask(state, player_id, legal, partial),
        "reorder_destination": _legal_reorder_destination_mask(state, player_id, legal, partial),
    }


def legal_action_mask(state: GameState, player_id: int) -> list[bool]:
    if state.winner is not None or state.active_player != player_id:
        return [False] * NUM_ACTIONS
    legal = legal_actions(state, player_id)
    mask = []
    for action_id in range(NUM_ACTIONS):
        action = action_id_to_engine_action(state, player_id, action_id)
        mask.append(action is not None and _matches_any_legal_action(action, legal))
    return mask


def _empty_factored_masks() -> dict[str, list[bool]]:
    return {
        "kind": [False] * ACTION_KIND_COUNT,
        "actor": [False] * MAX_TEAM_SIZE,
        "skill": [False] * MAX_SKILLS_PER_CHARACTER,
        "target": [False] * TARGET_CODE_COUNT,
        "random_chakra": [False] * RANDOM_CHAKRA_CODE_COUNT,
        "reorder_destination": [False] * REORDER_DESTINATION_COUNT,
    }


def _legal_kind_mask(state: GameState, player_id: int, legal: list[Action]) -> list[bool]:
    del state, player_id
    return [
        any(isinstance(action, EndTurnAction) for action in legal),
        any(isinstance(action, UseSkillAction) for action in legal),
        any(isinstance(action, ReorderSkillsAction) for action in legal),
    ]


def _legal_actor_mask(
    state: GameState,
    player_id: int,
    legal: list[Action],
    kind: ActionKind,
) -> list[bool]:
    mask = [False] * MAX_TEAM_SIZE
    for actor_slot, actor in enumerate(state.players[player_id].characters[:MAX_TEAM_SIZE]):
        if kind == ActionKind.USE_SKILL:
            mask[actor_slot] = any(
                isinstance(action, UseSkillAction) and action.actor_id == actor.instance_id
                for action in legal
            )
        elif kind == ActionKind.REORDER_SKILL:
            mask[actor_slot] = any(
                isinstance(action, ReorderSkillsAction) and action.character_id == actor.instance_id
                for action in legal
            )
    return mask


def _legal_skill_mask(
    state: GameState,
    player_id: int,
    legal: list[Action],
    partial: FactoredAction,
) -> list[bool]:
    mask = [False] * MAX_SKILLS_PER_CHARACTER
    if partial.actor_slot >= len(state.players[player_id].characters):
        return mask
    actor = state.players[player_id].characters[partial.actor_slot]
    for skill_slot, skill_id in enumerate(actor.skill_order[:MAX_SKILLS_PER_CHARACTER]):
        if partial.kind == ActionKind.USE_SKILL:
            try:
                resolved_skill_id = resolved_skill(state, actor.instance_id, skill_id).id
            except KeyError:
                continue
            mask[skill_slot] = any(
                isinstance(action, UseSkillAction)
                and action.actor_id == actor.instance_id
                and action.skill_id == resolved_skill_id
                for action in legal
            )
        elif partial.kind == ActionKind.REORDER_SKILL:
            mask[skill_slot] = any(
                isinstance(action, ReorderSkillsAction)
                and action.character_id == actor.instance_id
                and action.skill_id == skill_id
                for action in legal
            )
    return mask


def _legal_target_mask(
    state: GameState,
    player_id: int,
    legal: list[Action],
    partial: FactoredAction,
) -> list[bool]:
    mask = [False] * TARGET_CODE_COUNT
    if partial.kind != ActionKind.USE_SKILL:
        return mask
    skill = _skill_for_factored_slot(state, player_id, partial)
    if skill is None:
        return mask
    actor = state.players[player_id].characters[partial.actor_slot]
    legal_uses = [
        action
        for action in legal
        if isinstance(action, UseSkillAction)
        and action.actor_id == actor.instance_id
        and action.skill_id == skill.id
    ]
    for target_code in range(TARGET_CODE_COUNT):
        target_ids = _target_ids_for_code(
            state,
            player_id,
            actor.instance_id,
            skill.target_rule,
            target_code,
        )
        mask[target_code] = target_ids is not None and any(
            action.target_ids == target_ids for action in legal_uses
        )
    return mask


def _legal_random_chakra_mask(
    state: GameState,
    player_id: int,
    legal: list[Action],
    partial: FactoredAction,
) -> list[bool]:
    mask = [False] * RANDOM_CHAKRA_CODE_COUNT
    if partial.kind != ActionKind.USE_SKILL:
        return mask
    skill = _skill_for_factored_slot(state, player_id, partial)
    if skill is None:
        return mask
    actor = state.players[player_id].characters[partial.actor_slot]
    target_ids = _target_ids_for_code(
        state,
        player_id,
        actor.instance_id,
        skill.target_rule,
        partial.target_code,
    )
    if target_ids is None:
        return mask
    target_is_legal = any(
        isinstance(action, UseSkillAction)
        and action.actor_id == actor.instance_id
        and action.skill_id == skill.id
        and action.target_ids == target_ids
        for action in legal
    )
    if not target_is_legal:
        return mask
    for random_chakra_code in range(RANDOM_CHAKRA_CODE_COUNT):
        mask[random_chakra_code] = (
            _random_chakra_payment(state, player_id, skill.chakra_cost, random_chakra_code)
            is not None
        )
    return mask


def _legal_reorder_destination_mask(
    state: GameState,
    player_id: int,
    legal: list[Action],
    partial: FactoredAction,
) -> list[bool]:
    mask = [False] * REORDER_DESTINATION_COUNT
    if partial.kind != ActionKind.REORDER_SKILL:
        return mask
    for destination in range(REORDER_DESTINATION_COUNT):
        action = FactoredAction(
            ActionKind.REORDER_SKILL,
            actor_slot=partial.actor_slot,
            skill_slot=partial.skill_slot,
            reorder_to_end=bool(destination),
        )
        engine_action = factored_action_to_engine_action(state, player_id, action)
        mask[destination] = engine_action is not None and _matches_any_legal_action(
            engine_action,
            legal,
        )
    return mask


def _skill_for_factored_slot(
    state: GameState,
    player_id: int,
    action: FactoredAction,
):
    if action.actor_slot >= len(state.players[player_id].characters):
        return None
    actor = state.players[player_id].characters[action.actor_slot]
    if action.skill_slot >= len(actor.skill_order):
        return None
    try:
        return resolved_skill(state, actor.instance_id, actor.skill_order[action.skill_slot])
    except KeyError:
        return None


def random_chakra_payment(
    state: GameState,
    player_id: int,
    cost: ChakraCost,
    random_chakra_code: int = RANDOM_CHAKRA_NONE,
) -> dict[ChakraType, int]:
    if cost.random > 0 and random_chakra_code == RANDOM_CHAKRA_NONE:
        return _deterministic_random_chakra_payment(state, player_id, cost)
    return _random_chakra_payment(state, player_id, cost, random_chakra_code) or {}


def _random_chakra_payment(
    state: GameState,
    player_id: int,
    cost: ChakraCost,
    random_chakra_code: int,
) -> dict[ChakraType, int] | None:
    """Build a random-chakra payment from the policy's preferred chakra type."""

    if cost.random == 0:
        return {} if random_chakra_code == RANDOM_CHAKRA_NONE else None
    preferred = _chakra_type_for_random_code(random_chakra_code)
    if preferred is None:
        return None
    remaining = dict(state.players[player_id].chakra.amounts)
    for chakra_type, amount in cost.fixed.items():
        remaining[chakra_type] -= amount
    if remaining.get(preferred, 0) <= 0:
        return None
    payment = {chakra_type: 0 for chakra_type in ChakraType}
    payment[preferred] += 1
    remaining[preferred] -= 1
    for _ in range(cost.random - 1):
        available = [chakra_type for chakra_type, amount in remaining.items() if amount > 0]
        if not available:
            return None
        chosen = max(available, key=lambda chakra_type: (remaining[chakra_type], chakra_type.value))
        payment[chosen] += 1
        remaining[chosen] -= 1
    result = {chakra_type: amount for chakra_type, amount in payment.items() if amount}
    if not state.players[player_id].chakra.can_pay(cost, result):
        return None
    return result


def _deterministic_random_chakra_payment(
    state: GameState,
    player_id: int,
    cost: ChakraCost,
) -> dict[ChakraType, int]:
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


def _chakra_type_for_random_code(random_chakra_code: int) -> ChakraType | None:
    chakra_index = random_chakra_code - RANDOM_CHAKRA_OFFSET
    chakra_types = tuple(ChakraType)
    if not 0 <= chakra_index < len(chakra_types):
        return None
    return chakra_types[chakra_index]


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
    if isinstance(action, ReorderSkillsAction):
        return any(
            isinstance(candidate, ReorderSkillsAction)
            and candidate.character_id == action.character_id
            and candidate.skill_id == action.skill_id
            and candidate.new_index == action.new_index
            for candidate in legal
        )
    return False
