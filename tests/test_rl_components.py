from pathlib import Path

import pytest
import torch

from naruto_arena.data.characters import SAKURA_HARUNO, SASUKE_UCHIHA, UZUMAKI_NARUTO
from naruto_arena.engine.actions import ReorderSkillsAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.rl.action_space import (
    ACTION_CATALOG,
    ACTION_KIND_COUNT,
    MAX_TEAM_SIZE,
    NUM_ACTIONS,
    RANDOM_CHAKRA_OFFSET,
    REORDER_DESTINATION_COUNT,
    TARGET_CHARACTER_OFFSET,
    ActionKind,
    FactoredAction,
    action_id_to_engine_action,
    factored_action_to_engine_action,
    legal_action_mask,
    legal_factored_action_masks,
)
from naruto_arena.rl.env import NarutoArenaLearningEnv
from naruto_arena.rl.model import ActorCritic
from naruto_arena.rl.observation import encode_observation, observation_size


def test_rl_action_mask_always_exposes_end_turn() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    mask = legal_action_mask(state, state.active_player)

    assert len(mask) == NUM_ACTIONS
    assert mask[0]
    assert action_id_to_engine_action(state, state.active_player, 0) is not None


def test_partial_rl_observation_hides_enemy_chakra() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    state.players[1].chakra.add(ChakraType.NINJUTSU, 3)

    partial = encode_observation(state, 0, perfect_info=False)
    perfect = encode_observation(state, 0, perfect_info=True)

    assert partial[-5:] == [0.0] * 5
    assert perfect[-5:] != [0.0] * 5


def test_rl_use_skill_action_selects_random_chakra_payment() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    state.players[0].chakra = ChakraPool.from_counts(
        {ChakraType.TAIJUTSU: 1, ChakraType.NINJUTSU: 1}
    )
    sasuke = state.players[0].characters[2]
    skill_slot = sasuke.skill_order.index("lion_combo")
    action_id = _catalog_index(
        kind=ActionKind.USE_SKILL,
        actor_slot=2,
        skill_slot=skill_slot,
        target_code=TARGET_CHARACTER_OFFSET + MAX_TEAM_SIZE,
        random_chakra_code=_random_chakra_code(ChakraType.NINJUTSU),
    )

    action = action_id_to_engine_action(state, 0, action_id)

    assert isinstance(action, UseSkillAction)
    assert action.skill_id == "lion_combo"
    assert action.random_payment == {ChakraType.NINJUTSU: 1}
    assert legal_action_mask(state, 0)[action_id]


def test_rl_use_skill_action_rejects_unavailable_random_chakra_payment() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    state.players[0].chakra = ChakraPool.from_counts(
        {ChakraType.NINJUTSU: 1, ChakraType.TAIJUTSU: 1}
    )
    sasuke = state.players[0].characters[2]
    skill_slot = sasuke.skill_order.index("lion_combo")
    action_id = _catalog_index(
        kind=ActionKind.USE_SKILL,
        actor_slot=2,
        skill_slot=skill_slot,
        target_code=TARGET_CHARACTER_OFFSET + MAX_TEAM_SIZE,
        random_chakra_code=_random_chakra_code(ChakraType.GENJUTSU),
    )

    assert action_id_to_engine_action(state, 0, action_id) is None
    assert not legal_action_mask(state, 0)[action_id]


def test_rl_reorder_action_can_move_skill_to_start_or_end() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    naruto = state.players[0].characters[0]
    first_skill_id = naruto.skill_order[0]
    last_skill_id = naruto.skill_order[-1]
    move_first_to_end_id = _catalog_index(
        kind=ActionKind.REORDER_SKILL,
        actor_slot=0,
        skill_slot=0,
        reorder_to_end=True,
    )
    move_last_to_start_id = _catalog_index(
        kind=ActionKind.REORDER_SKILL,
        actor_slot=0,
        skill_slot=len(naruto.skill_order) - 1,
        reorder_to_end=False,
    )

    move_first_to_end = action_id_to_engine_action(state, 0, move_first_to_end_id)
    move_last_to_start = action_id_to_engine_action(state, 0, move_last_to_start_id)

    assert isinstance(move_first_to_end, ReorderSkillsAction)
    assert move_first_to_end.skill_id == first_skill_id
    assert move_first_to_end.new_index == len(naruto.skill_order) - 1
    assert legal_action_mask(state, 0)[move_first_to_end_id]
    assert isinstance(move_last_to_start, ReorderSkillsAction)
    assert move_last_to_start.skill_id == last_skill_id
    assert move_last_to_start.new_index == 0
    assert legal_action_mask(state, 0)[move_last_to_start_id]


def test_rl_factored_masks_expose_small_policy_heads() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    masks = legal_factored_action_masks(state, 0)

    assert len(masks["kind"]) == ACTION_KIND_COUNT
    assert len(masks["actor"]) == MAX_TEAM_SIZE
    assert len(masks["skill"]) == 5
    assert len(masks["target"]) == 10
    assert len(masks["random_chakra"]) == 5
    assert len(masks["reorder_destination"]) == REORDER_DESTINATION_COUNT
    assert masks["kind"][0]


def test_rl_factored_action_selects_random_chakra_payment() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    state.players[0].chakra = ChakraPool.from_counts(
        {ChakraType.TAIJUTSU: 1, ChakraType.NINJUTSU: 1}
    )
    sasuke = state.players[0].characters[2]
    skill_slot = sasuke.skill_order.index("lion_combo")
    action = FactoredAction(
        ActionKind.USE_SKILL,
        actor_slot=2,
        skill_slot=skill_slot,
        target_code=TARGET_CHARACTER_OFFSET + MAX_TEAM_SIZE,
        random_chakra_code=_random_chakra_code(ChakraType.NINJUTSU),
    )

    engine_action = factored_action_to_engine_action(state, 0, action)

    assert isinstance(engine_action, UseSkillAction)
    assert engine_action.random_payment == {ChakraType.NINJUTSU: 1}


def test_rl_opponent_requires_model_path() -> None:
    with pytest.raises(ValueError, match="opponent-model-path"):
        NarutoArenaLearningEnv(opponent="rl")


def test_rl_opponent_can_load_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "opponent.pt"
    model = ActorCritic(observation_size())
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": observation_size(),
            "policy_type": "factored",
            "perfect_info": False,
        },
        model_path,
    )

    env = NarutoArenaLearningEnv(opponent="rl", opponent_model_path=model_path)

    assert env.opponent_name == "rl"


def _random_chakra_code(chakra_type: ChakraType) -> int:
    return RANDOM_CHAKRA_OFFSET + tuple(ChakraType).index(chakra_type)


def _catalog_index(
    *,
    kind: ActionKind,
    actor_slot: int,
    skill_slot: int,
    target_code: int = 0,
    random_chakra_code: int = 0,
    reorder_to_end: bool = False,
) -> int:
    spec = next(
        spec
        for spec in ACTION_CATALOG
        if spec.kind == kind
        and spec.actor_slot == actor_slot
        and spec.skill_slot == skill_slot
        and spec.target_code == target_code
        and spec.random_chakra_code == random_chakra_code
        and spec.reorder_to_end == reorder_to_end
    )
    return ACTION_CATALOG.index(spec)
