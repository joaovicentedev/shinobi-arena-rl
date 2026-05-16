import copy
from pathlib import Path

import pytest
import torch

from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.data.characters import SAKURA_HARUNO, SASUKE_UCHIHA, UZUMAKI_NARUTO
from naruto_arena.engine.actions import (
    EndTurnAction,
    GetChakraAction,
    ReorderSkillsAction,
    UseSkillAction,
)
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action
from naruto_arena.rl.action_space import (
    ACTION_CATALOG,
    ACTION_KIND_COUNT,
    GET_CHAKRA_CODE_COUNT,
    MAX_SKILLS_PER_CHARACTER,
    MAX_STACK_SIZE,
    MAX_TEAM_SIZE,
    NUM_ACTIONS,
    RANDOM_CHAKRA_OFFSET,
    REORDER_DIRECTION_COUNT,
    TARGET_CHARACTER_OFFSET,
    ActionKind,
    FactoredAction,
    action_id_to_engine_action,
    factored_action_to_engine_action,
    legal_action_mask,
    legal_factored_action_masks,
)
from naruto_arena.rl.belief import ChakraBeliefTracker
from naruto_arena.rl.env import NarutoArenaLearningEnv
from naruto_arena.rl.model import (
    ActorCritic,
    RecurrentTransformerActorCritic,
    TransformerActorCritic,
)
from naruto_arena.rl.observation import (
    BASE_CHARACTER_FEATURE_SIZE,
    BASE_OBSERVATION_VERSION,
    CHARACTER_FEATURE_SIZE,
    CHARACTER_SLOTS,
    COMPACT_BASE_CHARACTER_FEATURE_SIZE,
    COMPACT_OBSERVATION_VERSION,
    SKILL_FEATURE_SIZE,
    encode_observation,
    observation_size,
)
from naruto_arena.rl.observation import (
    MAX_SKILLS_PER_CHARACTER as OBSERVATION_MAX_SKILLS_PER_CHARACTER,
)


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
    enemy_chakra_start = 7 + CHARACTER_SLOTS * CHARACTER_FEATURE_SIZE + 5

    assert partial[enemy_chakra_start : enemy_chakra_start + 4] != [3 / 12, 0.0, 0.0, 0.0]
    assert perfect[enemy_chakra_start : enemy_chakra_start + 4] != [3 / 12, 0.0, 0.0, 0.0]


def test_rl_observation_includes_skill_feature_map() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    observation = encode_observation(state, 0)
    sasuke_block = 7 + (2 * CHARACTER_FEATURE_SIZE)
    lion_combo_features = sasuke_block + COMPACT_BASE_CHARACTER_FEATURE_SIZE

    assert observation_size() == 7 + CHARACTER_SLOTS * CHARACTER_FEATURE_SIZE + 5 + 13 + (12 * 34)
    assert CHARACTER_FEATURE_SIZE < BASE_CHARACTER_FEATURE_SIZE + (
        OBSERVATION_MAX_SKILLS_PER_CHARACTER * SKILL_FEATURE_SIZE
    )
    assert observation[lion_combo_features] == 1.0
    assert observation[lion_combo_features + 6] == 0.25
    assert observation[lion_combo_features + 9] == 0.25
    assert observation[lion_combo_features + 32] == 0.30
    assert observation[lion_combo_features + SKILL_FEATURE_SIZE] == 1.0


def test_rl_observation_includes_visible_stacks_for_both_players() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    observation = encode_observation(state, 0)
    stack_start = 7 + CHARACTER_SLOTS * CHARACTER_FEATURE_SIZE + 5 + 13
    present_count = sum(
        observation[stack_start + (index * 34)] for index in range(MAX_STACK_SIZE)
    )

    assert present_count == min(
        len(state.players[0].skill_stack) + len(state.players[1].skill_stack),
        MAX_STACK_SIZE,
    )


def test_chakra_belief_tracker_updates_on_visible_enemy_turn_gain() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    tracker = ChakraBeliefTracker(observer_id=0)
    tracker.reset(state)
    before = copy.deepcopy(state)

    apply_action(state, EndTurnAction(0))
    tracker.observe_action(before, EndTurnAction(0), state)

    features = tracker.features(state)
    assert features[12] == 3 / 12


def test_rl_skill_vectors_allow_catalog_max_skill_slots() -> None:
    assert MAX_SKILLS_PER_CHARACTER == 9
    assert OBSERVATION_MAX_SKILLS_PER_CHARACTER == 9


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


def test_rl_reorder_action_can_move_stack_item_left_or_right() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SASUKE_UCHIHA, SAKURA_HARUNO],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    move_first_right_id = _catalog_index(
        kind=ActionKind.REORDER_STACK,
        stack_index=0,
        reorder_direction=1,
    )
    move_second_left_id = _catalog_index(
        kind=ActionKind.REORDER_STACK,
        stack_index=1,
        reorder_direction=0,
    )

    move_first_right = action_id_to_engine_action(state, 0, move_first_right_id)
    move_second_left = action_id_to_engine_action(state, 0, move_second_left_id)

    assert isinstance(move_first_right, ReorderSkillsAction)
    assert move_first_right.new_index == 1
    assert legal_action_mask(state, 0)[move_first_right_id]
    assert isinstance(move_second_left, ReorderSkillsAction)
    assert move_second_left.new_index == 0
    assert legal_action_mask(state, 0)[move_second_left_id]


def test_rl_get_chakra_action_exchanges_five_chakras_for_choice() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    state.players[0].chakra = ChakraPool.from_counts(
        {ChakraType.NINJUTSU: 2, ChakraType.TAIJUTSU: 2, ChakraType.GENJUTSU: 1}
    )
    action_id = _catalog_index(
        kind=ActionKind.GET_CHAKRA,
        get_chakra_code=tuple(ChakraType).index(ChakraType.BLOODLINE),
    )

    action = action_id_to_engine_action(state, 0, action_id)

    assert isinstance(action, GetChakraAction)
    assert action.chakra_type == ChakraType.BLOODLINE
    assert legal_action_mask(state, 0)[action_id]


def test_rl_factored_masks_expose_small_policy_heads() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    masks = legal_factored_action_masks(state, 0)

    assert len(masks["kind"]) == ACTION_KIND_COUNT
    assert len(masks["actor"]) == MAX_TEAM_SIZE
    assert len(masks["skill"]) == MAX_SKILLS_PER_CHARACTER
    assert len(masks["target"]) == 10
    assert len(masks["random_chakra"]) == 5
    assert len(masks["get_chakra"]) == GET_CHAKRA_CODE_COUNT
    assert len(masks["stack_index"]) == MAX_STACK_SIZE
    assert len(masks["reorder_direction"]) == REORDER_DIRECTION_COUNT
    assert masks["kind"][0]


def test_transformer_actor_critic_outputs_factored_policy_shapes() -> None:
    model = TransformerActorCritic(observation_size())
    observation = torch.tensor(
        encode_observation(
            create_initial_state(
                [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
                [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
            ),
            0,
        ),
        dtype=torch.float32,
    ).unsqueeze(0)

    policy, value = model(observation)

    assert policy.kind.shape == (1, ACTION_KIND_COUNT)
    assert policy.actor.shape == (1, MAX_TEAM_SIZE)
    assert policy.skill.shape == (1, MAX_SKILLS_PER_CHARACTER)
    assert policy.target.shape == (1, 10)
    assert policy.random_chakra.shape == (1, 5)
    assert policy.get_chakra.shape == (1, GET_CHAKRA_CODE_COUNT)
    assert policy.stack_index.shape == (1, MAX_STACK_SIZE)
    assert policy.reorder_direction.shape == (1, REORDER_DIRECTION_COUNT)
    assert value.shape == (1,)


def test_recurrent_transformer_actor_critic_outputs_factored_policy_shapes() -> None:
    model = RecurrentTransformerActorCritic(observation_size())
    observation = torch.tensor(
        encode_observation(
            create_initial_state(
                [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
                [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
            ),
            0,
        ),
        dtype=torch.float32,
    ).unsqueeze(0)
    hidden = model.initial_hidden(1, observation.device)

    policy, value, next_hidden = model(observation, hidden)

    assert policy.kind.shape == (1, ACTION_KIND_COUNT)
    assert policy.actor.shape == (1, MAX_TEAM_SIZE)
    assert policy.skill.shape == (1, MAX_SKILLS_PER_CHARACTER)
    assert policy.target.shape == (1, 10)
    assert policy.random_chakra.shape == (1, 5)
    assert policy.get_chakra.shape == (1, GET_CHAKRA_CODE_COUNT)
    assert policy.stack_index.shape == (1, MAX_STACK_SIZE)
    assert policy.reorder_direction.shape == (1, REORDER_DIRECTION_COUNT)
    assert value.shape == (1,)
    assert next_hidden.shape == hidden.shape


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


def test_rl_agent_can_load_transformer_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "opponent_transformer.pt"
    model = TransformerActorCritic(observation_size())
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": observation_size(),
            "policy_type": "factored_transformer",
            "model_arch": "transformer",
            "observation_version": COMPACT_OBSERVATION_VERSION,
            "perfect_info": False,
        },
        model_path,
    )

    agent = RlAgent(model_path)

    assert agent.model_arch == "transformer"


def test_rl_agent_can_load_legacy_base_observation_checkpoint(tmp_path: Path) -> None:
    model_path = tmp_path / "legacy_opponent.pt"
    legacy_obs_dim = observation_size(observation_version=BASE_OBSERVATION_VERSION)
    model = ActorCritic(legacy_obs_dim, character_feature_size=BASE_CHARACTER_FEATURE_SIZE)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": legacy_obs_dim,
            "policy_type": "factored",
            "perfect_info": False,
        },
        model_path,
    )

    agent = RlAgent(model_path)

    assert agent.model_arch == "mlp"
    assert agent.observation_version == BASE_OBSERVATION_VERSION


def _random_chakra_code(chakra_type: ChakraType) -> int:
    return RANDOM_CHAKRA_OFFSET + tuple(ChakraType).index(chakra_type)


def _catalog_index(
    *,
    kind: ActionKind,
    actor_slot: int = 0,
    skill_slot: int = 0,
    target_code: int = 0,
    random_chakra_code: int = 0,
    get_chakra_code: int = 0,
    stack_index: int = 0,
    reorder_direction: int = 0,
) -> int:
    spec = next(
        spec
        for spec in ACTION_CATALOG
        if spec.kind == kind
        and spec.actor_slot == actor_slot
        and spec.skill_slot == skill_slot
        and spec.target_code == target_code
        and spec.random_chakra_code == random_chakra_code
        and spec.get_chakra_code == get_chakra_code
        and spec.stack_index == stack_index
        and spec.reorder_direction == reorder_direction
    )
    return ACTION_CATALOG.index(spec)
