from naruto_arena.data.characters import SAKURA_HARUNO, SASUKE_UCHIHA, UZUMAKI_NARUTO
from naruto_arena.engine.chakra import ChakraType
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.rl.action_space import NUM_ACTIONS, action_id_to_engine_action, legal_action_mask
from naruto_arena.rl.observation import encode_observation


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
