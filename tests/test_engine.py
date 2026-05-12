import pytest

from naruto_arena.data.characters import (
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.actions import EndTurnAction, ReorderSkillsAction, UseSkillAction
from naruto_arena.engine.rules import RulesError, create_initial_state
from naruto_arena.engine.simulator import apply_action, can_use_skill, legal_actions


def test_team_cannot_have_duplicate_characters() -> None:
    with pytest.raises(RulesError):
        create_initial_state(
            [UZUMAKI_NARUTO, UZUMAKI_NARUTO, SAKURA_HARUNO],
            [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        )


def test_each_character_starts_with_100_hp() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    assert all(character.hp == 100 for character in state.all_characters())


def test_player_zero_starts_with_one_chakra() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    assert state.players[0].chakra.total() == 1


def test_chakra_gain_equals_number_of_living_characters_after_first_turn() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )

    apply_action(state, EndTurnAction(0))

    assert state.players[1].chakra.total() == 3
    state.players[1].characters[0].hp = 0
    apply_action(state, EndTurnAction(1))

    assert state.players[0].chakra.total() == 4


def test_random_chakra_gain_is_reproducible_with_same_seed() -> None:
    team = [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA]

    state_a = create_initial_state(team, team, rng_seed=42)
    state_b = create_initial_state(team, team, rng_seed=42)

    assert state_a.players[0].chakra.amounts == state_b.players[0].chakra.amounts


def test_reorder_skills_action_changes_skill_order() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    naruto = state.players[0].characters[0]

    apply_action(state, ReorderSkillsAction(0, naruto.instance_id, "sexy_technique", 0))

    assert naruto.skill_order[0] == "sexy_technique"


def test_reorder_skills_action_can_move_passive_skills() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    naruto = state.players[0].characters[0]

    apply_action(
        state,
        ReorderSkillsAction(0, naruto.instance_id, "kyuubi_chakra_awakening", 0),
    )

    assert naruto.skill_order[0] == "kyuubi_chakra_awakening"


def test_reorder_skills_action_is_limited_to_three_per_turn() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    naruto = state.players[0].characters[0]

    apply_action(state, ReorderSkillsAction(0, naruto.instance_id, "sexy_technique", 0))
    apply_action(state, ReorderSkillsAction(0, naruto.instance_id, "shadow_clones", 0))
    apply_action(state, ReorderSkillsAction(0, naruto.instance_id, "rasengan", 0))

    assert state.reorders_this_turn == 3
    assert not any(isinstance(action, ReorderSkillsAction) for action in legal_actions(state, 0))
    with pytest.raises(RulesError):
        apply_action(
            state,
            ReorderSkillsAction(0, naruto.instance_id, "kyuubi_chakra_awakening", 0),
        )


def test_reorder_skills_limit_resets_next_turn() -> None:
    state = create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    naruto = state.players[0].characters[0]

    apply_action(state, ReorderSkillsAction(0, naruto.instance_id, "sexy_technique", 0))
    apply_action(state, ReorderSkillsAction(0, naruto.instance_id, "shadow_clones", 0))
    apply_action(state, ReorderSkillsAction(0, naruto.instance_id, "rasengan", 0))
    apply_action(state, EndTurnAction(0))

    assert state.reorders_this_turn == 0


def test_character_can_use_only_one_new_skill_per_turn() -> None:
    state = create_initial_state(
        [SASUKE_UCHIHA, UZUMAKI_NARUTO, SAKURA_HARUNO],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )
    sasuke = state.players[0].characters[0]
    target = state.players[1].characters[0]

    apply_action(state, UseSkillAction(0, sasuke.instance_id, "sharingan", (target.instance_id,)))

    assert not can_use_skill(state, sasuke.instance_id, "lion_combo")
    assert all(
        not (
            isinstance(action, UseSkillAction)
            and action.actor_id == sasuke.instance_id
        )
        for action in legal_actions(state, 0)
    )
