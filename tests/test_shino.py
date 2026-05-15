from naruto_arena.data.characters import (
    ABURAME_SHINO,
    INUZUKA_KIBA,
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.actions import EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action, resolved_skill, resolve_pending_skill_stack


def make_state():
    return create_initial_state(
        [ABURAME_SHINO, INUZUKA_KIBA, SAKURA_HARUNO],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )


def set_chakra(state, player_id: int, **amounts: int) -> None:
    state.players[player_id].chakra = ChakraPool.empty()
    for name, amount in amounts.items():
        state.players[player_id].chakra.amounts[ChakraType(name)] = amount


def test_shino_definition_uses_requested_skills() -> None:
    assert ABURAME_SHINO.name == "Aburame Shino"
    assert [skill.name for skill in ABURAME_SHINO.skills if skill.replacement_for is None] == [
        "Chakra Leach",
        "Female Bug",
        "Bug Wall",
        "Bug Clone",
    ]


def test_chakra_leach_does_not_steal_when_target_has_no_chakra() -> None:
    state = make_state()
    shino = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, ninjutsu=1)
    set_chakra(state, 1)

    apply_action(state, UseSkillAction(0, shino.instance_id, "chakra_leach", (target.instance_id,)))
    resolve_pending_skill_stack(state, 0)

    assert target.hp == 80
    assert state.players[0].chakra.total() == 0
    assert state.players[1].chakra.total() == 0
    assert not shino.status.has_marker("chakra_leach_stolen_chakra")
    assert resolved_skill(state, shino.instance_id, "chakra_leach").chakra_cost.random == 0


def test_chakra_leach_steals_one_chakra_immediately_on_shinos_turn() -> None:
    state = make_state()
    shino = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, ninjutsu=1)
    set_chakra(state, 1, taijutsu=1)

    apply_action(state, UseSkillAction(0, shino.instance_id, "chakra_leach", (target.instance_id,)))
    resolve_pending_skill_stack(state, 0)

    assert state.players[0].chakra.total() == 1
    assert state.players[1].chakra.total() == 0
    assert shino.status.has_marker("chakra_leach_stolen_chakra")
    assert resolved_skill(state, shino.instance_id, "chakra_leach").chakra_cost.random == 1


def test_female_bug_stacks_chakra_leach_bonus_damage() -> None:
    state = make_state()
    shino = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, ninjutsu=1)

    marker_id = f"female_bug:{shino.instance_id}"
    target.status.active_markers[marker_id] = 4
    target.status.active_marker_stacks[marker_id] = 2
    apply_action(state, UseSkillAction(0, shino.instance_id, "chakra_leach", (target.instance_id,)))
    resolve_pending_skill_stack(state, 0)

    assert target.hp == 70


def test_female_bug_reduces_marked_enemy_non_affliction_damage() -> None:
    state = make_state()
    shino = state.players[0].characters[0]
    target_sasuke = state.players[1].characters[2]

    apply_action(
        state,
        UseSkillAction(0, shino.instance_id, "female_bug", (target_sasuke.instance_id,)),
    )
    apply_action(state, EndTurnAction(0))
    set_chakra(state, 1, taijutsu=1, genjutsu=1)

    apply_action(
        state,
        UseSkillAction(
            1,
            target_sasuke.instance_id,
            "lion_combo",
            (shino.instance_id,),
            {ChakraType.GENJUTSU: 1},
        ),
    )
    resolve_pending_skill_stack(state, 1)

    assert shino.hp == 75


def test_bug_wall_grants_all_allies_permanent_destructible_defense() -> None:
    state = make_state()
    shino = state.players[0].characters[0]
    ally_ids = tuple(character.instance_id for character in state.players[0].characters)
    set_chakra(state, 0, ninjutsu=1, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            shino.instance_id,
            "bug_wall",
            ally_ids,
            {ChakraType.BLOODLINE: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)

    assert [
        character.status.damage_reductions[-1].amount for character in state.players[0].characters
    ] == [20, 20, 20]


def test_bug_clone_makes_shino_invulnerable() -> None:
    state = make_state()
    shino = state.players[0].characters[0]
    set_chakra(state, 0, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            shino.instance_id,
            "bug_clone",
            (shino.instance_id,),
            {ChakraType.BLOODLINE: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)

    assert shino.status.invulnerable_turns == 1
