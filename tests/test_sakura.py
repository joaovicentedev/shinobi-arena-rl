from naruto_arena.data.characters import (
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.actions import EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action, can_use_skill


def make_state():
    return create_initial_state(
        [SAKURA_HARUNO, UZUMAKI_NARUTO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )


def set_chakra(state, player_id: int, **amounts: int) -> None:
    state.players[player_id].chakra = ChakraPool.empty()
    for name, amount in amounts.items():
        state.players[player_id].chakra.amounts[ChakraType(name)] = amount


def test_sakura_definition_uses_requested_skills() -> None:
    assert SAKURA_HARUNO.name == "Haruno Sakura"
    assert [skill.name for skill in SAKURA_HARUNO.skills] == [
        "KO Punch",
        "Cure",
        "Inner Sakura",
        "Sakura Replacement Technique",
    ]


def test_ko_punch_deals_20_and_stuns_physical_and_mental_skills() -> None:
    state = make_state()
    sakura = state.players[0].characters[0]
    target = state.players[1].characters[1]
    set_chakra(state, 0, taijutsu=1)

    apply_action(state, UseSkillAction(0, sakura.instance_id, "ko_punch", (target.instance_id,)))
    apply_action(state, EndTurnAction(0))
    set_chakra(state, 1, taijutsu=1, ninjutsu=1, genjutsu=1)

    assert target.hp == 80
    assert not can_use_skill(state, target.instance_id, "ko_punch")
    assert not can_use_skill(state, target.instance_id, "inner_sakura")
    assert can_use_skill(state, target.instance_id, "cure")


def test_cure_heals_sakura_or_an_ally_for_25() -> None:
    state = make_state()
    sakura = state.players[0].characters[0]
    ally = state.players[0].characters[1]
    ally.hp = 50
    set_chakra(state, 0, ninjutsu=1)

    apply_action(state, UseSkillAction(0, sakura.instance_id, "cure", (ally.instance_id,)))

    assert ally.hp == 75


def test_inner_sakura_boosts_ko_punch_and_adds_damage_reduction() -> None:
    state = make_state()
    sakura = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, taijutsu=1, genjutsu=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            sakura.instance_id,
            "inner_sakura",
            (sakura.instance_id,),
            {ChakraType.GENJUTSU: 1},
        ),
    )
    assert sakura.status.has_marker("inner_sakura")
    assert sakura.status.damage_reductions[0].amount == 10

    apply_action(state, EndTurnAction(0))
    apply_action(state, EndTurnAction(1))
    set_chakra(state, 0, taijutsu=1)
    apply_action(state, UseSkillAction(0, sakura.instance_id, "ko_punch", (target.instance_id,)))

    assert target.hp == 70


def test_inner_sakura_ignores_non_damage_effects_but_not_damage() -> None:
    state = make_state()
    sakura = state.players[0].characters[0]
    enemy_sakura = state.players[1].characters[1]
    sakura.status.active_markers["inner_sakura"] = 4
    set_chakra(state, 1, taijutsu=1)
    state.active_player = 1

    apply_action(
        state,
        UseSkillAction(1, enemy_sakura.instance_id, "ko_punch", (sakura.instance_id,)),
    )

    assert sakura.hp == 80
    assert sakura.status.class_stuns == {}


def test_sakura_replacement_technique_makes_sakura_invulnerable() -> None:
    state = make_state()
    sakura = state.players[0].characters[0]
    set_chakra(state, 0, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            sakura.instance_id,
            "sakura_replacement_technique",
            (sakura.instance_id,),
            {ChakraType.BLOODLINE: 1},
        ),
    )

    assert sakura.status.invulnerable_turns == 1
