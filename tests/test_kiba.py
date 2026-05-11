import pytest

from naruto_arena.data.characters import (
    INUZUKA_KIBA,
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.actions import EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.effects import ActiveDamageReduction
from naruto_arena.engine.rules import RulesError, create_initial_state
from naruto_arena.engine.simulator import apply_action, can_use_skill, legal_actions, resolved_skill


def make_state():
    return create_initial_state(
        [INUZUKA_KIBA, UZUMAKI_NARUTO, SAKURA_HARUNO],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )


def set_chakra(state, player_id: int, **amounts: int) -> None:
    state.players[player_id].chakra = ChakraPool.empty()
    for name, amount in amounts.items():
        state.players[player_id].chakra.amounts[ChakraType(name)] = amount


def test_kiba_definition_uses_requested_skills() -> None:
    assert INUZUKA_KIBA.name == "Inuzuka Kiba"
    assert [skill.name for skill in INUZUKA_KIBA.skills if skill.replacement_for is None] == [
        "Garouga",
        "Double Headed Wolf",
        "Dynamic Air Marking",
        "Smoke Bomb",
    ]


def test_double_headed_wolf_damages_all_enemies_over_time_and_reduces_damage() -> None:
    state = make_state()
    kiba = state.players[0].characters[0]
    enemy_ids = tuple(character.instance_id for character in state.players[1].characters)
    set_chakra(state, 0, taijutsu=1, bloodline=1)

    apply_action(
        state,
        UseSkillAction(0, kiba.instance_id, "double_headed_wolf", enemy_ids),
    )

    assert kiba.status.has_marker("double_headed_wolf")
    assert any(
        reduction.unpierceable and reduction.percent == 50
        for reduction in kiba.status.damage_reductions
    )

    apply_action(state, EndTurnAction(0))

    assert [enemy.hp for enemy in state.players[1].characters] == [85, 85, 85]


def test_double_headed_wolf_reduces_garouga_cost_by_one_random_chakra() -> None:
    state = make_state()
    kiba = state.players[0].characters[0]
    set_chakra(state, 0, taijutsu=1)

    assert not can_use_skill(state, kiba.instance_id, "garouga")

    kiba.status.active_markers["double_headed_wolf"] = 3

    assert can_use_skill(state, kiba.instance_id, "garouga")
    assert resolved_skill(state, kiba.instance_id, "garouga").chakra_cost.random == 0


def test_dynamic_air_marking_improves_kiba_damage_and_ignores_defenses() -> None:
    state = make_state()
    kiba = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, taijutsu=1, ninjutsu=1)

    apply_action(
        state,
        UseSkillAction(0, kiba.instance_id, "dynamic_air_marking", (target.instance_id,)),
    )
    target.status.invulnerable_turns = 1
    target.status.damage_reductions.append(ActiveDamageReduction(99, remaining_turns=1))

    apply_action(
        state,
        UseSkillAction(
            0,
            kiba.instance_id,
            "garouga",
            (target.instance_id,),
            {ChakraType.NINJUTSU: 1},
        ),
    )

    assert target.hp == 60


def test_dynamic_air_marking_prevents_new_reduction_and_invulnerability() -> None:
    state = make_state()
    kiba = state.players[0].characters[0]
    target = state.players[1].characters[0]

    apply_action(
        state,
        UseSkillAction(0, kiba.instance_id, "dynamic_air_marking", (target.instance_id,)),
    )

    from naruto_arena.engine.effects import DamageReduction, Invulnerability

    DamageReduction(10, duration=1, target_self=False).apply(
        state,
        kiba.instance_id,
        (target.instance_id,),
    )
    Invulnerability(1, target_self=False).apply(state, kiba.instance_id, (target.instance_id,))

    assert target.status.damage_reductions == []
    assert target.status.invulnerable_turns == 0


def test_dynamic_air_marking_cannot_target_enemy_already_marked_by_it() -> None:
    state = make_state()
    kiba = state.players[0].characters[0]
    target = state.players[1].characters[0]

    apply_action(
        state,
        UseSkillAction(0, kiba.instance_id, "dynamic_air_marking", (target.instance_id,)),
    )

    with pytest.raises(RulesError):
        apply_action(
            state,
            UseSkillAction(0, kiba.instance_id, "dynamic_air_marking", (target.instance_id,)),
        )

    actions = legal_actions(state, 0)
    air_marking_targets = [
        action.target_ids
        for action in actions
        if isinstance(action, UseSkillAction) and action.skill_id == "dynamic_air_marking"
    ]

    assert (target.instance_id,) not in air_marking_targets
