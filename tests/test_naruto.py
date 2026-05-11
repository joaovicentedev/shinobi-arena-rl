import pytest

from naruto_arena.data.characters import (
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.actions import EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.rules import create_initial_state, start_turn
from naruto_arena.engine.simulator import apply_action, can_use_skill, resolved_skill


def make_state():
    return create_initial_state(
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )


def naruto_state_parts():
    state = make_state()
    naruto = state.players[0].characters[0]
    enemy = state.players[1].characters[0]
    return state, naruto, enemy


def set_chakra(state, player_id: int, **amounts: int) -> None:
    state.players[player_id].chakra = ChakraPool.empty()
    for name, amount in amounts.items():
        state.players[player_id].chakra.amounts[ChakraType(name)] = amount


def test_rasengan_cannot_be_used_without_shadow_clones() -> None:
    state, naruto, _ = naruto_state_parts()
    set_chakra(state, 0, ninjutsu=1, taijutsu=1)

    assert not can_use_skill(state, naruto.instance_id, "rasengan")


def test_rasengan_can_be_used_with_shadow_clones() -> None:
    state, naruto, _ = naruto_state_parts()
    naruto.status.active_markers["shadow_clones"] = 5
    set_chakra(state, 0, ninjutsu=1, taijutsu=1)

    assert can_use_skill(state, naruto.instance_id, "rasengan")


def test_shadow_clones_improves_naruto_combo() -> None:
    state, naruto, enemy = naruto_state_parts()
    naruto.status.active_markers["shadow_clones"] = 5
    set_chakra(state, 0, taijutsu=1)

    apply_action(
        state,
        UseSkillAction(0, naruto.instance_id, "uzumaki_naruto_combo", (enemy.instance_id,)),
    )

    assert enemy.hp == 70


def test_kyuubi_awakening_activates_once_when_naruto_reaches_50_or_lower() -> None:
    state, naruto, enemy = naruto_state_parts()
    enemy_sasuke = state.players[1].characters[2]
    set_chakra(state, 1, taijutsu=2, ninjutsu=2)
    state.active_player = 1

    apply_action(
        state,
        UseSkillAction(
            1,
            enemy_sasuke.instance_id,
            "lion_combo",
            (naruto.instance_id,),
            {ChakraType.NINJUTSU: 1},
        ),
    )
    apply_action(
        state,
        UseSkillAction(
            1,
            enemy_sasuke.instance_id,
            "lion_combo",
            (naruto.instance_id,),
            {ChakraType.NINJUTSU: 1},
        ),
    )

    assert naruto.hp == 40
    assert naruto.passives["kyuubi_chakra_awakening"] is True
    naruto.hp = 40
    naruto.passives["kyuubi_chakra_awakening"] = False
    from naruto_arena.engine.rules import check_passive_triggers

    check_passive_triggers(state, naruto)
    assert naruto.passives["kyuubi_chakra_awakening"] is False


def test_kyuubi_awakening_heals_naruto_every_turn() -> None:
    state, naruto, _ = naruto_state_parts()
    naruto.hp = 45
    naruto.passives["kyuubi_chakra_awakening"] = True

    start_turn(state)

    assert naruto.hp == 50


def test_kyuubi_awakening_increases_naruto_damage_by_5() -> None:
    state, naruto, enemy = naruto_state_parts()
    naruto.passives["kyuubi_chakra_awakening"] = True
    set_chakra(state, 0, taijutsu=1)

    apply_action(
        state,
        UseSkillAction(0, naruto.instance_id, "uzumaki_naruto_combo", (enemy.instance_id,)),
    )

    assert enemy.hp == 75


def test_sexy_technique_is_replaced_after_kyuubi_awakening() -> None:
    state, naruto, _ = naruto_state_parts()
    naruto.passives["kyuubi_chakra_awakening"] = True

    skill = resolved_skill(state, naruto.instance_id, "sexy_technique")

    assert skill.name == "Kyuubi's Presence"
    assert "Mental" in {skill_class.value for skill_class in skill.classes}
