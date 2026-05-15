from naruto_arena.data.characters import (
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.actions import EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.effects import ActiveDamageReduction
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action, can_use_skill, resolve_pending_skill_stack


def make_state():
    return create_initial_state(
        [SASUKE_UCHIHA, UZUMAKI_NARUTO, SAKURA_HARUNO],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )


def set_chakra(state, player_id: int, **amounts: int) -> None:
    state.players[player_id].chakra = ChakraPool.empty()
    for name, amount in amounts.items():
        state.players[player_id].chakra.amounts[ChakraType(name)] = amount


def test_sasuke_definition_uses_requested_skills() -> None:
    assert SASUKE_UCHIHA.name == "Uchiha Sasuke"
    assert [skill.name for skill in SASUKE_UCHIHA.skills] == [
        "Lion Combo",
        "Chidori",
        "Sharingan",
        "Swift Block",
        "Cursed Seal Awakening",
    ]


def test_chidori_requires_sharingan_active_on_sasuke() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    set_chakra(state, 0, ninjutsu=1, taijutsu=1)

    assert not can_use_skill(state, sasuke.instance_id, "chidori")

    sasuke.status.active_markers["sharingan"] = 4

    assert can_use_skill(state, sasuke.instance_id, "chidori")


def test_lion_combo_deals_bonus_damage_to_sharingan_target() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, taijutsu=1, ninjutsu=1)

    target.status.active_markers[f"sharingan:{sasuke.instance_id}"] = 4
    apply_action(
        state,
        UseSkillAction(
            0,
            sasuke.instance_id,
            "lion_combo",
            (target.instance_id,),
            {ChakraType.NINJUTSU: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)

    assert target.hp == 55


def test_chidori_pierces_normal_damage_reduction() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, ninjutsu=1, taijutsu=1)
    target.status.damage_reductions.append(ActiveDamageReduction(20, remaining_turns=1))

    sasuke.status.active_markers["sharingan"] = 4
    target.status.active_markers[f"sharingan:{sasuke.instance_id}"] = 4
    apply_action(
        state,
        UseSkillAction(
            0,
            sasuke.instance_id,
            "chidori",
            (target.instance_id,),
            {ChakraType.TAIJUTSU: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)

    assert target.hp == 45


def test_chidori_cannot_be_used_on_sasukes_next_turn() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    target = state.players[1].characters[0]
    sasuke.status.active_markers["sharingan"] = 4
    set_chakra(state, 0, ninjutsu=1, taijutsu=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            sasuke.instance_id,
            "chidori",
            (target.instance_id,),
            {ChakraType.TAIJUTSU: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)
    apply_action(state, EndTurnAction(0))
    apply_action(state, EndTurnAction(1))
    set_chakra(state, 0, ninjutsu=1, taijutsu=1)

    assert not can_use_skill(state, sasuke.instance_id, "chidori")


def test_chidori_does_not_pierce_unpierceable_damage_reduction() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, ninjutsu=1, taijutsu=1)
    target.status.damage_reductions.append(
        ActiveDamageReduction(20, remaining_turns=1, unpierceable=True)
    )

    sasuke.status.active_markers["sharingan"] = 4
    target.status.active_markers[f"sharingan:{sasuke.instance_id}"] = 4
    apply_action(
        state,
        UseSkillAction(
            0,
            sasuke.instance_id,
            "chidori",
            (target.instance_id,),
            {ChakraType.TAIJUTSU: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)

    assert target.hp == 65


def test_swift_block_makes_sasuke_invulnerable() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    set_chakra(state, 0, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            sasuke.instance_id,
            "swift_block",
            (sasuke.instance_id,),
            {ChakraType.BLOODLINE: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)

    assert sasuke.status.invulnerable_turns == 1


def test_cursed_seal_awakens_once_and_grants_permanent_unpierceable_reduction() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    enemy_sasuke = state.players[1].characters[2]
    sasuke.hp = 70
    set_chakra(state, 1, taijutsu=1, ninjutsu=1)
    state.active_player = 1

    apply_action(
        state,
        UseSkillAction(
            1,
            enemy_sasuke.instance_id,
            "lion_combo",
            (sasuke.instance_id,),
            {ChakraType.NINJUTSU: 1},
        ),
    )
    resolve_pending_skill_stack(state, 1)

    assert sasuke.passives["cursed_seal_awakening"] is True
    assert sasuke.passive_triggered["cursed_seal_awakening"] is True
    assert any(
        reduction.unpierceable and reduction.percent == 25
        for reduction in sasuke.status.damage_reductions
    )


def test_cursed_seal_sharingan_target_cannot_reduce_damage_or_use_invulnerability() -> None:
    state = make_state()
    sasuke = state.players[0].characters[0]
    target = state.players[1].characters[0]
    sasuke.passives["cursed_seal_awakening"] = True
    target.status.invulnerable_turns = 1
    target.status.damage_reductions.append(ActiveDamageReduction(99, remaining_turns=1))
    set_chakra(state, 0, taijutsu=1, ninjutsu=1)

    target.status.active_markers[f"sharingan:{sasuke.instance_id}"] = 4
    apply_action(
        state,
        UseSkillAction(
            0,
            sasuke.instance_id,
            "lion_combo",
            (target.instance_id,),
            {ChakraType.NINJUTSU: 1},
        ),
    )
    resolve_pending_skill_stack(state, 0)

    assert target.hp == 55
