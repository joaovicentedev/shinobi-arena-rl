from naruto_arena.data.characters import (
    HYUUGA_HINATA,
    SAKURA_HARUNO,
    SASUKE_UCHIHA,
    UZUMAKI_NARUTO,
)
from naruto_arena.engine.actions import EndTurnAction, UseSkillAction
from naruto_arena.engine.chakra import ChakraPool, ChakraType
from naruto_arena.engine.effects import ActiveDamageReduction
from naruto_arena.engine.rules import create_initial_state
from naruto_arena.engine.simulator import apply_action


def make_state():
    return create_initial_state(
        [HYUUGA_HINATA, UZUMAKI_NARUTO, SAKURA_HARUNO],
        [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA],
    )


def set_chakra(state, player_id: int, **amounts: int) -> None:
    state.players[player_id].chakra = ChakraPool.empty()
    for name, amount in amounts.items():
        state.players[player_id].chakra.amounts[ChakraType(name)] = amount


def test_hinata_definition_uses_requested_skills() -> None:
    assert HYUUGA_HINATA.name == "Hyuuga Hinata"
    assert [skill.name for skill in HYUUGA_HINATA.skills] == [
        "Hinata Gentle Fist",
        "Eight Trigrams 64 Palms Protection",
        "Byakugan",
        "Hinata Block",
    ]


def test_hinata_gentle_fist_removes_chakra_during_byakugan() -> None:
    state = make_state()
    hinata = state.players[0].characters[0]
    target = state.players[1].characters[0]
    hinata.status.active_markers["byakugan"] = 4
    set_chakra(state, 0, taijutsu=1)
    set_chakra(state, 1, taijutsu=1, genjutsu=1)

    apply_action(
        state,
        UseSkillAction(0, hinata.instance_id, "hinata_gentle_fist", (target.instance_id,)),
    )

    assert target.hp == 80
    assert state.players[1].chakra.amounts[ChakraType.TAIJUTSU] == 0
    assert state.players[1].chakra.amounts[ChakraType.GENJUTSU] == 1


def test_hinata_gentle_fist_does_not_remove_chakra_without_byakugan() -> None:
    state = make_state()
    hinata = state.players[0].characters[0]
    target = state.players[1].characters[0]
    set_chakra(state, 0, taijutsu=1)
    set_chakra(state, 1, taijutsu=1)

    apply_action(
        state,
        UseSkillAction(0, hinata.instance_id, "hinata_gentle_fist", (target.instance_id,)),
    )

    assert target.hp == 80
    assert state.players[1].chakra.amounts[ChakraType.TAIJUTSU] == 1


def test_eight_trigrams_protection_damages_all_enemies_and_pierces_reduction() -> None:
    state = make_state()
    hinata = state.players[0].characters[0]
    enemy_ids = tuple(character.instance_id for character in state.players[1].characters)
    for enemy in state.players[1].characters:
        enemy.status.damage_reductions.append(ActiveDamageReduction(99, remaining_turns=2))
    set_chakra(state, 0, ninjutsu=1, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            hinata.instance_id,
            "eight_trigrams_64_palms_protection",
            enemy_ids,
            {ChakraType.BLOODLINE: 1},
        ),
    )
    apply_action(state, EndTurnAction(0))

    assert [enemy.hp for enemy in state.players[1].characters] == [90, 90, 90]


def test_eight_trigrams_protection_deals_more_damage_during_byakugan() -> None:
    state = make_state()
    hinata = state.players[0].characters[0]
    hinata.status.active_markers["byakugan"] = 4
    enemy_ids = tuple(character.instance_id for character in state.players[1].characters)
    set_chakra(state, 0, ninjutsu=1, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            hinata.instance_id,
            "eight_trigrams_64_palms_protection",
            enemy_ids,
            {ChakraType.BLOODLINE: 1},
        ),
    )
    apply_action(state, EndTurnAction(0))

    assert [enemy.hp for enemy in state.players[1].characters] == [85, 85, 85]


def test_eight_trigrams_protection_grants_invulnerability_after_ally_is_damaged() -> None:
    state = make_state()
    hinata = state.players[0].characters[0]
    ally = state.players[0].characters[1]
    enemy_sasuke = state.players[1].characters[2]
    enemy_ids = tuple(character.instance_id for character in state.players[1].characters)
    set_chakra(state, 0, ninjutsu=1, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            hinata.instance_id,
            "eight_trigrams_64_palms_protection",
            enemy_ids,
            {ChakraType.BLOODLINE: 1},
        ),
    )
    apply_action(state, EndTurnAction(0))
    set_chakra(state, 1, taijutsu=1, genjutsu=1)

    apply_action(
        state,
        UseSkillAction(
            1,
            enemy_sasuke.instance_id,
            "lion_combo",
            (ally.instance_id,),
            {ChakraType.GENJUTSU: 1},
        ),
    )

    assert ally.hp == 70
    assert ally.status.invulnerable_turns == 1


def test_byakugan_grants_hinata_damage_reduction() -> None:
    state = make_state()
    hinata = state.players[0].characters[0]
    set_chakra(state, 0, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            hinata.instance_id,
            "byakugan",
            (hinata.instance_id,),
            {ChakraType.BLOODLINE: 1},
        ),
    )

    assert hinata.status.has_marker("byakugan")
    assert any(reduction.percent == 50 for reduction in hinata.status.damage_reductions)


def test_hinata_block_makes_hinata_invulnerable() -> None:
    state = make_state()
    hinata = state.players[0].characters[0]
    set_chakra(state, 0, bloodline=1)

    apply_action(
        state,
        UseSkillAction(
            0,
            hinata.instance_id,
            "hinata_block",
            (hinata.instance_id,),
            {ChakraType.BLOODLINE: 1},
        ),
    )

    assert hinata.status.invulnerable_turns == 1
