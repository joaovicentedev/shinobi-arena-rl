from __future__ import annotations

from naruto_arena.data.json_characters import load_extra_characters
from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.effects import (
    ChakraRemoval,
    ChakraSteal,
    ConditionalDamageIncrease,
    DamageOverTime,
    DamageReduction,
    DirectDamage,
    Healing,
    Invulnerability,
    PassiveEffect,
    SelfDamage,
    StatusMarker,
    Stun,
)
from naruto_arena.engine.skills import SkillClass, SkillDefinition, TargetRule


def has_shadow_clones(state, actor_id: str) -> bool:
    return state.get_character(actor_id).status.has_marker("shadow_clones")


def has_kyuubi(state, actor_id: str) -> bool:
    return state.get_character(actor_id).passives.get("kyuubi_chakra_awakening", False)


def naruto_combo_effects(state, actor_id: str, skill: SkillDefinition):
    amount = 30 if has_shadow_clones(state, actor_id) else 20
    return [DirectDamage(amount)]


def shadow_clones_effects(state, actor_id: str, skill: SkillDefinition):
    return [DamageReduction(15, duration=5, unpierceable=has_kyuubi(state, actor_id))]


def has_inner_sakura(state, actor_id: str) -> bool:
    return state.get_character(actor_id).status.has_marker("inner_sakura")


def ko_punch_effects(state, actor_id: str, skill: SkillDefinition):
    amount = 30 if has_inner_sakura(state, actor_id) else 20
    return [
        DirectDamage(amount),
        Stun(1, classes=frozenset({SkillClass.PHYSICAL, SkillClass.MENTAL})),
    ]


def has_sharingan(state, actor_id: str) -> bool:
    return state.get_character(actor_id).status.has_marker("sharingan")


def has_cursed_seal(state, actor_id: str) -> bool:
    return state.get_character(actor_id).passives.get("cursed_seal_awakening", False)


def sasuke_damage(
    state,
    actor_id: str,
    amount: int,
    *,
    piercing: bool = False,
) -> DirectDamage:
    return DirectDamage(
        amount,
        piercing=piercing,
        conditional_marker_prefix="sharingan",
        conditional_bonus=15,
        ignore_defenses_marker_prefix="sharingan" if has_cursed_seal(state, actor_id) else None,
    )


def lion_combo_effects(state, actor_id: str, skill: SkillDefinition):
    return [sasuke_damage(state, actor_id, 30)]


def chidori_effects(state, actor_id: str, skill: SkillDefinition):
    return [sasuke_damage(state, actor_id, 40, piercing=True)]


def has_double_headed_wolf(state, actor_id: str) -> bool:
    return state.get_character(actor_id).status.has_marker("double_headed_wolf")


def target_not_marked_by_dynamic_air_marking(state, actor_id: str, target_id: str) -> bool:
    return not state.get_character(target_id).status.has_marker(f"dynamic_air_marking:{actor_id}")


def kiba_damage(amount: int) -> DirectDamage:
    return DirectDamage(
        amount,
        conditional_marker_prefix="dynamic_air_marking",
        conditional_bonus=10,
        ignore_defenses_marker_prefix="dynamic_air_marking",
    )


def garouga_effects(state, actor_id: str, skill: SkillDefinition):
    return [kiba_damage(30)]


def has_chakra_leach_stolen_chakra(state, actor_id: str) -> bool:
    return state.get_character(actor_id).status.has_marker("chakra_leach_stolen_chakra")


def shino_chakra_leach_effects(state, actor_id: str, skill: SkillDefinition):
    return [
        DirectDamage(
            20,
            conditional_marker_prefix="female_bug",
            conditional_bonus=5,
            conditional_bonus_per_stack=True,
        ),
        ChakraSteal(
            1,
            tuple(ChakraType),
            success_marker="chakra_leach_stolen_chakra",
            success_duration=1,
        ),
    ]


def has_byakugan(state, actor_id: str) -> bool:
    return state.get_character(actor_id).status.has_marker("byakugan")


def hinata_gentle_fist_effects(state, actor_id: str, skill: SkillDefinition):
    effects = [DirectDamage(20)]
    if has_byakugan(state, actor_id):
        effects.append(ChakraRemoval(1, (ChakraType.TAIJUTSU, ChakraType.NINJUTSU)))
    return effects


def eight_trigrams_64_palms_protection_effects(
    state,
    actor_id: str,
    skill: SkillDefinition,
):
    amount = 15 if has_byakugan(state, actor_id) else 10
    return [
        DamageOverTime(amount, duration=2, piercing=True),
        StatusMarker(
            "damage_triggers_invulnerability",
            duration=2,
            target_all_allies=True,
        ),
    ]


def target_not_marked_by_meditate(state, actor_id: str, target_id: str) -> bool:
    return not state.get_character(target_id).status.has_marker(f"meditate:{actor_id}")


def shikamaru_shadow_imitation_effects(state, actor_id: str, skill: SkillDefinition):
    del state, actor_id, skill
    return [
        Stun(
            1,
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.CHAKRA, SkillClass.AFFLICTION}),
        )
    ]


def chouji_pill_stacks(state, actor_id: str) -> int:
    return state.get_character(actor_id).status.marker_stacks("akimichi_pills")


def has_no_akimichi_pills(state, actor_id: str) -> bool:
    return chouji_pill_stacks(state, actor_id) == 0


def has_one_akimichi_pill(state, actor_id: str) -> bool:
    return chouji_pill_stacks(state, actor_id) == 1


def has_two_akimichi_pills(state, actor_id: str) -> bool:
    return chouji_pill_stacks(state, actor_id) == 2


def can_eat_akimichi_pill(state, actor_id: str) -> bool:
    return chouji_pill_stacks(state, actor_id) < 3


def partial_double_size_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    return [DirectDamage(20 + (20 * chouji_pill_stacks(state, actor_id)))]


def meat_tank_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    return [
        DamageOverTime(10 + (10 * chouji_pill_stacks(state, actor_id)), duration=2),
        Invulnerability(2),
    ]


def akimichi_pills_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    stacks = chouji_pill_stacks(state, actor_id)
    return [
        SelfDamage(15 + (5 * stacks), piercing=True, ignore_defenses=True),
        StatusMarker("akimichi_pills", duration=1_000_000, target_self=True, stackable=True),
    ]


def target_has_chakra_hair_trap(state, actor_id: str, target_id: str) -> bool:
    return state.get_character(target_id).status.has_marker(f"chakra_hair_strand_trap:{actor_id}")


def mind_body_disturbance_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    duration = (
        2
        if any(
            target_has_chakra_hair_trap(state, actor_id, enemy.instance_id)
            for enemy in state.players[1 - state.owner_of(actor_id)].living_characters()
        )
        else 1
    )
    return [
        Stun(1, classes=frozenset({SkillClass.PHYSICAL, SkillClass.CHAKRA})),
        StatusMarker("cannot_reduce_or_invulnerable", duration=duration),
    ]


def change_of_heart_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    duration = 3
    return [
        StatusMarker("cannot_reduce_or_invulnerable", duration=duration),
        StatusMarker("harmful_skills_stunned", duration=duration),
        StatusMarker("change_of_heart", duration=duration, source_scoped=True),
        StatusMarker("art_of_valentine_available", duration=2, target_self=True),
    ]


def art_of_valentine_effects(state, actor_id: str, skill: SkillDefinition):
    del state, actor_id, skill
    return [DirectDamage(25, conditional_marker_prefix="change_of_heart", conditional_bonus=5)]


def has_marker(marker: str):
    def requirement(state, actor_id: str) -> bool:
        return state.get_character(actor_id).status.has_marker(marker)

    return requirement


def self_marker(marker: str, duration: int) -> StatusMarker:
    return StatusMarker(marker, duration=duration, target_self=True)


def source_marker(marker: str, duration: int, *, stackable: bool = False) -> StatusMarker:
    return StatusMarker(marker, duration=duration, source_scoped=True, stackable=stackable)


def tenten_twin_rising_dragons_effects(state, actor_id: str, skill: SkillDefinition):
    duration = (
        3 if state.get_character(actor_id).status.has_marker("twin_rising_full_release") else 2
    )
    return [
        DamageOverTime(15, duration=duration),
        StatusMarker("twin_rising_dragons_followup", duration=2, target_self=True),
    ]


def tenten_trap_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    amount = (
        10 if state.get_character(actor_id).status.has_marker("twin_rising_full_release") else 0
    )
    return [
        DirectDamage(
            amount,
            conditional_marker_prefix="twin_rising_dragons",
            conditional_bonus=5,
            conditional_bonus_per_stack=True,
        ),
        Stun(1, classes=frozenset({SkillClass.PHYSICAL, SkillClass.CHAKRA, SkillClass.AFFLICTION})),
    ]


def gaara_desert_graveyard_effects(state, actor_id: str, skill: SkillDefinition):
    del state, actor_id, skill
    return [
        DirectDamage(
            50,
            piercing=True,
            conditional_marker_prefix="desert_coffin",
            conditional_bonus=25,
            conditional_bonus_per_stack=True,
        )
    ]


def kankuro_black_secret_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    amount = 35 if state.get_character(actor_id).status.has_marker("puppet_preparation") else 30
    return [DirectDamage(amount, piercing=True)]


def kankuro_poison_bomb_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    duration = 2 if state.get_character(actor_id).status.has_marker("puppet_preparation") else 1
    return [DamageOverTime(10, duration=duration, piercing=True)]


def rock_lee_fiery_spirit_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    character = state.get_character(actor_id)
    lost_thresholds = max(0, (character.max_hp - character.hp) // 25)
    return [Healing(10 + (10 * lost_thresholds), target_self=True)]


def has_fifth_gate_or_front_lotus(state, actor_id: str) -> bool:
    status = state.get_character(actor_id).status
    return status.has_marker("fifth_gate_opening") or status.has_marker("front_lotus")


def dosu_resonating_echo_drill_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    amount = 40 if state.get_character(actor_id).status.has_marker("melody_arm_tuning") else 20
    return [DirectDamage(amount), StatusMarker("reduced_physical_chakra_damage", duration=1)]


def dosu_sound_manipulation_effects(state, actor_id: str, skill: SkillDefinition):
    del skill
    amount = 20 if state.get_character(actor_id).status.has_marker("melody_arm_tuning") else 10
    return [DirectDamage(amount), Stun(1)]


UZUMAKI_NARUTO = CharacterDefinition(
    id="uzumaki_naruto",
    name="Uzumaki Naruto",
    description=(
        "A Genin from Team 7, Naruto is an orphan with the goal to one day become Hokage. "
        "Using his signature move, Shadow Clones, Naruto is able to perform powerful moves "
        "such as the Uzumaki Naruto Combo and the Rasengan."
    ),
    skills=(
        SkillDefinition(
            id="uzumaki_naruto_combo",
            name="Uzumaki Naruto Combo",
            description=(
                "Naruto's version of the Lion Combo. This skill deals 20 damage to one enemy. "
                'During "Shadow Clones", this skill will deal 10 additional damage.'
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=naruto_combo_effects,
        ),
        SkillDefinition(
            id="rasengan",
            name="Rasengan",
            description=(
                "Naruto hits one enemy with a ball of chakra, dealing 45 damage to them and "
                'stunning their skills for 1 turn. Requires "Shadow Clones".'
            ),
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(DirectDamage(45), Stun(1)),
            requirements=(has_shadow_clones,),
        ),
        SkillDefinition(
            id="shadow_clones",
            name="Shadow Clones",
            description=(
                "Naruto creates multiple shadow clones hiding his true self. Naruto gains 15 "
                'points of damage reduction for 5 turns. During this time, "Uzumaki Naruto '
                'Combo" is improved and will deal an additional 10 damage and "Rasengan" may '
                "be used."
            ),
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            duration=5,
            status_marker="shadow_clones",
            effect_factory=shadow_clones_effects,
        ),
        SkillDefinition(
            id="sexy_technique",
            name="Sexy Technique",
            description=(
                "This skill makes Uzumaki Naruto invulnerable for 1 turn. During "
                '"Kyuubi\'s Chakra Awakening", this skill will be replaced by "Kyuubi\'s '
                'Presence" and will be classed as Mental.'
            ),
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
        SkillDefinition(
            id="kyuubi_presence",
            name="Kyuubi's Presence",
            description="Kyuubi's chakra protects Naruto with an overwhelming mental presence.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
            requirements=(has_kyuubi,),
            replacement_for="sexy_technique",
        ),
        SkillDefinition(
            id="kyuubi_chakra_awakening",
            name="Kyuubi's Chakra Awakening",
            description=(
                "When Naruto reaches 50 health for the first time, Kyuubi's chakra will awaken, "
                "healing Naruto for 5 health every turn permanently. During this time, Naruto "
                'will deal 5 additional damage and "Shadow Clones" will grant him 15 '
                "unpierceable damage reduction instead. Passives cannot be removed."
            ),
            cooldown=0,
            chakra_cost=ChakraCost.none(),
            classes=frozenset({SkillClass.INSTANT, SkillClass.PASSIVE, SkillClass.UNREMOVABLE}),
            target_rule=TargetRule.NONE,
            effects=(PassiveEffect("kyuubi_chakra_awakening", "Kyuubi's Chakra Awakening"),),
            conditional_damage=(
                ConditionalDamageIncrease(
                    amount=5,
                    required_passive_id="kyuubi_chakra_awakening",
                ),
            ),
        ),
    ),
)


SAKURA_HARUNO = CharacterDefinition(
    id="sakura_haruno",
    name="Haruno Sakura",
    description=(
        "A Genin from Team 7, Sakura is very intelligent, but self-conscious about herself. "
        "Having just recently received training from Tsunade, Sakura is now able to deliver "
        "powerful punches and heal her own allies."
    ),
    skills=(
        SkillDefinition(
            id="ko_punch",
            name="KO Punch",
            description=(
                "Sakura punches one enemy with all her strength, dealing 20 damage to them and "
                "stunning their physical and mental skills for 1 turn. During 'Inner Sakura', "
                "this skill will deal 10 additional damage."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=ko_punch_effects,
        ),
        SkillDefinition(
            id="cure",
            name="Cure",
            description=(
                "Using basic healing techniques, Sakura heals herself or an ally for 25 health."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ALLY,
            effects=(Healing(25),),
        ),
        SkillDefinition(
            id="inner_sakura",
            name="Inner Sakura",
            description=(
                "Sakura's inner self surfaces and urges her on. For 4 turns, Sakura will gain "
                "10 points of damage reduction. During this time, Sakura will ignore non-damage "
                "effects and 'KO Punch' will deal 10 additional damage."
            ),
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.SELF,
            effects=(DamageReduction(10, duration=4),),
            duration=4,
            status_marker="inner_sakura",
        ),
        SkillDefinition(
            id="sakura_replacement_technique",
            name="Sakura Replacement Technique",
            description="This skill makes Haruno Sakura invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)
SASUKE_UCHIHA = CharacterDefinition(
    id="sasuke_uchiha",
    name="Uchiha Sasuke",
    description=(
        "A Genin from Team 7, Sasuke is the lone survivor of the Uchiha clan and has sworn "
        "vengeance against his brother. Using his sharingan, Sasuke is able to anticipate "
        "incoming attacks and is capable of advanced offensive moves."
    ),
    skills=(
        SkillDefinition(
            id="lion_combo",
            name="Lion Combo",
            description=(
                "Copying a taijutsu combo that Lee used on him, Sasuke deals 30 damage to one "
                "enemy. This skill will deal an additional 15 damage to an enemy affected by "
                "'Sharingan'."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=lion_combo_effects,
        ),
        SkillDefinition(
            id="chidori",
            name="Chidori",
            description=(
                "Using a lightning element attack jutsu, Sasuke deals 40 piercing damage to one "
                "enemy. This skill will deal an additional 15 damage to an enemy affected by "
                "'Sharingan'. Requires 'Sharingan' to be active on Sasuke."
            ),
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            requirements=(has_sharingan,),
            effect_factory=chidori_effects,
        ),
        SkillDefinition(
            id="sharingan",
            name="Sharingan",
            description=(
                "Sasuke targets one enemy. For 4 turns, Sasuke will gain 25% damage reduction "
                "and 'Chidori' may be used. During this time, that enemy will receive an "
                "additional 15 damage from 'Lion Combo' and 'Chidori'."
            ),
            cooldown=3,
            chakra_cost=ChakraCost.none(),
            classes=frozenset(
                {SkillClass.MENTAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(
                DamageReduction(0, duration=4, percent=25),
                StatusMarker("sharingan", duration=4, target_self=True),
                StatusMarker("sharingan", duration=4, source_scoped=True),
            ),
        ),
        SkillDefinition(
            id="swift_block",
            name="Swift Block",
            description="This skill makes Uchiha Sasuke invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
        SkillDefinition(
            id="cursed_seal_awakening",
            name="Cursed Seal Awakening",
            description=(
                "When Sasuke reaches 50 health for the first time, Cursed Seal will awaken, "
                "granting Sasuke 25% unpierceable damage reduction permanently. During this "
                "time, Sasuke's skills cannot be countered or reflected and 'Sharingan' will "
                "also make the enemy unable to reduce damage or become invulnerable."
            ),
            cooldown=0,
            chakra_cost=ChakraCost.none(),
            classes=frozenset({SkillClass.INSTANT, SkillClass.PASSIVE, SkillClass.UNREMOVABLE}),
            target_rule=TargetRule.NONE,
            effects=(PassiveEffect("cursed_seal_awakening", "Cursed Seal Awakening"),),
        ),
    ),
)

INUZUKA_KIBA = CharacterDefinition(
    id="inuzuka_kiba",
    name="Inuzuka Kiba",
    description=(
        "A Genin from Team 8, Kiba is a member of the Inuzuka clan, and is both "
        "short-tempered and impulsive. Using his dog, Akamaru, Kiba is capable of "
        "powerful taijutsu or fusing with Akamaru to become a deadly double headed dog."
    ),
    skills=(
        SkillDefinition(
            id="garouga",
            name="Garouga",
            description=(
                "Kiba deals 30 damage to one enemy. During 'Double Headed Wolf', this "
                "skill is improved and will cost 1 less random chakra."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=garouga_effects,
        ),
        SkillDefinition(
            id="double_headed_wolf",
            name="Double Headed Wolf",
            description=(
                "Kiba and Akamaru turn into a giant beast and attack all enemies, dealing "
                "15 damage to them for 3 turns. The following 3 turns, 'Garouga' is "
                "improved and will cost 1 less random chakra. During this time, Kiba "
                "gains 50% unpierceable damage reduction."
            ),
            cooldown=3,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1, ChakraType.BLOODLINE: 1}),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.ACTION, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ALL_ENEMIES,
            effects=(
                DamageOverTime(15, duration=3),
                DamageReduction(0, duration=3, unpierceable=True, percent=50),
            ),
            duration=3,
            status_marker="double_headed_wolf",
        ),
        SkillDefinition(
            id="dynamic_air_marking",
            name="Dynamic Air Marking",
            description=(
                "Akamaru sprays urine on one enemy who cannot reduce damage or become "
                "invulnerable for 3 turns. During this time, 'Double Headed Wolf' and "
                "'Garouga' will deal 10 additional damage to them. This skill may not "
                "be used on an enemy already affected by it."
            ),
            cooldown=0,
            chakra_cost=ChakraCost.none(),
            classes=frozenset(
                {
                    SkillClass.PHYSICAL,
                    SkillClass.RANGED,
                    SkillClass.INSTANT,
                    SkillClass.UNIQUE,
                    SkillClass.AFFLICTION,
                }
            ),
            target_rule=TargetRule.ONE_ENEMY,
            target_requirements=(target_not_marked_by_dynamic_air_marking,),
            effects=(
                StatusMarker("dynamic_air_marking", duration=3, source_scoped=True),
                StatusMarker("cannot_reduce_or_invulnerable", duration=3),
            ),
        ),
        SkillDefinition(
            id="smoke_bomb",
            name="Smoke Bomb",
            description="This skill makes Inuzuka Kiba invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
        SkillDefinition(
            id="garouga_during_double_headed_wolf",
            name="Garouga",
            description=(
                "Kiba deals 30 damage to one enemy. During 'Double Headed Wolf', this "
                "skill is improved and will cost 1 less random chakra."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            requirements=(has_double_headed_wolf,),
            effect_factory=garouga_effects,
            replacement_for="garouga",
        ),
    ),
)

ABURAME_SHINO = CharacterDefinition(
    id="aburame_shino",
    name="Aburame Shino",
    description=(
        "A Genin from Team 8, Shino is the successor of the Aburame clan, and a very "
        "reserved and tactical fighter. Using the bugs that live inside his body, Shino "
        "is able to leech the chakra of his enemies, or protect his entire team."
    ),
    skills=(
        SkillDefinition(
            id="chakra_leach",
            name="Chakra Leach",
            description=(
                "Shino directs his chakra draining bugs to one enemy, dealing 20 "
                "affliction damage and stealing 1 chakra. If this skill successfully "
                "steals a chakra from the opponent, this "
                "skill will cost an extra random chakra for 1 turn."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}),
            classes=frozenset(
                {
                    SkillClass.CHAKRA,
                    SkillClass.RANGED,
                    SkillClass.INSTANT,
                    SkillClass.UNIQUE,
                    SkillClass.AFFLICTION,
                }
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=shino_chakra_leach_effects,
        ),
        SkillDefinition(
            id="female_bug",
            name="Female Bug",
            description=(
                "Shino directs one of his female bugs to attach itself. For 4 turns, "
                "'Chakra Leach' will deal 5 additional damage to one enemy. During this "
                "time, if that enemy uses a new harmful skill, they will deal 5 less "
                "non-affliction damage for 1 turn. This skill stacks."
            ),
            cooldown=2,
            chakra_cost=ChakraCost.none(),
            classes=frozenset(
                {
                    SkillClass.PHYSICAL,
                    SkillClass.RANGED,
                    SkillClass.INSTANT,
                    SkillClass.UNIQUE,
                    SkillClass.AFFLICTION,
                }
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(StatusMarker("female_bug", duration=4, source_scoped=True, stackable=True),),
        ),
        SkillDefinition(
            id="bug_wall",
            name="Bug Wall",
            description=(
                "Shino calls millions of bugs to create a wall protecting himself and his "
                "allies and granting them 20 points of permanent destructible defense."
            ),
            cooldown=5,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.ALL_ALLIES,
            effects=(DamageReduction(20, duration=1_000_000, target_self=False),),
        ),
        SkillDefinition(
            id="bug_clone",
            name="Bug Clone",
            description="This skill makes Aburame Shino invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
        SkillDefinition(
            id="chakra_leach_after_steal",
            name="Chakra Leach",
            description=(
                "Shino directs his chakra draining bugs to one enemy, dealing 20 "
                "affliction damage and stealing 1 chakra. If this skill successfully "
                "steals a chakra from the opponent, this "
                "skill will cost an extra random chakra for 1 turn."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}, random=1),
            classes=frozenset(
                {
                    SkillClass.CHAKRA,
                    SkillClass.RANGED,
                    SkillClass.INSTANT,
                    SkillClass.UNIQUE,
                    SkillClass.AFFLICTION,
                }
            ),
            target_rule=TargetRule.ONE_ENEMY,
            requirements=(has_chakra_leach_stolen_chakra,),
            effect_factory=shino_chakra_leach_effects,
            replacement_for="chakra_leach",
        ),
    ),
)

HYUUGA_HINATA = CharacterDefinition(
    id="hyuuga_hinata",
    name="Hyuuga Hinata",
    description=(
        "A Genin from Team 8, Hinata is the next in line in the Hyuuga clan, but "
        "she is shy and very withdrawn. Using the trademark Byakugan of the Hyuuga, "
        "Hinata is able to delicately target an enemy's Chakra Points while defending "
        "the team."
    ),
    skills=(
        SkillDefinition(
            id="hinata_gentle_fist",
            name="Hinata Gentle Fist",
            description=(
                "Using the Hyuuga Clan's style of taijutsu, Hinata deals 20 damage to "
                "one enemy. During 'Byakugan', this skill will remove 1 taijutsu or "
                "ninjutsu chakra."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=hinata_gentle_fist_effects,
        ),
        SkillDefinition(
            id="eight_trigrams_64_palms_protection",
            name="Eight Trigrams 64 Palms Protection",
            description=(
                "Hinata deals 10 piercing damage to all enemies for 2 turns. For 1 turn, "
                "if Hinata or her allies is affected by a new damage skill, they will "
                "become invulnerable for 1 turn. During 'Byakugan', this skill will "
                "deal 5 additional damage."
            ),
            cooldown=2,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}, random=1),
            classes=frozenset(
                {
                    SkillClass.CHAKRA,
                    SkillClass.MELEE,
                    SkillClass.ACTION,
                    SkillClass.UNIQUE,
                    SkillClass.INSTANT,
                }
            ),
            target_rule=TargetRule.ALL_ENEMIES,
            effect_factory=eight_trigrams_64_palms_protection_effects,
        ),
        SkillDefinition(
            id="byakugan",
            name="Byakugan",
            description=(
                "Hinata activates her Byakugan and gains 50% damage reduction for 4 "
                "turns. During this time, 'Hinata Gentle Fist' and 'Eight Trigrams 64 "
                "Palms Protection' will be improved and 'Byakugan' will reveal any "
                "invisible skills used by the enemy team. This skill cannot be "
                "countered and it ends if Hinata dies."
            ),
            cooldown=3,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset(
                {SkillClass.MENTAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.SELF,
            effects=(DamageReduction(0, duration=4, percent=50),),
            duration=4,
            status_marker="byakugan",
        ),
        SkillDefinition(
            id="hinata_block",
            name="Hinata Block",
            description="This skill makes Hyuuga Hinata invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

NARA_SHIKAMARU = CharacterDefinition(
    id="nara_shikamaru",
    name="Nara Shikamaru",
    description=(
        "A Genin from Team 10, a member of the Nara clan, Shikamaru is considered "
        "to be the smartest Genin of all the Konoha 11. Using his bloodline, "
        "Shikamaru can manipulate the shadows in the battlefield to disable and "
        "attack his enemies."
    ),
    skills=(
        SkillDefinition(
            id="meditate",
            name="Meditate",
            description=(
                "Shikamaru begins thinking up a strategy against one enemy, marking "
                "them for 5 turns. During this time, the initial use of "
                "'Shadow-Neck Bind' and 'Shadow Imitation' will last 1 additional "
                "turn on them."
            ),
            cooldown=0,
            chakra_cost=ChakraCost.none(),
            classes=frozenset({SkillClass.MENTAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            target_requirements=(target_not_marked_by_meditate,),
            effects=(StatusMarker("meditate", duration=5, source_scoped=True),),
        ),
        SkillDefinition(
            id="shadow_neck_bind",
            name="Shadow-Neck Bind",
            description=(
                "Shikamaru chokes all enemies, making them unable to reduce damage "
                "or become invulnerable while dealing 15 damage to them for 1 turn."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.GENJUTSU: 1}),
            classes=frozenset(
                {
                    SkillClass.CHAKRA,
                    SkillClass.RANGED,
                    SkillClass.ACTION,
                    SkillClass.INSTANT,
                }
            ),
            target_rule=TargetRule.ALL_ENEMIES,
            effects=(
                DamageOverTime(15, duration=1),
                StatusMarker("cannot_reduce_or_invulnerable", duration=1),
            ),
        ),
        SkillDefinition(
            id="shadow_imitation",
            name="Shadow Imitation",
            description=(
                "Shikamaru captures all enemies in shadows, stunning their "
                "non-mental skills for 1 turn."
            ),
            cooldown=3,
            chakra_cost=ChakraCost({ChakraType.GENJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.RANGED, SkillClass.CONTROL}),
            target_rule=TargetRule.ALL_ENEMIES,
            effect_factory=shikamaru_shadow_imitation_effects,
        ),
        SkillDefinition(
            id="shikamaru_hide",
            name="Shikamaru Hide",
            description="This skill makes Nara Shikamaru invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

AKIMICHI_CHOUJI = CharacterDefinition(
    id="akimichi_chouji",
    name="Akimichi Chouji",
    description=(
        "A Genin from Team 10, Chouji is a member of the Akimichi clan, a large "
        "eater, and a close friend to his allies. While innately strong, Chouji "
        "is able to sacrifice his own life using special pills from his clan to "
        "become insanely powerful."
    ),
    skills=(
        SkillDefinition(
            id="partial_double_size",
            name="Partial Double Size",
            description=(
                "Chouji doubles the size of his arms and attacks one enemy, "
                "dealing 20 damage to them."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=partial_double_size_effects,
        ),
        SkillDefinition(
            id="meat_tank",
            name="Meat Tank",
            description=(
                "Chouji transforms into a meat tank, becoming invulnerable for 2 "
                "turns. If targetable, one enemy will be dealt 10 damage for 2 turns."
            ),
            cooldown=2,
            chakra_cost=ChakraCost({ChakraType.BLOODLINE: 1}),
            classes=frozenset(
                {
                    SkillClass.PHYSICAL,
                    SkillClass.ACTION,
                    SkillClass.INSTANT,
                    SkillClass.MELEE,
                }
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=meat_tank_effects,
        ),
        SkillDefinition(
            id="akimichi_pills",
            name="Akimichi Pills",
            description=(
                "Chouji eats a pill, taking 15 affliction damage. 'Partial Double "
                "Size' will deal 20 additional damage and 'Meat Tank' will deal "
                "10 additional damage permanently. Each use of this skill will "
                "deal 5 more affliction damage and will cost 2 additional random "
                "chakra. Chouji can only eat three pills."
            ),
            cooldown=0,
            chakra_cost=ChakraCost.none(),
            classes=frozenset(
                {SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.UNIQUE, SkillClass.AFFLICTION}
            ),
            target_rule=TargetRule.SELF,
            requirements=(can_eat_akimichi_pill, has_no_akimichi_pills),
            effect_factory=akimichi_pills_effects,
        ),
        SkillDefinition(
            id="effortless_block",
            name="Effortless Block",
            description="This skill makes Akimichi Chouji invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
        SkillDefinition(
            id="butterfly_mode",
            name="Passive: Butterfly Mode",
            description=(
                "When Chouji eats the three pills, he will activate the Butterfly "
                "Mode, gaining 75% unpierceable damage reduction permanently and "
                "gaining 1 random chakra every turn."
            ),
            cooldown=0,
            chakra_cost=ChakraCost.none(),
            classes=frozenset(
                {
                    SkillClass.CHAKRA,
                    SkillClass.INSTANT,
                    SkillClass.UNIQUE,
                    SkillClass.PASSIVE,
                    SkillClass.UNREMOVABLE,
                }
            ),
            target_rule=TargetRule.NONE,
            effects=(PassiveEffect("butterfly_mode", "Passive: Butterfly Mode"),),
        ),
        SkillDefinition(
            id="akimichi_pills_after_one",
            name="Akimichi Pills",
            description="Chouji eats his second Akimichi pill.",
            cooldown=0,
            chakra_cost=ChakraCost(random=2),
            classes=frozenset(
                {SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.UNIQUE, SkillClass.AFFLICTION}
            ),
            target_rule=TargetRule.SELF,
            requirements=(can_eat_akimichi_pill, has_one_akimichi_pill),
            effect_factory=akimichi_pills_effects,
            replacement_for="akimichi_pills",
        ),
        SkillDefinition(
            id="akimichi_pills_after_two",
            name="Akimichi Pills",
            description="Chouji eats his third Akimichi pill.",
            cooldown=0,
            chakra_cost=ChakraCost(random=4),
            classes=frozenset(
                {SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.UNIQUE, SkillClass.AFFLICTION}
            ),
            target_rule=TargetRule.SELF,
            requirements=(can_eat_akimichi_pill, has_two_akimichi_pills),
            effect_factory=akimichi_pills_effects,
            replacement_for="akimichi_pills",
        ),
    ),
)

YAMANAKA_INO = CharacterDefinition(
    id="yamanaka_ino",
    name="Yamanaka Ino",
    description=(
        "A Genin from Team 10, Ino is a member of the Yamanaka clan, and a very "
        "confident and vain girl. Ino is able to use a variety of abilities to "
        "take over and control her enemies, making it difficult to tell friend "
        "from foe."
    ),
    skills=(
        SkillDefinition(
            id="mind_body_disturbance",
            name="Mind Body Disturbance",
            description=(
                "Using this skill Ino stuns one enemy's physical and chakra skills "
                "for 1 turn. During this time, that enemy will be unable to reduce "
                "damage or become invulnerable."
            ),
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.GENJUTSU: 1}),
            classes=frozenset({SkillClass.MENTAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=mind_body_disturbance_effects,
        ),
        SkillDefinition(
            id="change_of_heart",
            name="Change of Heart",
            description=(
                "Ino takes over the mind of an enemy. For 3 turns, that enemy "
                "cannot reduce damage or become invulnerable and their harmful "
                "skills are stunned."
            ),
            cooldown=3,
            chakra_cost=ChakraCost({ChakraType.GENJUTSU: 2}),
            classes=frozenset({SkillClass.MENTAL, SkillClass.RANGED, SkillClass.CONTROL}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=change_of_heart_effects,
        ),
        SkillDefinition(
            id="chakra_hair_strand_trap",
            name="Chakra Hair Strand Trap",
            description=(
                "Ino creates a trap for an enemy. For 1 turn, if that enemy uses "
                "a new harmful skill, then for 2 turns, Ino's control skills are "
                "improved against that enemy. This skill is invisible."
            ),
            cooldown=1,
            chakra_cost=ChakraCost.none(),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.RANGED}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(StatusMarker("chakra_hair_strand_trap", duration=1, source_scoped=True),),
        ),
        SkillDefinition(
            id="ino_block",
            name="Ino Block",
            description="This skill makes Yamanaka Ino invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
        SkillDefinition(
            id="art_of_the_valentine",
            name="Art of the Valentine",
            description=(
                "Ino deals 25 damage to one enemy. If used on an enemy affected "
                "by 'Change of Heart', this skill will deal 30 damage instead."
            ),
            cooldown=0,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=art_of_valentine_effects,
        ),
    ),
)

TENTEN = CharacterDefinition(
    id="tenten",
    name="Tenten",
    description=(
        "A member of Team Gai, Tenten is a tomboyish weapon specialist who believes a "
        "kunoichi can be as strong as a male ninja."
    ),
    skills=(
        SkillDefinition(
            id="twin_rising_dragons",
            name="Twin Rising Dragons",
            description="Tenten deals 15 damage to all enemies for 2 turns.",
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.ACTION}),
            target_rule=TargetRule.ALL_ENEMIES,
            effect_factory=tenten_twin_rising_dragons_effects,
        ),
        SkillDefinition(
            id="twin_rising_dragons_trap",
            name="Twin Rising Dragons Trap",
            description=(
                "Tenten attacks all enemies and stuns harmful non-mental skills for 1 turn."
            ),
            cooldown=2,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ALL_ENEMIES,
            effect_factory=tenten_trap_effects,
        ),
        SkillDefinition(
            id="twin_rising_dragons_full_release",
            name="Twin Rising Dragons Full Release",
            description="Tenten empowers Twin Rising Dragons and becomes protected for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost.none(),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1), self_marker("twin_rising_full_release", 4)),
        ),
        SkillDefinition(
            id="spiked_boulder_shield",
            name="Spiked Boulder Shield",
            description="This skill makes Tenten invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

HYUUGA_NEJI = CharacterDefinition(
    id="hyuuga_neji",
    name="Hyuuga Neji",
    description=(
        "A member of Team Gai, Neji is the most talented member of the Hyuuga clan in "
        "both mind and body."
    ),
    skills=(
        SkillDefinition(
            id="neji_gentle_fist",
            name="Neji Gentle Fist",
            description="Neji deals 25 damage to one enemy for 2 turns.",
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}, random=1),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.ACTION, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(DamageOverTime(25, duration=2), self_marker("neji_gentle_fist", 1)),
        ),
        SkillDefinition(
            id="eight_trigram_heavenly_spin",
            name="Eight Trigram Heavenly Spin",
            description=(
                "Neji becomes invulnerable for 1 turn while dealing 15 damage to all enemies."
            ),
            cooldown=2,
            chakra_cost=ChakraCost({ChakraType.BLOODLINE: 1}),
            classes=frozenset(
                {SkillClass.CHAKRA, SkillClass.MELEE, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ALL_ENEMIES,
            effects=(DirectDamage(15), Invulnerability(1), self_marker("heavenly_spin", 1)),
        ),
        SkillDefinition(
            id="eight_trigram_sixty_four_palms",
            name="Eight Trigram Sixty-Four Palms",
            description="Neji deals 35 piercing damage and removes taijutsu or ninjutsu chakra.",
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1, ChakraType.BLOODLINE: 1}),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.ACTION, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(
                DirectDamage(35, piercing=True),
                ChakraRemoval(1, (ChakraType.TAIJUTSU, ChakraType.NINJUTSU)),
            ),
        ),
        SkillDefinition(
            id="neji_byakugan",
            name="Neji Byakugan",
            description="This skill makes Hyuuga Neji invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.UNIQUE, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

ROCK_LEE = CharacterDefinition(
    id="rock_lee",
    name="Rock Lee",
    description="A member of Team Gai, Lee has focused his life entirely on taijutsu.",
    skills=(
        SkillDefinition(
            id="high_speed_taijutsu",
            name="High Speed Taijutsu",
            description="Lee deals 25 piercing damage to one enemy.",
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(DirectDamage(25, piercing=True),),
        ),
        SkillDefinition(
            id="front_lotus",
            name="Front Lotus",
            description="Lee deals 35 piercing damage and enables Final Lotus for 1 turn.",
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(DirectDamage(35, piercing=True), self_marker("front_lotus", 1)),
        ),
        SkillDefinition(
            id="fifth_gate_opening",
            name="Fifth Gate Opening",
            description="Lee opens five chakra gates and unlocks Fiery Spirit.",
            cooldown=0,
            chakra_cost=ChakraCost.none(),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.SELF,
            effects=(self_marker("fifth_gate_opening", 1_000_000),),
        ),
        SkillDefinition(
            id="evasion",
            name="Evasion",
            description="This skill makes Rock Lee invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
        SkillDefinition(
            id="fiery_spirit",
            name="Fiery Spirit",
            description="Lee heals himself based on missing health.",
            cooldown=2,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            requirements=(has_marker("fifth_gate_opening"),),
            effect_factory=rock_lee_fiery_spirit_effects,
            replacement_for="fifth_gate_opening",
        ),
        SkillDefinition(
            id="final_lotus",
            name="Final Lotus",
            description="Lee deals 50 piercing damage to one enemy and takes 5 affliction damage.",
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 2}),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT, SkillClass.AFFLICTION}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            requirements=(has_fifth_gate_or_front_lotus,),
            effects=(DirectDamage(50, piercing=True), SelfDamage(5, piercing=True)),
            replacement_for="front_lotus",
        ),
    ),
)

GAARA_OF_THE_DESERT = CharacterDefinition(
    id="gaara_of_the_desert",
    name="Gaara of the Desert",
    description="A Sand Village jinchuuriki, Gaara manipulates sand to crush and defend.",
    skills=(
        SkillDefinition(
            id="desert_graveyard",
            name="Desert Graveyard",
            description="Gaara deals 50 piercing damage, empowered by Desert Coffin stacks.",
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.BLOODLINE: 1, ChakraType.NINJUTSU: 1}),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=gaara_desert_graveyard_effects,
        ),
        SkillDefinition(
            id="desert_coffin",
            name="Desert Coffin",
            description="Gaara stuns non-mental skills and stacks Desert Graveyard damage.",
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(
                Stun(
                    1,
                    classes=frozenset(
                        {SkillClass.PHYSICAL, SkillClass.CHAKRA, SkillClass.AFFLICTION}
                    ),
                ),
                source_marker("desert_coffin", 1_000_000, stackable=True),
            ),
        ),
        SkillDefinition(
            id="third_eye",
            name="Third Eye",
            description="Gaara watches the enemy and prepares sand defenses.",
            cooldown=3,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset(
                {SkillClass.CHAKRA, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.SELF,
            effects=(DamageReduction(15, duration=1_000_000), self_marker("third_eye", 1)),
        ),
        SkillDefinition(
            id="sand_shield",
            name="Sand Shield",
            description="This skill makes Gaara of the Desert invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

KANKURO = CharacterDefinition(
    id="kankuro",
    name="Kankuro",
    description="The brother of Gaara and a master puppeteer.",
    skills=(
        SkillDefinition(
            id="black_secret_machine_one_shot",
            name="Black Secret Machine One Shot",
            description="Kankuro deals 30 piercing damage, improved by Puppet Preparation.",
            cooldown=0,
            chakra_cost=ChakraCost(random=2),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=kankuro_black_secret_effects,
        ),
        SkillDefinition(
            id="poison_bomb",
            name="Poison Bomb",
            description="Kankuro deals 10 affliction damage to all enemies.",
            cooldown=1,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.AFFLICTION}
            ),
            target_rule=TargetRule.ALL_ENEMIES,
            effect_factory=kankuro_poison_bomb_effects,
        ),
        SkillDefinition(
            id="puppet_preparation",
            name="Puppet Preparation",
            description="Kankuro gains destructible defense and improves his puppet skills.",
            cooldown=3,
            chakra_cost=ChakraCost.none(),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.SELF,
            effects=(DamageReduction(10, duration=4), self_marker("puppet_preparation", 3)),
        ),
        SkillDefinition(
            id="puppet_replacement_technique",
            name="Puppet Replacement Technique",
            description="This skill makes Kankuro invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

TEMARI = CharacterDefinition(
    id="temari",
    name="Temari",
    description="The elder sister of Gaara and Kankuro, Temari fights with wind and her fan.",
    skills=(
        SkillDefinition(
            id="cutting_whirlwind",
            name="Cutting Whirlwind",
            description="Temari deals piercing wind damage and becomes protected for 1 turn.",
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ALL_ENEMIES,
            effects=(
                DirectDamage(10, piercing=True),
                Invulnerability(1),
                self_marker("cutting_whirlwind", 1),
            ),
        ),
        SkillDefinition(
            id="summoning_quick_beheading_dance",
            name="Summoning Quick Beheading Dance",
            description="Temari deals 35 damage to all enemies.",
            cooldown=2,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}, random=2),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ALL_ENEMIES,
            effects=(DirectDamage(35),),
        ),
        SkillDefinition(
            id="dust_wind",
            name="Dust Wind",
            description="Temari makes her team invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost({ChakraType.NINJUTSU: 1}, random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ALL_ALLIES,
            effects=(Invulnerability(1, target_self=False),),
        ),
        SkillDefinition(
            id="fan_defence_technique",
            name="Fan Defence Technique",
            description="This skill makes Temari invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

TSUCHI_KIN = CharacterDefinition(
    id="tsuchi_kin",
    name="Tsuchi Kin",
    description="One of the three sound genin, Kin uses needles and bells against enemies.",
    skills=(
        SkillDefinition(
            id="illusion_bell_needles",
            name="Illusion Bell Needles",
            description="One enemy receives 15 damage.",
            cooldown=0,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(DirectDamage(15), self_marker("illusion_bell_needles", 1)),
        ),
        SkillDefinition(
            id="needle_and_bell_trap",
            name="Needle and Bell Trap",
            description="One enemy will be stunned for 1 turn.",
            cooldown=1,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(Stun(1), self_marker("needle_and_bell_trap", 1)),
        ),
        SkillDefinition(
            id="unnerving_bells",
            name="Unnerving Bells",
            description="One enemy loses 1 random chakra.",
            cooldown=2,
            chakra_cost=ChakraCost({ChakraType.GENJUTSU: 1}),
            classes=frozenset({SkillClass.MENTAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(ChakraRemoval(1), self_marker("unnerving_bells", 1)),
        ),
        SkillDefinition(
            id="sharp_analysis",
            name="Sharp Analysis",
            description="This skill makes Tsuchi Kin invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

ABUMI_ZAKU = CharacterDefinition(
    id="abumi_zaku",
    name="Abumi Zaku",
    description="One of the three sound genin, Zaku creates waves of compressed air.",
    skills=(
        SkillDefinition(
            id="air_cutter",
            name="Air Cutter",
            description="Zaku deals 25 damage and enables Extreme Air Cutter.",
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.BLOODLINE: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effects=(DirectDamage(25), self_marker("air_cutter", 1)),
        ),
        SkillDefinition(
            id="wall_of_air",
            name="Wall of Air",
            description="Zaku protects one ally with an air wall.",
            cooldown=2,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT, SkillClass.UNIQUE}),
            target_rule=TargetRule.ONE_ALLY,
            effects=(Invulnerability(1, target_self=False),),
        ),
        SkillDefinition(
            id="extreme_air_cutter",
            name="Extreme Air Cutter",
            description="Zaku deals 45 damage to all enemies.",
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.BLOODLINE: 1}, random=2),
            classes=frozenset(
                {SkillClass.PHYSICAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ALL_ENEMIES,
            requirements=(has_marker("air_cutter"),),
            effects=(DirectDamage(45),),
        ),
        SkillDefinition(
            id="airwave_deflection",
            name="Airwave Deflection",
            description="This skill makes Abumi Zaku invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.CHAKRA, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

KINUTA_DOSU = CharacterDefinition(
    id="kinuta_dosu",
    name="Kinuta Dosu",
    description="Kinuta Dosu uses his implanted Melody Arm to amplify sound waves.",
    skills=(
        SkillDefinition(
            id="resonating_echo_drill",
            name="Resonating Echo Drill",
            description="Dosu deals damage and weakens physical and chakra skills for 1 turn.",
            cooldown=0,
            chakra_cost=ChakraCost({ChakraType.TAIJUTSU: 1}),
            classes=frozenset({SkillClass.PHYSICAL, SkillClass.MELEE, SkillClass.INSTANT}),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=dosu_resonating_echo_drill_effects,
        ),
        SkillDefinition(
            id="sound_manipulation",
            name="Sound Manipulation",
            description="Dosu deals damage and stuns one enemy for 1 turn.",
            cooldown=1,
            chakra_cost=ChakraCost({ChakraType.GENJUTSU: 1}),
            classes=frozenset(
                {SkillClass.MENTAL, SkillClass.MELEE, SkillClass.INSTANT, SkillClass.UNIQUE}
            ),
            target_rule=TargetRule.ONE_ENEMY,
            effect_factory=dosu_sound_manipulation_effects,
        ),
        SkillDefinition(
            id="melody_arm_tuning",
            name="Melody Arm Tuning",
            description="Dosu improves Resonating Echo Drill and Sound Manipulation for 5 turns.",
            cooldown=5,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(self_marker("melody_arm_tuning", 5),),
        ),
        SkillDefinition(
            id="dosu_hide",
            name="Dosu Hide",
            description="This skill makes Kinuta Dosu invulnerable for 1 turn.",
            cooldown=4,
            chakra_cost=ChakraCost(random=1),
            classes=frozenset({SkillClass.MENTAL, SkillClass.INSTANT}),
            target_rule=TargetRule.SELF,
            effects=(Invulnerability(1),),
        ),
    ),
)

HAND_AUTHORED_CHARACTERS = {
    character.id: character
    for character in (
        UZUMAKI_NARUTO,
        SAKURA_HARUNO,
        SASUKE_UCHIHA,
        INUZUKA_KIBA,
        ABURAME_SHINO,
        HYUUGA_HINATA,
        NARA_SHIKAMARU,
        AKIMICHI_CHOUJI,
        YAMANAKA_INO,
        TENTEN,
        HYUUGA_NEJI,
        ROCK_LEE,
        GAARA_OF_THE_DESERT,
        KANKURO,
        TEMARI,
        TSUCHI_KIN,
        ABUMI_ZAKU,
        KINUTA_DOSU,
    )
}

EXTRA_CHARACTERS = load_extra_characters(set(HAND_AUTHORED_CHARACTERS))
ALL_CHARACTERS = {
    **HAND_AUTHORED_CHARACTERS,
    **EXTRA_CHARACTERS,
}
