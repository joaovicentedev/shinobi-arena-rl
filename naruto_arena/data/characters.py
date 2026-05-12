from __future__ import annotations

from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.effects import (
    ConditionalDamageIncrease,
    ChakraRemoval,
    ChakraGainSteal,
    DamageReduction,
    DamageOverTime,
    DirectDamage,
    Healing,
    Invulnerability,
    PassiveEffect,
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
    return not state.get_character(target_id).status.has_marker(
        f"dynamic_air_marking:{actor_id}"
    )


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
        ChakraGainSteal(1),
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
            classes=frozenset(
                {SkillClass.INSTANT, SkillClass.PASSIVE, SkillClass.UNREMOVABLE}
            ),
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
            description="Using basic healing techniques, Sakura heals herself or an ally for 25 health.",
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
            classes=frozenset({SkillClass.MENTAL, SkillClass.RANGED, SkillClass.INSTANT, SkillClass.UNIQUE}),
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
                "affliction damage and stealing 1 chakra from their next chakra gain. "
                "If this skill successfully steals a chakra from the opponent, this "
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
                "affliction damage and stealing 1 chakra from their next chakra gain. "
                "If this skill successfully steals a chakra from the opponent, this "
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

ALL_CHARACTERS = {
    character.id: character
    for character in (
        UZUMAKI_NARUTO,
        SAKURA_HARUNO,
        SASUKE_UCHIHA,
        INUZUKA_KIBA,
        ABURAME_SHINO,
        HYUUGA_HINATA,
    )
}
