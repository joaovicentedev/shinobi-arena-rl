from __future__ import annotations

from naruto_arena.engine.chakra import ChakraCost, ChakraType
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.effects import DamageOverTime, DamageReduction, DirectDamage
from naruto_arena.engine.skills import SkillClass, SkillDefinition, TargetRule


def cost(
    *, tai: int = 0, nin: int = 0, gen: int = 0, blood: int = 0, random: int = 0
) -> ChakraCost:
    chakra = {
        ChakraType.TAIJUTSU: tai,
        ChakraType.NINJUTSU: nin,
        ChakraType.GENJUTSU: gen,
        ChakraType.BLOODLINE: blood,
    }
    return ChakraCost(
        {chakra_type: amount for chakra_type, amount in chakra.items() if amount}, random=random
    )


ATTACK = frozenset({SkillClass.PHYSICAL, SkillClass.INSTANT})
CHAKRA_ATTACK = frozenset({SkillClass.CHAKRA, SkillClass.INSTANT})
MENTAL_ATTACK = frozenset({SkillClass.MENTAL, SkillClass.INSTANT})
AFFLICTION = frozenset({SkillClass.AFFLICTION, SkillClass.INSTANT})
DEFENSE = frozenset({SkillClass.INSTANT})


def attack(
    skill_id: str,
    name: str,
    amount: int,
    chakra_cost: ChakraCost,
    cooldown: int = 0,
    *,
    classes: frozenset[SkillClass] = ATTACK,
    target_rule: TargetRule = TargetRule.ONE_ENEMY,
) -> SkillDefinition:
    return SkillDefinition(
        id=skill_id,
        name=name,
        description=f"Idea 0 attack. Deals {amount} damage.",
        cooldown=cooldown,
        chakra_cost=chakra_cost,
        classes=classes,
        target_rule=target_rule,
        effects=(DirectDamage(amount),),
    )


def dot(
    skill_id: str,
    name: str,
    amount: int,
    duration: int,
    chakra_cost: ChakraCost,
    cooldown: int = 1,
    *,
    target_rule: TargetRule = TargetRule.ONE_ENEMY,
) -> SkillDefinition:
    return SkillDefinition(
        id=skill_id,
        name=name,
        description=f"Idea 0 DOT. Deals {amount} damage for {duration} turns.",
        cooldown=cooldown,
        chakra_cost=chakra_cost,
        classes=AFFLICTION,
        target_rule=target_rule,
        effects=(DamageOverTime(amount, duration),),
    )


def defense(
    skill_id: str,
    name: str,
    amount: int,
    duration: int,
    chakra_cost: ChakraCost,
    cooldown: int = 2,
    *,
    target_rule: TargetRule = TargetRule.SELF,
) -> SkillDefinition:
    return SkillDefinition(
        id=skill_id,
        name=name,
        description=f"Idea 0 defense. Reduces damage by {amount} for {duration} turns.",
        cooldown=cooldown,
        chakra_cost=chakra_cost,
        classes=DEFENSE,
        target_rule=target_rule,
        effects=(DamageReduction(amount, duration, target_self=target_rule == TargetRule.SELF),),
    )


def character(
    char_id: str,
    name: str,
    skills: tuple[SkillDefinition, SkillDefinition, SkillDefinition],
) -> CharacterDefinition:
    return CharacterDefinition(
        id=char_id,
        name=name,
        description=(
            "Idea 0 curriculum character: only direct damage, simple defense, DOT, "
            "cooldowns, costs, and target selection are active."
        ),
        skills=skills,
    )


UZUMAKI_NARUTO = character(
    "uzumaki_naruto",
    "Uzumaki Naruto",
    (
        attack("naruto_combo", "Naruto Combo", 25, cost(tai=1)),
        dot("rasengan_pressure", "Rasengan Pressure", 10, 2, cost(nin=1), cooldown=1),
        defense("shadow_clone_guard", "Shadow Clone Guard", 15, 2, cost(random=0), cooldown=2),
    ),
)

SAKURA_HARUNO = character(
    "sakura_haruno",
    "Sakura Haruno",
    (
        attack("ko_punch", "K.O. Punch", 30, cost(tai=1), cooldown=1),
        attack(
            "shuriken_flurry",
            "Shuriken Flurry",
            15,
            cost(random=0),
            target_rule=TargetRule.ALL_ENEMIES,
        ),
        defense(
            "sakura_guard", "Sakura Guard", 20, 1, cost(gen=1), target_rule=TargetRule.ONE_ALLY
        ),
    ),
)

SASUKE_UCHIHA = character(
    "sasuke_uchiha",
    "Sasuke Uchiha",
    (
        attack("lion_combo", "Lion Combo", 25, cost(tai=1)),
        attack("chidori", "Chidori", 45, cost(nin=1, blood=1), cooldown=2, classes=CHAKRA_ATTACK),
        defense("evasion", "Evasion", 15, 2, cost(random=0), cooldown=2),
    ),
)

INUZUKA_KIBA = character(
    "inuzuka_kiba",
    "Inuzuka Kiba",
    (
        attack("fang_over_fang", "Fang over Fang", 25, cost(tai=1)),
        dot("dynamic_marking", "Dynamic Marking", 8, 3, cost(random=0), cooldown=1),
        defense(
            "beast_guard", "Beast Guard", 10, 2, cost(random=0), target_rule=TargetRule.ONE_ALLY
        ),
    ),
)

ABURAME_SHINO = character(
    "aburame_shino",
    "Aburame Shino",
    (
        dot("bug_swarm", "Bug Swarm", 10, 3, cost(nin=1), cooldown=1),
        attack(
            "chakra_leach_strike", "Chakra Leach Strike", 20, cost(random=1), classes=CHAKRA_ATTACK
        ),
        defense(
            "insect_wall", "Insect Wall", 15, 2, cost(random=0), target_rule=TargetRule.ONE_ALLY
        ),
    ),
)

HYUUGA_HINATA = character(
    "hyuuga_hinata",
    "Hyuuga Hinata",
    (
        attack("gentle_fist", "Gentle Fist", 25, cost(tai=1), classes=CHAKRA_ATTACK),
        dot("chakra_point_pressure", "Chakra Point Pressure", 12, 2, cost(blood=1), cooldown=1),
        defense(
            "protective_palms",
            "Protective Palms",
            20,
            1,
            cost(random=0),
            target_rule=TargetRule.ONE_ALLY,
        ),
    ),
)

NARA_SHIKAMARU = character(
    "nara_shikamaru",
    "Nara Shikamaru",
    (
        attack("shadow_clutch", "Shadow Clutch", 20, cost(gen=1), classes=MENTAL_ATTACK),
        dot("shadow_pressure", "Shadow Pressure", 10, 2, cost(random=1), cooldown=1),
        defense(
            "tactical_cover",
            "Tactical Cover",
            15,
            2,
            cost(random=0),
            target_rule=TargetRule.ALL_ALLIES,
        ),
    ),
)

AKIMICHI_CHOUJI = character(
    "akimichi_chouji",
    "Akimichi Chouji",
    (
        attack("partial_expansion", "Partial Expansion", 30, cost(tai=1), cooldown=1),
        attack(
            "meat_tank",
            "Meat Tank",
            18,
            cost(random=1),
            target_rule=TargetRule.ALL_ENEMIES,
            cooldown=2,
        ),
        defense("expanded_guard", "Expanded Guard", 20, 1, cost(random=0)),
    ),
)

YAMANAKA_INO = character(
    "yamanaka_ino",
    "Yamanaka Ino",
    (
        attack("mind_strike", "Mind Strike", 20, cost(gen=1), classes=MENTAL_ATTACK),
        dot("psychic_pressure", "Psychic Pressure", 8, 3, cost(random=1), cooldown=1),
        defense(
            "team_focus", "Team Focus", 10, 2, cost(random=0), target_rule=TargetRule.ALL_ALLIES
        ),
    ),
)

TENTEN = character(
    "tenten",
    "Tenten",
    (
        attack("weapon_barrage", "Weapon Barrage", 25, cost(tai=1)),
        attack(
            "twin_rising_dragons",
            "Twin Rising Dragons",
            15,
            cost(nin=1),
            target_rule=TargetRule.ALL_ENEMIES,
            cooldown=2,
        ),
        defense(
            "scroll_cover", "Scroll Cover", 15, 2, cost(random=0), target_rule=TargetRule.ONE_ALLY
        ),
    ),
)

HYUUGA_NEJI = character(
    "hyuuga_neji",
    "Hyuuga Neji",
    (
        attack("eight_trigrams", "Eight Trigrams", 30, cost(blood=1), cooldown=1),
        attack(
            "rotation_strike",
            "Rotation Strike",
            18,
            cost(tai=1),
            target_rule=TargetRule.ALL_ENEMIES,
            cooldown=2,
        ),
        defense("rotation_defense", "Rotation Defense", 25, 1, cost(random=1)),
    ),
)

ROCK_LEE = character(
    "rock_lee",
    "Rock Lee",
    (
        attack("leaf_hurricane", "Leaf Hurricane", 25, cost(tai=1)),
        attack("front_lotus", "Front Lotus", 40, cost(tai=1, random=1), cooldown=2),
        defense("quick_step", "Quick Step", 15, 2, cost(random=0)),
    ),
)

GAARA_OF_THE_DESERT = character(
    "gaara_of_the_desert",
    "Gaara of the Desert",
    (
        attack("sand_coffin", "Sand Coffin", 30, cost(nin=1), cooldown=1),
        dot("sand_burial", "Sand Burial", 12, 2, cost(blood=1), cooldown=1),
        defense("sand_shield", "Sand Shield", 25, 2, cost(random=1)),
    ),
)

KANKURO = character(
    "kankuro",
    "Kankuro",
    (
        attack("puppet_blade", "Puppet Blade", 25, cost(tai=1)),
        dot("poison_bomb", "Poison Bomb", 10, 3, cost(nin=1), cooldown=1),
        defense(
            "puppet_block", "Puppet Block", 15, 2, cost(random=0), target_rule=TargetRule.ONE_ALLY
        ),
    ),
)

TEMARI = character(
    "temari",
    "Temari",
    (
        attack("wind_scythe", "Wind Scythe", 25, cost(nin=1)),
        attack(
            "summoning_wind",
            "Summoning Wind",
            15,
            cost(random=1),
            target_rule=TargetRule.ALL_ENEMIES,
            cooldown=2,
        ),
        defense(
            "wind_screen", "Wind Screen", 10, 2, cost(random=0), target_rule=TargetRule.ALL_ALLIES
        ),
    ),
)

TSUCHI_KIN = character(
    "tsuchi_kin",
    "Tsuchi Kin",
    (
        attack("needle_bell", "Needle Bell", 20, cost(gen=1), classes=MENTAL_ATTACK),
        dot("bell_resonance", "Bell Resonance", 8, 3, cost(random=1), cooldown=1),
        defense("sound_screen", "Sound Screen", 15, 2, cost(random=0)),
    ),
)

ABUMI_ZAKU = character(
    "abumi_zaku",
    "Abumi Zaku",
    (
        attack("air_cutter", "Air Cutter", 25, cost(nin=1)),
        attack("extreme_air_cutter", "Extreme Air Cutter", 35, cost(nin=1, random=1), cooldown=2),
        defense(
            "pressure_barrier",
            "Pressure Barrier",
            15,
            2,
            cost(random=0),
            target_rule=TargetRule.ONE_ALLY,
        ),
    ),
)

KINUTA_DOSU = character(
    "kinuta_dosu",
    "Kinuta Dosu",
    (
        attack("echo_drill", "Echo Drill", 25, cost(tai=1), classes=MENTAL_ATTACK),
        dot("sound_pulse", "Sound Pulse", 10, 2, cost(gen=1), cooldown=1),
        defense("melody_guard", "Melody Guard", 15, 2, cost(random=0)),
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

ALL_CHARACTERS = HAND_AUTHORED_CHARACTERS
