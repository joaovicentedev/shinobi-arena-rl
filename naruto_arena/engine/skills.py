from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from naruto_arena.engine.chakra import ChakraCost
from naruto_arena.engine.effects import ConditionalDamageIncrease, Effect

if TYPE_CHECKING:
    from naruto_arena.engine.state import GameState


class SkillClass(StrEnum):
    PHYSICAL = "Physical"
    CHAKRA = "Chakra"
    MENTAL = "Mental"
    MELEE = "Melee"
    RANGED = "Ranged"
    INSTANT = "Instant"
    ACTION = "Action"
    CONTROL = "Control"
    STUN = "Stun"
    AFFLICTION = "Affliction"
    PASSIVE = "Passive"
    UNREMOVABLE = "Unremovable"
    UNIQUE = "Unique"


class TargetRule(StrEnum):
    SELF = "self"
    ONE_ENEMY = "one_enemy"
    ALL_ENEMIES = "all_enemies"
    ONE_ALLY = "one_ally"
    ALL_ALLIES = "all_allies"
    NONE = "none"


Requirement = Callable[["GameState", str], bool]
TargetRequirement = Callable[["GameState", str, str], bool]
EffectFactory = Callable[["GameState", str, "SkillDefinition"], list[Effect]]


@dataclass(frozen=True)
class SkillDefinition:
    id: str
    name: str
    description: str
    cooldown: int
    chakra_cost: ChakraCost
    classes: frozenset[SkillClass]
    target_rule: TargetRule
    effects: tuple[Effect, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    target_requirements: tuple[TargetRequirement, ...] = ()
    duration: int = 0
    status_marker: str | None = None
    effect_factory: EffectFactory | None = None
    conditional_damage: tuple[ConditionalDamageIncrease, ...] = ()
    replacement_for: str | None = None

    def is_passive(self) -> bool:
        return SkillClass.PASSIVE in self.classes

    def all_effects(self, state: "GameState", actor_id: str) -> list[Effect]:
        effects = list(self.effects)
        if self.effect_factory is not None:
            effects.extend(self.effect_factory(state, actor_id, self))
        return effects
