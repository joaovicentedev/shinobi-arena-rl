from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from naruto_arena.engine.state import GameState


class Effect(Protocol):
    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None: ...


@dataclass(frozen=True)
class DirectDamage:
    amount: int

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        from naruto_arena.engine.rules import deal_damage

        for target_id in target_ids:
            deal_damage(state, source_id, target_id, self.amount)


@dataclass(frozen=True)
class DamageReduction:
    amount: int
    duration: int
    target_self: bool = True

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        targets = (source_id,) if self.target_self else target_ids
        for target_id in targets:
            target = state.get_character(target_id)
            if target.is_alive:
                target.status.defenses.append(ActiveDefense(self.amount, self.duration))


@dataclass(frozen=True)
class DamageOverTime:
    amount: int
    duration: int

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        for target_id in target_ids:
            target = state.get_character(target_id)
            if target.is_alive:
                target.status.dots.append(ActiveDot(self.amount, self.duration, source_id))


@dataclass
class ActiveDefense:
    amount: int
    remaining_turns: int


@dataclass
class ActiveDot:
    amount: int
    remaining_turns: int
    source_id: str


@dataclass
class CharacterStatus:
    defenses: list[ActiveDefense] = field(default_factory=list)
    dots: list[ActiveDot] = field(default_factory=list)
