from __future__ import annotations

from dataclasses import dataclass, field

from naruto_arena.engine.chakra import ChakraType


class Action:
    player_id: int


@dataclass(frozen=True)
class UseSkillAction(Action):
    player_id: int
    actor_id: str
    skill_id: str
    target_ids: tuple[str, ...]
    random_payment: dict[ChakraType, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EndTurnAction(Action):
    player_id: int


@dataclass(frozen=True)
class ReorderSkillsAction(Action):
    player_id: int
    character_id: str
    skill_id: str
    new_index: int

