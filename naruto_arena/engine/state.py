from __future__ import annotations

import random
from dataclasses import dataclass, field

from naruto_arena.engine.chakra import ChakraPool
from naruto_arena.engine.characters import CharacterDefinition
from naruto_arena.engine.effects import CharacterStatus


@dataclass
class CharacterState:
    definition: CharacterDefinition
    owner: int
    instance_id: str
    hp: int = 100
    max_hp: int = 100
    skill_order: list[str] = field(default_factory=list)
    cooldowns: dict[str, int] = field(default_factory=dict)
    status: CharacterStatus = field(default_factory=CharacterStatus)
    passives: dict[str, bool] = field(default_factory=dict)
    passive_triggered: dict[str, bool] = field(default_factory=dict)
    used_skill_this_turn: bool = False

    def __post_init__(self) -> None:
        if not self.skill_order:
            self.skill_order = [
                skill.id for skill in self.definition.skills if skill.replacement_for is None
            ]
        for skill in self.definition.skills:
            if skill.is_passive():
                self.passives.setdefault(skill.id, False)
            self.cooldowns.setdefault(skill.id, 0)

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    def available_skills(self) -> list[str]:
        return list(self.skill_order)


@dataclass
class PlayerState:
    player_id: int
    characters: list[CharacterState]
    chakra: ChakraPool = field(default_factory=ChakraPool.empty)

    def living_characters(self) -> list[CharacterState]:
        return [character for character in self.characters if character.is_alive]


@dataclass
class GameState:
    players: tuple[PlayerState, PlayerState]
    active_player: int = 0
    turn_number: int = 1
    winner: int | None = None
    reorders_this_turn: int = 0
    rng_seed: int = 0
    rng: random.Random = field(default_factory=random.Random, repr=False, compare=False)

    def all_characters(self) -> list[CharacterState]:
        return [character for player in self.players for character in player.characters]

    def get_character(self, instance_id: str) -> CharacterState:
        for character in self.all_characters():
            if character.instance_id == instance_id:
                return character
        raise KeyError(instance_id)

    def owner_of(self, instance_id: str) -> int:
        return self.get_character(instance_id).owner
