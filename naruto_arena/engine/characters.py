from __future__ import annotations

from dataclasses import dataclass

from naruto_arena.engine.skills import SkillDefinition


@dataclass(frozen=True)
class CharacterDefinition:
    id: str
    name: str
    description: str
    skills: tuple[SkillDefinition, ...]

    def skill(self, skill_id: str) -> SkillDefinition:
        for skill in self.skills:
            if skill.id == skill_id:
                return skill
        raise KeyError(skill_id)

