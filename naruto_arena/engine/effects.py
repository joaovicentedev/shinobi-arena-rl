from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class Effect(Protocol):
    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        ...


@dataclass(frozen=True)
class DirectDamage:
    amount: int
    piercing: bool = False
    conditional_marker_prefix: str | None = None
    conditional_bonus: int = 0
    conditional_bonus_per_stack: bool = False
    ignore_defenses_marker_prefix: str | None = None

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        from naruto_arena.engine.rules import deal_damage

        for target_id in target_ids:
            target = state.get_character(target_id)
            amount = self.amount
            if (
                self.conditional_marker_prefix is not None
                and target.status.has_marker(f"{self.conditional_marker_prefix}:{source_id}")
            ):
                if self.conditional_bonus_per_stack:
                    stacks = target.status.marker_stacks(
                        f"{self.conditional_marker_prefix}:{source_id}"
                    )
                    amount += self.conditional_bonus * stacks
                else:
                    amount += self.conditional_bonus
            ignore_defenses = (
                self.ignore_defenses_marker_prefix is not None
                and target.status.has_marker(f"{self.ignore_defenses_marker_prefix}:{source_id}")
            )
            deal_damage(
                state,
                source_id,
                target_id,
                amount,
                piercing=self.piercing,
                ignore_defenses=ignore_defenses,
            )


@dataclass(frozen=True)
class Healing:
    amount: int
    target_self: bool = False

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        targets = (source_id,) if self.target_self else target_ids
        for target_id in targets:
            character = state.get_character(target_id)
            if character.is_alive:
                character.hp = min(character.max_hp, character.hp + self.amount)


@dataclass(frozen=True)
class Stun:
    duration: int
    classes: frozenset["SkillClass"] | None = None

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        for target_id in target_ids:
            target = state.get_character(target_id)
            if target.status.has_marker("ignore_non_damage_effects") or target.status.has_marker(
                "inner_sakura"
            ):
                continue
            if self.classes is None:
                target.status.stunned_turns = max(target.status.stunned_turns, self.duration)
                continue
            for skill_class in self.classes:
                target.status.class_stuns[skill_class.value] = max(
                    target.status.class_stuns.get(skill_class.value, 0),
                    self.duration,
                )


@dataclass(frozen=True)
class DamageReduction:
    amount: int
    duration: int
    unpierceable: bool = False
    percent: int = 0
    target_self: bool = True

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        targets = (source_id,) if self.target_self else target_ids
        for target_id in targets:
            if state.get_character(target_id).status.has_marker("cannot_reduce_or_invulnerable"):
                continue
            state.get_character(target_id).status.damage_reductions.append(
                ActiveDamageReduction(self.amount, self.duration, self.unpierceable, self.percent)
            )


@dataclass(frozen=True)
class Invulnerability:
    duration: int
    target_self: bool = True

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        targets = (source_id,) if self.target_self else target_ids
        for target_id in targets:
            if state.get_character(target_id).status.has_marker("cannot_reduce_or_invulnerable"):
                continue
            state.get_character(target_id).status.invulnerable_turns = max(
                state.get_character(target_id).status.invulnerable_turns,
                self.duration,
            )


@dataclass(frozen=True)
class DamageOverTime:
    amount: int
    duration: int
    piercing: bool = False

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        for target_id in target_ids:
            state.get_character(target_id).status.damage_over_time.append(
                ActiveDamageOverTime(self.amount, self.duration, source_id, self.piercing)
            )


@dataclass(frozen=True)
class ChakraRemoval:
    amount: int
    allowed_types: tuple["ChakraType", ...] = ()

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        owner = state.owner_of(source_id)
        for target_id in target_ids:
            target_owner = state.owner_of(target_id)
            if target_owner == owner:
                continue
            if self.allowed_types:
                state.players[target_owner].chakra.remove_from_types(
                    self.allowed_types,
                    self.amount,
                )
            else:
                state.players[target_owner].chakra.remove_any(self.amount)
            break


@dataclass(frozen=True)
class ChakraSteal:
    amount: int
    allowed_types: tuple["ChakraType", ...]
    success_marker: str | None = None
    success_duration: int = 0

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        source_owner = state.owner_of(source_id)
        removed: dict["ChakraType", int] = {}
        for target_id in target_ids:
            target_owner = state.owner_of(target_id)
            if target_owner == source_owner:
                continue
            removed = state.players[target_owner].chakra.remove_from_types(
                self.allowed_types,
                self.amount,
            )
            if removed:
                break
        for chakra_type, amount in removed.items():
            state.players[source_owner].chakra.add(chakra_type, amount)
        if removed and self.success_marker is not None and self.success_duration > 0:
            state.get_character(source_id).status.active_markers[
                self.success_marker
            ] = self.success_duration


@dataclass(frozen=True)
class ChakraGainSteal:
    amount: int
    duration: int = 1

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        for target_id in target_ids:
            target = state.get_character(target_id)
            marker_id = f"chakra_gain_steal:{source_id}"
            target.status.active_markers[marker_id] = self.duration
            target.status.active_marker_stacks[marker_id] = max(
                target.status.active_marker_stacks.get(marker_id, 0),
                self.amount,
            )


@dataclass(frozen=True)
class CooldownEffect:
    skill_id: str
    amount: int

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        state.get_character(source_id).cooldowns[self.skill_id] = self.amount


@dataclass(frozen=True)
class PassiveEffect:
    id: str
    name: str
    unremovable: bool = True

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        character = state.get_character(source_id)
        character.passives.setdefault(self.id, False)


@dataclass(frozen=True)
class StatusMarker:
    marker_id: str
    duration: int
    target_self: bool = False
    target_all_allies: bool = False
    source_scoped: bool = False
    stackable: bool = False

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        if self.target_all_allies:
            owner = state.owner_of(source_id)
            targets = tuple(
                character.instance_id for character in state.players[owner].living_characters()
            )
        else:
            targets = (source_id,) if self.target_self else target_ids
        marker_id = f"{self.marker_id}:{source_id}" if self.source_scoped else self.marker_id
        for target_id in targets:
            target = state.get_character(target_id)
            target.status.active_markers[marker_id] = self.duration
            if self.stackable:
                target.status.active_marker_stacks[marker_id] = (
                    target.status.active_marker_stacks.get(marker_id, 0) + 1
                )


@dataclass(frozen=True)
class ConditionalSkillReplacement:
    source_skill_id: str
    replacement_skill_id: str
    required_passive_id: str

    def apply(self, state: "GameState", source_id: str, target_ids: tuple[str, ...]) -> None:
        return None


@dataclass(frozen=True)
class ConditionalDamageIncrease:
    amount: int
    required_passive_id: str | None = None
    required_status_id: str | None = None
    skill_ids: frozenset[str] = field(default_factory=frozenset)

    def applies_to(self, character: "CharacterState", skill_id: str) -> bool:
        if self.skill_ids and skill_id not in self.skill_ids:
            return False
        if self.required_passive_id and not character.passives.get(
            self.required_passive_id, False
        ):
            return False
        if (
            self.required_status_id
            and self.required_status_id not in character.status.active_markers
        ):
            return False
        return True


@dataclass
class ActiveDamageReduction:
    amount: int
    remaining_turns: int
    unpierceable: bool = False
    percent: int = 0


@dataclass
class ActiveDamageOverTime:
    amount: int
    remaining_turns: int
    source_id: str
    piercing: bool = False


@dataclass
class CharacterStatus:
    stunned_turns: int = 0
    class_stuns: dict[str, int] = field(default_factory=dict)
    invulnerable_turns: int = 0
    damage_reductions: list[ActiveDamageReduction] = field(default_factory=list)
    damage_over_time: list[ActiveDamageOverTime] = field(default_factory=list)
    active_markers: dict[str, int] = field(default_factory=dict)
    active_marker_stacks: dict[str, int] = field(default_factory=dict)

    def has_marker(self, marker_id: str) -> bool:
        return marker_id in self.active_markers

    def marker_stacks(self, marker_id: str) -> int:
        if not self.has_marker(marker_id):
            return 0
        return max(1, self.active_marker_stacks.get(marker_id, 1))
