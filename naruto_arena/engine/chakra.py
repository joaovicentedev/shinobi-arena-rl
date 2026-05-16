from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping


class ChakraType(StrEnum):
    NINJUTSU = "ninjutsu"
    TAIJUTSU = "taijutsu"
    GENJUTSU = "genjutsu"
    BLOODLINE = "bloodline"


@dataclass(frozen=True)
class ChakraCost:
    fixed: Mapping[ChakraType, int] = field(default_factory=dict)
    random: int = 0

    @staticmethod
    def none() -> "ChakraCost":
        return ChakraCost()

    def is_free(self) -> bool:
        return self.random == 0 and all(amount == 0 for amount in self.fixed.values())


@dataclass
class ChakraPool:
    amounts: dict[ChakraType, int] = field(default_factory=lambda: {t: 0 for t in ChakraType})

    @classmethod
    def empty(cls) -> "ChakraPool":
        return cls()

    @classmethod
    def from_counts(cls, counts: Mapping[ChakraType, int]) -> "ChakraPool":
        pool = cls.empty()
        for chakra_type, amount in counts.items():
            pool.amounts[chakra_type] = amount
        return pool

    def copy(self) -> "ChakraPool":
        return ChakraPool(dict(self.amounts))

    def total(self) -> int:
        return sum(self.amounts.values())

    def add(self, chakra_type: ChakraType, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("Cannot add negative chakra.")
        self.amounts[chakra_type] += amount

    def can_pay(
        self,
        cost: ChakraCost,
        random_payment: Mapping[ChakraType, int] | None = None,
    ) -> bool:
        try:
            self._validate_payment(cost, random_payment or {})
        except ValueError:
            return False
        return True

    def can_afford(self, cost: ChakraCost) -> bool:
        remaining = dict(self.amounts)
        for chakra_type, amount in cost.fixed.items():
            if remaining.get(chakra_type, 0) < amount:
                return False
            remaining[chakra_type] -= amount
        return sum(remaining.values()) >= cost.random

    def pay(
        self,
        cost: ChakraCost,
        random_payment: Mapping[ChakraType, int] | None = None,
    ) -> None:
        payment = random_payment or {}
        self._validate_payment(cost, payment)
        for chakra_type, amount in cost.fixed.items():
            self.amounts[chakra_type] -= amount
        for chakra_type, amount in payment.items():
            self.amounts[chakra_type] -= amount

    def remove_any(self, amount: int) -> int:
        removed = 0
        for chakra_type in ChakraType:
            if removed == amount:
                break
            take = min(self.amounts[chakra_type], amount - removed)
            self.amounts[chakra_type] -= take
            removed += take
        return removed

    def can_exchange_for(self, amount: int = 5) -> bool:
        return self.total() >= amount

    def exchange_for(self, chakra_type: ChakraType, amount: int = 5) -> None:
        if not self.can_exchange_for(amount):
            raise ValueError("Not enough chakra to exchange.")
        removed = self.remove_any(amount)
        if removed != amount:
            raise ValueError("Not enough chakra to exchange.")
        self.add(chakra_type, 1)

    def remove_from_types(
        self,
        allowed_types: tuple[ChakraType, ...],
        amount: int,
    ) -> dict[ChakraType, int]:
        removed: dict[ChakraType, int] = {}
        remaining = amount
        for chakra_type in allowed_types:
            if remaining == 0:
                break
            take = min(self.amounts[chakra_type], remaining)
            if take == 0:
                continue
            self.amounts[chakra_type] -= take
            removed[chakra_type] = take
            remaining -= take
        return removed

    def _validate_payment(self, cost: ChakraCost, random_payment: Mapping[ChakraType, int]) -> None:
        if cost.random < 0 or any(amount < 0 for amount in cost.fixed.values()):
            raise ValueError("Chakra cost cannot be negative.")
        if any(amount < 0 for amount in random_payment.values()):
            raise ValueError("Random chakra payment cannot be negative.")
        if sum(random_payment.values()) != cost.random:
            raise ValueError("Random chakra payment does not match random cost.")

        remaining = dict(self.amounts)
        for chakra_type, amount in cost.fixed.items():
            if remaining.get(chakra_type, 0) < amount:
                raise ValueError(f"Not enough {chakra_type.value} chakra.")
            remaining[chakra_type] -= amount

        for chakra_type, amount in random_payment.items():
            if chakra_type not in ChakraType:
                raise ValueError("Invalid chakra type.")
            if remaining.get(chakra_type, 0) < amount:
                raise ValueError("Not enough chakra for random payment.")
            remaining[chakra_type] -= amount
