import pytest

from naruto_arena.engine.chakra import ChakraCost, ChakraPool, ChakraType


def test_fixed_chakra_costs_are_validated() -> None:
    pool = ChakraPool.from_counts({ChakraType.TAIJUTSU: 1})
    cost = ChakraCost({ChakraType.TAIJUTSU: 1})

    pool.pay(cost)

    assert pool.amounts[ChakraType.TAIJUTSU] == 0
    with pytest.raises(ValueError):
        pool.pay(cost)


def test_random_chakra_costs_can_use_any_type() -> None:
    pool = ChakraPool.from_counts({ChakraType.GENJUTSU: 1})
    cost = ChakraCost(random=1)

    pool.pay(cost, {ChakraType.GENJUTSU: 1})

    assert pool.total() == 0


def test_fixed_and_random_costs_are_paid_after_fixed_costs() -> None:
    pool = ChakraPool.from_counts({ChakraType.NINJUTSU: 1, ChakraType.TAIJUTSU: 1})
    cost = ChakraCost({ChakraType.NINJUTSU: 1}, random=1)

    pool.pay(cost, {ChakraType.TAIJUTSU: 1})

    assert pool.total() == 0
