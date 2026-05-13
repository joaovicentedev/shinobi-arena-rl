from __future__ import annotations

import random
from dataclasses import dataclass

from naruto_arena.data.characters import ALL_CHARACTERS
from naruto_arena.engine.characters import CharacterDefinition

DEFAULT_TEAM_IDS = ("uzumaki_naruto", "sakura_haruno", "sasuke_uchiha")
TRAINING_ROSTER = tuple(sorted(ALL_CHARACTERS.values(), key=lambda character: character.id))


@dataclass(frozen=True)
class BenchmarkMatchup:
    name: str
    team_a: tuple[str, str, str]
    team_b: tuple[str, str, str]


BENCHMARK_MATCHUPS = (
    BenchmarkMatchup(
        "team_7_mirror",
        ("uzumaki_naruto", "sakura_haruno", "sasuke_uchiha"),
        ("uzumaki_naruto", "sakura_haruno", "sasuke_uchiha"),
    ),
    BenchmarkMatchup(
        "team_7_vs_team_8",
        ("uzumaki_naruto", "sakura_haruno", "sasuke_uchiha"),
        ("inuzuka_kiba", "aburame_shino", "hyuuga_hinata"),
    ),
    BenchmarkMatchup(
        "team_8_vs_team_10",
        ("inuzuka_kiba", "aburame_shino", "hyuuga_hinata"),
        ("nara_shikamaru", "akimichi_chouji", "yamanaka_ino"),
    ),
    BenchmarkMatchup(
        "expanded_roster_opening",
        ("abumi_zaku", "aburame_shino_s", "akadou_yoroi"),
        ("akatsuchi_s", "akimichi_chouji_s", "akimichi_chouza_s"),
    ),
    BenchmarkMatchup(
        "expanded_roster_pressure",
        ("animal_path_pein_s", "ao_s", "asura_path_pein_s"),
        ("baki", "chiyo_s", "chojuro_s"),
    ),
)


def default_team() -> list[CharacterDefinition]:
    return team_from_ids(DEFAULT_TEAM_IDS)


def team_from_ids(ids: tuple[str, str, str]) -> list[CharacterDefinition]:
    return [ALL_CHARACTERS[character_id] for character_id in ids]


def random_teams(rng: random.Random) -> tuple[list[CharacterDefinition], list[CharacterDefinition]]:
    team_a = rng.sample(TRAINING_ROSTER, 3)
    team_b = rng.sample(TRAINING_ROSTER, 3)
    rng.shuffle(team_a)
    rng.shuffle(team_b)
    return team_a, team_b


def random_mirror_teams(
    rng: random.Random,
) -> tuple[list[CharacterDefinition], list[CharacterDefinition]]:
    team = rng.sample(TRAINING_ROSTER, 3)
    team_a = list(team)
    team_b = list(team)
    rng.shuffle(team_a)
    rng.shuffle(team_b)
    return team_a, team_b
