from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.rl.env import NarutoArenaLearningEnv
from naruto_arena.rl.model import create_actor_critic
from naruto_arena.rl.observation import OBSERVATION_VERSION, observation_size
from naruto_arena.rl.teams import BENCHMARK_MATCHUPS, TRAINING_ROSTER, random_teams
from scripts.train_rl_pytorch import collect_episode


def test_ppo_smoke_checkpoint_loads_for_evaluation(tmp_path: Path) -> None:
    save_path = tmp_path / "ppo.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_rl_pytorch.py",
            "--algorithm",
            "ppo",
            "--episodes",
            "2",
            "--batch-episodes",
            "1",
            "--max-actions",
            "20",
            "--save-path",
            str(save_path),
            "--device",
            "cpu",
        ],
        check=True,
    )

    checkpoint = torch.load(save_path, map_location="cpu")
    agent = RlAgent(save_path)

    assert checkpoint["algorithm"] == "ppo"
    assert checkpoint["training"]["algorithm"] == "ppo"
    assert agent.model_arch == checkpoint["model_arch"]


def test_recurrent_transformer_ppo_smoke_checkpoint_loads_for_evaluation(
    tmp_path: Path,
) -> None:
    save_path = tmp_path / "recurrent_ppo.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_rl_pytorch.py",
            "--algorithm",
            "ppo",
            "--model-arch",
            "recurrent_transformer",
            "--episodes",
            "1",
            "--batch-episodes",
            "1",
            "--max-actions",
            "10",
            "--save-path",
            str(save_path),
            "--device",
            "cpu",
        ],
        check=True,
    )

    checkpoint = torch.load(save_path, map_location="cpu")
    agent = RlAgent(save_path)

    assert checkpoint["algorithm"] == "ppo"
    assert checkpoint["model_arch"] == "recurrent_transformer"
    assert agent.model_arch == "recurrent_transformer"


def test_actor_critic_training_smoke(tmp_path: Path) -> None:
    save_path = tmp_path / "actor_critic.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_rl_pytorch.py",
            "--algorithm",
            "actor_critic",
            "--episodes",
            "2",
            "--batch-episodes",
            "1",
            "--max-actions",
            "20",
            "--save-path",
            str(save_path),
            "--device",
            "cpu",
        ],
        check=True,
    )

    assert save_path.exists()


def test_collected_factored_actions_remain_valid_under_masks() -> None:
    env = NarutoArenaLearningEnv(opponent="heuristic", max_actions=20, seed=11)
    model = create_actor_critic(observation_size(), observation_version=OBSERVATION_VERSION)

    trajectory = collect_episode(env, model, 0.99, 0.95, seed=12)

    assert trajectory["actions"]
    assert all(
        not mask_trace or any(mask_trace["kind"])
        for mask_trace in trajectory["action_masks"]
    )
    assert len(trajectory["actions"]) == len(trajectory["log_probs"])
    assert len(trajectory["actions"]) == len(trajectory["advantages"])


def test_invalid_env_step_info_preserves_winner_contract() -> None:
    env = NarutoArenaLearningEnv(opponent="heuristic", max_actions=20, seed=11)
    env.reset(seed=12)

    _, _, _, info = env.step(action_id=-1)

    assert "winner" in info
    assert info["invalid_action"] is True


def test_random_training_roster_uses_only_hand_authored_characters() -> None:
    training_ids = {character.id for character in TRAINING_ROSTER}

    assert training_ids == {
        "abumi_zaku",
        "aburame_shino",
        "akimichi_chouji",
        "gaara_of_the_desert",
        "hyuuga_hinata",
        "hyuuga_neji",
        "inuzuka_kiba",
        "kankuro",
        "kinuta_dosu",
        "nara_shikamaru",
        "rock_lee",
        "sakura_haruno",
        "sasuke_uchiha",
        "temari",
        "tenten",
        "tsuchi_kin",
        "uzumaki_naruto",
        "yamanaka_ino",
    }


def test_random_teams_are_sampled_from_training_roster() -> None:
    import random

    training_ids = {character.id for character in TRAINING_ROSTER}
    team_a, team_b = random_teams(random.Random(123))

    assert {character.id for character in team_a} <= training_ids
    assert {character.id for character in team_b} <= training_ids


def test_benchmark_matchups_use_only_training_roster() -> None:
    training_ids = {character.id for character in TRAINING_ROSTER}
    benchmark_ids = {
        character_id
        for matchup in BENCHMARK_MATCHUPS
        for team in (matchup.team_a, matchup.team_b)
        for character_id in team
    }

    assert benchmark_ids <= training_ids
