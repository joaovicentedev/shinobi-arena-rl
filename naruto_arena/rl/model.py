from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the rl extra installed.
    raise RuntimeError("Install RL dependencies with `uv sync --extra rl`.") from exc

from naruto_arena.rl.action_space import (
    ACTION_KIND_COUNT,
    MAX_SKILLS_PER_CHARACTER,
    MAX_TEAM_SIZE,
    RANDOM_CHAKRA_CODE_COUNT,
    REORDER_DESTINATION_COUNT,
    TARGET_CODE_COUNT,
)
from naruto_arena.rl.observation import CHARACTER_FEATURE_SIZE, CHARACTER_SLOTS


@dataclass(frozen=True)
class PolicyOutput:
    kind: torch.Tensor
    actor: torch.Tensor
    skill: torch.Tensor
    target: torch.Tensor
    random_chakra: torch.Tensor
    reorder_destination: torch.Tensor


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int | None = None,
        hidden_dim: int = 256,
        character_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        del num_actions
        self.character_hidden_dim = character_hidden_dim
        self.global_dim = obs_dim - (CHARACTER_SLOTS * CHARACTER_FEATURE_SIZE)
        if self.global_dim <= 0:
            raise ValueError("Observation size is too small for character feature layout.")
        self.character_encoder = nn.Sequential(
            nn.Linear(CHARACTER_FEATURE_SIZE, character_hidden_dim),
            nn.ReLU(),
            nn.Linear(character_hidden_dim, character_hidden_dim),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(self.global_dim + CHARACTER_SLOTS * character_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.kind_policy = nn.Linear(hidden_dim, ACTION_KIND_COUNT)
        self.actor_policy = nn.Linear(hidden_dim, MAX_TEAM_SIZE)
        self.skill_policy = nn.Linear(hidden_dim, MAX_SKILLS_PER_CHARACTER)
        self.target_policy = nn.Linear(hidden_dim, TARGET_CODE_COUNT)
        self.random_chakra_policy = nn.Linear(hidden_dim, RANDOM_CHAKRA_CODE_COUNT)
        self.reorder_destination_policy = nn.Linear(hidden_dim, REORDER_DESTINATION_COUNT)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[PolicyOutput, torch.Tensor]:
        hidden = self.shared(self._encode_observation(observations))
        return (
            PolicyOutput(
                kind=self.kind_policy(hidden),
                actor=self.actor_policy(hidden),
                skill=self.skill_policy(hidden),
                target=self.target_policy(hidden),
                random_chakra=self.random_chakra_policy(hidden),
                reorder_destination=self.reorder_destination_policy(hidden),
            ),
            self.value(hidden).squeeze(-1),
        )

    def _encode_observation(self, observations: torch.Tensor) -> torch.Tensor:
        global_prefix = observations[:, :4]
        character_start = 4
        character_end = character_start + CHARACTER_SLOTS * CHARACTER_FEATURE_SIZE
        character_features = observations[:, character_start:character_end].reshape(
            observations.shape[0],
            CHARACTER_SLOTS,
            CHARACTER_FEATURE_SIZE,
        )
        global_suffix = observations[:, character_end:]
        encoded_characters = self.character_encoder(character_features).reshape(
            observations.shape[0],
            CHARACTER_SLOTS * self.character_hidden_dim,
        )
        return torch.cat([global_prefix, encoded_characters, global_suffix], dim=-1)
