from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from naruto_arena.rl.observation import (
    BASE_CHARACTER_FEATURE_SIZE,
    BASE_OBSERVATION_VERSION,
    CHARACTER_FEATURE_SIZE,
    CHARACTER_ID_CODE_COUNT,
    CHARACTER_ID_FEATURE_INDEX,
    CHARACTER_SLOTS,
    COMPACT_CHARACTER_FEATURE_SIZE,
    COMPACT_OBSERVATION_VERSION,
    OBSERVATION_VERSION,
    SKILL_FEATURES_CHARACTER_FEATURE_SIZE,
    SKILL_FEATURES_OBSERVATION_VERSION,
)

MODEL_ARCH_MLP = "mlp"
MODEL_ARCH_TRANSFORMER = "transformer"
MODEL_ARCH_RECURRENT_TRANSFORMER = "recurrent_transformer"
MODEL_ARCHITECTURES = (MODEL_ARCH_MLP, MODEL_ARCH_TRANSFORMER, MODEL_ARCH_RECURRENT_TRANSFORMER)
POLICY_TYPE_FACTORED = "factored"
POLICY_TYPE_FACTORED_TRANSFORMER = "factored_transformer"
POLICY_TYPE_FACTORED_RECURRENT_TRANSFORMER = "factored_recurrent_transformer"


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
        character_feature_size: int = CHARACTER_FEATURE_SIZE,
    ) -> None:
        super().__init__()
        del num_actions
        self.character_hidden_dim = character_hidden_dim
        self.character_feature_size = character_feature_size
        self.global_dim = obs_dim - (CHARACTER_SLOTS * character_feature_size)
        if self.global_dim <= 0:
            raise ValueError("Observation size is too small for character feature layout.")
        self.character_encoder = nn.Sequential(
            nn.Linear(character_feature_size, character_hidden_dim),
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
        character_end = character_start + CHARACTER_SLOTS * self.character_feature_size
        character_features = observations[:, character_start:character_end].reshape(
            observations.shape[0],
            CHARACTER_SLOTS,
            self.character_feature_size,
        )
        global_suffix = observations[:, character_end:]
        encoded_characters = self.character_encoder(character_features).reshape(
            observations.shape[0],
            CHARACTER_SLOTS * self.character_hidden_dim,
        )
        return torch.cat([global_prefix, encoded_characters, global_suffix], dim=-1)


class TransformerActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int | None = None,
        hidden_dim: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        character_feature_size: int = CHARACTER_FEATURE_SIZE,
        character_id_code_count: int | None = CHARACTER_ID_CODE_COUNT,
        character_id_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        del num_actions
        self.character_feature_size = character_feature_size
        self.character_id_code_count = character_id_code_count
        self.character_id_feature_index = CHARACTER_ID_FEATURE_INDEX
        self.global_dim = obs_dim - (CHARACTER_SLOTS * character_feature_size)
        if self.global_dim <= 0:
            raise ValueError("Observation size is too small for character feature layout.")
        character_projection_input_size = character_feature_size
        if character_id_code_count is not None:
            character_projection_input_size = (
                character_feature_size - 1 + character_id_embedding_dim
            )
            self.character_id_embedding = nn.Embedding(
                character_id_code_count,
                character_id_embedding_dim,
            )
        else:
            self.character_id_embedding = None
        self.character_projection = nn.Linear(character_projection_input_size, d_model)
        self.context_projection = nn.Linear(self.global_dim, d_model)
        self.side_embedding = nn.Embedding(2, d_model)
        self.slot_embedding = nn.Embedding(3, d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.shared = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
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
        self.register_buffer(
            "character_sides",
            torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "character_slots",
            torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long),
            persistent=False,
        )

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
        character_end = character_start + CHARACTER_SLOTS * self.character_feature_size
        character_features = observations[:, character_start:character_end].reshape(
            observations.shape[0],
            CHARACTER_SLOTS,
            self.character_feature_size,
        )
        global_suffix = observations[:, character_end:]
        global_features = torch.cat([global_prefix, global_suffix], dim=-1)

        context_token = self.context_projection(global_features).unsqueeze(1)
        context_type = torch.zeros(
            observations.shape[0],
            1,
            dtype=torch.long,
            device=observations.device,
        )
        context_token = context_token + self.token_type_embedding(context_type)

        character_tokens = self.character_projection(
            self._encode_character_identity(character_features)
        )
        side_ids = self.character_sides.to(device=observations.device).unsqueeze(0)
        slot_ids = self.character_slots.to(device=observations.device).unsqueeze(0)
        character_type = torch.ones(
            observations.shape[0],
            CHARACTER_SLOTS,
            dtype=torch.long,
            device=observations.device,
        )
        character_tokens = (
            character_tokens
            + self.side_embedding(side_ids)
            + self.slot_embedding(slot_ids)
            + self.token_type_embedding(character_type)
        )

        tokens = torch.cat([context_token, character_tokens], dim=1)
        encoded_tokens = self.transformer(tokens)
        context_embedding = encoded_tokens[:, 0]
        character_embedding = encoded_tokens[:, 1:].mean(dim=1)
        return torch.cat([context_embedding, character_embedding], dim=-1)

    def _encode_character_identity(self, character_features: torch.Tensor) -> torch.Tensor:
        if self.character_id_embedding is None:
            return character_features

        character_ids = (
            character_features[:, :, self.character_id_feature_index]
            .round()
            .long()
            .clamp(0, self.character_id_code_count - 1)
        )
        numeric_features = torch.cat(
            [
                character_features[:, :, : self.character_id_feature_index],
                character_features[:, :, self.character_id_feature_index + 1 :],
            ],
            dim=-1,
        )
        return torch.cat([numeric_features, self.character_id_embedding(character_ids)], dim=-1)


class RecurrentTransformerActorCritic(TransformerActorCritic):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int | None = None,
        hidden_dim: int = 256,
        recurrent_hidden_dim: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(obs_dim, num_actions=num_actions, hidden_dim=hidden_dim, **kwargs)
        self.recurrent_hidden_dim = recurrent_hidden_dim
        self.memory_input = nn.Sequential(
            nn.Linear(hidden_dim, recurrent_hidden_dim),
            nn.ReLU(),
        )
        self.memory = nn.GRUCell(recurrent_hidden_dim, recurrent_hidden_dim)
        self.kind_policy = nn.Linear(recurrent_hidden_dim, ACTION_KIND_COUNT)
        self.actor_policy = nn.Linear(recurrent_hidden_dim, MAX_TEAM_SIZE)
        self.skill_policy = nn.Linear(recurrent_hidden_dim, MAX_SKILLS_PER_CHARACTER)
        self.target_policy = nn.Linear(recurrent_hidden_dim, TARGET_CODE_COUNT)
        self.random_chakra_policy = nn.Linear(recurrent_hidden_dim, RANDOM_CHAKRA_CODE_COUNT)
        self.reorder_destination_policy = nn.Linear(
            recurrent_hidden_dim,
            REORDER_DESTINATION_COUNT,
        )
        self.value = nn.Linear(recurrent_hidden_dim, 1)

    def forward(
        self,
        observations: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[PolicyOutput, torch.Tensor] | tuple[PolicyOutput, torch.Tensor, torch.Tensor]:
        encoded = self.shared(self._encode_observation(observations))
        memory_input = self.memory_input(encoded)
        if hidden is None:
            hidden = self.initial_hidden(observations.shape[0], observations.device)
            memory_hidden = self.memory(memory_input, hidden)
            return self._policy_and_value(memory_hidden)
        memory_hidden = self.memory(memory_input, hidden)
        policy, value = self._policy_and_value(memory_hidden)
        return policy, value, memory_hidden

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.recurrent_hidden_dim, device=device)

    def _policy_and_value(self, hidden: torch.Tensor) -> tuple[PolicyOutput, torch.Tensor]:
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


def create_actor_critic(
    obs_dim: int,
    model_arch: str = MODEL_ARCH_MLP,
    observation_version: str = OBSERVATION_VERSION,
) -> nn.Module:
    character_feature_size = character_feature_size_for_observation_version(
        observation_version,
    )
    if model_arch == MODEL_ARCH_MLP:
        return ActorCritic(obs_dim, character_feature_size=character_feature_size)
    if model_arch == MODEL_ARCH_TRANSFORMER:
        return TransformerActorCritic(
            obs_dim,
            character_feature_size=character_feature_size,
            character_id_code_count=character_id_code_count_for_observation_version(
                observation_version,
            ),
        )
    if model_arch == MODEL_ARCH_RECURRENT_TRANSFORMER:
        return RecurrentTransformerActorCritic(
            obs_dim,
            character_feature_size=character_feature_size,
            character_id_code_count=character_id_code_count_for_observation_version(
                observation_version,
            ),
        )
    raise ValueError(f"Unknown model architecture: {model_arch}")


def load_actor_critic_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError:
        model_state = model.state_dict()
        embedding_key = "character_id_embedding.weight"
        if embedding_key not in state_dict or embedding_key not in model_state:
            raise
        checkpoint_embedding = state_dict[embedding_key]
        model_embedding = model_state[embedding_key]
        if checkpoint_embedding.shape[1:] != model_embedding.shape[1:]:
            raise
        compatible_state_dict = dict(state_dict)
        resized_embedding = model_embedding.clone()
        rows = min(checkpoint_embedding.shape[0], resized_embedding.shape[0])
        resized_embedding[:rows] = checkpoint_embedding[:rows]
        compatible_state_dict[embedding_key] = resized_embedding
        model.load_state_dict(compatible_state_dict)


def character_feature_size_for_observation_version(observation_version: str) -> int:
    if observation_version == BASE_OBSERVATION_VERSION:
        return BASE_CHARACTER_FEATURE_SIZE
    if observation_version == SKILL_FEATURES_OBSERVATION_VERSION:
        return SKILL_FEATURES_CHARACTER_FEATURE_SIZE
    if observation_version == OBSERVATION_VERSION:
        return COMPACT_CHARACTER_FEATURE_SIZE
    raise ValueError(f"Unknown observation version: {observation_version}")


def character_id_code_count_for_observation_version(observation_version: str) -> int | None:
    if observation_version == COMPACT_OBSERVATION_VERSION:
        return CHARACTER_ID_CODE_COUNT
    if observation_version in {BASE_OBSERVATION_VERSION, SKILL_FEATURES_OBSERVATION_VERSION}:
        return None
    raise ValueError(f"Unknown observation version: {observation_version}")


def policy_type_for_model_arch(model_arch: str) -> str:
    if model_arch == MODEL_ARCH_MLP:
        return POLICY_TYPE_FACTORED
    if model_arch == MODEL_ARCH_TRANSFORMER:
        return POLICY_TYPE_FACTORED_TRANSFORMER
    if model_arch == MODEL_ARCH_RECURRENT_TRANSFORMER:
        return POLICY_TYPE_FACTORED_RECURRENT_TRANSFORMER
    raise ValueError(f"Unknown model architecture: {model_arch}")


def model_arch_from_checkpoint(checkpoint: dict[str, Any]) -> str:
    model_arch = checkpoint.get("model_arch")
    if model_arch is not None:
        if model_arch not in MODEL_ARCHITECTURES:
            raise ValueError(f"Checkpoint uses unknown model architecture: {model_arch}")
        return str(model_arch)

    policy_type = checkpoint.get("policy_type")
    if policy_type == POLICY_TYPE_FACTORED:
        return MODEL_ARCH_MLP
    if policy_type == POLICY_TYPE_FACTORED_TRANSFORMER:
        return MODEL_ARCH_TRANSFORMER
    if policy_type == POLICY_TYPE_FACTORED_RECURRENT_TRANSFORMER:
        return MODEL_ARCH_RECURRENT_TRANSFORMER
    raise ValueError("Checkpoint uses an unsupported policy type. Retrain the RL model.")


def is_recurrent_model(model: nn.Module) -> bool:
    return hasattr(model, "initial_hidden")
