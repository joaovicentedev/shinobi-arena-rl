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
    GET_CHAKRA_CODE_COUNT,
    MAX_SKILLS_PER_CHARACTER,
    MAX_STACK_SIZE,
    MAX_TEAM_SIZE,
    RANDOM_CHAKRA_CODE_COUNT,
    REORDER_DIRECTION_COUNT,
    TARGET_CODE_COUNT,
)
from naruto_arena.rl.observation import (
    ATTENTION_CHAR_NUMERIC_SIZE,
    ATTENTION_CHAR_TOKEN_SIZE,
    ATTENTION_GLOBAL_FEATURE_SIZE,
    ATTENTION_MAX_STACK_SIZE,
    ATTENTION_OBSERVATION_VERSION,
    ATTENTION_SKILL_NUMERIC_SIZE,
    ATTENTION_SKILL_TOKEN_SIZE,
    ATTENTION_STACK_NUMERIC_SIZE,
    ATTENTION_STACK_TOKEN_SIZE,
    BASE_CHARACTER_FEATURE_SIZE,
    BASE_OBSERVATION_VERSION,
    CHARACTER_FEATURE_SIZE,
    CHARACTER_ID_CODE_COUNT,
    CHARACTER_ID_FEATURE_INDEX,
    CHARACTER_SLOTS,
    COMPACT_CHARACTER_FEATURE_SIZE,
    COMPACT_OBSERVATION_VERSION,
    GLOBAL_FEATURE_SIZE,
    OBSERVATION_VERSION,
    SKILL_FEATURES_CHARACTER_FEATURE_SIZE,
    SKILL_FEATURES_OBSERVATION_VERSION,
    SKILL_ID_CODE_COUNT,
)

MODEL_ARCH_MLP = "mlp"
MODEL_ARCH_ATTENTION = "attention"
MODEL_ARCH_TRANSFORMER = "transformer"
MODEL_ARCH_RECURRENT_TRANSFORMER = "recurrent_transformer"
MODEL_ARCHITECTURES = (
    MODEL_ARCH_MLP,
    MODEL_ARCH_ATTENTION,
    MODEL_ARCH_TRANSFORMER,
    MODEL_ARCH_RECURRENT_TRANSFORMER,
)
POLICY_TYPE_FACTORED = "factored"
POLICY_TYPE_ATTENTION = "attention_factored"
POLICY_TYPE_FACTORED_TRANSFORMER = "factored_transformer"
POLICY_TYPE_FACTORED_RECURRENT_TRANSFORMER = "factored_recurrent_transformer"


@dataclass(frozen=True)
class PolicyOutput:
    kind: torch.Tensor
    actor: torch.Tensor
    skill: torch.Tensor
    target: torch.Tensor
    random_chakra: torch.Tensor
    get_chakra: torch.Tensor
    stack_index: torch.Tensor
    reorder_direction: torch.Tensor
    use_skill_joint: torch.Tensor | None = None
    reorder_joint: torch.Tensor | None = None


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


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
        self.get_chakra_policy = nn.Linear(hidden_dim, GET_CHAKRA_CODE_COUNT)
        self.stack_index_policy = nn.Linear(hidden_dim, MAX_STACK_SIZE)
        self.reorder_direction_policy = nn.Linear(hidden_dim, REORDER_DIRECTION_COUNT)
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
                get_chakra=self.get_chakra_policy(hidden),
                stack_index=self.stack_index_policy(hidden),
                reorder_direction=self.reorder_direction_policy(hidden),
            ),
            self.value(hidden).squeeze(-1),
        )

    def _encode_observation(self, observations: torch.Tensor) -> torch.Tensor:
        global_prefix = observations[:, :GLOBAL_FEATURE_SIZE]
        character_start = GLOBAL_FEATURE_SIZE
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
        self.get_chakra_policy = nn.Linear(hidden_dim, GET_CHAKRA_CODE_COUNT)
        self.stack_index_policy = nn.Linear(hidden_dim, MAX_STACK_SIZE)
        self.reorder_direction_policy = nn.Linear(hidden_dim, REORDER_DIRECTION_COUNT)
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
                get_chakra=self.get_chakra_policy(hidden),
                stack_index=self.stack_index_policy(hidden),
                reorder_direction=self.reorder_direction_policy(hidden),
            ),
            self.value(hidden).squeeze(-1),
        )

    def _encode_observation(self, observations: torch.Tensor) -> torch.Tensor:
        global_prefix = observations[:, :GLOBAL_FEATURE_SIZE]
        character_start = GLOBAL_FEATURE_SIZE
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
        self.get_chakra_policy = nn.Linear(recurrent_hidden_dim, GET_CHAKRA_CODE_COUNT)
        self.stack_index_policy = nn.Linear(recurrent_hidden_dim, MAX_STACK_SIZE)
        self.reorder_direction_policy = nn.Linear(recurrent_hidden_dim, REORDER_DIRECTION_COUNT)
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
                get_chakra=self.get_chakra_policy(hidden),
                stack_index=self.stack_index_policy(hidden),
                reorder_direction=self.reorder_direction_policy(hidden),
            ),
            self.value(hidden).squeeze(-1),
        )


class AttentionActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_actions: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del num_actions
        if obs_dim <= 0:
            raise ValueError("Observation size must be positive.")
        self.global_mlp = _mlp(ATTENTION_GLOBAL_FEATURE_SIZE, hidden_dim, hidden_dim)
        self.char_mlp = _mlp(ATTENTION_CHAR_NUMERIC_SIZE, hidden_dim, hidden_dim)
        self.skill_mlp = _mlp(ATTENTION_SKILL_NUMERIC_SIZE, hidden_dim, hidden_dim)
        self.stack_mlp = _mlp(ATTENTION_STACK_NUMERIC_SIZE, hidden_dim, hidden_dim)
        self.token_type_embedding = nn.Embedding(4, hidden_dim)
        self.side_embedding = nn.Embedding(2, hidden_dim)
        self.char_slot_embedding = nn.Embedding(MAX_TEAM_SIZE, hidden_dim)
        self.char_id_embedding = nn.Embedding(CHARACTER_ID_CODE_COUNT, hidden_dim)
        self.skill_slot_embedding = nn.Embedding(MAX_SKILLS_PER_CHARACTER, hidden_dim)
        self.skill_id_embedding = nn.Embedding(SKILL_ID_CODE_COUNT, hidden_dim)
        self.stack_position_embedding = nn.Embedding(ATTENTION_MAX_STACK_SIZE, hidden_dim)
        self.target_code_embedding = nn.Embedding(TARGET_CODE_COUNT, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.kind_policy = nn.Linear(hidden_dim, ACTION_KIND_COUNT)
        self.get_chakra_policy = nn.Linear(hidden_dim, GET_CHAKRA_CODE_COUNT)
        self.use_skill_pair_mlp = _mlp(hidden_dim * 4, hidden_dim, RANDOM_CHAKRA_CODE_COUNT)
        self.reorder_mlp = _mlp(hidden_dim * 2, hidden_dim, REORDER_DIRECTION_COUNT)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[PolicyOutput, torch.Tensor]:
        encoded = self._encode_tokens(observations)
        global_out = encoded[:, 0]
        my_skill_out = encoded[:, 7:34].reshape(
            observations.shape[0],
            MAX_TEAM_SIZE,
            MAX_SKILLS_PER_CHARACTER,
            -1,
        )
        target_out = self._target_representations(encoded, observations.device)
        pair = torch.cat(
            [
                my_skill_out[:, :, :, None, :].expand(-1, -1, -1, TARGET_CODE_COUNT, -1),
                target_out[:, None, None, :, :].expand(
                    -1,
                    MAX_TEAM_SIZE,
                    MAX_SKILLS_PER_CHARACTER,
                    -1,
                    -1,
                ),
                global_out[:, None, None, None, :].expand(
                    -1,
                    MAX_TEAM_SIZE,
                    MAX_SKILLS_PER_CHARACTER,
                    TARGET_CODE_COUNT,
                    -1,
                ),
                my_skill_out[:, :, :, None, :] * target_out[:, None, None, :, :],
            ],
            dim=-1,
        )
        use_skill_joint = self.use_skill_pair_mlp(pair)
        my_stack_out = encoded[:, 61 : 61 + ATTENTION_MAX_STACK_SIZE]
        reorder_joint = self.reorder_mlp(
            torch.cat(
                [
                    my_stack_out,
                    global_out[:, None, :].expand(-1, ATTENTION_MAX_STACK_SIZE, -1),
                ],
                dim=-1,
            )
        )
        actor = use_skill_joint.amax(dim=(2, 3, 4))
        skill = use_skill_joint.amax(dim=(1, 3, 4))
        target = use_skill_joint.amax(dim=(1, 2, 4))
        random_chakra = use_skill_joint.amax(dim=(1, 2, 3))
        stack_index = reorder_joint[:, :MAX_STACK_SIZE].amax(dim=2)
        reorder_direction = reorder_joint.amax(dim=1)
        return (
            PolicyOutput(
                kind=self.kind_policy(global_out),
                actor=actor,
                skill=skill,
                target=target,
                random_chakra=random_chakra,
                get_chakra=self.get_chakra_policy(global_out),
                stack_index=stack_index,
                reorder_direction=reorder_direction,
                use_skill_joint=use_skill_joint,
                reorder_joint=reorder_joint,
            ),
            self.value(global_out).squeeze(-1),
        )

    def _encode_tokens(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        offset = 0
        global_features = observations[:, offset : offset + ATTENTION_GLOBAL_FEATURE_SIZE]
        offset += ATTENTION_GLOBAL_FEATURE_SIZE
        tokens = [
            (self.global_mlp(global_features) + self.token_type_embedding.weight[0]).unsqueeze(1)
        ]

        char_flat = observations[:, offset : offset + CHARACTER_SLOTS * ATTENTION_CHAR_TOKEN_SIZE]
        offset += CHARACTER_SLOTS * ATTENTION_CHAR_TOKEN_SIZE
        char_data = char_flat.reshape(batch, CHARACTER_SLOTS, ATTENTION_CHAR_TOKEN_SIZE)
        char_tokens = self.char_mlp(char_data[:, :, :ATTENTION_CHAR_NUMERIC_SIZE])
        char_ids = char_data[:, :, ATTENTION_CHAR_NUMERIC_SIZE].round().long()
        char_ids = char_ids.clamp(0, CHARACTER_ID_CODE_COUNT - 1)
        char_slots = torch.tensor([0, 1, 2, 0, 1, 2], device=observations.device)
        char_sides = torch.tensor([0, 0, 0, 1, 1, 1], device=observations.device)
        char_tokens = (
            char_tokens
            + self.token_type_embedding.weight[1]
            + self.side_embedding(char_sides)
            + self.char_slot_embedding(char_slots)
            + self.char_id_embedding(char_ids)
        )
        tokens.append(char_tokens)

        skill_count = CHARACTER_SLOTS * MAX_SKILLS_PER_CHARACTER
        skill_flat = observations[:, offset : offset + skill_count * ATTENTION_SKILL_TOKEN_SIZE]
        offset += skill_count * ATTENTION_SKILL_TOKEN_SIZE
        skill_data = skill_flat.reshape(batch, skill_count, ATTENTION_SKILL_TOKEN_SIZE)
        skill_tokens = self.skill_mlp(skill_data[:, :, :ATTENTION_SKILL_NUMERIC_SIZE])
        skill_cat = skill_data[:, :, ATTENTION_SKILL_NUMERIC_SIZE:].round().long()
        skill_slots = skill_cat[:, :, 2].clamp(0, MAX_SKILLS_PER_CHARACTER - 1)
        skill_ids = skill_cat[:, :, 3].clamp(0, SKILL_ID_CODE_COUNT - 1)
        owner_slots = skill_cat[:, :, 0].clamp(0, MAX_TEAM_SIZE - 1)
        owner_char_ids = skill_cat[:, :, 1].clamp(0, CHARACTER_ID_CODE_COUNT - 1)
        skill_sides = torch.tensor([0] * 27 + [1] * 27, device=observations.device)
        skill_tokens = (
            skill_tokens
            + self.token_type_embedding.weight[2]
            + self.side_embedding(skill_sides)
            + self.char_slot_embedding(owner_slots)
            + self.char_id_embedding(owner_char_ids)
            + self.skill_slot_embedding(skill_slots)
            + self.skill_id_embedding(skill_ids)
        )
        tokens.append(skill_tokens)

        stack_flat = observations[:, offset : offset + 32 * ATTENTION_STACK_TOKEN_SIZE]
        stack_data = stack_flat.reshape(batch, 32, ATTENTION_STACK_TOKEN_SIZE)
        stack_tokens = self.stack_mlp(stack_data[:, :, :ATTENTION_STACK_NUMERIC_SIZE])
        stack_cat = stack_data[:, :, ATTENTION_STACK_NUMERIC_SIZE:].round().long()
        stack_sides = stack_cat[:, :, 0].clamp(0, 1)
        stack_slots = stack_cat[:, :, 1].clamp(0, MAX_TEAM_SIZE - 1)
        stack_char_ids = stack_cat[:, :, 2].clamp(0, CHARACTER_ID_CODE_COUNT - 1)
        stack_skill_slots = stack_cat[:, :, 3].clamp(0, MAX_SKILLS_PER_CHARACTER - 1)
        stack_skill_ids = stack_cat[:, :, 4].clamp(0, SKILL_ID_CODE_COUNT - 1)
        stack_positions = torch.arange(ATTENTION_MAX_STACK_SIZE, device=observations.device)
        stack_positions = stack_positions.repeat(2).unsqueeze(0)
        stack_tokens = (
            stack_tokens
            + self.token_type_embedding.weight[3]
            + self.side_embedding(stack_sides)
            + self.char_slot_embedding(stack_slots)
            + self.char_id_embedding(stack_char_ids)
            + self.skill_slot_embedding(stack_skill_slots)
            + self.skill_id_embedding(stack_skill_ids)
            + self.stack_position_embedding(stack_positions)
        )
        tokens.append(stack_tokens)
        return self.transformer(torch.cat(tokens, dim=1))

    def _target_representations(
        self,
        encoded: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        learned = self.target_code_embedding(torch.arange(TARGET_CODE_COUNT, device=device))
        learned = learned.unsqueeze(0).expand(encoded.shape[0], -1, -1).clone()
        learned[:, 4:10] = encoded[:, 1:7]
        return learned


def create_actor_critic(
    obs_dim: int,
    model_arch: str = MODEL_ARCH_MLP,
    observation_version: str = OBSERVATION_VERSION,
) -> nn.Module:
    if model_arch == MODEL_ARCH_ATTENTION:
        if observation_version != ATTENTION_OBSERVATION_VERSION:
            raise ValueError(
                "The attention architecture requires "
                f"{ATTENTION_OBSERVATION_VERSION} observations."
            )
        return AttentionActorCritic(obs_dim)
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
    if observation_version == ATTENTION_OBSERVATION_VERSION:
        return 0
    if observation_version == BASE_OBSERVATION_VERSION:
        return BASE_CHARACTER_FEATURE_SIZE
    if observation_version == SKILL_FEATURES_OBSERVATION_VERSION:
        return SKILL_FEATURES_CHARACTER_FEATURE_SIZE
    if observation_version == OBSERVATION_VERSION:
        return COMPACT_CHARACTER_FEATURE_SIZE
    raise ValueError(f"Unknown observation version: {observation_version}")


def character_id_code_count_for_observation_version(observation_version: str) -> int | None:
    if observation_version == ATTENTION_OBSERVATION_VERSION:
        return CHARACTER_ID_CODE_COUNT
    if observation_version == COMPACT_OBSERVATION_VERSION:
        return CHARACTER_ID_CODE_COUNT
    if observation_version in {BASE_OBSERVATION_VERSION, SKILL_FEATURES_OBSERVATION_VERSION}:
        return None
    raise ValueError(f"Unknown observation version: {observation_version}")


def policy_type_for_model_arch(model_arch: str) -> str:
    if model_arch == MODEL_ARCH_MLP:
        return POLICY_TYPE_FACTORED
    if model_arch == MODEL_ARCH_ATTENTION:
        return POLICY_TYPE_ATTENTION
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
    if policy_type == POLICY_TYPE_ATTENTION:
        return MODEL_ARCH_ATTENTION
    if policy_type == POLICY_TYPE_FACTORED_TRANSFORMER:
        return MODEL_ARCH_TRANSFORMER
    if policy_type == POLICY_TYPE_FACTORED_RECURRENT_TRANSFORMER:
        return MODEL_ARCH_RECURRENT_TRANSFORMER
    raise ValueError("Checkpoint uses an unsupported policy type. Retrain the RL model.")


def is_recurrent_model(model: nn.Module) -> bool:
    return hasattr(model, "initial_hidden")
