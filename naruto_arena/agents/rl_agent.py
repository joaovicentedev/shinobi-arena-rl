from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - only hit without the rl extra.
    raise RuntimeError("Install RL dependencies with `uv sync --extra rl`.") from exc

from naruto_arena.engine.actions import Action, EndTurnAction
from naruto_arena.engine.state import GameState
from naruto_arena.rl.action_space import (
    ACTION_KIND_ORDER,
    ActionKind,
    FactoredAction,
    factored_action_to_engine_action,
    legal_factored_action_masks,
)
from naruto_arena.rl.model import PolicyOutput, create_actor_critic, model_arch_from_checkpoint
from naruto_arena.rl.observation import (
    BASE_OBSERVATION_VERSION,
    OBSERVATION_VERSION,
    encode_observation,
    observation_size,
)


class RlAgent:
    def __init__(
        self,
        model_path: Path,
        *,
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        checkpoint = torch.load(model_path, map_location="cpu")
        obs_dim = int(checkpoint["obs_dim"])
        self.model_arch = model_arch_from_checkpoint(checkpoint)
        self.observation_version = _observation_version_from_checkpoint(checkpoint, obs_dim)
        self.model = create_actor_critic(
            obs_dim,
            self.model_arch,
            self.observation_version,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.perfect_info = bool(checkpoint.get("perfect_info", False))
        current_obs_dim = observation_size(
            perfect_info=self.perfect_info,
            observation_version=self.observation_version,
        )
        if obs_dim != current_obs_dim:
            raise ValueError(
                f"Checkpoint expects observation size {obs_dim}, but current rules "
                f"encode {current_obs_dim}. Retrain the RL model."
            )
        self.deterministic = deterministic
        self.generator = torch.Generator().manual_seed(seed)

    def choose_action(self, state: GameState, player_id: int) -> Action:
        kind_mask = legal_factored_action_masks(state, player_id)["kind"]
        if not any(kind_mask):
            return EndTurnAction(player_id)

        observation = encode_observation(
            state,
            player_id,
            perfect_info=self.perfect_info,
            observation_version=self.observation_version,
        )
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            policy, _ = self.model(obs)
            factored_action = self._choose_factored_action(state, player_id, policy)

        action = factored_action_to_engine_action(state, player_id, factored_action)
        return action if action is not None else EndTurnAction(player_id)

    def _choose_factored_action(
        self,
        state: GameState,
        player_id: int,
        policy: PolicyOutput,
    ) -> FactoredAction:
        kind = self._select(policy.kind, legal_factored_action_masks(state, player_id)["kind"])
        action_kind = ACTION_KIND_ORDER[kind]
        if action_kind == ActionKind.END_TURN:
            return FactoredAction(action_kind)

        partial = FactoredAction(action_kind)
        actor = self._select(
            policy.actor,
            legal_factored_action_masks(state, player_id, partial)["actor"],
        )
        partial = FactoredAction(action_kind, actor_slot=actor)
        skill = self._select(
            policy.skill,
            legal_factored_action_masks(state, player_id, partial)["skill"],
        )
        partial = FactoredAction(action_kind, actor_slot=actor, skill_slot=skill)
        if action_kind == ActionKind.REORDER_SKILL:
            destination = self._select(
                policy.reorder_destination,
                legal_factored_action_masks(state, player_id, partial)["reorder_destination"],
            )
            return FactoredAction(
                action_kind,
                actor_slot=actor,
                skill_slot=skill,
                reorder_to_end=bool(destination),
            )

        target = self._select(
            policy.target,
            legal_factored_action_masks(state, player_id, partial)["target"],
        )
        partial = FactoredAction(
            action_kind,
            actor_slot=actor,
            skill_slot=skill,
            target_code=target,
        )
        random_chakra = self._select(
            policy.random_chakra,
            legal_factored_action_masks(state, player_id, partial)["random_chakra"],
        )
        return FactoredAction(
            action_kind,
            actor_slot=actor,
            skill_slot=skill,
            target_code=target,
            random_chakra_code=random_chakra,
        )

    def _select(self, logits: torch.Tensor, mask_values: list[bool]) -> int:
        mask = torch.tensor(mask_values, dtype=torch.bool).unsqueeze(0)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        if self.deterministic:
            return int(torch.argmax(masked_logits, dim=-1).item())
        probabilities = torch.softmax(masked_logits.squeeze(0), dim=-1)
        return int(torch.multinomial(probabilities, 1, generator=self.generator).item())


def _observation_version_from_checkpoint(
    checkpoint: dict[str, object],
    obs_dim: int,
) -> str:
    observation_version = checkpoint.get("observation_version")
    if observation_version is not None:
        return str(observation_version)
    if obs_dim == observation_size(observation_version=BASE_OBSERVATION_VERSION):
        return BASE_OBSERVATION_VERSION
    return OBSERVATION_VERSION
