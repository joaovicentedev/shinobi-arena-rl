from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - only hit without the rl extra.
    raise RuntimeError("Install RL dependencies with `uv sync --extra rl`.") from exc

from naruto_arena.engine.actions import Action, EndTurnAction
from naruto_arena.engine.simulator import legal_actions
from naruto_arena.engine.state import GameState
from naruto_arena.rl.action_space import (
    ACTION_KIND_ORDER,
    ActionKind,
    FactoredAction,
    factored_action_to_engine_action,
    legal_factored_action_masks,
)
from naruto_arena.rl.model import (
    PolicyOutput,
    create_actor_critic,
    is_recurrent_model,
    load_actor_critic_state_dict,
    model_arch_from_checkpoint,
)
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
        load_actor_critic_state_dict(self.model, checkpoint["model_state_dict"])
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
        self.hidden_by_player: dict[int, torch.Tensor] = {}
        self.last_turn_number: int | None = None

    def observe_action(self, before: GameState, action: Action, after: GameState) -> None:
        del before, action, after

    def choose_action(self, state: GameState, player_id: int) -> Action:
        if self.last_turn_number is None or state.turn_number < self.last_turn_number:
            self.hidden_by_player.clear()
        self.last_turn_number = state.turn_number
        legal = legal_actions(state, player_id)
        kind_mask = legal_factored_action_masks(state, player_id, legal=legal)["kind"]
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
            if is_recurrent_model(self.model):
                hidden = self.hidden_by_player.get(player_id)
                if hidden is None:
                    hidden = self.model.initial_hidden(1, obs.device)  # type: ignore[attr-defined]
                policy, _, next_hidden = self.model(obs, hidden)  # type: ignore[misc]
                self.hidden_by_player[player_id] = next_hidden.detach()
            else:
                policy, _ = self.model(obs)
            factored_action = self._choose_factored_action(state, player_id, policy, legal)

        action = factored_action_to_engine_action(state, player_id, factored_action)
        return action if action is not None else EndTurnAction(player_id)

    def _choose_factored_action(
        self,
        state: GameState,
        player_id: int,
        policy: PolicyOutput,
        legal: list[Action],
    ) -> FactoredAction:
        kind = self._select(
            policy.kind,
            legal_factored_action_masks(state, player_id, legal=legal)["kind"],
        )
        action_kind = ACTION_KIND_ORDER[kind]
        if action_kind == ActionKind.END_TURN:
            return FactoredAction(action_kind)

        partial = FactoredAction(action_kind)
        if action_kind == ActionKind.GET_CHAKRA:
            chakra = self._select(
                policy.get_chakra,
                legal_factored_action_masks(state, player_id, partial, legal=legal)["get_chakra"],
            )
            return FactoredAction(action_kind, get_chakra_code=chakra)

        actor_logits = (
            policy.use_skill_joint.amax(dim=(2, 3, 4))
            if policy.use_skill_joint is not None
            else policy.actor
        )
        actor = self._select(
            actor_logits,
            legal_factored_action_masks(state, player_id, partial, legal=legal)["actor"],
        )
        partial = FactoredAction(action_kind, actor_slot=actor)
        skill_logits = (
            policy.use_skill_joint[:, actor].amax(dim=(2, 3))
            if policy.use_skill_joint is not None
            else policy.skill
        )
        skill = self._select(
            skill_logits,
            legal_factored_action_masks(state, player_id, partial, legal=legal)["skill"],
        )
        partial = FactoredAction(action_kind, actor_slot=actor, skill_slot=skill)
        target_logits = (
            policy.use_skill_joint[:, actor, skill].amax(dim=2)
            if policy.use_skill_joint is not None
            else policy.target
        )
        target = self._select(
            target_logits,
            legal_factored_action_masks(state, player_id, partial, legal=legal)["target"],
        )
        partial = FactoredAction(
            action_kind,
            actor_slot=actor,
            skill_slot=skill,
            target_code=target,
        )
        chakra_logits = (
            policy.use_skill_joint[:, actor, skill, target]
            if policy.use_skill_joint is not None
            else policy.random_chakra
        )
        random_chakra = self._select(
            chakra_logits,
            legal_factored_action_masks(state, player_id, partial, legal=legal)["random_chakra"],
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
