from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError as exc:  # pragma: no cover - only hit without the rl extra.
    raise RuntimeError("Install RL dependencies with `uv sync --extra rl`.") from exc

from naruto_arena.engine.actions import Action, EndTurnAction
from naruto_arena.engine.state import GameState
from naruto_arena.rl.action_space import NUM_ACTIONS, action_id_to_engine_action, legal_action_mask
from naruto_arena.rl.model import ActorCritic
from naruto_arena.rl.observation import encode_observation


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
        num_actions = int(checkpoint["num_actions"])
        if num_actions != NUM_ACTIONS:
            raise ValueError(
                f"Checkpoint expects {num_actions} actions, but code defines {NUM_ACTIONS}."
            )
        self.model = ActorCritic(obs_dim, num_actions)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.perfect_info = bool(checkpoint.get("perfect_info", False))
        self.deterministic = deterministic
        self.generator = torch.Generator().manual_seed(seed)

    def choose_action(self, state: GameState, player_id: int) -> Action:
        mask_values = legal_action_mask(state, player_id)
        if not any(mask_values):
            return EndTurnAction(player_id)

        observation = encode_observation(
            state,
            player_id,
            perfect_info=self.perfect_info,
        )
        with torch.no_grad():
            obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.model(obs)
            mask = torch.tensor(mask_values, dtype=torch.bool).unsqueeze(0)
            masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            if self.deterministic:
                action_id = int(torch.argmax(masked_logits, dim=-1).item())
            else:
                probabilities = torch.softmax(masked_logits.squeeze(0), dim=-1)
                action_id = int(
                    torch.multinomial(probabilities, 1, generator=self.generator).item()
                )

        action = action_id_to_engine_action(state, player_id, action_id)
        return action if action is not None else EndTurnAction(player_id)

