from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the rl extra installed.
    raise RuntimeError("Install RL dependencies with `uv sync --extra rl`.") from exc


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_dim, num_actions)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(observations)
        return self.policy(hidden), self.value(hidden).squeeze(-1)

