from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from naruto_arena.rl.action_space import NUM_ACTIONS
from naruto_arena.rl.env import NarutoArenaLearningEnv
from naruto_arena.rl.model import ActorCritic
from naruto_arena.rl.observation import observation_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure-PyTorch Naruto Arena RL agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-episodes", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opponent", choices=("random", "heuristic"), default="heuristic")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--save-path", default="models/naruto_actor_critic.pt")
    parser.add_argument(
        "--perfect-info",
        action="store_true",
        help="Debug mode: include hidden enemy chakra in the observation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    env = NarutoArenaLearningEnv(
        opponent=args.opponent,
        seed=args.seed,
        max_actions=args.max_actions,
        perfect_info=args.perfect_info,
    )
    model = ActorCritic(observation_size(perfect_info=args.perfect_info), NUM_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pending: list[dict[str, list[torch.Tensor] | list[float] | int | None]] = []
    recent_returns: list[float] = []
    recent_wins: list[float] = []
    start = time.monotonic()

    for episode in range(1, args.episodes + 1):
        trajectory = collect_episode(env, model, args.gamma, seed=args.seed + episode)
        pending.append(trajectory)
        recent_returns.append(float(sum(trajectory["rewards"])))  # type: ignore[arg-type]
        recent_wins.append(1.0 if trajectory["winner"] == env.learning_player else 0.0)
        if len(pending) >= args.batch_episodes:
            loss = update_model(model, optimizer, pending, args.value_coef, args.entropy_coef)
            pending.clear()
        else:
            loss = None
        if episode == 1 or episode % args.log_interval == 0 or episode == args.episodes:
            log_progress(episode, args.episodes, recent_returns, recent_wins, start, loss)
            recent_returns.clear()
            recent_wins.clear()

    if pending:
        update_model(model, optimizer, pending, args.value_coef, args.entropy_coef)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": observation_size(perfect_info=args.perfect_info),
            "num_actions": NUM_ACTIONS,
            "perfect_info": args.perfect_info,
            "opponent": args.opponent,
        },
        save_path,
    )
    print(f"saved_model={save_path}")


def collect_episode(
    env: NarutoArenaLearningEnv,
    model: ActorCritic,
    gamma: float,
    *,
    seed: int,
) -> dict[str, list[torch.Tensor] | list[float] | int | None]:
    observations = env.reset(seed=seed)
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    rewards: list[float] = []
    done = False
    winner = None
    while not done:
        obs = torch.tensor(observations, dtype=torch.float32).unsqueeze(0)
        logits, value = model(obs)
        mask = torch.tensor(env.action_mask(), dtype=torch.bool).unsqueeze(0)
        masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        distribution = Categorical(logits=masked_logits)
        action = distribution.sample()
        observations, reward, done, info = env.step(int(action.item()))
        log_probs.append(distribution.log_prob(action).squeeze(0))
        values.append(value.squeeze(0))
        entropies.append(distribution.entropy().squeeze(0))
        rewards.append(reward)
        winner = info["winner"]
    returns = discounted_returns(rewards, gamma)
    return {
        "log_probs": log_probs,
        "values": values,
        "entropies": entropies,
        "returns": returns,
        "rewards": rewards,
        "winner": winner,
    }


def update_model(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: list[dict[str, list[torch.Tensor] | list[float] | int | None]],
    value_coef: float,
    entropy_coef: float,
) -> float:
    del model
    log_probs = torch.cat(
        [torch.stack(item["log_probs"]) for item in batch]  # type: ignore[arg-type]
    )
    values = torch.cat([torch.stack(item["values"]) for item in batch])  # type: ignore[arg-type]
    entropies = torch.cat(
        [torch.stack(item["entropies"]) for item in batch]  # type: ignore[arg-type]
    )
    returns = torch.tensor(
        [value for item in batch for value in item["returns"]],  # type: ignore[union-attr]
        dtype=torch.float32,
    )
    advantages = returns - values.detach()
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_loss = -entropies.mean()
    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        1.0,
    )
    optimizer.step()
    return float(loss.detach().item())


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def log_progress(
    episode: int,
    total_episodes: int,
    returns: list[float],
    wins: list[float],
    start: float,
    loss: float | None,
) -> None:
    percent = 100 * episode / total_episodes
    elapsed = time.monotonic() - start
    avg_return = sum(returns) / max(1, len(returns))
    win_rate = 100 * sum(wins) / max(1, len(wins))
    loss_text = "n/a" if loss is None else f"{loss:.4f}"
    print(
        f"progress={percent:6.2f}% episode={episode}/{total_episodes} "
        f"avg_return={avg_return:+.3f} win_rate={win_rate:5.1f}% "
        f"loss={loss_text} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
