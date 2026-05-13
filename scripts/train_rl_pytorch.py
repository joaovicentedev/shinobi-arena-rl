from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from naruto_arena.rl.action_space import ACTION_KIND_ORDER, ActionKind, FactoredAction
from naruto_arena.rl.env import NarutoArenaLearningEnv
from naruto_arena.rl.model import (
    MODEL_ARCH_MLP,
    MODEL_ARCHITECTURES,
    PolicyOutput,
    create_actor_critic,
    model_arch_from_checkpoint,
    policy_type_for_model_arch,
)
from naruto_arena.rl.observation import OBSERVATION_VERSION, observation_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure-PyTorch Naruto Arena RL agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-episodes", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opponent", choices=("random", "heuristic", "rl"), default="heuristic")
    parser.add_argument(
        "--opponent-model-path",
        type=Path,
        default=None,
        help="Checkpoint path for --opponent rl.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--save-path", default="models/naruto_actor_critic.pt")
    parser.add_argument(
        "--model-arch",
        choices=MODEL_ARCHITECTURES,
        default=MODEL_ARCH_MLP,
        help="Policy/value network architecture.",
    )
    parser.add_argument(
        "--init-model-path",
        type=Path,
        default=None,
        help="Optional checkpoint to initialize model weights before training.",
    )
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
        opponent_model_path=args.opponent_model_path,
    )
    model = create_actor_critic(
        observation_size(perfect_info=args.perfect_info),
        args.model_arch,
        OBSERVATION_VERSION,
    )
    if args.init_model_path is not None:
        load_initial_model(
            model,
            args.init_model_path,
            perfect_info=args.perfect_info,
            model_arch=args.model_arch,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pending: list[dict[str, list[torch.Tensor] | list[float] | int | None]] = []
    recent_returns: list[float] = []
    recent_wins: list[float] = []
    episode_seed_rng = random.Random(args.seed)
    start = time.monotonic()

    for episode in range(1, args.episodes + 1):
        trajectory = collect_episode(
            env,
            model,
            args.gamma,
            seed=episode_seed_rng.randrange(2**32),
        )
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
            "observation_version": OBSERVATION_VERSION,
            "policy_type": policy_type_for_model_arch(args.model_arch),
            "model_arch": args.model_arch,
            "perfect_info": args.perfect_info,
            "opponent": args.opponent,
            "opponent_model_path": (
                None if args.opponent_model_path is None else str(args.opponent_model_path)
            ),
            "init_model_path": None if args.init_model_path is None else str(args.init_model_path),
        },
        save_path,
    )
    print(f"saved_model={save_path}")


def load_initial_model(
    model: torch.nn.Module,
    model_path: Path,
    *,
    perfect_info: bool,
    model_arch: str,
) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint_model_arch = model_arch_from_checkpoint(checkpoint)
    if checkpoint_model_arch != model_arch:
        raise ValueError(
            f"Initial checkpoint uses model architecture {checkpoint_model_arch}, "
            f"but current training uses {model_arch}."
        )
    checkpoint_observation_version = checkpoint.get("observation_version", OBSERVATION_VERSION)
    if checkpoint_observation_version != OBSERVATION_VERSION:
        raise ValueError(
            f"Initial checkpoint uses observation version {checkpoint_observation_version}, "
            f"but current training uses {OBSERVATION_VERSION}."
        )
    expected_obs_dim = observation_size(perfect_info=perfect_info)
    checkpoint_obs_dim = int(checkpoint["obs_dim"])
    if checkpoint_obs_dim != expected_obs_dim:
        raise ValueError(
            f"Initial checkpoint expects observation size {checkpoint_obs_dim}, "
            f"but current training uses {expected_obs_dim}."
        )
    checkpoint_perfect_info = bool(checkpoint.get("perfect_info", False))
    if checkpoint_perfect_info != perfect_info:
        raise ValueError("Initial checkpoint perfect_info setting does not match current training.")
    model.load_state_dict(checkpoint["model_state_dict"])


def collect_episode(
    env: NarutoArenaLearningEnv,
    model: torch.nn.Module,
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
        policy, value = model(obs)
        action, log_prob, entropy = sample_factored_action(env, policy)
        observations, reward, done, info = env.step(factored_action=action)
        log_probs.append(log_prob)
        values.append(value.squeeze(0))
        entropies.append(entropy)
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


def sample_factored_action(
    env: NarutoArenaLearningEnv,
    policy: PolicyOutput,
) -> tuple[FactoredAction, torch.Tensor, torch.Tensor]:
    kind, kind_log_prob, kind_entropy = _sample_masked(
        policy.kind,
        env.factored_action_masks()["kind"],
    )
    action_kind = ACTION_KIND_ORDER[int(kind.item())]
    log_prob = kind_log_prob
    entropy = kind_entropy
    if action_kind == ActionKind.END_TURN:
        return FactoredAction(action_kind), log_prob, entropy

    partial = FactoredAction(action_kind)
    actor, actor_log_prob, actor_entropy = _sample_masked(
        policy.actor,
        env.factored_action_masks(partial)["actor"],
    )
    partial = FactoredAction(action_kind, actor_slot=int(actor.item()))
    log_prob = log_prob + actor_log_prob
    entropy = entropy + actor_entropy

    skill, skill_log_prob, skill_entropy = _sample_masked(
        policy.skill,
        env.factored_action_masks(partial)["skill"],
    )
    partial = FactoredAction(
        action_kind,
        actor_slot=partial.actor_slot,
        skill_slot=int(skill.item()),
    )
    log_prob = log_prob + skill_log_prob
    entropy = entropy + skill_entropy

    if action_kind == ActionKind.REORDER_SKILL:
        destination, destination_log_prob, destination_entropy = _sample_masked(
            policy.reorder_destination,
            env.factored_action_masks(partial)["reorder_destination"],
        )
        return (
            FactoredAction(
                action_kind,
                actor_slot=partial.actor_slot,
                skill_slot=partial.skill_slot,
                reorder_to_end=bool(destination.item()),
            ),
            log_prob + destination_log_prob,
            entropy + destination_entropy,
        )

    target, target_log_prob, target_entropy = _sample_masked(
        policy.target,
        env.factored_action_masks(partial)["target"],
    )
    partial = FactoredAction(
        action_kind,
        actor_slot=partial.actor_slot,
        skill_slot=partial.skill_slot,
        target_code=int(target.item()),
    )
    chakra, chakra_log_prob, chakra_entropy = _sample_masked(
        policy.random_chakra,
        env.factored_action_masks(partial)["random_chakra"],
    )
    return (
        FactoredAction(
            action_kind,
            actor_slot=partial.actor_slot,
            skill_slot=partial.skill_slot,
            target_code=partial.target_code,
            random_chakra_code=int(chakra.item()),
        ),
        log_prob + target_log_prob + chakra_log_prob,
        entropy + target_entropy + chakra_entropy,
    )


def _sample_masked(
    logits: torch.Tensor,
    mask_values: list[bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.tensor(mask_values, dtype=torch.bool).unsqueeze(0)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    distribution = Categorical(logits=masked_logits)
    action = distribution.sample()
    return (
        action.squeeze(0),
        distribution.log_prob(action).squeeze(0),
        distribution.entropy().squeeze(0),
    )


def update_model(
    model: torch.nn.Module,
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
