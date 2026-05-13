from __future__ import annotations

import argparse
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from naruto_arena.rl.action_space import (
    ACTION_KIND_ORDER,
    ACTION_KIND_TO_INDEX,
    ActionKind,
    FactoredAction,
)
from naruto_arena.rl.env import NarutoArenaLearningEnv
from naruto_arena.rl.model import (
    MODEL_ARCH_MLP,
    MODEL_ARCHITECTURES,
    PolicyOutput,
    create_actor_critic,
    load_actor_critic_state_dict,
    model_arch_from_checkpoint,
    policy_type_for_model_arch,
)
from naruto_arena.rl.observation import OBSERVATION_VERSION, observation_size

MaskTrace = dict[str, list[bool]]
Trajectory = dict[str, list[Any] | list[float] | int | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure-PyTorch Naruto Arena RL agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-episodes", type=int, default=10)
    parser.add_argument("--max-actions", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--opponent", choices=("random", "heuristic", "rl"), default="heuristic")
    parser.add_argument(
        "--team-sampling",
        choices=("fixed", "random-roster", "random-mirror"),
        default="fixed",
        help="How to choose learner and opponent teams at episode reset.",
    )
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
        "--num-envs",
        type=int,
        default=1,
        help="Number of worker processes used to collect rollout episodes.",
    )
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
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for training: auto, cpu, cuda, cuda:0, etc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1.")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    env = NarutoArenaLearningEnv(
        opponent=args.opponent,
        seed=args.seed,
        max_actions=args.max_actions,
        perfect_info=args.perfect_info,
        opponent_model_path=args.opponent_model_path,
        team_sampling=args.team_sampling,
    )
    model = create_actor_critic(
        observation_size(perfect_info=args.perfect_info),
        args.model_arch,
        OBSERVATION_VERSION,
    )
    model.to(device)
    if args.init_model_path is not None:
        load_initial_model(
            model,
            args.init_model_path,
            perfect_info=args.perfect_info,
            model_arch=args.model_arch,
        )
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pending: list[Trajectory] = []
    recent_returns: list[float] = []
    recent_wins: list[float] = []
    episode_seed_rng = random.Random(args.seed)
    start = time.monotonic()
    loss: float | None = None

    if args.num_envs == 1:
        for episode in range(1, args.episodes + 1):
            trajectory = collect_episode(
                env,
                model,
                args.gamma,
                seed=episode_seed_rng.randrange(2**32),
            )
            pending.append(trajectory)
            recent_returns.append(float(sum(trajectory["rewards"])))
            recent_wins.append(1.0 if trajectory["winner"] == env.learning_player else 0.0)
            if len(pending) >= args.batch_episodes:
                loss = update_model(model, optimizer, pending, args.value_coef, args.entropy_coef)
                pending.clear()
            if episode == 1 or episode % args.log_interval == 0 or episode == args.episodes:
                log_progress(episode, args.episodes, recent_returns, recent_wins, start, loss)
                recent_returns.clear()
                recent_wins.clear()
    else:
        worker_config = {
            "opponent": args.opponent,
            "max_actions": args.max_actions,
            "perfect_info": args.perfect_info,
            "opponent_model_path": (
                None if args.opponent_model_path is None else str(args.opponent_model_path)
            ),
            "team_sampling": args.team_sampling,
            "model_arch": args.model_arch,
            "gamma": args.gamma,
        }
        completed_episodes = 0
        with ProcessPoolExecutor(max_workers=args.num_envs) as executor:
            while completed_episodes < args.episodes:
                batch_size = min(args.batch_episodes, args.episodes - completed_episodes)
                seeds = [episode_seed_rng.randrange(2**32) for _ in range(batch_size)]
                worker_seed_groups = _split_round_robin(seeds, args.num_envs)
                state_dict = _cpu_state_dict(model)
                futures = [
                    executor.submit(
                        collect_episode_worker_batch,
                        worker_config,
                        state_dict,
                        worker_seeds,
                    )
                    for worker_seeds in worker_seed_groups
                    if worker_seeds
                ]
                batch = [
                    trajectory
                    for future in futures
                    for trajectory in future.result()
                ]
                pending.extend(batch)
                loss = update_model(model, optimizer, pending, args.value_coef, args.entropy_coef)
                pending.clear()
                for trajectory in batch:
                    completed_episodes += 1
                    recent_returns.append(float(sum(trajectory["rewards"])))
                    recent_wins.append(
                        1.0 if trajectory["winner"] == env.learning_player else 0.0
                    )
                    if (
                        completed_episodes == 1
                        or completed_episodes % args.log_interval == 0
                        or completed_episodes == args.episodes
                    ):
                        log_progress(
                            completed_episodes,
                            args.episodes,
                            recent_returns,
                            recent_wins,
                            start,
                            loss,
                        )
                        recent_returns.clear()
                        recent_wins.clear()

    if pending:
        update_model(model, optimizer, pending, args.value_coef, args.entropy_coef)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
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
            "team_sampling": args.team_sampling,
            "init_model_path": None if args.init_model_path is None else str(args.init_model_path),
        },
        save_path,
    )
    print(f"saved_model={save_path}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return resolved


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
    load_actor_critic_state_dict(model, checkpoint["model_state_dict"])


def collect_episode(
    env: NarutoArenaLearningEnv,
    model: torch.nn.Module,
    gamma: float,
    *,
    seed: int,
) -> Trajectory:
    observations = env.reset(seed=seed)
    trajectory_observations: list[list[float]] = []
    actions: list[FactoredAction] = []
    action_masks: list[MaskTrace] = []
    rewards: list[float] = []
    done = False
    winner = None
    while not done:
        device = next(model.parameters()).device
        trajectory_observations.append(observations)
        with torch.no_grad():
            obs = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)
            policy, _ = model(obs)
            action, mask_trace = sample_factored_action(env, policy)
        observations, reward, done, info = env.step(factored_action=action)
        actions.append(action)
        action_masks.append(mask_trace)
        rewards.append(reward)
        winner = info["winner"]
    returns = discounted_returns(rewards, gamma)
    return {
        "observations": trajectory_observations,
        "actions": actions,
        "action_masks": action_masks,
        "returns": returns,
        "rewards": rewards,
        "winner": winner,
    }


def sample_factored_action(
    env: NarutoArenaLearningEnv,
    policy: PolicyOutput,
) -> tuple[FactoredAction, MaskTrace]:
    mask_trace: MaskTrace = {}
    kind_mask = env.factored_action_masks()["kind"]
    mask_trace["kind"] = kind_mask
    kind, _, _ = _sample_masked(
        policy.kind,
        kind_mask,
    )
    action_kind = ACTION_KIND_ORDER[int(kind.item())]
    if action_kind == ActionKind.END_TURN:
        return FactoredAction(action_kind), mask_trace

    partial = FactoredAction(action_kind)
    actor_mask = env.factored_action_masks(partial)["actor"]
    mask_trace["actor"] = actor_mask
    actor, _, _ = _sample_masked(
        policy.actor,
        actor_mask,
    )
    partial = FactoredAction(action_kind, actor_slot=int(actor.item()))

    skill_mask = env.factored_action_masks(partial)["skill"]
    mask_trace["skill"] = skill_mask
    skill, _, _ = _sample_masked(
        policy.skill,
        skill_mask,
    )
    partial = FactoredAction(
        action_kind,
        actor_slot=partial.actor_slot,
        skill_slot=int(skill.item()),
    )

    if action_kind == ActionKind.REORDER_SKILL:
        reorder_mask = env.factored_action_masks(partial)["reorder_destination"]
        mask_trace["reorder_destination"] = reorder_mask
        destination, _, _ = _sample_masked(
            policy.reorder_destination,
            reorder_mask,
        )
        return (
            FactoredAction(
                action_kind,
                actor_slot=partial.actor_slot,
                skill_slot=partial.skill_slot,
                reorder_to_end=bool(destination.item()),
            ),
            mask_trace,
        )

    target_mask = env.factored_action_masks(partial)["target"]
    mask_trace["target"] = target_mask
    target, _, _ = _sample_masked(
        policy.target,
        target_mask,
    )
    partial = FactoredAction(
        action_kind,
        actor_slot=partial.actor_slot,
        skill_slot=partial.skill_slot,
        target_code=int(target.item()),
    )
    chakra_mask = env.factored_action_masks(partial)["random_chakra"]
    mask_trace["random_chakra"] = chakra_mask
    chakra, _, _ = _sample_masked(
        policy.random_chakra,
        chakra_mask,
    )
    return (
        FactoredAction(
            action_kind,
            actor_slot=partial.actor_slot,
            skill_slot=partial.skill_slot,
            target_code=partial.target_code,
            random_chakra_code=int(chakra.item()),
        ),
        mask_trace,
    )


def _sample_masked(
    logits: torch.Tensor,
    mask_values: list[bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = torch.tensor(mask_values, dtype=torch.bool, device=logits.device).unsqueeze(0)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    distribution = Categorical(logits=masked_logits)
    action = distribution.sample()
    return (
        action.squeeze(0),
        distribution.log_prob(action).squeeze(0),
        distribution.entropy().squeeze(0),
    )


def _factored_action_log_prob_and_entropy(
    policy: PolicyOutput,
    index: int,
    action: FactoredAction,
    mask_trace: MaskTrace,
) -> tuple[torch.Tensor, torch.Tensor]:
    kind_index = ACTION_KIND_TO_INDEX[action.kind]
    log_prob, entropy = _selected_log_prob_and_entropy(
        policy.kind[index],
        mask_trace["kind"],
        kind_index,
    )
    if action.kind == ActionKind.END_TURN:
        return log_prob, entropy

    actor_log_prob, actor_entropy = _selected_log_prob_and_entropy(
        policy.actor[index],
        mask_trace["actor"],
        action.actor_slot,
    )
    skill_log_prob, skill_entropy = _selected_log_prob_and_entropy(
        policy.skill[index],
        mask_trace["skill"],
        action.skill_slot,
    )
    log_prob = log_prob + actor_log_prob + skill_log_prob
    entropy = entropy + actor_entropy + skill_entropy

    if action.kind == ActionKind.REORDER_SKILL:
        destination_log_prob, destination_entropy = _selected_log_prob_and_entropy(
            policy.reorder_destination[index],
            mask_trace["reorder_destination"],
            int(action.reorder_to_end),
        )
        return log_prob + destination_log_prob, entropy + destination_entropy

    target_log_prob, target_entropy = _selected_log_prob_and_entropy(
        policy.target[index],
        mask_trace["target"],
        action.target_code,
    )
    chakra_log_prob, chakra_entropy = _selected_log_prob_and_entropy(
        policy.random_chakra[index],
        mask_trace["random_chakra"],
        action.random_chakra_code,
    )
    return log_prob + target_log_prob + chakra_log_prob, entropy + target_entropy + chakra_entropy


def _selected_log_prob_and_entropy(
    logits: torch.Tensor,
    mask_values: list[bool],
    selected_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.tensor(mask_values, dtype=torch.bool, device=logits.device)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    distribution = Categorical(logits=masked_logits)
    selected = torch.tensor(selected_index, dtype=torch.long, device=logits.device)
    return distribution.log_prob(selected), distribution.entropy()


def update_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[Trajectory],
    value_coef: float,
    entropy_coef: float,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    observations = torch.tensor(
        [
            observation
            for item in batch
            for observation in item["observations"]
        ],
        dtype=torch.float32,
        device=device,
    )
    actions = [
        action
        for item in batch
        for action in item["actions"]
    ]
    action_masks = [
        mask_trace
        for item in batch
        for mask_trace in item["action_masks"]
    ]
    policy, values = model(observations)
    log_probs_and_entropies = [
        _factored_action_log_prob_and_entropy(policy, index, action, mask_trace)
        for index, (action, mask_trace) in enumerate(zip(actions, action_masks, strict=True))
    ]
    log_probs = torch.stack([item[0] for item in log_probs_and_entropies])
    entropies = torch.stack([item[1] for item in log_probs_and_entropies])
    returns = torch.tensor(
        [
            value
            for item in batch
            for value in item["returns"]
        ],
        dtype=torch.float32,
        device=device,
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


def collect_episode_worker_batch(
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    seeds: list[int],
) -> list[Trajectory]:
    env = NarutoArenaLearningEnv(
        opponent=str(config["opponent"]),
        seed=0,
        max_actions=int(config["max_actions"]),
        perfect_info=bool(config["perfect_info"]),
        opponent_model_path=(
            None
            if config["opponent_model_path"] is None
            else Path(str(config["opponent_model_path"]))
        ),
        team_sampling=str(config["team_sampling"]),
    )
    model = create_actor_critic(
        observation_size(perfect_info=bool(config["perfect_info"])),
        str(config["model_arch"]),
        OBSERVATION_VERSION,
    )
    load_actor_critic_state_dict(model, state_dict)
    model.eval()
    gamma = float(config["gamma"])
    return [collect_episode(env, model, gamma, seed=seed) for seed in seeds]


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }


def _split_round_robin(values: list[int], groups: int) -> list[list[int]]:
    split = [[] for _ in range(groups)]
    for index, value in enumerate(values):
        split[index % groups].append(value)
    return split


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
