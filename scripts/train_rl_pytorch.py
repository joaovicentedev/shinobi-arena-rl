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

from naruto_arena.agents.heuristic_agent import SimpleHeuristicAgent
from naruto_arena.engine.actions import (
    EndTurnAction,
    GetChakraAction,
    ReorderSkillsAction,
    UseSkillAction,
)
from naruto_arena.engine.simulator import resolved_skill
from naruto_arena.engine.skills import TargetRule
from naruto_arena.rl.action_space import (
    ACTION_KIND_ORDER,
    ACTION_KIND_TO_INDEX,
    RANDOM_CHAKRA_NONE,
    RANDOM_CHAKRA_OFFSET,
    TARGET_ALL_ALLIES,
    TARGET_ALL_ENEMIES,
    TARGET_CHARACTER_OFFSET,
    TARGET_NONE,
    TARGET_SELF,
    ActionKind,
    FactoredAction,
    factored_action_to_engine_action,
)
from naruto_arena.rl.env import NarutoArenaLearningEnv
from naruto_arena.rl.model import (
    MODEL_ARCH_MLP,
    MODEL_ARCHITECTURES,
    PolicyOutput,
    create_actor_critic,
    is_recurrent_model,
    load_actor_critic_state_dict,
    model_arch_from_checkpoint,
    policy_type_for_model_arch,
)
from naruto_arena.rl.observation import OBSERVATION_VERSION, OBSERVATION_VERSIONS, observation_size

MaskTrace = dict[str, list[bool]]
Trajectory = dict[str, list[Any] | list[float] | int | None]
UpdateStats = dict[str, float]

ALGORITHM_ACTOR_CRITIC = "actor_critic"
ALGORITHM_PPO = "ppo"
TRAINING_MODE_RL = "rl"
TRAINING_MODE_HEURISTIC_TEACHER = "heuristic-teacher"


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
    parser.add_argument(
        "--self-play-league-dir",
        type=Path,
        default=None,
        help="Directory of checkpoint snapshots to sample as RL opponents.",
    )
    parser.add_argument(
        "--self-play-snapshot-interval",
        type=int,
        default=0,
        help="Save a league snapshot every N episodes; 0 disables snapshots.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--algorithm",
        choices=(ALGORITHM_ACTOR_CRITIC, ALGORITHM_PPO),
        default=ALGORITHM_PPO,
    )
    parser.add_argument(
        "--training-mode",
        choices=(TRAINING_MODE_RL, TRAINING_MODE_HEURISTIC_TEACHER),
        default=TRAINING_MODE_RL,
        help=(
            "rl trains from sampled policy rollouts. heuristic-teacher trains the raw "
            "policy to imitate SimpleHeuristicAgent before later PPO fine-tuning."
        ),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ppo-clip", type=float, default=0.1)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-minibatch-size", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
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
        "--observation-version",
        choices=OBSERVATION_VERSIONS,
        default=OBSERVATION_VERSION,
        help="Observation encoder version.",
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
        observation_version=args.observation_version,
    )
    model = create_actor_critic(
        observation_size(
            perfect_info=args.perfect_info,
            observation_version=args.observation_version,
        ),
        args.model_arch,
        args.observation_version,
    )
    model.to(device)
    if args.init_model_path is not None:
        load_initial_model(
            model,
            args.init_model_path,
            perfect_info=args.perfect_info,
            model_arch=args.model_arch,
            observation_version=args.observation_version,
        )
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    pending: list[Trajectory] = []
    recent_returns: list[float] = []
    recent_wins: list[float] = []
    episode_seed_rng = random.Random(args.seed)
    start = time.monotonic()
    update_stats: UpdateStats | None = None

    if args.training_mode == TRAINING_MODE_HEURISTIC_TEACHER and args.num_envs != 1:
        raise ValueError("--training-mode heuristic-teacher currently requires --num-envs 1.")

    if args.training_mode == TRAINING_MODE_HEURISTIC_TEACHER:
        for episode in range(1, args.episodes + 1):
            teacher = SimpleHeuristicAgent(seed=args.seed + 30_000 + episode, allow_reorder=False)
            trajectory = collect_teacher_episode(
                env,
                model,
                teacher,
                seed=episode_seed_rng.randrange(2**32),
            )
            pending.append(trajectory)
            recent_returns.append(float(sum(trajectory["rewards"])))
            recent_wins.append(1.0 if trajectory["winner"] == env.learning_player else 0.0)
            if len(pending) >= args.batch_episodes:
                update_stats = update_behavior_cloning_model(
                    model,
                    optimizer,
                    pending,
                    max_grad_norm=args.max_grad_norm,
                )
                pending.clear()
            if episode == 1 or episode % args.log_interval == 0 or episode == args.episodes:
                log_progress(
                    episode,
                    args.episodes,
                    recent_returns,
                    recent_wins,
                    start,
                    update_stats,
                )
                recent_returns.clear()
                recent_wins.clear()
    elif args.num_envs == 1:
        for episode in range(1, args.episodes + 1):
            maybe_refresh_self_play_opponent(env, args, episode, episode_seed_rng)
            trajectory = collect_episode(
                env,
                model,
                args.gamma,
                args.gae_lambda,
                seed=episode_seed_rng.randrange(2**32),
            )
            pending.append(trajectory)
            recent_returns.append(float(sum(trajectory["rewards"])))
            recent_wins.append(1.0 if trajectory["winner"] == env.learning_player else 0.0)
            if len(pending) >= args.batch_episodes:
                update_stats = update_model(
                    model,
                    optimizer,
                    pending,
                    algorithm=args.algorithm,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                    ppo_clip=args.ppo_clip,
                    ppo_epochs=args.ppo_epochs,
                    ppo_minibatch_size=args.ppo_minibatch_size,
                    max_grad_norm=args.max_grad_norm,
                )
                pending.clear()
            if episode == 1 or episode % args.log_interval == 0 or episode == args.episodes:
                log_progress(
                    episode,
                    args.episodes,
                    recent_returns,
                    recent_wins,
                    start,
                    update_stats,
                )
                recent_returns.clear()
                recent_wins.clear()
            maybe_save_self_play_snapshot(model, args, episode)
    else:
        worker_config = {
            "opponent": args.opponent,
            "max_actions": args.max_actions,
            "perfect_info": args.perfect_info,
            "opponent_model_path": (
                None if args.opponent_model_path is None else str(args.opponent_model_path)
            ),
            "self_play_league_dir": (
                None if args.self_play_league_dir is None else str(args.self_play_league_dir)
            ),
            "team_sampling": args.team_sampling,
            "model_arch": args.model_arch,
            "observation_version": args.observation_version,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
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
                update_stats = update_model(
                    model,
                    optimizer,
                    pending,
                    algorithm=args.algorithm,
                    value_coef=args.value_coef,
                    entropy_coef=args.entropy_coef,
                    ppo_clip=args.ppo_clip,
                    ppo_epochs=args.ppo_epochs,
                    ppo_minibatch_size=args.ppo_minibatch_size,
                    max_grad_norm=args.max_grad_norm,
                )
                pending.clear()
                for trajectory in batch:
                    completed_episodes += 1
                    recent_returns.append(float(sum(trajectory["rewards"])))
                    recent_wins.append(
                        1.0 if trajectory["winner"] == env.learning_player else 0.0
                    )
                log_progress(
                    completed_episodes,
                    args.episodes,
                    recent_returns,
                    recent_wins,
                    start,
                    update_stats,
                )
                recent_returns.clear()
                recent_wins.clear()

    if pending:
        if args.training_mode == TRAINING_MODE_HEURISTIC_TEACHER:
            update_behavior_cloning_model(
                model,
                optimizer,
                pending,
                max_grad_norm=args.max_grad_norm,
            )
        else:
            update_model(
                model,
                optimizer,
                pending,
                algorithm=args.algorithm,
                value_coef=args.value_coef,
                entropy_coef=args.entropy_coef,
                ppo_clip=args.ppo_clip,
                ppo_epochs=args.ppo_epochs,
                ppo_minibatch_size=args.ppo_minibatch_size,
                max_grad_norm=args.max_grad_norm,
            )
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": observation_size(
                perfect_info=args.perfect_info,
                observation_version=args.observation_version,
            ),
            "observation_version": args.observation_version,
            "policy_type": policy_type_for_model_arch(args.model_arch),
            "model_arch": args.model_arch,
            "perfect_info": args.perfect_info,
            "opponent": args.opponent,
            "opponent_model_path": (
                None if args.opponent_model_path is None else str(args.opponent_model_path)
            ),
            "team_sampling": args.team_sampling,
            "init_model_path": None if args.init_model_path is None else str(args.init_model_path),
            "algorithm": args.algorithm,
            "training_mode": args.training_mode,
            "training": {
                "training_mode": args.training_mode,
                "algorithm": args.algorithm,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "ppo_clip": args.ppo_clip,
                "ppo_epochs": args.ppo_epochs,
                "ppo_minibatch_size": args.ppo_minibatch_size,
                "value_coef": args.value_coef,
                "entropy_coef": args.entropy_coef,
                "max_grad_norm": args.max_grad_norm,
            },
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
    observation_version: str,
) -> None:
    checkpoint = torch.load(model_path, map_location="cpu")
    checkpoint_model_arch = model_arch_from_checkpoint(checkpoint)
    if checkpoint_model_arch != model_arch:
        raise ValueError(
            f"Initial checkpoint uses model architecture {checkpoint_model_arch}, "
            f"but current training uses {model_arch}."
        )
    checkpoint_observation_version = checkpoint.get("observation_version", OBSERVATION_VERSION)
    if checkpoint_observation_version != observation_version:
        raise ValueError(
            f"Initial checkpoint uses observation version {checkpoint_observation_version}, "
            f"but current training uses {observation_version}."
        )
    expected_obs_dim = observation_size(
        perfect_info=perfect_info,
        observation_version=observation_version,
    )
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


def maybe_refresh_self_play_opponent(
    env: NarutoArenaLearningEnv,
    args: argparse.Namespace,
    episode: int,
    rng: random.Random,
) -> None:
    if args.self_play_league_dir is None:
        return
    snapshots = sorted(args.self_play_league_dir.glob("*.pt"))
    if not snapshots:
        return
    # Refresh once per episode so a rollout faces one stable opponent checkpoint.
    env.opponent_name = "rl"
    env.opponent_model_path = rng.choice(snapshots)
    env.opponent = env._make_opponent("rl", args.seed + 20_000 + episode)


def maybe_save_self_play_snapshot(
    model: torch.nn.Module,
    args: argparse.Namespace,
    episode: int,
) -> None:
    if args.self_play_league_dir is None or args.self_play_snapshot_interval <= 0:
        return
    if episode % args.self_play_snapshot_interval != 0:
        return
    args.self_play_league_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.self_play_league_dir / f"snapshot_{episode:08d}.pt"
    torch.save(
        {
            "model_state_dict": _cpu_state_dict(model),
            "obs_dim": observation_size(
                perfect_info=args.perfect_info,
                observation_version=args.observation_version,
            ),
            "observation_version": args.observation_version,
            "policy_type": policy_type_for_model_arch(args.model_arch),
            "model_arch": args.model_arch,
            "perfect_info": args.perfect_info,
            "opponent": args.opponent,
            "team_sampling": args.team_sampling,
            "algorithm": args.algorithm,
            "training": {
                "algorithm": args.algorithm,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "ppo_clip": args.ppo_clip,
                "ppo_epochs": args.ppo_epochs,
                "ppo_minibatch_size": args.ppo_minibatch_size,
                "value_coef": args.value_coef,
                "entropy_coef": args.entropy_coef,
                "max_grad_norm": args.max_grad_norm,
            },
        },
        snapshot_path,
    )


def collect_episode(
    env: NarutoArenaLearningEnv,
    model: torch.nn.Module,
    gamma: float,
    gae_lambda: float,
    *,
    seed: int,
) -> Trajectory:
    observations = env.reset(seed=seed)
    trajectory_observations: list[list[float]] = []
    actions: list[FactoredAction] = []
    action_masks: list[MaskTrace] = []
    log_probs: list[float] = []
    values: list[float] = []
    entropies: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    recurrent_hidden_states: list[list[float]] = []
    done = False
    winner = None
    hidden = None
    while not done:
        device = next(model.parameters()).device
        trajectory_observations.append(observations)
        with torch.no_grad():
            obs = torch.tensor(observations, dtype=torch.float32, device=device).unsqueeze(0)
            if is_recurrent_model(model):
                if hidden is None:
                    hidden = model.initial_hidden(1, device)  # type: ignore[attr-defined]
                recurrent_hidden_states.append(hidden.squeeze(0).detach().cpu().tolist())
                policy, value, hidden = model(obs, hidden)  # type: ignore[misc]
                hidden = hidden.detach()
            else:
                policy, value = model(obs)
            action, mask_trace, log_prob, entropy = sample_factored_action(env, policy)
        observations, reward, done, info = env.step(factored_action=action)
        actions.append(action)
        action_masks.append(mask_trace)
        log_probs.append(float(log_prob.cpu().item()))
        values.append(float(value.squeeze(0).cpu().item()))
        entropies.append(float(entropy.cpu().item()))
        rewards.append(reward)
        dones.append(done)
        winner = info.get("winner")
    returns = discounted_returns(rewards, gamma)
    advantages, gae_returns = generalized_advantage_estimates(
        rewards,
        values,
        dones,
        gamma,
        gae_lambda,
    )
    return {
        "observations": trajectory_observations,
        "actions": actions,
        "action_masks": action_masks,
        "log_probs": log_probs,
        "values": values,
        "entropies": entropies,
        "returns": returns,
        "gae_returns": gae_returns,
        "advantages": advantages,
        "rewards": rewards,
        "dones": dones,
        "recurrent_hidden_states": recurrent_hidden_states,
        "winner": winner,
    }


def collect_teacher_episode(
    env: NarutoArenaLearningEnv,
    model: torch.nn.Module,
    teacher: SimpleHeuristicAgent,
    *,
    seed: int,
) -> Trajectory:
    observations = env.reset(seed=seed)
    trajectory_observations: list[list[float]] = []
    actions: list[FactoredAction] = []
    action_masks: list[MaskTrace] = []
    rewards: list[float] = []
    dones: list[bool] = []
    recurrent_hidden_states: list[list[float]] = []
    done = False
    winner = None
    zero_hidden: list[float] | None = None
    if is_recurrent_model(model):
        recurrent_hidden_dim = int(model.recurrent_hidden_dim)  # type: ignore[attr-defined]
        zero_hidden = [0.0] * recurrent_hidden_dim
    while not done:
        assert env.state is not None
        trajectory_observations.append(observations)
        if zero_hidden is not None:
            recurrent_hidden_states.append(zero_hidden)
        teacher_action = teacher.choose_action(env.state, env.learning_player)
        factored_action = engine_action_to_factored(env.state, env.learning_player, teacher_action)
        converted_action = factored_action_to_engine_action(
            env.state,
            env.learning_player,
            factored_action,
        )
        if converted_action is None:
            raise RuntimeError(
                f"Teacher action could not be converted to a legal factored action: "
                f"{teacher_action!r} -> {factored_action!r}"
            )
        mask_trace = mask_trace_for_factored_action(env, factored_action)
        observations, reward, done, info = env.step(factored_action=factored_action)
        if info.get("invalid_action"):
            raise RuntimeError(
                f"Teacher produced invalid factored action: "
                f"{teacher_action!r} -> {factored_action!r}"
            )
        actions.append(factored_action)
        action_masks.append(mask_trace)
        rewards.append(reward)
        dones.append(done)
        winner = info.get("winner")
    return {
        "observations": trajectory_observations,
        "actions": actions,
        "action_masks": action_masks,
        "log_probs": [],
        "values": [],
        "entropies": [],
        "returns": [0.0] * len(rewards),
        "gae_returns": [0.0] * len(rewards),
        "advantages": [0.0] * len(rewards),
        "rewards": rewards,
        "dones": dones,
        "recurrent_hidden_states": recurrent_hidden_states,
        "winner": winner,
    }


def mask_trace_for_factored_action(
    env: NarutoArenaLearningEnv,
    action: FactoredAction,
) -> MaskTrace:
    mask_trace: MaskTrace = {"kind": env.factored_action_masks()["kind"]}
    if action.kind == ActionKind.END_TURN:
        return mask_trace
    partial = FactoredAction(action.kind)
    if action.kind == ActionKind.GET_CHAKRA:
        mask_trace["get_chakra"] = env.factored_action_masks(partial)["get_chakra"]
        return mask_trace
    if action.kind == ActionKind.REORDER_STACK:
        mask_trace["stack_index"] = env.factored_action_masks(partial)["stack_index"]
        partial = FactoredAction(action.kind, stack_index=action.stack_index)
        mask_trace["reorder_direction"] = env.factored_action_masks(partial)[
            "reorder_direction"
        ]
        return mask_trace
    mask_trace["actor"] = env.factored_action_masks(partial)["actor"]
    partial = FactoredAction(action.kind, actor_slot=action.actor_slot)
    mask_trace["skill"] = env.factored_action_masks(partial)["skill"]
    partial = FactoredAction(
        action.kind,
        actor_slot=action.actor_slot,
        skill_slot=action.skill_slot,
    )
    mask_trace["target"] = env.factored_action_masks(partial)["target"]
    partial = FactoredAction(
        action.kind,
        actor_slot=action.actor_slot,
        skill_slot=action.skill_slot,
        target_code=action.target_code,
    )
    mask_trace["random_chakra"] = env.factored_action_masks(partial)["random_chakra"]
    return mask_trace


def engine_action_to_factored(
    state,
    player_id: int,
    action,
) -> FactoredAction:
    if isinstance(action, EndTurnAction):
        return FactoredAction(ActionKind.END_TURN)
    if isinstance(action, GetChakraAction):
        chakra_types = tuple(type(action.chakra_type))
        return FactoredAction(
            ActionKind.GET_CHAKRA,
            get_chakra_code=chakra_types.index(action.chakra_type),
        )
    if isinstance(action, ReorderSkillsAction):
        player = state.players[player_id]
        stack_index = next(
            index
            for index, used_skill in enumerate(player.skill_stack)
            if used_skill.actor_id == action.character_id
            and used_skill.skill_id == action.skill_id
        )
        direction = 0 if action.new_index < stack_index else 1
        return FactoredAction(
            ActionKind.REORDER_STACK,
            stack_index=stack_index,
            reorder_direction=direction,
        )
    if not isinstance(action, UseSkillAction):
        raise ValueError(f"Unsupported teacher action: {action!r}")
    player = state.players[player_id]
    actor_slot = next(
        index
        for index, character in enumerate(player.characters)
        if character.instance_id == action.actor_id
    )
    actor = player.characters[actor_slot]
    skill_slot = next(
        index
        for index, skill_id in enumerate(actor.skill_order)
        if actor.definition.skill(skill_id).id == action.skill_id
    )
    skill = resolved_skill(state, action.actor_id, action.skill_id)
    random_chakra_code = RANDOM_CHAKRA_NONE
    if action.random_payment:
        chakra_type = max(action.random_payment, key=lambda key: action.random_payment[key])
        random_chakra_code = RANDOM_CHAKRA_OFFSET + tuple(type(chakra_type)).index(chakra_type)
    return FactoredAction(
        ActionKind.USE_SKILL,
        actor_slot=actor_slot,
        skill_slot=skill_slot,
        target_code=target_code_for_action(state, player_id, action, skill.target_rule),
        random_chakra_code=random_chakra_code,
    )


def target_code_for_action(
    state,
    player_id: int,
    action: UseSkillAction,
    target_rule: TargetRule,
) -> int:
    if not action.target_ids:
        return TARGET_NONE
    if target_rule == TargetRule.SELF and action.target_ids == (action.actor_id,):
        return TARGET_SELF
    if target_rule == TargetRule.ALL_ENEMIES:
        enemy_ids = tuple(
            character.instance_id for character in state.players[1 - player_id].living_characters()
        )
        if action.target_ids == enemy_ids:
            return TARGET_ALL_ENEMIES
    if target_rule == TargetRule.ALL_ALLIES:
        ally_ids = tuple(
            character.instance_id for character in state.players[player_id].living_characters()
        )
        if action.target_ids == ally_ids:
            return TARGET_ALL_ALLIES
    target = state.get_character(action.target_ids[0])
    if target.owner == player_id:
        side_slot = state.players[player_id].characters.index(target)
    else:
        side_slot = len(state.players[player_id].characters) + state.players[
            1 - player_id
        ].characters.index(target)
    return TARGET_CHARACTER_OFFSET + side_slot


def sample_factored_action(
    env: NarutoArenaLearningEnv,
    policy: PolicyOutput,
) -> tuple[FactoredAction, MaskTrace, torch.Tensor, torch.Tensor]:
    mask_trace: MaskTrace = {}
    kind_mask = env.factored_action_masks()["kind"]
    mask_trace["kind"] = kind_mask
    kind, log_prob, entropy = _sample_masked(
        policy.kind,
        kind_mask,
    )
    action_kind = ACTION_KIND_ORDER[int(kind.item())]
    if action_kind == ActionKind.END_TURN:
        return FactoredAction(action_kind), mask_trace, log_prob, entropy

    partial = FactoredAction(action_kind)
    if action_kind == ActionKind.GET_CHAKRA:
        chakra_mask = env.factored_action_masks(partial)["get_chakra"]
        mask_trace["get_chakra"] = chakra_mask
        chakra, chakra_log_prob, chakra_entropy = _sample_masked(policy.get_chakra, chakra_mask)
        return (
            FactoredAction(action_kind, get_chakra_code=int(chakra.item())),
            mask_trace,
            log_prob + chakra_log_prob,
            entropy + chakra_entropy,
        )

    if action_kind == ActionKind.REORDER_STACK:
        stack_mask = env.factored_action_masks(partial)["stack_index"]
        mask_trace["stack_index"] = stack_mask
        stack_logits = (
            policy.reorder_joint[:, : len(stack_mask)].amax(dim=2)
            if policy.reorder_joint is not None
            else policy.stack_index
        )
        stack_index, stack_log_prob, stack_entropy = _sample_masked(
            stack_logits,
            stack_mask,
        )
        log_prob = log_prob + stack_log_prob
        entropy = entropy + stack_entropy
        partial = FactoredAction(action_kind, stack_index=int(stack_index.item()))
        reorder_mask = env.factored_action_masks(partial)["reorder_direction"]
        mask_trace["reorder_direction"] = reorder_mask
        direction_logits = (
            policy.reorder_joint[:, partial.stack_index, :]
            if policy.reorder_joint is not None
            else policy.reorder_direction
        )
        direction, direction_log_prob, direction_entropy = _sample_masked(
            direction_logits,
            reorder_mask,
        )
        return (
            FactoredAction(
                action_kind,
                stack_index=partial.stack_index,
                reorder_direction=int(direction.item()),
            ),
            mask_trace,
            log_prob + direction_log_prob,
            entropy + direction_entropy,
        )

    actor_mask = env.factored_action_masks(partial)["actor"]
    mask_trace["actor"] = actor_mask
    actor_logits = (
        policy.use_skill_joint.amax(dim=(2, 3, 4))
        if policy.use_skill_joint is not None
        else policy.actor
    )
    actor, actor_log_prob, actor_entropy = _sample_masked(
        actor_logits,
        actor_mask,
    )
    log_prob = log_prob + actor_log_prob
    entropy = entropy + actor_entropy
    partial = FactoredAction(action_kind, actor_slot=int(actor.item()))

    skill_mask = env.factored_action_masks(partial)["skill"]
    mask_trace["skill"] = skill_mask
    skill_logits = (
        policy.use_skill_joint[:, partial.actor_slot].amax(dim=(2, 3))
        if policy.use_skill_joint is not None
        else policy.skill
    )
    skill, skill_log_prob, skill_entropy = _sample_masked(
        skill_logits,
        skill_mask,
    )
    log_prob = log_prob + skill_log_prob
    entropy = entropy + skill_entropy
    partial = FactoredAction(
        action_kind,
        actor_slot=partial.actor_slot,
        skill_slot=int(skill.item()),
    )

    target_mask = env.factored_action_masks(partial)["target"]
    mask_trace["target"] = target_mask
    target_logits = (
        policy.use_skill_joint[:, partial.actor_slot, partial.skill_slot].amax(dim=2)
        if policy.use_skill_joint is not None
        else policy.target
    )
    target, target_log_prob, target_entropy = _sample_masked(
        target_logits,
        target_mask,
    )
    log_prob = log_prob + target_log_prob
    entropy = entropy + target_entropy
    partial = FactoredAction(
        action_kind,
        actor_slot=partial.actor_slot,
        skill_slot=partial.skill_slot,
        target_code=int(target.item()),
    )
    chakra_mask = env.factored_action_masks(partial)["random_chakra"]
    mask_trace["random_chakra"] = chakra_mask
    chakra_logits = (
        policy.use_skill_joint[
            :,
            partial.actor_slot,
            partial.skill_slot,
            partial.target_code,
        ]
        if policy.use_skill_joint is not None
        else policy.random_chakra
    )
    chakra, chakra_log_prob, chakra_entropy = _sample_masked(
        chakra_logits,
        chakra_mask,
    )
    log_prob = log_prob + chakra_log_prob
    entropy = entropy + chakra_entropy
    return (
        FactoredAction(
            action_kind,
            actor_slot=partial.actor_slot,
            skill_slot=partial.skill_slot,
            target_code=partial.target_code,
            random_chakra_code=int(chakra.item()),
        ),
        mask_trace,
        log_prob,
        entropy,
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

    if action.kind == ActionKind.GET_CHAKRA:
        chakra_log_prob, chakra_entropy = _selected_log_prob_and_entropy(
            policy.get_chakra[index],
            mask_trace["get_chakra"],
            action.get_chakra_code,
        )
        return log_prob + chakra_log_prob, entropy + chakra_entropy

    if action.kind == ActionKind.REORDER_STACK:
        stack_logits = (
            policy.reorder_joint[index, : len(mask_trace["stack_index"])].amax(dim=1)
            if policy.reorder_joint is not None
            else policy.stack_index[index]
        )
        stack_log_prob, stack_entropy = _selected_log_prob_and_entropy(
            stack_logits,
            mask_trace["stack_index"],
            action.stack_index,
        )
        direction_logits = (
            policy.reorder_joint[index, action.stack_index]
            if policy.reorder_joint is not None
            else policy.reorder_direction[index]
        )
        direction_log_prob, direction_entropy = _selected_log_prob_and_entropy(
            direction_logits,
            mask_trace["reorder_direction"],
            action.reorder_direction,
        )
        return (
            log_prob + stack_log_prob + direction_log_prob,
            entropy + stack_entropy + direction_entropy,
        )

    actor_logits = (
        policy.use_skill_joint[index].amax(dim=(1, 2, 3))
        if policy.use_skill_joint is not None
        else policy.actor[index]
    )
    actor_log_prob, actor_entropy = _selected_log_prob_and_entropy(
        actor_logits,
        mask_trace["actor"],
        action.actor_slot,
    )
    skill_logits = (
        policy.use_skill_joint[index, action.actor_slot].amax(dim=(1, 2))
        if policy.use_skill_joint is not None
        else policy.skill[index]
    )
    skill_log_prob, skill_entropy = _selected_log_prob_and_entropy(
        skill_logits,
        mask_trace["skill"],
        action.skill_slot,
    )
    log_prob = log_prob + actor_log_prob + skill_log_prob
    entropy = entropy + actor_entropy + skill_entropy

    target_logits = (
        policy.use_skill_joint[index, action.actor_slot, action.skill_slot].amax(dim=1)
        if policy.use_skill_joint is not None
        else policy.target[index]
    )
    target_log_prob, target_entropy = _selected_log_prob_and_entropy(
        target_logits,
        mask_trace["target"],
        action.target_code,
    )
    chakra_logits = (
        policy.use_skill_joint[
            index,
            action.actor_slot,
            action.skill_slot,
            action.target_code,
        ]
        if policy.use_skill_joint is not None
        else policy.random_chakra[index]
    )
    chakra_log_prob, chakra_entropy = _selected_log_prob_and_entropy(
        chakra_logits,
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
    *,
    algorithm: str,
    value_coef: float,
    entropy_coef: float,
    ppo_clip: float,
    ppo_epochs: int,
    ppo_minibatch_size: int,
    max_grad_norm: float,
) -> UpdateStats:
    if algorithm == ALGORITHM_ACTOR_CRITIC:
        return update_actor_critic_model(
            model,
            optimizer,
            batch,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )
    if algorithm == ALGORITHM_PPO:
        return update_ppo_model(
            model,
            optimizer,
            batch,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            ppo_clip=ppo_clip,
            ppo_epochs=ppo_epochs,
            ppo_minibatch_size=ppo_minibatch_size,
            max_grad_norm=max_grad_norm,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def update_behavior_cloning_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[Trajectory],
    *,
    max_grad_norm: float,
) -> UpdateStats:
    model.eval()
    device = next(model.parameters()).device
    if is_recurrent_model(model):
        return update_recurrent_behavior_cloning_model(
            model,
            optimizer,
            batch,
            device=device,
            max_grad_norm=max_grad_norm,
        )
    rollout = flatten_rollout_batch(batch, device)
    observations = rollout["observations"]
    actions = rollout["actions"]
    action_masks = rollout["action_masks"]
    hidden_states = rollout["recurrent_hidden_states"]
    if hidden_states is not None:
        policy, _, _ = model(observations, hidden_states)  # type: ignore[misc]
    else:
        policy, _ = model(observations)
    log_probs_and_entropies = [
        _factored_action_log_prob_and_entropy(policy, index, action, mask_trace)
        for index, (action, mask_trace) in enumerate(zip(actions, action_masks, strict=True))
    ]
    log_probs = torch.stack([item[0] for item in log_probs_and_entropies])
    entropies = torch.stack([item[1] for item in log_probs_and_entropies])
    policy_loss = -log_probs.mean()
    entropy = entropies.mean()
    loss = policy_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_grad_norm,
    )
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": 0.0,
        "entropy": float(entropy.detach().item()),
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "explained_variance": 0.0,
    }


def update_recurrent_behavior_cloning_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[Trajectory],
    *,
    device: torch.device,
    max_grad_norm: float,
) -> UpdateStats:
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for trajectory in batch:
        hidden = model.initial_hidden(1, device)  # type: ignore[attr-defined]
        observations = trajectory["observations"]
        actions = trajectory["actions"]
        action_masks = trajectory["action_masks"]
        for observation, action, mask_trace in zip(
            observations,
            actions,
            action_masks,
            strict=True,
        ):
            obs = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            policy, _, hidden = model(obs, hidden)  # type: ignore[misc]
            log_prob, entropy = _factored_action_log_prob_and_entropy(
                policy,
                0,
                action,
                mask_trace,
            )
            log_probs.append(log_prob)
            entropies.append(entropy)
    if not log_probs:
        raise ValueError("Behavior cloning batch has no actions.")
    log_prob_tensor = torch.stack(log_probs)
    entropy_tensor = torch.stack(entropies)
    policy_loss = -log_prob_tensor.mean()
    entropy = entropy_tensor.mean()
    loss = policy_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_grad_norm,
    )
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": 0.0,
        "entropy": float(entropy.detach().item()),
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "explained_variance": 0.0,
    }


def update_actor_critic_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[Trajectory],
    *,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> UpdateStats:
    model.eval()
    device = next(model.parameters()).device
    rollout = flatten_rollout_batch(batch, device)
    observations = rollout["observations"]
    actions = rollout["actions"]
    action_masks = rollout["action_masks"]
    hidden_states = rollout["recurrent_hidden_states"]
    if hidden_states is not None:
        policy, values, _ = model(observations, hidden_states)  # type: ignore[misc]
    else:
        policy, values = model(observations)
    log_probs_and_entropies = [
        _factored_action_log_prob_and_entropy(policy, index, action, mask_trace)
        for index, (action, mask_trace) in enumerate(zip(actions, action_masks, strict=True))
    ]
    log_probs = torch.stack([item[0] for item in log_probs_and_entropies])
    entropies = torch.stack([item[1] for item in log_probs_and_entropies])
    returns = rollout["returns"]
    advantages = returns - values.detach()
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy = entropies.mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [parameter for group in optimizer.param_groups for parameter in group["params"]],
        max_grad_norm,
    )
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
        "explained_variance": explained_variance(values.detach(), returns),
    }


def update_ppo_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: list[Trajectory],
    *,
    value_coef: float,
    entropy_coef: float,
    ppo_clip: float,
    ppo_epochs: int,
    ppo_minibatch_size: int,
    max_grad_norm: float,
) -> UpdateStats:
    if ppo_epochs < 1:
        raise ValueError("--ppo-epochs must be at least 1.")
    if ppo_minibatch_size < 1:
        raise ValueError("--ppo-minibatch-size must be at least 1.")
    model.eval()
    device = next(model.parameters()).device
    rollout = flatten_rollout_batch(batch, device)
    observations = rollout["observations"]
    actions = rollout["actions"]
    action_masks = rollout["action_masks"]
    old_log_probs = rollout["old_log_probs"]
    returns = rollout["gae_returns"]
    advantages = rollout["advantages"]
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    sample_count = observations.shape[0]
    last_stats: UpdateStats = {}
    for _ in range(ppo_epochs):
        permutation = torch.randperm(sample_count, device=device)
        for start in range(0, sample_count, ppo_minibatch_size):
            indices = permutation[start : start + ppo_minibatch_size]
            hidden_states = rollout["recurrent_hidden_states"]
            if hidden_states is not None:
                policy, values, _ = model(  # type: ignore[misc]
                    observations[indices],
                    hidden_states[indices],
                )
            else:
                policy, values = model(observations[indices])
            minibatch_actions = [actions[int(index)] for index in indices.cpu()]
            minibatch_masks = [action_masks[int(index)] for index in indices.cpu()]
            log_probs_and_entropies = [
                _factored_action_log_prob_and_entropy(policy, index, action, mask_trace)
                for index, (action, mask_trace) in enumerate(
                    zip(minibatch_actions, minibatch_masks, strict=True)
                )
            ]
            log_probs = torch.stack([item[0] for item in log_probs_and_entropies])
            entropies = torch.stack([item[1] for item in log_probs_and_entropies])
            minibatch_old_log_probs = old_log_probs[indices]
            minibatch_returns = returns[indices]
            minibatch_advantages = advantages[indices]
            ratios = torch.exp(log_probs - minibatch_old_log_probs)
            unclipped_policy_loss = ratios * minibatch_advantages
            clipped_policy_loss = torch.clamp(
                ratios,
                1.0 - ppo_clip,
                1.0 + ppo_clip,
            ) * minibatch_advantages
            policy_loss = -torch.min(unclipped_policy_loss, clipped_policy_loss).mean()
            value_loss = F.mse_loss(values, minibatch_returns)
            entropy = entropies.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                max_grad_norm,
            )
            optimizer.step()
            with torch.no_grad():
                log_ratio = log_probs - minibatch_old_log_probs
                approx_kl = ((torch.exp(log_ratio) - 1.0) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratios - 1.0) > ppo_clip).to(torch.float32).mean()
                )
            last_stats = {
                "loss": float(loss.detach().item()),
                "policy_loss": float(policy_loss.detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "entropy": float(entropy.detach().item()),
                "approx_kl": float(approx_kl.detach().item()),
                "clip_fraction": float(clip_fraction.detach().item()),
                "explained_variance": explained_variance(values.detach(), minibatch_returns),
            }
    return last_stats


def flatten_rollout_batch(batch: list[Trajectory], device: torch.device) -> dict[str, Any]:
    hidden_states = [
        hidden_state
        for item in batch
        for hidden_state in item.get("recurrent_hidden_states", [])
    ]
    return {
        "observations": torch.tensor(
            [observation for item in batch for observation in item["observations"]],
            dtype=torch.float32,
            device=device,
        ),
        "actions": [action for item in batch for action in item["actions"]],
        "action_masks": [mask_trace for item in batch for mask_trace in item["action_masks"]],
        "old_log_probs": torch.tensor(
            [value for item in batch for value in item["log_probs"]],
            dtype=torch.float32,
            device=device,
        ),
        "returns": torch.tensor(
            [value for item in batch for value in item["returns"]],
            dtype=torch.float32,
            device=device,
        ),
        "gae_returns": torch.tensor(
            [value for item in batch for value in item["gae_returns"]],
            dtype=torch.float32,
            device=device,
        ),
        "advantages": torch.tensor(
            [value for item in batch for value in item["advantages"]],
            dtype=torch.float32,
            device=device,
        ),
        "recurrent_hidden_states": (
            torch.tensor(hidden_states, dtype=torch.float32, device=device)
            if hidden_states
            else None
        ),
    }


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def generalized_advantage_estimates(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    advantages: list[float] = [0.0 for _ in rewards]
    last_advantage = 0.0
    next_value = 0.0
    for index in reversed(range(len(rewards))):
        nonterminal = 0.0 if dones[index] else 1.0
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[index] = last_advantage
        next_value = values[index]
    returns = [advantage + value for advantage, value in zip(advantages, values, strict=True)]
    return advantages, returns


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    returns_variance = torch.var(returns, unbiased=False)
    if float(returns_variance.detach().item()) <= 1e-12:
        return 0.0
    residual_variance = torch.var(returns - values, unbiased=False)
    return float((1.0 - residual_variance / returns_variance).detach().item())


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
        observation_version=str(config["observation_version"]),
    )
    model = create_actor_critic(
        observation_size(
            perfect_info=bool(config["perfect_info"]),
            observation_version=str(config["observation_version"]),
        ),
        str(config["model_arch"]),
        str(config["observation_version"]),
    )
    load_actor_critic_state_dict(model, state_dict)
    model.eval()
    gamma = float(config["gamma"])
    gae_lambda = float(config["gae_lambda"])
    return [collect_episode(env, model, gamma, gae_lambda, seed=seed) for seed in seeds]


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
    stats: UpdateStats | None,
) -> None:
    percent = 100 * episode / total_episodes
    elapsed = time.monotonic() - start
    avg_return = sum(returns) / max(1, len(returns))
    win_rate = 100 * sum(wins) / max(1, len(wins))
    if stats is None:
        stats_text = (
            "loss=n/a policy_loss=n/a value_loss=n/a entropy=n/a approx_kl=n/a "
            "clip_fraction=n/a explained_variance=n/a"
        )
    else:
        stats_text = (
            f"loss={stats['loss']:.4f} "
            f"policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f} "
            f"approx_kl={stats['approx_kl']:.5f} "
            f"clip_fraction={stats['clip_fraction']:.3f} "
            f"explained_variance={stats['explained_variance']:+.3f}"
        )
    print(
        f"progress={percent:6.2f}% episode={episode}/{total_episodes} "
        f"avg_return={avg_return:+.3f} win_rate={win_rate:5.1f}% "
        f"{stats_text} elapsed={elapsed:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
