from __future__ import annotations

import random
from pathlib import Path

from naruto_arena.agents.heuristic_agent import SimpleHeuristicAgent
from naruto_arena.agents.random_agent import RandomAgent
from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.data.characters import SAKURA_HARUNO, SASUKE_UCHIHA, UZUMAKI_NARUTO
from naruto_arena.engine.actions import EndTurnAction
from naruto_arena.engine.rules import RulesError, create_initial_state
from naruto_arena.engine.simulator import apply_action
from naruto_arena.engine.state import GameState
from naruto_arena.rl.action_space import (
    FactoredAction,
    action_id_to_engine_action,
    factored_action_to_engine_action,
    legal_action_mask,
    legal_factored_action_masks,
)
from naruto_arena.rl.observation import encode_observation


class NarutoArenaLearningEnv:
    """Small single-learner environment with an automated opponent.

    Decision: this is not a Gym dependency. The trainer only needs reset, step,
    and action_mask, so keeping a tiny local API makes the pure-PyTorch path easier
    to understand and avoids framework assumptions.
    """

    def __init__(
        self,
        *,
        opponent: str = "heuristic",
        seed: int = 0,
        max_actions: int = 300,
        perfect_info: bool = False,
        opponent_model_path: Path | None = None,
    ) -> None:
        self.seed = seed
        self.max_actions = max_actions
        self.perfect_info = perfect_info
        self.learning_player = 0
        self.rng = random.Random(seed)
        self.opponent_name = opponent
        self.opponent_model_path = opponent_model_path
        self.opponent = self._make_opponent(opponent, seed + 10_000)
        self.state: GameState | None = None
        self.actions_taken = 0

    def reset(self, *, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        self.opponent = self._make_opponent(self.opponent_name, self.seed + 10_000)
        team = [UZUMAKI_NARUTO, SAKURA_HARUNO, SASUKE_UCHIHA]
        self.state = create_initial_state(team, team, rng_seed=self.seed)
        self.actions_taken = 0
        return self.observation()

    def observation(self) -> list[float]:
        assert self.state is not None
        return encode_observation(
            self.state,
            self.learning_player,
            perfect_info=self.perfect_info,
        )

    def action_mask(self) -> list[bool]:
        assert self.state is not None
        return legal_action_mask(self.state, self.learning_player)

    def factored_action_masks(
        self,
        partial: FactoredAction | None = None,
    ) -> dict[str, list[bool]]:
        assert self.state is not None
        return legal_factored_action_masks(self.state, self.learning_player, partial)

    def step(
        self,
        action_id: int | None = None,
        *,
        factored_action: FactoredAction | None = None,
    ) -> tuple[list[float], float, bool, dict[str, object]]:
        assert self.state is not None
        before = _score_state(self.state, self.learning_player)
        if factored_action is not None:
            action = factored_action_to_engine_action(
                self.state,
                self.learning_player,
                factored_action,
            )
            is_valid = action is not None
        elif action_id is not None:
            action = action_id_to_engine_action(self.state, self.learning_player, action_id)
            is_valid = action is not None and self.action_mask()[action_id]
        else:
            action = None
            is_valid = False
        if action is None or not is_valid:
            return self.observation(), -0.05, False, {"invalid_action": True}
        try:
            apply_action(self.state, action)
        except RulesError:
            return self.observation(), -0.05, False, {"invalid_action": True}
        self.actions_taken += 1
        self._play_opponent_turn_if_needed()
        after = _score_state(self.state, self.learning_player)
        terminated = self.state.winner is not None
        truncated = self.actions_taken >= self.max_actions
        reward = _shaped_reward(before, after, terminated, self.state.winner, self.learning_player)
        if isinstance(action, EndTurnAction):
            reward -= _unused_chakra_penalty(self.state, self.learning_player)
        return (
            self.observation(),
            reward,
            terminated or truncated,
            {
                "winner": self.state.winner,
                "truncated": truncated,
                "actions": self.actions_taken,
            },
        )

    def _play_opponent_turn_if_needed(self) -> None:
        assert self.state is not None
        while (
            self.state.winner is None
            and self.state.active_player != self.learning_player
            and self.actions_taken < self.max_actions
        ):
            action = self.opponent.choose_action(self.state, self.state.active_player)
            apply_action(self.state, action)
            self.actions_taken += 1

    def _make_opponent(self, name: str, seed: int):
        if name == "random":
            return RandomAgent(seed=seed, allow_reorder=False)
        if name == "heuristic":
            return SimpleHeuristicAgent(seed=seed, allow_reorder=False)
        if name == "rl":
            if self.opponent_model_path is None:
                raise ValueError("--opponent-model-path is required when --opponent rl.")
            return RlAgent(self.opponent_model_path, deterministic=True, seed=seed)
        raise ValueError(f"Unknown opponent: {name}")


def _score_state(state: GameState, player_id: int) -> dict[str, int]:
    enemy_id = 1 - player_id
    own_hp = sum(character.hp for character in state.players[player_id].characters)
    enemy_hp = sum(character.hp for character in state.players[enemy_id].characters)
    own_dead = sum(not character.is_alive for character in state.players[player_id].characters)
    enemy_dead = sum(not character.is_alive for character in state.players[enemy_id].characters)
    return {
        "own_hp": own_hp,
        "enemy_hp": enemy_hp,
        "own_dead": own_dead,
        "enemy_dead": enemy_dead,
    }


def _shaped_reward(
    before: dict[str, int],
    after: dict[str, int],
    terminated: bool,
    winner: int | None,
    player_id: int,
) -> float:
    # Decision: terminal reward dominates. HP and KO shaping are deliberately small
    # so the policy cannot outscore winning by farming damage or healing.
    reward = 0.0
    enemy_hp_delta = before["enemy_hp"] - after["enemy_hp"]
    own_hp_delta = before["own_hp"] - after["own_hp"]
    reward += 0.20 * ((enemy_hp_delta - own_hp_delta) / 300)
    reward += 0.15 * (after["enemy_dead"] - before["enemy_dead"])
    reward -= 0.15 * (after["own_dead"] - before["own_dead"])
    if terminated:
        if winner == player_id:
            reward += 1.0
        elif winner is not None:
            reward -= 1.0
    return reward


def _unused_chakra_penalty(state: GameState, player_id: int) -> float:
    total = state.players[player_id].chakra.total()
    return min(total, 12) * 0.001
