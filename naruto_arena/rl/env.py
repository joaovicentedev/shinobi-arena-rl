from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

from naruto_arena.agents.heuristic_agent import SimpleHeuristicAgent
from naruto_arena.agents.random_agent import RandomAgent
from naruto_arena.agents.rl_agent import RlAgent
from naruto_arena.engine.actions import Action, EndTurnAction
from naruto_arena.engine.rules import RulesError, create_initial_state
from naruto_arena.engine.simulator import apply_action, legal_actions
from naruto_arena.engine.state import GameState
from naruto_arena.rl.action_space import (
    FactoredAction,
    action_id_to_engine_action,
    factored_action_to_engine_action,
    legal_action_mask,
    legal_factored_action_masks,
)
from naruto_arena.rl.belief import ChakraBeliefTracker
from naruto_arena.rl.observation import encode_observation
from naruto_arena.rl.teams import default_team, random_mirror_teams, random_teams


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
        team_sampling: str = "fixed",
    ) -> None:
        self.seed = seed
        self.max_actions = max_actions
        self.perfect_info = perfect_info
        self.learning_player = 0
        self.rng = random.Random(seed)
        self.opponent_name = opponent
        self.opponent_model_path = opponent_model_path
        self.team_sampling = team_sampling
        self.opponent = self._make_opponent(opponent, seed + 10_000)
        self.state: GameState | None = None
        self.actions_taken = 0
        self.belief_tracker = ChakraBeliefTracker(self.learning_player)
        self._legal_actions_cache_key: tuple[int, int, int, int, int | None] | None = None
        self._legal_actions_cache: list[Action] | None = None

    def reset(self, *, seed: int | None = None) -> list[float]:
        if seed is not None:
            self.seed = seed
            self.rng.seed(seed)
        self.opponent = self._make_opponent(self.opponent_name, self.seed + 10_000)
        team_a, team_b = self._episode_teams()
        self.state = create_initial_state(team_a, team_b, rng_seed=self.seed)
        self.actions_taken = 0
        self.belief_tracker.reset(self.state)
        self._clear_legal_actions_cache()
        return self.observation()

    def observation(self) -> list[float]:
        assert self.state is not None
        return encode_observation(
            self.state,
            self.learning_player,
            perfect_info=self.perfect_info,
            enemy_chakra_belief=self.belief_tracker.features(self.state),
        )

    def action_mask(self) -> list[bool]:
        assert self.state is not None
        return legal_action_mask(self.state, self.learning_player)

    def factored_action_masks(
        self,
        partial: FactoredAction | None = None,
    ) -> dict[str, list[bool]]:
        assert self.state is not None
        return legal_factored_action_masks(
            self.state,
            self.learning_player,
            partial,
            legal=self._legal_actions_for_current_state(),
        )

    def step(
        self,
        action_id: int | None = None,
        *,
        factored_action: FactoredAction | None = None,
    ) -> tuple[list[float], float, bool, dict[str, object]]:
        assert self.state is not None
        before_state = deepcopy(self.state)
        before = _score_state(before_state, self.learning_player)
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
            return self.observation(), -0.05, False, self._step_info(invalid_action=True)
        try:
            apply_action(self.state, action)
        except RulesError:
            return self.observation(), -0.05, False, self._step_info(invalid_action=True)
        self.belief_tracker.observe_action(before_state, action, self.state)
        self._notify_opponent_observer(before_state, action)
        self.actions_taken += 1
        self._clear_legal_actions_cache()
        self._play_opponent_turn_if_needed()
        after = _score_state(self.state, self.learning_player)
        terminated = self.state.winner is not None
        truncated = self.actions_taken >= self.max_actions
        reward = -0.001
        if isinstance(action, EndTurnAction):
            reward += _shaped_reward(
                before,
                after,
                terminated,
                self.state.winner,
                self.learning_player,
            )
        return (
            self.observation(),
            reward,
            terminated or truncated,
            self._step_info(truncated=truncated),
        )

    def _play_opponent_turn_if_needed(self) -> None:
        assert self.state is not None
        while (
            self.state.winner is None
            and self.state.active_player != self.learning_player
            and self.actions_taken < self.max_actions
        ):
            action = self.opponent.choose_action(self.state, self.state.active_player)
            before_state = deepcopy(self.state)
            apply_action(self.state, action)
            self.belief_tracker.observe_action(before_state, action, self.state)
            self._notify_opponent_observer(before_state, action)
            self.actions_taken += 1
            self._clear_legal_actions_cache()

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

    def _notify_opponent_observer(self, before: GameState, action: Action) -> None:
        assert self.state is not None
        observer = getattr(self.opponent, "observe_action", None)
        if observer is not None:
            observer(before, action, self.state)

    def _episode_teams(self):
        if self.team_sampling == "fixed":
            team = default_team()
            return team, list(team)
        if self.team_sampling == "random-roster":
            return random_teams(self.rng)
        if self.team_sampling == "random-mirror":
            return random_mirror_teams(self.rng)
        raise ValueError(f"Unknown team sampling mode: {self.team_sampling}")

    def _legal_actions_for_current_state(self) -> list[Action]:
        assert self.state is not None
        cache_key = (
            id(self.state),
            self.state.active_player,
            self.state.turn_number,
            self.actions_taken,
            self.state.winner,
        )
        if cache_key != self._legal_actions_cache_key:
            self._legal_actions_cache_key = cache_key
            self._legal_actions_cache = legal_actions(self.state, self.learning_player)
        assert self._legal_actions_cache is not None
        return self._legal_actions_cache

    def _clear_legal_actions_cache(self) -> None:
        self._legal_actions_cache_key = None
        self._legal_actions_cache = None

    def _step_info(
        self,
        *,
        truncated: bool = False,
        invalid_action: bool = False,
    ) -> dict[str, object]:
        assert self.state is not None
        return {
            "winner": self.state.winner,
            "truncated": truncated,
            "actions": self.actions_taken,
            "invalid_action": invalid_action,
        }


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
