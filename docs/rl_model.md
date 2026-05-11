# Pure PyTorch RL Model

This document explains the current reinforcement learning implementation in
`naruto_arena/rl` and `scripts/train_rl_pytorch.py`.

The model is a first playable baseline, not the final competitive agent. It is
designed to prove that the engine can be wrapped for learning, that action
masking works, and that training progress can be inspected without adding an RL
framework dependency.

## Entry Points

Install the RL dependencies:

```bash
make setup-rl
```

Train:

```bash
make train-rl ARGS="--episodes 1000 --opponent heuristic --log-interval 25"
```

Useful debug run:

```bash
make train-rl ARGS="--episodes 10 --batch-episodes 1 --opponent random --log-interval 1"
```

The trainer prints progress as a percentage:

```text
progress= 50.00% episode=500/1000 avg_return=+0.123 win_rate= 42.0% loss=0.5312 elapsed=12.4s
```

The checkpoint is saved by default to:

```text
models/naruto_actor_critic.pt
```

## Tournament Reports

The trained checkpoint can be loaded once and used to play every team matchup:

```bash
make tournament-rl ARGS="--model-path models/naruto_actor_critic.pt --matches-per-pair 3"
```

The tournament uses the same model for both players and for every team. This
answers the question: "If this one policy controls any team, which team performs
best?"

The default report path is:

```text
reports/rl_tournament.json
```

The report follows the same structure as the minimax tournament report:

```text
metadata
characters
teams ranked by resolved_win_rate
```

Tournament progress is also logged as a percentage:

```text
progress= 50.00% games=200/400 elapsed=0.3s
```

Important limitation: the current training command trains on Naruto, Sakura,
and Sasuke mirrors. The tournament can evaluate all teams because the action
and observation encoders support all current characters, but the policy may be
weak on characters it did not see during training.

## Single Match Replay

To inspect whether the model is building a real strategy, run one model-vs-itself
match and save the full action timeline:

```bash
make simulate-rl ARGS="--model-path models/naruto_actor_critic.pt"
```

Select teams with comma-separated character ids:

```bash
make simulate-rl ARGS="\
  --team-a uzumaki_naruto,sakura_haruno,sasuke_uchiha \
  --team-b hyuuga_hinata,inuzuka_kiba,aburame_shino \
  --output reports/rl_match_custom.json"
```

List valid character ids:

```bash
make simulate-rl ARGS="--list-characters"
```

The default report path is:

```text
reports/rl_match.json
```

Each timeline entry includes:

- The selected action.
- The acting player.
- The full state before the action.
- The full state after the action.
- HP, chakra, cooldowns, passives, status markers, damage reductions,
  damage-over-time effects, stuns, and invulnerability.

## Model Type

The model is an actor-critic MLP implemented in `naruto_arena/rl/model.py`.

```text
observation -> shared MLP -> policy logits
                         -> state value
```

Architecture:

```text
Linear(obs_dim, 256)
ReLU
Linear(256, 256)
ReLU

Policy head: Linear(256, num_actions)
Value head:  Linear(256, 1)
```

The policy head chooses the next action. The value head estimates the expected
future return from the current state.

## Observation

Observations are encoded by `naruto_arena/rl/observation.py`.

The vector is perspective-based:

```text
current player team first
enemy team second
current player chakra
enemy chakra placeholder or debug enemy chakra
```

The default observation follows `docs/rules.md`: enemy chakra is hidden from the
player. For that reason, the normal encoder fills enemy chakra features with
zeroes.

For debugging, the trainer supports:

```bash
--perfect-info
```

That flag includes real enemy chakra in the observation. This is useful for
experiments, but it is not faithful to competitive hidden-information rules.

Per-character features include:

- HP and alive state.
- Total stun and class-specific stun duration.
- Invulnerability duration.
- Damage reduction and unpierceable reduction.
- Damage-over-time amount.
- Status marker duration and stacks.
- Passive active and passive-triggered counts.
- Character identity one-hot.
- Current skill cooldowns.

## Action Space

Actions are defined in `naruto_arena/rl/action_space.py`.

The policy uses a fixed discrete action catalog because the neural network needs
a stable output size.

Current action types:

```text
END_TURN
USE_SKILL
```

`USE_SKILL` is encoded as:

```text
actor_slot
skill_slot
target_code
```

Target codes support:

```text
none
self
all enemies
all allies
one concrete character slot
```

Skill reordering is intentionally excluded from the first training version.
The engine supports reordering, but it expands exploration significantly. It
should be added after the base combat agent is learning reliably.

## Action Masking

Before sampling an action, the trainer gets a legal action mask from the RL
environment. Invalid logits are replaced with a very negative value before
constructing the categorical distribution.

Invalid actions include examples such as:

- Using a dead actor.
- Using a missing skill.
- Using a passive as a manual action.
- Using a skill on cooldown.
- Using a skill without enough chakra.
- Choosing an invalid target for the skill rule.
- Violating skill-specific requirements.

The mask is derived from the engine's legal action system, so the engine remains
the source of truth.

## Random Chakra Payment

The current policy does not choose random chakra payment directly.

The RL adapter chooses payment deterministically:

```text
pay fixed chakra first
then pay random chakra from the most abundant remaining chakra type
```

This is a deliberate first-version simplification. It keeps the action space
small while preserving engine validation. Later, random chakra payment can become
part of the action so the model can learn deeper resource management.

## Training Loop

Training is implemented in `scripts/train_rl_pytorch.py`.

For each episode:

1. Reset the game with a deterministic seed.
2. Encode the observation.
3. Run the model to get policy logits and value.
4. Apply the legal action mask.
5. Sample an action from `torch.distributions.Categorical`.
6. Step the environment.
7. Store log probability, entropy, value, and reward.
8. Compute discounted returns.
9. Update the model after `--batch-episodes` episodes.

The loss is:

```text
policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
```

Where:

```text
policy_loss = -log_prob(action) * advantage
advantage = discounted_return - value.detach()
value_loss = mse(value, discounted_return)
entropy_loss = -entropy
```

This is a simple actor-critic update. It is intentionally smaller than PPO so it
is easy to inspect and modify.

## Opponents

The trainer currently supports one learning player against an automated
opponent:

```bash
--opponent random
--opponent heuristic
```

The default is `heuristic`.

The environment automatically plays the opponent's turn until control returns to
the learning player or the match ends.

## Reward

The reward is shaped but terminal-focused.

Terminal reward:

```text
+1.0 for winning
-1.0 for losing
```

Small shaping rewards are added for:

- Reducing enemy team HP.
- Preserving own team HP.
- Killing an enemy.
- Avoiding ally deaths.

There is also a very small penalty for ending a turn with unused chakra.

The shaping values are intentionally small so the model cannot score better by
farming damage or healing instead of winning.

## Current Limitations

- No self-play league yet.
- No PPO clipping yet.
- No recurrent memory for hidden-information inference.
- No learned random chakra payment.
- No skill reordering actions.
- Fixed team composition: Naruto, Sakura, Sasuke versus the same team.
- Observation uses hand-built features instead of learned skill embeddings.

## Recommended Next Steps

1. Add evaluation script for trained checkpoints.
2. Add PPO clipping while keeping the code pure PyTorch.
3. Add explicit random chakra payment actions.
4. Add self-play against previous checkpoints.
5. Add recurrent state or belief features for enemy chakra estimation.
6. Add skill reordering after combat behavior is stable.
