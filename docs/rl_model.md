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

Train against an existing RL checkpoint:

```bash
make train-rl ARGS="\
  --episodes 10000 \
  --opponent rl \
  --opponent-model-path models/naruto_actor_critic.pt \
  --save-path models/naruto_actor_critic_v2.pt"
```

Fine-tune an existing checkpoint while using it as the rival:

```bash
make train-rl ARGS="\
  --init-model-path models/naruto_actor_critic.pt \
  --opponent rl \
  --opponent-model-path models/naruto_actor_critic.pt \
  --save-path models/naruto_actor_critic_v2.pt \
  --episodes 25000 \
  --learning-rate 1e-4 \
  --log-interval 100"
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

## Model Comparison

To compare two RL checkpoints directly across the same 20 teams, run:

```bash
make compare-rl ARGS="\
  --model-a models/naruto_actor_critic.pt \
  --model-b models/naruto_actor_critic_v2.pt \
  --label-a v1 \
  --label-b v2 \
  --matches-per-pair 3"
```

For every ordered team pair, the script runs both model side assignments:

```text
v1 as player 0 vs v2 as player 1
v2 as player 0 vs v1 as player 1
```

The default report path is:

```text
reports/rl_compare.json
```

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

The model is an actor-critic network implemented in `naruto_arena/rl/model.py`.

```text
observation -> shared character encoder per character -> shared MLP -> policy heads
                                                                  -> state value
```

Architecture:

```text
Each 36-feature character block:
Linear(36, 64)
ReLU
Linear(64, 64)
ReLU

Encoded characters + global/chakra features:
Linear(398, 256)
ReLU
Linear(256, 256)
ReLU

Policy heads:
kind:                Linear(256, 3)
actor:               Linear(256, 3)
skill:               Linear(256, 5)
target:              Linear(256, 10)
random chakra:       Linear(256, 5)
reorder destination: Linear(256, 2)

Value head: Linear(256, 1)
```

The policy chooses a factored action by sampling only the heads relevant to the
selected action kind. The value head estimates the expected future return from
the current state.

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
- Whether the character already used a new skill this turn.
- Character identity one-hot.
- Current skill cooldowns.

## Action Space

Actions are defined in `naruto_arena/rl/action_space.py`.

The policy uses multiple small action heads instead of one large flat action
catalog.

Current action types:

```text
END_TURN
USE_SKILL
REORDER_SKILL
```

`USE_SKILL` is encoded as:

```text
actor_slot
skill_slot
target_code
random_chakra_code
```

Target codes support:

```text
none
self
all enemies
all allies
one concrete character slot
```

Random chakra codes support:

```text
none
one concrete chakra type
```

Skills without random chakra cost only accept `none`. Skills with random chakra
cost require the policy to choose a concrete chakra type that is available after
fixed costs are reserved.

`REORDER_SKILL` is encoded as:

```text
actor_slot
skill_slot
destination
```

Destination is limited to start or end of the skill stack. This keeps the
reorder surface small while still letting the policy alter timing-sensitive
damage, buff, passive, and modifier ordering.

The engine limits each player to 3 reorder actions per turn. After that, reorder
actions are masked until the next turn starts.

The flat catalog is still available as a compatibility/debug adapter, but the
trainer and RL agent use the factored policy heads. The policy emits 28 action
logits per state:

```text
3 + 3 + 5 + 10 + 5 + 2 = 28
```

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

The policy chooses the first random chakra type directly.

For current characters, random chakra costs are one chakra, so this fully
specifies payment. If a future skill costs more than one random chakra, the
adapter pays the selected type first and fills the remaining random cost
deterministically from the most abundant remaining chakra type.

The engine still validates the final payment:

```text
pay fixed chakra first
pay selected random chakra type
then validate the chosen payment
```

Invalid random chakra selections are masked before sampling.

## Training Loop

Training is implemented in `scripts/train_rl_pytorch.py`.

For each episode:

1. Reset the game with a fresh episode seed sampled from the trainer seed.
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
--opponent rl --opponent-model-path models/naruto_actor_critic.pt
```

The default is `heuristic`.

`--opponent rl` loads the checkpoint as a deterministic rival. Use a different
`--save-path` for the new model if you want to keep both checkpoints.

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
- Killing an enemy: `+0.15`.
- Losing an ally: `-0.15`.

There is also a very small penalty for ending a turn with unused chakra.

The shaping values are intentionally small so the model cannot score better by
farming damage or healing instead of winning.

## Current Limitations

- No self-play league yet.
- No PPO clipping yet.
- No recurrent memory for hidden-information inference.
- Random chakra payment only models the first chosen random chakra type.
- Fixed team composition: Naruto, Sakura, Sasuke versus the same team.
- Observation uses hand-built features instead of learned skill embeddings.
- Checkpoints must be retrained when engine rules change the observation shape
  or legal action behavior.

## Recent Experiment Notes

Recent changes made to the RL stack:

- Random chakra payment became an explicit policy decision.
- Skill reordering was added to the RL policy.
- Reorder was limited to 3 actions per player turn after early models looped on
  reorder actions.
- The flat 781-action policy was replaced by a factored policy:

```text
kind + actor + skill + target + random_chakra + reorder_destination
```

- The model now uses a shared character encoder for the 6 character feature
  blocks before the shared actor-critic trunk.
- Training can load an initial checkpoint with `--init-model-path`.
- Training can use a deterministic RL checkpoint as the opponent with
  `--opponent rl --opponent-model-path ...`.
- `compare-rl` was added to compare two checkpoints across all 20 current teams,
  alternating player side for fairness.

Observed v2 fine-tuning result:

```text
model | resolved_wr | overall_wr | wins | losses | unfinished | p0_wr | p1_wr
v1    | 0.541       | 0.527      | 1266 | 1076   | 58         | 0.527 | 0.527
v2    | 0.459       | 0.448      | 1076 | 1266   | 58         | 0.449 | 0.448
```

Interpretation: initializing v2 from v1 and training against deterministic v1
did not improve the checkpoint. The likely failure mode is policy degradation
from on-policy sampling and noisy actor-critic updates against a stronger fixed
opponent. Lower learning rate and lower entropy helped avoid large exploration
shifts, but the tested v2 still underperformed v1 in direct comparison.

This suggests the next major model/training direction should probably be a more
stable update method or league setup, not more fine-tuning against one fixed
checkpoint with the current actor-critic loop.

## Recommended Next Steps

1. Add evaluation script for trained checkpoints.
2. Add PPO clipping while keeping the code pure PyTorch.
3. Support multi-chakra random payment vectors if future skills need them.
4. Add self-play against previous checkpoints.
5. Add recurrent state or belief features for enemy chakra estimation.
6. Tune the reorder action frequency or add penalties if training overuses no-op setup.
