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

Train the experimental Transformer policy without replacing the current MLP
checkpoint:

```bash
make train-rl ARGS="\
  --model-arch transformer \
  --episodes 10000 \
  --opponent heuristic \
  --save-path models/naruto_actor_critic_transformer.pt"
```

Train the recurrent Transformer policy with PPO and simple checkpoint-league
self-play:

```bash
uv run --extra rl python scripts/train_rl_pytorch.py \
  --algorithm ppo \
  --model-arch recurrent_transformer \
  --episodes 50000 \
  --batch-episodes 128 \
  --num-envs 1 \
  --team-sampling random-roster \
  --opponent heuristic \
  --self-play-league-dir models/local/league_recurrent_transformer \
  --self-play-snapshot-interval 1000 \
  --learning-rate 1e-4 \
  --ppo-epochs 2 \
  --ppo-minibatch-size 256 \
  --ppo-clip 0.1 \
  --gae-lambda 0.95 \
  --gamma 0.99 \
  --entropy-coef 0.005 \
  --max-grad-norm 0.3 \
  --device cpu \
  --log-interval 250 \
  --save-path models/local/naruto_actor_critic_recurrent_transformer_ppo.pt
```

Train the Transformer on random 3v3 teams sampled from the 9 hand-authored
characters:

```bash
uv run --extra rl python scripts/train_rl_pytorch.py \
  --model-arch transformer \
  --episodes 5000 \
  --batch-episodes 128 \
  --num-envs 8 \
  --team-sampling random-roster \
  --opponent heuristic \
  --device cpu \
  --log-interval 250 \
  --save-path models/local/naruto_actor_critic_transformer_random_roster.pt
```

Run the fixed benchmark team set:

```bash
uv run --extra rl python scripts/evaluate_rl_benchmarks.py \
  --model-path models/local/naruto_actor_critic_transformer_random_roster.pt \
  --matches-per-benchmark 10 \
  --output reports/rl_benchmarks.json
```

Transformer checkpoints use `policy_type: factored_transformer`. Existing
`policy_type: factored` checkpoints continue to load as the current MLP model.
Recurrent Transformer checkpoints use `policy_type:
factored_recurrent_transformer`.

The `skill_features_v1` observation changes `obs_dim` from the legacy 230-float
layout to 1730 floats. New Transformer skill-feature models must be trained from
scratch or initialized from another `skill_features_v1` checkpoint. Legacy
checkpoints can still be used as opponents and comparisons; they are loaded with
the older `base_v1` observation encoder.

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

Important limitation: the default fixed-team training command trains on Naruto,
Sakura, and Sasuke mirrors. Use `--team-sampling random-roster` to sample random
learner and opponent teams from the expanded roster. The tournament can evaluate
all teams because the action and observation encoders support all current
characters, but the policy may be
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

The RL package has two actor-critic networks implemented in
`naruto_arena/rl/model.py`.

The default model is the original MLP-compatible architecture:

```text
observation -> shared character encoder per character -> shared MLP -> policy heads
                                                                  -> state value
```

Architecture:

```text
Each 286-feature character block:
Linear(286, 64)
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

The experimental Transformer architecture keeps the same observation vector and
the same factored policy heads, but treats each character as a token:

```text
observation -> 6 character tokens + 1 global/chakra context token
            -> side/slot/type embeddings
            -> TransformerEncoder
            -> pooled battle embedding
            -> policy heads
            -> state value
```

Default Transformer settings:

```text
d_model: 128
heads: 4
layers: 2
feedforward: 256
dropout: 0.1
```

This is meant to help the policy model relationships between allied and enemy
characters: threats, focus targets, pairs, and synergies. It does not change the
engine rules, observation size, action masks, or factored action format.

The recurrent Transformer architecture reuses that character/global Transformer
encoder, then adds a GRU memory before the policy and value heads:

```text
observation -> Transformer battle embedding
            -> GRU hidden state
            -> policy heads
            -> state value
```

The GRU state is reset at episode start and carried across decisions in the
match. PPO stores the hidden state that existed before each sampled action and
reuses it during updates, so the policy can condition on previous turns without
requiring a full historical Transformer over every past state.

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
- Per-skill feature maps for each current skill slot: cost, base cooldown, target
  rule, classes, direct damage, stun, healing, damage reduction, invulnerability,
  damage over time, chakra removal/steal, markers, requirements, and passives.

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

Destination is limited to start or end of the player's used-skill stack. This
keeps the reorder surface small while still letting the policy alter
timing-sensitive damage, buff, passive, and modifier ordering without changing
the character's fixed skill list.

The engine limits each player to 3 reorder actions per turn. The same character
skill can be reordered only once per turn, which prevents the policy from
bouncing one skill between the start and end of the stack. After the total limit
or per-skill limit is reached, those reorder actions are masked until the next
turn starts.

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

`--algorithm actor_critic` keeps this simple update path available. The
recommended trainer is `--algorithm ppo`, which collects the same factored
actions and legal masks, computes GAE advantages, and updates with the clipped
PPO objective.

Recommended PPO command:

```bash
uv run --extra rl python scripts/train_rl_pytorch.py \
  --algorithm ppo \
  --model-arch transformer \
  --episodes 50000 \
  --batch-episodes 128 \
  --num-envs 8 \
  --team-sampling random-roster \
  --opponent heuristic \
  --learning-rate 1e-4 \
  --ppo-epochs 2 \
  --ppo-minibatch-size 256 \
  --ppo-clip 0.1 \
  --gae-lambda 0.95 \
  --gamma 0.99 \
  --entropy-coef 0.005 \
  --max-grad-norm 0.3 \
  --device cpu \
  --log-interval 250 \
  --save-path models/local/naruto_actor_critic_transformer_ppo.pt
```

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

`REORDER_SKILL` receives a small `-0.01` action penalty. There is no general
penalty for ending a turn with unused chakra, because saving chakra can be
correct when preparing stronger later-turn combos.

The shaping values are intentionally small so the model cannot score better by
farming damage or healing instead of winning.

## Current Limitations

- Self-play league support is simple: the trainer samples fixed checkpoint
  snapshots from a directory and can periodically add new snapshots.
- Random chakra payment only models the first chosen random chakra type.
- Observation uses hand-built features instead of learned skill embeddings.
- Checkpoints must be retrained when engine rules change the observation shape
  or legal action behavior.

## Recent Experiment Notes

Recent changes made to the RL stack:

- A recurrent Transformer policy was added. It uses the same character/global
  Transformer encoder, then passes each decision through a GRU memory before the
  factored policy heads and value head.
- PPO can train recurrent checkpoints by storing the hidden state that existed
  before each sampled action and reusing it during policy updates.
- The trainer can maintain a simple checkpoint league with
  `--self-play-league-dir` and `--self-play-snapshot-interval`.
- Random chakra payment became an explicit policy decision.
- Skill reordering was added to the RL policy.
- Reorder was limited to 3 actions per player turn, and each character skill can
  only be reordered once per turn, after early models looped on reorder actions.
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
6. Continue tuning the reorder action frequency if training still overuses no-op setup.
