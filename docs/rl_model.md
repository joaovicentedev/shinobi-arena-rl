# Current RL Model

This document describes the current competitive RL model used by
`naruto_arena/rl` and `scripts/train_rl_pytorch.py`.

The active setup is a pure PyTorch MLP actor-critic trained with PPO. The model
uses partial-information observations: both teams, visible status/stack state,
own chakra, and an enemy chakra belief estimate. It does not expose the enemy's
real chakra pool.

## Train

Recommended command:

```bash
uv run --extra rl python scripts/train_rl_pytorch.py \
  --algorithm ppo \
  --model-arch mlp \
  --num-envs 8 \
  --batch-episodes 128 \
  --opponent heuristic \
  --team-sampling random-mirror \
  --episodes 50000 \
  --max-actions 300 \
  --learning-rate 1e-4 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --ppo-clip 0.1 \
  --ppo-epochs 2 \
  --ppo-minibatch-size 256 \
  --entropy-coef 0.005 \
  --value-coef 0.5 \
  --max-grad-norm 0.3 \
  --save-path models/naruto_mlp_ppo_stack_belief.pt
```

Do not use `--perfect-info` for competitive training. The current observation
already includes a belief estimate for enemy chakra without revealing the real
hidden chakra vector.

## Evaluate

Random mirror evaluation against the heuristic opponent:

```bash
uv run --extra rl python scripts/evaluate_rl_benchmarks.py \
  --model-path models/naruto_mlp_ppo_stack_belief.pt \
  --opponent heuristic \
  --random-mirror-matches 1000 \
  --paired-sides \
  --max-actions 300 \
  --output reports/eval_mlp_ppo_stack_belief.json
```

Single match replay:

```bash
uv run --extra rl python scripts/simulate_rl_match.py \
  --model-path models/naruto_mlp_ppo_stack_belief.pt \
  --team-a aburame_shino,uzumaki_naruto,sakura_haruno \
  --team-b sasuke_uchiha,hyuuga_hinata,inuzuka_kiba \
  --seed 42 \
  --max-actions 300 \
  --output reports/rl_match.json

uv run --extra rl python scripts/rl_match_json_to_txt.py \
  reports/rl_match.json \
  --output reports/rl_match.txt
```

## Observation

Current version:

```text
skill_features_compact_id_stack_v1
```

Observation layout:

```text
global[7]
my_chars[3]
enemy_chars[3]
my_chakra[5]
enemy_chakra_belief[13]
visible_stack[MAX_STACK_SIZE=12]
```

Global features:

```text
turn / 100
active_player_is_me
my_alive / 3
enemy_alive / 3
action_count_this_turn / 3
reorder_count_remaining / 3
pending_stack_count / 12
```

Each character block contains compact character identity, HP/alive state,
stuns, invulnerability, damage reduction, damage-over-time, marker summaries,
passive summaries, used-skill-this-turn, cooldowns, class stuns, and per-skill
feature maps for up to 9 skills.

Per-skill feature maps include:

```text
present
replacement/modified flag
can_use
cost
cooldown
duration
target rule
class tags
damage / piercing damage / conditional damage
healing
stun
reduction / invulnerability
damage over time
chakra removal / steal
status markers
requirements flags
```

Enemy chakra is hidden. The model receives a belief vector instead:

```text
enemy_chakra_belief[4]
enemy_chakra_min[4]
enemy_chakra_max[4]
enemy_chakra_total_estimate[1]
```

`ChakraBeliefTracker` updates this estimate from visible information: start-of-
turn chakra gain, visible skill costs, visible chakra exchange, and visible
spending. Invisible enemy skills are ignored. If no tracker history exists, the
encoder falls back to a conservative state-derived estimate.

## Visible Stack

The observation includes visible stack items from both players. Invisible enemy
skills are omitted.

Each stack item has 34 features:

```text
is_present
owner_is_me
source_char_slot one-hot[3]
source_skill_slot one-hot[9]
target_code one-hot[10]
remaining_duration
order_index
is_pending_this_turn
is_from_previous_turn
compact effect summary[6]
```

The stack contains both pending skills chosen this turn and active skills/effects
from previous turns. The `is_pending_this_turn` feature tells the policy whether
an item has not resolved yet.

## Action Space

The policy is factored into small heads. Action kinds:

```text
END_TURN
USE_SKILL
GET_CHAKRA
REORDER_STACK
```

`USE_SKILL`:

```text
actor_slot: 3 logits
skill_slot: 9 logits
target_code: 10 logits
random_chakra_code: 5 logits
```

`GET_CHAKRA`:

```text
get_chakra: 4 logits
```

This action exchanges 5 owned chakras for 1 chosen chakra type.

`REORDER_STACK`:

```text
stack_index: 12 logits
reorder_direction: 2 logits  # left/right
```

The engine allows at most 3 reorder actions per turn. Reorder works on concrete
stack items, not on character skill slots.

All action heads are masked by legal engine actions before sampling. Invalid
skills, invalid targets, unavailable chakra payments, invalid stack indices, and
illegal reorder directions are masked out.

## Model Architecture

The current model is `ActorCritic` in `naruto_arena/rl/model.py`.

```text
6 character blocks -> shared character encoder -> flatten
remaining global/chakra/stack features ---------------------> shared MLP
                                                              -> policy heads
                                                              -> value head
```

Character encoder:

```text
Linear(character_feature_size, 64)
ReLU
Linear(64, 64)
ReLU
```

Shared trunk:

```text
global/suffix features: 433
encoded character features: 6 * 64

Linear(817, 256)
ReLU
Linear(256, 256)
ReLU
```

Policy heads:

```text
kind:              4
actor:             3
skill:             9
target:            10
random_chakra:     5
get_chakra:        4
stack_index:       12
reorder_direction: 2
```

Value head:

```text
Linear(256, 1)
```

## PPO Training

The default algorithm is PPO.

For each episode:

1. Reset with a fresh episode seed.
2. Encode the current partial observation.
3. Run the policy/value model.
4. Apply legal action masks.
5. Sample a factored action.
6. Step the engine.
7. Store observation, action, masks, log probability, entropy, value, reward,
   and done flag.
8. Compute GAE advantages and returns.
9. Update with PPO after `--batch-episodes` episodes.

PPO uses:

```text
clipped policy objective
value MSE
entropy bonus
gradient clipping
```

## Reward

Intermediate actions receive only a tiny action cost:

```text
USE_SKILL:      -0.001
GET_CHAKRA:     -0.001
REORDER_STACK:  -0.001
```

`END_TURN` receives the real shaped turn reward:

```text
0.20 * ((enemy_hp_delta - own_hp_delta) / 300)
+0.15 per enemy KO
-0.15 per ally KO
+1.0 terminal win
-1.0 terminal loss
```

This matches the engine timing: `USE_SKILL` only queues a skill; the actual
damage/effects resolve on `END_TURN`. GAE propagates the turn reward back to the
skill, chakra, and reorder choices that led to it.

Invalid actions receive `-0.05`, but masks should prevent them during normal
training.

## Hidden Information

Competitive observations hide:

```text
enemy real chakra types
enemy invisible skills/effects
```

They expose:

```text
both teams' public character state
known skill definitions
visible cooldowns/statuses
visible stacks/effects
own real chakra
enemy chakra belief
```

## Notes

- Training is reproducible for a fixed `--seed`, but each episode receives a new
  episode seed. Chakra rolls vary across episodes and batches.
- Checkpoints store `observation_version`, `obs_dim`, `model_arch`, and
  `policy_type`. Retrain when observation shape or action heads change.
- The current best checkpoint name used in experiments is
  `models/naruto_mlp_ppo_stack_belief.pt`.
