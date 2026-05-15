# PyTorch Model Inputs

This document describes only the inputs used by the current pure PyTorch model.
The implementation lives in `naruto_arena/rl/observation.py` and
`naruto_arena/rl/action_space.py`.

## Input Tensor

The model receives one flat float vector:

```text
shape: [obs_dim]
dtype: float32
```

During training and inference, the script adds the batch dimension:

```text
shape: [batch_size, obs_dim]
```

The model does not receive Python objects, skill names, target ids, or action
objects directly. Everything is encoded into numeric features.

Internally, the model no longer treats the whole observation as one undifferentiated
MLP input. It slices the 6 character blocks, runs the same character encoder over
each block, then concatenates those embeddings with global and chakra features.
The Transformer model uses the same flat input vector, but projects the 6
character blocks into character tokens and combines them with one global/chakra
context token. In the current observation version, each character block contains
a compact character id code; the Transformer converts that code into a learned
embedding before projecting the rest of the numeric character and skill features.

Current observation version:

```text
skill_features_compact_id_v1
```

Legacy checkpoints without skill maps use:

```text
base_v1
```

Older skill-map checkpoints use:

```text
skill_features_v1
```

The RL agent selects the encoder from checkpoint metadata. Older checkpoints
without `observation_version` are treated as `base_v1` when their `obs_dim` is
230.

## Perspective

The observation is encoded from the current player's perspective.

That means the vector always uses this order:

```text
global features
my 3 characters
enemy 3 characters
my chakra
enemy chakra placeholder or debug enemy chakra
```

This avoids making the network learn that player 0 and player 1 are symmetric.

## Global Features

The first 4 features are:

```text
turn_number / MAX_TURN
is_current_player_active
my_living_characters / 3
enemy_living_characters / 3
```

Constants:

```text
MAX_TURN = 100
```

`is_current_player_active` should normally be `1.0` when the model is asked to
choose an action.

## Character Blocks

There are 6 character blocks:

```text
3 current-player characters
3 enemy characters
```

Each character block is currently 491 floats:

```text
32 character-state features
+ 9 skill slots * 51 skill features
```

The first 32 character-state features are:

```text
hp / max_hp
is_alive
stunned_turns / MAX_DURATION
invulnerable_turns / MAX_DURATION
total_flat_damage_reduction / 100
total_unpierceable_flat_damage_reduction / 100
max_damage_reduction_percent / 100
total_damage_over_time_amount / 100
total_status_marker_duration / (MAX_DURATION * 4)
total_status_marker_stacks / 5
active_passive_count / 5
triggered_passive_count / 5
used_skill_this_turn
character_id_code
cooldown_skill_slot_0 / MAX_COOLDOWN
cooldown_skill_slot_1 / MAX_COOLDOWN
cooldown_skill_slot_2 / MAX_COOLDOWN
cooldown_skill_slot_3 / MAX_COOLDOWN
cooldown_skill_slot_4 / MAX_COOLDOWN
class_stun_Physical / MAX_DURATION
class_stun_Chakra / MAX_DURATION
class_stun_Mental / MAX_DURATION
class_stun_Melee / MAX_DURATION
class_stun_Ranged / MAX_DURATION
class_stun_Instant / MAX_DURATION
class_stun_Action / MAX_DURATION
class_stun_Control / MAX_DURATION
class_stun_Stun / MAX_DURATION
class_stun_Affliction / MAX_DURATION
class_stun_Passive / MAX_DURATION
class_stun_Unremovable / MAX_DURATION
class_stun_Unique / MAX_DURATION
```

Each skill slot then adds 51 floats. Slots follow the character's fixed
`skill_order`. Skill reordering changes the player's used-skill stack, not the
character's skill slots. Replacement skills are resolved from the current state
before encoding.

```text
is_present
is_replacement_active
is_usable_now
is_passive
is_free
fixed_cost_ninjutsu / MAX_SKILL_COST
fixed_cost_taijutsu / MAX_SKILL_COST
fixed_cost_genjutsu / MAX_SKILL_COST
fixed_cost_bloodline / MAX_SKILL_COST
random_cost / MAX_SKILL_COST
base_cooldown / MAX_COOLDOWN
current_cooldown / MAX_COOLDOWN
duration / MAX_DURATION
target_rule_one_hot[6]
class_one_hot[13]
direct_damage / 100
piercing_direct_damage / 100
conditional_damage_bonus / 100
healing / 100
stun_duration / MAX_DURATION
class_stun_effect_count / 13
flat_damage_reduction / 100
percent_damage_reduction / 100
has_unpierceable_damage_reduction
invulnerability_duration / MAX_DURATION
damage_over_time_amount / 100
damage_over_time_duration / MAX_DURATION
has_piercing_damage_over_time
chakra_removal_amount / MAX_CHAKRA
chakra_steal_amount / MAX_CHAKRA
status_marker_count / 3
has_passive_effect
has_actor_requirement
has_target_requirement
```

Constants:

```text
MAX_SKILL_COST = 4
```

These skill features are a compact map of what the engine definition says the
skill can do. They are not a full symbolic simulator. Conditional factory effects
are encoded using the current state, so active setups such as replacements and
state-dependent damage are visible to the model.

Constants:

```text
MAX_DURATION = 5
MAX_COOLDOWN = 5
```

`used_skill_this_turn` is important because the rules allow each character to
use only one new skill per turn.

## Character Identity

Current `skill_features_compact_id_v1` stores identity as one scalar
`character_id_code` instead of a roster-width one-hot vector:

```text
0: unknown
1: uzumaki_naruto
2: sakura_haruno
3: sasuke_uchiha
4: inuzuka_kiba
5: aburame_shino
6: hyuuga_hinata
7: nara_shikamaru
8: akimichi_chouji
9: yamanaka_ino
```

For the Transformer, this scalar is not treated as an ordinal gameplay value.
`TransformerActorCritic` rounds and clamps it to an integer id, looks it up in an
`nn.Embedding`, removes the scalar from the numeric feature block, and appends
the learned identity embedding before projecting the character token.

This keeps the observation width stable when the roster grows. Adding 200
characters increases the embedding table rows, not every character block by 200
features.

## Cooldowns

Each character has up to 5 skill cooldown slots.

The slots follow the character's current `skill_order`.

```text
cooldown / MAX_COOLDOWN
```

Missing slots are padded with `0.0`.

The engine cooldown rule is turn-based. For example, a skill with cooldown 1
cannot be used on that character's next turn.

## Chakra Features

Each player chakra block has 5 floats:

```text
ninjutsu / MAX_CHAKRA
taijutsu / MAX_CHAKRA
genjutsu / MAX_CHAKRA
bloodline / MAX_CHAKRA
total_chakra / MAX_CHAKRA
```

Constant:

```text
MAX_CHAKRA = 12
```

The model always receives the current player's real chakra.

By default, enemy chakra is hidden because the rules say a player cannot observe
the opponent's exact chakra. In normal mode, the enemy chakra block is:

```text
0.0, 0.0, 0.0, 0.0, 0.0
```

With `--perfect-info`, the enemy chakra block contains the real enemy chakra.
That mode is for debugging, not competitive evaluation.

## Current Observation Size

The current default observation size is:

```text
4 global features
+ 6 character blocks * 491 features
+ 5 my chakra features
+ 5 enemy chakra features
= 2960 floats
```

You can verify this from Python:

```bash
uv run python -c "from naruto_arena.rl.observation import observation_size; \
print(observation_size())"
```

## Action Masks Are Separate

The action mask is not part of the observation vector.

The model outputs factored action logits:

```text
kind:                [3]
actor:               [3]
skill:               [9]
target:              [10]
random_chakra:       [5]
reorder_destination: [2]
```

Then the RL code applies conditional legal masks to each relevant head before
sampling or choosing an action.

This matters because many action ids are invalid in a given state:

- actor is dead;
- actor already used a skill this turn;
- skill is on cooldown;
- skill is passive;
- insufficient chakra;
- target is invalid;
- skill-specific requirements are not satisfied.

The masks come from engine legality checks, so the engine remains the source of
truth.

## Factored Action Policy

The current policy can produce:

```text
END_TURN
USE_SKILL(actor_slot, skill_slot, target_code, random_chakra_code)
REORDER_SKILL(actor_slot, skill_slot, destination)
```

Current constants:

```text
MAX_TEAM_SIZE = 3
MAX_SKILLS_PER_CHARACTER = 9
TARGET_CODE_COUNT = 10
RANDOM_CHAKRA_CODE_COUNT = 5
REORDER_DESTINATION_COUNT = 2
```

So the model emits:

```text
3 + 3 + 9 + 10 + 5 + 2 = 32 logits
```

For comparison, the equivalent fully-flat catalog would contain:

```text
1 + (3 * 9 * 10 * 5) + (3 * 9 * 2) = 1405 action ids
```

That flat catalog still exists as a compatibility/debug adapter, but training
and inference use the factored heads.

Target codes:

```text
0: none
1: self
2: all enemies
3: all allies
4-6: current-player character slots
7-9: enemy character slots
```

Random chakra codes:

```text
0: none
1-4: one concrete chakra type
```

Reorder destinations:

```text
start of skill stack
end of skill stack
```

The engine allows at most 3 reorder actions per player turn. Once the limit is
reached, `REORDER_SKILL` is masked until the next turn starts.

## Important Retraining Rule

If this input layout changes, old checkpoints should be considered invalid.
Changing only `--model-arch` between `mlp` and `transformer` does not change the
observation layout, but checkpoints are still architecture-specific because the
stored weights have different shapes.

The `skill_features_compact_id_v1` layout is incompatible with earlier
230-float and `skill_features_v1` checkpoints as an initialization source. Those
older checkpoints can still be loaded for evaluation or as fixed RL opponents
when their checkpoint metadata identifies the matching observation version.

Examples that require retraining:

- adding or removing observation features;
- changing character id codes;
- changing cooldown semantics;
- changing legal action rules;
- changing policy head sizes;
- changing random chakra payment encoding;
- changing reorder destinations.
