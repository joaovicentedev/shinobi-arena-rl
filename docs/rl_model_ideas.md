# Attention RL Model Implementation: `attention_skill_stack_no_belief_v1`

## Goal

Replace the current flat MLP observation encoder with a token-based attention model.

Keep:

```text
PPO training
partial-information game rules
own chakra visibility
enemy chakra hidden
legal action masks
existing engine action format
existing reward as baseline
```

Remove:

```text
enemy_chakra_belief
enemy_chakra_min
enemy_chakra_max
enemy_chakra_total_estimate
```

The model should learn from visible state only.

---

# Observation Version

```text
attention_skill_stack_no_belief_v1
```

---

# Token Layout

```text
tokens =
  global_token[1]
  my_char_tokens[3]
  enemy_char_tokens[3]
  my_skill_tokens[27]
  enemy_skill_tokens[27]
  stack_tokens[16]
```

Total:

```text
1 + 3 + 3 + 27 + 27 + 16 = 77 tokens
```

Dead characters are **not removed** and **not masked from attention**.

Instead:

```text
alive = 0
hp = 0
can_use = 0 for their skills
```

Legal action masks still prevent dead characters from acting.

---

# Embeddings

Transformers do not know order or ownership by default. So every token must receive semantic embeddings.

Use learned embeddings:

```text
token_type_embedding:
  GLOBAL
  CHAR
  SKILL
  STACK

side_embedding:
  ME
  ENEMY

char_slot_embedding:
  0
  1
  2

char_id_embedding:
  0..17

skill_slot_embedding:
  0..8

skill_id_embedding:
  one id per implemented skill

stack_position_embedding:
  0..15
```

Do **not** use classic sinusoidal positional encoding.
Use semantic position embeddings instead.

---

# Global Token

Features:

```text
turn / 100
active_player_is_me
my_alive / 3
enemy_alive / 3
action_count_this_turn / 3
reorder_count_remaining / 3
pending_stack_count / 16

my_chakra_bloodline
my_chakra_ninjutsu
my_chakra_taijutsu
my_chakra_genjutsu
my_chakra_total
```

No enemy chakra features.

Build:

```python
global_token = global_mlp(global_features)
global_token += token_type_emb[GLOBAL]
```

---

# Character Tokens

Six tokens:

```text
my_chars[3]
enemy_chars[3]
```

Each character token numeric features:

```text
alive
hp / 100
is_stunned
class_stun_summary
invulnerable
damage_reduction_flat
damage_reduction_percent
dot_damage
marker_summary
passive_summary
used_skill_this_turn
cooldown_summary
```

Build:

```python
char_token = char_mlp(char_numeric_features)

char_token += token_type_emb[CHAR]
char_token += side_emb[ME or ENEMY]
char_token += char_slot_emb[slot]
char_token += char_id_emb[char_id]
```

Do not mask dead chars.

---

# Skill Tokens

Use skills from both teams:

```text
my_skill_tokens = 3 * 9 = 27
enemy_skill_tokens = 3 * 9 = 27
```

Each skill token numeric features:

```text
present
owner_alive
can_use
used_this_turn
cooldown
duration

cost_bloodline
cost_ninjutsu
cost_taijutsu
cost_genjutsu
cost_random

target_rule

class_physical
class_chakra
class_mental
class_melee
class_ranged
class_instant
class_action
class_control
class_affliction
class_stun
class_passive
class_unique

damage
piercing_damage
conditional_damage
healing
stun
class_stun
damage_reduction
invulnerability
dot_damage
chakra_remove
chakra_steal
status_marker
requirement_flags
replacement_or_modified_flag
```

Build:

```python
skill_token = skill_mlp(skill_numeric_features)

skill_token += token_type_emb[SKILL]
skill_token += side_emb[ME or ENEMY]
skill_token += char_slot_emb[owner_slot]
skill_token += char_id_emb[owner_char_id]
skill_token += skill_slot_emb[skill_slot]
skill_token += skill_id_emb[skill_id]
```

Enemy skill definitions are public, so include enemy skill tokens too.

---

# Stack Tokens

Use visible stack only.

```text
max_stack_size = 16
```

Each stack token numeric features:

```text
is_present
remaining_duration
order_index / 16
is_pending_this_turn
is_from_previous_turn

effect_damage
effect_piercing_damage
effect_healing
effect_stun
effect_damage_reduction
effect_invulnerability
effect_dot
effect_chakra_remove
effect_chakra_steal
```

Categorical info:

```text
owner_side
source_char_slot
source_char_id
source_skill_slot
skill_id
target_code
stack_position
```

Build:

```python
stack_token = stack_mlp(stack_numeric_features)

stack_token += token_type_emb[STACK]
stack_token += side_emb[owner_side]
stack_token += char_slot_emb[source_char_slot]
stack_token += char_id_emb[source_char_id]
stack_token += skill_slot_emb[source_skill_slot]
stack_token += skill_id_emb[skill_id]
stack_token += stack_position_emb[stack_position]
target_embedding can be concatenated or added
```

Invisible enemy effects are not included.

---

# Encoder Architecture

Recommended first version:

```text
hidden_dim = 128
num_layers = 2
num_heads = 4
ff_dim = 512
dropout = 0.0 or 0.1
```

Architecture:

```python
tokens = build_tokens(observation)      # [batch, 77, hidden_dim]
tokens = transformer_encoder(tokens)    # [batch, 77, hidden_dim]
```

No padding mask needed if all 77 tokens always exist.

Stack empty slots remain tokens with:

```text
is_present = 0
```

---

# Policy Outputs

Keep the existing external action space:

```text
END_TURN
USE_SKILL
GET_CHAKRA
REORDER_STACK
```

But internally produce better logits.

---

## Action Kind Head

From global output token:

```python
kind_logits = kind_head(global_out)
```

Shape:

```text
[batch, 4]
```

Action kinds:

```text
0 END_TURN
1 USE_SKILL
2 GET_CHAKRA
3 REORDER_STACK
```

Apply existing legal kind mask.

---

## END_TURN

Can either use `kind_logits` only or have a dedicated scalar:

```python
end_turn_logit = end_turn_head(global_out)
```

Simplest: keep inside `kind_logits`.

---

## GET_CHAKRA

From global output token:

```python
get_chakra_logits = get_chakra_head(global_out)
```

Shape:

```text
[batch, 4]
```

Legal only if player has at least 5 chakra.

---

## USE_SKILL: Joint Skill-Target Scoring

Do not use fully independent actor/skill/target heads as the main logic.

Instead score:

```text
actor_slot x skill_slot x target_code x random_chakra_code
```

Shape:

```text
[batch, 3, 9, 10, 5]
```

Implementation:

```python
my_skill_out = outputs[my_skill_token_indices]   # [batch, 27, H]
target_out = build_target_representations(...)   # [batch, 10, H]
global_out = outputs[global_index]               # [batch, H]
```

For each skill-target pair:

```python
pair = concat(
    skill_out,
    target_out,
    global_out,
    skill_out * target_out,
)
```

Then:

```python
base_use_skill_logit = use_skill_pair_mlp(pair)
```

Shape:

```text
[batch, 27, 10]
```

For random chakra:

```python
random_chakra_logits = random_chakra_head(pair)
```

Shape:

```text
[batch, 27, 10, 5]
```

Final:

```text
use_skill_logits = [batch, 3, 9, 10, 5]
```

Apply old legal masks for:

```text
dead actor
used skill this turn
cooldown
stun
chakra availability
invalid target
invalid random chakra payment
```

---

# Target Representations

Target code count remains:

```text
10
```

Recommended target representations:

```text
target 0: none
target 1-3: my char slots
target 4-6: enemy char slots
target 7: all allies
target 8: all enemies
target 9: self / special depending current convention
```

For character targets, use the corresponding character output token.

For group/special targets, use learned embeddings:

```python
target_repr[all_allies] = learned_target_emb[ALL_ALLIES]
target_repr[all_enemies] = learned_target_emb[ALL_ENEMIES]
target_repr[none] = learned_target_emb[NONE]
```

---

# REORDER_STACK

Shape:

```text
[batch, 16, 2]
```

For each stack token:

```python
reorder_input = concat(
    stack_out,
    global_out,
)

reorder_logits = reorder_mlp(reorder_input)
```

Directions:

```text
0 left
1 right
```

Apply legal masks:

```text
stack item exists
item belongs to active player if required
direction is valid
reorder_count_remaining > 0
```

Add a stronger training penalty for useless reorder spam.

Recommended:

```text
REORDER_STACK base cost = -0.003 to -0.01
useless reorder extra penalty = -0.01
```

---

# Value Head

Use global token:

```python
value = value_head(global_out)
```

Optional:

```python
pooled = tokens_out.mean(dim=1)
value = value_head(concat(global_out, pooled))
```

Start with global only.

---

# PyTorch Module Sketch

```python
class AttentionActorCritic(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_chars: int = 18,
        num_skills: int = NUM_IMPLEMENTED_SKILLS,
        max_team_size: int = 3,
        max_skills_per_char: int = 9,
        max_stack_size: int = 16,
        num_target_codes: int = 10,
    ):
        super().__init__()

        emb_dim = hidden_dim

        self.token_type_emb = nn.Embedding(4, hidden_dim)
        self.side_emb = nn.Embedding(2, hidden_dim)
        self.char_slot_emb = nn.Embedding(3, hidden_dim)
        self.char_id_emb = nn.Embedding(num_chars, hidden_dim)
        self.skill_slot_emb = nn.Embedding(9, hidden_dim)
        self.skill_id_emb = nn.Embedding(num_skills, hidden_dim)
        self.stack_position_emb = nn.Embedding(max_stack_size, hidden_dim)
        self.target_code_emb = nn.Embedding(num_target_codes, hidden_dim)

        self.global_mlp = MLP(global_dim, hidden_dim, hidden_dim)
        self.char_mlp = MLP(char_numeric_dim, hidden_dim, hidden_dim)
        self.skill_mlp = MLP(skill_numeric_dim, hidden_dim, hidden_dim)
        self.stack_mlp = MLP(stack_numeric_dim, hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.kind_head = nn.Linear(hidden_dim, 4)
        self.get_chakra_head = nn.Linear(hidden_dim, 4)

        self.use_skill_pair_mlp = MLP(
            hidden_dim * 4,
            hidden_dim,
            5,
        )

        self.reorder_mlp = MLP(
            hidden_dim * 2,
            hidden_dim,
            2,
        )

        self.value_head = nn.Linear(hidden_dim, 1)
```

---

# Forward Output

The model should return the same kind of policy container your PPO code expects.

```python
return PolicyOutput(
    kind_logits=kind_logits,
    use_skill_logits=use_skill_logits,
    get_chakra_logits=get_chakra_logits,
    reorder_logits=reorder_logits,
    value=value,
)
```

Expected shapes:

```text
kind_logits:       [B, 4]
use_skill_logits:  [B, 3, 9, 10, 5]
get_chakra_logits: [B, 4]
reorder_logits:    [B, 16, 2]
value:             [B]
```

---

# PPO Compatibility

Reuse current PPO logic:

```text
collect rollout
store observation
store action
store legal masks
store log_prob
store entropy
store value
store reward
compute GAE
PPO clipped update
```

Only change:

```text
model forward
observation builder
action log-prob extraction for new USE_SKILL shape
```

The legal mask system should stay.

---

# Training Command

Example:

```bash
uv run --extra rl python scripts/train_rl_pytorch.py \
  --algorithm ppo \
  --model-arch attention \
  --observation-version attention_skill_stack_no_belief_v1 \
  --num-envs 8 \
  --batch-episodes 128 \
  --opponent heuristic \
  --team-sampling random-mirror \
  --episodes 50000 \
  --max-actions 300 \
  --learning-rate 5e-5 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --ppo-clip 0.1 \
  --ppo-epochs 2 \
  --ppo-minibatch-size 256 \
  --entropy-coef 0.005 \
  --value-coef 0.5 \
  --max-grad-norm 0.3 \
  --save-path models/naruto_attention_no_belief.pt
```

---

# Evaluation

Compare directly with the current MLP:

```bash
uv run --extra rl python scripts/evaluate_rl_benchmarks.py \
  --model-path models/naruto_attention_no_belief.pt \
  --opponent heuristic \
  --random-mirror-matches 1000 \
  --paired-sides \
  --max-actions 300 \
  --output reports/eval_attention_no_belief.json
```

Track:

```text
win_rate
avg_return
avg_match_turns
avg_actions_per_turn
avg_reorders_per_turn
avg_useless_reorders_per_turn
avg_skills_used_per_turn
avg_end_turn_unused_chakra
damage_to_lowest_hp_enemy_ratio
ally_low_hp_survival_rate
chakra_float_at_end_turn
```

---

# Implementation Order

1. Add new observation version:

```text
attention_skill_stack_no_belief_v1
```

2. Build token-based observation output.

3. Add `AttentionActorCritic`.

4. Add `--model-arch attention`.

5. Convert model outputs to current action sampling system.

6. Reuse old legal masks.

7. Train vs heuristic first.

8. Add reorder penalty.

9. Compare against current MLP.

10. Move to checkpoint league / self-play after basic behavior is sane.

---

# Main Design Decisions

```text
No enemy chakra belief.
Enemy chakra remains hidden.
Dead chars stay as tokens.
Semantic embeddings replace positional encoding.
Skill-target action is scored jointly.
Stack reorder is scored from stack tokens.
Legal masks remain authoritative.
PPO pipeline remains mostly unchanged.
```

This architecture should be much better for learning combos, target choice, visible stack timing, skill synergy, and defensive responses than the current flat MLP.


1. Yes — and this is extremely important.

The stack is not just “currently resolving skills”.

It becomes the model’s **short-term tactical memory**.

The model can learn things like:

```text
"enemy was marked last turn"
"my ally applied damage amplification"
"this control effect is still active"
"this combo starter already exists"
"this action skill is still channeling"
```

Without needing recurrence/LSTM.

So your stack tokens should contain BOTH:

```text
pending skills this turn
+
active effects from previous turns
```

Exactly like your current engine concept already does. 

That is one of the strongest parts of your current design.

I would even rename it conceptually:

```text
visible_effect_stack
```

because it is really:

```text
queued skills
+ ongoing effects
+ tactical state memory
```

So yes, the transformer can learn:

```text
Female Bug exists on target
-> next damage skill stronger

enemy has stun protection
-> avoid wasting stun

enemy marked by Dynamic Air Marking
-> Garouga valuable now
```

This is one of the biggest reasons attention should outperform flat MLP here.

---

2. Yes — I think two stacks is cleaner and better.

Instead of:

```text
stack_tokens[16]
```

Use:

```text
my_stack_tokens[16]
enemy_visible_stack_tokens[16]
```

This removes ambiguity.

Right now the model has to infer ownership from features. Separate stacks make relations easier.

New token layout:

```text
global_token[1]

my_char_tokens[3]
enemy_char_tokens[3]

my_skill_tokens[27]
enemy_skill_tokens[27]

my_stack_tokens[16]
enemy_visible_stack_tokens[16]
```

Total:

```text
1 + 3 + 3 + 27 + 27 + 16 + 16 = 93 tokens
```

Still totally fine.

---

# My Stack

Contains:

```text
pending skills this turn
ongoing own effects
ongoing own controls/actions
ongoing buffs/debuffs caused by me
```

This is full information because it is yours.

---

# Enemy Visible Stack

Contains only visible enemy information:

```text
visible queued enemy skills
visible enemy ongoing effects
visible enemy controls/actions
visible enemy buffs/debuffs
```

Do NOT include invisible enemy effects.

That keeps competitive partial information correct.

---

# Why Two Stacks Are Better

The transformer can now naturally learn:

```text
my_stack interacts with enemy_stack
```

Examples:

```text
enemy control exists
-> my action skill dangerous

my combo starter exists
-> follow-up valuable

enemy invulnerability exists
-> damage skill bad

my delayed damage exists
-> stall good
```

Much cleaner than mixed ownership.

---

# Recommended Stack Token Fields

Each stack token:

```text
is_present

owner_side
source_char_slot
source_char_id
source_skill_slot
skill_id

target_code

remaining_duration
order_index

is_pending_this_turn
is_from_previous_turn

effect_damage
effect_piercing_damage
effect_healing
effect_stun
effect_damage_reduction
effect_invulnerability
effect_dot
effect_chakra_remove
effect_chakra_steal

is_control
is_action
is_invisible
```

And embeddings:

```text
token_type_embedding
side_embedding
char_slot_embedding
char_id_embedding
skill_slot_embedding
skill_id_embedding
stack_position_embedding
```

---

Honestly, I think this change (persistent visible effect stacks + attention) is probably the highest-value architectural improvement you can make for this game.
