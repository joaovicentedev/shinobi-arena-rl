# Reinforcement Learning Plan for a Naruto Arena-Style Turn-Based Game

## Context

This document describes how to train an AI agent for a deterministic Python engine inspired by Naruto Arena.

The game is a turn-based battle between two teams of three characters. Each character has health, cooldowns, skills, active effects, and possibly passive abilities. Each turn, the current player receives chakra equal to the number of living characters on their team. Skills can consume fixed chakra types or random chakra, apply damage, healing, stun, damage reduction, invulnerability, chakra removal, delayed effects, or setup effects for future combos.

The goal is to build an agent that can learn through trial and error, without relying on Minimax as a teacher and without manually annotated data.

The recommended approach is reinforcement learning with self-play, action masking, and a fixed vector representation of the game state.

---

## Why Minimax Becomes Expensive

Minimax is expensive because the number of evaluated states grows exponentially with depth.

If the current player has around 40 legal actions:

```text
depth 1 = 40 states
depth 2 = 1,600 states
depth 3 = 64,000 states
depth 4 = 2,560,000 states
```

This is even worse if:

- skills have multiple possible targets;
- random chakra can be paid in multiple ways;
- actions include skill reordering;
- effects last multiple turns;
- the agent needs to simulate many possible futures;
- there are many teams and matchups.

Minimax is useful as a baseline, but it is usually too slow to be the main decision system during large-scale simulations or training.

---

## Why a Supervised MLP Is Not Enough

A simple supervised model like:

```text
MLP(state, action) -> score
```

can work if there is a teacher, such as Minimax or human-labeled data.

However, this has two major problems:

1. It moves the expensive computation to dataset generation.
2. The model only learns to imitate the teacher.

If the teacher is shallow Minimax, the model may also miss deeper strategic ideas such as:

- preparing combos;
- saving chakra;
- delaying a skill until a cooldown comes back;
- protecting a character before an enemy burst;
- using stun at the correct timing;
- using setup skills before high-damage skills.

For this game, the better long-term direction is reinforcement learning through self-play.

---

## Recommended Approach

The recommended first serious approach is:

```text
Gymnasium-style environment
+ PyTorch policy/value network
+ action masking
+ PPO or MaskablePPO
+ self-play
```

The core idea is:

```text
state -> neural network -> action probabilities + state value
```

The model plays games, receives rewards, and improves from the outcomes.

---

## Library Options

### Option 1: Pure PyTorch

Pure PyTorch gives maximum control.

Pros:

- full control over the environment;
- full control over action masking;
- easy to customize self-play;
- easy to debug model inputs and outputs;
- no dependency on RL framework assumptions.

Cons:

- you need to implement the RL algorithm yourself;
- PPO is not trivial to implement correctly;
- batching, rollout storage, advantage estimation, and action masking need care.

Recommended if you want to deeply understand the training loop.

---

### Option 2: Gymnasium + Stable-Baselines3 Contrib MaskablePPO

This is the most practical option.

Use:

```text
gymnasium
stable-baselines3
sb3-contrib
MaskablePPO
PyTorch
```

`MaskablePPO` supports invalid action masking. This matters a lot for this game because many actions are illegal depending on the current state.

Examples of invalid actions:

- using a skill on cooldown;
- using a skill without enough chakra;
- targeting a dead character;
- using Rasengan without Shadow Clones;
- using a stunned character;
- using a skill that requires a condition not currently active.

Pros:

- production-quality PPO implementation;
- built-in rollout collection;
- supports action masks;
- easier to get a first working RL baseline;
- PyTorch-based.

Cons:

- the environment must expose a fixed discrete action space;
- complex multi-discrete actions may need to be flattened into one integer action ID;
- self-play requires extra engineering.

Recommended as the first practical implementation.

---

### Option 3: Ray RLlib

Ray RLlib is powerful for large-scale distributed RL.

Pros:

- good for multi-agent and self-play setups;
- scalable;
- supports more complex training infrastructure.

Cons:

- heavier;
- more complex setup;
- overkill for the first version;
- more moving parts.

Recommended only later, if training becomes too slow or if you want distributed self-play.

---

## Recommended First Stack

Start with:

```text
Python engine already implemented
Gymnasium wrapper around the engine
sb3-contrib MaskablePPO
PyTorch
NumPy
```

The engine should stay independent from RL libraries.

Suggested project structure:

```text
naruto_arena/
  engine/
    state.py
    actions.py
    simulator.py
    characters.py
    skills.py
    effects.py

  rl/
    env.py
    action_space.py
    observation.py
    rewards.py
    masks.py
    self_play.py
    train_maskable_ppo.py
    evaluate.py

  agents/
    random_agent.py
    heuristic_agent.py
    ppo_agent.py
```

---

## Environment Design

The RL environment should wrap the existing engine.

The engine should remain the source of truth.

The environment should provide:

```python
reset() -> observation, info
step(action_id) -> observation, reward, terminated, truncated, info
action_masks() -> np.ndarray
```

The environment should not duplicate game logic. It should only:

1. convert engine state into a vector;
2. convert an integer action ID into an engine action;
3. apply the action using the engine;
4. compute reward;
5. expose legal action masks.

---

## Fixed Action Space

Neural networks need a fixed output size.

However, your game has variable legal actions. The solution is to define a large fixed action space and mask invalid actions.

A practical action format is:

```text
actor_slot
skill_slot
target_slot
chakra_payment_id
```

Where:

```text
actor_slot: 0..2
skill_slot: 0..MAX_SKILLS_PER_CHARACTER-1
target_slot: 0..5
chakra_payment_id: 0..N_PAYMENT_PATTERNS-1
```

Then flatten this tuple into a single integer:

```python
action_id = encode_action(actor_slot, skill_slot, target_slot, chakra_payment_id)
```

The model outputs:

```python
logits: shape [num_total_actions]
```

Then invalid actions are masked.

---

## Why Use Fixed Action IDs Instead of Variable Action Lists

A variable list of actions is natural for the engine, but awkward for most RL libraries.

A fixed action ID system is better because:

- the neural network output size is stable;
- PPO expects a fixed action distribution;
- action masks can remove invalid choices;
- batching is easier;
- evaluation is faster.

The engine can still internally use rich action objects. The RL layer simply maps:

```text
action_id -> engine Action object
```

---

## Chakra Payment Modeling

Chakra payment is important because random chakra choices can affect future turns.

Example:

```text
Skill cost: 1 ninjutsu + 1 random
Available chakra:
- 1 ninjutsu
- 1 taijutsu
- 1 genjutsu
- 0 bloodline
```

The random chakra could be paid with taijutsu or genjutsu. These choices are strategically different.

There are two possible designs.

---

### Design A: Include Chakra Payment in the Action

The action is:

```text
actor + skill + target + chakra_payment
```

Pros:

- the model learns which chakra type to spend;
- it can learn to preserve important chakra for future turns;
- more strategically accurate.

Cons:

- increases action space size;
- more invalid action combinations;
- slower training.

This is recommended if chakra management is central to the game.

---

### Design B: Let the Engine Auto-Pay Random Chakra

The action is:

```text
actor + skill + target
```

The engine chooses how to pay random chakra using a heuristic.

Example heuristic:

```text
spend the most abundant chakra first
preserve fixed chakra types needed by available skills
```

Pros:

- smaller action space;
- easier training;
- faster first prototype.

Cons:

- the agent does not directly learn chakra spending strategy;
- bad auto-payment heuristics may limit performance.

This is recommended for the first prototype.

---

### Recommended Chakra Strategy

Start with Design B.

Once the agent learns basic gameplay, upgrade to Design A.

This staged approach avoids making the first RL problem too hard.

---

## Skill Reordering Action

Your game supports moving skills to different positions to create combo timing.

This is strategically interesting, but it also increases action complexity.

For the first RL version, consider disabling skill reordering or limiting it.

Possible designs:

### Option 1: Disable Reordering for Initial Training

Pros:

- simpler action space;
- easier learning;
- faster debugging.

Cons:

- agent does not learn this mechanic yet.

Recommended for the first PPO prototype.

---

### Option 2: Add Reordering as a Separate Action Type

Add an action format like:

```text
action_type = USE_SKILL or REORDER_SKILL or END_TURN
```

For reorder:

```text
actor_slot
from_skill_slot
to_skill_slot
```

Pros:

- fully models the game;
- allows combo optimization.

Cons:

- significantly larger action space;
- harder exploration;
- may require more training.

Recommended after the basic agent works.

---

### Option 3: Treat Reordering as Pre-Match Deck Configuration

Instead of allowing reordering during the match, train another system to choose skill order before the match.

Pros:

- reduces in-game complexity;
- easier RL problem;
- still allows strategic skill order.

Cons:

- less faithful to the full game if reordering is allowed mid-match.

Good intermediate option.

---

## Observation Vector

The observation vector should represent everything the agent needs to make a decision.

It should not include hidden information unless both players have access to it.

Because the full game state is visible in Naruto Arena-style games, the observation can include both teams.

---

## Observation Layout

A practical observation vector:

```text
global features
current player features
enemy player features
character features for 6 characters
chakra features
cooldown features
active effect features
skill availability features
```

Use normalized numeric values whenever possible.

---

## Global Features

Examples:

```text
turn_number / MAX_TURNS
current_player_id
phase_id
```

If the game always uses alternating turns, `current_player_id` may be enough.

---

## Character Features

For each of the 6 character slots:

```text
hp / 100
is_alive
is_ally
is_current_player_character
is_stunned
is_invulnerable
damage_reduction / 100
unpierceable_damage_reduction / 100
damage_bonus / 100
healing_per_turn / 100
```

Also include character identity.

Two common options:

### Option A: One-Hot Character ID

If there are 20 characters:

```text
character_id_one_hot: length 20
```

Pros:

- simple;
- works well for a fixed roster;
- easy to debug.

Cons:

- does not generalize well to unseen characters.

Recommended for your first version.

---

### Option B: Character Stat/Skill Embeddings

Represent each character by features derived from skills.

Pros:

- better generalization to new characters;
- useful if roster grows a lot.

Cons:

- harder to design;
- more complex.

Recommended later.

---

## Cooldown Features

For each character and skill slot:

```text
cooldown_remaining / MAX_COOLDOWN
is_available
```

Example:

```text
cooldown skill 1
cooldown skill 2
cooldown skill 3
cooldown skill 4
```

If characters may have different numbers of skills, set:

```text
MAX_SKILLS_PER_CHARACTER = 5
```

For missing skills:

```text
exists = 0
cooldown = 0
is_available = 0
```

---

## Active Effect Features

Active effects are tricky because there may be many possible effects.

For the first version, do not encode every individual effect as text or objects.

Instead, aggregate them into numeric features per character:

```text
stun_duration / MAX_DURATION
invulnerability_duration / MAX_DURATION
damage_reduction_amount / 100
damage_reduction_duration / MAX_DURATION
unpierceable_reduction_amount / 100
unpierceable_reduction_duration / MAX_DURATION
damage_over_time_amount / 100
damage_over_time_duration / MAX_DURATION
healing_over_time_amount / 100
healing_over_time_duration / MAX_DURATION
damage_bonus_amount / 100
damage_bonus_duration / MAX_DURATION
```

This makes the vector stable and readable.

Later, if the game becomes more complex, use an effect embedding system.

---

## Chakra Features

For each player:

```text
ninjutsu / MAX_CHAKRA
taijutsu / MAX_CHAKRA
genjutsu / MAX_CHAKRA
bloodline / MAX_CHAKRA
total_chakra / MAX_CHAKRA
```

For both players:

```text
my chakra
enemy chakra
```

Always encode from the perspective of the current player:

```text
my_team
enemy_team
my_chakra
enemy_chakra
```

This reduces the need for the model to learn player symmetry.

---

## Skill Availability Features

For each character and skill slot:

```text
skill_exists
has_enough_chakra
cooldown_ready
requirements_satisfied
has_valid_target
is_usable
```

This helps the model learn faster.

The action mask already prevents invalid skills, but these features help the policy understand why some actions are good or bad.

---

## Action Mask

The action mask is a boolean vector:

```python
mask.shape == (num_total_actions,)
```

Each value means:

```text
True  -> action is legal
False -> action is illegal
```

Invalid actions include:

- actor is dead;
- actor is stunned;
- skill does not exist;
- skill is on cooldown;
- insufficient chakra;
- invalid chakra payment;
- target is dead;
- target is not allowed by the skill;
- skill-specific requirement is not satisfied;
- game is already terminal.

For MaskablePPO, the environment should expose action masks.

---

## Reward Design

The final objective is winning.

The simplest reward is:

```text
+1 for win
-1 for loss
0 otherwise
```

However, this can be too sparse early in training.

Use reward shaping carefully.

Recommended reward:

```text
+1.0 for winning
-1.0 for losing

+0.01 per damage dealt
-0.01 per damage received

+0.20 for killing an enemy
-0.20 for losing an ally

+0.03 for useful healing
+0.03 for applying stun to a target that can act
+0.02 for removing useful enemy chakra
-0.01 for ending turn with too much unused chakra
```

Important:

The win/loss reward should remain the most important signal.

Do not over-reward small actions so much that the agent learns to farm damage or healing instead of winning.

---

## Learning Cooldowns and Preparation

The model can learn cooldowns, setup skills, and delayed combos if the observation includes:

- cooldowns;
- active effect durations;
- skill availability;
- chakra;
- current HP;
- damage modifiers;
- stun/invulnerability states.

PPO learns from future returns.

Example:

```text
Turn 1: use Shadow Clones
Turn 2: use Rasengan
Turn 3: kill enemy
```

The final reward and intermediate rewards propagate backward through the trajectory, so the setup action can become more likely over time.

This is exactly why reinforcement learning is a better fit than a purely supervised model.

---

## Neural Network Architecture

Start simple.

Recommended first network:

```text
Input: observation vector
Shared MLP:
  Linear(obs_dim, 256)
  ReLU
  Linear(256, 256)
  ReLU

Policy head:
  Linear(256, num_actions)

Value head:
  Linear(256, 1)
```

The policy head chooses the action.

The value head estimates how good the current state is.

This is the standard actor-critic setup used by PPO.

---

## Why Not Use a Recurrent Model First

RNNs, LSTMs, and Transformers are not necessary at the start if the observation contains the full current game state.

The game is fully observable if the environment includes:

- HP;
- chakra;
- cooldowns;
- active effects;
- passives;
- buffs/debuffs;
- turn information.

If all relevant information is in the state vector, a feed-forward MLP is enough.

Use recurrent models only if:

- there is hidden information;
- the agent needs to remember previous unseen actions;
- some important state is not included in the observation.

For this game, the first version should use a normal MLP.

---

## Self-Play Curriculum

Do not start with pure self-play only.

A random agent versus itself may generate low-quality games for a long time.

Recommended training stages:

### Stage 1: PPO vs RandomAgent

The model learns basic rules:

- use legal skills;
- deal damage;
- avoid wasting turns;
- kill enemies.

### Stage 2: PPO vs SimpleHeuristicAgent

The model learns to beat a basic strategy:

- focus low-HP enemies;
- use damaging skills efficiently;
- avoid obvious mistakes.

### Stage 3: PPO vs Older Versions of Itself

Save model snapshots.

Train against a mixture of opponents:

```text
30% RandomAgent
30% HeuristicAgent
40% previous PPO checkpoints
```

This prevents the model from overfitting to one opponent.

### Stage 4: Full Self-Play League

Maintain a pool of historical agents.

Sample opponents from this pool during training.

This helps avoid forgetting and improves robustness.

---

## Evaluation

Always evaluate against fixed baselines.

Useful evaluations:

```text
PPO vs RandomAgent
PPO vs HeuristicAgent
PPO vs Minimax depth 1
PPO vs Minimax depth 2
PPO vs previous PPO checkpoint
```

Metrics:

```text
win rate
average turns to win
average damage dealt
average chakra wasted
skill usage distribution
team matchup win rate
```

Do not judge the agent only by training reward.

---

## Practical First Implementation Plan

### Step 1: Create a Gymnasium Environment

Implement:

```python
class NarutoArenaEnv(gym.Env):
    observation_space = gym.spaces.Box(...)
    action_space = gym.spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        ...

    def step(self, action_id):
        ...

    def action_masks(self):
        ...
```

### Step 2: Build Observation Encoder

Implement:

```python
encode_observation(state, current_player) -> np.ndarray
```

Use perspective-based encoding:

```text
current player's team first
enemy team second
```

### Step 3: Build Action Encoder

Implement:

```python
encode_action_tuple(...)
decode_action_id(action_id)
action_id_to_engine_action(state, action_id)
```

### Step 4: Build Legal Action Mask

Implement:

```python
get_action_mask(state, current_player) -> np.ndarray
```

This should call the engine's legal action validator.

### Step 5: Train MaskablePPO

Install:

```bash
pip install gymnasium stable-baselines3 sb3-contrib torch numpy
```

Training script:

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

env = NarutoArenaEnv()

model = MaskablePPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    gamma=0.99,
)

model.learn(total_timesteps=1_000_000)
model.save("models/naruto_maskable_ppo")
```

### Step 6: Evaluate

Run many games against RandomAgent and HeuristicAgent.

```text
If PPO cannot beat RandomAgent, the issue is probably:
- bad observation vector;
- bad action mask;
- reward too sparse;
- bug in environment step;
- invalid action encoding;
- training not long enough.
```

---

## Modeling Decisions Summary

### Use PPO Instead of Minimax

Reason:

Minimax is exponential in action count and depth. PPO learns a reusable policy that makes decisions in one forward pass.

---

### Use Action Masking

Reason:

The game has many invalid actions. Without masking, the model wastes training time trying illegal moves.

---

### Use Fixed Discrete Action IDs

Reason:

RL libraries need a fixed action space. The engine can still use rich action objects internally.

---

### Encode Observations from Current Player Perspective

Reason:

The same strategic pattern should work for both players. This reduces learning complexity.

---

### Start Without Chakra Payment as an Explicit Action

Reason:

Including chakra payment immediately increases action space. Auto-payment is easier for a first prototype.

Later, make chakra payment part of the action to learn deeper resource strategy.

---

### Start Without Skill Reordering

Reason:

Skill reordering increases exploration difficulty. First prove the agent can learn combat. Then add reordering.

---

### Use Aggregated Effect Features

Reason:

A fixed vector is easier to train than a variable list of effect objects.

---

### Use MLP Before RNN/Transformer

Reason:

The full game state is observable. If the observation vector contains cooldowns, effects, chakra, and HP, memory is not required.

---

## Future Improvements

After the first PPO agent works:

1. Add explicit random chakra payment decisions.
2. Add skill reordering actions.
3. Add team selection as a separate draft/meta-game.
4. Add self-play league training.
5. Add opponent sampling from historical checkpoints.
6. Add richer neural architectures:
   - character-wise shared encoders;
   - attention over characters;
   - skill embeddings;
   - graph neural networks.
7. Train separate models for:
   - in-game action selection;
   - team composition;
   - skill ordering.
8. Use population-based training for balance testing.

---

## Final Recommendation

For the current stage, the best practical path is:

```text
1. Keep the engine deterministic and independent.
2. Wrap it as a Gymnasium environment.
3. Use fixed Discrete action IDs.
4. Use action masks.
5. Train with MaskablePPO.
6. Start without explicit chakra payment and without skill reordering.
7. Evaluate against RandomAgent and HeuristicAgent.
8. Add deeper mechanics after the first agent learns to win basic games.
```

This gives a realistic path toward an agent that can learn cooldowns, setups, defensive timing, chakra management, and combos through trial and error.
