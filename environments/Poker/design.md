# Poker Design

## Purpose

This document explains the current poker implementation in PulseLib at a system-design level. It is meant to answer:

- what poker environment(s) exist in the repo
- what kind of agent is currently being trained
- how the training loop is wired
- what each major environment function is responsible for
- how state, action, reward, and termination are represented
- which files define the current poker stack

This is a description of the implementation that currently exists in the codebase. Where useful, it also notes important behavioral caveats that matter when reading results or extending the system.

## High-Level Architecture

There are two poker paths in this repository:

1. A GPU-first vectorized poker environment used for the current large-scale training setup.
2. A legacy CPU poker environment and tabular Q-learning path that appears to be an earlier implementation.

The primary training flow today is:

1. Load `config/pokerGPU.yaml`.
2. Instantiate a table of non-learning opponent agents.
3. Create a neural-network Q-learning agent (`PokerQNetwork`).
4. Insert that Q-agent into the seat list.
5. Build the `Pulse-Poker-GPU-v1` Gymnasium environment.
6. Run many vectorized games in parallel.
7. Use online temporal-difference updates to train the Q-network from environment transitions.

Conceptually, the GPU system is trying to do deep Q-learning over a poker state representation with:

- fixed discrete action space
- vectorized batched environment stepping
- dense reward shaping based on equity, pot size, and pot-odds style incentives
- rotating seat positions so the Q-agent does not always sit in the same place

## Main Files

The poker implementation is spread across a few core files:

- `environments/Poker/PokerGPU.py`
  - Main vectorized GPU environment.
- `scripts/Poker/trainGPU.py`
  - Main GPU training entrypoint.
- `environments/Poker/Player.py`
  - Player base class, heuristic opponents, and the neural Q-network.
- `environments/Poker/utils.py`
  - Agent loading, action routing, card helpers, and legacy reward helpers.
- `config/pokerGPU.yaml`
  - Hyperparameters and environment/training settings for the GPU path.

There is also a legacy CPU path:

- `environments/Poker/Poker.py`
- `scripts/Poker/train.py`
- `agents/TemperalDifference/PokerQLearning.py`
- `config/poker.yaml`

## Environment Registration

The poker environments are registered in `environments/__init__.py`.

- `Pulse-Poker-v1`
  - Entry point for the CPU environment.
- `Pulse-Poker-GPU-v1`
  - Entry point for the GPU environment.

The GPU training script uses `Pulse-Poker-GPU-v1`.

## Current Training Strategy

### Agent Type

The main learning agent is `PokerQNetwork` from `environments/Poker/Player.py`.

This is not a tabular Q-table. It is a neural-network function approximator that predicts Q-values for the discrete poker action space. In practice, it behaves like a DQN-style online learner with:

- a policy network
- a target network
- epsilon-greedy exploration
- Bellman-style bootstrap targets
- AdamW optimization

The opponents are not learning in the current GPU path. They are scripted agents with simple strategic styles:

- `RandomPlayer`
- `HeuristicHandsPlayerGPU`
- `TightAggressivePlayerGPU`
- `LoosePassivePlayerGPU`
- `SmallBallPlayerGPU`

### Why this setup exists

The design in the notes suggests the intent was:

- start the Q-agent against simpler opponents
- use those opponents to teach basic poker behavior
- rotate seats to avoid positional overfitting
- shape reward using poker-specific signals rather than only end-of-hand chip deltas
- eventually scale into stronger opponents or self-play

### Training Loop Behavior

The active GPU trainer is `scripts/Poker/trainGPU.py`.

At a high level, `train_agent(...)` does this per training episode:

1. Rotate the seat ordering of agents.
2. Reset the vectorized environment.
3. Record the Q-agent's initial stack in every parallel game.
4. Repeatedly:
   - build one action per parallel game
   - step the environment
   - filter the transitions where the acting seat was the Q-agent
   - run `PokerQNetwork.train_step(...)` on those transitions
5. Aggregate per-episode reward and chip-profit metrics.
6. After all episodes:
   - save network weights
   - plot reward and chip curves
   - write benchmark metadata

### Online Learning Style

This is online training rather than replay-buffer training.

That means:

- transitions are consumed immediately after they happen
- there is no experience replay layer between acting and learning
- the network learns from the live stream of transitions in the current batch

This keeps the implementation simpler, but it also means training quality depends heavily on transition correctness and reward correctness.

## Configuration: `config/pokerGPU.yaml`

The GPU config controls:

- result directory name
- environment id
- opponent composition
- number of vectorized games
- number of episodes
- starting stack size in big blinds
- number of opponent seats
- state/action dimensions expected by the Q-network
- reward-shaping constants
- target-network update frequency
- discount factor
- optimizer hyperparameters

Key values in the current config:

- `N_GAMES: 2000000`
  - extremely large batch count
- `EPISODES: 1000`
- `STARTING_BBS: 100`
- `NUM_PLAYERS: 9`
  - this is the number of non-Q opponents in the GPU script before the Q-agent is inserted
- `STATE_SPACE: 40`
- `ACTION_SPACE: 13`

The script then adds the Q-agent as an extra seat and builds the environment with `n_players = NUM_PLAYERS + 1`.

## State Representation

The GPU environment uses a flat tensor observation.

### Observation layout

The environment computes:

- first 13 values for hero/global information
- then `3 * (max_players - 1)` values for opponents

The layout used in `PokerGPU.get_obs()` is:

1. `0:5`
   - board cards
2. `5:7`
   - current player's hole cards
3. `7`
   - stage
4. `8`
   - position relative to button
5. `9`
   - pot size
6. `10`
   - amount required to call
7. `11`
   - current player's stack
8. `12`
   - current player's status
9. remaining slots in groups of 3
   - opponent stack
   - opponent status
   - opponent current-round bet

### Interpretation

The observation is always from the point of view of the seat whose turn it currently is.

That matters because:

- the same raw table state will look different depending on whose turn it is
- the training script must be very precise about which seat generated which transition
- rewards must align with the same actor perspective

### Card encoding

Cards are stored as integers. `environments/Poker/utils.py` provides:

- `encode_card(card)`
- `decode_card(card_int)`

The encoding scheme is rank plus suit offset.

## Action Space

The action space is discrete with 13 actions.

The current intended mapping is:

1. `0`
   - fold
2. `1`
   - check/call
3. `2`
   - min-raise
4. `3`
   - 25% pot raise
5. `4`
   - 33% pot raise
6. `5`
   - 50% pot raise
7. `6`
   - 75% pot raise
8. `7`
   - 100% pot raise
9. `8`
   - 150% pot raise
10. `9`
   - 200% pot raise
11. `10`
   - 300% pot raise
12. `11`
   - 400% pot raise
13. `12`
   - all-in

This is a deliberately discretized betting model. The design choice trades off realism for tractability:

- it avoids continuous action selection
- it makes Q-learning straightforward
- it reduces the search space
- it limits betting precision compared with real poker

## Reward Design

### Intended philosophy

The note files show the reward was designed to combine:

- hand quality / equity
- pot size
- pot odds
- strategic correctness of folding, calling, or raising
- some notion of money outcome without using only final chip result

The general goal is to avoid a reward function that only says "did you win the hand?" because that would be too sparse and too noisy.

### CPU-side helper

`environments/Poker/utils.py` contains a scalar `poker_reward(...)` helper used by the legacy CPU environment.

It combines:

- `m`
  - money-style term based on equity, pot, investment, and stack change
- `o`
  - pot-odds style term
- `s`
  - strategic term based on action type

Then it applies `tanh` scaling.

### GPU-side reward

The GPU environment uses `PokerGPU.poker_reward_gpu(...)`.

It computes:

- `active_counts`
  - number of active/all-in players
- `fair_shares`
  - equal-share baseline of the pot
- `call_costs`
  - amount needed to call
- `e`
  - current player's equity
- `m`
  - equity times pot
- `o`
  - pot-odds style ratio
- `s`
  - action-conditioned strategic term:
    - call/check
    - fold
    - raise

Then it returns a scaled `tanh` reward.

### Important design implication

Because the reward is dense and shaped, the quality of training is very sensitive to:

- correct actor indexing
- correct equity indexing
- correct stage semantics
- correct call-cost semantics

This implementation is much more reward-engineered than a simple end-of-episode chip-delta system.

## Equity Calculation

The GPU environment uses `HandRanks.dat` for fast evaluator-style lookup.

`PokerGPU.calculate_equities()` computes equities differently depending on street:

- preflop
  - fixed `0.5`
- flop
  - evaluator-based transformation of 5-card board-plus-hand state
- turn
  - evaluator-based transformation of 6-card state
- river
  - evaluator-based transformation of 7-card state

The notes make it clear this is an approximation-driven design choice. The implementation is trying to get a practical strength/equity proxy that is fast enough for large-scale training rather than perfect combinatorial equity against every exact opponent hand distribution.

## `PokerGPU` Class Walkthrough

This is the most important file for understanding the live poker system.

### `__init__(...)`

Responsibilities:

- store environment-wide constants
- store training/game parameters
- define action and observation spaces
- load `HandRanks.dat`
- allocate some reusable tensors and constants

Design intent:

- minimize Python overhead
- keep repeated operations inside tensor code
- support a large number of parallel games

### `set_agents(agents)`

Simple setter for replacing the agent list attached to the environment.

This exists so the environment can conceptually be reused with different seat lists, although the active training script currently relies more on external action routing than on calling this each episode.

### `reset(seed=None, options=None, rotation=0)`

Responsibilities:

- choose number of active players if variable-player mode is enabled
- record the Q-agent seat target
- build fresh decks and reset hand state
- refill or clamp stacks when necessary
- optionally rotate stack positions
- deal hole cards
- initialize per-hand betting state
- initialize button, blinds, aggressor, current index, and buffers
- return first observation and info dictionary

Important semantics:

- `options['q_agent_seat']`
  - intended to tell the environment which seat the learning agent should occupy in the current episode
- `options['rotation']`
  - intended to rotate seat-related state
- `options['active_players']`
  - allows randomizing table size

### `get_obs()`

Builds the flat observation tensor for every parallel game from the perspective of the current seat to act.

This function is the translation layer from internal environment state to RL state.

### `get_info()`

Returns auxiliary data used by the training script:

- active player count
- full stack tensor
- current acting seat per game

### `post_blinds()`

Applies the big blind posting logic to the current hand.

Current implementation detail:

- small blind is effectively treated as zero
- big blind is one unit from `bb_amounts`

### `deal_players_cards(n_cards)`

Deals contiguous card slices from each vectorized deck into player hands during preflop setup.

### `deal_cards(g, n_cards)`

Deals board cards for a specific subset of games.

This is used during street transitions and when resolving unfinished all-in/showdown paths.

### `execute_actions(actions)`

This is the main betting-state mutation function.

Responsibilities:

- compute call costs
- determine which current seats are eligible to act
- apply fold logic
- apply check/call logic
- apply raise logic
- update:
  - stacks
  - current-round bets
  - total invested
  - pot
  - player status
  - highest current bet
  - aggressor
  - acted counters
  - last raise size

This function is effectively the betting engine for the vectorized environment.

### `poker_reward_gpu(actions)`

Computes the reward tensor for the current batch of acting seats.

This is the main reward-shaping function for the GPU system.

### `resolve_fold_winners()`

Pays the pot immediately for games where only one non-folded player remains.

This is the fold-win fast path.

### `resolve_terminated_games()`

Handles hands that are done but still need board completion and showdown resolution.

Responsibilities:

- advance incomplete boards to the river when needed
- evaluate all remaining hands
- determine winners
- distribute pot

This is the main showdown-resolution path for the GPU environment.

### `calculate_equities()`

Computes or approximates equity/strength values for the active stage of each parallel game.

This supports reward shaping rather than direct action legality.

### `step(actions)`

This is the main environment transition function.

Responsibilities:

1. snapshot pre-action stack/bet state
2. calculate current equities
3. execute the chosen actions
4. find the next player to act in each game
5. detect round completion
6. handle street transitions
7. resolve completed games
8. clear finished-hand round state
9. compute rewards
10. return next observation, reward, done flags, truncation flags, and info

In design terms, `step()` is orchestrating all table-state progression. It does not just "apply an action"; it advances the entire batched poker state machine.

## `Player.py` Walkthrough

This file contains both agent definitions and the neural Q-network.

### `Player`

Abstract base class for poker players.

Fields:

- `id`
- `stack`
- `current_round_bet`
- `total_invested`
- `status`
- `hand`

This class mainly exists to give the CPU path and the opponent implementations a common interface.

### `RandomPlayer`

Takes a random action from the discrete action space.

Used as a weak baseline opponent.

### `HeuristicPlayer`

CPU-style heuristic player.

Uses:

- hole cards
- board
- simple hand-strength logic
- pot odds

This is a lightweight scripted baseline rather than a sophisticated poker bot.

### `HeuristicHandsPlayerGPU`

GPU batched opponent that mostly reacts to hole-card strength.

Simple behavior:

- fold weak low-rank combos
- raise with pairs/high-card strength

### `TightAggressivePlayerGPU`

Designed to play fewer hands and use stronger raises when it does continue.

### `LoosePassivePlayerGPU`

Designed to continue more often and raise less often.

### `SmallBallPlayerGPU`

Designed to prefer smaller aggressive actions and be more pot-sensitive.

### `PokerQNetwork`

This is the current learning agent.

Core components:

- feedforward MLP
- target network copied from the main network
- epsilon-greedy `get_actions(...)`
- Bellman target in `train_step(...)`
- AdamW optimizer

#### `__init__(...)`

Responsibilities:

- store RL hyperparameters
- build the policy network
- optionally load weights from disk
- clone the target network
- configure loss and optimizer

#### `forward(states)`

Returns one Q-value per action.

#### `get_actions(states)`

Performs epsilon-greedy action selection:

- with probability epsilon:
  - pick random discrete action
- otherwise:
  - choose `argmax(Q(s, a))`

It also decays epsilon over time.

#### `train_step(states, actions, rewards, next_states, dones)`

Performs the actual learning update.

Flow:

1. filter states considered valid for training
2. compute current `Q(s, a)`
3. compute bootstrap target:
   - `r + gamma * max_a' Q_target(s', a')`
4. compute MSE loss
5. backpropagate
6. clip gradients
7. optimizer step
8. periodically sync target network

This is the single most important learning function in the current system.

## `utils.py` Walkthrough

This file mixes helper logic for both poker paths.

### `calculate_equity(...)`

Legacy CPU helper using `eval7.py_hand_vs_range_monte_carlo(...)`.

This is used by the CPU environment, not by the main GPU environment.

### `decode_card(...)` / `encode_card(...)`

Card conversion helpers between integer state encoding and `eval7.Card`.

### `poker_reward(...)`

Legacy scalar reward helper used in the CPU path.

### `PokerAgentType`

Enum used to identify agent categories in the training scripts and action-routing logic.

### `load_agents(...)`

Legacy CPU agent loader.

Returns:

- player list
- agent-type list

### `build_actions(...)`

Given:

- state tensor
- current acting seat per game
- agent list
- agent type list

This function routes each active game to the correct agent implementation and fills the action tensor for the full batch.

This is the bridge between:

- environment seat state
- actual agent policy execution

### `load_gpu_agents(...)`

Creates the opponent roster for the GPU path.

This function currently loads only the non-learning scripted opponents. The Q-agent is created separately in `trainGPU.py` and inserted afterward.

### `debug_state(...)`

Human-readable state printer for debugging.

### `get_rotated_agents(...)`

Rotates agent order so the Q-agent can occupy different seats across episodes.

The design goal is to avoid hard-coding the learner to one absolute seat position.

## `trainGPU.py` Walkthrough

This is the current top-level training script.

### Module constants

Defines:

- config filename
- output filenames
- action-space width
- environment name

### `train_agent(...)`

This is the full training loop.

Responsibilities:

- run training episodes
- rotate agent seating
- build per-game actions
- advance the vectorized environment
- train the Q-network on Q-agent turns
- record reward/profit metrics
- save model weights and output artifacts

This function is the operational "trainer" for the current poker system.

### `__main__`

Bootstraps the whole run:

1. read config
2. create results directory
3. load device
4. load opponent agents
5. create the Q-network
6. insert Q-agent into the table
7. create Gym environment
8. profile and run training

## Legacy CPU Path

The CPU poker path appears to be the older design.

### `Poker.py`

Non-vectorized poker environment with:

- Python object-based players
- scalar step transitions
- `eval7` equity helper
- more explicit side-pot resolution logic at showdown

It is easier to read conceptually than the GPU version, but much slower and more Python-heavy.

### `PokerQLearning.py`

Legacy tabular Q-learning player.

Characteristics:

- state packed into a byte-string key
- Q-table stored in a Python dictionary
- epsilon-greedy action selection
- tabular Bellman update

This is a traditional Q-learning design rather than neural approximation.

### Why it still matters

Even though the GPU path is the active system, the CPU path still provides useful context:

- it documents earlier intended poker semantics
- it shows a side-pot-aware showdown implementation
- it shows what the original tabular Q-learning design looked like

## Design Goals Visible In The Note Files

The note files in `rl/notes.txt` and `rl/depth_notes.txt` show the reasoning behind the implementation.

Recurring goals include:

- avoid sparse reward
- keep the environment fast enough to scale
- batch many games in parallel
- use discrete betting to control state/action explosion
- rotate seats so the learner does not memorize one seat
- start against heuristic opponents before harder training setups
- treat poker as a partially observed decision problem where reward must still reflect hidden-information quality indirectly

The notes also show strong concern about:

- long or stalled games
- invalid actions
- chip-profit tracking versus reward tracking
- stack reset behavior
- equity-calculation fidelity

Those themes are reflected directly in the current code.

## Current Implementation Caveats

For context, the present implementation has several important behaviors that anyone extending or benchmarking the system should understand.

### 1. The GPU path is the real training path

If you want to understand current training results, read:

- `scripts/Poker/trainGPU.py`
- `environments/Poker/PokerGPU.py`
- `environments/Poker/Player.py`

The CPU path is useful context, but it is not the primary system.

### 2. Reward correctness is central

This trainer depends on dense shaped reward. That means small reward-indexing mistakes can have very large downstream effects on learning quality.

### 3. Showdown correctness matters for chip metrics

Because bankroll/profit are tracked, pot-distribution correctness is not just a game-rule detail. It directly affects reported performance.

### 4. Seat-rotation logic is part of the design

The system is explicitly trying to prevent positional overfitting by moving the learning agent around the table across episodes.

### 5. The implementation mixes "intended design" and "current behavior"

Some parts of the system clearly reflect design intent from the notes, while other parts reflect pragmatic implementation shortcuts needed for GPU batching and speed.

Anyone modifying the poker stack should read both:

- the live code
- the note files

before making structural changes.

## Recommended Reading Order For Future Work

If someone new needs to understand the poker system quickly, the best order is:

1. `config/pokerGPU.yaml`
2. `scripts/Poker/trainGPU.py`
3. `environments/Poker/PokerGPU.py`
4. `environments/Poker/Player.py`
5. `environments/Poker/utils.py`
6. `rl/depth_notes.txt`
7. `environments/Poker/Poker.py`
8. `agents/TemperalDifference/PokerQLearning.py`

This order moves from:

- what is configured
- to how training is launched
- to how the environment behaves
- to how the agent learns
- to historical context

## Files for this design.md file
C:\Users\422mi\Pulselib\config\pokerGPU.yaml
C:\Users\422mi\Pulselib\config\poker.yaml
C:\Users\422mi\Pulselib\scripts\Poker\trainGPU.py
C:\Users\422mi\Pulselib\scripts\Poker\train.py
C:\Users\422mi\Pulselib\environments\Poker\PokerGPU.py
C:\Users\422mi\Pulselib\environments\Poker\Poker.py
C:\Users\422mi\Pulselib\environments\Poker\Player.py
C:\Users\422mi\Pulselib\environments\Poker\utils.py
C:\Users\422mi\Pulselib\environments\__init__.py
C:\Users\422mi\Pulselib\agents\TemperalDifference\PokerQLearning.py
C:\Users\422mi\Pulselib\rl\notes.txt
C:\Users\422mi\Pulselib\rl\depth_notes.txt
