# 2048: environment and state design

This section implements human play and an inspectable environment shared with a three-agent learning experiment. See [the learning protocol](learning.md) for tabular Q-learning, dense DQN, CNN-DQN, recorded training animations, and evaluation results.

Open `index.html` in this directory directly in a browser, or follow **2048 Lab** from either existing dashboard. No installation, network connection, or build step is required.

## Game contract

- Board: 4 × 4, row-major order. Internally a cell is an integer exponent: 0 means empty, 1 means 2, 2 means 4, etc.
- Reset: two tiles at distinct randomly chosen empty cells.
- Actions: 0 = up, 1 = right, 2 = down, 3 = left.
- Transition: slide toward the chosen edge; merge equal neighboring tiles once per move, processing from that edge; compact again. A newly merged tile cannot merge again during the same move.
- Example: moving `[2, 2, 2, 2]` left produces `[4, 4, 0, 0]`, earning 8 points. Moving `[2, 2, 4, 0]` left produces `[4, 4, 0, 0]`, earning 4, not 8.
- Spawn: after a changed board, choose one empty cell uniformly and place 2 with probability 0.9 or 4 with probability 0.1. Reset uses this same distribution twice.
- Reward: sum of the values of tiles created by merging. Sliding without merging gives 0. Score is accumulated reward.
- Invalid move: no board change, spawn, score increment, move increment, or random-number draw. It returns reward 0. The UI disables illegal direction buttons, while keyboard input can demonstrate a no-op.
- Terminal: no action can change the board. A full board is not terminal if equal neighbors can merge.
- Reaching 2048 is a milestone, not a terminal transition; the game continues. Human play has no move cap. The experiment caps games at 2,000 moves and records truncations separately from game-over transitions.
- All agents select only legal actions. Stage 1 learning uses raw merge reward divided by 128 and discount 0.99. Stage 2 adds a potential-based corner-and-snake term without changing game transitions, observations, or original game scores. See [stage2.md](stage2.md).

## State choices

| Representation | Size | Preserves full board? | Tradeoff |
|---|---:|---|---|
| Tile values | 16 integers | Yes | Easiest to read; large numerical range |
| Tile exponents | 16 numbers, divided by 16 for the agents | Yes | Dense DQN uses a vector; CNN-DQN uses one 4 × 4 channel |
| Largest tile + empty cells | 2 integers | No | Actual Q-table key; no tile sum or action mask in the key |
| Extended summary | 3 statistics and legal-action list | No | Inspection only; not used by the agents |

No agent uses one-hot encoding. The earlier `encode(board, 'onehot')` utility remains available for compatibility, with an explicit rank 0–17 vocabulary and rejection of out-of-range custom boards; it is not an input to training or evaluation.

The complete board is a Markov state under the intended independent stochastic spawning model: the next-board distribution and immediate reward depend on the board and action. Score and move count do not change the transition rules. The summary is not an equivalent Markov state because it discards spatial information. Even adding the legal-action list does not recover that information.

Seed and pseudorandom-generator state are reproducibility metadata, not proposed model inputs. This simulator uses a deterministic seeded pseudorandom stream to approximate independent random spawning. Repeating a seed reproduces a game only if the action sequence also matches. No state encoding reveals future spawn draws.

## Environment API

`engine.js` exposes `Lab2048` in the browser and a CommonJS export in Node.

```js
const {Game, legal, encode} = require('./engine.js');
const game = new Game(42);
const observation = encode(game.board, 'exponents');
const actions = legal(game.board);
const transition = game.step(actions[0]);
// transition: board, reward, changed, spawned, done, reached2048
// game: board, score, moves
```

Returned observations and transition boards are copies. `slide(board, action)` is a pure deterministic merge operation with no spawn; use it to test rules. `Game.step(action)` performs the complete transition. Actions outside 0–3 are rejected. The Python environment in `src/game2048.py` matches the browser's transitions and pseudorandom generator; cross-language tests compare complete seeded games.

Run verification from the repository root:

```text
node --test tests/2048_environment.test.cjs
```

The browser still runs without dependencies. Reproducing training requires the project's Python dependencies, including PyTorch. Training and evaluation settings are in [the learning protocol](learning.md).
