> Saved model weights and standalone Q-table files are excluded from Git. Training code, final reports and browser replay data are included; regenerate models locally using the commands below.

# 2048: compact Q-learning, dense DQN, and CNN-DQN

The experiment follows the selected representations: **largest tile + empty-cell count** for the Q-table, and **tile exponents without one-hot encoding** for both neural agents. The environment rules are specified in [environment.md](environment.md).

This document describes the preserved **Stage 1** baseline. **Stage 2** retrains the same three designs from scratch using a corner-and-snake potential reward; see [stage2.md](stage2.md) for its exact formula and matched results. The dashboard's experiment-stage selector switches all replays, curves, final results, and learned table contents. A separate cross-stage table always shows both stages.

## What the dashboard does

The page works offline. It plays recorded trajectories from actual training and frozen-policy evaluation, with per-move boards, scores, selected actions, and Q-values. It does not train in the browser.

- Select one agent or watch all three synchronized by move index. Finished games remain visible while others finish.
- Choose checkpoints at episodes 0, 50, 100, 200, or 300. The learning curve reveals evaluation means through that checkpoint.
- **Frozen evaluation** uses no exploration and no updates. Choose one of the first three fixed evaluation seeds; these were selected in advance, not for performance.
- **Training episode** shows the actual checkpoint episode, including exploration and updates. At checkpoint 0, it shows untrained evaluation because no training episode has occurred.
- **Animate all checkpoints** advances through their games in order. Pause, speed, single-step and scrubbing are available.
- The Q-table displays the numeric update for the recorded transition. Neural losses belong to the latest replay minibatch and need not describe the displayed transition.
- Moves animate in order: each original tile slides, matching pairs merge with a pulse, and the new tile appears last. Direction captions and a NEW badge distinguish movement from spawning. Normal playback takes 0.9 seconds per move, with 1.5-second and 0.45-second options. Manual play animates each move in 0.48 seconds. Pause finishes the current move; scrubbing and checkpoint changes show snapshots immediately. Reduced-motion preferences bypass tile movement.

## Inputs and architectures

### Tabular Q-learning

The key is `(largest_tile_value, empty_cell_count)`. It stores four Q-values per observed key. The final run visited 85 keys, storing 340 Q-values. The environment supplies legal-action masks separately; masks and other statistics do not form part of the key.

This is state aggregation: different boards share a row. The observation is not Markov, and standard tabular convergence guarantees for fully observed MDPs do not apply. The compact table is a deliberate information-limited baseline.

### Dense DQN

Each cell is encoded as `log2(tile)/16`, with 0 for empty. The ordered 16-cell vector passes through:

`Linear(16,64) → ReLU → Linear(64,64) → ReLU → Linear(64,4)`

There are **5,508** trained weights and biases. There is no clipping or one-hot expansion.

### CNN-DQN

The same 16 values form a `(1,4,4)` grid:

`Conv2d(1,16,2) → ReLU → Conv2d(16,32,2) → ReLU → Flatten(128) → Linear(128,64) → ReLU → Linear(64,4)`

Convolutions have stride 1 and no padding. Spatial sizes are 4 × 4 → 3 × 3 → 2 × 2. There is no pooling. There are **10,676** trained weights and biases. The CNN uses exactly **one channel**, not one channel per tile value.

## Learning rules

For all agents, learning reward is raw merge points divided by 128. This constant scaling preserves the ordering of expected discounted returns; it does not add empty-cell, survival or terminal bonuses. Scores are reported in original game points.

The nonterminal target is `y = reward/128 + 0.99 * max_legal_a Q(next_state,a)`. Terminal targets omit the bootstrap term. Invalid actions are excluded from both action selection and bootstrap maxima.

Tabular updates use `Q ← Q + 0.1 * (y − Q)`.

Both DQNs use standard target-network DQN, not Double DQN:

- Adam, learning rate 0.0005; Huber loss (delta 1).
- Replay capacity 10,000 transitions, sampled uniformly with replacement.
- Batch size 64; warmup 256 transitions; one update every four transitions.
- Target network copied from online weights at initialization and every 250 optimizer updates; target parameters receive no gradients.
- Gradient norm clipped to 10.
- Input division by 16 is fixed, with no fitted normalization statistics.

Training exploration falls linearly from epsilon 1 to 0.05 over 300 episodes. Exploratory actions are sampled uniformly from legal actions; greedy ties are broken with a separate seeded random generator. There is no heuristic player, lookahead search, demonstration data, or pretrained network.

## Experiment and evaluation

The saved run uses training seed 42 and 300 episodes per agent. PyTorch runs on one CPU thread with deterministic algorithms. The report records library versions.

Training game seed for episode `e` is `42 + 1000 + e`; action randomness uses `42 + 200000 + e`. Networks and replay have independently initialized random streams. The Python game's Mulberry32 random stream matches the browser environment exactly, including tile spawning.

Checkpoint evaluation uses 20 games with seeds 8,000,000–8,000,019. These repeated games describe learning progress. Final evaluation uses 100 fresh seeds 9,000,000–9,000,099, after all training, without selecting the best checkpoint. Final policy parameters are frozen; evaluation never fills replay, updates weights, or creates table rows. A random-legal-action baseline uses the same evaluation seeds.

Identical seed and action sequences reproduce identical games. Matching seeds across different policies do not force the same later tile positions, because each policy creates different sets of empty cells.

The experiment caps each game at 2,000 valid moves. A cutoff is reported as truncation, not termination, and the last training target still bootstraps. There were no training or final-evaluation truncations in the saved run. Human play has no move cap.

Equal episode counts do not imply equal training samples or compute. This run used 35,879 transitions for Q-learning, 36,736 for dense DQN and 35,164 for CNN-DQN. Training time excludes checkpoint/final evaluation, imports, and artifact export.

## Saved results

| Agent | Final mean score | Across-game SD | Largest tile observed | Games reaching 2048 |
|---|---:|---:|---:|---:|
| Compact Q-learning | 1,011.96 | 508.19 | 256 | 0 / 100 |
| Dense DQN | 1,494.56 | 975.21 | 512 | 0 / 100 |
| CNN-DQN | 1,679.04 | 880.31 | 512 | 0 / 100 |
| Random legal actions | 1,071.68 | 522.87 | 256 | 0 / 100 |

CNN-DQN has the highest observed final mean in this run; the compact Q-table is below the random baseline. None solved 2048 in final evaluation. The dense network's checkpoint evaluation peaked at episode 200 and declined by episode 300: learning is not monotonic.

These are results from one training initialization and a short, fixed budget with no hyperparameter search. Game-score SD measures variation across games, not robustness across training seeds. The networks differ in capacity, and the Q-table loses information that the networks retain. This comparison does not isolate architecture or establish general superiority or statistical significance.

## Reproduction and files

Run from the repository root with the existing Python dependencies installed:

```text
python -m src.train_2048 --episodes 300
node --test tests/2048_environment.test.cjs tests/2048_results.test.cjs
python -m unittest tests.test_2048 -v
```

The trainer accepts `--seed`, `--eval-games`, `--final-games`, `--output` and `--dashboard`. Changing the run regenerates the dashboard data; the narrative numbers in this document and README describe the bundled default run and must be updated if replacing it.

- `src/game2048.py`: Python environment, cross-tested with the browser.
- `src/train_2048.py`: agents, learning, evaluation, recording and export.
- `results/2048/report.json`: configuration, history, checkpoint traces, final results and table.
- `results/2048/q_table.json`: final aggregated Q-table.
- `results/2048/dqn.pt` and `cnn.pt`: final network weights plus configuration and provenance. These are inference checkpoints, not full training-resume snapshots; optimizer and replay are not saved.
- `docs/2048/data.js`: browser copy of the report, allowing direct offline opening.

The trainer is a finite command, not a server. Any server used to preview the page must be stopped at task completion, as required by the repository's `AGENTS.md`.

## Replay comparison controls

Selecting one model shows synchronized Stage 1 (original reward) and Stage 2 (snake reward) boards. Checkpoint, evaluation seed, playback speed, stepping, restart and scrubbing apply to both; finished boards hold their last frame. The experiment-stage control still selects the summary tables and curves.

The default checkpoint is Fully trained (300 episodes), with frozen evaluation and Best recorded score. This chooses the shared seed with the highest individual final score among the visible policies and the three recorded checkpoint seeds; it is a highlight, not an unbiased performance estimate. Selecting a numbered seed overrides this choice. Untrained (0 episodes) shows initial policies before learning.


## Overall score and consistency

The middle-of-page overview ranks all 15 trained model/strategy combinations using final evaluation games. Balanced score = mean minus a user-selected weight times score standard deviation (default 1, adjustable 0 to 2). The overview also identifies the highest mean and smallest SD/mean ratio. These are descriptive preferences over game-to-game variation, not confidence bounds or evidence of stability across training seeds. The overview is independent of replay selections.

## New Stage 3: afterstate CNN

The approved replacement evaluates each pre-spawn move preview with a shared scalar-output CNN. Its final mean is 2,051.48 (SD 1,281.01), versus Stage 2 CNN 1,741.76. The single-seed comparison uses the same 300-episode budget; no evaluation game reached 2048. Stages 3–4 have a separate CNN-only replay with an expandable decision-preview inspector. Q and dense DQN remain Stage 2 references, not new Stage 3 runs. See [afterstate protocol](afterstate.md).

## Stage 4: hybrid afterstate CNN

The CNN grid branch is joined with a separate seven-statistic branch. Final mean: 5,262.88 versus Stage 3 2,051.48, on 100 matched games after 300 training episodes. Highest tile 1024; no 2048 wins. Final results now separate Stages 1–2 (three models each) from Stages 3–4 (CNN only). The two independent replay sections separate Stages 1–2 from Stages 3–4, and the Stage 4 preview inspector lists the actual scaled feature inputs. Run `python -m src.train_2048_hybrid`; see [Stage 4 protocol](hybrid.md).

### Independent experiment sections

The page now groups the complete experience, not just the final tables. Stages 1–2 has its own three-model replay, stage selector, checkpoints, controls, curves and six-row final results. Stages 3–4 has a separate CNN-only replay, stage selector, checkpoints, controls, curves and two-row final results. Playback and selections do not affect the other section. Single-model comparison in the first section never includes Stages 3–4. The overall ranking remains a separate cross-experiment summary between the groups.
