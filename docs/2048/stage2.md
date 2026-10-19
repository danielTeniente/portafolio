> Saved model weights and standalone Q-table files are excluded from Git. Training code, final reports and browser replay data are included; regenerate models locally using the commands below.

# Stage 2: teach the corner-and-snake strategy through reward shaping

Stage 1 remains unchanged. Stage 2 trains all three agents **from scratch**, using the same seed, initial network weights, episode budgets, exploration schedule, architectures, and evaluation game seeds. It is a second experimental condition, not additional training of Stage 1 checkpoints.

The only intended training-design change is the reward. The Q-table still uses exactly `(largest tile, empty-cell count)`. Both networks still use scaled tile exponents without one-hot encoding. No heuristic action selector or demonstrations are added.

## Exact board potential

We use a fixed top-left anchor and the user's column-first snake path. Row-major cell indices, beginning at zero:

```text
[0, 4, 8, 12, 13, 9, 5, 1, 2, 6, 10, 14, 15, 11, 7, 3]
```

The preferred descending-value order is:

```text
 1   8   9  16
 2   7  10  15
 3   6  11  14
 4   5  12  13
```

Numbers above are path positions, not tile values. The largest tile should be at position 1, followed by progressively smaller values down the first column, up the second, and so on. The anchor never changes to another corner during the game.

For an exponent board `b`, compute:

- **Corner score C:** `b[0] / max(b)`. This is a graded preference, not simply a binary largest-in-corner flag.
- **Snake score S:** `1 − sum(max(0, next_rank − current_rank)) / sum(b)` along the 15 consecutive path edges. Empty cells have rank zero, so a gap before a larger tile counts as a violation. This favors descending order but does not uniquely identify every ideal arrangement.
- **Empty-space score E:** `number_of_empty_cells / 16`.
- **Adjacent-pair score M:** count equal, nonempty horizontal or vertical neighbor pairs, divided by 24. Each pair is counted once. This is a local merge-opportunity proxy, not a search or a count of all possible merges after sliding.

Then `Φ(b) = 0.4 C + 0.3 S + 0.2 E + 0.1 M`. For an all-empty board, define Φ as zero. Each component lies in [0,1], so Φ is bounded in [0,1]. The coefficients and shaping strength were set before training and were not tuned after inspecting scores.

## Learning reward

```text
Stage 1: R = raw merge points / 128
Stage 2: R = raw merge points / 128 + λ [γ Φ(next board) − Φ(current board)]
λ = 1
γ = 0.99
```

The next board includes the random tile spawn. At a true game-over transition, the next-state potential is **zero**, regardless of the physical terminal board. A 2,000-move time-limit cutoff is a truncation, not a terminal: its potential and bootstrap are retained.

This is potential-based shaping, not a positive bonus paid repeatedly for staying in a favored layout. A bonus can be negative even when the move earns merge points. Over a terminated trajectory, discounted shaping terms telescope to a start-state constant. That property does not guarantee faster finite-budget learning, nor does it restore the Markov property lost by the compact Q-table. Evaluation always reports original game scores; shaped rewards and Q-value magnitudes are not directly compared across stages.

Recorded Stage 2 training moves include base reward, potential before/after, shaping bonus, and total learning reward. Evaluation applies no training updates or shaping to reported scores. The dashboard exposes the reward breakdown when viewing a training episode.

## Matched protocol and interpretation

Both stages use seed 42 and 300 training episodes per agent, with the existing DQN settings documented in [learning.md](learning.md). Identical untrained evaluation scores are verified, demonstrating matched initialization. Each policy's later actions and visited boards naturally diverge.

Checkpoint games use seeds 8,000,000–8,000,019. Final games use seeds 9,000,000–9,000,099. These final seeds are disjoint from training and checkpoint games, but were already inspected in Stage 1. Reusing them supports a matched descriptive comparison; it does not provide fresh confirmatory evidence. No best-checkpoint selection is used. Both stage curves remain visible, including declines after earlier improvements.

There is one training initialization, no hyperparameter search, and no claim of statistical significance. Equal episode counts do not imply equal transitions or compute. The potential uses the complete board when calculating rewards for every model, but the Q-table is still unable to distinguish many arrangements in its inputs.

## Results: original points on 100 matched final games

| Agent | Stage 1 mean | Stage 2 mean | Change | Stage 2 score SD | Stage 1 → Stage 2 highest tile |
|---|---:|---:|---:|---:|---|
| Compact Q-learning | 1,011.96 | 1,776.52 | +75.6% | 880.52 | 256 → 512 |
| Dense DQN | 1,494.56 | 1,572.04 | +5.2% | 785.66 | 512 → 256 |
| CNN-DQN | 1,679.04 | 1,741.76 | +3.7% | 1,140.35 | 512 → 512 |

Random legal actions score 1,071.68 in the same final games. All three observed means improved with shaping, but **none reached 2048** in either stage's final evaluation. Dense DQN improved its average while its largest observed tile declined: one metric does not summarize all behavior.

The shaped CNN's checkpoint mean was higher at episode 50 than at episode 300. We retain the final checkpoint for the comparison rather than selecting that earlier result. The results do not establish that the agents consistently execute the intended snake strategy; demonstrating that would require separate behavioral measurements.

## Files and reproduction

```text
python -m src.train_2048 --episodes 300 --reward-mode snake
python -m unittest tests.test_2048 tests.test_2048_shaping -v
node --test tests/2048_environment.test.cjs tests/2048_results.test.cjs tests/2048_motion.test.cjs
```

Stage 2 defaults to `results/2048-stage2/` for report, Q-table, and neural checkpoints, and `docs/2048/stage2-data.js` for the offline dashboard data. The trainer rejects Stage 2 output paths that point at the Stage 1 report directory or browser data file. Hashes in `results/2048-baseline-hashes.json` verify that the preserved Stage 1 report, weights, Q-table and browser data remain byte-for-byte unchanged.

Use **Experiment stage** on the dashboard to switch replays, scores, checkpoint curves, and the learned Q-table. The cross-stage table always shows both results. Stage 2 curves overlay Stage 1 as faint dashed reference lines with shared axes. All replay controls and the slide/merge/spawn animations work for either stage. The page works offline, without a server.
