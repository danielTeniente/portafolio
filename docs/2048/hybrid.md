> Saved model weights and standalone Q-table files are excluded from Git. Training code, final reports and browser replay data are included; regenerate models locally using the commands below.

# Stage 4: CNN plus seven board features

The reward and afterstate learning target are unchanged from [Stage 3](afterstate.md). This experiment changes the evaluator: a spatial CNN branch and a separate statistics branch learn jointly.

## Two input branches

For each legal action, calculate its deterministic afterstate before the random spawn. The board branch receives the 16 exponents divided by 16, reshaped into one 4x4 channel. The statistics branch receives seven values calculated from that exact same afterstate:

1. Empty cells divided by 16.
2. A flag indicating whether the maximum nonempty tile is at the top-left corner (ties allowed; all-empty returns 0).
3. Sum of upward exponent jumps along the fixed column-first snake path, divided by the sum of exponents (all-empty returns 0).
4. Merge points available by sliding up, divided by 128.
5. Merge points available by sliding right, divided by 128.
6. Merge points available by sliding down, divided by 128.
7. Merge points available by sliding left, divided by 128.

The first three values are bounded between 0 and 1. Merge features use the same fixed scaling as the base reward and are not clipped. Actual slide rules account for gaps, blockers and one-merge-per-tile behavior. These are hypothetical merge opportunities on a pre-spawn board, not promises about the next decision after the random tile arrives. Computing them adds deterministic rule calculations, not hidden random-state information.

```text
Grid:       1x4x4 -> Conv16(2x2) -> ReLU -> Conv32(2x2) -> ReLU -> Flatten128
Statistics: 7 -> Dense16 -> ReLU
Fusion:     concatenate 128+16 -> Dense64 -> ReLU -> Dense1
```

The statistics never enter the convolution layers. The final dense layers receive both branches. Prediction errors update both branches and the fusion layers; the formulas defining the original seven statistics stay fixed. All moves share one network. There are 11,633 parameters versus Stage 3's 10,481. Initial convolution weights match Stage 3 exactly; the new summary and fusion layers use seeded initialization.

## Protocol

A fresh CNN is trained for 300 episodes with seed 42. Same snake-shaped reward, afterstate target decomposition, gamma, Adam settings, replay size, warmup, batch size, exploration schedule, target-network update interval and gradient clipping as Stage 3. Checkpoints are 0, 50, 100, 200 and 300 episodes; 20 fixed evaluation games per checkpoint, then 100 matched final games. No checkpoint selection or post-result tuning.

The final-results UI separates six Stage 1/2 runs (three models each) from the two CNN-only Stage 3/4 runs. Stage 4's saved report contains only its newly trained CNN. Earlier reports and weights are unchanged. Eight distinct runs participate in the overall mean/consistency ranking; duplicated references are excluded. The CNN-only replay compares Stage 3 with Stage 4 with matching checkpoint and seed, and the Stage 4 preview inspector exposes its seven input features.

An improvement would support this complete hybrid approach under this budget, not prove that a larger network or these features alone caused it. Stage 4 adds both parameters and explicit merge-opportunity calculations. Equal episodes do not mean equal compute. Repeated training seeds and a larger CNN without summary features would help isolate those effects.

## Reproduce

```text
python -m src.train_2048_hybrid
python -m unittest tests.test_2048_hybrid
```

Results and weights: `results/2048-hybrid`. Browser data: `docs/2048/hybrid-data.js`.

## Final observed results

| Model | Mean score | Score SD | Relative variation | Highest tile | 2048 wins |
|---|---:|---:|---:|---:|---:|
| Stage 3 afterstate CNN | 2051.48 | 1281.01 | 62.4% | 512 | 0/100 |
| Stage 4 hybrid CNN | 5262.88 | 2578.94 | 49.0% | 1024 | 0/100 |

The observed mean increased 156.5%. Absolute variation increased, but variation relative to the mean fell. No final evaluation game reached 2048. This is a single-seed result, not an established general improvement.

### Independent experiment sections

The page now groups the complete experience, not just the final tables. Stages 1–2 has its own three-model replay, stage selector, checkpoints, controls, curves and six-row final results. Stages 3–4 has a separate CNN-only replay, stage selector, checkpoints, controls, curves and two-row final results. Playback and selections do not affect the other section. Single-model comparison in the first section never includes Stages 3–4. The overall ranking remains a separate cross-experiment summary between the groups.
