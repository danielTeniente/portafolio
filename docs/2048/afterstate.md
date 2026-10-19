> Saved model weights and standalone Q-table files are excluded from Git. Training code, final reports and browser replay data are included; regenerate models locally using the commands below.

# Stage 3: afterstate CNN

This replaces the retired Stage 3 duplicate-penalty idea. Old duplicate experiments remain saved separately and are excluded from active rankings.

## Architecture and decision

The input is one 4x4 grid of exponents divided by 16. Each legal move is simulated using deterministic slide/merge rules, without a random spawn. The same network evaluates all four previews in a batch; illegal actions are masked.

Architecture: Conv(1,16,2) → ReLU → Conv(16,32,2) → ReLU → Flatten(128) → Dense(64) → ReLU → Dense(1). No pooling or one-hot encoding. The convolutions and hidden-layer initialization match the Stage 2 CNN seed. The output has one continuation value, rather than four action values. There are 10,481 parameters.

The target keeps the original Stage 2 potential-based reward contract exactly. Let r be raw merge points/128, s the current full board, x its pre-spawn afterstate, and s' the actual sampled post-spawn next board:

```text
R = r + gamma * Phi(s') - Phi(s)
Q(s,a) = r - Phi(s) + U(x)
U target = R - r + Phi(s) + gamma * max_a' Q_target(s',a')
         = gamma * Phi(s') + gamma * max_a' Q_target(s',a')
```

Here gamma = 0.99 and U already includes the discount for the spawn continuation. Do not multiply U by gamma again when choosing an action. Both terminal potential and bootstrap are zero on true termination; truncations continue to bootstrap. The target network computes next action scores from next afterstates. With matching potentials, the next-state Phi terms cancel algebraically. Likewise, subtracting Phi(s) does not change the action ranking at a fixed current state. Thus the snake potential is not an extra hardcoded preference among previews; the network learns continuation value from rewards. This equivalence is specific to this decomposition, not proof of equal finite-training behavior to the DQN.

Replay stores afterstates, sampled next states' legal previews, their reward offsets and terminal masks. A sample uses the actual observed spawn, never a future random-generator state. Targets use the frozen target network. Immediate merge rewards are counted once. The preview inspector displays the real stored decision inputs before the shown move, not the current post-spawn board.

## Matched experiment

One new CNN trained fresh: seed 42, 300 episodes, epsilon 1 to 0.05, Adam 0.0005, batch 64, replay 10,000, warmup 256, one update every four moves, target copy every 250 updates, Huber loss and gradient clipping 10. Checkpoints: 0, 50, 100, 200, 300. Twenty fixed checkpoint games and 100 matched final games use the same seed ranges as earlier stages. No checkpoint selection or tuning after observing results.

The Stage 3 report reuses Stage 2 Q-learning and dense DQN as clearly labeled references. They are not retrained or counted as new candidates. The overall ranking contains seven distinct trained candidates. Selecting CNN displays Stage 1 CNN, Stage 2 CNN and Stage 3 afterstate CNN together. Other models retain their two-stage comparison.

## Final results

| CNN method | Mean original score | Score SD | 2048 wins |
|---|---:|---:|---:|
| Stage 1: direct action values | 1679.04 | 880.31 | 0/100 |
| Stage 2: direct action values + snake | 1741.76 | 1140.35 | 0/100 |
| Stage 3: afterstate values | 2051.48 | 1281.01 | 0/100 |

The new mean is 17.8% above Stage 2 CNN, with higher absolute game-to-game variation. This is a single training seed, so it does not establish general superiority. Stage 2 Q-learning remains the leader under the default mean-minus-SD ranking. Stage 3 consumes up to four preview evaluations per decision and a different bootstrap calculation; equal episodes are not equal compute. Current legal-action masking also uses game rules, but the new model additionally uses the full deterministic consequences as inputs.

## Reproduce

```text
python -m src.train_2048_afterstate
python -m unittest tests.test_2048_afterstate
```

Saved weights/report: `results/2048-afterstate`. Browser report: `docs/2048/afterstate-data.js`. Earlier stage reports and weights are unchanged. No server is needed for training.
