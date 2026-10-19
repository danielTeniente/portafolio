"""Train tabular Q, dense DQN and convolutional DQN; export auditable offline replays.

Run: python -m src.train_2048 --episodes 300
No one-hot features. Training, checkpoint evaluation and final evaluation seeds differ.
"""
import argparse
import copy
from collections import Counter
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.game2048 import Game, legal

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(gamma=.99, alpha=.1, learning_rate=.0005, batch=64,
              replay_capacity=10000, warmup=256, train_every=4,
              target_every=250, reward_scale=128, input_scale=16,
              max_moves=2000, epsilon_start=1., epsilon_end=.05)
DUPLICATE_WEIGHTS = {'mild': .25, 'medium': .5, 'strong': 1.}
NAMES = {'q': 'Q-learning', 'dqn': 'Dense DQN', 'cnn': 'CNN-DQN'}
SNAKE_PATH = (0, 4, 8, 12, 13, 9, 5, 1, 2, 6, 10, 14, 15, 11, 7, 3)
SHAPING = dict(weight=1., corner=.4, snake=.3, empty=.2, merges=.1,
               cornerIndex=0, path=list(SNAKE_PATH))


def potential(board):
    """Bounded board quality, fixed before Stage 2; exponents are not one-hot."""
    maximum = max(board)
    if not maximum:
        return 0.
    corner = board[0] / maximum
    ranks = [board[i] for i in SNAKE_PATH]
    violations = sum(max(0, b-a) for a, b in zip(ranks, ranks[1:]))
    snake = 1 - violations / sum(board)
    empty = board.count(0) / 16
    pairs = sum(board[i] != 0 and board[i] == board[j]
                for i in range(16) for j in (i+1, i+4)
                if j < 16 and (j == i+4 or i//4 == j//4))
    return SHAPING['corner']*corner + SHAPING['snake']*snake + SHAPING['empty']*empty + SHAPING['merges']*pairs/24


def duplicate_cost(board):
    """Extra-copy pairs, weighted by exponent/16 (capped); bounded [0,1]."""
    return sum(min(e, 16)/16 * n*(n-1)/2 for e, n in Counter(board).items() if e) / 120


def duplicate_count(board):
    return sum(max(0, n-1) for e, n in Counter(board).items() if e)


def reward_potential(board, mode):
    return potential(board) - DUPLICATE_WEIGHTS.get(mode, 0.) * duplicate_cost(board)


def learning_reward(state, raw_reward, next_state, done, reward_mode='baseline'):
    if reward_mode not in ('baseline', 'snake', 'mild', 'medium', 'strong'):
        raise ValueError('Unknown reward mode')
    base = raw_reward / CONFIG['reward_scale']
    before = reward_potential(state, reward_mode) if reward_mode != 'baseline' else 0.
    # Absorbing terminal potential is zero; time-limit truncations still bootstrap.
    after = reward_potential(next_state, reward_mode) if reward_mode != 'baseline' and not done else 0.
    bonus = SHAPING['weight'] * (CONFIG['gamma'] * after - before)
    return dict(baseReward=base, phiBefore=before, phiAfter=after,
                shapingBonus=bonus, learningReward=base+bonus)


def state_key(board):
    return (2 ** max(board) if max(board) else 0, board.count(0))


def choose(values, actions, rng, epsilon=0):
    if not actions:
        raise ValueError('Cannot act in a terminal state')
    exploratory = rng.random() < epsilon
    if exploratory:
        return rng.choice(actions), True
    best = max(values[a] for a in actions)
    ties = [a for a in actions if abs(values[a] - best) < 1e-7]
    return rng.choice(ties), False


class QAgent:
    def __init__(self):
        self.table = {}
        self.updates = 0
        self.last = None

    def values(self, board):
        return self.table.get(state_key(board), [0.] * 4).copy()

    def learn(self, state, action, reward, next_state, done):
        row = self.values(state)
        future = 0 if done else max(self.values(next_state)[a] for a in legal(next_state))
        target = reward + CONFIG['gamma'] * future
        before = row[action]
        row[action] += CONFIG['alpha'] * (target - before)
        self.table[state_key(state)] = row
        self.updates += 1
        self.last = dict(before=before, target=target, after=row[action], td=target-before)


def make_network(kind):
    if kind == 'dqn':
        return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64),
                             nn.ReLU(), nn.Linear(64, 4))
    if kind == 'cnn':
        return nn.Sequential(nn.Unflatten(1, (1, 4, 4)),
                             nn.Conv2d(1, 16, 2), nn.ReLU(),
                             nn.Conv2d(16, 32, 2), nn.ReLU(), nn.Flatten(),
                             nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 4))
    raise ValueError(kind)


def bellman_targets(rewards, dones, next_values, masks):
    # Terminal rows have no valid action; explicitly replace their maximum by zero.
    future = next_values.masked_fill(~masks, -torch.inf).max(dim=1).values
    future = torch.where(dones, torch.zeros_like(future), future)
    return rewards + CONFIG['gamma'] * future


class DQNAgent:
    def __init__(self, kind, seed):
        torch.manual_seed(seed)
        self.online = make_network(kind)
        self.target = copy.deepcopy(self.online).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=CONFIG['learning_rate'])
        self.rng = np.random.default_rng(seed + 500)
        n = CONFIG['replay_capacity']
        self.states = np.zeros((n, 16), dtype=np.float32)
        self.next_states = np.zeros_like(self.states)
        self.actions = np.zeros(n, dtype=np.int64)
        self.rewards = np.zeros(n, dtype=np.float32)
        self.dones = np.zeros(n, dtype=bool)
        self.masks = np.zeros((n, 4), dtype=bool)
        self.cursor = self.size = self.steps = self.updates = 0
        self.last = None

    @torch.no_grad()
    def values(self, board):
        x = torch.tensor([board], dtype=torch.float32) / CONFIG['input_scale']
        return self.online(x)[0].tolist()

    def learn(self, state, action, reward, next_state, done):
        i = self.cursor
        self.states[i] = np.asarray(state) / CONFIG['input_scale']
        self.next_states[i] = np.asarray(next_state) / CONFIG['input_scale']
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.masks[i] = False
        self.masks[i, legal(next_state)] = True
        self.cursor = (i + 1) % CONFIG['replay_capacity']
        self.size = min(self.size + 1, CONFIG['replay_capacity'])
        self.steps += 1
        if self.size < CONFIG['warmup'] or self.steps % CONFIG['train_every']:
            return
        ids = self.rng.integers(0, self.size, size=CONFIG['batch'])
        x = torch.from_numpy(self.states[ids])
        actions = torch.from_numpy(self.actions[ids])
        with torch.no_grad():
            target = bellman_targets(torch.from_numpy(self.rewards[ids]),
                                     torch.from_numpy(self.dones[ids]),
                                     self.target(torch.from_numpy(self.next_states[ids])),
                                     torch.from_numpy(self.masks[ids]))
        predictions = self.online(x).gather(1, actions[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        self.optimizer.step()
        self.updates += 1
        self.last = dict(loss=float(loss.detach()), updates=self.updates)
        if self.updates % CONFIG['target_every'] == 0:
            self.target.load_state_dict(self.online.state_dict())


def episode(agent, game_seed, action_seed, epsilon=0, training=False, record=False, reward_mode='baseline'):
    game, rng = Game(game_seed), random.Random(action_seed)
    duplicate_sum = cost_sum = merge_moves = 0
    frames = [dict(board=game.board.copy(), score=0, moves=0)] if record else []
    while legal(game.board) and game.moves < CONFIG['max_moves']:
        state = game.board.copy()
        values = agent.values(state) if agent is not None else [0.] * 4
        decision = copy.deepcopy(getattr(agent, 'decision', None)) if record else None
        action, exploratory = choose(values, legal(state), rng, epsilon)
        result = game.step(action)
        duplicate_sum += duplicate_count(game.board)
        cost_sum += duplicate_cost(game.board)
        merge_moves += result['reward'] > 0
        reward_parts = learning_reward(state, result['reward'], game.board, result['done'], reward_mode) if training else None
        if training:
            # A time-limit cutoff is not an MDP terminal: bootstrap on that transition.
            agent.learn(state, action, reward_parts['learningReward'], game.board, result['done'])
        if record:
            frames.append(dict(board=game.board.copy(), score=game.score, moves=game.moves,
                               action=action, reward=result['reward'], spawned=result['spawned'],
                               q=[round(v, 5) for v in values], legal=legal(state),
                               exploratory=exploratory, key=list(state_key(state)), decision=decision,
                               update=agent.last if training else None, rewardParts=reward_parts))
    return dict(seed=game_seed, score=game.score, moves=game.moves,
                meanDuplicates=duplicate_sum/max(1, game.moves), meanDuplicateCost=cost_sum/max(1, game.moves),
                mergeMoveFraction=merge_moves/max(1, game.moves),
                maxTile=2 ** max(game.board), terminated=not legal(game.board),
                truncated=bool(legal(game.board)), frames=frames)


def evaluate(agent, count, seed_start, trace_count=3):
    games = [episode(agent, seed_start+i, seed_start+i+100000, record=i < trace_count)
             for i in range(count)]
    scores = [g['score'] for g in games]
    return dict(mean=float(np.mean(scores)), sd=float(np.std(scores, ddof=1)) if count > 1 else 0.,
                scores=scores, meanDuplicates=float(np.mean([g['meanDuplicates'] for g in games])),
                meanDuplicateCost=float(np.mean([g['meanDuplicateCost'] for g in games])),
                mergeMoveFraction=float(np.mean([g['mergeMoveFraction'] for g in games])), maxTile=max(g['maxTile'] for g in games),
                wins=sum(g['maxTile'] >= 2048 for g in games),
                truncated=sum(g['truncated'] for g in games),
                traces=games[:trace_count])


def train(args):
    if args.episodes < 1 or args.eval_games < 2 or args.final_games < 2:
        raise ValueError('Use positive episodes and at least two evaluation games')
    reward_mode = getattr(args, 'reward_mode', 'baseline')
    if reward_mode not in ('baseline', 'snake', 'mild', 'medium', 'strong'):
        raise ValueError('Unknown reward mode')
    variant = reward_mode in DUPLICATE_WEIGHTS
    output = Path(args.output) if args.output else ROOT / 'results' / ('2048-stage3-' + reward_mode if variant else '2048-stage2' if reward_mode == 'snake' else '2048')
    dashboard = Path(args.dashboard) if args.dashboard else ROOT / 'docs' / '2048' / ('stage3-' + reward_mode + '-data.js' if variant else 'stage2-data.js' if reward_mode == 'snake' else 'data.js')
    # Never replace the preserved baseline when requesting the second experiment.
    if reward_mode == 'snake' and (output.resolve() == (ROOT/'results/2048').resolve() or dashboard.resolve() == (ROOT/'docs/2048/data.js').resolve()):
        raise ValueError('Stage 2 must use separate output paths from Stage 1')
    if variant and (output.resolve() in [(ROOT/'results'/n).resolve() for n in ('2048', '2048-stage2')] or dashboard.resolve() in [(ROOT/'docs/2048'/n).resolve() for n in ('data.js', 'stage2-data.js')]):
        raise ValueError('Stage 3 must preserve previous stages')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoints = sorted(set([0, max(1, args.episodes//6), max(1, args.episodes//3),
                              max(1, 2*args.episodes//3), args.episodes]))
    report = dict(schema=1, seed=args.seed, episodes=args.episodes, config=CONFIG,
                  stage=3 if variant else 2 if reward_mode == 'snake' else 1, rewardMode=reward_mode,
                  duplicateWeight=DUPLICATE_WEIGHTS.get(reward_mode, 0.),
                  initialization='fresh', shaping=SHAPING if reward_mode != 'baseline' else None,
                  checkpoints=checkpoints, evalGames=args.eval_games, finalGames=args.final_games,
                  evaluationSeeds=dict(checkpointStart=8000000, finalStart=9000000),
                  versions=dict(torch=torch.__version__, numpy=np.__version__), agents={})
    report['random'] = dict(checkpoint=evaluate(None, args.eval_games, 8000000),
                            final=evaluate(None, args.final_games, 9000000))
    output.mkdir(parents=True, exist_ok=True)
    for kind, name in NAMES.items():
        agent = QAgent() if kind == 'q' else DQNAgent(kind, args.seed)
        entry = dict(name=name, history=[], snapshots=[], trainingSeconds=0.,
                     parameters=0 if kind == 'q' else sum(p.numel() for p in agent.online.parameters()))
        entry['snapshots'].append(dict(episode=0, evaluation=evaluate(agent, args.eval_games, 8000000), training=None))
        for ep in range(1, args.episodes+1):
            epsilon = 1 - .95 * (ep-1) / max(1, args.episodes-1)
            start = time.perf_counter()
            result = episode(agent, args.seed+1000+ep, args.seed+200000+ep,
                             epsilon, True, ep in checkpoints, reward_mode)
            entry['trainingSeconds'] += time.perf_counter() - start
            entry['history'].append(dict(episode=ep, score=result['score'], moves=result['moves'],
                                         epsilon=epsilon, maxTile=result['maxTile'], truncated=result['truncated']))
            if ep in checkpoints:
                entry['snapshots'].append(dict(episode=ep, evaluation=evaluate(agent, args.eval_games, 8000000), training=result))
                print(f'{name}: episode {ep}/{args.episodes}; evaluation mean {entry["snapshots"][-1]["evaluation"]["mean"]:.1f}', flush=True)
        entry['final'] = evaluate(agent, args.final_games, 9000000)
        entry['updates'] = agent.updates
        if kind == 'q':
            entry['table'] = [dict(largest=k[0], empty=k[1], q=v) for k, v in sorted(agent.table.items())]
            entry['qEntries'] = len(agent.table) * 4
            (output / 'q_table.json').write_text(json.dumps(entry['table'], indent=2), encoding='utf-8')
        else:
            torch.save(dict(kind=kind, state_dict=agent.online.state_dict(), config=CONFIG,
                            episodes=args.episodes, seed=args.seed, reward_mode=reward_mode,
                            shaping=report['shaping'], duplicate_weight=report['duplicateWeight']), output / f'{kind}.pt')
        report['agents'][kind] = entry
    payload = json.dumps(report, separators=(',', ':'), allow_nan=False)
    (output / 'report.json').write_text(payload, encoding='utf-8')
    variable = 'LAB2048_STAGE3_' + reward_mode.upper() if variant else 'LAB2048_STAGE2' if reward_mode == 'snake' else 'LAB2048_RESULTS'
    dashboard.write_text(f'window.{variable}=' + payload + ';\n', encoding='utf-8')
    print(json.dumps({k: {'mean': v['final']['mean'], 'seconds': round(v['trainingSeconds'], 1)} for k, v in report['agents'].items()}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-games', type=int, default=20)
    parser.add_argument('--final-games', type=int, default=100)
    parser.add_argument('--reward-mode', choices=('baseline', 'snake', 'mild', 'medium', 'strong'), default='baseline')
    parser.add_argument('--output', default=None, help='Stage-specific result directory by default')
    parser.add_argument('--dashboard', default=None, help='Stage-specific browser data file by default')
    train(parser.parse_args())
