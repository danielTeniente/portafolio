"""Stage 3: CNN afterstate value learning, with the Stage 2 snake reward."""
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from src.game2048 import slide, legal
from src.train_2048 import (ROOT, CONFIG, SHAPING, DQNAgent, make_network,
                           potential, episode, evaluate)


def previews(board):
    outcomes = [slide(board, a) for a in range(4)]
    return ([x[0] for x in outcomes],
            [x[1]/CONFIG['reward_scale']-potential(board) for x in outcomes],
            [x[2] for x in outcomes])


class AfterstateAgent(DQNAgent):
    def __init__(self, seed):
        super().__init__('cnn', seed)
        # Reset seed so the convolution and hidden-layer initialization matches Stage 2.
        torch.manual_seed(seed)
        self.online = make_network('cnn')
        self.online[-1] = nn.Linear(64, 1)
        self.target = copy.deepcopy(self.online).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=CONFIG['learning_rate'])
        n = CONFIG['replay_capacity']
        self.next_previews = np.zeros((n,4,16), dtype=np.float32)
        self.next_offsets = np.zeros((n,4), dtype=np.float32)
        self.decision = None
        self.input_size = 16

    def encode(self, boards):
        return np.asarray(boards, dtype=np.float32)/16

    @torch.no_grad()
    def values(self, board):
        boards, offsets, masks = previews(board)
        continuation = self.online(torch.from_numpy(self.encode(boards))).flatten().tolist()
        self.decision = [dict(board=b, legal=valid, immediate=slide(board,a)[1]/128,
                              continuation=u, actionValue=offset+u)
                         for a,(b,offset,valid,u) in enumerate(zip(boards,offsets,masks,continuation))]
        if hasattr(self, 'board_features'):
            for item in self.decision:
                item['features'] = self.board_features(item['board'])
        return [offset+u for offset,u in zip(offsets,continuation)]

    def learn(self, state, action, reward, next_state, done):
        i = self.cursor
        after, raw, changed = slide(state, action)
        assert changed
        self.states[i] = self.encode([after])[0]
        boards, offsets, masks = previews(next_state)
        self.next_previews[i] = self.encode(boards)
        self.next_offsets[i] = offsets
        # Q(s,a)=r/128-Phi(s)+U(afterstate). U includes the discounted spawn continuation.
        self.rewards[i] = reward-raw/128+potential(state)
        self.dones[i], self.masks[i] = done, masks
        self.cursor = (i+1) % CONFIG['replay_capacity']
        self.size = min(self.size+1, CONFIG['replay_capacity'])
        self.steps += 1
        if self.size < CONFIG['warmup'] or self.steps % CONFIG['train_every']:
            return
        ids = self.rng.integers(0,self.size,size=CONFIG['batch'])
        with torch.no_grad():
            values = self.target(torch.from_numpy(self.next_previews[ids]).reshape(-1,self.input_size)).reshape(-1,4)
            values += torch.from_numpy(self.next_offsets[ids])
            future = values.masked_fill(~torch.from_numpy(self.masks[ids]),-torch.inf).max(1).values
            future = torch.where(torch.from_numpy(self.dones[ids]),torch.zeros_like(future),future)
            target = torch.from_numpy(self.rewards[ids]) + CONFIG['gamma']*future
        prediction = self.online(torch.from_numpy(self.states[ids])).flatten()
        loss = nn.functional.smooth_l1_loss(prediction,target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        self.optimizer.step()
        self.updates += 1
        self.last = dict(loss=float(loss.detach()),updates=self.updates)
        if self.updates % CONFIG['target_every'] == 0:
            self.target.load_state_dict(self.online.state_dict())


def train(stage=3, agent_factory=AfterstateAgent):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reference = json.loads((ROOT/'results/2048-stage2/report.json').read_text(encoding='utf-8'))
    report = copy.deepcopy(reference)
    hybrid = stage == 4
    architecture = 'hybrid-afterstate-cnn' if hybrid else 'afterstate-cnn'
    report.update(stage=stage, rewardMode='snake', architecture=architecture,
                  reusedAgents=['q','dqn'], initialization='fresh',
                  reference='Stage 2 Q and dense DQN are reused, not retrained.')
    if hybrid:
        report.update(agents={}, reusedAgents=[], reference='Stage 4 trains only the hybrid afterstate CNN.',
                      features=['empty_count/16', 'largest_in_top_left', 'snake_violations/sum_exponents',
                                'merge_up/128','merge_right/128','merge_down/128','merge_left/128'])
    agent = agent_factory(reference['seed'])
    entry = dict(name='Hybrid afterstate CNN' if hybrid else 'Afterstate CNN',history=[],snapshots=[],trainingSeconds=0.,
                 parameters=sum(p.numel() for p in agent.online.parameters()))
    entry['snapshots'].append(dict(episode=0,evaluation=evaluate(agent,20,8000000),training=None))
    for ep in range(1,301):
        epsilon = 1-.95*(ep-1)/299
        start = time.perf_counter()
        result = episode(agent,42+1000+ep,42+200000+ep,epsilon,True,ep in reference['checkpoints'],'snake')
        entry['trainingSeconds'] += time.perf_counter()-start
        entry['history'].append(dict(episode=ep,score=result['score'],moves=result['moves'],epsilon=epsilon,
                                    maxTile=result['maxTile'],truncated=result['truncated']))
        if ep in reference['checkpoints']:
            evaluation = evaluate(agent,20,8000000)
            entry['snapshots'].append(dict(episode=ep,evaluation=evaluation,training=result))
            print(f'{entry["name"]} {ep}/300: {evaluation["mean"]:.1f}',flush=True)
    entry['final'] = evaluate(agent,100,9000000)
    entry['updates'] = agent.updates
    report['agents']['cnn'] = entry
    output = ROOT/('results/2048-hybrid' if hybrid else 'results/2048-afterstate')
    output.mkdir(exist_ok=True)
    torch.save(dict(state_dict=agent.online.state_dict(),config=CONFIG,shaping=SHAPING,
                    architecture=architecture,episodes=300,seed=42),output/'cnn.pt')
    payload = json.dumps(report,separators=(',',':'),allow_nan=False)
    (output/'report.json').write_text(payload,encoding='utf-8')
    (ROOT/('docs/2048/hybrid-data.js' if hybrid else 'docs/2048/afterstate-data.js')).write_text(
        ('window.LAB2048_HYBRID=' if hybrid else 'window.LAB2048_AFTERSTATE=')+payload+';',encoding='utf-8')
    print({k:entry['final'][k] for k in ('mean','sd','wins','maxTile')},flush=True)


if __name__ == '__main__':
    train()
