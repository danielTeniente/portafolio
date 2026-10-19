import json
from pathlib import Path
import random
import subprocess
import unittest

import numpy as np
import torch

from src.game2048 import Game, legal
from src.train_2048 import CONFIG, DQNAgent, QAgent, bellman_targets, choose, evaluate, make_network, state_key


class Learning2048Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_python_browser_environment_parity(self):
        root = Path(__file__).resolve().parents[1]
        script = """const E=require('./docs/2048/engine.js'); const runs=[];
        for(const seed of [0,7,42,4294967295]) { const g=new E.Game(seed), frames=[g.board.slice()];
        for(let i=0;i<200;i++){const a=E.legal(g.board);if(!a.length)break;frames.push(g.step(a[i%a.length]));}
        runs.push({seed,frames}); } console.log(JSON.stringify(runs));"""
        runs = json.loads(subprocess.check_output(['node', '-e', script], cwd=root))
        for run in runs:
            game = Game(run['seed'])
            self.assertEqual(game.board, run['frames'][0])
            for i, expected in enumerate(run['frames'][1:]):
                actions = legal(game.board)
                self.assertEqual(game.step(actions[i % len(actions)]), expected)

    def test_table_key_uses_only_largest_and_empty(self):
        a = [1, 1, 2] + [0] * 13
        b = [1, 2, 1] + [0] * 13
        self.assertEqual(state_key(a), (4, 13))
        self.assertEqual(state_key(a), state_key(b))
        agent = QAgent()
        agent.learn(a, 3, 2., b, True)
        self.assertAlmostEqual(agent.values(b)[3], .2)
        self.assertEqual(len(agent.table), 1)

    def test_tabular_update_and_terminal_no_bootstrap(self):
        agent = QAgent()
        board = [1] + [0] * 15
        agent.table[state_key(board)] = [999., 2., 3., 1000.]
        # Only right/down legal: ignore the larger invalid-action estimates.
        agent.learn(board, 1, 1., board, False)
        self.assertAlmostEqual(agent.last['target'], 1 + .99 * 3)
        agent.learn(board, 1, 1., [1,2,1,2,2,1,2,1]*2, True)
        self.assertEqual(agent.last['target'], 1.)

    def test_neural_shapes_no_onehot_and_convolution(self):
        for kind in ('dqn', 'cnn'):
            network = make_network(kind)
            self.assertEqual(tuple(network(torch.zeros(5, 16)).shape), (5, 4))
        conv = next(m for m in make_network('cnn') if isinstance(m, torch.nn.Conv2d))
        self.assertEqual(conv.in_channels, 1)
        self.assertEqual(make_network('dqn')[0].in_features, 16)

    def test_bellman_targets_mask_invalid_and_terminal_actions(self):
        result = bellman_targets(torch.tensor([2., 3.]), torch.tensor([False, True]),
                                 torch.tensor([[99., 4., 5., 100.], [9., 9., 9., 9.]]),
                                 torch.tensor([[False, True, True, False], [False]*4]))
        torch.testing.assert_close(result, torch.tensor([2 + .99*5, 3.]))

    def test_network_training_updates_weights_and_target_is_delayed(self):
        board = [1] + [0] * 15
        for kind in ('dqn', 'cnn'):
            agent = DQNAgent(kind, 42)
            original = [p.clone() for p in agent.online.parameters()]
            for _ in range(CONFIG['warmup']):
                agent.learn(board, 1, 1., board, False)
            self.assertGreater(agent.updates, 0)
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(original, agent.online.parameters())))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(original, agent.target.parameters())))
            agent.updates = CONFIG['target_every'] - 1
            for _ in range(CONFIG['train_every']):
                agent.learn(board, 1, 1., board, False)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(agent.online.parameters(), agent.target.parameters())))
            self.assertTrue(np.isfinite(agent.last['loss']))

    def test_evaluation_is_reproducible_and_does_not_train(self):
        agent = DQNAgent('cnn', 7)
        before = [p.clone() for p in agent.online.parameters()]
        first = evaluate(agent, 2, 9000000, 1)
        self.assertEqual(first, evaluate(agent, 2, 9000000, 1))
        self.assertEqual(agent.updates, 0)
        self.assertEqual(agent.size, 0)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before, agent.online.parameters())))

    def test_actions_always_legal(self):
        rng = random.Random(42)
        for epsilon in (0, 1):
            for _ in range(30):
                action, _ = choose([100., 1., 2., 999.], [1, 2], rng, epsilon)
                self.assertIn(action, (1, 2))

    def test_saved_network_weights_reproduce_recorded_final_q_values(self):
        root = Path(__file__).resolve().parents[1]
        if not all((root/f'results/2048/{kind}.pt').exists() for kind in ('cnn','dqn')):
            self.skipTest('Saved models are excluded from Git; regenerate them to run this check')
        report = json.loads((root / 'results/2048/report.json').read_text())
        for kind in ('dqn', 'cnn'):
            saved = torch.load(root / f'results/2048/{kind}.pt', weights_only=True)
            network = make_network(kind).eval()
            network.load_state_dict(saved['state_dict'])
            for trace in report['agents'][kind]['final']['traces']:
                for previous, frame in zip(trace['frames'][:5], trace['frames'][1:6]):
                    with torch.no_grad():
                        actual = network(torch.tensor([previous['board']], dtype=torch.float32) / CONFIG['input_scale'])[0].numpy()
                    np.testing.assert_allclose(actual, frame['q'], atol=1e-5, rtol=0)


if __name__ == '__main__':
    unittest.main()
