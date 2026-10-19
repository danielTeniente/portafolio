import hashlib
import json
from pathlib import Path
import random
import unittest

import numpy as np
import torch

from src.game2048 import legal
from src.train_2048 import CONFIG, SHAPING, SNAKE_PATH, learning_reward, make_network, potential, state_key

ROOT = Path(__file__).resolve().parents[1]


class ShapingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.baseline = json.loads((ROOT/'results/2048/report.json').read_text(encoding='utf-8'))
        cls.shaped = json.loads((ROOT/'results/2048-stage2/report.json').read_text(encoding='utf-8'))

    def test_stage1_artifacts_unchanged(self):
        for entry in json.loads((ROOT/'results/2048-baseline-hashes.json').read_text()):
            path = ROOT/entry['path']
            if not path.exists() and (path.suffix == '.pt' or path.name == 'q_table.json'):
                continue  # Models are regenerated locally, not shipped in Git.
            self.assertEqual(hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest(), entry['sha256'])

    def test_potential_bounded_and_snake_covers_each_position(self):
        self.assertEqual(sorted(SNAKE_PATH), list(range(16)))
        self.assertEqual(list(SNAKE_PATH[:4]), [0,4,8,12])
        rng = random.Random(42)
        for _ in range(1000):
            value = potential([rng.randrange(12) for _ in range(16)])
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1)
        self.assertEqual(potential([0]*16), 0)

    def test_ordered_corner_has_higher_potential_than_reverse_order(self):
        ordered = [0]*16
        for rank, index in enumerate(SNAKE_PATH):
            ordered[index] = 16-rank
        reversed_board = ordered[::-1]
        self.assertGreater(potential(ordered), potential(reversed_board))
        self.assertEqual(state_key(ordered), state_key(reversed_board))

    def test_terminal_zero_potential_and_truncation_bootstrap(self):
        s = [1,1]+[0]*14
        next_s = [2,1]+[0]*14
        terminal = learning_reward(s, 4, next_s, True, 'snake')
        continuing = learning_reward(s, 4, next_s, False, 'snake')
        self.assertEqual(terminal['phiAfter'], 0)
        self.assertAlmostEqual(terminal['learningReward'], 4/128-potential(s))
        self.assertGreater(continuing['phiAfter'], 0)
        self.assertAlmostEqual(continuing['shapingBonus'], .99*potential(next_s)-potential(s))

    def test_discounted_bonus_telescopes(self):
        s0, s1, s2 = [1,1]+[0]*14, [2,1]+[0]*14, [2,2]+[0]*14
        b0 = learning_reward(s0, 4, s1, False, 'snake')['shapingBonus']
        b1 = learning_reward(s1, 4, s2, True, 'snake')['shapingBonus']
        self.assertAlmostEqual(b0+CONFIG['gamma']*b1, -potential(s0))
        base = learning_reward(s0, 4, s1, False, 'baseline')
        self.assertEqual(base['learningReward'], 4/128)
        self.assertEqual(base['shapingBonus'], 0)

    def test_matched_protocol_and_identical_untrained_policies(self):
        a, b = self.baseline, self.shaped
        for key in ('seed','episodes','config','checkpoints','evalGames','finalGames','evaluationSeeds'):
            self.assertEqual(a[key], b[key])
        self.assertEqual(b['rewardMode'], 'snake')
        self.assertEqual(b['initialization'], 'fresh')
        self.assertEqual(b['shaping'], SHAPING)
        self.assertEqual(a['random']['final']['scores'], b['random']['final']['scores'])
        for key in ('q','dqn','cnn'):
            self.assertEqual(a['agents'][key]['snapshots'][0]['evaluation']['scores'], b['agents'][key]['snapshots'][0]['evaluation']['scores'])
            self.assertEqual(a['agents'][key]['parameters'], b['agents'][key]['parameters'])

    def test_recorded_shaping_matches_formula_for_every_training_frame(self):
        for agent in self.shaped['agents'].values():
            for snapshot in agent['snapshots']:
                trace = snapshot['training']
                if not trace:
                    continue
                for previous, frame in zip(trace['frames'], trace['frames'][1:]):
                    expected = learning_reward(previous['board'], frame['reward'], frame['board'], not legal(frame['board']), 'snake')
                    for key, value in expected.items():
                        self.assertAlmostEqual(value, frame['rewardParts'][key])
                self.assertEqual(trace['score'], sum(f.get('reward',0) for f in trace['frames']))
            for trace in agent['final']['traces']:
                self.assertTrue(all(f.get('rewardParts') is None for f in trace['frames']))

    def test_stage2_checkpoints_reproduce_final_action_values(self):
        if not all((ROOT/f'results/2048-stage2/{kind}.pt').exists() for kind in ('cnn','dqn')):
            self.skipTest('Saved models are excluded from Git; regenerate them to run this check')
        for kind in ('dqn','cnn'):
            saved = torch.load(ROOT/f'results/2048-stage2/{kind}.pt', weights_only=True)
            self.assertEqual(saved['reward_mode'], 'snake')
            net = make_network(kind).eval()
            net.load_state_dict(saved['state_dict'])
            for trace in self.shaped['agents'][kind]['final']['traces']:
                for previous, frame in zip(trace['frames'][:5], trace['frames'][1:6]):
                    with torch.no_grad():
                        actual = net(torch.tensor([previous['board']], dtype=torch.float32)/16)[0].numpy()
                    np.testing.assert_allclose(actual, frame['q'], atol=1e-5, rtol=0)


if __name__ == '__main__':
    unittest.main()
