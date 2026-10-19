import json
import unittest
import numpy as np
import torch
from src.game2048 import slide, Game
from src.train_2048 import ROOT, potential, learning_reward, episode
from src.train_2048_afterstate import AfterstateAgent, previews


class AfterstateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_previews_are_deterministic_and_exclude_spawn(self):
        game = Game(42)
        before = game.board.copy()
        rng = game.random.state
        boards, offsets, masks = previews(before)
        self.assertEqual(game.board,before)
        self.assertEqual(game.random.state,rng)
        for action in range(4):
            after,reward,changed = slide(before,action)
            self.assertEqual(boards[action],after)
            self.assertEqual(masks[action],changed)
            self.assertAlmostEqual(offsets[action],reward/128-potential(before))
            self.assertEqual(sum(2**e for e in after if e),sum(2**e for e in before if e))

    def test_action_values_and_replay_target_account_for_reward_once(self):
        agent = AfterstateAgent(42)
        for p in agent.online.parameters():
            p.data.zero_()
        agent.online[-1].bias.data.fill_(2.)
        state = [4,4]+[0]*14
        values = agent.values(state)
        self.assertAlmostEqual(values[3],32/128-potential(state)+2.)
        after,raw,_ = slide(state,3)
        next_state=after.copy(); next_state[15]=1
        shaped=learning_reward(state,raw,next_state,False,'snake')['learningReward']
        agent.learn(state,3,shaped,next_state,False)
        np.testing.assert_allclose(agent.states[0],np.array(after)/16)
        self.assertAlmostEqual(float(agent.rewards[0]),.99*potential(next_state),places=6)
        shaped=learning_reward(state,raw,next_state,True,'snake')['learningReward']
        agent.learn(state,3,shaped,next_state,True)
        self.assertAlmostEqual(float(agent.rewards[1]),0.,places=6)
        self.assertTrue(agent.dones[1])

    def test_saved_run_matches_protocol_and_frozen_replays(self):
        if not (ROOT/'results/2048-afterstate/cnn.pt').exists():
            self.skipTest('Saved models are excluded from Git; regenerate them to run this check')
        report=json.loads((ROOT/'results/2048-afterstate/report.json').read_text())
        baseline=json.loads((ROOT/'results/2048-stage2/report.json').read_text())
        self.assertEqual(report['config'],baseline['config'])
        self.assertEqual(report['shaping'],baseline['shaping'])
        self.assertEqual(report['episodes'],300)
        for key in ('q','dqn'):
            self.assertEqual(report['agents'][key],baseline['agents'][key])
        agent=AfterstateAgent(42)
        agent.online.load_state_dict(torch.load(ROOT/'results/2048-afterstate/cnn.pt',weights_only=True)['state_dict'])
        self.assertEqual(agent.online[-1].out_features,1)
        for trace in report['agents']['cnn']['final']['traces']:
            result=episode(agent,trace['seed'],trace['seed']+100000,record=True)
            self.assertEqual(result,trace)
        for snapshot in report['agents']['cnn']['snapshots']:
            groups=[*snapshot['evaluation']['traces']]
            if snapshot['training']:
                groups.append(snapshot['training'])
            for trace in groups:
                for previous,frame in zip(trace['frames'],trace['frames'][1:]):
                    for action,preview in enumerate(frame['decision']):
                        after,raw,valid=slide(previous['board'],action)
                        self.assertEqual(preview['board'],after)
                        self.assertEqual(preview['legal'],valid)
                        self.assertAlmostEqual(preview['actionValue'],raw/128-potential(previous['board'])+preview['continuation'])
                        self.assertAlmostEqual(frame['q'][action],preview['actionValue'],places=4)
