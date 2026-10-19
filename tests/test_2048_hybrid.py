import hashlib
import json
import unittest
import numpy as np
import torch
from src.game2048 import slide
from src.train_2048 import ROOT, episode
from src.train_2048_afterstate import AfterstateAgent
from src.train_2048_hybrid import HybridAfterstateAgent, board_features


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_seven_features_ignore_empty_tiles_and_respect_blockers(self):
        self.assertEqual(board_features([0]*16),[1.,0.,0.,0.,0.,0.,0.])
        board=[4,4,0,0]+[0]*12
        f=board_features(board)
        self.assertEqual(len(f),7)
        self.assertEqual(f[:2],[14/16,1.])
        self.assertEqual(f[3:],[0.,.25,0.,.25])
        blocked=[4,3,4,0]+[0]*12
        self.assertEqual(board_features(blocked)[3:],[0.,0.,0.,0.])
        for action in range(4):
            self.assertEqual(f[3+action],slide(board,action)[1]/128)

    def test_shared_initial_convolution_and_both_branches_train(self):
        base=AfterstateAgent(42)
        agent=HybridAfterstateAgent(42)
        for old,new in zip(base.online[:6].parameters(),agent.online.convolution.parameters()):
            self.assertTrue(torch.equal(old,new))
        boards=[[4,4,2,0]+[0]*12,[0,2,3,4]+[0]*12]
        encoded=agent.encode(boards)
        self.assertEqual(encoded.shape,(2,23))
        np.testing.assert_allclose(encoded[:,16:],np.array([board_features(b) for b in boards]))
        output=agent.online(torch.from_numpy(encoded))
        self.assertEqual(tuple(output.shape),(2,1))
        (output-1).square().mean().backward()
        self.assertTrue(any(p.grad.abs().sum()>0 for p in agent.online.convolution.parameters()))
        self.assertTrue(any(p.grad.abs().sum()>0 for p in agent.online.summary.parameters()))

    def test_stage3_unchanged(self):
        for item in json.loads((ROOT/'results/2048-pre-stage4-hashes.json').read_text()):
            path = ROOT/item['path']
            if not path.exists() and path.suffix == '.pt':
                continue  # Models are regenerated locally, not shipped in Git.
            self.assertEqual(hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest(),item['sha256'])

    def test_saved_model_decisions_and_training_protocol(self):
        if not (ROOT/'results/2048-hybrid/cnn.pt').exists():
            self.skipTest('Saved models are excluded from Git; regenerate them to run this check')
        report=json.loads((ROOT/'results/2048-hybrid/report.json').read_text())
        previous=json.loads((ROOT/'results/2048-afterstate/report.json').read_text())
        self.assertEqual(list(report['agents']),['cnn'])
        for field in ('config','seed','episodes','checkpoints','evaluationSeeds','shaping'):
            self.assertEqual(report[field],previous[field])
        agent=HybridAfterstateAgent(42)
        agent.online.load_state_dict(torch.load(ROOT/'results/2048-hybrid/cnn.pt',weights_only=True)['state_dict'])
        self.assertEqual(sum(p.numel() for p in agent.online.parameters()),report['agents']['cnn']['parameters'])
        for trace in report['agents']['cnn']['final']['traces']:
            self.assertEqual(episode(agent,trace['seed'],trace['seed']+100000,record=True),trace)
        for snapshot in report['agents']['cnn']['snapshots']:
            traces=list(snapshot['evaluation']['traces'])
            if snapshot['training']:
                traces.append(snapshot['training'])
            for trace in traces:
                for previous_frame,frame in zip(trace['frames'],trace['frames'][1:]):
                    for action,preview in enumerate(frame['decision']):
                        self.assertEqual(preview['board'],slide(previous_frame['board'],action)[0])
                        self.assertEqual(preview['features'],board_features(preview['board']))
