"""Stage 4: afterstate CNN plus seven explicit board statistics."""
import copy
import numpy as np
import torch
from torch import nn
from src.game2048 import slide
from src.train_2048 import CONFIG, SNAKE_PATH
from src.train_2048_afterstate import AfterstateAgent, train


def board_features(board):
    ranks = [board[i] for i in SNAKE_PATH]
    violations = sum(max(0,b-a) for a,b in zip(ranks,ranks[1:]))
    return [board.count(0)/16, float(max(board)>0 and board[0]==max(board)),
            violations/max(1,sum(board)), *[slide(board,a)[1]/128 for a in range(4)]]


class HybridNetwork(nn.Module):
    def __init__(self, convolution):
        super().__init__()
        self.convolution = convolution
        self.summary = nn.Sequential(nn.Linear(7,16),nn.ReLU())
        self.head = nn.Sequential(nn.Linear(144,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self, inputs):
        spatial = self.convolution(inputs[:,:16])
        summary = self.summary(inputs[:,16:])
        return self.head(torch.cat([spatial,summary],dim=1))


class HybridAfterstateAgent(AfterstateAgent):
    board_features = staticmethod(board_features)

    def __init__(self, seed):
        super().__init__(seed)
        # Keep exactly the same initial convolution weights as the Stage 3 model.
        self.online = HybridNetwork(nn.Sequential(*list(self.online.children())[:6]))
        self.target = copy.deepcopy(self.online).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(),lr=CONFIG['learning_rate'])
        self.input_size = 23
        n = CONFIG['replay_capacity']
        self.states = np.zeros((n,23),dtype=np.float32)
        self.next_previews = np.zeros((n,4,23),dtype=np.float32)

    def encode(self, boards):
        return np.asarray([list(np.asarray(b)/16)+board_features(b) for b in boards],dtype=np.float32)


if __name__ == '__main__':
    train(stage=4,agent_factory=HybridAfterstateAgent)
