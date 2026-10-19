"""2048 environment matching docs/2048/engine.js, including seeded spawns."""
from functools import lru_cache

ACTIONS = ('Up', 'Right', 'Down', 'Left')


class Random:
    """Mulberry32, with the same unsigned arithmetic as the browser."""
    def __init__(self, seed):
        self.state = seed & 0xffffffff

    def __call__(self):
        self.state = (self.state + 0x6D2B79F5) & 0xffffffff
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xffffffff
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xffffffff)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff) / 4294967296


LINES = tuple(tuple(tuple(
    i * 4 + line if action == 0 else line * 4 + 3 - i if action == 1
    else (3 - i) * 4 + line if action == 2 else line * 4 + i
    for i in range(4)) for line in range(4)) for action in range(4))


@lru_cache(maxsize=65536)
def merge(row):
    values = [v for v in row if v]
    result, reward, i = [], 0, 0
    while i < len(values):
        if i + 1 < len(values) and values[i] == values[i + 1]:
            result.append(values[i] + 1)
            reward += 2 ** (values[i] + 1)
            i += 2
        else:
            result.append(values[i])
            i += 1
    return tuple(result + [0] * (4 - len(result))), reward


def slide(board, action):
    if action not in range(4):
        raise ValueError('Action must be 0..3')
    after, reward = list(board), 0
    for ids in LINES[action]:
        row, gain = merge(tuple(board[i] for i in ids))
        reward += gain
        for i, v in zip(ids, row):
            after[i] = v
    return after, reward, after != list(board)


def legal(board):
    return [a for a in range(4) if slide(board, a)[2]]


class Game:
    def __init__(self, seed=42):
        self.random = Random(seed)
        self.board = [0] * 16
        self.score = self.moves = 0
        self.spawn()
        self.spawn()

    def spawn(self):
        empty = [i for i, v in enumerate(self.board) if not v]
        if not empty:
            return None
        index = empty[int(self.random() * len(empty))]
        self.board[index] = 1 if self.random() < .9 else 2
        return {'index': index, 'value': 2 ** self.board[index]}

    def step(self, action):
        after, reward, changed = slide(self.board, action)
        spawned = None
        if changed:
            self.board = after
            self.score += reward
            self.moves += 1
            spawned = self.spawn()
        return dict(board=self.board.copy(), reward=reward, changed=changed,
                    spawned=spawned, done=not legal(self.board),
                    reached2048=max(self.board) >= 11)
