from collections import deque
from random import sample as random_sample

import numpy as np
from torch import BoolTensor, FloatTensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._buffer = deque(maxlen=capacity)

    def push(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor, BoolTensor]:
        batch = random_sample(self._buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            FloatTensor(state),
            FloatTensor(action),
            FloatTensor(reward).unsqueeze(1),
            FloatTensor(next_state),
            BoolTensor(done).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self._buffer)
