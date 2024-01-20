from layers import NN, set_shift
import numpy as np
from nums import numbers, binary
from time import time

def get_nn(*hidden: int, shift: float = None) -> NN:
    if shift is not None:
        set_shift(shift)
    nn = NN(15, 11, hidden)
    return nn

def train(nn: NN, train_for: int = 100):
    prev_state = nn.inputs.copy()
    start = time()
    print(f"training {train_for} times")
    for _ in range(train_for):
        for i in range(10):
            nn.set_input(np.matrix(numbers[i]))
            nn.update(i)
        nn.set_input(np.random.randint(2, size=(3, 5)))
        nn.update(10)
    print(f"training complete in {time()-start:.2f}s")
    nn.set_input(prev_state)