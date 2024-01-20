from layers import NN, set_shift
import numpy as np
from nums import numbers, binary

def get_nn(*hidden: int, repetitions: int = 100, shift: float = None) -> NN:
    if shift is not None:
        set_shift(shift)
    nn = NN(15, 10, hidden)

    for _ in range(repetitions):
        for i in range(10):
            nn.set_input(np.matrix(numbers[i]))
            nn.update(i)
    return nn