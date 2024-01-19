from layers import NN, set_shift
import numpy as np
from nums import numbers

nn = NN(15, 10)

for _ in range(50):
    for i in range(10):
        nn.set_input(np.matrix(numbers[i]))
        nn.update(i)

for i in range(10):
    nn.set_input(np.matrix(numbers[i]))
    print(nn.answers())
