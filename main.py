from layers import NN, set_shift
import numpy as np
from nums import numbers, binary

nn = NN(4, 2)

for _ in range(0):
    for i in range(10):
        nn.set_input(np.matrix(binary[i]))
        nn.update(i)

for i in range(2):
    nn.set_input(np.matrix(binary[i]))
    print(i)
    print(nn.answers())
    nn.update(i)
    print(nn.answers())
