from layers import NN, set_shift
import numpy as np
from nums import get_train, get_test
from time import time
import pickle

def get_nn(*hidden: int, shift: float = None) -> NN:
    if shift is not None:
        set_shift(shift)
    nn = NN(15, 11, hidden)
    return nn

def train(nn: NN, samples: int = 1024):
    prev_state = nn.inputs.copy()
    start = time()
    print(f"training {samples} samples out of 60'000")
    X, y = get_train()
    print(X, y)
    for _ in range(samples):
        nn.set_input()
        nn.update(10)
    print(f"training complete in {time()-start:.2f}s")
    nn.set_input(prev_state)

# def test(nn: NN):
#     prev_state = nn.inputs.copy()
#     start = time()
#     print(f"testing 10'000 samples")
#     X, y = get_test()
#     for 
#         nn.set_input()
#         nn.update(10)
#     print(f"training complete in {time()-start:.2f}s")
#     nn.set_input(prev_state)

def save_nn(nn: NN, path: str = "model.pkl"):
    prev_state = nn.inputs.copy()
    nn.clear_input()
    with open(path, "wb") as f:
        pickle.dump(nn, f)
    nn.set_input(prev_state)

def load_nn(path: str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)