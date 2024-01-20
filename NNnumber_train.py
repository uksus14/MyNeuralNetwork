from layers import NN, set_shift
import numpy as np
from nums import numbers, binary
from time import time
import pickle

def get_nn(*hidden: int, shift: float = None) -> NN:
    if shift is not None:
        set_shift(shift)
    nn = NN(15, 11, hidden)
    return nn

def hash_number(number: np.ndarray):
    answer = 0
    for i in range(number.shape[0]):
        for j in range(number.shape[1]):
            answer = (answer+number[i, j])*2
    return answer

hash_numbers = [hash_number(np.matrix(number)) for number in numbers]

def train(nn: NN, train_for: int = 100):
    prev_state = nn.inputs.copy()
    start = time()
    print(f"training {train_for} times")
    for _ in range(train_for):
        for i in range(10):
            nn.set_input(np.matrix(numbers[i]))
            nn.update(i)
        nothing = np.random.randint(2, size=(3, 5))
        while hash_number(nothing) in hash_numbers:
            nothing = np.random.randint(2, size=(3, 5))
        nn.set_input(nothing)
        nn.update(10)
    print(f"training complete in {time()-start:.2f}s")
    nn.set_input(prev_state)

def save_nn(nn: NN, path: str = "model.pkl"):
    prev_state = nn.inputs.copy()
    nn.clear_input()
    with open(path, "wb") as f:
        pickle.dump(nn, f)
    nn.set_input(prev_state)

def load_nn(path: str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)