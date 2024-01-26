from layers import NN, set_shift
from datetime import datetime
import pickle
import pandas as pd

samples = pd.read_csv("./samples/mnist_train.csv")
answers = samples.pop("label")
samples = samples / 255
samples["label"] = answers

tests = pd.read_csv("./samples/mnist_test.csv")
answers = tests.pop("label")
tests = tests / 255
tests["label"] = answers

def get_nn(*hidden: int, shift: float = None) -> NN:
    if shift is not None:
        set_shift(shift)
    nn = NN(28*28, 10, hidden)
    return nn

def train(nn: NN, n: int = 1024):
    prev_state = nn.inputs.copy()
    start = datetime.now()
    print(f"training {n} samples out of 60'000 at {start.time()}")
    portion = samples.sample(n, replace=True)
    losses = 0
    for _, data in portion.iterrows():
        answer = data.pop("label")
        input = data.to_numpy().reshape((28, 28))
        nn.set_input(input)
        losses += nn.update(int(answer))
    print(f"training complete in {(datetime.now()-start).total_seconds():.2f}s")
    print(f"cost is {losses/n}")
    nn.set_input(prev_state)
    return losses/n

def draw_zero(nn: NN):
    zero = tests[tests["label"] == 0].sample(1)
    for _, data in zero.iterrows():
        _ = data.pop("label")
        data = data.to_numpy().reshape((28, 28))
        nn.set_input(data)

def save_nn(nn: NN, path: str = "model.pkl"):
    prev_state = nn.inputs.copy()
    nn.clear_input()
    with open(path, "wb") as f:
        pickle.dump(nn, f)
    nn.set_input(prev_state)

def load_nn(path: str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)