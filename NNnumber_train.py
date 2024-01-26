from layers import NN, set_shift
from datetime import datetime
import pickle
import pandas as pd

samples = pd.read_csv("./samples/mnist_train.csv")
answers = samples["label"].copy()
samples = samples / 255
samples["label"] = answers

tests = pd.read_csv("./samples/mnist_test.csv")
answers = tests["label"].copy()
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
    last_answer = start
    print(f"training {n} samples out of 60'000 at {start.time().isoformat("seconds")}")
    portion = samples.sample(n, replace=True)
    done = 0
    losses = 0
    for _, data in portion.iterrows():
        answer = data.pop("label")
        input = data.to_numpy().reshape((28, 28))
        nn.set_input(input)
        losses += nn.update(int(answer))
        done += 1
        if (datetime.now()-last_answer).total_seconds()>30:
            last_answer = datetime.now()
            print(f"{100*done/n:.1f}% of samples are done")
    print(f"training complete in {(datetime.now()-start).total_seconds():.2f}s")
    print(f"cost is {losses/n:.2f}")
    nn.set_input(prev_state)
    return losses/n

def draw(nn: NN, answer: int):
    data = tests[tests["label"] == answer].sample(1)
    data.pop("label")
    nn.set_input(data.to_numpy().reshape((28, 28)))

def save_nn(nn: NN, path: str = "model.pkl"):
    prev_state = nn.inputs.copy()
    nn.clear_input()
    with open(path, "wb") as f:
        pickle.dump(nn, f)
    nn.set_input(prev_state)

def load_nn(path: str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)