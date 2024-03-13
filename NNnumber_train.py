from nn import NN
from data_nn import NNScheme
from number_train import get_data
from datetime import datetime
import pickle

samples, tests = get_data()

def get_nn(scheme: NNScheme) -> NN:
    return scheme.create()

def train(nn: NN, n: int = 1024):
    prev_state = nn.inputs.copy()
    start = datetime.now()
    last_answer = start
    print(f"training {n} samples out of 60'000 at {start.time().isoformat("seconds")}")
    portion = samples.sample(n, replace=True)
    done = 0
    losses = 0
    for _, data in portion.iterrows():
        answer = int(data.pop("label"))
        input = data.to_numpy().reshape((28, 28))
        nn.set_input(input)
        losses += nn.loss(answer)
        nn.update(answer)
        done += 1
        if (datetime.now()-last_answer).total_seconds()>30:
            last_answer = datetime.now()
            print(f"{100*done/n:.1f}% of samples are done in {(last_answer-start).total_seconds():.1f}s. Estimated time left is {n*(last_answer-start).total_seconds()/done-(last_answer-start).total_seconds():.1f}s ending at {(start+n*(last_answer-start)/done).time().isoformat("seconds")}")
    end = datetime.now()
    print(f"training complete in {(end-start).total_seconds():.2f}s at {end.time().isoformat("seconds")}")
    print(f"cost is {losses/n:.2f}")
    nn.set_input(prev_state)
    return losses/n

def draw(nn: NN, answer: int):
    data = tests[tests["label"] == answer].sample(1)
    data.pop("label")
    nn.set_input(data.to_numpy().reshape((28, 28)))

def save_nn(nn: NN, path: str = "model.pkl"):
    prev_state = nn.inputs.copy()
    nn.clear_input(28, 28)
    with open(path, "wb") as f:
        pickle.dump(nn, f)
    nn.set_input(prev_state)

def load_nn(path: str = "model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)