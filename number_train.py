import pandas as pd
def get_data():
    try:
        samples = pd.read_csv("./samples/mnist_train.csv")
    except FileNotFoundError:
        raise Exception("Download MNist samples from the link in the samples folder")
    answers = samples["label"].copy()
    samples = samples / 255
    samples["label"] = answers

    tests = pd.read_csv("./samples/mnist_test.csv")
    answers = tests["label"].copy()
    tests = tests / 255
    tests["label"] = answers
    return samples, tests