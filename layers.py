import numpy as np
from typing_extensions import Self
import cmath

from numpy import ndarray

ALPHA = .001
def sigmoid_(x: float) -> float:
    return 1/(1+cmath.exp(-x))
def shift_(x: float, dx: float) -> float:
    return x-ALPHA*dx
sigmoid = np.vectorize(sigmoid_)
shift = np.vectorize(shift_)

class Layer:
    def __init__(self, width: int, prev: Self):
        self.width = width
        self.prev = prev
        self.ks: np.ndarray = np.random.rand(())
        self.bs: np.ndarray = np.random.rand(())
        self.answers = None
    def request(self) -> tuple[np.ndarray, np.ndarray]:
        self.prev.ans()
        z = np.matmul(self.prev.answers[0], self.ks)+self.bs
        return sigmoid(z), z
    def ans(self):
        if self.answers is None:
            self.answers = self.request()
    def update(self, mul: np.ndarray):
        djdz = np.multiply(mul, np.multiply(self.answers[1], np.multiply(self.answers[0], self.answers[0])-self.answers[0]))

        self.bs += shift(self.bs, djdz)
        self.ks += shift(self.ks, np.matmul(self.prev.answers[0].T, djdz))
        self.prev.update(np.matmul(djdz, self.ks.T))
    def clear(self):
        self.answers = None
        self.prev.clear()
class ILayer(Layer):
    def __init__(self, width: int):
        self.width = width
        self.answers = np.zeros(())
    def request(self) -> tuple[ndarray, ndarray]:
        return self.answers
    def update(self, mul: np.ndarray):pass
    def clear(self):pass
    def set_input(self, input: np.ndarray):
        self.answers = input.reshape(())
class OLayer(Layer):
    def update(self, answers: np.ndarray):
        self.ans()
        real_answers = answers.reshape(())
        mul = (self.answers[0]-real_answers)/(self.answers[0]*(1-self.answers[0]))
        return super().update(mul)

class NN:
    def __init__(self, input: int, output: int, hidden: list[int]=[]):
        self.i = ILayer(input)
        self.layers = [self.i]
        for width in hidden:
            self.layers.append(Layer(width, self.layers[-1]))
        self.o = OLayer(output, self.layers[-1])
        self.layers.append(self.o)
    def clear(self):
        self.o.clear()
    def set_input(self, input: np.ndarray):
        self.i.set_input(input)
        self.clear()
    def update(self, answers: int|np.ndarray):
        if isinstance(answers, int):
            answers = self.get_answers(answers)
        self.o.update(answers)
        self.clear()
    def get_answers(self, index: int) -> np.ndarray:
        answers = np.zeros(())
        answers[0, index] = 1
        return answers