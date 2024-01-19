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
        self.ans()
        djdz = np.multiply(mul, np.multiply(self.answers[1], np.multiply(self.answers[0], self.answers[0])-self.answers[0]))

        self.bs += shift(self.bs, djdz)
        self.ks += shift(self.ks, np.matmul(self.prev.answers[0].T, djdz))
        self.prev.update(np.matmul(djdz, self.ks.T))
class ILayer(Layer):
    def __init__(self, width: int):
        self.width = width
        self.answers = np.zeros(())
    def request(self) -> tuple[ndarray, ndarray]:
        return self.answers
    def update(self, mul: np.ndarray):pass
    def set_input(self, input: np.ndarray):
        self.answers = input.reshape(())
class OLayer(Layer):
    def update(self, answers):
        #cost here
        return super().update(-answers)

class NN:
    def __init__(self) -> None:
        pass