import numpy as np
from typing_extensions import Self
from utils import Activation, Initialization, fixinf
from settings import alpha

class Layer:
    def __repr__(self) -> str:
        return f"Layer{self.width}"
    def __init__(self, prev: Self, width: int, initialization: Initialization = Initialization.He, activation: Activation = Activation.ReLU):
        self.width = width
        self.prev = prev
        self.ks: np.ndarray = initialization(self.prev.width, self.width)
        self.bs: np.ndarray = initialization(1, self.width)
        self.answers = None
        self.activation = activation
    def _request(self) -> np.ndarray:
        self.prev._ans()
        z = np.matmul(self.prev.answers, self.ks)+self.bs
        ans = self.activation.forward(z)
        print(repr(self), ans)
        return ans
    def _ans(self):
        if self.answers is None:
            self.answers = self._request()
    def _update(self, mul: np.ndarray):
        djdz = np.multiply(mul, self.activation.backward(self.answers))
        djdks = np.matmul(self.prev.answers.T, djdz)
        self.prev._update(np.matmul(djdz, self.ks.T))
        self.bs -= alpha*djdz
        self.ks -= alpha*djdks
    def _clear(self):
        self.answers = None
        self.prev._clear()
class ILayer(Layer):
    def __init__(self, width: int):
        self.width = width
        self.answers = np.zeros((1, self.width))
    def _update(self, mul: np.ndarray):pass
    def _clear(self):pass
    def set_input(self, input: np.ndarray):
        self.answers = input.reshape((1, self.width))
class OLayer(Layer):
    def __init__(self, prev: Self, width: int, initialization: Initialization = Initialization.He, activation: Activation = Activation.Sigmoid):
        super().__init__(prev, width, initialization, activation)
    def ans(self):
        super()._ans()
        return self.answers
    def clear(self):return super()._clear()
    def update(self, answers: np.ndarray):
        self._ans()
        real_answers = answers.reshape((1, self.width))
        mul = fixinf((1-real_answers)/(1-self.answers))-fixinf(real_answers/self.answers)
        super()._update(mul)
        self.clear()

    def loss(self, answers: np.ndarray) -> float:
        self._ans()
        real_answers = answers.reshape((1, self.width))
        loss = -np.multiply(real_answers, fixinf(np.log(self.answers))) - np.multiply(1-real_answers, fixinf(np.log(1-self.answers)))
        return loss.sum()