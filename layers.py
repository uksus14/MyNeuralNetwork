import numpy as np
from typing_extensions import Self
from utils import Activation, Initialization

alpha = .005
shift = lambda:None
def set_shift(alpha):
    global shift
    def shift_(x: float, dx: float) -> float:
        return x-alpha*dx
    shift = np.vectorize(shift_)
set_shift(alpha)

class Layer:
    all_layers = []
    def __init__(self, width: int, prev: Self, initialization: Initialization = Initialization.He, activation: Activation = Activation.ReLU):
        self.width = width
        self.prev = prev
        self.ks: np.ndarray = initialization(self.prev.width, self.width)
        self.bs: np.ndarray = np.zeros((1, self.width))
        self.answers = None
        self.activation = activation

        self.i = len(self.all_layers)
        self.all_layers.append(self)
    def request(self) -> np.ndarray:
        self.prev.ans()
        z = np.matmul(self.prev.answers, self.ks)+self.bs
        ans = self.activation.forward(z)
        return ans
    def ans(self):
        if self.answers is None:
            self.answers = self.request()
    def update(self, mul: np.ndarray):
        djdz = np.multiply(mul, self.activation.backward(self.answers))
        djdks = np.matmul(self.prev.answers.T, djdz)
        self.prev.update(np.matmul(djdz, self.ks.T))
        self.bs = shift(self.bs, djdz)
        self.ks = shift(self.ks, djdks)
    def clear(self):
        self.answers = None
        self.prev.clear()
class ILayer(Layer):
    def __init__(self, width: int):
        self.width = width
        self.answers = np.zeros((1, self.width))
    def update(self, mul: np.ndarray):pass
    def clear(self):pass
    def set_input(self, input: np.ndarray):
        self.answers = input.reshape((1, self.width))
class OLayer(Layer):
    def __init__(self, width: int, prev: Self, initialization: Initialization = Initialization.He):
        super().__init__(width, prev, initialization, Activation.Sigmoid)
    def update(self, answers: np.ndarray) -> float:
        self.ans()
        real_answers = answers.reshape((1, self.width))
        mul = (1-real_answers)/(1-self.answers)-real_answers/self.answers
        super().update(mul)
    def loss(self, answers: np.ndarray) -> float:
        self.ans()
        real_answers = answers.reshape((1, self.width))
        loss = -np.multiply(real_answers, np.log(self.answers)) - np.multiply(1-real_answers, np.log(1-self.answers))
        return loss.sum()

class NN:
    def __init__(self, input: int|tuple[int, int], output: int, hidden: list[int]=[], initialization: Initialization = Initialization.He, activation: Activation = Activation.ReLU):
        print(f"initializing an nn with layers widths of {', '.join(map(str, [input, *hidden, output]))}")
        self.input_shape = input
        if not isinstance(input, int):
            input = self.input_shape[0]*self.input_shape[1]
        self.i = ILayer(input)
        self.layers = [self.i]
        for width in hidden:
            self.layers.append(Layer(width, self.layers[-1], initialization, activation))
        self.o = OLayer(output, self.layers[-1], initialization)
        self.layers.append(self.o)
    def _flatten_input_coords(self, x: int, y: int = None) -> int:
        if y is not None:
            x = y*self.input_shape[1]+x
        return x
    def get_input(self, x: int, y: int = None) -> float|None:
        index = self._flatten_input_coords(x, y)
        if 0 <= index < self.i.width:
            return self.inputs[0, index]
        return None
    def clear_answers(self):
        self.o.clear()
    def clear_input(self):
        self.set_input(np.zeros((1, self.i.width)))
    def set_input(self, input: np.ndarray):
        self.i.set_input(input.reshape((1, self.i.width)))
        self.clear_answers()
    def change_input(self, value: int, x: int, y: int = None):
        index = self._flatten_input_coords(x, y)
        if self.inputs[0, index] == value:
            return
        self.inputs[0, index] = value
        self.clear_answers()
    def update(self, answers: int|np.ndarray):
        answers = self._get_answers(answers)
        self.o.update(answers)
        self.clear_answers()
    def loss(self, answers: int|np.ndarray):
        answers = self._get_answers(answers)
        return self.o.loss(answers)
    def _get_answers(self, answers: int|np.ndarray) -> np.ndarray:
        if isinstance(answers, int):
            new_answers = np.zeros((1, self.o.width))
            new_answers[0, answers] = 1
            answers = new_answers
        return answers
    def answers(self) -> np.ndarray:
        self.o.ans()
        return self.o.answers
    @property
    def inputs(self) -> np.ndarray:
        return self.i.answers