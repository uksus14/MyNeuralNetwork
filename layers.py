import numpy as np
from typing_extensions import Self

BIG_NUMBER = 9999999

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
    def __init__(self, width: int, prev: Self):
        self.width = width
        self.prev = prev
        self.ks: np.ndarray = np.random.rand(self.prev.width, self.width)-.5
        self.bs: np.ndarray = np.random.rand(1, self.width)-.5
        self.answers = None

        self.i = len(self.all_layers)
        self.all_layers.append(self)
    def request(self) -> tuple[np.ndarray, np.ndarray]:
        self.prev.ans()
        z = np.matmul(self.prev.answers[0], self.ks)+self.bs
        exp = np.exp(-z)
        exp[exp == np.inf] = BIG_NUMBER
        exp[exp == -np.inf] = -BIG_NUMBER
        return 1/(1+exp), exp
    def ans(self):
        if self.answers is None:
            self.answers = self.request()
    def update(self, mul: np.ndarray):
        djdz = np.multiply(mul, np.multiply(self.answers[1], np.multiply(self.answers[0], self.answers[0])))
        djdks = np.matmul(self.prev.answers[0].T, djdz)
        self.prev.update(np.matmul(djdz, self.ks.T))
        self.bs = shift(self.bs, djdz)
        self.ks = shift(self.ks, djdks)
    def clear(self):
        self.answers = None
        self.prev.clear()
class ILayer(Layer):
    def __init__(self, width: int):
        self.width = width
        self.answers = [np.zeros((1, self.width)), np.zeros((1, self.width))]
    def update(self, mul: np.ndarray):pass
    def clear(self):pass
    def set_input(self, input: np.ndarray):
        self.answers[0] = input.reshape((1, self.width))
class OLayer(Layer):
    def update(self, answers: np.ndarray) -> float:
        self.ans()
        real_answers = answers.reshape((1, self.width))
        mul = (1-real_answers)/(1-self.answers[0])-real_answers/self.answers[0]
        loss = -np.multiply(real_answers, np.log(self.answers[0])) - np.multiply(1-real_answers, np.log(1-self.answers[0]))
        super().update(mul)
        return loss.sum()

class NN:
    def __init__(self, input: int|tuple[int, int], output: int, hidden: list[int]=[]):
        print(f"initializing an nn with layers widths of {[input, *hidden, output]}")
        self.input_shape = input
        if not isinstance(input, int):
            input = self.input_shape[0]*self.input_shape[1]
        self.i = ILayer(input)
        self.layers = [self.i]
        for width in hidden:
            self.layers.append(Layer(width, self.layers[-1]))
        self.o = OLayer(output, self.layers[-1])
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
        if isinstance(answers, int):
            answers = self._get_answers(answers)
        loss = self.o.update(answers)
        self.clear_answers()
        return loss
    def _get_answers(self, index: int) -> np.ndarray:
        answers = np.zeros((1, self.o.width))
        answers[0, index] = 1
        return answers
    def answers(self) -> np.ndarray:
        self.o.ans()
        return self.o.answers[0]
    @property
    def inputs(self) -> np.ndarray:
        return self.i.answers[0]