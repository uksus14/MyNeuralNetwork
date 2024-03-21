from data_network_part import ConvScheme, FCScheme
from data_layers import OCLayerScheme
import numpy as np

class NN:
    def __init__(self, neural: FCScheme, conv: ConvScheme = None):
        self.neural = neural.create()
        if conv is None:
            conv = ConvScheme([OCLayerScheme(self.neural.i.width)])
        self.conv = conv.create()
        self.i = self.conv.i
        def clear_conv():
            self.conv.o.clear()
        def update_conv(answers: np.ndarray):
            self.conv.o.update(answers)
        def request_conv() -> np.ndarray:
            return self.conv.o.ans().reshape(1, self.neural.i.width)
        self.neural.i._clear = clear_conv
        self.neural.i._update = update_conv
        self.neural.i._request = request_conv
        self.o = self.neural.o
    def clear_answers(self):
        self.o.clear()
    def set_input(self, input: np.ndarray):
        self.i.set_input(input)
        self.clear_answers()
    def clear_input(self, height: int, width: int, depth: int = 1):
        self.conv.i.set_input(np.zeros((height, width, depth)))
    def _get_answers(self, answer: int|np.ndarray) -> np.ndarray:
        if not isinstance(answer, int):
            return answer
        answers = np.zeros((1, self.o.width))
        answers[0, answer] = 1
        return answers
    def loss(self, answers: int|np.ndarray) -> float:
        answers = self._get_answers(answers)
        return self.o.loss(answers)
    def change_input(self, value: int, x: int, y: int = 0):
        if self.inputs[y, x] == value:
            return
        self.inputs[y, x] = value
        self.clear_answers()
    def update(self, answers: int|np.ndarray):
        answers = self._get_answers(answers)
        self.o.update(answers)
        self.clear_answers()
    def answers(self) -> np.ndarray:
        return self.o.ans()
    @property
    def inputs(self) -> np.ndarray:

        return self.i.answers