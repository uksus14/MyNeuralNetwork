from abc import ABC
import numpy as np
from unions import layer_scheme, in_layer, mid_layer, out_layer
class NetworkPart(ABC):
    def __init__(self, *layers: layer_scheme):
        self.i: in_layer
        for layer in layers[:-1]:
            if layer._is_output:
                raise ValueError("all layers except the last must be out layers")
        if not layers[-1]._is_output:
            raise ValueError("last layer must be out layer")
        self.layers: list[in_layer|mid_layer|out_layer] = [self.i]
        for layer in layers:
            self.layers.append(layer.create(self.layers[-1]))
        self.o: out_layer = self.layers[-1]
    def clear_answers(self):
        self.o.clear()
    def set_input(self, input: np.ndarray):
        self.i.set_input(input)
        self.clear_answers()
    def update(self, answers: np.ndarray):
        self.o.update(answers)
    def answers(self) -> np.ndarray:
        return self.o.ans()
    @property
    def inputs(self) -> np.ndarray:
        return self.i.answers