import numpy as np
from math import sqrt
from enum import Enum
from typing import Callable

BIG_NUMBER = 9999999

def he_init(inputs: int, outputs: int) -> np.ndarray:
    return np.random.randn(inputs, outputs)*(2/inputs)**.5
def xavier_init(inputs: int, outputs: int) -> np.ndarray:
    return 2*sqrt(6)*(np.random.rand(inputs, outputs)-.5)/sqrt(inputs+outputs)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1+fixinf(np.exp(-z)))
def sigmoid_der(answers: np.ndarray) -> np.ndarray:
    return np.multiply(fixinf(1/answers)-1, np.square(answers))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0)
def relu_der(answers: np.ndarray) -> np.ndarray:
    return relu(answers)

def fixinf(arr: np.ndarray) -> np.ndarray:
    prev = arr.copy()
    arr[arr == np.inf] = BIG_NUMBER
    arr[arr == -np.inf] = -BIG_NUMBER
    # if (prev != arr).any(): 20/0
    return arr

class Activation(Enum):
    ReLU = (relu, relu_der)
    Sigmoid = (sigmoid, sigmoid_der)
    @property
    def backward(self) -> Callable:
        return self.value[1]
    @property
    def forward(self) -> Callable:
        return self.value[0]

class Initialization(Enum):
    He = he_init
    Xavier = xavier_init
    def __call__(self, prev: np.ndarray, next: np.ndarray) -> np.ndarray:
        return self.value(prev, next)