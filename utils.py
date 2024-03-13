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
    def __repr__(self) -> str:
        return self.name

class Initialization(Enum):
    He = he_init
    Xavier = xavier_init
    def __call__(self, prev: np.ndarray, next: np.ndarray) -> np.ndarray:
        return self.value(prev, next)
    def __repr__(self) -> str:
        return self.name

def get_tile(input: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    return input[y:y+height, x:x+width]

def get_conv_shape(input: tuple[int, int], tile: tuple[int, int], padding: int=0) -> tuple[int, int]|tuple[int, int, int]:
    output = [i+2*padding-k+1 for i, k in zip(input, tile)][:2]
    if len(tile) > 2 and tile[2] > 1:
        output.append(tile[2])
    elif len(input) > 2 and input[2] > 1:
        output.append(input[2])
    return tuple(output)
def reverse_conv_shape(output: tuple[int, int], tile: tuple[int, int], padding: int=0) -> tuple[int, int]:
    return get_conv_shape(output, [-k for k in tile], padding)[:2]
def kernel_conv_shape(input: tuple[int, int], output: tuple[int, int], padding: int=0) -> tuple[int, int]|tuple[int, int, int]:
    return get_conv_shape(output, input, padding)

def _pad(input: np.ndarray, padding: int) -> np.ndarray:
    if padding == 0: return input
    return np.pad(input, ((padding, padding), (padding, padding)), mode="constant", constant_values=0)#TODO breaks

def _for_tiles(input: np.ndarray, tilew: int, tileh: int, func: callable, padding: int=0, *args) -> np.ndarray:
    res = np.zeros(get_conv_shape(input.shape, (tileh, tilew), padding))
    padded = _pad(input, padding)

    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            tile = get_tile(padded, x, y, tilew, tileh)
            res[y, x] = func(tile, *args)
    return res
def maxpool_func(tile: np.ndarray) -> float:
    return np.max(tile)
def avrpool_func(tile: np.ndarray) -> float:
    return np.mean(tile)

def convolute(input: np.ndarray, kernel: np.ndarray, bs: np.ndarray, activation: Activation, padding: int=0) -> np.ndarray:
    ans = np.zeros(get_conv_shape(input.shape, kernel.shape, padding))

    def convolute_func(tile: np.ndarray, kernel: np.ndarray) -> float:
        return np.sum(np.multiply(tile, kernel))
    
    for kslice in range(kernel.shape[2]):
        for islice in range(input.shape[2]):
            ans[:, :, kslice] += _for_tiles(input[:, :, islice], kernel.shape[0], kernel.shape[1], convolute_func, padding, kernel[:, :, kslice])
        ans[:, :, kslice] += bs[kslice]
    ans = activation.forward(ans)
    return ans

def kernel_der(mul: np.ndarray, input: np.ndarray, kernel_shape: tuple[int, int, int], padding: int=0) -> np.ndarray:
    padded = _pad(input, padding)
    djdk = np.zeros(kernel_shape)
    for islice in range(padded.shape[2]):
        for kslice in range(kernel_shape[2]):
            for y in range(mul.shape[0]):
                for x in range(mul.shape[1]):
                    tile = get_tile(padded[:, :, islice], x, y, kernel_shape[0], kernel_shape[1])
                    djdk[:, :, kslice] += tile*mul[y, x, kslice]
    return djdk

def convolution_der(mul: np.ndarray, kernel: np.ndarray, inc: int, padding: int=0) -> np.ndarray:
    ishape = list(reverse_conv_shape(mul.shape, kernel.shape, padding))
    ishape.append(inc)
    djdi = np.zeros(ishape)
    for islice in range(inc):
        for kslice in range(kernel.shape[2]):
            for y in range(mul.shape[0]):
                for x in range(mul.shape[1]):
                    djdi[y:y+kernel.shape[0], x:x+kernel.shape[1], islice] += kernel[:, :, kslice]*mul[y, x, kslice]
    return djdi


def maxpool(input: np.ndarray, pool: tuple[int, int]) -> np.ndarray:
    ans = np.zeros(get_conv_shape(input.shape, pool))
    for islice in range(input.shape[2]):
        ans[:, :, islice] += _for_tiles(input[:, :, islice], *pool, maxpool_func)
    return ans
def maxpool_der(mul: np.ndarray, input: np.ndarray, pool: tuple[int, int]) -> np.ndarray:
    djdc = np.zeros_like(input)

    for islice in range(input.shape[2]):
        for y in range(mul.shape[0]):
            for x in range(mul.shape[1]):
                tile = input[y:y+pool[0], x:x+pool[1], islice]
                iy, ix = np.unravel_index(np.argmax(tile), tile.shape)
                djdc[iy+y, ix+x, islice] += mul[y, x, islice]
    return djdc

def avrpool(input: np.ndarray, pool: tuple[int, int]) -> np.ndarray:
    ans = np.zeros(get_conv_shape(input.shape, pool))
    for islice in range(input.shape[2]):
        ans[:, :, islice] += _for_tiles(input[:, :, islice], *pool, avrpool_func, 0)
    return ans
def avrpool_der(mul: np.ndarray, input: np.ndarray, pool: tuple[int, int]) -> np.ndarray:
    djdc = np.zeros_like(input)

    for y in range(mul.shape[0]):
        for x in range(mul.shape[1]):
            tile = mul[y, x]/(pool[0]*pool[1])
            djdc[y:y+pool[0], x:x+pool[1]] += tile
    return djdc

class Pool(Enum):
    Max = (maxpool, maxpool_der)
    Average = (avrpool, avrpool_der)
    No = (lambda x: x, lambda x, y, z: x)
    @property
    def backward(self) -> callable:
        return self.value[1]
    @property
    def forward(self) -> callable:
        return self.value[0]
    def __repr__(self) -> str:
        return self.name