import numpy as np
from typing import Self
from settings import alpha
from utils import Activation
from utils import Pool, convolute, convolution_der, kernel_conv_shape, kernel_der, get_conv_shape

class CLayer:
    def __init__(self, prev: Self, kernel: int|tuple[int, int], outc: int, pool_size: int|tuple[int, int], pool: Pool=Pool.Max, activation: Activation = Activation.ReLU, padding: int = 0):
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(pool_size, int):
            pool_size = (pool_size, pool_size)
        self.kernels = np.random.randn(*kernel, outc)
        self.pool_size = pool_size
        self.pool = pool
        self.bs = np.random.randn(outc)
        self.outc = outc
        self.prev = prev
        self.padding = padding
        self.activation = activation
        self.answers = None
        self._conv_answers = None

    def _request(self) -> tuple[np.ndarray]:
        self.prev._ans()
        cans = convolute(self.prev.answers, self.kernels, self.bs, self.activation, self.padding)
        ans = cans
        if self.pool is not None:
            ans = self.pool.forward(cans, self.pool_size)
        print(repr(self), ans)
        return cans, ans
    def _update(self, mul: np.ndarray):
        djda = self.pool.backward(mul, self.answers, self.pool_size)
        djdz = self.activation.backward(djda)
        djdb = np.sum(djdz, axis=(0, 1))
        djdk = kernel_der(djdz, self._conv_answers, self.kernels.shape, self.padding)
        djdi = convolution_der(djdz, self.kernels, self.prev.outc, self.padding)
        self.prev._update(djdi)
        self.bs -= alpha*djdb
        self.kernels -= alpha*djdk
    def _ans(self):
        if self.answers is None:
            self._conv_answers, self.answers = self._request()
    def _clear(self):
        self.answers = None
        self.prev._clear()
    @property
    def oshape(self):
        return get_conv_shape(get_conv_shape(self.prev.oshape, self.kernels.shape, self.padding), self.pool_size)
class ICLayer(CLayer):
    def __init__(self):
        self.answers = None
    def _update(self, mul: np.ndarray):pass
    def _clear(self):pass
    def set_input(self, input: np.ndarray):
        if len(input.shape) == 2:
            input = input.reshape((*input.shape, 1))
        self.answers = input
    def _request(self) -> np.ndarray:
        print("No input set")
        return None
    @property
    def outc(self):
        return self.answers.shape[2]
    @property
    def oshape(self):
        return self.answers.shape
class OCLayer(CLayer):
    def __init__(self, prev: CLayer, output_len: int, pool: Pool=Pool.Average):
        self.output_len = output_len
        self.prev = prev
        self.pool = pool
        if self.output_len % self.prev.outc != 0:
            raise ValueError("output length must be divisible by outc")
        self.outc = self.prev.outc

        slice_len = self.output_len // self.outc
        for h in range(int(slice_len ** .5), 0, -1):
            if slice_len % h == 0:
                height = h
                break
        width = slice_len // height
        depth = self.prev.outc
        self._oshape = height, width, depth

        self.answers = None
    @property
    def oshape(self):
        return self._oshape
    @property
    def pool_size(self):
        return kernel_conv_shape(self.prev.oshape, self._oshape)[:2]
    def update(self, mul: np.ndarray):
        mul = mul.reshape(self.oshape)
        self._ans()
        djda = self.pool.backward(mul, self.prev.answers, self.pool_size)
        self.prev._update(djda)
        self.clear()
    def _ans(self):
        if self.answers is None:
            self.answers = self._request()
    def ans(self):
        self._ans()
        return self.answers
    def clear(self):return super()._clear()
    def _request(self) -> np.ndarray:
        self.prev._ans()
        ans = self.pool.forward(self.prev.answers, self.pool_size)
        return ans