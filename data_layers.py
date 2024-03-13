from utils import Activation, Initialization, Pool
from convolution_layers import CLayer, OCLayer
from neural_layers import Layer, OLayer
from dataclasses import dataclass

@dataclass
class CLayerScheme:
    kernel: int
    outc: int
    pool_size: int
    pool: Pool = Pool.Max
    activation: Activation = Activation.ReLU
    padding: int = 0
    _is_output = False
    def create(self, prev: CLayer) -> CLayer:
        return CLayer(prev, self.kernel, self.outc, self.pool_size, self.pool, self.activation, self.padding)
@dataclass
class OCLayerScheme:
    len: int
    pool: Pool = Pool.Average
    _is_output = True
    def create(self, prev: CLayer) -> OCLayer:
        return OCLayer(prev, self.len, self.pool)

@dataclass
class LayerScheme:
    width: int
    initialization: Initialization = Initialization.He
    activation: Activation = Activation.ReLU
    _is_output = False
    def create(self, prev: Layer) -> Layer:
        return Layer(prev, self.width, self.initialization, self.activation)
@dataclass
class OLayerScheme(LayerScheme):
    _is_output = True
    def create(self, prev: Layer) -> OLayer:
        return OLayer(prev, self.width, self.initialization, self.activation)