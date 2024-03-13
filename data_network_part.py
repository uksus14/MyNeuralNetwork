from dataclasses import dataclass
from convolution import Conv
from neural import FC
from data_layers import LayerScheme, OLayerScheme, CLayerScheme, OCLayerScheme

@dataclass
class ConvScheme:
    layers: list[CLayerScheme|OCLayerScheme]
    def create(self) -> Conv:
        return Conv(*self.layers)

@dataclass
class FCScheme:
    input: int
    hidden: list[LayerScheme|OLayerScheme]
    def create(self) -> FC:
        return FC(self.input, *self.hidden)