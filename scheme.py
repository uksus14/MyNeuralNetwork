from data_nn import NNScheme
from data_network_part import FCScheme, ConvScheme
from data_layers import LayerScheme, OLayerScheme, OCLayerScheme, CLayerScheme

scheme = NNScheme(
    FCScheme(6*6*64, [
        LayerScheme(500),
        LayerScheme(100),
        OLayerScheme(10)
    ]),
    ConvScheme([ # 28x28x1
        CLayerScheme(5, 8, 5), # 20x20x8
        CLayerScheme(4, 16, 4), # 14x14x16
        CLayerScheme(3, 32, 3), # 10x10x16
        CLayerScheme(3, 64, 3), # 6x18x16
        OCLayerScheme(6*6*64) # 6x6x128
    ])
)