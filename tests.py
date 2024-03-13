from data_layers import OCLayerScheme
from convolution_layers import ICLayer

o = OCLayerScheme(2304).create(ICLayer())
o.