from data_layers import LayerScheme, OLayerScheme, CLayerScheme, OCLayerScheme
from convolution_layers import ICLayer, CLayer, OCLayer
from neural_layers import ILayer, Layer, OLayer
layer_scheme = CLayerScheme|OCLayerScheme|LayerScheme|OLayerScheme
in_layer = ICLayer|ILayer
mid_layer = CLayer|Layer
out_layer = OCLayer|OLayer