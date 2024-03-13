from convolution_layers import ICLayer
from network_part import NetworkPart
from data_layers import CLayerScheme, OCLayerScheme

class Conv(NetworkPart):
    def __init__(self, *layers: CLayerScheme|OCLayerScheme):
        print(f"initializing a convolution with convolution layers {layers}")
        self.i = ICLayer()
        super().__init__(*layers)