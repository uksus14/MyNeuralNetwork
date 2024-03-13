from network_part import NetworkPart
from neural_layers import ILayer
from data_layers import LayerScheme, OLayerScheme

class FC(NetworkPart):
    def __init__(self, input: int, *layers: LayerScheme|OLayerScheme):
        print(f"initializing a fc with layers widths of {input}, {', '.join(map(repr, layers))}")
        self.i = ILayer(input)
        super().__init__(*layers)