from dataclasses import dataclass
from data_network_part import ConvScheme, FCScheme
from nn import NN

@dataclass
class NNScheme:
    neural: FCScheme
    conv: ConvScheme = None
    def create(self) -> NN:
        return NN(self.neural, self.conv)