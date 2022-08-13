import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.encoder import *
from ..neck.necks import *

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.encoder = UNetEncoder(in_channels=1, filter_start=64, depth=5)
        self.neck = NetVLAD(num_clusters=512, dim=1024)

    def forward(self, x):
        x = self.encoder(x)
        x = self.neck(x)

        return x
