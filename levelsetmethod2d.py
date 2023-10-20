import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary


class NetLevelSetMethod2D(nn.Module):
    def __init__(self, lay):
        super(NetLevelSetMethod2D, self).__init__()
        self.net = nn.Sequential()
        for i in range(0, len(lay) - 1):
            self.net.add_module('Linear_layer_%d' % i, nn.Linear(lay[i], lay[i + 1]))
            if i < len(lay) - 2:
                self.net.add_module('Tanh_layer_%d' % i, nn.Tanh())

    def forward(self, x):
        return self.net(x)




layers = [3, 30, 30, 30, 1]
model = NetLevelSetMethod2D(layers)
print(model)
