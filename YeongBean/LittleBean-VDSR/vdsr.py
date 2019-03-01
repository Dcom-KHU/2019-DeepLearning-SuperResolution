import torch
import torch.nn as nn
from math import sqrt

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class Conv_ReLU_Block5(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block5, self).__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=2, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class Conv_ReLU_Block7(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block7, self).__init__()
        self.conv = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=3, padding=3, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.residual_layer = self.make_layers(Conv_ReLU_Block7, Conv_ReLU_Block5, Conv_ReLU_Block, 18, 6, 12)
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=7, stride=3, padding=3, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
        

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def make_layers(self, block1, block2, block3, num, mid1,mid2):
        layers = []
        for _ in range(mid1):
            layers.append(block1())
        for _ in range(mid1, mid2):
            layers.append(block2())
        for _ in range(mid2, num):
            layers.append(block3())

            return nn.Sequential(*layers)
        

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out
 