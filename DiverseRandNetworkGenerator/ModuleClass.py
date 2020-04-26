import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, padding):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, padding=padding, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, padding):
        super(DepthwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel, padding=padding, bias=False, groups=in_planes)
    def forward(self, x):
        x = self.conv(x)
        return x

class SkipConvolution(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, padding):
        super(SkipConvolution, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, padding=padding, bias=False)
    def forward(self, x):
        out1 = self.conv(x)
        return out1+x

class DiverseRandNetwork(nn.Module):
    def __init__(self, layerFeatures, paddingDict):
        super(DiverseRandNetwork, self).__init__()
        self.layerList = nn.ModuleList([])
        for i in range(len(layerFeatures)):
            layer = layerFeatures[i]
            if layer[0]=='conv':
                self.layerList.append(Conv2d(int(layer[2]), int(layer[3]), int(layer[4]), paddingDict[int(layer[4])]))
            elif layer[0]=='dconv':
                self.layerList.append(DepthwiseConv2d(int(layer[2]), int(layer[3]), int(layer[4]), paddingDict[int(layer[4])]))
            elif layer[0]=='rconv':
                self.layerList.append(SkipConvolution(int(layer[2]), int(layer[3]), int(layer[4]), paddingDict[int(layer[4])]))
            elif layer[0]=='relu':
                self.layerList.append(nn.ReLU(inplace=True))
            elif layer[0]=='pool':
                self.layerList.append(nn.MaxPool2d(2,2))
    def forward(self, x):
        for i in range(len(self.layerList)):
            x = self.layerList[i](x)
        return x

