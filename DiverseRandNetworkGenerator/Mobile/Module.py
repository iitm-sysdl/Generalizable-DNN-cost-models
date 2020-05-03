import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import csv
import platform
import random
from random import sample
from random import randrange
import os
# from ModuleClass import *

file1 = open('netFeaturesMob.csv', 'r')
csv_reader = csv.reader(file1)
data = list(csv_reader)
file1.close()

class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, padding):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, padding=padding, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

class MobileBottleNeck(nn.Module):
    def __init__(self, inplanes, expansion, kernel, padding, outplanes):
        super(MobileBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=kernel, padding=padding, bias=False, groups=inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, outplanes, kernel_size=1, bias=False)
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        return out

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, padding):
        super(DepthwiseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel, padding=padding, bias=False, groups=in_planes)
    def forward(self, x):
        x = self.conv(x)
        return x

class DiverseRandNetwork(nn.Module):
    def __init__(self, layerFeatures, paddingDict):
        super(DiverseRandNetwork, self).__init__()
        self.layerList = nn.ModuleList([])
        for i in range(len(layerFeatures)):
            layer = layerFeatures[i]
            if layer[0]=='conv':
                self.layerList.append(Conv2d(int(layer[2]), int(layer[3]), int(layer[4]), paddingDict[int(layer[4])]))
            elif layer[0]=='mbconv':
                self.layerList.append(MobileBottleNeck(int(layer[2]), int(layer[3]), int(layer[4]), paddingDict[int(layer[4])], int(layer[5])))
            elif layer[0]=='relu':
                self.layerList.append(nn.ReLU(inplace=True))
            elif layer[0]=='pool':
                self.layerList.append(nn.MaxPool2d(2,2))
            
    def forward(self, x):
        for i in range(len(self.layerList)):
            x = self.layerList[i](x)
        return x

paddingDict ={1:0, 3:1, 5:2, 7:3}

for i in range(len(data)):
    layerFeatures = [data[i][j * 6:(j + 1) * 6] for j in range((len(data[i]) + 5) // 6 )]
    inDim = int(layerFeatures[0][1])
    timeL=[]
    x = torch.randn([1, 3, inDim, inDim])
    net = DiverseRandNetwork(layerFeatures, paddingDict)
    print(net)
    y = net(x)