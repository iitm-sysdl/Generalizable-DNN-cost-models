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

file1 = open('netFeatures.csv', 'r')
csv_reader = csv.reader(file1)
data = list(csv_reader)
file1.close()

file2 = open('execTime.csv', 'w')
expEpoch = 30

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


paddingDict ={1:0, 3:1, 5:2, 7:3}

for i in range(len(data)):
    layerFeatures = [data[i][j * 5:(j + 1) * 5] for j in range((len(data[i]) + 4) // 5 )]
    inDim = int(layerFeatures[0][1])
    timeL=[]
    for l in range(expEpoch):
        x = torch.randn([1, 3, inDim, inDim])
        with torch.autograd.profiler.profile() as prof:
            for k in range(len(layerFeatures)):  
                layer = layerFeatures[k]
                if layer[0]=='conv' or layer[0] == 'rconv':
                    if layer[0] == 'rconv':
                        y = x 
                    net = Conv2d(int(layer[2]), int(layer[3]), int(layer[4]), paddingDict[int(layer[4])])
                    x = net(x)            
                elif layer[0]=='dconv':
                    net = DepthwiseConv2d(int(layer[2]), int(layer[3]), int(layer[4]), paddingDict[int(layer[4])])
                    x = net(x)
                elif layer[0]=='pool':
                    x=F.max_pool2d(x, kernel_size=2, stride=2)
                elif layer[0]=='relu':
                    x=F.relu(x)
                elif layer[0] == 'add':
                    x = x + y

        timeL.append(prof.self_cpu_time_total)
    mean_time = statistics.mean(timeL)
    vari_time = statistics.stdev(timeL)
    print(i, mean_time, vari_time)
    file2.write(str(mean_time)+ ',' + str(vari_time) + '\n')
    file2.flush()
    os.fsync(file2)
file2.close()
