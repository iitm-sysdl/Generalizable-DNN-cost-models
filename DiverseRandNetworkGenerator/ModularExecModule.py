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
from ModuleClass import *

file1 = open('netFeatures.csv', 'r')
csv_reader = csv.reader(file1)
data = list(csv_reader)
file1.close()

file2 = open('execTime.csv', 'w')
expEpoch = 30


paddingDict ={1:0, 3:1, 5:2, 7:3}

for i in range(len(data)):
    layerFeatures = [data[i][j * 5:(j + 1) * 5] for j in range((len(data[i]) + 4) // 5 )]
    inDim = int(layerFeatures[0][1])
    timeL=[]
    x = torch.randn([1, 3, inDim, inDim])
    net = DiverseRandNetwork(layerFeatures, paddingDict)
    for l in range(expEpoch):
        with torch.autograd.profiler.profile() as prof:
            y = net(x)
        timeL.append(prof.self_cpu_time_total)
    mean_time = statistics.mean(timeL)
    vari_time = statistics.stdev(timeL)
    print(i, mean_time, vari_time)
    file2.write(str(mean_time)+ ',' + str(vari_time) + '\n')
    file2.flush()
    os.fsync(file2)
file2.close()
