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
j = 0
for i in range(len(data)):
    layerFeatures = [data[i][j * 5:(j + 1) * 5] for j in range((len(data[i]) + 4) // 5 )]
    inDim = int(layerFeatures[0][1])
    if inDim == 224:
        timeL=[]
        x = torch.randn([1, 3, inDim, inDim])
        net = DiverseRandNetwork(layerFeatures, paddingDict)
        print(net)
        net.eval()
        traced_script_module = torch.jit.trace(net, x)
        traced_script_module.save("model" + str(j)+ ".pt")
        j+= 1
        if j == 5:
            exit()
file2.close()
