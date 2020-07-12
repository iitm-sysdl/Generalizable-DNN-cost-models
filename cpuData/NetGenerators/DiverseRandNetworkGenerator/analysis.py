#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import csv
import platform
import numpy as np
import random
from random import sample
from random import randrange
import pickle
import matplotlib.pyplot as plt
# %%
file1 = open('netFeatures.csv', 'r')
csv_reader = csv.reader(file1)
data = list(csv_reader)
file1.close()

# %%
inDimDict={224:0, 112:0, 56:0, 28:0, 14:0, 7:0}
numConv = np.zeros(10000*3)
numConv = np.reshape(numConv, (10000, 3))
convPerNetworkList=[]
for i in range(len(data)):
    layerFeatures = [data[i][j * 5:(j + 1) * 5] for j in range((len(data[i]) + 4) // 5 )]
    inDim = int(layerFeatures[0][1])
    inDimDict[inDim] = inDimDict[inDim] + 1
    numConvPerLayer=0
    for j in range(len(layerFeatures)):
        if layerFeatures[j][0] == 'conv':
            numConv[i][0] += 1
            numConvPerLayer += 1
        elif layerFeatures[j][0] == 'dconv':
            numConv[i][1] += 1
            numConvPerLayer += 1
        elif layerFeatures[j][0] == 'rconv':
            numConv[i][2] += 1
            numConvPerLayer += 1
    convPerNetworkList.append(numConvPerLayer)
# %%
plt.xlabel('Input Dimension of the Random Network')
plt.ylabel('Count of Networks')
plt.bar(range(len(inDimDict)), list(inDimDict.values()), align='center')
plt.xticks(range(len(inDimDict)), list(inDimDict.keys()))
plt.show()
# %%
# print(numConv)
# print(convPerNetworkList)
print("Avg number of Convolution in a network %f" %(statistics.mean(convPerNetworkList)))
print("Min Conv: %d" %(min(convPerNetworkList)))
print("Max Conv: %d" %(max(convPerNetworkList)))
bins = np.linspace(min(convPerNetworkList), max(convPerNetworkList), 20) # fixed number of bins
plt.hist(convPerNetworkList, bins=bins)
plt.xlabel('Number of Conv Per Layer')
plt.ylabel('count')
plt.show()
# %%


# %%
