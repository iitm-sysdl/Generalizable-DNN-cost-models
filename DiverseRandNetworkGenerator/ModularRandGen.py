import torch
import torch.nn as nn
import statistics
import csv
import platform
import psutil
import random
from random import sample
from random import randrange
import pickle

maxNumLayers = 20 
numSamples = 10000

masterFeatures = []
availableOperatorsList = ['nn.Conv2d', 'nn.DepthConv', 'nn.MaxPool2d', 'nn.ReLU']
redOperatorsList1 = ['nn.Conv2d', 'nn.DepthConv', 'nn.ReLU']
redOperatorsList2 = ['nn.Conv2d', 'nn.DepthConv', 'nn.MaxPool2d']
redOperatorsList3 = ['nn.Conv2d', 'nn.DepthConv']
channelList  = [3, 6, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 480, 576, 640, 720, 960]
inputDimList = [224, 112, 56, 28, 14, 7]
kernelList = [1, 3, 5, 7]
paddingDict ={1:0, 3:1, 5:2, 7:3}
possibleMaxPoolingDict={224:5, 112:4, 56:3, 28:2, 14:1, 7:0}
stride=1

file1 = open("netFeatures.csv",'w')
file2 = open("Embeddings.csv", 'w')
for i in range(numSamples):
    netFeatures = []
    netEmbedding = []
    numLayers = random.randint(1,maxNumLayers+1)
    inDim=random.choice(inputDimList)
    numMaxPool=possibleMaxPoolingDict[inDim]
    prevReLU=False
    inC=3
    outC=random.choice(channelList)
    k=random.choice(kernelList)
    netFeatures.append(['conv', inDim, inC, outC, k])
    netEmbedding.append([1, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*inC*k*k*2])
    for j in range(numLayers-1):
        if numMaxPool > 0:
            if prevReLU == True:
                operator = random.choice(redOperatorsList2)
                prevReLU=False
            else:
                operator = random.choice(availableOperatorsList)
        else:
            if prevReLU == True:
                operator = random.choice(redOperatorsList3)
                prevReLU=False
            else:
                operator = random.choice(redOperatorsList1)

        if operator == "nn.Conv2d":
            inC = outC
            outC=random.choice(channelList)
            k=random.choice(kernelList)
            netFeatures.append(['conv', inDim, inC, outC, k])
            netEmbedding.append([1, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*inC*k*k*2])
        elif operator == "nn.Conv2d":
            inC = outC
            outC = inC
            k=random.choice(kernelList)
            netFeatures.append(['dconv', inDim, inC, outC, k])
            netEmbedding.append([1, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*k*k*2])
        elif operator == "nn.MaxPool2d":
            netFeatures.append(['pool',inDim, 2, 2, 0])
            netEmbedding.append([0, 1, 0, inDim, inDim//2, outC, outC, k, stride, 0, inDim*inDim*outC])
            inDim = inDim//2
            numMaxPool = numMaxPool-1
        else:
            prevReLU=True
            netFeatures.append(['relu', inDim, 0, 0 ,0])
            netEmbedding.append([0, 0, 1, inDim, inDim, outC, outC, 0, 0, 0, inDim*inDim*outC])
    #print(numLayers)
    #print(netFeatures)
    #print(netEmbedding)
    data=''
    for itr in netFeatures:
        for itr2 in itr:
            data=data+str(itr2)+','
    data=data[:-1]
    data=data+'\n'
    file1.write(data)
    data=''
    for itr in netEmbedding:
        for itr2 in itr:
            data=data+str(itr2)+','
    data=data[:-1]
    data=data+'\n'
    file2.write(data)

file1.close()
file2.close()