import torch
import torch.nn as nn
import statistics
import csv
import platform
import psutil
from time import sleep
import random
from random import sample
from random import randrange
import torch.backends.cudnn as cudnn
import pickle

numLayers = 5
#numSamples=10
numSamples = 10


#name =  "20layerdataconv_" + platform.node() + ".csv"
# bfile = open(name,'a')
masterNetwork = []
masterFeatures = []
opList = []
opFeatures = []
availableOperatorsList = ['nn.Conv2d', 'nn.MaxPool2d', 'nn.ReLU']
redOperatorsList1 = ['nn.Conv2d', 'nn.ReLU']
redOperatorsList2 = ['nn.Conv2d', 'nn.MaxPool2d']
redOperatorsList3 = ['nn.Conv2d']
channelList  = [3, 6, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 480, 576, 640, 720, 960]
inputDimList = [224, 112, 56, 28, 14, 7]
kernelList = [3, 5, 7]
paddingDict ={3:1, 5:2, 7:3}
possibleMaxPoolingDict={224:5, 112:4, 56:3, 28:2, 14:1, 7:0}
stride=1
for i in range(numSamples):
    network = []
    fixedEmbedding = []
    inDim=random.choice(inputDimList)
    numMaxPool=possibleMaxPoolingDict[inDim]
    prevReLU = False
    inC=3
    outC=random.choice(channelList)
    k=random.choice(kernelList)
    #network.append(nn.Conv2d(inC, outC, k,  stride, paddingDict[k], bias=False))
    opList.append("nn.Conv2d")
    opFeatures.append([inC, outC, k, stride, paddingDict[k], 1, 1, False ])
    fixedEmbedding.append([1, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*inC*k*k])
    for j in range(4):
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
            #network.append(nn.Conv2d(inC, outC, k,  stride, paddingDict[k], bias=False))
            opList.append("nn.Conv2d")
            opFeatures.append([inC, outC, k, stride, paddingDict[k], 1, 1, False])
            fixedEmbedding.append([1, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*inC*k*k])
        elif operator == "nn.MaxPool2d":
            #network.append(nn.MaxPool2d(2,2))
            fixedEmbedding.append([0, 1, 0, inDim, inDim/2, outC, outC, k, stride, 0, inDim*inDim*outC])
            opList.append("nn.MaxPool2d")
            opFeatures.append([2,2])
            inDim = inDim/2
            numMaxPool = numMaxPool-1
        else:
            prevReLU=True
            #network.append(nn.ReLU(inplace=True))
            opList.append("nn.ReLU")
            opFeatures.append([True])
            fixedEmbedding.append([0, 0, 1, inDim, inDim, outC, outC, 0, 0, 0, inDim*inDim*outC])

    #masterNetwork.append(network)
    masterFeatures.append(fixedEmbedding)
#with open("Networks", "wb") as f:
#    pickle.dump(masterNetwork, f)

with open ("oplist.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(opList)

with open ("opFeatures.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(opFeatures)

with open("Features", "wb") as f:
    pickle.dump(masterFeatures, f)
