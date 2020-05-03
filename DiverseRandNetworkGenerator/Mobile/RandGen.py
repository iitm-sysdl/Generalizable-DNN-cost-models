import torch
import torch.nn as nn
import statistics
import csv
import platform
import random
from random import sample
from random import randrange

maxNumLayers = 20
numSamples = 5

masterFeatures = []
availableOperatorsList = ['nn.MobileBottle',  'nn.MaxPool2d']
redOperatorsList = ['nn.MobileBottle']

channelList  = [3, 6, 12, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 96, 104, 120]
inputDimList = [224]#, 112, 56, 28, 14, 7]
kernelList = [3, 5, 7]
expansionList = [1, 2, 4, 6]
paddingDict ={1:0, 3:1, 5:2, 7:3}
possibleMaxPoolingDict={224:5, 112:4, 56:3, 28:2, 14:1, 7:0}
stride=1

file1 = open("netFeaturesMob.csv",'w')
file2 = open("EmbeddingsMob.csv", 'w')

def convolution(outC, channelList, kernelList, expansionList, inDim, stride, paddingDict, netFeatures, netEmbedding):
    startChannels = outC
    inC = outC
    e = random.choice(expansionList)
    expansionRatio = e
    outC = inC*e
    k = 1
    netEmbedding.append([1, 0, 0, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*inC*k*k*2])
    netEmbedding.append([0, 0, 1, 0, 0, inDim, inDim, outC, outC, 0, 0, 0, inDim*inDim*outC])

    inC = outC
    k=random.choice(kernelList)
    kernelDepthwise = k
    netEmbedding.append([0, 1, 0, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*k*k*2])
    netEmbedding.append([0, 0, 1, 0, 0, inDim, inDim, outC, outC, 0, 0, 0, inDim*inDim*outC])

    outC = random.choice(channelList)
    endChannels = outC
    k = 1
    netEmbedding.append([1, 0, 0, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*inC*k*k*2])

    netFeatures.append(['mbconv', inDim, startChannels, expansionRatio, kernelDepthwise, endChannels])
    return outC

def maxpool(inDim, numMaxPool, netFeatures, netEmbedding):
    netFeatures.append(['pool',inDim, 2, 2, 0, 0])
    netEmbedding.append([0, 0, 0, 1, 0, inDim, inDim//2, outC, outC, 2, 2, 0, inDim*inDim*outC])
    inDim = inDim//2
    numMaxPool = numMaxPool-1
    return inDim, numMaxPool, True

for i in range(numSamples):
    netFeatures = []
    netEmbedding = []
    numLayers = random.randint(1,maxNumLayers+1)
    inDim=random.choice(inputDimList)
    numMaxPool=possibleMaxPoolingDict[inDim]
    prevMaxPool=False

    inC=3
    outC=random.choice(channelList)
    k=random.choice(kernelList)
    netFeatures.append(['conv', inDim, inC, outC, k, 0])
    netEmbedding.append([1, 0, 0, 0, 0, inDim, inDim, inC, outC, k, stride, paddingDict[k], inDim*inDim*outC*inC*k*k*2])

    inDim, numMaxPool, prevMaxPool = maxpool(inDim, numMaxPool, netFeatures, netEmbedding)

    for j in range(numLayers-1):
        if numMaxPool > 0:
            if prevMaxPool == True:
                operator = random.choice(redOperatorsList)
                prevMaxPool=False
            else:
                operator = random.choice(availableOperatorsList)
        else:
            operator = random.choice(redOperatorsList)

        if operator == "nn.MobileBottle":
            outC = convolution(outC, channelList, kernelList, expansionList, inDim, stride, paddingDict, netFeatures, netEmbedding)
        elif operator == "nn.MaxPool2d":
            inDim, numMaxPool, prevMaxPool = maxpool(inDim, numMaxPool, netFeatures, netEmbedding)
    
    print(netFeatures)
    # print(netEmbedding)
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