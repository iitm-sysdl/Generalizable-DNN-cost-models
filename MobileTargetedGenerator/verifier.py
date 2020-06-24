import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import csv
import platform
import random
from random import sample
from random import randrange
from torchprofile import profile_macs
from matplotlib import pyplot as plt
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow as tf
import onnx
import torch.onnx
# import onnx2keras
# from onnx2keras import onnx_to_keras
import math
random.seed(42)

numSamples = 100

#
modelSize = ['small', 'large', 'giant']
# 40% Probability 0.75
widthMultiplier = [0.75, 0.75, 1, 1, 1]

kernelList = {'small':[3, 5], 
              'large':[3, 5, 7],
              'giant': [3,5,7]
             }

expansionList = [3, 6]
paddingDict = {1:0, 3:1, 5:2, 7:3}

channelListPerDim = { 'small': {56:[ 16, 24 ], 28:[ 36, 40, 44, 48], 14:[80, 88, 96], 7:[576, 584], 1:[1024, 1064]}, 
                      'large': {112:[16, 24],  56:[ 32, 40, 48], 28:[80, 88, 96, 104, 112], 14:[160, 168, 176], 7:[920, 960], 1:[1280, 1320]},
                      'giant': {112:[16, 24, 32], 56:[24, 32, 40], 28:[80, 88, 96, 108], 14:[192, 216, 240, 320], 7:[960, 968], 1:[1280, 1320]}
                    }

numMBsPerDim = {'small': { 56:[2,3], 28:[3,4,5], 14:[3,4] },
                'large': {112:[2,3], 56:[3,4],   28:[5,6,7,8], 14:[3,4,5]},
                'giant': {112:[2,3], 56:[3,4,5], 28:[5,6,7,8,9], 14:[3,4,5]}
               }

# file = open("Embeddings.csv", 'w')
class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

class MobileBottleNeck(nn.Module):
    def __init__(self, inplanes, outplanes, expansion, kernel, stride, padding, skip, width):
        super(MobileBottleNeck, self).__init__()
        expandChannels = int(inplanes*expansion*width)
        outChannels = int(outplanes*width)
        self.conv1 = nn.Conv2d(inplanes, expandChannels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(expandChannels, expandChannels, kernel_size=kernel, stride=stride, padding=padding, bias=False, groups=expandChannels)
        self.conv3 = nn.Conv2d(expandChannels, outChannels, kernel_size=1, bias=False)
        self.skip = skip
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        if self.skip == True:
            out = out + x
        return out

def convolution(inDim, inC, outC, kernel, stride, padding, netEmbedding):
    netEmbedding.append([1,0,0,0,0, inDim, inDim/stride, inC, outC, kernel, stride, padding, (inDim/stride)*(inDim/stride)*outC*inC*kernel*kernel])
    inDim = inDim/stride
    return inDim

def depthConv(inDim, inC, kernel, stride, padding, netEmbedding):
    netEmbedding.append([0,1,0,0,0, inDim, inDim/stride, inC, inC, kernel, stride, padding, (inDim/stride)*(inDim/stride)*inC*kernel*kernel])
    inDim = inDim/stride
    return inDim

def mobileBottleneckConv(inDim, inC, outC, expansion, kernel, stride, padding, skip, width, netEmbedding):
    expandChannels = int(inC*expansion*width)
    outChannels = int(outC*width)

    inDim = convolution(inDim, inC, expandChannels, 1, 1, 0, netEmbedding)
    
    #ReLU Embedding
    netEmbedding.append([0,0,0,1,0, inDim, inDim, expandChannels, expandChannels, 0, 0, 0, inDim*inDim*expandChannels])
    
    inDim = depthConv(inDim, expandChannels, kernel, stride, padding, netEmbedding)
    
    #ReLU Embedding
    netEmbedding.append([0,0,0,1,0, inDim, inDim, expandChannels, expandChannels, 0, 0, 0, inDim*inDim*expandChannels])
    
    inDim = convolution(inDim, expandChannels, outChannels, 1, 1, 0, netEmbedding)
    
    if skip == True:
        netEmbedding.append([0,0,1,0,0, inDim, inDim, outChannels, outChannels, 0, 0, 0, inDim*inDim*outChannels])
    return inDim, outChannels

def pooling(inDim, kernel, channels, netEmbedding):
    netEmbedding.append([0,0,0,0,1, inDim, inDim/kernel, channels, channels, kernel, kernel, 0, (inDim/kernel)*(inDim/kernel)*outC])
    inDim = inDim/kernel
    return inDim
def onnxconvolution(inDim, inC, outC, kernel, stride, padding, groups, netEmbedding):
    outDim = math.floor((inDim-kernel+2*padding)/stride) + 1
    if groups == 1:
        netEmbedding.append([1,0,0,0,0, inDim, outDim, inC, outC, kernel, stride, padding, outDim*outDim*outC*inC*kernel*kernel])
    else:
        netEmbedding.append([0,1,0,0,0, inDim, outDim, outC, outC, kernel, stride, padding, outDim*outDim*outC*kernel*kernel])
    return outDim

def onnxrelu(inDim, channels, netEmbedding):
    netEmbedding.append([0,0,0,1,0, inDim, inDim, channels, channels, 0, 0, 0, inDim*inDim*channels])

def onnxglobalaveragepooling(inDim, channels, netEmbedding):
    outDim = 1
    netEmbedding.append([0,0,0,0,1, inDim, outDim, channels, channels, 0, 0, 0, outDim*outDim*channels])
    return outDim

def onnxskip(inDim, channels, netEmbedding):
    netEmbedding.append([0,0,1,0,0, inDim, inDim, channels, channels, 0, 0, 0, inDim*inDim*channels])

def onnxmaxpool(inDim, ceilmode, kernel, stride, padding, channels, netEmbedding):
    if ceilmode == 1:
        outdim = math.ceil((inDim - kernel+2*padding)/stride) + 1
    else:
        outdim = math.floor((inDim - kernel + 2*padding)/stride) + 1
    netEmbedding.append([0,0,0,0,1, inDim, outdim, channels, channels, kernel, stride, padding, outdim*outdim*channels])
    return outdim

def onnxparse(model):
    netEmbedding = []
    Dimension = 224
    Channels = 3
    for node in model.graph.node:
        if node.op_type == 'Conv':
            dialtion = node.attribute[0].ints 
            groups = node.attribute[1].i 
            kerenl_Size = node.attribute[2].ints
            padding = node.attribute[3].ints
            stride = node.attribute[4].ints
            # print(dialtion, groups, kerenl_Size, padding, stride)
            inits = model.graph.initializer
            for init in inits:
                if init.name == node.input[1]:
                    M, C, R, S = init.dims[0], init.dims[1], init.dims[2], init.dims[3]
                    break
                    # print(M, C, R, S)
            Dimension = onnxconvolution(Dimension, C, M, R, stride[0], padding[0], groups, netEmbedding)
            Channels = M
        elif node.op_type == 'Relu' or node.op_type == 'Clip':
            # print(node.input, node.output)
            onnxrelu(Dimension, Channels, netEmbedding)
        elif node.op_type == 'Add' :
            # print(node.input, node.output)
            onnxskip(Dimension, Channels, netEmbedding)
        elif node.op_type == 'GlobalAveragePool':
            # print(node.input, node.output)
            Dimension = onnxglobalaveragepooling(Dimension, Channels, netEmbedding)
        elif node.op_type == 'MaxPool':
            # print(node.attribute)
            ceilMode = node.attribute[0].i
            kerenl_Size = node.attribute[1].ints
            padding = node.attribute[2].ints
            stride = node.attribute[3].ints
            # print(ceilMode, kerenl_Size, padding, stride)
            Dimension = onnxmaxpool(Dimension, ceilMode, kerenl_Size[0], stride[0], padding[0], Channels, netEmbedding)
        elif node.op_type == "AveragePool":
            ceilMode = node.attribute[0].i
            kerenl_Size = node.attribute[1].ints
            padding = node.attribute[2].ints
            stride = node.attribute[3].ints
            # print(ceilMode, kerenl_Size, padding, stride)
            Dimension = onnxmaxpool(Dimension, ceilMode, kerenl_Size[0], stride[0], padding[0], Channels, netEmbedding)
        elif node.op_type == 'ReduceMean':
            axes = node.attribute[0].ints
            keepdim = node.attribute[1].i
            # print(axes, keepdim)
            Dimension = onnxglobalaveragepooling(Dimension, Channels, netEmbedding)
        elif node.op_type == 'Gemm':
            # print(node.input, node.output, node.input[1], node.input[2])
            inits = model.graph.initializer
            for init in inits:
                if init.name == node.input[1]:
                    inpF, outF = init.dims[1], init.dims[0]
                    break
                    # print(inpF, outF)
            Dimension = onnxconvolution(Dimension, inpF, outF, 1, 1, 0, 1, netEmbedding)
            Channels = outF
    
    return netEmbedding
def check(l1, l2, i):
    # print(len(l1), len(l2))
    for k in range(len(l1)):
        # print(len(l1[i]), len(l2[i]))
        temp = l1[k]
        for l in range(len(temp)):
            if int(l1[k][l]) != int(l2[k][l]):
                print("Failed Network %d. Layer %d"%(i, k))
                print(l1[k], l2[k])
                return False
flopsList = []
modelSizeList = []
inferenceTimeList = []
i = 0
while i < numSamples:
    if i < numSamples//3:
        size = 'small'
    elif i < 2*numSamples//3:
        size = 'large'
    else:
        size = 'giant'

    width = random.choice(widthMultiplier)
    netEmbedding = []
    network = []
    numSkip = 0
    numMB = 0
    ## Input Head 3 x 224 x 224
    inDim = 224
    inC = 3
    outC = 32
    k = 3
    s = 2
    network.append(Conv2d(inC, outC, k, s, paddingDict[k]))
    inDim = convolution(inDim, inC, outC, k, s, paddingDict[k], netEmbedding)
    
    inC = outC
    k = 3
    s = 2 if size == 'small' else 1
    outC = 16
    e = 1
    skip = False
    network.append(MobileBottleNeck(inC, outC, e, k, s, paddingDict[k], skip, width))
    inDim, outC = mobileBottleneckConv(inDim, inC, outC, e, k, s, paddingDict[k], skip, width, netEmbedding)
    numMB+=1
    ## 16 x 112 x 112
    while inDim != 7:
        numConvList = numMBsPerDim[size][inDim]
        outChannelList = channelListPerDim[size][inDim]
        numConv = random.choice(numConvList)
        for j in range(numConv):
            inC= outC
            outC = random.choice(outChannelList)
            k = random.choice(kernelList[size])
            s = 2 if j == 0 else 1
            e = random.choice(expansionList)
            skip = True if inC == int(outC*width) and s == 1 else False
            if skip == True:
                numSkip += 1
            network.append(MobileBottleNeck(inC, outC, e, k, s, paddingDict[k], skip, width))
            inDim, outC = mobileBottleneckConv(inDim, inC, outC, e, k, s, paddingDict[k], skip, width, netEmbedding)
            numMB+=1

    ## X x 7 x 7
    outChannelList = channelListPerDim[size][inDim]
    inC = outC
    outC = random.choice(outChannelList)
    network.append(Conv2d(inC, outC, 1, 1, 0))
    inDim = convolution(inDim, inC, outC, 1, 1, 0, netEmbedding)

    network.append(nn.MaxPool2d(7,7))
    inDim = pooling(inDim, 7, outC, netEmbedding)

    outChannelList = channelListPerDim[size][inDim]
    inC = outC
    outC = random.choice(outChannelList)
    network.append(Conv2d(inC, outC, 1, 1, 0))
    inDim = convolution(inDim, inC, outC, 1, 1, 0, netEmbedding)

    inC = outC
    outC = 1000
    network.append(Conv2d(inC, outC, 1, 1, 0))
    inDim = convolution(inDim, inC, outC, 1, 1, 0, netEmbedding)

    net = nn.Sequential(*network)
    # print(net)
    
    x = torch.rand([1,3,224,224])
    with torch.autograd.profiler.profile() as prof:
        y = net(x)
    macs = profile_macs(net, x)
    if  40000000 < macs < 400000000:
        i += 1
    else:
        continue
    params = sum([p.numel() for p in net.parameters()])
    flopsList.append(macs/1e6)
    modelSizeList.append((params*4.0)/(1024**2))
    inferenceTimeList.append(prof.self_cpu_time_total/1000.0)
    # print('Model ' + str(i) +' NumBottle:%d Skips: %d MACS: %f M   ModelSize: %f MB  Inference: %f ms' %(numMB, numSkip, macs/1e6, (params*4.0)/(1024**2), prof.self_cpu_time_total/1000.0))
    net.eval()
    
    torch.onnx.export(net, x, "temp.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"],dynamic_axes={"input" : {0: "batch_size"},"output" : {0: "batch_size"}})
    onnx_model = onnx.load("./temp.onnx")
    onnx.checker.check_model(onnx_model)
    netEmbedding2 = onnxparse(onnx_model)
    flag = check(netEmbedding, netEmbedding2, i-1)
    if flag==False:
        print(net)
        print(onnx.helper.printable_graph(onnx_model.graph))
        exit()
    # inpt = ['input']
    
    # keras_model = onnx_to_keras(onnx_model=onnx_model, input_names=inpt, change_ordering=True, verbose=False)
    # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    # tflite_model = converter.convert()
    # open('modelzoo/model_'+str(i)+'.tflite', "wb").write(tflite_model)
    
    # data=''
    # for itr in netEmbedding:
    #     for itr2 in itr:
    #         data=data+str(itr2)+','
    # data=data[:-1]
    # data=data+'\n'
    # file.write(data)
# file.close()

# plt.boxplot(flopsList)
# plt.savefig('FLOPSboxPlot.png')
# plt.figure()
# plt.boxplot(modelSizeList)
# plt.savefig('SizeboxPlot.png')
# plt.figure()
# plt.boxplot(inferenceTimeList)
# plt.savefig('TimeboxPlot.png')
# plt.show()
