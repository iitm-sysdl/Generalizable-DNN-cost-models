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
# import tensorflow as tf
# import onnx
# import torch.onnx
# import onnx2keras
# from onnx2keras import onnx_to_keras

numSamples = 50

kernelList = [3, 5, 7]
expansionList = [3, 6]
paddingDict ={1:0, 3:1, 5:2, 7:3}
channelListPerDim = {112:[16, 24, 32], 56:[24, 32, 40], 28:[80, 88, 96, 108], 14:[192, 216, 240, 320], 7:[960, 1024, 1280]}
numMBsPerDim = {112:[2,3], 56:[3,4,5], 28:[5,6,7,8,9], 14:[3,4,5]}

class Conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel, stride, padding):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=padding, bias=False)
    def forward(self, x):
        x = self.conv(x)
        return x

class MobileBottleNeck(nn.Module):
    def __init__(self, inplanes, outplanes, expansion, kernel, stride, padding):
        super(MobileBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes*expansion, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inplanes*expansion, inplanes*expansion, kernel_size=kernel, stride=stride, padding=padding, bias=False, groups=inplanes*expansion)
        self.conv3 = nn.Conv2d(inplanes*expansion, outplanes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        return out

def convolution(inDim, inC, outC, kernel, stride, padding, netEmbedding):
    # netEmbedding.append(Conv2d(inC, outC, kernel, stride, padding))
    inDim = inDim/stride
    return inDim

def depthConv(inDim, inC, kernel, stride, padding, netEmbedding):
    inDim = inDim/stride
    return inDim

def mobileBottleneckConv(inDim, inC, outC, expansion, kernel, stride, padding, netEmbedding):
    inDim = convolution(inDim, inC, expansion*inC, 1, 1, 0, netEmbedding)
    inDim = depthConv(inDim, inC*expansion, kernel, stride, padding, netEmbedding)
    inDim = convolution(inDim, inC*expansion, outC, 1, 1, 0, netEmbedding)
    return inDim

def pooling(inDim, kernel, netEmbedding):
    inDim = inDim/kernel
    return inDim

flopsList = []
modelSizeList = []
inferenceTimeList = []
for i in range(numSamples):
    netEmbedding = []
    network = []
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
    s = 1
    outC = 16
    e = 1
    network.append(MobileBottleNeck(inC, outC, e, k, s, paddingDict[k]))
    inDim = mobileBottleneckConv(inDim, inC, outC, e, k, s, paddingDict[k], netEmbedding)
    ## 16 x 112 x 112
    while inDim != 7:
        numConvList = numMBsPerDim[inDim]
        outChannelList = channelListPerDim[inDim]
        numConv = random.choice(numConvList)
        for j in range(numConv):
            inC= outC
            outC = random.choice(outChannelList)
            k = random.choice(kernelList)
            s = 2 if j == 0 else 1
            e = random.choice(expansionList)
            network.append(MobileBottleNeck(inC, outC, e, k, s, paddingDict[k]))
            inDim = mobileBottleneckConv(inDim, inC, outC, e, k, s, paddingDict[k], netEmbedding)
    
    ## X x 7 x 7
    outChannelList = channelListPerDim[inDim]
    inC = outC
    outC = random.choice(outChannelList)
    network.append(Conv2d(inC, outC, 1, 1, 0))
    inDim = convolution(inDim, inC, outC, 1, 1, 0, netEmbedding)

    network.append(nn.MaxPool2d(7,7))
    inDim = pooling(inDim, 7, netEmbedding)

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
    params = sum([p.numel() for p in net.parameters()])
    flopsList.append(macs/1e6)
    modelSizeList.append((params*4.0)/(1024**2))
    inferenceTimeList.append(prof.self_cpu_time_total/1000.0)
    print('Model ' + str(i) + '  MACS: %f M   ModelSize: %f MB  Inference: %f ms' %(macs/1e6, (params*4.0)/(1024**2), prof.self_cpu_time_total/1000.0))
    net.eval()
    # traced_script_module = torch.jit.trace(net, x)
    # traced_script_module.save("model" + str(i)+ ".pt")
    # torch.onnx.export(net, x, "temp.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"],dynamic_axes={"input" : {0: "batch_size"},"output" : {0: "batch_size"}})
    # onnx_model = onnx.load("./temp.onnx")
    # onnx.checker.check_model(onnx_model)
    # inpt = ['input']
    # keras_model = onnx_to_keras(onnx_model=onnx_model, input_names=inpt, change_ordering=True, verbose=False)
    # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # tflite_model = converter.convert()
    # open('model/model_'+str(i)+'.tflite', "wb").write(tflite_model)
plt.boxplot(flopsList)
plt.show()
plt.boxplot(modelSizeList)
plt.show()
plt.boxplot(inferenceTimeList)
plt.show()