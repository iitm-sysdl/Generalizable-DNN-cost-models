import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchprofile import profile_macs
from mobilenetv1 import *
import math
file = open("squeezeEmbeddings.csv", 'w')
## Conv Attributes: in_channels': 256, 'out_channels': 256, 'kernel_size': (1, 1), 'stride': (1, 1), 'padding': (0, 0), 'dilation': (1, 1), 'transposed': False, 'output_padding': (0, 0), 'groups': 1,
## FC Attributes: in_features': 256, 'out_features': 1000
## ReLU Attributes: 
def convolution(inDim, inC, outC, kernel, stride, padding, netEmbedding):
    outdim = (inDim - kernel + 2*padding)//stride + 1;
    netEmbedding.append([1,0,0,0,0, inDim, outdim, inC, outC, kernel, stride, padding, outdim*outdim*outC*inC*kernel*kernel*2])
    return outdim

def relu(inDim, channels, netEmbedding):
    netEmbedding.append([0,0,0,1,0, inDim, inDim, channels, channels, 0, 0, 0, inDim*inDim*channels])

def pooling(inDim, kernel, channels, netEmbedding):
    netEmbedding.append([0,0,0,0,1, inDim, inDim/kernel, channels, channels, 0, 0, 0, (inDim/kernel)*(inDim/kernel)*kernel*kernel*channels])
    inDim = inDim/kernel
    return inDim

def maxpool(inDim, kernel, stride, channels, netEmbedding):
    outdim = math.ceil((inDim - kernel)/stride) + 1;
    netEmbedding.append([0,0,0,0,1, inDim, outdim, channels, channels, 0, 0, 0, outdim*outdim*kernel*kernel*channels])
    return outdim

def generateEmbedding(model):
    inDim = 224
    netEmbedding = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            inDim = convolution(inDim, module.in_channels, module.out_channels, module.kernel_size[0], module.stride[0], module.padding[0], netEmbedding)

        elif isinstance(module, nn.ReLU6) or isinstance(module, nn.ReLU):
            channel = netEmbedding[-1][8]
            relu(inDim, channel, netEmbedding)
        
        elif isinstance(module, nn.Linear):
            inDim = convolution(inDim, module.in_features, module.out_features, 1, 1, 0, netEmbedding)
        
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            channel = netEmbedding[-1][8]
            inDim = pooling(inDim, inDim, channel, netEmbedding)
        
        elif isinstance(module, nn.MaxPool2d):
            channel = netEmbedding[-1][8]
            inDim = maxpool(inDim, module.kernel_size, module.stride, channel, netEmbedding)

    data=''
    for itr in netEmbedding:
        for itr2 in itr:
            data=data+str(itr2)+','
    data=data[:-1]
    data=data+'\n'
    file.write(data)

x = torch.rand([1,3,224,224])
# net = MobileNet(depth_mul=0.25)
# generateEmbedding(net)
# net = MobileNet(depth_mul=0.5)
# generateEmbedding(net)
# net = MobileNet(depth_mul=0.75)
# generateEmbedding(net)
# net = MobileNet(depth_mul=1.0)
# generateEmbedding(net)
# net = torchvision.models.mobilenet_v2(width_mult=1)
# net = torch.jit.trace(net, x)
# print(net)

# generateEmbedding(net)
# target_platform = "proxyless_mobile"
# net = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=False)
# generateEmbedding(net)
# net = timm.create_model('fbnetc_100', pretrained=False)
# net = timm.create_model('spnasnet_100', pretrained=False)
# target_platform = "proxyless_mobile_14"
# net = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=False)
# print(net)
# net = torchvision.models.mnasnet0_5()
# net = torchvision.models.mnasnet0_75()
# net = torchvision.models.mnasnet1_0()
# net = torchvision.models.mnasnet1_3()
net = torchvision.models.squeezenet1_0()
generateEmbedding(net)
net = torchvision.models.squeezenet1_1()
generateEmbedding(net)
# net = torchvision.models.mobilenet_v2(width_mult=0.75)
# net = torchvision.models.mobilenet_v2(width_mult=0.5)
# net = torchvision.models.mobilenet_v2(width_mult=0.25)
# net = efn.EfficientNetB0(weights='imagenet')
# net = efn.EfficientNetB1(weights='imagenet')
# net = efn.EfficientNetB2(weights='imagenet')
# net = efn.EfficientNetB3(weights='imagenet')
file.close()