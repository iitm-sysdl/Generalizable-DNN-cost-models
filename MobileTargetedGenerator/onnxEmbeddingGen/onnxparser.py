import torch
import torchvision
import onnx
import torch.onnx
import math
import timm

from mobilenetv1 import *
file = open("Embeddings101_118.csv", 'w')
def convolution(inDim, inC, outC, kernel, stride, padding, groups, netEmbedding):
    outDim = math.floor((inDim-kernel+2*padding)/stride) + 1
    if groups == 1:
        netEmbedding.append([1,0,0,0,0, inDim, outDim, inC, outC, kernel, stride, padding, outDim*outDim*outC*inC*kernel*kernel])
    else:
        netEmbedding.append([0,1,0,0,0, inDim, outDim, outC, outC, kernel, stride, padding, outDim*outDim*outC*kernel*kernel])
    return outDim

def relu(inDim, channels, netEmbedding):
    netEmbedding.append([0,0,0,1,0, inDim, inDim, channels, channels, 0, 0, 0, inDim*inDim*channels])

def globalaveragepooling(inDim, channels, netEmbedding):
    outDim = 1
    netEmbedding.append([0,0,0,0,1, inDim, outDim, channels, channels, inDim, inDim, 0, outDim*outDim*channels])
    return outDim

def skip(inDim, channels, netEmbedding):
    netEmbedding.append([0,0,1,0,0, inDim, inDim, channels, channels, 0, 0, 0, inDim*inDim*channels])

def maxpool(inDim, ceilmode, kernel, stride, padding, channels, netEmbedding):
    if ceilmode == 1:
        outdim = math.ceil((inDim - kernel+2*padding)/stride) + 1
    else:
        outdim = math.floor((inDim - kernel + 2*padding)/stride) + 1
    netEmbedding.append([0,0,0,0,1, inDim, outdim, channels, channels, kernel, stride, padding, outdim*outdim*channels])
    return outdim

def onnxparse(net, x):
    netEmbedding = []
    torch.jit.trace(net, x)
    torch.onnx.export(net, x, "temp.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"],verbose =True, dynamic_axes={"input" : {0: "batch_size"},"output" : {0: "batch_size"}})
    model = onnx.load("./temp.onnx")
    onnx.checker.check_model(model)
    # print(onnx.helper.printable_graph(model.graph))
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
            Dimension = convolution(Dimension, C, M, R, stride[0], padding[0], groups, netEmbedding)
            Channels = M
        elif node.op_type == 'Relu' or node.op_type == 'Clip':
            # print(node.input, node.output)
            relu(Dimension, Channels, netEmbedding)
        elif node.op_type == 'Add' :
            # print(node.input, node.output)
            skip(Dimension, Channels, netEmbedding)
        elif node.op_type == 'GlobalAveragePool':
            # print(node.input, node.output)
            Dimension = globalaveragepooling(Dimension, Channels, netEmbedding)
        elif node.op_type == 'MaxPool':
            # print(node.attribute)
            ceilMode = node.attribute[0].i
            kerenl_Size = node.attribute[1].ints
            padding = node.attribute[2].ints
            stride = node.attribute[3].ints
            # print(ceilMode, kerenl_Size, padding, stride)
            Dimension = maxpool(Dimension, ceilMode, kerenl_Size[0], stride[0], padding[0], Channels, netEmbedding)
        elif node.op_type == "AveragePool":
            ceilMode = node.attribute[0].i
            kerenl_Size = node.attribute[1].ints
            padding = node.attribute[2].ints
            stride = node.attribute[3].ints
            # print(ceilMode, kerenl_Size, padding, stride)
            Dimension = maxpool(Dimension, ceilMode, kerenl_Size[0], stride[0], padding[0], Channels, netEmbedding)
        elif node.op_type == 'ReduceMean':
            axes = node.attribute[0].ints
            keepdim = node.attribute[1].i
            # print(axes, keepdim)
            Dimension = globalaveragepooling(Dimension, Channels, netEmbedding)
        elif node.op_type == 'Gemm':
            # print(node.input, node.output, node.input[1], node.input[2])
            inits = model.graph.initializer
            for init in inits:
                if init.name == node.input[1]:
                    inpF, outF = init.dims[1], init.dims[0]
                    break
                    # print(inpF, outF)
            Dimension = convolution(Dimension, inpF, outF, 1, 1, 0, 1, netEmbedding)
            Channels = outF
        
    data=''
    for itr in netEmbedding:
        for itr2 in itr:
            data=data+str(itr2)+','
    data=data[:-1]
    data=data+'\n'
    file.write(data)


x = torch.rand([1,3,224,224])

net = MobileNet(depth_mul=0.25)
onnxparse(net, x)
net = MobileNet(depth_mul=0.5)
onnxparse(net, x)
net = MobileNet(depth_mul=0.75)
onnxparse(net, x)
net = MobileNet(depth_mul=1.0)
onnxparse(net, x)
target_platform = "proxyless_mobile"
net = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=False)
onnxparse(net, x)
net = timm.create_model('fbnetc_100', pretrained=False)
onnxparse(net, x)
net = timm.create_model('spnasnet_100', pretrained=False)
onnxparse(net, x)
target_platform = "proxyless_mobile_14"
net = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=False)
onnxparse(net, x)
net = torchvision.models.mnasnet0_5()
onnxparse(net, x)
net = torchvision.models.mnasnet0_75()
onnxparse(net, x)
net = torchvision.models.mnasnet1_0()
onnxparse(net, x)
net = torchvision.models.mnasnet1_3()
onnxparse(net, x)
net = torchvision.models.squeezenet1_0()
onnxparse(net, x)
net = torchvision.models.squeezenet1_1()
onnxparse(net, x)
net = torchvision.models.mobilenet_v2(width_mult=0.75)
onnxparse(net, x)
net = torchvision.models.mobilenet_v2(width_mult=0.5)
onnxparse(net, x)
net = torchvision.models.mobilenet_v2(width_mult=0.25)
onnxparse(net, x)
