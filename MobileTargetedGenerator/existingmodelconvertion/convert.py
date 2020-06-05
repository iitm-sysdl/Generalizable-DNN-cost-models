import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import csv
from torchprofile import profile_macs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import onnx
import torch.onnx
import onnx2keras
from onnx2keras import onnx_to_keras
#pip install -U git+https://github.com/qubvel/efficientnet
import efficientnet.tfkeras as efn 
from mobilenetv1 import *
#pip install timm
import timm

def convertFromTFKeras(net, i):
    converter = tf.lite.TFLiteConverter.from_keras_model(net)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    open('model_' +str(i)+ '.tflite', "wb").write(tflite_model)

def convertFromPytorch(net, x, i):
	net.eval()
	torch.onnx.export(net, x, "temp.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"],dynamic_axes={"input" : {0: "batch_size"},"output" : {0: "batch_size"}})
	onnx_model = onnx.load("./temp.onnx")
	onnx.checker.check_model(onnx_model)
	inpt = ['input']
	keras_model = onnx_to_keras(onnx_model=onnx_model, input_names=inpt, change_ordering=True, verbose=False)
	convertFromTFKeras(keras_model, i)
    

x = torch.rand([1,3,224,224])
net = MobileNet(depth_mul=0.25)
convertFromPytorch(net, x, 101)
net = MobileNet(depth_mul=0.5)
convertFromPytorch(net, x, 102)
net = MobileNet(depth_mul=0.75)
convertFromPytorch(net, x, 103)
net = MobileNet(depth_mul=1.0)
convertFromPytorch(net, x, 104)
net = torchvision.models.mobilenet_v2(width_mult=1)
convertFromPytorch(net, x, 105)
target_platform = "proxyless_mobile"
net = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=False)
convertFromPytorch(net, x, 106)
net = timm.create_model('fbnetc_100', pretrained=False)
convertFromPytorch(net, x, 107)
net = timm.create_model('spnasnet_100', pretrained=False)
convertFromPytorch(net, x, 108)
target_platform = "proxyless_mobile_14"
net = torch.hub.load('mit-han-lab/ProxylessNAS', target_platform, pretrained=False)
convertFromPytorch(net, x, 109)
net = torchvision.models.mnasnet0_5()
convertFromPytorch(net, x, 110)
net = torchvision.models.mnasnet0_75()
convertFromPytorch(net, x, 111)
net = torchvision.models.mnasnet1_0()
convertFromPytorch(net, x, 112)
net = torchvision.models.mnasnet1_3()
convertFromPytorch(net, x, 113)
net = torchvision.models.squeezenet1_0()
convertFromPytorch(net, x, 114)
net = torchvision.models.squeezenet1_1()
convertFromPytorch(net, x, 115)
net = torchvision.models.mobilenet_v2(width_mult=0.75)
convertFromPytorch(net, x, 116)
net = torchvision.models.mobilenet_v2(width_mult=0.5)
convertFromPytorch(net, x, 117)
net = torchvision.models.mobilenet_v2(width_mult=0.25)
convertFromPytorch(net, x, 118)
#############TFKeras
net = efn.EfficientNetB0(weights='imagenet')
convertFromTFKeras(net, 119)
net = efn.EfficientNetB1(weights='imagenet')
convertFromTFKeras(net, 120)
net = efn.EfficientNetB2(weights='imagenet')
convertFromTFKeras(net, 121)
net = efn.EfficientNetB3(weights='imagenet')
convertFromTFKeras(net, 122)
