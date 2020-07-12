import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import csv
import platform
import random
from random import sample
from random import randrange
#from pytorch2keras import pytorch_to_keras
import tensorflow as tf
import onnx
import torch.onnx
import onnx2keras
#from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
from ModuleClass import *

paddingDict ={1:0, 3:1, 5:2, 7:3}

file1 = open('netFeatures.csv', 'r')
csv_reader = csv.reader(file1)
data = list(csv_reader)
file1.close()

for i in range(len(data)):
    layerFeatures = [data[i][j * 5:(j + 1) * 5] for j in range((len(data[i]) + 4) // 5 )]
    inDim = int(layerFeatures[0][1])
    timeL=[]
    x = torch.randn([1, 3, inDim, inDim])
    net = DiverseRandNetwork(layerFeatures, paddingDict)
    #keras_model = pytorch_to_keras(net, x, [(3, 224, 224,)], verbose=True)
    ## Create ONNX
    print(net)
    torch.onnx.export(net, x, "temp.onnx", export_params=True, opset_version=10, do_constant_folding=True, input_names=["input"], output_names=["output"],dynamic_axes={"input" : {0: "batch_size"},"output" : {0: "batch_size"}})

    ## Verifying ONNX
    onnx_model = onnx.load("./temp.onnx")
    onnx.checker.check_model(onnx_model)
    inpt = ['input']
    keras_model = onnx_to_keras(onnx_model=onnx_model, input_names=inpt, change_ordering=True, verbose=False)
    #tf_rep = prepare(onnx_model)
    #print(tf_rep.tensor_dict)
    #tf_rep.export_graph('rand.pb')
    print("--------------------------------------%d-----------------------"%(i))
    #converter = tf.lite.TFLiteConverter.from_frozen_graph('rand.pb', input_arrays=['input'], output_arrays=['output'])
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()
    open('tflite_models/model_'+str(i)+'.tflite', "wb").write(tflite_model)

