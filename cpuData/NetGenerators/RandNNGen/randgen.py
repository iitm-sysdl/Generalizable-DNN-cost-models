import torch
import torch.nn as nn
import statistics
import csv
import platform
import psutil
from time import sleep
from random import sample
from random import randrange
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(comment = 'Layer')


num_layers = 0
max_layers = 19
num_samples = 1000000
ExpEpoch = 20

#TODO We should extend this to ResNet skip connections, MobileNet InvertedBottleneck layers, BottleNeck layers, etc.
#Let's generate CSV for the max layered network since that is also part of the search space now
available_operators = ['nn.Conv2d', 'nn.MaxPool2d', 'nn.ReLU']
name =  "20layerdataconv_" + platform.node() + ".csv"
bfile = open(name,'a')
Channels = [3, 6, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 480, 576, 640, 720, 960]
input_dim = [224, 112, 56, 28, 14, 7]
kernel = [3, 5, 7]
stride = 1
padding = 1
count = 0
flag = False
flag2 = True

for i in range(num_samples):
        vectoremb = []
        num_layers = randrange(0,max_layers)
        #num_layers_max = num_layers > num_layers_max? num_layers: num_layers_max #This will be used for the CSV generation
        inputChannels = [3]
        randomChannels = [3]
        random_network = []
        random_network.append(nn.Conv2d(3, 3, 3, 1, 1, bias=False))
        l = [1, 0, 0, 3, 3, 3, 1, 1, 0, 0, 0]
        vectoremb.append(l)
        for j in range(num_layers):
                randomOp = sample(available_operators, 1)
                inputChannels = randomChannels
                randominputDim = sample(input_dim, 1)
                randomKernel = sample(kernel, 1)
                if randomKernel[0]==5:
                        padding=2
                elif randomKernel[0]==7:
                        padding=3
                else:
                        padding=1
                #print(randomOp[0], available_operators[0], randomOp[0]==available_operators[0])
                if randomOp[0] == available_operators[0]:
                        randomChannels = sample(Channels, 1)
                        random_network.append(eval(randomOp[0])(inputChannels[0], randomChannels[0], randomKernel[0], stride, padding, bias=False))
                        l = [1, 0, 0, inputChannels[0], randomChannels[0], randomKernel[0], stride, padding, 0, 0, 0]
                        vectoremb.append(l)
                elif randomOp[0] == available_operators[1]:
                        random_network.append(eval(randomOp[0])(randomKernel[0], stride, padding))
                        l = [0, 1, 0, 0, 0, 0, 0, 0,randomKernel[0], stride, padding]
                        vectoremb.append(l)
                elif randomOp[0] == available_operators[2]:
                        random_network.append(eval(randomOp[0])(inplace=False))
                        l = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0]
                        vectoremb.append(l)
        for temp in range(num_layers, max_layers):
                vectoremb.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0])
        #print(random_network)
        model = nn.Sequential(*random_network)
        model.to('cuda')
        cudnn.benchmark=True
        print(model)
        x = torch.randn([1,3,224,224]).cuda()
        time = []
        for e in range(ExpEpoch):
            with torch.autograd.profiler.profile() as prof:
                y = model(x)
            time.append(prof.self_cpu_time_total)
        freq = psutil.cpu_freq().current
        temp = psutil.sensors_temperatures()
        writer.add_scalar('Frequency:', freq, count)
        for var in temp['coretemp']:
            writer.add_scalar(var.label, var.current, count)
            if var.current >= 85.0:
                flag = True
                flag2 = True
        if flag == True:
            print('Throttle')
            while(flag2):
                sleep(10)
                temp = psutil.sensors_temperatures()
                for var in temp['coretemp']:
                        if var.current <= 40:
                                flag2 = False
            flag = False
        mean_time = statistics.mean(time)
        vari_time = statistics.stdev(time)
        print(mean_time, vari_time)


        #point = [224, kernel, stride, padding, 3, Channels[i], 0, 0, mean_time, vari_time]
        writestring = ''
        for itr in vectoremb:
                for itr2 in itr:
                        writestring = writestring + str(itr2) + ','
        writestring += str(mean_time) + ',' + str(vari_time)+ '\n'
        bfile.write(writestring)
        count = count + 1






