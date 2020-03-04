import torch
import torch.nn as nn
import statistics
import csv
import platform
import psutil
from torch.utils.tensorboard import SummaryWriter
from time import sleep

writer = SummaryWriter(comment = '2Layer')
ExpEpoch = 30
input_dim = 224
input_channels = 3
batch_size = 1
kernel = 3
stride = 1
padding = 1
OutputLayer1 = [6, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
OutputLayer2 = [48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192,200, 208, 224, 240, 360, 480]
OutputLayer3 = [80, 96, 120, 144, 160, 184, 192, 224, 240, 360, 480, 720, 960]
input_channels = 3

# Input Tensor
x = torch.randn([batch_size, input_channels, input_dim, input_dim])

name =  "2layerconv_" + platform.node() + ".csv"
bfile = open(name,'a')
count = 0
flag =False
for i in OutputLayer1:
    for j in OutputLayer2:
        if j > i:
            model = nn.Sequential(nn.Conv2d(input_channels, i, kernel, stride, padding, bias=False), 
                                  nn.Conv2d( i , j, kernel, stride, padding, bias=False))
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
            if flag == True:
                print('Throttle')
                sleep(10)
                flag = False
            mean_time = statistics.mean(time)
            vari_time = statistics.stdev(time)
            point = [input_dim, kernel, stride, padding, input_channels, i, j, 0, mean_time, vari_time]
            writestring = ''
            for itr in point:
                writestring = writestring + str(itr) + ','
            writestring += '\n'
            bfile.write(writestring)
            count = count + 1

writer = SummaryWriter(comment = '3Layer')

name =  "3layerconv_" + platform.node() + ".csv"
bfile = open(name,'a')
count = 0
flag =False
for i in OutputLayer1:
    for j in OutputLayer2:
        for k in OutputLayer3:
            if j > i and k > j:
                model = nn.Sequential(  nn.Conv2d(input_channels, i, kernel, stride, padding, bias=False), 
                                        nn.Conv2d(i, j, kernel, stride, padding, bias=False),
                                        nn.Conv2d(j, k, kernel, stride, padding, bias=False)
                                    )
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
                if flag == True:
                    print('Throttle')
                    sleep(10)
                    flag = False
                mean_time = statistics.mean(time)
                vari_time = statistics.stdev(time)
                point = [input_dim, kernel, stride, padding, input_channels, i, j, k, mean_time, vari_time]
                writestring = ''
                for itr in point:
                    writestring = writestring + str(itr) + ','
                writestring += '\n'
                bfile.write(writestring)
                count = count + 1
