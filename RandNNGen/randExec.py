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

with open("Networks", "rb") as f:  
    networkList = pickle.load(f)
with open("Features", "rb") as f:
    featuresList=pickle.load(f)
expEpoch = 10

name =  "5layerconvnets_" + platform.node() + ".csv"
bFile = open(name,'a')

flag = False
flag2 = True

for i in range(len(networkList)):
        inputTensor = torch.randn([ 1, 3, featuresList[i][0][3], featuresList[i][0][3] ])
        #inputTensor.to('cuda')
        model = nn.Sequential(*networkList[i])
        #model.to('cuda')
        #cudnn.benchmark=True
        print(model)
        time = []
        for j in range(expEpoch):
            with torch.autograd.profiler.profile() as prof:
                y = model(inputTensor)
            time.append(prof.self_cpu_time_total)

        temp = psutil.sensors_temperatures()
        for var in temp['coretemp']:
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
        writestring = str(mean_time) + ',' + str(vari_time)+ '\n'
        bFile.write(writestring)
