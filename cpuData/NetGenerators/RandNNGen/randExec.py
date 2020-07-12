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

#with open("Networks", "rb") as f:
#    networkList = pickle.load(f)

with open("oplist.csv", "r") as f:
    oplist = list(csv.reader(f))

with open("opFeatures.csv", "r") as f:
    opFeatures = list(csv.reader(f))


with open("Features", "rb") as f:
    featuresList=pickle.load(f)

expEpoch = 10
print("Pickle Load done")
name =  "5layerconvnets_" + platform.node() + ".csv"
bFile = open(name,'a')
num_layers = 5
flag = False
flag2 = True
val = 0
for i in range(len(oplist[0])//num_layers):
        inputTensor = torch.randn([ 1, 3, featuresList[i][0][3], featuresList[i][0][3] ])
        #inputTensor.to('cuda')
        stitch_network = []
        model = []
        #model = nn.Sequential(*networkList[i])
        #model.to('cuda')
        #cudnn.benchmark=True
        for index in range(val, val+num_layers):
            stitch_network.append(eval(oplist[0][index])(*eval(opFeatures[0][index])))
        model = nn.Sequential(*stitch_network)    
        val = val+num_layers
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
        if i%100 == 0:
            print("100*%d epochs done" %(i//100))
