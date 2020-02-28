from __future__ import print_function
import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import statistics
import psutil
import time
from utils import progress_bar
from utils import get_model_complexity_info
from folder2lmdb import ImageFolderLMDB
from torch.utils.tensorboard import SummaryWriter

#-------------------------------------------------------------Arguments--------------------------------------------
torch.manual_seed(42)
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

parser = argparse.ArgumentParser(description='PyTorch IMAGENET Training')
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--valdir', type=str, required=True)
parser.add_argument('--numgpu', type=int)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--batchsize', type=int, required=True)
parser.add_argument('--numepochs', type=int, required=True)
args = parser.parse_args()
writer = SummaryWriter(comment = args.name)
#------------------------------------------------------------Data Loading-------------------------------------------
print('==> Preparing data..')
valdir   = args.valdir
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
val_transform = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])
val_dataset   = ImageFolderLMDB(valdir, val_transform)
val_loader    = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=True)
#-------------------------------------------------------------Model Loading and Modification-----------------------------------------------
#-------------------------------------------------------------Edit this Part Before Training ----------------------------------------------
print('==> Building model..')
if args.model=="mv3":
    from models.mobilenetv3 import MobileNetV3
    net = MobileNetV3(mode='small', dropout=0)
    state_dict = torch.load('./pretrainedmodels/mobilenetv3_small_67.4.pth.tar')
    net.load_state_dict(state_dict, strict=True)
elif args.model=="mv2":
    from models.mobilenetv2 import MobileNetV2
    net = MobileNetV2()
    state_dict = torch.load('./pretrainedmodels/mobilenet_v2.pth')
    net.load_state_dict(state_dict, strict=True)
elif args.model=="mv1":
    from models.mobilenetv1 import MobileNet
    net = MobileNet()
elif args.model=="resnet34":
    net=torchvision.models.resnet34()
elif args.model=="resnet18":
    net=torchvision.models.resnet18()
elif args.model=="resnet50":
    net=torchvision.models.resnet50()
elif args.model=="resnet101":
    net=torchvision.models.resnet101()
elif args.model=="resnet152":
    net=torchvision.models.resnet152()
elif args.model=="squeeze":
    from models.squeezenet import SqueezeNet
    net = SqueezeNet()
    state_dict = torch.load('./pretrainedmodels/squeezenet.pth')
    net.load_state_dict(state_dict, strict=True)
#---------------------------------------Printing Network Stats and Moving to CUDA--------------------------------------
flops, params = get_model_complexity_info(net, (224, 224), as_strings=False, print_per_layer_stat=False)
print('==> Model Flops:{}'.format(flops))
print('==> Model Params:' + str(params))
criterion = nn.CrossEntropyLoss()
net = net.to(device)
if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = False

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        time_stddev = []
        full_time = []
        curr_time = 0
        prev_time = 0
        step_time = 0
        flag = False
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            #if batch_idx > 10000:
            #    print("\n")
            #    break
            curr_time = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #torch.cuda.synchronize()
            step_time = time.time() - curr_time
            freq = psutil.cpu_freq().current
            #temp = psutil.sensors_temperatures()
            writer.add_scalar('Frequency:', freq, batch_idx)
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            #for var in temp['coretemp']:
            #    writer.add_scalar(var.label, var.current, batch_idx)
           #     if var.current >= 85.0:
           #         flag = True
           #         flag2 = True
           # if flag == True:
           #     print('Throttle')
           #     while(flag2):
           #         time.sleep(10)
           #         temp = psutil.sensors_temperatures()
           #         for var in temp['coretemp']:
           #             if var.current <= 40:
           #                 flag2 = False
           #         flag = False
            writer.add_scalar('Latency:', step_time, batch_idx)
            full_time.append(step_time)

            #writer.add_scalar('Mean-Latency:', meanV, batch_idx)
            #writer.add_scalar('Stddev:', stddevV, batch_idx)
        full_mean =  statistics.mean(full_time)
        if len(full_time)==1:
          full_stddev = 0
        else:
          full_stddev = statistics.stdev(full_time)
        print(full_mean, full_stddev)



#-------------------------------------Main Part---------------------
for epoch in range(args.numepochs):
	test(epoch)
#---------------------------------------

