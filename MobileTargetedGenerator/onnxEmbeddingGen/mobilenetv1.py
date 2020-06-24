'''MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out

class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, depth_mul = 1.0, num_classes=1000):
        super(MobileNet, self).__init__()
        self.depth_mul = depth_mul
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(int(1024 * self.depth_mul), num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7)
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = int(x * self.depth_mul) if isinstance(x, int) else int(x[0]*self.depth_mul)
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out

def test():
    net=MobileNetFriendly()
    x = torch.randn(1, 3, 224, 224)
    y = net(x)
    n2 = MobileNet()
    z = n2(x) 
    print(y.shape, z.shape)

# test()