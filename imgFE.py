import torch
import numpy as np
import torchvision
from torch import nn
from typing import List, Tuple
from torch.nn import functional as F
from collections import OrderedDict
from torch.nn.functional import relu, avg_pool2d
from torchvision.models import resnet18

# ## ------------ new implementation
class imgFE(nn.Module): 
    def __init__(self,output_size):
        super().__init__()
        self.channel_dim = 3
        output_size = output_size

        self.conv1 = nn.Conv2d(self.channel_dim, 16, stride=1, kernel_size=(3, 3), padding=1)
        self.batch1 =  nn.BatchNorm2d(num_features=16)
        self.activation = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(16, 32, stride=2, kernel_size=(3, 3), padding=1)
        self.batch2 = nn.BatchNorm2d(num_features=32)

        self.conv3 = nn.Conv2d(32, 64, stride=2, kernel_size=(3, 3), padding=1)
        self.batch3 = nn.BatchNorm2d(num_features=64)

        self.conv4 = nn.Conv2d(64, 64, stride=3, kernel_size=(3, 3), padding=0)
        self.batch4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(64, 128, stride=2, kernel_size=(2, 2))
        self.batch5 = nn.BatchNorm2d(num_features=128)
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(3200, 2000)
        self.linear2 = nn.Linear(2000, 1000)
        self.linear3 = nn.Linear(1000,output_size)

    def forward(self, x):
        conv1 = self.activation(self.batch1(self.conv1(x)))
        conv2 = self.activation(self.batch2(self.conv2(conv1)))
        conv3 = self.activation(self.batch3(self.conv3(conv2)))
        conv4_fm = self.batch4(self.conv4(conv3))
        conv4 = self.activation(conv4_fm)
        conv5 = self.batch5(self.conv5(conv4))
        flatten = self.flatten(conv5)
        linear1 = self.activation(self.linear1(flatten))
        linear2 = self.activation(self.linear2(linear1))
        linear3 = self.linear3(linear2)
        return linear3, conv5, conv4

def imageFeature_extractor():
    model = torch.load("pre_trained_models/GripperImageFE_32.pt")
    model.Linear1 = nn.Identity()
    model.Linear2 = nn.Identity()
    model.Linear3 = nn.Identity()
    return model



if __name__=="__main__":
    x = torch.randn(1,3,32,32)
    model = imgFE(10)
    print(model)
    model.eval()
    a,b,c = model(x)
    print(a.shape)


