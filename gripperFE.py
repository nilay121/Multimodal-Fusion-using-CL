import torch
from torch import nn
from collections import OrderedDict
from torchvision import transforms
from torch.nn import functional as F
from torch.nn.modules.activation import Softmax

# # ## -------------------- new model
class gripperFE(nn.Module): 
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=512)
        self.activation1 = nn.LeakyReLU(0.01,inplace=False)

        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=256)
        self.activation2 = nn.LeakyReLU(0.01,inplace=False)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.activation3 = nn.LeakyReLU(0.01,inplace=False)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,256)
        self.activation4 = nn.ReLU()
        self.linear2 = nn.Linear(256,output_dim)

    def forward(self,x):
        conv1 = self.activation1(self.batchnorm1(self.conv1(x)))
        conv2 = self.activation2(self.batchnorm2(self.conv2(conv1)))
        conv3 = self.activation3(self.batchnorm3(self.conv3(conv2)))
        flatten = self.flatten(conv3)
        linear1 = self.activation4(self.linear1(flatten))
        linear2 = self.linear2(linear1)
        return linear2, conv3, conv2
    
def signalFeature_extractor():
    model = torch.load("pre_trained_models/GripperFE.pt")
    model.linear1 = nn.Identity()
    model.linear2 = nn.Identity()
    model.activation4 = nn.Identity()
    return model
    
# if __name__=="__main__":
#     x = torch.randn(2, 600, 1)
#     sm = gripperFE(x.shape[1])
#     sm(x)

