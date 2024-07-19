import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from typing import Tuple
from collections import OrderedDict
from torchvision import transforms
from torch import nn
from torch.optim import Adam,SGD
import numpy as np
from tqdm import tqdm
from torchmetrics import Accuracy
from imgFE import imgFE
from preTrainDataset import preTrain_core50Data, preTrain_gripperImageData

class simpleMLP(nn.Module): 
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc4 = nn.Linear(512, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x):  
        x = self.activation(self.fc1(x))
        x = self.fc4(x)
        return x
       
#Load data
train_core50, test_core50 = preTrain_core50Data(mini=False)

# Train the custom feature extractor model
device = "cuda"
itr = 25
n_class = 50
fe_modelPer = imgFE(n_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
opt_per = SGD(fe_modelPer.parameters(), 1e-1)
train_dataloader = DataLoader(train_core50, num_workers=4, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_core50, num_workers=4, batch_size=32, shuffle=True)
for epochs in range(itr):
    loader_train = tqdm(train_dataloader)
    for data, targets in loader_train:
        data = data.to(device)
        targets = targets.to(device)
        opt_per.zero_grad()
        pred,_,_ = fe_modelPer(data)
        loss = criterion(pred, targets)
        loss.backward()
        opt_per.step()
        loader_train.set_description(f"Epoch {epochs}/{itr}")
        loader_train.set_postfix_str(f"loss {loss.item():4f} ")
torch.save(fe_modelPer.state_dict(), "pre_trained_models/GripperImageFE_128_visu.pt")

def overallTestAccuracy(test_dataloader, model):
    model.eval()
    cum_accu = []
    accuracy = Accuracy(task="multiclass", num_classes=n_class).to(device)
    accuracy_batch = 0
    batch_counter = 0
    with torch.no_grad():
        for data, targets in test_dataloader:
            data = data.to(device)
            targets = targets.to(device)
            preds,_,_ = model(data)
            temp = torch.argmax(preds,dim=1)
            accuracy_batch += accuracy(temp, targets)
            batch_counter += 1
    print(f"Pre_train accuracy is {(accuracy_batch/batch_counter)*100} %")

overallTestAccuracy(test_dataloader, fe_modelPer)
print("#"*40)
## Joint training
fe_model = torch.load("pre_trained_models/GripperImageFE_128_visu.pt")

## Load gripper image data
train_gripper, test_gripper = preTrain_gripperImageData()
# Remove the linear layers
fe_model.linear1 = nn.Identity()
fe_model.linear2 = nn.Identity()
fe_model.linear3 = nn.Identity()

itr = 70
n_classes = 10

model = simpleMLP(128, n_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
opt = SGD(model.parameters(), 1e-3, )
train_dataloader = DataLoader(train_gripper, num_workers=4, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_gripper, num_workers=4, batch_size=32, shuffle=True)
fe_model.eval()
for epochs in range(itr):
    loader_train = tqdm(train_dataloader)
    for data, targets in loader_train:
        data = data.float().to(device)
        with torch.no_grad():
            data_maps,_,_ = fe_model(data)
        current_batch = data.shape[0]
        targets = targets.to(device)
        opt.zero_grad()
        pred = model(data_maps)
        loss = criterion(pred, targets)
        loss.backward()
        opt.step()
        loader_train.set_description(f"Epoch {epochs}/{itr}")
        loader_train.set_postfix_str(f"loss {loss.item():4f} ")

def overallTestAccuracy(test_dataloader, model):
    model.eval()
    fe_model.eval()
    cum_accu = []
    accuracy = Accuracy(task="multiclass", num_classes=n_classes).to(device)
    accuracy_batch = 0
    batch_counter = 0
    with torch.no_grad():
        for data, targets in test_dataloader:
            current_batch = data.shape[0]
            data = data.float().to(device)
            data_maps,_,_ = fe_model(data)
            targets = targets.to(device)
            preds = model(data_maps)
            temp = torch.argmax(preds,dim=1)
            accuracy_batch += accuracy(temp, targets)
            batch_counter += 1
    print(f"Joint accuracy of the gripper Image is {accuracy_batch/batch_counter*100}")
overallTestAccuracy(test_dataloader, model)