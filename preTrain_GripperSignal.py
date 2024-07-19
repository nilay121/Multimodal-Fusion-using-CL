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
from gripperFE import gripperFE
from preTrainDataset import preTrain_gripperData, preTrain_gripperImageData


#Load data
train_gripper, test_gripper = preTrain_gripperData()

# Train the custom feature extractor model
device = "cuda"
itr = 300
n_class = 15
input_dim = 600
fe_modelPer = gripperFE(input_dim, n_class).to(device)
criterion = torch.nn.CrossEntropyLoss()
opt_per = Adam(fe_modelPer.parameters(), 1e-4)
train_dataloader = DataLoader(train_gripper, num_workers=4, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_gripper, num_workers=4, batch_size=32, shuffle=False)
for epochs in range(itr):
    loader_train = tqdm(train_dataloader)
    for data, targets in loader_train:
        data = data.reshape(-1, 600, 1).to(device)
        targets = targets.to(device)
        opt_per.zero_grad()
        pred,_,_ = fe_modelPer(data)
        loss = criterion(pred, targets)
        loss.backward()
        opt_per.step()
        loader_train.set_description(f"Epoch {epochs}/{itr}")
        loader_train.set_postfix_str(f"loss {loss.item():4f} ")
torch.save(fe_modelPer.state_dict(), "pre_trained_models/GripperFE.pt")

def overallTestAccuracy(test_dataloader, model):
    model.eval()
    cum_accu = []
    accuracy = Accuracy(task="multiclass", num_classes=n_class).to(device)
    accuracy_batch = 0
    batch_counter = 0
    with torch.no_grad():
        for data, targets in test_dataloader:
            data = data.reshape(-1, 600, 1).to(device)
            targets = targets.to(device)
            preds,_,_ = model(data)
            temp = torch.argmax(preds,dim=1)
            accuracy_batch += accuracy(temp, targets)
            batch_counter += 1
    print(f"Pre_train accuracy is {(accuracy_batch/batch_counter)*100} %")

overallTestAccuracy(test_dataloader, fe_modelPer)
