import torchvision
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Compose
from torchvision import datasets, transforms
from utility import dataPreprocess, CustomDatasetForDataLoader
from avalanche.benchmarks.datasets import CORe50Dataset

def preTrain_gripperData():
    # Gripper Dataset
    tactileData = pd.read_csv("dataset/BP_SensorData.csv", sep=",")
    labels = ['solid_square', 'solid_cylinder', 'stapler', 'airpods_case', 'gluestick', 'bottle', 'rectangular_block', 
            'hollow_cylinder','airpods', 'mobile_phone', 'smart_watch', 'air_guage', 'big_cuboid', 'plier', 'ball']

    gripper_trainData, gripper_trainTarget, gripper_testData, gipper_testTarget = dataPreprocess(labels, tactileData)
    # Create dataset
    gripper_train = CustomDatasetForDataLoader(gripper_trainData, gripper_trainTarget)
    gripper_test = CustomDatasetForDataLoader(gripper_testData, gipper_testTarget)
    return gripper_train, gripper_test

def preTrain_core50Data(mini):
    train_transforms = transforms.ToTensor()
    test_transforms = transforms.ToTensor()
    ## Implement the core 50 one
    core50_train = CORe50Dataset(train=True, transform=train_transforms, mini=mini)
    core50_test = CORe50Dataset(train=False, transform=test_transforms, mini=mini)
    return core50_train, core50_test

def preTrain_gripperImageData():
    # ------------Custom Gripper Objects --------------
    dataDir_train = "customDataset_ilfr/train"
    dataDir_test = "customDataset_ilfr/test"
    train_transform_gripperImage = Compose([transforms.Resize((32, 32)), transforms.ToTensor(),])
    test_transform_gripperImage = Compose([transforms.Resize((32, 32)),transforms.ToTensor(),])

    gripperImage_train = datasets.ImageFolder(root=dataDir_train, transform=train_transform_gripperImage)
    gripperImage_test = datasets.ImageFolder(root=dataDir_test, transform=test_transform_gripperImage)
    return gripperImage_train, gripperImage_test



if __name__=="__main__":
    preTrain_core50Data(True)