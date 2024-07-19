import pandas as pd
import numpy as np
from utility import dataPreprocess
from continuum import ClassIncremental
from continuum.datasets import ImageFolderDataset
from continuum.datasets import InMemoryDataset
from continuum.datasets import CIFAR100 as ICIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize,Resize

def IncrementalData(gripperSignal=False, gripperImage=False, enable_ssl=False, ssl_type=None, addNoise=False):
    if gripperImage:
        # Dataset Gripper Image 
        if ssl_type == "random":
            dataDir_train = "customDataset_random/train"
            dataDir_test = "customDataset_random/test"
        elif ssl_type == 'unique':
            dataDir_train = "customDataset_unique/train"
            dataDir_test = "customDataset_unique/test"
        elif ssl_type == "None":
            dataDir_train = "customDataset_ilfr/train"
            dataDir_test = "customDataset_ilfr/test"
        dataset_train = ImageFolderDataset(dataDir_train)
        dataset_test = ImageFolderDataset(dataDir_test)

        scenarioTrain_gripImg = ClassIncremental(dataset_train, increment=2, initial_increment=2, 
                                        transformations=[ToTensor(),Resize((32, 32)),], 
                                        class_order=np.arange(10).tolist())
        scenarioTest_gripImg = ClassIncremental(dataset_test, increment=2, initial_increment=2, 
                                        transformations=[ToTensor(),Resize((32, 32)),], 
                                        class_order=np.arange(10).tolist())
        return scenarioTrain_gripImg, scenarioTest_gripImg
    elif gripperSignal:
        # Dataset gripper
        ## Note solid cylinder refers to the gluestick
        tactileData = pd.read_csv("dataset/BP_SensorData.csv", sep=",")
        labels = ["stapler", "mobile_phone", "bottle", "solid_cylinder", "smart_watch", "plier", "airpods", "gluestick",
                    "airpods_case", "ball"]   
        gripper_trainData, gripper_trainTarget, gripper_testData, gipper_testTarget = dataPreprocess(labels, tactileData, addNoise)
        datasetTrain_gripper = InMemoryDataset(gripper_trainData, gripper_trainTarget)
        datasetTest_gripper = InMemoryDataset(gripper_testData, gipper_testTarget)

        scenarioTrain_gripper = ClassIncremental(datasetTrain_gripper, increment=2, initial_increment=2, 
                                        transformations=None, 
                                        class_order=[0,1,2,3,4,5,6,7,8,9])
        scenarioTest_gripper = ClassIncremental(datasetTest_gripper, increment=2, initial_increment=2, 
                                        transformations=None, 
                                        class_order=np.arange(10).tolist())
        return scenarioTrain_gripper, scenarioTest_gripper
    else:
        print("Incorrect selection!!")