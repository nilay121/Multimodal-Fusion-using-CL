import os
import shutil
import numpy as np
from utility import random_fileNames, strategic_fileNames
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize,Resize

def generateUnlabelledData_random():
    # Define the source and destination folders
    numb_samples = 150
    source_folder = 'customDataset/train/'
    destination_folder = 'customDataset/unsupervised/0_/'

    # # List all files in the source folder
    for i in range(10):
        source_folder = f'customDataset/train/object{i}'
        files = random_fileNames(numb_samples)
        for file in files:
            destination_fileName = str(i)+file
            source_file_path = os.path.join(source_folder, file)
            destination_file_path = os.path.join(destination_folder, destination_fileName)
            shutil.move(source_file_path, destination_file_path)
            print(f"Moved {file} of object {i} to {destination_folder}")

# def generateUnlabelledData_strategic():
#     # Define the source and destination folders
#     source_folder = 'customDataset/train/'
#     destination_folder = 'customDataset/unsupervised/'

#     for i in range(10):
#         source_folder = f'customDataset/train/object{i}'
#         files = strategic_fileNames()
#         for file in files:
#             destination_fileName = str(i)+file
#             source_file_path = os.path.join(source_folder, file)
#             destination_file_path = os.path.join(destination_folder, destination_fileName)
#             shutil.move(source_file_path, destination_file_path)
#             print(f"Moved {file} of object {i} to {destination_folder}")

def loadUnlabelledData(ssl_type):
    if ssl_type == "random":
        train_transform_gripperImage = Compose([ToTensor(),Resize((32, 32)),])
        gripperImage_train = datasets.ImageFolder(root="customDataset_random/unsupervised", 
                                                  transform=train_transform_gripperImage)
    elif ssl_type == "unique":
        train_transform_gripperImage = Compose([ToTensor(),Resize((32, 32)),])
        gripperImage_train = datasets.ImageFolder(root="customDataset_unique/unsupervised", 
                                                  transform=train_transform_gripperImage) 
    else:
        raise NotImplementedError("No such folder exists!!")       
    return gripperImage_train


if __name__=="__main__":
    generateUnlabelledData_random()
    print("Unlabelled data generation completed!! ")