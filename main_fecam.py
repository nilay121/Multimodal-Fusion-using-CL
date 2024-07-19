import torch
# import timm
import numpy as np
from torch import nn
from imgFE import imgFE
from gripperFE import gripperFE
from dataset import IncrementalData
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utility import save_list_to_txt
from argparse import ArgumentParser
from imgFE import imageFeature_extractor
from gripperFE import signalFeature_extractor
from train_test import train_phase, test_phase


def repeat(enable_ilfr, enable_ssl, ssl_type, gripperImg, gripperFE, 
           scenarioTrain_gripper, scenarioTrain_gripImg): 
    class_mean_set = {}
    accuracy_first_call = []
    accuracy_second_call = []
    cov_mat = {}
    shrink_cov_mat = {}

    shrink = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ## Train Test Gripper Signal
    for task_id, train_dataset in enumerate(scenarioTrain_gripper):
        class_mean_set, cov_mat, \
        shrink_cov_mat, accuracy_first_call = train_phase(train_dataset, scenarioTest_gripper, task_id, 
                                                    gripperFE, class_mean_set, cov_mat, 
                                                    shrink_cov_mat, shrink, "Signal", 40, 10, 0,
                                                    accuracy_first_call, enable_ilfr, enable_ssl,
                                                    ssl_type, device)
        
    print(f"Total incremental accuracy for Gripper Signal {round(np.mean(np.array(accuracy_first_call))* 100,2)} ")

    # -----------------------------------------------------------------------
    ## Train Phase Gripper Image
    for task_id, train_dataset in enumerate(scenarioTrain_gripImg):
        class_mean_set, cov_mat, \
        shrink_cov_mat, accuracy_first_call = train_phase(train_dataset, scenarioTest_gripImg, task_id, 
                                                    gripperImg, class_mean_set, cov_mat, 
                                                    shrink_cov_mat, shrink, "Image", 225, 50, 10,
                                                    accuracy_first_call, enable_ilfr, enable_ssl,
                                                    ssl_type, device)
        ## Gripper signal test
        '''
        Gripper signal accuracy : here though we are just checking the accuracy upto the 
        current class of the gripper image dataset, but technically we can check the accuracy 
        of all the classes because the model has already been trained on all the classes.
        Note: the accuracy of the of the previous classes will be somwhat similar to the 
        accuracy of the last class trained on the gripper signal dataset.  
        '''
        accuracy_second_call = test_phase(scenarioTest_gripper, task_id, gripperFE, 10, 0, class_mean_set, shrink_cov_mat,
                    accuracy_second_call, 5, "Signal 2nd", True, enable_ilfr, device)
    print(f"Total incremental accuracy {((np.array(accuracy_first_call)[5:].mean()+np.array(accuracy_second_call).mean())/2)*100} %")
    print("Mean matrix values are ", np.array(list(class_mean_set.values())).shape)
    print("Cov matrix values are ", np.array(list(cov_mat.values())).shape)
    
    return accuracy_first_call, accuracy_second_call

if __name__ == "__main__":

    parser = ArgumentParser(description="FeCAM custom dataset")
    parser.add_argument("--enable_ilfr", type=str, required=True, help="enable intra layer feature representation", choices=["True", "False"])
    parser.add_argument("--enable_ssl", type=str, required=True, help="enable semi supervised learning", choices=["True", "False"])
    parser.add_argument("--ssl_type", type=str, required=True, help="Type of semi-supervised learning experiment", choices=["None", "unique", "random"])
    parser.add_argument("--add_noise", type=str, required=True, help="Add gaussian noise to domain 1", choices=["True", "False"])

    args = parser.parse_args()
    
    repeat_accuracy = []
    count = 3
    enable_ilfr = args.enable_ilfr
    enable_ssl = args.enable_ssl
    ssl_type = args.ssl_type #choices ("unique", "random", "None")
    addNoise = args.add_noise

    '''
        Intra layer feature representation
            - True (enable)
            - False (disbale) ---> bench
        enable SSL
            - True (enable)
            - False (disbale)
        SSL type 
            - None ---> to perform the ILFR bench test
            - random ---> to perform random experiment for both ssl and bench
            - unique ---> to perform unique experiment for both ssl and bench
        Add noise 
            - True (adds noise to gripper signal data)
            - False
    '''

    for i in range(count):
        ## Pre trained model gripper image model
        gripperImg = imageFeature_extractor().cuda()
        gripperFE = signalFeature_extractor().cuda()

        ## Gripper signal dataset
        scenarioTrain_gripper, scenarioTest_gripper = IncrementalData(True, False, enable_ssl, ssl_type, addNoise)
        scenarioTrain_gripImg, scenarioTest_gripImg = IncrementalData(False, True, enable_ssl, ssl_type, addNoise)

        accu_first_call, accu_second_call = repeat(enable_ilfr, enable_ssl, ssl_type, gripperImg, gripperFE,
                                                   scenarioTrain_gripper, scenarioTrain_gripImg)
        repeat_accuracy.append(accu_first_call[5:]+accu_second_call)
    print("accu first call ", accu_first_call)
    print("accu second call ", accu_second_call)
    repeat_accuracy = np.array(repeat_accuracy)
    # Save the accuracy history to disk after N repeats
    np.savetxt(f"with_ifcl{enable_ilfr}_with_ssl{args.enable_ssl}_type{args.ssl_type}_noise{args.add_noise}.txt", repeat_accuracy)