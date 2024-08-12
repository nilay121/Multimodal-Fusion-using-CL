# !usr/bin/env python3
## it indicates that this is a python source file

import rospy
import time
import ast
import cv2 
import torch
import pickle
import torchvision
import numpy as np
import pandas as pd
from torch import nn
from sensor_msgs.msg import Image
## we will send messages in trhe form of images so we need to import image
from cv_bridge import CvBridge
from torch.nn import functional as F
from std_msgs.msg import String, Float32MultiArray
## cv bridge package conversts the opencv images into ros image message and vice-versa, and
## thus serves as a bridge between opencv and ros
'''
Technically we should have all the functions and classes in different scripts for neatness of the code
and to improve the readability, however using different custom modules is not that straight forward 
so we just write all the classes in a single file  
'''
###########################################################
## Load the pretrained model classes for feature extraction
###########################################################
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
        
        self.linear1 = nn.Linear(128, 2000)
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


## create the name of the publisher node 
subscriberNodeName = "camera_sensor_subscriber"

## make sure the same name is used in the source file of the publisher
topicName_video = "video_topic"
topicName_tactile = "tactile_topic"
topicName_publisher = "featureMaps_topic"

## Video publisher parameters
fps = 20.0
width = 640
height = 480
# object = "cube"
total_classes = 10
## sensor informations
sensor_data = []
sensorData_toPublish = []
visionData_toPublish = []

## Fecam stuffs
classes_map = ["stapler", "mobile_phone", "bottle", "solid_cylinder", "smart_watch", "plier", "airpods", "gluestick",
               "airpods_case", "ball"] 
trainingPhase = False

if trainingPhase:
    print("Object label is needed for training phase!! ")
    obj_name = str(input("Enter the object name: "))
    obj_idx = classes_map.index(obj_name)
else:
    print("Object label is needed for saving data!! ")
    obj_name = str(input("Enter the object name: "))

## saving video to file
current_time = time.time()
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # video codec
video_write = cv2.VideoWriter(f'ros_tests/videos/video_{obj_name}_{current_time}.mp4', fourcc, fps, (width, height))

## this function is a  callback function that is called every time the message arrives
def callbackFunctionVision(message):
    ## create a bridge object 
    bridgeObject = CvBridge()
    ## convert from cv_bridge to openCV image format
    convertedFrameBackToCV = bridgeObject.imgmsg_to_cv2(message)
    # Convert OpenCV image (NumPy array) to PyTorch tensor
    objImage_array = np.array(convertedFrameBackToCV)
    # objImage_tensor = torch.tensor(objImage_array).float()
    visionData_toPublish.append(objImage_array)
    # ##show thw image on a new window
    #cv2.imshow("camera", convertedFrameBackToCV)
    ## Save the video to a file
    video_write.write(convertedFrameBackToCV)
    ## Extract the feature maps 
    if (len(visionData_toPublish)%300==0) and len(visionData_toPublish)>0:
        print("Extracting the feature map for vision sensor......")
        model_img, _ = loadFeatureExtractors(gripperFE, imgFE)
        ## Feature extration and test phase
        imageFeatureExtractor(model_img, visionData_toPublish, False)

    ## Feature extraction and train phase
    if trainingPhase==True and len(visionData_toPublish)%1500==0:
        print("Starting the training phase --> vision part!!")
        model_img, _ = loadFeatureExtractors(gripperFE, imgFE)
        imageFeatureExtractor(model_img, visionData_toPublish, True)

    ## wait in miliseconds, 
    cv2.waitKey(1)

def callbackFunctionTactile(message):
    ## Convert the string back to a list
    message_data = ast.literal_eval(message.data)
    #rospy.loginfo(f"received sensor Data: {message}")
    sensor_data.append(message_data)
    ## when length becomes 150 apply preprocessing
    if (len(sensor_data)%150 == 0) and len(sensor_data)>0:
        struct_data = np.array(sensor_data).reshape(-1, 4).astype("float32")
        for j in range(len(sensorData_toPublish)*150, struct_data.shape[0],150):
            tempTrans = struct_data[j:j+150].reshape(1,-1)
            sensorData_toPublish.append(tempTrans[0])
        if len(sensorData_toPublish)>0:
            print("Extracting the feature map for tactile sensor......")
            _, model_sensor = loadFeatureExtractors(gripperFE, imgFE)
            ## feature extraction and test phase
            sensorSignalFeatureExtractor(model_sensor, sensorData_toPublish, False)
    ## Feature extraction and train phase
    if trainingPhase==True and len(sensor_data)%750==0:
        print("Starting the training phase --> tactile part!!")
        _, model_sensor = loadFeatureExtractors(gripperFE, imgFE)
        sensorSignalFeatureExtractor(model_sensor, sensorData_toPublish, True)

###########################################################
## Utility functions
###########################################################

def saveToCsv(data, objName):
    df = pd.DataFrame(data)
    df.to_csv(f"ros_tests/tactile/test_file_{objName}.csv", index=False)

def save_dictionary(data, dict_name):
    with open(f'saved_matrices/dict_{dict_name}.pkl', 'wb') as fp:
        pickle.dump(data, fp)
        print('dictionary saved successfully to file')

def load_dictionary(dict_name):
    with open(f'saved_matrices/{dict_name}.pkl', 'rb') as fp:
        dict_matrix = pickle.load(fp)
    return dict_matrix

def normalize_cov(cov_mat):
    norm_cov_mat = []
    for cov in cov_mat:
        sd = np.sqrt(np.diagonal(cov))  # standard deviations of the variables
        cov = cov/(np.matmul(np.expand_dims(sd,1),np.expand_dims(sd,0)) + 1e-10)
        norm_cov_mat.append(cov)

    return norm_cov_mat

def _mahalanobis(dist, cov=None, cov_dim=256):
    if cov is None:
        cov = np.eye(cov_dim)
    inv_covmat = np.linalg.pinv(cov) 
    left_term = np.matmul(dist, inv_covmat)
    mahal = np.matmul(left_term, dist.T) 
    return np.diagonal(mahal, 0)

def loadFecam_model():
    mean_matrix = load_dictionary("dict_class_mean_set")
    cov_matrix = load_dictionary("dict_cov_matrix")
    shrink_cov_matrix = load_dictionary("dict_shrink_cov_mat")
    return mean_matrix, cov_matrix, shrink_cov_matrix

def shrink_cov(cov):
    diag_mean = np.mean(np.diagonal(cov))
    off_diag = np.copy(cov)
    np.fill_diagonal(off_diag,0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / (mask.sum())
    iden = np.eye(cov.shape[0])
    alpha1 = 1
    alpha2  = 0
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_

def loadFeatureExtractors(gripperFE, imgFE):
    # # image feature extractor
    imgFE = imgFE(50)
    imgFE.load_state_dict(torch.load("pre_trained_models/GripperImageFE_32_realtime.pt", 
                                     map_location=torch.device("cpu")))
    imgFE.Linear1 = nn.Identity()
    imgFE.Linear2 = nn.Identity()
    imgFE.Linear3 = nn.Identity()
    # Sensor feature extractor
    gripperFE = gripperFE(600,15)
    gripperFE.load_state_dict(torch.load("pre_trained_models/GripperFE_realtime.pt", 
                                         map_location=torch.device("cpu")))
    gripperFE.linear1 = nn.Identity()
    gripperFE.linear2 = nn.Identity()
    gripperFE.activation4 = nn.Identity()
    print("model loaded!")
    return imgFE, gripperFE

def fecam_realtime_training(cov_mat, class_mean_set, shrink_cov_mat, offset, x_features, class_label):
    ## Check the unique class
    unique_class = class_label + offset
    class_labels = np.repeat([class_label], x_features.shape[0])
    print("extacted feature shape is ", x_features.shape)
    if unique_class not in class_mean_set.keys():
        # New class
        image_class_mask = (class_labels == (unique_class)-offset)
        class_mean_set[unique_class]= np.mean(x_features[image_class_mask],axis=0)
        cov = np.cov(x_features[image_class_mask].T)
        cov_mat[unique_class] = cov
        shrink_cov_mat[unique_class] = shrink_cov(cov_mat[unique_class])
    else:
        ## refine the previous mean, cov matrix
        print(f"Refining the knowledge base for class {classes_map[class_label]}")
        image_class_mask = (class_labels == (unique_class)-offset)
        temp_mean = np.mean(x_features[image_class_mask],axis=0)
        temp_cov = np.cov(x_features[image_class_mask].T)
        class_mean_set[unique_class] = (class_mean_set[unique_class] + temp_mean)/2
        cov_mat[unique_class] = (cov_mat[unique_class] + temp_cov)/2

    return class_mean_set, cov_mat, shrink_cov_mat


###########################################################
## Fecam realtime train test functions
###########################################################

def imageFeatureExtractor(model_img, img_batch, trainingPhase_flag):
    img_batch = torch.tensor(np.array(img_batch)).reshape(-1, 3, height, width).float()/255.0
    transform_image = torchvision.transforms.Resize((32,32))
    img_batch = transform_image(img_batch)
    currentBatch_size = img_batch.shape[0]
    model_img.eval()

    with torch.no_grad():
        _, last_layer, semi_last_layer = model_img(img_batch)
    last_layer = last_layer.reshape(currentBatch_size, -1)
    semi_last_layer = semi_last_layer.reshape(currentBatch_size, -1)
    out = F.normalize(torch.concat((last_layer, semi_last_layer), dim=1)).detach().numpy()
    print("Vision feature map shape is ", out.shape)
    mean_matrix, cov_matrix, shrink_cov_matrix = loadFecam_model()
    fecam_predictions(total_classes, True, out, mean_matrix, shrink_cov_matrix)
    
    if trainingPhase_flag:
        class_mean_set, cov_mat, shrink_cov_mat = fecam_realtime_training(cov_matrix, mean_matrix, 
                                                                          shrink_cov_matrix, 10,
                                                                          out, obj_idx)
        ## save the dictionary/overwrite it
        save_dictionary(class_mean_set,"class_mean_set")
        save_dictionary(cov_mat, "cov_matrix")
        save_dictionary(shrink_cov_mat, "shrink_cov_mat")


def fecam_predictions(total_classes, imgPred, out, class_mean_set, shrink_cov_matrix):
    maha_dist = []
    offset = 10 # number of initial classes in D1 
    norm_cov_mat = normalize_cov(shrink_cov_matrix.values())
    for cl in range(total_classes):
        cl = cl + offset if imgPred == True else cl
        distance = out - class_mean_set[cl]
        dist = _mahalanobis(distance, norm_cov_mat[cl])
        maha_dist.append(dist)
    maha_dist = np.array(maha_dist)
    pred = np.argmin(maha_dist.T, axis=1)[0]
    print(f"The prediction is {pred} and the corresponding class is {classes_map[pred]}")

def sensorSignalFeatureExtractor(model_sensor, signal_batch, trainingPhase_flag):
    signal_batch = torch.tensor(np.array(signal_batch).reshape(-1, 600, 1))
    currentBatch_size = signal_batch.shape[0]
    model_sensor.eval()
    
    with torch.no_grad():
        _, last_layer, semi_last_layer = model_sensor(signal_batch)
    last_layer = last_layer.reshape(currentBatch_size, -1)
    semi_last_layer = semi_last_layer.reshape(currentBatch_size, -1)
    out = F.normalize(torch.concat((last_layer, semi_last_layer), dim=1)).detach().numpy()
    print("Tactile feature map shape is ", out.shape)
    mean_matrix, cov_matrix, shrink_cov_matrix = loadFecam_model()
    fecam_predictions(total_classes, False, out, mean_matrix, shrink_cov_matrix)
    
    if trainingPhase_flag:
        class_mean_set, cov_mat, shrink_cov_mat = fecam_realtime_training(cov_matrix, mean_matrix, 
                                                                          shrink_cov_matrix, 0,
                                                                          out, obj_idx)
        ## save the dictionary/overwrite it
        save_dictionary(class_mean_set,"class_mean_set")
        save_dictionary(cov_mat, "cov_matrix")
        save_dictionary(shrink_cov_mat, "shrink_cov_mat")

###########################################################
## Fecam realtime re-training
###########################################################





###########################################################
## Ros Loop
###########################################################

try:
    ## ------------------------------------------- initialize the subscriber node -------------------------------------------
    rospy.init_node(subscriberNodeName, anonymous=True)
    ## we speicfy the topic name, type of the message we will receive, and the name of the callback function
    rospy.Subscriber(topicName_video, Image, callbackFunctionVision)
    ## susbscriber tactile data
    rospy.Subscriber(topicName_tactile, String, callbackFunctionTactile)
    rospy.spin()
    # ## Release the video writer
    video_write.release()
    # ## we destroy all windows
    cv2.destroyAllWindows()
    saveToCsv(sensorData_toPublish, obj_name) if trainingPhase==True else 0

    if rospy.is_shutdown():
        saveToCsv(sensorData_toPublish, obj_name) if trainingPhase==True else 0

except KeyboardInterrupt:
    ## Release the video writer
    video_write.release()
    ## we destroy all windows
    cv2.destroyAllWindows()
    saveToCsv(sensorData_toPublish, obj_name) if trainingPhase==True else 0

