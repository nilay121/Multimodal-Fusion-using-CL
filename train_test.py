import torch
import timm
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from utility import uniqueFunc, shrink_cov, normalize_cov,\
                    _mahalanobis
from unsupervised_dataset import loadUnlabelledData

def train_phase(train_dataset, scenario_test, task_id, featureExtractor, class_mean_set, 
                cov_mat, shrink_cov_mat, shrink, datasetType, train_batch, test_batch,
                offset, accuracy_history, enable_ilfr, enable_ssl, ssl_type, device):
    
    train_loader = DataLoader(train_dataset, batch_size=train_batch)
    offset_class = offset # number of classes of the previous domain
    n_classes_per_exp = 2
    temp_featureLabel_dict = {}
    store_features_per_class = 5
    num_cls_img = (task_id+1)*n_classes_per_exp 

    print(f"Gripper {datasetType} Task id is {task_id}, and num classes are {num_cls_img}")
    featureExtractor.eval()
    for (img_batch,label,t) in train_loader:
        img_batch = img_batch.to(device) if datasetType=="Image" else img_batch.reshape(-1, 600, 1).to(device)
        currentBatch_size = img_batch.shape[0]
        
        with torch.no_grad():
            _, last_layer, semi_last_layer = featureExtractor(img_batch)

        '''For custom dataset use single normalizer for both the layers'''
        last_layer = last_layer.reshape(currentBatch_size, -1)
        semi_last_layer = semi_last_layer.reshape(currentBatch_size, -1)
        
        # Perform intra laye feature representation?
        if enable_ilfr=="True":
            out = F.normalize(torch.concat((last_layer, semi_last_layer), dim=1)).detach().cpu().numpy()
        else:
            out = F.normalize(last_layer).detach().cpu().numpy()
        
        X_temp = out
        y_temp = label.detach().cpu().numpy()
        # prototype training
        class_mean_set, cov_mat, shrink_cov_mat = prototype_training(class_mean_set, cov_mat, 
                                                                     shrink_cov_mat, X_temp, y_temp, 
                                                                     offset_class, shrink)
        ## save some feature maps and labels for calculating the mahalanobis threshold for every class
        if datasetType == "Image":
            rand_idx = np.random.randint(0, currentBatch_size)
            label_temp = y_temp[rand_idx]
            if label_temp not in temp_featureLabel_dict.keys():
                temp_featureLabel_dict[label_temp] = [X_temp[rand_idx]]
            elif (label_temp in temp_featureLabel_dict.keys()) and (len(temp_featureLabel_dict[label_temp]) <= store_features_per_class):
                temp_featureLabel_dict[label_temp].append(X_temp[rand_idx])
    
    ## pseudo labelling using ssl
    if (datasetType=="Image") and (enable_ssl=="True"):
        if ssl_type == "random":
            threshold = 0.6
            start_class = num_cls_img - n_classes_per_exp
            class_mean_set, cov_mat, shrink_cov_mat = unlabelledPrediction(featureExtractor, start_class, num_cls_img, 
                                                                        offset_class, class_mean_set, cov_mat, 
                                                                        shrink_cov_mat, shrink, temp_featureLabel_dict, 
                                                                        threshold, enable_ilfr, ssl_type, device)
        elif ssl_type == "unique":
            threshold = 0.55
            start_class = num_cls_img - n_classes_per_exp
            class_mean_set, cov_mat, shrink_cov_mat = unlabelledPrediction(featureExtractor, start_class, num_cls_img, 
                                                                        offset_class, class_mean_set, cov_mat, 
                                                                        shrink_cov_mat, shrink, temp_featureLabel_dict, 
                                                                        threshold, enable_ilfr, ssl_type, device)
        else:
            print("Wrong ssl type enetered!!")
    else:
        pass    

    # Test phase
    accuracy_history = test_phase(scenario_test, task_id, featureExtractor, num_cls_img, 
               offset_class, class_mean_set, shrink_cov_mat, accuracy_history, test_batch,
               datasetType, True, enable_ilfr, device)
    return class_mean_set, cov_mat, shrink_cov_mat, accuracy_history


def prototype_training(class_mean_set, cov_mat, shrink_cov_mat, X_temp, 
                        y_temp, offset_class, shrink):
    ## Check the unique class
    unique_class = uniqueFunc(y_temp)[-1] + offset_class
    # print("unique class is ", unique_class)
    if unique_class not in class_mean_set.keys():
        # New class
        image_class_mask = (y_temp == (unique_class)-offset_class)
        class_mean_set[unique_class]= np.mean(X_temp[image_class_mask],axis=0)
        cov = np.cov(X_temp[image_class_mask].T)
        cov_mat[unique_class] = cov
        if shrink:
            shrink_cov_mat[unique_class] = shrink_cov(cov_mat[unique_class])
    else:
        image_class_mask = (y_temp == (unique_class)-offset_class)
        temp_mean = np.mean(X_temp[image_class_mask],axis=0)
        temp_cov = np.cov(X_temp[image_class_mask].T)
        class_mean_set[unique_class] = (class_mean_set[unique_class] + temp_mean)/2
        cov_mat[unique_class] = (cov_mat[unique_class] + temp_cov)/2
    return class_mean_set, cov_mat, shrink_cov_mat

def unlabelledPrediction(featureExtractor, start_class, num_cls_img, offset_class, class_mean_set, cov_mat,
                         shrink_cov_mat, shrink, temp_featureLabel_dict, threshold, enable_ilfr, ssl_type, device):
        dataset_unlabelled = loadUnlabelledData(ssl_type)
        norm_cov_mat = normalize_cov(shrink_cov_mat.values())
        cos_sim = torch.nn.CosineSimilarity(dim=1)
        featureExtractor.eval()

        for cl in range(start_class, num_cls_img):
            expUnsupLabels = []
            expUnsupFeatureMaps = []
            buffer_featureMaps = torch.tensor(np.array(temp_featureLabel_dict[cl]), dtype=torch.float32)
            unlabelled_loader = DataLoader(dataset_unlabelled, batch_size=buffer_featureMaps.shape[0], shuffle=False)
            cl = cl + offset_class  # add the offset class to the starting classes for domain 2
            for img_batch,_ in tqdm(unlabelled_loader):

                currentBatch_size = img_batch.shape[0]
                img_batch = img_batch.to(device)
                with torch.no_grad():
                    _, last_layer, semi_last_layer = featureExtractor(img_batch)

                '''For custom dataset use single normalizer for both the layers'''
                last_layer = last_layer.reshape(currentBatch_size, -1)
                semi_last_layer = semi_last_layer.reshape(currentBatch_size, -1)
                
                # Perform intra laye feature representation?
                if enable_ilfr=="True":
                    out = F.normalize(torch.concat((last_layer, semi_last_layer), dim=1)).detach().cpu().numpy()
                else:
                    out = F.normalize(last_layer).detach().cpu().numpy()

                ## Check the similarity between the previous labelled features and the unlabelled features
                out_tensor = torch.tensor(out, dtype=torch.float32)
                similarity = cos_sim(buffer_featureMaps, out_tensor)
                sim_map = (similarity.detach().numpy()>=threshold)
                if sim_map.sum() > 0:
                    expUnsupFeatureMaps.append(out[sim_map])
                    expUnsupLabels.append(np.repeat((cl-offset_class), sim_map.sum()))

            print(f"Unlabelled data extracted for class {cl}")
            if len(expUnsupLabels) > 0:
                print("Starting the unsupervised training.....")
                expUnsupLabels = np.concatenate(expUnsupLabels)
                expUnsupFeatureMaps = np.concatenate(expUnsupFeatureMaps)
                print("exp labels shape ", expUnsupLabels.shape)
                print("exp features shape ", expUnsupFeatureMaps.shape)
                class_mean_set, cov_mat, shrink_cov_mat = prototype_training(class_mean_set, cov_mat, 
                                                                            shrink_cov_mat, expUnsupFeatureMaps, 
                                                                            expUnsupLabels, offset_class, shrink)
        return class_mean_set, cov_mat, shrink_cov_mat

def test_phase(scenario_test, task_id, featureExtractor, num_cls_img,
               offset_class, class_mean_set, shrink_cov_mat, accuracy_history,
               test_batch, datasetType, store_accu, enable_ilfr, device):
    ## Test Phase Gripper Image
    norm_cov_mat = normalize_cov(shrink_cov_mat.values())
    test_ds = scenario_test[:task_id+1]
    test_loader = DataLoader(test_ds, batch_size=test_batch, shuffle=False)
    correct , total = 0 , 0
    featureExtractor.eval()

    for (img_batch,label,t) in tqdm(test_loader):
        img_batch = img_batch.to(device) if datasetType=="Image" else img_batch.reshape(img_batch.shape[0], 600, 1).to(device)
        currentBatch_size = img_batch.shape[0]
        with torch.no_grad():
            _, last_layer, semi_last_layer = featureExtractor(img_batch)

        '''For custom dataset use single normalizer for both the layers'''
        last_layer = last_layer.reshape(currentBatch_size, -1)
        semi_last_layer = semi_last_layer.reshape(currentBatch_size, -1)
        # Perform intra laye feature representation?
        if enable_ilfr=="True":
            out = F.normalize(torch.concat((last_layer, semi_last_layer), dim=1)).detach().cpu().numpy()
        else:
            out = F.normalize(last_layer).detach().cpu().numpy()
        
        predictions = []
        maha_dist = []
        for cl in range(num_cls_img):
            cl = cl + offset_class  # add the offset class to the starting classes for domain 2
            distance = out - class_mean_set[cl]
            dist = _mahalanobis(distance, norm_cov_mat[cl])
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)
        pred = np.argmin(maha_dist.T, axis=1)
        predictions.append(pred)

        predictions = torch.tensor(np.array(predictions))
        correct += (predictions.detach().cpu().numpy() == label.detach().cpu().numpy()).sum()
        total += label.shape[0]
    print(f"{'..'*10}Gripper {datasetType} accuracy at experience {task_id} is {correct/total}")
    accuracy_history.append(correct/total) if store_accu is True else 0
    return accuracy_history

