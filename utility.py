import torch
import numpy as np
import joblib
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class CustomDatasetForDataLoader(Dataset):
    def __init__(self,data,targets):
        # convet labels to 1 hot
        self.data = data
        self.targets = targets
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],self.targets[idx]
    

def dataPreprocess(labels, tactileData, addNoise):
        modifiedData = []
        modifiedLabels = []
        n_classes = len(labels)
        ## transform the data to desired format
        for i in range(len(labels)):
            dataToTransform = np.array(tactileData.loc[tactileData['object'] == labels[i], ['SensorVal1', 'SensorVal2', 'SensorVal3', 
                                                                                            'SensorVal4']])
            for j in range(0,dataToTransform.shape[0],150):
                tempTrans = dataToTransform[j:j+150].reshape(1,-1)
                modifiedLabels.append(i)
                modifiedData.append(tempTrans)

        modifiedLabels = np.array(modifiedLabels)
        modifiedData = np.array(modifiedData).squeeze(1)

        train_indices, val_indices, _, _ = train_test_split(range(len(modifiedLabels)),modifiedLabels,stratify=modifiedLabels,
                test_size=0.20)
        
        trainData = modifiedData[train_indices]
        trainLabels = modifiedLabels[train_indices]
        testData = modifiedData[val_indices]
        testLabels = modifiedLabels[val_indices]

        # # normalize features to range 0 to 1
        X_normalizedTrain = trainData.astype("float32")
        X_normalizedTest = testData.astype("float32")
        ## Add gaussian noise to features
        if addNoise:
            X_normalizedTrain = X_normalizedTrain + np.random.normal(0,0.2,(X_normalizedTrain.shape[0], X_normalizedTrain.shape[1]))

        X_normalizedTrain, trainLabels = data_restructure(X_normalizedTrain, trainLabels, n_classes)
        X_normalizedTest, testLabels = data_restructure(X_normalizedTest, testLabels, n_classes)

        return X_normalizedTrain, trainLabels, X_normalizedTest, testLabels

def uniqueFunc(list_numb):
    unique_list = []
    for i in list_numb:
        if i not in unique_list:
            unique_list.append(i)
    return unique_list

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

def normalize_cov(cov_mat):
    norm_cov_mat = []
    for cov in cov_mat:
        sd = np.sqrt(np.diagonal(cov))  # standard deviations of the variables
        cov = cov/(np.matmul(np.expand_dims(sd,1),np.expand_dims(sd,0)))
        norm_cov_mat.append(cov)

    return norm_cov_mat

def _mahalanobis(dist, cov=None, cov_dim=256):
    if cov is None:
        cov = np.eye(cov_dim)
    inv_covmat = np.linalg.pinv(cov) 
    left_term = np.matmul(dist, inv_covmat)
    mahal = np.matmul(left_term, dist.T) 
    return np.diagonal(mahal, 0)

def linearMlp_model(model, input_data, input_labels, opt, criterion, running_accuracy):
    opt.zero_grad()
    pred = model(input_data)
    loss = criterion(pred, input_labels)
    loss.backward()
    batch_accu = running_accuracy(pred, input_labels)
    opt.step()
    return loss, batch_accu

def expAccuracyExtractor(acc_dict):
    y_stable=[]
    cls_output = []

    for i in range(0,len(acc_dict)):
        y_stable.append(np.array(list(acc_dict.values())[i][0].cpu()))
    cls_output = [y_stable[outputs] for outputs in range(len(y_stable))]
    cls_output = np.array(cls_output)
    return np.round(cls_output, decimals=6)

def data_restructure(feature_data, targets_data, n_classes):
    aligned_data = []
    aligned_targets = []
    for i in range(n_classes):
        mask = (targets_data == i)
        aligned_data.append(feature_data[mask])
        aligned_targets.append(targets_data[mask])
    return np.concatenate(aligned_data), np.concatenate(aligned_targets)

def normalizer(data):
    zeroOneRange = ((data-data.min())/(data.max()-data.min()))
    return zeroOneRange

def random_fileNames(samples):
    rand_names = []
    while len(rand_names) < samples:
        temp_name = f"repeatation{np.random.randint(1,10)}_{np.random.randint(0,150)}.jpg"
        if temp_name not in rand_names:
            rand_names.append(temp_name)
        else:
            print("Same file name encountered!!")
    print("rand names size ", len(rand_names))
    return rand_names

def strategic_fileNames():
    file_names = []
    for i in range(7, 10):
        for j in range(0, 150):
            file_names.append(f"repeatation{i}_{j}.jpg")
    return file_names

def save_list_to_txt(filename, lst):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')



