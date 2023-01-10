import numpy as np
from scipy.io import arff
import torch
import pandas as pd
class Dataset:
    def __init__(self, path, val_size=(0.7, 0.1, 0.2)):
        self.path = path
        #self.test_path = path_test
        self.val_size = val_size
    def get_data(self):
        data = arff.loadarff(self.path)
        data = pd.DataFrame(data[0])
        data = data.values
        train_idx, val_idx = int(self.val_size[0]*len(data)), int((self.val_size[0]+self.val_size[1])*len(data))
        np.random.seed(225)
        np.random.shuffle(data)
        train_data, val_data, test_data = data[:train_idx].astype(np.float32), data[train_idx:val_idx].astype(np.float32), data[val_idx:].astype(np.float32) # val_ratio = 0.1
        X_train, Y_train = torch.from_numpy(train_data[:, :-4]).float(), torch.from_numpy(train_data[:, -4:]/train_data[:, -4:].max(axis=0)).float()
        X_val, Y_val = torch.from_numpy(val_data[:, :-4]).float(), torch.from_numpy(val_data[:, -4:]/val_data[:, -4:].max(axis=0)).float()
        X_test, Y_test = torch.from_numpy(test_data[:, :-4]).float(), torch.from_numpy(test_data[:, -4:]/test_data[:, -4:].max(axis=0)).float()
        train_set = torch.utils.data.TensorDataset(X_train, Y_train)
        val_set = torch.utils.data.TensorDataset(X_val, Y_val)
        test_set = torch.utils.data.TensorDataset(X_test, Y_test)
        return train_set, val_set, test_set

"""import numpy as np
from scipy.io import loadmat
import torch
class Dataset:
    def __init__(self, path_train,path_test, val_size=0):
        self.train_path = path_train
        self.test_path = path_test
        self.val_size = val_size
    def get_data(self):
        train_data = loadmat(self.train_path)['sarcos_inv'].astype(np.float32)
        np.random.shuffle(train_data)
        #val_data, train_data = train_data[:4448], train_data[4484:].astype(np.float32) # val_ratio = 0.1
        val_data, train_data = train_data[:4448], train_data[4448:].astype(np.float32) # val_ratio = 0.1
        test_data = loadmat(self.test_path)['sarcos_inv_test'].astype(np.float32)
        X_train, Y_train = torch.from_numpy(train_data[:, :21]).float(), torch.from_numpy(train_data[:, 21:]).float()
        X_val, Y_val = torch.from_numpy(val_data[:, :21]).float(), torch.from_numpy(val_data[:, 21:]).float()
        X_test, Y_test = torch.from_numpy(test_data[:, :21]).float(), torch.from_numpy(test_data[:, 21:]).float()
        train_set = torch.utils.data.TensorDataset(X_train, Y_train)
        val_set = torch.utils.data.TensorDataset(X_val, Y_val)
        test_set = torch.utils.data.TensorDataset(X_test, Y_test)
        return train_set, val_set, test_set"""