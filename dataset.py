import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
from sklearn import preprocessing


def data_load(path,featur_name,label_name):
    data = sio.loadmat(path)
    featur = data.get(featur_name)
    label = data.get(label_name)
    featur_rl = featur.real
    featur_rl = np.expand_dims(featur_rl,-1)
    featur_im = featur.imag
    featur_im = np.expand_dims(featur_im,-1)
    xinp = np.concatenate((featur_rl, featur_im),axis=2)
    xinp = xinp.astype(np.float32)
    lable_rl = label.real
    yrl = lable_rl.astype(np.float32)
    return xinp,yrl
def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch
def norm(data_train,data_test,version):
    if version == 1:
        ## 1.自定义max-min归一化
        train_max = np.max(data_train)
        train_min = np.min(data_train)

        train_norm = (data_train -train_min)/ (train_max - train_min)
        test_norm = (data_test -train_min)/ (train_max - train_min)
        ## 2.processing MinMax归一化
        # min_max_scaler = preprocessing.MinMaxScaler()
        # data_norm = min_max_scaler.fit_transform(data_pre)
        # data_norm = np.zeros(data_pre.shape, dtype=np.float64)
        ## 3.cv归一化
        # cv.normalize(data_pre, data_norm, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    if version == 2:
        ## 1.自定义标准化
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        data_shape = np.shape(data_train)
        for i in range(data_shape[-1]):
            train_mean = np.mean(data_train[:,:,:,i])
            train_std = np.std(data_train[:,:,:,i])
            data_train[:, :, :, i] = (data_train[:,:,:,i] - train_mean) / train_std
            data_test[:, :, :, i] = (data_test[:,:,:,i] - train_mean) / train_std
        ## 2.processing scale标准化
        # data_norm = preprocessing.scale(data_pre)
        ## 3.processing standardscaler标准化
        # data_norm = preprocessing.StandardScaler().fit(data_pre)
    return data_train,data_test
