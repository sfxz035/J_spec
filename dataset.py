import tensorflow as tf
import os
import numpy as np
import scipy.io as sio

# file_dir = '/data1/sf/J_spec/data/data1/data_train/'

def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch

def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 文件夹下的所有文件名
    file_name = []
    # 载入数据路径并写入标签值
    # list_dir = os.listdir(file_dir)
    for file in os.listdir(file_dir):
        file_name.append(file_dir+file)
    return file_name
def get_data(file_dir,input_name, label_name, is_norm = False,is_real = False):
    file_name = get_files(file_dir)
    sample_num = len(file_name)
    input_sequence, label_sequence = [], []
    for index in range(sample_num):
        data = sio.loadmat(file_name[index])
        data_x = data.get(input_name)
        if is_real == True:
            data_x = data_x.real
        data_y = data.get(label_name)
        if is_norm == True:
            data_x = norm(data_x)
            data_y = norm(data_y)
        input_sequence.append(data_x)
        label_sequence.append(data_y)
        if (index+1)%5000 == 0:
            print(index)
    data_input = np.asarray(input_sequence)
    data_label = np.asarray(label_sequence)
    return data_input,data_label
def norm(data):
    data_pre = data
    data_max = np.max(abs(data_pre))
    data = data/data_max
    return data

