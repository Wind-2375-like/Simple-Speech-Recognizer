# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import sys
# 注意在本机需要添加系统默认路径
sys.path.append("./")
import sklearn.utils as su
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from librosa.feature import mfcc
import sklearn.metrics as sm
import scipy.signal as signal
import os
import VAD
import utils
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
np.random.seed(7)
# hyperparameters
N, M = 256, 128

LR = 0.01
EPOCH = 10
BATCH_SIZE = 1
INPUT_SIZE = 120
DOWNLOAD_MNIST = False
file_path = "./dataset/processed/"

test_x = None
test_y = None

def raw_feature_pre(store, energy, amplitude, zerocrossingrate, label):
    d = []
    d.append(energy)
    d.append(amplitude)
    d.append(zerocrossingrate)
    d.append(label)
    store.append(d)

def padding_zeros_to(data: pd.Series, length):
    ret = data.copy()
    for i in range(len(ret)):
        t = np.array(data[i]).reshape(-1)
        ret[i] = np.concatenate([t, np.zeros(length-t.shape[0])], axis=0)
    return ret

def get_mfcc_features(wave_data: pd.Series, n_mfcc):
    """
    mfcc_feature
    """
    x = wave_data.apply(lambda d: (d-np.mean(d))/(np.std(d)))
    # x = wave_data
    x, max_length = utils.padding_to_max(x)
    features = []
    for i in range(x.shape[0]):
        t1 = mfcc(x[i], sr=16000, n_mfcc=n_mfcc)
        t2 = utils.diff(t1, axis=0)
        t3 = utils.diff(t1, axis=0, delta=2)
        t = np.concatenate([t1.T, t2.T, t3.T], axis=1)
        features.append(t)   
    return np.array(features)

def padding_to_max(data: pd.Series):
    max_length = np.max([len(w) for w in data.energy])
    data.energy = padding_zeros_to(data.energy, max_length)
    data.amplitude = padding_zeros_to(data.amplitude, max_length)
    data.zerocrossingrate = padding_zeros_to(data.zerocrossingrate, max_length)
    return data, max_length

class LSTMDataset(Dataset):
    # 自定义数据（数据集大小length，每个数据为size的向量），格式为torch
    def __init__(self, time_domain_f, label, transform=None):
        super(LSTMDataset, self).__init__()
        self.data_len = np.size(time_domain_f, 0)
        self.seq_len = np.size(time_domain_f, 1)
        self.input_size = np.size(time_domain_f, 2)
        data = []
        for i in range(self.data_len):
            data.append((time_domain_f[i], label[i]))
        self.transform = transform
        self.data = data
        self.time_domain_f = time_domain_f
        self.label = label

    #返回一个数据（返回值可以改写，这里是(index,self.data[index]),可以改成（self.data[index]））
    #参数index固定，代表当前数据的index
    def __getitem__(self, index):
        feature, label = self.data[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

    # 返回数据集长度
    def __len__(self):
        return self.data_len

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_c, h_h) = self.lstm(x, None)
        out = self.out(r_out[:, -1, :])
        return out

lstm = LSTM()
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
loss_fun = nn.CrossEntropyLoss()

for window_type in ["rect", "hamming", "hanning"]:
    if window_type == 'rect':
        winfunc = 1
    elif window_type == 'hamming':
        winfunc = signal.windows.hamming(N)
    else:
        winfunc = signal.windows.hann(N)
    print("training with window of {}...".format(window_type))

    path = os.path.join(file_path, "{}.pkl".format(window_type))
    df = utils.load_pkl(path)
    wave_data = df.wave_data
    label = np.array(df.content)
    store = []

    # 特征提取
    mfcc_f = get_mfcc_features(wave_data, 40)

    # # 开始降维
    # # time_domain_f 为特征向量
    # pca = PCA(n_components=11)
    # pca.fit(mfcc_f)
    # mfcc_f = pca.transform(mfcc_f)
    
    time_domain_f = mfcc_f
    n_components = time_domain_f.shape[1]

    train_data, test_data, train_label, test_label = train_test_split(
        time_domain_f, 
        label, 
        train_size=0.7, 
        test_size=0.3
    )

    train_data = LSTMDataset(train_data, train_label, transform=transforms.ToTensor())
    test_data = LSTMDataset(test_data, test_label, transform=transforms.ToTensor)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_x = torch.from_numpy(test_data.time_domain_f).type(torch.FloatTensor)
    test_y = test_data.label.flatten()

    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.view(-1, n_components, INPUT_SIZE).type(torch.FloatTensor)
            b_y = b_y.type(torch.LongTensor)
            r_out = lstm(b_x)
            
            loss = loss_fun(r_out, b_y.flatten())
            # print(r_out)
            # print(torch.max(r_out, 1)[1].data.numpy().flatten())
            # print(b_y.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 250 : break

            if step%50 == 0:
                test_out = lstm(test_x)
                pred_y = torch.max(test_out, 1)[1].data.numpy()
                print("test:", test_x)
                print("real:", test_y)
                print("pred:", pred_y)
                accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
                print("Epoch: ", epoch, "| train loss: ", loss.data.numpy(), "| test accuracy: %.2f" % accuracy)