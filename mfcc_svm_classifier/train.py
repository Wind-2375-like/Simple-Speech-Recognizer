# -*- coding:utf-8 -*-

import sys
# 注意在本机需要添加系统默认路径
sys.path.append("./")
from sklearn import svm
import sklearn.utils as su
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
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

N, M = 256, 128
# np.random.seed(24)
file_path = "./dataset/processed/"

def padding_zeros_to(data: pd.Series, length):
    ret = data.copy()
    for i in range(len(ret)):
        t = np.array(data[i]).reshape(-1)
        ret[i] = np.concatenate([t, np.zeros(length-t.shape[0])], axis=0)
    return ret


def padding_to_max(data: pd.Series):
    max_length = np.max([len(w) for w in data.energy])
    data.energy = padding_zeros_to(data.energy, max_length)
    data.amplitude = padding_zeros_to(data.amplitude, max_length)
    data.zerocrossingrate = padding_zeros_to(data.zerocrossingrate, max_length)
    return data, max_length

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
        t = np.concatenate([t1.T, t2.T, t3.T], axis=1).flatten()
        features.append(t)   
    return np.array(features)

def plot_classify_result(label, real, predict, filename):
    plt.figure(figsize=(8, 6))
    n_label = len(label)
    m = np.zeros([n_label, n_label])
    for r, p in zip(real, predict):
        m[int(r), int(p)] += 1
    plt.imshow(m)
    plt.xticks(label)
    plt.yticks(label)
    plt.xlabel("predict")
    plt.ylabel("real")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)


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
    # 去除噪声数据集
    # df = df[df["has_noisy"] == False]
    # df = df.reset_index()
    wave_data = df.wave_data
    label = df.content

    # 特征提取
    mfcc_f = get_mfcc_features(wave_data, 40)

    # 开始降维
    # mfcc_f 为特征向量
    pca = PCA(n_components=11)
    pca.fit(mfcc_f)
    mfcc_f = pca.transform(mfcc_f)

    # 开始训练
    x, y = su.shuffle(mfcc_f, np.array(label).astype(np.int).reshape(-1, 1))
    kf = RepeatedKFold(n_splits=5, n_repeats=5)
    train_acc = []
    test_acc = []
    f1_score = []
    max_accuracy = 0
    for train_index, test_index in kf.split(x):
        train_data, test_data, train_label, test_label = \
        x[train_index], x[test_index], y[train_index], y[test_index]
        classifier = svm.SVC(kernel='rbf', decision_function_shape='ovr')
        classifier.fit(train_data, train_label.ravel())
        train_acc.append(classifier.score(train_data,train_label))
        acc = classifier.score(test_data,test_label)
        test_acc.append(acc)
        test_m = test_label.reshape(1, -1).flatten()
        predict_m = classifier.predict(test_data)
        f1_score.append(sm.f1_score(test_m, predict_m, average=None))
        if acc > max_accuracy:
            test = test_label.reshape(1, -1).flatten()
            predict = classifier.predict(test_data)
            max_accuracy = acc

    print("train accuracy :",np.mean(np.array(train_acc)))
    print("test  accuracy :",np.mean(np.array(test_acc)))
    print("f1 score :", np.mean(np.array(f1_score), axis=0))
    print("plotting data...")
    plot_classify_result(range(10), test, predict, "./mfcc_svm_classifier/result_"+window_type+".png")
    print("done")