# -*- coding:utf-8 -*-

import sys
# 注意在本机需要添加系统默认路径
sys.path.append("./")
from sklearn import svm
import sklearn.utils as su
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
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


def padding_to_max(data: pd.Series):
    max_length = np.max([len(w) for w in data.energy])
    data.energy = padding_zeros_to(data.energy, max_length)
    data.amplitude = padding_zeros_to(data.amplitude, max_length)
    data.zerocrossingrate = padding_zeros_to(data.zerocrossingrate, max_length)
    return data, max_length

def downsampling(array, dim):
    array_len = array.shape[1]
    array = signal.savgol_filter(array, 59, 11, axis=1)
    if array_len >= dim:
        inc = int(array_len/dim)
        index = [i*inc for i in range(dim)]
        return array[:, index]
    else: 
        array = np.concatenate([array, np.zeros((array.shape[0], dim-array_len))], axis=1)
        return array

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
    store = []

    # # 减采样降维
    # for i in range(len(wave_data)):
    #     frames, num_frame = VAD.addWindow(wave_data[i].reshape(-1, 1), N, M, winfunc)
    #     energy = VAD.calEnergy(frames, N).reshape(1, num_frame).flatten()
    #     amplitude = VAD.calAmplitude(frames, N).reshape(1, num_frame).flatten()
    #     zerocrossingrate = VAD.calZeroCrossingRate(frames, N).reshape(1, num_frame).flatten()
    #     raw_feature_pre(store, energy, amplitude, zerocrossingrate, label[i])
    # raw_feature_df = pd.DataFrame(store, columns=["energy", "amplitude", "zerocrossingrate", "label"])
    # raw_feature_df, max_length = padding_to_max(raw_feature_df)

    # # 开始降维
    # # time_domain_f 为特征向量
    # energy_ss, amplitude_ss ,zerocrossingrate_ss = [], [], []
    # for i in range(len(raw_feature_df)):
    #     energy_ss.append(raw_feature_df["energy"][i])
    #     amplitude_ss.append(raw_feature_df["amplitude"][i])
    #     zerocrossingrate_ss.append(raw_feature_df["zerocrossingrate"][i])
    # energy_ss = np.array(energy_ss).reshape(-1, max_length)
    # amplitude_ss = np.array(amplitude_ss).reshape(-1, max_length)
    # zerocrossingrate_ss = np.array(zerocrossingrate_ss).reshape(-1, max_length)
    # dim = 99
    # new_energy_ss = downsampling(energy_ss, dim)
    # new_amplitude_ss = downsampling(amplitude_ss, dim)
    # new_zerocrossingrate_ss = downsampling(zerocrossingrate_ss, dim)
    # dim = 99
    # time_domain_f = np.concatenate([new_energy_ss, new_amplitude_ss, new_zerocrossingrate_ss], axis=1)
    # # time_domain_f = downsampling(time_domain_f, dim)

    # 开始填零
    for i in range(len(wave_data)):
        frames, num_frame = VAD.addWindow(wave_data[i].reshape(-1, 1), N, M, winfunc)
        energy = VAD.calEnergy(frames, N).reshape(1, num_frame).flatten()
        amplitude = VAD.calAmplitude(frames, N).reshape(1, num_frame).flatten()
        zerocrossingrate = VAD.calZeroCrossingRate(frames, N).reshape(1, num_frame).flatten()
        raw_feature_pre(store, energy, amplitude, zerocrossingrate, label[i])
    raw_feature_df = pd.DataFrame(store, columns=["energy", "amplitude", "zerocrossingrate", "label"])
    raw_feature_df, max_length = padding_to_max(raw_feature_df)

    # 开始降维
    # time_domain_f 为特征向量
    pca = PCA(n_components=33)
    energy_ss, amplitude_ss ,zerocrossingrate_ss = [], [], []
    for i in range(len(raw_feature_df)):
        energy_ss.append(raw_feature_df["energy"][i])
        amplitude_ss.append(raw_feature_df["amplitude"][i])
        zerocrossingrate_ss.append(raw_feature_df["zerocrossingrate"][i])
    energy_ss = np.array(energy_ss).reshape(-1, max_length)
    amplitude_ss = np.array(amplitude_ss).reshape(-1, max_length)
    zerocrossingrate_ss = np.array(zerocrossingrate_ss).reshape(-1, max_length)
    pca.fit(energy_ss)
    pca.fit(amplitude_ss)
    pca.fit(zerocrossingrate_ss)
    new_energy_ss = pca.transform(energy_ss)
    new_amplitude_ss = pca.transform(amplitude_ss)
    new_zerocrossingrate_ss = pca.transform(zerocrossingrate_ss)
    time_domain_f = np.concatenate([new_energy_ss, new_amplitude_ss, new_zerocrossingrate_ss], axis=1)
    pca = PCA(n_components=11)
    pca.fit(time_domain_f)
    time_domain_f = pca.transform(time_domain_f)

    # 开始训练
    x, y = su.shuffle(time_domain_f, np.array(raw_feature_df["label"]).astype(np.int).reshape(-1, 1))
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
    plot_classify_result(range(10), test, predict, "./td_svm_classifier/result_"+window_type+".png")
    print("done")