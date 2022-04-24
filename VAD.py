# -*- coding: utf-8 -*-
import struct
import os
import pandas as pd
import wave
import numpy as np
import json
import zipfile
import matplotlib.pyplot as plt
import pickle
import scipy.signal as signal

# 将record.wav输入信号 采样 切割 转化成若干processedi.pcm文件

# 通过高门限的一定是需要截取的音，但是只用高门限，会漏掉开始的清音
# 如果先用低门限，噪音可能不会被滤除掉
# 先用高门限确定是不是噪音，再用低门限确定语音开始

def calEnergy(frames, N):
    """ 返回每一帧的短时能量energy
        frames: 帧信号矩阵
        N: 一帧采样点个数
    """
    energy = []
    energy = np.sum(frames**2, axis=1)  # 计算帧信号矩阵每一行的平方和
    return energy

def calAmplitude(frames, N):
    """ 返回每一帧的短时幅度amplitude
        frames: 帧信号矩阵
        N: 一帧采样点个数
    """
    amplitude = []
    amplitude = np.sum(np.abs(frames), axis=1)  # 计算帧信号矩阵每一行的绝对值和
    return amplitude

def calZeroCrossingRate(frames, N):
    """ 返回每一帧的短时过零率zerocrossrate
        frames: 帧信号矩阵
        N: 一帧采样点个数
        T: 过滤低频
    """
    TH = np.mean(np.abs(frames))
    T = (np.mean(np.abs(frames[0])) + TH*4) / 4          

    zerocrossingrate = []
    zerocrossingrate = np.mean(
        np.abs(np.sign(frames[:, 1:N-1]-T)-np.sign(frames[:, 0:N-2]-T))+
        np.abs(np.sign(frames[:, 1:N-1]+T)-np.sign(frames[:, 0:N-2]+T)), axis=1)
    return zerocrossingrate

def detectEndPoint(wave_data, energy, zerocrossingrate):
    """ 利用短时能量/短时幅度，短时过零率，使用双门限法进行端点检测
        返回端点对应的帧序号endpoint0
        wave_data: 向量存储的语音信号
        energy: 一帧采样点个数
    """
    smooth_energy = energy
    smooth_zcr = zerocrossingrate
    # smooth_energy = signal.savgol_filter(energy, 29, 1)             # 利用savgol滤波器对能量和过零率进行平滑
    # smooth_zcr = signal.savgol_filter(zerocrossingrate, 29, 1)
    
    gap = int(len(wave_data)/20000)
    TH = np.mean(smooth_energy) / 4                                    # 较高能量门限
    TL = (np.mean(smooth_energy[:5]) / 5 + TH) / 4                # 较低能量门限
    T0 = np.mean(smooth_zcr) / 4      # 过零率门限
    endpointH = []  # 存储高能量门限 端点帧序号
    endpointL = []  # 存储低能量门限 端点帧序号
    endpoint0 = []  # 存储过零率门限 端点帧序号

    # 先利用较高能量门限 TH 筛选语音段
    flag = 0
    for i in range(len(smooth_energy)):
        # 左端点判断，第一个左端点加入
        if flag == 0 and smooth_energy[i] >= TH and len(endpointH) == 0:
            endpointH.append(i)
            flag = 1
        # 左端点判断，距离前一个右端点距离远则加入
        # 否则去掉这个左端点和前一个右端点，从上一个左端点重新计算，去除毛刺
        elif flag == 0 and smooth_energy[i] >= TH and i-gap > endpointH[len(endpointH)-1]:
            endpointH.append(i)
            flag = 1
        elif flag == 0 and smooth_energy[i] >= TH and i-gap <= endpointH[len(endpointH)-1]:
            endpointH = endpointH[:len(endpointH)-1]
            flag = 1

        # 右端点判断，检测帧长，太短则舍弃
        if flag == 1 and smooth_energy[i] < TH:
            if i - endpointH[len(endpointH)-1] <= gap:
                endpointH = endpointH[:len(endpointH)-1]
            else:
                endpointH.append(i)
            flag = 0

    # 再利用较低能量门限 TL 扩展语音段
    for j in range(len(endpointH)):
        i = endpointH[j]

        # 对右端点向右搜索
        if j % 2 == 1:
            while i < len(smooth_energy) and smooth_energy[i] >= TL:
                i = i + 1
            endpointL.append(i)

        # 对左端点向左搜索
        else:
            while i > 0 and smooth_energy[i] >= TL:
                i = i - 1
            endpointL.append(i)

    # 最后利用过零率门限 T0 得到最终语音段
    for j in range(len(endpointL)):
        i = endpointL[j]

        # 对右端点向右搜索
        if j % 2 == 1:
            while i < len(smooth_zcr) and smooth_zcr[i] >= T0:
                i = i + 1
            endpoint0.append(i)

        # 对左端点向左搜索
        else:
            while i > 0 and smooth_zcr[i] >= 3*T0:
                i = i - 1
            endpoint0.append(i)
    # 用于在VAD.detectEndPoint()中调试代码
    # plt.subplot(3, 1, 1)
    # plt.plot(np.arange(len(smooth_energy)), smooth_energy)
    # for i in endpointH: plt.axvline(x=i, color='r')
    # plt.axhline(y=TH, color='g')

    # plt.subplot(3, 1, 2)
    # plt.plot(np.arange(len(smooth_energy)), smooth_energy)
    # for i in endpointL: plt.axvline(x=i, color='r')
    # plt.axhline(y=TL, color='g')

    # plt.subplot(3, 1, 3)
    # for i in endpoint0: plt.axvline(x=i, color='r')
    # plt.plot(np.arange(len(smooth_zcr)), smooth_zcr)
    # plt.axhline(y=T0, color='g')

    # plt.show()            
    return endpoint0

def addWindow(wave_data, N, M, winfunc):
    """ 将音频信号转化为帧并加窗
        返回帧信号矩阵:维度(帧个数, N)以及帧数num_frame
        wave_data: 待处理语音信号
        N: 一帧采样点个数
        M: 帧移（帧交叠间隔）
        winfunc: 加窗函数
    """
    wav_length = len(wave_data)     # 音频信号总长度
    inc = N - M                     # 相邻帧的间隔
    if wav_length <= N:             # 若信号长度小于帧长度，则帧数num_frame=1
        num_frame = 1
    else:
        num_frame = int(np.ceil((1.0*wav_length-N)/inc + 1))
    pad_length = int((num_frame-1)*inc+N)               # 所有帧加起来铺平后长度
    zeros = np.zeros((pad_length-wav_length, 1))  # 不够的长度用0填补
    pad_wavedata = np.concatenate((wave_data, zeros))  # 填补后的信号
    indices = np.tile(np.arange(0, N), (num_frame, 1)) + \
        np.tile(np.arange(0, num_frame*inc, inc), (N, 1)
                ).T   # 用来对pad_wavedata进行抽取，得到num_frame*N的矩阵
    frames = pad_wavedata[indices].reshape(num_frame, N)            # 得到帧信号矩阵
    window = np.tile(winfunc, (num_frame, 1))
    return window * frames, num_frame                   # 加窗

def readWav(filename):
    """ 读取音频信号并转化为向量
        返回向量存储的语音信号wave_data及参数信息params
    """

    # 读入wav文件
    f = wave.open(filename, "rb")
    params = f.getparams()  # nchannels: 声道数, sampwidth: 量化位数, framerate: 采样频率, nframes: 采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)

    # 转成二字节数组形式（每个采样点占两个字节）
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data = np.reshape(wave_data, [nframes, nchannels])  # 转化为向量形式
    if nchannels == 2:
        wave_data = np.mean(wave_data, axis=1).reshape([-1, 1])
    print("采样点数目：" + str(len(wave_data)))  # 输出应为采样点数目
    f.close()

    return wave_data, params

def writeWav(store, person_identify, content, wave_data, endpoint, params, N, M,
             window_style="rectangle", has_noisy=False):
    """ 将切割好的语音信号输出
        生成多个切割好的wav文件
    """

    # 输出为 wav 格式
    i = 0
    inc = N - M                     # 相邻帧的间隔
    nchannels, sampwidth, framerate = params[:3]

    while i < len(endpoint):
        for s, e in np.array(endpoint, dtype=int).reshape([-1, 2]):
            d = []  # data, person, content, noisy
            d.append(wave_data[s*inc: e*inc].copy().reshape(-1))
            d.append(person_identify)
            d.append(content)
            d.append(has_noisy)
            store.append(d)

            i = i + 2
    return store

def plot_wave(wave_data, endpoint, N, M):
    n = np.arange(wave_data.shape[0])
    plt.figure(figsize=(12, 3))
    plt.plot(n, wave_data)
    # for s, e in np.array(endpoint).reshape([-1, 2]):
    #     plt.axvspan(s*(N-M), e*(N-M), color='gray', alpha=0.5)
    for i in endpoint:
        plt.axvline(i*(N-M), color='r')
    plt.show()

def unzip(from_file, to_dir):
    try:
        zip_file = zipfile.ZipFile(from_file)
    except zipfile.BadZipFile:
        print("error!!!")
        return
    if not os.path.exists(to_dir):
        os.makedirs(to_dir)
    for f in zip_file.namelist():
        zip_file.extract(f, to_dir)
    zip_file.close()
