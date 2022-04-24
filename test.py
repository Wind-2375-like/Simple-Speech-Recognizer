import scipy.signal as signal
import matplotlib.pyplot as plt
import VAD

# 读取音频信号
# unzip("./dataset/raw/ren1.zip", "./dataset/unzip/")
for i in range(10):
    plt.figure(i)
    wave_data, params = VAD.readWav("./dataset/unzip/ren1/" + str(i) + ".wav")

    # 语音信号分帧加窗
    N = 256         # 一帧时间 = N / framerate, 得 N 的范围: 160-480, 取最近2的整数次方 256
    M = 128         # M 的范围应在 N 的 0-1/2
    winfunc = signal.windows.hamming(N)     # 汉明窗
    # winfunc = signal.windows.hanning(N)     # 海宁窗
    # winfunc = 1                             # 矩形窗
    frames, num_frame = VAD.addWindow(wave_data, N, M, winfunc)

    # 时域特征值计算
    energy = VAD.calEnergy(frames, N).reshape(1, num_frame)
    amplitude = VAD.calAmplitude(frames, N).reshape(1, num_frame)
    zerocrossingrate = VAD.calZeroCrossingRate(frames, N).reshape(1, num_frame)

    # 端点检测
    endpoint = VAD.detectEndPoint(
        wave_data, energy[0], zerocrossingrate[0])    # 利用短时能量
    # endpoint = detectEndPoint(wave_data, amplitude[0], zerocrossingrate[0]) # 利用短时幅度

    sorted_endpoint = sorted(set(endpoint))

    VAD.plot_wave(wave_data, endpoint, N, M)

    # 输出为 wav 格式
    # store = []
    # writeWav(store, "1", "0", wave_data, sorted_endpoint, params, N, M)
    # df = pd.DataFrame(store, columns=['wave_data', 'person_id', 'content'])
    # with open("./dataset/processed/test.pkl", "wb") as f:
    #     pickle.dump(df, f)
    print("done")

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