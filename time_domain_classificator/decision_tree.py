import pickle
import pandas as pd
import numpy as np
import sklearn.tree as st
import sklearn.metrics as sm
import sklearn.ensemble as se
from sklearn.manifold import TSNE
import utils
import matplotlib.pyplot as plt


def get_features(wave_data: pd.Series, m, step):
    """
    feature
    """
    x = wave_data.apply(lambda d: (d-np.mean(d))/(np.std(d)))
    # x = wave_data
    x, max_length = utils.padding_to_max(x)
    n = int((max_length-m)/step) + 1
    indices = np.tile(np.arange(0, m), (n, 1)) +\
        np.tile(np.arange(0, n*step, step), (m, 1)).T
    features = np.empty([x.shape[0], 3, n], dtype=float)
    for i in range(x.shape[0]):
        windows = x[i][indices]
        features[i, 0] = np.mean(np.abs(windows), axis=1)
        features[i, 1] = np.mean(np.power(windows, 2), axis=1)
        features[i, 2] = np.mean(np.abs(windows[:, 1:m-1]-windows[:, 0:m-2]), axis=1)
    return features.reshape([len(wave_data), -1])


for win_type in ["rect", "hamming", "hanning"]:
    with open("../dataset/processed/{}.pkl".format(win_type), "rb") as f:
        df = pickle.load(f)
        df = df[~df.has_noisy]
        persons_id = set(df.person_id)
        predicts = []
        labels = []
        for test in persons_id:
            contents = set(df.content)
            train = [i for i in persons_id if i != test]
            train_data = df.apply(lambda d: d.person_id != test, axis=1)
            test_data = df.apply(lambda d: d.person_id == test, axis=1)
            features = get_features(df.wave_data, 200, 200)
            model = se.ExtraTreesClassifier(max_depth=6)
            model.fit(features[train_data], df[train_data].content)
            print("train_predict", sm.accuracy_score(df[train_data].content, model.predict(features[train_data])))
            predicts.append(model.predict(features[test_data]))
            labels.append(df[test_data].content)
        labels = np.array(labels)
        predicts = np.array(predicts)
        print(sm.accuracy_score(labels.reshape(-1), predicts.reshape(-1)))
        print(sm.f1_score(labels.reshape(-1), predicts.reshape(-1), average=None))
        utils.plot_classify_result(range(10), labels.reshape(-1), predicts.reshape(-1), "result{}.png".format(win_type))
        plt.close()
        # tsne = TSNE()
        # x = tsne.fit_transform(features, df.content)
        # plt.scatter(x[:, 0], x[:, 1], c=df.content)
        # plt.show()
