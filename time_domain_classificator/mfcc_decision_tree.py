from librosa.feature import mfcc
import pandas as pd
import numpy as np
import sklearn.tree as st
import sklearn.decomposition as sd
import sklearn.metrics as sm
import sklearn.ensemble as se
import utils
import matplotlib.pyplot as plt
import pickle


def get_features(wave_data: pd.Series, n_mfcc):
    """
    feature
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
            features = get_features(df.wave_data, 20)
            model = se.ExtraTreesClassifier(max_depth=2)
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