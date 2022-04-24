# Simple Speech Recognizer

This is a simple speech recognizer that classifies voices of sequences of numbers from `0` to `9`.

First, run `pip install -r requirements.txt` to install all dependencies.

## Data Preprocessing

Run `python split.py` and get audio pieces with three different type of windows "rectangle", "hanning", "hamming".

Save the data to `dataset/processed` and give them label. We use `.pkl` format, which is a dataframe form from `pandas`.

## Training

To run with time-domain feature SVM classification, just:

```shell
python td_svm_classifier/train.py
```

and get confounding matrices of three different windows-splitting techniques.

To run with mel frequency cepstral coefficients (MFCCs) feature SVM classification, just:

```shell
python mfcc_svm_classifier/train.py
```

To run LSTM model with time-domain features, just run:

```shell
python LSTM/train.py
```
