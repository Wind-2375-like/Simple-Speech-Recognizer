import AEN.model as model
import torch.nn as nn
import torch.optim as optim
import sklearn.utils as su
import pickle
import pandas
import utils

epoch = 10
data_dim = 1


# [TODO: 完成reader 函数]
def reader():

    def _reader():
        pass
    return _reader


data = utils.load_pkl("../dataset/processed/rect.pkl")
wave_data = data.wave_data[data.has_noisy is False]
label = data.content
wave_data, max_length = utils.padding_to_max(wave_data)
x, y = su.shuffle(wave_data, label)

train_reader = reader()


stack_coder = model.StackCoder(data_dim)
learning_rate = 0.1
criterion = nn.MSELoss()

optimizer = optim.Adam(params=stack_coder.parameters(), lr=0.01)
for i in range(epoch):
    for data in train_reader():
        optimizer.zero_grad()
        _, decode = stack_coder(data, flag=0)
        loss = criterion(decode, data)
        loss.backward()
        optimizer.step()
    for _ in range(2):
        for data in train_reader():
            optimizer.zero_grad()
            tmp_input, decode = stack_coder(data, flag=1)
            loss = criterion(decode, tmp_input)
            loss.backward()
            optimizer.step()

    if i % 20 == 0:
        print("generation {}: {}".format(i, loss))
