import torch
import torch.nn as nn


class AutoCoder(nn.Module):
    def __init__(self, input_dim=4, encoder_size=2):
        super(AutoCoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoder_size)
        self.decoder = nn.Linear(encoder_size, input_dim)

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class StackCoder(nn.Module):
    def __init__(self, data_dim, input_dim=1024, output_dim=128):
        super(StackCoder, self).__init__()
        self.pre_fc = nn.Linear(data_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, 4 * output_dim)
        self.auto_coder = AutoCoder(input_dim=4 * output_dim, encoder_size=output_dim)
        self.fc2 = nn.Linear(4 * output_dim, input_dim)

    def forward(self, x, flag=-1):
        x = self.pre_fc(x)
        if flag == 0:
            self.requires_grad_ = True
            self.auto_coder.requires_grad_ = False
            encode = self.fc1(x)
            decode = self.fc2(encode)
            return encode, decode
        elif flag == -1:
            self.requires_grad_ = False
            tmp = self.fc1(x)
            _, tmp = self.auto_coder(tmp)
            tmp = self.fc2(tmp)
            return tmp
        else:
            self.requires_grad_ = True
            self.fc1.requires_grad_ = False
            self.fc2.requires_grad_ = False
            tmp_input = self.fc1(x)
            noise = torch.randn_like(tmp_input) * 0.01
            tmp = tmp_input + noise
            encode, decode = self.auto_coder(tmp)
            return tmp_input, decode
