import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, hidden_size, nb_feature, num_layers=1, dropout=0):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)

    # def initHidden(self, batch_size):
    #     self.hidden_cell = (
    #         torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float),
    #         torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float)
    #     )

    def forward(self, input_seq):
        # self.initHidden(input_seq.shape[0])
        _, hidden_cell = self.lstm(input_seq)
        return hidden_cell[0]


class Decoder(nn.Module):
    def __init__(self, hidden_size, nb_feature, num_layers=1, dropout=0):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)

    def forward(self, input_seq, hidden_cell=None):
        # print('----\n', input_seq, hidden_cell)
        if hidden_cell is None:
            output, hidden_cell = self.lstm(input_seq)
        else:
            output, hidden_cell = self.lstm(input_seq, hidden_cell)
        return output, hidden_cell


class AutoEncoderLSTM_otoi(nn.Module):
    def __init__(self, hidden_size, nb_feature, num_layers=1, dropout=0, if_cuda=False, gpu=0):
        super(AutoEncoderLSTM_otoi, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(hidden_size, nb_feature, num_layers, dropout)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)
        self.if_cuda = if_cuda
        self.gpu = gpu

    def forward(self, input_seq):
        output = torch.zeros(size=(input_seq.shape[0], input_seq.shape[1], self.hidden_size), dtype=torch.double)
        if self.if_cuda:
            output = output.cuda(self.gpu)
        # print('asadsf', output.shape)
        hidden_cell = None
        input_decoder = self.encoder(input_seq)[-1:, :, :].transpose(0, 1)
        for i in range(input_seq.shape[1] - 1, -1, -1):
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)
            input_decoder = output_decoder
            output[:, i, :] = output_decoder[:, 0, :]
        return self.linear(output), input_decoder.squeeze(1)


class AutoEncoderLSTM_expo(nn.Module):
    def __init__(self, hidden_size, nb_feature, num_layers=1, dropout=0):
        super(AutoEncoderLSTM_expo, self).__init__()
        self.encoder = Encoder(hidden_size, nb_feature, num_layers, dropout)
        self.decoder = Decoder(hidden_size, hidden_size, num_layers, dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)

    def forward(self, input_seq):
        input_decoder = self.encoder(input_seq)[-1:, :, :].transpose(0, 1)
        # print(input_decoder.shape)
        input_decoder = input_decoder.expand(-1, input_seq.shape[1], -1)
        # print(input_decoder.shape)
        output, _ = self.decoder(input_decoder)
        return self.linear(output)
