import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


class SqueezeChannels(nn.Module):
    """
    Squeezes, in a three-dimensional tensor, the third dimension.
    """

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class Transpose(nn.Module):
    '''
    将一个tensor的两个维度进行交换
    '''

    def __init__(self, d1, d2):
        super(Transpose, self).__init__()
        self.d1 = d1
        self.d2 = d2

    def forward(self, x):
        return x.transpose(self.d1, self.d2)


class TCNBlock(nn.Module):
    '''
    一个TCN的block
    如果relu为True，则每个block块在输出之前经过一个激活函数
    '''

    def __init__(self, in_channels, out_channels, kernel_size, dilation, final_relu=False):
        super(TCNBlock, self).__init__()
        # 计算padding
        padding = (kernel_size - 1) * dilation

        # block内第一层，conv + weight norm + relu
        '''
        是否使用weight norm还是其他norm方式，待定
        目前使用WN，还可以考虑使用BN，不过LN不适合
        '''
        conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        ))
        chomp1 = Chomp1d(padding)
        relu1 = nn.LeakyReLU()

        # block内第二层，conv + weight norm + relu
        conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        ))
        chomp2 = Chomp1d(padding)
        relu2 = nn.LeakyReLU()

        # 一个TCN块
        self.tcn = nn.Sequential(
            conv1, chomp1, relu1, conv2, chomp2, relu2
        )

        # 残差连接，若in_channels不等于out_channels，需要进行上采样或降采样
        self.up_down_sampling = nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else None

        self.relu = nn.LeakyReLU() if final_relu else None

    def forward(self, x):
        out_tcn = self.tcn(x)
        res = x if self.up_down_sampling is None else self.up_down_sampling(x)
        return out_tcn + res if self.relu is None else self.relu(out_tcn + res)


class DilationTCN(nn.Module):
    '''
    堆叠的TCN层
    '''

    def __init__(self, in_channels, hidden_channels, out_channels, depth, kernel_size, final_relu):
        super(DilationTCN, self).__init__()

        layers = []
        dilation_size = 1
        for i in range(depth - 1):
            in_channels_block = in_channels if i == 0 else hidden_channels
            layers += [TCNBlock(
                in_channels_block, hidden_channels, kernel_size, dilation_size, final_relu
            )]
            dilation_size *= 2

        layers += [TCNBlock(
            hidden_channels, out_channels, kernel_size, dilation_size, final_relu
        )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNEncoder(nn.Module):
    '''
    对最后一层的TCN做处理
    目前采用的方式：先MaxPooling，之后加一个linear层
    '''

    def __init__(self, in_channels, hidden_channels, depth, kernel_size, vector_size, final_relu):
        super(TCNEncoder, self).__init__()
        dilation_tcn = DilationTCN(in_channels, hidden_channels, hidden_channels, depth, kernel_size, final_relu)
        # 先进行maxpooling
        pooling = nn.AdaptiveMaxPool1d(1)
        # 经过pooling后，L为1，交换C和L两个维度，输入线性层
        trans = Transpose(1, 2)
        # 线性层
        linear = nn.Linear(hidden_channels, vector_size)
        self.encoder = nn.Sequential(
            dilation_tcn, pooling, trans, linear
        )

    def forward(self, x):
        return self.encoder(x)


class ExpandVector(nn.Module):
    '''
    将vector进行扩展
    考虑两种方法：
    1. 线性变换
    2. 复制
    '''

    def __init__(self, input_size, output_size):
        super(ExpandVector, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


class TCNDecoder(nn.Module):
    '''
    TCN解码器，先对vector进行Transpose，之后进行扩展，再经过一个DilationTCN
    目前实现的是使用线性变换的方式
    '''

    def __init__(self, vector_size, expand_size, hidden_channels, out_channels, depth, kernel_size, final_relu):
        super(TCNDecoder, self).__init__()
        # 先Transpose
        trans = Transpose(1, 2)
        # 扩展
        linear = nn.Linear(1, expand_size)
        dilation_cnn = DilationTCN(vector_size, hidden_channels, out_channels, depth, kernel_size, final_relu)
        self.decoder = nn.Sequential(
            trans, linear, dilation_cnn
        )

    def forward(self, x):
        return self.decoder(x)


class AutoEncoderTCN(nn.Module):
    '''
    TCN自编码器
    包括：
    1. TCNEncoder
    2. TCNDecoder
    '''

    def __init__(self, in_channels=4, hidden_channels=10, depth=5, kernel_size=2, vector_size=5, expand_size=24,
                 final_relu=False):
        super(AutoEncoderTCN, self).__init__()
        encoder = TCNEncoder(in_channels, hidden_channels, depth, kernel_size, vector_size, final_relu)
        decoder = TCNDecoder(vector_size, expand_size, hidden_channels, in_channels, depth, kernel_size, final_relu)
        self.ae_tcn = nn.Sequential(encoder, decoder)

    def forward(self, x):
        return self.ae_tcn(x)
