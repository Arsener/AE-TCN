import os
import json
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import argparse
import time, datetime
import random
import matplotlib.pyplot as plt

from ae_tcn import *
from load_data import load_dataset
from utils import Dataset

# plt.rcParams['figure.dpi'] = 300


def create_model(file):
    # 从文件中读取超参数
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['cuda'] = args.cuda
    params['gpu'] = args.gpu

    # create model
    print('!!New model created')
    model = AutoEncoderTCN(
        in_channels=params['in_channels'],
        hidden_channels=params['hidden_channels'],
        depth=params['depth'],
        kernel_size=params['kernel_size'],
        vector_size=params['vector_size'],
        expand_size=params['expand_size'],
        final_relu=False  # params['final_relu']
    )
    # 将参数保存
    with open(
            os.path.join(
                args.save_path, 'hyperparameters.json'
            ), 'w'
    ) as fp:
        json.dump(params, fp)

    return model.double(), params


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', default='data',
                        help='dataset names')
    parser.add_argument('--data_path', type=str, metavar='PATH', default='data',
                        help='path where dataset is saved')
    parser.add_argument('--save_path', type=str, metavar='PATH', default='./model',
                        help='path where the estimator is/should be saved')
    parser.add_argument('--load_path', type=str, metavar='PATH', required=False, default=None,
                        help='path where the encoder is saved')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE',
                        default='hyperparameters.json',
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='the ratio of the train set')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    # 加载数据
    dataset = args.dataset
    gpu = args.gpu
    print('Processing', dataset)
    all_data = load_dataset(args.data_path, dataset)
    np.random.shuffle(all_data)

    # 将数据划分为训练集和测试集
    ratio = args.train_ratio
    nums, *_ = all_data.shape
    train_X, test_X = all_data[:int(nums * ratio)], all_data[int(nums * ratio):]
    print('Shape of train samples', train_X.shape)

    # 创建模型，获取参数
    model, params = create_model(args.hyper)

    # 训练模型
    train_data = torch.from_numpy(train_X)
    test_data = torch.from_numpy(test_X)
    if cuda:
        train_data = train_data.cuda(gpu)
        test_data = test_data.cuda(gpu)
        model = model.cuda(gpu)

    train_dataset = Dataset(train_X)
    train_generator = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    epoch = 0
    train_loss = []
    test_loss = []
    while epoch < params['epochs']:
        print('Epoch: ', epoch + 1)
        for batch in train_generator:
            if cuda:
                batch = batch.cuda(gpu)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            print(epoch, 'loss: ', loss)
            optimizer.step()

        epoch += 1
        with torch.no_grad():
            train_loss.append(criterion(train_data, model(train_data)).item())
            test_loss.append(criterion(test_data, model(test_data)).item())

    # 展示loss
    plt.plot(range(epoch), train_loss, 'b', label='train loss')
    plt.plot(range(epoch), test_loss, 'r', label='test loss')
    plt.xlabel("#Epochs")
    plt.ylabel("Mean Square Loss")
    plt.legend()
    plt.savefig('test.png')
    # plt.show()
    # 保存模型
    torch.save(model.state_dict(), os.path.join(args.save_path, dataset + '_AE_TCN.pth'))
    # model.save(os.path.join(args.save_path, dataset))
