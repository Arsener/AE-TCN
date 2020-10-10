import os
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import time, datetime
import matplotlib.pyplot as plt
import pandas as pd

from ae_lstm import *
from load_data import load_dataset, load_encoded_data
from utils import Dataset


# plt.rcParams['figure.dpi'] = 300


def create_model(file, model_type, cuda):
    # 从文件中读取超参数
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    params['cuda'] = cuda
    params['gpu'] = args.gpu

    # create model
    print('!!New model created')
    if model_type == 'otoi':
        model = AutoEncoderLSTM_otoi(
            hidden_size=params['hidden_size'],
            nb_feature=params['nb_feature'],
            if_cuda=params['cuda'],
            gpu=params['gpu']
        )
    elif model_type == 'expo':
        model = AutoEncoderLSTM_expo(
            hidden_size=params['hidden_size'],
            nb_feature=params['nb_feature']
        )

    # 将参数保存
    with open(
            os.path.join(
                args.save_path, 'lstm_{}_hyperparameters.json'.format(model_type)
            ), 'w'
    ) as fp:
        json.dump(params, fp)

    return model.double(), params


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--model', type=str, metavar='D', default='otoi',
                        help='model names')
    parser.add_argument('--dataset', type=str, metavar='D', default='data',
                        help='dataset names')
    parser.add_argument('--data_path', type=str, metavar='PATH', default='../all_sepsis_patient_data',
                        help='path where dataset is saved')
    parser.add_argument('--save_path', type=str, metavar='PATH', default='../model',
                        help='path where the estimator is/should be saved')
    parser.add_argument('--load_path', type=str, metavar='PATH', required=False, default=None,
                        help='path where the encoder is saved')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE',
                        default='otoi',
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='the ratio of the train set')

    return parser.parse_args()


gpu = 0


# def load_encoder(dataset, model_path, cuda, gpu=0):
#     hyper_para = None
#     with open(os.path.join(model_path, 'lstm_otoi_hyperparameters.json'),
#               'r', encoding='utf8') as f:
#         hyper_para = json.load(f)
#
#     model = AutoEncoderLSTM_otoi(
#         hidden_size=hyper_para['hidden_size'],
#         nb_feature=hyper_para['nb_feature']
#     )
#     model.load_state_dict(torch.load(
#         os.path.join(model_path, '{}_AE_LSTM_otoi.pth'.format(dataset))
#     ))
#
#     layers = list(model.children())
#     encoder = nn.Sequential(layers[0])
#     if cuda:
#         encoder = encoder.cuda(gpu)
#     return encoder


if __name__ == '__main__':
    args = parse_arguments()
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    # 加载数据
    dataset = args.dataset
    gpu = args.gpu
    model_type = args.model
    print('Processing', dataset)
    train_X, test_X = load_dataset(args.data_path, dataset, args.train_ratio)
    train_X = train_X.swapaxes(1, 2)
    test_X = test_X.swapaxes(1, 2)
    print('Shape of train samples', train_X.shape)

    # 创建模型，获取参数
    hyper = 'lstm_{}_hyperparameters.json'.format(args.hyper)
    model, params = create_model(hyper, model_type, cuda)

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

    # 记录开始时间
    t1 = time.time()
    d1 = datetime.datetime.now()
    epoch = 0
    train_loss = []
    test_loss = []
    with open('train_log.txt', 'w', encoding='utf8') as f:
        f.write('Start a new training process!!!\n')

    while epoch < params['epochs']:
        print('Epoch: ', epoch + 1)
        for batch in train_generator:
            if cuda:
                batch = batch.cuda(gpu)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            print(epoch + 1, 'loss: ', loss)
            optimizer.step()

        epoch += 1
        with torch.no_grad():
            train_loss.append(criterion(train_data, model(train_data)).item())
            test_loss.append(criterion(test_data, model(test_data)).item())
        with open('train_log.txt', 'a', encoding='utf8') as f:
            f.write("Total epochs: {}, current epoch: {}, train loss:{}, test_loss{}\n".format(params['epochs'], epoch, train_loss[-1], test_loss[-1]))

    t2 = time.time()
    with open('time cost.txt', 'w', encoding='utf8') as f:
        f.write('start time: {}\nend time: {}\ninterval: {}min'.format(d1, datetime.datetime.now(), (t2 - t1) / 60))

    # 展示loss
    plt.plot(range(epoch), train_loss, 'b', label='train loss')
    plt.plot(range(epoch), test_loss, 'r', label='test loss')
    plt.xlabel("#Epochs")
    plt.ylabel("Mean Square Loss")
    plt.legend()
    plt.savefig('test.png')
    # plt.show()
    # 保存模型
    torch.save(model.state_dict(), os.path.join(args.save_path, dataset + '_AE_LSTM_{}.pth'.format(model_type)))
    # model.save(os.path.join(args.save_path, dataset))

    # 加载患者出ICU时的数据
    X, label = load_encoded_data(args.data_path, dataset)
    X = X.swapaxes(1, 2)
    # 取出模型中encoder部分
    layers = list(model.children())
    encoder = nn.Sequential(layers[0])
    # 使用encoder将患者出ICU时的数据进行编码，并将编码结果保存。
    X = torch.from_numpy(X).cuda(gpu) if cuda else torch.from_numpy(X)
    vector = encoder(X).squeeze(0).detach()
    print(vector.shape)
    v = vector.cpu().numpy() if cuda else vector.numpy()
    d = np.concatenate([v, label], axis=1)
    encoded_df = pd.DataFrame(d, columns=['value{}'.format(i) for i in range(10)] + ['death'])
    encoded_df.to_csv('../all_sepsis_patient_data/encoded_{}_lstm_{}.csv'.format(dataset, model_type), index=False)



