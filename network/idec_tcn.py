from __future__ import print_function, division
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from torch.nn.parameter import Parameter

import os
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import time, datetime
import matplotlib.pyplot as plt
import pandas as pd

from load_data import *
from ae_tcn import *
from utils import *


# class AE(nn.Module):
#     '''
#         TCN自编码器
#         包括：
#         1. TCNEncoder
#         2. TCNDecoder
#         3. 一个线性层
#         '''
#
#     def __init__(self, in_channels=4, hidden_channels=10, depth=5, kernel_size=2, vector_size=5, expand_size=24,
#                  final_relu=False):
#         super(AE, self).__init__()
#         encoder = TCNEncoder(in_channels, hidden_channels, depth, kernel_size, vector_size, final_relu)
#         relu = nn.LeakyReLU()
#         decoder = TCNDecoder(vector_size, expand_size, hidden_channels, hidden_channels, depth, kernel_size, final_relu)
#         trans_1 = Transpose(1, 2)
#         linear = nn.Linear(hidden_channels, in_channels)
#         trans_2 = Transpose(1, 2)
#         self.ae_tcn = nn.Sequential(
#             encoder, relu, decoder, trans_1, linear, trans_2
#         )
#         self.enc = nn.Sequential(
#             encoder
#         )
#
#     def forward(self, x):
#         return self.ae_tcn(x), self.enc(x)


class IDEC(nn.Module):

    def __init__(self,
                 n_clusters,
                 alpha=1.0,
                 hyper='ae_tcn_hyperparameters',
                 model_name='train_data_AE_TCN',
                 model_path='../model'):
        super(IDEC, self).__init__()
        self.alpha = alpha
        self.n_clusters = n_clusters
        # 自编码器
        self.ae, self.n_z = self.load_ae(hyper=hyper, model_name=model_name, model_path=model_path)
        # cluster layer
        '''
        n_cluster个聚类中心（初始化方式？？？）
        '''
        self.cluster_layer = Parameter(torch.Tensor(self.n_clusters, self.n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    # 获得预训练过的自编码器
    def load_ae(self, hyper, model_name, model_path):
        hyper_para = None
        with open(os.path.join(model_path, '{}.json'.format(hyper)),
                  'r', encoding='utf8') as f:
            hyper_para = json.load(f)

        model = AutoEncoderTCN(
            in_channels=hyper_para['in_channels'],
            hidden_channels=hyper_para['hidden_channels'],
            depth=hyper_para['depth'],
            vector_size=hyper_para['vector_size'],
            expand_size=hyper_para['expand_size'],
            kernel_size=hyper_para['kernel_size'],
            final_relu=False  # params['final_relu']
        )
        model.load_state_dict(torch.load(
            os.path.join(model_path, '{}.pth'.format(model_name))
        ))

        return model.double(), hyper_para['vector_size']

    def forward(self, x):
        # x_bar: 重构结果，z: 编码特征
        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # 得到重构结果以及t分布（q）
        return x_bar, q


# 目标分布p
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )
    parser.add_argument('--dataset', type=str, metavar='D', default='train_data',
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
                        default='idec_tcn_hyperparameters.json',
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='the ratio of the train set')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    cuda = args.cuda
    if cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        cuda = False

    params = None
    with open(args.hyper, 'r', encoding='utf8') as f:
        params = json.load(f)
    gpu = args.gpu
    dataset = args.dataset
    print('Processing', dataset)
    train_X, train_y, *_ = load_labeled_dataset(args.data_path, dataset, args.train_ratio)
    print('Shape of train samples', train_X.shape, train_y.shape)

    model = IDEC(n_clusters=params['n_clusters'])
    train_data = torch.from_numpy(train_X)
    train_label = torch.from_numpy(train_y)
    if cuda:
        train_data = train_data.cuda(gpu)
        train_label = train_label.cuda(gpu)
        model = model.cuda(gpu)

    train_dataset = LabelledDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    x_bar, hidden = model.ae(train_data)
    if cuda:
        hidden = hidden.data.cpu().numpy()

    clf = KMeans(n_clusters=params['n_clusters'], n_init=20)
    y_pred = clf.fit_predict(hidden)
    # nmi_k = nmi_score(y_pred, y)
    # print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    if cuda:
        model.cluster_layer.data = torch.tensor(clf.cluster_centers_).cuda(gpu)
    else:
        model.cluster_layer.data = torch.tensor(clf.cluster_centers_)

    criterion1 = nn.MSELoss()
    criterion2 = nn.KLDivLoss()

    rec_loss, clu_loss, ttl_loss = [], [], []

    p = None
    for epoch in range(params['epochs']):

        if epoch % params['update_interval'] == 0:
            model.eval()
            _, tmp_q = model(train_data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            if cuda:
                tmp_q = tmp_q.cpu()
            y_pred = tmp_q.numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            print('Epoch: {}, delta_data: {}'.format(epoch, delta_label))

            # acc = cluster_acc(y, y_pred)
            # nmi = nmi_score(y, y_pred)
            # ari = ari_score(y, y_pred)
            # print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
            #       ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < params['tol']:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      params['tol'])
                print('Reached tolerance threshold. Stopping training.')

                # 保存编码后的数据、标签、聚类中心、模型
                # 保存模型
                torch.save(model.state_dict(), os.path.join(args.save_path, dataset + '_IDEC_TCN.pth'))
                # 加载患者出ICU时的数据
                X, label = load_encoded_data(args.data_path, dataset)
                # 取出模型中encoder部分
                # layers = list(model.ae.children())
                # encoder = nn.Sequential(*layers[:1])
                encoder = model.ae
                # 使用encoder将患者出ICU时的数据进行编码，并将编码结果保存。
                X = torch.from_numpy(X).cuda(gpu) if cuda else torch.from_numpy(X)
                # vector = encoder(X).squeeze(1).detach()
                vector = encoder(X)[1].detach()
                print(vector.shape)
                v = vector.cpu().numpy() if cuda else vector.numpy()
                d = np.concatenate([v, label], axis=1)

                # 获取预测的类别
                _, final_q = model(X)

                # update target distribution p
                final_q = final_q.data
                # final_p = target_distribution(final_q)

                # evaluate clustering performance
                if cuda:
                    final_q = final_q.cpu()
                y_pred = final_q.numpy().argmax(1).reshape(-1, 1)
                d = np.concatenate([d, y_pred], axis=1)

                encoded_df = pd.DataFrame(d, columns=['value{}'.format(i) for i in range(vector.shape[1])] + [
                    'death', 'label_pred'])
                encoded_df.to_csv('../all_sepsis_patient_data/encoded_{}_idec_tcn.csv'.format(dataset), index=False)
                break

        model.eval()
        with torch.no_grad():
            x_bar, q = model(train_data)
            rec_l = criterion1(train_data, x_bar).item()
            clu_l = criterion2(q.log(), p).item()
            ttl_l = params['gamma'] * clu_l + rec_l
            rec_loss.append(rec_l)
            clu_loss.append(clu_l)
            ttl_loss.append(ttl_l)

        model.train()
        for batch_idx, (batch_X, _, idx) in enumerate(train_loader):
            if cuda:
                batch_X = batch_X.cuda(gpu)
                idx = idx.cuda(gpu)

            optimizer.zero_grad()
            x_bar, q = model(batch_X)

            reconstr_loss = criterion1(x_bar, batch_X)
            kl_loss = criterion2(q.log(), p[idx])
            loss = params['gamma'] * kl_loss + reconstr_loss

            loss.backward()
            print(epoch + 1, 'loss: ', loss)
            optimizer.step()

    epoch = len(rec_loss)
    plt.plot(range(epoch), rec_loss, 'b', label='reconstruct loss')
    plt.plot(range(epoch), clu_loss, 'r', label='cluster loss')
    plt.plot(range(epoch), ttl_loss, 'g', label='total loss')
    plt.xlabel("#Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('test.png')

