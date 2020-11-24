import os
import json
import torch
import torch.nn as nn
import numpy as np
import argparse
import time, datetime
import matplotlib.pyplot as plt
import pandas as pd

from ae_tcn import *
from load_data import *
from utils import *
from pytorchtools import EarlyStopping


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
    model = TCN(
        in_channels=params['in_channels'],
        hidden_channels=params['hidden_channels'],
        depth=params['depth'],
        kernel_size=params['kernel_size'],
        vector_size=params['vector_size'],
        expand_size=params['expand_size'],
        output_size=params['output_size'],
        final_relu=False  # params['final_relu']
    )
    # 将参数保存
    with open(
            os.path.join(
                args.save_path, 'tcn_hyperparameters.json'
            ), 'w'
    ) as fp:
        json.dump(params, fp)

    return model.double(), params


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
                        default='tcn_hyperparameters.json',
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='the ratio of the train set')

    return parser.parse_args()


gpu = 0


# def load_encoder(dataset, model_path, cuda, gpu=0):
#     hyper_para = None
#     with open(os.path.join(model_path, 'hyperparameters.json'),
#               'r', encoding='utf8') as f:
#         hyper_para = json.load(f)

#     model = AutoEncoderTCN(
#         in_channels=hyper_para['in_channels'],
#         hidden_channels=hyper_para['hidden_channels'],
#         depth=hyper_para['depth'],
#         vector_size=hyper_para['vector_size'],
#         expand_size=hyper_para['expand_size'],
#         kernel_size=hyper_para['kernel_size'],
#         final_relu=False  # params['final_relu']
#     )
#     model.load_state_dict(torch.load(
#         os.path.join(model_path, '{}_TCN_CLF.pth'.format(dataset))
#     ))

#     layers = list(model.ae_tcn.children())
#     encoder = nn.Sequential(*layers[:1])
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
    print('Processing', dataset)
    train_X, train_y, test_X, test_y = load_labeled_dataset(args.data_path, dataset, args.train_ratio)
    print('Shape of train samples', train_X.shape, train_y.shape)

    # 创建模型，获取参数
    model, params = create_model(args.hyper)

    # 训练模型
    train_data = torch.from_numpy(train_X)
    train_label = torch.from_numpy(train_y)
    test_data = torch.from_numpy(test_X)
    test_label = torch.from_numpy(test_y)
    if cuda:
        train_data = train_data.cuda(gpu)
        train_label = train_label.cuda(gpu)
        test_data = test_data.cuda(gpu)
        test_label = test_label.cuda(gpu)
        model = model.cuda(gpu)

#     train_dataset = LabelledDataset(train_X)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_generator = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True
    )
    criterion1 = nn.MSELoss()
    criterion2 = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    early_stopping = EarlyStopping(params['early_stopping'], path=os.path.join(args.save_path, '{}_TCN.pth'.format(dataset)))

    # 记录开始时间
    t1 = time.time()
    d1 = datetime.datetime.now()
    epoch = 0
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    with open('train_log.txt', 'w', encoding='utf8') as f:
        f.write('Start a new training process!!!\n')

    def acc(label, prob):
        from sklearn.metrics import accuracy_score
        pred = torch.zeros(prob.shape)
        pred[prob >= .5] = 1
        return accuracy_score(label, pred)
    
    while epoch < params['epochs']:
        model.train()
        print('Epoch: ', epoch + 1)
        for batch_X, batch_y in train_generator:
            if cuda:
                batch_X = batch_X.cuda(gpu)
                batch_y = batch_y.cuda(gpu)
            optimizer.zero_grad()
            output, label_pre = model(batch_X)
            loss1 = criterion1(output, batch_X)
            loss2 = criterion2(label_pre, batch_y)
            loss = params['alpha'] * loss1 + params['beta'] * loss2
            loss.backward()
            print(epoch + 1, 'loss: ', loss)
            optimizer.step()

        epoch += 1
        model.eval()
        with torch.no_grad():
            train_pred = model(train_data)
            test_pred = model(test_data)
            tr_l = params['alpha'] * criterion1(train_pred[0], train_data).item() + params['beta'] * criterion2(train_pred[1], train_label).item()
            te_l = params['alpha'] * criterion1(test_pred[0], test_data).item() + params['beta'] * criterion2(test_pred[1], test_label).item()
            train_loss.append(tr_l)
            test_loss.append(te_l)
            early_stopping(te_l, model)
            
            tr_acc = acc(train_label.cpu(), train_pred[1].cpu())
            te_acc = acc(test_label.cpu(), test_pred[1].cpu())
            train_acc.append(tr_acc)
            test_acc.append(te_acc)
            print('train_acc: {:.6f}, test_acc: {:.6f}'.format(tr_acc, te_acc))
            # 若满足 early stopping 要求
            if early_stopping.early_stop:
                print("Early stopping")
                # 结束模型训练
                break
                
        with open('train_log.txt', 'a', encoding='utf8') as f:
            f.write("Total epochs: {}, current epoch: {}, train loss:{}, test_loss{}\n".format(params['epochs'], epoch, train_loss[-1], test_loss[-1]))

    t2 = time.time()
    with open('time cost.txt', 'w', encoding='utf8') as f:
        f.write('start time: {}\nend time: {}\ninterval: {}min'.format(d1, datetime.datetime.now(), (t2 - t1) / 60))

    # 展示loss
    plt.plot(range(epoch), train_loss, 'b', label='train loss')
    plt.plot(range(epoch), test_loss, 'r', label='test loss')
    plt.xlabel("#Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('test.png')
    plt.clf()
    plt.plot(range(epoch), train_acc, 'b', label='train acc')
    plt.plot(range(epoch), test_acc, 'r', label='test acc')
    plt.xlabel("#Epochs")
    plt.ylabel("Acc")
    plt.legend()
    plt.savefig('acc.png')
    # plt.show()
    # 保存模型
#     torch.save(model.state_dict(), os.path.join(args.save_path, dataset + '_AE_TCN.pth'))
    # model.save(os.path.join(args.save_path, dataset))

    # 加载患者出ICU时的数据
    X, label = load_encoded_data(args.data_path, dataset)
    # 取出模型中encoder部分
    layers = list(model.ae_tcn.children())
    encoder = nn.Sequential(*layers[:1])
    # 使用encoder将患者出ICU时的数据进行编码，并将编码结果保存。
    X = torch.from_numpy(X).cuda(gpu) if cuda else torch.from_numpy(X)
    vector = encoder(X).squeeze(1).detach()
#     vector = encoder(test_data).squeeze(1).detach()
    print(vector.shape)
    v = vector.cpu().numpy() if cuda else vector.numpy()
    d = np.concatenate([v, label.reshape(-1, 1)], axis=1)
#     d = np.concatenate([v, test_y.reshape(-1, 1)], axis=1)
    encoded_df = pd.DataFrame(d, columns=['value{}'.format(i) for i in range(params['vector_size'])] + ['death'])
    encoded_df.to_csv('../all_sepsis_patient_data/encoded_{}.csv'.format(dataset), index=False)



