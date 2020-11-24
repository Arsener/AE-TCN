import os
import pandas as pd
import numpy as np
import random

val = ['hr_value', 'rr_value', 'sp_value', 'mp_value']
# val = ['sp_value', 'mp_value']
dim = len(val)

def load_dataset(path, dataset, ratio):
    file_path = '{}/{}.csv'.format(path, dataset)
    if not os.path.exists(file_path):
        print(file_path + 'not exists')
        return

    # 读取某特征数据的标准化后的数据
    df = pd.read_csv(file_path)
    # 根据患者进行数据集的划分
    ids = list(set(df['icustay_id']))
    random.shuffle(ids)
    gap = int(ratio * len(ids))
    train_id, test_id = ids[:gap], ids[gap:]
    train_df = df[df.icustay_id.isin(train_id)]
    test_df = df[df.icustay_id.isin(test_id)]
    var = ['{}_{}'.format(v, t) for v in val for t in range(24)]
    train_X = train_df[var]
    test_X = test_df[var]
    train_X = np.array(train_X, dtype=np.float64).reshape((len(train_X), dim, 24))
    test_X = np.array(test_X, dtype=np.float64).reshape((len(test_X), dim, 24))
    np.random.shuffle(train_X)
    np.random.shuffle(test_X)

    return train_X, test_X


def load_labeled_dataset(path, dataset, ratio):
    file_path = '{}/{}.csv'.format(path, dataset)
    if not os.path.exists(file_path):
        print(file_path + 'not exists')
        return

    # 读取某特征数据的标准化后的数据
    df = pd.read_csv(file_path)
    # 根据患者进行数据集的划分
    ids = list(set(df['icustay_id']))
    random.shuffle(ids)
    gap = int(ratio * len(ids))
    train_id, test_id = ids[:gap], ids[gap:]
    train_df = df[df.icustay_id.isin(train_id)]
    train_df = train_df.sample(frac=1)
    test_df = df[df.icustay_id.isin(test_id)]
    test_df = test_df.sample(frac=1)    
    var = ['{}_{}'.format(v, t) for v in val for t in range(24)]
    train_X = train_df[var]
    train_y = train_df['death']
    test_X = test_df[var]
    test_y = test_df['death']
    train_X = np.array(train_X, dtype=np.float64).reshape((len(train_X), dim, 24))
    train_y = np.array(train_y, dtype=np.float64).reshape(-1, 1)
    test_X = np.array(test_X, dtype=np.float64).reshape((len(test_X), dim, 24))
    test_y = np.array(test_y, dtype=np.float64).reshape(-1, 1)

    return train_X, train_y, test_X, test_y


def load_encoded_data(path, dataset):
    file_path = '{}/{}.csv'.format(path, dataset)
    if not os.path.exists(file_path):
        print(file_path + 'not exists')
        return

    # 读取某特征数据的标准化后的数据
    d = pd.read_csv(file_path)
#     d = df[df.hours_to_ed == 0.0]
    label = np.array(d['death'])
    label = label.reshape(len(label), 1)
    var = ['{}_{}'.format(v, t) for v in val for t in range(24)]
    X = np.array(d[var])
    X = X.reshape((len(X), dim, 24))

    return X, label