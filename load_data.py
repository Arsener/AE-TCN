import os
import pandas as pd
import numpy as np

def load_dataset(path, dataset):
    file_path = '{}/{}.csv'.format(path, dataset)
    if not os.path.exists(file_path):
        print(file_path + 'not exists')
        return

    # 读取某特征数据的标准化后的数据
    df = pd.read_csv(file_path)
    var = ['{}_{}'.format(v, t) for v in ['hr_value', 'rr_value', 'sp_value', 'mp_value'] for t in range(24)]
    X = df[var]
    X = np.array(X, dtype=np.float64).reshape((len(X), 4, 24))

    return X