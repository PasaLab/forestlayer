# -*- coding:utf-8 -*-
"""
Tianchi AI contest. Intelligent Manufacturing.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Input
from keras.models import Model
from forestlayer.layers.layer import AutoGrowingCascadeLayer
from forestlayer.estimators.estimator_configs import ExtraRandomForestConfig, RandomForestConfig, XGBRegressorConfig, GBDTConfig
from forestlayer.backend.backend import set_base_dir
from forestlayer.datasets.dataset import get_dataset_dir
from forestlayer.utils.storage_utils import get_data_save_base, get_model_save_base
from forestlayer.utils.metrics import mse
import os.path as osp
import pickle

set_base_dir(osp.expanduser(osp.join('~', 'forestlayer')))

train = pd.read_excel(osp.join(get_dataset_dir(), 'tianchi', 'intelmanu', 'train.xlsx'))
testA = pd.read_excel(osp.join(get_dataset_dir(), 'tianchi', 'intelmanu', 'test_A.xlsx'))

train_test = pd.concat([train, testA], axis=0, ignore_index=True)


def func(x):
    try:
        return float(x)
    except:
        if x is None:
            return 0
        else:
            return x


label = train_test['Y'][:500]
train_test = train_test.fillna(0)
train_test.applymap(func)
feat_columns = list(train_test.columns.values)
feat_columns.remove("ID")
feat_columns.remove("Y")
# data 为除去ID和label后所有的特征
data = train_test[feat_columns]
label_id = train_test['ID']

cate_columns = data.select_dtypes(include=["object"]).columns
num_columns = data.select_dtypes(exclude=["object"]).columns
print("categorical feat num: ", len(cate_columns), "number feat num: ", len(num_columns))

feat_cate = data[cate_columns]
feat_nume = data[num_columns]

# categorical features: One-hot
feat_categorical_dummies = pd.get_dummies(feat_cate)
# number features: MinMax
feat_number_scale = pd.DataFrame(MinMaxScaler().fit_transform(feat_nume))
# Concatenate
feat_all = pd.concat([feat_number_scale, feat_categorical_dummies], axis=1)


def auto_encoder():
    if not osp.exists(osp.join(get_model_save_base(), 'feat_dim_120.pkl')):
        print("Cannot Find {}".format(osp.join(get_model_save_base(), 'feat_dim_120.pkl')))
        x_train, x_test, y_train, y_test = train_test_split(feat_all, feat_all, test_size=0.2, random_state=42)

        encoding_dim = 120
        input_ = Input(shape=(8051,))
        encoded = Dense(encoding_dim, activation='relu')(input_)
        decoded = Dense(8051, activation='sigmoid')(encoded)
        autoencoder = Model(input=input_, output=decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(x_train,
                        x_train,
                        nb_epoch=500,
                        batch_size=10,
                        shuffle=True,
                        validation_data=(x_test, x_test))

        # 根据上面我们训练的自编码器，截取其中编码部分定义为编码模型
        encoder = Model(input=input_, output=encoded)
        # 对特征进行编码降维
        feat_dim_120 = encoder.predict(feat_all)

        with open(osp.join(get_model_save_base(), 'feat_dim_120.pkl'), 'wb') as f:
            pickle.dump(feat_dim_120, f)
    else:
        print("Find {}".format(osp.join(get_model_save_base(), 'feat_dim_120.pkl')))
        with open(osp.join(get_model_save_base(), 'feat_dim_120.pkl'), 'rb') as f:
            feat_dim_120 = pickle.load(f)
    return feat_dim_120


def PCA():
    from sklearn.decomposition import PCA
    pca = PCA(n_components=120)
    x_train, x_test, y_train, y_test = train_test_split(feat_all, feat_all, test_size=0.2, random_state=42)
    pca.fit(x_train, y_train)
    feat_dim_120 = pca.transform(feat_all)
    print('PCA finished! feat_dim_120.shape = {}'.format(feat_dim_120.shape))
    return feat_dim_120


feat_dim_120 = auto_encoder()
# feat_dim_120 = PCA()


est_configs = [
    ExtraRandomForestConfig(),
    ExtraRandomForestConfig(),
    RandomForestConfig(n_estimators=100),
    RandomForestConfig(n_estimators=100),
    GBDTConfig(n_estimators=100),
    GBDTConfig(n_estimators=100),
    # XGBRegressorConfig(),
    # XGBRegressorConfig()
]

data_save_dir = osp.join(get_data_save_base(), 'tianchi', 'intelmanu')

agc = AutoGrowingCascadeLayer(task='regression',
                              est_configs=est_configs,
                              max_layers=1,
                              data_save_dir=data_save_dir,
                              keep_test_result=True)

agc.fit_transform(feat_dim_120[:500], label, feat_dim_120[500:])
result = agc.test_results

true_A = pd.read_csv(osp.join(get_dataset_dir(), 'tianchi', 'intelmanu', 'true_A_20180114.csv'), header=None)

true = true_A.iloc[:, 1]

print("MSE Score: {}".format(mse(result, true)))

# ret = pd.DataFrame()
# ret["ID"] = testA["ID"]
# ret["Y"] = result
# ret.to_csv(osp.join(data_save_dir, "result0112_PCA.csv"), index=False, header=False)



