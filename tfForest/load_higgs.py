# -*- coding:utf-8 -*-
# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import numpy as np
from sklearn.model_selection import train_test_split


def load_higgs():
	data_dir = "../"
	data = np.loadtxt(data_dir+'HIGGS.csv', delimiter=',')
	y = data[:, 0].astype('int')
	X = data[:, 1:].astype('float32')
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
	return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test = load_higgs()
	# np.save('imdb_x_train.npy', X_train)
	# np.save('imdb_x_test.npy', X_test)
	# np.save('imdb_y_train.npy', Y_train)
	# np.save('imdb_y_test.npy', Y_test)
	print(X_train.shape)
	print(X_test.shape)
	print(X_train.dtype)
	print(Y_train.shape)
