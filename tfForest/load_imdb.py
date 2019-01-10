# -*- coding:utf-8 -*-
"""
IMDB dataset loading.
"""

# Copyright 2017 Authors NJU PASA BigData Laboratory.
# Authors: Qiu Hu <huqiu00#163.com>
# License: Apache-2.0

import os
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence

"""
X_train.len: min,mean,max=11,238,2494
X_test.len: min,mean,max=7,230,2315
"""


def load_imdb(feature_type='tfidf'):
	"""
	Load IMDB data in several formats.

	:param feature_type: feature type, default is 'tfidf', others are 'origin', 'tfidf-seq'
	:return:
	"""
	data_dir = "../DeepForestTF_Data/"
	if (os.path.exists(data_dir + "imdb_x_train.npy") and os.path.exists(data_dir + "imdb_x_test.npy")
		  and os.path.exists(data_dir + "imdb_y_train.npy") and os.path.exists(data_dir + "imdb_y_test.npy")):
		x_train = np.load(data_dir + "imdb_x_train.npy")
		x_test = np.load(data_dir + "imdb_x_test.npy")
		y_train = np.load(data_dir + "imdb_y_train.npy")
		y_test = np.load(data_dir + "imdb_y_test.npy")
		x_train = x_train.reshape((x_train.shape[0], -1))
		x_test = x_test.reshape((x_test.shape[0], -1))
		return x_train, x_test, y_train, y_test

	max_features = 0
	if feature_type.startswith('tfidf'):
		max_features = 5000
		(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
	else:
		(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=None)
	if feature_type == 'origin':
		max_len = 400
		x_train = sequence.pad_sequences(x_train, maxlen=max_len)
		x_test = sequence.pad_sequences(x_test, maxlen=max_len)
	elif feature_type == 'tfidf':
		from sklearn.feature_extraction.text import TfidfTransformer
		transformer = TfidfTransformer(smooth_idf=True)
		x_train_bin = np.zeros((len(x_train), max_features), dtype=np.int16)
		x_test_bin = np.zeros((len(x_test), max_features), dtype=np.int16)
		for i, x_i in enumerate(x_train):
			x_train_bin[i, :] = np.bincount(x_i, minlength=max_features)
		for i, x_i in enumerate(x_test):
			x_test_bin[i, :] = np.bincount(x_i, minlength=max_features)
		transformer.fit_transform(x_train_bin)
		x_train = transformer.transform(x_train_bin)
		x_test = transformer.transform(x_test_bin)
		x_train = np.asarray(x_train.todense())
		x_test = np.asarray(x_test.todense())
	elif feature_type == 'tfidf-seq':
		from sklearn.feature_extraction.text import TfidfTransformer
		transformer = TfidfTransformer(smooth_idf=True)
		transformer2 = TfidfTransformer(smooth_idf=True)
		max_len = 400
		n_train = len(x_train)
		n_test = len(x_test)
		x_train_bin = np.zeros((n_train, max_features), dtype=np.int16)
		x_test_bin = np.zeros((n_test, max_features), dtype=np.int16)
		for i, x_i in enumerate(x_train):
			x_train_bin_i = np.bincount(x_i)
			x_train_bin[i, :len(x_train_bin_i)] = x_train_bin_i
		for i, x_i in enumerate(x_test):
			x_test_bin_i = np.bincount(x_i)
			x_test_bin[i, :len(x_test_bin_i)] = x_test_bin_i
		x_train_tfidf = transformer.fit_transform(x_train_bin)
		x_test_tfidf = transformer2.fit_transform(x_test_bin)
		x_train_tfidf = np.asarray(x_train_tfidf.todense())
		x_test_tfidf = np.asarray(x_test_tfidf.todense())
		x_train_id = sequence.pad_sequences(x_train, maxlen=max_len)
		x_test_id = sequence.pad_sequences(x_test, maxlen=max_len)
		x_train = np.zeros(x_train_id.shape, dtype=np.float32)
		x_test = np.zeros(x_test_id.shape, dtype=np.float32)
		for i in range(n_train):
			x_train[i, :] = x_train_tfidf[i][x_train_id[i]]
		for i in range(n_test):
			x_test[i, :] = x_test_tfidf[i][x_test_id[i]]
	else:
		raise ValueError('Unknown feature type: {}'.format(feature_type))

	x_train = x_train[:, np.newaxis, :, np.newaxis].astype('float32')
	x_test = x_test[:, np.newaxis, :, np.newaxis].astype('float32')
	return x_train, x_test, y_train.astype('int8'), y_test.astype('int8')


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test = load_imdb()
	# np.save('imdb_x_train.npy', X_train)
	# np.save('imdb_x_test.npy', X_test)
	# np.save('imdb_y_train.npy', Y_train)
	# np.save('imdb_y_test.npy', Y_test)
	print(X_train.shape)
	print(X_test.shape)
	print(X_train.dtype)
















