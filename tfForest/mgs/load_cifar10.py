from keras.datasets import cifar10
import numpy as np
import os


def load_cifar10(n_train=None, n_test=None):
	data_dir = "../DeepForestTF_Data/"
	if (os.path.exists(data_dir + "cifar10_x_train.npy") and os.path.exists(data_dir + "cifar10_x_test.npy")
		  and os.path.exists(data_dir + "cifar10_y_train.npy") and os.path.exists(data_dir + "cifar10_y_test.npy")):
		x_train = np.load(data_dir + "cifar10_x_train.npy")
		x_test = np.load(data_dir + "cifar10_x_test.npy")
		y_train = np.load(data_dir + "cifar10_y_train.npy")
		y_test = np.load(data_dir + "cifar10_y_test.npy")
	else:
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		x_train = x_train.transpose((0, 3, 1, 2))
		x_test = x_test.transpose((0, 3, 1, 2))
		y_train = y_train.reshape((y_train.shape[0]))
		y_test = y_test.reshape((y_test.shape[0]))

		x_train = x_train.reshape(50000, -1, 32, 32)
		x_test = x_test.reshape(10000, -1, 32, 32)

	if n_train is not None:
		x_train = x_train[:n_train, :, :, :]
		x_test = x_test[:n_test, :, :, :]
		y_train = y_train[:n_train]
		y_test = y_test[:n_test]

	y_train = y_train.astype(np.int)
	y_test = y_test.astype(np.int)

	return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test = load_cifar10()
	print(X_train.shape, X_test.shape)
	print(Y_train.shape, Y_train.shape)
	print(X_train.dtype, Y_train.dtype)
	pass

