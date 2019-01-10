from mgs.load_cifar10 import load_cifar10 as load_cifar10_y
import numpy as np


def load_cifar10():
	data_dir = "../DeepForestTF_Data/"
	x_train = [None, None, None]
	x_test = [None, None, None]

	for i in range(3):
		x_train[i] = np.load(data_dir + "cifar10_res_train_{}.npy".format(i))
		x_test[i] = np.load(data_dir + "cifar10_res_test_{}.npy".format(i))

	_, _, y_train, y_test = load_cifar10_y()

	return x_train, x_test, y_train, y_test


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test = load_cifar10()
	print(X_train[0].shape, X_test[0].shape)
	print(Y_train.shape, Y_train.shape)
	print(X_train[0].dtype, Y_train.dtype)
	pass

