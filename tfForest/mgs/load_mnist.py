import os
from keras.datasets import mnist
import numpy as np


def load_mnist():
	data_dir = "../DeepForestTF_Data/"
	if (os.path.exists(data_dir + "mnist_x_train.npy") and os.path.exists(data_dir + "mnist_x_test.npy")
		  and os.path.exists(data_dir + "mnist_y_train.npy") and os.path.exists(data_dir + "mnist_y_test.npy")):
		x_train = np.load(data_dir + "mnist_x_train.npy")
		x_test = np.load(data_dir + "mnist_x_test.npy")
		y_train = np.load(data_dir + "mnist_y_train.npy")
		y_test = np.load(data_dir + "mnist_y_test.npy")

		return x_train, x_test, y_train, y_test
	else:
		(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(60000, -1, 28, 28).astype('float32')
	x_test = x_test.reshape(10000, -1, 28, 28).astype('float32')
	x_train = x_train / 255.0
	x_test = x_test / 255.0

	return x_train, x_test, y_train.astype(np.int), y_test.astype(np.int)


if __name__ == '__main__':
	X_train, X_test, Y_train, Y_test = load_mnist()
	# np.save('mnist_x_train.npy', X_train)
	# np.save('mnist_x_test.npy', X_test)
	# np.save('mnist_y_train.npy', Y_train)
	# np.save('mnist_y_test.npy', Y_test)
	# x_train = x_train[:200, :, :, :]
	# x_test = x_test[:100, :, :, :]
	#
	# y_train = y_train[:200]
	# y_test = y_test[:100]
	print(X_train.shape, X_train.dtype)
	print(Y_train.dtype)

