import os
if 'http_proxy' in os.environ:
	os.environ.pop('http_proxy')
if 'https_proxy' in os.environ:
	os.environ.pop('https_proxy')

import argparse
import sys
import pickle
import tensorflow as tf
from time import sleep
import time
import numpy as np
import importlib
from kfoldwrapper import evaluate_performance
from mgs_helper import Window, MeanPooling, get_dim_from_window_and_pool, getmbof, scan, pool_shape, MGSWindow

# tf.logging.set_verbosity(tf.logging.INFO)
MAX_RAND_SEED = np.iinfo(np.int32).max
FLAGS = None
DTYPE = np.float64


def main(_):
	data_name = FLAGS.data   # data_name, one of 'letter', 'adult', 'yeast', 'imdb', ...
	load_xx = importlib.import_module('mgs.load_{}'.format(data_name))
	x_train, x_test, y_train, y_test = getattr(load_xx, 'load_{}'.format(data_name))()
	print(x_train.shape, x_test.shape)
	print(y_train.shape, y_test.shape)
	y_train = y_train.astype(np.int)
	y_test = y_test.astype(np.int)
	print("Num Classes: {}".format(len(set(y_test))))

	x_train_shape = x_train.shape
	x_test_shape = x_test.shape
	# num_features = x_test.shape[-1]
	if len(y_test.shape) == 1:
		num_classes = len(set(y_test))
	else:
		num_classes = y_test.shape[-1]
	n_estimators = 500
	max_nodes = 10000
	max_depth = 100

	ps_hosts = FLAGS.ps_hosts.split(",")
	worker_hosts = FLAGS.worker_hosts.split(",")
	merger_hosts = FLAGS.merger_hosts.split(",")

	cluster = tf.train.ClusterSpec({
		"ps": ps_hosts,
		"worker": worker_hosts,
		"merger": merger_hosts
	})

	num_split = FLAGS.numSplit

	if FLAGS.job_name == 'ps':
		parameter_server(num_workers=len(worker_hosts),
		                 num_mergers=len(merger_hosts),
		                 job_name=FLAGS.job_name,
		                 task_index=FLAGS.task_index,
		                 num_split=num_split,
		                 x_train_shape=x_train_shape,
		                 x_test_shape=x_test_shape,
		                 cluster=cluster,
		                 num_classes=num_classes,
		                 data_name=data_name)

	elif FLAGS.job_name == 'worker':
		# worker_type = 'E' if FLAGS.task_index < max(len(worker_hosts)//2, num_split) else 'R'
		worker_type = 'T'
		worker(num_workers=len(worker_hosts),
		       job_name=FLAGS.job_name,
		       task_index=FLAGS.task_index,
		       worker_type=worker_type,
		       num_split=num_split,
		       x_train_shape=x_train_shape,
		       x_test_shape=x_test_shape,
		       x_train=x_train,
		       x_test=x_test,
		       y_train=y_train,
		       y_test=y_test,
		       n_estimators=n_estimators,
		       max_depth=max_depth,
		       cluster=cluster,
		       num_classes=num_classes,
		       data_name=data_name
		       )
	elif FLAGS.job_name == 'merger':
		merger(num_workers=len(worker_hosts),
		       num_mergers=len(merger_hosts),
		       job_name=FLAGS.job_name,
		       task_index=FLAGS.task_index,
		       num_split=num_split,
		       x_train_shape=x_train_shape,
		       x_test_shape=x_test_shape,
		       cluster=cluster,
		       num_classes=num_classes,
		       data_name=data_name
		       )
	else:
		raise NotImplementedError("Not supported job_name: {}".format(FLAGS.job_name))


# Worker side
def get_prob(data_type="train", idx=0, x_train_shape=(), n_dims_train_win_i=0):
	with tf.variable_scope("prob_{}".format(data_type), reuse=True):
		x = tf.get_variable(name='x_{}'.format(idx),
		                    initializer=tf.zeros((x_train_shape[0], n_dims_train_win_i), dtype=DTYPE),
		                    dtype=DTYPE)
	return x


def get_windows(data_name):
	if data_name == 'cifar10':
		windows = [Window(win_x=8, win_y=8, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
		           Window(11, 11, 2, 2),
		           Window(16, 16, 2, 2)]
	elif data_name == 'mnist':
		windows = [Window(win_x=7, win_y=7, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
		           Window(10, 10, 2, 2),
		           Window(13, 13, 2, 2)]
	elif data_name == 'semg':
		windows = [MGSWindow((1, 187)),
		           MGSWindow((1, 375)),
		           MGSWindow((1, 750))]
	else:
		windows = []
	return windows


def parameter_server(num_workers=12,
                     num_mergers=1,
                     job_name="ps",
                     task_index=0,
                     num_split=1,
                     x_train_shape=(),
                     x_test_shape=(),
                     cluster=None,
                     num_classes=2,
                     data_name='cifar10'):
	windows = get_windows(data_name)
	pool = MeanPooling(2, 2)
	if data_name == 'semg':
		pool = MeanPooling(2, 1)
	n_dims_train = [get_dim_from_window_and_pool(x_train_shape=x_train_shape, window=windows[win_i],
				                                 pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
	n_dims_test = [get_dim_from_window_and_pool(x_train_shape=x_test_shape, window=windows[win_i],
				                                pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
	with tf.device("/job:{}/task:{}".format(job_name, task_index)):
		ready = tf.Variable([False for _ in range(num_workers)], name='ready')
		merged = tf.Variable([False for _ in range(num_mergers)], name='merged')

		prob_train_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_train"):
			for i in range(num_workers):
				win_i = i // (2 * num_split)
				prob_train_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                   initializer=tf.zeros((x_train_shape[0], n_dims_train[win_i]), dtype=DTYPE),
				                                   dtype=DTYPE)
		prob_test_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_test"):
			for i in range(num_workers):
				win_i = i // (2 * num_split)
				prob_test_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                  initializer=tf.zeros((x_test_shape[0], n_dims_test[win_i]), dtype=DTYPE),
				                                  dtype=DTYPE)
	start_time = time.time()
	server = tf.train.Server(cluster,
	                         job_name=job_name,
	                         task_index=task_index)
	sess = tf.Session(target=server.target)

	print("Parameter server: waiting for cluster connection...")
	sess.run(tf.report_uninitialized_variables())
	print("Parameter server: cluster ready!")

	print("Parameter server: initializing variables...")
	sess.run(tf.global_variables_initializer())
	print("Parameter server: variables initialized")

	while not sess.run(merged).all():
		sleep(1.0)
	print("All Ready!!!")

	res_train = [None for _ in range(len(windows))]
	res_test = [None for _ in range(len(windows))]

	for i in range(0, num_workers, 2*num_split):
		res_idx = i // (2*num_split)
		res_train[res_idx] = np.hstack((sess.run(prob_train_xs[i]), sess.run(prob_train_xs[i+num_split]))).astype(np.float32)
		res_test[res_idx] = np.hstack((sess.run(prob_test_xs[i]), sess.run(prob_test_xs[i+num_split]))).astype(np.float32)
		assert res_train[res_idx].shape == (x_train_shape[0], 2 * n_dims_train[res_idx]), ("Shape not consistent! " +
		                                                                                   "res_train[{}] shape = {} ".format(res_idx, res_train[res_idx].shape) +
		                                                                                   "should be {}".format((x_train_shape[0], 2 * n_dims_train[res_idx])))
		print(res_train[res_idx].shape)
		print("res_train_{}: ".format(res_idx), res_train[res_idx])

	[np.save("../DeepForestTF_Data/{}_res_train_{}.npy".format(data_name, i), res_train[i]) for i in range(len(windows))]
	[np.save("../DeepForestTF_Data/{}_res_test_{}.npy".format(data_name, i), res_test[i]) for i in range(len(windows))]

	print("Parameter server: blocking...")
	print("Time Cost: {}".format(time.time()-start_time))
	sess.close()
	sleep(5)
	exit(0)


def merger(num_workers=2,
           num_mergers=1,
           job_name="merger",
           task_index=0,
           num_split=1,
           x_train_shape=(),
           x_test_shape=(),
           cluster=None,
           num_classes=2,
           data_name='cifar10'):
	windows = get_windows(data_name)
	pool = MeanPooling(2, 2)
	if data_name == 'semg':
		pool = MeanPooling(2, 1)
	n_dims_train = [get_dim_from_window_and_pool(x_train_shape=x_train_shape, window=windows[win_i],
	                                             pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
	n_dims_test = [get_dim_from_window_and_pool(x_train_shape=x_test_shape, window=windows[win_i],
	                                            pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
	with tf.device("/job:ps/task:0"):
		ready = tf.Variable([False for _ in range(num_workers)], name='ready')
		merged = tf.Variable([False for _ in range(num_mergers)], name='merged')

		prob_train_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_train"):
			for i in range(num_workers):
				win_i = i // (2 * num_split)
				prob_train_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                   initializer=tf.zeros((x_train_shape[0], n_dims_train[win_i]),
				                                                        dtype=DTYPE),
				                                   dtype=DTYPE)
		prob_test_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_test"):
			for i in range(num_workers):
				win_i = i // (2 * num_split)
				prob_test_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                  initializer=tf.zeros((x_test_shape[0], n_dims_test[win_i]),
				                                                       dtype=DTYPE),
				                                  dtype=DTYPE)

	server = tf.train.Server(cluster,
	                         job_name=job_name,
	                         task_index=task_index)
	sess = tf.Session(target=server.target)

	print("Merger %d: waiting for cluster connection..." % task_index)
	sess.run(tf.report_uninitialized_variables())
	print("Merger %d: cluster ready!" % task_index)

	while sess.run(tf.report_uninitialized_variables()).any():
		print("Merger %d: waiting for variable initialization..." % task_index)
		sleep(1.0)
	print("Merger %d: variables initialized" % task_index)

	print("[Merger {}] Start merging ......".format(task_index))
	while not sess.run(ready).all():
		sleep(1.0)
	print("All Ready!!! Begin to Merge ...")

	# merge function ===============================================================
	for i in range(1, num_split):
		sess.run(prob_train_xs[task_index*num_split].assign_add(prob_train_xs[task_index*num_split + i]))
		sess.run(prob_test_xs[task_index*num_split].assign_add(prob_test_xs[task_index*num_split + i]))

	# setting merged flag ==========================================================
	indices = tf.constant([[task_index], ])
	updates = tf.constant([True])
	set_merged = merged.scatter_nd_update(indices, updates)
	sess.run(set_merged)
	# ==============================================================================

	print("Merger %d: blocking..." % task_index)
	sess.close()
	exit(0)


def worker(num_workers,
           job_name='worker',
           task_index=0,
           worker_type='R',
           num_split=1,
           x_train_shape=(),
           x_test_shape=(),
           x_train=None,
           x_test=None,
           y_train=None,
           y_test=None,
           n_estimators=500,
           max_depth=None,
           cluster=None,
           num_classes=2,
           data_name='cifar10'):
	windows = get_windows(data_name)
	pool = MeanPooling(2, 2)
	if data_name == 'semg':
		pool = MeanPooling(2, 1)
	n_dims_train = [get_dim_from_window_and_pool(x_train_shape=x_train_shape, window=windows[win_i],
	                                             pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
	n_dims_test = [get_dim_from_window_and_pool(x_train_shape=x_test_shape, window=windows[win_i],
	                                            pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
	with tf.device("/job:ps/task:0"):
		ready = tf.Variable([False for _ in range(num_workers)], name='ready')
		prob_train_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_train"):
			for i in range(num_workers):
				win_i = i // (2 * num_split)
				prob_train_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                   initializer=tf.zeros((x_train_shape[0], n_dims_train[win_i]),
				                                                        dtype=DTYPE),
				                                   dtype=DTYPE)
		prob_test_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_test"):
			for i in range(num_workers):
				win_i = i // (2 * num_split)
				prob_test_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                  initializer=tf.zeros((x_test_shape[0], n_dims_test[win_i]),
				                                                       dtype=DTYPE),
				                                  dtype=DTYPE)

	server = tf.train.Server(cluster,
	                         job_name=job_name,
	                         task_index=task_index)
	sess = tf.Session(target=server.target)

	print("Worker %d: waiting for cluster connection..." % task_index)
	sess.run(tf.report_uninitialized_variables())
	print("Worker %d: cluster ready!" % task_index)

	while sess.run(tf.report_uninitialized_variables()).any():
		print("Worker %d: waiting for variable initialization..." % task_index)
		sleep(1.0)
	print("Worker %d: variables initialized" % task_index)

	hash_dic = pickle.load(open('hash23_mgs', 'rb'))

	print("[Worker {}] Start training ......".format(task_index))

	this_win_i = task_index // (2 * num_split)

	# print(x_train[:10])

	x_win_train_wi = scan(windows[this_win_i], x_train)
	x_win_test_wi = scan(windows[this_win_i], x_test)

	print('X_win_train_{}: {}, size={}'.format(this_win_i, x_win_train_wi.shape, getmbof(x_win_train_wi)))
	print('X_win_test_{}: {}, size={}'.format(this_win_i, x_win_test_wi.shape, getmbof(x_win_test_wi)))
	# X_wins[wi] = (60000, 11, 11, 49)
	_, nh, nw, _ = x_win_train_wi.shape
	# (60000 * 121, 49)
	x_win_train_wi = x_win_train_wi.reshape((x_win_train_wi.shape[0], -1, x_win_train_wi.shape[-1]))
	# print("x_win_train_wi.shape = {}".format(x_win_train_wi.shape))
	y_win = y_train[:, np.newaxis].repeat(x_win_train_wi.shape[1], axis=1)
	# print("y_win previous = {}".format(y_win.shape))
	y_stratify = y_win[:, 0]
	# print("y_win[:, 0] shape = {}".format(y_stratify.shape))
	# y_win = y_win.reshape(-1)
	# print("y_win.shape = {}".format(y_win.shape))
	x_win_test_wi = x_win_test_wi.reshape((x_win_test_wi.shape[0], -1, x_win_test_wi.shape[-1]))
	y_win_test = y_test[:, np.newaxis].repeat(x_win_test_wi.shape[1], axis=1)
	# y_win_test = y_win_test.reshape(-1)
	# setting seed ================================================================
	est_name = 'win-{}-estimator-{}-{}folds'.format(this_win_i, (task_index % (2*num_split))//num_split, 3)
	# print("HASH[{}] = {}".format("[estimator] {}".format(est_name), hash_dic["[estimator] {}".format(est_name)]))
	seed = (0 + hash_dic["[estimator] {}".format(est_name)]) % 1000000007
	cv_seed = (0 + hash_dic["[estimator] {}".format(est_name)]) % 1000000007

	each_part = int(np.ceil(n_estimators / num_split))
	last_part_trees = n_estimators - each_part * (num_split-1)
	initial = (task_index % (2*num_split)) % num_split * each_part
	this_estimators = last_part_trees if (task_index % (2*num_split)) % num_split == num_split - 1 else each_part

	print("each part = {}\n".format(each_part) +
	      "last_part_trees = {}\n".format(last_part_trees) +
	      "initial = {}\n".format(initial) +
	      "this_estimators = {}\n".format(this_estimators))

	if worker_type in ['E', 'R']:
		seed_obj = np.random.RandomState(seed)
		if initial > 0:
			seed_obj.randint(MAX_RAND_SEED, size=initial)
	elif worker_type == 'T':
		seed_obj = seed
		seed_obj += initial
	else:
		raise NotImplementedError("Unsupported Worker Type: {} !".format(worker_type))
	# end setting seed ============================================================

	print("=============> I handle {} trees, my initial seed is {} <=============".format(this_estimators, seed_obj))

	# SKLearn KFoldWrapper ========================================================
	from kfoldwrapper import TFKFoldWrapper, SKKFoldWrapper, RFKFoldWrapper
	if worker_type == 'E':
		est_args = {'n_estimators': this_estimators, 'max_features': 'auto', 'max_depth': max_depth,
		            'n_jobs': -1, 'criterion': "gini",
		            'min_samples_split': 2, 'min_impurity_decrease': 0., 'min_samples_leaf': 10,
		            'random_state': None}
		kfw = SKKFoldWrapper(name=est_name, n_folds=3,
		                     task='classification',
		                     seed=seed_obj, dtype=DTYPE,
		                     est_args=est_args, cv_seed=cv_seed)
	elif worker_type == 'R':
		est_args = {'n_estimators': this_estimators, 'max_features': 'sqrt', 'max_depth': max_depth,
		            'n_jobs': -1, 'criterion': "gini",
		            'min_samples_split': 2, 'min_impurity_decrease': 0., 'min_samples_leaf': 10,
		            'random_state': None}
		kfw = RFKFoldWrapper(name=est_name, n_folds=3,
		                     task='classification',
		                     seed=seed_obj, dtype=DTYPE,
		                     est_args=est_args, cv_seed=cv_seed)
	else:
		max_nodes = 100000
		if data_name == 'mnist':
			max_nodes = 237000
		elif data_name == 'semg':
			max_nodes = 320000
		elif data_name == 'cifar10':
			max_nodes = 788000
		est_args = {'num_classes': num_classes, 'num_features': x_win_train_wi.shape[-1], 'regression': False,
		            'num_trees': this_estimators, 'max_nodes': max_nodes, 'valid_leaf_threshold': 10,
		            'base_random_seed': None, 'collate_examples': True}
		kfw = TFKFoldWrapper(name=est_name, n_folds=3,
		                     task='classification',
		                     seed=seed, dtype=DTYPE,
		                     est_args=est_args, cv_seed=seed)
	# =============================================================================

	x_win_train_wi = x_win_train_wi.astype(np.float32)
	x_win_test_wi = x_win_test_wi.astype(np.float32)

	y_proba_train, y_proba_test = kfw.fit_transform(X=x_win_train_wi, y=y_win, y_stratify=y_stratify,
	                                                x_test=x_win_test_wi, y_test=y_win_test)

	print("OUT y_proba_train.shape, y_proba_test.shape = {}, {}".format(y_proba_train.shape, y_proba_test.shape))

	y_proba_train = y_proba_train.reshape((-1, nh, nw, num_classes)).transpose((0, 3, 1, 2)).astype(DTYPE)
	y_proba_test = y_proba_test.reshape((-1, nh, nw, num_classes)).transpose((0, 3, 1, 2)).astype(DTYPE)

	print("Y_proba_train.shape, Y_proba_test.shape = {}, {}".format(y_proba_train.shape, y_proba_test.shape))

	# n, c, h, w
	y_proba_train = pool.fit_transform(y_proba_train)
	y_proba_test = pool.fit_transform(y_proba_test)

	print("POOLED Y_proba_train.shape, Y_proba_test.shape = {}, {}".format(y_proba_train.shape, y_proba_test.shape))

	y_proba_train = y_proba_train.reshape((y_proba_train.shape[0], -1))
	y_proba_test = y_proba_test.reshape((y_proba_test.shape[0], -1))

	print("RESHAPED Y_proba_train.shape, Y_proba_test.shape = {}, {}".format(y_proba_train.shape, y_proba_test.shape))

	x = get_prob("train", task_index, x_test_shape, n_dims_test[this_win_i])
	sess.run(x.assign(y_proba_train * this_estimators / n_estimators))

	x = get_prob("test", task_index, x_train_shape, n_dims_train[this_win_i])
	sess.run(x.assign(y_proba_test * this_estimators / n_estimators))

	# ready ========================================================================
	indices = tf.constant([[task_index], ])
	updates = tf.constant([True])
	set_ready = ready.scatter_nd_update(indices, updates)
	sess.run(set_ready)
	# ==============================================================================

	print("Worker %d: blocking..." % task_index)
	# server.join()
	sess.close()
	exit(0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.register('type', 'bool', lambda v: v.lower() == 'true')
	parser.add_argument(
		"--ps_hosts",
		type=str,
		default='',
		help="Comma-separated list of hostname:port pairs"
	)
	parser.add_argument(
	  "--worker_hosts",
	  type=str,
	  default="",
	  help="Comma-separated list of hostname:port pairs"
	)
	parser.add_argument(
		"--merger_hosts",
		type=str,
		default="",
		help="Comma-separated list of hostname:port pairs"
	)
	parser.add_argument(
	  "--job_name",
	  type=str,
	  default="",
	  help="One of 'ps', 'worker', 'merger'"
	)
	# Flags for defining the tf.train.Server
	parser.add_argument(
	  "--task_index",
	  type=int,
	  default=0,
	  help="Index of task within the job"
	)
	parser.add_argument(
		"--data",
		type=str,
		default='letter',
		help="Data to train and evaluate"
	)
	parser.add_argument(
		"--numSplit",
		type=int,
		default=1,
		help="How much groups to split"
	)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



