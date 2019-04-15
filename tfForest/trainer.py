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

# tf.logging.set_verbosity(tf.logging.INFO)
MAX_RAND_SEED = np.iinfo(np.int32).max
FLAGS = None
DTYPE = np.float64


def main(_):
	data_name = FLAGS.data   # data_name, one of 'letter', 'adult', 'yeast', 'imdb', ...
	load_xx = importlib.import_module('load_{}'.format(data_name))
	x_train, x_test, y_train, y_test = getattr(load_xx, 'load_{}'.format(data_name))()
	if isinstance(x_train, (tuple, list)):
		for i in range(len(x_train)):
			print(x_train[i].shape, x_test[i].shape)
		n_train = x_train[0].reshape((-1, x_train[0].shape[-1])).shape[0]
		n_test = x_test[0].reshape((-1, x_test[0].shape[-1])).shape[0]
	else:
		print(x_train.shape, x_test.shape)
		n_train = x_train.reshape((-1, x_train.shape[-1])).shape[0]
		n_test = x_test.reshape((-1, x_test.shape[-1])).shape[0]

	print(y_train.shape, y_test.shape)
	if len(y_test.shape) == 1:
		num_classes = len(set(y_test))
	else:
		num_classes = y_test.shape[-1]
	y_train = y_train.astype(np.int)
	y_test = y_test.astype(np.int)
	print("Num Classes: {}".format(len(set(y_test))))
	n_estimators = 500
	# max_leaf_nodes = 10000
	max_depth = 100
	early_stopping_rounds = 4

	if data_name == 'cifar10':
		n_estimators = 1000
		early_stopping_rounds = 8

	ps_hosts = FLAGS.ps_hosts.split(",")
	worker_hosts = FLAGS.worker_hosts.split(",")
	# merger_hosts = FLAGS.merger_hosts.split(",")

	cluster = tf.train.ClusterSpec({
		"ps": ps_hosts,
		"worker": worker_hosts,
		# "merger": merger_hosts
	})

	num_split = FLAGS.numSplit

	if FLAGS.job_name == 'ps':
		parameter_server(num_workers=len(worker_hosts),
		                 job_name=FLAGS.job_name,
		                 task_index=FLAGS.task_index,
		                 num_split=num_split,
		                 n_train=n_train,
		                 n_test=n_test,
		                 y_train=y_train,
		                 y_test=y_test,
		                 cluster=cluster,
		                 num_classes=num_classes,
		                 early_stopping_rounds=early_stopping_rounds)

	elif FLAGS.job_name == 'worker':
		# worker_type = 'E' if FLAGS.task_index < max(len(worker_hosts)//2, num_split) else 'R'
		worker_type = 'T'
		worker(num_workers=len(worker_hosts),
		       job_name=FLAGS.job_name,
		       task_index=FLAGS.task_index,
		       worker_type=worker_type,
		       num_split=num_split,
		       n_train=n_train,
		       n_test=n_test,
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
	else:
		raise NotImplementedError("Not supported job_name: {}".format(FLAGS.job_name))


# Worker side
def get_prob(data_type="train", idx=0, n_train=0, num_classes=0):
	with tf.variable_scope("prob_{}".format(data_type), reuse=True):
		x = tf.get_variable(name='x_{}'.format(idx),
		                    initializer=tf.zeros((n_train, num_classes), dtype=DTYPE),
		                    dtype=DTYPE)
	return x


def parameter_server(num_workers=2,
                     job_name="ps",
                     task_index=0,
                     num_split=1,
                     n_train=32561,
                     n_test=16281,
                     y_train=None,
                     y_test=None,
                     cluster=None,
                     num_classes=2,
                     early_stopping_rounds=4):
	with tf.device("/job:{}/task:{}".format(job_name, task_index)):
		cont = tf.Variable([True for _ in range(num_workers)], name='continue')
		shut_down = tf.Variable(False, name='shutdown')
		concat = tf.Variable(False, name='concat')
		ready = tf.Variable([False for _ in range(num_workers)], name='ready')
		prob_train_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_train"):
			for i in range(num_workers):
				prob_train_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                   initializer=tf.zeros((n_train, num_classes), dtype=DTYPE),
				                                   dtype=DTYPE)
		prob_test_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_test"):
			for i in range(num_workers):
				prob_test_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                  initializer=tf.zeros((n_test, num_classes), dtype=DTYPE),
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

	train_acc_records = []
	test_acc_records = []
	best_test_acc = 0.0
	best_train_acc = 0.0
	best_layer = 0
	early_stopping = False

	layer_train_prob = np.zeros((n_train, num_classes), dtype=np.float32)
	layer_test_prob = np.zeros((n_test, num_classes), dtype=np.float32)

	num_layer = 0
	while True:
		while not sess.run(ready).all():
			sleep(1.0)
		print("All Ready!!!")

		# clear layer_train_prob, layer_test_prob ==========================
		layer_train_prob.fill(0.0)
		layer_test_prob.fill(0.0)

		# merge ============================================================
		for i in range(0, num_workers, num_split):
			for j in range(0, num_split):
				sess.run(prob_train_xs[i].assign_add(prob_train_xs[i + j]))
				sess.run(prob_test_xs[i].assign_add(prob_test_xs[i + j]))
			layer_train_prob += sess.run(prob_train_xs[i])
			layer_test_prob += sess.run(prob_test_xs[i])
		# ==================================================================

		layer_train_prob /= num_workers
		layer_test_prob /= num_workers
		# print("Layer train Probability: {}".format(layer_train_prob))
		# print("Layer test  Probability: {}".format(layer_test_prob))
		# print("LAYER TRAIN DTYPE = {}".format(layer_train_prob.dtype))
		train_acc = evaluate_performance(layer_train_prob, y_train)
		test_acc = evaluate_performance(layer_test_prob, y_test)
		print("LAYER train={}, test={}".format(train_acc, test_acc))
		train_acc_records.append(train_acc)
		test_acc_records.append(test_acc)
		if test_acc > best_test_acc:
			best_train_acc = train_acc
			best_test_acc = test_acc
			best_layer = num_layer
		if num_layer - best_layer >= early_stopping_rounds:
			early_stopping = True

		each_train_acc = [evaluate_performance(sess.run(prob_train_xs[i]).astype(np.float32), y_train)
		                  for i in range(0, num_workers, num_split)]
		each_test_acc = [evaluate_performance(sess.run(prob_test_xs[i]).astype(np.float32), y_test)
		                 for i in range(0, num_workers, num_split)]
		print("Every train acc = {}".format(each_train_acc))
		print("Every test  acc = {}".format(each_test_acc))
		num_layer += 1

		sess.run(ready.assign([False for _ in range(num_workers)]))
		if num_layer > 0:  # to concat
			sess.run(concat.assign(True))
		if early_stopping or num_layer > 30:  # judge early-stopping, or max_layer encountered
			print("Early Stopped, best_layer = {}, train_acc = {}, test_acc = {}".format(best_layer,
			                                                                             best_train_acc,
			                                                                             best_test_acc))
			sess.run(shut_down.assign(True))
			break

		sess.run(cont.assign([True for _ in range(num_workers)]))
		print("Time Cost: {}".format(time.time()-start_time))
		print("Continue Training Layer {} ......".format(num_layer))

	print("Parameter server: blocking...")
	print("Time Cost: {}".format(time.time()-start_time))
	sess.close()
	sleep(10)
	exit(0)


def worker(num_workers,
           job_name='worker',
           task_index=0,
           worker_type='R',
           num_split=1,
           n_train=32561,
           n_test=16281,
           x_train=None,
           x_test=None,
           y_train=None,
           y_test=None,
           n_estimators=500,
           max_depth=100,
           cluster=None,
           num_classes=2,
           data_name='adult'):
	with tf.device("/job:ps/task:0"):
		cont = tf.Variable([True for _ in range(num_workers)], name='continue')
		shut_down = tf.Variable(False, name='shutdown')
		concat = tf.Variable(False, name='concat')
		ready = tf.Variable([False for _ in range(num_workers)], name='ready')
		prob_train_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_train"):
			for i in range(num_workers):
				prob_train_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                   initializer=tf.zeros((n_train, num_classes), dtype=DTYPE),
				                                   dtype=DTYPE)
		prob_test_xs = [None for _ in range(num_workers)]
		with tf.variable_scope("prob_test"):
			for i in range(num_workers):
				prob_test_xs[i] = tf.get_variable(name='x_{}'.format(i),
				                                  initializer=tf.zeros((n_test, num_classes), dtype=DTYPE),
				                                  dtype=DTYPE)

	server = tf.train.Server(cluster,
	                         job_name=job_name,
	                         task_index=task_index)
	sess = tf.Session(target=server.target)

	print("Worker %d: waiting for cluster connection..." % task_index)
	sess.run(tf.report_uninitialized_variables())
	print("Worker %d: cluster ready!" % task_index)

	c = 0
	while sess.run(tf.report_uninitialized_variables()).any():
		print("Worker %d: waiting for variable initialization..." % task_index)
		sleep(1.0)
		c += 1
		if c % 10 == 0:
			print(sess.run(tf.report_uninitialized_variables()))
	print("Worker %d: variables initialized" % task_index)

	hash_dic = pickle.load(open('hash23', 'rb'))

	if not isinstance(x_train, (tuple, list)):
		x_train = [x_train]
		x_test = [x_test]

	if data_name == 'cifar10':
		look_index_cycle = [[0], [1], [2], [0, 1, 2]]
	elif data_name in ['mnist', 'semg']:
		look_index_cycle = [[0], [1], [2]]
	else:
		look_index_cycle = [[0]]
	num_cycle = len(look_index_cycle)

	num_layer = 0
	while True:
		print("[Worker {}] Start training Layer {} ......".format(task_index, num_layer))

		# now continue, close continue tag ============================================
		indices = tf.constant([[task_index], ])
		updates = tf.constant([False])
		set_cont = cont.scatter_nd_update(indices, updates)
		sess.run(set_cont)
		# =============================================================================

		# concat ======================================================================
		need_concat = sess.run(concat)
		print("Need Concat ? {}".format(need_concat))

		if num_cycle > 1:
			chosed_input_train = np.hstack((x_train[idx] for idx in look_index_cycle[num_layer % num_cycle]))
			chosed_input_test = np.hstack((x_test[idx] for idx in look_index_cycle[num_layer % num_cycle]))
		else:
			chosed_input_train = x_train[0]
			chosed_input_test = x_test[0]

		if need_concat:
			true_x_train = np.hstack((chosed_input_train,) + tuple(sess.run(prob_train_xs[i]).astype(np.float32)
			                                                       for i in range(0, num_workers, num_split)))
			true_x_test = np.hstack((chosed_input_test,) + tuple(sess.run(prob_test_xs[i]).astype(np.float32)
			                                                     for i in range(0, num_workers, num_split)))
		else:
			true_x_train = chosed_input_train
			true_x_test = chosed_input_test
		print("true_x_train, true_x_test shape = {}, {}".format(true_x_train.shape, true_x_test.shape))
		print("true_x_train, true_x_test dtype = {}, {}".format(true_x_train.dtype, true_x_test.dtype))
		# print("true_x_train = {}".format(true_x_train))
		# end concat ==================================================================

		# setting seed ================================================================
		est_name = 'layer-{}-estimator-{}-{}folds'.format(num_layer, task_index//num_split, 3)
		seed = (0 + hash_dic["[estimator] {}".format(est_name)]) % 1000000007
		cv_seed = (0 + hash_dic["[estimator] {}".format(est_name)]) % 1000000007

		each_part = int(np.ceil(n_estimators / num_split))
		last_part_trees = n_estimators - each_part * (num_split-1)
		initial = task_index % num_split * each_part
		this_estimators = last_part_trees if task_index % num_split == num_split - 1 else each_part

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
			est_args = {'n_estimators': this_estimators, 'max_features': 1, 'max_depth': max_depth,
			            'n_jobs': -1, 'criterion': "gini",
			            'min_samples_split': 2, 'min_impurity_decrease': 0., 'min_samples_leaf': 1,
			            'random_state': None}
			kfw = SKKFoldWrapper(name=est_name, n_folds=3,
			                     task='classification',
			                     seed=seed_obj, dtype=DTYPE,
			                     est_args=est_args, cv_seed=cv_seed)
		elif worker_type == 'R':
			est_args = {'n_estimators': this_estimators, 'max_features': 'sqrt', 'max_depth': max_depth,
			            'n_jobs': -1, 'criterion': "gini",
			            'min_samples_split': 2, 'min_impurity_decrease': 0., 'min_samples_leaf': 1,
			            'random_state': None}
			kfw = RFKFoldWrapper(name=est_name, n_folds=3,
			                     task='classification',
			                     seed=seed_obj, dtype=DTYPE,
			                     est_args=est_args, cv_seed=cv_seed)
		else:
			max_nodes = 10000
			if data_name == 'cifar10':
				max_nodes = 60000
			elif data_name == 'semg':
				max_nodes = 1450
			elif data_name == 'mnist':
				max_nodes = 45000
			elif data_name == 'letter':
				max_nodes = 6000
			elif data_name == 'adult':
				max_nodes = 17000
			elif data_name == 'yeast':
				max_nodes = 1100
			print("Max Nodes: {}".format(max_nodes))
			est_args = {'num_classes': num_classes, 'num_features': true_x_train.shape[-1], 'regression': False,
			            'num_trees': this_estimators, 'max_nodes': max_nodes,
			            'base_random_seed': None, 'collate_examples': True}
			kfw = TFKFoldWrapper(name=est_name, n_folds=3,
			                     task='classification',
			                     seed=seed, dtype=DTYPE,
			                     est_args=est_args, cv_seed=seed)
		# =============================================================================

		y_proba_train, y_proba_test = kfw.fit_transform(
			X=true_x_train, y=y_train, x_test=true_x_test, y_test=y_test)
		# print("OUT y_proba_train.dtype = {}, y_proba_test = {}".format(y_proba_train.dtype, y_proba_test.dtype))
		# =============================================================================

		x = get_prob("test", task_index, n_train, num_classes)
		# print("Assign Adding {} ...".format(x.name))
		sess.run(x.assign(y_proba_test * this_estimators / n_estimators))
		# sess.run(layer_test_prob.assign_add(y_proba_test))
		# ==============================================================================

		x = get_prob("train", task_index, n_train, num_classes)
		# print("Assign Adding {} ...".format(x.name))
		sess.run(x.assign(y_proba_train * this_estimators / n_estimators))
		# sess.run(layer_train_prob.assign_add(y_proba_train))
		# ==============================================================================

		# ready ========================================================================
		indices = tf.constant([[task_index], ])
		updates = tf.constant([True])
		set_ready = ready.scatter_nd_update(indices, updates)
		sess.run(set_ready)
		# ==============================================================================

		# wait the indicator of the next run ===========================================
		print("[Worker {}] wait the indicator of the next run ...".format(task_index))
		is_shutdown = False
		while not sess.run(cont)[task_index]:  # while not continue
			time.sleep(1.0)
			print("cont indicator: ", sess.run(cont)[task_index])
			if sess.run(shut_down):
				is_shutdown = True
				break
		if is_shutdown:
			print("Worker %d shutting down..." % task_index)
			break
		num_layer += 1

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



