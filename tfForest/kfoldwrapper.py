import copy
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble.forest import ExtraTreesClassifier, RandomForestClassifier
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat


# Parameter Server side
def evaluate_performance(layer_prob, y_true):

	y_true = y_true.reshape(-1)
	y_pred = np.argmax(layer_prob.reshape((-1, layer_prob.shape[-1])), 1)
	assert len(y_true) > 0, "[evaluate performance] y_true length can not be zero!"
	acc = 100. * np.sum(y_true == y_pred) / len(y_true)
	return acc


class KFoldWrapper(object):
	def __init__(self, name, n_folds, task, seed=None, dtype=np.float32,
	             eval_metrics=None, est_args=None, cv_seed=None):
		self.name = name
		self.n_folds = n_folds
		self.task = task
		self.est_args = est_args if est_args is not None else {}
		self.seed = seed
		self.dtype = dtype
		self.cv_seed = cv_seed
		# assign a cv_seed
		if self.cv_seed is None:
			self.cv_seed = self.seed
		self.eval_metrics = eval_metrics if eval_metrics is not None else []
		self.n_dims = None

	def _init_estimator(self, k):
		raise NotImplementedError

	def _kfold_fit(self, est, X, y):
		raise NotImplementedError

	def _kfold_predict_proba(self, est, X):
		raise NotImplementedError

	def fit_transform(self, X, y, y_stratify=None, x_test=None, y_test=None):
		"""
		Fit and transform.

		:param X: (ndarray) n x k or n1 x n2 x k
							to support windows_layer, X could have dim >2
		:param y: (ndarray) y (ndarray):
							n or n1 x n2
		:param y_stratify: (list) used for StratifiedKFold or None means no stratify
		:param x_test:
		:param y_test:
		:return:
		"""
		assert 2 <= len(X.shape) <= 3, "X.shape should be n x k or n x n2 x k"
		assert len(X.shape) == len(y.shape) + 1

		# get y_stratify ================================
		if y_stratify is not None:
			assert X.shape[0] == len(y_stratify)
		else:
			y_stratify = y
		# ===============================================

		# K-Fold split ==================================
		n_stratify = X.shape[0]
		if self.n_folds == 1:
			cv = [(range(len(X)), range(len(X)))]
		else:
			if y_stratify is None:
				skf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
				cv = [(t, v) for (t, v) in skf.split(range(n_stratify))]
			else:
				skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.cv_seed)
				cv = [(t, v) for (t, v) in skf.split(range(n_stratify), y_stratify)]
		y_proba_train = None
		y_proba_test = None
		self.n_dims = X.shape[-1]
		# print("CV_seed = {}".format(self.cv_seed))
		for k in range(self.n_folds):

			est = self._init_estimator(k)

			train_idx, val_idx = cv[k]

			# print("Fold {}, x_train = {}".format(k, X[train_idx].reshape((-1, self.n_dims))))
			# print("shape = {}".format(X[train_idx].reshape((-1, self.n_dims)).shape))
			# print("y[train_idx].shape = ", y[train_idx].reshape(-1).shape)
			est = self._kfold_fit(est, X[train_idx].reshape((-1, self.n_dims)), y[train_idx].reshape(-1))

			y_proba = self._kfold_predict_proba(est, X[val_idx].reshape((-1, self.n_dims)))
			# print("y_proba: ", y_proba)
			if len(X.shape) == 3:
				y_proba = y_proba.reshape((len(val_idx), -1, y_proba.shape[-1]))

			self.log_metrics(self.name, y[val_idx], y_proba, "train_{}".format(k))
			# =========================================================================

			# merging result ==========================================================
			if k == 0:
				if len(X.shape) == 2:
					y_proba_cv = np.zeros((n_stratify, y_proba.shape[1]), dtype=self.dtype)
				else:
					y_proba_cv = np.zeros((n_stratify, y_proba.shape[1], y_proba.shape[2]), dtype=self.dtype)
				y_proba_train = y_proba_cv
			y_proba_train[val_idx, :] += y_proba
			# =========================================================================

			y_proba_t = self._kfold_predict_proba(est, x_test.reshape((-1, self.n_dims)))

			if len(X.shape) == 3:
				y_proba_t = y_proba_t.reshape((x_test.shape[0], x_test.shape[1], y_proba_t.shape[-1]))
			if k == 0:
				y_proba_test = y_proba_t
			else:
				y_proba_test += y_proba_t

		y_proba_test /= self.n_folds
		# =========================================================================

		# log train average
		self.log_metrics(self.name, y, y_proba_train, "train_avg")

		# y_test can be None
		if y_test is not None:
			self.log_metrics(self.name, y_test, y_proba_test, "test_avg")

		return y_proba_train, y_proba_test

	@staticmethod
	def log_metrics(est_name, y_true, y_proba, y_name):
		"""
		y_true (ndarray): n or n1 x n2
		y_proba (ndarray): n x n_classes or n1 x n2 x n_classes
		"""
		acc = evaluate_performance(y_proba, y_true)
		print("{}({} - {}) = {:.4f}{}".format(
			"Accuracy", est_name, y_name, acc, '%'))


class TFKFoldWrapper(KFoldWrapper):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def _init_estimator(self, k):
		est_args = self.est_args.copy()
		est_name = '{}/{}'.format(self.name, k)
		# TODO: consider if add a random_state, actually random_state of each estimator can be set in est_configs in
		# main program by users, so we need not to set random_state there.
		# More importantly, if some estimators have no random_state parameter, this assignment can throw problems.
		if est_args.get('base_random_seed', None) is None:
			est_args['base_random_seed'] = copy.deepcopy(self.seed)
		else:
			est_args['base_random_seed'] = est_args['base_random_seed'] + k ** 2

		params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(**est_args)
		estimator = SKCompat(tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params))

		return estimator

	def _kfold_fit(self, est, X, y, **kwargs):
		# fit on k-fold train =====================================================
		est.fit(X, y, **kwargs)
		return est
		# =========================================================================

	def _kfold_predict_proba(self, est, X):
		# predict on k-fold validation, this y_proba.dtype is float32 =============
		y_out = est.predict(X)
		# yyy = [x for x in y_out]
		y_proba = np.array(y_out['probabilities'])
		return y_proba


class SKKFoldWrapper(KFoldWrapper):
	def __init__(self, **kwargs):
		super(SKKFoldWrapper, self).__init__(**kwargs)

	def _init_estimator(self, k):
		est_args = self.est_args.copy()
		est_name = '{}/{}'.format(self.name, k)
		# TODO: consider if add a random_state, actually random_state of each estimator can be set in est_configs in
		# main program by users, so we need not to set random_state there.
		# More importantly, if some estimators have no random_state parameter, this assignment can throw problems.
		if est_args.get('random_state', None) is None:
			est_args['random_state'] = copy.deepcopy(self.seed)
		else:
			print("RED ALERT...(SKKFoldWrapper)")
			est_args['random_state'] = est_args['random_state'] + k ** 2

		# estimator = ExtraTreesClassifier(**est_args)
		estimator = ExtraTreesClassifier(**est_args)
		print("ESTIMATOR: ExtraTreesClassifier")

		return estimator

	def _kfold_fit(self, est, X, y):
		# fit on k-fold train =====================================================
		est.fit(X, y)
		# print([e_.tree_.node_count for e_ in est.estimators_])
		return est
		# =========================================================================

	def _kfold_predict_proba(self, est, X):
		# predict on k-fold validation, this y_proba.dtype is float32 =============
		y_proba = est.predict_proba(X)
		return y_proba


class RFKFoldWrapper(SKKFoldWrapper):
	def _init_estimator(self, k):
		est_args = self.est_args.copy()
		est_name = '{}/{}'.format(self.name, k)
		# TODO: consider if add a random_state, actually random_state of each estimator can be set in est_configs in
		# main program by users, so we need not to set random_state there.
		# More importantly, if some estimators have no random_state parameter, this assignment can throw problems.
		if est_args.get('random_state', None) is None:
			est_args['random_state'] = copy.deepcopy(self.seed)
		else:
			print("RED ALERT...(RFKFoldWrapper)")
			est_args['random_state'] = est_args['random_state'] + k ** 2

		# estimator = ExtraTreesClassifier(**est_args)
		estimator = RandomForestClassifier(**est_args)
		print("ESTIMATOR: RandomForestClassifier")

		return estimator


def t_kfold_wrapper():
	DTYPE = np.float64
	MAX_RAND_SEED = np.iinfo(np.int32).max
	import pickle
	from mgs_helper import Window, MeanPooling, get_dim_from_window_and_pool, getmbof, scan, pool_shape
	from load_cifar10 import load_cifar10
	x_train, x_test, y_train, y_test = load_cifar10()
	x_train_shape = x_train.shape
	x_test_shape = x_test.shape
	num_classes = 10
	task_index = 0
	num_split = 1
	n_estimators = 1000

	windows = [Window(win_x=8, win_y=8, stride_x=2, stride_y=2, pad_x=0, pad_y=0),
	           Window(11, 11, 2, 2),
	           Window(16, 16, 2, 2)]
	pool = MeanPooling(2, 2)
	n_dims_train = [get_dim_from_window_and_pool(x_train_shape=x_train_shape, window=windows[win_i],
	                                             pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
	n_dims_test = [get_dim_from_window_and_pool(x_train_shape=x_test_shape, window=windows[win_i],
	                                            pool=pool, n_classes=num_classes) for win_i in range(len(windows))]
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
	est_name = 'win-{}-estimator-{}-{}folds'.format(this_win_i, (task_index % (2 * num_split)) // num_split, 3)
	# print("HASH[{}] = {}".format("[estimator] {}".format(est_name), hash_dic["[estimator] {}".format(est_name)]))
	seed = (0 + hash_dic["[estimator] {}".format(est_name)]) % 1000000007
	cv_seed = (0 + hash_dic["[estimator] {}".format(est_name)]) % 1000000007

	each_part = int(np.ceil(n_estimators / num_split))
	last_part_trees = n_estimators - each_part * (num_split - 1)
	initial = (task_index % (2 * num_split)) % num_split * each_part
	this_estimators = last_part_trees if (task_index % (2 * num_split)) % num_split == num_split - 1 else each_part

	print("each part = {}\n".format(each_part) +
	      "last_part_trees = {}\n".format(last_part_trees) +
	      "initial = {}\n".format(initial) +
	      "this_estimators = {}\n".format(this_estimators))
	worker_type = 'E'
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
	max_depth = 100
	print("=============> I handle {} trees, my initial seed is {} <=============".format(this_estimators, seed_obj))

	# SKLearn KFoldWrapper ========================================================
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
		est_args = {'num_classes': num_classes, 'num_features': x_win_train_wi.shape[-1], 'regression': False,
		            'num_trees': this_estimators, 'max_leaf_nodes': 100000, 'valid_leaf_threshold': 10,
		            'base_random_seed': None}
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


if __name__ == '__main__':
	t_kfold_wrapper()
