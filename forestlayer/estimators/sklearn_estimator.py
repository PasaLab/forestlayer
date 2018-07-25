# -*- coding:utf-8 -*-
"""
Scikit-learn Estimators Definition.

If you want to add your new sklearn-based estimators, take regression task as an example,
you must define `__init__` function like follows:
```
class YOURRegressor(xSKLearnBaseEstimator):
    def __init__(self, name, kwargs):
        from xx import TrueRegressor
        super(YOURRegressor, self).__init__('regression', TrueRegressor, name, kwargs)
```
"""

from base_estimator import BaseEstimator
from sklearn.externals import joblib
import psutil


def forest_predict_batch_size(clf, X, task):
    """
    Decide predict batch size by calculating memory occupation.

    :param clf: classifier
    :param X: training data
    :param task: learning task
    :return: batch_size
    """
    # TODO: Different cluster need different batch size determination strategy.
    free_memory = psutil.virtual_memory().total - psutil.virtual_memory().used
    # LOGGER.debug('free_memory: {}'.format(free_memory))
    if free_memory < 2e9:
        free_memory = int(2e9)
    # max_mem_size = max(half of free memory, 10GB)
    max_mem_size = max(int(free_memory * 0.7), int(1e10))
    # LOGGER.debug('max_mem_size: {}'.format(max_mem_size))
    if task == 'regression':
        mem_size_1 = clf.n_estimators * 16
    else:
        # internally, estimator's predict_proba will convert probas to float64, thus 8 bytes, see forest.py: 584
        # set to 10, in order to not overflow free memory
        mem_size_1 = clf.n_classes_ * clf.n_estimators * 16
    batch_size = (max_mem_size - 1) / mem_size_1 + 1
    if batch_size < 10:
        batch_size = 10
    if batch_size >= X.shape[0]:
        return 0
    return batch_size


class SKLearnBaseEstimator(BaseEstimator):
    """
    SKlearn base estimators inherited from BaseEstimator.
    """
    def __init__(self, task=None, est_class=None, name=None, est_args=None):
        super(SKLearnBaseEstimator, self).__init__(task, est_class, name, est_args)

    def _save_model_to_disk(self, est, cache_path):
        """
        Save model to disk using joblib.

        :param est: estimator
        :param cache_path: cache path
        :return: None
        """
        joblib.dump(est, cache_path, compress=True)

    def _load_model_from_disk(self, cache_path):
        """
        Load model from disk using joblib.

        :param cache_path: cache path
        :return:
        """
        return joblib.load(cache_path)

    def copy(self):
        """
        copy

        :return:
        """
        return SKLearnBaseEstimator(est_class=self.est_class, **self.est_args)


"""
==============================================================
scikit-learn based Classifier definition
==============================================================
"""


class FLRFClassifier(SKLearnBaseEstimator):
    """
    Random Forest Classifier
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestClassifier
        super(FLRFClassifier, self).__init__('classification', RandomForestClassifier, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='classification'):
        return forest_predict_batch_size(est, X, task)


class FLCRFClassifier(SKLearnBaseEstimator):
    """
    Completely Random Forest Classifier
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesClassifier
        super(FLCRFClassifier, self).__init__('classification', ExtraTreesClassifier, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='classification'):
        return forest_predict_batch_size(est, X,  task)


class FLGBDTClassifier(SKLearnBaseEstimator):
    """
    Gradient Boosting Decision Tree Classifier.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import GradientBoostingClassifier
        super(FLGBDTClassifier, self).__init__('classification', GradientBoostingClassifier, name, kwargs)


class FLXGBoostClassifier(SKLearnBaseEstimator):
    """
    XGBoost Classifier using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from xgboost.sklearn import XGBClassifier
        super(FLXGBoostClassifier, self).__init__('classification', XGBClassifier, name, kwargs)


class FLLGBMClassifier(SKLearnBaseEstimator):
    """
    LightGBM Classifier using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from lightgbm.sklearn import LGBMClassifier
        super(FLLGBMClassifier, self).__init__('classifier', LGBMClassifier, name, kwargs)


"""
==============================================================
scikit-learn based Regressor definition
==============================================================
"""


class FLRFRegressor(SKLearnBaseEstimator):
    """
    Random Forest Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import RandomForestRegressor
        super(FLRFRegressor, self).__init__('regression', RandomForestRegressor, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='regression'):
        return forest_predict_batch_size(est, X, task)


class FLCRFRegressor(SKLearnBaseEstimator):
    """
    Completely Random Forest Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import ExtraTreesRegressor
        super(FLCRFRegressor, self).__init__('regression', ExtraTreesRegressor, name, kwargs)

    def _default_predict_batch_size(self, est, X, task='regression'):
        return forest_predict_batch_size(est, X, task)


class FLGBDTRegressor(SKLearnBaseEstimator):
    """
    Gradient Boosting Decision Tree Regressor.
    """
    def __init__(self, name, kwargs):
        from sklearn.ensemble import GradientBoostingRegressor
        super(FLGBDTRegressor, self).__init__('regression', GradientBoostingRegressor, name, kwargs)


class FLXGBoostRegressor(SKLearnBaseEstimator):
    """
    XGBoost Regressor using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from xgboost.sklearn import XGBRegressor
        super(FLXGBoostRegressor, self).__init__('regression', XGBRegressor, name, kwargs)


class FLLGBMRegressor(SKLearnBaseEstimator):
    """
    LightGBM Regressor using Sklearn interfaces.
    """
    def __init__(self, name, kwargs):
        from lightgbm.sklearn import LGBMRegressor
        super(FLLGBMRegressor, self).__init__('regression', LGBMRegressor, name, kwargs)











