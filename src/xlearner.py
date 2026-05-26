from econml.metalearners import XLearner
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
import numpy as np


class XlearnerWrapper(BaseEstimator):
    '''
    Wrapper for EconML Xlearner to make it compatible with sklearn API for HP tuning.

    Parameters
    ----------
    models : estimator or list of estimators
        Outcome estimators for control and treatment units.
    propensity_model : estimator
        Propensity score estimator.
    cate_models : estimator or list of estimators, optional (default=None)
        CATE estimators. If None, uses the same models as outcome estimators.
    '''
    def __init__(self, models, propensity_model, cate_models=None, **kwargs):
        self.models = models
        self.propensity_model = propensity_model
        self.cate_models = cate_models
        self._est = None

        # flatten nested params
        if models is not None and 'models__' in str(kwargs):
            models.set_params(**{k.replace('models__', ''): v for k, v in kwargs.items() if k.startswith('models__')})
        if propensity_model is not None and 'propensity_model__' in str(kwargs):
            propensity_model.set_params(**{k.replace('propensity_model__', ''): v for k, v in kwargs.items() if k.startswith('propensity_model__')})
        if cate_models is not None and 'cate_models__' in str(kwargs):
            cate_models.set_params(**{k.replace('cate_models__', ''): v for k, v in kwargs.items() if k.startswith('cate_models__')})


    def get_params(self, deep=True):
        return super().get_params(deep=deep)

    def fit(self, X, Y, **fit_params):
        W = fit_params.get("W", None)
        if W is None:
            raise ValueError("Treatment indicator W must be passed via fit parameter 'W'.")

        self._est = XLearner(
            models=self.models,
            propensity_model=self.propensity_model,
            cate_models=self.cate_models,
        )
        self._est.fit(Y, W, X=X)
        return self
    
    def predict(self, X):
        # CATE prediction, tau_hat
        return self._est.effect(X)
    
    def predict_outcome(self, X, W):
        # factual outcome prediction y_hat 
        X = np.asarray(X)
        W = np.asarray(W).reshape(-1)
        assert X.shape[0] == W.shape[0], "X and W must have same number of rows"

        y_hat = np.empty(W.shape[0], dtype=float)
        y_hat[W == 0] = self._est.models[0].predict(X[W == 0])
        y_hat[W == 1] = self._est.models[1].predict(X[W == 1])

        return y_hat
