from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from skopt.space import Integer, Real
from src.tuning import random_search, bayesian_search


def get_config():
    """
    Docstring for get_config
    """

    R = 30
    dgp_params = {"N": 1000, "d": 15, "alpha": 0.5}

    learners = [
        {
            "name": "x_rf",
            "models": RandomForestRegressor(random_state=0),
            "propensity_model": LogisticRegression(),
            "tuners": [
                {
                    "name": "random",
                    "fn": random_search,
                    "param_dist": {
                        "models__max_depth": Integer(3, 10),
                        "models__min_samples_leaf": Integer(1, 20),
                        "models__min_samples_split": Integer(2, 20),
                        "models__max_features": Real(0.1, 1.0),
                        "models__max_leaf_nodes": Integer(2, 50),
                        "models__bootstrap": [True, False]
                    },
                    "kwargs": {"cv": 3, "n_iter": 100}
                },
                {
                    "name": "bayes",
                    "fn": bayesian_search,
                    "param_dist": {
                        "models__max_depth": Integer(3, 10),
                        "models__min_samples_leaf": Integer(1, 20),
                        "models__min_samples_split": Integer(2, 20),
                        "models__max_features": Real(0.1, 1.0),
                        "models__max_leaf_nodes": Integer(2, 50),
                        "models__bootstrap": [True, False]
                    },
                    "kwargs": {"cv": 3, "n_iter": 100}
                },
            ]
        },
        {
            "name": "x_cb",
            "models": CatBoostRegressor(verbose=0, random_state=0),
            "propensity_model": LogisticRegression(),
            "tuners": [
                {
                    "name": "random",
                    "fn": random_search,
                    "param_dist": {
                        "models__learning_rate": Real(0.01, 0.15),
                        "models__depth": Integer(3, 10),
                        "models__iterations": Integer(50, 300),
                        "models__l2_leaf_reg": Integer(1, 10),
                        "models__random_strength": Real(0.01, 10),
                        "models__subsample": Real(0.4, 1.0)
                    },
                    "kwargs": {"cv": 3, "n_iter": 100}
                },
                {
                    "name": "bayes",
                    "fn": bayesian_search,
                    "param_dist": {
                        "models__learning_rate": Real(0.01, 0.15),
                        "models__depth": Integer(3, 10),
                        "models__iterations": Integer(50, 300),
                        "models__l2_leaf_reg": Integer(1, 10),
                        "models__random_strength": Real(0.01, 10),
                        "models__subsample": Real(0.4, 1.0)
                    },
                    "kwargs": {"cv": 3, "n_iter": 100}
                },
            ]
        }
    ]

    return {
        "name": "6d",
        "R": R,
        "dgp_params": dgp_params,
        "learners": learners,
        "base_seed": 64
    }