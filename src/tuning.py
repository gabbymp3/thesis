from itertools import product
import numpy as np
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.base import clone
from skopt import Optimizer
from skopt.space import Real, Categorical, Integer
from src.metrics_helpers import outcome_mse
from src.xlearner import XlearnerWrapper
from sklearn.gaussian_process import GaussianProcessRegressor



def expand_param_grid(param_grid):
    """
    Generator that expands a parameter grid into individual parameter dictionaries.

    Parameters
    ----------
    param_grid : dict
        Dictionary mapping parameter names to lists of candidate values.

    Yields
    ------
    params : dict
        A dictionary representing one combination of parameters from the grid.

    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))


def sample_from_distribution(dist, rng):
    """
    Sample from:
    - list / tuple / np.ndarray
    - scipy distributions with rvs()
    - skopt spaces (Integer, Real, Categorical)
    """
    # List, tuple, or ndarray
    if isinstance(dist, (list, tuple, np.ndarray)):
        if isinstance(dist, np.ndarray) and dist.size == 1:
            return dist.item()
        return rng.choice(dist)

    # If distribution has rvs (scipy or skopt)
    if hasattr(dist, "rvs"):
        seed = rng.integers(0, 1_000_000_000)
        val = dist.rvs(random_state=seed)
        # If val is a list or ndarray
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            else:
                return rng.choice(val)
        elif isinstance(val, list):
            if len(val) == 1:
                return val[0]
            else:
                return rng.choice(val)
        else:
            return val

    # skopt Integer / Real
    if isinstance(dist, Integer):
        return rng.integers(dist.low, dist.high + 1)
    if isinstance(dist, Real):
        return rng.uniform(dist.low, dist.high)
    if isinstance(dist, Categorical):
        return rng.choice(dist.categories)

    # Fallback
    raise ValueError(f"Unsupported distribution type: {type(dist)}")


def evaluate_params_cv(
    estimator,
    params,
    X,
    Y,
    W,
    cv=5,
    random_state=123
):
    """
    Evaluate a parameter configuration using K-fold CV and factual outcome MSE.

    Returns
    -------
    float
        Mean cross-validated MSE. Returns np.inf if no valid folds exist.
    """

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    fold_scores = []

    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        W_tr, W_te = W[train_idx], W[test_idx]

        # Ensure both treatment arms are present
        if (
            np.sum(W_tr == 0) == 0 or np.sum(W_tr == 1) == 0 or
            np.sum(W_te == 0) == 0 or np.sum(W_te == 1) == 0
        ):
            continue

        est = clone(estimator)
        est.set_params(**params)
        est.fit(X_tr, Y_tr, W=W_tr)

        mse = outcome_mse(est, X_te, Y_te, W_te)
        if not np.isnan(mse):
            fold_scores.append(mse)

    if len(fold_scores) == 0:
        return np.inf

    return np.mean(fold_scores)


######################################################
#-------------------TUNING FUNCTIONS -----------------
######################################################


def grid_search(
        estimator, 
        param_grid, 
        X, 
        Y, 
        W, 
        cv=5, 
        random_state=123, 
        verbose=False, 
        n_jobs=-1):
    """
    Manual grid search cross-validation for XlearnerWrapper.

    Performs an exhaustive search over all parameter combinations specified in
    `param_grid`, evaluating each configuration using K-fold cross-validation
    and factual outcome mean squared error (MSE).

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Base estimator (e.g., XlearnerWrapper) to be tuned.
    param_grid : dict
        Dictionary mapping parameter names to lists of values to be exhaustively searched.
    X : array-like of shape (n_samples, n_features)
        Covariate matrix.
    Y : array-like of shape (n_samples,)
        Observed outcomes.
    W : array-like of shape (n_samples,)
        Binary treatment indicator.
    cv : int, default=5
        Number of cross-validation folds.
    random_state : int, default=123
        Random seed used for shuffling cross-validation splits.
    verbose : bool, default=False
        If True, prints parameter configurations and corresponding scores.
    n_jobs : int, default=-1
        Included for API consistency; not used in this manual implementation.

    Returns
    -------
    best_estimator : estimator
        Fitted estimator with the best-performing hyperparameters.
    best_params : dict
        Dictionary of hyperparameters that achieved the lowest average MSE.
    best_score : float
        Best (lowest) cross-validated outcome MSE.

    """

    best_score = np.inf
    best_params = None
    best_estimator = None
    print("----Grid search starting...")
    for params in expand_param_grid(param_grid):
        score = evaluate_params_cv(
            estimator, params, X, Y, W, cv=cv, random_state=random_state
        )

        if verbose:
            print(f"Params {params} | MSE = {score:.4f}")

        if score < best_score:
            best_score = score
            best_params = params
            best_estimator = clone(estimator).set_params(**params)
            best_estimator.fit(X, Y, W=W)
    print("--------> Grid search complete.")
    return best_estimator, best_params, best_score


def random_search(
        estimator,
        param_dist, 
        X, 
        Y, 
        W, 
        cv=5, 
        n_iter=20, 
        random_state=123, 
        verbose=False, 
        n_jobs=-1):
    """
    Manual random search cross-validation for XlearnerWrapper.

    Samples hyperparameter configurations at random from specified distributions
    and evaluates each configuration using K-fold cross-validation and oMSE.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Base estimator (e.g., XlearnerWrapper) to be tuned.
    param_dist : dict
        Dictionary mapping parameter names to distributions or lists from which
        values are randomly sampled.
    X : array-like of shape (n_samples, n_features)
        Covariate matrix.
    Y : array-like of shape (n_samples,)
        Observed outcomes.
    W : array-like of shape (n_samples,)
        Binary treatment indicator.
    cv : int, default=5
        Number of cross-validation folds.
    n_iter : int, default=20
        Number of random parameter configurations to evaluate.
    random_state : int, default=123
        Random seed for parameter sampling and cross-validation splits.
    verbose : bool, default=False
        If True, prints parameter configurations and corresponding scores.
    n_jobs : int, default=-1
        Included for API consistency; not used in this manual implementation.

    Returns
    -------
    best_estimator : estimator
        Fitted estimator with the best-performing hyperparameters.
    best_params : dict
        Dictionary of hyperparameters that achieved the lowest average MSE.
    best_score : float
        Best (lowest) cross-validated outcome MSE.

    """

    best_score = np.inf
    best_params = None
    best_estimator = None

    history = []

    rng = np.random.default_rng(random_state)

    print("----Random search starting...")
    for i in range(n_iter):
        params = {
            key: sample_from_distribution(dist, rng)
            for key, dist in param_dist.items()
        }

        score = evaluate_params_cv(
            estimator, params, X, Y, W, cv=cv, random_state=random_state
        )

        history.append({"params": params, "score": score})

        if verbose:
            print(f"Params {params} | MSE = {score:.4f}")

        if score < best_score:
            best_score = score
            best_params = params
            best_estimator = clone(estimator).set_params(**params)
            best_estimator.fit(X, Y, W=W)
    print("--------> Random search complete.")
    return best_estimator, best_params, best_score, history



def bayesian_search_old(
    estimator,
    param_dist,
    X,
    Y,
    W,
    cv=5,
    n_iter=20,
    random_state=123,
    verbose=False
):
    """
    Manual Bayesian optimization for XlearnerWrapper using Gaussian processes.

    Sequentially proposes hyperparameter configurations using a Gaussian process
    surrogate model and an expected improvement acquisition function. Each
    configuration is evaluated using K-fold cross-validation and oMSE.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        Base estimator (e.g., XlearnerWrapper) to be tuned.
    param_dist : dict
        Dictionary mapping parameter names to skopt.space objects
        (e.g., Real, Integer, Categorical).
    X : array-like of shape (n_samples, n_features)
        Covariate matrix.
    Y : array-like of shape (n_samples,)
        Observed outcomes.
    W : array-like of shape (n_samples,)
        Binary treatment indicator.
    cv : int, default=5
        Number of cross-validation folds.
    n_iter : int, default=25
        Number of Bayesian optimization iterations.
    random_state : int, default=123
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, prints parameter configurations and corresponding scores.

    Returns
    -------
    best_estimator : estimator
        Fitted estimator with the best-performing hyperparameters.
    best_params : dict
        Dictionary of hyperparameters that achieved the lowest average MSE.
    best_score : float
        Best (lowest) cross-validated outcome MSE.

    """

    param_names = list(param_dist.keys())
    dimensions = list(param_dist.values())

    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="EI",
        random_state=random_state
    )

    best_score = np.inf
    best_params = None
    best_estimator = None
    history = []

    print("---- Bayesian search starting...")
    for it in range(n_iter):
        x = optimizer.ask()
        params = dict(zip(param_names, x))

        score = evaluate_params_cv(
            estimator, params, X, Y, W, cv=cv, random_state=random_state
        )

        history.append({"params": params, "score": score})

        optimizer.tell(x, score)

        if verbose:
            print(f"[Iter {it+1}/{n_iter}] Params {params} | MSE = {score:.4f}")

        if score < best_score:
            best_score = score
            best_params = params
            best_estimator = clone(estimator).set_params(**params)
            best_estimator.fit(X, Y, W=W)
    print("--------> Bayesian search complete.")
    return best_estimator, best_params, best_score, history




def bayesian_search(
    estimator,
    param_dist,
    X,
    Y,
    W,
    n_iter,
    cv=5,
    random_state=123,
    verbose=False
):
    param_names = list(param_dist.keys())
    dimensions = list(param_dist.values())

    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="gp_hedge",
        n_initial_points=10,
        initial_point_generator="sobol",
        random_state=random_state
    )

    best_score = np.inf
    best_params = None
    best_estimator = None
    history = []
    print("---- Bayesian search starting...")
    for it in range(n_iter):

        x = optimizer.ask()
        params = dict(zip(param_names, x))

        score = evaluate_params_cv(
            estimator, params, X, Y, W, cv=cv, random_state=random_state
        )

        optimizer.tell(x, score)

        # surrogate diagnostics
        surrogate = optimizer.base_estimator_

        if surrogate is not None and hasattr(surrogate, "predict"):

            if len(optimizer.Xi) > 1:
                X_obs = np.array(optimizer.Xi)
                y_obs = np.array(optimizer.yi)

                gp = GaussianProcessRegressor().fit(X_obs, y_obs)
                mu, std = gp.predict(np.array(x).reshape(1, -1), return_std=True)

            history.append({
                "iter": it,
                "params": params,
                "score": score,
                "pred_mu": float(mu) if len(optimizer.Xi) > 1 else None,
                "pred_std": float(std) if len(optimizer.Xi) > 1 else None,
            })
        else:
            history.append({
                "iter": it,
                "params": params,
                "score": score
            })

        if score < best_score:
            best_score = score
            best_params = params
            best_estimator = clone(estimator).set_params(**params)
            best_estimator.fit(X, Y, W=W)
    print("--------> Bayesian search complete.")
    return best_estimator, best_params, best_score, history
