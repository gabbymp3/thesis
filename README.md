# Bayesian Hyperparameter Tuning in Meta-Learners: Predictive vs. Causal Gains

This repository contains the implementation of my senior thesis for MMSS at Northwestern University. The objective of this study is to compare Bayesian and automatic hyperparameter tuning methods for X-learner models (from `EconML`) in the CATE setting, asking whether predictive gains in the tuning stage can improve end-goal causal estimation.

## Objective

This study seeks to address the issue of hyperparameter tuning in the development of causal ML models, asking whether a Bayesian approach to hyperparameter tuning can deliver better performance than standard frequentist-style automatic tuning methods. The analysis focuses on one Causal ML method, the X-learner, for estimating conditional average treatment effects to evaluate the performance of such tuning strategies. This thesis distinguishes between two sequential objectives in the causal ML model pipeline. The first is the tuning stage in which hyperparameters are selected to minimize cross-validated predictive error. The second is the evaluation stage in which the models are scored based on the accuracy of treatment effect estimation. It is important to note that the second stage is not a predictive task, but a causal one. Thus, the central question of this study is whether any predictive gains in Bayesian tuning during the first stage actually translate to concrete causal gains in the model evaluation stage.

## Repository overview

All main scripts are stored in the `src` module, with corresponding pytest modules that can be found in the `tests` folder.

```text
bayesian-tuning-metalearners/
├── src/
│   ├── dgp.py
│   └── xlearner.py
│   └── metrics_helpers.py
│   └── tuning.py
│   └── convergence.py
│   └── experiment.py
│   └── main.py
│   └── experiment_configs/
│       ├── config_1d.py
│       └── config_2d.py
│       └── config_4d.py
│       └── config_6d.py
├── pyproject.toml
└── README.md
```


`src/`
  `dgp.py` contains the `SimulatedDataset` class and the data generating function `simulate_dataset()` used in these simulations. The DGP is based on the procedure in Künzel et al. (2019) with the following modifications:

  - Confounding, prognostic, and effect modifier covariates are specified.
  - Correlation matrix is constructed from a randomly generated eigenvector.
  - Individual response functions (and thus the true treatment effect) and propensity score function are original.
  - Observed outcomes `Y0`, `Y1` constructed from their respective response functions `mu0`, `mu1` but share the same normally-distributed error term to control noise.

  `xlearner.py` contains the implentation of the X-learner model used in this study, the `XLearnerWrapper` class. This object inherits `BaseEstimator` from `sklearn.base` and wraps the XLearner object from `EconML`'s `metalearners` library.

  `metrics_helpers.py` contains the helper functions used in cross-validation and model evaluation, calculating observed outcome MSE, PEHE/PEHEplug, and TAUplug.

  `tuning.py` contains all tuning implementations, `grid_search()`, `random_search()`, and `bayesian_search()`. All tuning functions use the same internal cross-validation process, differing only in their search algorithms. Each returns the fitted model, parameters, best score achieved after tuning, and a history of scores and parameters at each iteration.

  `convergence.py` contains the `ConvergenceTracker` class, which is used to track the convergence of the tuning process. It stores the best score and parameters at each iteration, and can be used to generate convergence plots.

  `experiment.py` implements one Monte-Carlo simulation using R repetitions. The experimental workflow is as follows: 
  - Given a data-generating function, base learner configuration, tuner configuration, and R value:
    - For each Monte-Carlo repetition 1 through R, simulate a training and test dataset using the same DGP parameters `dgp_params` and a different random seed. Then construct an `Xlearner` with parameters given by the base learner setup, `learner_config`.
      - For each tuner configuration in `tuners`, tune an XLearner model on the training data. Then, use the test data to estimate CATE `tau_hat` and cross-predict `tau_plug`, and calculate PEHE and PEHE plugin values.
      - Store the learner-tuner combination and its resulting PEHE and PEHE plugin values in `raw_results`, and generate a summary table containing the mean and variance of both PEHE metrics across Monte-Carlo simulations.

  `main.py` conducts the entire pipeline, running all specified experiments, and storing their results in an output directory with the following format:

  ```text
bayesian-tuning-metalearners/
├── results_R_{R value}/
│   ├── x_cb/
│       ├── 1d/
│         └── raw_results.csv
│         └── summary.csv
│         └── convergence/
│             ├── random/
│             │   └── convergence_R{r}.csv
|                   ...
│             └── bayes/
│                 └── convergence_R{r}.csv
|                   ...
│       ├── 2d/ ...
│       ├── 4d/ ...
│       ├── 6d/ ...
│   └── x_rf/
│       ├── 1d/ ...
│       ├── 2d/ ...
│       ├── 4d/ ...
|       ├── 6d/ ...
```


## Replication

This module requires python version 3.10 or above and uses `poetry` for dependency management. To run the experiments on your local, do the following:

After cloning the repo, activate the virtual environment using:
```
source .venv/bin/activate
```
Next install dependencies:
```
poetry install
```
To run the experiment pipeline, run:
```
python -m src.main
```

Experiment configurations can be altered. To change these configurations, edit or create a new configuration file in `src/experiment_configs` and update the filepath in the first section of `main.py`. Happy experimentation :) 

