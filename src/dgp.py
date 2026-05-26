import numpy as np
from scipy.stats import random_correlation
from numpy.random import default_rng

class SimulatedDataset:

    """
    Simulated dataset generated using a similar procedure as in Künzel et al. (2019)
    with covariates and treatment effect heterogeneity; with the addition
    of specified confounding, prognostic, and effect modifier covariates.

    Parameters
    ----------
    N : int
        Number of observations to simulate.
    d : int
        Number of covariates (features). Must be at least 10 to respect the
        fixed index sets for confounders, prognostic variables, and effect
        modifiers.
    alpha : float
        Confounding strength parameter entering the propensity score logits.
        alpha = 0 corresponds to completely randomized treatment; increasing
        |alpha| increases confounding.
    seed : int, optional
        Random seed used to initialize the internal random number generator
        for reproducible simulation.

    This class generates:
        - X  : (N, d) feature matrix with a random correlation structure,
        - W  : (N,) binary treatment assignments,
        - Y  : (N,) observed outcomes,
        - Y0 : (N,) potential outcomes under control,
        - Y1 : (N,) potential outcomes under treatment,
        - mu0: (N,) conditional mean outcome under control E[Y | W=0, X],
        - mu1: (N,) conditional mean outcome under treatment E[Y | W=1, X],
        - tau: (N,) individual treatment effects (CATE) tau(X),
        - e  : (N,) propensity scores P(W=1 | X).

    Covariates are partitioned into:
        - Confounders      (indices [0, 1, 2, 3]): affect both treatment and outcome,
        - Prognostic vars  (indices [4, 5, 6]): affect outcome only,
        - Effect modifiers (indices [7, 8, 9]): drive treatment effect heterogeneity.

    Outcome model:
        - mu0(X) is a nonlinear function of confounders and prognostic covariates.
        - tau(X) is a nonlinear function of effect modifiers via rho(·).
        - mu1(X) = mu0(X) + tau(X).
        - Potential outcomes are Y0 = mu0 + ε, Y1 = mu1 + ε with ε ~ N(0, 1).

    Treatment assignment:
        - Propensity scores e(X) = P(W=1 | X) are a logistic function of confounders.
        - `alpha` controls the strength of confounding:
            * alpha = 0  → totally random treatment assignment,
            * larger |alpha| → stronger observed confounding.

    """

    def __init__(self, N,  d, alpha, seed=123):
        self.rng = default_rng(seed)
        self.N = N
        self.d = d
        self.alpha = alpha

        self.X = self.generate_features()
        self.x_confounders_idx, self.x_prognostic_idx,  self.x_effect_mod_idx = self.set_covariate_types()

        self.mu0 = self.set_mu0()
        self.tau = self.set_tau()
        self.mu1 = self.set_mu1()
        self.e = self.set_propensity_fn()
        self.Y0, self.Y1 = self.generate_potential_outcomes()

        self.W = self.generate_treatment()
        self.Y = self.generate_outcome()
        
    # --------------------------
    # CORRELATION STRUCTURE

    def generate_random_eigenvector(self):
        eigvals = self.rng.dirichlet(np.ones(self.d)) * self.d
        return eigvals

    def generate_random_correlation_matrix(self):
        eigvals = self.generate_random_eigenvector()
        corr_matrix = random_correlation.rvs(eigvals, random_state=self.rng)
        return corr_matrix

    def generate_features(self):
        corr_matrix = self.generate_random_correlation_matrix()
        X = self.rng.multivariate_normal(mean=np.zeros(self.d), cov=corr_matrix, size=self.N)
        return X
    
    def set_covariate_types(self):
        x_confounders_idx = [0, 1, 2, 3]
        x_prognostic_idx = [4, 5, 6]
        x_effect_mod_idx = [7, 8, 9]
        return x_confounders_idx, x_prognostic_idx, x_effect_mod_idx

    # --------------------------
    # STRUCTURAL FUNCTIONS

    def rho(self, x):
        # heterogeneity function
        return 2 / (1 + np.exp(-5 * (x - 0.35)))

    def set_mu0(self):
        # function of confounding and prognostic covariates
        Xc = self.X[:, self.x_confounders_idx]
        Xp = self.X[:, self.x_prognostic_idx]

        mu0 = (
            0.6 * Xc[:, 0]
            - 0.3 * Xc[:, 1]
            + 0.2 * Xc[:, 2]
            - 0.1 * Xc[:, 3]**2
            + 0.5 * Xp[:, 0]
            - 0.2 * Xp[:, 1]
            + 0.1 * Xp[:, 2]**2
        )
        return mu0
    
    def set_tau(self):
        # function of effect modifiers
        Xem = self.X[:, self.x_effect_mod_idx]
        tau = (
            0.8 * self.rho(Xem[:, 0])
            - 0.4 * self.rho(Xem[:, 1])
            + 0.2 * self.rho(Xem[:, 2]**2)
        )
        return tau
        

    def set_mu1(self):
        # mu1 = mu0 + tau
        return self.mu0 + self.set_tau()

    def generate_potential_outcomes(self):
        eps = self.rng.normal(0, 1, size=self.N)
        Y0 = self.mu0 + eps
        Y1 = self.mu1 + eps
        return Y0, Y1

    def set_propensity_fn(self):
        '''
        Propensity score function determining probability of positive treatment
            e(x) = P(W=1|X=x).
        '''
        Xc = self.X[:, self.x_confounders_idx]
        logits = (
            0.6 * Xc[:, 0]
            - 0.4 * Xc[:, 1]
            + 0.2 * Xc[:, 2]
            - 0.1 * Xc[:, 3]**2
        )
        return 1 / (1 + np.exp(-logits*self.alpha + 0.8))

    # -------------------------
    # OUTCOME AND TREATMENT

    def generate_treatment(self):
        return self.rng.binomial(1, self.e)

    def generate_outcome(self):
        return np.where(self.W == 1, self.Y1, self.Y0)


def simulate_dataset(dgp_params, seed=0):
    dgp = SimulatedDataset(
        N=dgp_params.get("N", 50),
        d=dgp_params.get("d", 5),
        alpha=dgp_params.get("alpha", 0.5),
        seed=seed
    )
    return dgp.X, dgp.W, dgp.Y, dgp.mu0, dgp.mu1, dgp.Y0, dgp.Y1, dgp.tau, dgp.e
