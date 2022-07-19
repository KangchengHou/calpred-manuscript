import pandas as pd
import numpy as np


def simulate_het_linear(
    X: np.ndarray, Z: np.ndarray, beta: np.ndarray, gamma: np.ndarray
):
    """
    Simulate a linear model with heteroskedasticity.

    Parameters
    ----------
    X : np.ndarray
        design matrix for mean effects
    Z : np.ndarray
        design matrix for variance effects
    beta : np.ndarray
        mean effect coefficients
    gamma : np.ndarray
        variance effect coefficients
    """
    assert X.ndim == 2
    assert Z.ndim == 2
    assert beta.ndim == 1
    assert gamma.ndim == 1
    assert X.shape[0] == Z.shape[0]
    assert X.shape[1] == beta.shape[0]
    assert Z.shape[1] == gamma.shape[0]

    return np.random.normal(loc=X @ beta, scale=np.sqrt(np.exp(Z @ gamma)))