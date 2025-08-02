import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import l0learn


def coordinate_descent_l0(X, y, lambda_, moral=None, max_iter=1000, tol=1e-6):
    # Initialization
    n, p = X.shape
    if moral is not None:
        if len(moral) == 0:
            return np.zeros(p)
        X = X[:, moral]
        n, p = X.shape

    beta = np.zeros(p)  # Initialize beta to be a zero vector

    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(p):
            # Compute the partial residual
            r = y - X @ beta + X[:, j] * beta[j]

            # Compute the univariate regression coefficient
            beta_j_new = X[:, j].T @ r / (X[:, j].T @ X[:, j])

            # Apply l0 thresholding
            threshold = np.sqrt(2 * lambda_ / (X[:, j].T @ X[:, j]))
            beta[j] = 0 if abs(beta_j_new) < threshold else beta_j_new

        # Check for convergence
        if np.linalg.norm(beta - beta_old, ord=2) < tol:
            break

    return beta


def regression(X, ordering, moral, lam):
    """
    Perform sparse regression on a data matrix with a specified variable order.

    Args:
        X (np.ndarray): Data matrix of shape (n_samples, n_features).
        ordering (list): Column indices defining the causal order (e.g., [2,0,1]).

    Returns:
        np.ndarray: Coefficient matrix B of shape (n_features, n_features), where B[i,j] is the
                    coefficient of variable i on variable j (after reordering).
    """
    # Reorder columns according to the specified ordering
    if type(X) == pd.DataFrame:
        X = X.values
    original_order = np.argsort(ordering)
    X_ordered = X[:, ordering]
    m = X_ordered.shape[1]
    B = np.zeros((m, m))

    for i in range(1, m):
        moral_i = np.where(moral[:i, i] != 0)[0]
        # Parents are all columns before i in the ordered matrix
        X_parents = X_ordered[:, moral_i]
        # Target is the i-th column in the ordered matrix
        X_target = X_ordered[:, i]

        # Fit Lasso regression
        estimated_Bi = coordinate_descent_l0(X_parents, X_target, lam, moral=moral_i)

        # Store coefficients in the i-th col, first i rows
        if len(moral_i) > 0:
            B[moral_i, i] = estimated_Bi
    B = B[original_order, :][:, original_order]

    return B

def regression_l0learn(X, ordering, moral):
    if type(X) == pd.DataFrame:
        X = X.values
    original_order = np.argsort(ordering)
    X_ordered = X[:, ordering]
    m = X_ordered.shape[1]
    B = np.zeros((m, m))
    moral = moral[ordering, :][:, ordering]

    for i in range(1, m):
        moral_i = np.where(moral[:i, i] != 0)[0]
        # Parents are all columns before i in the ordered matrix
        # X_parents = X_ordered[:, :i]
        X_parents = X_ordered[:, moral_i]
        # Target is the i-th column in the ordered matrix
        X_target = X_ordered[:, i]

        # Fit Lasso regression
        # Store coefficients in the i-th col, first i rows
        if len(moral_i) > 1:
            # fitted_model = l0learn.fit(X_parents, X_target, max_support_size=len(moral_i))
            fitted_model = l0learn.cvfit(X_parents, X_target)
            # v = 1/np.std(X_target - X_parents@fitted_model.coeffs[0][:,-1].A)
            # fitted_model = l0learn.cvfit(X_parents*v, X_target)
            B[moral_i, i] = fitted_model.coeffs[0][:, -1].A.T
            # B[moral_i, i] = fitted_model.coeffs[0][moral_i, -1].A.T
    B = B[original_order, :][:, original_order]

    return B