"""Least squares by the normal equations - the session 2 derivation, in code.

Deliberately the same interface as Assignment 1, so you can check your own implementation
against this one after the deadline (and not before).
"""

from __future__ import annotations


def solve(A, b):
    """Solve `A z = b` by Gaussian elimination with partial pivoting."""
    n = len(A)
    M = [list(map(float, row)) + [float(rhs)] for row, rhs in zip(A, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[pivot][col]) < 1e-12:
            raise ValueError("singular system - are two columns collinear?")
        M[col], M[pivot] = M[pivot], M[col]
        for row in range(col + 1, n):
            factor = M[row][col] / M[col][col]
            for k in range(col, n + 1):
                M[row][k] -= factor * M[col][k]
    z = [0.0] * n
    for row in reversed(range(n)):
        z[row] = (M[row][n] - sum(M[row][k] * z[k] for k in range(row + 1, n))) / M[row][row]
    return z


def fit(X, y, ridge: float = 0.0):
    """Fit `y = b0 + X b`, returning coefficients with the intercept first.

    Args:
        X: list of rows, no column of ones (this function adds it).
        y: list of targets.
        ridge: L2 penalty (session 5). The intercept is never penalised.

    Returns:
        A list of len(X[0]) + 1 coefficients.
    """
    design = [[1.0] + [float(v) for v in row] for row in X]
    p = len(design[0])
    xtx = [[sum(r[i] * r[j] for r in design) for j in range(p)] for i in range(p)]
    xty = [sum(r[i] * yi for r, yi in zip(design, y)) for i in range(p)]
    for i in range(1, p):                      # skip [0][0]: never penalise the intercept
        xtx[i][i] += ridge
    return solve(xtx, xty)


def predict(X, beta):
    """Fitted values for each row of `X`."""
    return [beta[0] + sum(b * v for b, v in zip(beta[1:], row)) for row in X]


def r_squared(y_true, y_pred):
    """1 - RSS/TSS. Negative when the fit is worse than the mean of `y_true`."""
    mean_y = sum(y_true) / len(y_true)
    rss = sum((a - p) ** 2 for a, p in zip(y_true, y_pred))
    tss = sum((a - mean_y) ** 2 for a in y_true)
    if tss == 0.0:
        raise ValueError("y_true has zero variance - R-squared is undefined")
    return 1.0 - rss / tss


def rmse(y_true, y_pred):
    """Root mean squared error, in the units of `y`. Report this to a reader."""
    return (sum((a - p) ** 2 for a, p in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def summary(beta, X, y, names=None) -> str:
    """A short, printable fit summary - coefficients, RMSE, R-squared."""
    names = list(names or [f"x{i + 1}" for i in range(len(beta) - 1)])
    y_hat = predict(X, beta)
    lines = [f"n = {len(y)}, p = {len(beta) - 1}",
             f"  intercept      {beta[0]:12.4f}"]
    lines += [f"  {name:<14} {value:12.4f}" for name, value in zip(names, beta[1:])]
    lines += [f"  RMSE           {rmse(y, y_hat):12.4f}",
              f"  R-squared      {r_squared(y, y_hat):12.4f}"]
    return "\n".join(lines)
