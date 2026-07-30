"""Two plots worth having, and nothing else.

matplotlib is imported INSIDE each function, so `import mlfdemo.plotting` succeeds in an
environment that does not have it - the failure then happens where it can be explained,
rather than at import time. Install with `pip install -e ".[plots]"`.
"""

from __future__ import annotations

from .linreg import predict


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "mlfdemo.plotting needs matplotlib: pip install -e '.[plots]'"
        ) from exc
    return plt


def loss_curve(history, ax=None, label: str | None = None, log: bool = True):
    """Plot loss against iteration - the plot session 3 says you must always look at.

    Args:
        history: the per-iteration loss values.
        ax: an existing axis to draw on (for comparing learning rates on one figure).
        label: legend entry, e.g. "alpha = 0.1".
        log: log-scale the y axis, which is almost always what you want.
    """
    plt = _pyplot()
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(len(history)), history, label=label)
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss" + (" (log scale)" if log else ""))
    if label:
        ax.legend()
    return ax


def residual_plot(beta, X, y, feature: int | None = None, ax=None):
    """Residuals against fitted values, or against one feature.

    Structure here means the functional form is wrong, not that the fit failed - the
    residuals are orthogonal to every feature by construction (session 2).
    """
    plt = _pyplot()
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 3.5))
    fitted = predict(X, beta)
    resid = [yi - yh for yi, yh in zip(y, fitted)]
    if feature is None:
        ax.scatter(fitted, resid)
        ax.set_xlabel("fitted value")
    else:
        ax.scatter([row[feature] for row in X], resid)
        ax.set_xlabel(f"feature {feature}")
    ax.axhline(0.0, ls="--", color="grey")
    ax.set_ylabel("residual")
    return ax
