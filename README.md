# mlfdemo

Teaching helpers for **Foundations of Machine Learning (Demo)** (E1234). Deliberately small,
pure-Python and dependency-free, so it installs anywhere and can be read in an afternoon.

```bash
pip install -e .            # from this directory
pip install -e ".[plots]"   # also install matplotlib, for mlfdemo.plotting
```

```python
from mlfdemo import datasets, linreg

X, y = datasets.housing()
model = linreg.fit(X, y)
print(linreg.summary(model, X, y))
```

## Phased release

Modules are released to the cohort as the term reaches them, so a partial checkout is
normal - which is why `mlfdemo/__init__.py` imports no submodule eagerly.

| Module | Released | Contents |
|---|---|---|
| `datasets` | session 1 | the twelve-flat housing extract and a synthetic classification set |
| `linreg` | session 2 | least squares by the normal equations, plus `summary` |
| `plotting` | session 3 | loss curves and residual plots (needs matplotlib) |

Everything here is written for legibility, not speed. Use NumPy and scikit-learn for real
work; use this to see what they are doing.
