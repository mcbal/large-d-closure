# Blog post code

Code for the numerical checks in the blog post *A Large-D Closure for Asymmetric Kinetic Vector-Spin Systems*. The only dependencies are NumPy and Matplotlib.

| File | What it does | Outputs |
|---|---|---|
| `core.py` | Shared library: mean-field, closure and sampled-closure recursions, exact vMF sampling (`simulate`, `stationary`), estimators, `error` / `error_interval` | — |
| `ring.py` | High-dimensional vMF geometry: the ring picture of a vMF spin | Fig. 1: `ring.{png,svg}` |
| `convergence.py` | Convergence in D, including the squared field-norm histogram | Figs. 2–3: `convergence.{png,svg,json}`, `fieldnorm.{png,svg}` |
| `mechanisms.py` | Norm inflation and shared inputs | Fig. 4: `mechanisms.{png,svg,json}` |
| `relaxation.py` | The stationary state: relaxation, and the stationary $(u,\beta)$ sweep behind the table and the low-temperature checks | Fig. 5: `relaxation.{png,svg,json}` |
| `sketch.py` | Carrying noise instead of a matrix (K > 1): the sampled closure | Fig. 6: `sketch.{png,svg,json}` |
| `subspace.py` | Supplementary, not in the post: fluctuations inside a signal subspace | `subspace.{png,svg,json}` |
| `checks/check_estimators.py` | Bias and interval coverage of the reference estimators on a tiny shared-input system, against the earlier `checks/core_before_unbiased.py` | printed table |

The `.json` files hold the results behind the post's figures and tables.

## Running

From this folder:

```bash
uv run --with numpy,matplotlib python ring.py
uv run --with numpy,matplotlib python convergence.py
uv run --with numpy,matplotlib python mechanisms.py
uv run --with numpy,matplotlib python relaxation.py
uv run --with numpy,matplotlib python sketch.py
uv run --with numpy,matplotlib python subspace.py
uv run --with numpy python checks/check_estimators.py
```

Each figure script writes its results to `<name>.json` next to the script, overwriting the included one, and its figure to `.png` and `.svg`. `--plot` redraws the figure from the JSON without resampling, so the post's figures can be reproduced in seconds:

```bash
uv run --with numpy,matplotlib python convergence.py --plot
```

Some scripts can rerun just one part:

```bash
uv run --with numpy,matplotlib python mechanisms.py --concentration   # only panel (a)
uv run --with numpy,matplotlib python sketch.py --physical            # only the closure-vs-exact scale lines
```

On a 4-core / 8-thread CPU:

| Script | Approximate time |
|---|---|
| `mechanisms.py` | 2 min |
| `subspace.py` | 3 min |
| `convergence.py` | 6 min |
| `relaxation.py` | 10 min |
| `sketch.py --physical` | 5 min |

Exact sampling runs in 4 worker processes (`workers=4` in `core.simulate` and `core.stationary`). The workers are started with `spawn`, so any script that calls these functions needs an `if __name__ == "__main__":` guard.
