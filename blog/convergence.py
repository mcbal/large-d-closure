"""Convergence in D at fixed couplings (N = 256, beta = 2, u = 1, sharpness 3, K = 4).

    python convergence.py           # runs the sweep, writes convergence.json and figures
    python convergence.py --plot    # redraws from convergence.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import core as C

HERE = Path(__file__).resolve().parent
DIMS = [16, 32, 64, 128, 256, 512, 1024]
SETUP = dict(N=256, beta=2.0, u=1.0, sharpness=3.0, K=4, k=16, seed=0)
NAMES = ["mean field", "diagonal C", "closure", "sampled"]


def run(D, N, beta, u, sharpness, K, k, seed):
    chains = 16000 if D < 512 else 8000
    x = C.fields(N, D, u, beta, [seed, D, 0])
    f = C.fields(N, D, 1.0, beta, [seed, D, 1])  # independent initializer fields
    J = C.softmax_couplings(N, sharpness, seed)
    M0 = C.exact_mean(f, beta)
    ne = C.n_eff(J)
    track = [int(np.argmin(ne)), int(np.argsort(ne)[N // 2])]  # most concentrated and a median site
    exact = C.simulate(x, f, J, beta, K, chains, seed=[seed, D, 2], track=track)
    models = {
        "mean field": C.mean_field(x, J, f, beta, K),
        "diagonal C": C.closure(x, J, M0, beta, K, diagonal=True),
        "closure": C.closure(x, J, M0, beta, K),
        "sampled": C.sampled_closure(x, J, M0, beta, K, k, [seed, D, 3]),
    }
    R2 = C.order(D)
    rows = []
    for step in range(1, K + 1):
        ex = exact[step]
        reps = C.observables(ex["m"], ex["C"], ex["D"], x, gram=ex["gram"])
        row = {"D": D, "step": step}
        for obs, rep in reps.items():
            row[f"floor {obs}"] = C.floor(rep)
            for name, res in models.items():
                pred = C.observables(res["m"][step - 1], res["C"][step - 1], res["D"][step - 1], x)
                row[f"{name} {obs}"], row[f"{name} {obs} 2se"] = C.error_interval(pred[obs], rep)
        # Field norms: sampled mean and spread, against mean field |hbar|^2 and the closure's |hbar|^2 + v.
        M_prev = M0 if step == 1 else models["closure"]["m"][step - 2]
        hbar = x + J @ M_prev
        v = models["closure"]["v"][step - 1]
        row["field"] = {"sites": track, "mean field": ((hbar * hbar).sum(1)[track] / R2).tolist(),
                        "closure": (((hbar * hbar).sum(1) + v)[track] / R2).tolist(),
                        "exact mean": ex["h2"][track].tolist(), "exact std": np.sqrt(ex["h2var"][track]).tolist(),
                        "samples": ex["track"][:, :4000].tolist() if step in (1, K) else None}
        row["v/R2 mean"] = float(v.mean() / R2)
        row["noise/meanfield"] = float(v.mean() / (hbar * hbar).sum(1).mean())
        rows.append(row)
        print({key: round(val, 4) for key, val in row.items() if isinstance(val, float) and "2se" not in key}
              | {"D": D, "step": step}, flush=True)
    return {"D": D, "chains": chains, "seconds": exact["seconds"], "rows": rows,
            "N_eff mean": float(ne.mean()), "N_eff min": float(ne.min()), "N_eff tracked": ne[track].tolist()}


def plot(results):
    import matplotlib.pyplot as plt

    plt.rcParams.update(C.PLOT_STYLE)
    ink, muted, grid = "#29313d", "#707783", "#d9d7d1"
    colors = {"mean field": "#d89170", "diagonal C": "#78a58f", "closure": "#4f7fa8", "sampled": "#9b7bb8"}
    labels = {"mean field": "mean field", "diagonal C": "closure, diagonal $C$", "closure": "closure",
              "sampled": f"sampled closure, $k={results['setup']['k']}$"}
    K, N = results["setup"]["K"], results["setup"]["N"]
    dims = np.array([r["D"] for r in results["runs"]])
    cols = {"q": "means (overlaps)", "D": "delayed correlations $\\mathcal{D}_{ij}$", "C": "equal-time $C_{i\\neq j}$"}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
    for i, step in enumerate((1, K)):
        rows = [r["rows"][step - 1] for r in results["runs"]]
        for j, (obs, title) in enumerate(cols.items()):
            ax = axes[i, j]
            for name, color in colors.items():
                if step == 1 and name in ("diagonal C", "sampled") and obs != "C":
                    continue  # identical to the closure at K = 1 from a product start
                if name == "sampled" and obs != "q" or name in ("mean field", "diagonal C") and obs == "C":
                    continue  # no off-diagonal C (mean field, diagonal); sampled entries: see sketch.py
                err = np.array([r[f"{name} {obs}"] for r in rows])
                band = np.array([r[f"{name} {obs} 2se"] for r in rows])
                resolved = err - band > 0  # interval excludes zero
                ax.plot(dims, np.where(err > 0, err, np.nan), "-", color=color, lw=2, label=labels[name])
                lower = np.where(resolved, band, 0)  # unresolved: upper bound only
                ax.errorbar(dims, err, yerr=[lower, band], fmt="none", ecolor=color, elinewidth=1, capsize=2)
                ax.plot(dims[resolved], err[resolved], "o", color=color, ms=4.5)
                ax.plot(dims[~resolved], err[~resolved], "o", color=color, mfc="white", ms=4.5)
            ref = rows[2][f"closure {obs}"]
            ax.plot(dims, ref * dims[2] / dims, ":", color=ink, lw=1, label="$\\propto 1/D$")
            ax.axvline(N, color=grid, lw=1)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(dims, [str(d) for d in dims])
            ax.grid(color=grid, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=10, color=ink)
            if i == 1:
                ax.set_xlabel("vector dimension $D$")
        axes[i, 0].set_ylabel(f"relative error after $K={step}$")
    handles, labs = axes[1, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls="none", marker="o", mfc="white", color=muted, ms=4.5))
    labs.append("interval includes zero")
    fig.legend(handles, labs, frameon=False, fontsize=8, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"convergence.{ext}", dpi=200)

    # Field norm of the most concentrated site at step 1, two dimensions.
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for D, color in ((64, "#b7c7d8"), (1024, "#4f7fa8")):
        run_ = next(r for r in results["runs"] if r["D"] == D)
        field = run_["rows"][0]["field"]
        ax.hist(field["samples"][0], bins=60, density=True, color=color, alpha=0.8,
                label=f"exact sampling, $D={D}$")
    ax.axvline(field["mean field"][0], color=colors["mean field"], lw=2,
               label="mean field $\\|\\bar{\\mathbf{h}}_i\\|^2$")
    ax.axvline(field["closure"][0], color=ink, lw=2, ls="--", label="closure $\\|\\bar{\\mathbf{h}}_i\\|^2+v_i$")
    ax.set_xlabel("$\\|\\mathbf{h}_i\\|^2/R^2$", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.1, 1.0))
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"fieldnorm.{ext}", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="only redraw from convergence.json")
    args = parser.parse_args()
    out = HERE / "convergence.json"
    if not args.plot:
        results = {"setup": SETUP, "runs": [run(D, **SETUP) for D in DIMS]}
        out.write_text(json.dumps(results))
    plot(json.loads(out.read_text()))
