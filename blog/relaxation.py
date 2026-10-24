"""The stationary state: relaxation and stationary errors (N = 64, the previous post's ensemble).

Trajectories to K = 100 at D in {64, 256} for diffuse (sharpness 1, the previous
post) and sharp (sharpness 3; "concentrated" in the JSON) couplings at (beta, u) = (2, 1), with the
stationary errors as the end point. A stationary table over a few (u, beta)
points and D in {16, 64, 256, 1024} for the diffuse ensemble, including the
housekeeping entropy production sigma = beta sum_ij (J - J^T)_ij D_ij.

    python relaxation.py           # runs, writes relaxation.json and figure
    python relaxation.py --plot    # redraws from relaxation.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import core as C

HERE = Path(__file__).resolve().parent
N, CHAINS, SEED = 64, 8000, 0
CHECKPOINTS = [1, 2, 4, 8, 16, 32, 64, 100]
ENSEMBLES = {"diffuse": 1.0, "concentrated": 3.0}
NAMES = ["mean field", "diagonal C", "closure"]


def setup(D, beta, u, sharpness):
    x = C.fields(N, D, u, beta, [SEED, D, 0])
    f = C.fields(N, D, 1.0, beta, [SEED, D, 1])
    return x, f, C.softmax_couplings(N, sharpness, SEED)


def recursions(x, f, J, beta, K):
    M0 = C.exact_mean(f, beta)
    return {"mean field": C.mean_field(x, J, f, beta, K),
            "diagonal C": C.closure(x, J, M0, beta, K, diagonal=True),
            "closure": C.closure(x, J, M0, beta, K)}


def entropy(J, D, beta):
    return beta * ((J - J.T) * D).sum((-2, -1))


def compare(x, J, beta, ex, res, index):
    """Errors of each recursion at ``index`` against replicate estimates ``ex``."""
    R2 = C.order(x.shape[1])
    reps = C.observables(ex["m"], ex["C"], ex["D"], x, gram=ex["gram"])
    reps["sigma"] = entropy(J, ex["D"], beta)[:, None] / R2
    row = {f"floor {obs}": C.floor(rep) for obs, rep in reps.items()}
    for name, r in res.items():
        pred = C.observables(r["m"][index], r["C"][index], r["D"][index], x)
        pred["sigma"] = np.array([entropy(J, r["D"][index], beta) / R2])
        for obs, rep in reps.items():
            row[f"{name} {obs}"], row[f"{name} {obs} 2se"] = C.error_interval(pred[obs], rep)
    row["sigma exact"] = float(reps["sigma"].mean())
    return row


def trajectory(ensemble, D, beta=2.0, u=1.0):
    x, f, J = setup(D, beta, u, ENSEMBLES[ensemble])
    exact = C.simulate(x, f, J, beta, CHECKPOINTS[-1], CHAINS, record=CHECKPOINTS, seed=[SEED, D, 2])
    res = recursions(x, f, J, beta, CHECKPOINTS[-1])
    rows = [{"step": k, **compare(x, J, beta, exact[k], res, k - 1)} for k in CHECKPOINTS]
    stat = stationary_row(x, f, J, beta)
    print(ensemble, D, f"{exact['seconds']:.0f}s",
          [(r["step"], round(r["closure q"], 4), round(r["closure D"], 4)) for r in rows], flush=True)
    return {"ensemble": ensemble, "D": D, "N_eff mean": float(C.n_eff(J).mean()), "rows": rows, "stationary": stat}


def stationary_row(x, f, J, beta, K=3000):
    ex = C.stationary(x, f, J, beta, seed=[SEED, x.shape[1], 5])
    res = recursions(x, f, J, beta, K)
    settled = {name: float(np.linalg.norm(r["m"][-1] - r["m"][-2]) / np.linalg.norm(r["m"][-1])) for name, r in res.items()}
    return {**compare(x, J, beta, ex, res, -1), "drift": ex["drift"], "recursion change": settled,
            "seconds": ex["seconds"]}


def table():
    rows = []
    for u, beta in [(1.0, 1.0), (1.0, 2.0), (0.5, 2.0), (1.0, 4.0)]:
        for D in [16, 64, 256, 1024]:
            x, f, J = setup(D, beta, u, ENSEMBLES["diffuse"])
            row = {"u": u, "beta": beta, "D": D, **stationary_row(x, f, J, beta)}
            rows.append(row)
            print(f"(u,beta)=({u},{beta}) D={D:5d}: " + " | ".join(
                f"{n}: q {row[f'{n} q']:.4f} D {row[f'{n} D']:.4f} sigma {row[f'{n} sigma']:.4f}" for n in NAMES)
                + f" | floors q {row['floor q']:.4f} D {row['floor D']:.4f} sigma {row['floor sigma']:.4f}"
                + f" | drift {row['drift']:.2f} | {row['seconds']:.0f}s", flush=True)
    return rows


def plot(results):
    import matplotlib.pyplot as plt

    plt.rcParams.update(C.PLOT_STYLE)
    ink, muted, grid = "#29313d", "#707783", "#d9d7d1"
    colors = {"mean field": "#d89170", "diagonal C": "#78a58f", "closure": "#4f7fa8"}
    labels = {"mean field": "mean field", "diagonal C": "closure, diagonal $C$", "closure": "closure"}
    rows = {"diffuse": "diffuse", "concentrated": "sharp"}
    cols = {"q": "means (overlaps)", "D": "delayed correlations $\\mathcal{D}_{ij}$"}
    inf = 200  # where the stationary point is drawn
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6), sharex=True)
    for i, ensemble in enumerate(ENSEMBLES):
        for j, (obs, title) in enumerate(cols.items()):
            ax = axes[i, j]
            for run in results["trajectories"]:
                if run["ensemble"] != ensemble:
                    continue
                solid = run["D"] == 256
                steps = [r["step"] for r in run["rows"]]
                for name, color in colors.items():
                    err = [r[f"{name} {obs}"] for r in run["rows"]]
                    ax.plot(steps, err, "-" if solid else "--", color=color, lw=2 if solid else 1, alpha=1 if solid else 0.7,
                            label=f"{labels[name]}" if solid and i == 0 and j == 0 else None)
                    err, band = run["stationary"][f"{name} {obs}"], run["stationary"][f"{name} {obs} 2se"]
                    ax.plot([inf], [err], "s", color=color, ms=5.5 if solid else 4, alpha=1 if solid else 0.7,
                            mfc=color if err - band > 0 else "white")  # hollow: interval includes zero, as in Fig. 3
                ne = run["N_eff mean"]
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks(CHECKPOINTS + [inf], [str(k) for k in CHECKPOINTS] + ["$\\infty$"])
            ax.grid(color=grid, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(title, fontsize=10, color=ink)
            if i == 1:
                ax.set_xlabel("update $K$")
        axes[i, 0].set_ylabel(f"{rows[ensemble]} ($N_{{\\mathrm{{eff}}}}\\approx{ne:.0f}$)\nrelative error")
    axes[0, 0].plot([], [], "-", color=muted, lw=2, label="$D=256$")
    axes[0, 0].plot([], [], "--", color=muted, lw=1, label="$D=64$")
    handles, labs = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labs, frameon=False, fontsize=8, loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"relaxation.{ext}", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="only redraw from relaxation.json")
    args = parser.parse_args()
    out = HERE / "relaxation.json"
    if not args.plot:
        results = {"trajectories": [trajectory(e, D) for e in ENSEMBLES for D in (64, 256)], "table": table()}
        out.write_text(json.dumps(results))
    plot(json.loads(out.read_text()))
