"""Sec. 4: carrying noise instead of a matrix (D = 64, beta = 2, u = 1, sharpness 3).

Sketch error of the scalar-channel closure against the full closure, for means,
delayed correlations D_ij and the entropy production sigma, versus the number of
channels k, at N in {256, 1024}: after K = 4 updates and at stationarity (one
snapshot, and a time average over 200 updates). Exact sampling gives the
closure's own error for scale. Also per-step runtimes.

    python sketch.py           # runs, writes sketch.json and figure
    python sketch.py --plot    # redraws from sketch.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import core as C

HERE = Path(__file__).resolve().parent
D, BETA, U, SHARP, SEED = 64, 2.0, 1.0, 3.0, 0
KS = [4, 16, 64, 256]
SEEDS = 8
K, BURN, LONG = 4, 100, 300


def readouts(x, J, state):
    R2 = C.order(D)
    obs = C.observables(state["m"], state["C"], state["D"], x)
    obs["sigma"] = np.array([BETA * ((J - J.T) * state["D"]).sum() / R2])
    return obs


def gap(pred, ref):
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def run(N):
    x = C.fields(N, D, U, BETA, [SEED, N, 0])
    f = C.fields(N, D, 1.0, BETA, [SEED, N, 1])
    J = C.softmax_couplings(N, SHARP, SEED)
    M0 = C.exact_mean(f, BETA)
    R2 = C.order(D)
    full = {"K=4": C.settle(C.closure_steps(x, J, M0, BETA), K),
            "stationary": C.settle(C.closure_steps(x, J, M0, BETA), 1000)}
    ref = {mode: readouts(x, J, s) for mode, s in full.items()}

    physical, seconds = scale(N, x, f, J, ref)

    rows = []
    for k in KS:
        for mode in ("K=4", "snapshot", "average"):
            errs = {obs: [] for obs in ("q", "D", "sigma", "m")}
            slack = []
            for seed in range(SEEDS):
                steps = C.sampled_steps(x, J, M0, BETA, k, [SEED, N, k, seed])
                if mode == "K=4":
                    state = C.settle(steps, K)
                elif mode == "snapshot":
                    state = C.settle(steps, LONG)
                else:
                    state = C.settle(steps, LONG, average_from=BURN + 1)
                    state["slack"] = None
                target = ref["K=4" if mode == "K=4" else "stationary"]
                pred = readouts(x, J, state)
                for obs in errs:
                    errs[obs].append(gap(pred[obs], target[obs]))
                if state.get("slack") is not None:
                    slack.append(float(np.abs(state["slack"] - (R2 - (state["m"] ** 2).sum(1))).mean() / R2))
            row = {"N": N, "k": k, "mode": mode, **{f"{obs}": float(np.mean(v)) for obs, v in errs.items()},
                   **{f"{obs} std": float(np.std(v)) for obs, v in errs.items()},
                   "sphere residual": float(np.mean(slack)) if slack else None}
            rows.append(row)
            print({key: (round(val, 4) if isinstance(val, float) else val) for key, val in row.items()
                   if "std" not in key}, flush=True)
    return {"N": N, "physical": physical, "rows": rows, "N_eff mean": float(C.n_eff(J).mean()),
            "exact seconds": seconds}


def scale(N, x, f, J, ref):
    """The closure's own error against exact sampling, for scale, with jackknife intervals."""
    R2 = C.order(D)
    chains = 16000 if N <= 256 else 8000
    sim = C.simulate(x, f, J, BETA, K, chains, record=(K,), seed=[SEED, N, 2])
    exs = C.stationary(x, f, J, BETA, seed=[SEED, N, 3])
    physical = {}
    for mode, ex in (("K=4", sim[K]), ("stationary", exs)):
        reps = C.observables(ex["m"], ex["C"], ex["D"], x, gram=ex["gram"])
        reps["sigma"] = (BETA * ((J - J.T) * ex["D"]).sum((-2, -1)) / R2)[:, None]
        physical[mode] = {}
        for obs, rep in reps.items():
            physical[mode][obs], physical[mode][f"{obs} 2se"] = C.error_interval(ref[mode][obs], rep)
    print(N, "closure vs exact", {m: {k: round(v, 4) for k, v in d.items()} for m, d in physical.items()}, flush=True)
    return physical, [sim["seconds"], exs["seconds"]]


def rescale(results):
    """Recompute only the closure-vs-exact scale of each run (the sketch errors need no sampling)."""
    for run_ in results["runs"]:
        N = run_["N"]
        x = C.fields(N, D, U, BETA, [SEED, N, 0])
        f = C.fields(N, D, 1.0, BETA, [SEED, N, 1])
        J = C.softmax_couplings(N, SHARP, SEED)
        M0 = C.exact_mean(f, BETA)
        full = {"K=4": C.settle(C.closure_steps(x, J, M0, BETA), K),
                "stationary": C.settle(C.closure_steps(x, J, M0, BETA), 1000)}
        ref = {mode: readouts(x, J, s_) for mode, s_ in full.items()}
        run_["physical"], run_["exact seconds"] = scale(N, x, f, J, ref)
    return results


def timings():
    rows = []
    for N in (256, 1024, 4096):
        x = C.fields(N, D, U, BETA, [SEED, N, 0])
        f = C.fields(N, D, 1.0, BETA, [SEED, N, 1])
        J = C.softmax_couplings(N, SHARP, SEED)
        M0 = C.exact_mean(f, BETA)
        gens = {"mean field (means only)": lambda: _mf_means(x, J, M0),
                "full closure": lambda: C.closure_steps(x, J, M0, BETA),
                "sampled k=16": lambda: C.sampled_steps(x, J, M0, BETA, 16, 0, readout=False),
                "sampled k=64": lambda: C.sampled_steps(x, J, M0, BETA, 64, 0, readout=False)}
        row = {"N": N}
        for name, make in gens.items():
            steps = make()
            next(steps)
            reps = 20 if N < 4096 else 3
            t0 = time.perf_counter()
            for _ in range(reps):
                next(steps)
            row[name] = (time.perf_counter() - t0) / reps * 1e3
        rows.append(row)
        print({key: (round(val, 2) if isinstance(val, float) else val) for key, val in row.items()}, "ms/step", flush=True)
    return rows


def _mf_means(x, J, M):
    while True:
        M = C.response(x + J @ M, BETA)[0]
        yield M


def plot(results):
    import matplotlib.pyplot as plt

    plt.rcParams.update(C.PLOT_STYLE)
    ink, muted, grid = "#29313d", "#707783", "#d9d7d1"
    colors = {"K=4": "#4f7fa8", "snapshot": "#9b7bb8", "average": "#c9a227"}
    labels = {"K=4": "after $K=4$", "snapshot": "stationary, one snapshot", "average": "stationary, time average"}
    cols = {"q": "means (overlaps)", "D": "delayed correlations $\\mathcal{D}_{ij}$", "sigma": "entropy production $\\sigma$"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    ks = np.array(KS)
    for ax, (obs, title) in zip(axes, cols.items()):
        for run in results["runs"]:
            solid = run["N"] == 1024
            for mode, color in colors.items():
                rows = [r for r in run["rows"] if r["mode"] == mode]
                ax.plot(ks, [r[obs] for r in rows], "o-" if solid else "o--", color=color, lw=2 if solid else 1,
                        ms=4 if solid else 3, alpha=1 if solid else 0.7, label=labels[mode] if solid and obs == "q" else None)
            if solid:
                for ref, mode in (("K=4", "K=4"), ("stationary", "average")):
                    ax.axhline(run["physical"][ref][obs], color=colors[mode], lw=1, ls=":",
                               label=f"closure vs exact, {'$K=4$' if ref == 'K=4' else ref}" if obs == "q" else None)
        ax.plot(ks, 0.5 * np.sqrt(ks[0] / ks), color=muted, lw=1, ls="-.", label="$\\propto k^{-1/2}$" if obs == "q" else None)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("channels $k$")
        ax.set_title(title, fontsize=10, color=ink)
        ax.grid(color=grid, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("sampled-closure error vs full closure")
    axes[0].plot([], [], "-", color=muted, lw=2, label="$N=1024$")
    axes[0].plot([], [], "--", color=muted, lw=1, label="$N=256$")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, frameon=False, fontsize=8, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"sketch.{ext}", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="only redraw from sketch.json")
    parser.add_argument("--physical", action="store_true", help="only recompute the closure-vs-exact scale")
    args = parser.parse_args()
    out = HERE / "sketch.json"
    if args.physical:
        out.write_text(json.dumps(rescale(json.loads(out.read_text()))))
    elif not args.plot:
        results = {"runs": [run(N) for N in (256, 1024)], "timings": timings()}
        out.write_text(json.dumps(results))
    plot(json.loads(out.read_text()))
