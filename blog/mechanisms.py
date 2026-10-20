"""Sec. 2: what sets the gaps (D = 64, N = 256).

(a) Concentration, K = 1, D = 256: causal softmax couplings span N_eff,i from 1
    to ~N in one run. Per-site errors of mean field and the closure against
    N_eff,i, for beta in {1, 2}, and the deterministic gap between the two
    recursions (no sampling, no fit).
(b) Sharing, K = 4: identical row weight profiles (same N_eff,i), with the large
    weights scattered over columns or placed on common inputs.
(c) Robustness: one knob at a time around (beta, u) = (2, 1), K = 1 and 4.

    python mechanisms.py           # runs, writes mechanisms.json and figures
    python mechanisms.py --plot    # redraws from mechanisms.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import core as C

HERE = Path(__file__).resolve().parent
N, D, SHARP, CHAINS = 256, 64, 3.0, 16000
D_CONCENTRATION = 256


def models(x, f, J, beta, K):
    M0 = C.exact_mean(f, beta)
    return {"mean field": C.mean_field(x, J, f, beta, K),
            "diagonal C": C.closure(x, J, M0, beta, K, diagonal=True),
            "closure": C.closure(x, J, M0, beta, K)}


def errors(x, exact, res, K):
    R2 = C.order(x.shape[1])
    out = {}
    for step in (1, K):
        ex = exact[step]
        reps = C.observables(ex["m"], ex["C"], ex["D"], x, gram=ex["gram"])
        for obs, rep in reps.items():
            out[f"floor {obs} {step}"] = C.floor(rep)
            for name, r in res.items():
                pred = C.observables(r["m"][step - 1], r["C"][step - 1], r["D"][step - 1], x)
                out[f"{name} {obs} {step}"], out[f"{name} {obs} {step} 2se"] = C.error_interval(pred[obs], rep)
    return out


def concentration(beta, dim=D_CONCENTRATION, u=1.0, seed=0):
    x = C.fields(N, dim, u, beta, [seed, dim, 0])
    f = C.fields(N, dim, 1.0, beta, [seed, dim, 1])
    J = C.softmax_couplings(N, SHARP, seed, causal=True)
    exact = C.simulate(x, f, J, beta, 1, CHAINS, seed=[seed, dim, 2, int(10 * beta)])
    res = models(x, f, J, beta, 1)
    reps = exact[1]["m"]
    site = {name: [C.error(r["m"][0][i], reps[:, i]) for i in range(N)] for name, r in res.items()}
    mf, cl = res["mean field"]["m"][0], res["closure"]["m"][0]
    site["gap"] = (np.linalg.norm(mf - cl, axis=1) / np.linalg.norm(cl, axis=1)).tolist()  # deterministic
    site["floor"] = [C.floor(reps[:, i]) for i in range(N)]
    return {"beta": beta, "D": dim, "N_eff": C.n_eff(J).tolist(), "site": site, "total": errors(x, exact, res, 1),
            "v/R2": (res["closure"]["v"][0] / C.order(dim)).tolist(), "seconds": exact["seconds"]}


def profiles(seed):
    """Sorted softmax row profiles placed on independent vs common column orders."""
    rng = np.random.default_rng([seed, 7])
    P = -np.sort(-C.softmax_couplings(N, SHARP, seed), axis=1)
    scattered = np.zeros((N, N))
    for i in range(N):
        scattered[i, rng.permutation(N)] = P[i]
    common = P[:, np.argsort(rng.permutation(N))]  # one column order shared by all rows
    return {"scattered": scattered, "common": common}


def sharing(beta=2.0, u=1.0, K=4, seed=0):
    x = C.fields(N, D, u, beta, [seed, D, 0])
    f = C.fields(N, D, 1.0, beta, [seed, D, 1])
    out = {}
    for label, J in profiles(seed).items():
        exact = C.simulate(x, f, J, beta, K, CHAINS, seed=[seed, D, 3, len(label)])
        res = models(x, f, J, beta, K)
        cl = res["closure"]
        R2 = C.order(D)
        C_prev = cl["C"][K - 2]
        v_diag = (J * J) @ np.diag(C_prev)
        out[label] = {"errors": errors(x, exact, res, K), "N_eff": C.n_eff(J).tolist(),
                      "self coupling": float(np.diag(J).mean()),
                      "v diag/R2": float(v_diag.mean() / R2), "v cross/R2": float((cl["v"][K - 1] - v_diag).mean() / R2),
                      "seconds": exact["seconds"]}
        e = out[label]["errors"]
        print(label, {k: round(e[f"{n} {o} {K}"], 4) for n in ("mean field", "diagonal C", "closure") for o in ("q", "D")
                      for k in [f"{n} {o}"]}, "v diag", round(out[label]["v diag/R2"], 3), "v cross",
              round(out[label]["v cross/R2"], 3), "self", round(out[label]["self coupling"], 4), flush=True)
    return out


def robustness(K=4, seed=0):
    rows = []
    J = C.softmax_couplings(N, SHARP, seed)
    for beta, u in [(2, 1), (1, 1), (2, 0.5), (2, 2)]:
        x = C.fields(N, D, u, beta, [seed, D, 0])
        f = C.fields(N, D, 1.0, beta, [seed, D, 1])
        exact = C.simulate(x, f, J, beta, K, CHAINS, seed=[seed, D, 4, beta, int(10 * u)])
        res = models(x, f, J, beta, K)
        rows.append({"beta": beta, "u": u, **errors(x, exact, res, K)})
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in rows[-1].items() if "floor" not in k and " m " not in k and "2se" not in k},
              flush=True)
    return rows


def plot(results):
    import matplotlib.pyplot as plt

    ink, muted, grid = "#29313d", "#707783", "#d9d7d1"
    colors = {"mean field": "#d89170", "diagonal C": "#78a58f", "closure": "#4f7fa8"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    for run in results["concentration"]:
        ne = np.array(run["N_eff"])
        site = run["site"]
        alpha = 0.9 if run["beta"] == 2 else 0.4
        first = run["beta"] == 2
        ax.scatter(ne, site["mean field"], s=10, color=colors["mean field"], alpha=alpha,
                   label="mean field vs exact" if first else None)
        ax.scatter(ne, site["closure"], s=10, color=colors["closure"], alpha=alpha,
                   label="closure vs exact" if first else None)
        ax.scatter(ne, site["gap"], s=7, marker="x", color=ink, lw=0.6,
                   label="mean field vs closure (no sampling)" if first else None)
        i = int(np.argmin(ne))
        ax.annotate(f"$\\beta={run['beta']:g}$", (ne[i], site["mean field"][i]), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=8, color=ink)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, None)
    ax.set_xlabel("effective number of inputs $N_{\\mathrm{eff},i}$ (causal attention)")
    ax.set_ylabel("per-site error of $\\mathbf{m}_i$, $K=1$")
    ax.set_title(f"(a) concentration, $D={results['concentration'][0]['D']}$", fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    share = results["sharing"]
    K = 4
    groups = [("q", "means"), ("D", "delayed $D_{ij}$")]
    width = 0.13
    for g, (obs, title) in enumerate(groups):
        for s, label in enumerate(("scattered", "common")):
            for n, name in enumerate(colors):
                pos = g + (s - 0.5) * 0.45 + (n - 1) * width
                val = share[label]["errors"][f"{name} {obs} {K}"]
                bx.bar(pos, val, width, color=colors[name], alpha=1 if label == "common" else 0.55,
                       label=name if g == 0 and s == 1 else None)
            bx.text(g + (s - 0.5) * 0.45, -0.02, label, ha="center", va="top", fontsize=7, color=muted,
                    transform=bx.get_xaxis_transform())
    bx.set_xticks([0, 1], [t for _, t in groups])
    bx.tick_params(axis="x", pad=14)
    bx.set_ylabel(f"relative error, $K={K}$")
    bx.set_title("(b) same $N_{\\mathrm{eff},i}$, scattered vs common inputs, $D=64$", fontsize=10, color=ink)
    bx.legend(frameon=False, fontsize=7)
    for a in (ax, bx):
        a.grid(color=grid, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"mechanisms.{ext}", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="only redraw from mechanisms.json")
    parser.add_argument("--concentration", action="store_true", help="only rerun (a), keep (b) and (c)")
    args = parser.parse_args()
    out = HERE / "mechanisms.json"
    if args.concentration:
        results = json.loads(out.read_text())
        results["concentration"] = [concentration(beta) for beta in (1.0, 2.0)]
        out.write_text(json.dumps(results))
    elif not args.plot:
        results = {"concentration": [concentration(beta) for beta in (1.0, 2.0)],
                   "sharing": sharing(), "robustness": robustness()}
        out.write_text(json.dumps(results))
    plot(json.loads(out.read_text()))
