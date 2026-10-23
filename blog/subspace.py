"""Appendix E: fluctuations inside the signal subspace (N = 64, beta = 2, u = 1, sharpness 3, K = 4).

Drives and initializer fields confined to a random r-dimensional subspace (r = 1, 8),
against a generic control (full-rank drives, a random subspace of the same rank). The
closure carries traces only, so the most it can say about a subspace is the
isotropic share (r/D) C_ii. Exact sampling measures the subspace variance. A
"resolved" recursion carries r x r subspace blocks of the covariance; a "naive"
variant puts the single-spin anisotropy (variance smaller by 1/gamma) along the
mean field and propagates with the plain gain.

    python subspace.py           # runs, writes subspace.json and figure
    python subspace.py --plot    # redraws from subspace.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import core as C

HERE = Path(__file__).resolve().parent
N, BETA, U, SHARP, K, CHAINS, SEED = 64, 2.0, 1.0, 3.0, 4, 16000, 0
DS = [64, 256, 1024]
CASES = {"signal": [1, 8], "generic": [None]}  # rank of the drives; None is full rank
RANKS = [1, 8]  # measured subspaces: the first r columns of the basis


def confined(basis, r, u, seed):
    """Fields in the span of the first r columns of basis (all of R^D if r is None), with beta |x_i| / R = u."""
    D = basis.shape[0]
    if r is None:
        return C.fields(N, D, u, BETA, seed)
    x = np.random.default_rng(seed).standard_normal((N, r)) @ basis[:, :r].T
    return x * (u * np.sqrt(C.order(D)) / BETA / np.linalg.norm(x, axis=1, keepdims=True))


def resolved(x, J, f, P, naive=False, sphere=True):
    """The closure plus r x r blocks G_ij = P^T Cov(s_i, s_j) P; returns tr G_ii, C_ii and gamma.

    Linear response of the projected spin to the previous fluctuations,
    L_ij = a_i J_ij I + 2 a'_i p_i (J_ij p_i - J_ij^2 q_j)^T, with p_i = P^T hbar_i and q_j = P^T m_j:
    the gain a(|h|^2) responds to |h|^2 = |hbar|^2 + 2 hbar.eta + |eta|^2, and the sphere fixes
    |ds_j|^2 = R^2 - |m_j|^2 - 2 m_j.ds_j, so |eta|^2 carries -2 sum_j J_ij^2 m_j.ds_j (``sphere``).
    The bilinear cross-site part of |eta|^2 is dropped.  Fresh noise has covariance
    (1/(1+gamma)) (I - (1 - 1/gamma) p p^T / r^2): the noise tilts each sample's field out of the
    subspace.  ``naive`` keeps only the single-spin anisotropy along the mean field and the plain gain.
    All of this needs the mean fields inside the subspace.
    """
    N, D = x.shape
    R2 = C.order(D)
    r = P.shape[1]
    I = np.eye(r)
    norm = np.linalg.norm(f, axis=1)
    kappa = BETA * np.sqrt(R2) * norm
    A = C.bessel_ratio(R2, kappa, terms=64)
    across, along = A / kappa, 1 - A**2 - (D - 1) * A / kappa
    p0 = (f / norm[:, None]) @ P
    G = np.zeros((N, N, r, r))
    G[np.arange(N), np.arange(N)] = R2 * (across[:, None, None] * I + (along - across)[:, None, None] * p0[:, :, None] * p0[:, None, :])
    M = C.exact_mean(f, BETA)
    Cfull = np.diag(R2 - (M * M).sum(1))
    for _ in range(K):
        H = x + J @ M
        v = np.einsum("ij,jk,ik->i", J, Cfull, J)
        s2 = (H * H).sum(1) + v
        g = C.gamma(s2, BETA, R2)
        a = BETA / (1 + g)
        p, q = H @ P, M @ P
        pp = p[:, :, None] * p[:, None, :]
        L = a[:, None, None, None] * J[:, :, None, None] * I
        if naive:
            fresh = (I - (1 - 1 / g)[:, None, None] * pp / (H * H).sum(1)[:, None, None]) / (1 + g)[:, None, None]
        else:
            da2 = -a * BETA**2 / (R2 * g * (1 + g))  # 2 a'(r^2)
            L += da2[:, None, None, None] * J[:, :, None, None] * pp[:, None]
            if sphere:
                L -= da2[:, None, None, None] * (J**2)[:, :, None, None] * p[:, None, :, None] * q[None, :, None, :]
            fresh = (I - (1 - 1 / g)[:, None, None] * pp / s2[:, None, None]) / (1 + g)[:, None, None]
        G = np.einsum("ilac,kldc->ikad", np.einsum("ijab,jlbc->ilac", L, G), L)
        G[np.arange(N), np.arange(N)] += fresh
        M = a[:, None] * H
        Cfull = a[:, None] * (J @ Cfull @ J.T) * a[None, :] + np.diag(2 * R2 / (1 + g))
    return np.einsum("iaa->i", G[np.arange(N), np.arange(N)]), np.diag(Cfull), g


def run(case, rank, D):
    basis = np.linalg.qr(np.random.default_rng([SEED, D, 3]).standard_normal((D, max(RANKS))))[0]
    x = confined(basis, rank, U, [SEED, D, 0])
    f = confined(basis, rank, 1.0, [SEED, D, 1])  # initializer fields in the same subspace
    J = C.softmax_couplings(N, SHARP, SEED)
    R2 = C.order(D)
    sim = C.simulate(x, f, J, BETA, K, CHAINS, record=(K,), basis=basis, seed=[SEED, D, 2])
    ex = sim[K]
    cl = C.closure(x, J, C.exact_mean(f, BETA), BETA, K)
    reps = C.observables(ex["m"], ex["C"], ex["D"], x, gram=ex["gram"])
    pred = C.observables(cl["m"][-1], cl["C"][-1], cl["D"][-1], x)
    out = {"case": case, "drive rank": rank, "D": D, "seconds": sim["seconds"]}
    for obs in ("m", "q", "C", "D"):
        out[f"closure {obs}"], out[f"closure {obs} 2se"] = C.error_interval(pred[obs], reps[obs])
        out[f"floor {obs}"] = C.floor(reps[obs])
    for r in RANKS if rank is None else [rank]:
        V = ex["subvar"][..., :r].sum(-1)  # (replicates, N): subspace variance
        iso_reps = r / D * (R2 - np.diagonal(ex["gram"], axis1=1, axis2=2))  # (replicates, N), same samples as V
        iso_exact = iso_reps.mean(0)  # isotropic share of the exact trace: tests isotropy alone
        res, Cii_cl, g = resolved(x, J, f, basis[:, :r])
        preds = {"isotropic": r / D * Cii_cl,
                 "naive": resolved(x, J, f, basis[:, :r], naive=True)[0],
                 "resolved, no sphere term": resolved(x, J, f, basis[:, :r], sphere=False)[0], "resolved": res}
        Vbar = V.mean(0)
        row = {"ratio exact/isotropic": (Vbar / iso_exact).tolist(), "gamma": g.tolist(), "floor": C.floor(V),
               "naive/isotropic": (preds["naive"] / preds["isotropic"]).tolist(),
               "resolved/isotropic": (res / preds["isotropic"]).tolist()}
        for name, pr in preds.items():
            row[f"{name} error"], row[f"{name} error 2se"] = C.error_interval(pr, V)
        name = "isotropic (exact trace)"  # estimated per replicate: pair the residuals
        row[f"{name} error"], row[f"{name} error 2se"] = C.error_interval(iso_reps, V, paired=True)
        out[f"r={r}"] = row
    print(case, rank, D, f"{sim['seconds']:.0f}s", {k: round(v, 4) for k, v in out.items() if isinstance(v, float) and "floor" not in k},
          {r: {k: round(v, 4) if isinstance(v, float) else round(float(np.mean(v)), 3) for k, v in out[r].items()}
           for r in out if r.startswith("r=")}, flush=True)
    return out


def plot(results):
    import matplotlib.pyplot as plt

    plt.rcParams.update(C.PLOT_STYLE)
    ink, muted, grid = "#29313d", "#707783", "#d9d7d1"
    colors = {("signal", 1): "#d89170", ("signal", 8): "#9b7bb8", ("generic", 1): "#78a58f", ("generic", 8): "#4f7fa8"}
    labels = {("signal", 1): "drives in rank 1", ("signal", 8): "drives in rank 8",
              ("generic", 1): "generic drives, random 1-dim", ("generic", 8): "generic drives, random 8-dim"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    gs = np.linspace(1, 2.7, 100)
    ax.plot(gs, 1 / gs, color=muted, lw=1, ls="-.", label="$1/\\gamma$ (single-spin anisotropy)")
    for row in results:
        if row["D"] != 1024:
            continue
        for key in (k for k in row if k.startswith("r=")):
            r = int(key[2:])
            ax.plot(row[key]["gamma"], row[key]["ratio exact/isotropic"], "o", ms=3, alpha=0.8,
                    color=colors[(row["case"], r)], label=labels[(row["case"], r)])
            if row["case"] == "signal":
                ax.plot(row[key]["gamma"], row[key]["resolved/isotropic"], "x", ms=4, mew=0.8, color=ink,
                        label="resolved recursion")
    ax.axhline(1, color=ink, lw=0.8, ls=":")
    ax.set_xlabel("$\\gamma_i$ (closure)")
    ax.set_ylabel("subspace variance, exact / isotropic")
    ax.set_title("per site, $D=1024$", fontsize=10, color=ink)
    ax = axes[1]
    for case, ranks in (("signal", [1, 8]), ("generic", [8])):
        for r in ranks:
            rows = [row for row in results if row["case"] == case and row["drive rank"] in (r, None) and f"r={r}" in row]
            Ds = [row["D"] for row in rows]
            col = colors[(case, r)]
            series = {"o-": [(row[f"r={r}"]["isotropic (exact trace) error"], row[f"r={r}"]["isotropic (exact trace) error 2se"]) for row in rows],
                      "s--": [(row[f"r={r}"]["resolved error"], row[f"r={r}"]["resolved error 2se"]) for row in rows],
                      "^:": [(row["closure D"], row["closure D 2se"]) for row in rows]}
            for style, vals in series.items():
                err, band = np.array(vals).T
                ok = err - band > 0  # interval excludes zero
                ax.plot(Ds, err, style[1:], color=col, lw=1.5 if style == "o-" else 1)
                ax.errorbar(Ds, err, yerr=[np.where(ok, band, 0), band], fmt="none", ecolor=col, elinewidth=0.8, capsize=2)
                Da = np.array(Ds)
                ax.plot(Da[ok], err[ok], style[0], color=col, ms=4 if style == "o-" else 3)
                ax.plot(Da[~ok], err[~ok], style[0], color=col, mfc="white", ms=4 if style == "o-" else 3)
    ax.plot([], [], "o-", color=muted, label="subspace variance, isotropic share")
    ax.plot([], [], "s--", color=muted, label="subspace variance, resolved")
    ax.plot([], [], "^:", color=muted, label="closure $\\mathcal{D}_{ij}$ (traces)")
    ax.plot([], [], "o", color=muted, mfc="white", ls="none", label="interval includes zero")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(DS, [str(d) for d in DS])
    ax.set_xlabel("$D$")
    ax.set_ylabel("relative error vs exact")
    ax.set_title("errors against exact sampling", fontsize=10, color=ink)
    for a in axes:
        a.grid(color=grid, lw=0.6)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    h0, l0 = axes[0].get_legend_handles_labels()
    seen = dict(zip(l0, h0))
    axes[0].legend(seen.values(), seen.keys(), frameon=False, fontsize=7, loc="lower left")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"subspace.{ext}", dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="only redraw from subspace.json")
    args = parser.parse_args()
    out = HERE / "subspace.json"
    if not args.plot:
        results = [run(case, rank, D) for case, ranks in CASES.items() for rank in ranks for D in DS]
        out.write_text(json.dumps(results))
    plot(json.loads(out.read_text()))
