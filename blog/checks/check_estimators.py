"""Bias and interval checks for the reference estimators, on a tiny shared-input system.

Mean over many small replicates (n chains each) against a high-precision reference.
The old estimator (same-sample centering) runs on identical chains via the saved copy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import core as C
import core_before_unbiased as OLD

N, D, BETA, K = 4, 8, 2.0, 3
J = np.array([[0.1, 0.0, 0.45, 0.45], [0.0, 0.1, 0.45, 0.45], [0.3, 0.3, 0.2, 0.2], [0.5, 0.0, 0.0, 0.5]])
x = C.fields(N, D, 1.0, BETA, 1)
f = C.fields(N, D, 1.0, BETA, 2)
basis = np.linalg.qr(np.c_[x[0], x[2]])[0]  # a 2-dim subspace containing two drives


def bias(est, ref):
    """Relative bias of the replicate mean and its standard error."""
    mean = est.mean(0)
    se = np.sqrt(((est - mean) ** 2).sum() / (len(est) - 1) / len(est))
    return np.linalg.norm(mean - ref) / np.linalg.norm(ref), se / np.linalg.norm(ref)


if __name__ == "__main__":
    n, Rr = 10, 4000
    ref = C.simulate(x, f, J, BETA, K, 2_000_000, replicates=8, seed=99, basis=basis)
    new = C.simulate(x, f, J, BETA, K, n * Rr, replicates=Rr, seed=7, basis=basis)
    old = OLD.simulate(x, f, J, BETA, K, n * Rr, replicates=Rr, seed=7, basis=basis)
    print(f"transient, n = {n} chains per replicate, {Rr} replicates (relative bias +- 1 s.e.)")
    for k in range(1, K + 1):
        r, a, b = ref[k], new[k], old[k]
        off = 1 - np.eye(N)
        oldsub = b["proj2"] - b["proj"] ** 2
        rows = {"C offdiag": (a["C"] * off, b["C"] * off, (r["C"] * off).mean(0)),
                "C diag": (np.diagonal(a["C"], 0, 1, 2), np.diagonal(b["C"], 0, 1, 2), np.diagonal(r["C"], 0, 1, 2).mean(0)),
                "D_ij": (a["D"], b["D"], r["D"].mean(0)),
                "gram": (a["gram"], b["gram"], r["gram"].mean(0)),
                "subspace var": (a["subvar"], oldsub, r["subvar"].mean(0))}
        for name, (e_new, e_old, e_ref) in rows.items():
            bn, sn = bias(e_new, e_ref); bo, so = bias(e_old, e_ref)
            print(f"  K={k} {name:12s} new {bn:.4f} +- {sn:.4f}   old {bo:.4f} +- {so:.4f}")

    # Interval calibration: a prediction with a known relative error against groups of 8 replicates.
    truth = 0.05
    for name in ("D_ij", "C offdiag"):
        e_ref = ref[K]["D"].mean(0) if name == "D_ij" else (ref[K]["C"] * (1 - np.eye(N))).mean(0)
        reps = new[K]["D"] if name == "D_ij" else new[K]["C"] * (1 - np.eye(N))
        rng = np.random.default_rng(0)
        pert = rng.standard_normal(e_ref.shape) * (1 - np.eye(N) if name != "D_ij" else 1)
        pred = e_ref + truth * np.linalg.norm(e_ref) * pert / np.linalg.norm(pert)
        groups = reps.reshape(-1, 8, N, N)[:200]
        res = np.array([C.error_interval(pred, g) for g in groups])
        cover = np.mean(np.abs(res[:, 0] - truth) <= res[:, 1])
        print(f"interval check {name}: true error {truth}, mean estimate {res[:, 0].mean():.4f},"
              f" mean 2 s.e. {res[:, 1].mean():.4f}, coverage {cover:.2f} over 200 groups of 8 replicates")

    # Stationary: few chains per replicate, short windows.
    st_ref = C.stationary(x, f, J, BETA, chains=4000, replicates=8, burn=60, T=2000, seed=11)
    st_new = C.stationary(x, f, J, BETA, chains=4, replicates=2000, burn=60, T=40, seed=12)
    st_old = OLD.stationary(x, f, J, BETA, chains=4, replicates=2000, burn=60, T=40, seed=12)
    print("stationary, 4 chains x 40 updates per replicate, 2000 replicates")
    off = 1 - np.eye(N)
    for name, key, mask in (("C offdiag", "C", off), ("D_ij", "D", 1), ("gram", "gram", 1)):
        e_ref = (st_ref[key] * mask).mean(0)
        bn, sn = bias(st_new[key] * mask, e_ref); bo, so = bias(st_old[key] * mask, e_ref)
        print(f"  {name:10s} new {bn:.4f} +- {sn:.4f}   old {bo:.4f} +- {so:.4f}")
