"""Shared core for the numerical checks: problems, recursions, exact sampling, errors.

Same model and conventions as largeD.py (R^2 = D/2 - 1, product start, raw
covariance traces), with a faster exact sampler:

- spins are stored as (N, B, D), so fields and Gram sums are single matmuls;
- chains run in worker processes with single-threaded BLAS;
- float32 sampling, float64 accumulation;
- Rao-Blackwellized readouts: m', off-diagonal C' and D_ij are read from the
  exact conditional mean mu_D(h) instead of the sampled spin, and C'_ii = R^2
  from the sphere, which removes the fresh sampling noise from every estimate.
"""

import multiprocessing as mp
import os
import time

import numpy as np


def order(D):
    """R^2 = D/2 - 1, which is also the Bessel order nu of the vMF normalizer."""
    return D / 2 - 1


# --- single site ------------------------------------------------------------


def bessel_ratio(nu, kappa, terms=16):
    """A = I_{nu+1}(kappa) / I_nu(kappa), by backward recurrence from an Amos-type start.

    Stable for any order and argument, unlike a ratio of scaled Bessel functions,
    which underflows at large order and moderate argument.
    """
    top = nu + terms
    r = kappa / (top + 0.5 + np.sqrt((top + 1.5) ** 2 + kappa**2))
    for j in range(terms, 0, -1):  # r_{mu-1} = kappa / (2 mu + kappa r_mu)
        r = kappa / (2 * (nu + j) + kappa * r)
    return r


def exact_mean(h, beta):
    """E[s | h] = R A(kappa) h/|h| with kappa = beta R |h|."""
    R2 = order(h.shape[-1])
    norm = np.linalg.norm(h, axis=-1, keepdims=True)
    return np.sqrt(R2) * bessel_ratio(R2, beta * np.sqrt(R2) * norm) * h / norm


def gamma(r2, beta, R2):
    return np.sqrt(1 + beta**2 * r2 / R2)


def response(h, beta, v=0.0):
    """Large-D response beta h / (1 + gamma), with gamma evaluated at |h|^2 + v."""
    R2 = order(h.shape[-1])
    g = gamma((h * h).sum(-1) + v, beta, R2)
    return (beta / (1 + g))[..., None] * h, g


def wood(kappa, D, rng):
    """Alignment w = mu . u of a unit vMF sample on S^{D-1} (Wood 1994), for flat kappa."""
    d1 = D - 1
    b = d1 / (2 * kappa + np.sqrt(4 * kappa**2 + d1**2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + d1 * np.log(1 - x0**2)
    w = np.empty_like(kappa)
    todo = np.arange(kappa.size)
    while todo.size:
        z = rng.beta(d1 / 2, d1 / 2, todo.size)
        prop = (1 - (1 + b[todo]) * z) / (1 - (1 - b[todo]) * z)
        accept = kappa[todo] * prop + d1 * np.log(1 - x0[todo] * prop) - c[todo] >= np.log(rng.uniform(size=todo.size))
        w[todo[accept]] = prop[accept]
        todo = todo[~accept]
    return w


def sample_given(H, norm, beta, rng):
    """One exact vMF draw per row of H (..., D) on the radius-R sphere, given |H| (...,)."""
    D = H.shape[-1]
    R = np.sqrt(order(D))
    w = wood((beta * R * norm).ravel(), D, rng).reshape(norm.shape)
    xi = rng.standard_normal(H.shape, dtype=H.dtype)
    along = np.einsum("...d,...d->...", xi, H) / norm               # xi . h_hat
    perp2 = np.einsum("...d,...d->...", xi, xi) - along**2           # |xi_perp|^2
    c2 = R * np.sqrt(1 - w**2) / np.sqrt(perp2)
    c1 = (R * w - c2 * along) / norm
    return c1.astype(H.dtype)[..., None] * H + c2.astype(H.dtype)[..., None] * xi


# --- problems ---------------------------------------------------------------


def fields(N, D, u, beta, seed):
    """Random directions with beta |x_i| / R = u."""
    x = np.random.default_rng(seed).standard_normal((N, D))
    return x * (u * np.sqrt(order(D)) / beta / np.linalg.norm(x, axis=1, keepdims=True))


def softmax_couplings(N, sharpness, seed, causal=False):
    """Row softmax of Gaussian logits (fixed across D); causal masks j > i."""
    logits = sharpness * np.random.default_rng(seed).standard_normal((N, N))
    if causal:
        logits = np.where(np.tril(np.ones((N, N), bool)), logits, -np.inf)
    J = np.exp(logits - logits.max(1, keepdims=True))
    return J / J.sum(1, keepdims=True)


def n_eff(J):
    return 1 / (J**2).sum(1)


# --- deterministic recursions -----------------------------------------------
# Each returns per-step arrays m (K, N, D), C (K, N, N), D (K, N, N), v (K, N).


def _cov_parts(h, beta):
    """Large-D conditional covariance Sigma = a I - b m m^T at field h."""
    m, g = response(h, beta)
    return 1 / (1 + g), 1 / (order(h.shape[-1]) * g), m


def mean_field_steps(x, J, f, beta):
    """Naive mean field (the neqnn update) from the product start in initializer fields f.

    Delayed correlations use the previous post's linear-response formula
    D_ij = beta J_ij Tr(Sigma_i Sigma_j^prev) in its large-D form.
    """
    N, D = x.shape
    R2 = order(D)
    M, h_prev = exact_mean(f, beta), f
    while True:
        h = x + J @ M
        a1, b1, m1 = _cov_parts(h, beta)
        a0, b0, m0 = _cov_parts(h_prev, beta)
        trace = (D * np.outer(a1, a0) - np.outer(a1, b0 * (m0 * m0).sum(1))
                 - np.outer(b1 * (m1 * m1).sum(1), a0) + np.outer(b1, b0) * (m1 @ m0.T) ** 2)
        M, h_prev = m1, h
        yield {"m": M, "C": np.diag(R2 - (M * M).sum(1)), "D": beta * J * trace, "v": np.zeros(N)}


def closure_steps(x, J, M0, beta, diagonal=False):
    """Large-D closure on M and C; ``diagonal`` drops cross-site covariance every step."""
    R2 = order(x.shape[1])
    M, C = M0, np.diag(R2 - (M0 * M0).sum(1))
    while True:
        H = x + J @ M
        JC = J @ C
        JCJ = JC @ J.T
        v = np.diag(JCJ).copy()
        g = gamma((H * H).sum(1) + v, beta, R2)
        a = beta / (1 + g)
        M = a[:, None] * H
        C = a[:, None] * JCJ * a[None, :] + np.diag(2 * R2 / (1 + g))
        if diagonal:
            C = np.diag(np.diag(C))
        yield {"m": M, "C": C, "D": a[:, None] * JC, "v": v}


def sampled_steps(x, J, M0, beta, k, seed, readout=True):
    """Scalar-channel closure: C_hat = diag(b) + Z Z^T / k with Z (N, k), cost O(N^2 D + k N^2).

    b holds only the fresh, independent variance; it is sampled into Z only once J
    mixes it into other sites.  C and D are materialized only for readout.
    """
    rng = np.random.default_rng(seed)
    N, D = x.shape
    R2 = order(D)
    M, b, Z = M0, R2 - (M0 * M0).sum(1), np.zeros((N, k))
    J2 = J * J
    while True:
        H = x + J @ M
        Y = J @ Z
        v = J2 @ b + (Y * Y).sum(1) / k
        g = gamma((H * H).sum(1) + v, beta, R2)
        a = beta / (1 + g)
        M = a[:, None] * H
        delayed = a[:, None] * (J * b[None, :] + Y @ Z.T / k) if readout else None
        Z = a[:, None] * (Y + J @ (np.sqrt(b)[:, None] * rng.standard_normal((N, k))))
        b = 2 * R2 / (1 + g)
        yield {"m": M, "C": np.diag(b) + Z @ Z.T / k if readout else None, "D": delayed, "v": v,
               "slack": b + (Z * Z).sum(1) / k}  # vs R^2 - |m|^2: the sphere identity in expectation


def collect(steps, K):
    """The first K steps, stacked: m (K, N, D), C (K, N, N), D (K, N, N), v (K, N)."""
    out = [next(steps) for _ in range(K)]
    return {key: np.array([o[key] for o in out]) for key in ("m", "C", "D", "v")}


def settle(steps, K, average_from=None):
    """State after K steps, or its time average over steps average_from..K, without storing the path."""
    total, count = None, 0
    for t in range(1, K + 1):
        state = next(steps)
        if average_from is not None and t >= average_from:
            total = {key: state[key] if total is None else total[key] + state[key] for key in ("m", "C", "D", "v")}
            count += 1
    return state if average_from is None else {key: val / count for key, val in total.items()}


def mean_field(x, J, f, beta, K):
    return collect(mean_field_steps(x, J, f, beta), K)


def closure(x, J, M0, beta, K, diagonal=False):
    return collect(closure_steps(x, J, M0, beta, diagonal), K)


def sampled_closure(x, J, M0, beta, K, k, seed):
    return collect(sampled_steps(x, J, M0, beta, k, seed), K)


# --- exact sampling ---------------------------------------------------------


def _chains(job):
    """Sums over one replicate's chains, in the (N, B, D) layout."""
    x, f, J, beta, K, record, n, batch, seed, track, dtype, basis = job
    N, D = x.shape
    rng = np.random.Generator(np.random.SFC64(seed))
    x_, J_ = x.astype(dtype), J.astype(dtype)
    r = 0 if basis is None else basis.shape[1]
    sums = {k: {"m": np.zeros((N, D)), "mm": np.zeros((N, N)), "lag": np.zeros((N, N)), "prev": np.zeros((N, D)),
                "mp": np.zeros((N, N)), "h2": np.zeros(N), "h4": np.zeros(N), "track": [],
                "proj": np.zeros((N, r)), "projmu2": np.zeros((N, r)), "proj2": np.zeros((N, r))} for k in record}
    mu0 = exact_mean(f, beta).astype(dtype)
    f_norm = np.linalg.norm(f, axis=1)
    for start in range(0, n, batch):
        B = min(batch, n - start)
        F = np.broadcast_to(f.astype(dtype)[:, None, :], (N, B, D))
        S = sample_given(F, np.broadcast_to(f_norm[:, None], (N, B)), beta, rng)
        mu_prev = np.broadcast_to(mu0[:, None, :], (N, B, D))
        for k in range(1, K + 1):
            H = x_[:, None, :] + (J_ @ S.reshape(N, B * D)).reshape(N, B, D)
            norm = np.sqrt(np.einsum("nbd,nbd->nb", H, H).astype(np.float64))
            if k in sums:
                acc = sums[k]
                R = np.sqrt(order(D))
                mu = (R * bessel_ratio(order(D), beta * R * norm) / norm).astype(dtype)[..., None] * H
                flat = mu.reshape(N, B * D)
                acc["m"] += mu.sum(1)
                acc["mm"] += flat @ flat.T
                acc["lag"] += flat @ S.reshape(N, B * D).T
                acc["prev"] += mu_prev.sum(1)
                acc["mp"] += flat @ np.ascontiguousarray(mu_prev).reshape(N, B * D).T
                r2 = norm**2 / order(D)
                acc["h2"] += r2.sum(1)
                acc["h4"] += (r2**2).sum(1)
                acc["track"].append(r2[list(track)])
                if basis is not None:
                    pm = (mu @ basis.astype(dtype)).astype(np.float64)
                    acc["proj"] += pm.sum(1)
                    acc["projmu2"] += (pm**2).sum(1)
                    acc["proj2"] += _projected_square(H, norm, beta, basis)
                mu_prev = mu
            elif k + 1 in sums:
                R = np.sqrt(order(D))
                mu_prev = (R * bessel_ratio(order(D), beta * R * norm) / norm).astype(dtype)[..., None] * H
            if k < K:
                S = sample_given(H, norm.astype(dtype), beta, rng)
    for acc in sums.values():
        acc["track"] = np.concatenate(acc["track"], axis=1) if acc["track"] else None
    return sums


def _projected_square(H, norm, beta, basis):
    """Sum over chains of E[(u_c . s)^2 | h] for the columns u_c of basis, from the exact vMF covariance.

    On the unit sphere the variance is A/kappa across h and dA/dkappa = 1 - A^2 - (D-1) A/kappa along it.
    """
    D = H.shape[-1]
    R2 = order(D)
    kappa = beta * np.sqrt(R2) * norm
    A = bessel_ratio(R2, kappa, terms=64)
    across = A / kappa
    along = 1 - A**2 - (D - 1) * A / kappa
    c = ((H / norm[..., None].astype(H.dtype)) @ basis.astype(H.dtype)).astype(np.float64) ** 2  # (N, B, r)
    return (R2 * ((A**2 + along - across)[..., None] * c + across[..., None])).sum(1)


def _map(fn, jobs, workers):
    """Run jobs in fresh interpreters with single-threaded BLAS: the workers parallelize."""
    if workers <= 1:
        return [fn(job) for job in jobs]
    env = {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    os.environ.update({k: "1" for k in env})
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            return pool.map(fn, jobs)
    finally:
        for k, val in env.items():
            os.environ.pop(k) if val is None else os.environ.__setitem__(k, val)


def simulate(x, f, J, beta, K, chains, *, record=None, replicates=8, workers=4, budget=2**24, seed=0,
             track=(), basis=None, dtype=np.float32):
    """Exact moments at the recorded steps, as independent replicate estimates.

    Returns {step: {"m": (R, N, D), "C": (R, N, N) raw, "D": (R, N, N) raw,
    "h2": (N,) mean |h|^2/R^2, "h2var": (N,), "track": (len(track), chains)}}.
    With an orthonormal ``basis`` (D, r), also "subvar" (R, N, r), the variance of each
    spin's coordinates along the basis vectors.

    Products of means are centered without bias within each replicate (chains are
    independent): E[a]E[b] is estimated by mean(a) mean(b) - (mean(ab) - mean(a) mean(b)) / (n - 1).
    The delayed covariance is centered with the previous conditional means mu_prev, so its
    correction uses mean(mu_i mu_prev_j), not the delayed product itself.
    """
    N, D = x.shape
    R2 = order(D)
    record = tuple(range(1, K + 1)) if record is None else tuple(record)
    n = chains // replicates
    batch = max(1, min(n, budget // (N * D)))
    seeds = np.random.SeedSequence(seed).generate_state(replicates)
    jobs = [(x, f, J, beta, K, record, n, batch, int(s), track, dtype, basis) for s in seeds]
    t0 = time.time()
    parts = _map(_chains, jobs, workers)
    out = {}
    for k in record:
        get = lambda key: np.array([p[k][key] / n for p in parts])  # noqa: E731
        m, mm, prev, lag, mp = get("m"), get("mm"), get("prev"), get("lag"), get("mp")
        gram = _unbiased(np.einsum("rid,rjd->rij", m, m), mm, n)  # m_i . m_j
        cross = _unbiased(np.einsum("rid,rjd->rij", m, prev), mp, n)  # m_i . m_prev_j
        C = mm - gram  # i != j: E[s_i . s_j] = E[mu_i . mu_j]
        C[:, np.arange(N), np.arange(N)] = R2 - np.diagonal(gram, axis1=1, axis2=2)  # sphere: E|s|^2 = R^2
        h2 = sum(p[k]["h2"] for p in parts) / (n * replicates)
        h4 = sum(p[k]["h4"] for p in parts) / (n * replicates)
        proj = get("proj")
        out[k] = {"m": m, "C": C, "D": lag - cross, "gram": gram, "n": n,
                  "h2": h2, "h2var": h4 - h2**2,
                  "track": np.concatenate([p[k]["track"] for p in parts], axis=1) if track else None,
                  "subvar": get("proj2") - _unbiased(proj**2, get("projmu2"), n)}
    out["seconds"] = time.time() - t0
    return out


def _stationary_chains(job):
    """Time sums over one replicate's chains after burn-in, split into two halves of the window.

    Means are kept per chain, (N, n, D), so that products of means can be centered across
    the replicate's independent chains."""
    x, f, J, beta, n, burn, T, seed, dtype = job
    N, D = x.shape
    R = np.sqrt(order(D))
    rng = np.random.Generator(np.random.SFC64(seed))
    x_, J_ = x.astype(dtype), J.astype(dtype)
    sums = [{"m": np.zeros((N, n, D)), "mm": np.zeros((N, N)), "lag": np.zeros((N, N))} for _ in range(2)]
    F = np.broadcast_to(f.astype(dtype)[:, None, :], (N, n, D))
    S = sample_given(F, np.broadcast_to(np.linalg.norm(f, axis=1)[:, None], (N, n)), beta, rng)
    for t in range(burn + T):
        H = x_[:, None, :] + (J_ @ S.reshape(N, n * D)).reshape(N, n, D)
        norm = np.sqrt(np.einsum("nbd,nbd->nb", H, H).astype(np.float64))
        if t >= burn:
            acc = sums[(t - burn) * 2 // T]
            mu = (R * bessel_ratio(order(D), beta * R * norm) / norm).astype(dtype)[..., None] * H
            flat = mu.reshape(N, n * D)
            acc["m"] += mu
            acc["mm"] += flat @ flat.T
            acc["lag"] += flat @ S.reshape(N, n * D).T
        S = sample_given(H, norm.astype(dtype), beta, rng)
    return sums


def stationary(x, f, J, beta, *, chains=32, replicates=8, burn=120, T=300, workers=4, seed=0, dtype=np.float32):
    """Stationary moments from time averages after burn-in, as independent replicate estimates.

    Defaults follow the previous post's estimator size (replicates x 32 chains, 120
    burn-in, 300 updates).  "drift" compares the two halves of the averaging
    window, relative to their Monte Carlo spread (about 1 if stationary).

    Samples are correlated in time, so products of means are centered across each
    replicate's independent chains: mean over chains c != c' of m_c . m_c', with m_c a
    chain's time average.  Replicates stay independent.
    """
    N, D = x.shape
    R2 = order(D)
    seeds = np.random.SeedSequence(seed).generate_state(replicates)
    t0 = time.time()
    parts = _map(_stationary_chains, [(x, f, J, beta, chains, burn, T, int(s), dtype) for s in seeds], workers)
    count = chains * T / 2
    out = {}
    for obs in ("mm", "lag"):
        out[obs] = np.array([(p[0][obs] + p[1][obs]) / (2 * count) for p in parts])
    halves = np.array([[p[h]["m"].mean(1) / (T / 2) for p in parts] for h in range(2)])  # (2, R, N, D)
    diff = halves[0] - halves[1]
    noise = np.sqrt(((diff - diff.mean(0)) ** 2).sum() / (replicates - 1) / replicates)
    per_chain = np.array([(p[0]["m"] + p[1]["m"]) / T for p in parts])  # (R, N, n, D)
    m = per_chain.mean(2)
    own = np.einsum("ricd,rjcd->rij", per_chain, per_chain) / chains
    gram = (chains * np.einsum("rid,rjd->rij", m, m) - own) / (chains - 1)  # mean over c != c'
    C = out["mm"] - gram
    C[:, np.arange(N), np.arange(N)] = R2 - np.diagonal(gram, axis1=1, axis2=2)
    return {"m": m, "C": C, "D": out["lag"] - gram, "gram": gram,
            "drift": float(np.linalg.norm(diff.mean(0)) / noise), "seconds": time.time() - t0}


# --- errors -----------------------------------------------------------------


def _unbiased(product, mean_of_product, n):
    """Unbiased E[a] E[b] from n independent samples: mean(a) mean(b) - (mean(ab) - mean(a) mean(b)) / (n - 1)."""
    return product - (mean_of_product - product) / (n - 1)


def _pair_sum(v):
    """mean_{r<s} <v_r, v_s> from the totals, for replicates v (R, ...)."""
    flat = v.reshape(len(v), -1)
    total = flat.sum(0)
    return ((total @ total) - (flat * flat).sum()) / (len(v) * (len(v) - 1))


def error(prediction, replicates, paired=False):
    """Noise-corrected relative Frobenius error: cross-replicate products cancel MC variance.

    ``paired``: the prediction is itself estimated per replicate, shape (R, ...), from the
    same samples; residuals are then formed within each replicate before pairing.
    """
    den = _pair_sum(replicates)
    num = _pair_sum((prediction if paired else prediction[None]) - replicates)
    return float(np.sqrt(max(num, 0) / den)) if den > 0 else np.nan


def error_interval(prediction, replicates, paired=False):
    """Noise-corrected error and two jackknife standard errors (leave one replicate out)."""
    R = len(replicates)
    drop = lambda a, r: np.delete(a, r, 0) if paired else a  # noqa: E731
    loo = np.array([error(drop(prediction, r), np.delete(replicates, r, 0), paired) for r in range(R)])
    return error(prediction, replicates, paired), float(2 * np.sqrt((R - 1) / R * ((loo - loo.mean()) ** 2).sum()))


def floor(replicates):
    """Standard error of the pooled estimate relative to its norm."""
    pooled = replicates.mean(0)
    spread = ((replicates - pooled) ** 2).sum() / (len(replicates) - 1)
    return float(np.sqrt(spread / len(replicates)) / np.linalg.norm(pooled))


def offdiag(A):
    return A * (1 - np.eye(A.shape[-1]))


def observables(m, C, D, x, gram=None):
    """The compared quantities, in units of R^2 where they are traces.

    m: component-wise means.  q: overlap-level means, the Gram of the means and
    their overlaps with the drives.  C: off-diagonal equal-time traces.  D: delayed traces.
    Pass the sampler's unbiased ``gram`` for exact estimates; the Gram of estimated means is biased.
    """
    R2 = order(x.shape[-1])
    if gram is None:
        gram = np.einsum("...id,...jd->...ij", m, m)
    xm = np.einsum("id,...jd->...ij", x, m)
    return {"m": m, "q": np.stack([gram, xm], -3) / R2, "C": offdiag(C) / R2, "D": D / R2}
