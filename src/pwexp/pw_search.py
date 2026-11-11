"""Best-found certified PW-Sum parameters.

Uses the formulas in pw_core_task.py:
weights (eq. weights), the parameter rule with p* = 1e-3 and the L += ln 2 retry (``instance``), zeta*(K) and K_min.

What this module adds:
* ``opt`` truncation: Theorem 1 holds for ANY integers t_j >= 1 with p_bar < 1 and gives delta_bar(t). Instead of the
  sufficient rule L = ln(1/(delta Z_lb (1-p*))), we take t_j(L) = 1 + ceil(2L/alpha_j) and bisect for the smallest
  L whose t(L) satisfies delta_bar(t(L)) <= delta (delta_bar and p_bar exactly as in Theorem 1, Z_lb from Lemma 4).
  delta_bar(t(L)) is non-increasing in L, so the bisection is exact up to float resolution; the result is then
  re-certified at 50 digits. This is a one-parameter family of t ("equal level"); it is not claimed to be the
  minimum over all t vectors.
* zeta = zeta*(K) - 1e-12 (eq. zetastar). For fixed K every alpha_j is non-decreasing in zeta and beta*sigma_bar is
  decreasing, so Z_lb, p_bar and delta_bar are monotone and the largest admissible zeta minimises M; the 1e-12 keeps
  the MSE certificate strictly below V_max after rounding.
* K is scanned over every integer K >= K_min until a provable floor on M exceeds the best value found:
  delta_bar >= e_x / (2 Z_lb) and Z_lb <= 2/(3e) give t_x >= 1 + 2 ln(1/delta)/alpha_x, so
  M >= 2K + sum_j 2 ln(1/delta)/alpha_j(zeta_inf) with zeta_inf = sup_K zeta*(K). Beyond the stop the floor is checked
  for every K up to K_safe, where the cruder floor 2K + 2 ln(1/delta)(K+1)/eps (alpha_j <= eps) alone exceeds the best.
"""
import math
import time

import mpmath as mp
import numpy as np

from . import pw_core_task as pc

ZETA_BACKOFF = 1e-12


def zeta_used(N, K, gamma, eps):
    z = pc.zeta_star(N, K, gamma, eps)
    return None if z is None or z <= ZETA_BACKOFF else z - ZETA_BACKOFF


def zlb_of(a, b, K):
    """Lemma 4 / eq. (zlb)."""
    sigbar = float(np.sum(np.abs(np.arange(K + 1) - K / 2) * 2 * math.sqrt(2) / a))
    return 2 / (3 * math.e) * min(1.0, 1 / (math.sqrt(3) * b * sigbar))


def t_of_L(a, L):
    return 1 + np.ceil(2 * L / a)


def dbar_pbar(a, t, zlb):
    """Theorem 1 in float64: p_bar and delta_bar = (sum of the two largest e^{-alpha_j(t_j-1)/2}) / (2 Z_lb (1-p_bar))."""
    pbar = float(np.sum(np.exp(-a * t / 2))) / zlb
    if pbar >= 1:
        return math.inf, pbar
    e = np.exp(-a * (t - 1) / 2)
    return float(np.partition(e, -2)[-2:].sum()) / (2 * zlb * (1 - pbar)), pbar


def opt_t(a, zlb, delta, L_hi):
    lo, hi = 0.0, L_hi
    assert dbar_pbar(a, t_of_L(a, hi), zlb)[0] <= delta
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if dbar_pbar(a, t_of_L(a, mid), zlb)[0] <= delta:
            hi = mid
        else:
            lo = mid
    return t_of_L(a, hi), hi


def at_K(N, gamma, delta, eps, K, mode="pw"):
    """Rule-based and certificate-optimised truncation at one K (float64; certified later at 50 digits)."""
    z = zeta_used(N, K, gamma, eps)
    if z is None:
        return None
    a, b = pc.weights(K, z, mode, eps)
    zlb = zlb_of(a, b, K)
    out = {"K": K, "zeta": z, "zlb": zlb}
    try:
        r = pc.instance(K, z, delta, mode, eps)
        out["rule"] = {"L": r["L"], "t": r["t"], "M": r["M"], "retries": r["retries"]}
        L_hi = r["L"]
    except pc.ParamRuleFailure as ex:
        out["rule"] = None
        out["rule_fail"] = str(ex)
        L_hi = math.log(1 / (delta * zlb))
        while dbar_pbar(a, t_of_L(a, L_hi), zlb)[0] > delta:
            L_hi *= 1.25
    t, L = opt_t(a, zlb, delta, L_hi)
    out["opt"] = {"L": L, "t": t, "M": int(t.sum()) + K - 1}
    return out


def m_floor(K, delta, eps, zinf, mode="pw"):
    a, _ = pc.weights(K, zinf, mode, eps)
    return 2 * K + float(np.sum(2 * math.log(1 / delta) / a))


def search(N, gamma, delta, eps, mode="pw", keep_trace=False):
    """Exhaustive scan over K (see module doc). Returns dict with the best point for each
    (truncation in {rule, opt}) x (objective in {M, bytes}) plus search bookkeeping."""
    t0 = time.time()
    kmin = pc.k_min(N, gamma, eps)
    zinf = 1 - math.sqrt(2 / (eps ** 2 * pc.vmax(gamma, eps)))
    best, trace, rule_fail = {}, [], []
    K = kmin
    while True:
        r = at_K(N, gamma, delta, eps, K, mode)
        if r is not None:
            if r["rule"] is None:
                rule_fail.append(K)
            for tr in ("rule", "opt"):
                x = r[tr]
                if x is None:
                    continue
                for obj, c in (("M", x["M"]), ("bytes", (2 * N + x["M"]) * pc.ell(K))):
                    if (tr, obj) not in best or c < best[(tr, obj)][0]:
                        best[(tr, obj)] = (c, {**r, "trunc": tr, "objective": obj})
            if keep_trace:
                trace.append((K, r["zeta"], r["rule"]["M"] if r["rule"] else -1, r["opt"]["M"]))
        mf = m_floor(K, delta, eps, zinf, mode)
        if len(best) == 4 and mf > max(best[(tr, "M")][0] for tr in ("rule", "opt")) and \
                (2 * N + mf) * pc.ELL_MIN > max(best[(tr, "bytes")][0] for tr in ("rule", "opt")):
            break
        K += 1
    k_stop = K
    # every K beyond the stop: exact floor up to K_safe, crude increasing floor afterwards
    thr_M = max(best[(tr, "M")][0] for tr in ("rule", "opt"))
    thr_B = max(best[(tr, "bytes")][0] for tr in ("rule", "opt"))
    lnd = math.log(1 / delta)
    k_safe = K
    while not (2 * k_safe + 2 * lnd * (k_safe + 1) / eps > thr_M and
               (2 * N + 2 * k_safe + 2 * lnd * (k_safe + 1) / eps) * pc.ELL_MIN > thr_B):
        k_safe += 1
    floor_ok = all(m_floor(k, delta, eps, zinf, mode) > thr_M and
                   (2 * N + m_floor(k, delta, eps, zinf, mode)) * pc.ELL_MIN > thr_B
                   for k in range(k_stop, k_safe + 1))
    return {"best": {k: v[1] for k, v in best.items()}, "K_min": kmin, "K_stop": k_stop, "K_safe": k_safe,
            "floor_ok_to_K_safe": floor_ok, "n_K": k_stop - kmin + 1, "rule_fail_K": rule_fail,
            "zeta_inf": zinf, "seconds": time.time() - t0, "trace": trace}


def certify(N, gamma, delta, eps, K, zeta, t, L, mode="pw"):
    """50-digit re-evaluation of Lemma 1 (pair condition), Lemma 4 (Z_lb), Theorem 1 (p_bar, delta_bar),
    Theorem 2 (real-valued MSE bound) and Theorem 3 (M upper bound) at the given float zeta and integer t."""
    mp.mp.dps = 50
    e, d, z = mp.mpf(eps), mp.mpf(delta), mp.mpf(zeta)
    t = [int(x) for x in t]
    if mode == "pw":
        a = [e * (z + 2 * (1 - z) * mp.mpf(min(j, K - j)) / K) for j in range(K + 1)]
    else:
        a = [e * z] * (K + 1)
    b = (1 - z) * e / K
    half = mp.mpf(K) / 2
    sigbar = mp.fsum(abs(j - half) * 2 * mp.sqrt(2) / a[j] for j in range(K + 1))
    zlb = 2 / (3 * mp.e) * min(mp.mpf(1), 1 / (mp.sqrt(3) * b * sigbar))
    pbar = mp.fsum(mp.exp(-aj * tj / 2) for aj, tj in zip(a, t)) / zlb
    tails = sorted((mp.exp(-aj * (tj - 1) / 2) for aj, tj in zip(a, t)), reverse=True)
    dbar = (tails[0] + tails[1]) / (2 * zlb * (1 - pbar)) if pbar < 1 else mp.inf
    pair_max = max((a[x] + a[y]) / 2 + b * abs(x - y)
                   for x in range(K + 1) for y in {0, K, K - x, K // 2, (K + 1) // 2} if y != x)
    mse = 2 / ((1 - z) ** 2 * e ** 2) + mp.mpf(N + 1) / (4 * mp.mpf(K) ** 2)
    V = 2 * (1 + mp.mpf(gamma)) / e ** 2
    M = sum(t) + K - 1
    zs = pc.mp_zeta_star(N, K, gamma, eps)
    m_ub = None
    if mode == "pw":
        # Theorem 3's proof only uses t_j <= 2 + 2L/alpha_j; take the smallest such L for the given t (50 digits)
        Lm = max(mp.mpf(0), max(aj * (tj - 2) / 2 for aj, tj in zip(a, t)))
        m_ub = 3 * (K + 1) + 2 * Lm / e * (2 / z + K / (1 - z) * mp.log(1 / z))
    return {
        "t_int_ge1": min(t) >= 1, "pair_slack": float(pair_max - e), "pair_ok": pair_max <= e * (1 + mp.mpf("1e-40")),
        "zlb": float(zlb), "pbar": float(pbar), "pbar_ok": pbar < 1, "dbar": float(dbar), "delta_ok": dbar <= d,
        "delta_margin_log10": float(mp.log10(d / dbar)), "mse_cert": float(mse), "vmax": float(V),
        "mse_ok": mse <= V, "mse_margin_rel": float((V - mse) / V), "zeta_le_zeta_star": z <= zs,
        "M": M, "thm3_M_ub": float(m_ub) if m_ub is not None else None,
        "thm3_ok": (M <= m_ub) if m_ub is not None else None,
    }
