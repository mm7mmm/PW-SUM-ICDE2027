"""Parameter rule of PW-Sum: weights, Z_lb, radii t_j, M = sum t + K - 1, zeta*(K) and the 50-digit certificate.

Retry rule: p* = 1e-3 is fixed; while p_bar > p*, L is increased by ln 2 and t is recomputed, at most 20 times; if the rule
still fails, the parameter point fails (ParamRuleFailure), and search() skips that K and counts it. The byte model ell(K)
is replaced by the caller with the measured ECIES ciphertext length."""
import math

import mpmath as mp
import numpy as np

mp.mp.dps = 50
EPS = 1.0
CT_OVERHEAD = 48
ELL_MIN = CT_OVERHEAD + 1
P_STAR = 1e-3             # fixed
MAX_RETRY = 20            # at most 20 increments of L by ln 2


class ParamRuleFailure(Exception):
    pass


def ell(K):
    """Default byte model (section 10); replaced by the measured length in a_params.py."""
    return CT_OVERHEAD + -(-int(K).bit_length() // 8)


def vmax(gamma, eps=EPS):
    return 2 * (1 + gamma) / eps ** 2


def zeta_star(N, K, gamma, eps=EPS):
    rest = vmax(gamma, eps) - (N + 1) / (4 * K * K)
    if rest <= 2 / eps ** 2:
        return None
    return 1 - math.sqrt(2 / (eps ** 2 * rest))


def k_min(N, gamma, eps=EPS):
    return math.floor(math.sqrt((N + 1) / (4 * (vmax(gamma, eps) - 2 / eps ** 2)))) + 1


def weights(K, zeta, mode, eps=EPS):
    j = np.arange(K + 1)
    if mode == "pw":
        a = eps * (zeta + 2 * (1 - zeta) * np.minimum(j, K - j) / K)
    else:
        a = np.full(K + 1, eps * zeta)
    return a, (1 - zeta) * eps / K


def instance(K, zeta, delta, mode, eps=EPS):
    """Parameter rule (paper section V) with the task-sheet retry rule."""
    a, b = weights(K, zeta, mode, eps)
    sigbar = float(np.sum(np.abs(np.arange(K + 1) - K / 2) * 2 * math.sqrt(2) / a))
    zlb = 2 / (3 * math.e) * min(1.0, 1 / (math.sqrt(3) * b * sigbar))
    L = math.log(1 / (delta * zlb * (1 - P_STAR)))
    for retries in range(MAX_RETRY + 1):
        t = 1 + np.ceil(2 * L / a)
        pbar = float(np.sum(np.exp(-a * t / 2))) / zlb
        if pbar <= P_STAR:
            break
        L += math.log(2)
    else:
        raise ParamRuleFailure(f"K={K}: p_bar={pbar:.3e} > p*={P_STAR} after {MAX_RETRY} retries")
    MT = int(t.sum())
    return {"K": K, "zeta": zeta, "L": L, "t": t, "MT": MT, "M": MT + K - 1,
            "p_star": P_STAR, "pbar": pbar, "zlb": zlb, "retries": retries}


def search(N, gamma, delta, mode, eps=EPS):
    """Search over K; K values whose parameter rule fails are skipped and counted.
    Returns (instance, bytes/user, K_stop, n_failed_K)."""
    zinf = 1 - math.sqrt(2 / (eps ** 2 * vmax(gamma, eps)))
    lnd = math.log(1 / delta)
    best, K, failed = None, k_min(N, gamma, eps), 0
    while True:
        z = zeta_star(N, K, gamma, eps)
        if z is not None:
            try:
                inst = instance(K, z, delta, mode, eps)
                cost = (2 * N + inst["M"]) * ell(K)
                if best is None or cost < best[0]:
                    best = (cost, inst)
            except ParamRuleFailure:
                failed += 1
        a_inf, _ = weights(K, zinf, mode, eps)
        if best is not None and (2 * N + float(np.sum(2 * lnd / a_inf))) * ELL_MIN > best[0]:
            return best[1], best[0] / N, K, failed
        K += 1


def l_eps_delta(eps, delta):
    e, d = mp.mpf(eps), mp.mpf(delta)
    tot, u = mp.mpf(0), 1
    while True:
        term = 1 - d * (mp.exp(u * e) - 1) / (mp.exp(e) - 1)
        if term <= 0:
            return tot
        tot += term
        u += 1


def mp_zeta_star(N, K, gamma, eps=EPS):
    e, g = mp.mpf(eps), mp.mpf(gamma)
    return 1 - mp.sqrt(2 / (e ** 2 * (2 * (1 + g) / e ** 2 - mp.mpf(N + 1) / (4 * mp.mpf(K) ** 2))))


def certify_point(K, zeta, delta, mode, eps=EPS):
    """50-digit re-evaluation with the same task-sheet retry rule."""
    e, d, z = mp.mpf(eps), mp.mpf(delta), mp.mpf(zeta)
    if mode == "pw":
        a = [e * (z + 2 * (1 - z) * mp.mpf(min(j, K - j)) / K) for j in range(K + 1)]
    else:
        a = [e * z] * (K + 1)
    b = (1 - z) * e / K
    half = mp.mpf(K) / 2
    sigbar = mp.fsum(abs(j - half) * 2 * mp.sqrt(2) / a[j] for j in range(K + 1))
    zlb = 2 / (3 * mp.e) * min(mp.mpf(1), 1 / (mp.sqrt(3) * b * sigbar))
    p_star = mp.mpf(P_STAR)
    L = mp.log(1 / (d * zlb * (1 - p_star)))
    for retries in range(MAX_RETRY + 1):
        t = [1 + int(mp.ceil(2 * L / aj)) for aj in a]
        pbar = mp.fsum(mp.exp(-aj * tj / 2) for aj, tj in zip(a, t)) / zlb
        if pbar <= p_star:
            break
        L += mp.log(2)
    else:
        raise ParamRuleFailure(f"K={K}: certificate p_bar > p* after {MAX_RETRY} retries")
    tails = sorted((mp.exp(-aj * (tj - 1) / 2) for aj, tj in zip(a, t)), reverse=True)
    dbar = (tails[0] + tails[1]) / (2 * zlb * (1 - pbar))
    pair_max = max((a[x] + a[y]) / 2 + b * abs(x - y)
                   for x in range(K + 1) for y in {0, K, K - x, K // 2, (K + 1) // 2} if y != x)
    MT = sum(t)
    M = MT + K - 1
    m_ub = (3 * (K + 1) + 2 * L / e * (2 / z + K / (1 - z) * mp.log(1 / z))) if mode == "pw" else None
    return {"K": K, "zeta": z, "alpha": a, "beta": b, "sigbar": sigbar, "zlb": zlb, "p_star": p_star,
            "L": L, "t": t, "pbar": pbar, "dbar": dbar, "pair_slack": pair_max - e, "MT": MT, "M": M,
            "m_ub": m_ub, "retries": retries}


def certify(N, gamma, delta, K, mode, eps=EPS):
    z = mp_zeta_star(N, K, gamma, eps)
    r = certify_point(K, z, delta, mode, eps)
    e = mp.mpf(eps)
    V = 2 * (1 + mp.mpf(gamma)) / e ** 2
    mse = 2 / ((1 - z) ** 2 * e ** 2) + mp.mpf(N + 1) / (4 * mp.mpf(K) ** 2)
    tiny = mp.mpf("1e-40")
    r.update({
        "N": N, "gamma": gamma, "delta": delta, "mode": mode, "mse": mse, "vmax": V,
        "bytes": (2 * N + r["M"]) * ell(K) / N,
        "pair_ok": r["pair_slack"] <= e * tiny, "pbar_ok": r["pbar"] <= r["p_star"],
        "delta_ok": r["dbar"] <= mp.mpf(delta), "mse_ok": mse <= V * (1 + tiny),
        "mub_ok": r["m_ub"] is None or r["M"] <= r["m_ub"],
    })
    return r
