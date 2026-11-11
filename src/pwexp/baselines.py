"""Pluggable methods for the comparison. Every method implements

    select_params(eps, delta, N, gamma) -> dict     (formulas and public quantities only)
    analytic_mse(p, qvar=None) -> float              (normalised units; qvar = exact quantisation variance of the
                                                      actual input at grid K, default = worst case N/(4K^2))
    simulate_error(y, p, rng, reps) -> ndarray       (estimate - true clipped sum, normalised units)
    comm_cost(p) -> dict                             (messages / bytes per user; 'analytic' or 'measured')

Specs: specs/<method>.md; review: specs/REVIEW.md. Units: inputs y in [0,1]; X_max = 1.
"""
import math

import numpy as np

CT = 48  # ECIES overhead (32-byte X25519 key + 16-byte tag)


def ell_bits(bits):
    return CT + max(1, -(-int(bits) // 8))


def ell_K(K):
    return CT + max(1, -(-int(K).bit_length() // 8))


def dlap_var(p):
    return 2 * p / (1 - p) ** 2


def dlap(rng, p, size):
    """Discrete Laplace with P(k) ∝ p^|k|, as the difference of two geometric variables (support >= 0)."""
    g = lambda: rng.geometric(1 - p, size=size) - 1  # noqa: E731
    return g() - g()


def quantise(y, K, rng):
    b = np.floor(y * K)
    return (b + (rng.random(len(y)) < y * K - b)).astype(np.int64)


def qvar_exact(y, K):
    f = y * K - np.floor(y * K)
    return float(np.sum(f * (1 - f))) / K ** 2


# ---------------------------------------------------------------- CDP / LDP
class CDP:
    name = "CDP-Laplace"
    trust = "trusted curator"

    def select_params(self, eps, delta, N, gamma):
        return {"eps": eps, "N": N}

    def analytic_mse(self, p, qvar=None):
        return 2 / p["eps"] ** 2

    def simulate_error(self, y, p, rng, reps):
        return rng.laplace(0, 1 / p["eps"], size=reps)

    def comm_cost(self, p):
        return {"messages_per_user": 1, "bytes_per_user": float("nan"), "kind": "not comparable (trusted curator)"}


class LDP:
    name = "LDP-Laplace"
    trust = "none (local)"

    def select_params(self, eps, delta, N, gamma):
        return {"eps": eps, "N": N}

    def analytic_mse(self, p, qvar=None):
        return 2 * p["N"] / p["eps"] ** 2

    def simulate_error(self, y, p, rng, reps):
        return np.array([rng.laplace(0, 1 / p["eps"], size=len(y)).sum() for _ in range(reps)])

    def comm_cost(self, p):
        return {"messages_per_user": 1, "bytes_per_user": 8.0, "kind": "analytic (one float64, one link)"}


# ---------------------------------------------------------------- GKMPS
def gkmps_atoms(D):
    """Theorem 12 / SM D.3: atoms and the dominating vector t. Returns list of (atom tuple, t_s, m)."""
    G = D * math.ceil(1 + math.log2(D))
    out = [((-1, 1), G, 1)]
    for m in range(2, D + 1):
        a, b = m // 2, -(-m // 2)
        out.append(((m, -a, -b), -(-G // m), m))
        out.append(((-m, a, b), -(-G // m), m))
    return out, G


def gkmps_right_inverse(D):
    """Build C (SM D.3) for small D and return (A_tilde @ C == I, max_i sum_s |C_si|/t_s)."""
    atoms, G = gkmps_atoms(D)
    idx = {a: k for k, (a, _, _) in enumerate(atoms)}
    rows = [i for i in range(-D, D + 1) if i not in (0, 1)]
    col = {}
    e = lambda k: np.eye(len(atoms), dtype=np.int64)[k]  # noqa: E731
    col[-1] = e(idx[(-1, 1)])
    col[1] = np.zeros(len(atoms), np.int64)
    for i in range(2, D + 1):
        i1, i2 = i // 2, -(-i // 2)
        col[i] = e(idx[(i, -i1, -i2)]) - col[-i1] - col[-i2]
        col[-i] = e(idx[(-i, i1, i2)]) - col[i1] - col[i2]
    A = np.array([[sum(1 for v in a if v == r) for a, _, _ in atoms] for r in rows])
    C = np.stack([col[r] for r in rows], axis=1)
    t = np.array([ts for _, ts, _ in atoms], float)
    dom = max(float(np.sum(np.abs(C[:, k]) / t)) for k, r in enumerate(rows) if r > 0)
    return bool(np.array_equal(A @ C, np.eye(len(rows), dtype=np.int64))), dom


class GKMPS:
    name = "GKMPS"
    trust = "shuffler; users add noise"

    def _cost(self, eps, delta, N, K, gG):
        D = K + 1
        es = (1 - gG) * eps
        e1 = e2 = min(1.0, gG * eps) / 2
        d1 = d2 = delta / 2
        pc = math.exp(-es / D)
        G = D * math.ceil(1 + math.log2(D))
        nS = 1 + 2 * (D - 1)
        rhat, phat = 3 * (1 + math.log(1 / d1)), math.exp(-0.2 * e1 / D)
        rs = 3 * (1 + math.log(nS / d2))
        m = np.arange(2, D + 1)
        ts = -(-G // m)
        ps = np.exp(-0.2 * e2 / (2 * ts))
        p1 = math.exp(-0.2 * e2 / (2 * G))
        tot = 2 * pc / (1 - pc)                                  # central: z^{+1}, z^{-1}
        tot += 2 * (rs * p1 / (1 - p1) + rhat * phat / (1 - phat))  # atom {-1,+1}: |s| = 2, D-tilde + D-hat
        tot += 2 * 3 * rs * float(np.sum(ps / (1 - ps)))         # two atoms of size 3 per m = 2..D
        msgs = 1 + tot / N
        bits = math.ceil(math.log2(2 * D + 1))
        return {"msgs": msgs, "bits": bits, "pc": pc, "ell": ell_bits(bits), "bytes": 2 * msgs * ell_bits(bits),
                "eps_star": es, "eps1": e1, "nS": nS, "Gamma": G}

    def select_params(self, eps, delta, N, gamma, n_g=120, fixed_gG=None):
        V = 2 * (1 + gamma) / eps ** 2
        best = None
        K = 1
        while N / (4 * K * K) >= V:
            K += 1
        k0, stall = K, 0
        while stall < 400 and K < 50 * k0 + 1000:
            room = (V - N / (4 * K * K)) * K * K   # allowed DLap variance
            # largest eps* with 2p/(1-p)^2 <= room  <=>  p <= p_max
            pm = (room + 1 - math.sqrt(2 * room + 1)) / room   # root of room p^2 - 2(room+1)p + room = 0
            es_min = -(K + 1) * math.log(pm)
            g_max = 1 - es_min / eps
            improved = False
            if g_max > 0:
                gs = [fixed_gG] if fixed_gG is not None else np.unique(np.r_[g_max * np.geomspace(1e-4, 1, n_g)])
                for gG in gs:
                    if gG <= 0 or gG > g_max or gG >= 1:
                        continue
                    c = self._cost(eps, delta, N, K, gG)
                    if best is None or c["bytes"] < best["bytes"]:
                        best = {"eps": eps, "delta": delta, "N": N, "gamma": gamma, "K": K, "gamma_G": float(gG), **c}
                        improved = True
            stall = 0 if improved else (stall + 1 if best is not None else 0)
            K += 1
        if best is None:
            # target unreachable within the scanned K: report the best achievable worst-case MSE (gamma_G -> 0)
            mse, Kb = min((dlap_var(math.exp(-eps / (k + 1))) / k ** 2 + N / (4 * k * k), k) for k in range(1, K))
            return {"eps": eps, "delta": delta, "N": N, "gamma": gamma, "feasible": False, "K": Kb,
                    "pc": math.exp(-eps / (Kb + 1)), "msgs": float("nan"), "bytes": float("nan"), "mse_worst": mse}
        best["feasible"] = True
        best["mse_worst"] = self.analytic_mse(best)
        return best

    def analytic_mse(self, p, qvar=None):
        K = p["K"]
        return dlap_var(p["pc"]) / K ** 2 + (p["N"] / (4 * K * K) if qvar is None else qvar)

    def simulate_error(self, y, p, rng, reps):
        K = p["K"]
        out = np.empty(reps)
        for r in range(reps):
            x = quantise(y, K, rng)
            out[r] = (x.sum() + dlap(rng, p["pc"], 1)[0]) / K - y.sum()
        return out

    def comm_cost(self, p):
        return {"messages_per_user": p["msgs"], "bytes_per_user": p["bytes"],
                "kind": "analytic (Observation 6 expectation; ECIES per message, both links)"}


# ---------------------------------------------------------------- BBGN
class BBGN:
    name = "BBGN"
    trust = "multi-Shuffler model (m parallel Shufflers plus one unshuffled message; stronger trust assumption than one Shuffler); users add noise"

    def select_params(self, eps, delta, N, gamma):
        V = 2 * (1 + gamma) / eps ** 2
        K = 1
        while dlap_var(math.exp(-eps / K)) / K ** 2 + N / (4 * K * K) > V:
            K += 1
        sigma = math.ceil(math.log2((1 + math.exp(eps)) / delta))
        q = math.ceil(2 * N * K)
        m = math.ceil((2 * sigma + math.log2(q)) / (math.log2(N) - math.log2(math.e))) + 1
        bits = math.ceil(math.log2(q))
        ok = m >= 3 and N >= 19 and sigma >= 1
        alpha = math.exp(-eps / K)
        p = {"eps": eps, "delta": delta, "N": N, "gamma": gamma, "K": K, "alpha": alpha, "sigma": sigma, "q": q,
             "m": m, "bits": bits, "preconditions_ok": ok, "delta_achieved": (1 + math.exp(eps)) * 2.0 ** -sigma,
             "wrap_prob_bound": 2 * alpha ** (N * K / 2) / (1 + alpha), "feasible": True}
        p["mse_worst"] = self.analytic_mse(p)
        return p

    def analytic_mse(self, p, qvar=None):
        K = p["K"]
        return dlap_var(p["alpha"]) / K ** 2 + (p["N"] / (4 * K * K) if qvar is None else qvar)

    def simulate_error(self, y, p, rng, reps):
        K, q, N = p["K"], p["q"], p["N"]
        out = np.empty(reps)
        for r in range(reps):
            x = quantise(y, K, rng)
            z = (int(x.sum()) + int(dlap(rng, p["alpha"], 1)[0])) % q
            if z > (N * K + q) / 2:
                z -= q
            out[r] = z / K - y.sum()
        return out

    def comm_cost(self, p):
        ell = ell_bits(p["bits"])
        return {"messages_per_user": p["m"] + 1, "bytes_per_user": (2 * p["m"] + 1) * ell,
                "kind": "analytic (Cor. 6.1 eq.(4); ECIES per message; shuffled shares on both links, unshuffled once)"}


# ---------------------------------------------------------------- ID-Sum (LNF adaptation)
def ageo_params(eps, beta, delta):
    ql = (math.exp(-eps / 2) - 1 + beta) / beta
    qr = beta / (math.exp(eps / 2) - 1 + beta)
    if ql <= 1e-15:   # S1Geo: beta = 1 - e^{-eps/2}, delta = 0, nu = 0
        return 0.0, qr, 0, 0.0
    c = 1 - math.exp(eps / 2) + beta * math.exp(eps / 2)
    nu = 0
    while True:
        kap = ql * (1 - ql ** nu) / (1 - ql) + 1 / (1 - qr)
        d = 2 / kap * ql ** nu * c
        if d <= delta:
            return ql, qr, nu, d
        nu += 1


def ageo_moments(ql, qr, nu):
    kap = (ql * (1 - ql ** nu) / (1 - ql) if nu else 0.0) + 1 / (1 - qr)
    k = np.arange(nu)
    wl = ql ** (nu - k) if nu else np.zeros(0)
    m1 = (np.sum(k * wl) + nu / (1 - qr) + qr / (1 - qr) ** 2) / kap
    m2 = (np.sum(k * k * wl) + nu * nu / (1 - qr) + 2 * nu * qr / (1 - qr) ** 2 + qr * (1 + qr) / (1 - qr) ** 3) / kap
    return m1, m2 - m1 * m1, kap


class IDSum:
    name = "ID-Sum (LNF adaptation)"
    trust = "augmented shuffle (same as PW-Sum)"

    def _eval(self, eps, delta, N, K, beta):
        ql, qr, nu, d = ageo_params(eps, beta, delta)
        mu, var, kap = ageo_moments(ql, qr, nu)
        mse = (1 - beta) * N / beta + var * (K + 1) * (2 * K + 1) / (6 * K * beta ** 2) + N / (4 * K * K)
        msgs_s2d = beta * N + (K + 1) * mu
        return {"K": K, "beta": beta, "ql": ql, "qr": qr, "nu": nu, "delta_achieved": d, "mu": mu, "sigma2": var,
                "mse_worst": mse, "dummies": (K + 1) * mu, "ell": ell_K(K),
                "bytes": (N + msgs_s2d) * ell_K(K) / N}

    def select_params(self, eps, delta, N, gamma):
        V = 2 * (1 + gamma) / eps ** 2
        b0 = 1 - math.exp(-eps / 2)
        betas = np.unique(np.r_[b0, 1 - (1 - b0) * np.geomspace(1, 1e-9, 60), 1.0])
        Ks = np.unique(np.r_[np.arange(1, 257), np.geomspace(256, 65536, 60).astype(int)])
        best_mse, best_ok = None, None
        for beta in betas:
            for K in Ks:
                r = self._eval(eps, delta, N, int(K), float(beta))
                if best_mse is None or r["mse_worst"] < best_mse["mse_worst"]:
                    best_mse = r
                if r["mse_worst"] <= V and (best_ok is None or r["bytes"] < best_ok["bytes"]):
                    best_ok = r
        p = best_ok if best_ok is not None else best_mse
        return {"eps": eps, "delta": delta, "N": N, "gamma": gamma, **p, "feasible": best_ok is not None,
                "target_V": V}

    def analytic_mse(self, p, qvar=None, sum_x2=None):
        K, b = p["K"], p["beta"]
        sx2 = p["N"] * K * K if sum_x2 is None else sum_x2
        q = p["N"] / (4 * K * K) if qvar is None else qvar
        return ((1 - b) / b * sx2 + p["sigma2"] / b ** 2 * K * (K + 1) * (2 * K + 1) / 6) / K ** 2 + q

    def _ageo(self, rng, p, size):
        ql, qr, nu = p["ql"], p["qr"], p["nu"]
        kmax = nu + int(math.ceil(60 / -math.log(qr))) + 1
        k = np.arange(kmax + 1)
        w = np.where(k < nu, ql ** np.maximum(nu - k, 0) if nu else 0.0, qr ** np.maximum(k - nu, 0))
        cdf = np.cumsum(w / w.sum())
        return np.searchsorted(cdf, rng.random(size))

    def simulate_error(self, y, p, rng, reps):
        K, b = p["K"], p["beta"]
        jj = np.arange(K + 1)
        out = np.empty(reps)
        for r in range(reps):
            g = np.bincount(quantise(y, K, rng), minlength=K + 1)
            T = rng.binomial(g, b) + self._ageo(rng, p, K + 1)
            out[r] = float(jj @ (T - p["mu"])) / b / K - y.sum()
        return out

    def comm_cost(self, p):
        return {"messages_per_user": 1 + (p["beta"] * p["N"] + p["dummies"]) / p["N"], "bytes_per_user": p["bytes"],
                "dummies": p["dummies"], "kind": "analytic (eq. (8) expectation; ECIES per message)"}


# ---------------------------------------------------------------- Prio3Sum + DP
def prio_upload_bytes(K):
    """Measured formula (prio 0.18.1): leader share (bits + 2 nextpow2(1+bits)) * 8, helper 32, public 0;
    DAP-17 framing with HPKE: report = 26 + 4 + 2*(1+2+32+4+16+6) + shares. Checked against the measured sizes."""
    bits = int(K).bit_length()
    lead = (bits + 2 * (1 << (bits).bit_length())) * 8
    return 26 + 4 + 2 * (1 + 2 + 32 + 4 + 16 + 6) + lead + 32


class Prio3DP:
    def __init__(self, variant):
        self.variant = variant
        self.name = f"Prio3Sum+DLap ({'A: collector adds noise' if variant == 'A' else 'B: aggregators add noise'})"
        self.trust = "two non-colluding aggregators" + (" + trusted collector" if variant == "A" else "")
        self.k = 1 if variant == "A" else 2

    def select_params(self, eps, delta, N, gamma):
        """Smallest K meeting V. Variant B has noise variance >= 2 * 2K^2/eps^2 / K^2 ~ 4/eps^2 > V for gamma < 1, so
        the target is unreachable; it then uses variant A's K (same quantisation) and is flagged infeasible."""
        V = 2 * (1 + gamma) / eps ** 2
        K, feasible = 1, True
        while dlap_var(math.exp(-eps / K)) / K ** 2 + N / (4 * K * K) > V:
            K += 1
        if self.k * dlap_var(math.exp(-eps / K)) / K ** 2 + N / (4 * K * K) > V:
            feasible = False
            if self.k == 2 and gamma >= 1:   # target may be reachable for B at large gamma: search further
                K2 = K
                while K2 < 100 * K and 2 * dlap_var(math.exp(-eps / K2)) / K2 ** 2 + N / (4 * K2 * K2) > V:
                    K2 += 1
                if 2 * dlap_var(math.exp(-eps / K2)) / K2 ** 2 + N / (4 * K2 * K2) <= V:
                    K, feasible = K2, True
        p = {"eps": eps, "delta": 0.0, "N": N, "gamma": gamma, "K": K, "p": math.exp(-eps / K), "feasible": feasible}
        p["mse_worst"] = self.analytic_mse(p)
        return p

    def analytic_mse(self, p, qvar=None):
        K = p["K"]
        return self.k * dlap_var(p["p"]) / K ** 2 + (p["N"] / (4 * K * K) if qvar is None else qvar)

    def simulate_error(self, y, p, rng, reps):
        K = p["K"]
        out = np.empty(reps)
        for r in range(reps):
            x = quantise(y, K, rng)
            out[r] = (x.sum() + dlap(rng, p["p"], self.k).sum()) / K - y.sum()
        return out

    def comm_cost(self, p):
        up = prio_upload_bytes(p["K"])
        s2s = 26 + 4 + (1 + 2 + 32 + 4 + 6 + 32 + 16) + 1 + 4 + 24 + 16 + 1 + 1 + 4
        return {"messages_per_user": 1, "bytes_per_user": up + s2s, "upload_bytes": up, "server_to_server": s2s,
                "kind": "analytic DAP-17 framing (cross-checked with the measured sizes)"}
