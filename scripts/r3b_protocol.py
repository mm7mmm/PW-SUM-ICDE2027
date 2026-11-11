"""Protocol-level malicious submissions, with real public-key encryption.

Message layer mirrors the Rust ECIES in rust/src/main.rs: ciphertext = epk (32) ||
ChaCha20-Poly1305(pt, nonce 0) || tag (16); key = HKDF-SHA256(salt = epk || server_pk, ikm = X25519(esk, server_pk),
info = HKDF_INFO). Plaintext = value as ceil(bit_length(K)/8) big-endian bytes.
Shuffler (Algorithm 1): accepts messages only from roster identities over the authenticated channel, keeps at most
ONE message per identity (the first), pads missing slots with fresh Enc(0), adds H_j fresh Enc(j), shuffles.
Server: decrypts; a message that fails to decrypt, has the wrong length, or decodes to a value > K counts as 0.
Every scenario reuses the same honest data, the same honest quantisation and the same Shuffler noise (V, xi).
Checks: the Server receives exactly N + M messages; |estimate - clean| <= (#controlled identities) * X_max
(Theorem 5, normalised units); decoded counts.
Writes results/m3/protocol.csv.
"""
import os as _os, sys as _sys  # repository paths: src/pwexp/paths.py
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
from pwexp.paths import PWSUM_ROOT, DATA_DIR, RUST_BIN  # noqa: E402,F401

import csv
import os
import random
import sys

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pwexp import pw_core_task as pc  # noqa: E402
from pwexp import pw_sampler as smp  # noqa: E402
import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location("s9", f"{ROOT}/scripts/s9_params_radius.py")
s9 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(s9)

HKDF_INFO = b"pw-sum-cost ecies x25519-hkdf-sha256-chacha20poly1305"
RAW = (Encoding.Raw, PublicFormat.Raw)
SEED = 20261108
N, EPS, GAMMA = 2000, 1.0, 0.5
FRAC = 0.05


def kdf(shared, epk, spk):
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=epk + spk, info=HKDF_INFO).derive(shared)


def enc(spk_obj, spk, pt):
    esk = X25519PrivateKey.generate()
    epk = esk.public_key().public_bytes(*RAW)
    key = kdf(esk.exchange(spk_obj), epk, spk)
    return epk + ChaCha20Poly1305(key).encrypt(b"\0" * 12, pt, b"")


def dec(ssk, spk, ct):
    if len(ct) < 48:
        return None
    try:
        epk = ct[:32]
        key = kdf(ssk.exchange(X25519PublicKey.from_public_bytes(epk)), epk, spk)
        return ChaCha20Poly1305(key).decrypt(b"\0" * 12, ct[32:], b"")
    except Exception:
        return None


def main():
    os.makedirs(f"{ROOT}/results/robust_v2", exist_ok=True)
    delta = 1 / N ** 2
    p = s9.work(("protocol", "any", N, EPS, "1/N^2", GAMMA))           # current paper rule
    K, zeta, t = p["K"], p["zeta"], np.array(p["t"], np.int64)
    nb = -(-K.bit_length() // 8)
    ssk = X25519PrivateKey.generate()
    spk_obj = ssk.public_key()
    spk = spk_obj.public_bytes(*RAW)
    rng = np.random.default_rng(SEED)
    y = rng.random(N)                                   # honest inputs (uniform)
    x = np.floor(y * K).astype(int) + (rng.random(N) < y * K - np.floor(y * K))
    a, b = pc.weights(K, zeta, "pw", EPS)
    _, _, V = smp.sample_s(a, b, t.astype(float), 1, rng, keep_v=1)
    U, _ = smp.integerize(V, rng.random(1))
    c = np.r_[0, np.ones(K - 1, int), 0]
    H = t + c + np.rint(U[0]).astype(int)
    M = int(H.sum())
    kappa = int(np.arange(K + 1) @ (t + c))
    roster = [f"user{i}" for i in range(N)]
    ctrl = list(rng.choice(N, int(FRAC * N), replace=False))
    dummies = [enc(spk_obj, spk, j.to_bytes(nb, "big")) for j in range(K + 1) for _ in range(H[j])]
    pyr = random.Random(SEED)

    def run(subs, label, n_ctrl):
        """subs: list of (identity, ciphertext bytes) in arrival order."""
        kept, seen, rejected_id, dup, badlen = {}, set(), 0, 0, 0
        roster_set = set(roster)
        for ident, ct in subs:
            if ident not in roster_set:
                rejected_id += 1
                continue
            if ident in seen:
                dup += 1
                continue
            seen.add(ident)
            if len(ct) != 48 + nb:          # fixed-length filter (Algorithm pw2): the slot is padded below
                badlen += 1
                continue
            kept[ident] = ct
        padded = [enc(spk_obj, spk, (0).to_bytes(nb, "big")) for ident in roster if ident not in kept]
        batch = list(kept.values()) + padded + dummies
        pyr.shuffle(batch)
        T = np.zeros(K + 1, int)
        fail = oor = wrong_len = 0
        for ct in batch:
            pt = dec(ssk, spk, ct)
            if pt is None:
                fail += 1
                v = 0
            elif len(pt) != nb:
                wrong_len += 1
                v = 0
            else:
                v = int.from_bytes(pt, "big")
                if v > K:
                    oor += 1
                    v = 0
            T[v] += 1
        est = (int(np.arange(K + 1) @ T) - kappa) / K
        return {"scenario": label, "N": N, "K": K, "M": M, "n_controlled": n_ctrl, "msgs_to_server": len(batch),
                "expected_msgs": N + M, "msgs_ok": len(batch) == N + M, "rejected_identity": rejected_id,
                "duplicates_dropped": dup, "bad_length_replaced": badlen, "padded_missing": len(padded), "decrypt_fail": fail,
                "wrong_length": wrong_len, "out_of_range": oor, "estimate": est}

    honest = {i: enc(spk_obj, spk, int(x[i]).to_bytes(nb, "big")) for i in range(N)}
    base = [(roster[i], honest[i]) for i in range(N)]
    scen = []
    clean = run(base, "clean", 0)
    scen.append(clean)

    def with_ctrl(f):
        subs = []
        for i in range(N):
            subs += f(i) if i in ctrl else [(roster[i], honest[i])]
        return subs

    scen.append(run(with_ctrl(lambda i: [(roster[i], os.urandom(48 + nb))]), "malformed ciphertext (random bytes)", len(ctrl)))
    scen.append(run(with_ctrl(lambda i: [(roster[i], honest[i][:20])]), "truncated ciphertext", len(ctrl)))
    big = (K + 1 + pyr.randrange(100)).to_bytes(nb, "big") if K + 101 < 256 ** nb else (256 ** nb - 1).to_bytes(nb, "big")
    scen.append(run(with_ctrl(lambda i: [(roster[i], enc(spk_obj, spk, big))]), "out-of-range value > K", len(ctrl)))
    scen.append(run(with_ctrl(lambda i: [(roster[i], enc(spk_obj, spk, b"\x01" * (nb + 3)))]), "oversize plaintext", len(ctrl)))
    scen.append(run(with_ctrl(lambda i: [(roster[i], enc(spk_obj, spk, K.to_bytes(nb, "big")))] * 1 +
                              [(roster[i], enc(spk_obj, spk, K.to_bytes(nb, "big"))) for _ in range(20)]),
                    "duplicate identity flooding (21 msgs of K)", len(ctrl)))
    scen.append(run(with_ctrl(lambda i: [(roster[i], enc(spk_obj, spk, K.to_bytes(nb, "big")))]), "in-range poisoning: all K", len(ctrl)))
    scen.append(run(with_ctrl(lambda i: [(roster[i], enc(spk_obj, spk, (0).to_bytes(nb, "big")))]), "in-range poisoning: all 0", len(ctrl)))
    scen.append(run(base + [(f"sybil{j}", enc(spk_obj, spk, K.to_bytes(nb, "big"))) for j in range(500)],
                    "500 non-roster identities", 0))
    scen.append(run(with_ctrl(lambda i: []), "controlled users send nothing", len(ctrl)))
    for r in scen:
        r["shift"] = r["estimate"] - clean["estimate"]
        r["bound_theorem5"] = float(r["n_controlled"])
        r["within_bound"] = abs(r["shift"]) <= r["bound_theorem5"] + 1e-9
        r["truth_sum_y"] = float(y.sum())
    with open(f"{ROOT}/results/robust_v2/protocol.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scen[0]))
        w.writeheader()
        w.writerows(scen)
    for r in scen:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
