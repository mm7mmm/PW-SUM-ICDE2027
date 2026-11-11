# CDP-Laplace, LDP-Laplace, PW-Sum (reference specs)

## CDP-Laplace
- Trusted curator computes S = Σ y_i (clipped, normalised, y_i ∈ [0,1]) and releases S + Lap(1/ε).
- Privacy: Laplace mechanism (Dwork–Roth 2014, Thm 3.6), sensitivity 1 under replace-one ⇒ pure ε-DP.
- MSE = 2/ε² exactly; no quantisation. Communication: one value per user to the curator (not comparable).
- Simulation: Laplace draw per repetition (60/60 runs within 5 % of 2/ε²).

## LDP-Laplace
- User releases y_i + Lap(1/ε) (ε-LDP, hence ε-DP for replace-one); server sums.
- MSE = 2N/ε² exactly. Simulated per user (N draws per repetition), no shortcut.
- Collusion: each user's privacy is local, unaffected by any collusion.

## PW-Sum
- Paper: Algorithm 1, position weights, integerization, Theorem 1 (full-view privacy), Theorem 2 (utility),
  Theorem 3 (messages), Proposition 1 (bounded poisoning).
- Parameters: the rule of Section V, checked with interval arithmetic (`scripts/s9_params_radius.py`).
- Error: Δ_q ⌊s(V) + ξ⌋ + quantisation; V from the exact endpoint-anchor sampler; validated end-to-end (`scripts/pu_e2e.py`).
