# ID-Sum — per-category independent dummies, an adaptation of the LNF framework to summation

**This is our adaptation, not the original LNF algorithm** (LNF/SAGeo-Shuffle estimate frequencies, not sums).
Source of the framework: Murakami, Sei, Eriguchi, "Augmented Shuffle Protocols for Accurate and Robust Frequency
Estimation under Differential Privacy", IEEE S&P 2025, arXiv:2504.07362v1 (sha256 prefix
959bbe6e01a2fd57).

## Protocol (augmented shuffle, same trust model as PW-Sum)
- Categories j ∈ {0..K} (d = K+1), user sends Enc(x̃_i) of the quantised value (same quantiser as PW-Sum).
- Shuffler (Algorithm 1, G_{D,β}): keeps each input independently with probability β; for each category j adds
  z_j ~ D dummies of value j; shuffles. D = AGeo(ν, q_l, q_r) (SAGeo-Shuffle, Sec. 4.5) or, at β = 1 − e^{−ε/2},
  the one-sided geometric (S1Geo-Shuffle, Sec. 4.6, δ = 0).
- Server: histogram T_j; estimator (our adaptation of eq. (5) to the sum): Ŝ = Σ_j j (T_j − μ)/β, real estimate Ŝ/K.

## Privacy (theorem numbers in arXiv:2504.07362v1)
- Lemma 1, Lemma 2: the released histogram is T = Σ_i a_i e_{x_i} + z, a_i ~ Ber(β).
- **Theorem 2**: if the binary-input mechanism M_{D,β}(x) = a x + z is (ε/2, δ/2)-DP then the protocol is (ε,δ)-DP,
  also for Ω-neighbouring databases (collusion with users); Ŝ is post-processing.
- **Theorem 6**: q_l = (e^{−ε/2} − 1 + β)/β, q_r = β/(e^{ε/2} − 1 + β); M_{AGeo,β} is (ε/2, δ/2)-DP with
  δ = 0 if β = 1 − e^{−ε/2}, else δ = (2/κ) q_l^ν (1 − e^{ε/2} + β e^{ε/2}); κ = q_l(1−q_l^ν)/(1−q_l) + 1/(1−q_r).
  Parameter ν = smallest integer with δ ≤ δ_target (paper: "uniquely determined as a minimum value").
- AGeo pmf (Sec. 4.5): Pr(k) = q_l^{ν−k}/κ for k = 0..ν−1, q_r^{k−ν}/κ for k ≥ ν.
- Neighbouring: replace-one.

## Analytic MSE (exact for given data)
Var(Σ_j j T_j) = β(1−β) Σ_i x̃_i² + σ² Σ_j j², so with σ² = Var(AGeo) (computed exactly from the pmf, not the
Proposition 2 upper bound):
MSE_real = [(1−β)/β · Σ_i x̃_i² + σ²/β² · K(K+1)(2K+1)/6] / K² + Σ_i f_i(1−f_i)/K².
Worst case over inputs (x̃_i = K): (1−β)N/β + σ² (K+1)(2K+1)/(6 K β²) + N/(4K²).
(Unbiased: E T_j = β g_j + μ.) The σ² Σ j² term is Θ(K³)/K² — the paper's Theorem 3 (independent dummies) applies.

## Communication (eq. (8) adapted)
User→shuffler N messages, shuffler→server βN + (K+1)μ in expectation; bytes with ECIES ℓ(K) per message.

## Parameter selection
For each ε, δ, N: grid over K (1..4096, then sparse) and β ∈ {1 − e^{−ε/2}} ∪ (1 − e^{−ε/2}, 1] (log-grid of 1−β,
200 points), ν from Theorem 6; minimise the worst-case MSE (reported as the best achievable if the target V cannot be
met) and, among points meeting V (if any), the bytes.

## Simulation (no shortcut needed)
T is simulated exactly: Binomial thinning of the per-category counts g_j (equivalent to per-user Bernoulli sampling,
since only counts matter) and per-category AGeo draws by inverse CDF; estimator as above.

## Deviations
- Sum estimator instead of frequency estimator; quantised numeric domain as categories.
- We use SAGeo/S1Geo only (SBin is dominated per the paper's own Sec. 5).

## Questions
- Q: is β < 1 ever useful for sums? Sampling adds (1−β)/β·Σx̃²/K² ≈ (1−β)N/β·E[y²]; for N ≥ 1e4 it forces β ≈ 1.
  Handled by the search; reported.
