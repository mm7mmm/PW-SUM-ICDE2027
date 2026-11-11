# Prio3Sum + DP noise (two variants)

Sources: VDAF Prio3Sum as implemented by `prio` 0.18.1 (VDAF draft-18 per crate docs); discrete Laplace / geometric mechanism (Ghosh, Roughgarden, Sundararajan 2009).

## Protocol
- Client: quantise y_i to x̃_i ∈ {0..K} (same quantiser), shard into two input shares with an FLP validity proof,
  HPKE-encrypt each share to its aggregator (DAP: X25519-HKDF-SHA256, HKDF-SHA256, AES-128-GCM).
- Leader and Helper: decrypt, verify, aggregate; Collector: unshard → exact Σ x̃_i.
- **Variant A (trusted collector adds noise)**: collector adds DLap(ε/K) to the exact sum. The collector sees the exact
  sum; DP holds only for the released output, not against the collector.
- **Variant B (collector only gets a noisy result)**: each aggregator adds an independent DLap(ε/K) to its aggregate
  share before sending it; the collector sees Σ x̃ + Z_L + Z_H. DP against the collector (and against one aggregator
  colluding with it) holds as long as ONE aggregator is honest (its noise alone gives ε-DP; the other noise is
  post-processing). Cost: MSE noise term doubles.

## Privacy
Pure ε-DP via the discrete Laplace mechanism with sensitivity K on Z under replace-one neighbours (values in {0..K}
are enforced by the FLP proof — input validity, not privacy). δ = 0. Real-valued inputs: quantisation kernel mixing
(as in PW-Sum Theorem 1 step 6) keeps ε.

## Analytic MSE
v(K) = 2q/(1−q)², q = e^{−ε/K}. Variant A: v/K² + Σf(1−f)/K². Variant B: 2v/K² + Σf(1−f)/K².

## Communication
Measured with the Rust implementation in `rust/`: upload per report (DAP-17 framing, real HPKE), server-to-server per report with the
Leader forwarding the Helper's encrypted share. Parameter K: smallest K meeting V = 2(1+γ)/ε² (upload bytes non-decreasing in K).

## Simulation
Error = Z (A) or Z_L + Z_H (B) plus per-user quantisation error, Z ~ DLap(ε/K) sampled as the difference of two
geometric variables; the secret-sharing layer is exact (aggregate equals Σ x̃, verified in the Rust runs).

## Deviations / questions
- The noise sampler is floating-point; not a hardened DP sampler.
- Trust models differ from PW-Sum (two non-colluding aggregators vs. non-colluding Shuffler/Server).
