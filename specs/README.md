# Baseline specifications

One file per method: the source paper (arXiv version and checksum), the theorems used for privacy and parameters, the estimator,
the communication accounting, and every deviation from the original construction.

| Method | File | Privacy and parameters | Notes |
|---|---|---|---|
| CDP, LDP (Laplace) | `cdp_ldp_pw.md` | Laplace mechanism | no tunable parameter |
| PW-Sum | `cdp_ldp_pw.md` | Theorems 1-3 and Proposition 1 of the paper | parameter rule of Section V |
| GKMPS | `gkmps.md` | Theorems 2, 8, 10-12 of Ghazi et al. | explicit analytic recipe; the paper's numerical parameter search is not reproduced, so its message count is an upper bound |
| BBGN | `bbgn.md` | Corollary 6.1 of Balle et al. | parallel-Shuffler model; preconditions m >= 3, N >= 19, sigma >= 1 checked per configuration |
| ID-Sum | `id_sum.md` | Lemmas 1-2 and Theorems 2, 6 of LNF | our adaptation of LNF to summation |
| Prio3Sum + discrete Laplace | `prio3sum_dp.md` | geometric mechanism; VDAF Prio3Sum (libprio 0.18.1) | variants A and B differ in who adds noise |

The trust models differ across methods; the experiments do not claim equal collusion guarantees.
