//! Cost benchmarks for PW-Sum (ECIES message path) and Prio3Sum (libprio).
//!
//! Subcommands (all print one JSON object per line on stdout; raw repetition times are emitted so
//! that medians/IQRs are computed downstream):
//!   ecies-len <K...>                      measured ciphertext length l(K)
//!   pw <N> <K> <M> <reps> <seed>          PW-Sum user / Shuffler / Server timings
//!   pw-e2e <params.json> <seed>           end-to-end correctness with a real dummy histogram H
//!   prio <N> <K> <reps> <seed>            Prio3Sum bytes and timings
//!   prio-check <N> <K> <seed>             Prio3Sum correctness (aggregate before noise == true sum)

mod cost2;

use std::time::Instant;

use chacha20poly1305::{AeadInOut, ChaCha20Poly1305, Key, KeyInit, Nonce};
use hkdf::Hkdf;
use prio::codec::{Encode, ParameterizedDecode};
use prio::vdaf::prio3::{Prio3InputShare, Prio3PublicShare, Prio3Sum};
use prio::vdaf::{Aggregatable, Aggregator, Client, Collector, VerifyTransition};
use rand::RngExt;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::Sha256;
use x25519_dalek::{PublicKey, StaticSecret};

const HKDF_INFO: &[u8] = b"pw-sum-cost ecies x25519-hkdf-sha256-chacha20poly1305";

// ---------------------------------------------------------------- ECIES
//
// Ciphertext = ephemeral X25519 public key (32) || ChaCha20-Poly1305(plaintext) || tag (16).
// The AEAD key is HKDF-SHA256(salt = epk || server_pk, ikm = X25519(esk, server_pk)).
// The nonce is fixed to zero: every message uses a fresh ephemeral key pair, hence a fresh,
// independent AEAD key, so a (key, nonce) pair is never reused.

struct Server {
    sk: StaticSecret,
    pk: PublicKey,
}

fn derive_cipher(shared: &[u8; 32], epk: &[u8; 32], server_pk: &[u8; 32]) -> ChaCha20Poly1305 {
    let mut salt = [0u8; 64];
    salt[..32].copy_from_slice(epk);
    salt[32..].copy_from_slice(server_pk);
    let hk = Hkdf::<Sha256>::new(Some(&salt), shared);
    let mut key = [0u8; 32];
    hk.expand(HKDF_INFO, &mut key).expect("hkdf expand");
    ChaCha20Poly1305::new(&Key::from(key))
}

fn ecies_encrypt(rng: &mut ChaCha20Rng, server_pk: &PublicKey, pt: &[u8]) -> Vec<u8> {
    let esk = StaticSecret::random_from_rng(rng);
    let epk = PublicKey::from(&esk);
    let shared = esk.diffie_hellman(server_pk);
    let cipher = derive_cipher(shared.as_bytes(), epk.as_bytes(), server_pk.as_bytes());
    let mut buf = Vec::with_capacity(pt.len() + 16);
    buf.extend_from_slice(pt);
    cipher.encrypt_in_place(&Nonce::default(), b"", &mut buf).expect("aead encrypt");
    let mut out = Vec::with_capacity(32 + buf.len());
    out.extend_from_slice(epk.as_bytes());
    out.extend_from_slice(&buf);
    out
}

fn ecies_decrypt(server: &Server, ct: &[u8]) -> Option<Vec<u8>> {
    if ct.len() < 48 {
        return None;
    }
    let mut epk_bytes = [0u8; 32];
    epk_bytes.copy_from_slice(&ct[..32]);
    let epk = PublicKey::from(epk_bytes);
    let shared = server.sk.diffie_hellman(&epk);
    let cipher = derive_cipher(shared.as_bytes(), &epk_bytes, server.pk.as_bytes());
    let mut buf = ct[32..].to_vec();
    cipher.decrypt_in_place(&Nonce::default(), b"", &mut buf).ok()?;
    Some(buf)
}

/// Plaintext length in bytes: ceil(bitlen(K)/8) == ceil(log2(K+1)/8) for K >= 1.
fn pt_len(k: u64) -> usize {
    ((64 - k.leading_zeros() as usize) + 7) / 8
}

fn encode_value(v: u64, len: usize) -> Vec<u8> {
    v.to_be_bytes()[8 - len..].to_vec()
}

/// Bounded decoder of Algorithm 1: decryption failure, wrong length or value > K count as 0.
fn bounded_decode(server: &Server, ct: &[u8], k: u64, len: usize) -> u64 {
    match ecies_decrypt(server, ct) {
        Some(pt) if pt.len() == len => {
            let mut b = [0u8; 8];
            b[8 - len..].copy_from_slice(&pt);
            let v = u64::from_be_bytes(b);
            if v <= k { v } else { 0 }
        }
        _ => 0,
    }
}

fn new_server(seed: u64) -> Server {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let sk = StaticSecret::random_from_rng(&mut rng);
    let pk = PublicKey::from(&sk);
    Server { sk, pk }
}

/// Uniform Fisher-Yates shuffle driven by a ChaCha20 CSPRNG.
fn fisher_yates<T>(v: &mut [T], rng: &mut ChaCha20Rng) {
    for i in (1..v.len()).rev() {
        let j = rng.random_range(0..=i);
        v.swap(i, j);
    }
}

/// Server side: decrypt all ciphertexts, bounded-decode, count, return (counts, estimate).
fn server_count(server: &Server, cts: &[Vec<u8>], k: u64, len: usize, kappa: i128) -> (Vec<u64>, i128) {
    let mut counts = vec![0u64; k as usize + 1];
    for ct in cts {
        counts[bounded_decode(server, ct, k, len) as usize] += 1;
    }
    let s: i128 = counts.iter().enumerate().map(|(j, &c)| j as i128 * c as i128).sum();
    (counts, s - kappa)
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn cmd_ecies_len(ks: &[u64]) {
    let server = new_server(1);
    let mut rng = ChaCha20Rng::seed_from_u64(2);
    for &k in ks {
        let len = pt_len(k);
        let ct = ecies_encrypt(&mut rng, &server.pk, &encode_value(k, len));
        assert_eq!(bounded_decode(&server, &ct, k, len), k);
        println!("{}", json!({"cmd": "ecies-len", "K": k, "pt_len": len, "ct_len": ct.len()}));
    }
}

fn cmd_pw(n: usize, k: u64, m: usize, reps: usize, seed: u64) {
    let server = new_server(seed);
    let len = pt_len(k);
    // Dataset (not timed): user inputs and their ciphertexts, built in parallel.
    let mut rng = ChaCha20Rng::seed_from_u64(seed + 1);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    let user_cts: Vec<Vec<u8>> = xs
        .par_iter()
        .enumerate()
        .map(|(i, &x)| {
            let mut r = ChaCha20Rng::seed_from_u64(seed ^ (0x9e37_79b9 + i as u64));
            ecies_encrypt(&mut r, &server.pk, &encode_value(x, len))
        })
        .collect();
    // Dummy histogram used for timing only (values do not affect cost): M messages spread over 0..=K.
    let dummy_vals: Vec<u64> = (0..m).map(|i| (i as u64) % (k + 1)).collect();
    let kappa: i128 = dummy_vals.iter().map(|&v| v as i128).sum();
    let s_true: i128 = xs.iter().map(|&x| x as i128).sum();

    // Warm-up.
    {
        let mut r = ChaCha20Rng::seed_from_u64(seed + 7);
        for _ in 0..2000 {
            let ct = ecies_encrypt(&mut r, &server.pk, &encode_value(1, len));
            std::hint::black_box(bounded_decode(&server, &ct, k, len));
        }
    }

    let enc_batch = 20_000usize;
    let mut user_enc = Vec::new();
    let mut dummy_gen = Vec::new();
    let mut shuffle = Vec::new();
    let mut server_t = Vec::new();
    let mut last_est = 0i128;
    for rep in 0..reps {
        let mut r = ChaCha20Rng::seed_from_u64(seed + 100 + rep as u64);
        let t = Instant::now();
        for i in 0..enc_batch {
            std::hint::black_box(ecies_encrypt(&mut r, &server.pk, &encode_value(i as u64 % (k + 1), len)));
        }
        user_enc.push(secs(t) / enc_batch as f64);

        let t = Instant::now();
        let dummies: Vec<Vec<u8>> = dummy_vals
            .iter()
            .map(|&v| ecies_encrypt(&mut r, &server.pk, &encode_value(v, len)))
            .collect();
        dummy_gen.push(secs(t));

        let mut all: Vec<Vec<u8>> = Vec::with_capacity(n + m);
        all.extend(user_cts.iter().cloned());
        all.extend(dummies);
        let t = Instant::now();
        fisher_yates(&mut all, &mut r);
        shuffle.push(secs(t));

        let t = Instant::now();
        let (_, est) = server_count(&server, &all, k, len, kappa);
        server_t.push(secs(t));
        last_est = est;
    }
    // Multi-threaded throughput (reference only), one repetition each.
    let threads = rayon::current_num_threads();
    let mut r = ChaCha20Rng::seed_from_u64(seed + 999);
    let t = Instant::now();
    let dummies_mt: Vec<Vec<u8>> = dummy_vals
        .par_iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut rr = ChaCha20Rng::seed_from_u64(seed ^ (0x5151_0000 + i as u64));
            ecies_encrypt(&mut rr, &server.pk, &encode_value(v, len))
        })
        .collect();
    let dummy_mt = secs(t);
    let mut all: Vec<Vec<u8>> = user_cts.iter().cloned().chain(dummies_mt).collect();
    fisher_yates(&mut all, &mut r);
    let t = Instant::now();
    let decoded: Vec<u64> = all.par_iter().map(|ct| bounded_decode(&server, ct, k, len)).collect();
    let server_mt = secs(t);
    let est_mt: i128 = decoded.iter().map(|&v| v as i128).sum::<i128>() - kappa;
    println!(
        "{}",
        json!({
            "cmd": "pw", "N": n, "K": k, "M": m, "reps": reps, "seed": seed, "ct_len": user_cts[0].len(),
            "user_enc_per_msg_s": user_enc, "shuffler_dummy_gen_s": dummy_gen, "shuffler_shuffle_s": shuffle,
            "server_decrypt_count_s": server_t, "threads": threads,
            "mt_dummy_gen_s": dummy_mt, "mt_server_decrypt_s": server_mt,
            "est_minus_true_matches": last_est == s_true && est_mt == s_true
        })
    );
}

fn cmd_pw_e2e(path: &str, seed: u64) {
    let p: Value = serde_json::from_str(&std::fs::read_to_string(path).expect("read params")).expect("json");
    let k = p["K"].as_u64().unwrap();
    let n = p["N"].as_u64().unwrap() as usize;
    let h: Vec<u64> = p["H"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let kappa = p["kappa"].as_i64().unwrap() as i128;
    let expected_err = p["expected_err"].as_i64().unwrap() as i128;
    assert_eq!(h.len() as u64, k + 1);
    let server = new_server(seed);
    let len = pt_len(k);
    let mut rng = ChaCha20Rng::seed_from_u64(seed + 1);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    let s_true: i128 = xs.iter().map(|&x| x as i128).sum();
    let mut all: Vec<Vec<u8>> = xs.iter().map(|&x| ecies_encrypt(&mut rng, &server.pk, &encode_value(x, len))).collect();
    for (j, &hj) in h.iter().enumerate() {
        for _ in 0..hj {
            all.push(ecies_encrypt(&mut rng, &server.pk, &encode_value(j as u64, len)));
        }
    }
    let total = all.len();
    fisher_yates(&mut all, &mut rng);
    let (counts, est) = server_count(&server, &all, k, len, kappa);
    let ok = est == s_true + expected_err && counts.iter().sum::<u64>() as usize == total;
    println!(
        "{}",
        json!({"cmd": "pw-e2e", "N": n, "K": k, "M": total - n, "S": s_true as i64, "estimate": est as i64,
               "expected_err": expected_err as i64, "estimate_equals_S_plus_err": ok, "seed": seed})
    );
}

// ---------------------------------------------------------------- Prio3Sum

const CTX: &[u8] = b"pw-sum-cost";

struct Report {
    nonce: [u8; 16],
    public_share: Vec<u8>,
    shares: [Vec<u8>; 2],
}

fn shard_all(vdaf: &Prio3Sum, xs: &[u64], rng: &mut ChaCha20Rng) -> Vec<Report> {
    xs.iter()
        .map(|x| {
            let nonce: [u8; 16] = rng.random();
            let (ps, is) = vdaf.shard(CTX, x, &nonce).expect("shard");
            Report {
                nonce,
                public_share: ps.get_encoded().unwrap(),
                shares: [is[0].get_encoded().unwrap(), is[1].get_encoded().unwrap()],
            }
        })
        .collect()
}

type AggShare = <Prio3Sum as prio::vdaf::Vdaf>::AggregateShare;

/// Both aggregators: decode, verify_init, combine verifier shares, verify_next, accumulate.
/// Returns (the two aggregate shares, verifier-share bytes per aggregator, verifier-message bytes).
fn aggregate_all(vdaf: &Prio3Sum, vk: &[u8; 32], reports: &[Report]) -> ([AggShare; 2], usize, usize) {
    let mut agg = [vdaf.aggregate_init(&()), vdaf.aggregate_init(&())];
    let (mut vs_len, mut vm_len) = (0usize, 0usize);
    for r in reports {
        let ps = Prio3PublicShare::get_decoded_with_param(vdaf, &r.public_share).expect("ps");
        let mut states = Vec::with_capacity(2);
        let mut vshares = Vec::with_capacity(2);
        for agg_id in 0..2 {
            let is = Prio3InputShare::get_decoded_with_param(&(vdaf, agg_id), &r.shares[agg_id]).expect("is");
            let (st, vs) = vdaf.verify_init(vk, CTX, agg_id, &(), &r.nonce, &ps, &is).expect("verify_init");
            vs_len = vs.get_encoded().unwrap().len();
            states.push(st);
            vshares.push(vs);
        }
        let msg = vdaf.verifier_shares_to_message(CTX, &(), vshares).expect("message");
        vm_len = msg.get_encoded().unwrap().len();
        for (agg_id, st) in states.into_iter().enumerate() {
            match vdaf.verify_next(CTX, st, msg.clone()).expect("verify_next") {
                VerifyTransition::Finish(out) => agg[agg_id].accumulate(&out).expect("accumulate"),
                _ => panic!("Prio3 is one-round"),
            }
        }
    }
    (agg, vs_len, vm_len)
}

fn unshard(vdaf: &Prio3Sum, agg: [AggShare; 2], n: usize) -> u64 {
    vdaf.unshard(&(), agg, n).expect("unshard")
}

/// Discrete Laplace with parameter eps/K, sampled as the difference of two geometric variables.
/// (Inverse-CDF floating-point sampling; adequate for timing, not a hardened DP sampler.)
fn dlap(rng: &mut ChaCha20Rng, eps_over_k: f64) -> i64 {
    let q = (-eps_over_k).exp();
    let geo = |r: &mut ChaCha20Rng| -> i64 {
        let u: f64 = r.random();
        ((1.0 - u).ln() / q.ln()).floor() as i64
    };
    geo(rng) - geo(rng)
}

fn cmd_prio(n: usize, k: u64, reps: usize, seed: u64) {
    let vdaf = Prio3Sum::new_sum(2, k).expect("new_sum");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    let mut vk = [0u8; 32];
    rng.fill(&mut vk[..]);
    // Warm-up on a small batch.
    let warm = shard_all(&vdaf, &xs[..n.min(2000)], &mut rng);
    let (wagg, _, _) = aggregate_all(&vdaf, &vk, &warm);
    let _ = unshard(&vdaf, wagg, warm.len());

    let (mut t_shard, mut t_agg, mut t_unshard, mut t_noise) = (vec![], vec![], vec![], vec![]);
    let mut sizes = json!({});
    let mut ok = true;
    let s_true: u64 = xs.iter().sum();
    for _ in 0..reps {
        let t = Instant::now();
        let reports = shard_all(&vdaf, &xs, &mut rng);
        t_shard.push(secs(t));
        let t = Instant::now();
        let (agg, vs_len, vm_len) = aggregate_all(&vdaf, &vk, &reports);
        t_agg.push(secs(t));
        let t = Instant::now();
        let res = unshard(&vdaf, agg, reports.len());
        t_unshard.push(secs(t));
        ok &= res == s_true;
        let t = Instant::now();
        let noise = dlap(&mut rng, 1.0 / k as f64);
        std::hint::black_box(res as i64 + noise);
        t_noise.push(secs(t));
        sizes = json!({"public_share": reports[0].public_share.len(), "input_share_leader": reports[0].shares[0].len(),
                       "input_share_helper": reports[0].shares[1].len(), "verifier_share": vs_len,
                       "verifier_message": vm_len});
    }
    // HPKE-equivalent cost (reference): one ECIES encryption per input share at the client, one
    // decryption per aggregator, with the same primitive as PW-Sum and plaintext = encoded share.
    let server = new_server(seed + 5);
    let lens = [sizes["input_share_leader"].as_u64().unwrap() as usize, sizes["input_share_helper"].as_u64().unwrap() as usize];
    let reps_h = 20_000usize;
    let mut hpke_client = vec![];
    let mut hpke_agg = vec![];
    for _ in 0..reps {
        let pts: Vec<Vec<u8>> = lens.iter().map(|&l| vec![7u8; l]).collect();
        let t = Instant::now();
        let mut cts = Vec::with_capacity(2 * reps_h);
        for _ in 0..reps_h {
            for pt in &pts {
                cts.push(ecies_encrypt(&mut rng, &server.pk, pt));
            }
        }
        hpke_client.push(secs(t) / reps_h as f64);
        let t = Instant::now();
        for ct in &cts {
            std::hint::black_box(ecies_decrypt(&server, ct));
        }
        hpke_agg.push(secs(t) / reps_h as f64);
    }
    println!(
        "{}",
        json!({"cmd": "prio", "N": n, "K": k, "reps": reps, "seed": seed, "vdaf": "Prio3Sum (prio 0.18.1, VDAF draft-18)",
               "encoded_sizes": sizes, "client_shard_all_s": t_shard, "aggregators_verify_aggregate_all_s": t_agg,
               "unshard_s": t_unshard, "noise_s": t_noise,
               "hpke_equiv_client_per_report_s": hpke_client, "hpke_equiv_aggregators_per_report_s": hpke_agg,
               "aggregate_equals_true_sum": ok})
    );
}

fn cmd_prio_check(n: usize, k: u64, seed: u64) {
    let vdaf = Prio3Sum::new_sum(2, k).expect("new_sum");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    let mut vk = [0u8; 32];
    rng.fill(&mut vk[..]);
    let reports = shard_all(&vdaf, &xs, &mut rng);
    let (agg, _, _) = aggregate_all(&vdaf, &vk, &reports);
    let res = unshard(&vdaf, agg, reports.len());
    let s: u64 = xs.iter().sum();
    println!("{}", json!({"cmd": "prio-check", "N": n, "K": k, "aggregate": res, "true_sum": s, "equal": res == s, "seed": seed}));
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let u = |i: usize| a[i].parse::<u64>().expect("integer argument");
    match a.get(1).map(String::as_str) {
        Some("ecies-len") => cmd_ecies_len(&a[2..].iter().map(|s| s.parse().unwrap()).collect::<Vec<u64>>()),
        Some("pw") => cmd_pw(u(2) as usize, u(3), u(4) as usize, u(5) as usize, u(6)),
        Some("pw-e2e") => cmd_pw_e2e(&a[2], u(3)),
        Some("prio") => cmd_prio(u(2) as usize, u(3), u(4) as usize, u(5)),
        Some("prio-check") => cmd_prio_check(u(2) as usize, u(3), u(4)),
        Some("pw2") => cost2::cmd_pw2(u(2) as usize, u(3), u(4) as usize, u(5) as usize, u(6) as usize, u(7)),
        Some("prio2") => cost2::cmd_prio2(u(2) as usize, u(3), u(4) as usize, u(5) as usize, u(6)),
        Some("pw3") => cost2::cmd_pw3(u(2) as usize, u(3), u(4) as usize, u(5) as usize, u(6) as usize, u(7)),
        Some("pw4") => cost2::cmd_pw4(u(2) as usize, u(3), &std::env::args().nth(4).unwrap(), u(5) as usize, u(6) as usize, u(7)),
        Some("dbq_pw") => cost2::cmd_dbq_pw(&std::env::args().nth(2).unwrap(), u(3), &std::env::args().nth(4).unwrap(), u(5) as usize, std::env::args().nth(6).unwrap().parse().unwrap(), u(7)),
        Some("dbq_prio") => cost2::cmd_dbq_prio(&std::env::args().nth(2).unwrap(), u(3), u(4) as usize, std::env::args().nth(5).unwrap().parse().unwrap(), u(6)),
        _ => eprintln!("usage: see module docs"),
    }
}
