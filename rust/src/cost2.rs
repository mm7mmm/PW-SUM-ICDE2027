//! Cost benchmarks: role-separated timings, real HPKE for Prio3Sum, multi-threading.
//!
//!   pw2 <N> <K> <M> <reps> <threads> <seed>     PW-Sum: user per message; Shuffler (dummy encryption, shuffle);
//!                                               Server (decrypt + bounded decode + count). threads = 1 -> single
//!                                               thread; > 1 -> rayon pool of that size for dummy encryption and
//!                                               Server decryption (shuffle stays sequential Fisher-Yates).
//!   prio2 <N> <K> <reps> <threads> <seed>       Prio3Sum with real HPKE (RFC 9180, X25519-HKDF-SHA256 /
//!                                               HKDF-SHA256 / AES-128-GCM, the DAP-mandated suite). Roles:
//!                                               client per report, Leader, Helper, Collector.
//! Every repetition's raw time is printed; medians are taken downstream. Nothing is summed over users to form an
//! "end-to-end" time.
use std::time::Instant;

use hpke::aead::AesGcm128;
use hpke::kdf::HkdfSha256;
use hpke::kem::X25519HkdfSha256;
use hpke::{Deserializable, Kem as KemTrait, OpModeR, OpModeS, Serializable};
use prio::codec::{Encode, ParameterizedDecode};
use prio::vdaf::prio3::{Prio3InputShare, Prio3PublicShare, Prio3Sum};
use prio::vdaf::{Aggregatable, Aggregator, Client, Collector, VerifyTransition};
use rand::RngExt;
use rand_chacha::ChaCha20Rng;
use rand_chacha_09::rand_core::SeedableRng as Seedable09;
use rand_core::SeedableRng;
use rayon::prelude::*;
use serde_json::json;

use crate::{bounded_decode, dlap, ecies_encrypt, encode_value, fisher_yates, new_server, pt_len, secs, CTX};

type Kem = X25519HkdfSha256;
type HRng = rand_chacha_09::ChaCha20Rng;

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("pool")
}

// ---------------------------------------------------------------- PW-Sum

pub fn cmd_pw2(n: usize, k: u64, m: usize, reps: usize, threads: usize, seed: u64) {
    let tp = pool(threads.max(1));
    let server = new_server(seed);
    let len = pt_len(k);
    let mut rng = ChaCha20Rng::seed_from_u64(seed + 1);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    // dataset (not timed)
    let user_cts: Vec<Vec<u8>> = pool(64).install(|| {
        xs.par_iter()
            .enumerate()
            .map(|(i, &x)| {
                let mut r = ChaCha20Rng::seed_from_u64(seed ^ (0x9e37_79b9 + i as u64));
                ecies_encrypt(&mut r, &server.pk, &encode_value(x, len))
            })
            .collect()
    });
    let dummy_vals: Vec<u64> = (0..m).map(|i| (i as u64) % (k + 1)).collect();
    let kappa: i128 = dummy_vals.iter().map(|&v| v as i128).sum();
    let s_true: i128 = xs.iter().map(|&x| x as i128).sum();
    {
        let mut r = ChaCha20Rng::seed_from_u64(seed + 7);
        for _ in 0..2000 {
            let ct = ecies_encrypt(&mut r, &server.pk, &encode_value(1, len));
            std::hint::black_box(bounded_decode(&server, &ct, k, len));
        }
    }
    let (mut user_t, mut dummy_t, mut shuf_t, mut serv_t) = (vec![], vec![], vec![], vec![]);
    let mut ok = true;
    let batch = 20_000usize;
    for rep in 0..reps {
        let mut r = ChaCha20Rng::seed_from_u64(seed + 100 + rep as u64);
        let t = Instant::now();
        for i in 0..batch {
            std::hint::black_box(ecies_encrypt(&mut r, &server.pk, &encode_value(i as u64 % (k + 1), len)));
        }
        user_t.push(secs(t) / batch as f64);

        let t = Instant::now();
        let dummies: Vec<Vec<u8>> = if threads <= 1 {
            dummy_vals.iter().map(|&v| ecies_encrypt(&mut r, &server.pk, &encode_value(v, len))).collect()
        } else {
            tp.install(|| {
                dummy_vals
                    .par_iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let mut rr = ChaCha20Rng::seed_from_u64((seed + rep as u64) ^ (0x5151_0000 + i as u64));
                        ecies_encrypt(&mut rr, &server.pk, &encode_value(v, len))
                    })
                    .collect()
            })
        };
        dummy_t.push(secs(t));

        let mut all: Vec<Vec<u8>> = Vec::with_capacity(n + m);
        all.extend(user_cts.iter().cloned());
        all.extend(dummies);
        let t = Instant::now();
        fisher_yates(&mut all, &mut r);
        shuf_t.push(secs(t));

        let t = Instant::now();
        let counts: Vec<u64> = if threads <= 1 {
            let mut c = vec![0u64; k as usize + 1];
            for ct in &all {
                c[bounded_decode(&server, ct, k, len) as usize] += 1;
            }
            c
        } else {
            tp.install(|| {
                all.par_iter()
                    .fold(|| vec![0u64; k as usize + 1], |mut c, ct| {
                        c[bounded_decode(&server, ct, k, len) as usize] += 1;
                        c
                    })
                    .reduce(|| vec![0u64; k as usize + 1], |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x += y;
                        }
                        a
                    })
            })
        };
        let est: i128 = counts.iter().enumerate().map(|(j, &c)| j as i128 * c as i128).sum::<i128>() - kappa;
        serv_t.push(secs(t));
        ok &= est == s_true && counts.iter().sum::<u64>() as usize == n + m;
    }
    println!(
        "{}",
        json!({"cmd": "pw2", "N": n, "K": k, "M": m, "reps": reps, "threads": threads, "seed": seed,
               "ct_len": user_cts[0].len(), "user_enc_per_msg_s": user_t, "shuffler_dummy_enc_s": dummy_t,
               "shuffler_shuffle_s": shuf_t, "server_decrypt_count_s": serv_t, "estimate_ok": ok})
    );
}

// ---------------------------------------------------------------- Prio3Sum + HPKE (DAP-17 framing)

struct Rep {
    metadata: Vec<u8>,       // ReportMetadata: report_id(16) || time(8) || public_extensions len (2) = 26 B
    report_id: [u8; 16],
    public_share: Vec<u8>,
    enc: [Vec<u8>; 2],       // HPKE encapsulated keys
    ct: [Vec<u8>; 2],        // HPKE ciphertexts of PlaintextInputShare
}

const INFO_PREFIX: &[u8] = b"dap-17 input share";

fn info(role: u8) -> Vec<u8> {
    let mut v = INFO_PREFIX.to_vec();
    v.push(0x01); // sender: client
    v.push(role); // receiver: 0x02 leader, 0x03 helper
    v
}

fn aad(task_id: &[u8; 32], metadata: &[u8], ps: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(32 + metadata.len() + 4 + ps.len());
    v.extend_from_slice(task_id);
    v.extend_from_slice(metadata);
    v.extend_from_slice(&(ps.len() as u32).to_be_bytes());
    v.extend_from_slice(ps);
    v
}

fn plaintext_input_share(share: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(6 + share.len());
    v.extend_from_slice(&0u16.to_be_bytes()); // empty private_extensions
    v.extend_from_slice(&(share.len() as u32).to_be_bytes());
    v.extend_from_slice(share);
    v
}

fn make_report(vdaf: &Prio3Sum, x: u64, task_id: &[u8; 32], pks: &[<Kem as KemTrait>::PublicKey; 2],
               rng: &mut ChaCha20Rng, hrng: &mut HRng) -> Rep {
    let report_id: [u8; 16] = rng.random();
    let mut metadata = report_id.to_vec();
    metadata.extend_from_slice(&1_700_000_000u64.to_be_bytes());
    metadata.extend_from_slice(&0u16.to_be_bytes());
    let (ps, is) = vdaf.shard(CTX, &x, &report_id).expect("shard");
    let ps = ps.get_encoded().unwrap();
    let a = aad(task_id, &metadata, &ps);
    let mut enc: [Vec<u8>; 2] = [vec![], vec![]];
    let mut ct: [Vec<u8>; 2] = [vec![], vec![]];
    for j in 0..2 {
        let pt = plaintext_input_share(&is[j].get_encoded().unwrap());
        let (e, c) = hpke::single_shot_seal::<AesGcm128, HkdfSha256, Kem, _>(
            &OpModeS::Base, &pks[j], &info(0x02 + j as u8), &pt, &a, hrng).expect("seal");
        enc[j] = e.to_bytes().to_vec();
        ct[j] = c;
    }
    Rep { metadata, report_id, public_share: ps, enc, ct }
}

fn open_share(vdaf: &Prio3Sum, task_id: &[u8; 32], sk: &<Kem as KemTrait>::PrivateKey, r: &Rep, j: usize)
              -> Prio3InputShare<prio::field::Field64, 32> {
    let a = aad(task_id, &r.metadata, &r.public_share);
    let ek = <Kem as KemTrait>::EncappedKey::from_bytes(&r.enc[j]).expect("enc");
    let pt = hpke::single_shot_open::<AesGcm128, HkdfSha256, Kem>(&OpModeR::Base, sk, &ek, &info(0x02 + j as u8),
                                                                    &r.ct[j], &a).expect("open");
    let n = u32::from_be_bytes(pt[2..6].try_into().unwrap()) as usize;
    Prio3InputShare::get_decoded_with_param(&(vdaf, j), &pt[6..6 + n]).expect("input share")
}

fn hpke_ct_bytes(r: &Rep, j: usize) -> usize {
    1 + 2 + r.enc[j].len() + 4 + r.ct[j].len() // HpkeCiphertext{config_id, enc<1..2^16-1>, payload<1..2^32-1>}
}

pub fn cmd_prio2(n: usize, k: u64, reps: usize, threads: usize, seed: u64) {
    let tp = pool(threads.max(1));
    let build = pool(64);
    let vdaf = Prio3Sum::new_sum(2, k).expect("new_sum");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut hrng = HRng::seed_from_u64(seed + 1);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    let s_true: u64 = xs.iter().sum();
    let mut vk = [0u8; 32];
    rng.fill(&mut vk[..]);
    let task_id: [u8; 32] = rng.random();
    let (sk_l, pk_l) = Kem::gen_keypair(&mut hrng);
    let (sk_h, pk_h) = Kem::gen_keypair(&mut hrng);
    let pks = [pk_l, pk_h];
    let sks = [sk_l, sk_h];
    let make_all = |base: u64| -> Vec<Rep> {
        (if threads > 1 { &tp } else { &build }).install(|| {
            xs.par_iter()
                .enumerate()
                .map(|(i, &x)| {
                    let mut r = ChaCha20Rng::seed_from_u64(base ^ (0x7777_0000 + i as u64));
                    let mut h = HRng::seed_from_u64(base ^ (0x8888_0000 + i as u64));
                    make_report(&vdaf, x, &task_id, &pks, &mut r, &mut h)
                })
                .collect()
        })
    };
    // warm-up
    for i in 0..500 {
        let r = make_report(&vdaf, i % (k + 1), &task_id, &pks, &mut rng, &mut hrng);
        std::hint::black_box(open_share(&vdaf, &task_id, &sks[0], &r, 0));
    }
    let batch = 5000usize;
    let (mut client_t, mut client_all_t, mut leader_t, mut helper_t, mut coll_t) = (vec![], vec![], vec![], vec![], vec![]);
    let mut ok = true;
    let mut bytes = json!({});
    for rep in 0..reps {
        // client, per report (always single thread): shard + encode + 2 HPKE seals
        let t = Instant::now();
        for i in 0..batch {
            std::hint::black_box(make_report(&vdaf, xs[i % n], &task_id, &pks, &mut rng, &mut hrng));
        }
        client_t.push(secs(t) / batch as f64);
        // all N reports (dataset for the aggregators), built with the pool; the time is a client-side throughput
        // figure only when threads > 1 (single-thread client cost is client_per_report_s above)
        let t = Instant::now();
        let reports = make_all(seed + 1000 + rep as u64);
        client_all_t.push(if threads > 1 { json!(secs(t)) } else { json!(null) });
        // Leader step 1: open own share, decode, verify_init  (Helper step 1 likewise)
        let step1 = |j: usize| {
            let f = |r: &Rep| {
                let ps = Prio3PublicShare::get_decoded_with_param(&vdaf, &r.public_share).expect("ps");
                let is = open_share(&vdaf, &task_id, &sks[j], r, j);
                vdaf.verify_init(&vk, CTX, j, &(), &r.report_id, &ps, &is).expect("verify_init")
            };
            if threads <= 1 { reports.iter().map(f).collect::<Vec<_>>() } else { tp.install(|| reports.par_iter().map(f).collect()) }
        };
        let t = Instant::now();
        let lead = step1(0);
        let tl1 = secs(t);
        let t = Instant::now();
        let help = step1(1);
        let th1 = secs(t);
        let vs_len = lead[0].1.get_encoded().unwrap().len();
        // Helper: combine verifier shares into the verifier message (DAP ping-pong: Helper finishes), verify_next
        let t = Instant::now();
        macro_rules! comb {
            ($l:expr, $h:expr) => {{
                let (hs, hv) = $h;
                let msg = vdaf.verifier_shares_to_message(CTX, &(), [$l.1.clone(), hv]).expect("msg");
                let out = match vdaf.verify_next(CTX, hs, msg.clone()).expect("next") {
                    VerifyTransition::Finish(o) => o,
                    _ => panic!("one round"),
                };
                (msg, out)
            }};
        }
        let hres: Vec<_> = if threads <= 1 {
            lead.iter().zip(help.into_iter()).map(|(l, h)| comb!(l, h)).collect()
        } else {
            tp.install(|| lead.par_iter().zip(help.into_par_iter()).map(|(l, h)| comb!(l, h)).collect())
        };
        let mut agg_h = vdaf.aggregate_init(&());
        for (_, o) in &hres {
            agg_h.accumulate(o).expect("acc");
        }
        let th2 = secs(t);
        let vm_len = hres[0].0.get_encoded().unwrap().len();
        // Leader step 2: verify_next with the message, accumulate
        let t = Instant::now();
        let mut agg_l = vdaf.aggregate_init(&());
        for ((st, _), (msg, _)) in lead.into_iter().zip(hres.into_iter()) {
            match vdaf.verify_next(CTX, st, msg).expect("next") {
                VerifyTransition::Finish(o) => agg_l.accumulate(&o).expect("acc"),
                _ => panic!("one round"),
            }
        }
        let tl2 = secs(t);
        leader_t.push(json!({"open_decode_verify_init_s": tl1, "verify_next_accumulate_s": tl2, "total_s": tl1 + tl2}));
        helper_t.push(json!({"open_decode_verify_init_s": th1, "combine_verify_next_accumulate_s": th2, "total_s": th1 + th2}));
        // Collector: unshard + discrete Laplace (trusted-collector variant)
        let t = Instant::now();
        let res = vdaf.unshard(&(), [agg_l, agg_h], n).expect("unshard");
        let noisy = res as i64 + dlap(&mut rng, 1.0 / k as f64);
        std::hint::black_box(noisy);
        coll_t.push(secs(t));
        ok &= res == s_true;
        let r0 = &reports[0];
        let ps_len = r0.public_share.len();
        let upload = 26 + 4 + ps_len + hpke_ct_bytes(r0, 0) + hpke_ct_bytes(r0, 1);
        let l2h = 26 + 4 + ps_len + hpke_ct_bytes(r0, 1) + 1 + 4 + vs_len; // PrepareInit{ReportShare, Message::initialize}
        let h2l = 16 + 1 + 1 + 4 + vm_len;                                  // PrepareResp{report_id, continue, Message::finish}
        bytes = json!({"public_share": ps_len, "hpke_enc": r0.enc[0].len(), "hpke_ct_leader": r0.ct[0].len(),
                       "hpke_ct_helper": r0.ct[1].len(), "verifier_share": vs_len, "verifier_message": vm_len,
                       "upload_report_bytes": upload, "leader_to_helper_per_report": l2h,
                       "helper_to_leader_per_report": h2l, "server_to_server_per_report": l2h + h2l});
    }
    println!(
        "{}",
        json!({"cmd": "prio2", "N": n, "K": k, "reps": reps, "threads": threads, "seed": seed,
               "hpke_suite": "DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM (hpke 0.13)",
               "vdaf": "Prio3Sum (prio 0.18.1)", "bytes": bytes, "client_per_report_s": client_t,
               "client_all_reports_s": client_all_t, "leader": leader_t, "helper": helper_t,
               "collector_unshard_noise_s": coll_t, "aggregate_equals_true_sum": ok})
    );
}

// ---------------------------------------------------------------- PW-Sum with HPKE (supplement 4)
//
//   pw3 <N> <K> <M> <reps> <threads> <seed>
// Same as pw2, but every user message and every dummy is HPKE-encrypted with the suite used for Prio3Sum:
// DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM, base mode (hpke 0.13). Serialized ciphertext =
// encapsulated key (32) || AEAD ciphertext (plaintext + 16-byte tag). Bounded decoding as in Algorithm 1:
// decapsulation/decryption failure, wrong length or value > K count as 0.

const PW_INFO: &[u8] = b"pw-sum message";

fn hpke_seal_msg(pk: &<Kem as KemTrait>::PublicKey, pt: &[u8], rng: &mut HRng) -> Vec<u8> {
    let (e, c) = hpke::single_shot_seal::<AesGcm128, HkdfSha256, Kem, _>(&OpModeS::Base, pk, PW_INFO, pt, b"", rng)
        .expect("seal");
    let mut v = e.to_bytes().to_vec();
    v.extend_from_slice(&c);
    v
}

fn hpke_bounded_decode(sk: &<Kem as KemTrait>::PrivateKey, ct: &[u8], k: u64, len: usize) -> u64 {
    if ct.len() < 32 + 16 {
        return 0;
    }
    let ek = match <Kem as KemTrait>::EncappedKey::from_bytes(&ct[..32]) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    match hpke::single_shot_open::<AesGcm128, HkdfSha256, Kem>(&OpModeR::Base, sk, &ek, PW_INFO, &ct[32..], b"") {
        Ok(pt) if pt.len() == len => {
            let mut b = [0u8; 8];
            b[8 - len..].copy_from_slice(&pt);
            let v = u64::from_be_bytes(b);
            if v <= k { v } else { 0 }
        }
        _ => 0,
    }
}

pub fn cmd_pw3(n: usize, k: u64, m: usize, reps: usize, threads: usize, seed: u64) {
    let tp = pool(threads.max(1));
    let mut hrng = HRng::seed_from_u64(seed);
    let (sk, pk) = Kem::gen_keypair(&mut hrng);
    let len = pt_len(k);
    let mut rng = ChaCha20Rng::seed_from_u64(seed + 1);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    let user_cts: Vec<Vec<u8>> = pool(64).install(|| {
        xs.par_iter()
            .enumerate()
            .map(|(i, &x)| {
                let mut r = HRng::seed_from_u64(seed ^ (0x9e37_79b9 + i as u64));
                hpke_seal_msg(&pk, &encode_value(x, len), &mut r)
            })
            .collect()
    });
    let dummy_vals: Vec<u64> = (0..m).map(|i| (i as u64) % (k + 1)).collect();
    let kappa: i128 = dummy_vals.iter().map(|&v| v as i128).sum();
    let s_true: i128 = xs.iter().map(|&x| x as i128).sum();
    for _ in 0..2000 {
        let ct = hpke_seal_msg(&pk, &encode_value(1, len), &mut hrng);
        std::hint::black_box(hpke_bounded_decode(&sk, &ct, k, len));
    }
    let (mut user_t, mut dummy_t, mut shuf_t, mut serv_t) = (vec![], vec![], vec![], vec![]);
    let mut ok = true;
    let batch = 20_000usize;
    for rep in 0..reps {
        let mut r = ChaCha20Rng::seed_from_u64(seed + 100 + rep as u64);
        let mut hr = HRng::seed_from_u64(seed + 200 + rep as u64);
        let t = Instant::now();
        for i in 0..batch {
            std::hint::black_box(hpke_seal_msg(&pk, &encode_value(i as u64 % (k + 1), len), &mut hr));
        }
        user_t.push(secs(t) / batch as f64);
        let t = Instant::now();
        let dummies: Vec<Vec<u8>> = if threads <= 1 {
            dummy_vals.iter().map(|&v| hpke_seal_msg(&pk, &encode_value(v, len), &mut hr)).collect()
        } else {
            tp.install(|| {
                dummy_vals
                    .par_iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let mut rr = HRng::seed_from_u64((seed + rep as u64) ^ (0x5151_0000 + i as u64));
                        hpke_seal_msg(&pk, &encode_value(v, len), &mut rr)
                    })
                    .collect()
            })
        };
        dummy_t.push(secs(t));
        let mut all: Vec<Vec<u8>> = Vec::with_capacity(n + m);
        all.extend(user_cts.iter().cloned());
        all.extend(dummies);
        let t = Instant::now();
        fisher_yates(&mut all, &mut r);
        shuf_t.push(secs(t));
        let t = Instant::now();
        let counts: Vec<u64> = if threads <= 1 {
            let mut c = vec![0u64; k as usize + 1];
            for ct in &all {
                c[hpke_bounded_decode(&sk, ct, k, len) as usize] += 1;
            }
            c
        } else {
            tp.install(|| {
                all.par_iter()
                    .fold(|| vec![0u64; k as usize + 1], |mut c, ct| {
                        c[hpke_bounded_decode(&sk, ct, k, len) as usize] += 1;
                        c
                    })
                    .reduce(|| vec![0u64; k as usize + 1], |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x += y;
                        }
                        a
                    })
            })
        };
        let est: i128 = counts.iter().enumerate().map(|(j, &c)| j as i128 * c as i128).sum::<i128>() - kappa;
        serv_t.push(secs(t));
        ok &= est == s_true && counts.iter().sum::<u64>() as usize == n + m;
    }
    // malformed-input sanity check of the bounded decoder (not timed)
    let bad_ok = hpke_bounded_decode(&sk, &vec![7u8; 32 + 16 + len], k, len) == 0
        && hpke_bounded_decode(&sk, &user_cts[0][..20], k, len) == 0
        && hpke_bounded_decode(&sk, &hpke_seal_msg(&pk, &encode_value(k + 1, len.max(pt_len(k + 1))), &mut hrng), k, len) == 0;
    println!(
        "{}",
        json!({"cmd": "pw3", "N": n, "K": k, "M": m, "reps": reps, "threads": threads, "seed": seed,
               "hpke_suite": "DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM, base mode (hpke 0.13)",
               "ct_len": user_cts[0].len(), "user_enc_per_msg_s": user_t, "shuffler_dummy_enc_s": dummy_t,
               "shuffler_shuffle_s": shuf_t, "server_decrypt_count_s": serv_t, "estimate_ok": ok,
               "bounded_decode_malformed_ok": bad_ok})
    );
}


// ---------------------------------------------------------------- PW-Sum, real noise path
//
//   pw4 <N> <K> <H file> <reps> <threads> <seed>
// H file: first line "W kappa M", second line the K+1 dummy counts H_j = b_j + U_j produced by Algorithm pw2
// (sampling and integerization are done and timed by the Python driver). Per repetition the whole Shuffler step (expand H into
// dummy plaintexts, encrypt, merge with the users' ciphertexts, shuffle) and the whole Server step (decrypt, bounded decode,
// count, estimate) are timed as one interval each. Check per repetition: sum_j j T_j - kappa == sum x + W, and N + M messages.
pub fn cmd_pw4(n: usize, k: u64, hfile: &str, reps: usize, threads: usize, seed: u64) {
    let txt = std::fs::read_to_string(hfile).expect("H file");
    let mut lines = txt.lines();
    let head: Vec<i128> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (w, kappa, m) = (head[0], head[1], head[2] as usize);
    let h: Vec<u64> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    assert_eq!(h.len(), k as usize + 1);
    assert_eq!(h.iter().sum::<u64>() as usize, m);
    let tp = pool(threads.max(1));
    let mut hrng = HRng::seed_from_u64(seed);
    let (sk, pk) = Kem::gen_keypair(&mut hrng);
    let len = pt_len(k);
    let mut rng = ChaCha20Rng::seed_from_u64(seed + 1);
    let xs: Vec<u64> = (0..n).map(|_| rng.random_range(0..=k)).collect();
    let user_cts: Vec<Vec<u8>> = pool(64).install(|| {
        xs.par_iter()
            .enumerate()
            .map(|(i, &x)| {
                let mut r = HRng::seed_from_u64(seed ^ (0x9e37_79b9 + i as u64));
                hpke_seal_msg(&pk, &encode_value(x, len), &mut r)
            })
            .collect()
    });
    let s_true: i128 = xs.iter().map(|&x| x as i128).sum();
    for _ in 0..2000 {
        let ct = hpke_seal_msg(&pk, &encode_value(1, len), &mut hrng);
        std::hint::black_box(hpke_bounded_decode(&sk, &ct, k, len));
    }
    let (mut user_t, mut shuf_total, mut serv_total, mut enc_t, mut shf_t) = (vec![], vec![], vec![], vec![], vec![]);
    let mut ok = true;
    let batch = 20_000usize;
    for rep in 0..reps {
        let mut r = ChaCha20Rng::seed_from_u64(seed + 100 + rep as u64);
        let mut hr = HRng::seed_from_u64(seed + 200 + rep as u64);
        let t = Instant::now();
        for i in 0..batch {
            std::hint::black_box(hpke_seal_msg(&pk, &encode_value(i as u64 % (k + 1), len), &mut hr));
        }
        user_t.push(secs(t) / batch as f64);
        let t0 = Instant::now();
        let dummy_vals: Vec<u64> = h.iter().enumerate().flat_map(|(j, &c)| std::iter::repeat(j as u64).take(c as usize)).collect();
        let t = Instant::now();
        let dummies: Vec<Vec<u8>> = if threads <= 1 {
            dummy_vals.iter().map(|&v| hpke_seal_msg(&pk, &encode_value(v, len), &mut hr)).collect()
        } else {
            tp.install(|| {
                dummy_vals
                    .par_iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        let mut rr = HRng::seed_from_u64((seed + rep as u64) ^ (0x5151_0000 + i as u64));
                        hpke_seal_msg(&pk, &encode_value(v, len), &mut rr)
                    })
                    .collect()
            })
        };
        enc_t.push(secs(t));
        let mut all: Vec<Vec<u8>> = Vec::with_capacity(n + m);
        all.extend(user_cts.iter().cloned());
        all.extend(dummies);
        let t = Instant::now();
        fisher_yates(&mut all, &mut r);
        shf_t.push(secs(t));
        shuf_total.push(secs(t0));
        let t = Instant::now();
        let counts: Vec<u64> = if threads <= 1 {
            let mut c = vec![0u64; k as usize + 1];
            for ct in &all {
                c[hpke_bounded_decode(&sk, ct, k, len) as usize] += 1;
            }
            c
        } else {
            tp.install(|| {
                all.par_iter()
                    .fold(|| vec![0u64; k as usize + 1], |mut c, ct| {
                        c[hpke_bounded_decode(&sk, ct, k, len) as usize] += 1;
                        c
                    })
                    .reduce(|| vec![0u64; k as usize + 1], |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x += y;
                        }
                        a
                    })
            })
        };
        let est: i128 = counts.iter().enumerate().map(|(j, &c)| j as i128 * c as i128).sum::<i128>() - kappa;
        serv_total.push(secs(t));
        ok &= est == s_true + w && counts.iter().sum::<u64>() as usize == n + m;
    }
    println!(
        "{}",
        json!({"cmd": "pw4", "N": n, "K": k, "M": m, "W": w as i64, "reps": reps, "threads": threads, "seed": seed, "ct_len": user_cts[0].len(),
               "user_enc_per_msg_s": user_t, "shuffler_total_s": shuf_total, "shuffler_dummy_enc_s": enc_t, "shuffler_shuffle_s": shf_t,
               "server_total_s": serv_total, "estimate_equals_sum_plus_W": ok})
    );
}


// ---------------------------------------------------------------- Database workload: partitioned bounded SUM (next-stage experiment)
//
//   dbq_pw   <values.u32> <K> <H file> <threads> <bandwidth Mbit/s, 0 = unlimited> <seed>
//   dbq_prio <values.u32> <K> <threads> <bandwidth Mbit/s, 0 = unlimited> <seed>
// values.u32: the quantised inputs of one query (little-endian u32, one per record), produced by the Python driver from the
// partitioned scan. Every network hop sends the real serialised messages (4-byte length prefix each) over a loopback TCP connection
// paced to the given bandwidth; the receiver counts every byte. Stages are timed separately; nothing is overlapped.

fn read_u32s(path: &str) -> Vec<u64> {
    let b = std::fs::read(path).expect("values file");
    b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64).collect()
}

fn paced_send<I: Iterator<Item = Vec<u8>>>(msgs: I, bw_mbps: f64) -> (f64, usize) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().unwrap();
    let rx = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().expect("accept");
        let mut buf = vec![0u8; 1 << 20];
        let mut total = 0usize;
        loop {
            let n = s.read(&mut buf).expect("read");
            if n == 0 {
                break;
            }
            total += n;
        }
        total
    });
    let mut s = std::net::TcpStream::connect(addr).expect("connect");
    let rate = bw_mbps * 1e6 / 8.0;
    let t0 = Instant::now();
    let mut chunk: Vec<u8> = Vec::with_capacity(1 << 21);
    let mut sent = 0usize;
    let mut flush = |chunk: &mut Vec<u8>, sent: &mut usize, s: &mut std::net::TcpStream| {
        s.write_all(chunk).expect("write");
        *sent += chunk.len();
        chunk.clear();
        if rate > 0.0 {
            let target = *sent as f64 / rate;
            let el = secs(t0);
            if target > el {
                std::thread::sleep(std::time::Duration::from_secs_f64(target - el));
            }
        }
    };
    for m in msgs {
        chunk.extend_from_slice(&(m.len() as u32).to_be_bytes());
        chunk.extend_from_slice(&m);
        if chunk.len() >= (1 << 20) {
            flush(&mut chunk, &mut sent, &mut s);
        }
    }
    if !chunk.is_empty() {
        flush(&mut chunk, &mut sent, &mut s);
    }
    s.shutdown(std::net::Shutdown::Write).expect("shutdown");
    let got = rx.join().expect("rx");
    assert_eq!(got, sent);
    (secs(t0), sent)
}

pub fn cmd_dbq_pw(path: &str, k: u64, hfile: &str, threads: usize, bw: f64, seed: u64) {
    let xs = read_u32s(path);
    let n = xs.len();
    let txt = std::fs::read_to_string(hfile).expect("H file");
    let mut lines = txt.lines();
    let head: Vec<i128> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    let (w, kappa, m) = (head[0], head[1], head[2] as usize);
    let h: Vec<u64> = lines.next().unwrap().split_whitespace().map(|v| v.parse().unwrap()).collect();
    assert_eq!(h.len(), k as usize + 1);
    let tp = pool(threads.max(1));
    let mut hrng = HRng::seed_from_u64(seed);
    let (sk, pk) = Kem::gen_keypair(&mut hrng);
    let len = pt_len(k);
    let s_true: i128 = xs.iter().map(|&x| x as i128).sum();
    // users (data owners of the partitions): encrypt one message per record
    let t = Instant::now();
    let user_cts: Vec<Vec<u8>> = tp.install(|| {
        xs.par_iter().enumerate().map(|(i, &x)| {
            let mut r = HRng::seed_from_u64(seed ^ (0x9e37_79b9 + i as u64));
            hpke_seal_msg(&pk, &encode_value(x, len), &mut r)
        }).collect()
    });
    let t_enc = secs(t);
    let (t_up, b_up) = paced_send(user_cts.iter().cloned(), bw);
    // Shuffler: dummies, merge, shuffle
    let t = Instant::now();
    let dummy_vals: Vec<u64> = h.iter().enumerate().flat_map(|(j, &c)| std::iter::repeat(j as u64).take(c as usize)).collect();
    let dummies: Vec<Vec<u8>> = tp.install(|| {
        dummy_vals.par_iter().enumerate().map(|(i, &v)| {
            let mut rr = HRng::seed_from_u64(seed ^ (0x5151_0000 + i as u64));
            hpke_seal_msg(&pk, &encode_value(v, len), &mut rr)
        }).collect()
    });
    let mut all = user_cts;
    all.extend(dummies);
    let mut r = ChaCha20Rng::seed_from_u64(seed + 7);
    fisher_yates(&mut all, &mut r);
    let t_shuf = secs(t);
    let (t_fwd, b_fwd) = paced_send(all.iter().cloned(), bw);
    // Server: decrypt, count, estimate
    let t = Instant::now();
    let counts: Vec<u64> = tp.install(|| {
        all.par_iter()
            .fold(|| vec![0u64; k as usize + 1], |mut c, ct| {
                c[hpke_bounded_decode(&sk, ct, k, len) as usize] += 1;
                c
            })
            .reduce(|| vec![0u64; k as usize + 1], |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            })
    });
    let est: i128 = counts.iter().enumerate().map(|(j, &c)| j as i128 * c as i128).sum::<i128>() - kappa;
    let t_srv = secs(t);
    println!("{}", json!({"cmd": "dbq_pw", "N": n, "K": k, "M": m, "threads": threads, "bw_mbps": bw, "t_encrypt": t_enc, "t_upload": t_up, "bytes_upload": b_up,
        "t_shuffler": t_shuf, "t_forward": t_fwd, "bytes_forward": b_fwd, "t_server": t_srv, "estimate": est as i64, "true_sum": s_true as i64, "W": w as i64,
        "estimate_equals_sum_plus_W": est == s_true + w && counts.iter().sum::<u64>() as usize == n + m}));
}

pub fn cmd_dbq_prio(path: &str, k: u64, threads: usize, bw: f64, seed: u64) {
    let xs = read_u32s(path);
    let n = xs.len();
    let tp = pool(threads.max(1));
    let vdaf = Prio3Sum::new_sum(2, k).expect("new_sum");
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut hrng = HRng::seed_from_u64(seed + 1);
    let s_true: u64 = xs.iter().sum();
    let mut vk = [0u8; 32];
    rng.fill(&mut vk[..]);
    let task_id: [u8; 32] = rng.random();
    let (sk_l, pk_l) = Kem::gen_keypair(&mut hrng);
    let (sk_h, pk_h) = Kem::gen_keypair(&mut hrng);
    let pks = [pk_l, pk_h];
    let sks = [sk_l, sk_h];
    // clients: shard, encode, two HPKE seals per record
    let t = Instant::now();
    let reports: Vec<Rep> = tp.install(|| {
        xs.par_iter().enumerate().map(|(i, &x)| {
            let mut r = ChaCha20Rng::seed_from_u64(seed ^ (0x7777_0000 + i as u64));
            let mut h = HRng::seed_from_u64(seed ^ (0x8888_0000 + i as u64));
            make_report(&vdaf, x, &task_id, &pks, &mut r, &mut h)
        }).collect()
    });
    let t_enc = secs(t);
    let upload = |r: &Rep| -> Vec<u8> {
        // DAP Report: metadata, public_share<u32>, two HpkeCiphertext{config_id, enc<u16>, payload<u32>}
        let mut v = r.metadata.clone();
        v.extend_from_slice(&(r.public_share.len() as u32).to_be_bytes());
        v.extend_from_slice(&r.public_share);
        for j in 0..2 {
            v.push(0);
            v.extend_from_slice(&(r.enc[j].len() as u16).to_be_bytes());
            v.extend_from_slice(&r.enc[j]);
            v.extend_from_slice(&(r.ct[j].len() as u32).to_be_bytes());
            v.extend_from_slice(&r.ct[j]);
        }
        v
    };
    let (t_up, b_up) = paced_send(reports.iter().map(upload), bw);
    // Leader step 1
    let step1 = |j: usize| {
        let f = |r: &Rep| {
            let ps = Prio3PublicShare::get_decoded_with_param(&vdaf, &r.public_share).expect("ps");
            let is = open_share(&vdaf, &task_id, &sks[j], r, j);
            vdaf.verify_init(&vk, CTX, j, &(), &r.report_id, &ps, &is).expect("verify_init")
        };
        tp.install(|| reports.par_iter().map(f).collect::<Vec<_>>())
    };
    let t = Instant::now();
    let lead = step1(0);
    let t_l1 = secs(t);
    // Leader -> Helper: report share for the Helper plus the Leader's verifier share
    let l2h = |(r, l): (&Rep, &(_, _))| -> Vec<u8> {
        let (_, vs): &(_, prio::vdaf::prio3::Prio3VerifierShare<prio::field::Field64, 32>) = l;
        // DAP PrepareInit: ReportShare{metadata, public_share<u32>, Helper HpkeCiphertext} + Message::initialize(verifier share<u32>)
        let vsb = vs.get_encoded().unwrap();
        let mut v = r.metadata.clone();
        v.extend_from_slice(&(r.public_share.len() as u32).to_be_bytes());
        v.extend_from_slice(&r.public_share);
        v.push(0);
        v.extend_from_slice(&(r.enc[1].len() as u16).to_be_bytes());
        v.extend_from_slice(&r.enc[1]);
        v.extend_from_slice(&(r.ct[1].len() as u32).to_be_bytes());
        v.extend_from_slice(&r.ct[1]);
        v.push(0);
        v.extend_from_slice(&(vsb.len() as u32).to_be_bytes());
        v.extend_from_slice(&vsb);
        v
    };
    let (t_l2h, b_l2h) = paced_send(reports.iter().zip(lead.iter()).map(l2h), bw);
    // Helper: step 1, combine, verify_next, accumulate
    let t = Instant::now();
    let help = step1(1);
    macro_rules! comb {
        ($l:expr, $h:expr) => {{
            let (hs, hv) = $h;
            let msg = vdaf.verifier_shares_to_message(CTX, &(), [$l.1.clone(), hv]).expect("msg");
            let out = match vdaf.verify_next(CTX, hs, msg.clone()).expect("next") {
                VerifyTransition::Finish(o) => o,
                _ => panic!("one round"),
            };
            (msg, out)
        }};
    }
    let hres: Vec<_> = tp.install(|| lead.par_iter().zip(help.into_par_iter()).map(|(l, h)| comb!(l, h)).collect());
    let mut agg_h = vdaf.aggregate_init(&());
    for (_, o) in &hres {
        agg_h.accumulate(o).expect("acc");
    }
    let t_help = secs(t);
    // DAP PrepareResp: report_id, continue, Message::finish(verifier message<u32>)
    let (t_h2l, b_h2l) = paced_send(reports.iter().zip(hres.iter()).map(|(r, (m, _))| {
        let mb = m.get_encoded().unwrap();
        let mut v = r.report_id.to_vec();
        v.push(0);
        v.push(2);
        v.extend_from_slice(&(mb.len() as u32).to_be_bytes());
        v.extend_from_slice(&mb);
        v
    }), bw);
    // Leader step 2
    let t = Instant::now();
    let mut agg_l = vdaf.aggregate_init(&());
    for ((st, _), (msg, _)) in lead.into_iter().zip(hres.into_iter()) {
        match vdaf.verify_next(CTX, st, msg).expect("next") {
            VerifyTransition::Finish(o) => agg_l.accumulate(&o).expect("acc"),
            _ => panic!("one round"),
        }
    }
    let t_l2 = secs(t);
    // Collector: unshard + discrete Laplace (variant A)
    let t = Instant::now();
    let res = vdaf.unshard(&(), [agg_l, agg_h], n).expect("unshard");
    let noisy = res as i64 + dlap(&mut rng, 1.0 / k as f64);
    let t_col = secs(t);
    println!("{}", json!({"cmd": "dbq_prio", "N": n, "K": k, "threads": threads, "bw_mbps": bw, "t_encrypt": t_enc, "t_upload": t_up, "bytes_upload": b_up,
        "t_leader1": t_l1, "t_leader_to_helper": t_l2h, "bytes_leader_to_helper": b_l2h, "t_helper": t_help, "t_helper_to_leader": t_h2l, "bytes_helper_to_leader": b_h2l,
        "t_leader2": t_l2, "t_collector": t_col, "estimate": noisy, "aggregate": res as i64, "true_sum": s_true as i64, "aggregate_equals_true_sum": res == s_true}));
}
