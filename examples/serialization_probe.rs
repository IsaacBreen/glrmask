use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use glrmask::{Constraint, Vocab};
use glrmask::__private::{ConstraintStateExt, VocabExt};

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn load_vocab(path: &Path) -> Vocab {
    let raw = std::fs::read_to_string(path).unwrap();
    let entries: BTreeMap<u32, String> = serde_json::from_str(&raw).unwrap();
    Vocab::new(
        entries
            .into_iter()
            .map(|(id, hex)| (id, hex_to_bytes(&hex)))
            .collect(),
    )
}

fn median_ns(mut xs: Vec<u128>) -> u128 {
    xs.sort_unstable();
    xs[xs.len() / 2]
}

fn ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn percentile_us(sorted_ns: &[u128], p: f64) -> f64 {
    assert!(!sorted_ns.is_empty());
    let pos = (sorted_ns.len() - 1) as f64 * p;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    let value = sorted_ns[lo] as f64 * (1.0 - frac) + sorted_ns[hi] as f64 * frac;
    value / 1_000.0
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    match args.get(1).map(String::as_str) {
        Some("bench-vocab") => {
            let started = Instant::now();
            let raw = std::fs::read_to_string(&args[2]).unwrap();
            let read_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            let entries: BTreeMap<u32, String> = serde_json::from_str(&raw).unwrap();
            let vocab = Vocab::new(
                entries
                    .into_iter()
                    .map(|(id, hex)| (id, hex_to_bytes(&hex)))
                    .collect(),
            );
            let construct_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = Instant::now();
            vocab.prepare_for_compile();
            let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
            println!(
                "read_ms={read_ms:.3} construct_ms={construct_ms:.3} prepare_ms={prepare_ms:.3} total_ms={:.3} entries={}",
                read_ms + construct_ms + prepare_ms,
                vocab.len(),
            );
        }
        Some("generate-glrm") => {
            let grammar = std::fs::read_to_string(&args[2]).unwrap();
            let vocab = load_vocab(Path::new(&args[3]));
            vocab.prepare_for_compile();
            let started = Instant::now();
            let constraint = Constraint::from_glrm_grammar(&grammar, &vocab).unwrap();
            eprintln!("compile_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
            let started = Instant::now();
            let saved = constraint.save();
            eprintln!(
                "save_ms={:.3} artifact_bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                saved.len()
            );
            std::fs::write(&args[4], saved).unwrap();
        }
        Some("generate-schema") => {
            let schema = std::fs::read_to_string(&args[2]).unwrap();
            let vocab = load_vocab(Path::new(&args[3]));
            vocab.prepare_for_compile();
            let started = Instant::now();
            let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
            eprintln!("compile_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
            let started = Instant::now();
            let saved = constraint.save();
            eprintln!(
                "save_ms={:.3} artifact_bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                saved.len()
            );
            std::fs::write(&args[4], saved).unwrap();
        }
        Some("batch-schema") => {
            let vocab = load_vocab(Path::new(&args[2]));
            vocab.prepare_for_compile();
            for schema_path in &args[3..] {
                let schema = std::fs::read_to_string(schema_path).unwrap();
                let started = Instant::now();
                let constraint = Constraint::from_json_schema(&schema, &vocab).unwrap();
                let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
                let started = Instant::now();
                let saved = constraint.save();
                let save_ms = started.elapsed().as_secs_f64() * 1000.0;
                println!(
                    "schema={} compile_ms={compile_ms:.3} save_ms={save_ms:.3} total_ms={:.3} artifact_bytes={}",
                    schema_path,
                    compile_ms + save_ms,
                    saved.len(),
                );
                std::hint::black_box(saved);
                std::hint::black_box(constraint);
            }
        }
        Some("bench") => {
            let path = Path::new(&args[2]);
            let iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(21usize);
            let mut reads = Vec::with_capacity(iters);
            let mut loads = Vec::with_capacity(iters);
            let mut saves = Vec::with_capacity(iters);
            let mut artifact_bytes = 0usize;
            let mut resaved_bytes = 0usize;
            for _ in 0..iters {
                let started = Instant::now();
                let bytes = std::fs::read(path).unwrap();
                reads.push(started.elapsed().as_nanos());
                artifact_bytes = bytes.len();

                let started = Instant::now();
                let constraint = Constraint::load(&bytes).unwrap();
                loads.push(started.elapsed().as_nanos());

                let started = Instant::now();
                let saved = constraint.save();
                saves.push(started.elapsed().as_nanos());
                resaved_bytes = saved.len();
                std::hint::black_box(saved);
                std::hint::black_box(constraint);
            }
            println!(
                "artifact={} artifact_bytes={} resaved_bytes={} iters={} read_median_ms={:.3} load_median_ms={:.3} save_median_ms={:.3}",
                path.display(),
                artifact_bytes,
                resaved_bytes,
                iters,
                ms(median_ns(reads)),
                ms(median_ns(loads)),
                ms(median_ns(saves)),
            );
        }
        Some("bench-owned") => {
            let path = Path::new(&args[2]);
            let iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(21usize);
            let mut reads = Vec::with_capacity(iters);
            let mut loads = Vec::with_capacity(iters);
            let mut saves = Vec::with_capacity(iters);
            let mut artifact_bytes = 0usize;
            let mut resaved_bytes = 0usize;
            for _ in 0..iters {
                let started = Instant::now();
                let bytes = std::fs::read(path).unwrap();
                reads.push(started.elapsed().as_nanos());
                artifact_bytes = bytes.len();

                let started = Instant::now();
                let constraint = Constraint::load_owned(bytes).unwrap();
                loads.push(started.elapsed().as_nanos());

                let started = Instant::now();
                let saved = constraint.save();
                saves.push(started.elapsed().as_nanos());
                resaved_bytes = saved.len();
                std::hint::black_box(saved);
                std::hint::black_box(constraint);
            }
            println!(
                "artifact={} artifact_bytes={} resaved_bytes={} iters={} read_median_ms={:.3} load_owned_median_ms={:.3} save_median_ms={:.3}",
                path.display(),
                artifact_bytes,
                resaved_bytes,
                iters,
                ms(median_ns(reads)),
                ms(median_ns(loads)),
                ms(median_ns(saves)),
            );
        }
        Some("load-once") => {
            let bytes = std::fs::read(&args[2]).unwrap();
            let started = Instant::now();
            let constraint = Constraint::load(&bytes).unwrap();
            println!("load_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(constraint);
        }
        Some("mask-bench") => {
            let bytes = std::fs::read(&args[2]).unwrap();
            let constraint = Constraint::load(&bytes).unwrap();
            let state = constraint.start();
            let iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000usize);
            let mut mask = state.mask();
            for _ in 0..100 {
                std::hint::black_box(state.fill_mask_timed_ns(&mut mask));
            }
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                samples.push(state.fill_mask_timed_ns(&mut mask) as u128);
            }
            samples.sort_unstable();
            let percentile = |p: f64| -> f64 {
                let index = ((samples.len() - 1) as f64 * p).round() as usize;
                samples[index] as f64 / 1_000.0
            };
            println!(
                "iters={} mask_p50_us={:.3} mask_p90_us={:.3} mask_p99_us={:.3} mask_max_us={:.3}",
                iters,
                percentile(0.50),
                percentile(0.90),
                percentile(0.99),
                percentile(1.0),
            );
            std::hint::black_box(mask);
        }
        Some("commit-bench") => {
            let bytes = std::fs::read(&args[2]).unwrap();
            let constraint = Constraint::load_owned(bytes).unwrap();
            let state = constraint.start();
            let iters = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000usize);
            let mask = state.mask();
            let token = mask
                .iter()
                .enumerate()
                .find_map(|(word_index, &word)| {
                    (word != 0).then(|| {
                        (word_index as u32) * 32 + word.trailing_zeros()
                    })
                })
                .expect("start state should allow at least one token");
            for _ in 0..100 {
                let mut sample = state.clone();
                std::hint::black_box(sample.commit_token_timed_ns(token).unwrap());
            }
            let mut samples = Vec::with_capacity(iters);
            for _ in 0..iters {
                let mut sample = state.clone();
                samples.push(sample.commit_token_timed_ns(token).unwrap() as u128);
            }
            samples.sort_unstable();
            let percentile = |p: f64| -> f64 {
                let index = ((samples.len() - 1) as f64 * p).round() as usize;
                samples[index] as f64 / 1_000.0
            };
            println!(
                "token={} iters={} commit_p50_us={:.3} commit_p90_us={:.3} commit_p99_us={:.3} commit_max_us={:.3}",
                token,
                iters,
                percentile(0.50),
                percentile(0.90),
                percentile(0.99),
                percentile(1.0),
            );
        }
        Some("profile-token") => {
            let artifact = std::fs::read(&args[2]).unwrap();
            let constraint = Constraint::load_owned(artifact).unwrap();
            let token: u32 = args[3].parse().unwrap();
            let mut state = constraint.start();
            let started = Instant::now();
            let profile = state.commit_token_profiled(token).unwrap();
            println!("wall_us={:.3} profile={profile:#?}", started.elapsed().as_secs_f64() * 1e6);
        }
        Some("replay-token-ids") => {
            let artifact = std::fs::read(&args[2]).unwrap();
            let token_sequences: Vec<Vec<u32>> =
                serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();
            let runs = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50usize);
            let sample_count = token_sequences.iter().map(Vec::len).sum::<usize>();
            let mut best_mask = vec![u128::MAX; sample_count];
            let mut best_commit = vec![u128::MAX; sample_count];
            let mut first_mask = vec![0u128; sample_count];
            let mut first_commit = vec![0u128; sample_count];
            let mut first_starts = Vec::<u128>::with_capacity(token_sequences.len());
            let mut best_starts = vec![u128::MAX; token_sequences.len()];
            let mut mask_words = 0usize;

            // Every measured traversal gets a fresh Constraint loaded from a fresh
            // artifact allocation. Nothing mutable survives from run N to N+1;
            // multiple runs exist only to reduce scheduler noise.
            for run in 0..runs {
                let constraint = Constraint::load_owned(artifact.clone()).unwrap();
                mask_words = constraint.mask_len();
                let mut sample_index = 0usize;
                for (example_index, tokens) in token_sequences.iter().enumerate() {
                    let started = Instant::now();
                    let mut state = constraint.start();
                    let start_ns = started.elapsed().as_nanos();
                    if run == 0 {
                        first_starts.push(start_ns);
                    }
                    best_starts[example_index] = best_starts[example_index].min(start_ns);
                    let mut mask = vec![0u32; mask_words];
                    for &token in tokens {
                        let mask_ns = state.fill_mask_timed_ns(&mut mask) as u128;
                        let word = token as usize / 32;
                        let bit = token as usize % 32;
                        assert!(
                            word < mask.len() && ((mask[word] >> bit) & 1) != 0,
                            "recorded token {token} is not allowed at sample {sample_index}"
                        );
                        let commit_ns = state.commit_token_timed_ns(token).unwrap() as u128;
                        if run == 0 {
                            first_mask[sample_index] = mask_ns;
                            first_commit[sample_index] = commit_ns;
                        }
                        best_mask[sample_index] = best_mask[sample_index].min(mask_ns);
                        best_commit[sample_index] = best_commit[sample_index].min(commit_ns);
                        sample_index += 1;
                    }
                }
                assert_eq!(sample_index, sample_count);
            }

            let mut first_tbm = first_mask
                .iter()
                .zip(&first_commit)
                .map(|(&mask, &commit)| mask + commit)
                .collect::<Vec<_>>();
            let locate = |sample: usize| {
                let mut base = 0usize;
                for (example, tokens) in token_sequences.iter().enumerate() {
                    if sample < base + tokens.len() {
                        return (example, sample - base, tokens[sample - base]);
                    }
                    base += tokens.len();
                }
                unreachable!()
            };
            for (name, samples) in [("first_mask", &first_mask), ("first_commit", &first_commit), ("first_tbm", &first_tbm)] {
                let (sample, &ns) = samples.iter().enumerate().max_by_key(|&(_, ns)| ns).unwrap();
                let (example, token_index, token_id) = locate(sample);
                eprintln!("worst {name}: ns={ns} sample={sample} example={example} token_index={token_index} token_id={token_id}");
            }
            let mut best_tbm = best_mask
                .iter()
                .zip(&best_commit)
                .map(|(&mask, &commit)| mask + commit)
                .collect::<Vec<_>>();
            first_mask.sort_unstable();
            first_commit.sort_unstable();
            first_tbm.sort_unstable();
            first_starts.sort_unstable();
            best_mask.sort_unstable();
            best_commit.sort_unstable();
            best_tbm.sort_unstable();
            best_starts.sort_unstable();
            let print = |name: &str, samples: &[u128]| {
                println!(
                    "{name} samples={} p50_us={:.3} p90_us={:.3} p99_us={:.3} p99_9_us={:.3} p99_99_us={:.3} p100_us={:.3}",
                    samples.len(),
                    percentile_us(samples, 0.50),
                    percentile_us(samples, 0.90),
                    percentile_us(samples, 0.99),
                    percentile_us(samples, 0.999),
                    percentile_us(samples, 0.9999),
                    percentile_us(samples, 1.0),
                );
            };
            println!("examples={} runs={} mask_words={}", token_sequences.len(), runs, mask_words);
            println!("--- first independent run ---");
            print("start", &first_starts);
            print("mask", &first_mask);
            print("commit", &first_commit);
            print("tbm", &first_tbm);
            println!("--- elementwise best of independent fresh-load runs ---");
            print("start", &best_starts);
            print("mask", &best_mask);
            print("commit", &best_commit);
            print("tbm", &best_tbm);
        }
        Some("compare-replay") => {
            let left = Constraint::load_owned(std::fs::read(&args[2]).unwrap()).unwrap();
            let right = Constraint::load_owned(std::fs::read(&args[3]).unwrap()).unwrap();
            let token_sequences: Vec<Vec<u32>> =
                serde_json::from_slice(&std::fs::read(&args[4]).unwrap()).unwrap();
            assert_eq!(left.mask_len(), right.mask_len());
            let mut sample_index = 0usize;
            for (example_index, tokens) in token_sequences.iter().enumerate() {
                let mut left_state = left.start();
                let mut right_state = right.start();
                let mut left_mask = vec![0u32; left.mask_len()];
                let mut right_mask = vec![0u32; right.mask_len()];
                for (token_index, &token) in tokens.iter().enumerate() {
                    left_state.fill_mask(&mut left_mask);
                    right_state.fill_mask(&mut right_mask);
                    if left_mask != right_mask {
                        let differing_words = left_mask
                            .iter()
                            .zip(&right_mask)
                            .enumerate()
                            .filter_map(|(word, (&a, &b))| (a != b).then_some((word, a, b)))
                            .take(16)
                            .collect::<Vec<_>>();
                        eprintln!(
                            "mask divergence sample={sample_index} example={example_index} token_index={token_index} next_token={token} differing_words={differing_words:?}"
                        );
                        eprintln!("left_stacks={:#?}", left_state.debug_parser_stacks());
                        eprintln!("right_stacks={:#?}", right_state.debug_parser_stacks());
                        return;
                    }
                    let left_result = left_state.commit_token(token);
                    let right_result = right_state.commit_token(token);
                    if left_result.is_err() != right_result.is_err() {
                        eprintln!(
                            "commit result divergence sample={sample_index} example={example_index} token_index={token_index} token={token} left={left_result:?} right={right_result:?}"
                        );
                        return;
                    }
                    sample_index += 1;
                }
            }
            println!("no divergence across {sample_index} samples");
        }
        Some("resave") => {
            let bytes = std::fs::read(&args[2]).unwrap();
            let constraint = Constraint::load(&bytes).unwrap();
            let started = Instant::now();
            let saved = constraint.save();
            println!(
                "save_ms={:.3} artifact_bytes={}",
                started.elapsed().as_secs_f64() * 1000.0,
                saved.len()
            );
            std::fs::write(&args[3], saved).unwrap();
        }
        _ => panic!("usage: serialization_probe <bench-vocab|generate-glrm|generate-schema|batch-schema|bench|bench-owned|load-once|mask-bench|commit-bench|profile-token|replay-token-ids|compare-replay|resave> ..."),
    }
}
