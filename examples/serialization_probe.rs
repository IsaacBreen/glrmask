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
        _ => panic!("usage: serialization_probe <bench-vocab|generate-glrm|generate-schema|batch-schema|bench|bench-owned|load-once|mask-bench|commit-bench|resave> ..."),
    }
}
