use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use glrmask::__private::{ConstraintStateExt, VocabExt};
use glrmask::{Constraint, Grammar, Vocab};

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

fn percentile_us(sorted_ns: &[u128], p: f64) -> f64 {
    let index = ((sorted_ns.len() - 1) as f64 * p).round() as usize;
    sorted_ns[index] as f64 / 1_000.0
}

fn print_samples(name: &str, mut samples: Vec<u128>) {
    samples.sort_unstable();
    println!(
        "{name} p50={:.3}us p90={:.3}us p99={:.3}us p99.9={:.3}us max={:.3}us",
        percentile_us(&samples, 0.50),
        percentile_us(&samples, 0.90),
        percentile_us(&samples, 0.99),
        percentile_us(&samples, 0.999),
        percentile_us(&samples, 1.0),
    );
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let grammar = std::fs::read_to_string(&args[1]).unwrap();
    let vocab = load_vocab(Path::new(&args[2]));
    vocab.prepare_for_compile();
    let token_sequences: Vec<Vec<u32>> =
        serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();

    let started = Instant::now();
    let constraint = Constraint::compile(Grammar::glrm(&grammar), &vocab).unwrap();
    eprintln!("compile_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);

    let save_started = Instant::now();
    let saved = constraint.save();
    let save_ms = save_started.elapsed().as_secs_f64() * 1000.0;
    let artifact_bytes = saved.len();
    let load_started = Instant::now();
    let loaded = Constraint::load_owned(saved).unwrap();
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "save_ms={save_ms:.3} artifact_bytes={artifact_bytes} load_owned_ms={load_ms:.3}"
    );

    let mut compiled_masks = Vec::<u128>::new();
    let mut compiled_commits = Vec::<u128>::new();
    let mut compiled_tbm = Vec::<u128>::new();
    let mut loaded_masks = Vec::<u128>::new();
    let mut loaded_commits = Vec::<u128>::new();
    let mut loaded_tbm = Vec::<u128>::new();
    let mut absolute_sample = 0usize;
    for (example_index, tokens) in token_sequences.iter().enumerate() {
        let mut state = constraint.start();
        let mut loaded_state = loaded.start();
        let mut mask = vec![0u32; constraint.mask_len()];
        let mut loaded_mask = vec![0u32; loaded.mask_len()];
        for (token_index, &token) in tokens.iter().enumerate() {
            // Alternate call order so neither representation systematically
            // gets the warmer instruction/data-cache position in this probe.
            let (compiled_mask_ns, loaded_mask_ns) = if absolute_sample & 1 == 0 {
                (
                    state.fill_mask_timed_ns(&mut mask) as u128,
                    loaded_state.fill_mask_timed_ns(&mut loaded_mask) as u128,
                )
            } else {
                let loaded_ns = loaded_state.fill_mask_timed_ns(&mut loaded_mask) as u128;
                let compiled_ns = state.fill_mask_timed_ns(&mut mask) as u128;
                (compiled_ns, loaded_ns)
            };
            if mask != loaded_mask {
                let diff_words = mask
                    .iter()
                    .zip(&loaded_mask)
                    .filter(|(left, right)| left != right)
                    .count();
                eprintln!(
                    "mask divergence sample={} example={} token_index={} token_id={} diff_words={}",
                    absolute_sample, example_index, token_index, token, diff_words
                );
                panic!("compiled/loaded masks diverged");
            }
            let timed_commit = |result: Result<u64, _>, which: &str| match result {
                Ok(ns) => ns as u128,
                Err(err) => {
                    eprintln!(
                        "{which} reject sample={} example={} token_index={} token_id={} error={err}",
                        absolute_sample, example_index, token_index, token
                    );
                    panic!("{which} replay rejected token");
                }
            };
            let (compiled_commit_ns, loaded_commit_ns) = if absolute_sample & 1 == 0 {
                let compiled_ns = timed_commit(state.commit_token_timed_ns(token), "compiled");
                let loaded_ns = timed_commit(loaded_state.commit_token_timed_ns(token), "loaded");
                (compiled_ns, loaded_ns)
            } else {
                let loaded_ns = timed_commit(loaded_state.commit_token_timed_ns(token), "loaded");
                let compiled_ns = timed_commit(state.commit_token_timed_ns(token), "compiled");
                (compiled_ns, loaded_ns)
            };
            compiled_masks.push(compiled_mask_ns);
            compiled_commits.push(compiled_commit_ns);
            compiled_tbm.push(compiled_mask_ns + compiled_commit_ns);
            loaded_masks.push(loaded_mask_ns);
            loaded_commits.push(loaded_commit_ns);
            loaded_tbm.push(loaded_mask_ns + loaded_commit_ns);
            absolute_sample += 1;
        }
    }
    println!("examples={} samples={}", token_sequences.len(), compiled_tbm.len());
    print_samples("compiled_mask", compiled_masks);
    print_samples("compiled_commit", compiled_commits);
    print_samples("compiled_tbm", compiled_tbm);
    print_samples("loaded_mask", loaded_masks);
    print_samples("loaded_commit", loaded_commits);
    print_samples("loaded_tbm", loaded_tbm);
}
