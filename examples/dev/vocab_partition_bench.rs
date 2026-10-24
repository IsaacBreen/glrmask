//! Focused study harness for the grammar-specific vocabulary partition API.
//!
//! Usage:
//!   cargo run --release --example vocab_partition_bench -- \
//!     target-vocab-partition/cohort-manifest.json \
//!     C:/path/to/llama3_vocab.json \
//!     target-vocab-partition/vocab-partition-results.json \
//!     5

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

use glrmask::__private::{ConstraintExt, VocabExt};
use glrmask::{Constraint, Grammar, Vocab, VocabPartition};
use serde_json::{Value, json};

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0, "hex token must have even length");
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).expect("hex is ASCII");
            u8::from_str_radix(text, 16).expect("invalid token hex")
        })
        .collect()
}

fn load_vocab(path: &Path) -> Vec<(u32, Vec<u8>)> {
    let raw: BTreeMap<String, String> =
        serde_json::from_slice(&fs::read(path).expect("read vocab JSON"))
            .expect("parse vocab JSON");
    raw.into_iter()
        .map(|(token, bytes)| {
            (
                token.parse::<u32>().expect("numeric token id"),
                decode_hex(&bytes),
            )
        })
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) * 0.5
    }
}

fn reset_global_compile_caches() {
    <Constraint as ConstraintExt>::clear_stale_weights();
    <Constraint as ConstraintExt>::clear_weight_op_caches();
    <Constraint as ConstraintExt>::clear_weight_interners();
}

fn final_partition_relation(
    fast: &VocabPartition,
    final_map: &[u32],
) -> (usize, usize) {
    let mut refining_classes = 0usize;
    let mut split_by_final = 0usize;
    for class in fast.classes() {
        let mut seen = None;
        let mut refines = true;
        for &token in class {
            let target = final_map.get(token as usize).copied().unwrap_or(u32::MAX);
            if let Some(previous) = seen {
                if previous != target {
                    refines = false;
                    break;
                }
            } else {
                seen = Some(target);
            }
        }
        if refines {
            refining_classes += 1;
        } else {
            split_by_final += 1;
        }
    }
    (refining_classes, split_by_final)
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(args.len() >= 4, "expected manifest, vocab JSON, output JSON, [reps]");
    let manifest_path = Path::new(&args[1]);
    let vocab_path = Path::new(&args[2]);
    let output_path = Path::new(&args[3]);
    let reps = args
        .get(4)
        .map(|value| value.parse::<usize>().expect("reps is integer"))
        .unwrap_or(5);
    assert!(reps > 0);

    let manifest: Vec<Value> =
        serde_json::from_slice(&fs::read(manifest_path).expect("read manifest"))
            .expect("parse manifest");
    let entries = load_vocab(vocab_path);
    eprintln!("loaded {} vocabulary tokens", entries.len());

    // Keep method-local vocabulary caches independent while still removing the
    // vocabulary-only construction cost from each grammar build.
    let static_vocab = Vocab::new(entries.clone());
    let partition_vocab = Vocab::new(entries);
    static_vocab.prepare_for_compile();
    partition_vocab.prepare_for_compile();

    let mut output = Vec::<Value>::new();
    for record in manifest {
        let problem_id = record["problem_id"].as_str().expect("problem id");
        let band = record["band"].as_str().expect("band");
        let schema_path = record["schema_path"].as_str().expect("schema path");
        let schema = fs::read_to_string(schema_path).expect("read normalized schema");
        let mut static_ms = Vec::with_capacity(reps);
        let mut partition_ms = Vec::with_capacity(reps);
        let mut fast_classes = 0usize;
        let mut final_classes = 0usize;
        let mut refining_classes = 0usize;
        let mut split_by_final = 0usize;

        for rep in 0..reps {
            let static_first = rep % 2 == 0;
            for mode_static in [static_first, !static_first] {
                reset_global_compile_caches();
                if mode_static {
                    let started = Instant::now();
                    let constraint = Constraint::compile(Grammar::json_schema(&schema), &static_vocab)
                        .expect("static compile");
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    static_ms.push(elapsed);
                    final_classes = constraint.final_internal_token_count();
                    black_box(constraint);
                } else {
                    let started = Instant::now();
                    let partition = VocabPartition::compile(
                        Grammar::json_schema(&schema),
                        &partition_vocab,
                    )
                    .expect("vocab partition compile");
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    partition_ms.push(elapsed);
                    fast_classes = partition.num_classes();
                    black_box(partition);
                }
            }
        }

        // Keep the diagnostic partition-relation check outside the timed reps;
        // otherwise it would warm grammar-specific data before one timing mode.
        reset_global_compile_caches();
        let constraint = Constraint::compile(Grammar::json_schema(&schema), &static_vocab)
            .expect("static compile for relation check");
        let final_map = constraint.final_original_token_map();
        final_classes = constraint.final_internal_token_count();
        reset_global_compile_caches();
        let fast = VocabPartition::compile(Grammar::json_schema(&schema), &partition_vocab)
            .expect("partition compile for relation check");
        fast_classes = fast.num_classes();
        (refining_classes, split_by_final) = final_partition_relation(&fast, &final_map);
        black_box((constraint, fast));

        let static_median = median(static_ms.clone());
        let partition_median = median(partition_ms.clone());
        let ratio = partition_median / static_median;
        eprintln!(
            "{band:>3} {problem_id:<70} static={static_median:8.3}ms partition={partition_median:8.3}ms ratio={ratio:6.3} classes={fast_classes}/{final_classes} final_split_fast_classes={split_by_final}",
        );
        output.push(json!({
            "band": band,
            "problem_id": problem_id,
            "rank": record["rank"],
            "cfa_static_ms": record["cfa_static_ms"],
            "same_machine_static_ms": static_ms,
            "same_machine_static_median_ms": static_median,
            "vocab_partition_ms": partition_ms,
            "vocab_partition_median_ms": partition_median,
            "partition_over_static": ratio,
            "vocab_partition_classes": fast_classes,
            "final_runtime_classes": final_classes,
            "fast_classes_refining_final_runtime": refining_classes,
            "fast_classes_split_by_final_runtime": split_by_final,
            "vocab_size": partition_vocab.len(),
        }));
    }

    fs::write(
        output_path,
        serde_json::to_vec_pretty(&output).expect("serialize results"),
    )
    .expect("write results");
    eprintln!("wrote {}", output_path.display());
}
