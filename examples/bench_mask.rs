//! Rust-native mask benchmark.
//!
//! Usage:
//!   cargo run --release --example bench_mask
//!   cargo run --release --example bench_mask -- --schema /path/to/schema.json
//!   cargo run --release --example bench_mask -- --compiled tests/data/kb143_gpt2.bin --tokens tests/data/kb143_gpt2_prefix.json
//!   cargo run --release --example bench_mask -- --iters 500

use glrmask::{Constraint, Vocab};
use std::time::Instant;

fn byte_vocab() -> Vocab {
    let entries: Vec<(u32, Vec<u8>)> = (0..=255u32).map(|b| (b, vec![b as u8])).collect();
    Vocab::new(entries, None)
}

fn default_schema() -> &'static str {
    r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" },
            "active": { "type": "boolean" },
            "tags": { "type": "array", "items": { "type": "string" } },
            "address": {
                "type": "object",
                "properties": {
                    "street": { "type": "string" },
                    "city": { "type": "string" },
                    "zip": { "type": "string" }
                },
                "required": ["street", "city"]
            }
        },
        "required": ["name", "age"]
    }"#
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut schema_path: Option<String> = None;
    let mut compiled_path: Option<String> = None;
    let mut tokens_path: Option<String> = None;
    let mut iters: usize = 500;
    let mut warmup: usize = 50;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--schema" => { i += 1; schema_path = Some(args[i].clone()); }
            "--compiled" => { i += 1; compiled_path = Some(args[i].clone()); }
            "--tokens" => { i += 1; tokens_path = Some(args[i].clone()); }
            "--iters" => { i += 1; iters = args[i].parse().unwrap(); }
            "--warmup" => { i += 1; warmup = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    // --- Load or compile constraint ---
    let constraint = if let Some(path) = &compiled_path {
        eprintln!("Loading pre-compiled constraint from {}...", path);
        let data = std::fs::read(path).expect("Failed to read compiled constraint");
        let t0 = Instant::now();
        let c = Constraint::load(&data).expect("Failed to deserialize constraint");
        eprintln!("Loaded in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        c
    } else {
        let vocab = byte_vocab();
        let schema = schema_path.as_ref()
            .map(|p| std::fs::read_to_string(p).expect("Failed to read schema"))
            .unwrap_or_else(|| default_schema().to_string());
        eprintln!("Compiling schema ({} bytes) with byte vocab...", schema.len());
        let t0 = Instant::now();
        let c = Constraint::from_json_schema(&schema, &vocab).expect("Schema compilation failed");
        eprintln!("Compiled in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
        c
    };
    eprintln!("Mask length: {} u32 words", constraint.mask_len());

    // --- Build token prefix ---
    let prefix_tokens: Vec<u32> = if let Some(path) = &tokens_path {
        eprintln!("Loading token prefix from {}...", path);
        let json_str = std::fs::read_to_string(path).expect("Failed to read tokens file");
        let tokens: Vec<u32> = serde_json::from_str(&json_str).expect("Failed to parse tokens JSON");
        let mut s = constraint.start();
        let mut valid = Vec::new();
        for &t in &tokens {
            let mask = s.mask();
            let word = t as usize / 32;
            let bit = t % 32;
            if word < mask.len() && (mask[word] >> bit) & 1 != 0 {
                valid.push(t);
                if s.commit_token(t).is_err() { break; }
                if s.is_finished() { break; }
            } else {
                eprintln!("  token {} rejected at step {}, stopping", t, valid.len());
                break;
            }
        }
        eprintln!("Validated {} / {} tokens", valid.len(), tokens.len());
        valid
    } else {
        // Greedy generation following a target JSON structure.
        let target = b"{\"name\":\"hello\",\"age\":42,\"active\":true,\"tags\":[\"a\",\"b\"],\"address\":{\"street\":\"Main St\",\"city\":\"NYC\",\"zip\":\"10001\"}}";
        let mut s = constraint.start();
        let mut tokens = Vec::new();
        let mut tidx = 0;
        for _ in 0..300 {
            let mask = s.mask();
            let pick = |t: u32, m: &[u32]| -> bool {
                let w = t as usize / 32;
                w < m.len() && (m[w] >> (t % 32)) & 1 != 0
            };
            let token = if tidx < target.len() && pick(target[tidx] as u32, &mask) {
                tidx += 1;
                Some(target[tidx - 1] as u32)
            } else {
                (0x20..=0x7Eu32).chain(0..0x20u32).find(|&t| pick(t, &mask))
            };
            match token {
                Some(t) => {
                    tokens.push(t);
                    if s.commit_token(t).is_err() || s.is_finished() { break; }
                }
                None => break,
            }
        }
        let bytes: Vec<u8> = tokens.iter().map(|&t| t as u8).collect();
        eprintln!("Generated {} tokens: {:?}", tokens.len(),
            String::from_utf8_lossy(&bytes[..bytes.len().min(80)]));
        tokens
    };
    eprintln!("Prefix: {} tokens", prefix_tokens.len());

    // --- Profile each step to find the hottest ---
    let mask_len = constraint.mask_len();
    let mut buf = vec![0u32; mask_len];
    let mut step_times: Vec<(usize, u64)> = Vec::new();
    {
        let mut s = constraint.start();
        for (i, &t) in prefix_tokens.iter().enumerate() {
            let t0 = Instant::now();
            s.fill_mask(&mut buf);
            step_times.push((i, t0.elapsed().as_nanos() as u64));
            let _ = s.commit_token(t);
        }
    }
    step_times.sort_by_key(|&(_, ns)| std::cmp::Reverse(ns));
    let hot_step = step_times[0].0;
    eprintln!("Hottest step: {} ({} ns), top-5: {:?}", hot_step, step_times[0].1,
        step_times.iter().take(5).map(|&(i, ns)| format!("s{}={}ns", i, ns)).collect::<Vec<_>>());

    // --- Benchmark the hot step ---
    let build_state = || {
        let mut s = constraint.start();
        for &t in &prefix_tokens[..hot_step] {
            let _ = s.commit_token(t);
        }
        s
    };

    for _ in 0..warmup {
        let s = build_state();
        s.fill_mask(&mut buf);
    }

    let mut times_ns: Vec<u64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = build_state();
        let t0 = Instant::now();
        s.fill_mask(&mut buf);
        times_ns.push(t0.elapsed().as_nanos() as u64);
    }
    times_ns.sort();

    let p50 = times_ns[iters / 2];
    let p90 = times_ns[iters * 9 / 10];
    let p99 = times_ns[iters * 99 / 100];
    let min = times_ns[0];
    let max = times_ns[iters - 1];
    let mean: u64 = times_ns.iter().sum::<u64>() / iters as u64;

    println!("=== fill_mask benchmark (step {}, {} iters) ===", hot_step, iters);
    println!("  min:  {:>7} ns", min);
    println!("  p50:  {:>7} ns", p50);
    println!("  mean: {:>7} ns", mean);
    println!("  p90:  {:>7} ns", p90);
    println!("  p99:  {:>7} ns", p99);
    println!("  max:  {:>7} ns", max);

    let s = build_state();
    let metrics = s.debug_mask_metrics();
    println!("\n=== debug metrics (step {}) ===", hot_step);
    println!("  total_ns:               {:>7}", metrics.total_ns);
    println!("  seed_ns:                {:>7}", metrics.seed_ns);
    println!("  bfs_loop_ns:            {:>7}", metrics.bfs_loop_ns);
    println!("  transition_gss_ns:      {:>7}", metrics.transition_gss_ns);
    println!("  transition_intersect_ns:{:>7}", metrics.transition_intersect_ns);
    println!("  transition_enqueue_ns:  {:>7}", metrics.transition_enqueue_ns);
    println!("  queue_pop_ns:           {:>7}", metrics.queue_pop_ns);
    println!("  final_weight_ns:        {:>7}", metrics.final_weight_ns);
    println!("  state: {:?}", metrics.state_summary);
}
