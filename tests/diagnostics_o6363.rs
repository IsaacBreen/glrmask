/// Diagnostic test for o6363: dump table stats and track max parser stack depth.
/// Run with: cargo test --release --test diagnostics_o6363 -- --nocapture

use glrmask::{Constraint, Vocab};
use std::fs;

fn build_realistic_vocab() -> Vocab {
    let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut next_id = 0u32;

    // All single-byte tokens (0x00..0xFF)
    for b in 0..=255u8 {
        entries.push((next_id, vec![b]));
        next_id += 1;
    }

    // 2-byte tokens for common JSON patterns
    for pair in &[
        b": " as &[u8], b", ", b"\":", b"\",", b"{\n", b"}\n", b"[\n", b"]\n",
        b"  ", b"\"\"", b"\\\"", b"\\n", b"\\t", b"nu", b"ll", b"tr", b"ue",
        b"fa", b"ls", b"se",
    ] {
        entries.push((next_id, pair.to_vec()));
        next_id += 1;
    }

    // Common JSON multi-byte tokens
    for tok in &[
        b"null" as &[u8], b"true", b"false", b"    ", b"      ",
        b": \"", b"\": ", b"\": \"", b",\n", b"{\n  ", b"}\n",
    ] {
        entries.push((next_id, tok.to_vec()));
        next_id += 1;
    }

    Vocab::new(entries, None)
}

#[test]
fn dump_o6363_table_stats() {
    // Try multiple paths
    let paths = [
        "data/sources/jsonschemabench/data/Github_ultra/o6363.json",
        "../constraint-framework-analysis/data/sources/jsonschemabench/data/Github_ultra/o6363.json",
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/data/Github_ultra/o6363.json",
    ];

    let schema_str = paths.iter()
        .find_map(|p| fs::read_to_string(p).ok())
        .expect("Could not find o6363.json schema file");

    let vocab = build_realistic_vocab();
    let constraint = Constraint::from_json_schema(&schema_str, &vocab).unwrap();

    println!("\n=== o6363 Table Statistics ===");
    println!("{:?}", constraint.debug_rules());
    println!("=== Parser DWA Statistics ===");
    println!("{}", constraint.debug_parser_dwa_num_states());
    println!("=== Replace Context Statistics ===");
    println!("{}", constraint.debug_num_states());
    println!("=== Replace Equivalence Statistics ===");
    println!("{}", constraint.debug_num_terminals());

    // Now process some tokens and track max stack depth
    let mut state = constraint.start();
    let mask_started_at = std::time::Instant::now();
    let mask = state.mask();
    let mut total_mask_ns = mask_started_at.elapsed().as_nanos() as u64;
    let mut mask_calls = 1u64;

    // Find first allowed token to start
    let first_token = (0..mask.len() as u32 * 32)
        .find(|&id| {
            let word = id / 32;
            let bit = id % 32;
            if (word as usize) < mask.len() {
                mask[word as usize] & (1 << bit) != 0
            } else {
                false
            }
        });

    println!("First allowed token: {:?}", first_token);

    // Generate a valid JSON string by following forced tokens
    let mut max_stack_depth = 0usize;
    let mut total_steps = 0usize;
    let mut token_sequence = Vec::new();
    let mut total_path_count = 0usize;
    let mut max_path_count = 0usize;

    // Prefer structural tokens that create nesting
    // Strategy: close strings quickly with ", prefer { over [, prefer : after property names
    let nesting_tokens: Vec<u32> = vec![
        123, // {
        91,  // [
    ];
    let closer_tokens: Vec<u32> = vec![
        34,  // " (close string)
        58,  // : (property separator)
        44,  // , (next item)
        125, // } (close object)
        93,  // ] (close array)
    ];

    for _ in 0..2000 {
        let stacks = state.debug_parser_stacks();
        let max_depth = stacks.iter()
            .flat_map(|(_, stacks)| stacks.iter().map(|(stack, _)| stack.len()))
            .max()
            .unwrap_or(0);

        let path_count = state.parser_path_count(10000);
        total_path_count += path_count;
        if path_count > max_path_count {
            max_path_count = path_count;
        }

        if max_depth > max_stack_depth {
            max_stack_depth = max_depth;
            println!("  Step {}: new max stack depth = {} (paths={})", total_steps, max_stack_depth, path_count);
        }

        let mask_started_at = std::time::Instant::now();
        let mask = state.mask();
        total_mask_ns += mask_started_at.elapsed().as_nanos() as u64;
        mask_calls += 1;
        // Count allowed tokens
        let mut allowed: Vec<u32> = Vec::new();
        for word_idx in 0..mask.len() {
            let word = mask[word_idx];
            if word == 0 { continue; }
            for bit in 0..32 {
                if word & (1 << bit) != 0 {
                    allowed.push((word_idx as u32) * 32 + bit);
                }
            }
        }

        if allowed.is_empty() {
            println!("No allowed tokens at step {} — generation complete", total_steps);
            break;
        }

        // Prefer tokens that create nesting, then closers, then first allowed
        let token = nesting_tokens.iter()
            .find(|t| allowed.contains(t))
            .or_else(|| closer_tokens.iter().find(|t| allowed.contains(t)))
            .copied()
            .unwrap_or(allowed[0]);

        token_sequence.push(token);
        if let Err(e) = state.commit_token(token) {
            println!("Error at step {}: {}", total_steps, e);
            break;
        }
        total_steps += 1;
    }

    println!("\n=== Stack Depth Summary ===");
    println!("Max parser stack depth (greedy): {}", max_stack_depth);
    println!("Max parser path count: {}", max_path_count);
    println!("Avg parser path count: {:.1}", total_path_count as f64 / total_steps.max(1) as f64);
    println!("Total tokens processed: {}", total_steps);
    let total_mask_ms = total_mask_ns as f64 / 1_000_000.0;
    println!("Mask calls: {}", mask_calls);
    println!("Total mask time (ms): {:.3}", total_mask_ms);
    println!("Mask throughput (calls/sec): {:.1}", mask_calls as f64 / (total_mask_ns as f64 / 1_000_000_000.0).max(1e-9));

    // Now feed a known deeply-nested JSON to find max stack depth
    println!("\n=== Deep Nesting Test ===");
    let mut state2 = constraint.start();
    // {"apiVersion":{"a":{"a":{"a":{"a":{"a":{"a":"x"}}}}}}}
    let nested_json = r#"{"apiVersion":"x","kind":"x","metadata":{"name":"x","labels":{"a":{"b":{"c":"d"}}}}}"#;
    let mut max_depth_nested = 0usize;
    for (i, byte) in nested_json.bytes().enumerate() {
        let mask_started_at = std::time::Instant::now();
        let mask = state2.mask();
        total_mask_ns += mask_started_at.elapsed().as_nanos() as u64;
        mask_calls += 1;
        let token = byte as u32;
        let word = token / 32;
        let bit = token % 32;
        if (word as usize) < mask.len() && mask[word as usize] & (1 << bit) != 0 {
            state2.commit_token(token).unwrap();
            let stacks = state2.debug_parser_stacks();
            let depth = stacks.iter()
                .flat_map(|(_, s)| s.iter().map(|(stack, _)| stack.len()))
                .max()
                .unwrap_or(0);
            if depth > max_depth_nested {
                max_depth_nested = depth;
                println!("  Byte {}: '{}' stack depth = {}", i, byte as char, depth);
            }
        } else {
            println!("  Byte {}: '{}' NOT ALLOWED, skipping", i, byte as char);
            break;
        }
    }
    println!("Max stack depth (nested JSON): {}", max_depth_nested);

    // Also compute theoretical max reduce chain from rules
    let rules = constraint.debug_rules();
    println!("\n=== Rule Statistics ===");
    println!("Total rules: {}", rules.len());
    let max_rhs = rules.iter().map(|(_, len, _)| *len).max().unwrap_or(0);
    println!("Max RHS length: {}", max_rhs);
    let avg_rhs = rules.iter().map(|(_, len, _)| *len as f64).sum::<f64>() / rules.len() as f64;
    println!("Avg RHS length: {:.2}", avg_rhs);
}
