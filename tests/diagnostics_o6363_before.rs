/// Diagnostic test for o6363 on the ORIGINAL (pre-refactor) code.
/// Run with: cargo test --release --test diagnostics_o6363_before -- --nocapture

use glrmask::{Constraint, Vocab};
use std::fs;

fn build_realistic_vocab() -> Vocab {
    let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut next_id = 0u32;

    for b in 0..=255u8 {
        entries.push((next_id, vec![b]));
        next_id += 1;
    }

    for pair in &[
        b": " as &[u8], b", ", b"\":", b"\",", b"{\n", b"}\n", b"[\n", b"]\n",
        b"  ", b"\"\"", b"\\\"", b"\\n", b"\\t", b"nu", b"ll", b"tr", b"ue",
        b"fa", b"ls", b"se",
    ] {
        entries.push((next_id, pair.to_vec()));
        next_id += 1;
    }

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
fn dump_o6363_before_stats() {
    let paths = [
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/data/Github_ultra/o6363.json",
    ];

    let schema_str = paths.iter()
        .find_map(|p| fs::read_to_string(p).ok())
        .expect("Could not find o6363.json schema file");

    let vocab = build_realistic_vocab();
    let constraint = Constraint::from_json_schema(&schema_str, &vocab).unwrap();

    let num_states = constraint.debug_num_states();
    let num_terminals = constraint.debug_num_terminals();
    let rules = constraint.debug_rules();

    // Count actions by parsing debug strings
    let mut total_shifts = 0usize;
    let mut total_reduces = 0usize;
    let mut total_splits = 0usize;
    let mut reduce_len_counts = std::collections::BTreeMap::new();

    for state in 0..num_states {
        let actions = constraint.debug_actions_for_state(state);
        for (_, action_str) in &actions {
            if action_str.starts_with("Shift") {
                total_shifts += 1;
            } else if action_str.starts_with("Reduce") {
                total_reduces += 1;
                // Parse Reduce(rule_id) to get rule_id
                if let Some(start) = action_str.find('(') {
                    if let Some(end) = action_str.find(')') {
                        if let Ok(rule_id) = action_str[start+1..end].parse::<usize>() {
                            if rule_id < rules.len() {
                                let rhs_len = rules[rule_id].1;
                                *reduce_len_counts.entry(rhs_len).or_insert(0usize) += 1;
                            }
                        }
                    }
                }
            } else if action_str.starts_with("Split") {
                total_splits += 1;
                // Count reduces within split
                let reduce_count = action_str.matches("Reduce").count();
                // This is approximate for splits
            }
        }
    }

    // Compute reduce lengths from rules (for all rules)
    let max_rhs = rules.iter().map(|(_, len, _)| *len).max().unwrap_or(0);
    let avg_rhs = rules.iter().map(|(_, len, _)| *len as f64).sum::<f64>() / rules.len().max(1) as f64;

    // Compute total gotos 
    // Can't directly get goto count from debug APIs, but we can get it from rules

    println!("\n=== BEFORE (pre-refactor) o6363 Table Statistics ===");
    println!("States: {}", num_states);
    println!("Terminals: {}", num_terminals);
    println!("Total shifts: {}", total_shifts);
    println!("Total reduces: {}", total_reduces);
    println!("Total splits: {}", total_splits);
    println!("Total rules: {}", rules.len());
    println!("Max RHS length: {}", max_rhs);
    println!("Avg RHS length: {:.2}", avg_rhs);
    println!("Reduce length distribution (from table actions):");
    let total = reduce_len_counts.values().sum::<usize>().max(1);
    for (&len, &count) in &reduce_len_counts {
        println!("  len={}: {} ({:.1}%)", len, count, 100.0 * count as f64 / total as f64);
    }

    // Stack depth test
    let mut state = constraint.start();
    let mut max_stack_depth = 0usize;

    for step in 0..2000 {
        let stacks = state.debug_parser_stacks();
        let depth = stacks.iter()
            .flat_map(|(_, s)| s.iter().map(|(stack, _)| stack.len()))
            .max()
            .unwrap_or(0);
        if depth > max_stack_depth {
            max_stack_depth = depth;
            println!("  Step {}: new max stack depth = {}", step, depth);
        }

        let mask = state.mask();
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
        if allowed.is_empty() { break; }

        // Prefer { to get nesting
        let nesting = vec![123u32, 91];
        let closers = vec![34u32, 58, 44, 125, 93];
        let token = nesting.iter().find(|t| allowed.contains(t))
            .or_else(|| closers.iter().find(|t| allowed.contains(t)))
            .copied()
            .unwrap_or(allowed[0]);

        if state.commit_token(token).is_err() { break; }
    }

    println!("\nMax parser stack depth: {}", max_stack_depth);
}
