/// Measure replace-action metrics for o6363.
/// Run with: cargo test --release --test replace_metrics_o6363 -- --nocapture

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

fn load_schema() -> String {
    let paths = [
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/data/Github_ultra/o6363.json",
        "../constraint-framework-analysis/data/sources/jsonschemabench/data/Github_ultra/o6363.json",
    ];

    paths.iter()
        .find_map(|p| fs::read_to_string(p).ok())
        .expect("Could not find o6363.json schema file")
}

fn parse_glr_metrics(table_json: &str) -> (u64, u32, u32, u32, u32, u64, u32, u32) {
    let table: serde_json::Value = serde_json::from_str(table_json).unwrap();
    let actions = table["action"].as_array().unwrap();
    let gotos = table["goto"].as_array().unwrap();
    let num_states = table["num_states"].as_u64().unwrap();

    let mut total_shifts = 0u32;
    let mut replace_shifts = 0u32;
    let mut total_reduces = 0u32;
    let mut total_reduce_length = 0u64;

    for state_actions in actions {
        for (_terminal, action) in state_actions.as_object().unwrap() {
            if let Some(obj) = action.as_object() {
                if let Some(arr) = obj.get("Shift") {
                    let arr = arr.as_array().unwrap();
                    total_shifts += 1;
                    if arr[1].as_bool().unwrap_or(false) { replace_shifts += 1; }
                }
                if let Some(arr) = obj.get("Reduce") {
                    let arr = arr.as_array().unwrap();
                    total_reduces += 1;
                    total_reduce_length += arr[1].as_u64().unwrap();
                }
                if let Some(split) = obj.get("Split") {
                    if let Some(shift) = split.get("shift") {
                        if !shift.is_null() {
                            let arr = shift.as_array().unwrap();
                            total_shifts += 1;
                            if arr[1].as_bool().unwrap_or(false) { replace_shifts += 1; }
                        }
                    }
                    if let Some(reduces) = split.get("reduces") {
                        for red in reduces.as_array().unwrap() {
                            let arr = red.as_array().unwrap();
                            total_reduces += 1;
                            total_reduce_length += arr[1].as_u64().unwrap();
                        }
                    }
                }
            }
        }
    }

    let mut total_gotos = 0u32;
    let mut replace_gotos = 0u32;
    for state_gotos in gotos {
        for (_nt, entry) in state_gotos.as_object().unwrap() {
            let arr = entry.as_array().unwrap();
            total_gotos += 1;
            if arr[1].as_bool().unwrap_or(false) { replace_gotos += 1; }
        }
    }

    (num_states, total_shifts, replace_shifts, total_gotos, replace_gotos, total_reduce_length, total_reduces, 0)
}

fn max_stack_depth(state: &glrmask::ConstraintState) -> usize {
    let stacks = state.debug_parser_stacks();
    stacks.iter()
        .flat_map(|(_, stack_list)| stack_list.iter().map(|(stack, _)| stack.len()))
        .max()
        .unwrap_or(0)
}

#[test]
fn measure_replace_metrics() {
    let schema_str = load_schema();
    let table_json = glrmask::dump_json_schema_glr_table(&schema_str).unwrap();
    let (num_states, total_shifts, replace_shifts, total_gotos, replace_gotos, total_reduce_length, total_reduces, _) = parse_glr_metrics(&table_json);

    let avg_reduce_length = if total_reduces > 0 {
        total_reduce_length as f64 / total_reduces as f64
    } else { 0.0 };

    println!("\n=== o6363 Replace Metrics ===");
    println!("Table states: {}", num_states);
    println!("Total shifts: {}, replace shifts: {} ({:.1}%)",
        total_shifts, replace_shifts,
        if total_shifts > 0 { replace_shifts as f64 / total_shifts as f64 * 100.0 } else { 0.0 });
    println!("Total gotos: {}, replace gotos: {} ({:.1}%)",
        total_gotos, replace_gotos,
        if total_gotos > 0 { replace_gotos as f64 / total_gotos as f64 * 100.0 } else { 0.0 });
    println!("Total reduces: {}, avg reduce length: {:.2}", total_reduces, avg_reduce_length);
    println!("============================\n");
}

#[test]
fn compare_replace_vs_no_replace() {
    let schema_str = load_schema();
    let vocab = build_realistic_vocab();

    // With replace (default)
    let constraint_with = Constraint::from_json_schema(&schema_str, &vocab).unwrap();
    let dwa_states_with = constraint_with.debug_parser_dwa_num_states();
    let dwa_trans_with = constraint_with.debug_parser_dwa_num_transitions();
    let glr_states_with = constraint_with.debug_num_states();

    // Without replace
    unsafe { std::env::set_var("GLRMASK_DISABLE_REPLACE", "1"); }
    let constraint_without = Constraint::from_json_schema(&schema_str, &vocab).unwrap();
    let dwa_states_without = constraint_without.debug_parser_dwa_num_states();
    let dwa_trans_without = constraint_without.debug_parser_dwa_num_transitions();
    let glr_states_without = constraint_without.debug_num_states();
    unsafe { std::env::remove_var("GLRMASK_DISABLE_REPLACE"); }

    println!("\n=== o6363 Replace vs No-Replace Comparison ===");
    println!("                  | With Replace | Without Replace | Ratio");
    println!("GLR states        | {:>12} | {:>15} | {:.2}x",
        glr_states_with, glr_states_without,
        glr_states_without as f64 / glr_states_with as f64);
    println!("Parser DWA states | {:>12} | {:>15} | {:.2}x",
        dwa_states_with, dwa_states_without,
        dwa_states_without as f64 / dwa_states_with as f64);
    println!("Parser DWA trans  | {:>12} | {:>15} | {:.2}x",
        dwa_trans_with, dwa_trans_without,
        dwa_trans_without as f64 / dwa_trans_with as f64);

    // Measure max stack depth during parsing of a sample JSON
    let sample_json = br#"{"apiVersion":"skaffold/v1"}"#;
    let mut max_depth_with = 0usize;
    let mut max_depth_without = 0usize;
    {
        let mut s = constraint_with.start();
        for &b in sample_json.iter() {
            if s.commit_bytes(&[b]).is_err() { break; }
            let d = max_stack_depth(&s);
            if d > max_depth_with { max_depth_with = d; }
        }
    }
    {
        let mut s = constraint_without.start();
        for &b in sample_json.iter() {
            if s.commit_bytes(&[b]).is_err() { break; }
            let d = max_stack_depth(&s);
            if d > max_depth_without { max_depth_without = d; }
        }
    }

    println!("Max stack depth   | {:>12} | {:>15} | {:.2}x",
        max_depth_with, max_depth_without,
        max_depth_without as f64 / max_depth_with.max(1) as f64);
    println!("=================================================\n");
}

#[test]
fn dump_table_and_null_reductions() {
    let schema_str = load_schema();
    let table_json = glrmask::dump_json_schema_glr_table(&schema_str).unwrap();
    let table: serde_json::Value = serde_json::from_str(&table_json).unwrap();

    // Save table
    fs::write("/tmp/o6363_glr_table.json", serde_json::to_string_pretty(&table).unwrap()).unwrap();

    let actions = table["action"].as_array().unwrap();
    let gotos = table["goto"].as_array().unwrap();

    let mut null_reduces = 0u32;
    let mut total_reduces = 0u32;
    let mut reduce_len_histogram: std::collections::BTreeMap<u64, u32> = std::collections::BTreeMap::new();
    let mut replace_shift_count = 0u32;
    let mut replace_goto_count = 0u32;
    let mut total_shift_count = 0u32;
    let mut total_goto_count = 0u32;

    for state_actions in actions {
        for (_terminal, action) in state_actions.as_object().unwrap() {
            if let Some(obj) = action.as_object() {
                if let Some(arr) = obj.get("Reduce") {
                    let arr = arr.as_array().unwrap();
                    total_reduces += 1;
                    let len = arr[1].as_u64().unwrap();
                    *reduce_len_histogram.entry(len).or_insert(0) += 1;
                    if len == 0 { null_reduces += 1; }
                }
                if let Some(arr) = obj.get("Shift") {
                    let arr = arr.as_array().unwrap();
                    total_shift_count += 1;
                    if arr[1].as_bool().unwrap_or(false) { replace_shift_count += 1; }
                }
                if let Some(split) = obj.get("Split") {
                    if let Some(shift) = split.get("shift") {
                        if !shift.is_null() {
                            let arr = shift.as_array().unwrap();
                            total_shift_count += 1;
                            if arr[1].as_bool().unwrap_or(false) { replace_shift_count += 1; }
                        }
                    }
                    if let Some(reduces) = split.get("reduces") {
                        for red in reduces.as_array().unwrap() {
                            let arr = red.as_array().unwrap();
                            total_reduces += 1;
                            let len = arr[1].as_u64().unwrap();
                            *reduce_len_histogram.entry(len).or_insert(0) += 1;
                            if len == 0 { null_reduces += 1; }
                        }
                    }
                }
            }
        }
    }

    for state_gotos in gotos {
        for (_nt, entry) in state_gotos.as_object().unwrap() {
            let arr = entry.as_array().unwrap();
            total_goto_count += 1;
            if arr[1].as_bool().unwrap_or(false) { replace_goto_count += 1; }
        }
    }

    println!("\n=== o6363 Detailed Table Stats ===");
    println!("Table states: {}", table["num_states"]);
    println!("Shifts: {} total, {} replace ({:.1}%)", total_shift_count, replace_shift_count,
        replace_shift_count as f64 / total_shift_count.max(1) as f64 * 100.0);
    println!("Gotos: {} total, {} replace ({:.1}%)", total_goto_count, replace_goto_count,
        replace_goto_count as f64 / total_goto_count.max(1) as f64 * 100.0);
    println!("Reduces: {} total, {} null (len=0) ({:.1}%)", total_reduces, null_reduces,
        null_reduces as f64 / total_reduces.max(1) as f64 * 100.0);
    println!("Reduce length histogram:");
    for (len, count) in &reduce_len_histogram {
        println!("  len={}: {} ({:.1}%)", len, count, *count as f64 / total_reduces.max(1) as f64 * 100.0);
    }
    println!("Table saved to /tmp/o6363_glr_table.json");
    println!("==================================\n");
}
