/// Measure replace-action metrics for o6363.
/// Run with: cargo test --release --test replace_metrics_o6363 -- --nocapture

use glrmask::Vocab;
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
fn measure_replace_metrics() {
    let paths = [
        "/Users/isaacbreen/Projects2/constraint-framework-analysis/data/sources/jsonschemabench/data/Github_ultra/o6363.json",
        "../constraint-framework-analysis/data/sources/jsonschemabench/data/Github_ultra/o6363.json",
    ];

    let schema_str = paths.iter()
        .find_map(|p| fs::read_to_string(p).ok())
        .expect("Could not find o6363.json schema file");

    // Dump the GLR table as JSON and parse action/goto stats from it
    let table_json = glrmask::dump_json_schema_glr_table(&schema_str).unwrap();
    let table: serde_json::Value = serde_json::from_str(&table_json).unwrap();

    let actions = table["action"].as_array().unwrap();
    let gotos = table["goto"].as_array().unwrap();

    let mut total_shifts = 0u32;
    let mut replace_shifts = 0u32;
    let mut total_reduces = 0u32;
    let mut total_reduce_length = 0u64;

    for state_actions in actions {
        for (_terminal, action) in state_actions.as_object().unwrap() {
            if let Some(obj) = action.as_object() {
                // Shift: {"Shift": [target, replace_bool]}
                if let Some(arr) = obj.get("Shift") {
                    let arr = arr.as_array().unwrap();
                    total_shifts += 1;
                    if arr[1].as_bool().unwrap_or(false) {
                        replace_shifts += 1;
                    }
                }
                // Reduce: {"Reduce": [nt, len]}
                if let Some(arr) = obj.get("Reduce") {
                    let arr = arr.as_array().unwrap();
                    total_reduces += 1;
                    total_reduce_length += arr[1].as_u64().unwrap();
                }
                // Split: {"Split": {"shift": [target, replace], "reduces": [[nt, len], ...], "accept": bool}}
                if let Some(split) = obj.get("Split") {
                    if let Some(shift) = split.get("shift") {
                        if !shift.is_null() {
                            let arr = shift.as_array().unwrap();
                            total_shifts += 1;
                            if arr[1].as_bool().unwrap_or(false) {
                                replace_shifts += 1;
                            }
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
            if arr[1].as_bool().unwrap_or(false) {
                replace_gotos += 1;
            }
        }
    }

    let avg_reduce_length = if total_reduces > 0 {
        total_reduce_length as f64 / total_reduces as f64
    } else {
        0.0
    };

    let num_states = table["num_states"].as_u64().unwrap();

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
