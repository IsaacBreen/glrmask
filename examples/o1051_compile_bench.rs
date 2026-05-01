use glrmask::{Constraint, Vocab};
use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() {
    let vocab_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".cache/vocab_cache/llama3_vocab.json");
    let vocab_json = fs::read_to_string(&vocab_path)
        .unwrap_or_else(|err| panic!("failed to read vocab: {err}"));
    let vocab_map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&vocab_json).expect("parse vocab json");
    let entries = vocab_map
        .into_iter()
        .map(|(token_id, token_hex)| {
            let token_id = token_id.parse::<u32>().unwrap();
            let token_hex = token_hex.as_str().unwrap();
            (token_id, decode_hex_bytes(token_hex))
        })
        .collect();
    let vocab = Vocab::new(entries, None);

    let schema_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../constraint-framework-analysis/data/sources/jsonschemabench/data/Github_hard/o1051.json");
    let schema_json = fs::read_to_string(&schema_path)
        .unwrap_or_else(|err| panic!("failed to read schema: {err}"));
    let schema_value: serde_json::Value = serde_json::from_str(&schema_json)
        .expect("parse schema json");
    let schema_payload = schema_value.get("schema").unwrap_or(&schema_value);
    let schema_str = serde_json::to_string(schema_payload).unwrap();

    let started = Instant::now();
    let constraint = Constraint::from_json_schema(&schema_str, &vocab).unwrap();
    let elapsed = started.elapsed().as_secs_f64();

    eprintln!("[o1051_compile_bench] total_s={:.3} parser_states={}", elapsed, constraint.num_parser_states());
}

fn decode_hex_bytes(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0);
    (0..hex.len())
        .step_by(2)
        .map(|offset| u8::from_str_radix(&hex[offset..offset + 2], 16).unwrap())
        .collect()
}
