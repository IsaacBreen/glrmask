use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glrmask::{Constraint, Vocab};
use glrmask::__private::ConstraintExt as _;

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).unwrap())
        .collect()
}

fn load_llama3_vocab() -> Vocab {
    let path = std::env::var_os("GLRMASK_LLAMA3_VOCAB_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join(".cache/vocab_cache/llama3_vocab.json")
        });
    let raw = std::fs::read_to_string(&path).unwrap();
    let id_to_hex: BTreeMap<u32, String> = serde_json::from_str(&raw).unwrap();
    Vocab::new(
        id_to_hex
            .into_iter()
            .map(|(token_id, hex)| (token_id, hex_to_bytes(&hex)))
            .collect())
}

fn main() {
    let schema_path = std::env::var("GLR_SCHEMA")
        .unwrap_or_else(|_| "/tmp/catalog_512.schema.json".to_string());
    let schema = std::fs::read_to_string(&schema_path).unwrap();
    let vocab = load_llama3_vocab();
    let iters: usize = std::env::var("GLR_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    // Warm up
    let c = Constraint::from_json_schema(&schema, &vocab).unwrap();
    std::hint::black_box(&c);

    let import_only = std::env::var("GLR_IMPORT_ONLY").is_ok();
    for _ in 0..iters {
        if import_only {
            glrmask::Constraint::profile_json_schema_import(&schema).unwrap();
            continue;
        }
        Constraint::clear_weight_interners();
        Constraint::clear_weight_op_caches();
        let c = Constraint::from_json_schema(&schema, &vocab).unwrap();
        std::hint::black_box(&c);
    }
}
