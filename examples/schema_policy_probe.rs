//! JSON-lines acceptance probe for scripts/schema_policy_differential.py.
//! Each input is {"schema": <schema>, "cases": [<JSON values>]}. No benchmarks.
use glrmask::{BuildOptions, Grammar, MaskStrategy, Vocab};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn error_kind(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("unsatisfiable") || lower.contains("empty language") {
        "reported_empty"
    } else if lower.contains("resource limit") || lower.contains("schema too large")
        || lower.contains("budget") || lower.contains("size limit") {
        "resource_limit"
    } else if lower.contains("unsupported") || lower.contains("not supported")
        || lower.contains("unimplemented") {
        "unsupported"
    } else if lower.contains("malformed") || lower.contains("invalid")
        || lower.contains("must be") || lower.contains("syntax") {
        "malformed"
    } else {
        "other_build_error"
    }
}

// Match the importer's existing canonical separator spelling. This changes
// serialization only; schema/instance semantics and fixture values are intact.
fn canonical_json(value: &Value) -> String {
    match value {
        Value::Array(items) => format!("[{}]", items.iter().map(canonical_json).collect::<Vec<_>>().join(", ")),
        Value::Object(items) => format!("{{{}}}", items.iter().map(|(key, value)| format!("{}: {}", serde_json::to_string(key).unwrap(), canonical_json(value))).collect::<Vec<_>>().join(", ")),
        _ => serde_json::to_string(value).unwrap(),
    }
}

fn probe(request: &Value, vocab: &Vocab, mode: MaskStrategy) -> Value {
    let Some(schema) = request.get("schema") else {
        return json!({"status":"input_error", "error":"missing schema"});
    };
    let Some(cases) = request.get("cases").and_then(Value::as_array) else {
        return json!({"status":"input_error", "error":"cases must be an array"});
    };
    let text = serde_json::to_string(schema).unwrap();
    let compiled = match Grammar::from_json_schema(&text).unwrap()
        .compile_with_build_options(vocab, BuildOptions::default().optimization(mode)) {
        Ok(compiled) => compiled,
        Err(error) => {
            let message = error.to_string();
            return json!({"status":"build_error", "kind":error_kind(&message), "error":message});
        }
    };
    let mut accepts = Vec::new();
    let mut runtime_errors = Vec::new();
    for (index, value) in cases.iter().enumerate() {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            let mut state = compiled.start();
            let bytes = canonical_json(value).into_bytes();
            for byte in bytes {
                let id = u32::from(byte);
                let mask = state.mask();
                if !mask.get(id as usize / 32).is_some_and(|word| word & (1 << (id % 32)) != 0) {
                    return Ok(false);
                }
                if !state.commit_token(id) {
                    return Ok(false);
                }
                if state.mask().iter().all(|word| *word == 0) {
                    return Err("mask allowed a token whose commit rejected the state".to_string());
                }
            }
            Ok(state.is_accepting())
        }));
        match outcome {
            Ok(Ok(accepted)) => accepts.push(json!(accepted)),
            Ok(Err(error)) => {
                accepts.push(Value::Null);
                runtime_errors.push(json!({"case":index, "error":error}));
            }
            Err(_) => {
                accepts.push(Value::Null);
                runtime_errors.push(json!({"case":index, "error":"runtime panic"}));
            }
        }
    }
    json!({"status":"built", "accepts":accepts, "runtime_errors":runtime_errors})
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = match args.windows(2).find(|pair| pair[0] == "--mode").map(|pair| pair[1].as_str()) {
        None | Some("fast-build") => MaskStrategy::TokenTrie,
        Some("fast-runtime") => MaskStrategy::ParserDwa,
        Some(other) => { eprintln!("unknown --mode {other:?}"); std::process::exit(2); }
    };
    let vocab = Vocab::new((0..=255u32).map(|id| (id, vec![id as u8])).collect());
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let result = match line {
            Ok(line) => match serde_json::from_str::<Value>(&line) {
                Ok(request) => catch_unwind(AssertUnwindSafe(|| probe(&request, &vocab, mode)))
                    .unwrap_or_else(|_| json!({"status":"build_panic", "error":"compile panic"})),
                Err(error) => json!({"status":"input_error", "error":error.to_string()}),
            },
            Err(error) => json!({"status":"input_error", "error":error.to_string()}),
        };
        writeln!(stdout, "{result}").unwrap();
        stdout.flush().unwrap();
    }
}
