use glrmask::{
    compile_grammar_def_json,
    dump_json_schema_grammar_glrm,
    dump_json_schema_prepared_grammar_def,
    Constraint,
    Vocab,
};
use serde_json::{Map, Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::fs;
use std::path::Path;

fn token_allowed(mask: &[u32], id: usize) -> bool {
    let word = id / 32;
    if word >= mask.len() {
        return false;
    }
    (mask[word] >> (id % 32)) & 1 != 0
}

fn build_vocab(entries: &[Vec<u8>]) -> Vocab {
    let mapped: Vec<(u32, Vec<u8>)> = entries
        .iter()
        .enumerate()
        .map(|(i, entry)| (i as u32, entry.clone()))
        .collect();
    Vocab::new(mapped, None)
}

fn predicate_from_schema(schema: &Value, prefix: &[u8], vocab_entries: &[Vec<u8>]) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let schema_json = match serde_json::to_string(schema) {
            Ok(value) => value,
            Err(_) => return false,
        };
        let vocab = build_vocab(vocab_entries);
        let constraint = match Constraint::from_json_schema(&schema_json, &vocab) {
            Ok(value) => value,
            Err(_) => return false,
        };

        let mut mask_state = constraint.start();
        if mask_state.commit_bytes(prefix).is_err() {
            return false;
        }
        let mask_accepts = token_allowed(&mask_state.mask(), 0);

        let mut commit_state = constraint.start();
        if commit_state.commit_bytes(prefix).is_err() {
            return false;
        }
        let commit_accepts = match catch_unwind(AssertUnwindSafe(|| commit_state.commit_token(0))) {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => true,
        };

        !mask_accepts && commit_accepts
    }))
    .unwrap_or(false)
}

fn predicate_from_grammar_json(grammar_json: &str, prefix: &[u8], vocab_entries: &[Vec<u8>]) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        let vocab = build_vocab(vocab_entries);
        let constraint = match compile_grammar_def_json(grammar_json, &vocab) {
            Ok(value) => value,
            Err(_) => return false,
        };

        let mut mask_state = constraint.start();
        if mask_state.commit_bytes(prefix).is_err() {
            return false;
        }
        let mask_accepts = token_allowed(&mask_state.mask(), 0);

        let mut commit_state = constraint.start();
        if commit_state.commit_bytes(prefix).is_err() {
            return false;
        }
        let commit_accepts = match catch_unwind(AssertUnwindSafe(|| commit_state.commit_token(0))) {
            Ok(Ok(())) => true,
            Ok(Err(_)) => false,
            Err(_) => true,
        };

        !mask_accepts && commit_accepts
    }))
    .unwrap_or(false)
}

fn ddmin_bytes<F>(bytes: &mut Vec<u8>, predicate: F)
where
    F: Fn(&[u8]) -> bool,
{
    if bytes.is_empty() {
        return;
    }

    let mut granularity = 2usize;
    while bytes.len() >= 2 {
        let chunk = bytes.len().div_ceil(granularity);
        let mut reduced = false;

        for index in 0..granularity {
            let start = index * chunk;
            if start >= bytes.len() {
                break;
            }
            let end = ((index + 1) * chunk).min(bytes.len());
            let mut candidate = bytes.clone();
            candidate.drain(start..end);
            if candidate.is_empty() {
                continue;
            }
            if predicate(&candidate) {
                *bytes = candidate;
                granularity = 2;
                reduced = true;
                break;
            }
        }

        if reduced {
            continue;
        }

        if granularity >= bytes.len() {
            break;
        }
        granularity = (granularity * 2).min(bytes.len());
    }
}

fn minimize_vocab<F>(vocab_entries: &mut Vec<Vec<u8>>, predicate: F)
where
    F: Fn(&[Vec<u8>]) -> bool,
{
    loop {
        let mut changed = false;

        for index in (1..vocab_entries.len()).rev() {
            let mut candidate = vocab_entries.clone();
            candidate.remove(index);
            if predicate(&candidate) {
                *vocab_entries = candidate;
                changed = true;
                break;
            }
        }

        if !changed {
            break;
        }
    }
}

fn string_candidates(value: &str) -> Vec<String> {
    let mut out = Vec::new();
    for candidate in ["", "a", "id", "string", "object", "true"] {
        if candidate != value {
            out.push(candidate.to_string());
        }
    }
    if !value.is_empty() {
        out.push(value[..value.len() / 2].to_string());
        out.push(value.chars().take(1).collect());
    }
    out.sort();
    out.dedup();
    out
}

fn scalar_replacements(value: &Value) -> Vec<Value> {
    match value {
        Value::Null => Vec::new(),
        Value::Bool(current) => [true, false]
            .into_iter()
            .filter(|candidate| candidate != current)
            .map(Value::Bool)
            .collect(),
        Value::Number(current) => [0, 1, 2, 4, 8, 16]
            .into_iter()
            .filter_map(|candidate| {
                let value = Value::from(candidate);
                (value != Value::Number(current.clone())).then_some(value)
            })
            .collect(),
        Value::String(current) => string_candidates(current)
            .into_iter()
            .map(Value::String)
            .collect(),
        _ => Vec::new(),
    }
}

fn try_reduce_value(value: &mut Value, predicate: &dyn Fn(&Value) -> bool) -> bool {
    match value {
        Value::Object(map) => try_reduce_object(map, predicate),
        Value::Array(array) => try_reduce_array(array, predicate),
        _ => {
            for replacement in scalar_replacements(value) {
                if predicate(&replacement) {
                    *value = replacement;
                    return true;
                }
            }
            false
        }
    }
}

fn try_reduce_object(map: &mut Map<String, Value>, predicate: &dyn Fn(&Value) -> bool) -> bool {
    let keys: Vec<String> = map.keys().cloned().collect();
    for key in &keys {
        let mut candidate = map.clone();
        candidate.remove(key);
        if predicate(&Value::Object(candidate.clone())) {
            *map = candidate;
            return true;
        }
    }

    for key in keys {
        let Some(original_child) = map.get(&key).cloned() else {
            continue;
        };

        for replacement in scalar_replacements(&original_child) {
            let mut candidate = map.clone();
            candidate.insert(key.clone(), replacement);
            if predicate(&Value::Object(candidate.clone())) {
                *map = candidate;
                return true;
            }
        }

        let mut child = original_child.clone();
        if try_reduce_value(&mut child, &|reduced_child| {
            let mut candidate = map.clone();
            candidate.insert(key.clone(), reduced_child.clone());
            predicate(&Value::Object(candidate))
        }) {
            map.insert(key.clone(), child);
            return true;
        }
    }

    false
}

fn try_reduce_array(array: &mut Vec<Value>, predicate: &dyn Fn(&Value) -> bool) -> bool {
    for index in (0..array.len()).rev() {
        let mut candidate = array.clone();
        candidate.remove(index);
        if predicate(&Value::Array(candidate.clone())) {
            *array = candidate;
            return true;
        }
    }

    for index in 0..array.len() {
        let original_child = array[index].clone();

        for replacement in scalar_replacements(&original_child) {
            let mut candidate = array.clone();
            candidate[index] = replacement;
            if predicate(&Value::Array(candidate.clone())) {
                *array = candidate;
                return true;
            }
        }

        let mut child = original_child.clone();
        if try_reduce_value(&mut child, &|reduced_child| {
            let mut candidate = array.clone();
            candidate[index] = reduced_child.clone();
            predicate(&Value::Array(candidate))
        }) {
            array[index] = child;
            return true;
        }
    }

    false
}

fn minimize_json_value(value: &mut Value, predicate: &dyn Fn(&Value) -> bool) {
    while try_reduce_value(value, predicate) {}
}

fn write_output(path: &Path, content: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    fs::write(path, content).expect("write output file");
}

fn build_object_string_prefix(key: &str, content: &[u8]) -> Vec<u8> {
    let mut prefix = format!("{{\"{key}\":\"").into_bytes();
    prefix.extend_from_slice(content);
    prefix
}

fn main() {
    std::panic::set_hook(Box::new(|_| {}));
    let schema_only = std::env::var_os("GLRMASK_MINIMIZE_SCHEMA_ONLY").is_some();

    let mut schema = json!({
        "type": "object",
        "properties": {
            "aside": { "type": "boolean" },
            "autoplay": { "type": "boolean" },
            "css_class": {
                "type": "string",
                "pattern": "^[\\w\\s-]+$"
            },
            "description": {
                "type": "string",
                "minLength": 0,
                "maxLength": 5000
            }
        },
        "required": ["id"],
        "additionalProperties": true
    });

    let phrase = b"This is a Vimeo video block. ".to_vec();
    let mut content = std::iter::repeat(phrase.as_slice())
        .take(79)
        .flatten()
        .copied()
        .collect::<Vec<_>>();
    content.extend_from_slice(b"This is a");
    let mut prefix = Vec::from(
        b"{\"aside\": true, \"autoplay\": false, \"css_class\": \"vimeo-video-block\", \"description\": \"".as_slice(),
    );
    prefix.extend_from_slice(&content);

    let mut vocab_entries = vec![b"'];?>\"".to_vec(), b" Vimeo".to_vec()];

    assert!(
        predicate_from_schema(&schema, &prefix, &vocab_entries),
        "starting case must reproduce mask=false, commit=true"
    );

    minimize_vocab(&mut vocab_entries, |candidate_vocab| {
        predicate_from_schema(&schema, &prefix, candidate_vocab)
    });

    for candidate_schema in [
        json!({"type": "object", "required": ["a"], "additionalProperties": true}),
        json!({"type": "object", "properties": {}, "required": ["a"], "additionalProperties": true}),
    ] {
        if predicate_from_schema(&candidate_schema, &prefix, &vocab_entries) {
            schema = candidate_schema;
            break;
        }
    }

    let mut short_prefix = build_object_string_prefix("b", &content);
    if predicate_from_schema(&schema, &short_prefix, &vocab_entries) {
        prefix = short_prefix.clone();
    }

    let mut low = 0usize;
    let mut high = 79usize;
    let suffix = b"This is a".to_vec();
    while low < high {
        let mid = (low + high) / 2;
        let mut candidate_content = std::iter::repeat(phrase.as_slice())
            .take(mid)
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        candidate_content.extend_from_slice(&suffix);
        let candidate_prefix = build_object_string_prefix("b", &candidate_content);
        if predicate_from_schema(&schema, &candidate_prefix, &vocab_entries) {
            high = mid;
            content = candidate_content;
            prefix = candidate_prefix;
        } else {
            low = mid + 1;
        }
    }

    ddmin_bytes(&mut content, |candidate_content| {
        let candidate_prefix = build_object_string_prefix("b", candidate_content);
        predicate_from_schema(&schema, &candidate_prefix, &vocab_entries)
    });
    short_prefix = build_object_string_prefix("b", &content);
    if predicate_from_schema(&schema, &short_prefix, &vocab_entries) {
        prefix = short_prefix;
    }

    minimize_json_value(&mut schema, &|candidate_schema| {
        predicate_from_schema(candidate_schema, &prefix, &vocab_entries)
    });
    ddmin_bytes(&mut prefix, |candidate_prefix| {
        predicate_from_schema(&schema, candidate_prefix, &vocab_entries)
    });

    let schema_json = serde_json::to_string_pretty(&schema).expect("serialize minimized schema");
    let vocab_json = serde_json::to_string_pretty(
        &vocab_entries
            .iter()
            .map(|bytes| String::from_utf8_lossy(bytes).to_string())
            .collect::<Vec<_>>()
    )
    .expect("serialize minimized vocab");

    println!("=== Minimized Vocab ===");
    for (index, token) in vocab_entries.iter().enumerate() {
        println!("token[{index}] = {:?}", String::from_utf8_lossy(token));
    }
    println!("\n=== Minimized Prefix Bytes ===\n{:?}", String::from_utf8_lossy(&prefix));
    println!("\n=== Minimized Schema ===\n{schema_json}");

    write_output(
        Path::new("/Users/isaacbreen/Projects2/glrmask2/tmp/o82710_minimized_schema.json"),
        &schema_json,
    );
    write_output(
        Path::new("/Users/isaacbreen/Projects2/glrmask2/tmp/o82710_minimized_prefix.txt"),
        &String::from_utf8_lossy(&prefix),
    );
    write_output(
        Path::new("/Users/isaacbreen/Projects2/glrmask2/tmp/o82710_minimized_vocab.json"),
        &vocab_json,
    );

    if schema_only {
        return;
    }

    let prepared_grammar_json = dump_json_schema_prepared_grammar_def(&schema_json)
        .expect("dump prepared grammar def");
    let mut grammar_value: Value = serde_json::from_str(&prepared_grammar_json)
        .expect("parse prepared grammar def json");
    assert!(
        predicate_from_grammar_json(&prepared_grammar_json, &prefix, &vocab_entries),
        "prepared grammar form should preserve the mismatch predicate"
    );
    minimize_json_value(&mut grammar_value, &|candidate_grammar| {
        let candidate_json = serde_json::to_string(candidate_grammar).expect("serialize grammar candidate");
        predicate_from_grammar_json(&candidate_json, &prefix, &vocab_entries)
    });
    let minimized_grammar_json = serde_json::to_string_pretty(&grammar_value)
        .expect("serialize minimized grammar def");

    let glrm_text = dump_json_schema_grammar_glrm(&schema_json)
        .expect("dump GLRM grammar text for minimized schema");

    println!("\n=== Minimized Prepared GrammarDef JSON ===\n{minimized_grammar_json}");
    println!("\n=== GLRM Grammar (from minimized schema) ===\n{glrm_text}");

    write_output(
        Path::new("/Users/isaacbreen/Projects2/glrmask2/tmp/o82710_minimized_prepared_grammar.json"),
        &minimized_grammar_json,
    );
    write_output(
        Path::new("/Users/isaacbreen/Projects2/glrmask2/tmp/o82710_minimized_schema.glrm"),
        &glrm_text,
    );
}