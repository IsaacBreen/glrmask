use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use glrmask::{Constraint, Vocab};
use glrmask::__private::ConstraintExt as _;
use glrmask_runtime::RuntimeArtifact;
use serde_json::{json, Value};

fn byte_decoder() -> BTreeMap<char, u8> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|&byte| byte as u32).collect();
    let mut next = 0u32;
    for byte in 0u8..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            chars.push(256 + next);
            next += 1;
        }
    }
    chars
        .into_iter()
        .zip(bytes)
        .filter_map(|(codepoint, byte)| char::from_u32(codepoint).map(|ch| (ch, byte)))
        .collect()
}

fn load_gpt_style_vocab(tokenizer_path: &Path) -> Result<Vocab, String> {
    let text = fs::read_to_string(tokenizer_path)
        .map_err(|error| format!("read {}: {error}", tokenizer_path.display()))?;
    let tokenizer: Value = serde_json::from_str(&text)
        .map_err(|error| format!("parse {}: {error}", tokenizer_path.display()))?;
    let vocab = tokenizer
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| "tokenizer.json does not contain model.vocab".to_owned())?;
    let decoder = byte_decoder();
    let mut entries = Vec::with_capacity(vocab.len());
    for (token, id_value) in vocab {
        let id = id_value
            .as_u64()
            .ok_or_else(|| format!("non-numeric token id for {token:?}"))? as u32;
        let bytes = token
            .chars()
            .map(|ch| {
                decoder.get(&ch).copied().ok_or_else(|| {
                    format!("token {token:?} contains non-GPT-byte character {ch:?}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        entries.push((id, bytes));
    }
    Ok(Vocab::new(entries, None))
}

fn story_json_grammar() -> String {
    // {"story":"..."}; this bounded terminal only admits raw bytes which are
    // directly legal inside the JSON string. Its finite upper bound guarantees
    // that live constrained sampling eventually has to close the quote/object.
    json!({
        "rules": [
            {"lhs": 0, "rhs": [{"Terminal": 0}, {"Terminal": 1}, {"Terminal": 2}, {"Terminal": 3}, {"Terminal": 5}, {"Terminal": 3}, {"Terminal": 4}]}
        ],
        "start": 0,
        "terminals": [
            {"Literal": {"id": 0, "bytes": [123]}},
            {"Literal": {"id": 1, "bytes": [34, 115, 116, 111, 114, 121, 34]}},
            {"Literal": {"id": 2, "bytes": [58]}},
            {"Literal": {"id": 3, "bytes": [34]}},
            {"Literal": {"id": 4, "bytes": [125]}},
            {"Pattern": {"id": 5, "pattern": "[A-Za-z0-9 .,!?'-]{1,96}", "utf8": false}}
        ],
        "nonterminal_names": {"0": "json_story"},
        "terminal_names": {"0": "{", "1": "\\\"story\\\"", "2": ":", "3": "\\\"", "4": "}", "5": "story chars"},
        "ignore_terminal": null
    })
    .to_string()
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let tokenizer = args
        .next()
        .ok_or_else(|| "usage: glrmask-browser-artifact <tokenizer.json> <output.glrmaskc>".to_owned())?;
    let output = args
        .next()
        .ok_or_else(|| "usage: glrmask-browser-artifact <tokenizer.json> <output.glrmaskc>".to_owned())?;
    if args.next().is_some() {
        return Err("usage: glrmask-browser-artifact <tokenizer.json> <output.glrmaskc>".to_owned());
    }

    let vocab = load_gpt_style_vocab(Path::new(&tokenizer))?;
    eprintln!("loaded {} TinyStories BPE tokens", vocab.len());
    let constraint = Constraint::compile_grammar_def_json(&story_json_grammar(), &vocab)
        .map_err(|error| error.to_string())?;
    let artifact = RuntimeArtifact::from_runtime_payload_v2(constraint.save_runtime_payload_v2());
    let output = Path::new(&output);
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    fs::write(output, artifact.as_bytes())
        .map_err(|error| format!("write {}: {error}", output.display()))?;
    eprintln!("wrote {} bytes to {}", artifact.as_bytes().len(), output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
