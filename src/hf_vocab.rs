use crate::{Error, Result, Vocab, error::ErrorExt};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};
use toktrie_hf_tokenizers::ByteTokenizer;

/// Hugging Face vocabulary constructors (feature `hf`).
/// Import this trait alongside `Vocab`. Special and padded IDs have empty
/// byte spellings. SentencePiece's initial dummy space remains in the bytes.
pub trait VocabFromTokenizer: Sized {
    /// Loads a serialized Hugging Face tokenizer, optionally padding to the model vocabulary size.
    fn from_tokenizer_json(json: &str, vocab_size: Option<usize>) -> Result<Self>;
    /// Loads a local tokenizer.json file.
    fn from_tokenizer_file(path: impl AsRef<Path>, vocab_size: Option<usize>) -> Result<Self>;
}
impl VocabFromTokenizer for Vocab {
    fn from_tokenizer_json(json: &str, vocab_size: Option<usize>) -> Result<Self> {
        Self::new(extract(json, vocab_size)?)
    }
    fn from_tokenizer_file(path: impl AsRef<Path>, vocab_size: Option<usize>) -> Result<Self> {
        let json = std::fs::read_to_string(path).map_err(|e| invalid(e.to_string()))?;
        Self::from_tokenizer_json(&json, vocab_size)
    }
}
fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidInput(message.into())
}
fn extract(json: &str, vocab_size: Option<usize>) -> Result<Vec<(u32, Vec<u8>)>> {
    const LIMIT: usize = crate::vocab::MAX_TOKEN_ID as usize + 1;
    if vocab_size.is_some_and(|n| n > LIMIT) {
        return Err(invalid("vocabulary size exceeds supported range"));
    }
    let spec: Value = serde_json::from_str(json).map_err(|e| invalid(e.to_string()))?;
    let model = &spec["model"];
    let mut ids = BTreeMap::new();
    if let Some(vocab) = model["vocab"].as_object() {
        for (name, id) in vocab {
            let id = id.as_u64().filter(|&id| id < LIMIT as u64)
                .ok_or_else(|| invalid("invalid tokenizer vocabulary ID"))? as u32;
            if ids.insert(id, name.as_str()).is_some() {
                return Err(invalid("duplicate tokenizer vocabulary ID"));
            }
        }
    } else {
        let vocab = model["vocab"].as_array().ok_or_else(|| invalid("missing tokenizer vocabulary"))?;
        if vocab.len() > LIMIT { return Err(invalid("vocabulary size exceeds supported range")); }
        for (id, row) in vocab.iter().enumerate() {
            let name = row[0].as_str().ok_or_else(|| invalid("invalid tokenizer vocabulary entry"))?;
            ids.insert(id as u32, name);
        }
    }
    let mut added = BTreeMap::new();
    if let Some(tokens) = spec["added_tokens"].as_array() {
        for token in tokens {
            let id = token["id"].as_u64().filter(|&id| id < LIMIT as u64)
                .ok_or_else(|| invalid("invalid added token ID"))? as u32;
            let text = token["content"].as_str().ok_or_else(|| invalid("invalid added token content"))?;
            let special = token["special"].as_bool().ok_or_else(|| invalid("invalid added token special flag"))?;
            if ids.insert(id, text).is_some_and(|old| old != text) || added.insert(id, (text, special)).is_some() {
                return Err(invalid("conflicting added token ID"));
            }
        }
    }
    let size = ids.len();
    // Upstream indexes a dense vector by ID.
    if ids.keys().copied().ne(0..size as u32) {
        return Err(invalid("sparse tokenizer vocabulary is unsupported"));
    }
    let target = vocab_size.unwrap_or(size);
    if target < size { return Err(invalid("vocab_size is smaller than tokenizer vocabulary")); }
    let bt = std::panic::catch_unwind(|| ByteTokenizer::from_json_bytes(json.as_bytes()))
        .map_err(|_| invalid("unsupported or malformed tokenizer"))?
        .map_err(|e| invalid(e.to_string()))?;
    let bytes = bt.token_bytes();
    if bytes.len() != size { return Err(invalid("tokenizer vocabulary size mismatch")); }
    let unk = model["unk_token"].as_str();
    let unk_id = model["unk_id"].as_u64();
    let mut entries = Vec::with_capacity(target);
    for (id, mut bytes) in bytes.into_iter().enumerate() {
        let id = id as u32;
        // Upstream's marker (0xff) is also a valid ordinary byte. Use
        // metadata, including ordinary angle-bracketed added tokens.
        if let Some(&(text, special)) = added.get(&id) {
            bytes = if special { Vec::new() } else { text.as_bytes().to_vec() };
        } else if (unk.is_some() && unk == ids.get(&id).copied()) || unk_id == Some(id as u64) {
            bytes.clear();
        } else if bytes.is_empty() {
            return Err(invalid(format!("no byte spelling for token {id}")));
        }
        entries.push((id, bytes));
    }
    entries.extend((size..target).map(|id| (id as u32, Vec::new())));
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_tokenizers() {
        let Some(root) = std::env::var_os("GLRMASK_HF_FIXTURES") else {
            eprintln!("Set GLRMASK_HF_FIXTURES to run the real-model suite");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let cases: Value = serde_json::from_str(&std::fs::read_to_string(root.join("cases.json")).unwrap()).unwrap();
        for model in cases.as_array().unwrap() {
            let name = model["model"].as_str().unwrap();
            let source = std::fs::read_to_string(root.join(model["file"].as_str().unwrap())).unwrap();
            let bytes = extract(&source, None).unwrap_or_else(|e| panic!("{name}: {e}"));
            for case in model["cases"].as_array().unwrap() {
                let mut got = Vec::new();
                for id in case["ids"].as_array().unwrap() {
                    let id = id.as_u64().unwrap() as usize;
                    assert!(!bytes[id].1.is_empty(), "{name}: unexpected byte-less ID {id}");
                    got.extend_from_slice(&bytes[id].1);
                }
                let expected = case["expected"].as_str().unwrap().as_bytes();
                if model["dummy_prefix"].as_bool().unwrap() && got != expected && got.starts_with(b" ") {
                    got.remove(0);
                }
                assert_eq!(got, expected, "{name}: {:?}", case["expected"]);
            }
            println!("{name}: {} reference cases OK", model["cases"].as_array().unwrap().len());
        }
    }

    #[test]
    fn special_and_padding_preserve_raw_ff() {
        let json = r#"{
            "version":"1.0","truncation":null,"padding":null,
            "added_tokens":[
              {"id":2,"content":"<special>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},
              {"id":3,"content":"<ordinary>","special":false,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}
            ],
            "normalizer":null,"pre_tokenizer":null,"post_processor":null,
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
            "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"vocab":{"a":0,"ÿ":1},"merges":[]}
        }"#;
        assert_eq!(extract(json, Some(5)).unwrap(), vec![
            (0,b"a".to_vec()), (1,vec![255]), (2,vec![]),
            (3,b"<ordinary>".to_vec()), (4,vec![]),
        ]);
        assert!(extract(json, Some(3)).is_err());
        assert!(extract("{}", None).is_err());
        assert!(extract(&json.replace("\"ÿ\":1", "\"ÿ\":8"), None).is_err());
    }
}
