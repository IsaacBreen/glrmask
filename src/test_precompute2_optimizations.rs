// This file contains tests for optimization passes on the Precompute2 trie.
// It verifies that optimizations produce a semantically equivalent tree.

use crate::constraint::{
    are_precompute2_trees_equivalent, clone_trie2_graph, context_aware_merge_trie2, GrammarConstraint, Precomputed2,
};
use crate::json_serialization::JSONConvertible;
use crate::interface::{CompiledGrammar, GrammarDefinition};
use crate::tokenizer::{LLMTokenID, LLMTokenMap};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::Arc;
use reqwest::blocking;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::error::Error;

// Helper function copied from test_constraint_js.rs
fn load_or_download_gpt2_vocab(
    cache_dir: &Path,
    file_name: &str,
    url: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    fs::create_dir_all(cache_dir)?;
    let cache_path = cache_dir.join(file_name);

    let vocab_map: BTreeMap<String, u32> = if cache_path.exists() {
        println!("Loading GPT-2 vocab from cache: {:?}", cache_path);
        let file = File::open(cache_path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)?
    } else {
        println!("Downloading GPT-2 vocab from: {}", url);
        let response = blocking::get(url)?.error_for_status()?;
        let content = response.text()?;

        let mut file = File::create(&cache_path)?;
        file.write_all(content.as_bytes())?;
        println!("Saved GPT-2 vocab to cache: {:?}", cache_path);

        serde_json::from_str(&content)?
    };
    Ok(vocab_map.into_keys().collect())
}

// --- Compact Helpers for Inline EBNF-based tests ---

fn compiled_grammar_from_inline_ebnf(ebnf_src: &str) -> Result<CompiledGrammar, Box<dyn Error>> {
    // Persist inline EBNF to a stable cache file (based on a hash) and compile from that path.
    let cache_dir = Path::new(".cache/test_precompute2/inline_ebnf");
    fs::create_dir_all(cache_dir)?;
    let mut hasher = DefaultHasher::new();
    ebnf_src.hash(&mut hasher);
    let hash = hasher.finish();
    let ebnf_path = cache_dir.join(format!("inline_{}.ebnf", hash));
    fs::write(&ebnf_path, ebnf_src)?;
    let grammar_definition = GrammarDefinition::from_ebnf_file(ebnf_path.to_str().unwrap())?;
    Ok(CompiledGrammar::from_definition(Arc::new(grammar_definition)))
}

fn build_llm_map(tokens: &[&str]) -> (LLMTokenMap, usize) {
    let mut map = LLMTokenMap::new();
    let mut max_id = 0usize;
    for (i, tok) in tokens.iter().enumerate() {
        map.insert(tok.as_bytes().to_vec(), LLMTokenID(i));
        max_id = max_id.max(i);
    }
    (map, max_id)
}

fn precomputed2_from_ebnf_and_tokens(ebnf_src: &str, tokens: &[&str]) -> Result<Precomputed2, Box<dyn Error>> {
    let compiled = compiled_grammar_from_inline_ebnf(ebnf_src)?;
    let (llm_token_map, max_original_llm_token_id) = build_llm_map(tokens);
    let gc = GrammarConstraint::from_compiled_grammar(
        compiled,
        llm_token_map,
        LLMTokenID(0), // dummy EOF placeholder
        max_original_llm_token_id,
    );
    Ok(gc.precomputed2)
}

fn optimize_and_assert_equivalent(pre2: &Precomputed2) {
    // Deep-clone the original tree graph-wise (preserve DAG sharing if any)
    let mut optimized_pre2: Precomputed2 = BTreeMap::new();
    for (sid, root_arc) in pre2.iter() {
        let (cloned_root, _node_map) = clone_trie2_graph(root_arc);
        optimized_pre2.insert(*sid, cloned_root);
    }
    // Apply optimization passes
    context_aware_merge_trie2(&mut optimized_pre2);

    // Compare semantic equivalence per tokenizer-state root
    assert_eq!(
        pre2.len(),
        optimized_pre2.len(),
        "Number of tokenizer roots differs after optimization"
    );
    for sid in pre2.keys() {
        let a = pre2.get(sid).unwrap();
        let b = optimized_pre2.get(sid).unwrap();
        assert!(
            are_precompute2_trees_equivalent(a, b),
            "Mismatch for tokenizer state ID {}",
            sid.0
        );
    }
}

#[cfg(not(rustrover))]
#[test]
fn test_precompute2_optimizations_are_equivalent() -> Result<(), Box<dyn std::error::Error>> {
    // --- Setup Phase ---
    println!("--- Setting up for Precompute2 Optimization Equivalence Test ---");

    const FORCE_RECOMPUTE: bool = false;
    const SAVE_TO_CACHE: bool = false;
    use flate2::read::GzDecoder;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use crate::json_serialization;
    let cache_dir = Path::new(".cache/test_precompute2");
    fs::create_dir_all(cache_dir)?;
    let precomputed2_cache_path = cache_dir.join("precomputed2_js_gpt2_small.json.gz");

    // 1. Load and compile the JavaScript grammar.
    let grammar_path = "src/js.ebnf";
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path)?;
    println!("Compiling GrammarDefinition into CompiledGrammar...");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Successfully compiled GrammarDefinition.");

    // 2. Load a small, representative vocabulary.
    println!("\nLoading GPT-2 vocabulary...");
    let vocab_cache_dir = Path::new(".cache/test_vocabs");
    let vocab_url = "https://huggingface.co/openai-community/gpt2/raw/main/vocab.json";
    let vocab_file_name = "gpt2_vocab.json";
    let mut gpt2_raw_vocab = load_or_download_gpt2_vocab(vocab_cache_dir, vocab_file_name, vocab_url)?;
    // Keep a smaller subset to speed up the test
    gpt2_raw_vocab.retain(|s| s.len() < 5);
    println!("Using a subset of {} tokens for the test.", gpt2_raw_vocab.len());

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id_val: usize = 0;
    for (i, token_str) in gpt2_raw_vocab.iter().enumerate() {
        let id_val = i;
        let processed_token_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
        let token_bytes = processed_token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(id_val));
        max_original_llm_token_id_val = max_original_llm_token_id_val.max(id_val);
    }

    // 3. Construct the GrammarConstraint to get the baseline precomputed2 tree.
    let original_precomputed2: Precomputed2;

    if !FORCE_RECOMPUTE && precomputed2_cache_path.exists() {
        println!("\nLoading Precomputed2 from cache: {:?}", precomputed2_cache_path);
        let file = File::open(&precomputed2_cache_path)?;
        let decompressor = GzDecoder::new(BufReader::new(file));
        // Reading is not fully streamed to avoid complexity, but this avoids storing the huge uncompressed file on disk.
        original_precomputed2 = Precomputed2::from_json_reader(decompressor)?;
        println!("Successfully loaded Precomputed2 from cache.");
    } else {
        println!("\nConstructing GrammarConstraint (will generate Precomputed2)...");
        let grammar_constraint = GrammarConstraint::from_compiled_grammar(
            compiled_grammar.clone(),
            llm_token_map.clone(),
            LLMTokenID(0), // dummy_eof_placeholder
            max_original_llm_token_id_val,
        );
        println!("GrammarConstraint constructed successfully.");
        original_precomputed2 = grammar_constraint.precomputed2;

        println!("\nSerializing and saving Precomputed2 to cache: {:?}", precomputed2_cache_path);
        if SAVE_TO_CACHE {
            let file = File::create(&precomputed2_cache_path)?;
            let writer = BufWriter::new(file);
            let mut encoder = GzEncoder::new(writer, Compression::default());
            json_serialization::write_precomputed2_to_stream(&original_precomputed2, &mut encoder)?;
        }
        println!("Successfully saved Precomputed2 to cache.");
    }

    // 4. Deep-clone the original precomputed2 tree.
    println!("\nCloning the original precomputed2 tree...");
    let mut optimized_precomputed2: Precomputed2 = BTreeMap::new();
    for (sid, root_arc) in original_precomputed2.iter() {
        let (cloned_root, _map) = clone_trie2_graph(root_arc);
        optimized_precomputed2.insert(*sid, cloned_root);
    }
    println!("Cloning complete.");

    // 5. Apply optimization passes to the cloned tree.
    println!("\nApplying optimization passes to the cloned tree...");
    context_aware_merge_trie2(&mut optimized_precomputed2);
    println!("Optimization passes applied.");

    // 6. Compare the original and optimized trees for semantic equivalence.
    println!("\nComparing original and optimized trees for equivalence...");
    assert_eq!(original_precomputed2.len(), optimized_precomputed2.len(), "Number of tokenizer states with trees differs.");

    for sid in original_precomputed2.keys() {
        let original_root = original_precomputed2.get(sid).unwrap();
        let optimized_root = optimized_precomputed2.get(sid).unwrap();

        assert!(are_precompute2_trees_equivalent(original_root, optimized_root), "Mismatch found for tokenizer state ID: {}", sid.0);
        println!("  - OK for tokenizer state ID: {}", sid.0);
    }

    println!("\nEquivalence test passed: All trees are semantically equivalent after optimization.");

    Ok(())
}
