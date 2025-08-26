// This file contains tests for optimization passes on the Precompute2 trie.
// It verifies that optimizations produce a semantically equivalent tree.

use crate::constraint::{are_precompute2_trees_equivalent, clone_trie2_graph, context_aware_merge_trie2, GrammarConstraint, Precomputed2};
use crate::json_serialization::{JSONConvertible, JSONNode};
use crate::interface::{CompiledGrammar, GrammarDefinition};
use crate::tokenizer::{LLMTokenID, LLMTokenMap};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::{Arc, RwLock};
use reqwest::blocking;
use crate::datastructures::gss::PrecomputeNode2;

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

/// Custom streaming serializer for a single Trie to avoid high memory usage.
/// This function serializes a Trie graph into a self-contained JSON object to a writer.
fn stream_trie_to_writer<W: Write>(
    root_arc: &Arc<RwLock<PrecomputeNode2>>,
    mut writer: W,
) -> Result<(), String> {
    // Pass 1: Discover all nodes via BFS and assign unique indices.
    let mut ptr_to_idx: HashMap<*const RwLock<PrecomputeNode2>, usize> = HashMap::new();
    let mut idx_to_arc: Vec<Arc<RwLock<PrecomputeNode2>>> = Vec::new();

    ptr_to_idx.insert(Arc::as_ptr(root_arc), 0);
    idx_to_arc.push(root_arc.clone());

    let mut head = 0;
    while head < idx_to_arc.len() {
        let current_arc = &idx_to_arc[head];
        head += 1;

        let guard = current_arc.read().map_err(|_| "RwLock poisoned during discovery".to_string())?;
        for child_map in guard.children().values() {
            for node_ptr in child_map.keys() {
                if let Some(child_arc) = node_ptr.upgrade() {
                    let child_ptr = Arc::as_ptr(&child_arc);
                    if !ptr_to_idx.contains_key(&child_ptr) {
                        let new_idx = idx_to_arc.len();
                        ptr_to_idx.insert(child_ptr, new_idx);
                        idx_to_arc.push(child_arc);
                    }
                }
            }
        }
    }

    // Pass 2: Write nodes to stream one by one.
    write!(writer, "{{\"nodes\":[",).map_err(|e| e.to_string())?;

    for (i, node_arc) in idx_to_arc.iter().enumerate() {
        if i > 0 {
            write!(writer, ",").map_err(|e| e.to_string())?;
        }
        let guard = node_arc.read().map_err(|_| "RwLock poisoned during serialization".to_string())?;

        // Manually construct the JSON for this single node.
        let mut children_json_data = Vec::new();
        let mut weak_children_json_data = Vec::new();

        for (edge_key, destinations_map) in guard.children() {
            let ek_json = edge_key.to_json();
            let mut strong_dests_json = Vec::new();
            let mut weak_dests_json = Vec::new();

            for (node_ptr, edge_val) in destinations_map {
                if let Some(child_arc) = node_ptr.upgrade() {
                    let child_idx = ptr_to_idx.get(&Arc::as_ptr(&child_arc)).unwrap();
                    let dest_entry = JSONNode::Array(vec![child_idx.to_json(), edge_val.to_json()]);
                    if node_ptr.is_strong() {
                        strong_dests_json.push(dest_entry);
                    } else {
                        weak_dests_json.push(dest_entry);
                    }
                }
            }
            if !strong_dests_json.is_empty() {
                children_json_data.push(JSONNode::Array(vec![ek_json.clone(), JSONNode::Array(strong_dests_json)]));
            }
            if !weak_dests_json.is_empty() {
                weak_children_json_data.push(JSONNode::Array(vec![ek_json, JSONNode::Array(weak_dests_json)]));
            }
        }

        let node_json = JSONNode::Object(BTreeMap::from_iter(vec![
            ("value".to_string(), guard.value.to_json()),
            ("max_depth".to_string(), guard.max_depth.to_json()),
            ("children".to_string(), JSONNode::Array(children_json_data)),
            ("weak_children".to_string(), JSONNode::Array(weak_children_json_data)),
        ]));

        node_json.to_writer(&mut writer)?;
    }

    write!(writer, "],\"root_idx\":0}}").map_err(|e| e.to_string())?;
    Ok(())
}

/// Custom streaming serializer for `Precomputed2`.
fn write_precomputed2_to_stream<W: Write>(
    precomputed2: &Precomputed2,
    mut writer: W,
) -> Result<(), String> {
    writer.write_all(b"[").map_err(|e| e.to_string())?;
    let mut first = true;
    for (key, trie_root_arc) in precomputed2 {
        if !first {
            writer.write_all(b",").map_err(|e| e.to_string())?;
        }
        first = false;
        writer.write_all(b"[").map_err(|e| e.to_string())?;
        key.to_json().to_writer(&mut writer)?;
        writer.write_all(b",").map_err(|e| e.to_string())?;
        stream_trie_to_writer(trie_root_arc, &mut writer)?;
        writer.write_all(b"]").map_err(|e| e.to_string())?;
    }
    writer.write_all(b"]").map_err(|e| e.to_string())?;
    Ok(())
}

#[test]
fn test_precompute2_optimizations_are_equivalent() -> Result<(), Box<dyn std::error::Error>> {
    // --- Setup Phase ---
    println!("--- Setting up for Precompute2 Optimization Equivalence Test ---");

    const FORCE_RECOMPUTE: bool = false;
    const SAVE_TO_CACHE: bool = false;
    use flate2::read::GzDecoder;
    use flate2::write::GzEncoder;
    use flate2::Compression;
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
            write_precomputed2_to_stream(&original_precomputed2, &mut encoder)?;
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
