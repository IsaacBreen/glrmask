use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::Arc;
use flate2::read::GzDecoder;
use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use sep1::constraint::{GrammarConstraint, GrammarConstraintConfig, Precompute0Cache};
use sep1::interface::{CompiledGrammar, GrammarDefinition};
use sep1::json_serialization::JSONConvertible;
use sep1::tokenizer::{LLMTokenID, LLMTokenMap};

/// Compiles a grammar and vocabulary into a GrammarConstraint object.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the EBNF grammar file.
    #[arg(short, long)]
    grammar: PathBuf,

    /// Path to the JSON vocabulary file.
    /// The file should be a JSON object mapping token strings to integer IDs.
    #[arg(short, long)]
    vocab: PathBuf,

    /// Path for the output compressed JSON file (.json.gz).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optional: load a precompute0 cache from this file to skip rebuilding Trie0.
    #[arg(long, value_name = "FILE")]
    load_precompute0: Option<PathBuf>,

    /// Optional: save the precompute0 stage to this file for reuse.
    #[arg(long, value_name = "FILE")]
    save_precompute0: Option<PathBuf>,

    /// If specified, only compute and save the precompute0 cache, then exit.
    #[arg(long, requires = "save_precompute0")]
    precompute0_only: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if !args.precompute0_only && args.output.is_none() {
        return Err("Error: --output is required unless --precompute0-only is specified.".into());
    }

    // 1. Load and compile the grammar.
    println!("Loading grammar from: {:?}", args.grammar);
    let grammar_path_str = args.grammar.to_str().ok_or_else(|| format!("Path is not valid UTF-8: {:?}", args.grammar))?;
    let grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path_str)?;
    println!("Compiling grammar...");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_definition));
    println!("Grammar compiled successfully.");

    // 2. Load the vocabulary.
    println!("Loading vocabulary from: {:?}", args.vocab);
    let vocab_file = File::open(&args.vocab)?;
    let reader = BufReader::new(vocab_file);
    let vocab: BTreeMap<String, usize> = serde_json::from_reader(reader)?;

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id = 0;

    for (token_str, token_id) in vocab {
        let processed_token_str = token_str.replace("Ġ", " ").replace("ą", "\n").replace("Ċ", "\n");
        let token_bytes = processed_token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(token_id));
        max_original_llm_token_id = max_original_llm_token_id.max(token_id);
    }
    println!("Vocabulary loaded ({} tokens, max ID: {}).", llm_token_map.len(), max_original_llm_token_id);

    // 3. Construct the GrammarConstraint.
    let _dummy_eof_token_id = LLMTokenID(max_original_llm_token_id + 1);
    println!("\nConstructing GrammarConstraint...");
    let mut loaded_pc0: Option<Precompute0Cache> = None;
    if let Some(path) = args.load_precompute0.as_ref() {
        println!("Attempting to load precompute0 cache from: {:?}", path);
        match File::open(path) {
            Ok(f) => {
                let mut dec = GzDecoder::new(f);
                match Precompute0Cache::from_reader(&mut dec) {
                    Ok(cache) => {
                        println!("Loaded precompute0 cache successfully.");
                        loaded_pc0 = Some(cache);
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse precompute0 cache: {}. Will recompute.", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Could not open precompute0 cache file: {}. Will recompute.", e);
            }
        }
    }

    let mut config = GrammarConstraintConfig::default();
    config.precompute0_only = args.precompute0_only;

    let grammar_constraint = GrammarConstraint::new_with_config_and_precompute0_cache(
        compiled_grammar.tokenizer,
        compiled_grammar.glr_parser,
        llm_token_map,
        compiled_grammar.definition.terminal_to_group_id().clone(),
        max_original_llm_token_id,
        &config,
        loaded_pc0,
    );
    println!("GrammarConstraint constructed successfully.");
    if let Some(path) = args.save_precompute0.as_ref() {
        println!("Saving precompute0 cache to: {:?}", path);
        let output_file = File::create(path)?;
        let writer = BufWriter::new(output_file);
        let mut encoder = GzEncoder::new(writer, Compression::default());
        grammar_constraint.export_precompute0_cache().to_writer(&mut encoder)?;
        encoder.finish()?;
    }

    if args.precompute0_only {
        println!("--precompute0-only specified. Exiting after saving cache.");
        return Ok(());
    }

    // 4. Save the GrammarConstraint to a compressed file.
    if let Some(output_path) = args.output {
        println!("Saving GrammarConstraint to: {:?}", output_path);
        let output_file = File::create(&output_path)?;
        let writer = BufWriter::new(output_file);
        let mut encoder = GzEncoder::new(writer, Compression::default());
        grammar_constraint.to_writer(&mut encoder)?;
        encoder.finish()?;
        println!("Successfully saved constraint to {:?}", output_path);
    }

    Ok(())
}
