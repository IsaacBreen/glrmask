use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use sep1::constraint::{GrammarConstraint, GrammarConstraintConfig};
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
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 1. Load and compile the grammar.
    println!("Loading grammar from: {:?}", args.grammar);
    let grammar_definition = GrammarDefinition::from_ebnf_file(&args.grammar)?;
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
    let dummy_eof_token_id = LLMTokenID(max_original_llm_token_id + 1);
    println!("\nConstructing GrammarConstraint...");
    let grammar_constraint = GrammarConstraint::from_compiled_grammar_with_config(
        compiled_grammar,
        llm_token_map,
        dummy_eof_token_id,
        max_original_llm_token_id,
        &GrammarConstraintConfig::default(),
    );
    println!("GrammarConstraint constructed successfully.");

    // 4. Save the GrammarConstraint to a compressed file.
    println!("Saving GrammarConstraint to: {:?}", args.output);
    let output_file = File::create(&args.output)?;
    let writer = BufWriter::new(output_file);
    let mut encoder = GzEncoder::new(writer, Compression::default());
    grammar_constraint.to_writer(&mut encoder)?;
    encoder.finish()?;
    println!("Successfully saved constraint to {:?}", args.output);

    Ok(())
}
