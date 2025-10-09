use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;
use std::sync::Arc;

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
    #[arg(short, long, required_unless_present = "dump_precompute0")]
    output: Option<PathBuf>,

    /// If specified, compute up to precompute0 and dump it to this file.
    #[arg(long)]
    dump_precompute0: Option<PathBuf>,

    /// If specified, load a precompute0 cache and continue compilation.
    #[arg(long, conflicts_with = "dump_precompute0")]
    load_precompute0: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.output.is_none() && args.dump_precompute0.is_none() {
        // This case is prevented by clap's `required_unless_present`
        eprintln!("Error: You must specify an output file with --output OR a cache file to dump with --dump-precompute0.");
        std::process::exit(1);
    }

    if args.load_precompute0.is_some() && args.dump_precompute0.is_some() {
        eprintln!("Error: --load-precompute0 and --dump-precompute0 cannot be used at the same time.");
        std::process::exit(1);
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
    let dummy_eof_token_id = LLMTokenID(max_original_llm_token_id + 1);
    let config = GrammarConstraintConfig::default();

    // 3a. Handle dumping precompute0 cache if requested
    if let Some(dump_path) = &args.dump_precompute0 {
        println!("\nBuilding precompute0 cache...");
        let pc0_cache = GrammarConstraint::build_precompute0_cache(
            compiled_grammar.clone(), // Clone grammar for potential later use
            llm_token_map.clone(), // Clone map for potential later use
            dummy_eof_token_id,
            max_original_llm_token_id,
            &config,
        );
        println!("Saving precompute0 cache to: {:?}", dump_path);
        let output_file = File::create(dump_path)?;
        let writer = BufWriter::new(output_file);
        let mut encoder = GzEncoder::new(writer, Compression::default());
        pc0_cache.to_writer(&mut encoder)?;
        encoder.finish()?;
        println!("Successfully saved precompute0 cache to {:?}", dump_path);
    }

    // 3b. Handle building and saving the final constraint file if requested
    if let Some(output_path) = &args.output {
        println!("\nConstructing GrammarConstraint...");
        let grammar_constraint = if let Some(load_path) = &args.load_precompute0 {
            println!("Loading precompute0 cache from: {:?}", load_path);
            let cache_file = File::open(load_path)?;
            let reader = BufReader::new(flate2::read::GzDecoder::new(cache_file));
            let cache = Precompute0Cache::from_reader(reader)?;
            println!("Finished loading cache. Building rest of constraint...");

            GrammarConstraint::from_compiled_grammar_and_precompute0_cache(
                compiled_grammar,
                llm_token_map,
                dummy_eof_token_id,
                max_original_llm_token_id,
                &config,
                cache,
            )
        } else {
            GrammarConstraint::from_compiled_grammar_with_config(
                compiled_grammar,
                llm_token_map,
                dummy_eof_token_id,
                max_original_llm_token_id,
                &config,
            )
        };
        println!("GrammarConstraint constructed successfully.");

        // 4. Save the GrammarConstraint to a compressed file.
        println!("Saving GrammarConstraint to: {:?}", output_path);
        let output_file = File::create(output_path)?;
        let writer = BufWriter::new(output_file);
        let mut encoder = GzEncoder::new(writer, Compression::default());
        grammar_constraint.to_writer(&mut encoder)?;
        encoder.finish()?;
        println!("Successfully saved constraint to {:?}", output_path);
    }

    Ok(())
}
