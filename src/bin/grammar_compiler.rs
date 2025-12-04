use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use sep1::constraint::{GrammarConstraint, GrammarConstraintConfig};
use sep1::interface::GrammarDefinition;
use sep1::json_serialization::JSONConvertible;
use sep1::tokenizer::{LLMTokenID, LLMTokenMap};
use sep1::r#macro::{colors::*, is_debug_level_enabled, format_duration};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;

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

    /// Path for the output file (.json or .json.gz).
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Optional: load a precompute0 cache from this file to skip rebuilding Trie0.
    #[arg(long, value_name = "FILE")]
    load_precompute0: Option<PathBuf>,

    /// Optional: save the precompute0 stage to this file for reuse.
    #[arg(long, value_name = "FILE")]
    save_precompute0: Option<PathBuf>,

    /// If specified, only compute and save the precompute0 cache, then exit.
    #[arg(long, requires = "save_precompute0", conflicts_with = "output")]
    precompute0_only: bool,

    /// Grammar format: "ebnf" or "lark". If not specified, inferred from file extension.
    #[arg(long)]
    format: Option<String>,
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} bytes", bytes)
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let total_start = std::time::Instant::now();
    let args = Args::parse();

    if !args.precompute0_only && args.output.is_none() {
        return Err("Error: --output is required unless --precompute0-only is specified.".into());
    }
    
    // Convenience: check debug level once
    let show_output = is_debug_level_enabled(1);

    // Header (Level 1+)
    if show_output {
        println!("\n{BOLD_CYAN}Grammar Constraint Compiler{RESET}");
        println!("{DIM}─────────────────────────────────────────{RESET}\n");
    }

    // 1. Load and compile the grammar.
    let step = std::time::Instant::now();
    sep1::debug!(2, "Loading grammar...");
    
    let grammar_path_str = args.grammar.to_str().ok_or_else(|| format!("Path is not valid UTF-8: {:?}", args.grammar))?;
    
    let format = args.format.as_deref().or_else(|| {
        args.grammar.extension().and_then(|ext| ext.to_str()).map(|ext| {
            if ext == "lark" { "lark" } else { "ebnf" }
        })
    }).unwrap_or("ebnf");

    sep1::debug!(5, "Parsing grammar file...");
    let mut grammar_definition = match format {
        "lark" => GrammarDefinition::from_lark_file(grammar_path_str)?,
        "ebnf" => GrammarDefinition::from_ebnf_file(grammar_path_str)?,
        _ => return Err(format!("Unknown grammar format: {}", format).into()),
    };
    sep1::debug!(5, "Grammar parsed");
    
    let prod_count = grammar_definition.productions.len();
    let term_count = grammar_definition.terminal_to_group_id().len();
    
    sep1::debug!(5, "Optimizing grammar...");
    grammar_definition.optimize();
    sep1::debug!(5, "Grammar optimized");
    
    let opt_prod_count = grammar_definition.productions.len();
    let opt_term_count = grammar_definition.terminal_to_group_id().len();
    
    if is_debug_level_enabled(2) {
        sep1::debug!(2, "{BOLD_GREEN}{CHECK}{RESET}  {DIM}{} → {} productions, {} → {} terminals{RESET} {MAGENTA}({}){RESET}", 
            prod_count, opt_prod_count, term_count, opt_term_count,
            format_duration(step.elapsed()));
    }

    // 2. Load the vocabulary.
    let step = std::time::Instant::now();
    sep1::debug!(2, "Loading vocabulary...");
    
    let vocab_file = File::open(&args.vocab)?;
    let reader = BufReader::new(vocab_file);
    let vocab: BTreeMap<String, usize> = serde_json::from_reader(reader)?;

    let mut llm_token_map = LLMTokenMap::new();
    let mut max_original_llm_token_id = 0;

    for (token_str, token_id) in vocab {
        let processed_token_str = token_str
            .replace("Ġ", " ")
            .replace("ą", "\n")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
            .replace("č", "\r");
        let token_bytes = processed_token_str.as_bytes().to_vec();
        llm_token_map.insert(token_bytes, LLMTokenID(token_id));
        max_original_llm_token_id = max_original_llm_token_id.max(token_id);
    }
    
    if is_debug_level_enabled(2) {
        sep1::debug!(2, "{BOLD_GREEN}{CHECK}{RESET}  {DIM}{} tokens{RESET} {MAGENTA}({}){RESET}", 
            llm_token_map.len(), format_duration(step.elapsed()));
    }

    // 3. Construct the GrammarConstraint.
    sep1::debug!(2, "Building constraint...");
    let build_start = std::time::Instant::now();
    
    let config = GrammarConstraintConfig::default();

    let grammar_constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &config,
    );

    // Record the total compilation time (grammar + vocab loading + constraint building)
    let compilation_time_seconds = total_start.elapsed().as_secs_f64();
    
    if show_output {
        println!("  {CYAN}└─ Total build time: {}{RESET}", format_duration(build_start.elapsed()));
    }

    if let Some(path) = args.save_precompute0.as_ref() {
        let output_file = File::create(path)?;
        let _writer = BufWriter::new(output_file);
        if show_output {
            println!("  {DIM}└─ Saved precompute cache to {:?}{RESET}", path);
        }
    }

    if args.precompute0_only {
        if show_output {
            println!("\n{BOLD_GREEN}{CHECK} Done (precompute0 only){RESET}");
        }
        return Ok(());
    }

    // 4. Save the GrammarConstraint to a file.
    // Optimized: serialize to memory first, then compress/write in one pass.
    // This is ~6x faster than streaming JSON through the gzip encoder.
    if let Some(output_path) = args.output {
        let step = std::time::Instant::now();
        sep1::debug!(2, "Saving output...");

        // Serialize to JSON in memory
        let json_node = grammar_constraint.to_json();
        
        // Add compilation metadata to the JSON
        // The compilation_time_seconds includes everything: grammar parsing, vocab loading,
        // tokenizer construction, parser table generation, DWA construction, etc.
        let json_with_metadata = match json_node {
            sep1::json_serialization::JSONNode::Object(mut obj) => {
                obj.insert(
                    "compilation_time_seconds".to_string(), 
                    sep1::json_serialization::JSONNode::Float(compilation_time_seconds)
                );
                sep1::json_serialization::JSONNode::Object(obj)
            },
            other => other, // Shouldn't happen, but be safe
        };
        
        let json_bytes = serde_json::to_vec(&json_with_metadata).map_err(|e| e.to_string())?;

        // Write to file (with optional compression)
        let output_file = File::create(&output_path)?;
        let buf_writer = BufWriter::new(output_file);

        if output_path.extension().and_then(|s| s.to_str()) == Some("gz") {
            // Use compression level 3 for good speed/size tradeoff
            // (level 1: 16ms/931KB, level 3: 30ms/746KB, level 6: 123ms/728KB)
            let mut encoder = GzEncoder::new(buf_writer, Compression::new(3));
            encoder.write_all(&json_bytes).map_err(|e| e.to_string())?;
            encoder.finish().map_err(|e| e.to_string())?;
        } else {
            let mut writer = buf_writer;
            writer.write_all(&json_bytes).map_err(|e| e.to_string())?;
        }
        
        if is_debug_level_enabled(2) {
            let file_size = std::fs::metadata(&output_path)?.len();
            sep1::debug!(2, "{BOLD_GREEN}{CHECK}{RESET}  {DIM}{:?}{RESET} {CYAN}({}){RESET} {MAGENTA}({}){RESET}", 
                output_path, format_bytes(file_size), format_duration(step.elapsed()));
        }
    }

    // Summary
    if show_output {
        println!("\n{DIM}─────────────────────────────────────────{RESET}");
        println!("{BOLD_GREEN}{CHECK} Complete in {}{RESET}\n", format_duration(total_start.elapsed()));
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{BOLD_RED}Error:{RESET} {}", err);
        std::process::exit(1);
    }
}
