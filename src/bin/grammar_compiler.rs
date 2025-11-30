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
    if show_output { println!("  {BOLD_WHITE}Loading grammar...{RESET}"); }
    
    let grammar_path_str = args.grammar.to_str().ok_or_else(|| format!("Path is not valid UTF-8: {:?}", args.grammar))?;
    let mut grammar_definition = GrammarDefinition::from_ebnf_file(grammar_path_str)?;
    
    let prod_count = grammar_definition.productions.len();
    let term_count = grammar_definition.terminal_to_group_id().len();
    
    grammar_definition.optimize();
    
    let opt_prod_count = grammar_definition.productions.len();
    let opt_term_count = grammar_definition.terminal_to_group_id().len();
    
    if show_output {
        println!("  {BOLD_GREEN}{CHECK}{RESET}  {DIM}{} → {} productions, {} → {} terminals{RESET} {MAGENTA}({}){RESET}", 
            prod_count, opt_prod_count, term_count, opt_term_count,
            format_duration(step.elapsed()));
    }

    // 2. Load the vocabulary.
    let step = std::time::Instant::now();
    if show_output { println!("  {BOLD_WHITE}Loading vocabulary...{RESET}"); }
    
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
    
    if show_output {
        println!("  {BOLD_GREEN}{CHECK}{RESET}  {DIM}{} tokens{RESET} {MAGENTA}({}){RESET}", 
            llm_token_map.len(), format_duration(step.elapsed()));
    }

    // 3. Construct the GrammarConstraint.
    if show_output { println!("\n  {BOLD_WHITE}Building constraint...{RESET}"); }
    let build_start = std::time::Instant::now();
    
    let config = GrammarConstraintConfig::default();

    let grammar_constraint = GrammarConstraint::new_from_grammar_definition(
        Arc::new(grammar_definition),
        llm_token_map,
        max_original_llm_token_id,
        &config,
    );

    if show_output {
        println!("\n  {DIM}└─ Total build time: {}{RESET}", format_duration(build_start.elapsed()));
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
    if let Some(output_path) = args.output {
        let step = std::time::Instant::now();
        if show_output { println!("\n  {BOLD_WHITE}Saving output...{RESET}"); }

        let output_file = File::create(&output_path)?;
        let buf_writer = BufWriter::new(output_file);

        let mut writer: Box<dyn Write> =
            if output_path.extension().and_then(|s| s.to_str()) == Some("gz") {
                Box::new(GzEncoder::new(buf_writer, Compression::default()))
            } else {
                Box::new(buf_writer)
            };

        grammar_constraint.to_writer(&mut writer)?;
        
        if show_output {
            let file_size = std::fs::metadata(&output_path)?.len();
            println!("  {BOLD_GREEN}{CHECK}{RESET}  {DIM}{:?}{RESET} {CYAN}({}){RESET} {MAGENTA}({}){RESET}", 
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
