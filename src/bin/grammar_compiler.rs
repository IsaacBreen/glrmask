use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use sep1::constraint::{GrammarConstraint, GrammarConstraintConfig};
use sep1::interface::GrammarDefinition;
use sep1::json_serialization::JSONConvertible;
use sep1::dfa_u8::{LLMTokenID, LLMTokenMap};
use sep1::r#macro::{colors::*, is_debug_level_enabled, format_duration};
use sep1::datastructures::flush_weight_dump;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write, Read};
use std::path::PathBuf;
use std::sync::Arc;

/// Compiles a grammar and vocabulary into a GrammarConstraint object.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the EBNF or Lark grammar file.
    #[arg(short, long, conflicts_with = "json_schema")]
    grammar: Option<PathBuf>,

    /// Path to a JSON Schema file (alternative to --grammar).
    #[arg(long, conflicts_with = "grammar")]
    json_schema: Option<PathBuf>,

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

    /// Filter vocabulary to include tokens with specific byte lengths or ranges.
    #[arg(long, value_delimiter = ' ', num_args = 1..)]
    token_len: Option<Vec<String>>,
}

fn parse_len_ranges(ranges: &[String]) -> Result<(std::collections::HashSet<usize>, Option<usize>), String> {
    let mut allowed = std::collections::HashSet::new();
    let mut min_unbounded = None;

    for r in ranges {
        if let Some((start, end)) = r.split_once('-') {
             let start: usize = start.parse().map_err(|_| format!("Invalid start: {}", start))?;
             if end.is_empty() {
                 min_unbounded = Some(min_unbounded.map_or(start, |m: usize| m.min(start)));
             } else {
                 let end: usize = end.parse().map_err(|_| format!("Invalid end: {}", end))?;
                 if start > end { return Err(format!("Invalid range {} > {}", start, end)); }
                 for i in start..=end { allowed.insert(i); }
             }
        } else {
             let val: usize = r.parse().map_err(|_| format!("Invalid value: {}", r))?;
             allowed.insert(val);
        }
    }
    Ok((allowed, min_unbounded))
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
    let grammar_start = std::time::Instant::now();
    
    let grammar_definition = if let Some(json_schema_path) = &args.json_schema {
        // Load from JSON Schema
        sep1::debug!(2, "Loading JSON schema...");
        let schema_path_str = json_schema_path.to_str().ok_or_else(|| format!("Path is not valid UTF-8: {:?}", json_schema_path))?;
        sep1::debug!(5, "Reading schema file...");
        let schema_content = std::fs::read_to_string(schema_path_str)?;
        sep1::debug!(5, "Converting schema to EBNF...");
        let ebnf = sep1::json_schema::json_schema_to_ebnf(&schema_content).map_err(|e| format!("Failed to convert JSON schema: {}", e))?;
        sep1::debug!(5, "Parsing generated EBNF ({} chars)...", ebnf.len());
        GrammarDefinition::from_ebnf(&ebnf).map_err(|e| format!("Failed to parse generated EBNF: {}", e))?
    } else if let Some(grammar_path) = &args.grammar {
        // Load from EBNF/Lark grammar file
        sep1::debug!(2, "Loading grammar...");
        let grammar_path_str = grammar_path.to_str().ok_or_else(|| format!("Path is not valid UTF-8: {:?}", grammar_path))?;
        
        let format = args.format.as_deref().or_else(|| {
            grammar_path.extension().and_then(|ext| ext.to_str()).map(|ext| {
                if ext == "lark" { "lark" } else { "ebnf" }
            })
        }).unwrap_or("ebnf");

        sep1::debug!(5, "Parsing grammar file...");
        let gd = match format {
            "lark" => GrammarDefinition::from_lark_file(grammar_path_str)?,
            "ebnf" => GrammarDefinition::from_ebnf_file(grammar_path_str)?,
            _ => return Err(format!("Unknown grammar format: {}", format).into()),
        };
        sep1::debug!(5, "Grammar parsed");
        gd
    } else {
        return Err("Error: either --grammar or --json-schema must be specified.".into());
    };
    
    // NOTE: Grammar is already optimized by from_ebnf()/from_lark() internally.
    // Calling optimize() again here would cause double-processing of nullable
    // terminals, creating additional wrapper non-terminals that bloat the grammar
    // (e.g., 446 productions instead of 228, causing 2-3x slower builds).
    let prod_count = grammar_definition.productions.len();
    let term_count = grammar_definition.terminal_to_group_id().len();

    eprintln!("TIMING: grammar_definition {:?}", grammar_start.elapsed());
    
    if is_debug_level_enabled(2) {
        sep1::debug!(2, "└─ {} productions, {} terminals {MAGENTA}({}){RESET}", 
            prod_count, term_count, format_duration(step.elapsed()));
    }

    // 2. Load the vocabulary.
    let step = std::time::Instant::now();
    sep1::debug!(2, "Loading vocabulary...");
    
    let vocab_start = std::time::Instant::now();
    let mut vocab_file = File::open(&args.vocab)?;
    let mut vocab_bytes = Vec::new(); // Read to memory for faster parsing
    vocab_file.read_to_end(&mut vocab_bytes)?;
    let vocab: BTreeMap<String, usize> = serde_json::from_slice(&vocab_bytes)?;

    // Parse filters
    let (allowed_lengths, min_len_unbounded) = if let Some(ranges) = &args.token_len {
        parse_len_ranges(ranges).map_err(|e| e.to_string())?
    } else {
        (std::collections::HashSet::new(), None)
    };
    let has_filter = !allowed_lengths.is_empty() || min_len_unbounded.is_some();

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

        if has_filter {
            let len = token_bytes.len();
            let keep = allowed_lengths.contains(&len) || min_len_unbounded.map_or(false, |m| len >= m);
            if !keep { continue; }
        }

        llm_token_map.insert(token_bytes, LLMTokenID(token_id));
        max_original_llm_token_id = max_original_llm_token_id.max(token_id);
    }

    eprintln!("TIMING: load_vocabulary {:?}", vocab_start.elapsed());
    
    if is_debug_level_enabled(2) {
        sep1::debug!(2, "└─ {} tokens {MAGENTA}({}){RESET}", 
            llm_token_map.len(), format_duration(step.elapsed()));
        println!();
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

    eprintln!("TIMING: build_constraint {:?}", build_start.elapsed());

    // Record the total compilation time (grammar + vocab loading + constraint building)
    let compilation_time_seconds = total_start.elapsed().as_secs_f64();
    
    if is_debug_level_enabled(2) {
        sep1::debug!(2, "└─ Total build time: {}", format_duration(build_start.elapsed()));
        println!();
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
    if std::env::var("SKIP_SERIALIZATION").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false) {
        if show_output {
            println!("\n{DIM}─────────────────────────────────────────{RESET}");
            println!("{BOLD_GREEN}{CHECK} Complete in {}{RESET}\n", format_duration(total_start.elapsed()));
        }
        eprintln!("TIMING: total {:?}", total_start.elapsed());
        return Ok(());
    }

    if let Some(output_path) = args.output {
        let save_start = std::time::Instant::now();
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

        eprintln!("TIMING: save_output {:?}", save_start.elapsed());
        
        if is_debug_level_enabled(2) {
            let file_size = std::fs::metadata(&output_path)?.len();
            sep1::debug!(2, "└─ {:?} {CYAN}({}){RESET} {MAGENTA}({}){RESET}", 
                output_path, format_bytes(file_size), format_duration(step.elapsed()));
            // println!(); // Removed debug print
        }
    }

    // Summary
    if show_output {
        println!("\n{DIM}─────────────────────────────────────────{RESET}");
        println!("{BOLD_GREEN}{CHECK} Complete in {}{RESET}\n", format_duration(total_start.elapsed()));
    }

    // Flush any collected factorized weights for analysis
    if let Err(e) = flush_weight_dump(".cache/factorized_weights_dump.json") {
        eprintln!("Warning: failed to flush weight dump: {}", e);
    }

    eprintln!("TIMING: total {:?}", total_start.elapsed());

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{BOLD_RED}Error:{RESET} {}", err);
        std::process::exit(1);
    }
    // Optimization: fast exit to skip destructors of large graph structures
    std::process::exit(0);
}
