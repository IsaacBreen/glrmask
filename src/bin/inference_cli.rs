use clap::{Parser, Subcommand};
use flate2::read::GzDecoder;
use sep1::constraint::GrammarConstraint;
use sep1::json_serialization::JSONConvertible;
use sep1::datastructures::bitset::Bitset;
use sep1::dfa_u8::LLMTokenID;
use sep1::r#macro::colors::*;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Grammar constraint inference CLI - test masks and token sequences
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the compiled grammar constraint file (.json or .json.gz)
    #[arg(short, long)]
    constraint: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Get the initial token mask (no tokens committed)
    InitialMask {
        /// Show timing information
        #[arg(short, long)]
        timing: bool,
        
        /// Output format: count, list, or ranges
        #[arg(short, long, default_value = "count")]
        format: String,
    },
    
    /// Commit a sequence of tokens and show masks at each step
    Sequence {
        /// Token IDs to commit (comma-separated or space-separated)
        #[arg(required = true)]
        tokens: Vec<String>,
        
        /// Show timing information
        #[arg(short, long)]
        timing: bool,
    },
    
    /// Interactive mode - commit tokens one at a time
    Interactive,
    
    /// Validate a token sequence (check if all tokens are allowed)
    Validate {
        /// Token IDs to validate (comma-separated or space-separated)
        #[arg(required = true)]
        tokens: Vec<String>,
    },
    
    /// Show constraint statistics
    Stats,
}

fn load_constraint(path: &PathBuf) -> Result<GrammarConstraint, Box<dyn std::error::Error>> {
    let start = Instant::now();
    
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    
    let constraint = if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        let decoder = GzDecoder::new(reader);
        let buf_reader = BufReader::new(decoder);
        GrammarConstraint::from_reader(buf_reader)?
    } else {
        GrammarConstraint::from_reader(reader)?
    };
    
    eprintln!("{DIM}Loaded constraint in {:.2?}{RESET}", start.elapsed());
    Ok(constraint)
}

fn parse_token_ids(tokens: &[String]) -> Result<Vec<u32>, String> {
    let mut result = Vec::new();
    for token_str in tokens {
        // Handle comma-separated values
        for part in token_str.split(',') {
            let part = part.trim();
            if part.is_empty() { continue; }
            match part.parse::<u32>() {
                Ok(id) => result.push(id),
                Err(_) => return Err(format!("Invalid token ID: '{}'", part)),
            }
        }
    }
    Ok(result)
}

fn format_mask_output(mask: &Bitset, format: &str, vocab_size: usize) -> String {
    match format {
        "count" => {
            let count = mask.len();
            format!("{} tokens allowed out of {}", count, vocab_size)
        }
        "list" => {
            let ids: Vec<usize> = mask.iter().collect();
            if ids.len() > 100 {
                format!("{:?}... ({} total)", &ids[..100], ids.len())
            } else {
                format!("{:?}", ids)
            }
        }
        "ranges" => {
            let ids: Vec<usize> = mask.iter().collect();
            if ids.is_empty() {
                return "No tokens allowed".to_string();
            }
            let mut ranges = Vec::new();
            let mut start = ids[0];
            let mut end = ids[0];
            for &id in &ids[1..] {
                if id == end + 1 {
                    end = id;
                } else {
                    if start == end {
                        ranges.push(format!("{}", start));
                    } else {
                        ranges.push(format!("{}-{}", start, end));
                    }
                    start = id;
                    end = id;
                }
            }
            if start == end {
                ranges.push(format!("{}", start));
            } else {
                ranges.push(format!("{}-{}", start, end));
            }
            format!("{} ranges: {}", ranges.len(), ranges.join(", "))
        }
        _ => format!("Unknown format: {}. Use 'count', 'list', or 'ranges'.", format),
    }
}

fn cmd_initial_mask(constraint: &GrammarConstraint, timing: bool, format: &str) {
    let start = Instant::now();
    let state = constraint.init();
    let init_time = start.elapsed();
    
    let start = Instant::now();
    let mask = state.get_mask();
    let mask_time = start.elapsed();
    
    let vocab_size = constraint.parser_dwa_vocab.max_original_llm_token_id + 1;
    println!("{}", format_mask_output(&mask, format, vocab_size));
    
    if timing {
        eprintln!("{DIM}Init: {:.2?}, Mask: {:.2?}{RESET}", init_time, mask_time);
    }
}

fn cmd_sequence(constraint: &GrammarConstraint, tokens: &[String], timing: bool) -> Result<(), String> {
    let token_ids = parse_token_ids(tokens)?;
    
    let mut state = constraint.init();
    let _vocab_size = constraint.parser_dwa_vocab.max_original_llm_token_id + 1;
    
    println!("Processing {} tokens...\n", token_ids.len());
    
    for (i, &token_id) in token_ids.iter().enumerate() {
        let start = Instant::now();
        let mask = state.get_mask();
        let mask_time = start.elapsed();
        
        let is_allowed = mask.contains(token_id as usize);
        let count = mask.len();
        
        print!("Step {}: ", i);
        if is_allowed {
            print!("{BOLD_GREEN}✓{RESET} ");
        } else {
            print!("{BOLD_RED}✗{RESET} ");
        }
        print!("Token {} - ", token_id);
        
        if is_allowed {
            println!("{} tokens allowed", count);
        } else {
            println!("{BOLD_RED}NOT ALLOWED{RESET} ({} tokens allowed)", count);
            return Err(format!("Token {} not allowed at step {}", token_id, i));
        }
        
        if timing {
            eprintln!("{DIM}  Mask: {:.2?}{RESET}", mask_time);
        }
        
        let start = Instant::now();
        state.commit(LLMTokenID(token_id as usize)).unwrap();
        if timing {
            eprintln!("{DIM}  Commit: {:.2?}{RESET}", start.elapsed());
        }
    }
    
    // Show final mask
    let final_mask = state.get_mask();
    let final_count = final_mask.len();
    println!("\nFinal state: {} tokens allowed", final_count);
    
    Ok(())
}

fn cmd_interactive(constraint: &GrammarConstraint) {
    let mut state = constraint.init();
    let vocab_size = constraint.parser_dwa_vocab.max_original_llm_token_id + 1;
    let stdin = io::stdin();
    
    println!("Interactive mode. Enter token IDs to commit, 'mask' to show current mask, 'reset' to restart, 'quit' to exit.\n");
    
    loop {
        let mask = state.get_mask();
        let count = mask.len();
        print!("{BOLD_CYAN}[{} allowed]>{RESET} ", count);
        io::stdout().flush().unwrap();
        
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).is_err() {
            break;
        }
        let line = line.trim();
        
        match line {
            "quit" | "exit" | "q" => break,
            "reset" | "r" => {
                state = constraint.init();
                println!("State reset.");
            }
            "mask" | "m" => {
                println!("{}", format_mask_output(&mask, "list", vocab_size));
            }
            "ranges" => {
                println!("{}", format_mask_output(&mask, "ranges", vocab_size));
            }
            "" => continue,
            _ => {
                match parse_token_ids(&[line.to_string()]) {
                    Ok(ids) => {
                        for id in ids {
                            if mask.contains(id as usize) {
                                state.commit(LLMTokenID(id as usize)).unwrap();
                                println!("{BOLD_GREEN}✓{RESET} Committed token {}", id);
                            } else {
                                println!("{BOLD_RED}✗{RESET} Token {} not allowed!", id);
                            }
                        }
                    }
                    Err(e) => println!("{BOLD_RED}Error:{RESET} {}", e),
                }
            }
        }
    }
}

fn cmd_validate(constraint: &GrammarConstraint, tokens: &[String]) -> Result<(), String> {
    let token_ids = parse_token_ids(tokens)?;
    
    let mut state = constraint.init();
    
    for (i, &token_id) in token_ids.iter().enumerate() {
        let mask = state.get_mask();
        if !mask.contains(token_id as usize) {
            println!("{BOLD_RED}✗{RESET} Invalid at step {}: token {} not allowed", i, token_id);
            let count = mask.len();
            println!("  ({} tokens were allowed)", count);
            return Err(format!("Validation failed at step {}", i));
        }
        state.commit(LLMTokenID(token_id as usize)).unwrap();
    }
    
    println!("{BOLD_GREEN}✓{RESET} All {} tokens are valid", token_ids.len());
    Ok(())
}

fn cmd_stats(constraint: &GrammarConstraint) {
    println!("Constraint Statistics:");
    println!("  Vocabulary size: {}", constraint.parser_dwa_vocab.max_original_llm_token_id + 1);
    println!("  Internal tokens: {}", constraint.parser_dwa_vocab.internal_to_original.len());
    println!("  DWA: {}", constraint.parser_dwa.stats());
    println!("  Parser productions: {}", constraint.parser.productions.len());
    println!("  Tokenizer DFA states: {}", constraint.tokenizer.dfa().states.len());
}

fn main() {
    let args = Args::parse();
    
    let constraint = match load_constraint(&args.constraint) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{BOLD_RED}Error loading constraint:{RESET} {}", e);
            std::process::exit(1);
        }
    };
    
    let result = match args.command {
        Commands::InitialMask { timing, format } => {
            cmd_initial_mask(&constraint, timing, &format);
            Ok(())
        }
        Commands::Sequence { tokens, timing } => {
            cmd_sequence(&constraint, &tokens, timing)
        }
        Commands::Interactive => {
            cmd_interactive(&constraint);
            Ok(())
        }
        Commands::Validate { tokens } => {
            cmd_validate(&constraint, &tokens)
        }
        Commands::Stats => {
            cmd_stats(&constraint);
            Ok(())
        }
    };
    
    if let Err(e) = result {
        eprintln!("{BOLD_RED}Error:{RESET} {}", e);
        std::process::exit(1);
    }
}
