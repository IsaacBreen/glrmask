use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use std::path::PathBuf;
use clap::Parser;

use sep1::interface::{CompiledGrammar, GrammarDefinition};
use sep1::glr::grammar::Terminal;
use sep1::glr::table::TerminalID;
use sep1::precompute4::characterize::compute_all_characterizations;
use sep1::precompute4::template_nwa::{build_template_dwas, build_ignore_terminal_dwa};
use sep1::tokenizer::LLMTokenID;
use sep1::constraint_precompute::run_precompute1;
use sep1::json_serialization::JSONConvertible;
use serde_json::json;

#[derive(Parser)]
#[command(author, version, about = "Dump pipeline artifacts for visualization")]
struct Cli {
    /// Path to grammar file (.ebnf)
    #[arg(short, long)]
    grammar: PathBuf,
    
    /// Output JSON file
    #[arg(short, long, default_value = "pipeline_artifacts.json")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Read grammar file
    let grammar_text = std::fs::read_to_string(&cli.grammar)?;
    
    println!("Compiling grammar from {:?}...", cli.grammar);
    let grammar_def = GrammarDefinition::from_ebnf(&grammar_text).expect("Failed to parse EBNF");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_def));
    let parser = &compiled_grammar.glr_parser;
    let tokenizer = &compiled_grammar.tokenizer;
    
    // Build terminal name map
    let mut terminal_names: BTreeMap<usize, String> = BTreeMap::new();
    for (term, tid) in parser.terminal_map.iter() {
        let name = match term {
            Terminal::Literal(bytes) => {
                format!("'{}'", String::from_utf8_lossy(bytes))
            },
            Terminal::RegexName(name) => name.clone(),
        };
        terminal_names.insert(tid.0, name);
    }
    
    // Build nonterminal name map
    let mut nonterminal_names: BTreeMap<usize, String> = BTreeMap::new();
    for (nt, ntid) in parser.non_terminal_map.iter() {
        nonterminal_names.insert(ntid.0, nt.0.clone());
    }
    
    // Compute ALL Characterizations
    println!("Computing all characterizations...");
    let all_chars = compute_all_characterizations(parser);
    let char_map: BTreeMap<String, String> = all_chars.iter()
        .map(|(tid, c)| {
            let term_name = terminal_names.get(&tid.0).cloned().unwrap_or_else(|| format!("T{}", tid.0));
            (term_name, format!("{:?}", c))
        })
        .collect();

    // Build ALL Template DFAs
    println!("Building all Template DFAs...");
    let template_dwas = build_template_dwas(parser).expect("Failed to build template DWAs");
    let _ignore_dwa = build_ignore_terminal_dwa();
    
    let template_map: BTreeMap<String, String> = template_dwas.iter()
        .map(|(tid, dwa)| {
            let term_name = terminal_names.get(&tid.0).cloned().unwrap_or_else(|| format!("T{}", tid.0));
            (term_name, format!("{}", dwa))
        })
        .collect();

    // Build Skeleton DWA (Terminal DWA / Precompute1)
    println!("Building Skeleton DWA (Terminal DWA / Precompute1)...");
    
    let mut internal_llm_token_map: BTreeMap<Vec<u8>, LLMTokenID> = BTreeMap::new();
    let mut token_id_counter = 0;
    let mut terminal_to_token_id: BTreeMap<TerminalID, LLMTokenID> = BTreeMap::new();

    for (term, tid) in parser.terminal_map.iter() {
        if let Terminal::Literal(bytes) = term {
            let token_id = LLMTokenID(token_id_counter);
            internal_llm_token_map.insert(bytes.clone(), token_id);
            terminal_to_token_id.insert(*tid, token_id);
            token_id_counter += 1;
        }
    }
    let internal_max_llm_token = if token_id_counter > 0 { token_id_counter - 1 } else { 0 };
    let terminals_count = parser.terminal_map.len();
    let active_states = vec![tokenizer.initial_state_id()];

    let skeleton_dwa = run_precompute1(
        tokenizer,
        Some(parser),
        &internal_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        active_states,
    );

    // Get tokenizer DFA info from JSON
    let tokenizer_json = tokenizer.to_json().to_json_string();
    let tokenizer_data: serde_json::Value = serde_json::from_str(&tokenizer_json)?;
    
    // Get parser table from JSON
    let compiled_json = compiled_grammar.to_json().to_json_string();
    let compiled_data: serde_json::Value = serde_json::from_str(&compiled_json)?;
    let parser_data = compiled_data.get("glr_parser").cloned().unwrap_or(json!({}));
    
    // Dump Everything
    println!("Dumping artifacts to {:?}...", cli.output);
    
    let output = json!({
        "grammar_text": grammar_text,
        "terminal_names": terminal_names,
        "nonterminal_names": nonterminal_names,
        "tokenizer_dfa": tokenizer_data,
        "parse_table": parser_data.get("stage_7_table"),
        "terminal_map": parser_data.get("terminal_map"),
        "non_terminal_map": parser_data.get("non_terminal_map"),
        "productions": parser_data.get("productions"),
        "characterizations": char_map,
        "template_dfas": template_map,
        "skeleton_dwa_repr": format!("{}", skeleton_dwa),
    });

    let mut file = File::create(&cli.output)?;
    file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;

    println!("✓ Artifacts dumped to {:?}", cli.output);
    Ok(())
}
