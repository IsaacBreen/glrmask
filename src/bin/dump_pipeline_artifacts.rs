use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use sep1::interface::{CompiledGrammar, GrammarDefinition};
use sep1::glr::grammar::Terminal;
use sep1::glr::table::TerminalID;
use sep1::precompute4::characterize::compute_below_bottom_characterization;
use sep1::precompute4::template_nwa::build_template_nwa_from_characterization;
use sep1::precompute4::resolve_negatives::resolve_negative_codes_in_nwa;
use sep1::precompute4::weighted_automata::{NWA, DWA};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define Grammar
    // S -> A $
    // A -> 'a' B
    // B -> 'b' C
    // C -> 'c' A | 'c'
    let ebnf = r#"
        s ::= a "$";
        a ::= 'a' b;
        b ::= 'b' c;
        c ::= 'c' a | 'c';
    "#;

    // 2. Compile Grammar
    println!("Compiling grammar...");
    let grammar_def = GrammarDefinition::from_ebnf(ebnf).expect("Failed to parse EBNF");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_def));
    let parser = &compiled_grammar.glr_parser;

    // 3. Find Terminal ID for "$"
    let dollar_terminal = Terminal::Literal(b"$".to_vec());
    let dollar_id = *parser.terminal_map.get_by_left(&dollar_terminal)
        .expect("Could not find '$' terminal");
    
    println!("Found '$' terminal ID: {:?}", dollar_id);

    // 4. Compute Characterization
    println!("Computing characterization...");
    let char_data = compute_below_bottom_characterization(parser, dollar_id);
    
    // 5. Build Template NWA
    println!("Building Template NWA...");
    let template_nwa = build_template_nwa_from_characterization(&char_data)
        .expect("Failed to build template NWA");

    // 6. Build Template DFA (Determinize + Simplify)
    println!("Building Template DFA...");
    let mut template_dfa = template_nwa.determinize();
    template_dfa.simplify();

    // 7. Resolved NWA (Simulated)
    println!("Building Resolved NWA...");
    let mut final_nwa = NWA::from_dwa(&template_dfa);
    resolve_negative_codes_in_nwa(&mut final_nwa);

    // 8. Dump Data
    println!("Dumping artifacts...");
    let output = json!({
        "grammar_ebnf": ebnf,
        "lalr_table": format!("{:?}", parser.table),
        "characterization": format!("{:?}", char_data),
        "template_nwa": format!("{}", template_nwa),
        "template_dfa": format!("{}", template_dfa),
        "final_nwa": format!("{}", final_nwa),
    });

    let mut file = File::create("pipeline_artifacts.json")?;
    file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;

    println!("Artifacts dumped to pipeline_artifacts.json");
    Ok(())
}
