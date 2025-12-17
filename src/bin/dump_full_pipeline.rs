use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::path::PathBuf;
use clap::Parser;

use sep1::interface::{CompiledGrammar, GrammarDefinition};
use sep1::glr::grammar::Terminal;
use sep1::glr::table::{TerminalID, Stage7ShiftsAndReducesLookaheadValue};
use sep1::precompute4::characterize::compute_all_characterizations;
use sep1::precompute4::template_nwa::{build_terminal_dwas, build_ignore_terminal_dwa};
use sep1::precompute4::weighted_automata::{NWA, NWABody, NWAState, NWAStates, Weight};
use sep1::precompute4::weighted_automata::common::Label;
use sep1::precompute4::full_dwa::finalize_and_optimize_and_determinize;
use sep1::constraint_precompute::run_precompute1;
use sep1::tokenizer::LLMTokenID;
use serde_json::json;
use sep1::precompute4::resolve_negatives::resolve_negative_codes_in_nwa;
use sep1::r#macro::is_debug_level_enabled;

#[derive(Parser)]
#[command(author, version, about = "Dump full pipeline artifacts for visualization")]
struct Cli {
    /// Path to grammar file (.ebnf)
    #[arg(short, long)]
    grammar: PathBuf,

    /// Output JSON file
    #[arg(short, long, default_value = "pipeline_full_artifacts.json")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // 1. Load Grammar
    let grammar_text = std::fs::read_to_string(&cli.grammar)?;

    println!("Compiling grammar from {:?}...", cli.grammar);
    // Use from_ebnf_no_optimize to preserve original grammar structure for visualization
    let grammar_def = GrammarDefinition::from_ebnf_no_optimize(&grammar_text).expect("Failed to parse EBNF");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_def));
    let parser = &compiled_grammar.glr_parser;
    let tokenizer = &compiled_grammar.tokenizer;

    // Build terminal name map
    let mut terminal_names: BTreeMap<usize, String> = BTreeMap::new();
    for (term, tid) in parser.terminal_map.iter() {
        let name = match term {
            Terminal::Literal(bytes) => {
                format!("'{}'", String::from_utf8_lossy(bytes))
            }
            Terminal::RegexName(name) => name.clone(),
        };
        terminal_names.insert(tid.0, name);
    }

    // Build nonterminal name map
    let mut nonterminal_names: BTreeMap<usize, String> = BTreeMap::new();
    for (nt, ntid) in parser.non_terminal_map.iter() {
        nonterminal_names.insert(ntid.0, nt.0.clone());
    }

    // 2. Compute ALL Characterizations
    println!("Computing all characterizations...");
    let all_chars = compute_all_characterizations(parser);
    let char_map: BTreeMap<String, String> = all_chars.iter()
        .map(|(tid, c)| (format!("{:?}", tid), format!("{:?}", c)))
        .collect();

    if is_debug_level_enabled(4) {
        println!("Terminals:");
        for (term, tid) in &parser.terminal_map {
            println!("  {}: {}", tid.0, term);
        }
        println!("Nonterminals:");
        for (nt, ntid) in &parser.non_terminal_map {
            println!("  {:?}: {}", ntid.0, nt);
        }
        println!("Characterizations:");
        for (tid, c) in &all_chars {
            println!("{}", c);
        }
    }

    // 3. Build ALL Template DFAs
    println!("Building all Template DFAs...");
    let template_dwas = build_terminal_dwas(parser).expect("Failed to build template DWAs");
    let ignore_dwa = build_ignore_terminal_dwa();

    if is_debug_level_enabled(4) {
        println!("Template DFAs:");
        for (tid, dwa) in &template_dwas {
            let default_name = format!("t{}", tid.0);
            let name = terminal_names.get(&tid.0).unwrap_or(&default_name);
            println!("  Template for {} ({:?}):", name, tid);
            for line in format!("{}", dwa).lines() {
                println!("    {}", line);
            }
        }
    }

    let template_map: BTreeMap<String, String> = template_dwas.iter()
        .map(|(tid, dwa)| (format!("{:?}", tid), format!("{}", dwa)))
        .collect();

    // 4. Build Terminal DWA (Precompute1)
    println!("Building Terminal DWA (Precompute1)...");

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

    // Handle case where there are no literal terminals
    let internal_max_llm_token = if token_id_counter > 0 { token_id_counter - 1 } else { 0 };
    let terminals_count = parser.terminal_map.len();
    let active_states = vec![tokenizer.initial_state_id()];

    let mut terminal_dwa = run_precompute1(
        tokenizer,
        Some(parser),
        &internal_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        active_states,
    );
    terminal_dwa.simplify();
    if is_debug_level_enabled(4) {
        println!("Terminal DWA:");
        println!("{}", terminal_dwa);
    }

    // 5. Build Unresolved NWA (Simple Edge-Based Construction)
    // For each edge in the terminal DWA labeled with a terminal ID,
    // copy the corresponding template DFA and connect it.
    println!("Building Unresolved NWA...");

    let offset = parser.terminal_map.len() as Label;

    // Build individual template NWAs for each terminal
    let mut individual_template_nwas: HashMap<Option<TerminalID>, NWA> = HashMap::new();
    for (term_id, template_dwa) in &template_dwas {
        individual_template_nwas.insert(Some(*term_id), NWA::from_dwa(template_dwa));
    }
    individual_template_nwas.insert(None, NWA::from_dwa(&ignore_dwa));

    // Track which states come from which template for visualization
    let template_regions: Arc<Mutex<Vec<(usize, usize, Option<usize>)>>> = Arc::new(Mutex::new(Vec::new()));

    // Create NWA states arena
    let mut nwa_states = NWAStates::default();

    // Create one NWA node for each terminal DWA state
    let tdwa_to_nwa: Vec<usize> = (0..terminal_dwa.states.len())
        .map(|_| nwa_states.add_state())
        .collect();

    // Create combined start state for tokenizer state transitions
    let combined_start_state = nwa_states.add_state();

    // Entry node mapping for visualization (now a simple 1:1 correspondence)
    let entry_node_mapping: Vec<serde_json::Value> = tdwa_to_nwa.iter().enumerate()
        .map(|(tdwa_state, &nwa_node)| {
            json!({
                "terminal_dwa_state": tdwa_state,
                "nwa_entry_node": nwa_node
            })
        })
        .collect();

    // Process each terminal DWA state
    for (tdwa_state_id, tdwa_state) in terminal_dwa.states.0.iter().enumerate() {
        let nwa_entry = tdwa_to_nwa[tdwa_state_id];

        // Copy final weight from terminal DWA
        if let Some(fw) = &tdwa_state.final_weight {
            nwa_states[nwa_entry].final_weight = Some(fw.clone());
        }

        // Process each outgoing edge
        for (&label, &dest_tdwa_state) in &tdwa_state.transitions {
            let edge_weight = tdwa_state.trans_weights.get(&label)
                .cloned()
                .unwrap_or_else(Weight::all);

            if label >= offset {
                // Tokenizer state transition - add to combined start state
                let tsid_label = label;
                let dest_nwa = tdwa_to_nwa[dest_tdwa_state];
                nwa_states.add_transition(combined_start_state, tsid_label, dest_nwa, edge_weight).unwrap();
            } else {
                // Terminal edge - copy template DFA and connect
                let terminal_id = TerminalID(label as usize);
                let template_nwa = individual_template_nwas.get(&Some(terminal_id))
                    .expect("Template must exist for terminal");

                // Copy template into NWA states
                let template_offset = nwa_states.len();
                let template_end = template_offset + template_nwa.states.len();

                // Record this template region for visualization
                {
                    let mut regions = template_regions.lock().unwrap();
                    regions.push((template_offset, template_end, Some(terminal_id.0)));
                }

                // Copy all template states
                let dest_nwa = tdwa_to_nwa[dest_tdwa_state];
                for old_state in &template_nwa.states.0 {
                    let mut new_state = NWAState::default();

                    // Copy transitions (adjusting state indices)
                    for (lbl, targets) in &old_state.transitions {
                        let new_targets: Vec<(usize, Weight)> = targets.iter()
                            .map(|(t, w)| (*t + template_offset, w.clone()))
                            .collect();
                        if !new_targets.is_empty() {
                            new_state.transitions.insert(*lbl, new_targets);
                        }
                    }

                    // Copy epsilons (adjusting state indices)
                    for (target, w) in &old_state.epsilons {
                        new_state.epsilons.push((*target + template_offset, w.clone()));
                    }

                    // For final states: create epsilon edges to destination with edge weight
                    if old_state.final_weight.is_some() {
                        new_state.epsilons.push((dest_nwa, edge_weight.clone()));
                    }

                    nwa_states.0.push(new_state);
                }

                // Create epsilon edge from entry node to template start (full weight)
                let template_start_states: Vec<usize> = template_nwa.body.start_states.iter()
                    .map(|s| s + template_offset)
                    .collect();
                for s in template_start_states {
                    nwa_states[nwa_entry].epsilons.push((s, Weight::all()));
                }
            }
        }
    }

    let mut unresolved_nwa = NWA { 
        states: nwa_states, 
        body: NWABody { start_states: vec![combined_start_state] } 
    };
    if is_debug_level_enabled(4) {
        println!("Unresolved NWA:");
        println!("{}", unresolved_nwa);
    }

    // Build combined_start_mapping for compatibility
    let combined_start_mapping = json!({
        "terminal_dwa_state": 0,
        "nwa_entry_node": combined_start_state
    });

    // 6. Build Final DWA
    println!("Building Final DWA...");
    let mut resolved_nwa = unresolved_nwa.clone();
    resolve_negative_codes_in_nwa(&mut resolved_nwa);
    if is_debug_level_enabled(4) {
        println!("Resolved NWA:");
        println!("{}", resolved_nwa);
    }
    let mut final_dwa = finalize_and_optimize_and_determinize(parser, resolved_nwa.clone());
    final_dwa.simplify();
    if is_debug_level_enabled(4) {
        println!("Final DWA:");
        println!("{}", final_dwa);
    }

    // Optimize DWA/NWA for visualization
    final_dwa.simplify();
    terminal_dwa.optimize_for_visualization();
    unresolved_nwa.optimize_for_visualization();
    resolved_nwa.optimize_for_visualization();
    final_dwa.optimize_for_visualization();

    // 7. Dump Everything
    println!("Dumping artifacts to {:?}...", cli.output);

    // Convert template regions to serializable format
    let template_regions_data: Vec<serde_json::Value> = template_regions.lock().unwrap()
        .iter()
        .map(|(start, end, term_id)| {
            json!({
                "start_state": start,
                "end_state": end,
                "terminal_id": term_id
            })
        })
        .collect();

    // combined_start_mapping is already defined above

    // Build parse table representation for visualization
    let parse_table_data: Vec<serde_json::Value> = parser.table.iter()
        .map(|(state_id, row)| {
            // Build action map (terminals -> shift/reduce)
            let actions: BTreeMap<String, serde_json::Value> = row.get_shifts_and_reduces_map()
                .iter()
                .map(|(term_id, action)| {
                    let term_name = terminal_names.get(&term_id.0)
                        .cloned()
                        .unwrap_or_else(|| format!("t{}", term_id.0));
                    let action_val = match action {
                        Stage7ShiftsAndReducesLookaheadValue::Shift(target) => {
                            json!({"type": "shift", "target": target.0})
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                            let nt_name = nonterminal_names.get(&nonterminal_id.0)
                                .cloned()
                                .unwrap_or_else(|| format!("N{}", nonterminal_id.0));
                            let prod_ids: Vec<usize> = production_ids.iter().map(|p| p.0).collect();
                            json!({"type": "reduce", "nonterminal": nt_name, "len": len, "production_ids": prod_ids})
                        }
                        Stage7ShiftsAndReducesLookaheadValue::Split { shift, reduces } => {
                            // GLR split - both shift and reduce possible
                            let mut parts = Vec::new();
                            if let Some(target) = shift {
                                parts.push(json!({"type": "shift", "target": target.0}));
                            }
                            for (len, nts) in reduces {
                                for (nt_id, prod_ids) in nts {
                                    let nt_name = nonterminal_names.get(&nt_id.0)
                                        .cloned()
                                        .unwrap_or_else(|| format!("N{}", nt_id.0));
                                    let pids: Vec<usize> = prod_ids.iter().map(|p| p.0).collect();
                                    parts.push(json!({"type": "reduce", "nonterminal": nt_name, "len": len, "production_ids": pids}));
                                }
                            }
                            json!({"type": "split", "actions": parts})
                        }
                    };
                    (term_name, action_val)
                })
                .collect();

            // Build goto map (nonterminals -> goto state)
            let gotos: BTreeMap<String, serde_json::Value> = row.get_gotos()
                .iter()
                .filter_map(|(nt_id, goto)| {
                    goto.state_id.map(|sid| {
                        let nt_name = nonterminal_names.get(&nt_id.0)
                            .cloned()
                            .unwrap_or_else(|| format!("N{}", nt_id.0));
                        (nt_name, json!({"target": sid.0, "accept": goto.accept}))
                    })
                })
                .collect();

            // Handle default reduce
            let default_reduce = row.default_reduce.as_ref().map(|action| {
                match action {
                    Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                        let nt_name = nonterminal_names.get(&nonterminal_id.0)
                            .cloned()
                            .unwrap_or_else(|| format!("N{}", nonterminal_id.0));
                        let pids: Vec<usize> = production_ids.iter().map(|p| p.0).collect();
                        json!({"type": "reduce", "nonterminal": nt_name, "len": len, "production_ids": pids})
                    }
                    _ => json!(null)
                }
            });

            json!({
                "state_id": state_id.0,
                "actions": actions,
                "gotos": gotos,
                "default_reduce": default_reduce
            })
        })
        .collect();

    // Build productions list for visualization (indexed by production ID)
    let productions_data: Vec<serde_json::Value> = parser.productions.iter()
        .enumerate()
        .map(|(idx, prod)| {
            let lhs = prod.lhs.0.clone();
            let rhs: Vec<String> = prod.rhs.iter().map(|sym| {
                match sym {
                    sep1::glr::grammar::Symbol::Terminal(t) => match t {
                        Terminal::Literal(bytes) => format!("'{}'", String::from_utf8_lossy(bytes)),
                        Terminal::RegexName(name) => name.clone(),
                    },
                    sep1::glr::grammar::Symbol::NonTerminal(nt) => nt.0.clone(),
                }
            }).collect();
            json!({
                "id": idx,
                "lhs": lhs,
                "rhs": rhs
            })
        })
        .collect();

    // Build internal_to_token mapping for vocab ID lookup
    // Maps internal LLM token ID -> token string
    let internal_to_token: BTreeMap<usize, String> = internal_llm_token_map.iter()
        .map(|(bytes, llm_id)| (llm_id.0, String::from_utf8_lossy(bytes).to_string()))
        .collect();

    let output = json!({
        "grammar_text": grammar_text,
        "terminal_names": terminal_names,
        "nonterminal_names": nonterminal_names,
        "productions": productions_data,
        "characterizations": char_map,
        "template_dfas": template_map,
        "terminal_dwa": format!("{}", terminal_dwa),
        "unresolved_nwa": format!("{}", unresolved_nwa),
        "template_regions": template_regions_data,
        "entry_node_mapping": entry_node_mapping,
        "combined_start_mapping": combined_start_mapping,
        "final_dwa": format!("{}", final_dwa),
        "terminal_map": terminal_to_token_id.iter().map(|(k, v)| (format!("{:?}", k), v.0)).collect::<BTreeMap<_, _>>(),
        "parse_table": parse_table_data,
        "internal_to_token": internal_to_token,
    });

    let mut file = File::create(&cli.output)?;
    file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;

    println!("✓ Artifacts dumped to {:?}", cli.output);
    Ok(())
}
