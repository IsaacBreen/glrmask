use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::path::PathBuf;
use clap::Parser;

use sep1::interface::{CompiledGrammar, GrammarDefinition};
use sep1::glr::grammar::Terminal;
use sep1::glr::table::TerminalID;
use sep1::precompute4::characterize::compute_all_characterizations;
use sep1::precompute4::template_nwa::{build_template_dwas, build_ignore_terminal_dwa};
use sep1::precompute4::weighted_automata::{NWA, NWABody, NWAState, NWAStates, Weight};
use sep1::precompute4::weighted_automata::common::Label;
use sep1::precompute4::full_dwa::{
    nwa_special_map, finalize_and_optimize_and_determinize,
};
use sep1::constraint_precompute::run_precompute1;
use sep1::tokenizer::{LLMTokenID, TokenizerStateID};
use sep1::constraint::LLMTokenBV;
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

    // 2. Compute ALL Characterizations
    println!("Computing all characterizations...");
    let all_chars = compute_all_characterizations(parser);
    let char_map: BTreeMap<String, String> = all_chars.iter()
        .map(|(tid, c)| (format!("{:?}", tid), format!("{:?}", c)))
        .collect();

    // 3. Build ALL Template DFAs
    println!("Building all Template DFAs...");
    let template_dwas = build_template_dwas(parser).expect("Failed to build template DWAs");
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

    // 4. Build Skeleton DWA (Terminal DWA / Precompute1)
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
    
    // Handle case where there are no literal terminals
    let internal_max_llm_token = if token_id_counter > 0 { token_id_counter - 1 } else { 0 };
    let terminals_count = parser.terminal_map.len();
    let active_states = vec![tokenizer.initial_state_id()];

    let mut skeleton_dwa = run_precompute1(
        tokenizer,
        Some(parser),
        &internal_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        active_states,
    );
    skeleton_dwa.simplify();
    if is_debug_level_enabled(4) {
        println!("Skeleton DWA (Terminal DWA):");
        println!("{}", skeleton_dwa);
    }

    // 5. Build Unresolved NWA (WITHOUT canonicalize_bundle)
    // This creates a cleaner NWA where each terminal's template DFA is copied directly,
    // with epsilon edges connecting entry nodes to template starts and template ends to bodies.
    println!("Building Unresolved NWA...");
    
    let input_nwa = NWA::from_dwa(&skeleton_dwa);
    let reversed_nwa = input_nwa.reverse();
    let traversal_data = reversed_nwa.compute_traversal_data();

    let offset = parser.terminal_map.len() as Label;

    // Build individual template NWAs for each terminal (no combining/determinization)
    let mut individual_template_nwas: HashMap<Option<TerminalID>, NWA> = HashMap::new();
    for (term_id, template_dwa) in &template_dwas {
        individual_template_nwas.insert(Some(*term_id), NWA::from_dwa(template_dwa));
    }
    individual_template_nwas.insert(None, NWA::from_dwa(&ignore_dwa));

    // Track which states come from which template for visualization
    // Maps state_id -> Option<TerminalID> (None = entry node or other)
    let template_regions: Arc<Mutex<Vec<(usize, usize, Option<usize>)>>> = Arc::new(Mutex::new(Vec::new()));

    // Pass 2 Traversal with new data structure:
    // BTreeMap<Option<TerminalID>, BTreeMap<NWABody, Weight>>
    // This groups by terminal first, then by body - opposite of old approach.
    let states_arena = RefCell::new(NWAStates::default());
    let initial_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_states: vec![start] }
    };
    
    // Initial: terminal=None maps to initial_body with full weight
    let initial_term_body_map: BTreeMap<Option<TerminalID>, BTreeMap<NWABody, Weight>> = 
        BTreeMap::from([(None, BTreeMap::from([(initial_body.clone(), Weight::all())]))]);
    
    let initial_values_full: Vec<(usize, (BTreeMap<Option<TerminalID>, BTreeMap<NWABody, Weight>>, LLMTokenBV))> =
        reversed_nwa.body.start_states.iter()
            .map(|&s| (s, (initial_term_body_map.clone(), LLMTokenBV::max_ones())))
            .collect();

    let final_bodies_arc: Arc<Mutex<BTreeMap<TokenizerStateID, Vec<(NWABody, Weight)>>>> = Arc::new(Mutex::new(BTreeMap::new()));
    let template_regions_clone = template_regions.clone();

    nwa_special_map(
        &reversed_nwa, &traversal_data, initial_values_full,
        |current_val: &(BTreeMap<Option<TerminalID>, BTreeMap<NWABody, Weight>>, LLMTokenBV), edge_label, transitions| {
            let (current_term_bodies, current_tokens) = current_val;
            if let Some(lbl) = edge_label {
                if lbl >= offset {
                    // This is a tokenizer state transition - record final bodies
                    let tsid = TokenizerStateID((lbl - offset) as usize);
                    let mut fb = final_bodies_arc.lock().unwrap();
                    let list = fb.entry(tsid).or_default();
                    for (_dest, weight) in transitions {
                        let w_bv: LLMTokenBV = weight.clone().into();
                        let intersection_bv = current_tokens & &w_bv;
                        if !intersection_bv.is_empty() {
                            let final_w = Weight::from_rsb(intersection_bv.inner().clone());
                            // Collect all bodies from all terminals
                            for body_map in current_term_bodies.values() {
                                for body in body_map.keys() {
                                    list.push((body.clone(), final_w.clone()));
                                }
                            }
                        }
                    }
                    return Vec::new();
                }
            }
            let terminal_id = edge_label.map(|l| TerminalID(l as usize));
            let mut results = Vec::new();
            for (dest_id, weight) in transitions {
                let edge_bv: LLMTokenBV = weight.clone().into();
                let next_tokens = current_tokens & &edge_bv;
                if next_tokens.is_empty() { continue; }
                
                // New structure: BTreeMap<Option<TerminalID>, BTreeMap<NWABody, Weight>>
                let mut new_term_bodies = BTreeMap::new();
                // For each existing body across all terminals, record it under this terminal
                for body_map in current_term_bodies.values() {
                    for body in body_map.keys() {
                        new_term_bodies.entry(terminal_id)
                            .or_insert_with(BTreeMap::new)
                            .insert(body.clone(), weight.clone());
                    }
                }
                results.push((*dest_id, (new_term_bodies, next_tokens)));
            }
            results
        },
        |val1, val2| {
            let (term_bodies1, tokens1) = val1;
            let (term_bodies2, tokens2) = val2;
            for (term_id, body_map2) in term_bodies2 {
                let body_map1 = term_bodies1.entry(term_id).or_default();
                for (body, weight2) in body_map2 {
                    *body_map1.entry(body).or_insert_with(Weight::zeros) |= &weight2;
                }
            }
            *tokens1 |= &tokens2;
        },
        |_, val| {
            // NEW PROCESS FUNCTION: No canonicalize_bundle!
            // For each terminal DWA node, create an unresolved NWA structure:
            // - Create entry node A
            // - For each terminal_id: copy template DFA, connect A -> S (full weight), E -> B (weight W)
            let (term_bodies_map, tokens) = val;
            
            let mut states = states_arena.borrow_mut();
            
            // Create entry node A for this terminal DWA node
            let entry_node_a = states.add_state();
            
            // Loop over (Option<TerminalID>, BTreeMap<NWABody, Weight>) entries
            for (term_id, body_weight_map) in term_bodies_map {
                // Get the template DFA for this terminal
                let template_nwa = individual_template_nwas.get(&term_id)
                    .expect("Template must exist for terminal");
                
                // Copy template into NWA states, returning its start/end nodes
                // The "start" is template_nwa.body.start_states
                // The "end" nodes are those with final_weight
                let template_offset = states.len();
                let template_end = template_offset + template_nwa.states.len();
                
                // Record this template region for visualization
                {
                    let mut regions = template_regions_clone.lock().unwrap();
                    regions.push((template_offset, template_end, term_id.map(|t| t.0)));
                }
                
                // Copy all template states
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
                    
                    // For final states: instead of copying final_weight,
                    // create epsilon edges to each NWA body B with weight W
                    if old_state.final_weight.is_some() {
                        for (body, weight) in &body_weight_map {
                            for &b_start in &body.start_states {
                                new_state.epsilons.push((b_start, weight.clone()));
                            }
                        }
                    }
                    
                    states.0.push(new_state);
                }
                
                // Create epsilon edge A -> S (template start) with FULL weight
                let template_start_states: Vec<usize> = template_nwa.body.start_states.iter()
                    .map(|s| s + template_offset)
                    .collect();
                for s in template_start_states {
                    states[entry_node_a].epsilons.push((s, Weight::all()));
                }
            }
            
            if !tokens.is_empty() {
                let result_body = NWABody { start_states: vec![entry_node_a] };
                let mut next_term_bodies = BTreeMap::new();
                next_term_bodies.insert(None, BTreeMap::from([(result_body, Weight::all())]));
                Some((next_term_bodies, tokens))
            } else { 
                None 
            }
        },
    );

    let final_bodies = Arc::try_unwrap(final_bodies_arc).unwrap().into_inner().unwrap();
    let mut combined_nwa_states = states_arena.into_inner();
    let combined_start_state = combined_nwa_states.add_state();
    for (tsid, list) in final_bodies {
        let label = tsid.0 as Label;
        for (body, weight) in list {
            for &s in &body.start_states {
                combined_nwa_states.add_transition(combined_start_state, label, s, weight.clone()).unwrap();
            }
        }
    }

    let mut unresolved_nwa = NWA { states: combined_nwa_states, body: NWABody { start_states: vec![combined_start_state] } };
    if is_debug_level_enabled(4) {
        println!("Unresolved NWA:");
        println!("{}", unresolved_nwa);
    }

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
    skeleton_dwa.optimize_for_visualization();
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
    
    let output = json!({
        "grammar_text": grammar_text,
        "terminal_names": terminal_names,
        "nonterminal_names": nonterminal_names,
        "characterizations": char_map,
        "template_dfas": template_map,
        "skeleton_dwa": format!("{}", skeleton_dwa),
        "unresolved_nwa": format!("{}", unresolved_nwa),
        "template_regions": template_regions_data,
        "final_dwa": format!("{}", final_dwa),
        "terminal_map": terminal_to_token_id.iter().map(|(k, v)| (format!("{:?}", k), v.0)).collect::<BTreeMap<_, _>>(),
    });

    let mut file = File::create(&cli.output)?;
    file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;

    println!("✓ Artifacts dumped to {:?}", cli.output);
    Ok(())
}
