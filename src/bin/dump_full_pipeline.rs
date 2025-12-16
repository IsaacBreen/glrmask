use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;
use std::path::PathBuf;
use clap::Parser;

use sep1::interface::{CompiledGrammar, GrammarDefinition};
use sep1::glr::grammar::Terminal;
use sep1::glr::table::{TerminalID, Stage7ShiftsAndReducesLookaheadValue};
use sep1::precompute4::characterize::compute_all_characterizations;
use sep1::precompute4::template_nwa::{build_terminal_dwas, build_ignore_terminal_dwa};
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

    // 5. Build Unresolved NWA (WITHOUT canonicalize_bundle)
    // This creates a cleaner NWA where each terminal's template DFA is copied directly,
    // with epsilon edges connecting entry nodes to template starts and template ends to bodies.
    println!("Building Unresolved NWA...");

    let input_nwa = NWA::from_dwa(&terminal_dwa);
    let reversed_nwa = input_nwa.reverse();
    let traversal_data = reversed_nwa.compute_traversal_data();

    // The super_start state (created by reverse()) is at index terminal_dwa.states.len()
    // We don't want to create entry nodes for this artificial state
    let super_start_state = terminal_dwa.states.len();

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

    // Track which terminal DWA states we've created entry nodes for
    // Maps terminal DWA state ID -> entry node ID in combined NWA
    let entry_nodes_for_tdwa_state: RefCell<HashMap<usize, usize>> = RefCell::new(HashMap::new());

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
        |tdwa_state_id, val| {
            // NEW PROCESS FUNCTION: No canonicalize_bundle!
            // For each terminal DWA node, create an unresolved NWA structure:
            // - Create entry node A (or reuse if we've already created one for this terminal DWA state)
            // - For each terminal_id: copy template DFA, connect A -> S (full weight), E -> B (weight W)
            let (term_bodies_map, tokens) = val;

            // Skip creating entry nodes for the super_start state (it's just an artifact of reversal)
            if tdwa_state_id == super_start_state {
                // Just propagate values forward without creating structure
                if !tokens.is_empty() {
                    // Return the same term_bodies_map but packaged for propagation
                    return Some((term_bodies_map.clone(), tokens));
                } else {
                    return None;
                }
            }

            let mut states = states_arena.borrow_mut();
            let mut entry_nodes_map = entry_nodes_for_tdwa_state.borrow_mut();

            // Get or create entry node A for this terminal DWA state
            let is_new = !entry_nodes_map.contains_key(&tdwa_state_id);
            let entry_node_a = *entry_nodes_map.entry(tdwa_state_id).or_insert_with(|| states.add_state());
            if is_new && is_debug_level_enabled(4) {
                println!("  Created entry node {} for terminal DWA state {}", entry_node_a, tdwa_state_id);
            }

            // Loop over (Option<TerminalID>, BTreeMap<NWABody, Weight>) entries
            for (term_id, body_weight_map) in term_bodies_map {
                // Get the template DFA for this terminal
                let template_nwa = individual_template_nwas.get(&term_id)
                    .expect("Template must exist for terminal");

                // Copy template into NWA states, returning its start/end nodes
                // The "start" is template_nwa.body.start_states
                // The "end" nodes are those with final_weight
                let template_offset = states.len();
                
                // Create funnel exit node E for this template
                // This ensures each template has exactly ONE exit node
                let exit_node_e = states.add_state();
                
                let template_end = states.len() + template_nwa.states.len();

                // Record this template region for visualization (includes exit node)
                {
                    let mut regions = template_regions_clone.lock().unwrap();
                    regions.push((template_offset, template_end, term_id.map(|t| t.0)));
                }

                // Copy all template states
                for old_state in &template_nwa.states.0 {
                    let mut new_state = NWAState::default();

                    // Copy transitions (adjusting state indices - account for exit node)
                    for (lbl, targets) in &old_state.transitions {
                        let new_targets: Vec<(usize, Weight)> = targets.iter()
                            .map(|(t, w)| (*t + template_offset + 1, w.clone()))
                            .collect();
                        if !new_targets.is_empty() {
                            new_state.transitions.insert(*lbl, new_targets);
                        }
                    }

                    // Copy epsilons (adjusting state indices - account for exit node)
                    for (target, w) in &old_state.epsilons {
                        new_state.epsilons.push((*target + template_offset + 1, w.clone()));
                    }

                    // For final states: connect to exit node E with full weight
                    // (instead of directly to each body)
                    if old_state.final_weight.is_some() {
                        new_state.epsilons.push((exit_node_e, Weight::all()));
                    }

                    states.0.push(new_state);
                }

                // Connect exit node E to all destination bodies
                for (body, weight) in &body_weight_map {
                    for &b_start in &body.start_states {
                        states[exit_node_e].epsilons.push((b_start, weight.clone()));
                    }
                }

                // Create epsilon edge A -> S (template start) with FULL weight
                // Account for exit node offset (+1)
                let template_start_states: Vec<usize> = template_nwa.body.start_states.iter()
                    .map(|s| s + template_offset + 1)
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

    // Output the entry node mapping for debugging
    if is_debug_level_enabled(4) {
        let entry_map = entry_nodes_for_tdwa_state.borrow();
        println!("Entry node mapping (terminal DWA state -> NWA entry node):");
        for (tdwa_state, nwa_entry) in entry_map.iter() {
            println!("  terminal DWA state {} -> NWA entry node {}", tdwa_state, nwa_entry);
        }
    }

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

    // Validate: each template region should have exactly 1 entry node receiving incoming edges
    // and exactly 1 exit node sending outgoing edges
    {
        let regions = template_regions.lock().unwrap();
        for (start, end, term_id) in regions.iter() {
            if term_id.is_none() {
                continue; // Skip non-terminal regions
            }
            
            // Count incoming edges to nodes in this region (from outside)
            let mut entry_node_incoming: HashMap<usize, usize> = HashMap::new();
            // Count outgoing edges from nodes in this region (to outside)
            let mut exit_node_outgoing: HashMap<usize, usize> = HashMap::new();
            
            for (state_id, state) in unresolved_nwa.states.0.iter().enumerate() {
                let src_in_region = state_id >= *start && state_id < *end;
                
                // Check epsilon edges
                for (target, _weight) in &state.epsilons {
                    let dst_in_region = *target >= *start && *target < *end;
                    
                    if !src_in_region && dst_in_region {
                        // Incoming edge to region
                        *entry_node_incoming.entry(*target).or_insert(0) += 1;
                    } else if src_in_region && !dst_in_region {
                        // Outgoing edge from region
                        *exit_node_outgoing.entry(state_id).or_insert(0) += 1;
                    }
                }
                
                // Check labeled transitions
                for (_label, targets) in &state.transitions {
                    for (target, _weight) in targets {
                        let dst_in_region = *target >= *start && *target < *end;
                        
                        if !src_in_region && dst_in_region {
                            *entry_node_incoming.entry(*target).or_insert(0) += 1;
                        } else if src_in_region && !dst_in_region {
                            *exit_node_outgoing.entry(state_id).or_insert(0) += 1;
                        }
                    }
                }
            }
            
            // Validate: exactly 1 entry node and 1 exit node
            let entry_nodes: Vec<_> = entry_node_incoming.keys().collect();
            let exit_nodes: Vec<_> = exit_node_outgoing.keys().collect();
            
            if entry_nodes.len() != 1 {
                panic!(
                    "Template region {:?} (states {}-{}) has {} entry nodes (expected 1): {:?}",
                    term_id, start, end, entry_nodes.len(), entry_nodes
                );
            }
            
            if exit_nodes.len() != 1 {
                panic!(
                    "Template region {:?} (states {}-{}) has {} exit nodes (expected 1): {:?}",
                    term_id, start, end, exit_nodes.len(), exit_nodes
                );
            }
            
            if is_debug_level_enabled(4) {
                println!(
                    "  Template {:?} (states {}-{}): entry node {:?}, exit node {:?}",
                    term_id, start, end, entry_nodes[0], exit_nodes[0]
                );
            }
        }
        
        if is_debug_level_enabled(3) {
            println!("✓ All template regions have exactly 1 entry and 1 exit node");
        }
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
    terminal_dwa.optimize_for_visualization();
    // NOTE: We do NOT call optimize_for_visualization() on unresolved_nwa because
    // it would eliminate epsilon chains, breaking the funnel exit node structure
    // that we carefully constructed. Each template region has exactly 1 entry and 1 exit node.
    // unresolved_nwa.optimize_for_visualization();
    resolved_nwa.optimize_for_visualization();
    final_dwa.optimize_for_visualization();

    // 7. Dump Everything
    println!("Dumping artifacts to {:?}...", cli.output);

    // NEW: Build explicit Stack-Based Structure for Visualization
    // Group transitions by (src, dst)
    let mut stacks_map: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
    for (src_idx, state) in terminal_dwa.states.iter().enumerate() {
        for (label, target) in &state.transitions {
            let term_id = *label as usize;
            stacks_map.entry((src_idx, *target))
                .or_default()
                .push(term_id);
        }
    }

    let stacks_json: Vec<serde_json::Value> = stacks_map.into_iter()
        .map(|((src, dst), terms)| {
            json!({
                "src": src,
                "dst": dst,
                "terminals": terms
            })
        }).collect();

    let nodes_json: Vec<serde_json::Value> = terminal_dwa.states.iter().enumerate()
        .map(|(idx, s)| {
            json!({
                "id": idx,
                "is_start": idx == 0, // Assuming 0 is start? No, check initial_state_id
                "final_weight": s.final_weight.is_some()
            })
        })
        .collect();

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

    // Build entry node mapping for validation
    // Maps terminal DWA state ID -> NWA entry node ID
    let entry_node_mapping: Vec<serde_json::Value> = entry_nodes_for_tdwa_state.borrow()
        .iter()
        .map(|(tdwa_state, nwa_entry)| {
            json!({
                "terminal_dwa_state": tdwa_state,
                "nwa_entry_node": nwa_entry
            })
        })
        .collect();

    // Also include the combined_start mapping (terminal DWA state 0 -> NWA combined_start)
    let combined_start_mapping = json!({
        "terminal_dwa_state": 0,
        "nwa_entry_node": combined_start_state
    });

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
        "nwa_structure": {
             "nodes": nodes_json,
             "stacks": stacks_json
        }
    });

    let mut file = File::create(&cli.output)?;
    file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;

    println!("✓ Artifacts dumped to {:?}", cli.output);
    Ok(())
}
