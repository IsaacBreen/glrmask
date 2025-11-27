use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

use sep1::interface::{CompiledGrammar, GrammarDefinition};
use sep1::glr::grammar::Terminal;
use sep1::glr::table::TerminalID;
use sep1::precompute4::characterize::compute_all_characterizations;
use sep1::precompute4::template_nwa::{build_template_dwas, build_ignore_terminal_dwa};
use sep1::precompute4::weighted_automata::{NWA, NWABody, NWAStates, Weight};
use sep1::precompute4::weighted_automata::common::Label;
use sep1::precompute4::full_dwa::{
    canonicalize_bundle, instantiate_nwa_template_into, nwa_special_map, 
    precompute_token_bvs_and_signatures, resolve_negatives_and_optimize_and_determinize,
    Signature
};
use sep1::constraint_precompute::run_precompute1;
use sep1::tokenizer::{LLMTokenID, TokenizerStateID};
use sep1::constraint::LLMTokenBV;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Define Grammar (use uppercase in output, lowercase for parsing)
    let ebnf = r#"
        s ::= a "$";
        a ::= 'a' b;
        b ::= 'b' c;
        c ::= 'c' a | 'c';
    "#;

    println!("Compiling grammar...");
    let grammar_def = GrammarDefinition::from_ebnf(ebnf).expect("Failed to parse EBNF");
    let compiled_grammar = CompiledGrammar::from_definition(Arc::new(grammar_def));
    let parser = &compiled_grammar.glr_parser;
    let tokenizer = &compiled_grammar.tokenizer;

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
    
    let template_map: BTreeMap<String, String> = template_dwas.iter()
        .map(|(tid, dwa)| (format!("{:?}", tid), format!("{}", dwa)))
        .collect();

    // 4. Build Skeleton DWA (FULL Terminal DWA / Precompute1)
    println!("Building Skeleton DWA (Terminal DWA / Precompute1)...");
    
    // Create LLM token map where each terminal is a token
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
    let internal_max_llm_token = (token_id_counter - 1) as usize;
    let terminals_count = parser.terminal_map.len();
    let active_states = vec![tokenizer.initial_state_id()];

    let skeleton_dwa = run_precompute1(
        tokenizer,
        Some(parser),
        None,
        &internal_llm_token_map,
        internal_max_llm_token,
        terminals_count,
        active_states,
    );

    // 5. Build Flattened NWA (Replicating Precompute4 Pass 2)
    println!("Building Flattened NWA...");
    
    let input_nwa = NWA::from_dwa(&skeleton_dwa);
    let reversed_nwa = input_nwa.reverse();
    let traversal_data = reversed_nwa.compute_traversal_data();

    let initial_tokens = LLMTokenBV::max_ones();
    let mut initial_values_bv = Vec::new();
    for &start in &reversed_nwa.body.start_states {
        initial_values_bv.push((start, initial_tokens.clone()));
    }

    let offset = parser.terminal_map.len() as Label;
    let (node_tokens, mut unique_signatures) = precompute_token_bvs_and_signatures(&reversed_nwa, &traversal_data, initial_values_bv, offset);
    unique_signatures.insert(vec![vec![None]]);

    // Populate template cache
    let mut template_cache = HashMap::new();
    for sig in unique_signatures {
        let terminals = &sig[0];
        let mut combined_nwa = NWA::new_empty();
        for term_opt in terminals {
            let template = match term_opt {
                Some(term_id) => {
                    if Some(*term_id) == parser.ignore_terminal_id {
                        &ignore_dwa
                    } else {
                        template_dwas.get(term_id).unwrap_or(&ignore_dwa)
                    }
                },
                None => &ignore_dwa,
            };
            NWA::union_assign(&mut combined_nwa, &NWA::from_dwa(template));
        }
        let mut dwa = combined_nwa.determinize();
        dwa.simplify_lightweight();
        template_cache.insert(sig, NWA::from_dwa(&dwa));
    }

    // Pass 2 Traversal
    let states_arena = RefCell::new(NWAStates::default());
    let initial_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_states: vec![start] }
    };
    let initial_term_map: BTreeMap<Option<TerminalID>, Weight> = BTreeMap::from([(None, Weight::all())]);
    let initial_values_full: Vec<(usize, (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV))> =
        reversed_nwa.body.start_states.iter().map(|&s| (s, (BTreeMap::from([(initial_body.clone(), initial_term_map.clone())]), LLMTokenBV::max_ones()))).collect();

    let final_bodies_arc: Arc<Mutex<BTreeMap<TokenizerStateID, Vec<(NWABody, Weight)>>>> = Arc::new(Mutex::new(BTreeMap::new()));

    nwa_special_map(
        &reversed_nwa, &traversal_data, initial_values_full,
        |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV), edge_label, transitions| {
            let (current_bodies, current_tokens) = current_val;
            if let Some(lbl) = edge_label {
                if lbl >= offset {
                    let tsid = TokenizerStateID((lbl - offset) as usize);
                    let mut fb = final_bodies_arc.lock().unwrap();
                    let list = fb.entry(tsid).or_default();
                    for (_dest, weight) in transitions {
                        let w_bv: LLMTokenBV = weight.clone().into();
                        let intersection_bv = current_tokens & &w_bv;
                        if !intersection_bv.is_empty() {
                            // FIX: Use .inner() method instead of .inner field
                            let final_w = Weight::from_rsb(intersection_bv.inner().clone());
                            for body in current_bodies.keys() {
                                list.push((body.clone(), final_w.clone()));
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
                let mut terminal_map = BTreeMap::new();
                terminal_map.insert(terminal_id, weight.clone());
                let mut body_map = BTreeMap::new();
                for body in current_bodies.keys() { body_map.insert(body.clone(), terminal_map.clone()); }
                results.push((*dest_id, (body_map, next_tokens)));
            }
            results
        },
        |val1, val2| {
            let (bodies1, tokens1) = val1;
            let (bodies2, tokens2) = val2;
            for (right_body, term_map2) in bodies2 {
                let term_map1 = bodies1.entry(right_body.clone()).or_default();
                for (term, weight2) in term_map2 { *term_map1.entry(term).or_insert_with(Weight::zeros) |= &weight2; }
            }
            *tokens1 |= &tokens2;
        },
        |_, val| {
            let (nwa_bodies_map, tokens) = val;
            let mut nwa_body = NWABody { start_states: vec![] };
            for (right_body, terminal_map) in nwa_bodies_map {
                let (signature, concrete_weights) = canonicalize_bundle(terminal_map);
                let template_nwa = template_cache.get(&signature).expect("Template must exist");
                let mut states = states_arena.borrow_mut();
                let composed_body = instantiate_nwa_template_into(template_nwa, &concrete_weights, &mut states, &right_body);
                nwa_body = NWABody::union(&nwa_body, &composed_body);
            }
            if !tokens.is_empty() {
                let mut next_body_map = BTreeMap::new(); next_body_map.insert(nwa_body, BTreeMap::new());
                Some((next_body_map, tokens))
            } else { None }
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

    let flattened_nwa = NWA { states: combined_nwa_states, body: NWABody { start_states: vec![combined_start_state] } };

    // 6. Build Final DWA
    println!("Building Final DWA...");
    let mut final_dwa = resolve_negatives_and_optimize_and_determinize(parser, flattened_nwa.clone());
    final_dwa.simplify();

    // 7. Dump Everything
    println!("Dumping artifacts...");
    
    let output = json!({
        "grammar_ebnf": ebnf.replace("s ::=", "S ::=").replace("a ::=", "A ::=").replace("b ::=", "B ::=").replace("c ::=", "C ::="),
        "lalr_table": format!("{:?}", parser.table),
        "characterizations_all": char_map,
        "template_dfas_all": template_map,
        "skeleton_dwa": format!("{}", skeleton_dwa),
        "flattened_nwa": format!("{}", flattened_nwa),
        "final_dwa": format!("{}", final_dwa),
        "terminal_map": terminal_to_token_id.iter().map(|(k, v)| (format!("{:?}", k), v.0)).collect::<BTreeMap<_, _>>(),
    });

    let mut file = File::create("pipeline_artifacts.json")?;
    file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;

    println!("Artifacts dumped to pipeline_artifacts.json");
    Ok(())
}
