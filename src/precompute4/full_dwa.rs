use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::time::Instant;

use chrono::Local;

use crate::constraint::{LLMTokenBV, PrecomputeNode1Index, Trie1GodWrapper};
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::json_serialization::JSONConvertible;
use crate::precompute4::nwa_optimizations::{prune_continuations_from_final_states, simplify_default_transitions};
use crate::precompute4::resolve_negatives::{apply_cancellations, apply_finality_fixpoint, remove_negative_transitions};
use crate::precompute4::template_nwa::{build_epsilon_dwa, build_ignore_terminal_dwa, build_template_dwas};
use crate::precompute4::weighted_automata::{DWA, NWA, NWABody, NWAStates, Weight};
use crate::r#macro::is_debug_level_enabled;
use crate::tokenizer::TokenizerStateID;

struct SimplifyRustfstConfig {
    rm_epsilon: bool,
    determinize: bool,
}

impl SimplifyRustfstConfig {
    fn default() -> Self { Self { rm_epsilon: false, determinize: false } }
    fn with_rm_epsilon(mut self, val: bool) -> Self { self.rm_epsilon = val; self }
    fn with_determinize(mut self, val: bool) -> Self { self.determinize = val; self }
}

impl NWA {
    pub fn determinize_to_dwa_with_rustfst(&self) -> DWA { self.determinize_to_dwa() }
    pub fn simplify_rustfst(&mut self) { self.simplify(); }
    pub fn simplify_rustfst_with_config(&mut self, _config: SimplifyRustfstConfig) { self.simplify(); }
}

// Re-export for backward compatibility: `FullDWABuildError` used to be defined here.
pub use crate::precompute4::template_nwa::FullDWABuildError;

pub type Precomputed4 = DWA;

/// Public API: precompute4 using an NWA-first approach, determinizing at the end.
pub fn precompute4(
    parser: &GLRParser,
    precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) -> DWA {
    let now_total = Instant::now();
    let now = Instant::now();
    crate::debug!(5, "Starting precompute4...");

    // 1. Build template DWAs for all terminals.
    let template_dwas = match build_template_dwas(parser) {
        Ok(m) => m,
        Err(e) => panic!("Failed to build template DWAs: {:?}", e),
    };
    // Print the template DWA for terminal ''`''
    // println!("parser: {}", parser);
    let allowed = [0, 69, 79, 101, 131, 151, 161, 165, 166, 279, 280, 286, 300, 310, 371, 374, 375, 376, 400, 422, 423, 429, 436, 437, 438, 458, 459, 476];
    let mut template_dwa = template_dwas.get(&parser.terminal_map.get_by_left(&crate::glr::grammar::Terminal::Literal(b"`".to_vec())).unwrap()).expect_else(|| "No template DWA for terminal ''`''".to_string()).clone();
    template_dwa.states.0.iter_mut().for_each(|st| st.transitions.retain(|&label, _| allowed.contains(&label) || allowed.contains(&-label) || label == 0));
    template_dwa.simplify();
    println!("Template DWA for terminal ''`'':\n{}", template_dwa);
    let ignore_dwa = build_ignore_terminal_dwa();
    crate::debug!(4, "Built {} template DWAs in {:?}", template_dwas.len(), now.elapsed());
    if is_debug_level_enabled(5) {
        for (term, dwa) in template_dwas.iter().take(5) {
            crate::debug!(5, "Stats for template DWA for terminal {:?}:\n{}", term, dwa.stats());
        }
    }

    // 2. Shared NWA state arena.
    let states_arena = RefCell::new(NWAStates::default());

    // 3. Reverse the precompute1 trie.
    let trie1_roots: Vec<_> = precomputed1.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &trie1_roots);

    let leaf_node = all_nodes
        .iter()
        .find_map(|&idx| idx.read(trie1_god).and_then(|g| if g.value.end { Some(idx) } else { None }))
        .expect("Precompute1 trie must have a single leaf node.");

    let reversed_trie1_god = Trie::reverse(trie1_god, &trie1_roots);
    let reversed_trie_root = leaf_node;

    // 4. Traverse the reversed trie with NWA bodies.
    let initial_nwa_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_state: start }
    };
    let initial_tokens = LLMTokenBV::max_ones();
    use crate::glr::table::TerminalID;
    let initial_term_map: BTreeMap<Option<TerminalID>, Weight> = BTreeMap::from([(None, Weight::all())]);
    let initial_body_map = BTreeMap::from([(initial_nwa_body, initial_term_map)]);
    let initial_values: Vec<(Trie2Index, (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV))> =
        vec![(reversed_trie_root, (initial_body_map, initial_tokens))];

    let traversal_data =
        Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root]).expect("Failed to compute traversal data for reversed trie1");

    let mut original_trie1_roots_map: BTreeMap<PrecomputeNode1Index, Vec<TokenizerStateID>> = BTreeMap::new();
    for (k, v) in precomputed1.iter() {
        original_trie1_roots_map.entry(*v).or_default().push(*k);
    }

    let options = crate::datastructures::trie::PrettyPrintOptions::default().omit_nodes().omit_depth();
    crate::debug!(5, "Trie:\n{}", Trie::pretty_print_with_options(&trie1_god, &trie1_roots, &options));
    crate::debug!(5, "Reversed trie:\n{}", Trie::pretty_print_with_options(&reversed_trie1_god, &[reversed_trie_root], &options));

    let mut final_bodies: BTreeMap<TokenizerStateID, NWABody> = BTreeMap::new();

    let now = Instant::now();
    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values,
        // step function
        |current_val: &(NWABody, LLMTokenBV), edge_terminal_opt, dest_map| {
            let (current_nwa_body, current_tokens) = current_val;
            let terminal_id = *edge_terminal_opt;

            let mut results = Vec::new();
            for (dest_idx, llm_token_bv) in dest_map.iter() {
                let next_tokens = current_tokens & llm_token_bv;
                if next_tokens.is_empty() {
                    continue;
                }
                let weight = Weight::from_rsb(llm_token_bv.inner.as_ref().clone());
                let mut terminal_map = BTreeMap::new();
                terminal_map.insert(terminal_id, weight);
                let mut body_map = BTreeMap::new();
                body_map.insert(*current_nwa_body, terminal_map);
                results.push((*dest_idx, (body_map, next_tokens.clone())));
            }
            results
        },
        // merge function: union via epsilon
        |val1: &mut (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV),
         val2: (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV)| {
            let (bodies1, tokens1) = val1;
            let (bodies2, tokens2) = val2;
            for (right_body, term_map2) in bodies2 {
                let term_map1 = bodies1.entry(right_body).or_default();
                for (term, weight2) in term_map2 {
                    *term_map1.entry(term).or_insert_with(Weight::zeros) |= &weight2;
                }
            }
            *tokens1 |= &tokens2;
        },
        // process function: capture at original roots
        |_node_data,
         node_idx,
         val: (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV)| {
            let (nwa_bodies_map, tokens) = val;

            // Combine all left bodies into a single NWA body via union (epsilon)
            let mut nwa_body = {
                let mut states = states_arena.borrow_mut();
                let start = states.add_state();
                NWABody { start_state: start }
            };

            crate::debug!(6, "NWA states:\n{}", states_arena.borrow());
            crate::debug!(6, "{:?}", nwa_bodies_map);

            for (right_body, terminal_map) in nwa_bodies_map {
                let mut left_bodies = Vec::new();
                for (terminal_id_opt, weight) in terminal_map {
                    if weight.is_empty() {
                        continue;
                    }
                    let template_dwa: &DWA = match terminal_id_opt {
                        Some(terminal_id) if Some(terminal_id) != parser.ignore_terminal_id => {
                            template_dwas.get(&terminal_id).expect_else(|| format!("No template DWA for terminal {:?}", terminal_id))
                        }
                        _ => &ignore_dwa,
                    };
                    let left_body = instantiate_template_nwa_with_weight(&states_arena, template_dwa, weight);
                    left_bodies.push(left_body);
                }
                if left_bodies.is_empty() {
                    continue;
                }
                let left_bodies_union = union_left_bodies(&states_arena, left_bodies);
                let mut states = states_arena.borrow_mut();
                let composed_body =
                    NWA::concatenate_components(&mut states, &left_bodies_union, &right_body, &Weight::all());
                nwa_body = NWA::union_components(&mut states, &nwa_body, &composed_body);
            }

            crate::debug!(
                6,
                "At trie node {:?}, obtained NWA body with start state {} and {} states.",
                node_idx,
                nwa_body.start_state,
                states_arena.borrow().len()
            );
            crate::debug!(6, "NWA body:\n{}", nwa_body);
            crate::debug!(6, "NWA states:\n{}", states_arena.borrow());

            if !tokens.is_empty() {
                if let Some(tokenizer_state_ids) = original_trie1_roots_map.get(&node_idx) {
                    for tokenizer_state_id in tokenizer_state_ids {
                        final_bodies.insert(*tokenizer_state_id, nwa_body.clone());
                    }
                }
                Some((nwa_body, tokens))
            } else {
                None
            }
        },
    );
    crate::debug!(4, "Reversed trie traversal (special_map_grouped) took: {:?}", now.elapsed());

    // Combine all final NWA bodies into a single NWA
    let mut combined_nwa_states = states_arena.into_inner();
    let combined_start_state = combined_nwa_states.add_state();

    for (tok_id, body) in final_bodies {
        let label = tok_id.0 as i16;
        combined_nwa_states
            .add_transition(combined_start_state, label, body.start_state, Weight::all())
            .unwrap();
    }

    let combined_nwa = NWA { states: combined_nwa_states, body: NWABody { start_state: combined_start_state } };
    crate::debug!(4, "Combined NWA has {} states after merging all final bodies.", combined_nwa.states.len());

    let final_dwa = resolve_negatives_and_optimize_and_determinize(parser, combined_nwa);
    crate::debug!(3, "Total precompute4 time: {:?}", now_total.elapsed());
    final_dwa
}

fn resolve_negatives_and_optimize_and_determinize(parser: &GLRParser, mut combined_nwa: NWA) -> DWA {
    let allowed = [0, 69, 79, 101, 131, 151, 161, 165, 166, 279, 280, 286, 300, 310, 371, 374, 375, 376, 400, 422, 423, 429, 436, 437, 438, 458, 459, 476];
    combined_nwa.states.0.iter_mut().for_each(|st| st.transitions.retain(|&label, _| allowed.contains(&label) || allowed.contains(&-label) || label == 0));
    crate::debug!(4, "Starting resolve negatives and optimization and determinization of combined NWA...");
    combined_nwa.simplify_rustfst();
    println!("Combined NWA after filtering transitions:\n{}", combined_nwa);
    crate::debug!(5, "Resolving negative codes in combined NWA: {}", combined_nwa);
    crate::debug!(4, "Combined NWA has {} states.", combined_nwa.states.len());
    crate::debug!(4, "Stats for combined NWA before negative resolution:\n{}", combined_nwa.stats());

    let now = Instant::now();
    crate::debug!(4, "Starting negative code resolution...");
    apply_cancellations(&mut combined_nwa);
    apply_finality_fixpoint(&mut combined_nwa);
    remove_negative_transitions(&mut combined_nwa);
    println!("Combined NWA after filtering transitions:\n{}", combined_nwa);
    combined_nwa.simplify_rustfst();
    crate::debug!(
        4,
        "Negative code resolution took: {:?}. NWA now has {} states.",
        now.elapsed(),
        combined_nwa.states.len()
    );
    crate::debug!(4, "Stats for combined NWA after negative resolution:\n{}", combined_nwa.stats());

    let now = Instant::now();
    crate::debug!(4, "Pruning continuations from final states...");
    prune_continuations_from_final_states(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    crate::debug!(
        4,
        "Pruning and simplifying took: {:?}. NWA now has {} states.",
        now.elapsed(),
        combined_nwa.states.len()
    );
    crate::debug!(4, "Stats for combined NWA after pruning:\n{}", combined_nwa.stats());

    let now = Instant::now();
    crate::debug!(4, "Simplifying default transitions...");
    simplify_default_transitions(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    crate::debug!(
        4,
        "Default transition simplification took: {:?}. NWA now has {} states.",
        now.elapsed(),
        combined_nwa.states.len()
    );
    crate::debug!(4, "Stats for combined NWA after default simplification:\n{}", combined_nwa.stats());

    crate::debug!(4, "Starting simplification before final determinization...");
    let now = Instant::now();
    simplify_remove_epsilon(&mut combined_nwa);
    combined_nwa.simplify();
    simplify_remove_epsilon(&mut combined_nwa);
    crate::debug!(
        4,
        "Simplification before final determinization took: {:?}. NWA now has {} states.",
        now.elapsed(),
        combined_nwa.states.len()
    );
    crate::debug!(4, "Stats for combined NWA before final determinization:\n{}", combined_nwa.stats());

    if env::var("RLLM_DUMP_NWA").is_ok() {
        let timestamp = Local::now().format("%Y%m%d-%H%M%S");
        let filename = format!("nwa_dump_before_final_det_{}.json", timestamp);
        eprintln!("Dumping NWA to {} before final determinization...", filename);
        let f = std::fs::File::create(&filename).expect("Unable to create NWA dump file");
        serde_json::to_writer_pretty(f, &combined_nwa).expect("Unable to write NWA to file");
        eprintln!("NWA dump complete.");
        let parser_filename = format!("parser_dump_before_final_det_{}.json", timestamp);
        eprintln!("Dumping parser to {}...", parser_filename);
        let parser_f = std::fs::File::create(&parser_filename).expect("Unable to create parser dump file");
        let parser_json = parser.to_json();
        serde_json::to_writer_pretty(parser_f, &parser_json).expect("Unable to write parser to file");
        eprintln!("Parser dump complete.");
    }

    let now = Instant::now();
    crate::debug!(4, "Determinizing final combined NWA...");
    combined_nwa = NWA::from_dwa(&combined_nwa.determinize_to_dwa());
    crate::debug!(4, "Stats after final NWA determinization:\n{}", combined_nwa.stats());
    combined_nwa.simplify_rustfst();
    crate::debug!(
        4,
        "Final NWA simplification took: {:?}. NWA now has {} states.",
        now.elapsed(),
        combined_nwa.states.len()
    );
    crate::debug!(4, "Stats for final NWA before DWA determinization:\n{}", combined_nwa.stats());
    let mut final_dwa = combined_nwa.determinize_to_dwa_with_rustfst();
    crate::debug!(
        4,
        "Final determinize & simplify took: {:?}. Final DWA has {} states.",
        now.elapsed(),
        final_dwa.states.len()
    );
    crate::debug!(4, "Stats for final DWA:\n{}", final_dwa.stats());

    final_dwa
}

fn instantiate_template_nwa_with_weight(
    states_arena: &RefCell<NWAStates>,
    template_dwa: &DWA,
    weight: Weight,
) -> NWABody {
    crate::debug!(5, "Applying template NWA with weight {:?}...", weight);
    let concatenated_dwa = template_dwa.concatenate(&build_epsilon_dwa(weight));
    let template_nwa = NWA::from_dwa(&concatenated_dwa);
    let mut states = states_arena.borrow_mut();
    let (template_start_in_arena, _) = states.copy_subgraph_from(&template_nwa.states, template_nwa.body.start_state);
    crate::debug!(5, "Template NWA copied into arena. Current arena size: {} states.", states.0.len());
    NWABody { start_state: template_start_in_arena }
}

fn union_left_bodies(states_arena: &RefCell<NWAStates>, left_bodies: Vec<NWABody>) -> NWABody {
    let mut queue: VecDeque<NWA> = VecDeque::with_capacity(left_bodies.len());
    {
        let arena = states_arena.borrow();
        for left_body in left_bodies {
            let mut states = NWAStates::default();
            let start = states.copy_subgraph_from_and_return_body(&arena, left_body).start_state;
            queue.push_back(NWA { states, body: NWABody { start_state: start } });
        }
    }

    if queue.len() == 1 {
        simplify_and_determinize_nwa(queue.front_mut().unwrap());
    }

    while queue.len() > 1 {
        let nwa1 = queue.pop_front().unwrap();
        let nwa2 = queue.pop_front().unwrap();
        let mut states = nwa1.states;
        let (start2, _) = states.copy_subgraph_from(&nwa2.states, nwa2.body.start_state);
        let union_body = NWA::union_components(&mut states, &nwa1.body, &NWABody { start_state: start2 });
        let mut res = NWA { states, body: union_body };
        simplify_and_determinize_nwa(&mut res);
        queue.push_back(res);
    }

    let NWA { states: states2, body: left_bodies_union2 } = queue.pop_front().unwrap();
    let mut states = states_arena.borrow_mut();
    states.copy_subgraph_from_and_return_body(&states2, left_bodies_union2)
}

fn simplify_and_determinize_nwa(nwa: &mut NWA) {
    nwa.simplify_rustfst_with_config(
        SimplifyRustfstConfig::default()
            .with_rm_epsilon(true)
            .with_determinize(true),
    );
}

fn simplify_remove_epsilon(nwa: &mut NWA) {
    nwa.simplify_rustfst_with_config(SimplifyRustfstConfig::default().with_rm_epsilon(true));
}
