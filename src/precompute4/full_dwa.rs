use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::env;
use std::time::Instant;

use chrono::Local;
use rustfst::Label;
use crate::constraint::{LLMTokenBV, PrecomputeNode1, PrecomputeNode1Index, PrecomputedNodeContents, Trie1GodWrapper};
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::json_serialization::JSONConvertible;
use crate::precompute4::nwa_optimizations::{prune_continuations_from_final_states, simplify_default_transitions};
use crate::precompute4::resolve_negatives::{apply_cancellations, apply_finality_fixpoint, remove_negative_transitions};
use crate::precompute4::template_nwa::{build_epsilon_dwa, build_ignore_terminal_dwa, build_template_dwas};
use crate::precompute4::weighted_automata::{DWA, NWA, NWABody, NWAStateID, NWAStates, Weight, StateID};
use crate::r#macro::is_debug_level_enabled;
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
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
    pub fn determinize_to_dwa_with_rustfst(&self) -> DWA {
        determinize_nwa_to_dwa(self)
    }
    pub fn simplify_rustfst(&mut self) { self.simplify(); }
    pub fn simplify_rustfst_with_config(&mut self, _config: SimplifyRustfstConfig) { self.simplify(); }
}

// Re-export for backward compatibility: `FullDWABuildError` used to be defined here.
pub use crate::precompute4::template_nwa::FullDWABuildError;
use crate::precompute4::weighted_automata::determinization_rustfst::determinize_nwa_to_dwa;

pub type Precomputed4 = DWA;

fn convert_node_to_nwa(
    node_idx: PrecomputeNode1Index,
    god: &Trie1GodWrapper,
    nwa: &mut NWA,
    cache: &mut HashMap<PrecomputeNode1Index, StateID>,
) -> StateID {
    if let Some(&sid) = cache.get(&node_idx) {
        return sid;
    }

    let sid = nwa.add_state();
    cache.insert(node_idx, sid);

    let guard = node_idx.read(god).unwrap();

    // Map live_tokens to final_weight (representing valid tokens at this state)
    if guard.value.end {
        nwa.states[sid].final_weight = Some(Weight::all());
    }

    let children = guard.children().clone();
    drop(guard);

    for (edge_key, child_map) in children {
        for (child_idx, edge_bv) in child_map {
            let child_sid = convert_node_to_nwa(child_idx, god, nwa, cache);

            let trans_w: Weight = edge_bv.into();

            // Add transition (NWA allows multiple transitions for same label)
            if let Some(label) = edge_key {
                nwa.add_transition(sid, label.0 as Label, child_sid, trans_w).unwrap();
            } else {
                nwa.add_epsilon(sid, child_sid, trans_w);
            }
        }
    }
    sid
}

fn convert_precompute1_to_nwa(
    precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) -> NWA {
    let mut nwa = NWA::new();
    nwa.states.0.clear(); // Clear default start state
    let start_state = nwa.states.add_state();
    nwa.body.start_state = start_state;

    let mut node_cache = HashMap::new();

    for (sid, root_idx) in precomputed1 {
        let root_state = convert_node_to_nwa(*root_idx, trie1_god, &mut nwa, &mut node_cache);
        nwa.add_transition(start_state, sid.0 as Label, root_state, Weight::all()).unwrap();
    }
    nwa
}

fn convert_dwa_to_precompute1(
    dwa: &DWA,
    max_llm_token_id: usize,
) -> (BTreeMap<TokenizerStateID, PrecomputeNode1Index>, Trie1GodWrapper) {
    let god = Trie1GodWrapper::new();
    let mut result = BTreeMap::new();
    let mut state_cache = HashMap::new();
    let end_node_contents = PrecomputedNodeContents { end: true, live_tokens: LLMTokenBV::ones(max_llm_token_id + 1) };
    let end_node = PrecomputeNode1::new(end_node_contents);
    let end_node_idx = PrecomputeNode1Index::new(god.insert(end_node));

    let start_state = dwa.body.start_state;
    if start_state >= dwa.states.len() {
        return (result, god);
    }

    let start_node = &dwa.states[start_state];
    for (label, target) in &start_node.transitions {
        let sid = TokenizerStateID(*label as usize);
        let root_idx = convert_dwa_state_to_trie_node(*target, dwa, &god, &mut state_cache, max_llm_token_id, end_node_idx);

        let weight = start_node.trans_weights.get(label).cloned().unwrap_or_else(Weight::all);
        let edge_bv: LLMTokenBV = LLMTokenBV::from(weight) & LLMTokenBV::ones(max_llm_token_id + 1);

        let contents = PrecomputedNodeContents { end: false, live_tokens: edge_bv.clone() };
        let wrapper_node = PrecomputeNode1::new(contents);
        let wrapper_idx = PrecomputeNode1Index::new(god.insert(wrapper_node));
        god.insert_edge_simple(wrapper_idx, root_idx, None, edge_bv);
        result.insert(sid, wrapper_idx);
    }

    (result, god)
}

fn convert_dwa_state_to_trie_node(
    state_id: StateID,
    dwa: &DWA,
    god: &Trie1GodWrapper,
    cache: &mut HashMap<StateID, PrecomputeNode1Index>,
    max_llm_token_id: usize,
    end_node_idx: PrecomputeNode1Index,
) -> PrecomputeNode1Index {
    if let Some(&idx) = cache.get(&state_id) {
        return idx;
    }


    let contents = PrecomputedNodeContents { end: false, live_tokens: LLMTokenBV::ones(max_llm_token_id + 1) };
    let node = PrecomputeNode1::new(contents);
    let idx = PrecomputeNode1Index::new(god.insert(node));
    cache.insert(state_id, idx);

    let state = &dwa.states[state_id];
    if let Some(fw) = &state.final_weight {
        god.insert_edge_simple(idx, end_node_idx, None, fw.clone().into());
    }

    for (label, target) in &state.transitions {
        let target_idx = convert_dwa_state_to_trie_node(*target, dwa, god, cache, max_llm_token_id, end_node_idx);
        let weight = state.trans_weights.get(label).cloned().unwrap_or_else(Weight::all);
        let edge_bv: LLMTokenBV = LLMTokenBV::from(weight) & LLMTokenBV::ones(max_llm_token_id + 1);
        let term_id = GrammarTokenID(*label as usize);
        god.insert_edge_simple(idx, target_idx, Some(term_id), edge_bv);
    }

    idx
}

/// Public API: precompute4 using an NWA-first approach, determinizing at the end.
pub fn precompute4(
    parser: &GLRParser,
    precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    max_llm_token_id: usize,
) -> DWA {
    crate::debug!(4, "Optimizing precomputed1 via NWA/DWA conversion...");
    let mut nwa = convert_precompute1_to_nwa(precomputed1, trie1_god);
    crate::debug!(5, "Optimizing precomputed1 via NWA/DWA conversion... done.");
    nwa.simplify();
    crate::debug!(5, "Simplified precomputed1 NWA... done.");
    let mut dwa = nwa.determinize();
    crate::debug!(5, "Determinized precomputed1 NWA... done.");
    dwa.minimize_with_rustfst();

    crate::debug!(4, "Unrolling cycles in precomputed1 DWA...");
    let mut unrolled = dwa.unroll_cycles();
    unrolled.minimize_with_rustfst();
    dwa = unrolled;

    crate::debug!(
        5,
        "Optimized precomputed1 DWA has {} states and {} transitions.",
        dwa.states.len(),
        dwa.num_transitions(),
    );

    let (optimized_precomputed1, optimized_trie1_god) = convert_dwa_to_precompute1(&dwa, max_llm_token_id);
    let precomputed1 = &optimized_precomputed1;
    let trie1_god = &optimized_trie1_god;

    let now_total = Instant::now();
    let now = Instant::now();
    crate::debug!(5, "Starting precompute4...");

    // 1. Build template DWAs for all terminals.
    let template_dwas = match build_template_dwas(parser) {
        Ok(m) => m,
        Err(e) => panic!("Failed to build template DWAs: {:?}", e),
    };
    let ignore_dwa = build_ignore_terminal_dwa();
    crate::debug!(4, "Built {} template DWAs in {:?}", template_dwas.len(), now.elapsed());
    if is_debug_level_enabled(5) {
        for (term, dwa) in template_dwas.iter().take(5) {
            crate::debug!(5, "Stats for template DWA for terminal {:?}:\n{}", term, dwa.stats());
        }
    }

    // Build a "super DWA" that contains all templates, distinguished by weights.
    let mut term_to_bit = BTreeMap::new();
    let mut bit_to_term: Vec<Option<TerminalID>> = Vec::new();

    let mut all_terminals: BTreeSet<TerminalID> = template_dwas.keys().cloned().collect();
    if let Some(ignore_term) = parser.ignore_terminal_id {
        all_terminals.insert(ignore_term);
    }

    term_to_bit.insert(None, 0);
    bit_to_term.push(None);
    for (i, term_id) in all_terminals.iter().enumerate() {
        term_to_bit.insert(Some(*term_id), i + 1);
        bit_to_term.push(Some(*term_id));
    }

    let now_super_dwa = Instant::now();
    let mut super_nwa_states = NWAStates::default();
    let super_nwa_start = super_nwa_states.add_state();

    for (term_id_opt, bit) in &term_to_bit {
        let mut weight = Weight::zeros();
        weight.set(*bit, true);

        let template_dwa = match term_id_opt {
            Some(term_id) if Some(*term_id) != parser.ignore_terminal_id => template_dwas.get(term_id).unwrap(),
            _ => &ignore_dwa,
        };

        let mut weighted_dwa = template_dwa.clone();
        weighted_dwa.apply_weight_inplace(&weight);

        let nwa = NWA::from_dwa(&weighted_dwa);
        let (start, _) = super_nwa_states.copy_subgraph_from(&nwa.states, nwa.body.start_state);
        super_nwa_states.add_epsilon(super_nwa_start, start, Weight::all());
    }

    let mut super_nwa = NWA { states: super_nwa_states, body: NWABody { start_state: super_nwa_start } };
    super_nwa.simplify();
    let mut super_dwa = super_nwa.determinize_to_dwa();
    super_dwa.simplify();
    crate::debug!(
        4,
        "Built super DWA with {} states in {:?}",
        super_dwa.states.len(),
        now_super_dwa.elapsed()
    );

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

            let mut now_step = Instant::now();
            for (right_body, terminal_map) in nwa_bodies_map {
                let mut effective_terminal_map = BTreeMap::new();
                for (terminal_id_opt, weight) in terminal_map {
                    effective_terminal_map.insert(terminal_id_opt, weight);
                }

                if effective_terminal_map.is_empty() {
                    continue;
                }

                let mut left_dwa = specialize_dwa(&super_dwa, &effective_terminal_map, &bit_to_term);
                left_dwa.simplify();
                let left_nwa = NWA::from_dwa(&left_dwa);

                let mut states = states_arena.borrow_mut();
                let (left_body_start, remap) =
                    states.copy_subgraph_from(&left_nwa.states, left_nwa.body.start_state);

                let new_states_filter: HashSet<NWAStateID> = remap.values().cloned().collect();

                let left_body = NWABody { start_state: left_body_start };

                let composed_body = NWA::_concatenate_components(&mut states, &left_body, &right_body, &Weight::all());

                if !new_states_filter.is_empty() {
                    apply_cancellations(&mut states, &new_states_filter);
                    apply_finality_fixpoint(&mut states, &new_states_filter);
                    remove_negative_transitions(&mut states, &new_states_filter);
                }

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
        let label = tok_id.0 as Label;
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
    let now = Instant::now();
    crate::debug!(4, "Starting resolve negatives and optimization and determinization of combined NWA...");
    combined_nwa.simplify_rustfst();
    crate::debug!(4, "Initial simplification took: {:?}. NWA now has {} states.", now.elapsed(), combined_nwa.states.len());

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
    combined_nwa = NWA::from_dwa(&combined_nwa._determinize());
    crate::debug!(4, "Stats after final NWA determinization:\n{}", combined_nwa.stats());
    combined_nwa.simplify_rustfst();
    crate::debug!(
        4,
        "Final NWA simplification took: {:?}. NWA now has {} states.",
        now.elapsed(),
        combined_nwa.states.len()
    );
    crate::debug!(4, "Stats for final NWA before DWA determinization:\n{}", combined_nwa.stats());
    let mut final_dwa = combined_nwa.determinize_to_dwa();
    crate::debug!(
        4,
        "Final determinize & simplify took: {:?}. Final DWA has {} states.",
        now.elapsed(),
        final_dwa.states.len()
    );
    crate::debug!(4, "Stats for final DWA:\n{}", final_dwa.stats());

    final_dwa
}

fn specialize_dwa(
    super_dwa: &DWA,
    bundle: &BTreeMap<Option<TerminalID>, Weight>,
    bit_to_term: &[Option<TerminalID>],
) -> DWA {
    let mut specialized_dwa = super_dwa.clone();
    let mut weight_cache: HashMap<Weight, Weight> = HashMap::new();

    for state in &mut specialized_dwa.states.0 {
        if let Some(fw) = &mut state.final_weight {
            *fw = specialize_weight(fw, bundle, bit_to_term, &mut weight_cache);
            if fw.is_empty() {
                state.final_weight = None;
            }
        }
        if let Some(sw) = &mut state.state_weight {
            *sw = specialize_weight(sw, bundle, bit_to_term, &mut weight_cache);
            if sw.is_empty() {
                state.state_weight = None;
            }
        }
        for tw in state.trans_weights.values_mut() {
            *tw = specialize_weight(tw, bundle, bit_to_term, &mut weight_cache);
        }
        state.trans_weights.retain(|_, w| !w.is_empty());
        state.transitions.retain(|k, _| state.trans_weights.contains_key(k));
    }
    specialized_dwa
}

fn specialize_weight(
    weight: &Weight,
    bundle: &BTreeMap<Option<TerminalID>, Weight>,
    bit_to_term: &[Option<TerminalID>],
    cache: &mut HashMap<Weight, Weight>,
) -> Weight {
    if let Some(cached) = cache.get(weight) {
        return cached.clone();
    }

    let mut new_weight = Weight::zeros();
    for bit_idx in weight.iter_up_to(bit_to_term.len()) {
        if let Some(term_id_opt) = bit_to_term.get(bit_idx) {
            if let Some(bundle_weight) = bundle.get(term_id_opt) {
                new_weight |= bundle_weight;
            }
        }
    }

    cache.insert(weight.clone(), new_weight.clone());
    new_weight
}

fn simplify_remove_epsilon(nwa: &mut NWA) {
    nwa.simplify_rustfst_with_config(SimplifyRustfstConfig::default().with_rm_epsilon(true));
}
