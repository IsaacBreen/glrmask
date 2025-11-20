use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::env;
use std::time::Instant;
use std::sync::{Arc, Mutex};

use chrono::Local;
use crate::constraint::{LLMTokenBV, PrecomputeNode1, PrecomputeNode1Index, PrecomputedNodeContents, Trie1GodWrapper};
use crate::datastructures::trie::{Trie, Trie2Index};
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::json_serialization::JSONConvertible;
use crate::precompute4::nwa_optimizations::{prune_continuations_from_final_states, simplify_default_transitions};
use crate::precompute4::resolve_negatives::{apply_cancellations, apply_finality_fixpoint, remove_negative_transitions};
use crate::precompute4::template_nwa::{build_epsilon_dwa, build_ignore_terminal_dwa, build_template_dwas};
use crate::precompute4::weighted_automata::{DWA, NWA, NWABody, NWAStateID, NWAStates, Weight, StateID, SimpleBitset};
use crate::r#macro::is_debug_level_enabled;
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::tokenizer::TokenizerStateID;
use crate::precompute4::weighted_automata::common::Label;
use crate::precompute4::weighted_automata::determinization_rustfst::determinize_nwa_to_dwa;

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

// Re-export for backward compatibility
pub use crate::precompute4::template_nwa::FullDWABuildError;

pub type Precomputed4 = DWA;
type Signature = Vec<Vec<Option<TerminalID>>>;

/// Helper to index a signature for fast compatibility checks.
struct SignatureIndex {
    term_to_group: HashMap<Option<TerminalID>, usize>,
    total_terms: usize,
}

impl SignatureIndex {
    fn new(sig: &Signature) -> Self {
        let mut map = HashMap::new();
        let mut count = 0;
        for (g_idx, group) in sig.iter().enumerate() {
            for term in group {
                map.insert(*term, g_idx);
                count += 1;
            }
        }
        Self { term_to_group: map, total_terms: count }
    }

    fn get_group(&self, term: &Option<TerminalID>) -> Option<usize> {
        self.term_to_group.get(term).cloned()
    }
}

/// Checks if `parent` signature can derive `child`, returning the weight mapping if so.
/// Mapping[i] = Weight to replace bit `i` with.
fn can_derive(
    parent: &Signature,
    child_index: &SignatureIndex,
) -> Option<Vec<Weight>> {
    let mut mapping = Vec::with_capacity(parent.len());
    let mut matched_terms = 0;

    for group in parent {
        if group.is_empty() {
            mapping.push(Weight::zeros());
            continue;
        }

        let first_term = &group[0];
        let expected_g = child_index.get_group(first_term);

        for term in &group[1..] {
            let g = child_index.get_group(term);
            if g != expected_g {
                return None; // Parent group splits across child groups or mixing present/missing
            }
        }

        if let Some(g) = expected_g {
            mapping.push(Weight::from_item(g));
            matched_terms += group.len();
        } else {
            mapping.push(Weight::zeros());
        }
    }

    // Ensure parent covers all terminals in child
    if matched_terms != child_index.total_terms {
        return None;
    }

    Some(mapping)
}

/// Creates a new DWA from a parent DWA by remapping its abstract weights according to `mapping`.
fn specialize_dwa_relative(
    parent_dwa: &DWA,
    mapping: &[Weight],
) -> DWA {
    let mut specialized_dwa = parent_dwa.clone();
    let mut cache: HashMap<Weight, SimpleBitset> = HashMap::new();

    // Helper to map a bitset of parent indices to a bitset of child indices
    let mut map_weight = |w: &Weight| -> Weight {
        if let Some(cw) = cache.get(w) {
            return cw.clone();
        }
        let mut new_w = Weight::zeros();
        for bit in w.iter_up_to(mapping.len()) {
            if let Some(target_w) = mapping.get(bit) {
                new_w |= target_w;
            }
        }
        cache.insert(w.clone(), new_w.clone());
        new_w
    };

    for state in &mut specialized_dwa.states.0 {
        if let Some(fw) = &mut state.final_weight {
            *fw = map_weight(fw);
            if fw.is_empty() { state.final_weight = None; }
        }
        if let Some(sw) = &mut state.state_weight {
            *sw = map_weight(sw);
            if sw.is_empty() { state.state_weight = None; }
        }
        for tw in state.trans_weights.values_mut() {
            *tw = map_weight(tw);
        }
        // Prune now-empty transitions
        state.trans_weights.retain(|_, w| !w.is_empty());
        state.transitions.retain(|k, _| state.trans_weights.contains_key(k));
    }

    specialized_dwa
}

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

    // Traversal Data
    let traversal_data =
        Trie::compute_traversal_data(&reversed_trie1_god, &[reversed_trie_root]).expect("Failed to compute traversal data for reversed trie1");

    let initial_tokens = LLMTokenBV::max_ones();
    let initial_values_bv: Vec<(Trie2Index, LLMTokenBV)> = vec![(reversed_trie_root, initial_tokens.clone())];

    // Pass 1: Compute Token Bitsets and Collect Signatures
    let start_pass1 = Instant::now();
    let (node_tokens, unique_signatures) = precompute_token_bvs_and_signatures(
        &reversed_trie1_god,
        &traversal_data,
        initial_values_bv,
    );
    crate::debug!(4, "Pass 1 (Token BVs & Signatures) took: {:?}. Found {} unique signatures.", start_pass1.elapsed(), unique_signatures.len());

    // Collect all terminals actually used in signatures (some might be uncharacterized in grammar but present in trie)
    let mut used_terminals: BTreeSet<TerminalID> = BTreeSet::new();
    for sig in &unique_signatures {
        for group in sig {
            for term_opt in group {
                if let Some(term) = term_opt {
                    used_terminals.insert(*term);
                }
            }
        }
    }

    // Build a "super DWA" that contains all templates, distinguished by weights.
    let mut term_to_bit = BTreeMap::new();
    let mut bit_to_term: Vec<Option<TerminalID>> = Vec::new();

    let mut all_terminals: BTreeSet<TerminalID> = template_dwas.keys().cloned().collect();
    if let Some(ignore_term) = parser.ignore_terminal_id {
        all_terminals.insert(ignore_term);
    }
    all_terminals.extend(used_terminals);

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
            Some(term_id) => {
                 if Some(*term_id) == parser.ignore_terminal_id {
                     &ignore_dwa
                 } else {
                     template_dwas.get(term_id).unwrap_or(&ignore_dwa)
                 }
            },
            None => &ignore_dwa,
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

    // Precompute Templates via waterfall derivation
    let start_templates = Instant::now();
    let mut template_cache = HashMap::new();

    // 1. Construct explicit super signature and put super_dwa in the pool.
    // Super signature maps index i -> {bit_to_term[i]}.
    let super_signature: Signature = bit_to_term.iter().map(|t| vec![*t]).collect();
    let mut pool: Vec<(Signature, DWA)> = Vec::new();
    pool.push((super_signature, super_dwa.clone()));

    // 2. Sort target signatures by complexity (groups, terminals) descending.
    // This ensures we build larger/more-complex templates first, which can then serve as parents for smaller ones.
    let mut signatures_vec: Vec<Signature> = unique_signatures.into_iter().collect();
    signatures_vec.sort_by(|a, b| {
        // Heuristic: Larger number of groups ~ more granularity.
        let groups_a = a.len();
        let groups_b = b.len();
        if groups_a != groups_b {
            return groups_b.cmp(&groups_a);
        }
        // Tie-break: number of terminals
        let terms_a: usize = a.iter().map(|g| g.len()).sum();
        let terms_b: usize = b.iter().map(|g| g.len()).sum();
        terms_b.cmp(&terms_a)
    });

    // 3. Greedy derivation
    for target_sig in signatures_vec {
        let target_idx = SignatureIndex::new(&target_sig);

        // Find best parent in pool
        let mut best_parent: Option<(usize, Vec<Weight>)> = None; // (pool_index, mapping)
        let mut best_score = usize::MAX; // Minimize parent groups

        for (p_idx, (p_sig, _)) in pool.iter().enumerate() {
            // Strict check: Parent must derive child exactly (merge only, no splitting/guessing)
            if let Some(mapping) = can_derive(p_sig, &target_idx) {
                let score = p_sig.len();
                if score < best_score {
                    best_score = score;
                    best_parent = Some((p_idx, mapping));
                }
            }
        }

        let (parent_idx, mapping) = best_parent.expect("Super signature should always be a valid parent");
        let parent_dwa = &pool[parent_idx].1;

        // Derive
        let mut derived_dwa = specialize_dwa_relative(parent_dwa, &mapping);
        derived_dwa.simplify(); // Crucial step: reduce the DWA for the pool

        template_cache.insert(target_sig.clone(), NWA::from_dwa(&derived_dwa));
        pool.push((target_sig, derived_dwa));
    }

    crate::debug!(4, "Precomputed {} templates in: {:?}", template_cache.len(), start_templates.elapsed());


    // Pass 2: Build NWAs
    let initial_nwa_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_state: start }
    };

    let initial_term_map: BTreeMap<Option<TerminalID>, Weight> = BTreeMap::from([(None, Weight::all())]);
    let initial_body_map_full = BTreeMap::from([(initial_nwa_body, initial_term_map)]);
    let initial_values_full: Vec<(Trie2Index, (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV))> =
        vec![(reversed_trie_root, (initial_body_map_full, initial_tokens))];

    let mut original_trie1_roots_map: BTreeMap<PrecomputeNode1Index, Vec<TokenizerStateID>> = BTreeMap::new();
    for (k, v) in precomputed1.iter() {
        original_trie1_roots_map.entry(*v).or_default().push(*k);
    }

    let mut final_bodies: BTreeMap<TokenizerStateID, NWABody> = BTreeMap::new();

    let now_traversal = Instant::now();

    Trie::special_map_grouped(
        &reversed_trie1_god,
        &traversal_data,
        initial_values_full,
        // step
        |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV), edge_terminal_opt, dest_map| {
             let (current_bodies, current_tokens) = current_val;
             let terminal_id = *edge_terminal_opt;
             let mut results = Vec::new();
             for (dest_idx, llm_token_bv) in dest_map.iter() {
                 let next_tokens = current_tokens & llm_token_bv;
                 if next_tokens.is_empty() { continue; }
                 let weight = Weight::from_rsb(llm_token_bv.inner.as_ref().clone());
                 let mut terminal_map = BTreeMap::new();
                 terminal_map.insert(terminal_id, weight);
                 let mut body_map = BTreeMap::new();
                 for body in current_bodies.keys() {
                     body_map.insert(*body, terminal_map.clone());
                 }
                 results.push((*dest_idx, (body_map, next_tokens)));
             }
             results
        },
        // merge
        |val1, val2| {
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
        // process
        |_, node_idx, val| {
            let (nwa_bodies_map, tokens) = val;
            let mut nwa_body = {
                let mut states = states_arena.borrow_mut();
                let start = states.add_state();
                NWABody { start_state: start }
            };

            for (right_body, terminal_map) in nwa_bodies_map {
                // 1. Canonicalize Bundle
                let (signature, concrete_weights) = canonicalize_bundle(terminal_map);

                // 2. Lookup Template
                let template_nwa = template_cache.get(&signature).expect("Template must exist for precomputed signature");

                // 3. Instantiate
                let mut states = states_arena.borrow_mut();
                let (left_body_start, remap) = instantiate_nwa_template_into_arena(
                    template_nwa,
                    &concrete_weights,
                    &mut states
                );

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

            if !tokens.is_empty() {
                if let Some(tokenizer_state_ids) = original_trie1_roots_map.get(&node_idx) {
                    for tokenizer_state_id in tokenizer_state_ids {
                        final_bodies.insert(*tokenizer_state_id, nwa_body.clone());
                    }
                }

                let mut next_body_map = BTreeMap::new();
                next_body_map.insert(nwa_body, BTreeMap::new());
                Some((next_body_map, tokens))
            } else {
                None
            }
        }
    );

    println!("=== Traversal Statistics (Pass 2) ===");
    println!("Total Duration: {:?}", now_traversal.elapsed());
    println!("=====================================");

    // Combine final bodies...
    let mut combined_nwa_states = states_arena.into_inner();
    let combined_start_state = combined_nwa_states.add_state();
    for (tok_id, body) in final_bodies {
        let label = tok_id.0 as Label;
        combined_nwa_states
            .add_transition(combined_start_state, label, body.start_state, Weight::all())
            .unwrap();
    }
    let combined_nwa = NWA { states: combined_nwa_states, body: NWABody { start_state: combined_start_state } };

    let final_dwa = resolve_negatives_and_optimize_and_determinize(parser, combined_nwa);
    crate::debug!(3, "Total precompute4 time: {:?}", now_total.elapsed());
    final_dwa
}

fn canonicalize_bundle(
    terminal_map: BTreeMap<Option<TerminalID>, Weight>
) -> (Signature, Vec<Weight>) {
    let mut weight_groups: HashMap<Weight, Vec<Option<TerminalID>>> = HashMap::new();
    for (term, weight) in terminal_map {
        if !weight.is_empty() {
            weight_groups.entry(weight).or_default().push(term);
        }
    }
    let mut groups_vec: Vec<(Weight, Vec<Option<TerminalID>>)> = weight_groups.into_iter().collect();
    for (_, terms) in &mut groups_vec {
        terms.sort();
    }
    groups_vec.sort_by(|a, b| a.1.cmp(&b.1));

    let signature: Vec<Vec<Option<TerminalID>>> = groups_vec.iter().map(|(_, terms)| terms.clone()).collect();
    let concrete_weights: Vec<Weight> = groups_vec.into_iter().map(|(w, _)| w).collect();
    (signature, concrete_weights)
}

fn precompute_token_bvs_and_signatures(
    reversed_trie: &crate::datastructures::trie::Arena<Trie<Option<TerminalID>, LLMTokenBV, PrecomputedNodeContents>>,
    traversal_data: &crate::datastructures::trie::TrieTraversalData,
    initial_values: Vec<(Trie2Index, LLMTokenBV)>
) -> (HashMap<Trie2Index, LLMTokenBV>, HashSet<Signature>) {

    let node_tokens: Arc<Mutex<HashMap<Trie2Index, LLMTokenBV>>> = Arc::new(Mutex::new(HashMap::new()));
    let signatures: Arc<Mutex<HashSet<Signature>>> = Arc::new(Mutex::new(HashSet::new()));

    // We use the trie traversal to propagate tokens AND collect signatures on the fly.
    // Value type V = LLMTokenBV.

    Trie::special_map_grouped(
        reversed_trie,
        traversal_data,
        initial_values,
        // step: propagate tokens
        |tokens: &LLMTokenBV, _edge_term, dest_map| {
            let mut results = Vec::new();
            for (dest_idx, edge_bv) in dest_map.iter() {
                let next = tokens & edge_bv;
                if !next.is_empty() {
                    results.push((*dest_idx, next));
                }
            }
            results
        },
        // merge: union
        |t1, t2| { *t1 |= &t2; },
        // process: capture signatures
        |node_guard, node_idx, tokens| {
            node_tokens.lock().unwrap().insert(node_idx, tokens.clone());

            // To form signatures for the NEXT step (outgoing edges from here),
            // we need to look at the children of the current node in the reversed trie.
            // In `reversed_trie`, children map Key -> Dest -> Weight.
            // We group by Dest.

            let mut bundles_by_dest: HashMap<Trie2Index, BTreeMap<Option<TerminalID>, Weight>> = HashMap::new();

            for (term_opt, dest_map) in node_guard.children() {
                for (dest_idx, edge_bv) in dest_map.iter() {
                     let combined = &tokens & edge_bv;
                     if !combined.is_empty() {
                         let w = Weight::from_rsb(edge_bv.inner.as_ref().clone());
                         bundles_by_dest.entry(*dest_idx).or_default().insert(term_opt.clone(), w);
                     }
                }
            }

            let mut sigs = signatures.lock().unwrap();
            for (_, bundle) in bundles_by_dest {
                 let (sig, _) = canonicalize_bundle(bundle);
                 sigs.insert(sig);
            }

            Some(tokens)
        }
    );

    let final_tokens = Arc::try_unwrap(node_tokens).unwrap().into_inner().unwrap();
    let final_sigs = Arc::try_unwrap(signatures).unwrap().into_inner().unwrap();
    (final_tokens, final_sigs)
}

fn resolve_negatives_and_optimize_and_determinize(parser: &GLRParser, mut combined_nwa: NWA) -> DWA {
    println!("=== Post-Processing Statistics ===");
    let start_total = Instant::now();

    let start = Instant::now();
    combined_nwa.simplify_rustfst();
    let t_initial = start.elapsed();
    println!("Initial Simplify: {:?}", t_initial);
    crate::debug!(4, "Initial simplification took: {:?}. NWA now has {} states.", t_initial, combined_nwa.states.len());

    let start = Instant::now();
    prune_continuations_from_final_states(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    let t_prune = start.elapsed();
    println!("Prune Continuations: {:?}", t_prune);
    crate::debug!(4, "Pruning took: {:?}. NWA now has {} states.", t_prune, combined_nwa.states.len());

    let start = Instant::now();
    simplify_default_transitions(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    let t_defaults = start.elapsed();
    println!("Simplify Defaults: {:?}", t_defaults);
    crate::debug!(4, "Simplify defaults took: {:?}. NWA now has {} states.", t_defaults, combined_nwa.states.len());

    let start = Instant::now();
    simplify_remove_epsilon(&mut combined_nwa);
    combined_nwa.simplify();
    simplify_remove_epsilon(&mut combined_nwa);
    let t_pre_det = start.elapsed();
    println!("Pre-Determinization Simplify: {:?}", t_pre_det);
    crate::debug!(4, "Pre-det simplify took: {:?}. NWA now has {} states.", t_pre_det, combined_nwa.states.len());

    if env::var("RLLM_DUMP_NWA").is_ok() {
        let timestamp = Local::now().format("%Y%m%d-%H%M%S");
        let filename = format!("nwa_dump_before_final_det_{}.json", timestamp);
        eprintln!("Dumping NWA to {} before final determinization...", filename);
        let f = std::fs::File::create(&filename).expect("Unable to create NWA dump file");
        serde_json::to_writer_pretty(f, &combined_nwa).expect("Unable to write NWA to file");
        eprintln!("NWA dump complete.");
    }

    let start = Instant::now();
    crate::debug!(4, "Determinizing final combined NWA...");
    combined_nwa = NWA::from_dwa(&combined_nwa._determinize());
    combined_nwa.simplify_rustfst();
    let mut final_dwa = combined_nwa.determinize_to_dwa();
    final_dwa.minimize_with_rustfst();
    let t_det = start.elapsed();
    println!("Final Determinize & Simplify: {:?}", t_det);
    crate::debug!(4, "Final determinize took: {:?}. Final DWA has {} states.", t_det, final_dwa.states.len());

    println!("Total Post-Processing: {:?}", start_total.elapsed());
    println!("================================");

    final_dwa
}

fn instantiate_nwa_template_into_arena(
    template: &NWA,
    ordered_weights: &[Weight],
    arena: &mut NWAStates,
) -> (NWAStateID, HashMap<NWAStateID, NWAStateID>) {
    let mut union_cache: HashMap<Weight, Weight> = HashMap::new();

    let mut map_abstract_weight = |w: &Weight| -> Weight {
        if w.is_empty() {
            return Weight::zeros();
        }
        if let Some(res) = union_cache.get(w) {
            return res.clone();
        }
        let mut concrete = Weight::zeros();
        for idx in w.iter_up_to(ordered_weights.len()) {
            if let Some(concrete_w) = ordered_weights.get(idx) {
                concrete |= concrete_w;
            }
        }
        union_cache.insert(w.clone(), concrete.clone());
        concrete
    };

    let start_offset = arena.len();
    let template_len = template.states.len();
    let mut map: Vec<NWAStateID> = Vec::with_capacity(template_len);

    for _ in 0..template_len {
        map.push(arena.add_state());
    }

    let mut id_map = HashMap::with_capacity(template_len);
    for (i, &new_id) in map.iter().enumerate() {
        id_map.insert(i, new_id);
    }

    for (old_id, old_state) in template.states.0.iter().enumerate() {
        let new_id = map[old_id];
        let new_state = &mut arena[new_id];

        if let Some(fw) = &old_state.final_weight {
            let concrete = map_abstract_weight(fw);
            if !concrete.is_empty() {
                new_state.final_weight = Some(concrete);
            }
        }

        for (lbl, targets) in &old_state.transitions {
            let new_targets = new_state.transitions.entry(*lbl).or_default();
            for (target, w) in targets {
                let concrete = map_abstract_weight(w);
                if !concrete.is_empty() {
                    new_targets.push((map[*target], concrete));
                }
            }
        }

        for (target, w) in &old_state.epsilons {
            let concrete = map_abstract_weight(w);
            if !concrete.is_empty() {
                new_state.epsilons.push((map[*target], concrete));
            }
        }
    }

    (map[template.body.start_state], id_map)
}

fn simplify_remove_epsilon(nwa: &mut NWA) {
    nwa.simplify_rustfst_with_config(SimplifyRustfstConfig::default().with_rm_epsilon(true));
}