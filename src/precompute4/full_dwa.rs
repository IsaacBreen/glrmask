use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rayon::prelude::*;
use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};



use crate::constraint::LLMTokenBV;
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::precompute4::nwa_optimizations::prune_continuations_from_final_states;
use crate::precompute4::resolve_negatives::{
    apply_cancellations_range, apply_finality_fixpoint_range, remove_negative_transitions_range
    // Note: remove_redundant_default_transitions is called once at the end in finalize_and_optimize_and_determinize,
    // not per-range here, since it requires a global pass over all states.
};
use crate::precompute4::template_nwa::{build_ignore_terminal_dwa, build_template_dwas};
use crate::precompute4::weighted_automata::{
    common::Label, determinization_rustfst::determinize_nwa_to_dwa, DWA, NWA, NWABody, NWAStateID, NWAStates,
    StateID, Weight,
};
use crate::tokenizer::TokenizerStateID;
use crate::types::{TerminalID, TerminalID as GrammarTokenID};

struct MinimizeRustfstConfig {
    rm_epsilon: bool,
    #[allow(dead_code)]
    determinize: bool,
}

impl MinimizeRustfstConfig {
    fn default() -> Self { Self { rm_epsilon: false, determinize: false } }
    fn with_rm_epsilon(mut self, val: bool) -> Self { self.rm_epsilon = val; self }
}

pub use crate::precompute4::template_nwa::FullDWABuildError;

/// The Parser DWA - the final precomputed artifact used for get_mask queries.
/// This is a deterministic weighted automaton where weights are sparse bitvectors
/// over LLM token equivalence classes.
pub type ParserDWA = DWA;

/// Type alias for backward compatibility
#[deprecated(since = "0.3.0", note = "Use ParserDWA instead")]
pub type Precomputed4 = DWA;

pub type Signature = Vec<Vec<Option<TerminalID>>>;

pub struct NwaTraversalData {
    pub comp_id: Vec<usize>,
    pub sccs: Vec<Vec<usize>>,
    pub topo: Vec<usize>,
}

impl NWA {
    pub fn compute_traversal_data(&self) -> NwaTraversalData {
        let (sccs, comp_id) = compute_sccs(self);
        let scc_count = sccs.len();
        let mut scc_adj = vec![HashSet::new(); scc_count];
        let mut indeg = vec![0; scc_count];

        for u in 0..self.states.len() {
            let u_scc = comp_id[u];
            let state = &self.states[u];
            let mut neighbors = Vec::new();
            for targets in state.transitions.values() {
                for (v, _) in targets { neighbors.push(*v); }
            }
            for (v, _) in &state.epsilons { neighbors.push(*v); }

            for v in neighbors {
                let v_scc = comp_id[v];
                if u_scc != v_scc && !scc_adj[u_scc].contains(&v_scc) {
                    scc_adj[u_scc].insert(v_scc);
                    indeg[v_scc] += 1;
                }
            }
        }

        let mut topo = Vec::with_capacity(scc_count);
        let mut q = VecDeque::new();
        for i in 0..scc_count { if indeg[i] == 0 { q.push_back(i); } }

        while let Some(u) = q.pop_front() {
            topo.push(u);
            for &v in &scc_adj[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 { q.push_back(v); }
            }
        }
        NwaTraversalData { comp_id, sccs, topo }
    }
}

fn compute_sccs(nwa: &NWA) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = nwa.states.len();
    let mut adj = vec![Vec::new(); n];
    let mut rev_adj = vec![Vec::new(); n];

    for (u, state) in nwa.states.0.iter().enumerate() {
        let mut neighbors = Vec::new();
        for targets in state.transitions.values() {
            for (v, _) in targets { neighbors.push(*v); }
        }
        for (v, _) in &state.epsilons { neighbors.push(*v); }

        for v in neighbors {
            adj[u].push(v);
            rev_adj[v].push(u);
        }
    }

    let mut order = Vec::new();
    let mut visited = vec![false; n];
    for i in 0..n {
        if !visited[i] {
            let mut stack = vec![(i, false)];
            while let Some((u, processed)) = stack.pop() {
                if processed { order.push(u); } else {
                    if visited[u] { continue; }
                    visited[u] = true;
                    stack.push((u, true));
                    for &v in &adj[u] { if !visited[v] { stack.push((v, false)); } }
                }
            }
        }
    }

    let mut comp_id = vec![usize::MAX; n];
    let mut sccs = Vec::new();
    let mut current_scc_id = 0;

    for &u in order.iter().rev() {
        if comp_id[u] == usize::MAX {
            let mut component = Vec::new();
            let mut stack = vec![u];
            comp_id[u] = current_scc_id;
            while let Some(curr) = stack.pop() {
                component.push(curr);
                for &prev in &rev_adj[curr] {
                    if comp_id[prev] == usize::MAX {
                        comp_id[prev] = current_scc_id;
                        stack.push(prev);
                    }
                }
            }
            sccs.push(component);
            current_scc_id += 1;
        }
    }
    (sccs, comp_id)
}

pub fn nwa_special_map<V, U, I>(
    nwa: &NWA,
    traversal_data: &NwaTraversalData,
    initial_values: Vec<(StateID, V)>,
    mut step: impl FnMut(&U, Option<Label>, &[(StateID, Weight)]) -> I,
    mut merge: impl FnMut(&mut V, V),
    mut process: impl FnMut(StateID, V) -> Option<U>,
) where
    V: Clone,
    I: IntoIterator<Item = (StateID, V)>,
{
    let mut values: FxHashMap<StateID, V> = FxHashMap::default();
    let mut stopped_nodes: FxHashSet<StateID> = FxHashSet::default();

    for (state, v) in initial_values {
        values.entry(state).and_modify(|old| merge(old, v.clone())).or_insert(v);
    }

    let mut in_queue = FxHashSet::default();


    for &scc_idx in &traversal_data.topo {
        let scc_nodes = &traversal_data.sccs[scc_idx];
        let mut local_queue: VecDeque<StateID> = VecDeque::new();

        for &u in scc_nodes {
            if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                local_queue.push_back(u);
                in_queue.insert(u);
            }
        }
        if local_queue.is_empty() { continue; }

        while let Some(u) = local_queue.pop_front() {
            in_queue.remove(&u);

            if stopped_nodes.contains(&u) { continue; }

            let agg_v = match values.remove(&u) { Some(v) => v, None => continue };
            let proceed_val = match process(u, agg_v.clone()) {
                Some(val) => val,
                None => { stopped_nodes.insert(u); continue; }
            };
            let state = &nwa.states[u];

            if !state.epsilons.is_empty() {
                for (v, new_v) in step(&proceed_val, None, &state.epsilons) {
                    if stopped_nodes.contains(&v) { continue; }
                    values.entry(v).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                    if traversal_data.comp_id[v] == scc_idx && !in_queue.contains(&v) {
                        local_queue.push_back(v);
                        in_queue.insert(v);
                    }
                }
            }
            for (&label, targets) in &state.transitions {
                for (v, new_v) in step(&proceed_val, Some(label), targets) {
                    if stopped_nodes.contains(&v) { continue; }
                    values.entry(v).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                    if traversal_data.comp_id[v] == scc_idx && !in_queue.contains(&v) {
                        local_queue.push_back(v);
                        in_queue.insert(v);
                    }
                }
            }
            if values.contains_key(&u) && !stopped_nodes.contains(&u) && !in_queue.contains(&u) {
                local_queue.push_back(u);
                in_queue.insert(u);
            }
        }
    }
}

struct SignatureIndex {
    term_to_group: HashMap<Option<TerminalID>, usize>,
    total_terms: usize,
}

impl SignatureIndex {
    fn new(sig: &Signature) -> Self {
        let mut map = HashMap::new();
        let mut count = 0;
        for (g_idx, group) in sig.iter().enumerate() {
            for term in group { map.insert(*term, g_idx); count += 1; }
        }
        Self { term_to_group: map, total_terms: count }
    }
    fn get_group(&self, term: &Option<TerminalID>) -> Option<usize> { self.term_to_group.get(term).cloned() }
}

fn can_derive(parent: &Signature, child_index: &SignatureIndex) -> Option<Vec<Weight>> {
    let mut mapping = Vec::with_capacity(parent.len());
    let mut matched_terms = 0;
    for group in parent {
        if group.is_empty() { mapping.push(Weight::zeros()); continue; }
        let expected_g = child_index.get_group(&group[0]);
        for term in &group[1..] {
            if child_index.get_group(term) != expected_g { return None; }
        }
        if let Some(g) = expected_g { mapping.push(Weight::from_item(g)); matched_terms += group.len(); }
        else { mapping.push(Weight::zeros()); }
    }
    if matched_terms != child_index.total_terms { return None; }
    Some(mapping)
}

fn specialize_dwa_relative(parent_dwa: &DWA, mapping: &[Weight], parent_unique_weights: &[Weight]) -> DWA {
    // Pre-compute the mapping for all unique weights in the parent DWA.
    // This avoids re-computing the mapping for the same weight multiple times across different states
    // and allows us to use a read-only map during the parallel state construction.
    let weight_map: HashMap<Weight, Weight> = parent_unique_weights.iter()
        .map(|w| {
            // OPTIMIZATION: Accumulate in a local RangeSetBlaze to avoid SimpleBitset lock contention
            let mut accumulator = RangeSetBlaze::new();
            let mut is_all = false;

            for bit in w.iter_up_to(mapping.len()) {
                if let Some(target_w) = mapping.get(bit) {
                    if target_w.is_all_fast() {
                        is_all = true;
                        break;
                    }
                    // Access the inner RSB directly to avoid locking
                    accumulator |= &target_w.rsb;
                }
            }

            let new_w = if is_all {
                Weight::all()
            } else {
                Weight::from_rsb(accumulator)
            };
            (w.clone(), new_w)
        })
        .collect();

    specialize_dwa_relative_with_map(parent_dwa, &weight_map)
}

fn specialize_dwa_relative_with_map(parent_dwa: &DWA, weight_map: &HashMap<Weight, Weight>) -> DWA {
    // Optimized version that uses a pre-computed weight_map.
    // This is faster when specializing many DWAs from the same parent with different mappings,
    // as the weight_map can be computed once and reused.
    
    // We construct the new states in parallel using the pre-computed map.
    let new_states_vec: Vec<crate::precompute4::weighted_automata::dwa::DWAState> = parent_dwa.states.0.par_iter().map(|state| {
        let map_weight = |w: &Weight| -> Weight {
            if let Some(cw) = weight_map.get(w) { return cw.clone(); }
            // Fallback should not happen if weight_map is complete, but safe to keep
            Weight::zeros() 
        };

        let mut new_state = crate::precompute4::weighted_automata::dwa::DWAState::default();
        
        // Final weight
        if let Some(fw) = &state.final_weight {
            let new_fw = map_weight(fw);
            if !new_fw.is_empty() { new_state.final_weight = Some(new_fw); }
        }

        // Transitions
        for (label, w) in &state.trans_weights {
            let new_w = map_weight(w);
            if !new_w.is_empty() {
                new_state.trans_weights.insert(*label, new_w);
                if let Some(target) = state.transitions.get(label) {
                    new_state.transitions.insert(*label, *target);
                }
            }
        }
        
        new_state
    }).collect();

    DWA {
        states: crate::precompute4::weighted_automata::dwa::DWAStates(new_states_vec),
        body: parent_dwa.body.clone(),
    }
}

pub fn canonicalize_bundle(terminal_map: BTreeMap<Option<TerminalID>, Weight>) -> (Signature, Vec<Weight>) {
    let mut weight_groups: HashMap<Weight, Vec<Option<TerminalID>>> = HashMap::new();
    for (term, weight) in terminal_map {
        if !weight.is_empty() { weight_groups.entry(weight).or_default().push(term); }
    }
    let mut groups_vec: Vec<(Weight, Vec<Option<TerminalID>>)> = weight_groups.into_iter().collect();
    for (_, terms) in &mut groups_vec { terms.sort(); }
    groups_vec.sort_by(|a, b| a.1.cmp(&b.1));
    (groups_vec.iter().map(|(_, terms)| terms.clone()).collect(), groups_vec.into_iter().map(|(w, _)| w).collect())
}

/// Build the Parser DWA from the GLR parser and lexical NWA.
/// 
/// This is the main precomputation function that:
/// 1. Builds terminal DWAs from terminal characterizations
/// 2. Composes them with the lexical NWA
/// 3. Determinizes the result into the final Parser DWA
/// 
/// The resulting DWA is used at runtime for O(1) mask queries.
pub fn build_parser_dwa(parser: &GLRParser, terminal_nwa: &NWA) -> DWA {
    crate::debug!(4, "Starting Parser DWA construction");
    let now = Instant::now();
    let terminal_dwas = match build_template_dwas(parser) { Ok(m) => m, Err(e) => panic!("Failed to build terminal DWAs: {:?}", e), };
    let ignore_dwa = build_ignore_terminal_dwa();
    crate::debug!(4, "Built {} terminal DWAs in {:?}", terminal_dwas.len(), now.elapsed());

    // Check if we're in symbol-heavy mode (tsid encoded as labels, not weights)
    let is_symbol_heavy = !crate::constraint_precompute::is_weight_heavy_enabled();
    let terminals_count = parser.terminal_map.len();
    
    // In symbol-heavy mode, identify the original start state and tsid-labeled incoming edges
    // These will be used to reconstruct tsid-labeled transitions at the end
    let original_start_state = terminal_nwa.body.start_states[0];
    let tsid_to_root: BTreeMap<Label, StateID> = if is_symbol_heavy {
        let start_transitions = &terminal_nwa.states[original_start_state].transitions;
        
        // Collect tsid-labeled transitions (labels >= terminals_count)
        let mut mapping = BTreeMap::new();
        for (&label, targets) in start_transitions {
            if label as usize >= terminals_count {
                // This is a tsid transition: start --[tsid_label]--> root
                for &(target, _) in targets {
                    mapping.insert(label, target);
                }
            }
        }
        crate::debug!(4, "Symbol-heavy mode: found {} tsid transitions from original start state", mapping.len());
        mapping
    } else {
        BTreeMap::new()
    };

    let reversed_nwa = terminal_nwa.reverse();
    let traversal_data = reversed_nwa.compute_traversal_data();
    
    // In symbol-heavy mode, build a map of OUTGOING tsid-labeled edges FROM each root state
    // In the reversed NWA, root --[tsid_label]--> original_start
    // We need: root -> [(tsid_label, edge_weight), ...]
    let outgoing_tsid_edges: BTreeMap<StateID, Vec<(Label, Weight)>> = if is_symbol_heavy {
        let mut outgoing: BTreeMap<StateID, Vec<(Label, Weight)>> = BTreeMap::new();
        for (src, state) in reversed_nwa.states.0.iter().enumerate() {
            for (&label, targets) in &state.transitions {
                if label as usize >= terminals_count {
                    // This is a tsid-labeled transition
                    for (dst, weight) in targets {
                        if *dst == original_start_state {
                            outgoing.entry(src).or_default().push((label, weight.clone()));
                        }
                    }
                }
            }
        }
        crate::debug!(5, "Symbol-heavy mode: {} root states with tsid edges", outgoing.len());
        for (src, edges) in &outgoing {
            crate::debug!(6, "  Root state {} has tsid edges: {:?}", src, edges.iter().map(|(l,_)|*l).collect::<Vec<_>>());
        }
        outgoing
    } else {
        BTreeMap::new()
    };

    let initial_tokens = LLMTokenBV::max_ones();
    let mut initial_values_bv = Vec::new();
    for &start in &reversed_nwa.body.start_states {
        initial_values_bv.push((start, initial_tokens.clone()));
    }

    let start_pass1 = Instant::now();
    let (node_tokens, mut unique_signatures) = precompute_token_bvs_and_signatures(&reversed_nwa, &traversal_data, initial_values_bv);
    unique_signatures.insert(vec![vec![None]]);
    crate::debug!(4, "Pass 1: Tokens & Signatures ({} sigs, {:.2?})", unique_signatures.len(), start_pass1.elapsed());
    let mut unique_term_ids_in_sigs = BTreeSet::new();
    for sig in &unique_signatures {
        for terms in sig {
            for term in terms {
                if let Some(term_id) = term {
                    unique_term_ids_in_sigs.insert(term_id.0);
                }
            }
        }
    }

    let mut template_cache: FxHashMap<Signature, NWA> = FxHashMap::default();

    // OPTIMIZATION START: Split signatures into Simple (Direct Union) and Complex (Bitvector Derivation)
    let mut simple_signatures = Vec::new();
    let mut complex_signatures = Vec::new();

    for sig in unique_signatures {
        if sig.len() == 1 {
            simple_signatures.push(sig);
        } else {
            complex_signatures.push(sig);
        }
    }

    crate::debug!(4, "Optimization: {} simple signatures (direct build), {} complex signatures (derivation)",
        simple_signatures.len(), complex_signatures.len());

    // 1. FAST PATH: Handle simple signatures via direct Union
    // A signature of length 1 means all terminals in it map to the same logical state transition.
    // We don't need bitmasks; we just Union the Templates.
    // NOTE: Parallelizing this was tested but memory contention makes serial faster (143-169ms vs 121ms serial).

    for sig in simple_signatures {
        let terminals = &sig[0];
        let mut combined_nwa = NWA::new_empty();

        // If there are many terminals, this might look expensive, but NWA union is cheap (just adding edges/start states).
        // Determinization handles the complexity.
        for term_opt in terminals {
            let term_dwa = match term_opt {
                Some(term_id) => {
                    if parser.ignore_terminal_ids.contains(term_id) {
                        &ignore_dwa
                    } else {
                        terminal_dwas.get(term_id).unwrap_or(&ignore_dwa)
                    }
                },
                None => &ignore_dwa,
            };
            // We can convert DWA to NWA cheaply and union
            NWA::union_assign(&mut combined_nwa, &NWA::from_dwa(term_dwa));
        }

        // OPTIMIZATION: Skip NWA minimization for simple signatures.
        // Determinization will merge states anyway, so pre-minimization has minimal benefit.
        // Use lightweight minimization on the DWA to avoid expensive minimization.
        let mut dwa = combined_nwa.determinize();
        dwa.minimize_lightweight();

        template_cache.insert(sig, NWA::from_dwa(&dwa));

    }

    // 2. SLOW PATH: Handle complex signatures via Super DWA
    // Only run this logic if we actually have complex signatures.
    if !complex_signatures.is_empty() {
        crate::debug!(4, "Building Super DWA for {} complex signatures", complex_signatures.len());

        let mut used_terminals: BTreeSet<TerminalID> = BTreeSet::new();
        for sig in &complex_signatures {
            for group in sig {
                for term in group { if let Some(term) = term { used_terminals.insert(*term); } }
            }
        }
        crate::debug!(5, "  Used terminals: {}", used_terminals.len());

        let mut term_to_bit = BTreeMap::new();
        let mut bit_to_term: Vec<Option<TerminalID>> = Vec::new();
        // We ONLY include terminals relevant to the complex signatures to keep bitvectors small
        let mut all_terminals: BTreeSet<TerminalID> = used_terminals;

        // Note: Unlike original code, we don't force ALL terminal_dwas keys into the Super DWA,
        // only those needed for the complex pool. This makes the Super DWA smaller.

        term_to_bit.insert(None, 0);
        bit_to_term.push(None);
        for (i, term_id) in all_terminals.iter().enumerate() {
            term_to_bit.insert(Some(*term_id), i + 1);
            bit_to_term.push(Some(*term_id));
        }

        let start_super_nwa = std::time::Instant::now();
        let mut super_nwa = NWA::new_empty();
        for (term_id_opt, bit) in &term_to_bit {
            let mut weight = Weight::zeros();
            weight.set(*bit, true);
            let term_dwa = match term_id_opt {
                Some(term_id) => if parser.ignore_terminal_ids.contains(term_id) { &ignore_dwa } else { terminal_dwas.get(term_id).unwrap_or(&ignore_dwa) },
                None => &ignore_dwa,
            };
            let mut weighted_dwa = term_dwa.clone();
            weighted_dwa.apply_weight_inplace(&weight);
            NWA::union_assign(&mut super_nwa, &NWA::from_dwa(&weighted_dwa));
        }
        crate::debug!(5, "  Super NWA construction: {:?}, {} states, {} transitions", 
            start_super_nwa.elapsed(), super_nwa.states.len(), super_nwa.states.num_transitions());

        // OPTIMIZATION: Use lightweight minimization for super DWA construction.
        // Full minimization is expensive and not critical for intermediate results.
        let start_det = std::time::Instant::now();
        let super_dwa = super_nwa.determinize_and_minimize("SuperDWA");
        crate::debug!(5, "  Super DWA det+min: {:?}, {} states", start_det.elapsed(), super_dwa.states.len());

        let super_signature: Signature = bit_to_term.iter().map(|t| vec![*t]).collect();
        
        // Collect all unique weights from super_dwa once
        let start_weights = std::time::Instant::now();
        let super_dwa_unique_weights: Vec<Weight> = super_dwa.states.0.par_iter()
            .fold(HashSet::new, |mut acc, s| {
                 if let Some(w) = &s.final_weight { acc.insert(w.clone()); }
                 for w in s.trans_weights.values() { acc.insert(w.clone()); }
                 acc
            })
            .reduce(HashSet::new, |mut a, b| {
                for w in b { a.insert(w); }
                a
            })
            .into_iter().collect();
        crate::debug!(5, "  Collected {} unique weights in {:?}", super_dwa_unique_weights.len(), start_weights.elapsed());

        // PRE-COMPUTE: Build all weight mappings for all complex signatures upfront
        // This avoids redundant computation inside specialize_dwa_relative, which was creating
        // a new weight_map HashMap for each of the 199 complex signatures.
        let start_mappings = std::time::Instant::now();
        let all_mappings: Vec<(Signature, Vec<Weight>, HashMap<Weight, Weight>)> = complex_signatures.par_iter().map(|target_sig| {
            let target_idx = SignatureIndex::new(target_sig);
            let mapping = can_derive(&super_signature, &target_idx).expect("Super signature must derive target");
            
            // Pre-compute the weight mapping for this target signature
            let weight_map: HashMap<Weight, Weight> = super_dwa_unique_weights.iter()
                .map(|w| {
                    let mut accumulator = RangeSetBlaze::new();
                    let mut is_all = false;

                    for bit in w.iter_up_to(mapping.len()) {
                        if let Some(target_w) = mapping.get(bit) {
                            if target_w.is_all_fast() {
                                is_all = true;
                                break;
                            }
                            accumulator |= &target_w.rsb;
                        }
                    }

                    let new_w = if is_all {
                        Weight::all()
                    } else {
                        Weight::from_rsb(accumulator)
                    };
                    (w.clone(), new_w)
                })
                .collect();
            
            (target_sig.clone(), mapping, weight_map)
        }).collect();
        crate::debug!(5, "  Weight mappings: {:?}", start_mappings.elapsed());

        // PARALLEL OPTIMIZATION: Specialize DWAs using pre-computed weight mappings
        let start_specialize = std::time::Instant::now();
        let results: Vec<(Signature, NWA)> = all_mappings.par_iter().map(|(target_sig, _mapping, weight_map)| {
            let mut derived_dwa = specialize_dwa_relative_with_map(&super_dwa, weight_map);
            derived_dwa.minimize_lightweight();
            (target_sig.clone(), NWA::from_dwa(&derived_dwa))
        }).collect();
        crate::debug!(5, "  Specialization (149 DWAs): {:?}", start_specialize.elapsed());

        for (sig, nwa) in results {
            template_cache.insert(sig, nwa);
        }
    }
    // OPTIMIZATION END

    crate::debug!(4, "Finished DWA specialization");

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

    // Store (NWABody, Weight, node_id) - node_id is the state ID in reversed NWA where we collected this body
    // In symbol-heavy mode, we also collect tsid-specific bodies separately
    let final_bodies_arc: Arc<Mutex<Vec<(NWABody, Weight, StateID)>>> = Arc::new(Mutex::new(Vec::new()));
    
    // For symbol-heavy mode: collect (NWABody, Weight, tsid_label) for each tsid-labeled transition
    // These are transitions from root states to the original start state in the reversed NWA
    let tsid_bodies_arc: Arc<Mutex<Vec<(NWABody, Weight, Label)>>> = Arc::new(Mutex::new(Vec::new()));

    crate::debug!(4, "Beginning NWA traversal");

    // Clone references for use in closures
    let tsid_bodies_for_process = tsid_bodies_arc.clone();
    let template_cache_ref = &template_cache;
    let states_arena_ref = &states_arena;

    nwa_special_map(
        &reversed_nwa, &traversal_data, initial_values_full,
        |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV), edge_label, transitions| {
            let (current_bodies, current_tokens) = current_val;
            let mut results = Vec::new();
            
            // In symbol-heavy mode, skip tsid-labeled transitions in normal traversal
            // These will be handled in the process callback when we're at a root state
            if is_symbol_heavy {
                if let Some(label) = edge_label {
                    if label as usize >= terminals_count {
                        // This is a tsid-labeled transition - skip it
                        return results;
                    }
                }
            }
            
            let terminal_id = edge_label.map(|l| TerminalID(l as usize));
            for (dest_id, weight) in transitions {
                let edge_bv_tokens: LLMTokenBV = weight.clone().into();
                let next_tokens = current_tokens & &edge_bv_tokens;
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
        |node_id, val| {
            static PROCESS_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            static INSTANTIATE_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            static TOTAL_TEMPLATE_STATES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            
            let proc_count = PROCESS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if proc_count % 10000 == 0 {
                crate::debug!(4, "Process callback #{}, instantiate count: {}, total_states: {}", proc_count, 
                    INSTANTIATE_COUNT.load(std::sync::atomic::Ordering::Relaxed),
                    TOTAL_TEMPLATE_STATES.load(std::sync::atomic::Ordering::Relaxed));
            }
            
            let (nwa_bodies_map, tokens) = val;
            let bodies_count = nwa_bodies_map.len();
            let mut nwa_body = NWABody { start_states: vec![] };
            for (right_body, terminal_map) in &nwa_bodies_map {
                let (signature, concrete_weights) = canonicalize_bundle(terminal_map.clone());
                let cached_nwa = template_cache.get(&signature).expect_else(|| format!("Template must exist for signature {:?}", signature));
                
                let template_size = cached_nwa.states.len();
                let count = INSTANTIATE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                TOTAL_TEMPLATE_STATES.fetch_add(template_size, std::sync::atomic::Ordering::Relaxed);
                if count % 10000 == 0 {
                    crate::debug!(4, "Template instantiation #{}: {} total states so far, template size {}, bodies_count: {}", 
                        count, TOTAL_TEMPLATE_STATES.load(std::sync::atomic::Ordering::Relaxed), template_size, bodies_count);
                }
                
                let mut states = states_arena.borrow_mut();
                let new_states_offset = states.len();
                let composed_body = instantiate_nwa_template_into(cached_nwa, &concrete_weights, &mut states, right_body);
                let range = new_states_offset..states.len();
                if !range.is_empty() {
                    apply_cancellations_range(&mut states, range.clone());
                    apply_finality_fixpoint_range(&mut states, range.clone());
                    remove_negative_transitions_range(&mut states, range);
                    // Note: remove_redundant_default_transitions is NOT called here because it 
                    // requires a global pass over all states. It's called once at the end.
                }
                nwa_body = NWABody::union(&nwa_body, &composed_body);
            }
            
            // In symbol-heavy mode, check if this is a root state (has tsid-labeled edges to original_start_state)
            // If so, collect the body for each tsid
            if is_symbol_heavy {
                if let Some(tsid_edges) = outgoing_tsid_edges.get(&node_id) {
                    for (tsid_label, edge_weight) in tsid_edges {
                        let edge_bv: LLMTokenBV = edge_weight.clone().into();
                        let intersection_bv = &tokens & &edge_bv;
                        if !intersection_bv.is_empty() && !nwa_body.start_states.is_empty() {
                            let final_w = Weight::from_rsb(intersection_bv.inner.as_ref().clone());
                            crate::debug!(6, "Collecting tsid body at root {} for label {} with {} tokens", 
                                node_id, tsid_label, final_w.len());
                            let mut tb = tsid_bodies_for_process.lock().unwrap();
                            tb.push((nwa_body.clone(), final_w, *tsid_label));
                        }
                    }
                }
            }
            
            // Check if this is a final state in the reversed NWA (original start state or root state)
            // In symbol-heavy mode, we handle the original_start_state specially via tsid_bodies
            // so don't collect it here
            let should_collect = if is_symbol_heavy {
                // In symbol-heavy mode, only collect for states OTHER than original_start_state
                // (the original start is handled via tsid-labeled transitions)
                node_id != original_start_state && reversed_nwa.states[node_id].final_weight.is_some()
            } else {
                reversed_nwa.states[node_id].final_weight.is_some()
            };
            
            if should_collect {
                if let Some(fw) = &reversed_nwa.states[node_id].final_weight {
                    let fw_bv: LLMTokenBV = fw.clone().into();
                    let intersection_bv = &tokens & &fw_bv;
                    if !intersection_bv.is_empty() {
                        let final_w = Weight::from_rsb(intersection_bv.inner.as_ref().clone());
                        let mut fb = final_bodies_arc.lock().unwrap();
                        fb.push((nwa_body.clone(), final_w, node_id));
                    }
                }
            }
            
            if !tokens.is_empty() {
                let mut next_body_map = BTreeMap::new(); next_body_map.insert(nwa_body, BTreeMap::new());
                Some((next_body_map, tokens))
            } else { None }
        },
    );
    // Drop the process closure's reference to tsid_bodies
    drop(tsid_bodies_for_process);

    crate::debug!(4, "Finished Pass 2");
    let final_bodies = Arc::try_unwrap(final_bodies_arc).unwrap().into_inner().unwrap();
    let tsid_bodies = Arc::try_unwrap(tsid_bodies_arc).unwrap().into_inner().unwrap();
    crate::debug!(4, "Collected {} final bodies, {} tsid bodies, states_arena has {} states", 
        final_bodies.len(), tsid_bodies.len(), states_arena.borrow().len());
    let mut combined_nwa_states = states_arena.into_inner();
    let combined_start_state = combined_nwa_states.add_state();
    
    if is_symbol_heavy && !tsid_bodies.is_empty() {
        // Symbol-heavy mode: add labeled transitions with tsid labels
        // Use the tsid_bodies collected during traversal
        for (body, weight, tsid_label) in tsid_bodies {
            crate::debug!(6, "Adding tsid body with label={}, weight len={}", tsid_label, weight.len());
            for &s in &body.start_states {
                combined_nwa_states.add_transition(combined_start_state, tsid_label, s, weight.clone()).unwrap();
            }
        }
        crate::debug!(4, "Symbol-heavy mode: added {} tsid-labeled transitions", 
            combined_nwa_states[combined_start_state].transitions.values().map(|v| v.len()).sum::<usize>());
    } else {
        // Weight-heavy mode: no tsid labels, just add epsilon transitions with weights
        // The weights encode tsid info (positions in N×M space)
        for (body, weight, _node_id) in final_bodies {
            for &s in &body.start_states {
                combined_nwa_states.add_epsilon(combined_start_state, s, weight.clone());
            }
        }
    }

    let combined_nwa = NWA { states: combined_nwa_states, body: NWABody { start_states: vec![combined_start_state] } };
    let mut final_dwa = finalize_and_optimize_and_determinize(parser, combined_nwa);
    // SKIP final minimization to test performance impact
    // final_dwa.minimize();
    crate::debug!(4, "Parser DWA construction complete. Stats: {}", final_dwa.stats());

    final_dwa
}

/// Deprecated alias for build_parser_dwa
#[deprecated(since = "0.3.0", note = "Use build_parser_dwa instead")]
pub fn precompute4(parser: &GLRParser, terminal_nwa: &NWA) -> DWA {
    build_parser_dwa(parser, terminal_nwa)
}

pub fn precompute_token_bvs_and_signatures(reversed_nwa: &NWA, traversal_data: &NwaTraversalData, initial_values: Vec<(StateID, LLMTokenBV)>) -> (HashMap<StateID, LLMTokenBV>, HashSet<Signature>) {
    let node_tokens: Arc<Mutex<HashMap<StateID, LLMTokenBV>>> = Arc::new(Mutex::new(HashMap::new()));
    let signatures: Arc<Mutex<HashSet<Signature>>> = Arc::new(Mutex::new(HashSet::new()));

    let node_tokens_clone = node_tokens.clone();
    let signatures_clone = signatures.clone();

    nwa_special_map(reversed_nwa, traversal_data, initial_values,
        move |tokens: &LLMTokenBV, _edge_label, transitions| {
            let mut results = Vec::new();
            for (dest_id, weight) in transitions {
                let edge_bv: LLMTokenBV = weight.clone().into();
                let next = tokens & &edge_bv;
                if !next.is_empty() { results.push((*dest_id, next)); }
            }
            results
        },
        |t1, t2| { *t1 |= &t2; },
        move |node_id, tokens| {
            node_tokens_clone.lock().unwrap().insert(node_id, tokens.clone());
            let mut bundles_by_dest: HashMap<StateID, BTreeMap<Option<TerminalID>, Weight>> = HashMap::new();
            let state = &reversed_nwa.states[node_id];
            for (label, targets) in &state.transitions {
                let term = Some(TerminalID(*label as usize));
                for (v, w) in targets {
                    let edge_bv: LLMTokenBV = w.clone().into();
                    let combined = &tokens & &edge_bv;
                    if !combined.is_empty() {
                        let w_weight = Weight::from_rsb(edge_bv.inner.as_ref().clone());
                        bundles_by_dest.entry(*v).or_default().insert(term, w_weight);
                    }
                }
            }
            for (v, w) in &state.epsilons {
                let edge_bv: LLMTokenBV = w.clone().into();
                let combined = &tokens & &edge_bv;
                if !combined.is_empty() {
                    let w_weight = Weight::from_rsb(edge_bv.inner.as_ref().clone());
                    bundles_by_dest.entry(*v).or_default().insert(None, w_weight);
                }
            }
            let mut sigs = signatures_clone.lock().unwrap();
            for (_, bundle) in bundles_by_dest {
                let (sig, _) = canonicalize_bundle(bundle);
                sigs.insert(sig);
            }
            Some(tokens)
        },
    );
    (Arc::try_unwrap(node_tokens).unwrap().into_inner().unwrap(), Arc::try_unwrap(signatures).unwrap().into_inner().unwrap())
}

pub fn finalize_and_optimize_and_determinize(parser: &GLRParser, mut combined_nwa: NWA) -> DWA {
    crate::debug!(4, "Resolving negatives and optimizing for NWA with {} states and {} transitions...", combined_nwa.states.len(), combined_nwa.states.num_transitions());
    prune_continuations_from_final_states(&mut combined_nwa);
    crate::debug!(4, "Pruned continuations from final states. NWA with {} states and {} transitions remaining.", combined_nwa.states.len(), combined_nwa.states.num_transitions());
    
    // Use unified determinize_and_minimize with "FinalDWA" profile
    // Pipeline: determinize → prune_dead_ends → minimize
    let dwa = combined_nwa.determinize_and_minimize("FinalDWA");
    crate::debug!(4, "Final DWA minimization complete. {} states and {} transitions.", dwa.states.len(), dwa.states.num_transitions());
    dwa
}

pub fn instantiate_nwa_template_into(
    template: &NWA,
    ordered_weights: &[Weight],
    states: &mut NWAStates,
    right_body: &NWABody,
) -> NWABody {
    let offset = states.len();
    states.0.reserve(template.states.len());

    let mut union_cache: HashMap<Weight, Weight> = HashMap::new();
    let mut map_abstract_weight = |w: &Weight| -> Weight {
        if w.is_empty() { return Weight::zeros(); }
        if let Some(res) = union_cache.get(w) { return res.clone(); }
        let mut concrete = Weight::zeros();
        for idx in w.iter_up_to(ordered_weights.len()) {
            if let Some(concrete_w) = ordered_weights.get(idx) { concrete |= concrete_w; }
        }
        union_cache.insert(w.clone(), concrete.clone());
        concrete
    };

    for old_state in &template.states.0 {
        let mut new_state = crate::precompute4::weighted_automata::nwa::NWAState::default();
        
        // Transitions
        for (lbl, targets) in &old_state.transitions {
            let mut new_targets = Vec::with_capacity(targets.len());
            for (target, w) in targets {
                let concrete = map_abstract_weight(w);
                if !concrete.is_empty() {
                    new_targets.push((*target + offset, concrete));
                }
            }
            if !new_targets.is_empty() {
                new_state.transitions.insert(*lbl, new_targets);
            }
        }

        // Epsilons
        for (target, w) in &old_state.epsilons {
            let concrete = map_abstract_weight(w);
            if !concrete.is_empty() {
                new_state.epsilons.push((*target + offset, concrete));
            }
        }

        // Final Weight -> Epsilon to right_body starts
        if let Some(fw) = &old_state.final_weight {
            let concrete = map_abstract_weight(fw);
            if !concrete.is_empty() {
                for &r_start in &right_body.start_states {
                    new_state.epsilons.push((r_start, concrete.clone()));
                }
            }
        }

        states.0.push(new_state);
    }

    NWABody {
        start_states: template.body.start_states.iter().map(|s| s + offset).collect()
    }
}

fn minimize_remove_epsilon(nwa: &mut NWA) {
    nwa.minimize()
}