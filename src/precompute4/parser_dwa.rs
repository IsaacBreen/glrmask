use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use profiler_macro::{time_it, timeit};



use crate::glr::parser::{ExpectElse, GLRParser};
use crate::precompute4::resolve_negatives::{
    apply_cancellations_range, apply_finality_fixpoint_range, remove_negative_transitions_range
    // Note: remove_redundant_default_transitions is called once at the end in finalize_and_optimize_and_determinize,
    // not per-range here, since it requires a global pass over all states.
};
use crate::precompute4::template_dfa::{build_ignore_terminal_dwa, build_template_dwas};
use crate::dwa_i32::{
    common::Label, determinization_rustfst::determinize_nwa_to_dwa, dwa::DWAStats, DeterminizeAndMinimizeProfile, DwaOptimizeConfig,
    DWA, NWA, NWABody, NWAStateID, NWAStates, StateID, Weight,
};
use crate::dfa_u8::TokenizerStateID;
use crate::types::{TerminalID, TerminalID as GrammarTokenID};
use crate::datastructures::abstract_weight::{BackendChoice, override_backend, restore_backend};

struct MinimizeRustfstConfig {
    rm_epsilon: bool,
    #[allow(dead_code)]
    determinize: bool,
}

impl MinimizeRustfstConfig {
    fn default() -> Self { Self { rm_epsilon: false, determinize: false } }
    fn with_rm_epsilon(mut self, val: bool) -> Self { self.rm_epsilon = val; self }
}

pub use crate::precompute4::template_dfa::FullDWABuildError;

/// The Parser DWA - the final precomputed artifact used for get_mask queries.
/// This is a deterministic weighted automaton where weights are sparse bitvectors
/// over LLM token equivalence classes.
pub type ParserDWA = DWA;

/// Type alias for backward compatibility
#[deprecated(since = "0.3.0", note = "Use ParserDWA instead")]
pub type Precomputed4 = DWA;

pub type Signature = Vec<Vec<Option<TerminalID>>>;

struct Pass2Profile {
    process_total_us: AtomicU64,
    process_count: AtomicU64,
    template_count: AtomicU64,
    canonicalize_us: AtomicU64,
    cache_lookup_us: AtomicU64,
    cache_insert_us: AtomicU64,
    dynamic_derive_us: AtomicU64,
    instantiate_us: AtomicU64,
    apply_cancellations_us: AtomicU64,
    apply_finality_us: AtomicU64,
    remove_negative_us: AtomicU64,
    union_us: AtomicU64,
    tsid_collect_us: AtomicU64,
    final_collect_us: AtomicU64,
}

impl Pass2Profile {
    fn new() -> Self {
        Self {
            process_total_us: AtomicU64::new(0),
            process_count: AtomicU64::new(0),
            template_count: AtomicU64::new(0),
            canonicalize_us: AtomicU64::new(0),
            cache_lookup_us: AtomicU64::new(0),
            cache_insert_us: AtomicU64::new(0),
            dynamic_derive_us: AtomicU64::new(0),
            instantiate_us: AtomicU64::new(0),
            apply_cancellations_us: AtomicU64::new(0),
            apply_finality_us: AtomicU64::new(0),
            remove_negative_us: AtomicU64::new(0),
            union_us: AtomicU64::new(0),
            tsid_collect_us: AtomicU64::new(0),
            final_collect_us: AtomicU64::new(0),
        }
    }

    fn log(&self) {
        let process_count = self.process_count.load(Ordering::Relaxed);
        if process_count == 0 {
            return;
        }
        let template_count = self.template_count.load(Ordering::Relaxed);
        let process_total_us = self.process_total_us.load(Ordering::Relaxed);
        let canonicalize_us = self.canonicalize_us.load(Ordering::Relaxed);
        let cache_lookup_us = self.cache_lookup_us.load(Ordering::Relaxed);
        let cache_insert_us = self.cache_insert_us.load(Ordering::Relaxed);
        let dynamic_derive_us = self.dynamic_derive_us.load(Ordering::Relaxed);
        let instantiate_us = self.instantiate_us.load(Ordering::Relaxed);
        let apply_cancellations_us = self.apply_cancellations_us.load(Ordering::Relaxed);
        let apply_finality_us = self.apply_finality_us.load(Ordering::Relaxed);
        let remove_negative_us = self.remove_negative_us.load(Ordering::Relaxed);
        let union_us = self.union_us.load(Ordering::Relaxed);
        let tsid_collect_us = self.tsid_collect_us.load(Ordering::Relaxed);
        let final_collect_us = self.final_collect_us.load(Ordering::Relaxed);

        crate::debug!(
            4,
            "Pass2 profile: process_total={:?} ({} calls), templates={}, canonicalize={:?}, cache_lookup={:?}, cache_insert={:?}, dynamic_derive={:?}, instantiate={:?}, cancellations={:?}, finality_fixpoint={:?}, remove_negative={:?}, union={:?}, tsid_collect={:?}, final_collect={:?}",
            std::time::Duration::from_micros(process_total_us),
            process_count,
            template_count,
            std::time::Duration::from_micros(canonicalize_us),
            std::time::Duration::from_micros(cache_lookup_us),
            std::time::Duration::from_micros(cache_insert_us),
            std::time::Duration::from_micros(dynamic_derive_us),
            std::time::Duration::from_micros(instantiate_us),
            std::time::Duration::from_micros(apply_cancellations_us),
            std::time::Duration::from_micros(apply_finality_us),
            std::time::Duration::from_micros(remove_negative_us),
            std::time::Duration::from_micros(union_us),
            std::time::Duration::from_micros(tsid_collect_us),
            std::time::Duration::from_micros(final_collect_us),
        );
    }
}

struct WeightBackendOverride {
    previous: Option<BackendChoice>,
}

impl WeightBackendOverride {
    fn new(backend: &str) -> Self {
        let choice = match backend {
            "rangeset" | "rsb" => BackendChoice::RangeSet,
            _ => BackendChoice::Factorized,
        };
        let previous = override_backend(choice);
        Self { previous }
    }
}

impl Drop for WeightBackendOverride {
    fn drop(&mut self) {
        restore_backend(self.previous);
    }
}

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
                for &v in &rev_adj[curr] {
                    if comp_id[v] == usize::MAX {
                        comp_id[v] = current_scc_id;
                        stack.push(v);
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
    mut merge: impl FnMut(&mut V, V) -> bool,
    mut process: impl FnMut(StateID, V) -> Option<U>,
) where
    V: Clone,
    I: IntoIterator<Item = (StateID, V)>,
{
    let mut values: FxHashMap<StateID, V> = FxHashMap::default();
    let mut stopped_nodes: FxHashSet<StateID> = FxHashSet::default();

    for (state, v) in initial_values {
        match values.entry(state) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                merge(entry.get_mut(), v);
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(v);
            }
        }
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

            let agg_v = match values.get(&u) { Some(v) => v.clone(), None => continue };
            let proceed_val = match process(u, agg_v.clone()) {
                Some(val) => val,
                None => { stopped_nodes.insert(u); continue; }
            };
            let state = &nwa.states[u];

            if !state.epsilons.is_empty() {
                for (v, new_v) in step(&proceed_val, None, &state.epsilons) {
                    if stopped_nodes.contains(&v) { continue; }
                    let changed = match values.entry(v) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            merge(entry.get_mut(), new_v)
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(new_v);
                            true
                        }
                    };
                    if changed && traversal_data.comp_id[v] == scc_idx && !in_queue.contains(&v) {
                        local_queue.push_back(v);
                        in_queue.insert(v);
                    }
                }
            }
            for (&label, targets) in &state.transitions {
                for (v, new_v) in step(&proceed_val, Some(label), targets) {
                    if stopped_nodes.contains(&v) { continue; }
                    let changed = match values.entry(v) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            merge(entry.get_mut(), new_v)
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(new_v);
                            true
                        }
                    };
                    if changed && traversal_data.comp_id[v] == scc_idx && !in_queue.contains(&v) {
                        local_queue.push_back(v);
                        in_queue.insert(v);
                    }
                }
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
            // OPTIMIZATION: Accumulate in a local Weight to avoid SimpleBitset lock contention
            let mut accumulator = Weight::zeros();
            let mut is_all = false;

            for bit in w.iter_up_to_allow_expansion(mapping.len()) {
                if let Some(target_w) = mapping.get(bit) {
                    if target_w.is_all_fast() {
                        is_all = true;
                        break;
                    }
                    // Access the inner RSB directly to avoid locking
                    accumulator |= target_w;
                }
            }

            let new_w = if is_all {
                Weight::all()
            } else {
                accumulator
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
    let new_states_vec: Vec<crate::dwa_i32::dwa::DWAState> = parent_dwa.states.0.par_iter().map(|state| {
        let map_weight = |w: &Weight| -> Weight {
            if let Some(cw) = weight_map.get(w) { return cw.clone(); }
            // Fallback should not happen if weight_map is complete, but safe to keep
            Weight::zeros() 
        };

        let mut new_state = crate::dwa_i32::dwa::DWAState::default();
        
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
        states: crate::dwa_i32::dwa::DWAStates(new_states_vec),
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
/// 1. Builds template DWAs from terminal characterizations (one per terminal group)
/// 2. Composes them with the lexical NWA
/// 3. Determinizes the result into the final Parser DWA
/// 
/// The resulting DWA is used at runtime for O(1) mask queries.
#[time_it("build_parser_dwa")]
pub fn build_parser_dwa(parser: &GLRParser, terminal_nwa: &NWA) -> DWA {
    crate::debug!(5, "build_parser_dwa: start");
    crate::debug!(3, "Starting Parser DWA construction. Input terminal_nwa: {}", 
        terminal_nwa.stats());
    
    // Handle empty terminal NWA (no valid tokens for this grammar/vocabulary combination)
    // Return a minimal DWA with one state and no transitions (always returns empty mask)
    if terminal_nwa.states.0.is_empty() || terminal_nwa.body.start_states.is_empty() {
        crate::debug!(3, "Terminal NWA is empty - returning empty Parser DWA");
        let mut empty_dwa = DWA::new_empty();
        // Add a single start state with no final weight (no tokens are valid)
        let start_state = empty_dwa.states.add_state();
        empty_dwa.body.start_state = start_state;
        crate::debug!(5, "build_parser_dwa: end (empty terminal NWA)");
        return empty_dwa;
    }
    
    let template_dwas = timeit!("build_template_dwas", {
        match build_template_dwas(parser) {
            Ok(m) => m,
            Err(e) => panic!("Failed to build template DWAs: {:?}", e),
        }
    });
    let ignore_dwa = timeit!("build_ignore_terminal_dwa", {
        build_ignore_terminal_dwa()
    });

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

    // Debug: dump input terminal NWA
    crate::debug!(5, "Input terminal NWA: {}, start_states={:?}", terminal_nwa.stats(), terminal_nwa.body.start_states);
    for (i, state) in terminal_nwa.states.0.iter().enumerate() {
        crate::debug!(6, "  Input State {}: final_weight={:?}, epsilons={}, transitions={:?}", 
            i, 
            state.final_weight.as_ref().map(|w| format!("len={}, ranges={}", w.len(), w.ranges_len())),
            state.epsilons.len(),
            state.transitions.iter().map(|(&l, targets)| format!("{}:{}", l, targets.len())).collect::<Vec<_>>()
        );
    }

    let reversed_nwa = terminal_nwa.reverse();
    crate::debug!(5, "Reversed NWA: {}, start_states={:?}", reversed_nwa.stats(), reversed_nwa.body.start_states);
    for (i, state) in reversed_nwa.states.0.iter().enumerate() {
        crate::debug!(6, "  State {}: final_weight={:?}, epsilons={:?}, transitions={}", 
            i, 
            state.final_weight.as_ref().map(|w| format!("len={}, ranges={}", w.len(), w.ranges_len())),
            state.epsilons.iter().take(3).map(|(v, w)| format!("->{}(len={}, ranges={})", v, w.len(), w.ranges_len())).collect::<Vec<_>>(),
            state.transitions.len()
        );
    }
    let traversal_data = timeit!("parser_dwa::compute_traversal_data", {
        reversed_nwa.compute_traversal_data()
    });
    
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

    let initial_tokens = Weight::all();
    let mut initial_values_bv = Vec::new();
    for &start in &reversed_nwa.body.start_states {
        initial_values_bv.push((start, initial_tokens.clone()));
    }

    let start_pass1 = Instant::now();
    let (node_tokens, mut unique_signatures) = timeit!("parser_dwa::pass1_precompute", {
        precompute_token_bvs_and_signatures(&reversed_nwa, &traversal_data, initial_values_bv)
    });
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

    let mut super_dwa_opt: Option<DWA> = None;
    let mut super_signature_opt: Option<Signature> = None;
    let mut super_dwa_unique_weights_opt: Option<Vec<Weight>> = None;

    let template_cache: RefCell<FxHashMap<Signature, Arc<NWA>>> = RefCell::new(FxHashMap::default());

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

    timeit!("parser_dwa::simple_signatures", {
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
                            template_dwas.get(term_id).unwrap_or(&ignore_dwa)
                        }
                    },
                    None => &ignore_dwa,
                };
                // We can convert DWA to NWA cheaply and union
                NWA::union_assign(&mut combined_nwa, &NWA::from_dwa(term_dwa));
            }

            // OPTIMIZATION: Skip NWA minimization for simple signatures.
            // Determinization will merge states anyway, so pre-minimization has minimal benefit.
            // Use basic pruning on the DWA to avoid expensive minimization.
            let mut dwa = combined_nwa.determinize();
            dwa.prune_basic();

            template_cache.borrow_mut().insert(sig, Arc::new(NWA::from_dwa(&dwa)));
        }
    });

    // 2. SLOW PATH: Handle complex signatures via Super DWA
    // Only run this logic if we actually have complex signatures.
    if !complex_signatures.is_empty() {
        timeit!("parser_dwa::complex_signatures", {
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

        // Note: Unlike original code, we don't force ALL template_dwas keys into the Super DWA,
        // only those needed for the complex pool. This makes the Super DWA smaller.

        term_to_bit.insert(None, 0);
        bit_to_term.push(None);
        for (i, term_id) in all_terminals.iter().enumerate() {
            term_to_bit.insert(Some(*term_id), i + 1);
            bit_to_term.push(Some(*term_id));
        }

        let _rangeset_backend = WeightBackendOverride::new("rangeset");
        let template_dwas_rsb = match build_template_dwas(parser) {
            Ok(m) => m,
            Err(e) => panic!("Failed to build template DWAs for Super DWA: {:?}", e),
        };
        let ignore_dwa_rsb = build_ignore_terminal_dwa();
        crate::debug!(5, "  Built {} template DWAs with rangeset backend", template_dwas_rsb.len());

        let mut super_nwa = timeit!("parser_dwa::super_nwa_build", {
            let start_super_nwa = std::time::Instant::now();
            let mut super_nwa = NWA::new_empty();
            for (term_id_opt, bit) in &term_to_bit {
                let mut weight = Weight::zeros();
                weight.set(*bit, true);
                let term_dwa = match term_id_opt {
                    Some(term_id) => {
                        if parser.ignore_terminal_ids.contains(term_id) {
                            &ignore_dwa_rsb
                        } else {
                            template_dwas_rsb.get(term_id).unwrap_or(&ignore_dwa_rsb)
                        }
                    }
                    None => &ignore_dwa_rsb,
                };
                let mut weighted_dwa = term_dwa.clone();
                weighted_dwa.apply_weight_inplace(&weight);
                NWA::union_assign(&mut super_nwa, &NWA::from_dwa(&weighted_dwa));
            }
            crate::debug!(5, "  Super NWA construction: {:?}, {}", 
                start_super_nwa.elapsed(), super_nwa.stats());
            super_nwa
        });

        // OPTIMIZATION: Use lightweight minimization for super DWA construction.
        // Full minimization is expensive and not critical for intermediate results.
        let super_dwa = timeit!("parser_dwa::super_dwa_det_min", {
            let start_det = std::time::Instant::now();
            let super_dwa = super_nwa.determinize_and_minimize(DeterminizeAndMinimizeProfile::Super);
            crate::debug!(5, "  Super DWA det+min: {:?}, {}", start_det.elapsed(), super_dwa.stats());
            super_dwa
        });

        let super_signature: Signature = bit_to_term.iter().map(|t| vec![*t]).collect();
        
        // Collect all unique weights from super_dwa once
        let super_dwa_unique_weights: Vec<Weight> = timeit!("parser_dwa::super_dwa_collect_weights", {
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
            super_dwa_unique_weights
        });

        // PRE-COMPUTE: Build all weight mappings for all complex signatures upfront
        // This avoids redundant computation inside specialize_dwa_relative, which was creating
        // a new weight_map HashMap for each of the 199 complex signatures.
        let all_mappings: Vec<(Signature, Vec<Weight>, HashMap<Weight, Weight>)> = timeit!(
            "parser_dwa::weight_mappings",
            {
                let start_mappings = std::time::Instant::now();
                let all_mappings: Vec<(Signature, Vec<Weight>, HashMap<Weight, Weight>)> = complex_signatures.par_iter().map(|target_sig| {
                    let target_idx = SignatureIndex::new(target_sig);
                    let mapping = can_derive(&super_signature, &target_idx).expect("Super signature must derive target");
                    
                    // Pre-compute the weight mapping for this target signature
                    let weight_map: HashMap<Weight, Weight> = super_dwa_unique_weights.iter()
                        .map(|w| {
                            let mut accumulator = Weight::zeros();
                            let mut is_all = false;

                            for bit in w.iter_up_to_allow_expansion(mapping.len()) {
                                if let Some(target_w) = mapping.get(bit) {
                                    if target_w.is_all_fast() {
                                        is_all = true;
                                        break;
                                    }
                                    accumulator |= target_w;
                                }
                            }

                            let new_w = if is_all {
                                Weight::all()
                            } else {
                                accumulator
                            };
                            (w.clone(), new_w)
                        })
                        .collect();
                    
                    (target_sig.clone(), mapping, weight_map)
                }).collect();
                crate::debug!(5, "  Weight mappings: {:?}", start_mappings.elapsed());
                all_mappings
            }
        );

        // PARALLEL OPTIMIZATION: Specialize DWAs using pre-computed weight mappings
        let results: Vec<(Signature, NWA, DWAStats, DWAStats)> = timeit!(
            "parser_dwa::specialize_dw_as",
            {
                let start_specialize = std::time::Instant::now();
                let results: Vec<(Signature, NWA, DWAStats, DWAStats)> = all_mappings.par_iter().map(|(target_sig, _mapping, weight_map)| {
                    let mut derived_dwa = specialize_dwa_relative_with_map(&super_dwa, weight_map);
                    let before_stats = derived_dwa.stats();
                    // Skip expensive minimization - just prune
                    // Rely on final determinization/minimize to compress
                    derived_dwa.optimize(DwaOptimizeConfig::SpecializedSuper);
                    let after_stats = derived_dwa.stats();
                    (target_sig.clone(), NWA::from_dwa(&derived_dwa), before_stats, after_stats)
                }).collect();
                let mut before_total_stats = DWAStats {
                    states: 0,
                    transitions: 0,
                    unique_state_pairs: 0,
                    ranges: 0,
                    ranges_interned: 0,
                    transition_multiplicity_hist: BTreeMap::new(),
                };
                let mut after_total_stats = DWAStats {
                    states: 0,
                    transitions: 0,
                    unique_state_pairs: 0,
                    ranges: 0,
                    ranges_interned: 0,
                    transition_multiplicity_hist: BTreeMap::new(),
                };
                let mut reduced_count = 0usize;
                let mut max_state_delta = 0usize;
                let mut max_transition_delta = 0usize;
                let mut accumulate_stats = |total: &mut DWAStats, stats: &DWAStats| {
                    total.states += stats.states;
                    total.transitions += stats.transitions;
                    total.unique_state_pairs += stats.unique_state_pairs;
                    total.ranges += stats.ranges;
                    total.ranges_interned += stats.ranges_interned;
                    for (k, v) in &stats.transition_multiplicity_hist {
                        *total.transition_multiplicity_hist.entry(*k).or_insert(0) += v;
                    }
                };
                for (_, _, before_stats, after_stats) in &results {
                    accumulate_stats(&mut before_total_stats, before_stats);
                    accumulate_stats(&mut after_total_stats, after_stats);
                    let state_delta = before_stats.states.saturating_sub(after_stats.states);
                    let transition_delta = before_stats.transitions.saturating_sub(after_stats.transitions);
                    if state_delta > 0 || transition_delta > 0 {
                        reduced_count += 1;
                    }
                    if state_delta > max_state_delta {
                        max_state_delta = state_delta;
                    }
                    if transition_delta > max_transition_delta {
                        max_transition_delta = transition_delta;
                    }
                }
                let total_state_delta = before_total_stats.states.saturating_sub(after_total_stats.states);
                let total_transition_delta = before_total_stats.transitions.saturating_sub(after_total_stats.transitions);
                let reduction_pct = if before_total_stats.states == 0 {
                    0.0
                } else {
                    100.0 * (1.0 - after_total_stats.states as f64 / before_total_stats.states as f64)
                };
                crate::debug!(5, "  Specialization ({} DWAs): {:?}, before={}, after={} ({:.1}% reduction)", 
                    results.len(), start_specialize.elapsed(), before_total_stats, after_total_stats, reduction_pct);
                crate::debug!(5, "  Specialization reduction: reduced_dw_as={}/{}, total_state_delta={}, total_transition_delta={}, max_state_delta={}, max_transition_delta={}",
                    reduced_count,
                    results.len(),
                    total_state_delta,
                    total_transition_delta,
                    max_state_delta,
                    max_transition_delta,
                );
                for (idx, (_, _, before_stats, after_stats)) in results.iter().enumerate() {
                    crate::debug!(6, "  Specialized DWA #{}: before={}, after={}", idx, before_stats, after_stats);
                }
                results
            }
        );

        for (sig, nwa, _, _) in results {
            template_cache.borrow_mut().insert(sig, Arc::new(nwa));
        }

        super_dwa_opt = Some(super_dwa);
        super_signature_opt = Some(super_signature);
        super_dwa_unique_weights_opt = Some(super_dwa_unique_weights);
        });
    }
    // OPTIMIZATION END

    // Log template cache stats
    let template_cache_snapshot = template_cache.borrow();
    let template_sizes: Vec<usize> = template_cache_snapshot.values().map(|nwa| nwa.states.len()).collect();
    let total_template_states: usize = template_sizes.iter().sum();
    let max_template: usize = template_sizes.iter().copied().max().unwrap_or(0);
    let avg_template: f64 = total_template_states as f64 / template_sizes.len().max(1) as f64;
    crate::debug!(4, "Template cache: {} templates, {} total states, max={}, avg={:.1}", 
        template_cache_snapshot.len(), total_template_states, max_template, avg_template);
    drop(template_cache_snapshot);

    crate::debug!(4, "Finished DWA specialization");

    let states_arena = RefCell::new(NWAStates::default());
    let initial_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_states: vec![start] }
    };
    let initial_term_map: BTreeMap<Option<TerminalID>, Weight> = BTreeMap::from([(None, Weight::all())]);
    let initial_values_full: Vec<(usize, (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, Weight))> =
        reversed_nwa.body.start_states.iter().map(|&s| (s, (BTreeMap::from([(initial_body.clone(), initial_term_map.clone())]), Weight::all()))).collect();

    // Store (NWABody, Weight, node_id) - node_id is the state ID in reversed NWA where we collected this body
    // In symbol-heavy mode, we also collect tsid-specific bodies separately
    let final_bodies_arc: Arc<Mutex<Vec<(NWABody, Weight, StateID)>>> = Arc::new(Mutex::new(Vec::new()));
    
    // For symbol-heavy mode: collect (NWABody, Weight, tsid_label) for each tsid-labeled transition
    // These are transitions from root states to the original start state in the reversed NWA
    let tsid_bodies_arc: Arc<Mutex<Vec<(NWABody, Weight, Label)>>> = Arc::new(Mutex::new(Vec::new()));

    crate::debug!(4, "Beginning NWA traversal");

    let pass2_start = Instant::now();
    let pass2_profile = Arc::new(Pass2Profile::new());

    // Clone references for use in closures
    let tsid_bodies_for_process = tsid_bodies_arc.clone();
    let pass2_profile_for_process = pass2_profile.clone();
    let template_cache_ref = &template_cache;
    let super_dwa_opt_ref = &super_dwa_opt;
    let super_signature_opt_ref = &super_signature_opt;
    let super_dwa_unique_weights_opt_ref = &super_dwa_unique_weights_opt;
    let states_arena_ref = &states_arena;

    timeit!("parser_dwa::pass2_traversal", {
        nwa_special_map(
            &reversed_nwa, &traversal_data, initial_values_full,
            |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, Weight), edge_label, transitions| {
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
                    let next_tokens = current_tokens & weight;
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
                let mut changed = false;
                for (right_body, term_map2) in bodies2 {
                    let term_map1 = bodies1.entry(right_body.clone()).or_default();
                    for (term, weight2) in term_map2 {
                        let entry = term_map1.entry(term).or_insert_with(Weight::zeros);
                        if !weight2.is_subset_of(entry) {
                            *entry |= &weight2;
                            changed = true;
                        }
                    }
                }
                if !tokens2.is_subset_of(tokens1) {
                    *tokens1 |= &tokens2;
                    changed = true;
                }
                changed
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
            
            let process_start = Instant::now();
            pass2_profile_for_process
                .process_count
                .fetch_add(1, Ordering::Relaxed);

            let (nwa_bodies_map, tokens) = val;
            let bodies_count = nwa_bodies_map.len();
            let mut nwa_body = NWABody { start_states: vec![] };
            for (right_body, terminal_map) in &nwa_bodies_map {
                let canon_start = Instant::now();
                let (signature, concrete_weights) = canonicalize_bundle(terminal_map.clone());
                pass2_profile_for_process
                    .canonicalize_us
                    .fetch_add(canon_start.elapsed().as_micros() as u64, Ordering::Relaxed);

                let cached_nwa = {
                    let cache_lookup_start = Instant::now();
                    let cached = template_cache_ref.borrow().get(&signature).cloned();
                    pass2_profile_for_process.cache_lookup_us.fetch_add(
                        cache_lookup_start.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                    if let Some(nwa) = cached { nwa } else {
                    let dynamic_start = Instant::now();
                    let _rangeset_backend = WeightBackendOverride::new("rangeset");
                    crate::debug!(5, "Dynamic derivation for signature {:?}", signature);

                    let super_dwa = super_dwa_opt_ref
                        .as_ref()
                        .expect("Super DWA missing for dynamic derivation");
                    let super_signature = super_signature_opt_ref
                        .as_ref()
                        .expect("Super signature missing for dynamic derivation");
                    let super_dwa_unique_weights = super_dwa_unique_weights_opt_ref
                        .as_ref()
                        .expect("Super DWA weights missing for dynamic derivation");

                    let target_idx = SignatureIndex::new(&signature);
                    let mapping = can_derive(super_signature, &target_idx)
                        .expect("Super signature must derive target");

                    let weight_map: HashMap<Weight, Weight> = super_dwa_unique_weights
                        .iter()
                        .map(|w| {
                            let mut accumulator = Weight::zeros();
                            let mut is_all = false;

                            for bit in w.iter_up_to_allow_expansion(mapping.len()) {
                                if let Some(target_w) = mapping.get(bit) {
                                    if target_w.is_all_fast() {
                                        is_all = true;
                                        break;
                                    }
                                    accumulator |= target_w;
                                }
                            }

                            let new_w = if is_all { Weight::all() } else { accumulator };
                            (w.clone(), new_w)
                        })
                        .collect();

                    let mut derived_dwa = specialize_dwa_relative_with_map(super_dwa, &weight_map);
                    derived_dwa.optimize(DwaOptimizeConfig::SpecializedSuper);
                    let nwa = Arc::new(NWA::from_dwa(&derived_dwa));
                    pass2_profile_for_process.dynamic_derive_us.fetch_add(
                        dynamic_start.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                    let cache_insert_start = Instant::now();
                    template_cache_ref
                        .borrow_mut()
                        .insert(signature.clone(), nwa.clone());
                    pass2_profile_for_process.cache_insert_us.fetch_add(
                        cache_insert_start.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                    nwa
                    }
                };
                let cached_nwa = cached_nwa.as_ref();

                pass2_profile_for_process
                    .template_count
                    .fetch_add(1, Ordering::Relaxed);
                
                let template_size = cached_nwa.states.len();
                let count = INSTANTIATE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                TOTAL_TEMPLATE_STATES.fetch_add(template_size, std::sync::atomic::Ordering::Relaxed);
                if count % 10000 == 0 {
                    crate::debug!(4, "Template instantiation #{}: {} total states so far, template size {}, bodies_count: {}", 
                        count, TOTAL_TEMPLATE_STATES.load(std::sync::atomic::Ordering::Relaxed), template_size, bodies_count);
                }
                
                let mut states = states_arena.borrow_mut();
                let new_states_offset = states.len();
                let instantiate_start = Instant::now();
                let composed_body = instantiate_nwa_template_into(cached_nwa, &concrete_weights, &mut states, right_body);
                pass2_profile_for_process.instantiate_us.fetch_add(
                    instantiate_start.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );
                let range = new_states_offset..states.len();
                if !range.is_empty() {
                    let total_states = states.len();
                    let pass2_skip_cancellations_threshold = std::env::var("NWA_PASS2_SKIP_CANCELLATIONS_THRESHOLD")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    let pass2_skip_finality_threshold = std::env::var("NWA_PASS2_SKIP_FINALITY_FIXPOINT_THRESHOLD")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0);
                    let skip_cancellations = pass2_skip_cancellations_threshold > 0
                        && total_states >= pass2_skip_cancellations_threshold;
                    let skip_finality = pass2_skip_finality_threshold > 0
                        && total_states >= pass2_skip_finality_threshold;
                    let cancel_start = Instant::now();
                    if skip_cancellations {
                        crate::debug!(4, "Pass2: skipping cancellations for range {:?} (states={}, threshold={})", range, total_states, pass2_skip_cancellations_threshold);
                    } else {
                        apply_cancellations_range(&mut states, range.clone());
                    }
                    pass2_profile_for_process.apply_cancellations_us.fetch_add(
                        cancel_start.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                    let finality_start = Instant::now();
                    if skip_finality {
                        crate::debug!(4, "Pass2: skipping finality fixpoint for range {:?} (states={}, threshold={})", range, total_states, pass2_skip_finality_threshold);
                    } else {
                        apply_finality_fixpoint_range(&mut states, range.clone());
                    }
                    pass2_profile_for_process.apply_finality_us.fetch_add(
                        finality_start.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                    let remove_start = Instant::now();
                    remove_negative_transitions_range(&mut states, range);
                    pass2_profile_for_process.remove_negative_us.fetch_add(
                        remove_start.elapsed().as_micros() as u64,
                        Ordering::Relaxed,
                    );
                    // Note: remove_redundant_default_transitions is NOT called here because it 
                    // requires a global pass over all states. It's called once at the end.
                }
                let union_start = Instant::now();
                nwa_body = NWABody::union(&nwa_body, &composed_body);
                pass2_profile_for_process.union_us.fetch_add(
                    union_start.elapsed().as_micros() as u64,
                    Ordering::Relaxed,
                );
            }
            
            // In symbol-heavy mode, check if this is a root state (has tsid-labeled edges to original_start_state)
            // If so, collect the body for each tsid
            if is_symbol_heavy {
                if let Some(tsid_edges) = outgoing_tsid_edges.get(&node_id) {
                    for (tsid_label, edge_weight) in tsid_edges {
                        let intersection_w = &tokens & edge_weight;
                        if !intersection_w.is_empty() && !nwa_body.start_states.is_empty() {
                            let final_w = intersection_w;
                            crate::debug!(6, "Collecting tsid body at root {} for label {} with {} tokens", 
                                node_id, tsid_label, final_w.len());
                            let tsid_collect_start = Instant::now();
                            let mut tb = tsid_bodies_for_process.lock().unwrap();
                            tb.push((nwa_body.clone(), final_w, *tsid_label));
                            pass2_profile_for_process.tsid_collect_us.fetch_add(
                                tsid_collect_start.elapsed().as_micros() as u64,
                                Ordering::Relaxed,
                            );
                        }
                    }
                }
            }
            
            // Check if this is a final state in the reversed NWA (original start state or root state)
            // In symbol-heavy mode, we handle the original_start_state specially via tsid_bodies
            // so don't collect it here
            let has_final_weight = reversed_nwa.states[node_id].final_weight.is_some();
            crate::debug!(7, "Process node_id={}, is_symbol_heavy={}, has_final_weight={}", node_id, is_symbol_heavy, has_final_weight);
            let should_collect = if is_symbol_heavy {
                // In symbol-heavy mode, only collect for states OTHER than original_start_state
                // (the original start is handled via tsid-labeled transitions)
                node_id != original_start_state && reversed_nwa.states[node_id].final_weight.is_some()
            } else {
                reversed_nwa.states[node_id].final_weight.is_some()
            };
            
            if should_collect {
                if let Some(fw) = &reversed_nwa.states[node_id].final_weight {
                    let intersection_w = &tokens & fw;
                    crate::debug!(7, "Final body candidate: node_id={}, tokens_len={}, tokens_ranges={}, fw_len={}, fw_ranges={}, intersection_len={}, intersection_ranges={}", 
                        node_id, tokens.len(), tokens.ranges_len(), fw.len(), fw.ranges_len(), intersection_w.len(), intersection_w.ranges_len());
                    if !intersection_w.is_empty() {
                        let final_w = intersection_w;
                        let final_collect_start = Instant::now();
                        let mut fb = final_bodies_arc.lock().unwrap();
                        fb.push((nwa_body.clone(), final_w, node_id));
                        pass2_profile_for_process.final_collect_us.fetch_add(
                            final_collect_start.elapsed().as_micros() as u64,
                            Ordering::Relaxed,
                        );
                    }
                }
            }
            
            let process_elapsed = process_start.elapsed();
            pass2_profile_for_process.process_total_us.fetch_add(
                process_elapsed.as_micros() as u64,
                Ordering::Relaxed,
            );

            if !tokens.is_empty() {
                let mut next_body_map = BTreeMap::new(); next_body_map.insert(nwa_body, BTreeMap::new());
                Some((next_body_map, tokens))
            } else { None }
            },
        );
        pass2_profile.log();
        crate::debug!(4, "Pass 2 (nwa_special_map) in {:?}", pass2_start.elapsed());
        // Drop the process closure's reference to tsid_bodies
        drop(tsid_bodies_for_process);

        crate::debug!(4, "Finished Pass 2");
    });
    let final_bodies = Arc::try_unwrap(final_bodies_arc).unwrap().into_inner().unwrap();
    let tsid_bodies = Arc::try_unwrap(tsid_bodies_arc).unwrap().into_inner().unwrap();
    let avg_template_size = states_arena.borrow().len() as f64 / (final_bodies.len() + tsid_bodies.len()).max(1) as f64;
    crate::debug!(4, "Collected {} final bodies, {} tsid bodies, states_arena has {} states (avg {:.0} states/body)", 
        final_bodies.len(), tsid_bodies.len(), states_arena.borrow().len(), avg_template_size);
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
    crate::debug!(3, "Combined NWA before determinization: {}, is_symbol_heavy={}", 
        combined_nwa.stats(), is_symbol_heavy);
    let mut final_dwa = timeit!("parser_dwa::finalize_and_determinize", {
        finalize_and_optimize_and_determinize(parser, combined_nwa)
    });
    // SKIP final minimization to test performance impact
    // final_dwa.minimize();
    crate::debug!(4, "Parser DWA construction complete. Stats: {}", final_dwa.stats());
    if let Some(avg_path_len) = final_dwa.average_path_length() {
        crate::debug!(4, "Parser DWA average path length: {:.2}", avg_path_len);
    }

    crate::debug!(5, "build_parser_dwa: end");
    final_dwa
}

/// Deprecated alias for build_parser_dwa
#[deprecated(since = "0.3.0", note = "Use build_parser_dwa instead")]
pub fn precompute4(parser: &GLRParser, terminal_nwa: &NWA) -> DWA {
    build_parser_dwa(parser, terminal_nwa)
}

pub fn precompute_token_bvs_and_signatures(reversed_nwa: &NWA, traversal_data: &NwaTraversalData, initial_values: Vec<(StateID, Weight)>) -> (HashMap<StateID, Weight>, HashSet<Signature>) {
    let node_tokens: Arc<Mutex<HashMap<StateID, Weight>>> = Arc::new(Mutex::new(HashMap::new()));
    let signatures: Arc<Mutex<HashSet<Signature>>> = Arc::new(Mutex::new(HashSet::new()));

    let node_tokens_clone = node_tokens.clone();
    let signatures_clone = signatures.clone();

    nwa_special_map(reversed_nwa, traversal_data, initial_values,
        move |tokens: &Weight, _edge_label, transitions| {
            let mut results = Vec::new();
            for (dest_id, weight) in transitions {
                let next = tokens & weight;
                if !next.is_empty() { results.push((*dest_id, next)); }
            }
            results
        },
        |t1, t2| {
            if t2.is_subset_of(t1) {
                false
            } else {
                *t1 |= &t2;
                true
            }
        },
        move |node_id, tokens| {
            node_tokens_clone.lock().unwrap().insert(node_id, tokens.clone());
            let mut bundles_by_dest: HashMap<StateID, BTreeMap<Option<TerminalID>, Weight>> = HashMap::new();
            let state = &reversed_nwa.states[node_id];
            for (label, targets) in &state.transitions {
                let term = Some(TerminalID(*label as usize));
                for (v, w) in targets {
                    let combined = &tokens & w;
                    if !combined.is_empty() {
                        bundles_by_dest.entry(*v).or_default().insert(term, w.clone());
                    }
                }
            }
            for (v, w) in &state.epsilons {
                let combined = &tokens & w;
                if !combined.is_empty() {
                    bundles_by_dest.entry(*v).or_default().insert(None, w.clone());
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
    crate::debug!(4, "Pruning continuations from final states for NWA with {}...", combined_nwa.stats());
    let prune_final_start = std::time::Instant::now();
    combined_nwa.subtract_final_weights_from_outgoing();
    crate::debug!(5, "subtract_final_weights_from_outgoing in {:?}", prune_final_start.elapsed());
    crate::debug!(4, "Pruned continuations from final states. NWA now {}.", combined_nwa.stats());
    
    // After pruning continuations, some transitions may become empty and states may become unreachable.
    // Prune dead ends before determinization to reduce the NWA size significantly.
    let before_prune = combined_nwa.stats();
    let prune_start = std::time::Instant::now();
    combined_nwa.prune_dead_ends();
    let prune_dead_time = prune_start.elapsed();
    let prune_unreachable_start = std::time::Instant::now();
    combined_nwa.prune_unreachable();
    let prune_unreachable_time = prune_unreachable_start.elapsed();
    crate::debug!(5, "prune_dead_ends in {:?}, prune_unreachable in {:?}", prune_dead_time, prune_unreachable_time);
    crate::debug!(4, "After pruning dead ends: NWA {} -> {}", 
        before_prune, combined_nwa.stats());

    // NWA minimization is expensive - skip it for now
    // The DWA minimization will handle the reduction anyway
    // let nwa_states = combined_nwa.states.len();
    // if nwa_states > 100_000 {
    //     let minimize_start = std::time::Instant::now();
    //     combined_nwa.minimize_internal();
    //     let minimize_time = minimize_start.elapsed();
    //     crate::debug!(4, "NWA minimization: {} -> {} states in {:.2?}", 
    //         nwa_states, combined_nwa.states.len(), minimize_time);
    // }
    
    let disable_minimize = std::env::var("PARSER_DWA_MINIMIZE")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false);

    if disable_minimize {
        crate::debug!(4, "Parser DWA minimize disabled (PARSER_DWA_MINIMIZE=0)");
        let det_start = std::time::Instant::now();
        let dwa = combined_nwa.determinize();
        crate::debug!(5, "determinize(Parser) in {:?}", det_start.elapsed());
        crate::debug!(4, "Parser DWA determinize complete. {}", dwa.stats());
        return dwa;
    }

    crate::debug!(4, "Running parser DWA minimize");
    // Use unified determinize_and_minimize with "Parser" profile
    // Pipeline: determinize → prune_dead_ends → minimize
    let det_min_start = std::time::Instant::now();
    let dwa = combined_nwa.determinize_and_minimize(DeterminizeAndMinimizeProfile::Parser);
    crate::debug!(5, "determinize_and_minimize(Parser) in {:?}", det_min_start.elapsed());
    crate::debug!(4, "Parser DWA minimization complete. {}", dwa.stats());
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
        for idx in w.iter_up_to_allow_expansion(ordered_weights.len()) {
            if let Some(concrete_w) = ordered_weights.get(idx) {
                if matches!(concrete, Weight::Factorized(_)) {
                    if let Weight::RangeSet(rsb) = concrete_w {
                        let converted = Weight::from_rsb(rsb.inner().clone());
                        concrete |= &converted;
                        continue;
                    }
                }
                concrete |= concrete_w;
            }
        }
        union_cache.insert(w.clone(), concrete.clone());
        concrete
    };

    for old_state in &template.states.0 {
        let mut new_state = crate::dwa_i32::nwa::NWAState::default();
        
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