use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::env;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use chrono::Local;
use kdam::{tqdm, BarExt};

use crate::constraint::{
    LLMTokenBV, PrecomputeNode1Index, Trie1GodWrapper,
};
use crate::datastructures::trie::Trie2Index;
use crate::glr::parser::GLRParser;
use crate::precompute4::nwa_optimizations::{
    prune_continuations_from_final_states, simplify_default_transitions,
};
use crate::precompute4::resolve_negatives::{
    apply_cancellations, apply_finality_fixpoint, remove_negative_transitions,
};
use crate::precompute4::template_nwa::{
    build_ignore_terminal_dwa, build_template_dwas,
};
use crate::precompute4::weighted_automata::{
    common::Label, determinization_rustfst::determinize_nwa_to_dwa, DWA, NWA, NWABody,
    NWAStateID, NWAStates, SimpleBitset, StateID, Weight,
};
use crate::tokenizer::TokenizerStateID;
use crate::types::{TerminalID, TerminalID as GrammarTokenID};

struct SimplifyRustfstConfig {
    rm_epsilon: bool,
    #[allow(dead_code)]
    determinize: bool,
}

impl SimplifyRustfstConfig {
    fn default() -> Self {
        Self {
            rm_epsilon: false,
            determinize: false,
        }
    }
    fn with_rm_epsilon(mut self, val: bool) -> Self {
        self.rm_epsilon = val;
        self
    }
}

impl NWA {
    pub fn determinize_to_dwa_with_rustfst(&self) -> DWA {
        determinize_nwa_to_dwa(self)
    }
    pub fn simplify_rustfst(&mut self) {
        self.simplify();
    }
    pub fn simplify_rustfst_with_config(&mut self, _config: SimplifyRustfstConfig) {
        self.simplify();
    }
}

// Re-export for backward compatibility
pub use crate::precompute4::template_nwa::FullDWABuildError;

pub type Precomputed4 = DWA;
type Signature = Vec<Vec<Option<TerminalID>>>;

// ---------------------------------------------------------------------------
// NWA Traversal Data & Algorithms
// ---------------------------------------------------------------------------

pub struct NwaTraversalData {
    pub comp_id: Vec<usize>,   // StateID -> SCC ID
    pub sccs: Vec<Vec<usize>>, // SCC ID -> Vec<StateID>
    pub topo: Vec<usize>,      // SCC IDs in topological order
}

impl NWA {
    /// Creates a new NWA with all edges reversed.
    /// Useful for backward propagation (e.g. propagating valid tokens from End states back to Start).
    pub fn reverse(&self) -> NWA {
        let mut reversed = NWA::new();
        reversed.states.0.clear();
        for _ in 0..self.states.len() {
            reversed.add_state();
        }

        for (u, state) in self.states.0.iter().enumerate() {
            // Reverse labeled transitions: u -> v becomes v -> u
            for (label, targets) in &state.transitions {
                for (v, w) in targets {
                    reversed
                        .add_transition(*v, *label, u, w.clone())
                        .unwrap();
                }
            }
            // Reverse epsilon transitions
            for (v, w) in &state.epsilons {
                reversed.add_epsilon(*v, u, w.clone());
            }
        }
        reversed
    }

    /// Computes SCCs and topological order for the NWA.
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
                for (v, _) in targets {
                    neighbors.push(*v);
                }
            }
            for (v, _) in &state.epsilons {
                neighbors.push(*v);
            }

            for v in neighbors {
                let v_scc = comp_id[v];
                if u_scc != v_scc {
                    if !scc_adj[u_scc].contains(&v_scc) {
                        scc_adj[u_scc].insert(v_scc);
                        indeg[v_scc] += 1;
                    }
                }
            }
        }

        let mut topo = Vec::with_capacity(scc_count);
        let mut q = VecDeque::new();
        for i in 0..scc_count {
            if indeg[i] == 0 {
                q.push_back(i);
            }
        }

        while let Some(u) = q.pop_front() {
            topo.push(u);
            for &v in &scc_adj[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    q.push_back(v);
                }
            }
        }

        NwaTraversalData {
            comp_id,
            sccs,
            topo,
        }
    }
}

fn compute_sccs(nwa: &NWA) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = nwa.states.len();
    let mut adj = vec![Vec::new(); n];
    let mut rev_adj = vec![Vec::new(); n];

    for (u, state) in nwa.states.0.iter().enumerate() {
        let mut neighbors = Vec::new();
        for targets in state.transitions.values() {
            for (v, _) in targets {
                neighbors.push(*v);
            }
        }
        for (v, _) in &state.epsilons {
            neighbors.push(*v);
        }

        for v in neighbors {
            adj[u].push(v);
            rev_adj[v].push(u);
        }
    }

    // 1. DFS order
    let mut order = Vec::new();
    let mut visited = vec![false; n];
    for i in 0..n {
        if !visited[i] {
            let mut stack = vec![(i, false)];
            while let Some((u, processed)) = stack.pop() {
                if processed {
                    order.push(u);
                } else {
                    if visited[u] {
                        continue;
                    }
                    visited[u] = true;
                    stack.push((u, true));
                    for &v in &adj[u] {
                        if !visited[v] {
                            stack.push((v, false));
                        }
                    }
                }
            }
        }
    }

    // 2. Reverse DFS
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

/// A generalized fixpoint traversal for NWA, analogous to `Trie::special_map_grouped`.
/// It processes SCCs in topological order and iterates within SCCs until convergence.
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
    let mut values: HashMap<StateID, V> = HashMap::new();
    let mut stopped_nodes: HashSet<StateID> = HashSet::new();

    for (state, v) in initial_values {
        values
            .entry(state)
            .and_modify(|old| merge(old, v.clone()))
            .or_insert(v);
    }

    let mut in_queue = HashSet::new();
    let mut pb = tqdm!(
        total = nwa.states.len(),
        desc = "NWA Traversal",
        disable = !crate::profiler::PROGRESS_BAR_ENABLED,
        leave = false
    );

    for &scc_idx in &traversal_data.topo {
        let scc_nodes = &traversal_data.sccs[scc_idx];
        let mut local_queue: VecDeque<StateID> = VecDeque::new();

        // Seed local queue
        for &u in scc_nodes {
            if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                local_queue.push_back(u);
                in_queue.insert(u);
            }
        }

        if local_queue.is_empty() {
            continue;
        }

        while let Some(u) = local_queue.pop_front() {
            in_queue.remove(&u);
            let _ = pb.update(1);

            if stopped_nodes.contains(&u) {
                continue;
            }

            let agg_v = match values.remove(&u) {
                Some(v) => v,
                None => continue,
            };

            let proceed_val = match process(u, agg_v.clone()) {
                Some(val) => val,
                None => {
                    stopped_nodes.insert(u);
                    continue;
                }
            };

            let state = &nwa.states[u];

            // Propagate via Epsilons (Label = None)
            if !state.epsilons.is_empty() {
                let res = step(&proceed_val, None, &state.epsilons);
                for (v, new_v) in res {
                    if stopped_nodes.contains(&v) {
                        continue;
                    }
                    values
                        .entry(v)
                        .and_modify(|old| merge(old, new_v.clone()))
                        .or_insert(new_v);

                    if traversal_data.comp_id[v] == scc_idx {
                        if !in_queue.contains(&v) {
                            local_queue.push_back(v);
                            in_queue.insert(v);
                        }
                    }
                }
            }

            // Propagate via Transitions (Label = Some)
            for (&label, targets) in &state.transitions {
                let res = step(&proceed_val, Some(label), targets);
                for (v, new_v) in res {
                    if stopped_nodes.contains(&v) {
                        continue;
                    }
                    values
                        .entry(v)
                        .and_modify(|old| merge(old, new_v.clone()))
                        .or_insert(new_v);

                    if traversal_data.comp_id[v] == scc_idx {
                        if !in_queue.contains(&v) {
                            local_queue.push_back(v);
                            in_queue.insert(v);
                        }
                    }
                }
            }

            // Fixpoint check: if new values arrived at u while processing
            if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                if !in_queue.contains(&u) {
                    local_queue.push_back(u);
                    in_queue.insert(u);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Signature Logic (Moved from precompute4 logic)
// ---------------------------------------------------------------------------

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
        Self {
            term_to_group: map,
            total_terms: count,
        }
    }

    fn get_group(&self, term: &Option<TerminalID>) -> Option<usize> {
        self.term_to_group.get(term).cloned()
    }
}

fn can_derive(parent: &Signature, child_index: &SignatureIndex) -> Option<Vec<Weight>> {
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
                return None;
            }
        }

        if let Some(g) = expected_g {
            mapping.push(Weight::from_item(g));
            matched_terms += group.len();
        } else {
            mapping.push(Weight::zeros());
        }
    }

    if matched_terms != child_index.total_terms {
        return None;
    }

    Some(mapping)
}

fn specialize_dwa_relative(parent_dwa: &DWA, mapping: &[Weight]) -> DWA {
    let mut specialized_dwa = parent_dwa.clone();
    let mut cache: HashMap<Weight, SimpleBitset> = HashMap::new();

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
            if fw.is_empty() {
                state.final_weight = None;
            }
        }
        if let Some(sw) = &mut state.state_weight {
            *sw = map_weight(sw);
            if sw.is_empty() {
                state.state_weight = None;
            }
        }
        for tw in state.trans_weights.values_mut() {
            *tw = map_weight(tw);
        }
        state.trans_weights.retain(|_, w| !w.is_empty());
        state
            .transitions
            .retain(|k, _| state.trans_weights.contains_key(k));
    }

    specialized_dwa
}

// ---------------------------------------------------------------------------
// Conversion Helpers (Precompute1 -> NWA)
// ---------------------------------------------------------------------------

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
    if guard.value.end {
        nwa.states[sid].final_weight = Some(Weight::all());
    }
    let children = guard.children().clone();
    drop(guard);

    for (edge_key, child_map) in children {
        for (child_idx, edge_bv) in child_map {
            let child_sid = convert_node_to_nwa(child_idx, god, nwa, cache);
            let trans_w: Weight = edge_bv.into();
            if let Some(label) = edge_key {
                nwa.add_transition(sid, label.0 as Label, child_sid, trans_w)
                    .unwrap();
            } else {
                nwa.add_epsilon(sid, child_sid, trans_w);
            }
        }
    }
    sid
}

pub fn convert_precompute1_to_nwa(
    precomputed1: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) -> NWA {
    let mut nwa = NWA::new();
    nwa.states.0.clear();
    let start_state = nwa.states.add_state();
    nwa.body.start_state = start_state;

    let mut node_cache = HashMap::new();

    for (sid, root_idx) in precomputed1 {
        let root_state = convert_node_to_nwa(*root_idx, trie1_god, &mut nwa, &mut node_cache);
        nwa.add_transition(start_state, sid.0 as Label, root_state, Weight::all())
            .unwrap();
    }
    nwa
}

fn canonicalize_bundle(
    terminal_map: BTreeMap<Option<TerminalID>, Weight>,
) -> (Signature, Vec<Weight>) {
    let mut weight_groups: HashMap<Weight, Vec<Option<TerminalID>>> = HashMap::new();
    for (term, weight) in terminal_map {
        if !weight.is_empty() {
            weight_groups.entry(weight).or_default().push(term);
        }
    }
    let mut groups_vec: Vec<(Weight, Vec<Option<TerminalID>>)> =
        weight_groups.into_iter().collect();
    for (_, terms) in &mut groups_vec {
        terms.sort();
    }
    groups_vec.sort_by(|a, b| a.1.cmp(&b.1));

    let signature: Vec<Vec<Option<TerminalID>>> = groups_vec
        .iter()
        .map(|(_, terms)| terms.clone())
        .collect();
    let concrete_weights: Vec<Weight> = groups_vec.into_iter().map(|(w, _)| w).collect();
    (signature, concrete_weights)
}

// ---------------------------------------------------------------------------
// Main Precompute4 Logic
// ---------------------------------------------------------------------------

pub fn precompute4(
    parser: &GLRParser,
    input_nwa: &NWA,
    // Argument kept for compatibility if needed, though unused in this version
    _max_llm_token_id: usize,
) -> DWA {
    crate::debug!(3, "Starting precompute4 (DWA construction)");

    // 1. Build template DWAs
    let now = Instant::now();
    let template_dwas = match build_template_dwas(parser) {
        Ok(m) => m,
        Err(e) => panic!("Failed to build template DWAs: {:?}", e),
    };
    let ignore_dwa = build_ignore_terminal_dwa();
    crate::debug!(
        3,
        "Built {} template DWAs in {:?}",
        template_dwas.len(),
        now.elapsed()
    );

    // 2. Reverse NWA for backward propagation
    // In the original logic, we propagated from "End" (leaf of Trie) backwards.
    // In NWA, states with `final_weight` are the "End".
    let reversed_nwa = input_nwa.reverse();
    let traversal_data = reversed_nwa.compute_traversal_data();

    // Identify 'roots' for reverse traversal (states with final weights in original NWA)
    let initial_tokens = LLMTokenBV::max_ones();
    let mut initial_values_bv = Vec::new();
    for (id, state) in input_nwa.states.0.iter().enumerate() {
        if state.final_weight.is_some() {
            initial_values_bv.push((id, initial_tokens.clone()));
        }
    }

    // Pass 1: Token propagation and Signature collection
    let start_pass1 = Instant::now();
    let (node_tokens, unique_signatures) = precompute_token_bvs_and_signatures(
        &reversed_nwa,
        &traversal_data,
        initial_values_bv,
    );
    crate::debug!(
        3,
        "Pass 1: Tokens & Signatures ({} sigs, {:.2?})",
        unique_signatures.len(),
        start_pass1.elapsed()
    );

    // 3. Build Super DWA / Template Derivation Pool

    // 3. Build Super DWA / Template Derivation Pool
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
            }
            None => &ignore_dwa,
        };
        let mut weighted_dwa = template_dwa.clone();
        weighted_dwa.apply_weight_inplace(&weight);
        let nwa = NWA::from_dwa(&weighted_dwa);
        let (start, _) = super_nwa_states.copy_subgraph_from(&nwa.states, nwa.body.start_state);
        super_nwa_states.add_epsilon(super_nwa_start, start, Weight::all());
    }

    let mut super_nwa = NWA {
        states: super_nwa_states,
        body: NWABody {
            start_state: super_nwa_start,
        },
    };
    super_nwa.simplify();
    let mut super_dwa = super_nwa.determinize_to_dwa();
    super_dwa.simplify();

    // Template Cache
    let mut template_cache = HashMap::new();
    let super_signature: Signature = bit_to_term.iter().map(|t| vec![*t]).collect();
    let mut pool: Vec<(Signature, DWA)> = vec![(super_signature, super_dwa.clone())];
    let mut signatures_vec: Vec<Signature> = unique_signatures.into_iter().collect();
    signatures_vec.sort_by(|a, b| {
        let groups_a = a.len();
        let groups_b = b.len();
        if groups_a != groups_b {
            return groups_b.cmp(&groups_a);
        }
        let terms_a: usize = a.iter().map(|g| g.len()).sum();
        let terms_b: usize = b.iter().map(|g| g.len()).sum();
        terms_b.cmp(&terms_a)
    });

    for target_sig in signatures_vec {
        let target_idx = SignatureIndex::new(&target_sig);
        let mut best_parent: Option<(usize, Vec<Weight>)> = None;
        let mut best_score = usize::MAX;
        for (p_idx, (p_sig, _)) in pool.iter().enumerate() {
            if let Some(mapping) = can_derive(p_sig, &target_idx) {
                let score = p_sig.len();
                if score < best_score {
                    best_score = score;
                    best_parent = Some((p_idx, mapping));
                }
            }
        }
        let (parent_idx, mapping) = best_parent
            .expect("Super signature should always be a valid parent");
        let parent_dwa = &pool[parent_idx].1;
        let mut derived_dwa = specialize_dwa_relative(parent_dwa, &mapping);
        derived_dwa.simplify();
        template_cache.insert(target_sig.clone(), NWA::from_dwa(&derived_dwa));
        pool.push((target_sig, derived_dwa));
    }

    // Pass 2: Build Final NWA
    let states_arena = RefCell::new(NWAStates::default());

    // Initial values for Pass 2
    let initial_nwa_body = {
        let mut states = states_arena.borrow_mut();
        let start = states.add_state();
        states[start].final_weight = Some(Weight::all());
        NWABody { start_state: start }
    };
    let initial_term_map: BTreeMap<Option<TerminalID>, Weight> =
        BTreeMap::from([(None, Weight::all())]);
    let initial_body_map_full = BTreeMap::from([(initial_nwa_body, initial_term_map)]);

    let mut initial_values_full = Vec::new();
    for (id, state) in input_nwa.states.0.iter().enumerate() {
        if state.final_weight.is_some() {
            if let Some(tokens) = node_tokens.get(&id) {
                initial_values_full.push((
                    id,
                    (initial_body_map_full.clone(), tokens.clone()),
                ));
            }
        }
    }

    // We need to capture the final bodies computed at the "root states" of the original input NWA.
    // The `input_nwa.body.start_state` transitions point to these roots.
    let start_state_id = input_nwa.body.start_state;
    let mut root_to_tokenizer_ids: HashMap<NWAStateID, Vec<TokenizerStateID>> = HashMap::new();
    for (label, targets) in &input_nwa.states[start_state_id].transitions {
        for (root_state_id, _) in targets {
            root_to_tokenizer_ids
                .entry(*root_state_id)
                .or_default()
                .push(TokenizerStateID(*label as usize));
        }
    }

    let root_to_tok = Arc::new(root_to_tokenizer_ids);
    let final_bodies_arc = Arc::new(Mutex::new(BTreeMap::new()));

    nwa_special_map(
        &reversed_nwa,
        &traversal_data,
        initial_values_full,
        // step
        |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV), edge_label, transitions| {
            let (current_bodies, current_tokens) = current_val;
            // Convert i32 label back to TerminalID
            let terminal_id = edge_label.map(|l| TerminalID(l as usize));
            let mut results = Vec::new();

            // In reversed NWA, transitions are (source, weight) but variable name is dest_id.
            // Meaning edge u->v in original is v->u in reversed.
            // transitions contains the 'u's.
            for (dest_id, weight) in transitions {
                // weight corresponds to edge_bv
                let edge_bv_tokens: LLMTokenBV = weight.clone().into();
                let next_tokens = current_tokens & &edge_bv_tokens;
                if next_tokens.is_empty() {
                    continue;
                }

                let mut terminal_map = BTreeMap::new();
                terminal_map.insert(terminal_id, weight.clone());
                let mut body_map = BTreeMap::new();
                for body in current_bodies.keys() {
                    body_map.insert(*body, terminal_map.clone());
                }
                results.push((*dest_id, (body_map, next_tokens)));
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
                    *term_map1
                        .entry(term)
                        .or_insert_with(Weight::zeros) |= &weight2;
                }
            }
            *tokens1 |= &tokens2;
        },
        // process
        |node_idx, val| {
            let (nwa_bodies_map, tokens) = val;
            let mut nwa_body = {
                let mut states = states_arena.borrow_mut();
                let start = states.add_state();
                NWABody { start_state: start }
            };

            for (right_body, terminal_map) in nwa_bodies_map {
                let (signature, concrete_weights) = canonicalize_bundle(terminal_map);
                let template_nwa = template_cache
                    .get(&signature)
                    .expect("Template must exist");

                let mut states = states_arena.borrow_mut();
                let (left_body_start, remap) = instantiate_nwa_template_into_arena(
                    template_nwa,
                    &concrete_weights,
                    &mut states,
                );
                let new_states_filter: HashSet<NWAStateID> = remap.values().cloned().collect();
                let left_body = NWABody {
                    start_state: left_body_start,
                };
                let composed_body = NWA::_concatenate_components(
                    &mut states,
                    &left_body,
                    &right_body,
                    &Weight::all(),
                );

                if !new_states_filter.is_empty() {
                    apply_cancellations(&mut states, &new_states_filter);
                    apply_finality_fixpoint(&mut states, &new_states_filter);
                    remove_negative_transitions(&mut states, &new_states_filter);
                }
                nwa_body = NWA::union_components(&mut states, &nwa_body, &composed_body);
            }

            // Capture results if this node is a root for a tokenizer state
            if !tokens.is_empty() {
                if let Some(tok_ids) = root_to_tok.get(&node_idx) {
                    let mut fb = final_bodies_arc.lock().unwrap();
                    for tid in tok_ids {
                        fb.insert(*tid, nwa_body.clone());
                    }
                }

                let mut next_body_map: BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>> = BTreeMap::new();
                next_body_map.insert(nwa_body, BTreeMap::new());
                Some((next_body_map, tokens))
            } else {
                None
            }
        },
    );

    crate::debug!(3, "Finished Pass 2");

    // Combine collected final bodies
    let final_bodies = Arc::try_unwrap(final_bodies_arc)
        .unwrap()
        .into_inner()
        .unwrap();

    let mut combined_nwa_states = states_arena.into_inner();
    let combined_start_state = combined_nwa_states.add_state();
    for (tok_id, body) in final_bodies {
        let label = tok_id.0 as Label;
        combined_nwa_states
            .add_transition(
                combined_start_state,
                label,
                body.start_state,
                Weight::all(),
            )
            .unwrap();
    }
    let combined_nwa = NWA {
        states: combined_nwa_states,
        body: NWABody {
            start_state: combined_start_state,
        },
    };

    let final_dwa = resolve_negatives_and_optimize_and_determinize(parser, combined_nwa);
    crate::debug!(3, "Precomputation complete");
    final_dwa
}

fn precompute_token_bvs_and_signatures(
    reversed_nwa: &NWA,
    traversal_data: &NwaTraversalData,
    initial_values: Vec<(StateID, LLMTokenBV)>,
) -> (HashMap<StateID, LLMTokenBV>, HashSet<Signature>) {
    let node_tokens: Arc<Mutex<HashMap<StateID, LLMTokenBV>>> =
        Arc::new(Mutex::new(HashMap::new()));
    let signatures: Arc<Mutex<HashSet<Signature>>> = Arc::new(Mutex::new(HashSet::new()));

    nwa_special_map(
        reversed_nwa,
        traversal_data,
        initial_values,
        // step: propagate tokens
        |tokens: &LLMTokenBV, _edge_label, transitions| {
            let mut results = Vec::new();
            for (dest_id, weight) in transitions {
                let edge_bv: LLMTokenBV = weight.clone().into();
                let next = tokens & &edge_bv;
                if !next.is_empty() {
                    results.push((*dest_id, next));
                }
            }
            results
        },
        // merge
        |t1, t2| {
            *t1 |= &t2;
        },
        // process
        |node_id, tokens| {
            node_tokens.lock().unwrap().insert(node_id, tokens.clone());

            // Collect signatures from outgoing edges in reversed nwa
            let mut bundles_by_dest: HashMap<
                StateID,
                BTreeMap<Option<TerminalID>, Weight>,
            > = HashMap::new();
            let state = &reversed_nwa.states[node_id];

            for (label, targets) in &state.transitions {
                let term = Some(TerminalID(*label as usize));
                for (v, w) in targets {
                    let edge_bv: LLMTokenBV = w.clone().into();
                    let combined = &tokens & &edge_bv;
                    if !combined.is_empty() {
                        let w_weight = Weight::from_rsb(edge_bv.inner.as_ref().clone());
                        bundles_by_dest
                            .entry(*v)
                            .or_default()
                            .insert(term, w_weight);
                    }
                }
            }
            for (v, w) in &state.epsilons {
                let edge_bv: LLMTokenBV = w.clone().into();
                let combined = &tokens & &edge_bv;
                if !combined.is_empty() {
                    let w_weight = Weight::from_rsb(edge_bv.inner.as_ref().clone());
                    bundles_by_dest
                        .entry(*v)
                        .or_default()
                        .insert(None, w_weight);
                }
            }

            let mut sigs = signatures.lock().unwrap();
            for (_, bundle) in bundles_by_dest {
                let (sig, _) = canonicalize_bundle(bundle);
                sigs.insert(sig);
            }

            Some(tokens)
        },
    );

    let final_tokens = Arc::try_unwrap(node_tokens)
        .unwrap()
        .into_inner()
        .unwrap();
    let final_sigs = Arc::try_unwrap(signatures)
        .unwrap()
        .into_inner()
        .unwrap();
    (final_tokens, final_sigs)
}

fn resolve_negatives_and_optimize_and_determinize(
    parser: &GLRParser,
    mut combined_nwa: NWA,
) -> DWA {
    combined_nwa.simplify_rustfst();
    prune_continuations_from_final_states(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    simplify_default_transitions(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    combined_nwa.simplify();
    simplify_remove_epsilon(&mut combined_nwa);
    combined_nwa = NWA::from_dwa(&combined_nwa._determinize());
    combined_nwa.simplify_rustfst();
    let mut final_dwa = combined_nwa.determinize_to_dwa();
    final_dwa.minimize_with_rustfst();
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
