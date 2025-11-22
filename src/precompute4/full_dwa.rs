use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use kdam::{tqdm, BarExt};

use crate::constraint::LLMTokenBV;
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::precompute4::nwa_optimizations::{prune_continuations_from_final_states, simplify_default_transitions};
use crate::precompute4::resolve_negatives::{apply_cancellations, apply_finality_fixpoint, remove_negative_transitions};
use crate::precompute4::template_nwa::{build_ignore_terminal_dwa, build_template_dwas};
use crate::precompute4::weighted_automata::{
    common::Label, determinization_rustfst::determinize_nwa_to_dwa, DWA, NWA, NWABody, NWAStateID, NWAStates, SimpleBitset,
    StateID, Weight,
};
use crate::tokenizer::TokenizerStateID;
use crate::types::{TerminalID, TerminalID as GrammarTokenID};

struct SimplifyRustfstConfig {
    rm_epsilon: bool,
    #[allow(dead_code)]
    determinize: bool,
}

impl SimplifyRustfstConfig {
    fn default() -> Self { Self { rm_epsilon: false, determinize: false } }
    fn with_rm_epsilon(mut self, val: bool) -> Self { self.rm_epsilon = val; self }
}

pub use crate::precompute4::template_nwa::FullDWABuildError;

pub type Precomputed4 = DWA;
type Signature = Vec<Vec<Option<TerminalID>>>;

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
    let mut values: HashMap<StateID, V> = HashMap::new();
    let mut stopped_nodes: HashSet<StateID> = HashSet::new();

    for (state, v) in initial_values {
        values.entry(state).and_modify(|old| merge(old, v.clone())).or_insert(v);
    }

    let mut in_queue = HashSet::new();
    let mut pb = tqdm!(total = nwa.states.len(), desc = "NWA Traversal", disable = !crate::profiler::PROGRESS_BAR_ENABLED, leave = false);

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
            let _ = pb.update(1);
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

fn specialize_dwa_relative(parent_dwa: &DWA, mapping: &[Weight]) -> DWA {
    let mut specialized_dwa = parent_dwa.clone();
    let mut cache: HashMap<Weight, SimpleBitset> = HashMap::new();
    let mut map_weight = |w: &Weight| -> Weight {
        if let Some(cw) = cache.get(w) { return cw.clone(); }
        let mut new_w = Weight::zeros();
        for bit in w.iter_up_to(mapping.len()) {
            if let Some(target_w) = mapping.get(bit) { new_w |= target_w; }
        }
        cache.insert(w.clone(), new_w.clone());
        new_w
    };
    for state in &mut specialized_dwa.states.0 {
        if let Some(fw) = &mut state.final_weight { *fw = map_weight(fw); if fw.is_empty() { state.final_weight = None; } }
        if let Some(sw) = &mut state.state_weight { *sw = map_weight(sw); if sw.is_empty() { state.state_weight = None; } }
        for tw in state.trans_weights.values_mut() { *tw = map_weight(tw); }
        state.trans_weights.retain(|_, w| !w.is_empty());
        state.transitions.retain(|k, _| state.trans_weights.contains_key(k));
    }
    specialized_dwa
}

fn canonicalize_bundle(terminal_map: BTreeMap<Option<TerminalID>, Weight>) -> (Signature, Vec<Weight>) {
    let mut weight_groups: HashMap<Weight, Vec<Option<TerminalID>>> = HashMap::new();
    for (term, weight) in terminal_map {
        if !weight.is_empty() { weight_groups.entry(weight).or_default().push(term); }
    }
    let mut groups_vec: Vec<(Weight, Vec<Option<TerminalID>>)> = weight_groups.into_iter().collect();
    for (_, terms) in &mut groups_vec { terms.sort(); }
    groups_vec.sort_by(|a, b| a.1.cmp(&b.1));
    (groups_vec.iter().map(|(_, terms)| terms.clone()).collect(), groups_vec.into_iter().map(|(w, _)| w).collect())
}

pub fn precompute4(parser: &GLRParser, input_nwa: &NWA) -> DWA {
    crate::debug!(3, "Starting precompute4 (DWA construction)");
    let now = Instant::now();
    let template_dwas = match build_template_dwas(parser) { Ok(m) => m, Err(e) => panic!("Failed to build template DWAs: {:?}", e), };
    let ignore_dwa = build_ignore_terminal_dwa();
    crate::debug!(3, "Built {} template DWAs in {:?}", template_dwas.len(), now.elapsed());

    let reversed_nwa = input_nwa.reverse();
    let traversal_data = reversed_nwa.compute_traversal_data();

    let initial_tokens = LLMTokenBV::max_ones();
    let mut initial_values_bv = Vec::new();
    for &start in &reversed_nwa.body.start_states {
        initial_values_bv.push((start, initial_tokens.clone()));
    }

    let start_pass1 = Instant::now();
    let (node_tokens, mut unique_signatures) = precompute_token_bvs_and_signatures(&reversed_nwa, &traversal_data, initial_values_bv);
    unique_signatures.insert(vec![vec![None]]);
    crate::debug!(3, "Pass 1: Tokens & Signatures ({} sigs, {:.2?})", unique_signatures.len(), start_pass1.elapsed());

    let mut used_terminals: BTreeSet<TerminalID> = BTreeSet::new();
    for sig in &unique_signatures {
        for group in sig {
            for term in group { if let Some(term) = term { used_terminals.insert(*term); } }
        }
    }

    let mut term_to_bit = BTreeMap::new();
    let mut bit_to_term: Vec<Option<TerminalID>> = Vec::new();
    let mut all_terminals: BTreeSet<TerminalID> = template_dwas.keys().cloned().collect();
    if let Some(ignore_term) = parser.ignore_terminal_id { all_terminals.insert(ignore_term); }
    all_terminals.extend(used_terminals);

    term_to_bit.insert(None, 0);
    bit_to_term.push(None);
    for (i, term_id) in all_terminals.iter().enumerate() {
        term_to_bit.insert(Some(*term_id), i + 1);
        bit_to_term.push(Some(*term_id));
    }

    let mut super_nwa = NWA::new_empty();
    for (term_id_opt, bit) in &term_to_bit {
        let mut weight = Weight::zeros();
        weight.set(*bit, true);
        let template_dwa = match term_id_opt {
            Some(term_id) => if Some(*term_id) == parser.ignore_terminal_id { &ignore_dwa } else { template_dwas.get(term_id).unwrap_or(&ignore_dwa) },
            None => &ignore_dwa,
        };
        let mut weighted_dwa = template_dwa.clone();
        weighted_dwa.apply_weight_inplace(&weight);
        super_nwa = NWA::union(&super_nwa, &NWA::from_dwa(&weighted_dwa));
    }
    super_nwa.simplify();
    let mut super_dwa = super_nwa.determinize();
    super_dwa.simplify();

    let mut template_cache = HashMap::new();
    let super_signature: Signature = bit_to_term.iter().map(|t| vec![*t]).collect();
    let mut pool: Vec<(Signature, DWA)> = vec![(super_signature, super_dwa.clone())];
    let mut signatures_vec: Vec<Signature> = unique_signatures.into_iter().collect();
    signatures_vec.sort_by(|a, b| {
        let groups_a = a.len(); let groups_b = b.len();
        if groups_a != groups_b { return groups_b.cmp(&groups_a); }
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
                if score < best_score { best_score = score; best_parent = Some((p_idx, mapping)); }
            }
        }
        let (parent_idx, mapping) = best_parent.expect("Super signature should always be a valid parent");
        let parent_dwa = &pool[parent_idx].1;
        let mut derived_dwa = specialize_dwa_relative(parent_dwa, &mapping);
        derived_dwa.simplify();
        template_cache.insert(target_sig.clone(), NWA::from_dwa(&derived_dwa));
        pool.push((target_sig, derived_dwa));
    }

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
    
    let offset = parser.terminal_map.len() as Label;
    let final_bodies_arc = Arc::new(Mutex::new(BTreeMap::new()));

    nwa_special_map(
        &reversed_nwa, &traversal_data, initial_values_full,
        |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV), edge_label, transitions| {
            let (current_bodies, current_tokens) = current_val;
            if let Some(lbl) = edge_label {
                if lbl >= offset {
                    let tsid = TokenizerStateID((lbl - offset) as usize);
                    let mut fb = final_bodies_arc.lock().unwrap();
                    for body in current_bodies.keys() { fb.insert(tsid, body.clone()); }
                    return Vec::new();
                }
            }
            let terminal_id = edge_label.map(|l| TerminalID(l as usize));
            let mut results = Vec::new();
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
        |_, val| {
            let (nwa_bodies_map, tokens) = val;
            let mut nwa_body = NWABody { start_states: vec![] };
            for (right_body, terminal_map) in nwa_bodies_map {
                let (signature, concrete_weights) = canonicalize_bundle(terminal_map);
                let template_nwa = template_cache.get(&signature).expect_else(|| format!("Template must exist for signature {:?}", signature));
                let instantiated_template = instantiate_nwa_template(template_nwa, &concrete_weights);
                let mut states = states_arena.borrow_mut();
                let new_states_offset = states.len();
                let composed_body = states.concatenate_in_place(&instantiated_template, &right_body);
                let new_states_filter: HashSet<NWAStateID> = (new_states_offset..states.len()).collect();
                if !new_states_filter.is_empty() {
                    apply_cancellations(&mut states, &new_states_filter);
                    apply_finality_fixpoint(&mut states, &new_states_filter);
                    remove_negative_transitions(&mut states, &new_states_filter);
                }
                nwa_body = NWABody::union(&nwa_body, &composed_body);
            }
            if !tokens.is_empty() {
                let mut next_body_map = BTreeMap::new(); next_body_map.insert(nwa_body, BTreeMap::new());
                Some((next_body_map, tokens))
            } else { None }
        },
    );

    crate::debug!(3, "Finished Pass 2");
    let final_bodies = Arc::try_unwrap(final_bodies_arc).unwrap().into_inner().unwrap();
    let mut combined_nwa_states = states_arena.into_inner();
    let combined_start_state = combined_nwa_states.add_state();
    for (tok_id, body) in final_bodies {
        let label = tok_id.0 as Label;
        for &s in &body.start_states {
            combined_nwa_states.add_transition(combined_start_state, label, s, Weight::all()).unwrap();
        }
    }
    let combined_nwa = NWA { states: combined_nwa_states, body: NWABody { start_states: vec![combined_start_state] } };
    let final_dwa = resolve_negatives_and_optimize_and_determinize(parser, combined_nwa);
    crate::debug!(3, "Precomputation complete. Final DWA stats:\n{}", final_dwa.stats());
    final_dwa
}

fn precompute_token_bvs_and_signatures(reversed_nwa: &NWA, traversal_data: &NwaTraversalData, initial_values: Vec<(StateID, LLMTokenBV)>) -> (HashMap<StateID, LLMTokenBV>, HashSet<Signature>) {
    let node_tokens: Arc<Mutex<HashMap<StateID, LLMTokenBV>>> = Arc::new(Mutex::new(HashMap::new()));
    let signatures: Arc<Mutex<HashSet<Signature>>> = Arc::new(Mutex::new(HashSet::new()));

    nwa_special_map(reversed_nwa, traversal_data, initial_values,
        |tokens: &LLMTokenBV, _, transitions| {
            let mut results = Vec::new();
            for (dest_id, weight) in transitions {
                let edge_bv: LLMTokenBV = weight.clone().into();
                let next = tokens & &edge_bv;
                if !next.is_empty() { results.push((*dest_id, next)); }
            }
            results
        },
        |t1, t2| { *t1 |= &t2; },
        |node_id, tokens| {
            node_tokens.lock().unwrap().insert(node_id, tokens.clone());
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
            let mut sigs = signatures.lock().unwrap();
            for (_, bundle) in bundles_by_dest {
                let (sig, _) = canonicalize_bundle(bundle);
                sigs.insert(sig);
            }
            Some(tokens)
        },
    );
    (Arc::try_unwrap(node_tokens).unwrap().into_inner().unwrap(), Arc::try_unwrap(signatures).unwrap().into_inner().unwrap())
}

fn resolve_negatives_and_optimize_and_determinize(parser: &GLRParser, mut combined_nwa: NWA) -> DWA {
    combined_nwa.simplify();
    prune_continuations_from_final_states(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    simplify_default_transitions(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    simplify_remove_epsilon(&mut combined_nwa);
    combined_nwa.simplify();
    simplify_remove_epsilon(&mut combined_nwa);
    combined_nwa = NWA::from_dwa(&combined_nwa.determinize());
    combined_nwa.simplify();
    let mut final_dwa = combined_nwa.determinize();
    final_dwa.minimize_with_rustfst();
    final_dwa
}

fn instantiate_nwa_template(template: &NWA, ordered_weights: &[Weight]) -> NWA {
    let mut new_nwa = NWA::new_empty();
    new_nwa.states.0.reserve(template.states.len());
    for _ in 0..template.states.len() { new_nwa.add_state(); }

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

    for (old_id, old_state) in template.states.0.iter().enumerate() {
        let new_state = &mut new_nwa.states[old_id];
        if let Some(fw) = &old_state.final_weight {
            let concrete = map_abstract_weight(fw);
            if !concrete.is_empty() { new_state.final_weight = Some(concrete); }
        }
        for (lbl, targets) in &old_state.transitions {
            for (target, w) in targets {
                let concrete = map_abstract_weight(w);
                if !concrete.is_empty() { new_nwa.states.add_transition(old_id, *lbl, *target, concrete).unwrap(); }
            }
        }
        for (target, w) in &old_state.epsilons {
            let concrete = map_abstract_weight(w);
            if !concrete.is_empty() { new_nwa.states.add_epsilon(old_id, *target, concrete); }
        }
    }
    new_nwa.body = template.body.clone();
    new_nwa
}

fn simplify_remove_epsilon(nwa: &mut NWA) {
    nwa.simplify()
}
