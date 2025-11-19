use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque, BinaryHeap};
use std::env;
use std::time::Instant;
use std::cmp::Reverse;

use chrono::Local;

use crate::constraint::LLMTokenBV;
use crate::glr::parser::{ExpectElse, GLRParser};
use crate::json_serialization::JSONConvertible;
use crate::precompute4::nwa_optimizations::{prune_continuations_from_final_states, simplify_default_transitions};
use crate::precompute4::resolve_negatives::{apply_cancellations, apply_finality_fixpoint, remove_negative_transitions};
use crate::precompute4::template_nwa::{build_epsilon_dwa, build_ignore_terminal_dwa, build_template_dwas};
use crate::precompute4::weighted_automata::{DWA, NWA, NWABody, NWAStateID, NWAStates, Weight, StateID};
use crate::r#macro::is_debug_level_enabled;
use crate::types::TerminalID;
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

use kdam::{tqdm, BarExt};
use crate::profiler::PROGRESS_BAR_ENABLED;

pub type Precomputed4 = DWA;

/// Public API: precompute4 using an NWA-first approach, determinizing at the end.
pub fn precompute4(
    parser: &GLRParser,
    mut nwa: NWA,
    nwa_roots: &BTreeMap<TokenizerStateID, NWAStateID>,
    _max_llm_token_id: usize,
) -> DWA {
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
    let reversed_nwa = reverse_nwa(&NWA::from_dwa(&dwa));
    // The leaf node of the original DWA (which had final weights) becomes the root of the reversed NWA.
    // In `reverse_nwa`, we don't set a single start state.
    // However, `dwa` from precompute1 should have states with final weights.
    // We collect those states to be the roots for the traversal.
    let mut reversed_roots = Vec::new();
    for (i, st) in dwa.states.0.iter().enumerate() {
        if st.final_weight.is_some() {
            reversed_roots.push(i);
        }
    }
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
    
    let mut initial_values = Vec::new();
    for &root in &reversed_roots {
        initial_values.push((root, (initial_body_map.clone(), initial_tokens.clone())));
    }

    let traversal_data =
        compute_nwa_traversal_data(&reversed_nwa, &reversed_roots).expect("Failed to compute traversal data for reversed NWA");

    let mut original_trie1_roots_map: BTreeMap<NWAStateID, Vec<TokenizerStateID>> = BTreeMap::new();
    for (k, v) in nwa_roots.iter() {
        original_trie1_roots_map.entry(*v).or_default().push(*k);
    }

    // crate::debug!(5, "Reversed NWA:\n{}", reversed_nwa);

    let mut final_bodies: BTreeMap<TokenizerStateID, NWABody> = BTreeMap::new();

    let now = Instant::now();
    special_map_grouped_nwa(
        &reversed_nwa,
        &traversal_data,
        initial_values,
        // step function
        |current_val: &(BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV), 
         edge_terminal_opt: Option<i16>, 
         dest_list: &Vec<(NWAStateID, Weight)>| {
            let (current_nwa_body, current_tokens) = current_val;
            let terminal_id = edge_terminal_opt;

            let mut results = Vec::new();
            for (dest_idx, llm_token_bv) in dest_list.iter() {
                let next_tokens = current_tokens & &LLMTokenBV::from(llm_token_bv.clone());
                if next_tokens.is_empty() {
                    continue;
                }
                let weight = Weight::from_rsb(llm_token_bv.rsb.clone());
                let mut terminal_map = BTreeMap::new();
                terminal_map.insert(terminal_id, weight);
                // We need to return a map of bodies, but here we are processing one body at a time?
                // Wait, the input `current_val` is the aggregated value (map of bodies).
                // But the signature of `step` in Trie::special_map_grouped was `FnMut(&U, &EK, &OrderedHashMap) -> I`.
                // Here I am defining `step` to take the aggregated value.
                let mut body_map = BTreeMap::new();
                body_map.insert(current_nwa_body.clone(), terminal_map);
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
        |node_idx: NWAStateID,
         val: (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV)| -> Option<(NWABody, LLMTokenBV)> {
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

fn reverse_nwa(nwa: &NWA) -> NWA {
    let mut reversed = NWA::new();
    reversed.states.0.clear();
    for _ in 0..nwa.states.len() {
        reversed.states.add_state();
    }
    
    for (src, state) in nwa.states.0.iter().enumerate() {
        for (&label, targets) in &state.transitions {
            for &(dst, ref w) in targets {
                reversed.add_transition(dst, label, src, w.clone()).unwrap();
            }
        }
        for &(dst, ref w) in &state.epsilons {
            reversed.add_epsilon(dst, src, w.clone());
        }
        // final_weight in original becomes something we track via roots in traversal
    }
    reversed
}

struct NWATraversalData {
    nodes: Vec<NWAStateID>,
    pos_of_u: HashMap<usize, usize>,
    comp_id: Vec<usize>,
    sccs: Vec<Vec<usize>>,
    topo: Vec<usize>,
}

fn compute_nwa_traversal_data(nwa: &NWA, roots: &[NWAStateID]) -> Option<NWATraversalData> {
    let mut nodes = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    for &root in roots {
        if visited.insert(root) {
            queue.push_back(root);
        }
    }
    while let Some(u) = queue.pop_front() {
        nodes.push(u);
        if u >= nwa.states.len() { continue; }
        let st = &nwa.states[u];
        for &(v, _) in &st.epsilons {
            if visited.insert(v) {
                queue.push_back(v);
            }
        }
        for targets in st.transitions.values() {
            for &(v, _) in targets {
                if visited.insert(v) {
                    queue.push_back(v);
                }
            }
        }
    }
    
    if nodes.is_empty() { return None; }
    
    let n = nodes.len();
    let mut pos_of_u = HashMap::with_capacity(n);
    for (i, &u) in nodes.iter().enumerate() {
        pos_of_u.insert(u, i);
    }

    let mut adj = vec![Vec::new(); n];
    let mut radj = vec![Vec::new(); n];
    for (i, &u) in nodes.iter().enumerate() {
        let st = &nwa.states[u];
        let mut neighbors = Vec::new();
        for &(v, _) in &st.epsilons { neighbors.push(v); }
        for targets in st.transitions.values() {
            for &(v, _) in targets { neighbors.push(v); }
        }
        for v in neighbors {
            if let Some(&j) = pos_of_u.get(&v) {
                adj[i].push(j);
                radj[j].push(i);
            }
        }
    }

    // Kosaraju
    let mut visited_scc = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for u in 0..n {
        if !visited_scc[u] {
            let mut stack = vec![(u, 0)];
            visited_scc[u] = true;
            while let Some((node, next_i)) = stack.last_mut() {
                if *next_i < adj[*node].len() {
                    let v = adj[*node][*next_i];
                    *next_i += 1;
                    if !visited_scc[v] {
                        visited_scc[v] = true;
                        stack.push((v, 0));
                    }
                } else {
                    order.push(*node);
                    stack.pop();
                }
            }
        }
    }

    let mut comp_id = vec![usize::MAX; n];
    let mut cid = 0;
    for &u in order.iter().rev() {
        if comp_id[u] == usize::MAX {
            let mut stack = vec![u];
            comp_id[u] = cid;
            while let Some(x) = stack.pop() {
                for &v in &radj[x] {
                    if comp_id[v] == usize::MAX {
                        comp_id[v] = cid;
                        stack.push(v);
                    }
                }
            }
            cid += 1;
        }
    }

    let scc_count = cid;
    let mut sccs = vec![Vec::new(); scc_count];
    for i in 0..n {
        sccs[comp_id[i]].push(i);
    }

    let mut scc_adj = vec![BTreeSet::new(); scc_count];
    let mut indeg = vec![0; scc_count];
    for u in 0..n {
        let cu = comp_id[u];
        for &v in &adj[u] {
            let cv = comp_id[v];
            if cu != cv {
                if scc_adj[cu].insert(cv) {
                    indeg[cv] += 1;
                }
            }
        }
    }

    let mut topo = Vec::with_capacity(scc_count);
    let mut q_scc = VecDeque::new();
    for s in 0..scc_count {
        if indeg[s] == 0 {
            q_scc.push_back(s);
        }
    }
    while let Some(s) = q_scc.pop_front() {
        topo.push(s);
        for &t in &scc_adj[s] {
            indeg[t] -= 1;
            if indeg[t] == 0 {
                q_scc.push_back(t);
            }
        }
    }

    Some(NWATraversalData { nodes, pos_of_u, comp_id, sccs, topo })
}

fn special_map_grouped_nwa<V, S>(
    nwa: &NWA,
    traversal_data: &NWATraversalData,
    initial_values: Vec<(NWAStateID, V)>,
    mut step: S, // (current_val, edge_label_opt, dest_list) -> Vec<(dest_id, new_val)>
    mut merge: impl FnMut(&mut V, V),
    mut process: impl FnMut(NWAStateID, V) -> Option<(NWABody, LLMTokenBV)>,
)
where
    V: Clone,
    S: FnMut(&(NWABody, LLMTokenBV), Option<i16>, &Vec<(NWAStateID, Weight)>) -> Vec<(NWAStateID, V)>,
{
    let mut values: HashMap<usize, V> = HashMap::new();
    let mut stopped_nodes: HashSet<usize> = HashSet::new();

    let total_edges = nwa.num_transitions() + nwa.num_epsilons();
    let mut pb = tqdm!(total = total_edges, desc = "Traversing NWA", disable = !PROGRESS_BAR_ENABLED, leave=false);

    for (node_idx, v0) in initial_values {
        values.entry(node_idx).and_modify(|old| merge(old, v0.clone())).or_insert(v0);
    }

    let nodes = &traversal_data.nodes;
    let pos_of_u = &traversal_data.pos_of_u;
    let comp_id = &traversal_data.comp_id;
    let sccs = &traversal_data.sccs;
    let topo = &traversal_data.topo;

    let mut in_queue: HashSet<usize> = HashSet::new();

    for &s in topo {
        let mut local_queue: VecDeque<usize> = VecDeque::new();
        for &pos in &sccs[s] {
            let u = nodes[pos];
            if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                if in_queue.insert(u) {
                    local_queue.push_back(pos);
                }
            }
        }
        if local_queue.is_empty() { continue; }

        while let Some(pos) = local_queue.pop_front() {
            pb.set_postfix(&format!("pending: {}", values.len()));
            let u = nodes[pos];
            in_queue.remove(&u);

            if stopped_nodes.contains(&u) { continue; }
            let agg_v = match values.remove(&u) {
                Some(v) => v,
                None => continue,
            };

            let processed_value = process(u, agg_v);
            let proceed_value = match processed_value {
                Some(val) => val,
                None => {
                    stopped_nodes.insert(u);
                    continue;
                }
            };

            let st = &nwa.states[u];
            
            // Epsilons
            if !st.epsilons.is_empty() {
                let _ = pb.update(st.epsilons.len());
                let new_vals = step(&proceed_value, None, &st.epsilons);
                for (child_u, new_v) in new_vals {
                    if stopped_nodes.contains(&child_u) { continue; }
                    values.entry(child_u).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                    if let Some(&child_pos) = pos_of_u.get(&child_u) {
                        if comp_id[child_pos] == s {
                            if in_queue.insert(child_u) {
                                local_queue.push_back(child_pos);
                            }
                        }
                    }
                }
            }

            // Transitions
            for (&label, targets) in &st.transitions {
                let _ = pb.update(targets.len());
                let new_vals = step(&proceed_value, Some(label), targets);
                for (child_u, new_v) in new_vals {
                    if stopped_nodes.contains(&child_u) { continue; }
                    values.entry(child_u).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v);
                    if let Some(&child_pos) = pos_of_u.get(&child_u) {
                        if comp_id[child_pos] == s {
                            if in_queue.insert(child_u) {
                                local_queue.push_back(child_pos);
                            }
                        }
                    }
                }
            }

            if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                if in_queue.insert(u) {
                    local_queue.push_back(pos);
                }
            }
        }
    }
    pb.set_postfix("");
}
