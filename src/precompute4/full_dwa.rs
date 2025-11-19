use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::time::Instant;

use chrono::Local;

use crate::constraint::LLMTokenBV;
use crate::glr::parser::{ExpectElse, GLRParser};
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

/// Public API: precompute4 using an NWA-first approach, determinizing at the end.
pub fn precompute4(
    parser: &GLRParser,
    precomputed1_nwa: &NWA,
    _max_llm_token_id: usize,
) -> DWA {
    crate::debug!(4, "Optimizing precomputed1 via NWA/DWA conversion...");
    let mut nwa = precomputed1_nwa.clone();
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

    // 3. Reverse the optimized precomputed1 DWA (converted back to NWA for traversal).
    // The DWA from precomputed1 has a start state (root).
    // We need to reverse it so we can traverse from leaves (which are now roots) up to the start.
    // The leaves of the original DWA are states with final weights.
    // In the reversed graph, these become the start states.
    // The original start state becomes a leaf (or multiple if we consider all reachable).
    let nwa_for_reversal = NWA::from_dwa(&dwa);
    let reversed_nwa = reverse_nwa(&nwa_for_reversal);
    
    // The roots for traversal in the reversed graph are the original final states.
    // `reverse_nwa` puts all original final states as reachable from its new start state via epsilon if needed,
    // or we can just find them.
    // Actually `reverse_nwa` implementation below will create a single start state connected to original final states.
    let reversed_nwa_root = reversed_nwa.body.start_state;
    let reversed_nwa_roots = vec![reversed_nwa_root];

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
        vec![(reversed_nwa_root, (initial_body_map, initial_tokens))];

    let traversal_data =
        compute_nwa_traversal_data(&reversed_nwa, &reversed_nwa_roots).expect("Failed to compute traversal data for reversed NWA");

    // Map from reversed NWA state (which corresponds to original state) to TokenizerStateID.
    // The original DWA start state corresponds to the tokenizer states.
    // In the reversed NWA, the original start state is a leaf.
    // We need to know which TokenizerStateID corresponds to the original start state's transitions.
    // Actually, the precomputed1 DWA start state has transitions labeled with TokenizerStateID.
    // So in the reversed NWA, the edges coming *into* the original start state (now outgoing from it)
    // are labeled with TokenizerStateID.
    // Wait, `precomputed1` DWA structure: Start -> (TokenizerStateID) -> Root of specific trie.
    // So the original DWA has a single start state, and transitions on `TokenizerStateID` to the roots of the per-state graphs.
    // In the reversed NWA, we start from the leaves (original finals) and go backwards.
    // Eventually we reach the original roots. From there, we have transitions labeled `TokenizerStateID` back to the original start.
    // So when we process the edge labeled `TokenizerStateID`, we capture the result.

    let mut final_bodies: BTreeMap<TokenizerStateID, NWABody> = BTreeMap::new();

    let now = Instant::now();
    nwa_special_map_grouped(
        &reversed_nwa,
        &traversal_data,
        initial_values,
        // step function
        |current_val: &(NWABody, LLMTokenBV), edge_label_i16, targets| {
            let (current_nwa_body, current_tokens) = current_val;
            let terminal_id = if edge_label_i16 >= 0 { Some(GrammarTokenID(edge_label_i16 as usize)) } else { None };

            let mut results = Vec::new();
            for (dest_idx, weight) in targets {
                let llm_token_bv = LLMTokenBV::from(weight.clone());
                let next_tokens = current_tokens & &llm_token_bv;
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
            for (right_body, mut term_map2) in bodies2 {
                let term_map1 = bodies1.entry(right_body).or_default();
                for (term, weight2) in term_map2 {
                    *term_map1.entry(term).or_insert_with(Weight::zeros) |= &weight2;
                }
            }
            *tokens1 |= &tokens2;
        },
        // process function: capture at original roots
        |_guard,
         node_idx,
         val: (BTreeMap<NWABody, BTreeMap<Option<TerminalID>, Weight>>, LLMTokenBV)| {
            let (nwa_bodies_map, tokens) = val;
            
            // In the reversed NWA, the node_idx corresponds to a state in the original DWA.
            // The original DWA start state is where we collect results.
            // But wait, we need to know which TokenizerStateID it corresponds to.

            // Combine all left bodies into a single NWA body via union (epsilon)
            let mut nwa_body = {
                let mut states = states_arena.borrow_mut();
                let start = states.add_state();
                NWABody { start_state: start }
            };

            // crate::debug!(6, "NWA states:\n{}", states_arena.borrow());
            // crate::debug!(6, "{:?}", nwa_bodies_map);

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
            // crate::debug!(6, "NWA body:\n{}", nwa_body);
            // crate::debug!(6, "NWA states:\n{}", states_arena.borrow());

            if !tokens.is_empty() {
                // Check if this node has a transition to the original start state in the reversed graph.
                // In the reversed graph, this corresponds to an edge FROM the original start state TO this node.
                // The label on that edge is the TokenizerStateID.
                // We need to find edges in `reversed_nwa` from `node_idx` that go to the `reversed_nwa`'s leaf (original start).
                // Wait, `reversed_nwa` structure: Original Final -> ... -> Original Roots -> Original Start.
                // The edges from Original Roots to Original Start are labeled with TokenizerStateID.
                // So if `node_idx` is an Original Root, it has an outgoing edge to Original Start.
                
                // We can inspect the outgoing edges of `node_idx` in `reversed_nwa`.
                for (label, targets) in &reversed_nwa.states[node_idx].transitions {
                    // If the target is the original start state (which is now a leaf in reversed graph)
                    if targets.iter().any(|(t, _)| *t == dwa.body.start_state) {
                        // This label is the TokenizerStateID
                        let tokenizer_id = TokenizerStateID(*label as usize);
                        final_bodies.insert(tokenizer_id, nwa_body.clone());
                    }
                }
                Some((nwa_body, tokens))
            } else { None } },
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

use crate::datastructures::trie::Trie2Index;

// NWA Traversal Utilities

#[derive(Debug, Clone)]
pub struct NWATraversalData {
    pub nodes: Vec<NWAStateID>,
    pub pos_of_u: HashMap<NWAStateID, usize>,
    pub comp_id: Vec<usize>,
    pub sccs: Vec<Vec<usize>>,
    pub topo: Vec<usize>,
}

pub fn compute_nwa_traversal_data(nwa: &NWA, roots: &[NWAStateID]) -> Option<NWATraversalData> {
    // BFS to find reachable nodes
    let mut nodes = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    for &root in roots {
        if visited.insert(root) {
            queue.push_back(root);
        }
    }
    while let Some(u) = queue.pop_front() {
        nodes.push(u);
        if u < nwa.states.len() {
            let st = &nwa.states[u];
            for targets in st.transitions.values() {
                for (v, _) in targets {
                    if visited.insert(*v) {
                        queue.push_back(*v);
                    }
                }
            }
            for (v, _) in &st.epsilons {
                if visited.insert(*v) {
                    queue.push_back(*v);
                }
            }
        }
    }

    if nodes.is_empty() {
        return None;
    }

    let n = nodes.len();
    let mut pos_of_u = HashMap::with_capacity(n);
    for (i, &u) in nodes.iter().enumerate() {
        pos_of_u.insert(u, i);
    }

    // Build adjacency for SCC
    let mut adj = vec![Vec::new(); n];
    let mut radj = vec![Vec::new(); n];
    for (i, &u) in nodes.iter().enumerate() {
        if u < nwa.states.len() {
            let st = &nwa.states[u];
            let mut neighbors = Vec::new();
            for targets in st.transitions.values() {
                for (v, _) in targets {
                    neighbors.push(*v);
                }
            }
            for (v, _) in &st.epsilons {
                neighbors.push(*v);
            }
            
            for v in neighbors {
                if let Some(&j) = pos_of_u.get(&v) {
                    adj[i].push(j);
                    radj[j].push(i);
                }
            }
        }
    }

    // Kosaraju
    let mut visited_scc = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for i in 0..n {
        if !visited_scc[i] {
            fn dfs1(u: usize, adj: &[Vec<usize>], visited: &mut Vec<bool>, order: &mut Vec<usize>) {
                visited[u] = true;
                for &v in &adj[u] {
                    if !visited[v] {
                        dfs1(v, adj, visited, order);
                    }
                }
                order.push(u);
            }
            dfs1(i, &adj, &mut visited_scc, &mut order);
        }
    }

    let mut comp_id = vec![usize::MAX; n];
    let mut cid = 0;
    for &u in order.iter().rev() {
        if comp_id[u] == usize::MAX {
            fn dfs2(u: usize, radj: &[Vec<usize>], comp_id: &mut Vec<usize>, cid: usize) {
                comp_id[u] = cid;
                for &v in &radj[u] {
                    if comp_id[v] == usize::MAX {
                        dfs2(v, radj, comp_id, cid);
                    }
                }
            }
            dfs2(u, &radj, &mut comp_id, cid);
            cid += 1;
        }
    }

    let scc_count = cid;
    let mut sccs = vec![Vec::new(); scc_count];
    for i in 0..n {
        sccs[comp_id[i]].push(i);
    }

    // Topo sort of SCCs
    let mut scc_adj = vec![HashSet::new(); scc_count];
    let mut indeg = vec![0; scc_count];
    for u in 0..n {
        let cu = comp_id[u];
        for &v in &adj[u] {
            let cv = comp_id[v];
            if cu != cv && scc_adj[cu].insert(cv) {
                indeg[cv] += 1;
            }
        }
    }

    let mut topo = Vec::with_capacity(scc_count);
    let mut q = std::collections::VecDeque::new();
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

    Some(NWATraversalData { nodes, pos_of_u, comp_id, sccs, topo })
}

pub fn reverse_nwa(nwa: &NWA) -> NWA {
    let mut rev = NWA::new();
    rev.states.0.clear();
    for _ in 0..nwa.states.len() {
        rev.states.add_state();
    }
    
    // Original start state becomes a target for transitions from original roots.
    // Original final states become the new start states (connected via epsilon from new super-start).
    let new_start = rev.states.add_state();
    rev.body.start_state = new_start;

    for (u, st) in nwa.states.0.iter().enumerate() {
        if st.final_weight.is_some() {
            rev.add_epsilon(new_start, u, Weight::all());
        }
        for (&label, targets) in &st.transitions {
            for &(v, ref w) in targets {
                rev.add_transition(v, label, u, w.clone()).unwrap();
            }
        }
        for &(v, ref w) in &st.epsilons {
            rev.add_epsilon(v, u, w.clone());
        }
    }
    rev
}

pub fn nwa_special_map_grouped<V, U, S, I>(
    nwa: &NWA,
    traversal_data: &NWATraversalData,
    initial_nodes_and_values: Vec<(NWAStateID, V)>,
    mut step: S,
    mut merge: impl FnMut(&mut V, V),
    mut process: impl FnMut(&crate::precompute4::weighted_automata::nwa::NWAState, NWAStateID, V) -> Option<U>,
)
where
    V: Clone,
    S: FnMut(&U, i16, &Vec<(NWAStateID, Weight)>) -> I,
    I: IntoIterator<Item = (NWAStateID, V)>,
{
    let mut values: HashMap<NWAStateID, V> = HashMap::new();
    let mut stopped_nodes: HashSet<NWAStateID> = HashSet::new();

    for (node_idx, v0) in initial_nodes_and_values {
        values.entry(node_idx).and_modify(|old| merge(old, v0.clone())).or_insert(v0);
    }

    let nodes = &traversal_data.nodes;
    let pos_of_u = &traversal_data.pos_of_u;
    let comp_id = &traversal_data.comp_id;
    let sccs = &traversal_data.sccs;
    let topo = &traversal_data.topo;

    let mut in_queue: HashSet<NWAStateID> = HashSet::new();
    
    for &s in topo {
        let mut local_queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
        for &pos in &sccs[s] {
            let u = nodes[pos];
            if values.contains_key(&u) && !stopped_nodes.contains(&u) {
                if in_queue.insert(u) {
                    local_queue.push_back(pos);
                }
            }
        }

        while let Some(pos) = local_queue.pop_front() {
            let u = nodes[pos];
            in_queue.remove(&u);
            if stopped_nodes.contains(&u) { continue; }
            
            let agg_v = match values.remove(&u) {
                Some(v) => v,
                None => continue,
            };

            let processed_value = process(&nwa.states[u], u, agg_v);
            let proceed_value = match processed_value {
                Some(val) => val,
                None => {
                    stopped_nodes.insert(u);
                    continue;
                }
            };

            // Propagate
            // Labeled transitions
            for (&label, targets) in &nwa.states[u].transitions {
                for (child_u, new_v) in step(&proceed_value, label, targets) {
                    if stopped_nodes.contains(&child_u) { continue; }
                    values.entry(child_u).and_modify(|old| merge(old, new_v.clone())).or_insert(new_v.clone());
                    
                    if let Some(&child_pos) = pos_of_u.get(&child_u) {
                        if comp_id[child_pos] == s {
                            if in_queue.insert(child_u) {
                                local_queue.push_back(child_pos);
                            }
                        }
                    }
                }
            }
            // Epsilons - treat as label -1 (or special)? 
            // The step function signature takes i16. We can pass -1 or handle separately.
            // But `step` expects `i16`. Let's use a convention or modify `step` sig.
            // The caller provided `step` expects `i16`.
            // In `precompute4`, we used `edge_label_i16`.
            // We can pass -1 for epsilon if that's unused, or handle epsilons separately.
            // Given `precompute4` logic, it handles `edge_label_i16 >= 0` as terminal.
            // Epsilons in NWA are usually for union/concat.
            // In the reversed graph, epsilons exist.
            // Let's assume `step` handles -1 as epsilon or we pass a dummy.
            // Actually, `precompute4` logic: `if edge_label_i16 >= 0 ... else None`.
            // So passing -1 for epsilon is safe.
            if !nwa.states[u].epsilons.is_empty() {
                for (child_u, new_v) in step(&proceed_value, -1, &nwa.states[u].epsilons) {
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
}
