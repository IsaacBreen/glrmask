use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher, DefaultHasher};
use std::cmp::Reverse;
use std::ops::BitOrAssign;
use std::sync::Arc;
use bitvec::macros::internal::funty::Fundamental;
use range_set_blaze::RangeSetBlaze;
use ordered_hash_map::OrderedHashMap;
use kdam::tqdm;
use crate::constraint::{GrammarConstraintConfig, PrecomputeNode0Index, PrecomputeNode1, PrecomputedNodeContents, Trie0GodWrapper};
use crate::profiler::PROGRESS_BAR_ENABLED;
use crate::constraint::{StageVocab, PrecomputeNode1Index, Trie1GodWrapper};
use crate::constraint_extra::PrecomputeStats;
use deterministic_hash::DeterministicHasher;
use crate::datastructures::EntryApi;
use crate::constraint::LLMTokenBV;
use crate::datastructures::trie::Trie;
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::tokenizer::TokenizerStateID;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::ordered_hash_map::Retain;

fn constrain_bitvecs_trie1(
    trie1_god: &Trie1GodWrapper,
    roots: &[PrecomputeNode1Index],
    max_llm_token_id: usize,
) {
    crate::debug!(3, "Trie1: constraining LLM token bitvectors and removing empty edges...");
    let all_nodes = Trie::all_nodes(trie1_god, roots);
    for n in all_nodes {
        let mut w = n.write(trie1_god).expect("write");
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<Option<GrammarTokenID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
        for (ek, dest_map) in old_children {
            let mut new_dest_map = OrderedHashMap::new();
            for (dst, mut bv) in dest_map {
                bv.constrain(max_llm_token_id);
                if !bv.is_empty() {
                    new_dest_map.insert(dst, bv);
                }
            }
            if !new_dest_map.is_empty() {
                new_children.insert(ek, new_dest_map);
            }
        }
        *w.children_mut() = new_children;
    }
}

// Flatten None-key (epsilon-like) chains when safe:
// U --(None, B1)--> V, and V is non-end with exactly one outgoing edge (None, B2) -> W (and no other edges)
// becomes U --(None, B1 ∩ B2)--> W
// This is exact because None edges do not perform grammar steps (in get_mask1),
// and we do not bypass any 'end' union points (we forbid compressing through end nodes).
fn shortcut_none_chains_trie1(
    trie1_god: &Trie1GodWrapper,
    roots: &[PrecomputeNode1Index],
) {
    crate::debug!(3, "Trie1: shortcutting (flattening) None-key chains where safe...");
    let nodes = Trie::all_nodes(trie1_god, roots);
    if nodes.is_empty() { return; }

    type DestList = Vec<(PrecomputeNode1Index, LLMTokenBV)>;
    type EdgeList = Vec<(Option<GrammarTokenID>, DestList)>;

    // Build summary snapshot
    let mut summary: HashMap<PrecomputeNode1Index, (bool, EdgeList)> = HashMap::new();
    for u in &nodes {
        let g = u.read(trie1_god).expect("read");
        let edges: EdgeList = g.children()
            .iter()
            .map(|(ek, dm)| {
                let dests = dm.iter().map(|(d, bv)| (*d, bv.clone())).collect::<DestList>();
                (*ek, dests)
            })
            .collect();
        summary.insert(*u, (g.value.end, edges));
    }

    #[derive(Clone)]
    struct ChainRes {
        last: PrecomputeNode1Index,
        llm: LLMTokenBV,
    }

    // Follow a chain of (None, B) with single outgoing None edge and non-end middle nodes.
    fn follow_none_chain(
        v: PrecomputeNode1Index,
        trie1_god: &Trie1GodWrapper,
        summary: &HashMap<PrecomputeNode1Index, (bool, EdgeList)>,
        memo: &mut HashMap<PrecomputeNode1Index, Option<ChainRes>>,
    ) -> Option<ChainRes> {
        if let Some(cached) = memo.get(&v) {
            return cached.clone();
        }
        let (is_end, edges) = match summary.get(&v) {
            Some(x) => x,
            None => {
                memo.insert(v, None);
                return None;
            }
        };
        if *is_end { memo.insert(v, None); return None; }

        // Count only None edges
        let mut none_edges = edges.iter().filter(|(k, _)| k.is_none());
        let next = match none_edges.next() {
            Some(x) => x,
            None => {
                memo.insert(v, None);
                return None;
            }
        };
        // Must be the only outgoing edge and have exactly one destination.
        if edges.len() != 1 || next.1.len() != 1 {
            memo.insert(v, None);
            return None;
        }
        let (dst, bv2) = &next.1[0];

        let res = if let Some(tail) = follow_none_chain(*dst, trie1_god, summary, memo) {
            Some(ChainRes { last: tail.last, llm: bv2 & &tail.llm })
        } else {
            Some(ChainRes { last: *dst, llm: bv2.clone() })
        };
        memo.insert(v, res.clone());
        res
    }

    let mut memo: HashMap<PrecomputeNode1Index, Option<ChainRes>> = HashMap::new();

    for u in &nodes {
        // Snapshot current None edges for this node
        let none_edges_snapshot: Vec<(PrecomputeNode1Index, LLMTokenBV)> = {
            let g = u.read(trie1_god).expect("read");
            match g.children().get(&None) {
                Some(dm) => dm.iter().map(|(d, bv)| (*d, bv.clone())).collect(),
                None => Vec::new(),
            }
        };
        if none_edges_snapshot.is_empty() { continue; }

        let mut w = u.write(trie1_god).expect("write");
        for (v, b1) in none_edges_snapshot {
            if let Some(chain) = follow_none_chain(v, trie1_god, &summary, &mut memo) {
                // Remove U --(None)--> V
                if let Some(dm) = w.children_mut().get_mut(&None) {
                    dm.remove(&v);
                }
                // Add U --(None)--> chain.last with intersected BV
                let new_bv = b1 & &chain.llm;
                if !new_bv.is_empty() {
                    w.children_mut()
                        .entry(None)
                        .or_default()
                        .entry(chain.last)
                        .and_modify(|e| *e |= &new_bv)
                        .or_insert(new_bv);
                }
            }
        }
    }
}

// Backward liveness pruning for Trie1:
// - Seed end nodes with ALL internal tokens (conservative upper bound).
// - Propagate back along edges with intersection & edge BV.
// - Remove edges with empty propagated BVs; set node.value.live_tokens to the liveness result.
fn prune_dead_paths_trie1(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    max_llm_token_id: usize,
) {
    crate::debug!(3, "Trie1: pruning dead paths (conservative, token-only)...");
    let all_nodes = Trie::all_nodes(trie1_god, &roots.values().cloned().collect::<Vec<_>>());
    if all_nodes.is_empty() { return; }

    let all_tokens = LLMTokenBV::ones(max_llm_token_id + 1);
    let mut live: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();
    let mut preds: HashMap<PrecomputeNode1Index, Vec<(PrecomputeNode1Index, Option<GrammarTokenID>, LLMTokenBV)>> = HashMap::new();
    let mut worklist: VecDeque<PrecomputeNode1Index> = VecDeque::new();

    // Initialize live sets and predecessors
    for u in &all_nodes {
        live.insert(*u, LLMTokenBV::zeros());
        let g = u.read(trie1_god).expect("read");
        if g.value.end {
            live.insert(*u, all_tokens.clone()); // conservative seed
            worklist.push_back(*u);
        }
        for (ek, dm) in g.children() {
            for (v, bv) in dm {
                preds.entry(*v).or_default().push((*u, *ek, bv.clone()));
            }
        }
    }

    // Backward propagation
    while let Some(v) = worklist.pop_front() {
        let live_v = live.get(&v).unwrap().clone();
        if let Some(in_edges) = preds.get(&v) {
            for (u, _ek, edge_bv) in in_edges {
                let add = &live_v & edge_bv;
                if add.is_empty() { continue; }
                let entry = live.get_mut(u).unwrap();
                let before = entry.len();
                *entry |= &add;
                if entry.len() > before {
                    worklist.push_back(*u);
                }
            }
        }
    }

    // Prune edges based on liveness intersections and write node live_tokens
    for u in &all_nodes {
        let mut w = u.write(trie1_god).expect("write");
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<Option<GrammarTokenID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in old_children {
            let mut new_dm = OrderedHashMap::new();
            for (v, edge_bv) in dm {
                let live_v = live.get(&v).unwrap();
                let pass = &edge_bv & live_v;
                if !pass.is_empty() {
                    new_dm.insert(v, pass);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
        // Update node live tokens for diagnostics/consumers
        if let Some(l) = live.get(u) {
            w.value.live_tokens = l.clone();
        }
    }
}

// Exact signature-based DAG minimization for Trie1:
// Two nodes are equivalent iff:
//   - their end flags match
//   - for each edge key and destination class, the aggregated LLMTokenBV is identical
// We reconstruct representative nodes and redirect all references to them.
fn merge_nodes_trie1(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) {
    crate::debug!(3, "Trie1: merging identical subgraphs by signature (partition refinement)...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    // Dense indexing
    let mut dense_of: HashMap<PrecomputeNode1Index, usize> = HashMap::new();
    let mut old_of: Vec<PrecomputeNode1Index> = Vec::with_capacity(all_nodes.len());
    for (i, idx) in all_nodes.iter().enumerate() {
        dense_of.insert(*idx, i);
        old_of.push(*idx);
    }
    let n = old_of.len();

    let mut ends: Vec<bool> = vec![false; n];
    type RawEdge1 = (Option<GrammarTokenID>, LLMTokenBV, usize);
    let mut raw_edges: Vec<Vec<RawEdge1>> = vec![Vec::new(); n];

    for (u_dense, u_idx) in old_of.iter().enumerate() {
        let g = u_idx.read(trie1_god).expect("read");
        ends[u_dense] = g.value.end;
        for (ek, dm) in g.children() {
            for (v_idx, bv) in dm {
                if let Some(&v_dense) = dense_of.get(v_idx) {
                    raw_edges[u_dense].push((*ek, bv.clone(), v_dense));
                }
            }
        }
    }

    // Initial partition by end flag
    let mut prev_class: Vec<usize> = (0..n).map(|i| if ends[i] { 1 } else { 0 }).collect();

    const MAX_ITERS: usize = 40;
    for _it in 0..MAX_ITERS {
        // Signature: (end_flag, Vec<((edge_key, dest_class), llm_bv_union)>)
        type SigEdgeKey = (Option<GrammarTokenID>, usize);
        type Signature1 = (bool, Vec<(SigEdgeKey, LLMTokenBV)>);

        let mut sig_to_id: HashMap<Signature1, usize> = HashMap::new();
        let mut new_class = vec![0; n];
        let mut next_id = 0;
        let mut changes = 0;

        for u in 0..n {
            let mut aggr: BTreeMap<SigEdgeKey, LLMTokenBV> = BTreeMap::new();
            for (ek, llm_bv, v_dense) in &raw_edges[u] {
                let dest_class = prev_class[*v_dense];
                let key = (*ek, dest_class);
                aggr.entry(key)
                    .and_modify(|e| *e |= llm_bv)
                    .or_insert_with(|| llm_bv.clone());
            }
            let agg_vec: Vec<(SigEdgeKey, LLMTokenBV)> = aggr.into_iter().collect();
            let sig: Signature1 = (ends[u], agg_vec);

            let cid = *sig_to_id.entry(sig).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            new_class[u] = cid;
            if new_class[u] != prev_class[u] {
                changes += 1;
            }
        }

        prev_class = new_class;
        if changes == 0 { break; }
    }

    // Representatives per class
    let num_classes = prev_class.iter().max().map_or(0, |m| m + 1);
    let mut rep_of_class: Vec<Option<PrecomputeNode1Index>> = vec![None; num_classes];
    for (u_dense, &cid) in prev_class.iter().enumerate() {
        if rep_of_class[cid].is_none() {
            rep_of_class[cid] = Some(old_of[u_dense]);
        }
    }

    let mut node_to_rep: HashMap<PrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();
    for (u_dense, &cid) in prev_class.iter().enumerate() {
        node_to_rep.insert(old_of[u_dense], rep_of_class[cid].unwrap());
    }

    // Rebuild edges of representatives from aggregated signatures
    for cid in 0..num_classes {
        if let Some(rep_idx) = rep_of_class[cid] {
            let sample_dense = prev_class.iter().position(|&c| c == cid).unwrap();
            // Aggregate edges by (edge_key, dest_class)
            let mut aggr: BTreeMap<(Option<GrammarTokenID>, usize), LLMTokenBV> = BTreeMap::new();
            for (ek, llm_bv, v_dense) in &raw_edges[sample_dense] {
                let dst_c = prev_class[*v_dense];
                aggr.entry((*ek, dst_c))
                    .and_modify(|e| *e |= llm_bv)
                    .or_insert(llm_bv.clone());
            }

            let mut new_children: BTreeMap<Option<GrammarTokenID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
            for ((ek, dest_class), llm_bv) in aggr {
                if llm_bv.is_empty() { continue; }
                let dest_rep = rep_of_class[dest_class].unwrap();
                new_children.entry(ek)
                    .or_default()
                    .entry(dest_rep)
                    .and_modify(|e| *e |= &llm_bv)
                    .or_insert(llm_bv);
            }

            let mut w = rep_idx.write(trie1_god).expect("write");
            *w.children_mut() = new_children;
            // Optionally, set live_tokens to union of outgoing BVs for diagnostics.
            let mut union_bv = LLMTokenBV::zeros();
            for (_ek, dm) in w.children().iter() {
                for (_dst, bv) in dm.iter() {
                    union_bv |= bv;
                }
            }
            w.value.live_tokens |= &union_bv;
        }
    }

    // Redirect roots to representatives
    for r in roots.values_mut() {
        if let Some(rep) = node_to_rep.get(r) {
            *r = *rep;
        }
    }

    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie1_god, &roots_vec2);
}

pub fn optimize_trie1_size(
    precomputed1: &mut BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    trie0_god: &Trie0GodWrapper,
    node0_to_node1_map: &HashMap<PrecomputeNode0Index, PrecomputeNode1Index>,
    ignore_terminal_id: Option<TerminalID>,
    internal_max_llm_token: usize,
    terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    config: &GrammarConstraintConfig,
    stage_vocab: &mut StageVocab,
    token_name_map: &bimap::BiBTreeMap<crate::glr::grammar::Terminal, usize>,
) {
    crate::debug!(2, "Starting Trie1 size optimization...");

    crate::debug!(2, "Initial Trie1 stats:");
    let mut stats = PrecomputeStats::default();
    crate::constraint_extra::calculate_final_stats1(precomputed1, &mut stats, trie1_god);
    crate::constraint_extra::print_precompute_stats1(&stats, token_name_map, trie1_god);

    // === Pass 1: Initial Simplification and Pruning ===
    simplify_none_edges_to_former_end_nodes_trie1(precomputed1, trie1_god, trie0_god, node0_to_node1_map);
    replace_ignore_token_edges_with_none_edges_trie1(precomputed1, trie1_god, ignore_terminal_id);
    // if config.optimize_trie1_early_flatten_epsilon {
    //     flatten_all_none_edges_trie1(precomputed1, trie1_god);
    // } else {
    //     shortcut_none_chains_trie1(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());
    // }
    constrain_bitvecs_trie1(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>(), internal_max_llm_token);
    prune_on_no_terminal_follow_trie1(precomputed1, trie1_god, terminal_follow_map, ignore_terminal_id);
    prune_nodes_not_reaching_end_trie1(precomputed1, trie1_god);
    prune_dead_paths_trie1(precomputed1, trie1_god, internal_max_llm_token);

    // === Pass 2: Minimization and further cleanup ===
    if config.optimize_trie1_minimize_by_signature {
        merge_nodes_trie1(precomputed1, trie1_god);
    }
    // if !config.optimize_trie1_early_flatten_epsilon {
    //     flatten_all_none_edges_trie1(precomputed1, trie1_god);
    // }
    // prune_nodes_not_reaching_end_trie1(precomputed1, trie1_god);
    // prune_dead_paths_trie1(precomputed1, trie1_god, internal_max_llm_token);

    // === Pass 3: Token-level optimizations ===
    if config.optimize_trie1_merge_equivalent_llm_tokens {
        merge_equivalent_llm_tokens_trie1(precomputed1, trie1_god, stage_vocab);
    }
    if config.optimize_trie1_reorder_llm_tokens {
        reorder_llm_tokens_for_range_minimization_trie1(precomputed1, trie1_god, stage_vocab);
    }

    // === Pass 4: Final Minimization and GC ===
    if config.optimize_trie1_minimize_by_signature {
        merge_nodes_trie1(precomputed1, trie1_god);
    }
    // prune_nodes_not_reaching_end_trie1(precomputed1, trie1_god);
    prune_dead_paths_trie1(precomputed1, trie1_god, internal_max_llm_token);
    Trie::gc(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());
    Trie::recompute_all_max_depths(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());

    crate::debug!(2, "Final Trie1 stats:");
    let mut stats = PrecomputeStats::default();
    crate::constraint_extra::calculate_final_stats1(precomputed1, &mut stats, trie1_god);
    crate::constraint_extra::print_precompute_stats1(&stats, token_name_map, trie1_god);
}

fn simplify_none_edges_to_former_end_nodes_trie1(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    trie0_god: &Trie0GodWrapper,
    node0_to_node1_map: &HashMap<PrecomputeNode0Index, PrecomputeNode1Index>,
) {
    crate::debug!(2, "Simplifying None edges to former end nodes in Trie1...");

    let mut former_end_nodes1: HashSet<PrecomputeNode1Index> = HashSet::new();
    for (node0_idx, node1_idx) in node0_to_node1_map {
        let node0_guard = node0_idx.read(trie0_god).unwrap();
        if node0_guard.value.final_tokenizer_state.is_some() {
            former_end_nodes1.insert(*node1_idx);
        }
    }

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    let mut nodes_to_make_end: HashSet<PrecomputeNode1Index> = HashSet::new();
    for a_arc in all_nodes {
        let mut edges_to_add = Vec::new();
        let mut none_edges_to_b_to_remove = Vec::new();

        { // read lock scope
            let a_guard = a_arc.read(trie1_god).unwrap();
            if let Some(none_dest_map) = a_guard.children().get(&None) {
                for (b_arc_wrapper, bv_ab) in none_dest_map {
                    if former_end_nodes1.contains(b_arc_wrapper) {
                        let b_arc = b_arc_wrapper.as_arc();
                        let b_guard = b_arc.read(trie1_god).unwrap();

                        if !b_guard.children().is_empty() {
                            // This is a candidate for simplification: A -(None)-> B, where B has children.
                            none_edges_to_b_to_remove.push(b_arc_wrapper.clone());

                            // B's children are the edges to add to A.
                            for (term_opt, c_dest_map) in b_guard.children() {
                                for (c_arc_wrapper, _bv_bc) in c_dest_map {
                                    // New edge: A -(term_opt)-> C
                                    // New BV is bv_ab, since bv_bc is all_tokens.
                                    let new_bv = bv_ab.clone();
                                    if !new_bv.is_empty() {
                                        edges_to_add.push((term_opt.clone(), c_arc_wrapper.clone(), new_bv));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } // end read lock

        if !none_edges_to_b_to_remove.is_empty() {
            let mut a_guard = a_arc.write(trie1_god).unwrap();

            // Remove the None edges that point to former end nodes that we are shortcutting.
            if let Some(none_dest_map) = a_guard.children_mut().get_mut(&None) {
                for b_to_remove in none_edges_to_b_to_remove {
                    none_dest_map.remove(&b_to_remove);
                }
            }

            // If the None edge map is now empty, remove it.
            if a_guard.children().get(&None).map_or(false, |m| m.is_empty()) {
                a_guard.children_mut().remove(&None);
            }

            // Add the new shortcut edges.
            for (term_opt, c_arc_wrapper, new_bv) in edges_to_add {
                let dest_map = a_guard.children_mut().entry(term_opt).or_default();
                dest_map.entry(c_arc_wrapper).or_insert_with(LLMTokenBV::zeros).bitor_assign(&new_bv);
            }
        }
    }

    crate::debug!(2, "Done simplifying None edges to former end nodes in Trie1.");
}

fn replace_ignore_token_edges_with_none_edges_trie1(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    ignore_terminal_id: Option<TerminalID>,
) {
    let ignore_tid = if let Some(id) = ignore_terminal_id {
        id
    } else {
        return;
    };

    crate::debug!(2, "Replacing ignore token edges with None edges in Trie1...");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);

    for node_arc in all_nodes {
        let mut node_guard = node_arc.write(trie1_god).expect("poison");
        if let Some(dest_map_to_move) = node_guard.children_mut().remove(&Some(ignore_tid)) {
            let dest_map_for_new_key = node_guard.children_mut().entry(None).or_default();
            for (dest_wrapper, edge_bv) in dest_map_to_move {
                if let Some(existing_bv) = dest_map_for_new_key.get_mut(&dest_wrapper) {
                    *existing_bv |= &edge_bv;
                } else {
                    dest_map_for_new_key.insert(dest_wrapper, edge_bv);
                }
            }
        }
    }
    crate::debug!(2, "Done replacing ignore token edges in Trie1.");
}
fn count_total_ranges_trie1(
    all_nodes: &[PrecomputeNode1Index],
    trie1_god: &Trie1GodWrapper,
) -> usize {
    let mut count = 0;
    for n in all_nodes {
        let g = n.read(trie1_god).expect("read");
        count += g.value.live_tokens.inner().ranges_len();
        for (_ek, dm) in g.children() {
            for (_dst, bv) in dm {
                count += bv.inner().ranges_len();
            }
        }
    }
    count
}

fn remap_llm_bv_many_to_one(bv: &LLMTokenBV, map_old_to_new: &BTreeMap<usize, usize>, max_token_id: usize) -> LLMTokenBV {
    if bv.is_empty() { return LLMTokenBV::zeros(); }

    let mut ranges_to_add = Vec::new();
    let mut elements_to_add = Vec::new();

    let process_range = |range: std::ops::RangeInclusive<usize>,
                         ranges_to_add: &mut Vec<_>,
                         elements_to_add: &mut Vec<_>| {
        let (mut current, end) = (*range.start(), *range.end());
        for (&k, &v) in map_old_to_new.range(current..=end) {
            if current < k {
                ranges_to_add.push(current..=k - 1);
            }
            elements_to_add.push(v);
            current = k + 1;
        }
        if current <= end {
            ranges_to_add.push(current..=end);
        }
    };

    if *bv == LLMTokenBV::max_ones() {
        process_range(0..=max_token_id, &mut ranges_to_add, &mut elements_to_add);
    } else {
        for range in bv.inner().ranges() {
            process_range(*range.start()..=*range.end(), &mut ranges_to_add, &mut elements_to_add);
        }
    }

    let mut new_set = RangeSetBlaze::from_iter(ranges_to_add);
    new_set.extend(elements_to_add);

    LLMTokenBV { inner: crate::datastructures::cache::intern_l1(new_set) }
}

fn remap_llm_bv_permutation(bv: &LLMTokenBV, map_old_to_new: &BTreeMap<usize, usize>, _max_token_id: usize) -> LLMTokenBV {
    // Fast‑paths for empty or full‑universe bitvectors.
    if bv.is_empty() { return LLMTokenBV::zeros(); }
    if *bv == LLMTokenBV::max_ones() { return LLMTokenBV::max_ones(); }

    let mut ranges_to_add = Vec::new();
    let mut elements_to_add = Vec::new();

    for range in bv.inner().ranges() {
        let (mut current, end) = (*range.start(), *range.end());

        for (&k, &v) in map_old_to_new.range(current..=end) {
            if current < k {
                // Identity mapping for elements not in map_old_to_new
                ranges_to_add.push(current..=k - 1);
            }
            elements_to_add.push(v);
            current = k + 1;
        }

        if current <= end {
            // Identity mapping for the rest of the range
            ranges_to_add.push(current..=end);
        }
    }

    let mut new_set = RangeSetBlaze::from_iter(ranges_to_add);
    new_set.extend(elements_to_add);

    LLMTokenBV { inner: crate::datastructures::cache::intern_l1(new_set) }
}

/// Merge equivalent internal LLM token ids in Trie1:
/// Two tokens are equivalent if they appear together in every occurrence across:
/// - node.value.live_tokens
/// - every edge's LLMTokenBV
///
/// Applies a many-to-one id mapping and merges masks accordingly.
pub fn merge_equivalent_llm_tokens_trie1(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    crate::debug!(2, "Merging equivalent LLM tokens in Trie1...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    if all_nodes.is_empty() { return; }

    // 1) Collect all unique bitsets to use as splitters.
    let mut all_bvs = HashSet::new();
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Merge Tokens (Collect BVs)", disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let g = n.read(trie1_god).expect("read");
        if !g.value.live_tokens.is_empty() {
            all_bvs.insert(g.value.live_tokens.clone());
        }
        for (_ek, dm) in g.children() {
            for (_dst, bv) in dm {
                if !bv.is_empty() {
                    all_bvs.insert(bv.clone());
                }
            }
        }
    }
    if all_bvs.is_empty() { return; }

    // 2) Partition refinement.
    let max_tok = stage_vocab.internal_max_llm_token;
    let mut token_to_class: Vec<usize> = vec![0; max_tok + 1];
    let mut class_to_tokens: HashMap<usize, Vec<usize>> = HashMap::new();
    class_to_tokens.insert(0, (0..=max_tok).collect());
    let mut num_classes = 1;

    #[cfg(not(rustrover))]
    let it = tqdm!(all_bvs.iter(), desc = "Trie1 Merge Tokens (Refine)", disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_bvs.iter();
    for splitter_bv in it {
        if *splitter_bv == LLMTokenBV::max_ones() { continue; }

        let mut members_in_splitter_by_class: HashMap<usize, Vec<usize>> = HashMap::new();
        for token in splitter_bv.iter() {
            if token <= max_tok {
                let class_id = token_to_class[token];
                members_in_splitter_by_class.entry(class_id).or_default().push(token);
            }
        }

        for (old_class_id, tokens_for_new_class) in members_in_splitter_by_class {
            let old_class_size = class_to_tokens.get(&old_class_id).map_or(0, |v| v.len());
            if old_class_size == 0 { continue; }

            if !tokens_for_new_class.is_empty() && tokens_for_new_class.len() < old_class_size {
                let new_class_id = num_classes;
                num_classes += 1;

                for &token in &tokens_for_new_class {
                    token_to_class[token] = new_class_id;
                }

                let old_class_tokens = class_to_tokens.get_mut(&old_class_id).unwrap();
                let moved_tokens_set: HashSet<_> = tokens_for_new_class.iter().cloned().collect();
                old_class_tokens.retain(|t| !moved_tokens_set.contains(t));
                
                class_to_tokens.insert(new_class_id, tokens_for_new_class);
            }
        }
    }

    // 3) Build many-to-one mapping from the final partition.
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    let mut merged_count = 0;
    for (_class_id, group) in &class_to_tokens {
        if group.len() <= 1 { continue; }
        let rep = *group.iter().min().unwrap();
        for &t in group {
            if t != rep {
                old_to_new.insert(t, rep);
                merged_count += 1;
            }
        }
    }
    let tokens_before = max_tok + 1;
    let tokens_after = num_classes;
    crate::debug!(2, "Trie1: merged LLM tokens. Before: {}, After: {}. ({} merged)", tokens_before, tokens_after, merged_count);
    if merged_count == 0 { return; }

    // Memoization cache for remapping.
    let mut memo: HashMap<LLMTokenBV, LLMTokenBV> = HashMap::new();

    // Precompute the mapped universal set once (used when a set equals max_ones())
    let mut mapped_universe = LLMTokenBV::zeros();
    for t in 0..=max_tok {
        let rep = old_to_new.get(&t).copied().unwrap_or(t);
        mapped_universe.insert(rep);
    }

    // Identify which concrete bitvectors are affected (by Arc pointer identity)
    let mut affected_ptrs: HashSet<*const RangeSetBlaze<usize>> = HashSet::new();
    for splitter in all_bvs {
        affected_ptrs.insert(Arc::as_ptr(&splitter.inner));
    }

    // 4) Apply mapping to trie in‑place, only where needed
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Merge (Remap In‑Place)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        // Quick check: does this node reference any affected bitvector?
        let needs_update = {
            let r = n.read(trie1_god).expect("read");
            let lv_ptr = Arc::as_ptr(&r.value.live_tokens.inner);
            if affected_ptrs.contains(&lv_ptr) {
                true
            } else {
                let mut touched = false;
                for (_ek, dm) in r.children() {
                    for (_dst, bv) in dm {
                        let bv_ptr = Arc::as_ptr(&bv.inner);
                        if affected_ptrs.contains(&bv_ptr) {
                            touched = true;
                            break;
                        }
                    }
                    if touched { break; }
                }
                touched
            }
        };
        if !needs_update { continue; }

        let mut w = n.write(trie1_god).expect("write");

        // Remap live_tokens if needed
        if !w.value.live_tokens.is_empty() {
            if w.value.live_tokens == LLMTokenBV::max_ones() {
                w.value.live_tokens = mapped_universe.clone();
            } else {
                let lv_ptr = Arc::as_ptr(&w.value.live_tokens.inner);
                if affected_ptrs.contains(&lv_ptr) {
                    let original_bv = w.value.live_tokens.clone();
                    w.value.live_tokens = memo.entry(original_bv)
                        .or_insert_with_key(|bv| remap_llm_bv_many_to_one(bv, &old_to_new, max_tok))
                        .clone();
                }
            }
        }

        // Remap children edge masks
        let old_children = std::mem::take(w.children_mut());
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in old_children {
            let mut new_dm: OrderedHashMap<PrecomputeNode1Index, LLMTokenBV> = OrderedHashMap::new();
            for (dst, bv) in dm {
                let bv_ptr = Arc::as_ptr(&bv.inner);
                let mapped = if bv.is_empty() {
                    LLMTokenBV::zeros()
                } else if bv == LLMTokenBV::max_ones() {
                    mapped_universe.clone()
                } else if affected_ptrs.contains(&bv_ptr) {
                    memo.entry(bv.clone())
                        .or_insert_with_key(|bv| remap_llm_bv_many_to_one(bv, &old_to_new, max_tok))
                        .clone()
                } else {
                    bv.clone()
                };
                if !mapped.is_empty() {
                    new_dm.entry(dst)
                        .and_modify(|e| *e |= &mapped)
                        .or_insert(mapped);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
	}
	// 5) Update stage vocab
	// Merge internal_to_original for tokens mapped into representatives
	#[cfg(not(rustrover))]
	let it = tqdm!(old_to_new.iter(), desc = "Trie1 Merge (Update Vocab)", total = old_to_new.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
	#[cfg(rustrover)]
	let it = old_to_new.iter();
	for (old, new_rep) in it {
		if old == new_rep { continue; }
		if let Some(moved) = stage_vocab.internal_to_original.remove(old) {
			let entry = stage_vocab.internal_to_original.entry(*new_rep).or_default();
			*entry |= &moved;
			for o in moved.iter() {
				stage_vocab.original_to_internal.insert(o, *new_rep);
			}
		}
	}
	// internal_max_llm_token stays the same here (holes may appear). A later reorder can compact.
}

/// Reorder internal LLM tokens (permutation) to reduce ranges in masks by clustering co-occurring tokens.
/// Conservative heuristic: sort by (descending frequency, then by id).
pub fn reorder_llm_tokens_for_range_minimization_trie1(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    stage_vocab: &mut StageVocab,
) {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    if all_nodes.is_empty() { return; }
    let ranges_before = count_total_ranges_trie1(&all_nodes, trie1_god);

    let max_tok = stage_vocab.internal_max_llm_token;

    // 1. Collect unique BV counts to optimize frequency calculation.
    let mut bv_counts: HashMap<LLMTokenBV, usize> = HashMap::new();
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Reorder (Collect BVs)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let g = n.read(trie1_god).expect("read");
        if !g.value.live_tokens.is_empty() {
            *bv_counts.entry(g.value.live_tokens.clone()).or_default() += 1;
        }
        for (_ek, dm) in g.children() {
            for (_dst, bv) in dm {
                if !bv.is_empty() {
                    *bv_counts.entry(bv.clone()).or_default() += 1;
                }
            }
        }
    }

    // 2. Compute token frequencies from unique BV counts.
    let mut freq: Vec<usize> = vec![0; max_tok + 1];
    #[cfg(not(rustrover))]
    let it = tqdm!(bv_counts.iter(), desc = "Trie1 Reorder (Count Frequencies)", total = bv_counts.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = bv_counts.iter();
    for (bv, &count) in it {
        if bv.is_all() {
            for t in 0..=max_tok {
                freq[t] += count;
            }
        } else {
            for t in bv.iter() {
                if t <= max_tok {
                    freq[t] += count;
                }
            }
        }
    }
    crate::debug!(2, "Done computing frequencies.");

    // Build ordering: tokens present at least once, sorted by (freq desc, id asc)
    let mut present: Vec<usize> = (0..=max_tok).filter(|t| freq[*t] > 0).collect();
    if present.is_empty() { return; }
    present.sort_by_key(|&t| (std::cmp::Reverse(freq[t]), t));

    // Build permutation
    let mut old_to_new: BTreeMap<usize, usize> = BTreeMap::new();
    for (new_id, old_id) in present.iter().enumerate() {
        old_to_new.insert(*old_id, new_id);
    }

    // Memoization cache
    let mut memo: HashMap<LLMTokenBV, LLMTokenBV> = HashMap::new();

    // Apply mapping to trie
    let mut new_states = Vec::with_capacity(all_nodes.len());
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Reorder (Remap Read)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let r = n.read(trie1_god).expect("read");
        let new_live_tokens = if r.value.live_tokens.is_empty() {
            r.value.live_tokens.clone()
        } else {
            memo.entry(r.value.live_tokens.clone())
                .or_insert_with_key(|bv| remap_llm_bv_permutation(bv, &old_to_new, max_tok))
                .clone()
        };
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in r.children() {
            let mut new_dm: OrderedHashMap<PrecomputeNode1Index, LLMTokenBV> = OrderedHashMap::new();
            for (dst, bv) in dm {
                let mapped = memo.entry(bv.clone())
                    .or_insert_with_key(|bv| remap_llm_bv_permutation(bv, &old_to_new, max_tok))
                    .clone();
                if !mapped.is_empty() {
                    new_dm.insert(dst.clone(), mapped);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek.clone(), new_dm);
            }
        }
        new_states.push((new_live_tokens, new_children));
    }
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter().enumerate(), desc = "Trie1 Reorder (Remap Write)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = all_nodes.iter().enumerate();
    for (i, n) in it {
        let mut w = n.write(trie1_god).expect("write");
        let (live_tokens, children) = &new_states[i];
        w.value.live_tokens = live_tokens.clone();
        *w.children_mut() = children.clone();
    }
	let ranges_after = count_total_ranges_trie1(&all_nodes, trie1_god);

	// Update stage vocab (pure permutation)
	let mut new_internal_to_original: BTreeMap<usize, LLMTokenBV> = BTreeMap::new();
	#[cfg(not(rustrover))]
	let it = tqdm!(stage_vocab.internal_to_original.clone().into_iter(), desc = "Trie1 Reorder (Vocab 1)", disable = !PROGRESS_BAR_ENABLED, leave = true);
	#[cfg(rustrover)]
	let it = stage_vocab.internal_to_original.clone().into_iter();
	for (old_id, setv) in it {
		if let Some(new_id) = old_to_new.get(&old_id) {
			new_internal_to_original.insert(*new_id, setv);
        }
    }
    stage_vocab.internal_to_original = new_internal_to_original;
    let mut new_original_to_internal: BTreeMap<usize, usize> = BTreeMap::new();
    #[cfg(not(rustrover))]
    let it = tqdm!(stage_vocab.original_to_internal.clone().into_iter(), desc = "Trie1 Reorder (Vocab 2)", disable = !PROGRESS_BAR_ENABLED, leave = true);
    #[cfg(rustrover)]
    let it = stage_vocab.original_to_internal.clone().into_iter();
    for (orig, old_internal) in it {
        if let Some(new_internal) = old_to_new.get(&old_internal) {
            new_original_to_internal.insert(orig, *new_internal);
        }
    }
    stage_vocab.original_to_internal = new_original_to_internal;
    stage_vocab.internal_max_llm_token = present.len().saturating_sub(1);
    crate::debug!(2, "Trie1 reordering complete. Ranges reduced from {} to {}. New max internal token ID: {}", ranges_before, ranges_after, stage_vocab.internal_max_llm_token);
}

/// Remove edges to nodes that cannot reach any end node, then GC unreachable nodes.
/// This is an exact pruning that preserves correctness: nodes that cannot reach an end
/// can never contribute tokens to the final mask in get_mask1, so their edges are dead.
fn prune_nodes_not_reaching_end_trie1(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
)
{
    crate::debug!(2, "Pruning Trie1 nodes that cannot reach any end node (reverse reachability)...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if roots_vec.is_empty() {
        return;
    }

    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // Build reverse adjacency: dest -> sources having an edge to dest (for any key)
    let mut incoming: HashMap<PrecomputeNode1Index, Vec<PrecomputeNode1Index>> = HashMap::new();
    for src in &all_nodes {
        let g = src.read(trie1_god).expect("read");
        for (_ek, dm) in g.children() {
            for (dst, _bv) in dm {
                incoming.entry(*dst).or_default().push(*src);
            }
        }
    }

    // Initialize worklist with all end nodes
    let mut productive: HashSet<PrecomputeNode1Index> = HashSet::new();
    let mut q: VecDeque<PrecomputeNode1Index> = VecDeque::new();
    let mut end_nodes_count = 0usize;
    for n in &all_nodes {
        let r = n.read(trie1_god).expect("read");
        if r.value.end {
            end_nodes_count += 1;
            if productive.insert(*n) {
                q.push_back(*n);
            }
        }
    }
    if end_nodes_count == 0 {
        // No end nodes present: nothing to prune under this criterion.
        crate::debug!(2, "No end nodes found in Trie1; skipping end-reachability pruning.");
        return;
    }

    // Reverse BFS: mark all nodes that can reach some end node
    while let Some(d) = q.pop_front() {
        if let Some(srcs) = incoming.get(&d) {
            for s in srcs {
                if productive.insert(*s) {
                    q.push_back(*s);
                }
            }
        }
    }

    let total_nodes = all_nodes.len();
    let productive_nodes = productive.len();
    let prunable = total_nodes.saturating_sub(productive_nodes);
    crate::debug!(2, "Trie1 end-reachability: total={}, productive={}, prunable={}", total_nodes, productive_nodes, prunable);
    if prunable == 0 {
        return;
    }

    // Remove any edge to a non-productive destination, recompute node live_tokens
    for n in &all_nodes {
        let mut w = n.write(trie1_god).expect("write");
        let mut new_children: BTreeMap<Option<GrammarTokenID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in w.children().clone() {
            let mut new_dm: OrderedHashMap<PrecomputeNode1Index, LLMTokenBV> = OrderedHashMap::new();
            for (dst, bv) in dm {
                if productive.contains(&dst) {
                    new_dm.insert(dst, bv);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
        // Recompute live_tokens = union of outgoing edge masks
        let mut lt = LLMTokenBV::zeros();
        for dm in w.children().values() {
            for bv in dm.values() {
                lt |= bv;
            }
        }
        w.value.live_tokens = lt;
    }

    // GC everything now unreachable from roots and recompute depths
    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::gc(trie1_god, &roots_vec2);
    Trie::recompute_all_max_depths(trie1_god, &roots_vec2);

    crate::debug!(2, "Finished end-reachability pruning in Trie1.");
}

fn prune_on_no_terminal_follow_trie1(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    terminal_follow_map: &BTreeMap<GrammarTokenID, BTreeSet<GrammarTokenID>>,
    ignore_terminal_id: Option<TerminalID>,
) {
    crate::debug!(2, "Pruning Trie1 based on terminal follow sets.");

    let initial_nodes_and_values: Vec<_> = roots.values()
        .map(|root_arc| (root_arc.clone(), None))
        .collect();

    type NodePtr = *const PrecomputeNode1;
    let mut edges_to_keep: HashMap<NodePtr, BTreeSet<Option<GrammarTokenID>>> = HashMap::new();

    Trie::special_map(
        trie1_god,
        initial_nodes_and_values,
        |predecessors: &Option<BTreeSet<GrammarTokenID>>, edge_key: &Option<GrammarTokenID>, _edge_bv, _child_node| {
            match edge_key {
                Some(t) if Some(*t) == ignore_terminal_id => Some(predecessors.clone()),
                Some(t) => Some(Some(BTreeSet::from([*t]))),
                None => Some(predecessors.clone()),
            }
        },
        |existing_set, new_set| {
            match (existing_set, new_set) {
                (None, _) => {},
                (existing_set @ _, None) => *existing_set = None,
                (Some(existing), Some(new)) => existing.extend(new),
            }
        },
        |node, maybe_all_immediate_predecessors| {
            if maybe_all_immediate_predecessors.is_none() {
                return true;
            }

            let mut allowed_follow_terminals = BTreeSet::new();
            if let Some(all_immediate_predecessors) = &*maybe_all_immediate_predecessors {
                for preceding_terminal in all_immediate_predecessors {
                    if let Some(follow_set) = terminal_follow_map.get(preceding_terminal) {
                        allowed_follow_terminals.extend(follow_set.iter().cloned());
                    }
                }
            }

            let keys_to_keep: BTreeSet<_> = node.children().keys().filter(|edge_key| {
                match edge_key {
                    Some(edge_terminal) => allowed_follow_terminals.contains(edge_terminal) || Some(*edge_terminal) == ignore_terminal_id,
                    None => true,
                }
            }).cloned().collect();

            let node_ptr: NodePtr = node;
            edges_to_keep.insert(node_ptr, keys_to_keep);
            true
        },
    );

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    for node_arc in all_nodes {
        let node_ptr: NodePtr = {
            let guard = node_arc.read(trie1_god).expect("poison");
            &*guard as *const _
        };
        if let Some(keys_to_keep) = edges_to_keep.get(&node_ptr) {
            let mut node_guard = node_arc.write(trie1_god).unwrap();
            node_guard.children_mut().retain(|k, _| keys_to_keep.contains(k));
        }
    }

    crate::debug!(2, "Finished pruning Trie1 based on terminal follow sets.");
}

/// Flattens None edges (epsilon-closure) by computing a per-node closure and rewriting
/// outgoing edges. Keyed edges from None-children are composed (bv_intersection) and pushed
/// up to the current node. Additionally, we preserve only those None edges that are required
/// to maintain reachability to end nodes via None-only paths.
///
/// For a node A:
/// - Start from A's existing keyed edges (Some(t) -> {dst -> bv}).
/// - For each None-edge A -(None, bv1)-> B, compose B's already-flattened keyed edges
///   with bv1 via intersection: bv_final = bv1 & bv_child_edge, and union into A's map.
/// - Preserve the None-edge A -(None)-> B if B can (via a None-only path) reach an end node.
///   This avoids dropping necessary epsilon paths to end/leaf nodes.
/// - Replace A's children with this new map (keyed edges plus the preserved None edges),
///   and recompute live_tokens.
///
/// Notes:
/// - We never turn A into an end node via None-closure. That would be unsound for partial masks.
///   End detection remains tied to explicit keyed paths to leaf/end nodes, and epsilon paths preserved explicitly.
/// - We process nodes in descending max_depth so children are processed (flattened) before parents.
fn flatten_all_none_edges_trie1(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) {
    crate::debug!(2, "Flattening None edges in Trie1 via epsilon-closure (keyed-only rewrite)...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie1_god, &roots_vec);

    let mut all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    // Sort by descending depth so that children are processed (flattened) first.
    all_nodes.sort_by_key(|idx| {
        idx.read(trie1_god).expect("read").max_depth
    });
    all_nodes.reverse();

    // Cache of flattened edges per node (primarily keyed edges; we may also keep
    // some None edges if necessary to preserve reachability to end).
    // Key: node index -> map: Some(terminal) -> OrderedHashMap<dst, bv>
    let mut flat_cache: HashMap<
        PrecomputeNode1Index,
        BTreeMap<Option<GrammarTokenID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>>
    > = HashMap::new();

    // Bottom-up memo: whether a node can reach an end node using only None edges (or is itself end).
    let mut none_reaches_end: HashMap<PrecomputeNode1Index, bool> = HashMap::new();

    // Work backwards from deepest leaves
    for node_idx in &all_nodes {
        // Snapshot current children to avoid lock re-entrancy
        let (children_snapshot, node_is_end) = {
            let g = node_idx.read(trie1_god).expect("read");
            (g.children().clone(), g.value.end)
        };

        let mut has_none_path_to_end_from_children = false;

        // Start with direct keyed edges
        let mut new_children: BTreeMap<Option<GrammarTokenID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in &children_snapshot {
            if ek.is_some() {
                let dest_map = new_children.entry(ek.clone()).or_default();
                for (dst, bv) in dm {
                    dest_map.entry(*dst)
                        .and_modify(|e| *e |= bv)
                        .or_insert(bv.clone());
                }
            }
        }

        // Compose None edges with child's flattened keyed edges (if any)
        if let Some(none_dest_map) = children_snapshot.get(&None) {
            for (child_idx, bv_none_edge) in none_dest_map {
                if bv_none_edge.is_empty() { continue; }

                // Check if child has a None-path to an end node.
                // Since we process bottom-up, this should be in the memo.
                let child_none_to_end = *none_reaches_end.get(child_idx).unwrap_or(&false);
                has_none_path_to_end_from_children |= child_none_to_end;

                // If it does, preserve the None edge.
                if child_none_to_end {
                    let dest_map_none = new_children.entry(None).or_default();
                    dest_map_none
                        .entry(*child_idx)
                        .and_modify(|e| *e |= bv_none_edge)
                        .or_insert(bv_none_edge.clone());
                }

                // Now, compose its keyed edges, regardless of whether we preserved the None edge.
                let child_keyed_edges_to_compose = if let Some(child_flat) = flat_cache.get(child_idx) {
                    child_flat.clone()
                } else {
                    // Fallback: use child's immediate keyed edges.
                    let child_children_snapshot: BTreeMap<Option<GrammarTokenID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = {
                        let cg = child_idx.read(trie1_god).expect("read");
                        cg.children().clone()
                    };
                    child_children_snapshot
                };

                for (ek2, dm2) in child_keyed_edges_to_compose {
                    if ek2.is_none() { continue; } // Only compose keyed edges
                    let dest_map = new_children.entry(ek2.clone()).or_default();
                    for (dst2, bv_child) in dm2 {
                        let mut composed = bv_none_edge.clone();
                        composed &= &bv_child;
                        if composed.is_empty() { continue; }
                        dest_map.entry(dst2)
                            .and_modify(|e| *e |= &composed)
                            .or_insert(composed);
                    }
                }
            }
        }

        // Finalize the memo for this node: it can reach an end via None-only path if it's an end
        // or if any of its None-children could reach an end via None-only paths.
        let this_none_reaches_end = node_is_end || has_none_path_to_end_from_children;
        none_reaches_end.insert(*node_idx, this_none_reaches_end);

        // Write back: replace children with new map and recompute live_tokens
        {
            let mut w = node_idx.write(trie1_god).expect("write");
            *w.children_mut() = new_children.clone();
            // Recompute live_tokens as union of all edge masks
            let mut lt = LLMTokenBV::zeros();
            for dm in w.children().values() {
                for bv in dm.values() {
                    lt |= bv;
                }
            }
            w.value.live_tokens = lt;
            // Do NOT change w.value.end here. End detection remains via keyed paths and preserved None paths.
        }

        flat_cache.insert(*node_idx, new_children);
    }

    // Cleanup: recompute depths and GC to drop eliminated intermediates
    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie1_god, &roots_vec2);
    Trie::gc(trie1_god, &roots_vec2);
    Trie::recompute_all_max_depths(trie1_god, &roots_vec2);

    crate::debug!(2, "Done flattening None edges in Trie1.");
}
