use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
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
use crate::datastructures::EntryApi;
use crate::datastructures::gss::LLMTokenBV;
use crate::datastructures::trie::Trie;
use crate::types::{TerminalID as GrammarTokenID, TerminalID};
use crate::tokenizer::TokenizerStateID;
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::ordered_hash_map::Retain;

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

    simplify_none_edges_to_former_end_nodes_trie1(
        precomputed1,
        trie1_god,
        trie0_god,
        node0_to_node1_map,
    );
    replace_ignore_token_edges_with_none_edges_trie1(precomputed1, trie1_god, ignore_terminal_id);
    simplify_none_edges_trie1(precomputed1, trie1_god);
    Trie::recompute_all_max_depths(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());

    prune_dead_paths_trie1(precomputed1, trie1_god, internal_max_llm_token);
    prune_on_no_terminal_follow_trie1(precomputed1, trie1_god, terminal_follow_map, ignore_terminal_id);
    prune_dead_paths_trie1(precomputed1, trie1_god, internal_max_llm_token);

    Trie::recompute_all_max_depths(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());
    factor_common_destinations_trie1(precomputed1, trie1_god);
    merge_nodes_trie1(precomputed1, trie1_god);
    Trie::gc(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());

    Trie::recompute_all_max_depths(trie1_god, &precomputed1.values().cloned().collect::<Vec<_>>());

    if config.optimize_trie1_merge_equivalent_llm_tokens {
        merge_equivalent_llm_tokens_trie1(precomputed1, trie1_god, stage_vocab);
    }
    if config.optimize_trie1_reorder_llm_tokens {
        reorder_llm_tokens_for_range_minimization_trie1(precomputed1, trie1_god, stage_vocab);
    }

    // Rerun token optimizations at the end.
    if config.optimize_trie1_merge_equivalent_llm_tokens {
        merge_equivalent_llm_tokens_trie1(precomputed1, trie1_god, stage_vocab);
    }
    // Always run normalization pass after potential token changes.
    optimize_state_masks_and_edges_trie1(precomputed1, trie1_god);
    if config.optimize_trie1_reorder_llm_tokens {
        reorder_llm_tokens_for_range_minimization_trie1(precomputed1, trie1_god, stage_vocab);
    }

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

                        if b_guard.children().is_empty() {
                            // This former end node has no successors, so it's a valid end point.
                            // Mark it to become a true end node. The edge is preserved.
                            nodes_to_make_end.insert(b_arc_wrapper.clone());
                        } else {
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

    for node_to_make_end in nodes_to_make_end {
        let mut guard = node_to_make_end.write(trie1_god).unwrap();
        guard.value.end = true;
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

fn simplify_none_edges_trie1(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) {
    crate::debug!(2, "Simplifying None edges in Trie1...");
    let root_node_ptrs: HashSet<PrecomputeNode1Index> = roots.values().cloned().collect();
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    let mut arc_by_ptr: HashMap<PrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();
    for n in &all_nodes {
        arc_by_ptr.insert(*n, n.clone());
    }

    let mut incoming: HashMap<
        PrecomputeNode1Index,
        Vec<(PrecomputeNode1Index, Option<GrammarTokenID>, LLMTokenBV)>,
    > = HashMap::new();
    let mut none_edges_from: HashMap<
        PrecomputeNode1Index,
        Vec<(PrecomputeNode1Index, LLMTokenBV)>,
    > = HashMap::new();
    let mut none_union: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();

    for src_arc in &all_nodes {
        let src_ptr = src_arc;
        let guard = src_arc.read(trie1_god).expect("poison");
        for (ek, dest_map) in guard.children().iter() {
            for (child_wrap, ev_bv) in dest_map.iter() {
                let child_arc = child_wrap.as_arc().clone();
                let child_ptr = child_arc;
                incoming.entry(child_ptr)
                    .or_default()
                    .push((src_arc.clone(), ek.clone(), ev_bv.clone()));
            }
        }
        if let Some(dest_map) = guard.children().get(&None) {
            let list = none_edges_from.entry(*src_ptr).or_default();
            for (child_wrap, ev_bv) in dest_map.iter() {
                list.push((child_wrap.as_arc().clone(), ev_bv.clone()));
                let entry = none_union.entry(*src_ptr).or_insert_with(LLMTokenBV::zeros);
                *entry |= ev_bv;
            }
        }
    }

    for (b_ptr, none_edges) in none_edges_from.into_iter() {
        let union_mask = match none_union.get(&b_ptr) {
            Some(bv) if !bv.is_empty() => bv.clone(),
            _ => continue,
        };
        let in_edges = match incoming.get(&b_ptr) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => {
                if root_node_ptrs.contains(&b_ptr) {
                    continue;
                }
                if let Some(b_arc) = arc_by_ptr.get(&b_ptr).cloned() {
                    let mut b_guard = b_arc.write(trie1_god).expect("poison");
                    b_guard.children_mut().remove(&None);
                }
                continue;
            }
        };

        let b_arc = arc_by_ptr.get(&b_ptr).unwrap().clone();
        let b_key = b_arc.clone();

        for (a_arc, edge_key, bv1_original) in in_edges.into_iter() {
            if edge_key.is_none() { continue; }

            let mut total_to_move = bv1_original.clone();
            total_to_move &= &union_mask;
            if total_to_move.is_empty() {
                continue;
            }

            let mut a_guard = a_arc.write(trie1_god).expect("poison");
            let dest_map = a_guard.children_mut().entry(edge_key.clone()).or_default();

            for (c_arc, bv2) in &none_edges {
                let mut to_move_for_c = bv1_original.clone();
                to_move_for_c &= bv2;
                if to_move_for_c.is_empty() {
                    continue;
                }
                let c_key = c_arc.clone();
                if let Some(existing_ev) = dest_map.get_mut(&c_key) {
                    *existing_ev |= &to_move_for_c;
                } else {
                    dest_map.insert(c_key, to_move_for_c);
                }
            }

            let mut remove_b_edge = false;
            if let Some(ev_ab) = dest_map.get_mut(&b_key) {
                *ev_ab -= &total_to_move;
                remove_b_edge = ev_ab.is_empty();
            }
            if remove_b_edge {
                dest_map.remove(&b_key);
            }
        }

        {
            let mut b_guard = b_arc.write(trie1_god).expect("poison");
            b_guard.children_mut().remove(&None);
        }
    }
    crate::debug!(2, "Done simplifying None edges in Trie1.");
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
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Merge Tokens (Collect BVs)", disable = !PROGRESS_BAR_ENABLED, leave = false);
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
    let it = tqdm!(all_bvs.iter(), desc = "Trie1 Merge Tokens (Refine)", disable = !PROGRESS_BAR_ENABLED, leave = false);
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
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Merge (Remap In‑Place)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false);
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
	let it = tqdm!(old_to_new.iter(), desc = "Trie1 Merge (Update Vocab)", total = old_to_new.len(), disable = !PROGRESS_BAR_ENABLED, leave = false);
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

    // Count frequencies directly to avoid slow HashMap<LLMTokenBV, ...>
    let mut freq: Vec<usize> = vec![0; max_tok + 1];
    #[cfg(not(rustrover))]
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Reorder (Count Frequencies)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false);
    #[cfg(rustrover)]
    let it = all_nodes.iter();
    for n in it {
        let g = n.read(trie1_god).expect("read");
        let live_tokens = &g.value.live_tokens;
        if !live_tokens.is_empty() {
            if live_tokens.is_all() {
                for t in 0..=max_tok { freq[t] += 1; }
            } else {
                for t in live_tokens.iter() {
                    if t <= max_tok { freq[t] += 1; }
                }
            }
        }
        for (_ek, dm) in g.children() {
            for (_dst, bv) in dm {
                if !bv.is_empty() {
                    if bv.is_all() {
                        for t in 0..=max_tok { freq[t] += 1; }
                    } else {
                        for t in bv.iter() {
                            if t <= max_tok { freq[t] += 1; }
                        }
                    }
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
    let it = tqdm!(all_nodes.iter(), desc = "Trie1 Reorder (Remap Read)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false);
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
    let it = tqdm!(all_nodes.iter().enumerate(), desc = "Trie1 Reorder (Remap Write)", total = all_nodes.len(), disable = !PROGRESS_BAR_ENABLED, leave = false);
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
	let it = tqdm!(stage_vocab.internal_to_original.clone().into_iter(), desc = "Trie1 Reorder (Vocab 1)", disable = !PROGRESS_BAR_ENABLED, leave = false);
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
    let it = tqdm!(stage_vocab.original_to_internal.clone().into_iter(), desc = "Trie1 Reorder (Vocab 2)", disable = !PROGRESS_BAR_ENABLED, leave = false);
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

/// Conservative normalization pass for Trie1:
/// - Coalesce duplicate destination entries (union LLMBV) for same child under a terminal key.
/// - Remove empty masks.
pub fn optimize_state_masks_and_edges_trie1(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    for n in &all_nodes {
        let mut w = n.write(trie1_god).expect("write");
        let mut new_children: BTreeMap<Option<crate::types::TerminalID>, OrderedHashMap<PrecomputeNode1Index, LLMTokenBV>> = BTreeMap::new();
        for (ek, dm) in w.children().clone() {
            // Union masks for same dst
            let mut coalesced: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();
            for (dst, bv) in dm {
                let entry = coalesced.entry(dst).or_insert_with(LLMTokenBV::zeros);
                *entry |= &bv;
            }
            let mut new_dm: OrderedHashMap<PrecomputeNode1Index, LLMTokenBV> = OrderedHashMap::new();
            for (dst, bv) in coalesced {
                if !bv.is_empty() {
                    new_dm.insert(dst, bv);
                }
            }
            if !new_dm.is_empty() {
                new_children.insert(ek, new_dm);
            }
        }
        *w.children_mut() = new_children;
    }
}

fn prune_dead_paths_trie1(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
    internal_max_llm_token: usize,
) {
    crate::debug!(2, "Pruning dead paths from Trie1.");
    let mut live_tokens_cache: HashMap<PrecomputeNode1Index, LLMTokenBV> = HashMap::new();
    let all_llm_tokens = HybridBitset::ones(internal_max_llm_token + 1);
    for root_arc in roots.values() {
        get_live_tokens_and_prune_trie1(root_arc.clone(), &mut live_tokens_cache, trie1_god, &all_llm_tokens);
    }
    crate::debug!(2, "Finished pruning dead paths from Trie1.");
}

fn get_live_tokens_and_prune_trie1(
    node_wrapper: PrecomputeNode1Index,
    live_tokens_cache: &mut HashMap<PrecomputeNode1Index, LLMTokenBV>,
    trie1_god: &Trie1GodWrapper,
    all_llm_tokens: &LLMTokenBV,
) -> LLMTokenBV {
    if let Some(cached_bv) = live_tokens_cache.get(&node_wrapper) {
        return cached_bv.clone();
    }
    live_tokens_cache.insert(node_wrapper.clone(), LLMTokenBV::zeros());

    let node_arc = node_wrapper.as_arc().clone();

    let children_to_check: Vec<PrecomputeNode1Index> = {
        let node_guard = node_arc.read(trie1_god).unwrap();
        node_guard.children().values().flat_map(|dest_map| dest_map.keys().cloned()).collect()
    };

    for child_wrapper in children_to_check {
        get_live_tokens_and_prune_trie1(child_wrapper, live_tokens_cache, trie1_god, all_llm_tokens);
    }

    let mut node_guard = node_arc.write(trie1_god).unwrap();

    node_guard.children_mut().retain(|_edge_key, dest_map| {
        dest_map.retain(|child_wrapper, edge_value_bv| {
            let live_tokens_from_child = live_tokens_cache.get(child_wrapper)
                .expect("Child not found in live_tokens_cache. Logic error.");
            let live_tokens_for_this_edge = &*edge_value_bv & live_tokens_from_child;
            if live_tokens_for_this_edge.is_empty() {
                false
            } else {
                *edge_value_bv = live_tokens_for_this_edge;
                true
            }
        });
        !dest_map.is_empty()
    });

    let mut current_node_live_tokens = LLMTokenBV::zeros();
    for dest_map in node_guard.children().values() {
        for edge_bv in dest_map.values() {
            current_node_live_tokens |= edge_bv;
        }
    }
    node_guard.value.live_tokens = current_node_live_tokens.clone();

    let is_end_node = node_guard.value.end;
    drop(node_guard);

    let returned_live_tokens = if is_end_node {
        all_llm_tokens.clone()
    } else {
        current_node_live_tokens
    };

    live_tokens_cache.insert(node_wrapper, returned_live_tokens.clone());
    returned_live_tokens
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

fn factor_common_destinations_trie1(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) {
    crate::debug!(2, "Factoring out common destinations in Trie1.");
    const MIN_INCOMING_EDGES_FOR_FACTORING: usize = 3;

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie1_god, &roots_vec);
    let arc_map: HashMap<_, _> = all_nodes.iter().map(|n| (n, n.clone())).collect();

    let mut incoming_map: HashMap<
        PrecomputeNode1Index,
        HashMap<
            Option<GrammarTokenID>,
            Vec<(PrecomputeNode1Index, LLMTokenBV)>,
        >,
    > = HashMap::new();

    for src_arc in &all_nodes {
        let src_ptr = src_arc;
        let guard = src_arc.read(trie1_god).expect("poison");
        for (edge_key, dest_map) in guard.children() {
            if edge_key.is_some() {
                for (dest_wrapper, bv) in dest_map {
                    let dest_arc = dest_wrapper.as_arc();
                    let dest_ptr = dest_arc;
                    incoming_map.entry(*dest_ptr).or_default().entry(edge_key.clone()).or_default().push((*src_ptr, bv.clone()));
                }
            }
        }
    }

    for (dest_ptr, edges_by_key) in incoming_map {
        for (edge_key, sources) in edges_by_key {
            if sources.len() >= MIN_INCOMING_EDGES_FOR_FACTORING {
                let dest_arc = arc_map.get(&dest_ptr).unwrap().clone();
                let intermediate_node = PrecomputeNode1Index::new(trie1_god.insert(PrecomputeNode1::new(PrecomputedNodeContents::internal())));

                let mut union_bv = LLMTokenBV::zeros();
                for (_, bv) in &sources {
                    union_bv |= bv;
                }

                {
                    let mut intermediate_guard = intermediate_node.write(trie1_god).expect("poison");
                    let mut edge_val_opt = Some(union_bv.clone());
                    intermediate_guard.try_insert_unchecked(edge_key.clone(), &mut edge_val_opt, dest_arc.clone());
                    intermediate_guard.value.live_tokens |= &union_bv;
                }

                for (src_ptr, bv) in &sources {
                    let src_arc = arc_map.get(src_ptr).unwrap();
                    let mut src_guard = src_arc.write(trie1_god).expect("poison");

                    if let Some(dest_map_for_key) = src_guard.children_mut().get_mut(&edge_key) {
                        dest_map_for_key.remove(&dest_arc.clone());
                        if dest_map_for_key.is_empty() {
                            src_guard.children_mut().remove(&edge_key);
                        }
                    }

                    let mut edge_val_opt = Some(bv.clone());
                    src_guard.try_insert_unchecked(None, &mut edge_val_opt, intermediate_node.clone());
                    src_guard.value.live_tokens |= bv;
                }
            }
        }
    }
    crate::debug!(2, "Finished factoring common destinations in Trie1.");
}

fn merge_nodes_trie1(
    roots: &mut BTreeMap<TokenizerStateID, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) {
    crate::debug!(2, "Merging identical subtrees in Trie1.");
    let mut canonical_nodes: HashMap<PrecomputeNode1, PrecomputeNode1Index> = HashMap::new();
    let mut visited: HashMap<PrecomputeNode1Index, PrecomputeNode1Index> = HashMap::new();

    let mut new_roots = BTreeMap::new();
    for (sid, root_arc) in roots.iter() {
        let canonical_root = deduplicate_recursive_trie1(root_arc.clone(), &mut canonical_nodes, &mut visited, trie1_god);
        new_roots.insert(*sid, canonical_root);
    }
    *roots = new_roots;
    crate::debug!(2, "Finished merging subtrees in Trie1. Canonical nodes: {}", canonical_nodes.len());
}

fn deduplicate_recursive_trie1(
    node_arc: PrecomputeNode1Index,
    canonical_nodes: &mut HashMap<PrecomputeNode1, PrecomputeNode1Index>,
    visited: &mut HashMap<PrecomputeNode1Index, PrecomputeNode1Index>,
    trie1_god: &Trie1GodWrapper,
) -> PrecomputeNode1Index {
    let node_ptr = node_arc;
    if let Some(canonical_arc) = visited.get(&node_ptr) {
        return canonical_arc.clone();
    }

    // Mark as visited early to break potential cycles.
    visited.insert(node_ptr, node_arc.clone());

    // Snapshot children under a short-lived read lock, then drop it before recursing.
    let children_snapshot: Vec<(
        Option<GrammarTokenID>,
        Vec<(PrecomputeNode1Index, LLMTokenBV)>,
    )> = {
        let g = node_arc.read(trie1_god).unwrap();
        g.children()
            .iter()
            .map(|(ek, dest_map)| {
                let entries = dest_map
                    .iter()
                    .map(|(node_ptr, ev)| (node_ptr.clone(), ev.clone()))
                    .collect::<Vec<_>>();
                (ek.clone(), entries)
            })
            .collect()
    };

    // Rebuild children map with canonicalized children (no locks held on the current node).
    let mut new_children_map = BTreeMap::new();
    let mut children_changed = false;
    for (edge_key, entries) in children_snapshot {
        let mut new_dest_map = OrderedHashMap::new();
        for (child_arc, edge_val) in entries {
            let canonical_child_arc = deduplicate_recursive_trie1(
                child_arc.clone(),
                canonical_nodes,
                visited,
                trie1_god,
            );
            if child_arc != canonical_child_arc {
                children_changed = true;
            }
            new_dest_map.insert(canonical_child_arc, edge_val);
        }
        if !new_dest_map.is_empty() {
            new_children_map.insert(edge_key, new_dest_map);
        }
    }

    // Write back updated children; avoid recompute_max_depth here to prevent lock re-entrancy.
    if children_changed {
        let mut g = node_arc.write(trie1_god).unwrap();
        *g.children_mut() = new_children_map;
        // Depths are recomputed globally after merging:
        // Trie::recompute_all_max_depths(...) is invoked by the caller.
    }

    // Canonicalize the current node by content after potential child rewrites.
    let canonical_arc = {
        let g = node_arc.read(trie1_god).unwrap();
        let node_content = (*g).clone();
        canonical_nodes
            .entry(node_content)
            .or_insert_with(|| node_arc.clone())
            .clone()
    };

    visited.insert(node_ptr, canonical_arc.clone());
    canonical_arc
}
