use std::collections::{BTreeMap, HashMap};

use ordered_hash_map::OrderedHashMap;
use range_set_blaze::RangeSetBlaze;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::trie::Trie;
use crate::datastructures::EntryApi;

/// Helper: Build a global set of token "atoms" per 'pop' by splitting the universe with every
/// LLMTokenBV mask used by any edge at that pop. Caps the number of atoms per pop to avoid
/// blowups; if exceeded, falls back to a single-universe atom at that pop.
pub fn build_global_token_atoms_by_pop(
    trie3_god: &Trie3GodWrapper,
    nodes: &[PrecomputeNode3Index],
    max_llm_token_id: usize,
    max_atoms_per_pop: usize,
) -> BTreeMap<isize, Vec<LLMTokenBV>> {
    let mut by_pop_masks: BTreeMap<isize, Vec<LLMTokenBV>> = BTreeMap::new();
    for n in nodes {
        let g = n.read(trie3_god).expect("read");
        for ((pop, llm_bv), _dm) in g.children() {
            if !llm_bv.is_empty() {
                by_pop_masks.entry(*pop).or_default().push(llm_bv.clone());
            }
        }
    }
    let universe = LLMTokenBV::ones(max_llm_token_id + 1);
    let mut atoms_by_pop: BTreeMap<isize, Vec<LLMTokenBV>> = BTreeMap::new();
    for (pop, masks) in by_pop_masks {
        // Start with a single block, split iteratively by every mask.
        let mut blocks = vec![universe.clone()];
        let mut aborted = false;
        for m in masks {
            let mut next_blocks: Vec<LLMTokenBV> = Vec::with_capacity(blocks.len().saturating_mul(2));
            for b in blocks.iter() {
                let inter = b & &m;
                if !inter.is_empty() {
                    next_blocks.push(inter);
                }
                let diff = b - &m;
                if !diff.is_empty() {
                    next_blocks.push(diff);
                }
            }
            if next_blocks.len() > max_atoms_per_pop {
                aborted = true;
                break;
            }
            blocks = next_blocks;
            if blocks.is_empty() {
                break;
            }
        }
        atoms_by_pop.insert(
            pop,
            if aborted || blocks.is_empty() {
                vec![universe.clone()]
            } else {
                blocks
            },
        );
    }
    atoms_by_pop
}

/// Behaviorally minimize per-node/per-pop edges by refining tokens into "semantic atoms".
/// For each node U and each pop P, consider the family of LLM masks {L_i} attached to the
/// outgoing edges and the corresponding destination maps D_i (dest -> SIDs). For a token t,
/// the one-step behavior is F_t(dest) = union over i: t ∈ L_i of D_i(dest). We:
///  - Partition the token universe U = ⋃_i L_i into atoms by iteratively splitting by each L_i.
///  - For each atom B, compute F_B(dest) = ⋃_{i: B ∩ L_i ≠ ∅} D_i(dest).
///  - Merge atoms with identical F_B (canonical dest->SIDs vector) and union their token-sets.
///  - Emit the resulting minimal set of edges for pop P.
/// We only rewrite when it improves a local size metric (sum of LLM range-counts + dest entries).
pub fn refine_edges_to_token_atoms_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
    max_blocks: usize,
) {
    crate::debug!(
        2,
        "Refining token-sets to semantic atoms in Trie3 (cap = {} blocks)...",
        max_blocks
    );
    if max_blocks == 0 {
        crate::debug!(3, "Refinement disabled due to max_blocks=0.");
        return;
    }
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    // Helper: compute range-count for a mask (bounded and constrained for safety).
    #[inline]
    fn ranges_len_bounded(m: &LLMTokenBV, max_id: usize) -> usize {
        let mut c = m.clone();
        c.constrain(max_id);
        c.inner().ranges_len()
    }

    let mut nodes_changed = 0usize;
    let mut total_improvement = 0isize;

    for node_idx in &all_nodes {
        let mut w = if let Some(guard) = node_idx.write(trie3_god) {
            guard
        } else {
            continue;
        };
        if w.children().is_empty() {
            continue;
        }

        // Take ownership of children for processing.
        let old_children = std::mem::take(w.children_mut());

        // Group by pop.
        use std::collections::BTreeMap as BTM;
        let mut by_pop: BTM<
            isize,
            Vec<(
                LLMTokenBV,
                OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
            )>,
        > = BTM::new();
        for ((pop, llm_bv), dm) in old_children {
            // Constrain to avoid spurious universe bits; drop empty.
            let mut l = llm_bv.clone();
            l.constrain(max_llm_token_id);
            if l.is_empty() {
                continue;
            }
            // Constrain SIDs in the map and drop empties
            let mut dm2: OrderedHashMap<PrecomputeNode3Index, StateIDBV> = OrderedHashMap::new();
            for (dst, mut sids) in dm {
                sids.constrain(max_state_id);
                if !sids.is_empty() {
                    dm2.insert(dst, sids);
                }
            }
            if dm2.is_empty() {
                continue;
            }
            by_pop.entry(pop).or_default().push((l, dm2));
        }

        // Build new children here
        let mut new_children: BTM<
            (isize, LLMTokenBV),
            OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
        > = BTM::new();
        let mut changed_locally = false;
        let mut local_improvement = 0isize;

        for (pop, mut entries) in by_pop {
            if entries.is_empty() {
                continue;
            }
            if entries.len() == 1 {
                // Nothing to refine; forward as is.
                let (l, dm) = entries.pop().unwrap();
                new_children.insert((pop, l), dm);
                continue;
            }

            // Universe U
            let mut universe = LLMTokenBV::zeros();
            for (l, _) in &entries {
                universe |= l;
            }
            universe.constrain(max_llm_token_id);
            if universe.is_empty() {
                continue;
            }

            // Atomization: iteratively split by each L_i
            let mut blocks: Vec<LLMTokenBV> = vec![universe.clone()];
            let mut aborted = false;
            for (l, _) in &entries {
                let mut next_blocks: Vec<LLMTokenBV> =
                    Vec::with_capacity(blocks.len().saturating_mul(2));
                for b in blocks.iter() {
                    let inter = b & l;
                    if !inter.is_empty() {
                        next_blocks.push(inter);
                    }
                    let diff = b - l;
                    if !diff.is_empty() {
                        next_blocks.push(diff);
                    }
                }
                if next_blocks.len() > max_blocks {
                    aborted = true;
                    break;
                }
                blocks = next_blocks;
                if blocks.is_empty() {
                    break;
                }
            }

            // Local size metric before:
            let mut old_cost_ranges = 0usize;
            let mut old_cost_dests = 0usize;
            for (l, dm) in &entries {
                old_cost_ranges += ranges_len_bounded(l, max_llm_token_id);
                old_cost_dests += dm.len();
            }
            let old_cost = (old_cost_ranges + old_cost_dests) as isize;

            if aborted || blocks.is_empty() {
                // Fallback: keep original entries
                for (l, dm) in entries {
                    new_children.insert((pop, l), dm);
                }
                continue;
            }

            // For each block, compute unioned dest->SIDs, then group by identical canonical dest-vector.
            use std::collections::HashMap as HM;
            let mut grouped: HM<Vec<(PrecomputeNode3Index, StateIDBV)>, LLMTokenBV> = HM::new();

            for b in blocks.iter() {
                let mut dest_agg: BTM<PrecomputeNode3Index, StateIDBV> = BTM::new();
                for (l, dm) in &entries {
                    if (b & l).is_empty() {
                        continue;
                    }
                    for (dst, sids) in dm {
                        dest_agg
                            .entry(*dst)
                            .and_modify(|e| *e |= sids)
                            .or_insert_with(|| sids.clone());
                    }
                }
                if dest_agg.is_empty() {
                    continue;
                }
                let dest_vec: Vec<(PrecomputeNode3Index, StateIDBV)> = dest_agg.into_iter().collect();
                let entry = grouped.entry(dest_vec).or_insert_with(LLMTokenBV::zeros);
                *entry |= b;
            }

            // Compute new cost
            let mut new_cost_ranges = 0usize;
            let mut new_cost_dests = 0usize;
            for (dest_vec, lnew) in &grouped {
                let mut ln = lnew.clone();
                ln.constrain(max_llm_token_id);
                if ln.is_empty() {
                    continue;
                }
                new_cost_ranges += ranges_len_bounded(&ln, max_llm_token_id);
                new_cost_dests += dest_vec.len();
            }
            let new_cost = (new_cost_ranges + new_cost_dests) as isize;

            if new_cost < old_cost {
                changed_locally = true;
                local_improvement += old_cost - new_cost;
                // Emit refined edges
                for (dest_vec, lnew) in grouped {
                    let mut ln = lnew.clone();
                    ln.constrain(max_llm_token_id);
                    if ln.is_empty() {
                        continue;
                    }
                    let mut dm_out: OrderedHashMap<PrecomputeNode3Index, StateIDBV> =
                        OrderedHashMap::new();
                    for (dst, sids) in dest_vec {
                        if !sids.is_empty() {
                            dm_out.insert(dst, sids);
                        }
                    }
                    if !dm_out.is_empty() {
                        new_children.insert((pop, ln), dm_out);
                    }
                }
            } else {
                // Keep originals (no improvement)
                for (l, dm) in entries {
                    new_children.insert((pop, l), dm);
                }
            }
        }

        // Recompute live tokens as union of all outgoing LLM masks at this node.
        let mut new_live = LLMTokenBV::zeros();
        for ((_, l), _) in &new_children {
            new_live |= l;
        }
        w.value.live_tokens = new_live;
        *w.children_mut() = new_children;

        if changed_locally {
            nodes_changed += 1;
            total_improvement += local_improvement;
        }
    }

    crate::debug!(
        2,
        "Refine token-atoms: nodes_changed={}, total_improvement(metric)={}",
        nodes_changed,
        total_improvement
    );
}

/// Exact token-atom refinement (per node and per pop) using the Boolean algebra of the outgoing
/// LLMTokenBV masks. This constructs atoms by iteratively splitting with each L_i and groups
/// atoms by identical one-step semantics F_B (dest -> union SIDs). The candidate result is adopted
/// only if it strictly decreases a local cost metric:
///   cost = sum over edges of (llm_bv.ranges_len + number_of_dest_entries).
/// This guarantees no blow-up while ensuring semantic minimality when adopted.
pub fn refine_edges_to_token_atoms_trie3_exact(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
) {
    crate::debug!(2, "Refining token-sets to semantic atoms in Trie3 (exact)...");
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    #[inline]
    fn ranges_len_bounded(m: &LLMTokenBV, max_id: usize) -> usize {
        let mut c = m.clone();
        c.constrain(max_id);
        c.inner().ranges_len()
    }

    for node_idx in &all_nodes {
        let mut w = if let Some(guard) = node_idx.write(trie3_god) {
            guard
        } else {
            continue;
        };
        if w.children().is_empty() {
            continue;
        }

        // Take old children and canonicalize inputs
        let old_children = std::mem::take(w.children_mut());
        use std::collections::BTreeMap as BTM;
        let mut by_pop: BTM<
            isize,
            Vec<(
                LLMTokenBV,
                OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
            )>,
        > = BTM::new();

        for ((pop, mut llm_bv), dm) in old_children {
            llm_bv.constrain(max_llm_token_id);
            if llm_bv.is_empty() {
                continue;
            }
            let mut dm2: OrderedHashMap<PrecomputeNode3Index, StateIDBV> = OrderedHashMap::new();
            for (dst, mut sids) in dm {
                sids.constrain(max_state_id);
                if !sids.is_empty() {
                    dm2.insert(dst, sids);
                }
            }
            if dm2.is_empty() {
                continue;
            }
            by_pop.entry(pop).or_default().push((llm_bv, dm2));
        }

        // Build a reconstruction of the old children (for cost comparison or fallback)
        let mut old_as_children: BTM<
            (isize, LLMTokenBV),
            OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
        > = BTM::new();
        for (pop, entries) in &by_pop {
            for (l, dm) in entries {
                old_as_children.insert((*pop, l.clone()), dm.clone());
            }
        }
        let mut old_cost_ranges = 0usize;
        let mut old_cost_dests = 0usize;
        for ((_, l), dm) in &old_as_children {
            old_cost_ranges += ranges_len_bounded(l, max_llm_token_id);
            old_cost_dests += dm.len();
        }
        let old_cost = (old_cost_ranges + old_cost_dests) as isize;

        // Candidate exact representation
        let mut new_children: BTM<
            (isize, LLMTokenBV),
            OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
        > = BTM::new();

        for (pop, entries) in by_pop {
            if entries.is_empty() {
                continue;
            }
            if entries.len() == 1 {
                let (l, dm) = entries.into_iter().next().unwrap();
                new_children.insert((pop, l), dm);
                continue;
            }

            // Universe of tokens for this (node,pop)
            let mut universe = LLMTokenBV::zeros();
            for (l, _) in &entries {
                universe |= l;
            }
            universe.constrain(max_llm_token_id);
            if universe.is_empty() {
                continue;
            }

            // Build atoms by iterative splitting
            let mut blocks: Vec<LLMTokenBV> = vec![universe];
            for (l, _) in &entries {
                let mut next_blocks: Vec<LLMTokenBV> =
                    Vec::with_capacity(blocks.len().saturating_mul(2));
                for b in blocks.into_iter() {
                    let inter = &b & l;
                    if !inter.is_empty() {
                        next_blocks.push(inter);
                    }
                    let diff = &b - l;
                    if !diff.is_empty() {
                        next_blocks.push(diff);
                    }
                }
                blocks = next_blocks;
                if blocks.is_empty() {
                    break;
                }
            }
            if blocks.is_empty() {
                continue;
            }

            // Group atoms by their semantic effect (canonical dest->SIDs vector)
            use std::collections::HashMap as HM;
            let mut grouped: HM<Vec<(PrecomputeNode3Index, StateIDBV)>, LLMTokenBV> = HM::new();
            for b in blocks.into_iter() {
                let mut dest_agg: BTM<PrecomputeNode3Index, StateIDBV> = BTM::new();
                for (l, dm) in &entries {
                    if (&b & l).is_empty() {
                        continue;
                    }
                    for (dst, sids) in dm {
                        dest_agg
                            .entry(*dst)
                            .and_modify(|e| *e |= sids)
                            .or_insert_with(|| sids.clone());
                    }
                }
                if dest_agg.is_empty() {
                    continue;
                }
                let dest_vec: Vec<(PrecomputeNode3Index, StateIDBV)> = dest_agg.into_iter().collect();
                if let Some(acc) = grouped.get_mut(&dest_vec) {
                    *acc |= &b;
                } else {
                    grouped.insert(dest_vec, b.clone());
                }
            }

            // Emit grouped atoms as candidate edges
            for (dest_vec, mut ln) in grouped {
                ln.constrain(max_llm_token_id);
                if ln.is_empty() {
                    continue;
                }
                let mut dm_out: OrderedHashMap<PrecomputeNode3Index, StateIDBV> =
                    OrderedHashMap::new();
                for (dst, sids) in dest_vec {
                    if !sids.is_empty() {
                        dm_out.insert(dst, sids);
                    }
                }
                if !dm_out.is_empty() {
                    new_children.insert((pop, ln), dm_out);
                }
            }
        }

        // Compare costs
        let mut new_cost_ranges = 0usize;
        let mut new_cost_dests = 0usize;
        for ((_, l), dm) in &new_children {
            new_cost_ranges += ranges_len_bounded(l, max_llm_token_id);
            new_cost_dests += dm.len();
        }
        let new_cost = (new_cost_ranges + new_cost_dests) as isize;

        // Adopt only if strictly better; otherwise restore original
        let chosen_children = if new_cost < old_cost {
            &new_children
        } else {
            &old_as_children
        };

        // Recompute live tokens and commit
        let mut new_live = LLMTokenBV::zeros();
        for ((_, l), _) in chosen_children {
            new_live |= l;
        }
        w.value.live_tokens = new_live;
        *w.children_mut() = chosen_children.clone();
    }
    crate::debug!(2, "Finished exact token-atom refinement.");
}
