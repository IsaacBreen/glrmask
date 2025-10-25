use std::collections::BTreeMap as OrderedBTreeMap;

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{
    LLMTokenBV, PrecomputeNode3Index, PrecomputedNodeContents, StateIDBV, Trie3GodWrapper,
};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::Trie;

/// Root fanout factoring (super-pass).
/// For each root r and pop p>0, we:
///  - Build token-atoms B_1..B_k that partition the union of all masks at that pop
///    (with a configurable cap; fallback to a single block if exceeded).
///  - For each atom B_j, aggregate D_j(dest) = ⋃_{edges i: B_j∩L_i≠∅} D_i(dest).
///  - Create an intermediate node A_j with a single outgoing key (p, B_j) and dest-map D_j.
///  - Add a single edge root -> A_j with (0, B_j) and SIDs = all-states.
/// Pop=0 edges out of the root are left unchanged. This preserves semantics exactly and
/// reduces root out-degree from “sum of dest entries across p>0 edges” to “#atoms per pop”.
pub fn factor_root_fanout_via_atoms(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
    max_llm_token_id: usize,
    max_state_id: usize,
    max_atoms_per_pop: usize,
) {
    crate::debug!(
        2,
        "Factoring root fanout via token atoms (p>0 only, cap={} atoms/pop)...",
        max_atoms_per_pop
    );
    if roots.is_empty() {
        return;
    }

    let all_states_bv = StateIDBV::ones(max_state_id + 1);

    // Iterate over roots one-by-one; roots are few so cloning their children is cheap.
    for root_idx in roots.values() {
        // Snapshot original children to allow safe fallback.
        let original_children = if let Some(r) = root_idx.read(trie3_god) {
            r.children().clone()
        } else {
            continue;
        };
        if original_children.is_empty() {
            continue;
        }

        use std::collections::BTreeMap as BTM;
        // Separate pop=0 edges (kept as-is) and pop>0 edges (to be factored).
        let mut keep_children: BTM<(isize, LLMTokenBV), OrderedHashMap<PrecomputeNode3Index, StateIDBV>> =
            BTM::new();
        let mut by_pop: BTM<
            isize,
            Vec<(
                LLMTokenBV,
                OrderedHashMap<PrecomputeNode3Index, StateIDBV>,
            )>,
        > = BTM::new();
        let mut any_p_gt_0 = false;

        for ((pop, llm_bv), dm) in original_children {
            if pop <= 0 {
                keep_children.insert((pop, llm_bv), dm);
            } else {
                any_p_gt_0 = true;
                // Constrain bitsets for safety and drop empties.
                let mut l = llm_bv.clone();
                l.constrain(max_llm_token_id);
                if l.is_empty() {
                    continue;
                }
                let mut dm2: OrderedHashMap<PrecomputeNode3Index, StateIDBV> =
                    OrderedHashMap::new();
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
        }

        if !any_p_gt_0 {
            // Nothing to factor for this root.
            continue;
        }

        let mut new_children = keep_children;
        let mut made_progress = false;

        for (pop, entries) in by_pop {
            if entries.is_empty() {
                continue;
            }
            // Compute the union of all masks at this pop.
            let mut universe = LLMTokenBV::zeros();
            for (l, _) in &entries {
                universe |= l;
            }
            universe.constrain(max_llm_token_id);
            if universe.is_empty() {
                continue;
            }

            // Compute token-atoms with a cap; fallback to a single block if exceeded.
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
                if next_blocks.len() > max_atoms_per_pop && max_atoms_per_pop > 0 {
                    aborted = true;
                    break;
                }
                blocks = next_blocks;
                if blocks.is_empty() {
                    break;
                }
            }
            if aborted || blocks.is_empty() {
                blocks = vec![universe.clone()];
            }

            // For each atom, aggregate destination maps and build an intermediate node.
            for b in blocks {
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

                // Create an intermediate node with a single outgoing key (pop, b).
                let mid = PrecomputeNode3Index::new(
                    trie3_god.insert(Trie::new(PrecomputedNode3Contents::internal())),
                );
                {
                    let mut mw = mid.write(trie3_god).expect("write");
                    let mut dm_out: OrderedHashMap<PrecomputeNode3Index, StateIDBV> =
                        OrderedHashMap::new();
                    for (dst, sids) in dest_agg {
                        dm_out.insert(dst, sids);
                    }
                    mw.children_mut().insert((pop, b.clone()), dm_out);
                    mw.value.live_tokens = b.clone();
                }
                // Root -> mid uses (0, b) and SIDs = all states.
                new_children
                    .entry((0, b.clone()))
                    .or_insert_with(OrderedHashMap::new)
                    .insert(mid, all_states_bv.clone());
                made_progress = true;
            }
        }

        if made_progress {
            if let Some(mut rw) = root_idx.write(trie3_god) {
                // Recompute live tokens as the union of outgoing LLM masks.
                let mut live = LLMTokenBV::zeros();
                for ((_, l), _) in &new_children {
                    live |= l;
                }
                rw.value.live_tokens = live;
                *rw.children_mut() = new_children;
            }
        }
    }
}
