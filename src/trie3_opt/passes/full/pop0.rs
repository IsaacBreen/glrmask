use std::collections::{BTreeMap, HashMap, HashSet};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};

#[inline]
pub fn pop_is_zero(ek: &(isize, LLMTokenBV)) -> bool {
    ek.0 == 0
}

/// Assert that no pop=0 edges exist from non-root nodes.
/// When enabled in config, this runs after eliminating pop0 edges.
pub fn assert_no_pop0_nonroot_edges_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    let roots_set: HashSet<PrecomputeNode3Index> = roots.values().cloned().collect();
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    for n in all_nodes {
        if roots_set.contains(&n) {
            continue;
        }
        let r = n.read(trie3_god).expect("read");
        for ((pop, _llm_bv), _dm) in r.children() {
            if *pop == 0 {
                panic!(
                    "Invariant violated: found pop=0 edge originating from non-root node {}",
                    n
                );
            }
        }
    }
}

/// Remove all pop=0 edges whose source is not a root.
/// For each 0-pop edge B --(0, L_bc, S_bc)--> C and each incoming edge
/// A --(p_ab, L_ab, S_ab)--> B, add/update A --(p_ab, L_ab∧L_bc, S_ab∧S_bc)--> C.
/// Then remove the original B->C (0, L_bc) mapping. Iterate until no such edges remain.
pub fn eliminate_pop0_edges_except_roots_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(
        2,
        "Eliminating pop=0 edges from non-root nodes in Trie3..."
    );

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    if roots_vec.is_empty() {
        return;
    }
    let roots_set: HashSet<PrecomputeNode3Index> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    let mut total_removed = 0usize;
    let mut iter = 0usize;

    loop {
        iter += 1;

        // Snapshot incoming edges and current pop=0 edges.
        let mut incoming: HashMap<
            PrecomputeNode3Index,
            Vec<(PrecomputeNode3Index, (isize, LLMTokenBV), StateIDBV)>,
        > = HashMap::new();
        let mut pop0_edges: Vec<(
            PrecomputeNode3Index,
            LLMTokenBV,
            PrecomputeNode3Index,
            StateIDBV,
        )> = Vec::new();

        for src in &all_nodes {
            if let Some(r) = src.read(trie3_god) {
                for (ek, dm) in r.children() {
                    for (dst, sids) in dm {
                        incoming
                            .entry(*dst)
                            .or_default()
                            .push((*src, (ek.0, ek.1.clone()), sids.clone()));
                        if pop_is_zero(ek) {
                            pop0_edges.push((*src, ek.1.clone(), *dst, sids.clone()));
                        }
                    }
                }
            }
        }

        // Early exit if there isn't any 0-pop edge at all.
        if pop0_edges.is_empty() {
            break;
        }

        let mut removed_this_iter = 0usize;

        for (b, llm_bc, c, s_bc) in pop0_edges {
            if roots_set.contains(&b) {
                // pop=0 edge from a root is allowed by design, keep it
                continue;
            }
            if let Some(preds) = incoming.get(&b) {
                for (a, (p_ab, llm_ab), s_ab) in preds {
                    // Compose constraints at the same position (pop=0 does not advance position)
                    let new_llm = llm_ab & &llm_bc;
                    if new_llm.is_empty() {
                        continue;
                    }
                    let new_sids = s_ab & &s_bc;
                    if new_sids.is_empty() {
                        continue;
                    }
                    // Create/update A -> C with (p_ab, new_llm) and SIDs = intersection
                    if let Some(mut aw) = a.write(trie3_god) {
                        let key_llm = new_llm.clone();
                        let entry =
                            aw.children_mut().entry((*p_ab, key_llm.clone())).or_default();
                        entry
                            .entry(c)
                            .and_modify(|e| *e |= &new_sids)
                            .or_insert(new_sids.clone());
                        // keep live_tokens conservative; we'll recompute exactly later anyway
                        aw.value.live_tokens |= &key_llm;
                    }
                }
            }

            // Remove specific B -> C mapping under (0, llm_bc).
            if let Some(mut bw) = b.write(trie3_god) {
                let key = (0isize, llm_bc.clone());
                let mut removed_one = false;
                if let Some(dm) = bw.children_mut().get_mut(&key) {
                    if dm.remove(&c).is_some() {
                        removed_one = true;
                    }
                    if dm.is_empty() {
                        bw.children_mut().remove(&key);
                    }
                }
                if removed_one {
                    removed_this_iter += 1;
                    // Recompute live tokens conservatively for B; exact recomputation follows post-loop
                    let mut new_live = LLMTokenBV::zeros();
                    for ((_, llm_bv), _) in bw.children() {
                        new_live |= llm_bv;
                    }
                    bw.value.live_tokens = new_live;
                }
            }
        }

        total_removed += removed_this_iter;
        crate::debug!(
            3,
            "Pop0-elim iter {}: removed {} pop=0 edges.",
            iter,
            removed_this_iter
        );
        if removed_this_iter == 0 {
            break;
        }
    }

    // Final recomputation of live_tokens for all nodes.
    for n in &all_nodes {
        if let Some(mut w) = n.write(trie3_god) {
            let mut new_live = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in w.children() {
                new_live |= llm_bv;
            }
            w.value.live_tokens = new_live;
        }
    }

    let roots_vec2: Vec<_> = roots.values().cloned().collect();
    Trie::recompute_all_max_depths(trie3_god, &roots_vec2);
    crate::debug!(
        2,
        "Eliminated {} pop=0 edges from non-root nodes.",
        total_removed
    );
}
