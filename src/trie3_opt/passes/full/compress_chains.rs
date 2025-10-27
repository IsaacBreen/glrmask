use std::collections::{BTreeMap, HashMap, HashSet};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper};
use crate::datastructures::EntryApi;
use crate::datastructures::trie::{Trie, Trie2Index};

/// Contract unary chains where a node U has exactly one incoming edge and one outgoing edge,
/// both under the same (pop, LLMTokenBV) key, U is not an end node or a root, and the outgoing
/// edge maps to exactly one destination. The rewrite composes SIDs as intersection and rewires
/// the predecessor P to the child V, deleting the P->U entry. This preserves semantics and is
/// cycle-safe (may reduce multi-node cycles to a self-loop with intersected SIDs).
pub fn compress_unary_chains_trie3(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "Contracting unary chains in Trie3...");

    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }

    let roots_set: HashSet<PrecomputeNode3Index> = roots.values().cloned().collect();

    // Iterate a few times until no more rewrites are possible.
    let mut total_rewired = 0usize;
    let max_outer_iters = 8usize;
    for _iter in 0..max_outer_iters {
        // Build indegree counts and track unique predecessor + edge-key for nodes with indegree == 1.
        let mut indegree: HashMap<PrecomputeNode3Index, usize> = HashMap::new();
        let mut unique_parent: HashMap<
            PrecomputeNode3Index,
            (PrecomputeNode3Index, (isize, LLMTokenBV)),
        > = HashMap::new();

        for src in &all_nodes {
            if let Some(g) = src.read(trie3_god) {
                for (ek, dm) in g.children() {
                    for (dst, _sid) in dm {
                        let cnt = indegree.entry(*dst).or_insert(0);
                        *cnt += 1;
                        if *cnt == 1 {
                            unique_parent.insert(*dst, (*src, ek.clone()));
                        } else {
                            // More than one distinct incoming, not a unique-predecessor case.
                            unique_parent.remove(dst);
                        }
                    }
                }
            }
        }

        // For outdegree, record nodes that have exactly one outgoing destination overall
        // (across all keys combined) and are not end nodes; also capture that single edge.
        let mut out_info: HashMap<
            PrecomputeNode3Index,
            ((isize, LLMTokenBV), PrecomputeNode3Index, StateIDBV),
        > = HashMap::new();

        for u in &all_nodes {
            if let Some(g) = u.read(trie3_god) {
                if g.value.end {
                    continue; // Don't contract through accept nodes.
                }
                let mut total_dests = 0usize;
                let mut only_key: Option<(isize, LLMTokenBV)> = None;
                let mut only_dst: Option<PrecomputeNode3Index> = None;
                let mut only_sid: Option<StateIDBV> = None;

                for (ek, dm) in g.children() {
                    for (dst, sid) in dm {
                        total_dests += 1;
                        if total_dests == 1 {
                            only_key = Some(ek.clone());
                            only_dst = Some(*dst);
                            only_sid = Some(sid.clone());
                        }
                    }
                }
                if total_dests == 1 {
                    out_info.insert(*u, (only_key.unwrap(), only_dst.unwrap(), only_sid.unwrap()));
                }
            }
        }

        // Find candidates: nodes with indegree==1, outdegree==1, not roots, and matching keys on both sides.
        let mut rewired_this_iter = 0usize;
        for u in &all_nodes {
            let u_idx = *u;
            if roots_set.contains(&u_idx) {
                continue; // Never rewire through a root.
            }
            let Some((out_key, v_idx, s_uv)) = out_info.get(&u_idx).cloned() else { continue };
            let Some((p_idx, in_key)) = unique_parent.get(&u_idx).cloned() else { continue };
            if p_idx == u_idx {
                continue; // Avoid attempting to rewire a pure self-loop through itself.
            }
            // Only contract if the incoming and outgoing (pop, LLMTokenBV) keys match exactly.
            if in_key != out_key {
                continue;
            }

            // Perform the rewrite: in parent P, remove the U mapping under in_key,
            // and add/union V mapping with SIDs intersection S(P->U) ∧ S(U->V).
            if let Some(mut pw) = p_idx.write(trie3_god) {
                let (pop, llm) = in_key.clone();
                if let Some(dm) = pw.children_mut().get_mut(&(pop, llm.clone())) {
                    if let Some(s_pu_actual) = dm.remove(&u_idx) {
                        let s_comp = &s_pu_actual & &s_uv;
                        if !s_comp.is_empty() {
                            dm.entry(v_idx)
                                .and_modify(|e| *e |= &s_comp)
                                .or_insert(s_comp);
                        }
                        if dm.is_empty() {
                            pw.children_mut().remove(&(pop, llm.clone()));
                        }
                        // Recompute live tokens as union of outgoing LLM masks for robustness.
                        let mut new_live = LLMTokenBV::zeros();
                        for ((_, llm_bv), _) in pw.children() {
                            new_live |= llm_bv;
                        }
                        pw.value.live_tokens = new_live;
                        rewired_this_iter += 1;
                    }
                }
            }
        }

        total_rewired += rewired_this_iter;
        if rewired_this_iter == 0 {
            break;
        }
    }

    crate::debug!(2, "Unary-chain contraction rewired {} edges.", total_rewired);
}
