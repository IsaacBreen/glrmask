use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use ordered_hash_map::OrderedHashMap;

use crate::constraint::{
    LLMTokenBV, PrecomputeNode3Index, StateIDBV, Trie3GodWrapper,
};
use crate::datastructures::trie::{PathComparison, Trie};
use crate::datastructures::EntryApi;

/// Detect and assert absence of true cycles over pop=0 edges (token-and-state compatible).
pub fn has_true_cycle_trie3(arena: &Trie3GodWrapper, roots: &[PrecomputeNode3Index]) {
    let all_nodes = Trie::all_nodes(arena, roots);
    if all_nodes.is_empty() {
        return;
    }

    const UNVISITED: u8 = 0;
    const VISITING: u8 = 1;
    const VISITED: u8 = 2;

    let mut states: HashMap<PrecomputeNode3Index, u8> =
        all_nodes.iter().map(|&idx| (idx, UNVISITED)).collect();

    let mut path: Vec<PrecomputeNode3Index> = Vec::new();

    for node_idx in all_nodes {
        if states.get(&node_idx) == Some(&UNVISITED) {
            if let Some(report) =
                detect_true_cycle_recursive_trie3_new(node_idx, arena, &mut states, &mut path)
            {
                panic!("{}", report);
            }
        }
    }
}

fn detect_true_cycle_recursive_trie3_new(
    node_idx: PrecomputeNode3Index,
    arena: &Trie3GodWrapper,
    states: &mut HashMap<PrecomputeNode3Index, u8>,
    path: &mut Vec<PrecomputeNode3Index>,
) -> Option<String> {
    const UNVISITED: u8 = 0;
    const VISITING: u8 = 1;
    const VISITED: u8 = 2;

    states.insert(node_idx, VISITING);
    path.push(node_idx);

    let children_to_visit = if let Some(guard) = node_idx.read(arena) {
        guard
            .children()
            .iter()
            .filter(|((pop, _), _)| *pop == 0)
            .map(|(_ek, dm)| dm.clone())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    for dest_map in children_to_visit {
        for child_idx in dest_map.keys() {
            match states.get(child_idx).copied().unwrap_or(UNVISITED) {
                VISITING => {
                    if let Some(cycle_start_index) = path.iter().position(|&p| p == *child_idx) {
                        let cycle_nodes = &path[cycle_start_index..];

                        let mut llm_intersection = LLMTokenBV::max_ones();
                        let mut sids_intersection = StateIDBV::max_ones();

                        for i in 0..cycle_nodes.len() {
                            let u = cycle_nodes[i];
                            let v = if i + 1 < cycle_nodes.len() {
                                cycle_nodes[i + 1]
                            } else {
                                *child_idx
                            };

                            let u_guard = u.read(arena).unwrap();
                            let mut edge_llm_union = LLMTokenBV::zeros();
                            let mut edge_sids_union = StateIDBV::zeros();

                            for ((pop, llm_bv), dm) in u_guard.children() {
                                if *pop == 0 {
                                    if let Some(sids) = dm.get(&v) {
                                        edge_llm_union |= llm_bv;
                                        edge_sids_union |= sids;
                                    }
                                }
                            }

                            llm_intersection &= &edge_llm_union;
                            sids_intersection &= &edge_sids_union;
                        }

                        if !llm_intersection.is_empty() && !sids_intersection.is_empty() {
                            let first_violating_token = llm_intersection.iter().next().unwrap();
                            let first_violating_state = sids_intersection.iter().next().unwrap();

                            let mut report = format!(
                                "LLM-compatible cycle with pop=0 detected in precompute3 trie. Example violation: internal LLM token ID {} and state ID {}.\nCycle path:\n",
                                first_violating_token, first_violating_state
                            );
                            for i in 0..cycle_nodes.len() {
                                let u = cycle_nodes[i];
                                let v = if i + 1 < cycle_nodes.len() {
                                    cycle_nodes[i + 1]
                                } else {
                                    *child_idx
                                };
                                report.push_str(&format!("  {} --> {}\n", u, v));
                            }
                            return Some(report);
                        }
                    }
                }
                UNVISITED => {
                    if let Some(report) = detect_true_cycle_recursive_trie3_new(
                        *child_idx,
                        arena,
                        states,
                        path,
                    ) {
                        return Some(report);
                    }
                }
                _ => {}
            }
        }
    }

    path.pop();
    states.insert(node_idx, VISITED);
    None
}

pub type CycleReport3 =
    (Vec<(PrecomputeNode3Index, Option<(isize, LLMTokenBV)>)>, usize);

pub fn has_true_cycle_trie3_llm_only(
    arena: &Trie3GodWrapper,
    roots: &[PrecomputeNode3Index],
    internal_max_llm_token: usize,
) {
    // We check for each token individually. This is slow but thorough.
    for llm_token_id in 0..=internal_max_llm_token {
        let mut visited: HashSet<PrecomputeNode3Index> = HashSet::new();
        for &root in roots {
            if visited.contains(&root) {
                continue;
            }
            if let Some((cycle_path, _)) = detect_true_cycle_recursive_trie3_llm_only(
                root,
                None,
                llm_token_id,
                arena,
                &mut HashMap::new(),
                &mut visited,
                &mut Vec::new(),
            ) {
                let mut report = format!(
                    "LLM-compatible cycle with pop=0 detected in precompute3 trie for internal LLM token ID {}.\nCycle path:\n",
                    llm_token_id
                );
                for i in 0..cycle_path.len() {
                    let (node_idx, _) = cycle_path[i];
                    let next_i = (i + 1) % cycle_path.len();
                    let (next_node_idx, edge_to_next_opt) = &cycle_path[next_i];
                    let edge_str = edge_to_next_opt.as_ref().map_or_else(
                        || " (root edge)".to_string(),
                        |ek| format!("pop={}, llm_bv=[...]", ek.0),
                    );
                    report.push_str(&format!(
                        "  {} --[{}]--> {}\n",
                        node_idx, edge_str, next_node_idx
                    ));
                }
                panic!("{}", report);
            }
        }
    }
}

fn detect_true_cycle_recursive_trie3_llm_only(
    node_idx: PrecomputeNode3Index,
    edge_key_opt: Option<(isize, LLMTokenBV)>,
    llm_token_id: usize,
    arena: &Trie3GodWrapper,
    recursion_stack: &mut HashMap<PrecomputeNode3Index, usize>,
    visited: &mut HashSet<PrecomputeNode3Index>,
    path: &mut Vec<(PrecomputeNode3Index, Option<(isize, LLMTokenBV)>)>,
) -> Option<CycleReport3> {
    path.push((node_idx, edge_key_opt));

    const UNVISITED: i8 = -1;

    if let Some(&path_start_idx) = recursion_stack.get(&node_idx) {
        let cycle_path = path[path_start_idx..].to_vec();
        path.pop();
        return Some((cycle_path, llm_token_id));
    }

    if visited.contains(&node_idx) {
        path.pop();
        return None;
    }

    recursion_stack.insert(node_idx, path.len() - 1);

    let children_to_visit = if let Some(guard) = node_idx.read(arena) {
        guard.children().clone()
    } else {
        recursion_stack.remove(&node_idx);
        path.pop();
        return None;
    };

    for (edge_key, dest_map) in children_to_visit.iter() {
        let (pop, llm_bv) = edge_key;
        if *pop == 0 && llm_bv.contains(llm_token_id) {
            for child_idx in dest_map.keys() {
                if let Some(report) = detect_true_cycle_recursive_trie3_llm_only(
                    *child_idx,
                    Some(edge_key.clone()),
                    llm_token_id,
                    arena,
                    recursion_stack,
                    visited,
                    path,
                ) {
                    return Some(report);
                }
            }
        }
    }

    recursion_stack.remove(&node_idx);
    visited.insert(node_idx);
    path.pop();
    None
}

/// Extra assert: ensure pop=0 paths to end are short or redundant.
pub fn assert_pop0_paths_to_end_are_short(
    roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    crate::debug!(2, "Asserting that all pop=0 paths to an end node are of length <= 1, or are redundant.");

    for root_node in roots.values() {
        // 1. Find all short pop=0 paths from this root and collect their semantics.
        let mut short_path_semantics: Vec<(LLMTokenBV, StateIDBV)> = Vec::new();
        let r = root_node.read(trie3_god).expect("root must exist");
        for ((pop, llm_bv), dm) in r.children() {
            if *pop == 0 {
                for (dest, sids_bv) in dm {
                    if dest.read(trie3_god).expect("dest must exist").value.end {
                        short_path_semantics.push((llm_bv.clone(), sids_bv.clone()));
                    }
                }
            }
        }

        // 2. Find all long pop=0 paths and check if they are covered.
        // A long path starts with a pop=0 edge from the root to a non-end node.
        for ((pop, llm_bv1), dm) in r.children() {
            if *pop == 0 {
                for (dest, sids_bv1) in dm {
                    if dest.read(trie3_god).expect("dest must exist").value.end {
                        continue; // This is a short path, which is fine.
                    }

                    // This is the start of a potential long path from root -> dest.
                    // Find all pop=0 paths from `dest` to any end node.
                    let mut q: VecDeque<(
                        PrecomputeNode3Index,
                        LLMTokenBV,
                        StateIDBV,
                        Vec<PrecomputeNode3Index>,
                    )> = VecDeque::new();

                    q.push_back((
                        *dest,
                        llm_bv1.clone(),
                        sids_bv1.clone(),
                        vec![*root_node, *dest],
                    ));

                    while let Some((curr, path_llm, path_sids, path_nodes)) = q.pop_front() {
                        let curr_guard = curr.read(trie3_god).expect("node must exist");

                        if curr_guard.value.end {
                            // We found a long path. Check if it's covered.
                            let long_llm = path_llm;
                            let long_sids = path_sids;

                            if long_llm.is_empty() || long_sids.is_empty() {
                                continue; // Path is not actually traversable.
                            }

                            // Check coverage for every token in long_llm
                            for t in long_llm.iter() {
                                let mut sids_covered_for_t = StateIDBV::zeros();
                                for (short_llm, short_sids) in &short_path_semantics {
                                    if short_llm.contains(t) {
                                        sids_covered_for_t |= short_sids;
                                    }
                                }

                                if !long_sids.is_subset(&sids_covered_for_t) {
                                    let violating_sids = &long_sids - &sids_covered_for_t;
                                    let first_violating_sid =
                                        violating_sids.iter().next().unwrap();
                                    let path_str = path_nodes
                                        .iter()
                                        .map(|n| n.to_string())
                                        .collect::<Vec<_>>()
                                        .join(" -> ");
                                    panic!(
                                        "Found a non-redundant pop=0 path of length > 1 to an end node.\n\
                                        Path: {}\n\
                                        The pair (llm_token={}, state_id={}) is accepted by this long path, \
                                        but not by any short (length=1) pop=0 path from the same root.",
                                        path_str, t, first_violating_sid
                                    );
                                }
                            }
                            // This long path is covered. Don't traverse further from an end node.
                            continue;
                        }

                        // Continue traversal
                        for ((pop2, llm_bv2), dm2) in curr_guard.children() {
                            if *pop2 == 0 {
                                for (next, sids_bv2) in dm2 {
                                    let next_path_llm = &path_llm & llm_bv2;
                                    let next_path_sids = &path_sids & sids_bv2;

                                    if !next_path_llm.is_empty() && !next_path_sids.is_empty() {
                                        // Simple cycle check on the current path being explored.
                                        if path_nodes.contains(next) {
                                            continue;
                                        }
                                        let mut next_path_nodes = path_nodes.clone();
                                        next_path_nodes.push(*next);
                                        q.push_back((
                                            *next,
                                            next_path_llm,
                                            next_path_sids,
                                            next_path_nodes,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
