use crate::constraint::{IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3GodWrapper, IntermediateTrie3EdgeKey, IntermediatePrecomputedNodeContents3};
use crate::datastructures::trie::{Trie};
use crate::tokenizer::TokenizerStateID;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Display;

pub fn eliminate_pushes_and_pops(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    // This is a complex graph transformation. We will do it iteratively until a fixed point is reached.
    // The core idea is to find a sequence A --push--> B --op--> C and replace it.
    // We select B nodes that have no outgoing push edges to ensure that part of the graph is "stable".
    loop {
        let all_nodes = Trie::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>());
        let mut predecessors: HashMap<IntermediatePrecomputeNode3Index, Vec<(IntermediatePrecomputeNode3Index, IntermediateTrie3EdgeKey)>> = HashMap::new();
        let mut outgoing_push_counts: HashMap<IntermediatePrecomputeNode3Index, usize> = HashMap::new();

        for &node_idx in &all_nodes {
            outgoing_push_counts.entry(node_idx).or_insert(0);
            if let Some(guard) = node_idx.read(god) {
                for (edge_key, dest_map) in guard.children() {
                    if let IntermediateTrie3EdgeKey::Push(_) = edge_key {
                        *outgoing_push_counts.get_mut(&node_idx).unwrap() += dest_map.len();
                    }
                    for (child_idx, _) in dest_map {
                        predecessors.entry(*child_idx).or_default().push((node_idx, edge_key.clone()));
                    }
                }
            }
        }

        let mut nodes_to_process: Vec<IntermediatePrecomputeNode3Index> = Vec::new();
        for node_idx in &all_nodes {
            if outgoing_push_counts.get(node_idx).cloned().unwrap_or(0) == 0 {
                if let Some(guard) = node_idx.read(god) {
                    if !guard.value.end {
                        nodes_to_process.push(*node_idx);
                    }
                }
            }
        }

        let mut changed = false;

        for b_idx in nodes_to_process {
            let incoming_pushes = if let Some(preds) = predecessors.get(&b_idx) {
                preds.iter().filter_map(|(pred_idx, edge_key)| {
                    if let IntermediateTrie3EdgeKey::Push(s) = edge_key {
                        Some((*pred_idx, s.clone()))
                    } else {
                        None
                    }
                }).collect::<Vec<_>>()
            } else {
                continue;
            };

            if incoming_pushes.is_empty() {
                continue;
            }

            let outgoing_edges = {
                let b_guard = b_idx.read(god).unwrap();
                b_guard.children().clone().into_iter().flat_map(|(ek, dm)| {
                    dm.keys().map(move |c_idx| (ek.clone(), *c_idx)).collect::<Vec<_>>()
                }).collect::<Vec<_>>()
            };

            if !outgoing_edges.is_empty() {
                changed = true;
            }

            for (a_idx, s) in incoming_pushes {
                // Remove the edge A --push(S)--> B
                if let Some(mut a_guard) = a_idx.write(god) {
                    if let Some(dest_map) = a_guard.children_mut().get_mut(&IntermediateTrie3EdgeKey::Push(s.clone())) {
                        dest_map.remove(&b_idx);
                        if dest_map.is_empty() {
                            a_guard.children_mut().remove(&IntermediateTrie3EdgeKey::Push(s.clone()));
                        }
                    }
                }

                for (op, c_idx) in &outgoing_edges {
                    match op {
                        IntermediateTrie3EdgeKey::Push(_) => unreachable!("Node B has an outgoing push edge but was selected for processing."),
                        IntermediateTrie3EdgeKey::Pop(0, s_prime) => {
                            let intersection = &s & s_prime;
                            if !intersection.is_empty() {
                                let new_edge_key = IntermediateTrie3EdgeKey::Push(intersection);
                                a_idx.write(god).unwrap().force_insert_to_node(new_edge_key, (), *c_idx);
                            }
                        }
                        IntermediateTrie3EdgeKey::Pop(1, s_prime) => {
                            if !s.is_disjoint(s_prime) {
                                a_idx.write(god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::NoOp, (), *c_idx);
                            }
                        }
                        IntermediateTrie3EdgeKey::Pop(n, s_prime) if *n > 1 => {
                            let new_edge_key = IntermediateTrie3EdgeKey::Pop(*n - 1, s_prime.clone());
                            a_idx.write(god).unwrap().force_insert_to_node(new_edge_key, (), *c_idx);
                        }
                        IntermediateTrie3EdgeKey::CheckLLM(l) => {
                            let b_prime_idx = IntermediatePrecomputeNode3Index::new(god.insert(IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())));
                            a_idx.write(god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::CheckLLM(l.clone()), (), b_prime_idx);
                            b_prime_idx.write(god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(s.clone()), (), *c_idx);
                        }
                        IntermediateTrie3EdgeKey::NoOp => {
                            a_idx.write(god).unwrap().force_insert_to_node(IntermediateTrie3EdgeKey::Push(s.clone()), (), *c_idx);
                        }
                        _ => {}
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    Trie::gc(god, &roots.values().cloned().collect::<Vec<_>>());
}

pub fn assert_no_pops_reachable_from_pushes(
    roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    let all_nodes = Trie::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>());
    
    let mut pop_reachable_memo: HashMap<IntermediatePrecomputeNode3Index, bool> = HashMap::new();

    for &node_idx in &all_nodes {
        is_pop_reachable_from(node_idx, god, &mut pop_reachable_memo, &mut HashSet::new());
    }

    for &node_idx in &all_nodes {
        if let Some(guard) = node_idx.read(god) {
            for (edge_key, dest_map) in guard.children() {
                if let IntermediateTrie3EdgeKey::Push(_) = edge_key {
                    for child_idx in dest_map.keys() {
                        if *pop_reachable_memo.get(child_idx).unwrap_or(&false) {
                            let path = find_path_to_pop( *child_idx, god, &pop_reachable_memo);
                            panic!("Assertion failed: Pop is reachable from a Push edge. Path: Node {} --Push--> Node {} --> ... --> Pop. Path to pop: {:?}", node_idx, child_idx, path);
                        }
                    }
                }
            }
        }
    }
}

fn is_pop_reachable_from(
    node: IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    memo: &mut HashMap<IntermediatePrecomputeNode3Index, bool>,
    visiting: &mut HashSet<IntermediatePrecomputeNode3Index>
) -> bool {
    if let Some(&result) = memo.get(&node) {
        return result;
    }
    if !visiting.insert(node) {
        return false;
    }

    if let Some(guard) = node.read(god) {
        for (edge_key, dest_map) in guard.children() {
            if let IntermediateTrie3EdgeKey::Pop(_, _) = edge_key {
                visiting.remove(&node);
                memo.insert(node, true);
                return true;
            }
            for child_idx in dest_map.keys() {
                if is_pop_reachable_from(*child_idx, god, memo, visiting) {
                    visiting.remove(&node);
                    memo.insert(node, true);
                    return true;
                }
            }
        }
    }

    visiting.remove(&node);
    memo.insert(node, false);
    false
}

fn find_path_to_pop(
    start_node: IntermediatePrecomputeNode3Index,
    god: &IntermediateTrie3GodWrapper,
    pop_reachable_memo: &HashMap<IntermediatePrecomputeNode3Index, bool>
) -> Vec<(IntermediatePrecomputeNode3Index, String)> {
    let mut path = vec![];
    let mut current_node = start_node;
    let mut visited = HashSet::new();

    while visited.insert(current_node) {
        if let Some(guard) = current_node.read(god) {
            let mut found_next = false;
            for (edge_key, dest_map) in guard.children() {
                if let IntermediateTrie3EdgeKey::Pop(_, _) = edge_key {
                    path.push((current_node, format!("{}", edge_key)));
                    return path;
                }
                for child_idx in dest_map.keys() {
                    if *pop_reachable_memo.get(child_idx).unwrap_or(&false) {
                        path.push((current_node, format!("{}", edge_key)));
                        current_node = *child_idx;
                        found_next = true;
                        break;
                    }
                }
                if found_next {
                    break;
                }
            }
            if !found_next {
                break;
            }
        } else {
            break;
        }
    }
    path
}
