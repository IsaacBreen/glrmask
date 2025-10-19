use crate::constraint::{IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediateTrie3GodWrapper, IntermediateTrie3EdgeKey, IntermediatePrecomputedNodeContents3, StateIDBV, LLMTokenBV};
use crate::datastructures::trie::{Trie, GodWrapper, Trie2Index, MergeableEdgeValue};
use crate::tokenizer::TokenizerStateID;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Intermediate2Trie3EdgeKey {
    Pop(usize, StateIDBV),
    Push(StateIDBV),
    NoOp,
}

impl Display for Intermediate2Trie3EdgeKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Helper to format HybridBitset into ranges
        fn format_bv(bv: &StateIDBV) -> String {
            if bv.is_empty() {
                return "[]".to_string();
            }
            if bv.is_all() {
                return "[ALL]".to_string();
            }

            const MAX_RANGES_TO_SHOW: usize = 10;
            let total_ranges = bv.inner().ranges_len();

            let mut parts: Vec<String> = bv.iter_ranges().take(MAX_RANGES_TO_SHOW).map(|(start, end)| {
                if start == end {
                    format!("{}", start)
                } else if end == usize::MAX {
                    format!("{}..", start)
                } else {
                    format!("{}..={}", start, end)
                }
            }).collect();

            if total_ranges > MAX_RANGES_TO_SHOW {
                parts.push(format!("... ({} more ranges)", total_ranges - MAX_RANGES_TO_SHOW));
            }

            if total_ranges > 1 {
                format!("[{}]", parts.join(", "))
            } else {
                parts.join(", ")
            }
        }

        match self {
            Intermediate2Trie3EdgeKey::Pop(n, bv) => write!(f, "Pop({}, {})", n, format_bv(bv)),
            Intermediate2Trie3EdgeKey::Push(bv) => write!(f, "Push({})", format_bv(bv)),
            Intermediate2Trie3EdgeKey::NoOp => write!(f, "NoOp"),
        }
    }
}

impl MergeableEdgeValue for LLMTokenBV {
    fn merge(&mut self, other: Self) {
        *self |= &other;
    }
}

pub type Intermediate2PrecomputeNode3 = Trie<Intermediate2Trie3EdgeKey, LLMTokenBV, IntermediatePrecomputedNodeContents3>;
pub type Intermediate2PrecomputeNode3Index = Trie2Index;
pub type Intermediate2Trie3GodWrapper = GodWrapper<Intermediate2Trie3EdgeKey, LLMTokenBV, IntermediatePrecomputedNodeContents3>;

fn convert_to_intermediate2(
    roots1: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god1: &IntermediateTrie3GodWrapper,
) -> (BTreeMap<TokenizerStateID, Intermediate2PrecomputeNode3Index>, Intermediate2Trie3GodWrapper) {
    let god2 = Intermediate2Trie3GodWrapper::new();
    let mut roots2 = BTreeMap::new();
    let mut node_map: HashMap<IntermediatePrecomputeNode3Index, Intermediate2PrecomputeNode3Index> = HashMap::new();
    let mut q: std::collections::VecDeque<IntermediatePrecomputeNode3Index> = std::collections::VecDeque::new();

    for (sid, root1) in roots1 {
        let root2 = Intermediate2PrecomputeNode3Index::new(god2.insert(
            Intermediate2PrecomputeNode3::new(root1.read(god1).unwrap().value.clone())
        ));
        roots2.insert(*sid, root2);
        node_map.insert(*root1, root2);
        q.push_back(*root1);
    }

    let mut visited = HashSet::new();
    while let Some(idx1) = q.pop_front() {
        if !visited.insert(idx1) { continue; }

        let idx2 = *node_map.get(&idx1).unwrap();
        let guard1 = idx1.read(god1).unwrap();

        for (edge_key1, dest_map1) in guard1.children() {
            for (child1_idx, _) in dest_map1 {
                let child2_idx = *node_map.entry(*child1_idx).or_insert_with(|| {
                    let new_node = Intermediate2PrecomputeNode3Index::new(god2.insert(
                        Intermediate2PrecomputeNode3::new(child1_idx.read(god1).unwrap().value.clone())
                    ));
                    q.push_back(*child1_idx);
                    new_node
                });

                let (edge_key2, edge_value2) = match edge_key1 {
                    IntermediateTrie3EdgeKey::Pop(n, s) => (Intermediate2Trie3EdgeKey::Pop(*n, s.clone()), LLMTokenBV::max_ones()),
                    IntermediateTrie3EdgeKey::Push(s) => (Intermediate2Trie3EdgeKey::Push(s.clone()), LLMTokenBV::max_ones()),
                    IntermediateTrie3EdgeKey::NoOp => (Intermediate2Trie3EdgeKey::NoOp, LLMTokenBV::max_ones()),
                    IntermediateTrie3EdgeKey::CheckLLM(bv) => (Intermediate2Trie3EdgeKey::NoOp, bv.clone()),
                };

                god2.insert_edge_simple(idx2, child2_idx, edge_key2, edge_value2);
            }
        }
    }

    (roots2, god2)
}

fn convert_from_intermediate2(
    roots2: &BTreeMap<TokenizerStateID, Intermediate2PrecomputeNode3Index>,
    god2: &Intermediate2Trie3GodWrapper,
) -> (BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>, IntermediateTrie3GodWrapper) {
    let god1 = IntermediateTrie3GodWrapper::new();
    let mut roots1 = BTreeMap::new();
    let mut node_map: HashMap<Intermediate2PrecomputeNode3Index, IntermediatePrecomputeNode3Index> = HashMap::new();
    let mut q: std::collections::VecDeque<Intermediate2PrecomputeNode3Index> = std::collections::VecDeque::new();

    for (sid, root2) in roots2 {
        let root1 = IntermediatePrecomputeNode3Index::new(god1.insert(
            IntermediatePrecomputeNode3::new(root2.read(god2).unwrap().value.clone())
        ));
        roots1.insert(*sid, root1);
        node_map.insert(*root2, root1);
        q.push_back(*root2);
    }

    let mut visited = HashSet::new();
    while let Some(idx2) = q.pop_front() {
        if !visited.insert(idx2) { continue; }

        let idx1 = *node_map.get(&idx2).unwrap();
        let guard2 = idx2.read(god2).unwrap();

        for (edge_key2, dest_map2) in guard2.children() {
            for (child2_idx, edge_value2) in dest_map2 {
                let child1_idx = *node_map.entry(*child2_idx).or_insert_with(|| {
                    let new_node = IntermediatePrecomputeNode3Index::new(god1.insert(
                        IntermediatePrecomputeNode3::new(child2_idx.read(god2).unwrap().value.clone())
                    ));
                    q.push_back(*child2_idx);
                    new_node
                });

                if edge_value2.is_all() {
                    let edge_key1 = match edge_key2 {
                        Intermediate2Trie3EdgeKey::Pop(n, s) => IntermediateTrie3EdgeKey::Pop(*n, s.clone()),
                        Intermediate2Trie3EdgeKey::Push(s) => IntermediateTrie3EdgeKey::Push(s.clone()),
                        Intermediate2Trie3EdgeKey::NoOp => IntermediateTrie3EdgeKey::NoOp,
                    };
                    god1.insert_edge_simple(idx1, child1_idx, edge_key1, ());
                } else {
                    match edge_key2 {
                        Intermediate2Trie3EdgeKey::NoOp => {
                            god1.insert_edge_simple(idx1, child1_idx, IntermediateTrie3EdgeKey::CheckLLM(edge_value2.clone()), ());
                        }
                        _ => {
                            // This case requires inserting an intermediate node.
                            let intermediate_node = IntermediatePrecomputeNode3Index::new(god1.insert(
                                IntermediatePrecomputeNode3::new(IntermediatePrecomputedNodeContents3::internal())
                            ));
                            let edge_key1_op = match edge_key2 {
                                Intermediate2Trie3EdgeKey::Pop(n, s) => IntermediateTrie3EdgeKey::Pop(*n, s.clone()),
                                Intermediate2Trie3EdgeKey::Push(s) => IntermediateTrie3EdgeKey::Push(s.clone()),
                                _ => unreachable!(),
                            };
                            god1.insert_edge_simple(idx1, intermediate_node, edge_key1_op, ());
                            god1.insert_edge_simple(intermediate_node, child1_idx, IntermediateTrie3EdgeKey::CheckLLM(edge_value2.clone()), ());
                        }
                    }
                }
            }
        }
    }

    (roots1, god1)
}

pub fn eliminate_pushes_and_pops(
    roots: &mut BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god: &IntermediateTrie3GodWrapper,
) {
    let (mut roots2, god2) = convert_to_intermediate2(roots, god);

    let mut iteration = 0;
    let mut prev_num_nodes = usize::MAX;
    loop {
        iteration += 1;
        Trie::gc(&god2, &roots2.values().cloned().collect::<Vec<_>>());
        let all_nodes = Trie::all_nodes(&god2, &roots2.values().cloned().collect::<Vec<_>>());
        let current_num_nodes = all_nodes.len();

        if iteration > 1 {
            assert!(
                current_num_nodes <= prev_num_nodes,
                "Node count increased in iteration {}. Previous: {}, Current: {}",
                iteration, prev_num_nodes, current_num_nodes
            );
        }
        prev_num_nodes = current_num_nodes;

        let mut predecessors: HashMap<Intermediate2PrecomputeNode3Index, Vec<(Intermediate2PrecomputeNode3Index, Intermediate2Trie3EdgeKey, LLMTokenBV)>> = HashMap::new();
        let mut outgoing_push_counts: HashMap<Intermediate2PrecomputeNode3Index, usize> = HashMap::new();

        for &node_idx in &all_nodes {
            outgoing_push_counts.entry(node_idx).or_insert(0);
            if let Some(guard) = node_idx.read(&god2) {
                for (edge_key, dest_map) in guard.children() {
                    if let Intermediate2Trie3EdgeKey::Push(_) = edge_key {
                        *outgoing_push_counts.get_mut(&node_idx).unwrap() += dest_map.len();
                    }
                    for (child_idx, edge_val) in dest_map {
                        predecessors.entry(*child_idx).or_default().push((node_idx, edge_key.clone(), edge_val.clone()));
                    }
                }
            }
        }

        let mut nodes_to_process: Vec<Intermediate2PrecomputeNode3Index> = Vec::new();
        for node_idx in &all_nodes {
            if outgoing_push_counts.get(node_idx).cloned().unwrap_or(0) == 0 {
                if let Some(guard) = node_idx.read(&god2) {
                    if !guard.value.end {
                        nodes_to_process.push(*node_idx);
                    }
                }
            }
        }

        let mut changed = false;
        let num_nodes_to_process = nodes_to_process.len();

        for b_idx in nodes_to_process {
            let incoming_pushes = if let Some(preds) = predecessors.get(&b_idx) {
                preds.iter().filter_map(|(pred_idx, edge_key, edge_val)| {
                    if let Intermediate2Trie3EdgeKey::Push(s) = edge_key {
                        Some((*pred_idx, s.clone(), edge_val.clone()))
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
                let b_guard = b_idx.read(&god2).unwrap();
                b_guard.children().clone().into_iter().flat_map(|(ek, dm)| {
                    dm.iter().map(move |(c_idx, ev)| (ek.clone(), *c_idx, ev.clone())).collect::<Vec<_>>()
                }).collect::<Vec<_>>()
            };

            if !outgoing_edges.is_empty() {
                changed = true;
            }

            for (a_idx, s, bv_ab) in incoming_pushes {
                // Remove the edge A --push(S)--> B
                if let Some(mut a_guard) = a_idx.write(&god2) {
                    if let Some(dest_map) = a_guard.children_mut().get_mut(&Intermediate2Trie3EdgeKey::Push(s.clone())) {
                        dest_map.remove(&b_idx);
                        if dest_map.is_empty() {
                            a_guard.children_mut().remove(&Intermediate2Trie3EdgeKey::Push(s.clone()));
                        }
                    }
                }

                for (op, c_idx, bv_bc) in &outgoing_edges {
                    let new_bv = &bv_ab & bv_bc;
                    if new_bv.is_empty() { continue; }

                    match op {
                        Intermediate2Trie3EdgeKey::Push(_) => unreachable!("Node B has an outgoing push edge but was selected for processing."),
                        Intermediate2Trie3EdgeKey::Pop(0, s_prime) => {
                            let intersection = &s & s_prime;
                            if !intersection.is_empty() {
                                let new_edge_key = Intermediate2Trie3EdgeKey::Push(intersection);
                                god2.insert_edge_simple(a_idx, *c_idx, new_edge_key, new_bv);
                            }
                        }
                        Intermediate2Trie3EdgeKey::Pop(1, s_prime) => {
                            if !s.is_disjoint(s_prime) {
                                god2.insert_edge_simple(a_idx, *c_idx, Intermediate2Trie3EdgeKey::NoOp, new_bv);
                            }
                        }
                        Intermediate2Trie3EdgeKey::Pop(n @ 2.., s_prime) => {
                            let new_edge_key = Intermediate2Trie3EdgeKey::Pop(*n - 1, s_prime.clone());
                            god2.insert_edge_simple(a_idx, *c_idx, new_edge_key, new_bv);
                        }
                        Intermediate2Trie3EdgeKey::NoOp => {
                            god2.insert_edge_simple(a_idx, *c_idx, Intermediate2Trie3EdgeKey::Push(s.clone()), new_bv);
                        },
                    }
                }
            }
        }

        if changed {
            crate::debug!(2, "Eliminating pushes/pops: iteration {}, processed {} nodes.", iteration, num_nodes_to_process);
        }

        if !changed {
            break;
        }
    }

    let (final_roots1_map, final_god1) = convert_from_intermediate2(&roots2, &god2);

    let sids: Vec<_> = final_roots1_map.keys().cloned().collect();
    let old_roots: Vec<_> = final_roots1_map.values().cloned().collect();

    god.clear();
    let (new_roots_vec, _map) = Trie::deep_copy_subtrees_into(&final_god1, god, &old_roots);

    roots.clear();
    for (sid, new_root) in sids.into_iter().zip(new_roots_vec.into_iter()) {
        roots.insert(sid, new_root);
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
                            let mut options = crate::datastructures::trie::PrettyPrintOptions::default()
                                .display_edge_keys_only()
                                .omit_depth()
                                ;
                            eprintln!("Full graph:");
                            eprintln!("{}", Trie::pretty_print_with_options(god, roots.values().cloned().collect::<Vec<_>>().as_slice(), &options));
                            eprintln!("Segment:");
                            eprintln!("{}", Trie::pretty_print_with_options(god, &[node_idx], &options));
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
