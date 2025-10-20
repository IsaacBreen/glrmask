use crate::constraint::{
    IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediatePrecomputedNodeContents3,
    IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper, LLMTokenBV, StateIDBV,
};
use crate::datastructures::trie::{GodWrapper, MergeableEdgeValue, Trie, Trie2Index};
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
        fn format_bv(bv: &StateIDBV) -> String {
            if bv.is_empty() {
                return "[]".to_string();
            }
            if bv.is_all() {
                return "[ALL]".to_string();
            }

            const MAX_RANGES_TO_SHOW: usize = 10;
            let total_ranges = bv.inner().ranges_len();

            let mut parts: Vec<String> = bv
                .iter_ranges()
                .take(MAX_RANGES_TO_SHOW)
                .map(|(start, end)| {
                    if start == end {
                        format!("{}", start)
                    } else if end == usize::MAX {
                        format!("{}..", start)
                    } else {
                        format!("{}..={}", start, end)
                    }
                })
                .collect();

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

pub type Intermediate2PrecomputeNode3 =
    Trie<Intermediate2Trie3EdgeKey, LLMTokenBV, IntermediatePrecomputedNodeContents3>;
pub type Intermediate2PrecomputeNode3Index = Trie2Index;
pub type Intermediate2Trie3GodWrapper =
    GodWrapper<Intermediate2Trie3EdgeKey, LLMTokenBV, IntermediatePrecomputedNodeContents3>;

// This conversion function remains the same. It simplifies the graph by
// moving token constraints into edge values.
fn convert_to_intermediate2(
    roots1: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    god1: &IntermediateTrie3GodWrapper,
) -> (
    BTreeMap<TokenizerStateID, Intermediate2PrecomputeNode3Index>,
    Intermediate2Trie3GodWrapper,
) {
    let god2 = Intermediate2Trie3GodWrapper::new();
    let mut roots2 = BTreeMap::new();
    let mut node_map: HashMap<IntermediatePrecomputeNode3Index, Intermediate2PrecomputeNode3Index> =
        HashMap::new();
    let mut q: std::collections::VecDeque<IntermediatePrecomputeNode3Index> =
        std::collections::VecDeque::new();

    for (sid, root1) in roots1 {
        let root2 = Intermediate2PrecomputeNode3Index::new(god2.insert(Intermediate2PrecomputeNode3::new(
            root1.read(god1).unwrap().value.clone(),
        )));
        roots2.insert(*sid, root2);
        node_map.insert(*root1, root2);
        q.push_back(*root1);
    }

    let mut visited = HashSet::new();
    while let Some(idx1) = q.pop_front() {
        if !visited.insert(idx1) {
            continue;
        }
        let idx2 = *node_map.get(&idx1).unwrap();
        let guard1 = idx1.read(god1).unwrap();

        for (edge_key1, dest_map1) in guard1.children() {
            for (child1_idx, _) in dest_map1 {
                let child2_idx = *node_map.entry(*child1_idx).or_insert_with(|| {
                    let new_node = Intermediate2PrecomputeNode3Index::new(god2.insert(
                        Intermediate2PrecomputeNode3::new(child1_idx.read(god1).unwrap().value.clone()),
                    ));
                    q.push_back(*child1_idx);
                    new_node
                });

                let (edge_key2, edge_value2) = match edge_key1 {
                    IntermediateTrie3EdgeKey::Pop(n, s) => {
                        (Intermediate2Trie3EdgeKey::Pop(*n, s.clone()), LLMTokenBV::max_ones())
                    }
                    IntermediateTrie3EdgeKey::Push(s) => {
                        (Intermediate2Trie3EdgeKey::Push(s.clone()), LLMTokenBV::max_ones())
                    }
                    IntermediateTrie3EdgeKey::NoOp => {
                        (Intermediate2Trie3EdgeKey::NoOp, LLMTokenBV::max_ones())
                    }
                    IntermediateTrie3EdgeKey::CheckLLM(bv) => {
                        (Intermediate2Trie3EdgeKey::NoOp, bv.clone())
                    }
                };

                god2.insert_edge_simple(idx2, child2_idx, edge_key2, edge_value2);
            }
        }
    }

    (roots2, god2)
}

// This conversion function also remains the same.
fn convert_from_intermediate2(
    roots2: &BTreeMap<TokenizerStateID, Intermediate2PrecomputeNode3Index>,
    god2: &Intermediate2Trie3GodWrapper,
) -> (
    BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
    IntermediateTrie3GodWrapper,
) {
    let god1 = IntermediateTrie3GodWrapper::new();
    let mut roots1 = BTreeMap::new();
    let mut node_map: HashMap<Intermediate2PrecomputeNode3Index, IntermediatePrecomputeNode3Index> =
        HashMap::new();
    let mut q: std::collections::VecDeque<Intermediate2PrecomputeNode3Index> =
        std::collections::VecDeque::new();

    for (sid, root2) in roots2 {
        let root1 = IntermediatePrecomputeNode3Index::new(god1.insert(IntermediatePrecomputeNode3::new(
            root2.read(god2).unwrap().value.clone(),
        )));
        roots1.insert(*sid, root1);
        node_map.insert(*root2, root1);
        q.push_back(*root2);
    }

    let mut visited = HashSet::new();
    while let Some(idx2) = q.pop_front() {
        if !visited.insert(idx2) {
            continue;
        }
        let idx1 = *node_map.get(&idx2).unwrap();
        let guard2 = idx2.read(god2).unwrap();

        for (edge_key2, dest_map2) in guard2.children() {
            for (child2_idx, edge_value2) in dest_map2 {
                let child1_idx = *node_map.entry(*child2_idx).or_insert_with(|| {
                    let new_node = IntermediatePrecomputeNode3Index::new(god1.insert(
                        IntermediatePrecomputeNode3::new(child2_idx.read(god2).unwrap().value.clone()),
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
                            god1.insert_edge_simple(
                                idx1,
                                child1_idx,
                                IntermediateTrie3EdgeKey::CheckLLM(edge_value2.clone()),
                                (),
                            );
                        }
                        _ => {
                            // Need an intermediate node to separate op and CheckLLM
                            let inter = IntermediatePrecomputeNode3Index::new(god1.insert(
                                IntermediatePrecomputeNode3::new(
                                    IntermediatePrecomputedNodeContents3::internal(),
                                ),
                            ));
                            let edge_key1_op = match edge_key2 {
                                Intermediate2Trie3EdgeKey::Pop(n, s) => IntermediateTrie3EdgeKey::Pop(*n, s.clone()),
                                Intermediate2Trie3EdgeKey::Push(s) => IntermediateTrie3EdgeKey::Push(s.clone()),
                                _ => unreachable!(),
                            };
                            god1.insert_edge_simple(idx1, inter, edge_key1_op, ());
                            god1.insert_edge_simple(
                                inter,
                                child1_idx,
                                IntermediateTrie3EdgeKey::CheckLLM(edge_value2.clone()),
                                (),
                            );
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
    println!("Starting push/pop elimination...");
    let initial_nodes =
        IntermediatePrecomputeNode3::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>())
            .len();
    println!("Initial graph node count: {}", initial_nodes);

    // 1. Convert to Intermediate2 format, which moves token constraints to edge values.
    let (roots2, god2) = convert_to_intermediate2(roots, god);
    let intermediate_nodes = Intermediate2PrecomputeNode3::all_nodes(
        &god2,
        &roots2.values().cloned().collect::<Vec<_>>(),
    )
    .len();
    println!(
        "Intermediate graph node count: {}",
        intermediate_nodes
    );

    // --- New Optimized Implementation ---

    // 2. One-time precomputation of reverse adjacency list and other properties.
    let mut reverse_adj: HashMap<
        Intermediate2PrecomputeNode3Index,
        HashMap<Intermediate2PrecomputeNode3Index, Vec<(Intermediate2Trie3EdgeKey, LLMTokenBV)>>,
    > = HashMap::new();

    let all_nodes =
        Intermediate2PrecomputeNode3::all_nodes(&god2, &roots2.values().cloned().collect::<Vec<_>>());

    for &u in &all_nodes {
        if let Some(u_guard) = u.read(&god2) {
            for (edge_key, dest_map) in u_guard.children() {
                for (&v, edge_val) in dest_map {
                    reverse_adj
                        .entry(v)
                        .or_default()
                        .entry(u)
                        .or_default()
                        .push((edge_key.clone(), edge_val.clone()));
                }
            }
        }
    }

    // 3. Main loop: iteratively process candidates using a worklist.
    let mut worklist: Vec<_> = all_nodes.into_iter().collect();
    let mut processed_nodes = HashSet::new();
    let mut pass = 0;

    loop {
        pass += 1;
        let mut changed_in_pass = false;
        processed_nodes.clear();

        println!("\nPass {}: worklist size {}", pass, worklist.len());

        // --- Standard Candidate Processing ---
        let mut i = 0;
        while i < worklist.len() {
            let b_idx = worklist[i];
            i += 1;

            if processed_nodes.contains(&b_idx) { continue; }

            let b_guard = b_idx.read(&god2).unwrap();
            if b_guard.value.end { continue; }

            let has_outgoing_push = b_guard.children().keys().any(|k| matches!(k, Intermediate2Trie3EdgeKey::Push(_)));
            if has_outgoing_push { continue; }

            let incoming_pushes: Vec<_> = reverse_adj.get(&b_idx).map_or(Vec::new(), |preds| {
                preds.iter().flat_map(|(&a_idx, edges)| {
                    edges.iter().filter(|(k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_))).map(move |(k, v)| (a_idx, k.clone(), v.clone()))
                }).collect()
            });

            if incoming_pushes.is_empty() { continue; }

            // Is a standard candidate, so process it.
            changed_in_pass = true;
            processed_nodes.insert(b_idx);

            let outgoing_edges: Vec<_> = b_guard.children().iter().flat_map(|(k, dest_map)| {
                dest_map.iter().map(move |(&c_idx, val)| (k.clone(), c_idx, val.clone()))
            }).collect();

            // Remove all incoming push edges to B.
            for (a_idx, edge_key, _) in &incoming_pushes {
                god2.remove_edge(*a_idx, b_idx, edge_key);
                worklist.push(*a_idx);
            }

            // Create new "shortcut" edges from A to C.
            for (a_idx, push_key, tokens_a_b) in &incoming_pushes {
                let s = match push_key {
                    Intermediate2Trie3EdgeKey::Push(s) => s,
                    _ => unreachable!(),
                };

                for (op_key, c_idx, tokens_b_c) in &outgoing_edges {
                    let new_tokens = tokens_a_b & tokens_b_c;
                    if new_tokens.is_empty() { continue; }

                    let new_key_opt = match op_key {
                        Intermediate2Trie3EdgeKey::Pop(0, s_prime) => (!s.is_disjoint(s_prime)).then_some(Intermediate2Trie3EdgeKey::Push(s & s_prime)),
                        Intermediate2Trie3EdgeKey::Pop(1, s_prime) => (!s.is_disjoint(s_prime)).then_some(Intermediate2Trie3EdgeKey::NoOp),
                        Intermediate2Trie3EdgeKey::Pop(n, s_prime) => Some(Intermediate2Trie3EdgeKey::Pop(n - 1, s_prime.clone())),
                        Intermediate2Trie3EdgeKey::NoOp => Some(Intermediate2Trie3EdgeKey::Push(s.clone())),
                        Intermediate2Trie3EdgeKey::Push(_) => unreachable!("Node to process should not have outgoing pushes"),
                    };

                    if let Some(new_key) = new_key_opt {
                        god2.insert_edge_simple(*a_idx, *c_idx, new_key, new_tokens);
                        worklist.push(*a_idx);
                        worklist.push(*c_idx);
                    }
                }
            }
        }
        worklist.clear();

        if changed_in_pass {
            // Rebuild reverse_adj and continue with standard candidates
            reverse_adj.clear();
            let all_nodes = Intermediate2PrecomputeNode3::all_nodes(&god2, &roots2.values().cloned().collect::<Vec<_>>());
            for &u in &all_nodes {
                if let Some(u_guard) = u.read(&god2) {
                    for (edge_key, dest_map) in u_guard.children() {
                        for (&v, edge_val) in dest_map {
                            reverse_adj.entry(v).or_default().entry(u).or_default().push((edge_key.clone(), edge_val.clone()));
                        }
                    }
                }
                worklist.push(u);
            }
            continue;
        }

        // --- Split Candidate Processing ---
        let mut split_candidates = Vec::new();
        let all_nodes = Intermediate2PrecomputeNode3::all_nodes(&god2, &roots2.values().cloned().collect::<Vec<_>>());
        for b_idx in all_nodes {
            let b_guard = b_idx.read(&god2).unwrap();
            if b_guard.value.end { continue; }

            let has_outgoing_push = b_guard.children().keys().any(|k| matches!(k, Intermediate2Trie3EdgeKey::Push(_)));
            let has_outgoing_nonpush = b_guard.children().keys().any(|k| !matches!(k, Intermediate2Trie3EdgeKey::Push(_)));

            if has_outgoing_push && has_outgoing_nonpush {
                if let Some(preds) = reverse_adj.get(&b_idx) {
                    if preds.values().any(|edges| edges.iter().any(|(k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_)))) {
                        split_candidates.push(b_idx);
                    }
                }
            }
        }

        if !split_candidates.is_empty() {
            println!("  No standard candidates. Found {} split candidates.", split_candidates.len());
            for b_idx in split_candidates {
                let b_value = b_idx.read(&god2).unwrap().value.clone();
                let b_np_idx = Intermediate2PrecomputeNode3Index::new(god2.insert(Intermediate2PrecomputeNode3::new(b_value)));

                let mut to_move = Vec::new();
                if let Some(b_guard) = b_idx.read(&god2) {
                    for (edge_key, dest_map) in b_guard.children() {
                        if !matches!(edge_key, Intermediate2Trie3EdgeKey::Push(_)) {
                            for (&c_idx, val) in dest_map {
                                to_move.push((edge_key.clone(), c_idx, val.clone()));
                            }
                        }
                    }
                }
                for (k, c_idx, val) in to_move {
                    god2.insert_edge_simple(b_np_idx, c_idx, k.clone(), val.clone());
                    god2.remove_edge(b_idx, c_idx, &k);
                }

                if let Some(preds) = reverse_adj.get(&b_idx).cloned() {
                    for (a_idx, edges) in preds {
                        for (k, val) in edges {
                            god2.insert_edge_simple(a_idx, b_np_idx, k, val);
                        }
                    }
                }
            }

            // Rebuild reverse_adj and restart with standard candidate processing
            reverse_adj.clear();
            let all_nodes = Intermediate2PrecomputeNode3::all_nodes(&god2, &roots2.values().cloned().collect::<Vec<_>>());
            for &u in &all_nodes {
                if let Some(u_guard) = u.read(&god2) {
                    for (edge_key, dest_map) in u_guard.children() {
                        for (&v, edge_val) in dest_map {
                            reverse_adj.entry(v).or_default().entry(u).or_default().push((edge_key.clone(), edge_val.clone()));
                        }
                    }
                }
                worklist.push(u);
            }
            continue;
        }

        // No changes from standard or split candidates, so we are done.
        println!("  No more candidates to process or split. Exiting loop.");
        break;
    }

    // 4. Convert back to the original Trie format.
    let (new_roots1_map, new_god1) = convert_from_intermediate2(&roots2, &god2);
    println!(
        "Finished elimination loop. Final intermediate graph has {} nodes.",
        IntermediatePrecomputeNode3::all_nodes(
            &new_god1,
            &new_roots1_map
                .values()
                .cloned()
                .collect::<Vec<_>>()
        )
        .len()
    );

    // 5. The function signature requires modifying `god` in place.
    // We clear the original `god` and deep-copy the new graph into it.
    let mut sids_in_order = Vec::new();
    let mut new_roots_vec = Vec::new();
    for (&sid, &root_idx) in &new_roots1_map {
        sids_in_order.push(sid);
        new_roots_vec.push(root_idx);
    }

    god.clear();
    let (final_roots_vec, _map) =
        IntermediatePrecomputeNode3::deep_copy_subtrees_into(&new_god1, god, &new_roots_vec);

    roots.clear();
    for (sid, final_root_idx) in sids_in_order.iter().zip(final_roots_vec.iter()) {
        roots.insert(*sid, *final_root_idx);
    }
    println!(
        "Final graph node count: {}",
        IntermediatePrecomputeNode3::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>())
            .len()
    );
}

// --- Assertion and Test Helpers (Unchanged) ---

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
                            let path = find_path_to_pop(*child_idx, god, &pop_reachable_memo);
                            let mut options =
                                crate::datastructures::trie::PrettyPrintOptions::default()
                                    .display_edge_keys_only()
                                    .omit_depth();
                            eprintln!("Full graph:");
                            eprintln!(
                                "{}",
                                Trie::pretty_print_with_options(
                                    god,
                                    roots.values().cloned().collect::<Vec<_>>().as_slice(),
                                    &options
                                )
                            );
                            eprintln!("Segment:");
                            eprintln!("{}", Trie::pretty_print_with_options(god, &[node_idx], &options));
                            panic!(
                                "Assertion failed: Pop is reachable from a Push edge. Path: Node {} --Push--> Node {} --> ... --> Pop. Path to pop: {:?}",
                                node_idx, child_idx, path
                            );
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
    visiting: &mut HashSet<IntermediatePrecomputeNode3Index>,
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
    pop_reachable_memo: &HashMap<IntermediatePrecomputeNode3Index, bool>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{
        IntermediatePrecomputedNodeContents3, IntermediateTrie3EdgeKey, LLMTokenBV, StateIDBV,
    };
    use crate::datastructures::trie::Trie;
    use crate::tokenizer::TokenizerStateID;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn test_eliminate_push_pop_failure_case() {
        let god = IntermediateTrie3GodWrapper::new();

        let mut node_map = HashMap::new();
        let node_ids = vec![5, 6, 7, 8, 9, 13, 14, 15, 16];
        for id in node_ids {
            node_map.insert(
                id,
                Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal()))),
            );
        }

        let n = |id: usize| -> Trie2Index { *node_map.get(&id).unwrap() };

        // Segment that previously failed:
        // 13 --Push--> 14 --Pop--> 15
        // 14 also has another outgoing Push to 16
        god.insert_edge_simple(n(13), n(14), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(1)), ());
        god.insert_edge_simple(n(14), n(15), IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()), ());
        god.insert_edge_simple(n(14), n(16), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(4)), ());
        god.insert_edge_simple(n(15), n(5), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(0)), ());
        god.insert_edge_simple(n(5), n(6), IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)), ());
        god.insert_edge_simple(n(6), n(7), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(1)), ());
        god.insert_edge_simple(n(7), n(8), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(2)), ());
        god.insert_edge_simple(n(8), n(9), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(0)), ());
        god.insert_edge_simple(n(16), n(9), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(2)), ());

        let mut roots = BTreeMap::new();
        roots.insert(TokenizerStateID(0), n(13));

        eliminate_pushes_and_pops(&mut roots, &god);

        // This assertion should now pass.
        assert_no_pops_reachable_from_pushes(&roots, &god);
    }

    #[test]
    fn test_eliminate_push_noop_pop_failure_case() {
        let god = IntermediateTrie3GodWrapper::new();

        let mut node_map = HashMap::new();
        // N101: Push source (Root)
        // N102: Problematic node (Push in, Push out, NoOp out to Pop path)
        // N103: NoOp target (Pop source)
        // N104: Pop target
        // N105: Push target / Cycle target
        let node_ids = vec![101, 102, 103, 104, 105];
        for id in node_ids {
            node_map.insert(
                id,
                Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal()))),
            );
        }

        let n = |id: usize| -> Trie2Index { *node_map.get(&id).unwrap() };

        // N101 --Push(1)--> N102
        god.insert_edge_simple(n(101), n(102), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(1)), ());
        // N102 --NoOp--> N103
        god.insert_edge_simple(n(102), n(103), IntermediateTrie3EdgeKey::NoOp, ());
        // N102 --Push(2)--> N105
        god.insert_edge_simple(n(102), n(105), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(2)), ());
        // N103 --Pop(1, ALL)--> N104
        god.insert_edge_simple(n(103), n(104), IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()), ());
        // N105 --CheckLLM(1)--> N101 (cycle)
        god.insert_edge_simple(n(105), n(101), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(1)), ());

        let mut roots = BTreeMap::new();
        roots.insert(TokenizerStateID(0), n(101));

        eliminate_pushes_and_pops(&mut roots, &god);
        assert_no_pops_reachable_from_pushes(&roots, &god);
    }
}