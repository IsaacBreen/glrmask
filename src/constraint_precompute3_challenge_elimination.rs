use crate::constraint::{
    IntermediatePrecomputeNode3, IntermediatePrecomputeNode3Index, IntermediatePrecomputedNodeContents3,
    IntermediateTrie3EdgeKey, IntermediateTrie3GodWrapper, LLMTokenBV, StateIDBV,
};
use crate::datastructures::trie::{GodWrapper, MergeableEdgeValue, Trie, Trie2Index};
use crate::tokenizer::TokenizerStateID;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
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
    // 1) Convert to Intermediate2 format, which moves token constraints to edge values.
    let (roots2, god2) = convert_to_intermediate2(roots, god);

    // 2) Build the set of reachable nodes once.
    let all_nodes = Intermediate2PrecomputeNode3::all_nodes(
        &god2,
        &roots2.values().cloned().collect::<Vec<_>>(),
    );

    // 3) Build reverse adjacency once and keep it incrementally updated.
    let mut rev: HashMap<
        Intermediate2PrecomputeNode3Index,
        Vec<(
            Intermediate2PrecomputeNode3Index,
            Intermediate2Trie3EdgeKey,
            LLMTokenBV,
        )>,
    > = HashMap::new();
    for &u in &all_nodes {
        if let Some(u_guard) = u.read(&god2) {
            for (edge_key, dest_map) in u_guard.children() {
                for (&v, edge_val) in dest_map {
                    rev.entry(v)
                        .or_default()
                        .push((u, edge_key.clone(), edge_val.clone()));
                }
            }
        }
    }

    // Helpers
    let has_outgoing_push = |node: Intermediate2PrecomputeNode3Index| -> bool {
        node.read(&god2)
            .map_or(false, |g| g.children().keys().any(|k| matches!(k, Intermediate2Trie3EdgeKey::Push(_))))
    };
    let has_outgoing_nonpush = |node: Intermediate2PrecomputeNode3Index| -> bool {
        node.read(&god2)
            .map_or(false, |g| g.children().keys().any(|k| !matches!(k, Intermediate2Trie3EdgeKey::Push(_))))
    };
    let is_end = |node: Intermediate2PrecomputeNode3Index| -> bool {
        node.read(&god2).map_or(false, |g| g.value.end)
    };
    // Keep rev in sync for a single (u, v, k) after insert/remove.
    let mut sync_rev_entry_for = |u: Intermediate2PrecomputeNode3Index,
                                  v: Intermediate2PrecomputeNode3Index,
                                  k: &Intermediate2Trie3EdgeKey| {
        if let Some(vec) = rev.get_mut(&v) {
            vec.retain(|(uu, kk, _)| !(*uu == u && *kk == *k));
        }
        if let Some(guard) = u.read(&god2) {
            if let Some(dest_map) = guard.children().get(k) {
                if let Some(val) = dest_map.get(&v) {
                    rev.entry(v)
                        .or_default()
                        .push((u, k.clone(), val.clone()));
                }
            }
        }
    };

    // 4) Prepare an initial work queue: nodes that are not end, have at least one incoming push, and no outgoing push.
    let mut queue: VecDeque<Intermediate2PrecomputeNode3Index> = VecDeque::new();
    let mut in_queue: HashSet<Intermediate2PrecomputeNode3Index> = HashSet::new();
    for &b in &all_nodes {
        if is_end(b) {
            continue;
        }
        let has_incoming_push = rev
            .get(&b)
            .map_or(false, |incoming| incoming.iter().any(|(_, k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_))));
        if has_incoming_push && !has_outgoing_push(b) {
            queue.push_back(b);
            in_queue.insert(b);
        }
    }

    // Maintain a set of potential split candidates incrementally.
    let mut split_candidates: HashSet<Intermediate2PrecomputeNode3Index> = HashSet::new();
    for &b in &all_nodes {
        if is_end(b) {
            continue;
        }
        let has_incoming_push = rev
            .get(&b)
            .map_or(false, |incoming| incoming.iter().any(|(_, k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_))));
        if has_incoming_push && has_outgoing_push(b) && has_outgoing_nonpush(b) {
            split_candidates.insert(b);
        }
    }

    // 5) Event-driven elimination loop.
    loop {
        while let Some(b) = queue.pop_front() {
            in_queue.remove(&b);
            if is_end(b) {
                continue;
            }
            if has_outgoing_push(b) {
                // Not eligible anymore (maybe changed); skip.
                continue;
            }

            // Snapshot incoming pushes to b
            let incoming_list = rev.get(&b).cloned().unwrap_or_default();
            let incoming_pushes: Vec<_> = incoming_list
                .into_iter()
                .filter(|(_, k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_)))
                .collect();
            if incoming_pushes.is_empty() {
                continue;
            }

            // Gather outgoing non-push edges from b
            let mut outgoing_nonpush: Vec<(Intermediate2Trie3EdgeKey, Intermediate2PrecomputeNode3Index, LLMTokenBV)> = Vec::new();
            if let Some(b_guard) = b.read(&god2) {
                for (k, dest_map) in b_guard.children() {
                    if matches!(k, Intermediate2Trie3EdgeKey::Push(_)) {
                        continue;
                    }
                    for (&c, val) in dest_map {
                        outgoing_nonpush.push((k.clone(), c, val.clone()));
                    }
                }
            }

            // Remove all incoming push edges to b
            for (a, push_k, _) in &incoming_pushes {
                god2.remove_edge(*a, b, push_k);
                sync_rev_entry_for(*a, b, push_k);
            }

            // Create shortcut edges from a to c
            for (a, push_k, tokens_a_b) in incoming_pushes {
                let s = match &push_k {
                    Intermediate2Trie3EdgeKey::Push(s) => s.clone(),
                    _ => unreachable!(),
                };
                for (op_k, c, tokens_b_c) in &outgoing_nonpush {
                    let new_tokens = tokens_a_b.clone() & tokens_b_c.clone();
                    if new_tokens.is_empty() {
                        continue;
                    }
                    let new_key_opt = match op_k {
                        Intermediate2Trie3EdgeKey::Pop(0, s_prime) => {
                            (!s.is_disjoint(s_prime)).then_some(Intermediate2Trie3EdgeKey::Push(s.clone() & s_prime.clone()))
                        }
                        Intermediate2Trie3EdgeKey::Pop(1, s_prime) => {
                            (!s.is_disjoint(s_prime)).then_some(Intermediate2Trie3EdgeKey::NoOp)
                        }
                        Intermediate2Trie3EdgeKey::Pop(n, s_prime) => {
                            Some(Intermediate2Trie3EdgeKey::Pop(n - 1, s_prime.clone()))
                        }
                        Intermediate2Trie3EdgeKey::NoOp => {
                            Some(Intermediate2Trie3EdgeKey::Push(s.clone()))
                        }
                        Intermediate2Trie3EdgeKey::Push(_) => unreachable!(),
                    };
                    if let Some(new_k) = new_key_opt {
                        god2.insert_edge_simple(a, *c, new_k.clone(), new_tokens.clone());
                        // Keep reverse adjacency accurate (dedupe existing entry).
                        sync_rev_entry_for(a, *c, &new_k);
                        if let Intermediate2Trie3EdgeKey::Push(_) = new_k {
                            // If c has no outgoing push, it's a candidate to process next.
                            if !is_end(*c) && !has_outgoing_push(*c) && in_queue.insert(*c) {
                                queue.push_back(*c);
                            }
                            // If c has both push and non-push outgoing edges, it's a split candidate.
                            if has_outgoing_push(*c) && has_outgoing_nonpush(*c) {
                                split_candidates.insert(*c);
                            }
                        }
                    }
                }
            }
        }

        // Queue is empty. If there are split candidates that still qualify, split them in batch.
        let mut to_split: Vec<Intermediate2PrecomputeNode3Index> = Vec::new();
        for &b in split_candidates.iter() {
            if is_end(b) {
                continue;
            }
            let has_incoming_push = rev
                .get(&b)
                .map_or(false, |incoming| incoming.iter().any(|(_, k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_))));
            if has_incoming_push && has_outgoing_push(b) && has_outgoing_nonpush(b) {
                to_split.push(b);
            }
        }
        if to_split.is_empty() {
            break;
        }

        for b in to_split {
            split_candidates.remove(&b);
            // Create the new "non-push" clone node
            let b_value = b.read(&god2).unwrap().value.clone();
            let b_np = Intermediate2PrecomputeNode3Index::new(
                god2.insert(Intermediate2PrecomputeNode3::new(b_value)),
            );

            // Move all non-push outgoing edges from b -> (...) to b_np -> (...)
            let mut to_move: Vec<(Intermediate2Trie3EdgeKey, Intermediate2PrecomputeNode3Index, LLMTokenBV)> = Vec::new();
            if let Some(b_guard) = b.read(&god2) {
                for (edge_key, dest_map) in b_guard.children() {
                    if !matches!(edge_key, Intermediate2Trie3EdgeKey::Push(_)) {
                        for (&c, val) in dest_map {
                            to_move.push((edge_key.clone(), c, val.clone()));
                        }
                    }
                }
            }
            for (k, c, val) in &to_move {
                god2.insert_edge_simple(b_np, *c, k.clone(), val.clone());
                sync_rev_entry_for(b_np, *c, &k);
            }
            for (k, c, _) in &to_move {
                god2.remove_edge(b, *c, &k);
                sync_rev_entry_for(b, *c, &k);
            }

            // Duplicate all incoming edges so both b (push-only) and b_np (non-push-only) are reachable
            if let Some(incoming) = rev.get(&b).cloned() {
                for (a, k, val) in incoming {
                    god2.insert_edge_simple(a, b_np, k.clone(), val.clone());
                    sync_rev_entry_for(a, b_np, &k);
                    if let Intermediate2Trie3EdgeKey::Push(_) = k {
                        if !is_end(b_np) && !has_outgoing_push(b_np) && in_queue.insert(b_np) {
                            queue.push_back(b_np);
                        }
                    }
                }
            }
        }
    }

    // 6) Convert back to the original Trie format.
    let (new_roots1_map, new_god1) = convert_from_intermediate2(&roots2, &god2);

    // 7) The function signature requires modifying `god` in place.
    // Clear the original `god` and deep-copy the new graph into it.
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