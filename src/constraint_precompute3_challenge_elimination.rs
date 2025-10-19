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

    // Reusable clone per node (a "no-pop-path" view for incoming pushes).
    let mut clone_for: HashMap<Intermediate2PrecomputeNode3Index, Intermediate2PrecomputeNode3Index> = HashMap::new();

    loop {
        Trie::gc(&god2, &roots2.values().cloned().collect::<Vec<_>>());
        let all_nodes = Trie::all_nodes(&god2, &roots2.values().cloned().collect::<Vec<_>>());

        // Build map of incoming Push predecessors for each node.
        let mut preds: HashMap<Intermediate2PrecomputeNode3Index, Vec<(Intermediate2PrecomputeNode3Index, StateIDBV, LLMTokenBV)>> = HashMap::new();
        for &src in &all_nodes {
            if let Some(guard) = src.read(&god2) {
                for (ek, dm) in guard.children() {
                    if let Intermediate2Trie3EdgeKey::Push(s) = ek {
                        for (dst, ev) in dm {
                            preds.entry(*dst).or_default().push((src, s.clone(), ev.clone()));
                        }
                    }
                }
            }
        }

        // Memo: whether a Pop is reachable from a node (along any edges).
        let mut pop_memo: HashMap<Intermediate2PrecomputeNode3Index, bool> = HashMap::new();
        fn pop_reachable_from(
            node: Intermediate2PrecomputeNode3Index,
            god2: &Intermediate2Trie3GodWrapper,
            memo: &mut HashMap<Intermediate2PrecomputeNode3Index, bool>,
            visiting: &mut HashSet<Intermediate2PrecomputeNode3Index>,
        ) -> bool {
            if let Some(&res) = memo.get(&node) { return res; }
            if !visiting.insert(node) { return false; }
            let mut res = false;
            if let Some(guard) = node.read(god2) {
                'outer: for (ek, dm) in guard.children() {
                    if matches!(ek, Intermediate2Trie3EdgeKey::Pop(_, _)) {
                        res = true;
                        break 'outer;
                    }
                    for child in dm.keys() {
                        if pop_reachable_from(*child, god2, memo, visiting) {
                            res = true;
                            break 'outer;
                        }
                    }
                }
            }
            visiting.remove(&node);
            memo.insert(node, res);
            res
        }
        for &n in &all_nodes {
            pop_reachable_from(n, &god2, &mut pop_memo, &mut HashSet::new());
        }

        let mut changed = false;

        // For each node B with incoming pushes and Pop reachable from B:
        for &b in &all_nodes {
            let incoming_pushes = match preds.get(&b) {
                Some(v) if !v.is_empty() => v.clone(),
                _ => continue,
            };
            if !*pop_memo.get(&b).unwrap_or(&false) {
                continue;
            }

            // Classify outgoing edges from B.
            let mut keep_edges: Vec<(Intermediate2Trie3EdgeKey, Intermediate2PrecomputeNode3Index, LLMTokenBV)> = Vec::new();
            let mut split_edges: Vec<(Intermediate2Trie3EdgeKey, Intermediate2PrecomputeNode3Index, LLMTokenBV)> = Vec::new();
            if let Some(guard) = b.read(&god2) {
                for (ek, dm) in guard.children() {
                    for (c, ev) in dm {
                        let on_pop_path = matches!(ek, Intermediate2Trie3EdgeKey::Pop(_, _))
                            || *pop_memo.get(c).unwrap_or(&false);
                        if on_pop_path && !matches!(ek, Intermediate2Trie3EdgeKey::Push(_)) {
                            split_edges.push((ek.clone(), *c, ev.clone()));
                        } else {
                            keep_edges.push((ek.clone(), *c, ev.clone()));
                        }
                    }
                }
            }

            if split_edges.is_empty() {
                continue;
            }
            changed = true;

            // Optionally create/reuse a clone that keeps only non-pop-path edges.
            let clone_idx_opt = if keep_edges.is_empty() {
                None
            } else {
                let clone_idx = *clone_for.entry(b).or_insert_with(|| {
                    let b_value = b.read(&god2).unwrap().value.clone();
                    let new_node = Intermediate2PrecomputeNode3Index::new(
                        god2.insert(Intermediate2PrecomputeNode3::new(b_value))
                    );
                    for (ek, c, ev) in &keep_edges {
                        god2.insert_edge_simple(new_node, *c, ek.clone(), ev.clone());
                    }
                    new_node
                });
                Some(clone_idx)
            };

            // Redirect/remove A --Push(S)--> B to A --Push(S)--> clone (if clone exists).
            for (a, s, bv_ab) in &incoming_pushes {
                god2.remove_edge(*a, b, &Intermediate2Trie3EdgeKey::Push(s.clone()));
                if let Some(clone_idx) = clone_idx_opt {
                    god2.insert_edge_simple(*a, clone_idx, Intermediate2Trie3EdgeKey::Push(s.clone()), bv_ab.clone());
                }
            }

            // Splice compositions for split edges (NoOp/Pop encountered on a pop-path).
            for (op, c, bv_bc) in &split_edges {
                for (a, s, bv_ab) in &incoming_pushes {
                    let new_bv = bv_ab & bv_bc;
                    if new_bv.is_empty() { continue; }
                    match op {
                        Intermediate2Trie3EdgeKey::NoOp => {
                            god2.insert_edge_simple(*a, *c, Intermediate2Trie3EdgeKey::Push(s.clone()), new_bv.clone());
                        }
                        Intermediate2Trie3EdgeKey::Pop(0, s_prime) => {
                            let inter = s & s_prime;
                            if !inter.is_empty() {
                                god2.insert_edge_simple(*a, *c, Intermediate2Trie3EdgeKey::Push(inter), new_bv.clone());
                            }
                        }
                        Intermediate2Trie3EdgeKey::Pop(1, s_prime) => {
                            if !s.is_disjoint(s_prime) {
                                god2.insert_edge_simple(*a, *c, Intermediate2Trie3EdgeKey::NoOp, new_bv.clone());
                            }
                        }
                        Intermediate2Trie3EdgeKey::Pop(n, s_prime) => {
                            if *n >= 2 {
                                god2.insert_edge_simple(*a, *c, Intermediate2Trie3EdgeKey::Pop(*n - 1, s_prime.clone()), new_bv.clone());
                            }
                        }
                        Intermediate2Trie3EdgeKey::Push(_) => { /* not part of split_edges */ }
                    }
                }
            }

            // Remove split edges from B; keep only the non-pop-path surface.
            for (op, c, _) in &split_edges {
                god2.remove_edge(b, *c, op);
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{IntermediatePrecomputedNodeContents3, IntermediateTrie3EdgeKey, LLMTokenBV, StateIDBV};
    use crate::datastructures::trie::Trie;
    use crate::tokenizer::TokenizerStateID;
    use std::collections::{BTreeMap, HashMap};

    #[test]
    fn test_eliminate_push_pop_failure_case() {
        let god = IntermediateTrie3GodWrapper::new();
        
        let mut node_map = HashMap::new();
        let node_ids = vec![5, 6, 7, 8, 9, 13, 14, 15, 16];
        for id in node_ids {
            node_map.insert(id, Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal()))));
        }

        let n = |id: usize| -> Trie2Index { *node_map.get(&id).unwrap() };

        // Add edges to replicate the graph structure from the panic log's "Segment"
        // Path that causes the issue: 13 --Push--> 14 --Pop--> 15
        // Node 14 also has another outgoing Push to 16, which prevents the current
        // implementation of `eliminate_pushes_and_pops` from processing node 14.
        
        // From node 13
        god.insert_edge_simple(n(13), n(14), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(1)), ());
        
        // From node 14
        god.insert_edge_simple(n(14), n(15), IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()), ());
        god.insert_edge_simple(n(14), n(16), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(4)), ());

        // From node 15
        god.insert_edge_simple(n(15), n(5), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(0)), ());
        
        // From node 5
        god.insert_edge_simple(n(5), n(6), IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)), ());

        // From node 6
        god.insert_edge_simple(n(6), n(7), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(1)), ());
        
        // From node 7
        god.insert_edge_simple(n(7), n(8), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(2)), ());
        
        // From node 8
        god.insert_edge_simple(n(8), n(9), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(0)), ());
        
        // From node 16
        god.insert_edge_simple(n(16), n(9), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(2)), ());

        let mut roots = BTreeMap::new();
        roots.insert(TokenizerStateID(0), n(13));

        // This call is expected to fail to eliminate the push->pop sequence
        eliminate_pushes_and_pops(&mut roots, &god);

        // This assertion will then panic, which is expected by the test.
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
            node_map.insert(id, Trie2Index::from(god.insert(Trie::new(IntermediatePrecomputedNodeContents3::internal()))));
        }

        let n = |id: usize| -> Trie2Index { *node_map.get(&id).unwrap() };

        // N101 --Push(1)--> N102 (Incoming Push to N102)
        god.insert_edge_simple(n(101), n(102), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(1)), ());
        
        // N102 --NoOp--> N103 (NoOp out)
        god.insert_edge_simple(n(102), n(103), IntermediateTrie3EdgeKey::NoOp, ());
        
        // N102 --Push(2)--> N105 (Push out - prevents N102 from being processed in the first loop)
        god.insert_edge_simple(n(102), n(105), IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(2)), ());

        // N103 --Pop(1, ALL)--> N104 (Pop out - makes Pop reachable from N102)
        god.insert_edge_simple(n(103), n(104), IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()), ());

        // N105 --CheckLLM(1)--> N101 (Cycle to keep nodes alive)
        god.insert_edge_simple(n(105), n(101), IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(1)), ());

        let mut roots = BTreeMap::new();
        roots.insert(TokenizerStateID(0), n(101));

        eliminate_pushes_and_pops(&mut roots, &god);

        assert_no_pops_reachable_from_pushes(&roots, &god);
    }
}
