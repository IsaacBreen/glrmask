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
        let root2 = *node_map.entry(*root1).or_insert_with(|| {
            let new_node = Intermediate2PrecomputeNode3Index::new(god2.insert(
                Intermediate2PrecomputeNode3::new(root1.read(god1).unwrap().value.clone()),
            ));
            q.push_back(*root1);
            new_node
        });
        roots2.insert(*sid, root2);
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
        let root1 = *node_map.entry(*root2).or_insert_with(|| {
            let new_node = IntermediatePrecomputeNode3Index::new(god1.insert(
                IntermediatePrecomputeNode3::new(root2.read(god2).unwrap().value.clone()),
            ));
            q.push_back(*root2);
            new_node
        });
        roots1.insert(*sid, root1);
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
    const DEBUG: bool = true;
    if DEBUG {
        let initial_nodes =
            IntermediatePrecomputeNode3::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>())
                .len();
        println!("Starting push/pop elimination... nodes: {}", initial_nodes);
    }

    // Convert to Intermediate2 format (moves token constraints to edge values).
    let (roots2, god2) = convert_to_intermediate2(roots, god);
    if DEBUG {
        let inter_nodes = Intermediate2PrecomputeNode3::all_nodes(
            &god2,
            &roots2.values().cloned().collect::<Vec<_>>(),
        )
        .len();
        println!("Intermediate graph node count: {}", inter_nodes);
    }

    // Build initial reverse adjacency and per-node stats once. We'll maintain them incrementally.
    let mut known_nodes_vec = Intermediate2PrecomputeNode3::all_nodes(
        &god2,
        &roots2.values().cloned().collect::<Vec<_>>(),
    );
    let mut known_nodes: HashSet<Intermediate2PrecomputeNode3Index> =
        known_nodes_vec.iter().cloned().collect();

    // Reverse adjacency: for each node v, store list of (u, key, val) where u --key(val)--> v.
    let mut reverse_adj: HashMap<
        Intermediate2PrecomputeNode3Index,
        Vec<(
            Intermediate2PrecomputeNode3Index,
            Intermediate2Trie3EdgeKey,
            LLMTokenBV,
        )>,
    > = HashMap::new();

    // Per-node stats
    let mut outgoing_push_count: HashMap<Intermediate2PrecomputeNode3Index, usize> = HashMap::new();
    let mut has_outgoing_push: HashMap<Intermediate2PrecomputeNode3Index, bool> = HashMap::new();
    let mut has_outgoing_nonpush: HashMap<Intermediate2PrecomputeNode3Index, bool> = HashMap::new();
    let mut incoming_push_count: HashMap<Intermediate2PrecomputeNode3Index, usize> = HashMap::new();
    let mut is_end: HashSet<Intermediate2PrecomputeNode3Index> = HashSet::new();

    for &u in &known_nodes_vec {
        if let Some(guard) = u.read(&god2) {
            if guard.value.end {
                is_end.insert(u);
            }
            let mut push_count_u = 0usize;
            let mut has_push = false;
            let mut has_nonpush = false;
            for (edge_key, dest_map) in guard.children() {
                let is_push_key = matches!(edge_key, Intermediate2Trie3EdgeKey::Push(_));
                if is_push_key {
                    has_push = true;
                    push_count_u += dest_map.len();
                } else {
                    has_nonpush = true;
                }
                for (&v, val) in dest_map {
                    reverse_adj.entry(v).or_default().push((u, edge_key.clone(), val.clone()));
                }
            }
            outgoing_push_count.insert(u, push_count_u);
            has_outgoing_push.insert(u, has_push);
            has_outgoing_nonpush.insert(u, has_nonpush);
        }
    }
    for (v, incomings) in &reverse_adj {
        let cnt = incomings
            .iter()
            .filter(|(_, k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_)))
            .count();
        incoming_push_count.insert(*v, cnt);
    }

    // Candidate worklist: nodes with at least one incoming Push and no outgoing Push.
    let mut queue: std::collections::VecDeque<Intermediate2PrecomputeNode3Index> =
        std::collections::VecDeque::new();
    for &b in &known_nodes {
        if !is_end.contains(&b)
            && !has_outgoing_push.get(&b).copied().unwrap_or(false)
            && incoming_push_count.get(&b).copied().unwrap_or(0) > 0
        {
            queue.push_back(b);
        }
    }

    loop {
        let mut made_progress = false;

        // Process all standard candidates available in the queue.
        while let Some(b_idx) = queue.pop_front() {
            if is_end.contains(&b_idx) {
                continue;
            }
            if has_outgoing_push.get(&b_idx).copied().unwrap_or(false) {
                continue;
            }
            let incoming_list = reverse_adj.get(&b_idx).cloned().unwrap_or_default();
            let incoming_pushes: Vec<_> = incoming_list
                .into_iter()
                .filter(|(_, k, _)| matches!(k, Intermediate2Trie3EdgeKey::Push(_)))
                .collect();
            if incoming_pushes.is_empty() {
                continue;
            }

            let outgoing_edges: Vec<_> = if let Some(guard) = b_idx.read(&god2) {
                guard
                    .children()
                    .iter()
                    .flat_map(|(k, dest_map)| {
                        dest_map
                            .iter()
                            .map(move |(&c_idx, val)| (k.clone(), c_idx, val.clone()))
                    })
                    .collect()
            } else {
                Vec::new()
            };

            made_progress = true;

            for (a_idx, push_key, tokens_a_b) in incoming_pushes {
                // Remove A --Push--> B edge completely.
                god2.remove_edge(a_idx, b_idx, &push_key);
                if let Some(vec) = reverse_adj.get_mut(&b_idx) {
                    vec.retain(|(uu, kk, _)| !(*uu == a_idx && *kk == push_key));
                }
                if let Some(cnt) = incoming_push_count.get_mut(&b_idx) {
                    if *cnt > 0 {
                        *cnt -= 1;
                    }
                }
                // Update A's outgoing push count and enqueue A if it becomes a candidate.
                if matches!(push_key, Intermediate2Trie3EdgeKey::Push(_)) {
                    let count_a = outgoing_push_count.entry(a_idx).or_insert(0);
                    if *count_a > 0 {
                        *count_a -= 1;
                    }
                    if *count_a == 0 {
                        has_outgoing_push.insert(a_idx, false);
                        if !is_end.contains(&a_idx)
                            && incoming_push_count.get(&a_idx).copied().unwrap_or(0) > 0
                        {
                            queue.push_back(a_idx);
                        }
                    }
                }

                // Compose A --Push(s)--> B --op--> C into A --new_key--> C
                let s = match &push_key {
                    Intermediate2Trie3EdgeKey::Push(s) => s.clone(),
                    _ => unreachable!(),
                };
                for (op_key, c_idx, tokens_b_c) in &outgoing_edges {
                    let new_tokens = tokens_a_b.clone() & tokens_b_c.clone();
                    if new_tokens.is_empty() {
                        continue;
                    }
                    let new_key_opt = match op_key {
                        Intermediate2Trie3EdgeKey::Pop(0, s_prime) => {
                            if !s.is_disjoint(s_prime) {
                                Some(Intermediate2Trie3EdgeKey::Push(s.clone() & s_prime.clone()))
                            } else {
                                None
                            }
                        }
                        Intermediate2Trie3EdgeKey::Pop(1, s_prime) => {
                            if !s.is_disjoint(s_prime) {
                                Some(Intermediate2Trie3EdgeKey::NoOp)
                            } else {
                                None
                            }
                        }
                        Intermediate2Trie3EdgeKey::Pop(n, s_prime) => {
                            Some(Intermediate2Trie3EdgeKey::Pop(n - 1, s_prime.clone()))
                        }
                        Intermediate2Trie3EdgeKey::NoOp => {
                            Some(Intermediate2Trie3EdgeKey::Push(s.clone()))
                        }
                        Intermediate2Trie3EdgeKey::Push(_) => {
                            // By invariant, B has no outgoing pushes.
                            continue;
                        }
                    };
                    if let Some(new_key) = new_key_opt {
                        // Check if (A --new_key--> C) already existed before inserting (to keep counts consistent).
                        let existed_before = if let Some(aguard) = a_idx.read(&god2) {
                            aguard
                                .children()
                                .get(&new_key)
                                .map_or(false, |dm| dm.contains_key(c_idx))
                        } else {
                            false
                        };
                        god2.insert_edge_simple(a_idx, *c_idx, new_key.clone(), new_tokens.clone());
                        if let Some(aguard) = a_idx.read(&god2) {
                            if let Some(dest_map) = aguard.children().get(&new_key) {
                                if let Some(val) = dest_map.get(c_idx) {
                                    let vec = reverse_adj.entry(*c_idx).or_default();
                                    if let Some(pos) =
                                        vec.iter().position(|(uu, kk, _)| *uu == a_idx && *kk == new_key)
                                    {
                                        vec[pos].2 = val.clone();
                                    } else {
                                        vec.push((a_idx, new_key.clone(), val.clone()));
                                    }
                                }
                            }
                        }
                        match &new_key {
                            Intermediate2Trie3EdgeKey::Push(_) => {
                                if !existed_before {
                                    let count = outgoing_push_count.entry(a_idx).or_insert(0);
                                    *count += 1;
                                    has_outgoing_push.insert(a_idx, true);
                                    let cnt_in = incoming_push_count.entry(*c_idx).or_insert(0);
                                    *cnt_in += 1;
                                    if !is_end.contains(c_idx)
                                        && !has_outgoing_push.get(c_idx).copied().unwrap_or(false)
                                    {
                                        queue.push_back(*c_idx);
                                    }
                                }
                            }
                            _ => {
                                has_outgoing_nonpush.insert(a_idx, true);
                            }
                        }
                    }
                }
            }
        }

        if made_progress {
            continue;
        }

        // No more standard candidates. Try a split to create one.
        let mut split_candidate: Option<Intermediate2PrecomputeNode3Index> = None;
        for &b in &known_nodes {
            if is_end.contains(&b) {
                continue;
            }
            let has_push = has_outgoing_push.get(&b).copied().unwrap_or(false);
            let has_non = has_outgoing_nonpush.get(&b).copied().unwrap_or(false);
            let inc_p = incoming_push_count.get(&b).copied().unwrap_or(0);
            if has_push && has_non && inc_p > 0 {
                split_candidate = Some(b);
                break;
            }
        }

        if let Some(b_idx) = split_candidate {
            // Clone the node for non-push outgoing edges.
            let b_value = b_idx.read(&god2).unwrap().value.clone();
            let b_np_idx =
                Intermediate2PrecomputeNode3Index::new(god2.insert(Intermediate2PrecomputeNode3::new(
                    b_value,
                )));
            known_nodes.insert(b_np_idx);
            if is_end.contains(&b_idx) {
                is_end.insert(b_np_idx);
            }

            // Move all non-push outgoing edges from b_idx to b_np_idx.
            let to_move: Vec<(Intermediate2Trie3EdgeKey, Intermediate2PrecomputeNode3Index, LLMTokenBV)> =
                if let Some(b_guard) = b_idx.read(&god2) {
                    let mut v = Vec::new();
                    for (edge_key, dest_map) in b_guard.children() {
                        if !matches!(edge_key, Intermediate2Trie3EdgeKey::Push(_)) {
                            for (&c_idx, val) in dest_map {
                                v.push((edge_key.clone(), c_idx, val.clone()));
                            }
                        }
                    }
                    v
                } else {
                    Vec::new()
                };
            for (k, c_idx, val) in &to_move {
                god2.insert_edge_simple(b_np_idx, *c_idx, k.clone(), val.clone());
                if let Some(aguard) = b_np_idx.read(&god2) {
                    if let Some(dest_map) = aguard.children().get(k) {
                        if let Some(val) = dest_map.get(c_idx) {
                            let vec = reverse_adj.entry(*c_idx).or_default();
                            if let Some(pos) =
                                vec.iter().position(|(uu, kk, _)| *uu == b_np_idx && kk == k)
                            {
                                vec[pos].2 = val.clone();
                            } else {
                                vec.push((b_np_idx, k.clone(), val.clone()));
                            }
                        }
                    }
                }
                god2.remove_edge(b_idx, *c_idx, k);
                if let Some(vec) = reverse_adj.get_mut(&c_idx) {
                    vec.retain(|(uu, kk, _)| !(*uu == b_idx && *kk == *k));
                }
            }
            has_outgoing_nonpush.insert(b_idx, false);
            has_outgoing_push.insert(b_np_idx, false);
            has_outgoing_nonpush.insert(b_np_idx, !to_move.is_empty());

            // Duplicate all incoming edges so both b_idx (push-only) and b_np_idx (non-push-only) are reachable.
            if let Some(incoming) = reverse_adj.get(&b_idx).cloned() {
                for (a_idx, k, val) in incoming {
                    god2.insert_edge_simple(a_idx, b_np_idx, k.clone(), val.clone());
                    if let Some(aguard) = a_idx.read(&god2) {
                        if let Some(dest_map) = aguard.children().get(&k) {
                            if let Some(val) = dest_map.get(&b_np_idx) {
                                let vec = reverse_adj.entry(b_np_idx).or_default();
                                if let Some(pos) =
                                    vec.iter().position(|(uu, kk, _)| *uu == a_idx && *kk == k)
                                {
                                    vec[pos].2 = val.clone();
                                } else {
                                    vec.push((a_idx, k.clone(), val.clone()));
                                }
                            }
                        }
                    }
                    if matches!(k, Intermediate2Trie3EdgeKey::Push(_)) {
                        *incoming_push_count.entry(b_np_idx).or_insert(0) += 1;
                    }
                }
            }

            // After splitting, the new node is a standard candidate (no outgoing Push) if it has incoming pushes.
            if !is_end.contains(&b_np_idx)
                && !has_outgoing_push.get(&b_np_idx).copied().unwrap_or(false)
                && incoming_push_count.get(&b_np_idx).copied().unwrap_or(0) > 0
            {
                queue.push_back(b_np_idx);
            }
            continue;
        } else {
            // Finished: no more candidates and no split possible.
            break;
        }
    }

    // Convert back to the original Trie format.
    let (new_roots1_map, new_god1) = convert_from_intermediate2(&roots2, &god2);
    if DEBUG {
        let final_nodes = IntermediatePrecomputeNode3::all_nodes(
            &new_god1,
            &new_roots1_map.values().cloned().collect::<Vec<_>>(),
        )
        .len();
        println!("Finished elimination. Final intermediate nodes: {}", final_nodes);
    }

    // Deep-copy the new graph into the provided `god` and update roots.
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
    if DEBUG {
        let final_nodes =
            IntermediatePrecomputeNode3::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>())
                .len();
        println!("Final graph node count: {}", final_nodes);
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
    use std::collections::BTreeSet;

    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct NormalizedPath {
        llm_bv: LLMTokenBV,
        pops: BTreeMap<usize, StateIDBV>,
    }

    impl fmt::Display for NormalizedPath {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "NormalizedPath {{ llm_bv: {}, pops: {{", self.llm_bv)?;
            let mut first = true;
            for (pos, bv) in &self.pops {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", pos, bv)?;
                first = false;
            }
            write!(f, "}} }}")
        }
    }

    impl fmt::Debug for NormalizedPath {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self)
        }
    }

    fn normalize_path_challenges(path_keys: &mut Vec<IntermediateTrie3EdgeKey>) -> bool {
        loop {
            let last_push_idx =
                path_keys.iter().rposition(|k| matches!(k, IntermediateTrie3EdgeKey::Push(_)));

            if last_push_idx.is_none() {
                return true; // No more pushes, done.
            }
            let push_idx = last_push_idx.unwrap();

            let first_pop_after_push_idx = path_keys
                .iter()
                .skip(push_idx + 1)
                .position(|k| matches!(k, IntermediateTrie3EdgeKey::Pop(_, _)));

            if first_pop_after_push_idx.is_none() {
                // This push and any subsequent ones have no following pops.
                // The path is as reduced as it can be.
                return true;
            }
            let pop_idx = push_idx + 1 + first_pop_after_push_idx.unwrap();

            let push_key = path_keys[push_idx].clone();
            let pop_key = path_keys[pop_idx].clone();

            let s_push = match push_key {
                IntermediateTrie3EdgeKey::Push(s) => s,
                _ => unreachable!(),
            };
            let (n, s_pop) = match pop_key {
                IntermediateTrie3EdgeKey::Pop(n, s) => (n, s),
                _ => unreachable!(),
            };

            if n == 0 {
                if s_push.is_disjoint(&s_pop) {
                    return false;
                }
                path_keys[push_idx] = IntermediateTrie3EdgeKey::Push(s_push & s_pop);
                path_keys.remove(pop_idx);
            } else if n == 1 {
                if s_push.is_disjoint(&s_pop) {
                    return false;
                }
                path_keys.remove(pop_idx);
                path_keys.remove(push_idx);
            } else {
                // n > 1
                path_keys.remove(push_idx);
                path_keys[pop_idx - 1] = IntermediateTrie3EdgeKey::Pop(n - 1, s_pop);
            }
        }
    }

    fn get_normalized_paths(
        roots: &BTreeMap<TokenizerStateID, IntermediatePrecomputeNode3Index>,
        god: &IntermediateTrie3GodWrapper,
    ) -> BTreeSet<NormalizedPath> {
        let all_paths = Trie::get_all_paths(
            god,
            &roots.values().cloned().collect::<Vec<_>>(),
            |_, node| node.value.end,
        );

        let mut normalized_set = BTreeSet::new();

        for (_root_val, path_edges) in all_paths {
            let mut path_keys: Vec<IntermediateTrie3EdgeKey> =
                path_edges.into_iter().map(|(ek, _, _)| ek).collect();

            if !normalize_path_challenges(&mut path_keys) {
                continue; // Invalid path
            }

            let mut llm_bv = LLMTokenBV::max_ones();
            let mut pops = BTreeMap::new();
            let mut pop_pos = 0;

            for key in path_keys {
                match key {
                    IntermediateTrie3EdgeKey::CheckLLM(bv) => {
                        llm_bv &= &bv;
                    }
                    IntermediateTrie3EdgeKey::Pop(n, s) => {
                        pop_pos += n;
                        pops.entry(pop_pos)
                            .or_insert_with(StateIDBV::max_ones)
                            .intersects(&s);
                        pop_pos += 1;
                    }
                    IntermediateTrie3EdgeKey::Push(_) | IntermediateTrie3EdgeKey::NoOp => {
                        // ignore
                    }
                }
            }

            if !llm_bv.is_empty() {
                normalized_set.insert(NormalizedPath { llm_bv, pops });
            }
        }

        normalized_set
    }

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

        let paths_before = get_normalized_paths(&roots, &god);

        eliminate_pushes_and_pops(&mut roots, &god);

        let paths_after = get_normalized_paths(&roots, &god);
        assert_eq!(paths_before, paths_after);

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

        let paths_before = get_normalized_paths(&roots, &god);

        eliminate_pushes_and_pops(&mut roots, &god);

        let paths_after = get_normalized_paths(&roots, &god);
        assert_eq!(paths_before, paths_after);

        assert_no_pops_reachable_from_pushes(&roots, &god);
    }

    #[test]
    fn test_eliminate_complex_trie_from_prompt() {
        let god = IntermediateTrie3GodWrapper::new();

        let mut node_map = HashMap::new();
        let node_ids = vec![
            202, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 217, 218, 219, 220, 221,
            222, 223, 224, 225, 226, 227, 228, 229, 230, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 242, 243, 244, 245, 246, 250, 251, 252, 253, 254, 255, 256, 257, 258,
            259, 260, 261, 262, 263,
        ];
        for id in node_ids {
            let contents = if id == 204 {
                IntermediatePrecomputedNodeContents3::leaf()
            } else {
                IntermediatePrecomputedNodeContents3::internal()
            };
            node_map.insert(id, Trie2Index::from(god.insert(Trie::new(contents))));
        }

        let n = |id: usize| -> Trie2Index { *node_map.get(&id).unwrap() };

        // Root
        god.insert_edge_simple(n(202), n(206), IntermediateTrie3EdgeKey::NoOp, ());
        god.insert_edge_simple(n(202), n(217), IntermediateTrie3EdgeKey::NoOp, ());

        // Path from 206
        god.insert_edge_simple(
            n(206),
            n(207),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_iter([0, 3, 7])),
            (),
        );
        god.insert_edge_simple(
            n(207),
            n(211),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(9)),
            (),
        );
        god.insert_edge_simple(
            n(211),
            n(233),
            IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(1)),
            (),
        );

        // Path from 233
        god.insert_edge_simple(
            n(233),
            n(234),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(234),
            n(239),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(239),
            n(237),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(7)),
            (),
        );
        god.insert_edge_simple(
            n(237),
            n(241),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(241),
            n(244),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_iter([0, 3, 7])),
            (),
        );
        god.insert_edge_simple(
            n(244),
            n(237),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(7)),
            (),
        ); // Cycle
        god.insert_edge_simple(n(244), n(246), IntermediateTrie3EdgeKey::NoOp, ());
        god.insert_edge_simple(
            n(246),
            n(240),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(240),
            n(243),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(10)),
            (),
        );
        god.insert_edge_simple(
            n(243),
            n(250),
            IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(0)),
            (),
        );

        // Path from 250
        god.insert_edge_simple(
            n(250),
            n(251),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(251),
            n(256),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(256),
            n(254),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(7)),
            (),
        );
        god.insert_edge_simple(
            n(254),
            n(258),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(258),
            n(261),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_iter([0, 3, 7])),
            (),
        );
        god.insert_edge_simple(
            n(261),
            n(254),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(7)),
            (),
        ); // Cycle
        god.insert_edge_simple(n(261), n(263), IntermediateTrie3EdgeKey::NoOp, ());
        god.insert_edge_simple(
            n(263),
            n(257),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(257),
            n(260),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(10)),
            (),
        );
        god.insert_edge_simple(
            n(260),
            n(204),
            IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(0)),
            (),
        ); // To END
        god.insert_edge_simple(n(251), n(257), IntermediateTrie3EdgeKey::NoOp, ());
        god.insert_edge_simple(
            n(250),
            n(252),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(6)),
            (),
        );
        god.insert_edge_simple(
            n(252),
            n(256),
            IntermediateTrie3EdgeKey::Pop(2, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(250),
            n(253),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(8)),
            (),
        );
        god.insert_edge_simple(
            n(253),
            n(258),
            IntermediateTrie3EdgeKey::Pop(2, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(250),
            n(254),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(9)),
            (),
        );
        god.insert_edge_simple(
            n(250),
            n(255),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(10)),
            (),
        );
        god.insert_edge_simple(
            n(255),
            n(259),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(259),
            n(262),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(262),
            n(256),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(243),
            n(204),
            IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(1)),
            (),
        ); // To END

        // More from 234
        god.insert_edge_simple(n(234), n(240), IntermediateTrie3EdgeKey::NoOp, ());

        // More from 233
        god.insert_edge_simple(
            n(233),
            n(235),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(6)),
            (),
        );
        god.insert_edge_simple(
            n(235),
            n(239),
            IntermediateTrie3EdgeKey::Pop(2, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(233),
            n(236),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(8)),
            (),
        );
        god.insert_edge_simple(
            n(236),
            n(241),
            IntermediateTrie3EdgeKey::Pop(2, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(233),
            n(237),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(9)),
            (),
        );
        god.insert_edge_simple(
            n(233),
            n(238),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(10)),
            (),
        );
        god.insert_edge_simple(
            n(238),
            n(242),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(242),
            n(245),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(245),
            n(239),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );

        // More from 206
        god.insert_edge_simple(
            n(206),
            n(208),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(208),
            n(212),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(212),
            n(214),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(3)),
            (),
        );
        god.insert_edge_simple(
            n(214),
            n(207),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(0)),
            (),
        );
        god.insert_edge_simple(
            n(212),
            n(208),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(7)),
            (),
        ); // Cycle
        god.insert_edge_simple(
            n(206),
            n(209),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(6)),
            (),
        );
        god.insert_edge_simple(
            n(209),
            n(212),
            IntermediateTrie3EdgeKey::Pop(2, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(206),
            n(209),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(8)),
            (),
        );
        god.insert_edge_simple(
            n(206),
            n(208),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(9)),
            (),
        );
        god.insert_edge_simple(
            n(206),
            n(210),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(10)),
            (),
        );
        god.insert_edge_simple(
            n(210),
            n(213),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(213),
            n(208),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );

        // Path from 217
        god.insert_edge_simple(
            n(217),
            n(218),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(218),
            n(223),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(223),
            n(221),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(7)),
            (),
        );
        god.insert_edge_simple(
            n(221),
            n(225),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(225),
            n(228),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_iter([0, 3, 7])),
            (),
        );
        god.insert_edge_simple(
            n(228),
            n(221),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(7)),
            (),
        ); // Cycle
        god.insert_edge_simple(n(228), n(230), IntermediateTrie3EdgeKey::NoOp, ());
        god.insert_edge_simple(
            n(230),
            n(224),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(224),
            n(227),
            IntermediateTrie3EdgeKey::Push(StateIDBV::from_item(10)),
            (),
        );
        god.insert_edge_simple(
            n(227),
            n(233),
            IntermediateTrie3EdgeKey::CheckLLM(LLMTokenBV::from_item(0)),
            (),
        );
        god.insert_edge_simple(n(218), n(224), IntermediateTrie3EdgeKey::NoOp, ());
        god.insert_edge_simple(
            n(217),
            n(219),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(6)),
            (),
        );
        god.insert_edge_simple(
            n(219),
            n(223),
            IntermediateTrie3EdgeKey::Pop(2, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(217),
            n(220),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(8)),
            (),
        );
        god.insert_edge_simple(
            n(220),
            n(225),
            IntermediateTrie3EdgeKey::Pop(2, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(217),
            n(221),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(9)),
            (),
        );
        god.insert_edge_simple(
            n(217),
            n(222),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(10)),
            (),
        );
        god.insert_edge_simple(
            n(222),
            n(226),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );
        god.insert_edge_simple(
            n(226),
            n(229),
            IntermediateTrie3EdgeKey::Pop(0, StateIDBV::from_item(5)),
            (),
        );
        god.insert_edge_simple(
            n(229),
            n(223),
            IntermediateTrie3EdgeKey::Pop(1, StateIDBV::max_ones()),
            (),
        );

        let mut roots = BTreeMap::new();
        roots.insert(TokenizerStateID(0), n(202));

        let paths_before = get_normalized_paths(&roots, &god);

        eliminate_pushes_and_pops(&mut roots, &god);

        let paths_after = get_normalized_paths(&roots, &god);
        assert_eq!(paths_before, paths_after);

        assert_no_pops_reachable_from_pushes(&roots, &god);
    }
}
