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
    const VERBOSE: bool = false;
    if VERBOSE {
        println!("Starting push/pop elimination...");
        let initial_nodes =
            IntermediatePrecomputeNode3::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>())
                .len();
        println!("Initial graph node count: {}", initial_nodes);
    }

    // 1. Convert to Intermediate2 format, which moves token constraints to edge values.
    let (roots2, god2) = convert_to_intermediate2(roots, god);
    if VERBOSE {
        let intermediate_nodes = Intermediate2PrecomputeNode3::all_nodes(
            &god2,
            &roots2.values().cloned().collect::<Vec<_>>(),
        )
        .len();
        println!("Intermediate graph node count: {}", intermediate_nodes);
    }

    // 2. Build adjacency and per-node metadata once; then use a worklist algorithm.
    type Node = Intermediate2PrecomputeNode3Index;
    type Key2 = Intermediate2Trie3EdgeKey;
    let all_nodes2 = Intermediate2PrecomputeNode3::all_nodes(
        &god2,
        &roots2.values().cloned().collect::<Vec<_>>(),
    );

    let is_push = |k: &Key2| matches!(k, Intermediate2Trie3EdgeKey::Push(_));

    struct GraphState<'a> {
        god2: &'a Intermediate2Trie3GodWrapper,
        fwd: HashMap<Node, HashMap<Key2, HashMap<Node, LLMTokenBV>>>,
        rev: HashMap<Node, HashMap<Key2, HashMap<Node, LLMTokenBV>>>,
        out_push_count: HashMap<Node, usize>,
        out_nonpush_count: HashMap<Node, usize>,
        in_push_count: HashMap<Node, usize>,
        is_end: HashMap<Node, bool>,
        split_done: HashSet<Node>,
    }

    impl<'a> GraphState<'a> {
        fn new(god2: &'a Intermediate2Trie3GodWrapper, all_nodes2: &[Node]) -> Self {
            let mut fwd = HashMap::new();
            let mut rev = HashMap::new();
            let mut out_push_count = HashMap::new();
            let mut out_nonpush_count = HashMap::new();
            let mut in_push_count = HashMap::new();
            let mut is_end = HashMap::new();

            for &u in all_nodes2 {
                if let Some(uguard) = u.read(god2) {
                    is_end.insert(u, uguard.value.end);
                    for (key, dest_map) in uguard.children() {
                        for (&v, val) in dest_map {
                            fwd.entry(u)
                                .or_default()
                                .entry(key.clone())
                                .or_default()
                                .insert(v, val.clone());
                            rev.entry(v)
                                .or_default()
                                .entry(key.clone())
                                .or_default()
                                .insert(u, val.clone());
                            if is_push(key) {
                                *out_push_count.entry(u).or_default() += 1;
                                *in_push_count.entry(v).or_default() += 1;
                            } else {
                                *out_nonpush_count.entry(u).or_default() += 1;
                            }
                        }
                    }
                }
            }
            Self {
                god2,
                fwd,
                rev,
                out_push_count,
                out_nonpush_count,
                in_push_count,
                is_end,
                split_done: HashSet::new(),
            }
        }

        fn remove_edge(&mut self, u: Node, v: Node, key: &Key2) {
            if let Some(by_key) = self.fwd.get_mut(&u) {
                if let Some(dest_map) = by_key.get_mut(key) {
                    if dest_map.remove(&v).is_some() {
                        if is_push(key) {
                            if let Some(c) = self.out_push_count.get_mut(&u) {
                                *c = c.saturating_sub(1);
                            }
                            if let Some(by_k) = self.rev.get_mut(&v) {
                                if let Some(src_map) = by_k.get_mut(key) {
                                    if src_map.remove(&u).is_some() {
                                        if let Some(c) = self.in_push_count.get_mut(&v) {
                                            *c = c.saturating_sub(1);
                                        }
                                    }
                                    if src_map.is_empty() {
                                        by_k.remove(key);
                                    }
                                }
                            }
                        } else {
                            if let Some(c) = self.out_nonpush_count.get_mut(&u) {
                                *c = c.saturating_sub(1);
                            }
                            if let Some(by_k) = self.rev.get_mut(&v) {
                                if let Some(src_map) = by_k.get_mut(key) {
                                    src_map.remove(&u);
                                    if src_map.is_empty() {
                                        by_k.remove(key);
                                    }
                                }
                            }
                        }
                    }
                    if dest_map.is_empty() {
                        by_key.remove(key);
                    }
                }
                if by_key.is_empty() {
                    self.fwd.remove(&u);
                }
            }
            self.god2.remove_edge(u, v, key);
        }

        fn insert_edge(&mut self, u: Node, v: Node, key: Key2, val: LLMTokenBV) {
            if val.is_empty() {
                return;
            }
            let entry_by_key = self.fwd.entry(u).or_default();
            let entry_dest = entry_by_key.entry(key.clone()).or_default();
            let existed_before = entry_dest.contains_key(&v);
            if let Some(prev) = entry_dest.get_mut(&v) {
                *prev = prev.clone() | val.clone();
            } else {
                entry_dest.insert(v, val.clone());
                if is_push(&key) {
                    *self.out_push_count.entry(u).or_default() += 1;
                } else {
                    *self.out_nonpush_count.entry(u).or_default() += 1;
                }
            }

            let rev_by_key = self.rev.entry(v).or_default();
            let rev_src = rev_by_key.entry(key.clone()).or_default();
            if let Some(prev) = rev_src.get_mut(&u) {
                *prev = prev.clone() | val.clone();
            } else {
                rev_src.insert(u, val.clone());
                if is_push(&key) && !existed_before {
                    *self.in_push_count.entry(v).or_default() += 1;
                }
            }

            self.god2.insert_edge_simple(u, v, key, val);
        }

        fn split_node(&mut self, b_idx: Node) -> Option<Node> {
            if *self.is_end.get(&b_idx).unwrap_or(&false) {
                return None;
            }
            if self.split_done.contains(&b_idx) {
                return None;
            }
            let has_out_push = *self.out_push_count.get(&b_idx).unwrap_or(&0) > 0;
            let has_out_nonpush = *self.out_nonpush_count.get(&b_idx).unwrap_or(&0) > 0;
            if !(has_out_push && has_out_nonpush) {
                return None;
            }

            let b_value = b_idx.read(self.god2).unwrap().value.clone();
            let b_np_idx = Intermediate2PrecomputeNode3Index::new(
                self.god2.insert(Intermediate2PrecomputeNode3::new(b_value)),
            );
            self.is_end.insert(b_np_idx, false);

            let mut to_move: Vec<(Key2, Node, LLMTokenBV)> = Vec::new();
            if let Some(by_key) = self.fwd.get(&b_idx) {
                for (edge_key, dest_map) in by_key {
                    if !is_push(edge_key) {
                        for (&c_idx, val) in dest_map {
                            to_move.push((edge_key.clone(), c_idx, val.clone()));
                        }
                    }
                }
            }
            for (k, c_idx, val) in &to_move {
                self.insert_edge(b_np_idx, *c_idx, k.clone(), val.clone());
            }
            for (k, c_idx, _) in &to_move {
                self.remove_edge(b_idx, *c_idx, k);
            }

            if let Some(by_key) = self.rev.get(&b_idx).cloned() {
                for (k, src_map) in by_key {
                    for (a_idx, val) in src_map {
                        self.insert_edge(a_idx, b_np_idx, k.clone(), val.clone());
                    }
                }
            }

            self.split_done.insert(b_idx);
            Some(b_np_idx)
        }
    }

    let mut state = GraphState::new(&god2, &all_nodes2);
    // Initial batch-split: nodes that already have incoming Push and both Push and non-Push outgoing.
    {
        let mut initial_split_targets: Vec<Node> = Vec::new();
        for &b in &all_nodes2 {
            if *state.is_end.get(&b).unwrap_or(&false) {
                continue;
            }
            let in_push = *state.in_push_count.get(&b).unwrap_or(&0) > 0;
            let out_push = *state.out_push_count.get(&b).unwrap_or(&0) > 0;
            let out_nonpush = *state.out_nonpush_count.get(&b).unwrap_or(&0) > 0;
            if in_push && out_push && out_nonpush {
                initial_split_targets.push(b);
            }
        }
        for b in initial_split_targets {
            let _ = state.split_node(b);
        }
    }

    // Worklist of standard candidates: incoming Push and no outgoing Push.
    let mut queue: std::collections::VecDeque<Node> = std::collections::VecDeque::new();
    for (&node, &end_flag) in &state.is_end {
        if !end_flag
            && *state.in_push_count.get(&node).unwrap_or(&0) > 0
            && *state.out_push_count.get(&node).unwrap_or(&0) == 0
        {
            queue.push_back(node);
        }
    }

    // Main worklist loop
    while let Some(b_idx) = queue.pop_front() {
        if *state.is_end.get(&b_idx).unwrap_or(&false) {
            continue;
        }
        if *state.out_push_count.get(&b_idx).unwrap_or(&0) > 0 {
            continue; // Not a standard candidate anymore
        }

        // Gather incoming Push edges into B
        let mut incoming_pushes: Vec<(Node, Key2, LLMTokenBV)> = Vec::new();
        if let Some(by_key) = state.rev.get(&b_idx) {
            for (k, src_map) in by_key {
                if matches!(k, Intermediate2Trie3EdgeKey::Push(_)) {
                    for (&a_idx, val) in src_map {
                        incoming_pushes.push((a_idx, k.clone(), val.clone()));
                    }
                }
            }
        }
        if incoming_pushes.is_empty() {
            continue;
        }

        // Gather all outgoing edges from B (ignore Push edges)
        let mut outgoing_edges: Vec<(Key2, Node, LLMTokenBV)> = Vec::new();
        if let Some(by_key) = state.fwd.get(&b_idx) {
            for (op_key, dest_map) in by_key {
                if matches!(op_key, Intermediate2Trie3EdgeKey::Push(_)) {
                    continue;
                }
                for (&c_idx, val) in dest_map {
                    outgoing_edges.push((op_key.clone(), c_idx, val.clone()));
                }
            }
        }

        // Create all shortcut edges from A to C
        let mut new_push_targets: HashSet<Node> = HashSet::new();
        for (a_idx, push_key, tokens_a_b) in &incoming_pushes {
            let s = match push_key {
                Intermediate2Trie3EdgeKey::Push(s) => s,
                _ => unreachable!(),
            };
            for (op_key, c_idx, tokens_b_c) in &outgoing_edges {
                let new_tokens = tokens_a_b & tokens_b_c;
                if new_tokens.is_empty() {
                    continue;
                }
                let new_key_opt = match op_key {
                    Intermediate2Trie3EdgeKey::Pop(0, s_prime) => {
                        (!s.is_disjoint(s_prime))
                            .then_some(Intermediate2Trie3EdgeKey::Push(s & s_prime))
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
                    Intermediate2Trie3EdgeKey::Push(_) => {
                        unreachable!("Standard candidate should not have outgoing pushes")
                    }
                };
                if let Some(new_key) = new_key_opt {
                    if matches!(new_key, Intermediate2Trie3EdgeKey::Push(_)) {
                        new_push_targets.insert(*c_idx);
                    }
                    state.insert_edge(*a_idx, *c_idx, new_key, new_tokens);
                }
            }
        }

        // Dynamic follow-ups for any node that just received a Push:
        for c in new_push_targets {
            if *state.is_end.get(&c).unwrap_or(&false) {
                continue;
            }
            let out_push = *state.out_push_count.get(&c).unwrap_or(&0);
            let out_nonpush = *state.out_nonpush_count.get(&c).unwrap_or(&0);
            let in_push = *state.in_push_count.get(&c).unwrap_or(&0);
            if out_push == 0 && in_push > 0 {
                queue.push_back(c);
            } else if in_push > 0 && out_push > 0 && out_nonpush > 0 {
                if let Some(c_np) = state.split_node(c) {
                    // The split duplicate includes the new incoming push; b_np has no outgoing pushes
                    if *state.in_push_count.get(&c_np).unwrap_or(&0) > 0
                        && *state.out_push_count.get(&c_np).unwrap_or(&0) == 0
                    {
                        queue.push_back(c_np);
                    }
                }
            }
        }

        // Remove all incoming push edges into B
        for (a_idx, push_key, _) in incoming_pushes {
            state.remove_edge(a_idx, b_idx, &push_key);
        }
    }

    // 3. Convert back to the original Trie format.
    let (new_roots1_map, new_god1) = convert_from_intermediate2(&roots2, &god2);
    if VERBOSE {
        println!(
            "Finished elimination. Final intermediate graph has {} nodes.",
            IntermediatePrecomputeNode3::all_nodes(
                &new_god1,
                &new_roots1_map
                    .values()
                    .cloned()
                    .collect::<Vec<_>>()
            )
            .len()
        );
    }

    // 4. The function signature requires modifying `god` in place.
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
    if VERBOSE {
        println!(
            "Final graph node count: {}",
            IntermediatePrecomputeNode3::all_nodes(god, &roots.values().cloned().collect::<Vec<_>>())
                .len()
        );
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