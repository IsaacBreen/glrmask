use std::collections::{BTreeMap, BTreeSet, VecDeque, HashMap};

/// Compact node id for the mini trie.
pub type NodeId = u32;

/// A compact sorted set of usize indices with deterministic ordering.
/// Backed by Vec<usize> kept sorted and deduplicated.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Default, Hash)]
pub struct SortedSet {
    pub elems: Vec<usize>,
}

impl SortedSet {
    pub fn new() -> Self {
        Self { elems: Vec::new() }
    }
    pub fn from_iter<I: IntoIterator<Item = usize>>(it: I) -> Self {
        let mut v: Vec<usize> = it.into_iter().collect();
        v.sort_unstable();
        v.dedup();
        Self { elems: v }
    }
    pub fn insert(&mut self, x: usize) {
        match self.elems.binary_search(&x) {
            Ok(_) => {}
            Err(pos) => self.elems.insert(pos, x),
        }
    }
    pub fn union_inplace(&mut self, other: &SortedSet) {
        if other.elems.is_empty() {
            return;
        }
        let mut out = Vec::with_capacity(self.elems.len() + other.elems.len());
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.elems.len() && j < other.elems.len() {
            let a = self.elems[i];
            let b = other.elems[j];
            if a < b {
                out.push(a);
                i += 1;
            } else if a > b {
                out.push(b);
                j += 1;
            } else {
                out.push(a);
                i += 1;
                j += 1;
            }
        }
        while i < self.elems.len() {
            out.push(self.elems[i]);
            i += 1;
        }
        while j < other.elems.len() {
            out.push(other.elems[j]);
            j += 1;
        }
        self.elems = out;
    }
    pub fn intersect(&self, other: &SortedSet) -> SortedSet {
        let mut out = Vec::new();
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.elems.len() && j < other.elems.len() {
            let a = self.elems[i];
            let b = other.elems[j];
            if a < b {
                i += 1;
            } else if a > b {
                j += 1;
            } else {
                out.push(a);
                i += 1;
                j += 1;
            }
        }
        SortedSet { elems: out }
    }
    pub fn difference(&self, other: &SortedSet) -> SortedSet {
        let mut out = Vec::new();
        let mut i = 0;
        let mut j = 0;
        while i < self.elems.len() && j < other.elems.len() {
            if self.elems[i] < other.elems[j] {
                out.push(self.elems[i]);
                i += 1;
            } else if self.elems[i] > other.elems[j] {
                j += 1;
            } else {
                i += 1;
                j += 1;
            }
        }
        while i < self.elems.len() {
            out.push(self.elems[i]);
            i += 1;
        }
        SortedSet { elems: out }
    }
    /// Returns true if self and other share at least one common element.
    /// This is a fast, allocation-free check using a merged scan on the sorted vectors.
    pub fn intersects(&self, other: &SortedSet) -> bool {
        let mut i = 0usize;
        let mut j = 0usize;
        while i < self.elems.len() && j < other.elems.len() {
            let a = self.elems[i];
            let b = other.elems[j];
            if a < b {
                i += 1;
            } else if a > b {
                j += 1;
            } else {
                return true;
            }
        }
        false
    }
    pub fn is_empty(&self) -> bool {
        self.elems.is_empty()
    }
    pub fn len(&self) -> usize {
        self.elems.len()
    }
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.elems.iter().cloned()
    }
}

/// Edge key for the mini trie: pop delta and a token set.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
#[derive(Hash)]
pub struct EdgeKey {
    pub pop: isize,
    pub tokens: SortedSet,
}

impl EdgeKey {
    pub fn new(pop: isize, tokens: SortedSet) -> Self {
        Self { pop, tokens }
    }
}

/// A node in the mini trie.
#[derive(Clone, Debug)]
pub struct Node {
    id: NodeId,
    end: bool,
    // key: (pop, tokens) -> dest map: dest node -> state-set
    children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>>,
    // src map: src node -> (key: (pop, tokens) -> state-set)
    parents: BTreeMap<NodeId, BTreeMap<EdgeKey, SortedSet>>,
}

impl Node {
    pub fn new(id: NodeId, end: bool) -> Self {
        Self {
            id,
            end,
            children: BTreeMap::new(),
            parents: BTreeMap::new(),
        }
    }
    pub fn id(&self) -> NodeId {
        self.id
    }
    pub fn is_end(&self) -> bool {
        self.end
    }
    pub fn children(&self) -> &BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>> {
        &self.children
    }
    pub fn parents(&self) -> &BTreeMap<NodeId, BTreeMap<EdgeKey, SortedSet>> {
        &self.parents
    }
    pub fn out_degree(&self) -> usize {
        self.children.values().map(|m| m.len()).sum()
    }
    pub fn in_degree(&self) -> usize {
        self.parents.values().map(|m| m.len()).sum()
    }
}

/// A compact, no-generics mini trie for precompute3 optimization.
#[derive(Clone, Debug)]
pub struct MiniTrie {
    nodes: BTreeMap<NodeId, Node>,
    pub root_ids: BTreeSet<NodeId>,
    /// Counter to generate unique node IDs.
    next_node_id: NodeId,
}

impl MiniTrie {
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            root_ids: BTreeSet::new(),
            next_node_id: 0,
        }
    }
    pub fn add_node(&mut self, end: bool) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.insert(id, Node::new(id, end));
        id
    }
    pub fn add_edge(
        &mut self,
        src: NodeId,
        key: EdgeKey,
        dst: NodeId,
        states: SortedSet,
    ) {
        // Update children of src node
        if let Some(src_node) = self.nodes.get_mut(&src) {
            let dm = src_node.children.entry(key.clone()).or_default();
            dm.entry(dst)
                .and_modify(|e| e.union_inplace(&states))
                .or_insert(states.clone());
        }

        // Update parents of dst node
        if let Some(dst_node) = self.nodes.get_mut(&dst) {
            let parent_edges = dst_node.parents.entry(src).or_default();
            parent_edges.entry(key)
                .and_modify(|e| e.union_inplace(&states))
                .or_insert(states);
        }
    }
    pub fn add_root(&mut self, id: NodeId) {
        self.root_ids.insert(id);
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn set_end(&mut self, node_id: NodeId, is_end: bool) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            node.end = is_end;
        }
    }

    /// Removes a specific destination from an edge. Returns the state set of the removed edge destination.
    pub fn remove_edge_dest(&mut self, src: NodeId, key: &EdgeKey, dst: NodeId) -> Option<SortedSet> {
        let removed_sids;
        if let Some(src_node) = self.nodes.get_mut(&src) {
            if let Some(dm) = src_node.children.get_mut(key) {
                removed_sids = dm.remove(&dst);
                if dm.is_empty() {
                    src_node.children.remove(key);
                }
            } else {
                removed_sids = None;
            }
        } else {
            removed_sids = None;
        }

        if removed_sids.is_some() {
            if let Some(dst_node) = self.nodes.get_mut(&dst) {
                if let Some(parent_edges) = dst_node.parents.get_mut(&src) {
                    parent_edges.remove(key);
                    if parent_edges.is_empty() {
                        dst_node.parents.remove(&src);
                    }
                }
            }
        }
        removed_sids
    }

    /// Removes all outgoing edges from a node.
    pub fn clear_children(&mut self, node_id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            let old_children = std::mem::take(&mut node.children);
            for (ek, dm) in old_children {
                for (dst, _sids) in dm {
                    if let Some(dst_node) = self.nodes.get_mut(&dst) {
                        if let Some(parent_edges) = dst_node.parents.get_mut(&node_id) {
                            parent_edges.remove(&ek);
                            if parent_edges.is_empty() {
                                dst_node.parents.remove(&node_id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Replaces all outgoing edges for a node.
    pub fn set_children(&mut self, node_id: NodeId, new_children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>>) {
        self.clear_children(node_id);
        for (ek, dm) in new_children {
            for (dst, sids) in dm {
                self.add_edge(node_id, ek.clone(), dst, sids);
            }
        }
    }

    /// Compute set of nodes reachable from any root.
    pub fn reachable_from_roots(&self) -> BTreeSet<NodeId> {
        let mut seen: BTreeSet<NodeId> = BTreeSet::new();
        let mut q: VecDeque<NodeId> = self.root_ids.iter().cloned().collect();
        while let Some(u) = q.pop_front() {
            if !seen.insert(u) {
                continue;
            }
            let node = self.nodes.get(&u).unwrap();
            for (_ek, dm) in node.children.iter() {
                for (v, _s) in dm.iter() {
                    if !seen.contains(v) {
                        q.push_back(*v);
                    }
                }
            }
        }
        seen
    }

    /// Compute set of nodes that can reach an end node (reverse reachability).
    pub fn can_reach_end(&self) -> BTreeSet<NodeId> {
        let mut productive: BTreeSet<NodeId> = BTreeSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        for n in self.nodes() {
            if n.is_end() {
                productive.insert(n.id());
                q.push_back(n.id());
            }
        }
        while let Some(v_id) = q.pop_front() {
            if let Some(v_node) = self.nodes.get(&v_id) {
                for &u_id in v_node.parents.keys() {
                    if productive.insert(u_id) {
                        q.push_back(u_id);
                    }
                }
            }
        }
        productive
    }

    /// Removes nodes not reachable from any root.
    pub fn gc(&mut self) {
        let live_nodes = self.reachable_from_roots();

        if live_nodes.len() == self.nodes.len() {
            return;
        }

        // Remove dead nodes from the main nodes map.
        self.nodes.retain(|id, _| live_nodes.contains(id));

        // Clean up edges in remaining live nodes to not point to dead nodes.
        for node in self.nodes.values_mut() {
            // Clean children: remove destinations that are not live.
            for (_ek, dm) in node.children.iter_mut() {
                dm.retain(|dst, _| live_nodes.contains(dst));
            }
            node.children.retain(|_ek, dm| !dm.is_empty());

            // Clean parents: remove parents that are not live.
            node.parents.retain(|parent_id, _| live_nodes.contains(parent_id));
        }
    }

    /// Compute the set of all LLM tokens on outgoing edges for each node.
    pub fn live_llm_tokens(&self) -> HashMap<NodeId, SortedSet> {
        let mut live_tokens: HashMap<NodeId, SortedSet> = HashMap::new();
        for node in self.nodes() {
            let mut node_live_tokens = SortedSet::new();
            for (ek, _) in node.children() {
                node_live_tokens.union_inplace(&ek.tokens);
            }
            live_tokens.insert(node.id(), node_live_tokens);
        }
        live_tokens
    }

    /// Compute set of LLM tokens at each node that can start a path to an END node.
    pub fn productive_llm_tokens(&self) -> HashMap<NodeId, SortedSet> {
        let mut productive_tokens: HashMap<NodeId, SortedSet> =
            self.node_ids().map(|id| (id, SortedSet::new())).collect();
        let mut productive_nodes: BTreeSet<NodeId> = BTreeSet::new();
        let mut worklist: VecDeque<NodeId> = VecDeque::new();

        for node in self.nodes() {
            if node.is_end() {
                worklist.push_back(node.id());
                productive_nodes.insert(node.id());
            }
        }

        while let Some(v_id) = worklist.pop_front() {
            if let Some(v_node) = self.get_node(v_id) {
                for (u_id, edges_from_u) in v_node.parents() {
                    let u_prod = productive_tokens.get_mut(u_id).unwrap();
                    let old_len = u_prod.len();

                    for (ek, _sids) in edges_from_u {
                        u_prod.union_inplace(&ek.tokens);
                    }

                    if u_prod.len() > old_len {
                        // If u's productive set grew, u is now productive.
                        // If it wasn't productive before, we need to propagate this.
                        if productive_nodes.insert(*u_id) {
                            worklist.push_back(*u_id);
                        }
                    }
                }
            }
        }

        productive_tokens
    }
}
