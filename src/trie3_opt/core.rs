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
    pub id: NodeId,
    pub end: bool,
    // key: (pop, tokens) -> dest map: dest node -> state-set
    pub children: BTreeMap<EdgeKey, BTreeMap<NodeId, SortedSet>>,
    // src map: src node -> (key: (pop, tokens) -> state-set)
    pub parents: BTreeMap<NodeId, BTreeMap<EdgeKey, SortedSet>>,
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
    pub nodes: BTreeMap<NodeId, Node>,
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
        for n in self.nodes.values() {
            if n.end {
                productive.insert(n.id);
                q.push_back(n.id);
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
}
