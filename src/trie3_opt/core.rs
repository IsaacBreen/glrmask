use std::collections::{BTreeMap, BTreeSet, VecDeque};

/// Compact node id for the mini trie.
pub type NodeId = u32;

/// A compact sorted set of usize indices with deterministic ordering.
/// Backed by Vec<usize> kept sorted and deduplicated.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Default)]
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
}

impl Node {
    pub fn new(id: NodeId, end: bool) -> Self {
        Self {
            id,
            end,
            children: BTreeMap::new(),
        }
    }
    pub fn out_degree(&self) -> usize {
        self.children.values().map(|m| m.len()).sum()
    }
}

/// A compact, no-generics mini trie for precompute3 optimization.
#[derive(Clone, Debug)]
pub struct MiniTrie {
    pub nodes: Vec<Node>,
    pub root_ids: BTreeSet<NodeId>,
}

impl MiniTrie {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_ids: BTreeSet::new(),
        }
    }
    pub fn add_node(&mut self, end: bool) -> NodeId {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node::new(id, end));
        id
    }
    pub fn add_edge(
        &mut self,
        src: NodeId,
        key: EdgeKey,
        dst: NodeId,
        states: SortedSet,
    ) {
        let n = &mut self.nodes[src as usize];
        let dm = n.children.entry(key).or_insert_with(BTreeMap::new);
        dm.entry(dst)
            .and_modify(|e| e.union_inplace(&states))
            .or_insert(states);
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
            let node = &self.nodes[u as usize];
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
        let mut incoming: BTreeMap<NodeId, Vec<NodeId>> = BTreeMap::new();
        for n in &self.nodes {
            for (_ek, dm) in &n.children {
                for (dst, _s) in dm {
                    incoming.entry(*dst).or_default().push(n.id);
                }
            }
        }
        let mut productive: BTreeSet<NodeId> = BTreeSet::new();
        let mut q: VecDeque<NodeId> = VecDeque::new();
        for n in &self.nodes {
            if n.end {
                productive.insert(n.id);
                q.push_back(n.id);
            }
        }
        while let Some(v) = q.pop_front() {
            if let Some(srcs) = incoming.get(&v) {
                for &u in srcs {
                    if productive.insert(u) {
                        q.push_back(u);
                    }
                }
            }
        }
        productive
    }
}
