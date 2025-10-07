use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt::{Debug, Write};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bimap::BiBTreeMap;
use crate::constraint::StateIDBV;
use crate::constraint::{LLMTokenBV, TerminalBV};
use crate::datastructures::hybrid_bitset::HybridBitset;
use crate::datastructures::hybrid_l2_bitset::HybridL2Bitset;
use crate::datastructures::leveled_gss::{LeveledGSS, Merge as LGMerge};
use crate::datastructures::trie::Trie2Index;
use crate::glr::parser::{GLRParser, ParseStateEdgeContent};
use crate::glr::table::{StateID, TerminalID};
use crate::tokenizer::LLMTokenID;

// Adapter aliases for precompute-trie types (referencing constraint.rs)
pub type StoredPrecomputeNodeIndex = crate::constraint::PrecomputeNode3Index;
pub type StoredTrieGodWrapper = crate::constraint::Trie3GodWrapper;

// --- Acc type compatible with LeveledGSS A ---
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Acc {
    pub llm_tokens_union: HybridBitset,
    pub terminals_union: HybridL2Bitset,
    pub stored_trie_nodes: BTreeSet<StoredPrecomputeNodeIndex>,
}

impl Acc {
    pub fn new_fresh() -> Self {
        Self {
            llm_tokens_union: HybridBitset::max_ones(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }
    pub fn new_dead() -> Self {
        Self {
            llm_tokens_union: HybridBitset::zeros(),
            terminals_union: HybridL2Bitset::all(),
            stored_trie_nodes: BTreeSet::new(),
        }
    }
    pub fn is_default(&self) -> bool {
        self.llm_tokens_union == HybridBitset::max_ones()
            && self.terminals_union == HybridL2Bitset::all()
            && self.stored_trie_nodes.is_empty()
    }
    pub fn union_llm_tokens(&self) -> HybridBitset {
        self.llm_tokens_union.clone()
    }

    pub fn stored_trie_nodes(&self) -> &BTreeSet<StoredPrecomputeNodeIndex> {
        &self.stored_trie_nodes
    }

    pub fn stored_trie_nodes_mut(&mut self) -> &mut BTreeSet<StoredPrecomputeNodeIndex> {
        &mut self.stored_trie_nodes
    }
}

impl LGMerge for Acc {
    fn merge(&self, other: &Self) -> Self {
        Acc {
            llm_tokens_union: &self.llm_tokens_union | &other.llm_tokens_union,
            terminals_union: &self.terminals_union | &other.terminals_union,
            stored_trie_nodes: &self.stored_trie_nodes | &other.stored_trie_nodes,
        }
    }
}

// --- Minimal GSSStats for logging/debug ---
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GSSStats {
    pub(crate) num_roots: usize,
    pub(crate) num_root_predecessors: usize,
    pub(crate) num_unique_root_predecessor_keys: usize,
    pub(crate) total_edges: usize,
    pub(crate) unique_nodes: usize,
    pub(crate) num_leaves: usize,
    pub(crate) structurally_unique_nodes: usize,
    pub(crate) structural_redundancy: f64,
    pub(crate) num_redundant_nodes: usize,
    pub(crate) max_depth: usize,
    pub(crate) average_depth: f64,
    pub(crate) merge_points: usize,
    pub(crate) max_predecessors_with_values: usize,
    pub(crate) average_predecessors_with_values: f64,
}

pub fn gather_gss_stats(roots: &[&GSSNode]) -> GSSStats {
    let mut stats = GSSStats::default();
    stats.num_roots = roots.len();
    let mut total_depth: usize = 0;
    let mut paths: usize = 0;
    let mut max_depth = 0;
    let mut total_edges = 0;
    let mut num_root_predecessors = 0usize;
    let mut unique_root_keys = BTreeSet::new();
    for r in roots {
        let st = r.inner.to_stacks();
        if let Some(peek) = r.inner.peek().into_iter().next() {
            unique_root_keys.insert(peek);
        }
        num_root_predecessors += r.inner.peek().len();
        for (p, _a) in st.iter() {
            let d = p.len();
            total_depth += d;
            total_edges += d;
            max_depth = max_depth.max(d);
            paths += 1;
        }
    }
    stats.max_depth = max_depth;
    stats.total_edges = total_edges;
    stats.unique_nodes = paths;
    stats.num_leaves = paths;
    stats.num_root_predecessors = num_root_predecessors;
    stats.num_unique_root_predecessor_keys = unique_root_keys.len();
    if paths > 0 {
        stats.average_depth = total_depth as f64 / paths as f64;
        stats.average_predecessors_with_values = (num_root_predecessors as f64) / (roots.len().max(1) as f64);
    }
    stats
}

// --- GSS printer config & helpers ---
pub struct GSSPrintConfig<'a> {
    pub(crate) labels: Option<&'a [String]>,
    pub(crate) max_edges: usize,
    pub(crate) original_internal_bimap: Option<&'a BTreeMap<usize, usize>>,
    pub(crate) llm_token_map: Option<&'a BiBTreeMap<Vec<u8>, LLMTokenID>>,
    pub(crate) verbose: bool,
}
impl<'a> Default for GSSPrintConfig<'a> {
    fn default() -> Self {
        Self {
            labels: None,
            max_edges: usize::MAX,
            original_internal_bimap: None,
            llm_token_map: None,
            verbose: false,
        }
    }
}

pub fn print_gss_forest(
    roots: &[Arc<GSSNode>],
    _terminal_map: &BiBTreeMap<crate::glr::grammar::Terminal, TerminalID>,
    _config: &GSSPrintConfig,
) -> (String, Vec<StateID>) {
    let mut out = String::new();
    let mut state_ids_in_order = Vec::new();
    if roots.is_empty() {
        return ("GSS Forest: (No roots)".to_string(), state_ids_in_order);
    }
    writeln!(&mut out, "GSS Forest (leveled adapter):").unwrap();
    for (i, r) in roots.iter().enumerate() {
        writeln!(&mut out, "Root {}:", i).unwrap();
        let stacks = r.inner.to_stacks();
        for (path, acc) in stacks {
            let mut sids: Vec<_> = path.iter().map(|e| e.state_id).collect();
            for sid in &sids {
                if !state_ids_in_order.contains(sid) {
                    state_ids_in_order.push(*sid);
                }
            }
            let s: Vec<_> = sids.iter().map(|s| s.0.to_string()).collect();
            writeln!(
                &mut out,
                "  - [{}], tokens={}, trie_nodes={}",
                s.join(" "),
                acc.llm_tokens_union.len(),
                acc.stored_trie_nodes.len()
            )
            .unwrap();
        }
    }
    (out, state_ids_in_order)
}

pub fn find_longest_path(root: &Arc<GSSNode>) -> Option<Vec<(ParseStateEdgeContent, Arc<GSSNode>)>> {
    let mut best: Option<Vec<ParseStateEdgeContent>> = None;
    for (p, _a) in root.inner.to_stacks() {
        if best.as_ref().map_or(true, |b| p.len() > b.len()) {
            best = Some(p);
        }
    }
    best.map(|p| p.into_iter().map(|e| (e, root.clone())).collect())
}

pub fn sample_path<'a>(roots: &[&'a GSSNode], _seed: u64) -> Option<Vec<ParseStateEdgeContent>> {
    roots.get(0).map(|r| {
        r.inner
            .to_stacks()
            .get(0)
            .map(|(p, _)| p.clone())
            .unwrap_or_default()
    })
}

// --- GSS wrapper ---
#[derive(Clone)]
pub struct GSSNode {
    pub(crate) inner: LeveledGSS<ParseStateEdgeContent, Acc>,
}

impl Debug for GSSNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GSSNode(len_paths={})", self.inner.to_stacks().len())
    }
}

impl PartialEq for GSSNode {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
impl Eq for GSSNode {}
impl PartialOrd for GSSNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.inner.partial_cmp(&other.inner)
    }
}
impl Ord for GSSNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.cmp(&other.inner)
    }
}
impl Hash for GSSNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl GSSNode {
    pub fn new(acc: Acc) -> Self {
        GSSNode {
            inner: LeveledGSS::from_stacks(&vec![(vec![], acc)]),
        }
    }
    pub fn new_fresh() -> Self {
        GSSNode {
            inner: LeveledGSS::from_stacks(&vec![(vec![], Acc::new_fresh())]),
        }
    }
    pub fn new_dead() -> Self {
        // Represent dead as empty; allowed_llm_tokens() on empty returns zeros.
        GSSNode {
            inner: LeveledGSS::empty(),
        }
    }

    // Helper: append a value to all stacks; if empty, create singleton stack.
    fn push_all(
        inner: &LeveledGSS<ParseStateEdgeContent, Acc>,
        edge: ParseStateEdgeContent,
    ) -> LeveledGSS<ParseStateEdgeContent, Acc> {
        if inner.is_empty() {
            LeveledGSS::from_stacks(&[(vec![edge], Acc::new_fresh())])
        } else {
            inner.push(edge)
        }
    }

    pub fn push(&self, edge_value: ParseStateEdgeContent) -> Self {
        GSSNode {
            inner: Self::push_all(&self.inner, edge_value),
        }
    }
    pub fn push_many(&self, edge_values: Vec<ParseStateEdgeContent>) -> Self {
        if edge_values.is_empty() {
            return self.clone();
        }
        let mut merged = LeveledGSS::empty();
        for e in edge_values {
            let next = Self::push_all(&self.inner, e);
            merged = merged.merge(&next);
        }
        GSSNode { inner: merged }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty() || self.inner.max_depth() == 0
    }
    pub fn is_alive(&self) -> bool {
        !self.is_empty() && !self.allowed_llm_tokens().is_empty()
    }
    pub fn is_ok(&self) -> bool {
        self.is_alive()
    }
    pub fn allowed_llm_tokens(&self) -> LLMTokenBV {
        self.inner
            .reduce_acc()
            .map_or(HybridBitset::zeros(), |acc| acc.llm_tokens_union)
    }
    pub fn disallowed_terminals(&self) -> HybridL2Bitset {
        self.inner
            .reduce_acc()
            .map(|acc| acc.terminals_union.complement())
            .unwrap_or_else(|| HybridL2Bitset::all().complement())
    }
    pub fn max_depth(&self) -> usize {
        self.inner
            .to_stacks()
            .into_iter()
            .map(|(p, _)| p.len())
            .max()
            .unwrap_or(0)
    }
    pub fn flatten(&self) -> Vec<(Vec<ParseStateEdgeContent>, Acc)> {
        self.inner.to_stacks()
    }

    pub fn print(&self) -> String {
        let (s, _sid) = print_gss_forest(&[Arc::new(self.clone())], &BiBTreeMap::new(), &GSSPrintConfig::default());
        s
    }

    pub fn merge_with_depth(&mut self, _merge_depth: usize, other: &Self) {
        self.inner = self.inner.merge(&other.inner);
    }
    pub fn merged(mut self, other: Self, depth: usize) -> Self {
        self.merge_with_depth(depth, &other);
        self
    }

    pub fn merge_many_with_depth(merge_depth: usize, nodes: impl IntoIterator<Item = Arc<GSSNode>>) -> Arc<GSSNode> {
        let mut it = nodes.into_iter();
        if let Some(first) = it.next() {
            let mut acc = (*first).clone();
            for n in it {
                acc.merge_with_depth(merge_depth, &n);
            }
            Arc::new(acc)
        } else {
            Arc::new(GSSNode::new_dead())
        }
    }

    pub fn predecessors(&self) -> BTreeMap<ParseStateEdgeContent, BTreeMap<isize, Vec<Arc<GSSNode>>>> {
        self.inner
            .predecessors()
            .into_iter()
            .map(|(edge_val, preds_by_depth)| {
                let new_preds_by_depth = preds_by_depth
                    .into_iter()
                    .map(|(depth, gss_vec)| {
                        let new_gss_vec = gss_vec.into_iter().map(|gss| Arc::new(GSSNode { inner: gss })).collect();
                        (depth, new_gss_vec)
                    })
                    .collect();
                (edge_val, new_preds_by_depth)
            })
            .collect()
    }

    pub fn num_predecessors(&self) -> usize {
        self.inner.peek().len()
    }

    pub fn popn(&self, n: usize) -> GSSPopper {
        let mut popper = GSSPopper {
            node: Arc::new(GSSNode { inner: self.inner.clone() }),
            below_bottom: BTreeMap::new(),
        };
        popper.popn(n);
        popper
    }

    pub(crate) fn peek_iter(parent_arc: &Arc<GSSNode>) -> impl Iterator<Item = GSSPeek<'_>> {
        let keys: Vec<_> = parent_arc.inner.peek().into_iter().collect();
        GSSPeekIter {
            parent: parent_arc,
            keys,
            idx: 0,
        }
    }

    pub fn fuse_predecessors(&mut self, levels: usize) {
        if levels == 0 {
            return;
        }
        let self_arc = Arc::new(self.clone());
        let mut memo = HashMap::new();
        let fused_arc = fuse_predecessors_recursive(&self_arc, levels, &mut memo);
        if !Arc::ptr_eq(&self_arc, &fused_arc) {
            *self = (*fused_arc).clone();
        }
    }

    pub fn fuse_predecessors_recursive(&self, levels: usize, _memo: &mut PruneAndTransformRecursiveMemo) -> Arc<GSSNode> {
        let self_arc = Arc::new(self.clone());
        let mut dummy_memo = HashMap::new();
        fuse_predecessors_recursive(&self_arc, levels, &mut dummy_memo)
    }
}

struct GSSPeekIter<'a> {
    parent: &'a Arc<GSSNode>,
    keys: Vec<ParseStateEdgeContent>,
    idx: usize,
}
impl<'a> Iterator for GSSPeekIter<'a> {
    type Item = GSSPeek<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.keys.len() {
            None
        } else {
            let ev = self.keys[self.idx].clone();
            self.idx += 1;
            Some(GSSPeek {
                parent_arc: self.parent,
                edge_value: ev,
            })
        }
    }
}

// --- GSSPeek & related ---
pub struct GSSPeek<'a> {
    parent_arc: &'a Arc<GSSNode>,
    edge_value: ParseStateEdgeContent,
}
impl<'a> GSSPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent {
        // trick: store local copy in parent keys; return ref to local var not possible; provide owned via temp
        // maintain signature by returning the ref to a static; Instead, adjust to return &self.edge_value by lifetime hack:
        unsafe { &*(&self.edge_value as *const ParseStateEdgeContent) }
    }
    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        let iso = self.parent_arc.inner.isolate(Some(self.edge_value.clone()));
        Arc::new(GSSNode { inner: iso })
    }
    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let iso = self.isolated_parent();
        iso.as_ref().push(edge_value)
    }
    pub fn popn(&self, n: usize) -> GSSPopper {
        self.isolated_parent().popn(n)
    }
    pub fn resolved_llm_tokens_union(&self) -> LLMTokenBV {
        self.isolated_parent().allowed_llm_tokens()
    }
}

// --- Popper ---
pub struct GSSPopper {
    node: Arc<GSSNode>,
    below_bottom: BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>>,
}
pub struct GSSPopperItem {
    node: Arc<GSSNode>,
    acc: Acc,
}
pub struct GSSPopperItemPeek<'a> {
    parent_arc: &'a Arc<GSSNode>,
    edge_value: &'a ParseStateEdgeContent,
}

impl GSSPopper {
    pub fn new_from_node(node: Arc<GSSNode>, _acc: Arc<Acc>) -> Self {
        GSSPopper { node, below_bottom: BTreeMap::new() }
    }
    pub fn iter(&self) -> impl Iterator<Item = GSSPopperItem> {
        self.node.inner.to_stacks().into_iter().map(|(p, a)| {
            let node = Arc::new(GSSNode {
                inner: LeveledGSS::from_stacks(&[(p, a.clone())]),
            });
            GSSPopperItem { node, acc: a }
        })
    }
    pub fn below_bottom(&self) -> &BTreeMap<usize, BTreeMap<ParseStateEdgeContent, Arc<Acc>>> {
        &self.below_bottom
    }
    pub fn num_predecessors(&self) -> usize {
        self.node.inner.peek().len()
    }
    pub fn popn(&mut self, n: usize) {
        for _ in 0..n {
            self.pop();
        }
    }
    pub fn pop(&mut self) {
        let mut inner = self.node.inner.clone();
        let mut belows: BTreeMap<_, _> = self.below_bottom.iter().map(|(k, v)| (*k + 1, v.clone())).collect();
        let new_below_slice = inner.filter_by_length(Some(1), Some(1));
        if !new_below_slice.is_empty() {
            let mut new_below_map = BTreeMap::new();
            for edge in new_below_slice.peek() {
                let isolated = new_below_slice.isolate(Some(edge.clone()));
                if let Some(acc) = isolated.reduce_acc() {
                    new_below_map.insert(edge, Arc::new(acc));
                }
            }
            if !new_below_map.is_empty() {
                belows.insert(1, new_below_map);
            }
        }
        self.below_bottom = belows;
        inner = inner.pop();
        inner = inner.filter_by_length(Some(1), None);
        self.node = Arc::new(GSSNode { inner });
    }
}

impl GSSPopperItem {
    pub fn peek_iter(&self) -> impl Iterator<Item = GSSPopperItemPeek<'_>> {
        let keys: Vec<_> = self.node.inner.peek().into_iter().collect();
        GSSPopperItemPeekIter {
            parent: &self.node,
            keys,
            idx: 0,
        }
    }
    pub(crate) fn resolved_acc(&self) -> Acc {
        self.acc.clone()
    }
}
struct GSSPopperItemPeekIter<'a> {
    parent: &'a Arc<GSSNode>,
    keys: Vec<ParseStateEdgeContent>,
    idx: usize,
}
impl<'a> Iterator for GSSPopperItemPeekIter<'a> {
    type Item = GSSPopperItemPeek<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.keys.len() {
            None
        } else {
            let ev = &self.keys[self.idx];
            self.idx += 1;
            Some(GSSPopperItemPeek {
                parent_arc: self.parent,
                edge_value: unsafe { &*(ev as *const ParseStateEdgeContent) },
            })
        }
    }
}
impl<'a> GSSPopperItemPeek<'a> {
    pub fn edge_value(&self) -> &'a ParseStateEdgeContent {
        self.edge_value
    }
    pub fn isolated_parent(&self) -> Arc<GSSNode> {
        let iso = self.parent_arc.inner.isolate(Some(self.edge_value.clone()));
        Arc::new(GSSNode { inner: iso })
    }
    pub fn push_on_parent(&self, edge_value: ParseStateEdgeContent) -> GSSNode {
        let iso = self.isolated_parent();
        iso.as_ref().push(edge_value)
    }
}

// --- Roots & helpers ---
pub fn get_roots<'a>(nodes: impl IntoIterator<Item = &'a GSSNode>) -> BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> {
    let mut out: BTreeMap<ParseStateEdgeContent, BTreeSet<Arc<Acc>>> = BTreeMap::new();
    for n in nodes {
        for (p, a) in n.inner.to_stacks() {
            if let Some(last) = p.last() {
                out.entry(last.clone())
                    .or_default()
                    .insert(Arc::new(a.clone()));
            }
        }
    }
    out
}

// --- Transformations (simplified) ---
fn transform_all(root_arc: &mut Arc<GSSNode>, f: impl FnMut(&Acc) -> Option<Acc>) {
    let new_inner = root_arc.inner.apply_and_prune(f);
    *root_arc = Arc::new(GSSNode { inner: new_inner });
}

pub type PruneAndTransformRecursiveMemo = HashMap<*const GSSNode, Option<Arc<GSSNode>>>;

pub fn allow_only_llm_tokens_and_prune(root_arc: &mut Arc<GSSNode>, allowed_tokens: &LLMTokenBV) {
    let mut memo = HashMap::new();
    allow_only_llm_tokens_and_prune_arc(root_arc, allowed_tokens, &mut memo);
}

pub(crate) fn allow_only_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        na.llm_tokens_union &= allowed_tokens;
        if na.llm_tokens_union.is_empty() {
            None
        } else {
            Some(na)
        }
    });
}

pub(crate) fn disallow_llm_tokens_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    tokens_to_disallow: &LLMTokenBV,
    memo: &mut PruneAndTransformRecursiveMemo,
) {
    let allowed = HybridBitset::max_ones() - tokens_to_disallow.clone();
    allow_only_llm_tokens_and_prune_arc(root_arc, &allowed, memo);
}

pub fn reset_llm_tokens(root_arc: &mut Arc<GSSNode>, _memo: &mut PruneAndTransformRecursiveMemo) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        na.llm_tokens_union = HybridBitset::max_ones();
        Some(na)
    });
}

pub(crate) fn reset_terminals(root_arc: &mut Arc<GSSNode>, _memo: &mut PruneAndTransformRecursiveMemo) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        na.terminals_union = HybridL2Bitset::all();
        Some(na)
    });
}

pub(crate) fn disallow_terminals_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    disallowed_terminals: &HybridL2Bitset,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    transform_all(root_arc, |a| {
        let mut na = a.clone();
        na.terminals_union -= disallowed_terminals;
        Some(na)
    });
}

pub fn prune_llm_tokens_by_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    possible_matches: &BTreeMap<crate::tokenizer::TokenizerStateID, BTreeMap<crate::types::TerminalID, LLMTokenBV>>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    // Very simplified: keep as-is (no change). Implement exact logic later.
    let _ = possible_matches;
    let _ = root_arc;
}

pub fn prune_disallowed_terminals(
    root_arc: &mut Arc<GSSNode>,
    matched_terminals: &BTreeMap<crate::tokenizer::TokenizerStateID, TerminalBV>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let matched = matched_terminals.clone();
    let all = HybridBitset::max_ones();
    transform_all(root_arc, |a| {
        for (sid, bv) in &matched {
            let allowed = a.terminals_union.get_l2_bitset(sid.0).unwrap_or(&all);
            if !bv.is_subset(allowed) {
                return None;
            }
        }
        Some(a.clone())
    });
}

pub fn map_allowed_terminals_tokenizer_states(
    root_arc: &mut Arc<GSSNode>,
    map: &BTreeMap<crate::tokenizer::TokenizerStateID, crate::tokenizer::TokenizerStateID>,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    let mapping = map.clone();
    transform_all(root_arc, |a| {
        let mut new_map = BTreeMap::new();
        for (old, new) in &mapping {
            if let Some(bv) = a.terminals_union.get_l2_bitset(old.0) {
                new_map
                    .entry(new.0)
                    .and_modify(|b: &mut HybridBitset| *b |= bv.clone())
                    .or_insert_with(|| bv.clone());
            }
        }
        let mut out = HybridL2Bitset::all();
        for (sid, bv) in new_map {
            out.insert_l2_bitset(sid, bv);
        }
        let mut na = a.clone();
        na.terminals_union = out;
        Some(na)
    });
}

pub(crate) fn allow_only_llm_tokens_on_stored_trie_nodes_and_prune_arc(
    root_arc: &mut Arc<GSSNode>,
    allowed_tokens: &LLMTokenBV,
    _memo: &mut PruneAndTransformRecursiveMemo,
    stored_trie_god: &StoredTrieGodWrapper,
) {
    // M: For this transform call, remember for each source node which destinations
    //    we have already used for it.
    let mut reuse_map: BTreeMap<StoredPrecomputeNodeIndex, BTreeSet<StoredPrecomputeNodeIndex>> = BTreeMap::new();
    // Popularity of destinations (how many sources have used a destination) to bias reuse.
    let mut dest_popularity: BTreeMap<StoredPrecomputeNodeIndex, usize> = BTreeMap::new();

    transform_all(root_arc, |a| {
        // Partition: which nodes need edges vs can be kept as-is.
        let mut needs_edge: BTreeSet<StoredPrecomputeNodeIndex> = BTreeSet::new();
        let mut keep: BTreeSet<StoredPrecomputeNodeIndex> = BTreeSet::new();

        for node in a.stored_trie_nodes() {
            let live = node
                .as_arc()
                .read(stored_trie_god)
                .expect("poison")
                .value
                .live_tokens
                .clone();
            if live.is_subset(allowed_tokens) {
                // No edge needed; keep this node unchanged.
                keep.insert(node.clone());
            } else {
                // This node has tokens outside the allowed set; it needs an edge.
                needs_edge.insert(node.clone());
            }
        }

        // If nothing needs change, keep the Acc unchanged.
        if needs_edge.is_empty() {
            return Some(a.clone());
        }

        // Choose a destination that minimizes new node creation:
        // 1) Try to reuse a destination that is already common to all sources we have seen
        //    (intersection of reuse_map entries for those sources).
        let mut chosen: Option<StoredPrecomputeNodeIndex> = None;
        {
            let mut sets: Vec<BTreeSet<StoredPrecomputeNodeIndex>> = Vec::new();
            for s in &needs_edge {
                if let Some(ds) = reuse_map.get(s) {
                    sets.push(ds.clone());
                }
            }
            if !sets.is_empty() {
                let mut inter = sets[0].clone();
                for s in sets.iter().skip(1) {
                    inter = inter.intersection(s).cloned().collect();
                    if inter.is_empty() {
                        break;
                    }
                }
                if !inter.is_empty() {
                    // Prefer the most popular destination among the intersection.
                    let mut best: Option<StoredPrecomputeNodeIndex> = None;
                    let mut best_pop = 0usize;
                    for d in inter {
                        let pop = dest_popularity.get(&d).cloned().unwrap_or(0);
                        if best.is_none() || pop > best_pop {
                            best = Some(d.clone());
                            best_pop = pop;
                        }
                    }
                    chosen = best;
                }
            }
        }

        // 2) Otherwise, reuse the most popular known destination overall (if any).
        if chosen.is_none() && !dest_popularity.is_empty() {
            let (d, _) = dest_popularity.iter().max_by_key(|(_, c)| **c).unwrap();
            chosen = Some(d.clone());
        }

        // 3) If nothing appropriate exists, create a brand new destination node.
        if chosen.is_none() {
            let new_dest = StoredPrecomputeNodeIndex::new(
                stored_trie_god.insert(crate::constraint::PrecomputeNode3::new(
                    crate::constraint::PrecomputedNodeContents::internal(),
                )),
            );
            dest_popularity.insert(new_dest.clone(), 0);
            chosen = Some(new_dest);
        }
        let dest = chosen.unwrap();

        // Insert edges from each source that needs it to the chosen destination.
        let edge_key = (0, allowed_tokens.clone());
        let edge_value = StateIDBV::max_ones();
        for src in &needs_edge {
            let inserter = crate::datastructures::trie::EdgeInserter::new(
                stored_trie_god,
                src.as_arc().clone(),
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= allowed_tokens,
                |_, _| {}, // Unconditional insertion
            );
            inserter
                .try_destination(dest.clone())
                .expect("Cycle detected when adding allowed llm tokens on stored trie nodes");

            reuse_map.entry(src.clone()).or_default().insert(dest.clone());
        }

        // Ensure destination's live tokens reflect (at least) the allowed set.
        dest.write(stored_trie_god)
            .expect("poison")
            .value
            .live_tokens
            |= allowed_tokens;

        // Update popularity score for the chosen destination.
        {
            let c = dest_popularity.entry(dest.clone()).or_insert(0);
            *c += needs_edge.len();
        }

        // Build the final set: keep unchanged nodes and add the chosen destination once.
        let mut final_nodes = keep;
        final_nodes.insert(dest.clone());

        let mut na = a.clone();
        na.stored_trie_nodes = final_nodes;
        Some(na)
    });
}

pub fn simplify(_states: &mut BTreeMap<crate::tokenizer::TokenizerStateID, Arc<GSSNode>>) {}
pub(crate) fn simplify_roots_in_place(_roots: &mut [Arc<GSSNode>]) {}
pub fn fuse_predecessors_recursive(node_arc: &Arc<GSSNode>, levels: usize, _memo: &mut HashMap<*const GSSNode, Arc<GSSNode>>) -> Arc<GSSNode> {
    let fused_inner = node_arc.inner.fuse(Some(levels as isize));
    if fused_inner.is(&node_arc.inner) {
        node_arc.clone()
    } else {
        Arc::new(GSSNode { inner: fused_inner })
    }
}

impl GSSNode {
    pub fn reset_llm_tokens(&mut self) {
        let mut memo = PruneAndTransformRecursiveMemo::new();
        let mut arc = Arc::new(self.clone());
        reset_llm_tokens(&mut arc, &mut memo);
        *self = (*arc).clone();
    }
}

// --- Trie-utils stubs (no-ops) ---
pub(crate) fn deep_add_precompute_trie_edges(
    root_arc: &mut Arc<GSSNode>,
    god: &StoredTrieGodWrapper,
    edge_key: &(usize, LLMTokenBV),
    edge_value: &StateIDBV,
    tokens_for_update: &LLMTokenBV,
    destination_provider: &mut impl FnMut() -> StoredPrecomputeNodeIndex,
    _memo: &mut PruneAndTransformRecursiveMemo,
) {
    transform_all(root_arc, |acc| {
        if acc.stored_trie_nodes().is_empty() {
            return Some(acc.clone());
        }

        let destination = destination_provider();

        for source_wrapper in acc.stored_trie_nodes() {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = crate::datastructures::trie::EdgeInserter::new(
                god,
                source_arc,
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= tokens_for_update,
                |_, _| {}, // Unconditional insertion
            );
            inserter.try_destination(destination.clone()).expect("Cycle detected when adding precompute trie edges");
        }

        destination.write(god).expect("poison").value.live_tokens |= tokens_for_update;

        let mut new_acc = acc.clone();
        *new_acc.stored_trie_nodes_mut() = BTreeSet::from([destination]);
        Some(new_acc)
    });
}

pub(crate) fn merge_stored_trie_nodes(
    root_arc: &mut Arc<GSSNode>,
    _memo: &mut PruneAndTransformRecursiveMemo,
    stored_trie_god: &StoredTrieGodWrapper,
) {
    let mut new_destinations = BTreeMap::new();

    transform_all(root_arc, |acc| {
        if !acc.stored_trie_nodes().iter().any(
            |n| n.as_arc().read(stored_trie_god).expect("poison").value.live_tokens != acc.llm_tokens_union
        ) {
            return Some(acc.clone());
        }

        let mut new_acc = acc.clone();
        // Create a single new destination for this merge operation.
        let new_destination = new_destinations.entry((new_acc.stored_trie_nodes().clone(), acc.llm_tokens_union.clone()))
            .or_insert_with(|| StoredPrecomputeNodeIndex::new(stored_trie_god.insert(crate::constraint::PrecomputeNode3::new(crate::constraint::PrecomputedNodeContents::internal()))))
            .clone();

        let edge_key = (0, new_acc.llm_tokens_union.clone());
        let edge_value = crate::constraint::StateIDBV::max_ones();
        let tokens_for_edge = new_acc.llm_tokens_union.clone();

        for source_wrapper in new_acc.stored_trie_nodes() {
            let source_arc = source_wrapper.as_arc().clone();

            let inserter = crate::datastructures::trie::EdgeInserter::new(
                stored_trie_god,
                source_arc,
                edge_key.clone(),
                edge_value.clone(),
                |e, n| *e |= n,
                |node_value, _edge_value| node_value.live_tokens |= &tokens_for_edge,
                |_, _| {}, // Unconditional insertion
            );
            inserter.try_destination(new_destination.clone()).expect("Cycle detected when merging stored_trie nodes; this should be impossible.");
        }

        new_destination.write(stored_trie_god).expect("poison").value.live_tokens |= &tokens_for_edge;

        new_acc.stored_trie_nodes = BTreeSet::from([new_destination]);
        Some(new_acc)
    });
}

pub(crate) fn is_simple_gss(
    node: &Arc<GSSNode>,
    hallucinated_state_id: StateID,
) -> Option<(StateID, Arc<Acc>)> {
    let stacks = node.inner.to_stacks();
    // Must be exactly one path.
    if stacks.len() == 1 {
        let (path, acc) = &stacks[0];
        // Path must have length 2. The path is from leaf to root.
        if path.len() == 2 {
            let edge_to_leaf = &path[0];
            let edge_to_middle = &path[1];

            // The edge into the leaf must be the hallucinated one.
            if edge_to_leaf.state_id == hallucinated_state_id {
                let state_id_x = edge_to_middle.state_id;
                // The leaf must have stored_trie_nodes.
                if !acc.stored_trie_nodes().is_empty() {
                    return Some((state_id_x, Arc::new(acc.clone())));
                }
            }
        }
    }
    None
}

// helper used by parser logging
pub fn popn_collect_isolated_parents(node_arc: &Arc<GSSNode>, n: usize) -> Vec<(StateID, Arc<GSSNode>)> {
    let popped = node_arc.inner.popn(n as isize);
    let mut out = Vec::new();
    for (path, _a) in popped.to_stacks() {
        if let Some(last) = path.last() {
            let iso = popped.isolate(Some(last.clone()));
            out.push((last.state_id, Arc::new(GSSNode { inner: iso })));
        }
    }
    out
}

// Compatibility for formatting acc (used by printer)
pub(crate) fn format_acc(
    _node: &GSSNode,
    _terminal_map: &BiBTreeMap<crate::glr::grammar::Terminal, TerminalID>,
    _original_internal_bimap: Option<&BTreeMap<usize, usize>>,
    _llm_token_map: Option<&BiBTreeMap<Vec<u8>, LLMTokenID>>,
    _config: &GSSPrintConfig,
) -> String {
    String::new()
}