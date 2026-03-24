//! Trie-based reference vocab equivalence analysis.
//!
//! Partitions tokens by DFA behavior using a byte-level trie and per-token hashing.
//! The trie amortizes DFA transitions: tokens sharing a prefix share the walk.
//!
//! Algorithm:
//! 1. Build flat DFA from tokenizer
//! 2. Build byte-level trie from vocabulary
//! 3. Walk the trie depth-first, carrying all initial DFA states simultaneously
//! 4. At each token leaf, compute a signature from end states, match positions,
//!    and suffix DAG structure
//! 5. Group tokens by signature → equivalence classes

// Do NOT add caching shortcuts that skip states/tokens. Full correctness mandatory.

use super::super::compat::TokenizerView;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{BuildHasher, Hasher};

use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

type EdgeList = SmallVec<[(usize, usize); 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;
// ---- Deterministic hashing ----

static HASH_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_STATE.build_hasher()
}

#[inline]
fn hash_group_list(iter: impl ExactSizeIterator<Item = usize>) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    h.write_u64(iter.len() as u64);
    for v in iter {
        h.write_u64(v as u64);
    }
    h.finish()
}

// ---- Flat DFA ----

struct Dfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<SmallVec<[usize; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    possible_future_groups: Vec<SmallVec<[usize; 4]>>,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    /// Per-state bitset: which bytes cause a self-loop (transition back to same state).
    self_loop_bytes: Vec<U8Set>,
    /// Precomputed hash for suffix DAG at end-of-token (empty suffix).
    empty_suffix_hash: u64,
    disallowed_follows: Vec<BitSet>,
}

impl Dfa {
    #[inline]
    fn completion(&self, state: usize) -> u64 {
        if state < self.completion_hash.len() {
            self.completion_hash[state]
        } else {
            self.none_completion_hash
        }
    }

    #[inline]
    fn completion_with_disallowed(&self, state: usize, disallowed: Option<&BitSet>) -> u64 {
        let Some(disallowed) = disallowed.filter(|bits| !bits.is_zero()) else {
            return self.completion(state);
        };
        if state >= self.possible_future_groups.len() {
            return self.none_completion_hash;
        }
        let mut h = new_hasher();
        h.write_u8(2);
        h.write_u64(disallowed.len() as u64);
        for &word in disallowed.words() {
            h.write_u64(word);
        }
        h.write_u64(hash_filtered_group_list(&self.possible_future_groups[state], disallowed));
        h.finish()
    }

    #[inline]
    fn disallowed_for(&self, gid: usize) -> &BitSet {
        &self.disallowed_follows[gid]
    }

    #[inline]
    fn empty_suffix_hash_for(&self, gid: usize) -> u64 {
        let disallowed = self.disallowed_for(gid);
        if disallowed.is_zero() {
            return self.empty_suffix_hash;
        }
        let end_state = if self.is_dead_end[self.start_state] {
            STATE_NONE
        } else {
            self.start_state
        };
        let mut h = new_hasher();
        h.write_u64(self.completion_with_disallowed(end_state, Some(disallowed)));
        h.finish()
    }
}

fn normalize_disallowed_follows(
    num_groups: usize,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Vec<BitSet> {
    let mut normalized = vec![BitSet::new(num_groups); num_groups];
    for gid in 0..num_groups {
        if let Some(bits) = disallowed_follows.get(&(gid as u32)) {
            let mut out = BitSet::new(num_groups);
            for bit in bits.iter() {
                if bit < num_groups {
                    out.set(bit);
                }
            }
            normalized[gid] = out;
        }
    }
    normalized
}

#[inline]
fn hash_filtered_group_list(groups: &[usize], disallowed: &BitSet) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    let mut count = 0usize;
    for &gid in groups {
        if !disallowed.contains(gid) {
            count += 1;
        }
    }
    h.write_u64(count as u64);
    for &gid in groups {
        if !disallowed.contains(gid) {
            h.write_u64(gid as u64);
        }
    }
    h.finish()
}

fn build_dfa(regex: &TokenizerView, disallowed_follows: &BTreeMap<u32, BitSet>) -> Dfa {
    let dfa = regex.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");

    let num_groups = dfa
        .states
        .iter()
        .flat_map(|s| {
            s.finalizers
                .iter()
                .copied()
                .chain(s.possible_future_group_ids.iter().copied())
        })
        .max()
        .map_or(0, |m| m + 1);

    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut possible_future_groups = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE; 256];
        for (byte_idx, &target) in state.transitions.iter().enumerate() {
            table[byte_idx] = target;
        }
        transitions.push(table);

        finalizers.push(state.finalizers.iter().copied().collect());

        is_dead_end.push(state.possible_future_group_ids.is_empty());
        let future_groups: SmallVec<[usize; 4]> =
            state.possible_future_group_ids.iter().copied().collect();
        completion_hash.push(hash_group_list(future_groups.iter().copied()));
        possible_future_groups.push(future_groups);
    }

    let none_completion_hash = {
        let mut h = new_hasher();
        h.write_u8(0);
        h.finish()
    };

    // Precompute self-loop byte sets per state
    let self_loop_bytes: Vec<U8Set> = (0..transitions.len())
        .map(|s| {
            let mut bits = U8Set::empty();
            for b in 0..=255u8 {
                if transitions[s][b as usize] == s as u32 {
                    bits.insert(b);
                }
            }
            bits
        })
        .collect();

    // Precompute empty suffix hash (suffix DAG at end-of-token, where remaining = "")
    let empty_suffix_hash = {
        let end = if is_dead_end[dfa.start_state] {
            STATE_NONE
        } else {
            dfa.start_state
        };
        let ch = if end < completion_hash.len() {
            completion_hash[end]
        } else {
            none_completion_hash
        };
        let mut h = new_hasher();
        h.write_u64(ch);
        h.finish()
    };

    Dfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        is_dead_end,
        num_groups,
        possible_future_groups,
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
        empty_suffix_hash,
        disallowed_follows: normalize_disallowed_follows(num_groups, disallowed_follows),
    }
}

fn intersect_node_disallowed(scratch: &mut Scratch, pos: usize, incoming: &BitSet) {
    if scratch.dag_disallowed_generation[pos] == scratch.dag_generation {
        scratch.dag_disallowed[pos] = scratch.dag_disallowed[pos].intersection(incoming);
    } else {
        scratch.dag_disallowed[pos] = incoming.clone();
        scratch.dag_disallowed_generation[pos] = scratch.dag_generation;
    }
}

fn node_disallows_gid(scratch: &Scratch, pos: usize, gid: usize) -> bool {
    scratch.dag_disallowed_generation[pos] == scratch.dag_generation
        && scratch.dag_disallowed[pos].contains(gid)
}

struct ProgressReporter {
}

impl ProgressReporter {
    fn new(_total: usize) -> Self {
        ProgressReporter {}
    }

    #[inline]
    fn record(&mut self, _count: usize) {
    }
}

// ---- Vocab Trie ----

struct TrieNode {
    children: SmallVec<[(u8, u32); 4]>,
    token_idx: u32, // u32::MAX if not a token endpoint
    /// Bitset of all bytes reachable from descendant edges of this node.
    subtree_bytes: U8Set,
}

struct VocabTrie {
    nodes: Vec<TrieNode>,
}

impl VocabTrie {
    fn build<S: AsRef<[u8]>>(tokens: &[S]) -> Self {
        // Estimate node count: total bytes across all tokens (upper bound for trie nodes)
        let total_bytes: usize = tokens.iter().map(|t| t.as_ref().len()).sum();
        let mut nodes = Vec::with_capacity(total_bytes + 1);
        nodes.push(TrieNode {
            children: SmallVec::new(),
            token_idx: u32::MAX,
            subtree_bytes: U8Set::empty(),
        });

        // Flat lookup tables for root (depth 0) and depth-1 nodes to avoid linear search
        let mut root_children = [u32::MAX; 256];
        // d1_children[byte0][byte1] -> node_id at depth 2
        let mut d1_children = vec![[u32::MAX; 256]; 256];

        for (idx, token) in tokens.iter().enumerate() {
            let bytes = token.as_ref();
            if bytes.is_empty() {
                nodes[0].token_idx = idx as u32;
                continue;
            }

            // Depth 0 → 1: flat O(1) lookup
            let b0 = bytes[0] as usize;
            let mut cur = if root_children[b0] != u32::MAX {
                root_children[b0]
            } else {
                let new_idx = nodes.len() as u32;
                nodes.push(TrieNode {
                    children: SmallVec::new(),
                    token_idx: u32::MAX,
                    subtree_bytes: U8Set::empty(),
                });
                root_children[b0] = new_idx;
                new_idx
            };

            if bytes.len() == 1 {
                nodes[cur as usize].token_idx = idx as u32;
                continue;
            }

            // Depth 1 → 2: flat O(1) lookup
            let b1 = bytes[1] as usize;
            cur = if d1_children[b0][b1] != u32::MAX {
                d1_children[b0][b1]
            } else {
                let new_idx = nodes.len() as u32;
                nodes.push(TrieNode {
                    children: SmallVec::new(),
                    token_idx: u32::MAX,
                    subtree_bytes: U8Set::empty(),
                });
                d1_children[b0][b1] = new_idx;
                new_idx
            };

            if bytes.len() == 2 {
                nodes[cur as usize].token_idx = idx as u32;
                continue;
            }

            // Depth 2+: binary search (children are small at this depth)
            for &byte in &bytes[2..] {
                let result = nodes[cur as usize]
                    .children
                    .binary_search_by_key(&byte, |&(b, _)| b);
                cur = match result {
                    Ok(p) => nodes[cur as usize].children[p].1,
                    Err(p) => {
                        let new_idx = nodes.len() as u32;
                        nodes.push(TrieNode {
                            children: SmallVec::new(),
                            token_idx: u32::MAX,
                            subtree_bytes: U8Set::empty(),
                        });
                        nodes[cur as usize].children.insert(p, (byte, new_idx));
                        new_idx
                    }
                };
            }
            nodes[cur as usize].token_idx = idx as u32;
        }

        // Convert flat lookup tables to sorted children lists
        nodes[0].children = (0..=255u8)
            .filter(|&b| root_children[b as usize] != u32::MAX)
            .map(|b| (b, root_children[b as usize]))
            .collect();

        // Depth-1 nodes: build children from d1_children table
        for b0 in 0..256 {
            let parent = root_children[b0];
            if parent != u32::MAX {
                nodes[parent as usize].children = (0..=255u8)
                    .filter(|&b| d1_children[b0][b as usize] != u32::MAX)
                    .map(|b| (b, d1_children[b0][b as usize]))
                    .collect();
            }
        }

        // Depth 2+ children are already sorted (inserted via binary_search)

        // Compute subtree byte sets (post-order)
        fn compute_subtree_bytes(nodes: &mut [TrieNode], idx: u32) -> U8Set {
            let mut bits = U8Set::empty();
            let num_children = nodes[idx as usize].children.len();
            for i in 0..num_children {
                let (byte, child_idx) = nodes[idx as usize].children[i];
                bits.insert(byte);
                let child_bits = compute_subtree_bytes(nodes, child_idx);
                bits |= child_bits;
            }
            nodes[idx as usize].subtree_bytes = bits;
            bits
        }
        compute_subtree_bytes(&mut nodes, 0);

        VocabTrie { nodes }
    }
}

// ---- Suffix DAG ----

/// Run DFA on a suffix from start_state, returning (end_state, edges to match positions).
fn run_suffix(
    dfa: &Dfa,
    slice: &[u8],
    base_pos: usize,
    mp: &mut [u32],
) -> (Option<usize>, EdgeList) {
    let ng = dfa.num_groups;
    mp[..ng].fill(NONE);
    let mut cur = dfa.start_state;
    let mut done = dfa.is_dead_end[cur];

    for &gid in &dfa.finalizers[cur] {
        if gid < ng && mp[gid] == NONE {
            mp[gid] = 0;
        }
    }

    for (i, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }
        let ns = dfa.transitions[cur][byte as usize];
        if ns == NONE {
            done = true;
            break;
        }
        cur = ns as usize;
        let pos = (i + 1) as u32;
        for &gid in &dfa.finalizers[cur] {
            if gid < ng {
                mp[gid] = pos;
            }
        }
        if dfa.is_dead_end[cur] {
            done = true;
        }
    }

    let end = if done { None } else { Some(cur) };
    let edges: EdgeList = (0..ng)
        .filter_map(|gid| {
            let pv = mp[gid];
            (pv != NONE && pv != 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect();
    (end, edges)
}

/// Build suffix DAG via BFS from target positions and hash bottom-up.
/// Uses scratch.targets as input positions.
fn hash_suffixes(dfa: &Dfa, slice: &[u8], scratch: &mut Scratch) {
    let len = slice.len();
    scratch.dag_generation = scratch.dag_generation.wrapping_add(1);
    scratch.queue.clear();

    // Ensure dag is large enough
    if len + 2 > scratch.dag.len() {
        scratch.dag.resize_with(len + 2, || DagEntry {
            hash: 0,
            edges: EdgeList::new(),
            generation: 0,
        });
        scratch.dag_end_states.resize(len + 2, STATE_NONE);
        scratch
            .dag_disallowed
            .resize_with(len + 2, || BitSet::new(scratch.num_groups));
        scratch.dag_disallowed_generation.resize(len + 2, 0);
    }

    for ti in 0..scratch.targets.len() {
        let pos = scratch.targets[ti];
        if pos <= len && !scratch.dag_contains(pos) {
            scratch.activate_dag_node(pos);
        }
    }

    let mut cursor = 0;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;
        let (end, edges) = run_suffix(dfa, &slice[pos..], pos, &mut scratch.tmp_mp);
        for &(_, target) in &edges {
            if target <= len && !scratch.dag_contains(target) {
                scratch.activate_dag_node(target);
            }
        }
        scratch.dag_end_states[pos] = end.unwrap_or(STATE_NONE);
        scratch.dag[pos].hash = 0;
        scratch.dag[pos].edges = edges;
    }

    for ei in 0..scratch.root_edges.len() {
        let (gid, pos) = scratch.root_edges[ei];
        if pos <= len && scratch.dag_contains(pos) {
            intersect_node_disallowed(scratch, pos, dfa.disallowed_for(gid));
        }
    }

    scratch.queue.sort_unstable();
    for idx in 0..scratch.queue.len() {
        let pos = scratch.queue[idx];
        for ei in 0..scratch.dag[pos].edges.len() {
            let (gid, target) = scratch.dag[pos].edges[ei];
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            if target <= len && scratch.dag_contains(target) {
                intersect_node_disallowed(scratch, target, dfa.disallowed_for(gid));
            }
        }
    }

    scratch.queue.sort_unstable_by(|a, b| b.cmp(a));
    for idx in 0..scratch.queue.len() {
        let pos = scratch.queue[idx];
        let mut h = new_hasher();
        h.write_u64(
            dfa.completion_with_disallowed(
                scratch.dag_end_states[pos],
                (scratch.dag_disallowed_generation[pos] == scratch.dag_generation)
                    .then_some(&scratch.dag_disallowed[pos]),
            ),
        );
        // Need to iterate edges without borrowing dag mutably at the same time
        for ei in 0..scratch.dag[pos].edges.len() {
            let (gid, target) = scratch.dag[pos].edges[ei];
            if node_disallows_gid(scratch, pos, gid) {
                continue;
            }
            h.write_u64(gid as u64);
            h.write_u64(scratch.dag_get_hash(target));
        }
        scratch.dag[pos].hash = h.finish();
    }
}

// ---- Scratch workspace ----

struct DagEntry {
    hash: u64,
    edges: EdgeList,
    generation: u32,
}

struct Scratch {
    dag: Vec<DagEntry>,
    dag_end_states: Vec<usize>,
    dag_generation: u32,
    queue: Vec<usize>,
    tmp_mp: Vec<u32>,
    targets: Vec<usize>,
    root_edges: Vec<(usize, usize)>,
    dag_disallowed: Vec<BitSet>,
    dag_disallowed_generation: Vec<u32>,
    num_groups: usize,
}

impl Scratch {
    fn new(ng: usize, max_token_len: usize) -> Self {
        let cap = max_token_len + 2;
        let mut dag = Vec::with_capacity(cap);
        for _ in 0..cap {
            dag.push(DagEntry {
                hash: 0,
                edges: EdgeList::new(),
                generation: 0,
            });
        }
        Scratch {
            dag,
            dag_end_states: vec![STATE_NONE; cap],
            dag_generation: 0,
            queue: Vec::new(),
            tmp_mp: vec![NONE; ng],
            targets: Vec::new(),
            root_edges: Vec::new(),
            dag_disallowed: vec![BitSet::new(ng); cap],
            dag_disallowed_generation: vec![0; cap],
            num_groups: ng,
        }
    }

    #[inline]
    fn activate_dag_node(&mut self, pos: usize) {
        self.queue.push(pos);
        self.dag[pos].hash = 0;
        self.dag[pos].edges.clear();
        self.dag[pos].generation = self.dag_generation;
        self.dag_end_states[pos] = STATE_NONE;
        self.dag_disallowed[pos].clear_all();
        self.dag_disallowed_generation[pos] = self.dag_generation.wrapping_sub(1);
    }

    #[inline]
    fn dag_contains(&self, pos: usize) -> bool {
        pos < self.dag.len() && self.dag[pos].generation == self.dag_generation
    }

    #[inline]
    fn dag_get_hash(&self, pos: usize) -> u64 {
        if pos < self.dag.len() && self.dag[pos].generation == self.dag_generation {
            self.dag[pos].hash
        } else {
            0
        }
    }
}

// ---- Core: recursive trie walk with inline signature computation ----

/// Assign the same hash to all tokens in a trie subtree.
fn assign_hash_to_subtree(
    trie: &VocabTrie,
    node: u32,
    hash: u64,
    hashes: &mut [u64],
    progress: &mut ProgressReporter,
) {
    let n = &trie.nodes[node as usize];
    if n.token_idx != u32::MAX {
        hashes[n.token_idx as usize] = hash;
        progress.record(1);
    }
    for &(_, child) in &n.children {
        assign_hash_to_subtree(trie, child, hash, hashes, progress);
    }
}

/// Walk the trie depth-first, carrying DFA states for all initial states.
/// At each token leaf, computes the token's signature and writes to `hashes`.
///
/// Layout: `states[depth * ni + si]`, `mp[(depth * ni + si) * ng + gid]`
fn walk_trie<S: AsRef<[u8]>>(
    trie: &VocabTrie,
    node: u32,
    dfa: &Dfa,
    states: &mut [u32],
    mp: &mut [u32],
    depth: usize,
    ni: usize,
    ng: usize,
    max_depth: usize,
    strings: &[S],
    scratch: &mut Scratch,
    hashes: &mut [u64],
    progress: &mut ProgressReporter,
) {
    let n = &trie.nodes[node as usize];

    // At token leaf: compute signature
    if n.token_idx != u32::MAX {
        let ti = n.token_idx as usize;
        let bytes = strings[ti].as_ref();

        // Collect suffix targets across all initial states
        scratch.targets.clear();
        scratch.root_edges.clear();
        for si in 0..ni {
            let base = (depth * ni + si) * ng;
            for gid in 0..ng {
                let pv = mp[base + gid];
                if pv != NONE && pv > 0 {
                    scratch.targets.push(pv as usize);
                    scratch.root_edges.push((gid, pv as usize));
                }
            }
        }
        scratch.targets.sort_unstable();
        scratch.targets.dedup();

        if !scratch.targets.is_empty() {
            hash_suffixes(dfa, bytes, scratch);
        }

        // Fold per-state signatures into token hash
        let mut hash = HASH_SEED3;
        for si in 0..ni {
            let es = states[depth * ni + si];
            let base = (depth * ni + si) * ng;
            let mp_slice = &mp[base..base + ng];

            let completion = if es == NONE {
                dfa.none_completion_hash
            } else {
                dfa.completion_hash[es as usize]
            };

            let sig = if mp_slice.iter().any(|&pv| pv != NONE) {
                let mut h = new_hasher();
                h.write_u64(completion);
                for (gid, &pv) in mp_slice.iter().enumerate() {
                    if pv != NONE && pv > 0 {
                        h.write_u64(gid as u64);
                        h.write_u64(scratch.dag_get_hash(pv as usize));
                    }
                }
                h.finish()
            } else {
                completion
            };

            hash = hash.wrapping_mul(HASH_SEED1).wrapping_add(sig);
        }
        hashes[ti] = hash;
        progress.record(1);
    }

    // Recurse into children
    for &(byte, child) in &n.children {
        let cd = depth + 1;
        if cd >= max_depth {
            continue;
        }

        for si in 0..ni {
            let ps = states[depth * ni + si];
            let parent_mp = (depth * ni + si) * ng;
            let child_mp = (cd * ni + si) * ng;

            // Copy parent match positions to child
            mp.copy_within(parent_mp..parent_mp + ng, child_mp);

            if ps == NONE {
                states[cd * ni + si] = NONE;
            } else {
                let ns = dfa.transitions[ps as usize][byte as usize];
                if ns == NONE {
                    states[cd * ni + si] = NONE;
                } else {
                    let ns_u = ns as usize;
                    // Apply finalizers at new state
                    for &gid in &dfa.finalizers[ns_u] {
                        if gid < ng {
                            mp[child_mp + gid] = cd as u32;
                        }
                    }
                    states[cd * ni + si] = if dfa.is_dead_end[ns_u] { NONE } else { ns };
                }
            }
        }

        // Self-loop optimization: if all alive states self-loop on every byte
        // reachable from the child subtree, then all tokens in the subtree will
        // end in the same states and can potentially share one signature.
        let child_node = &trie.nodes[child as usize];
        if !child_node.subtree_bytes.is_empty() {
            // Intersect self_loop_bytes across all alive states at child depth
            let mut sl_inter = U8Set::all();
            let mut any_alive = false;
            for si in 0..ni {
                let cs = states[cd * ni + si];
                if cs != NONE {
                    any_alive = true;
                    sl_inter &= dfa.self_loop_bytes[cs as usize];
                }
            }

            if any_alive && child_node.subtree_bytes.is_subset(&sl_inter) {
                // All alive states self-loop on all descendant bytes.
                // Check if bulk-assign is safe: every mp > 0 must be for a group
                // where the current state has a greedy finalizer (so mp advances
                // to token_length with empty suffix, producing the same hash).
                let can_bulk = (0..ni).all(|si| {
                    let cs = states[cd * ni + si];
                    let base = (cd * ni + si) * ng;
                    (0..ng).all(|gid| {
                        let pv = mp[base + gid];
                        if pv > 0 && pv != NONE {
                            // For alive states: needs greedy finalizer to advance mp to L.
                            // For dead states: suffix depends on token content → NOT safe.
                            cs != NONE
                                && dfa.finalizers[cs as usize]
                                    .iter()
                                    .any(|&state_gid| state_gid == gid)
                        } else {
                            true
                        }
                    })
                });

                if can_bulk {
                    // Compute the signature that all tokens in the subtree share.
                    // End states are the same (self-loop). Greedy mp → L (token length),
                    // suffix from L is empty → empty_suffix_hash.
                    let mut hash = HASH_SEED3;
                    for si in 0..ni {
                        let es = states[cd * ni + si];
                        let base = (cd * ni + si) * ng;
                        let completion = if es == NONE {
                            dfa.none_completion_hash
                        } else {
                            dfa.completion_hash[es as usize]
                        };

                        let has_any = (0..ng).any(|gid| mp[base + gid] != NONE);
                        let sig = if has_any {
                            let mut h = new_hasher();
                            h.write_u64(completion);
                            for gid in 0..ng {
                                let pv = mp[base + gid];
                                if pv != NONE && pv > 0 {
                                    // Greedy finalizer: mp will advance to L,
                                    // suffix from L is empty.
                                    h.write_u64(gid as u64);
                                    h.write_u64(dfa.empty_suffix_hash_for(gid));
                                }
                            }
                            h.finish()
                        } else {
                            completion
                        };
                        hash = hash.wrapping_mul(HASH_SEED1).wrapping_add(sig);
                    }

                    assign_hash_to_subtree(trie, child, hash, hashes, progress);
                    continue;
                }
            }
        }

        walk_trie(
            trie, child, dfa, states, mp, cd, ni, ng, max_depth, strings, scratch, hashes, progress,
        );
    }
}

// ---- Public API ----

pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> VocabEquivalenceResult {
    let dfa = build_dfa(regex, disallowed_follows);
    let nt = strings.len();
    let ni = initial_states.len();

    if ni == 0 || nt == 0 {
        return BTreeSet::from_iter(vec![(0..nt).collect()]);
    }

    let ng = dfa.num_groups;
    let trie = VocabTrie::build(strings);
    let max_depth: usize = 256;

    let mut hashes = vec![HASH_SEED3; nt];
    let mut states = vec![NONE; max_depth * ni];
    let mut mp = vec![NONE; max_depth * ni * ng];
    let max_token_len = strings.iter().map(|s| s.as_ref().len()).max().unwrap_or(0);
    let mut scratch = Scratch::new(ng, max_token_len);
    let mut progress = ProgressReporter::new(nt);

    // Initialize depth 0: set initial DFA states and their finalizers
    for (si, &s) in initial_states.iter().enumerate() {
        let mp_base = si * ng;
        for &gid in &dfa.finalizers[s] {
            if gid < ng && mp[mp_base + gid] == NONE {
                mp[mp_base + gid] = 0;
            }
        }
        states[si] = if dfa.is_dead_end[s] { NONE } else { s as u32 };
    }

    walk_trie(
        &trie,
        0,
        &dfa,
        &mut states,
        &mut mp,
        0,
        ni,
        ng,
        max_depth,
        strings,
        &mut scratch,
        &mut hashes,
        &mut progress,
    );

    // Group tokens by hash → equivalence classes
    let mut groups: HashMap<u64, Vec<usize>> = HashMap::with_capacity(nt / 4);
    for (ti, &h) in hashes.iter().enumerate() {
        groups.entry(h).or_default().push(ti);
    }
    groups.into_values().collect()
}

// ---- Partition comparison utilities ----

/// Returns true if `finer` is at least as fine as `coarser`.
///
/// Every class in `finer` must be a subset of some class in `coarser`.
pub fn partition_is_at_least_as_fine(
    finer: &VocabEquivalenceResult,
    coarser: &VocabEquivalenceResult,
) -> bool {
    finer
        .iter()
        .all(|fc| coarser.iter().any(|cc| fc.iter().all(|t| cc.contains(t))))
}

/// Returns true if one partition refines the other (or they are equal).
pub fn partitions_are_comparable(a: &VocabEquivalenceResult, b: &VocabEquivalenceResult) -> bool {
    partition_is_at_least_as_fine(a, b) || partition_is_at_least_as_fine(b, a)
}

/// Returns true if both partitions have identical classes.
#[cfg(test)]
pub fn partitions_are_equivalent(a: &VocabEquivalenceResult, b: &VocabEquivalenceResult) -> bool {
    a == b
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    // NOTE: test_a_plus_equivalence remains disabled until the legacy byte-DFA
    // fixtures used by this test are restored.
    // Original test used: crate::dfa_u8::{eat_u8, greedy_group, rep1, Tokenizer}

    #[test]
    fn test_partition_reflexive() {
        let p: VocabEquivalenceResult = BTreeSet::from([vec![0, 1], vec![2, 3]]);
        assert!(partition_is_at_least_as_fine(&p, &p));
        assert!(partitions_are_comparable(&p, &p));
        assert!(partitions_are_equivalent(&p, &p));
    }

    #[test]
    fn test_partition_finer() {
        let coarse: VocabEquivalenceResult = BTreeSet::from([vec![0, 1, 2], vec![3, 4]]);
        let fine: VocabEquivalenceResult = BTreeSet::from([vec![0, 1], vec![2], vec![3, 4]]);
        assert!(partition_is_at_least_as_fine(&fine, &coarse));
        assert!(!partition_is_at_least_as_fine(&coarse, &fine));
        assert!(partitions_are_comparable(&fine, &coarse));
        assert!(!partitions_are_equivalent(&fine, &coarse));
    }

    #[test]
    fn test_partition_incomparable() {
        let a: VocabEquivalenceResult = BTreeSet::from([vec![0, 1], vec![2, 3]]);
        let b: VocabEquivalenceResult = BTreeSet::from([vec![0, 2], vec![1, 3]]);
        assert!(!partition_is_at_least_as_fine(&a, &b));
        assert!(!partition_is_at_least_as_fine(&b, &a));
        assert!(!partitions_are_comparable(&a, &b));
    }
}
