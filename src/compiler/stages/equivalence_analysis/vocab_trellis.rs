//! Trellis-based vocab equivalence analysis (ported from sep1/grammars2024).
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
//!
//! Ported from grammars2024/src/equivalence_analysis/vocab_equivalence_analysis_fast_simple.rs.
//! Adapted to use glrmask DFA types (BitSet finalizers, CharTransitions<u32>, per-state
//! non_greedy_finalizers).

// Do NOT add caching shortcuts that skip states/tokens. Full correctness mandatory.

use crate::automata::lexer::tokenizer::Tokenizer;
use ahash::{AHasher, RandomState};
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::hash::{BuildHasher, Hasher};
use std::sync::LazyLock;

use super::ManyToOneIdMap;

type EdgeList = SmallVec<[(usize, usize); 4]>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;

// ---- Deterministic hashing ----

static HASH_STATE: LazyLock<RandomState> =
    LazyLock::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_STATE.build_hasher()
}

#[inline]
fn hash_group_list(iter: impl Iterator<Item = usize>) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    let mut count = 0u64;
    // We hash each element and count them; the count goes at the end to
    // distinguish different-length sequences.
    for v in iter {
        h.write_u64(v as u64);
        count += 1;
    }
    h.write_u64(count);
    h.finish()
}

// ---- Flat DFA ----

#[derive(Clone, Copy)]
struct Finalizer {
    gid: usize,
    non_greedy: bool,
}

struct Dfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<SmallVec<[Finalizer; 4]>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    self_loop_bytes: Vec<[u64; 4]>,
    empty_suffix_hash: u64,
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
}

fn build_dfa(tokenizer: &Tokenizer) -> Dfa {
    let dfa = &tokenizer.dfa;
    let states = dfa.states();
    let n = states.len();
    assert!(n <= u32::MAX as usize, "DFA too large");

    // Determine number of groups from state metadata.
    // Use DFA accessors for `possible_future_group_ids` (private field).
    let num_groups = {
        let mut max_gid = 0usize;
        for (i, s) in states.iter().enumerate() {
            for gid in s.finalizers.iter() {
                if gid + 1 > max_gid { max_gid = gid + 1; }
            }
            for gid in s.non_greedy_finalizers.iter() {
                if gid + 1 > max_gid { max_gid = gid + 1; }
            }
            for gid in dfa.possible_future_group_ids(i as u32).iter() {
                if gid + 1 > max_gid { max_gid = gid + 1; }
            }
        }
        max_gid
    };

    // Build non-greedy flags: a group is non-greedy if ANY state has it in non_greedy_finalizers.
    let mut non_greedy_flags = vec![false; num_groups];
    for state in states {
        for gid in state.non_greedy_finalizers.iter() {
            if gid < num_groups {
                non_greedy_flags[gid] = true;
            }
        }
    }

    let mut transitions = Vec::with_capacity(n);
    let mut finalizers_out = Vec::with_capacity(n);
    let mut is_dead_end = Vec::with_capacity(n);
    let mut completion_hash = Vec::with_capacity(n);

    for (si, state) in states.iter().enumerate() {
        let mut table = [NONE; 256];
        for (byte, &target) in state.transitions.iter() {
            table[byte as usize] = target;
        }
        transitions.push(table);

        let mut state_finals: SmallVec<[Finalizer; 4]> = SmallVec::new();
        for gid in state.finalizers.iter() {
            state_finals.push(Finalizer {
                gid,
                non_greedy: non_greedy_flags.get(gid).copied().unwrap_or(false),
            });
        }
        // Also include non-greedy finalizers not already covered
        for gid in state.non_greedy_finalizers.iter() {
            if !state.finalizers.iter().any(|g| g == gid) {
                state_finals.push(Finalizer {
                    gid,
                    non_greedy: true,
                });
            }
        }
        finalizers_out.push(state_finals);

        let pfg = dfa.possible_future_group_ids(si as u32);
        is_dead_end.push(pfg.is_empty());
        completion_hash.push(hash_group_list(pfg.iter()));
    }

    let none_completion_hash = {
        let mut h = new_hasher();
        h.write_u8(0);
        h.finish()
    };

    let self_loop_bytes: Vec<[u64; 4]> = (0..transitions.len())
        .map(|s| {
            let mut bits = [0u64; 4];
            for b in 0..=255u8 {
                if transitions[s][b as usize] == s as u32 {
                    bits[b as usize >> 6] |= 1u64 << (b & 63);
                }
            }
            bits
        })
        .collect();

    let start = tokenizer.start_state() as usize;
    let empty_suffix_hash = {
        let end = if is_dead_end[start] {
            STATE_NONE
        } else {
            start
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
        start_state: start,
        transitions,
        finalizers: finalizers_out,
        is_dead_end,
        num_groups,
        completion_hash,
        none_completion_hash,
        self_loop_bytes,
        empty_suffix_hash,
    }
}

// ---- Vocab Trie ----

struct TrieNode {
    children: SmallVec<[(u8, u32); 4]>,
    token_idx: u32,
    subtree_bytes: [u64; 4],
}

struct VocabTrie {
    nodes: Vec<TrieNode>,
}

impl VocabTrie {
    fn build(tokens: &[&[u8]]) -> Self {
        let total_bytes: usize = tokens.iter().map(|t| t.len()).sum();
        let mut nodes = Vec::with_capacity(total_bytes + 1);
        nodes.push(TrieNode {
            children: SmallVec::new(),
            token_idx: u32::MAX,
            subtree_bytes: [0u64; 4],
        });

        let mut root_children = [u32::MAX; 256];
        let mut d1_children = vec![[u32::MAX; 256]; 256];

        for (idx, token) in tokens.iter().enumerate() {
            let bytes = *token;
            if bytes.is_empty() {
                nodes[0].token_idx = idx as u32;
                continue;
            }

            let b0 = bytes[0] as usize;
            let mut cur = if root_children[b0] != u32::MAX {
                root_children[b0]
            } else {
                let new_idx = nodes.len() as u32;
                nodes.push(TrieNode {
                    children: SmallVec::new(),
                    token_idx: u32::MAX,
                    subtree_bytes: [0u64; 4],
                });
                root_children[b0] = new_idx;
                new_idx
            };

            if bytes.len() == 1 {
                nodes[cur as usize].token_idx = idx as u32;
                continue;
            }

            let b1 = bytes[1] as usize;
            cur = if d1_children[b0][b1] != u32::MAX {
                d1_children[b0][b1]
            } else {
                let new_idx = nodes.len() as u32;
                nodes.push(TrieNode {
                    children: SmallVec::new(),
                    token_idx: u32::MAX,
                    subtree_bytes: [0u64; 4],
                });
                d1_children[b0][b1] = new_idx;
                new_idx
            };

            if bytes.len() == 2 {
                nodes[cur as usize].token_idx = idx as u32;
                continue;
            }

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
                            subtree_bytes: [0u64; 4],
                        });
                        nodes[cur as usize].children.insert(p, (byte, new_idx));
                        new_idx
                    }
                };
            }
            nodes[cur as usize].token_idx = idx as u32;
        }

        nodes[0].children = (0..=255u8)
            .filter(|&b| root_children[b as usize] != u32::MAX)
            .map(|b| (b, root_children[b as usize]))
            .collect();

        for b0 in 0..256 {
            let parent = root_children[b0];
            if parent != u32::MAX {
                nodes[parent as usize].children = (0..=255u8)
                    .filter(|&b| d1_children[b0][b as usize] != u32::MAX)
                    .map(|b| (b, d1_children[b0][b as usize]))
                    .collect();
            }
        }

        fn compute_subtree_bytes(nodes: &mut [TrieNode], idx: u32) -> [u64; 4] {
            let mut bits = [0u64; 4];
            let num_children = nodes[idx as usize].children.len();
            for i in 0..num_children {
                let (byte, child_idx) = nodes[idx as usize].children[i];
                bits[byte as usize >> 6] |= 1u64 << (byte & 63);
                let child_bits = compute_subtree_bytes(nodes, child_idx);
                for j in 0..4 {
                    bits[j] |= child_bits[j];
                }
            }
            nodes[idx as usize].subtree_bytes = bits;
            bits
        }
        compute_subtree_bytes(&mut nodes, 0);

        VocabTrie { nodes }
    }
}

// ---- Suffix DAG ----

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

    for f in &dfa.finalizers[cur] {
        if f.gid < ng && mp[f.gid] == NONE {
            mp[f.gid] = 0;
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
        for f in &dfa.finalizers[cur] {
            if f.gid < ng {
                if !f.non_greedy || mp[f.gid] == NONE {
                    mp[f.gid] = pos;
                }
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

struct DagEntry {
    hash: u64,
    edges: EdgeList,
    generation: u32,
}

struct Scratch {
    dag: Vec<DagEntry>,
    dag_generation: u32,
    queue: Vec<usize>,
    tmp_mp: Vec<u32>,
    targets: Vec<usize>,
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
            dag_generation: 0,
            queue: Vec::new(),
            tmp_mp: vec![NONE; ng],
            targets: Vec::new(),
        }
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

fn hash_suffixes(dfa: &Dfa, slice: &[u8], scratch: &mut Scratch) {
    let len = slice.len();
    scratch.dag_generation = scratch.dag_generation.wrapping_add(1);
    scratch.queue.clear();

    if len + 2 > scratch.dag.len() {
        scratch.dag.resize_with(len + 2, || DagEntry {
            hash: 0,
            edges: EdgeList::new(),
            generation: 0,
        });
    }

    for ti in 0..scratch.targets.len() {
        let pos = scratch.targets[ti];
        if pos <= len && !scratch.dag_contains(pos) {
            scratch.queue.push(pos);
            scratch.dag[pos].hash = 0;
            scratch.dag[pos].edges.clear();
            scratch.dag[pos].generation = scratch.dag_generation;
        }
    }

    let mut cursor = 0;
    while cursor < scratch.queue.len() {
        let pos = scratch.queue[cursor];
        cursor += 1;
        let (end, edges) = run_suffix(dfa, &slice[pos..], pos, &mut scratch.tmp_mp);
        for &(_, target) in &edges {
            if target <= len && !scratch.dag_contains(target) {
                scratch.queue.push(target);
                scratch.dag[target].hash = 0;
                scratch.dag[target].edges.clear();
                scratch.dag[target].generation = scratch.dag_generation;
            }
        }
        scratch.dag[pos].hash = dfa.completion(end.unwrap_or(STATE_NONE));
        scratch.dag[pos].edges = edges;
    }

    scratch.queue.sort_unstable_by(|a, b| b.cmp(a));
    for idx in 0..scratch.queue.len() {
        let pos = scratch.queue[idx];
        let mut h = new_hasher();
        h.write_u64(scratch.dag[pos].hash);
        for ei in 0..scratch.dag[pos].edges.len() {
            let (gid, target) = scratch.dag[pos].edges[ei];
            h.write_u64(gid as u64);
            h.write_u64(scratch.dag_get_hash(target));
        }
        scratch.dag[pos].hash = h.finish();
    }
}

// ---- Core: recursive trie walk with inline signature computation ----

#[inline]
fn u8set_is_subset(a: &[u64; 4], b: &[u64; 4]) -> bool {
    (a[0] & !b[0]) == 0 && (a[1] & !b[1]) == 0 && (a[2] & !b[2]) == 0 && (a[3] & !b[3]) == 0
}

fn assign_hash_to_subtree(trie: &VocabTrie, node: u32, hash: u64, hashes: &mut [u64]) {
    let n = &trie.nodes[node as usize];
    if n.token_idx != u32::MAX {
        hashes[n.token_idx as usize] = hash;
    }
    for &(_, child) in &n.children {
        assign_hash_to_subtree(trie, child, hash, hashes);
    }
}

fn walk_trie(
    trie: &VocabTrie,
    node: u32,
    dfa: &Dfa,
    states: &mut [u32],
    mp: &mut [u32],
    depth: usize,
    ni: usize,
    ng: usize,
    max_depth: usize,
    strings: &[&[u8]],
    scratch: &mut Scratch,
    hashes: &mut [u64],
) {
    let n = &trie.nodes[node as usize];

    if n.token_idx != u32::MAX {
        let ti = n.token_idx as usize;
        let bytes = strings[ti];

        scratch.targets.clear();
        for si in 0..ni {
            let base = (depth * ni + si) * ng;
            for gid in 0..ng {
                let pv = mp[base + gid];
                if pv != NONE && pv > 0 {
                    scratch.targets.push(pv as usize);
                }
            }
        }
        scratch.targets.sort_unstable();
        scratch.targets.dedup();

        if !scratch.targets.is_empty() {
            hash_suffixes(dfa, bytes, scratch);
        }

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
    }

    for &(byte, child) in &n.children {
        let cd = depth + 1;
        if cd >= max_depth {
            continue;
        }

        for si in 0..ni {
            let ps = states[depth * ni + si];
            let parent_mp = (depth * ni + si) * ng;
            let child_mp = (cd * ni + si) * ng;

            mp.copy_within(parent_mp..parent_mp + ng, child_mp);

            if ps == NONE {
                states[cd * ni + si] = NONE;
            } else {
                let ns = dfa.transitions[ps as usize][byte as usize];
                if ns == NONE {
                    states[cd * ni + si] = NONE;
                } else {
                    let ns_u = ns as usize;
                    for f in &dfa.finalizers[ns_u] {
                        if f.gid < ng {
                            if !f.non_greedy || mp[child_mp + f.gid] == NONE {
                                mp[child_mp + f.gid] = cd as u32;
                            }
                        }
                    }
                    states[cd * ni + si] = if dfa.is_dead_end[ns_u] { NONE } else { ns };
                }
            }
        }

        let child_node = &trie.nodes[child as usize];
        if child_node.subtree_bytes != [0u64; 4] {
            let mut sl_inter = [!0u64; 4];
            let mut any_alive = false;
            for si in 0..ni {
                let cs = states[cd * ni + si];
                if cs != NONE {
                    any_alive = true;
                    let sl = &dfa.self_loop_bytes[cs as usize];
                    for i in 0..4 {
                        sl_inter[i] &= sl[i];
                    }
                }
            }

            if any_alive && u8set_is_subset(&child_node.subtree_bytes, &sl_inter) {
                let can_bulk = (0..ni).all(|si| {
                    let cs = states[cd * ni + si];
                    let base = (cd * ni + si) * ng;
                    (0..ng).all(|gid| {
                        let pv = mp[base + gid];
                        if pv > 0 && pv != NONE {
                            cs != NONE
                                && dfa.finalizers[cs as usize]
                                    .iter()
                                    .any(|f| f.gid == gid && !f.non_greedy)
                        } else {
                            true
                        }
                    })
                });

                if can_bulk {
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
                                    h.write_u64(gid as u64);
                                    h.write_u64(dfa.empty_suffix_hash);
                                }
                            }
                            h.finish()
                        } else {
                            completion
                        };
                        hash = hash.wrapping_mul(HASH_SEED1).wrapping_add(sig);
                    }

                    assign_hash_to_subtree(trie, child, hash, hashes);
                    continue;
                }
            }
        }

        walk_trie(
            trie, child, dfa, states, mp, cd, ni, ng, max_depth, strings, scratch, hashes,
        );
    }
}

// ---- Public API ----

/// Compute vocab equivalence classes using trellis-based analysis.
///
/// Groups tokens that produce identical DFA behavior across all given initial states.
/// Returns a `ManyToOneIdMap` mapping original token IDs to internal equivalence class IDs.
pub(crate) fn analyze_vocab_equivalences_trellis(
    tokenizer: &Tokenizer,
    vocab: &crate::Vocab,
    initial_states: &[u32],
) -> ManyToOneIdMap {
    let max_token_id = vocab
        .entries
        .iter()
        .map(|(token_id, _)| *token_id)
        .max()
        .unwrap_or(0);

    // Build ordered list of (token_id, bytes) and a parallel slice-of-slices for the trie.
    let ordered_entries: Vec<(u32, &[u8])> = vocab
        .entries
        .iter()
        .map(|(id, bytes)| (*id, bytes.as_slice()))
        .collect();
    let strings: Vec<&[u8]> = ordered_entries.iter().map(|(_, b)| *b).collect();

    let dfa = build_dfa(tokenizer);
    let nt = strings.len();
    let ni = initial_states.len();
    let ng = dfa.num_groups;

    if ni == 0 || nt == 0 {
        // All tokens in one class
        let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
        let mut originals = Vec::new();
        for &(token_id, _) in &ordered_entries {
            if let Some(slot) = original_to_internal.get_mut(token_id as usize) {
                *slot = 0;
            }
            originals.push(token_id);
        }
        return ManyToOneIdMap {
            original_to_internal,
            internal_to_originals: vec![originals],
        };
    }

    let trie = VocabTrie::build(&strings);
    let max_depth: usize = 256;

    let mut hashes = vec![HASH_SEED3; nt];
    let mut states_buf = vec![NONE; max_depth * ni];
    let mut mp = vec![NONE; max_depth * ni * ng];
    let max_token_len = strings.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut scratch = Scratch::new(ng, max_token_len);

    // Convert initial states from u32 to usize for the flat DFA
    let initial_states_usize: Vec<usize> = initial_states.iter().map(|&s| s as usize).collect();

    // Initialize depth 0
    for (si, &s) in initial_states_usize.iter().enumerate() {
        let mp_base = si * ng;
        for f in &dfa.finalizers[s] {
            if f.gid < ng && mp[mp_base + f.gid] == NONE {
                mp[mp_base + f.gid] = 0;
            }
        }
        states_buf[si] = if dfa.is_dead_end[s] { NONE } else { s as u32 };
    }

    walk_trie(
        &trie, 0, &dfa, &mut states_buf, &mut mp, 0, ni, ng, max_depth, &strings, &mut scratch,
        &mut hashes,
    );

    // The trie stores only one token_idx per unique byte sequence.
    // Tokens with identical bytes must share the same hash. Propagate
    // the computed hash from the trie-stored token to all duplicates.
    {
        let mut bytes_to_hash: BTreeMap<&[u8], u64> = BTreeMap::new();
        for (idx, &(_, bytes)) in ordered_entries.iter().enumerate() {
            if hashes[idx] != HASH_SEED3 {
                bytes_to_hash.entry(bytes).or_insert(hashes[idx]);
            }
        }
        for (idx, &(_, bytes)) in ordered_entries.iter().enumerate() {
            if let Some(&h) = bytes_to_hash.get(bytes) {
                hashes[idx] = h;
            }
        }
    }

    // Group tokens by hash → equivalence classes, producing ManyToOneIdMap
    let mut hash_to_internal: BTreeMap<u64, u32> = BTreeMap::new();
    let mut internal_to_originals: Vec<Vec<u32>> = Vec::new();
    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];

    for (entry_idx, &hash) in hashes.iter().enumerate() {
        let token_id = ordered_entries[entry_idx].0;

        let internal_id = if let Some(&existing) = hash_to_internal.get(&hash) {
            existing
        } else {
            let next = internal_to_originals.len() as u32;
            hash_to_internal.insert(hash, next);
            internal_to_originals.push(Vec::new());
            next
        };

        if let Some(slot) = original_to_internal.get_mut(token_id as usize) {
            *slot = internal_id;
        }
        internal_to_originals[internal_id as usize].push(token_id);
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::compile::build_tokenizer;
    use crate::compiler::grammar_def::{GrammarDef, Rule, Symbol, Terminal};
    use crate::Vocab;
    use crate::compiler::stages::equivalence_analysis::state_analysis::analyze_state_equivalences;

    #[test]
    fn test_trellis_simple_two_tokens() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal { id: 0, bytes: b"{".to_vec() },
                Terminal::Literal { id: 1, bytes: b"}".to_vec() },
            ],
            ..Default::default()
        };
        let tok = build_tokenizer(&gdef);
        let vocab = Vocab::new(
            vec![(0, b"{".to_vec()), (1, b"}".to_vec())],
            None,
        );
        let state_map = analyze_state_equivalences(&tok);
        let rep_states: Vec<u32> = state_map
            .internal_to_originals
            .iter()
            .filter_map(|o| o.first().copied())
            .collect();
        let result = analyze_vocab_equivalences_trellis(&tok, &vocab, &rep_states);
        // { and } should be in different classes
        assert_eq!(result.internal_to_originals.len(), 2);
    }

    #[test]
    fn test_trellis_identical_bytes_merge() {
        let gdef = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: vec![Terminal::Literal { id: 0, bytes: b"a".to_vec() }],
            ..Default::default()
        };
        let tok = build_tokenizer(&gdef);
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec())],
            None,
        );
        let state_map = analyze_state_equivalences(&tok);
        let rep_states: Vec<u32> = state_map
            .internal_to_originals
            .iter()
            .filter_map(|o| o.first().copied())
            .collect();
        let result = analyze_vocab_equivalences_trellis(&tok, &vocab, &rep_states);
        // tokens 0 and 1 have byte "a" -> same class; token 2 has "b" -> different class
        assert_eq!(result.internal_to_originals.len(), 2);
        // Verify token 0 and 1 are in the same class
        let class_of_0 = result.original_to_internal[0];
        let class_of_1 = result.original_to_internal[1];
        let class_of_2 = result.original_to_internal[2];
        assert_eq!(class_of_0, class_of_1);
        assert_ne!(class_of_0, class_of_2);
    }
}
