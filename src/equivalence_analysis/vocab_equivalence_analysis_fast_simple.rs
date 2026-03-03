//! Simplified fast vocab equivalence analysis.
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

use crate::dfa_u8::Tokenizer;
use ahash::{AHasher, RandomState};
use hashbrown::HashMap;
use once_cell::sync::Lazy;
use smallvec::SmallVec;
use std::collections::BTreeSet;
use std::hash::{BuildHasher, Hasher};

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

fn build_dfa(regex: &Tokenizer) -> Dfa {
    let dfa = regex.dfa();
    assert!(dfa.states.len() <= u32::MAX as usize, "DFA too large");

    let num_groups = dfa
        .states
        .iter()
        .flat_map(|s| {
            s.finalizers
                .iter()
                .chain(s.possible_future_group_ids.iter().copied())
        })
        .chain(dfa.non_greedy_finalizers.iter().copied())
        .max()
        .map_or(0, |m| m + 1);

    let mut non_greedy_flags = vec![false; num_groups];
    for &gid in &dfa.non_greedy_finalizers {
        if gid < num_groups {
            non_greedy_flags[gid] = true;
        }
    }

    let mut transitions = Vec::with_capacity(dfa.states.len());
    let mut finalizers = Vec::with_capacity(dfa.states.len());
    let mut is_dead_end = Vec::with_capacity(dfa.states.len());
    let mut completion_hash = Vec::with_capacity(dfa.states.len());

    for state in &dfa.states {
        let mut table = [NONE; 256];
        for (byte, &target) in state.transitions.iter() {
            table[byte as usize] = target as u32;
        }
        transitions.push(table);

        finalizers.push(
            state
                .finalizers
                .iter()
                .map(|gid| Finalizer {
                    gid,
                    non_greedy: non_greedy_flags.get(gid).copied().unwrap_or(false),
                })
                .collect(),
        );

        is_dead_end.push(state.possible_future_group_ids.is_empty());
        completion_hash.push(hash_group_list(
            state.possible_future_group_ids.iter().copied(),
        ));
    }

    let none_completion_hash = {
        let mut h = new_hasher();
        h.write_u8(0);
        h.finish()
    };

    Dfa {
        start_state: dfa.start_state,
        transitions,
        finalizers,
        is_dead_end,
        num_groups,
        completion_hash,
        none_completion_hash,
    }
}

// ---- Vocab Trie ----

struct TrieNode {
    children: SmallVec<[(u8, u32); 4]>,
    token_idx: u32, // u32::MAX if not a token endpoint
}

struct VocabTrie {
    nodes: Vec<TrieNode>,
}

impl VocabTrie {
    fn build<S: AsRef<[u8]>>(tokens: &[S]) -> Self {
        let mut nodes = vec![TrieNode {
            children: SmallVec::new(),
            token_idx: u32::MAX,
        }];

        for (idx, token) in tokens.iter().enumerate() {
            let mut cur = 0u32;
            for &byte in token.as_ref() {
                let pos = nodes[cur as usize]
                    .children
                    .iter()
                    .position(|&(b, _)| b == byte);
                cur = match pos {
                    Some(p) => nodes[cur as usize].children[p].1,
                    None => {
                        let new_idx = nodes.len() as u32;
                        nodes.push(TrieNode {
                            children: SmallVec::new(),
                            token_idx: u32::MAX,
                        });
                        nodes[cur as usize].children.push((byte, new_idx));
                        new_idx
                    }
                };
            }
            nodes[cur as usize].token_idx = idx as u32;
        }

        // Sort children for deterministic traversal
        for node in &mut nodes {
            node.children.sort_unstable_by_key(|&(b, _)| b);
        }

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

/// Build suffix DAG via BFS from target positions and hash bottom-up.
fn hash_suffixes(
    dfa: &Dfa,
    slice: &[u8],
    targets: &[usize],
    dag: &mut HashMap<usize, (u64, EdgeList)>,
    queue: &mut Vec<usize>,
    tmp_mp: &mut Vec<u32>,
) {
    let len = slice.len();
    dag.clear();
    queue.clear();

    for &pos in targets {
        if pos <= len && !dag.contains_key(&pos) {
            queue.push(pos);
            dag.insert(pos, (0, EdgeList::new()));
        }
    }

    let mut cursor = 0;
    while cursor < queue.len() {
        let pos = queue[cursor];
        cursor += 1;
        let (end, edges) = run_suffix(dfa, &slice[pos..], pos, tmp_mp);
        for &(_, target) in &edges {
            if target <= len && !dag.contains_key(&target) {
                queue.push(target);
                dag.insert(target, (0, EdgeList::new()));
            }
        }
        dag.insert(pos, (dfa.completion(end.unwrap_or(STATE_NONE)), edges));
    }

    queue.sort_unstable_by(|a, b| b.cmp(a));
    for idx in 0..queue.len() {
        let pos = queue[idx];
        let (ch, edges) = dag[&pos].clone();
        let mut h = new_hasher();
        h.write_u64(ch);
        for &(gid, target) in &edges {
            h.write_u64(gid as u64);
            h.write_u64(dag.get(&target).map_or(0, |e| e.0));
        }
        dag.get_mut(&pos).unwrap().0 = h.finish();
    }
}

// ---- Scratch workspace ----

struct Scratch {
    dag: HashMap<usize, (u64, EdgeList)>,
    queue: Vec<usize>,
    tmp_mp: Vec<u32>,
    targets: Vec<usize>,
}

impl Scratch {
    fn new(ng: usize) -> Self {
        Scratch {
            dag: HashMap::new(),
            queue: Vec::new(),
            tmp_mp: vec![NONE; ng],
            targets: Vec::new(),
        }
    }
}

// ---- Core: recursive trie walk with inline signature computation ----

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
) {
    let n = &trie.nodes[node as usize];

    // At token leaf: compute signature
    if n.token_idx != u32::MAX {
        let ti = n.token_idx as usize;
        let bytes = strings[ti].as_ref();

        // Collect suffix targets across all initial states
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
            hash_suffixes(
                dfa,
                bytes,
                &scratch.targets,
                &mut scratch.dag,
                &mut scratch.queue,
                &mut scratch.tmp_mp,
            );
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
                        h.write_u64(
                            scratch.dag.get(&(pv as usize)).map_or(0, |e| e.0),
                        );
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

        walk_trie(
            trie, child, dfa, states, mp, cd, ni, ng, max_depth, strings, scratch, hashes,
        );
    }
}

// ---- Public API ----

pub fn find_vocab_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
) -> VocabEquivalenceResult {
    find_vocab_equivalence_classes_with_follow(regex, strings, initial_states, None, None, None)
}

/// Find vocab equivalence classes. The last three parameters are accepted for
/// API compatibility but unused.
pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    _suffix_group_mask: Option<&[bool]>,
    _ever_allowed_by_group: Option<&[Vec<bool>]>,
    _group_to_class: Option<&[usize]>,
) -> VocabEquivalenceResult {
    let dfa = build_dfa(regex);
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
    let mut scratch = Scratch::new(ng);

    // Initialize depth 0: set initial DFA states and their finalizers
    for (si, &s) in initial_states.iter().enumerate() {
        let mp_base = si * ng;
        for f in &dfa.finalizers[s] {
            if f.gid < ng && mp[mp_base + f.gid] == NONE {
                mp[mp_base + f.gid] = 0;
            }
        }
        states[si] = if dfa.is_dead_end[s] { NONE } else { s as u32 };
    }

    walk_trie(
        &trie, 0, &dfa, &mut states, &mut mp, 0, ni, ng, max_depth, strings, &mut scratch,
        &mut hashes,
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
    finer.iter().all(|fc| {
        coarser
            .iter()
            .any(|cc| fc.iter().all(|t| cc.contains(t)))
    })
}

/// Returns true if one partition refines the other (or they are equal).
pub fn partitions_are_comparable(
    a: &VocabEquivalenceResult,
    b: &VocabEquivalenceResult,
) -> bool {
    partition_is_at_least_as_fine(a, b) || partition_is_at_least_as_fine(b, a)
}

/// Returns true if both partitions have identical classes.
pub fn partitions_are_equivalent(
    a: &VocabEquivalenceResult,
    b: &VocabEquivalenceResult,
) -> bool {
    a == b
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

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
