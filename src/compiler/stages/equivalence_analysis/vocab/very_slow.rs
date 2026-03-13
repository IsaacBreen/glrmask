//! Trellis-based vocab equivalence analysis (very slow ground truth).
//!
//! For each token × initial-state pair this module builds a full suffix DAG
//! capturing group-match segmentation of the token, applies disallowed-terminal
//! pruning, then hashes bottom-up. Tokens whose combined hashes across all
//! initial states are identical land in the same equivalence class.
//!
//! This is deliberately simple: no trie, no batching, no self-loop optimization.
//! Each token is processed independently.
//!
//! **Complexity:** O(tokens × states × suffix_dag_size_per_token).
//! Use only for validation / small problems.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::BuildHasher;
use std::time::{Duration, Instant};

use ahash::{AHasher, RandomState};
use once_cell::sync::Lazy;
use std::hash::Hasher;

use super::super::compat::{GroupID, Sep1Tokenizer};
use crate::ds::bitset::BitSet;

pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

// ---- Deterministic hashing (same seeds as slow.rs) ----

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;

static HASH_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_STATE.build_hasher()
}

const PROGRESS_ENV: &str = "VERY_SLOW_VOCAB_EQUIV_PROGRESS";
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let t = v.trim();
            !t.is_empty() && t != "0" && !t.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

// ---- Flat DFA (extracted once, reused for all tokens) ----

struct Dfa {
    start_state: usize,
    transitions: Vec<[u32; 256]>,
    finalizers: Vec<Vec<usize>>,
    is_dead_end: Vec<bool>,
    num_groups: usize,
    possible_future_groups: Vec<Vec<usize>>,
    completion_hash: Vec<u64>,
    none_completion_hash: u64,
    disallowed_follows: Vec<BitSet>,
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

#[inline]
fn hash_filtered_group_list(groups: &[usize], disallowed: &BitSet) -> u64 {
    let mut h = new_hasher();
    h.write_u8(1);
    let count = groups.iter().filter(|&&gid| !disallowed.contains(gid)).count();
    h.write_u64(count as u64);
    for &gid in groups {
        if !disallowed.contains(gid) {
            h.write_u64(gid as u64);
        }
    }
    h.finish()
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
        hash_filtered_group_list(&self.possible_future_groups[state], disallowed)
    }

    #[inline]
    fn disallowed_for(&self, gid: usize) -> &BitSet {
        &self.disallowed_follows[gid]
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

fn build_dfa(regex: &Sep1Tokenizer, disallowed_follows: &BTreeMap<u32, BitSet>) -> Dfa {
    let dfa = regex.dfa();

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
        finalizers.push(state.finalizers.clone());
        is_dead_end.push(state.possible_future_group_ids.is_empty());
        let future_groups = state.possible_future_group_ids.clone();
        completion_hash.push(hash_group_list(future_groups.iter().copied()));
        possible_future_groups.push(future_groups);
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
        possible_future_groups,
        completion_hash,
        none_completion_hash,
        disallowed_follows: normalize_disallowed_follows(num_groups, disallowed_follows),
    }
}

// ---- Per-token suffix DAG ----

type Edge = (usize, usize);

/// Flat DAG node: end_state + edges. Disallowed and hash computed separately.
struct FlatNode {
    end_state: usize,
    edges: Vec<Edge>,
}

/// Walk the DFA from `start_state` on `slice`, returning end state and
/// last-match-position per group in `mp`. Matches slow.rs semantics exactly.
fn walk_dfa(
    dfa: &Dfa,
    slice: &[u8],
    start_state: usize,
    mp: &mut [u32],
) -> usize {
    let ng = dfa.num_groups;
    mp[..ng].fill(NONE);
    let mut cur = start_state;
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

    if done {
        STATE_NONE
    } else {
        cur
    }
}

/// Extract edges from mp: (gid, base_pos + match_pos) for groups with pv > 0.
fn edges_from_mp(mp: &[u32], ng: usize, base_pos: usize) -> Vec<Edge> {
    (0..ng)
        .filter_map(|gid| {
            let pv = mp[gid];
            (pv != NONE && pv > 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect()
}

/// Build suffix DAG, apply reachability pruning, and hash using recursive
/// tree-based disallowed-follows propagation. Matches the grammars2024
/// trellis_equivalence_analysis approach.
fn hash_token_for_state(dfa: &Dfa, token: &[u8], initial_state: usize, tmp_mp: &mut Vec<u32>) -> u64 {
    let ng = dfa.num_groups;
    let len = token.len();
    tmp_mp.resize(ng, NONE);

    // ---- Root walk: DFA from initial_state on full token ----
    let root_end = walk_dfa(dfa, token, initial_state, tmp_mp);
    let root_edges = edges_from_mp(tmp_mp, ng, 0);

    // ---- BFS to build flat DAG ----
    let mut dag: BTreeMap<usize, FlatNode> = BTreeMap::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    // Root node at position 0
    dag.insert(0, FlatNode {
        end_state: root_end,
        edges: root_edges.clone(),
    });

    for &(_, pos) in &root_edges {
        if pos <= len && !dag.contains_key(&pos) {
            queue.push_back(pos);
            dag.insert(pos, FlatNode {
                end_state: STATE_NONE,
                edges: Vec::new(),
            });
        }
    }

    while let Some(pos) = queue.pop_front() {
        let end = walk_dfa(dfa, &token[pos..], dfa.start_state, tmp_mp);
        let edges = edges_from_mp(tmp_mp, ng, pos);

        for &(_, target) in &edges {
            if target <= len && !dag.contains_key(&target) {
                queue.push_back(target);
                dag.insert(target, FlatNode {
                    end_state: STATE_NONE,
                    edges: Vec::new(),
                });
            }
        }

        let node = dag.get_mut(&pos).unwrap();
        node.end_state = end;
        node.edges = edges;
    }

    // ---- Reachability pruning: only keep nodes/edges that can reach token end ----
    // A node at position `len` (end of token) is always reachable.
    // Walk backward through edges to find all reachable nodes.
    let mut reverse_edges: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&src, node) in &dag {
        for &(_, dst) in &node.edges {
            reverse_edges.entry(dst).or_default().push(src);
        }
    }

    let mut can_reach_end: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack: Vec<usize> = Vec::new();
    if dag.contains_key(&len) {
        can_reach_end.insert(len);
        stack.push(len);
    }
    while let Some(pos) = stack.pop() {
        if let Some(preds) = reverse_edges.get(&pos) {
            for &pred in preds {
                if can_reach_end.insert(pred) {
                    stack.push(pred);
                }
            }
        }
    }

    // Prune edges that don't reach end
    for (_pos, node) in dag.iter_mut() {
        node.edges.retain(|&(_, target)| can_reach_end.contains(&target));
    }

    // If root can't reach end, produce a degenerate hash
    if !can_reach_end.contains(&0) {
        // No valid parse path through the whole token
        let mut h = new_hasher();
        h.write_u8(0); // None end_state marker
        return h.finish();
    }

    // ---- Recursive tree-based hashing with per-parent disallowed context ----
    // This matches grammars2024's prune_trellis_disallowed_follows + hash_trellis:
    // at each level, only edges whose gid is NOT disallowed by the parent are kept,
    // and the disallowed context for each child depends on the edge's gid.
    fn hash_recursive(
        dfa: &Dfa,
        dag: &BTreeMap<usize, FlatNode>,
        pos: usize,
        parent_disallowed: Option<&BitSet>,
        memo: &mut HashMap<(usize, Option<u64>), u64>,
    ) -> u64 {
        // Memo key: (position, hash of parent_disallowed for context)
        let dis_key = parent_disallowed.map(|d| {
            let mut h = new_hasher();
            for bit in d.iter() {
                h.write_u64(bit as u64);
            }
            h.finish()
        });
        if let Some(&cached) = memo.get(&(pos, dis_key)) {
            return cached;
        }

        let node = match dag.get(&pos) {
            Some(n) => n,
            None => {
                let mut h = new_hasher();
                h.write_u8(0);
                return h.finish();
            }
        };

        let mut h = new_hasher();

        // Hash end_state (completion): possible_future_group_ids, filtered by disallowed
        h.write_u64(dfa.completion_with_disallowed(node.end_state, parent_disallowed));

        // Hash edges: only those not disallowed by parent
        let edge_count = node.edges.iter()
            .filter(|&&(gid, _)| {
                parent_disallowed.map_or(true, |d| !d.contains(gid))
            })
            .count();
        h.write_u64(edge_count as u64);

        for &(gid, target) in &node.edges {
            if let Some(d) = parent_disallowed {
                if d.contains(gid) {
                    continue;
                }
            }
            h.write_u64(gid as u64);
            // Child's disallowed context comes from THIS edge's gid
            let child_disallowed = dfa.disallowed_for(gid);
            let child_dis = if child_disallowed.is_zero() {
                None
            } else {
                Some(child_disallowed)
            };
            let child_hash = hash_recursive(dfa, dag, target, child_dis, memo);
            h.write_u64(child_hash);
        }

        let result = h.finish();
        memo.insert((pos, dis_key), result);
        result
    }

    let mut memo: HashMap<(usize, Option<u64>), u64> = HashMap::new();
    // Root: no parent disallowed context (all edges allowed at root)
    hash_recursive(dfa, &dag, 0, None, &mut memo)
}

// ---- Public API ----

pub fn find_vocab_equivalence_classes_with_follow<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> VocabEquivalenceResult {
    let dfa = build_dfa(regex, disallowed_follows);
    let nt = strings.len();
    let ns = initial_states.len();

    if ns == 0 || nt == 0 {
        return BTreeSet::from_iter(vec![(0..nt).collect()]);
    }

    let show_progress = env_flag_enabled(PROGRESS_ENV);
    let started = Instant::now();
    let mut last_report = started;

    let mut tmp_mp = vec![NONE; dfa.num_groups];
    let mut hashes: Vec<u64> = Vec::with_capacity(nt);

    for (ti, token_ref) in strings.iter().enumerate() {
        let token = token_ref.as_ref();

        let mut combined = HASH_SEED3;
        for &state in initial_states {
            let sig = hash_token_for_state(&dfa, token, state, &mut tmp_mp);
            combined = combined.wrapping_mul(HASH_SEED1).wrapping_add(sig);
        }

        hashes.push(combined);

        if show_progress {
            let now = Instant::now();
            if now.duration_since(last_report) >= PROGRESS_INTERVAL || ti + 1 == nt {
                eprintln!(
                    "[very_slow vocab equiv] processed {}/{} tokens in {:.1}s",
                    ti + 1,
                    nt,
                    now.duration_since(started).as_secs_f64(),
                );
                last_report = now;
            }
        }
    }

    let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
    for (ti, h) in hashes.into_iter().enumerate() {
        groups.entry(h).or_default().push(ti);
    }

    groups.into_values().collect()
}
