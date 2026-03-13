//! Reference equivalence analysis (strongest ground truth).
//!
//! For each (token, initial_state) pair this module:
//!
//! 1. Builds a full trellis DAG capturing group-match segmentation.
//! 2. Converts the DAG to an NFA with context-dependent states
//!    `(position, Option<parent_gid>)`.
//! 3. Replaces ignore-terminal edges with epsilon transitions (transparent).
//! 4. Encodes completion (possible future groups) and disallowed-follows
//!    pruning directly in the NFA structure.
//! 5. Determinizes → minimizes the resulting DFA.
//! 6. Computes a canonical hash of the minimal DFA (invariant to state
//!    renumbering) via recursive structural hashing.
//!
//! The per-(token, state) hashes form a matrix. Rows (across states) give
//! vocab equivalence; columns (across tokens) give state equivalence.
//!
//! **Complexity:** O(tokens × states × 2^{trellis_nodes × groups}).
//! Use only for validation on small problems.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::BuildHasher;
use std::time::{Duration, Instant};

use ahash::{AHasher, RandomState};
use once_cell::sync::Lazy;
use std::hash::Hasher;

use super::compat::{FlatDfa, Sep1Tokenizer};
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::dfa::{Label, DFA};
use crate::automata::unweighted_u32::minimize::minimize;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::ds::bitset::BitSet;

use super::state::fast::StateEquivalenceResult;
pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

// ---- Deterministic hashing ----

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

const PROGRESS_ENV: &str = "REFERENCE_EQUIV_PROGRESS";
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let t = v.trim();
            !t.is_empty() && t != "0" && !t.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

// ---- Label encoding ----
//
// NFA labels encode two kinds of information:
//   - Edge labels (group IDs from the trellis): `gid as Label` (non-negative)
//   - Completion labels (possible future groups at a node):
//     `-(gid as Label + 1)` for each future group
//     `ALIVE_MARKER` for alive nodes with no (allowed) future groups
//
// Edge labels and completion labels never overlap.

/// Marker label for a live DFA state that has no possible future groups.
const ALIVE_MARKER: Label = i32::MIN;

#[inline]
fn completion_label(gid: usize) -> Label {
    -(gid as Label + 1)
}

// ---- Precomputed data (derived from FlatDfa, reused across all tokens) ----

struct PrecomputedData {
    num_groups: usize,
    disallowed_follows: Vec<BitSet>,
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

fn precompute(dfa: &FlatDfa, disallowed_follows: &BTreeMap<u32, BitSet>) -> PrecomputedData {
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

    PrecomputedData {
        num_groups,
        disallowed_follows: normalize_disallowed_follows(num_groups, disallowed_follows),
    }
}

// ---- Trellis DAG construction ----

type Edge = (usize, usize); // (group_id, target_position)

struct FlatNode {
    end_state: usize,
    edges: Vec<Edge>,
}

/// Walk the tokenizer DFA from `start_state` on `slice`, recording the
/// last-match position per group. Returns the end state (or STATE_NONE if
/// the DFA reaches a dead end).
fn walk_tokenizer_dfa(
    dfa: &FlatDfa,
    ng: usize,
    slice: &[u8],
    start_state: usize,
    mp: &mut [u32],
) -> usize {
    mp[..ng].fill(NONE);
    let mut cur = start_state;
    let mut done = dfa.states[cur].possible_future_group_ids.is_empty();

    for &gid in &dfa.states[cur].finalizers {
        if gid < ng && mp[gid] == NONE {
            mp[gid] = 0;
        }
    }

    for (i, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }
        let ns = dfa.states[cur].transitions[byte as usize];
        if ns == NONE {
            done = true;
            break;
        }
        cur = ns as usize;
        let pos = (i + 1) as u32;
        for &gid in &dfa.states[cur].finalizers {
            if gid < ng {
                mp[gid] = pos;
            }
        }
        if dfa.states[cur].possible_future_group_ids.is_empty() {
            done = true;
        }
    }

    if done { STATE_NONE } else { cur }
}

fn edges_from_mp(mp: &[u32], ng: usize, base_pos: usize) -> Vec<Edge> {
    (0..ng)
        .filter_map(|gid| {
            let pv = mp[gid];
            (pv != NONE && pv > 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect()
}

fn build_trellis_dag(
    dfa: &FlatDfa,
    ng: usize,
    token: &[u8],
    initial_state: usize,
    tmp_mp: &mut Vec<u32>,
) -> BTreeMap<usize, FlatNode> {
    let len = token.len();
    tmp_mp.resize(ng, NONE);

    let root_end = walk_tokenizer_dfa(dfa, ng, token, initial_state, tmp_mp);
    let root_edges = edges_from_mp(tmp_mp, ng, 0);

    let mut dag: BTreeMap<usize, FlatNode> = BTreeMap::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    dag.insert(0, FlatNode { end_state: root_end, edges: root_edges.clone() });

    for &(_, pos) in &root_edges {
        if pos <= len && !dag.contains_key(&pos) {
            queue.push_back(pos);
            dag.insert(pos, FlatNode { end_state: STATE_NONE, edges: Vec::new() });
        }
    }

    while let Some(pos) = queue.pop_front() {
        let end = walk_tokenizer_dfa(dfa, ng, &token[pos..], dfa.start_state, tmp_mp);
        let edges = edges_from_mp(tmp_mp, ng, pos);

        for &(_, target) in &edges {
            if target <= len && !dag.contains_key(&target) {
                queue.push_back(target);
                dag.insert(target, FlatNode { end_state: STATE_NONE, edges: Vec::new() });
            }
        }

        let node = dag.get_mut(&pos).unwrap();
        node.end_state = end;
        node.edges = edges;
    }

    dag
}

fn prune_reachable(dag: &mut BTreeMap<usize, FlatNode>, token_len: usize) {
    let mut reverse_edges: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&src, node) in dag.iter() {
        for &(_, dst) in &node.edges {
            reverse_edges.entry(dst).or_default().push(src);
        }
    }

    let mut can_reach_end: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut stack: Vec<usize> = Vec::new();
    if dag.contains_key(&token_len) {
        can_reach_end.insert(token_len);
        stack.push(token_len);
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

    for (_pos, node) in dag.iter_mut() {
        node.edges.retain(|&(_, target)| can_reach_end.contains(&target));
    }
}

// ---- NFA construction from trellis DAG ----

/// NFA state key: (position_in_token, Option<parent_group_that_led_here>)
type NfaStateKey = (usize, Option<usize>);

fn build_nfa_from_trellis(
    dfa: &FlatDfa,
    pre: &PrecomputedData,
    dag: &BTreeMap<usize, FlatNode>,
    ignore_terminal: Option<usize>,
) -> NFA {
    let mut nfa = NFA::new_empty();
    let mut state_map: HashMap<NfaStateKey, u32> = HashMap::new();
    let mut worklist: VecDeque<NfaStateKey> = VecDeque::new();

    // Global accepting sink for completion transitions
    let accept_sink = nfa.add_state();
    nfa.set_accepting(accept_sink);

    let mut get_or_create = |nfa: &mut NFA,
                             state_map: &mut HashMap<NfaStateKey, u32>,
                             worklist: &mut VecDeque<NfaStateKey>,
                             key: NfaStateKey|
     -> u32 {
        if let Some(&id) = state_map.get(&key) {
            id
        } else {
            let id = nfa.add_state();
            state_map.insert(key, id);
            worklist.push_back(key);
            id
        }
    };

    // Root: position 0, no parent context
    let root_key: NfaStateKey = (0, None);
    let root_id = get_or_create(&mut nfa, &mut state_map, &mut worklist, root_key);
    nfa.start_states = vec![root_id];

    while let Some(key) = worklist.pop_front() {
        let (pos, parent_gid) = key;
        let nfa_state = state_map[&key];

        let node = match dag.get(&pos) {
            Some(n) => n,
            None => continue,
        };

        let disallowed = parent_gid.map(|pgid| &pre.disallowed_follows[pgid]);

        // --- Completion transitions ---
        if node.end_state != STATE_NONE && node.end_state < dfa.states.len() {
            let future_groups = &dfa.states[node.end_state].possible_future_group_ids;
            let mut has_any = false;
            for &gid in future_groups {
                if let Some(d) = disallowed {
                    if d.contains(gid) {
                        continue;
                    }
                }
                has_any = true;
                nfa.add_transition(nfa_state, completion_label(gid), accept_sink);
            }
            if !has_any {
                nfa.add_transition(nfa_state, ALIVE_MARKER, accept_sink);
            }
        }

        // --- Edge transitions ---
        for &(gid, target_pos) in &node.edges {
            if let Some(d) = disallowed {
                if d.contains(gid) {
                    continue;
                }
            }

            let is_ignore = ignore_terminal.map_or(false, |ig| gid == ig);

            if is_ignore {
                // Epsilon: ignore terminal is transparent, inherit parent context
                let child_key: NfaStateKey = (target_pos, parent_gid);
                let child_id = get_or_create(&mut nfa, &mut state_map, &mut worklist, child_key);
                nfa.add_epsilon(nfa_state, child_id);
            } else {
                // Labeled transition: child gets this gid as context
                let child_key: NfaStateKey = (target_pos, Some(gid));
                let child_id = get_or_create(&mut nfa, &mut state_map, &mut worklist, child_key);
                nfa.add_transition(nfa_state, gid as Label, child_id);
            }
        }
    }

    nfa
}

// ---- Canonical hash of minimized DFA ----

fn canonical_hash(dfa: &DFA) -> u64 {
    if dfa.states.is_empty() {
        let mut h = new_hasher();
        h.write_u8(0);
        return h.finish();
    }
    let mut memo: HashMap<u32, u64> = HashMap::new();
    hash_dfa_state(dfa, dfa.start_state, &mut memo)
}

fn hash_dfa_state(dfa: &DFA, state: u32, memo: &mut HashMap<u32, u64>) -> u64 {
    if let Some(&cached) = memo.get(&state) {
        return cached;
    }
    let s = &dfa.states[state as usize];
    let mut h = new_hasher();
    h.write_u8(if s.is_accepting { 1 } else { 0 });
    h.write_u64(s.transitions.len() as u64);
    // BTreeMap iterates in sorted label order — deterministic
    for (&label, &target) in &s.transitions {
        h.write_i32(label);
        h.write_u64(hash_dfa_state(dfa, target, memo));
    }
    let result = h.finish();
    memo.insert(state, result);
    result
}

// ---- Per-(token, state) processing ----

fn process_token_for_state(
    dfa: &FlatDfa,
    pre: &PrecomputedData,
    token: &[u8],
    initial_state: usize,
    ignore_terminal: Option<usize>,
    tmp_mp: &mut Vec<u32>,
) -> u64 {
    tmp_mp.resize(pre.num_groups, NONE);

    let mut dag = build_trellis_dag(dfa, pre.num_groups, token, initial_state, tmp_mp);
    prune_reachable(&mut dag, token.len());

    let nfa = build_nfa_from_trellis(dfa, pre, &dag, ignore_terminal);
    let det_dfa = determinize(&nfa);
    let min_dfa = minimize(&det_dfa);
    canonical_hash(&min_dfa)
}

// ---- Public API ----

/// Result of reference equivalence analysis.
pub struct ReferenceEquivalenceResult {
    pub vocab_classes: VocabEquivalenceResult,
    pub state_classes: StateEquivalenceResult,
}

pub fn find_equivalence_classes<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<usize>,
) -> ReferenceEquivalenceResult {
    let dfa = regex.dfa();
    let pre = precompute(dfa, disallowed_follows);
    let nt = strings.len();
    let ns = initial_states.len();

    if ns == 0 || nt == 0 {
        let vocab_classes = BTreeSet::from_iter(vec![(0..nt).collect()]);
        let state_classes: StateEquivalenceResult = initial_states
            .iter()
            .map(|&s| std::iter::once(s).collect())
            .collect();
        return ReferenceEquivalenceResult { vocab_classes, state_classes };
    }

    let show_progress = env_flag_enabled(PROGRESS_ENV);
    let started = Instant::now();
    let mut last_report = started;

    let mut tmp_mp = vec![NONE; pre.num_groups];

    // Compute per-(token, state) hash matrix
    // Layout: hashes[ti * ns + si] = hash for token ti at state initial_states[si]
    let mut hashes: Vec<u64> = Vec::with_capacity(nt * ns);

    for (ti, token_ref) in strings.iter().enumerate() {
        let token = token_ref.as_ref();
        for &state in initial_states {
            let sig = process_token_for_state(dfa, &pre, token, state, ignore_terminal, &mut tmp_mp);
            hashes.push(sig);
        }

        if show_progress {
            let now = Instant::now();
            if now.duration_since(last_report) >= PROGRESS_INTERVAL || ti + 1 == nt {
                eprintln!(
                    "[reference equiv] processed {}/{} tokens in {:.1}s",
                    ti + 1,
                    nt,
                    now.duration_since(started).as_secs_f64(),
                );
                last_report = now;
            }
        }
    }

    // --- Vocab equivalence: combine per-state hashes for each token ---
    let mut vocab_groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    for ti in 0..nt {
        let row: Vec<u64> = (0..ns).map(|si| hashes[ti * ns + si]).collect();
        vocab_groups.entry(row).or_default().push(ti);
    }
    let vocab_classes: VocabEquivalenceResult = vocab_groups.into_values().collect();

    // --- State equivalence: combine per-token hashes for each state ---
    let mut state_groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    for (si, &state) in initial_states.iter().enumerate() {
        let col: Vec<u64> = (0..nt).map(|ti| hashes[ti * ns + si]).collect();
        state_groups.entry(col).or_default().push(state);
    }
    let state_classes: StateEquivalenceResult = state_groups
        .into_values()
        .map(|states| states.into_iter().collect::<BTreeSet<usize>>())
        .collect();

    ReferenceEquivalenceResult { vocab_classes, state_classes }
}
