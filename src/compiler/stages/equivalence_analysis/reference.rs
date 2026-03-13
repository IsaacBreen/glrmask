//! Reference equivalence analysis (strongest ground truth).
//!
//! For each (token, initial_state) pair this module:
//!
//! 1. Builds a full trellis DAG capturing group-match segmentation.
//! 2. Converts the DAG to an NFA with context-dependent states
//!    `(position, Option<parent_gid>)`.
//! 3. Replaces ignore-terminal edges with epsilon transitions (transparent).
//! 4. Encodes completion (possible future groups) directly in the NFA.
//! 5. Determinizes → minimizes the resulting DFA.
//! 6. Subtracts a precomputed disallowed-follow DFA and minimizes again.
//! 7. Computes a canonical hash of the minimal DFA (invariant to state
//!    renumbering) via recursive structural hashing.
//!
//! The per-(token, state) hashes form a matrix. Rows (across states) give
//! vocab equivalence; columns (across tokens) give state equivalence.
//!
//! **Complexity:** O(tokens × states × 2^{trellis_nodes × groups}).
//! Use only for validation on small problems.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ahash::{AHasher, RandomState};
use once_cell::sync::Lazy;
use std::hash::Hasher;

use super::compat::{FlatDfa, FlatDfaState, Sep1Tokenizer};
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::dfa::{Label, DFA};
use crate::automata::unweighted_u32::minimize::minimize;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::automata::unweighted_u32::subtract::subtract;
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
//
// Edge labels and completion labels never overlap.

#[inline]
fn completion_label(gid: usize) -> Label {
    -(gid as Label + 1)
}

#[inline]
fn terminal_label(gid: usize) -> Label {
    gid as Label
}

#[inline]
fn future_groups_cover_all_terminals(future_groups: &[usize], num_groups: usize) -> bool {
    future_groups.len() == num_groups && future_groups.iter().copied().eq(0..num_groups)
}

// ---- Precomputed data (derived from FlatDfa, reused across all tokens) ----

struct PrecomputedData {
    num_groups: usize,
    disallowed_detector: Option<DFA>,
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

    let normalized = normalize_disallowed_follows(num_groups, disallowed_follows);
    let disallowed_detector = normalized
        .iter()
        .any(|bits| !bits.is_zero())
        .then(|| build_disallowed_follow_dfa(&normalized));

    PrecomputedData {
        num_groups,
        disallowed_detector,
    }
}

fn build_disallowed_follow_dfa(disallowed_follows: &[BitSet]) -> DFA {
    let num_groups = disallowed_follows.len();
    if num_groups == 0 {
        return DFA::new();
    }

    let mut dfa = DFA::new();
    let start = dfa.start_state;
    let accept = dfa.add_state();
    dfa.set_accepting(accept, true);

    let mut previous_terminal_states = Vec::with_capacity(num_groups);
    for _ in 0..num_groups {
        previous_terminal_states.push(dfa.add_state());
    }

    for prev_gid in 0..num_groups {
        let prev_state = previous_terminal_states[prev_gid];
        dfa.add_transition(start, terminal_label(prev_gid), prev_state);

        for next_gid in 0..num_groups {
            let target = if disallowed_follows[prev_gid].contains(next_gid) {
                accept
            } else {
                previous_terminal_states[next_gid]
            };
            dfa.add_transition(prev_state, terminal_label(next_gid), target);
        }
    }

    for gid in 0..num_groups {
        dfa.add_transition(accept, terminal_label(gid), accept);
    }

    minimize(&dfa)
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

// ---- NFA construction from trellis DAG ----

fn build_nfa_from_trellis(
    dfa: &FlatDfa,
    dag: &BTreeMap<usize, FlatNode>,
    num_groups: usize,
    ignore_terminal: Option<usize>,
) -> NFA {
    let mut nfa = NFA::new_empty();
    let mut state_map: HashMap<usize, u32> = HashMap::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();

    // Global accepting sink for completion transitions
    let accept_sink = nfa.add_state();
    nfa.set_accepting(accept_sink);

    let mut get_or_create = |nfa: &mut NFA,
                             state_map: &mut HashMap<usize, u32>,
                             worklist: &mut VecDeque<usize>,
                             pos: usize|
     -> u32 {
        if let Some(&id) = state_map.get(&pos) {
            id
        } else {
            let id = nfa.add_state();
            state_map.insert(pos, id);
            worklist.push_back(pos);
            id
        }
    };

    // Root: position 0
    let root_id = get_or_create(&mut nfa, &mut state_map, &mut worklist, 0);
    nfa.start_states = vec![root_id];

    while let Some(pos) = worklist.pop_front() {
        let nfa_state = state_map[&pos];

        let node = match dag.get(&pos) {
            Some(n) => n,
            None => continue,
        };

        // --- Completion transitions (no disallowed filtering) ---
        if node.end_state != STATE_NONE && node.end_state < dfa.states.len() {
            let future_groups = &dfa.states[node.end_state].possible_future_group_ids;
            if future_groups_cover_all_terminals(future_groups, num_groups) {
                nfa.add_epsilon(nfa_state, accept_sink);
            } else {
                for &gid in future_groups {
                    nfa.add_transition(nfa_state, completion_label(gid), accept_sink);
                }
            }
        }

        // --- Edge transitions (no disallowed filtering) ---
        for &(gid, target_pos) in &node.edges {
            let is_ignore = ignore_terminal.map_or(false, |ig| gid == ig);

            if is_ignore {
                // Epsilon: ignore terminal is transparent
                let child_id = get_or_create(&mut nfa, &mut state_map, &mut worklist, target_pos);
                nfa.add_epsilon(nfa_state, child_id);
            } else {
                // Labeled transition
                let child_id = get_or_create(&mut nfa, &mut state_map, &mut worklist, target_pos);
                nfa.add_transition(nfa_state, gid as Label, child_id);
            }
        }
    }

    nfa
}

// ---- Canonical hash of minimized DFA ----

fn canonical_nfa_hash(nfa: &NFA) -> u64 {
    if nfa.states.is_empty() {
        let mut h = new_hasher();
        h.write_u8(0);
        return h.finish();
    }

    let mut memo: HashMap<u32, u64> = HashMap::new();
    let mut start_hashes: Vec<u64> = nfa
        .start_states
        .iter()
        .map(|&state| hash_nfa_state(nfa, state, &mut memo))
        .collect();
    start_hashes.sort_unstable();

    let mut h = new_hasher();
    h.write_u64(start_hashes.len() as u64);
    for state_hash in start_hashes {
        h.write_u64(state_hash);
    }
    h.finish()
}

fn hash_nfa_state(nfa: &NFA, state: u32, memo: &mut HashMap<u32, u64>) -> u64 {
    if let Some(&cached) = memo.get(&state) {
        return cached;
    }

    let s = &nfa.states[state as usize];
    let mut h = new_hasher();
    h.write_u8(if s.is_accepting { 1 } else { 0 });

    let mut epsilon_hashes: Vec<u64> = s
        .epsilons
        .iter()
        .map(|&target| hash_nfa_state(nfa, target, memo))
        .collect();
    epsilon_hashes.sort_unstable();
    h.write_u64(epsilon_hashes.len() as u64);
    for target_hash in epsilon_hashes {
        h.write_u64(target_hash);
    }

    h.write_u64(s.transitions.len() as u64);
    for (&label, targets) in &s.transitions {
        let mut target_hashes: Vec<u64> = targets
            .iter()
            .map(|&target| hash_nfa_state(nfa, target, memo))
            .collect();
        target_hashes.sort_unstable();
        h.write_i32(label);
        h.write_u64(target_hashes.len() as u64);
        for target_hash in target_hashes {
            h.write_u64(target_hash);
        }
    }

    let result = h.finish();
    memo.insert(state, result);
    result
}

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
    hash_memo: &Mutex<HashMap<u64, u64>>,
    tmp_mp: &mut Vec<u32>,
) -> u64 {
    tmp_mp.resize(pre.num_groups, NONE);

    let mut dag = build_trellis_dag(dfa, pre.num_groups, token, initial_state, tmp_mp);

    let mut nfa = build_nfa_from_trellis(dfa, &dag, pre.num_groups, ignore_terminal);
    let nfa_hash = canonical_nfa_hash(&nfa);

    if let Some(&cached) = hash_memo.lock().unwrap().get(&nfa_hash) {
        return cached;
    }

    let det_dfa = determinize(&nfa);
    let min_dfa = minimize(&det_dfa);
    let pruned_dfa = match &pre.disallowed_detector {
        Some(disallowed_detector) => minimize(&subtract(&min_dfa, disallowed_detector)),
        None => min_dfa,
    };
    let final_hash = canonical_hash(&pruned_dfa);

    hash_memo.lock().unwrap().insert(nfa_hash, final_hash);
    final_hash
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
    find_equivalence_classes_with_progress(
        regex,
        strings,
        initial_states,
        disallowed_follows,
        ignore_terminal,
        env_flag_enabled(PROGRESS_ENV),
    )
}

pub(super) fn find_equivalence_classes_with_progress<S: AsRef<[u8]> + Sync>(
    regex: &Sep1Tokenizer,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<usize>,
    show_progress: bool,
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

    let started = Instant::now();
    let hash_memo = Arc::new(Mutex::new(HashMap::<u64, u64>::new()));

    // Compute per-(token, state) hash matrix in parallel.
    // Layout: hashes[ti * ns + si] = hash for token ti at state initial_states[si]
    let counter = Arc::new(AtomicUsize::new(0));

    let progress_thread = if show_progress {
        let counter = counter.clone();
        let nt = nt;
        let started = started;
        Some(std::thread::spawn(move || {
            loop {
                std::thread::sleep(PROGRESS_INTERVAL);
                let done = counter.load(Ordering::Relaxed);
                eprintln!(
                    "[reference equiv] processed {}/{} tokens in {:.1}s",
                    done,
                    nt,
                    started.elapsed().as_secs_f64(),
                );
                if done >= nt {
                    break;
                }
            }
        }))
    } else {
        None
    };

    #[cfg(feature = "rayon")]
    let hashes: Vec<u64> = {
        use rayon::prelude::*;
        let hash_memo = hash_memo.clone();
        let rows: Vec<Vec<u64>> = strings
            .par_iter()
            .map(|token_ref| {
                let token = token_ref.as_ref();
                let mut tmp_mp = vec![NONE; pre.num_groups];
                let row: Vec<u64> = initial_states
                    .iter()
                    .map(|&state| {
                        process_token_for_state(
                            dfa,
                            &pre,
                            token,
                            state,
                            ignore_terminal,
                            &hash_memo,
                            &mut tmp_mp,
                        )
                    })
                    .collect();
                counter.fetch_add(1, Ordering::Relaxed);
                row
            })
            .collect();
        rows.into_iter().flatten().collect()
    };

    #[cfg(not(feature = "rayon"))]
    let hashes: Vec<u64> = {
        let mut tmp_mp = vec![NONE; pre.num_groups];
        let hash_memo = hash_memo.clone();
        strings
            .iter()
            .flat_map(|token_ref| {
                let token = token_ref.as_ref();
                let row: Vec<u64> = initial_states
                    .iter()
                    .map(|&state| {
                        process_token_for_state(
                            dfa,
                            &pre,
                            token,
                            state,
                            ignore_terminal,
                            &hash_memo,
                            &mut tmp_mp,
                        )
                    })
                    .collect();
                counter.fetch_add(1, Ordering::Relaxed);
                row
            })
            .collect()
    };

    if let Some(t) = progress_thread {
        counter.store(nt, Ordering::Relaxed);
        let _ = t.join();
    }

    if show_progress {
        eprintln!(
            "[reference equiv] finished {}/{} tokens in {:.1}s",
            nt,
            nt,
            started.elapsed().as_secs_f64(),
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, class, seq, star};
    use crate::compiler::compile::{build_tokenizer, build_tokenizer_from_exprs, compute_disallowed_follows};
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::stages::equivalence_analysis::compat::Sep1Tokenizer;
    use crate::ds::u8set::U8Set;

    #[test]
    fn test_reference_simple_ab_with_disallowed_follow() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b")]);
        let sep1_tok = Sep1Tokenizer::new(&tokenizer);
        let tokens = vec![b"a".to_vec(), b"b".to_vec()];
        let initial_states = vec![sep1_tok.initial_state_id()];

        let mut disallowed = BTreeMap::new();
        let mut after_a = BitSet::new(2);
        after_a.set(0);
        disallowed.insert(0u32, after_a);

        let result = find_equivalence_classes(
            &sep1_tok,
            &tokens,
            &initial_states,
            &disallowed,
            None,
        );

        assert_eq!(result.vocab_classes, BTreeSet::from([vec![0], vec![1]]));
        assert_eq!(
            result.state_classes,
            BTreeSet::from([BTreeSet::from([sep1_tok.initial_state_id()])]),
        );
    }

    fn build_live_minimal_tokenizer_fixture() -> (Sep1Tokenizer, BTreeMap<u32, BitSet>, usize, Vec<Vec<u8>>) {
        let b_or_c = class(U8Set::from_bytes(b"bc"));
        let tokenizer = build_tokenizer_from_exprs(&[
            star(b_or_c.clone()),
            seq(vec![star(b_or_c), bytes(b"b")]),
        ]);
        let sep1 = Sep1Tokenizer::new(&tokenizer);

        let mut disallowed_follows = BTreeMap::new();
        let mut all_groups = BitSet::new(2);
        all_groups.set(0);
        all_groups.set(1);
        disallowed_follows.insert(0, all_groups.clone());
        disallowed_follows.insert(1, all_groups);

        let tokens = vec![b"ba".to_vec(), b"bca".to_vec()];
        let initial_state = sep1.initial_state_id();
        (sep1, disallowed_follows, initial_state, tokens)
    }

    #[test]
    fn test_live_minimal_tokenizer_fast_reference_mismatch() {
        let (sep1, disallowed_follows, initial_state, tokens) =
            build_live_minimal_tokenizer_fixture();

        let fast_classes = crate::compiler::stages::equivalence_analysis::vocab::fast::find_vocab_equivalence_classes_with_follow(
            &sep1,
            &tokens,
            &[initial_state],
            &disallowed_follows,
        );
        let reference = find_equivalence_classes(&sep1, &tokens, &[initial_state], &disallowed_follows, None);

        assert_eq!(fast_classes, BTreeSet::from([vec![0, 1]]));
        assert_eq!(reference.vocab_classes, BTreeSet::from([vec![0, 1]]));
    }

    #[test]
    fn test_live_minimal_tokenizer_reference_hashes_differ() {
        let (sep1, disallowed_follows, initial_state, tokens) =
            build_live_minimal_tokenizer_fixture();
        let dfa = sep1.dfa();
        let pre = precompute(&dfa, &disallowed_follows);
        let hash_memo = Mutex::new(HashMap::new());

        let left_hash = process_token_for_state(
            &dfa,
            &pre,
            &tokens[0],
            initial_state,
            None,
            &hash_memo,
            &mut Vec::new(),
        );
        let right_hash = process_token_for_state(
            &dfa,
            &pre,
            &tokens[1],
            initial_state,
            None,
            &hash_memo,
            &mut Vec::new(),
        );

        assert_ne!(left_hash, right_hash);
    }

    #[test]
    #[ignore = "expected to fail: current live reference mismatch on minimal tokenizer expressions"]
    fn repro_live_minimal_tokenizer_reference_should_merge() {
        let (sep1, disallowed_follows, initial_state, tokens) =
            build_live_minimal_tokenizer_fixture();

        let reference = find_equivalence_classes(&sep1, &tokens, &[initial_state], &disallowed_follows, None);
        assert_eq!(reference.vocab_classes, BTreeSet::from([vec![0, 1]]));
    }

    /// Diagnostic: trace the trellis + NFA + DFA for the live witness pair at state 9686.
    /// Run with: cargo test --lib trace_witness_tokens -- --nocapture --ignored
    #[test]
    #[ignore]
    fn trace_witness_tokens() {
        let lark = include_str!("../../../../tests/fixtures/github_hard_o56012_split_quotes.lark");
        let grammar = crate::import::lark::parse_lark(lark).expect("lark should parse");
        let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
        let disallowed_follows = compute_disallowed_follows(&analyzed);
        let tokenizer = build_tokenizer(&grammar);
        let sep1_tok = Sep1Tokenizer::new(&tokenizer);
        let dfa = sep1_tok.dfa();
        let pre = precompute(dfa, &disallowed_follows);
        let state = 9686usize;
        let tokens: [&[u8]; 2] = [b",\"", b",\'\""];

        for (ti, token) in tokens.iter().enumerate() {
            println!("\n{}", "=".repeat(60));
            println!("=== Token {} : {:?} (hex: {:02X?}) ===", ti, String::from_utf8_lossy(token), token);

            // Step 1: Walk DFA from initial state
            let mut mp = vec![NONE; pre.num_groups];
            let end_state = walk_tokenizer_dfa(dfa, pre.num_groups, token, state, &mut mp);
            println!("\nDFA walk from state {}:", state);
            println!("  State {}: finalizers={:?}, futures={:?}",
                state, dfa.states[state].finalizers, dfa.states[state].possible_future_group_ids.len());
            let mut cur = state;
            for (i, &byte) in token.iter().enumerate() {
                let ns = dfa.states[cur].transitions[byte as usize];
                if ns == u32::MAX {
                    println!("  byte {} (0x{:02X}): DEAD", i, byte);
                    break;
                }
                cur = ns as usize;
                println!("  byte {} (0x{:02X} '{}'): -> state {} finalizers={:?} futures_len={}",
                    i, byte,
                    if byte.is_ascii_graphic() || byte == b' ' { char::from(byte).to_string() } else { format!("\\x{:02X}", byte) },
                    cur,
                    dfa.states[cur].finalizers,
                    dfa.states[cur].possible_future_group_ids.len());
            }
            println!("  end_state = {}", if end_state == STATE_NONE { "DEAD".to_string() } else { end_state.to_string() });

            // Match positions
            let matches: Vec<(usize, u32)> = (0..pre.num_groups).filter_map(|gid| {
                let p = mp[gid];
                if p != NONE { Some((gid, p)) } else { None }
            }).collect();
            println!("  match positions: {:?}", matches);

            // Step 2: Build trellis
            let mut tmp_mp = vec![NONE; pre.num_groups];
            let dag = build_trellis_dag(dfa, pre.num_groups, token, state, &mut tmp_mp);
            println!("\nTrellis DAG ({} nodes):", dag.len());
            for (&pos, node) in &dag {
                let edge_str: Vec<String> = node.edges.iter().map(|(gid, tgt)| format!("gid{}->pos{}", gid, tgt)).collect();
                println!("  pos {}: end_state={}, edges=[{}]",
                    pos,
                    if node.end_state == STATE_NONE { "DEAD".to_string() } else { node.end_state.to_string() },
                    edge_str.join(", "));
            }

            // Step 3: Build NFA
            let mut dag2 = build_trellis_dag(dfa, pre.num_groups, token, state, &mut tmp_mp);
            let nfa = build_nfa_from_trellis(dfa, &dag2, pre.num_groups, None);
            println!("\nNFA: {} states", nfa.num_states());

            // Step 4: Determinize + minimize
            let det_dfa = determinize(&nfa);
            let min_dfa = minimize(&det_dfa);
            println!("Determinized DFA: {} states", det_dfa.num_states());
            println!("Minimized DFA: {} states", min_dfa.num_states());

            // Step 5: Subtract disallowed
            let pruned_dfa = match &pre.disallowed_detector {
                Some(dd) => {
                    let s = subtract(&min_dfa, dd);
                    let m = minimize(&s);
                    println!("After subtract+minimize: {} states", m.num_states());
                    m
                },
                None => {
                    println!("No disallowed follows detector");
                    min_dfa.clone()
                }
            };

            // Step 6: Hash
            let hash = canonical_hash(&pruned_dfa);
            println!("Canonical hash: 0x{:016X}", hash);

            // Print the pruned DFA structure
            println!("\nPruned DFA states:");
            for sid in 0..pruned_dfa.num_states() {
                let s = &pruned_dfa.states[sid as usize];
                let trans: Vec<String> = s.transitions.iter()
                    .map(|(&label, &target)| format!("{}->s{}", label, target))
                    .collect();
                println!("  s{}: accepting={} trans=[{}]",
                    sid, s.is_accepting, trans.join(", "));
            }

            // Print info about the trellis end states
            println!("\nTrellis node end state details:");
            for (&pos, node) in &dag {
                if node.end_state != STATE_NONE && node.end_state < dfa.states.len() {
                    let st = &dfa.states[node.end_state];
                    let all_futures = st.possible_future_group_ids.len() == pre.num_groups;
                    println!("  pos {} end_state={}: finalizers={:?} futures_len={} all_futures={}",
                        pos, node.end_state, st.finalizers, st.possible_future_group_ids.len(), all_futures);
                } else {
                    println!("  pos {} end_state=DEAD", pos);
                }
            }
        }
    }

    /// Extracts the reachable sub-DFA from state 9686 and prints it as
    /// JSON so we can create a tiny fixture without needing the Lark grammar.
    /// Run with: cargo test --lib extract_subdfa_from_9686 -- --nocapture --ignored
    #[test]
    #[ignore]
    fn extract_subdfa_from_9686() {
        use std::collections::VecDeque;

        let lark = include_str!("../../../../tests/fixtures/github_hard_o56012_split_quotes.lark");
        let grammar = crate::import::lark::parse_lark(lark).expect("lark should parse");
        let analyzed = AnalyzedGrammar::from_grammar_def(&grammar);
        let disallowed_follows = compute_disallowed_follows(&analyzed);
        let tokenizer = build_tokenizer(&grammar);
        let sep1_tok = Sep1Tokenizer::new(&tokenizer);
        let dfa = sep1_tok.dfa();
        let start = 9686usize;

        // BFS to find all reachable states
        let mut visited = vec![false; dfa.states.len()];
        let mut queue = VecDeque::new();
        let mut reachable_order = Vec::new();
        visited[start] = true;
        queue.push_back(start);
        while let Some(s) = queue.pop_front() {
            reachable_order.push(s);
            for &target in &dfa.states[s].transitions {
                if target != u32::MAX {
                    let t = target as usize;
                    if t < dfa.states.len() && !visited[t] {
                        visited[t] = true;
                        queue.push_back(t);
                    }
                }
            }
        }

        // Build old→new state mapping
        let mut state_map = HashMap::new();
        for (new_idx, &old_idx) in reachable_order.iter().enumerate() {
            state_map.insert(old_idx, new_idx);
        }

        // Collect all group IDs that appear
        let mut all_groups = BTreeSet::new();
        for &old_s in &reachable_order {
            let st = &dfa.states[old_s];
            for &g in &st.finalizers {
                all_groups.insert(g);
            }
            for &g in &st.possible_future_group_ids {
                all_groups.insert(g);
            }
        }
        let group_list: Vec<usize> = all_groups.iter().copied().collect();
        let mut group_map = HashMap::new();
        for (new_gid, &old_gid) in group_list.iter().enumerate() {
            group_map.insert(old_gid, new_gid);
        }

        // Remap disallowed_follows
        let mut new_disallowed: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &old_gid in &group_list {
            if let Some(bits) = disallowed_follows.get(&(old_gid as u32)) {
                let mapped: Vec<usize> = bits
                    .iter()
                    .filter(|&b| group_map.contains_key(&b))
                    .map(|b| group_map[&b])
                    .collect();
                if !mapped.is_empty() {
                    new_disallowed.insert(group_map[&old_gid] as u32, mapped);
                }
            }
        }

        // Print summary
        println!("\n=== Sub-DFA extraction from state {} ===", start);
        println!("Reachable states: {}", reachable_order.len());
        println!("Groups used: {} (remapped from {:?})", group_list.len(), group_list);
        println!("Disallowed follows entries: {}", new_disallowed.len());

        // Print as JSON-ish for fixture creation
        println!("\n--- States (new_idx: transitions, finalizers, future_groups) ---");
        for (new_idx, &old_idx) in reachable_order.iter().enumerate() {
            let st = &dfa.states[old_idx];
            // Compact transitions: only non-dead
            let transitions: Vec<(u8, usize)> = (0..=255u16)
                .filter_map(|b| {
                    let target = st.transitions[b as usize];
                    if target != u32::MAX {
                        Some((b as u8, state_map[&(target as usize)]))
                    } else {
                        None
                    }
                })
                .collect();
            let finalizers: Vec<usize> = st.finalizers.iter().map(|g| group_map[g]).collect();
            let futures: Vec<usize> = st
                .possible_future_group_ids
                .iter()
                .map(|g| group_map[g])
                .collect();
            println!(
                "  state {}: trans={}, finalizers={:?}, futures={:?}",
                new_idx,
                transitions.len(),
                finalizers,
                futures
            );
            // Print first few transitions for context
            for &(byte, target) in transitions.iter().take(10) {
                println!(
                    "    0x{:02X} ({}) -> {}",
                    byte,
                    if byte.is_ascii_graphic() || byte == b' ' {
                        char::from(byte).to_string()
                    } else {
                        format!("\\x{:02X}", byte)
                    },
                    target
                );
            }
            if transitions.len() > 10 {
                println!("    ... and {} more", transitions.len() - 10);
            }
        }

        println!("\n--- Disallowed follows (remapped) ---");
        for (gid, disallowed) in &new_disallowed {
            println!("  group {} -> disallowed: {:?}", gid, disallowed);
        }

        // Print JSON for fixture
        println!("\n--- JSON fixture ---");
        println!("{{");
        println!("  \"start_state\": 0,");
        println!("  \"num_groups\": {},", group_list.len());
        println!("  \"group_mapping\": {:?},", group_list);
        println!("  \"states\": [");
        for (new_idx, &old_idx) in reachable_order.iter().enumerate() {
            let st = &dfa.states[old_idx];
            let transitions: Vec<(u8, usize)> = (0..=255u16)
                .filter_map(|b| {
                    let target = st.transitions[b as usize];
                    if target != u32::MAX {
                        Some((b as u8, state_map[&(target as usize)]))
                    } else {
                        None
                    }
                })
                .collect();
            let finalizers: Vec<usize> = st.finalizers.iter().map(|g| group_map[g]).collect();
            let futures: Vec<usize> = st
                .possible_future_group_ids
                .iter()
                .map(|g| group_map[g])
                .collect();
            let trans_str: Vec<String> = transitions
                .iter()
                .map(|&(b, t)| format!("[{},{}]", b, t))
                .collect();
            let comma = if new_idx + 1 < reachable_order.len() {
                ","
            } else {
                ""
            };
            println!(
                "    {{\"transitions\":[{}],\"finalizers\":{:?},\"futures\":{:?}}}{}",
                trans_str.join(","),
                finalizers,
                futures,
                comma
            );
        }
        println!("  ],");
        println!("  \"disallowed_follows\": {{");
        let disallowed_entries: Vec<_> = new_disallowed.iter().collect();
        for (i, (gid, disallowed)) in disallowed_entries.iter().enumerate() {
            let comma = if i + 1 < disallowed_entries.len() {
                ","
            } else {
                ""
            };
            println!("    \"{}\": {:?}{}", gid, disallowed, comma);
        }
        println!("  }}");
        println!("}}");

        // Verify the mismatch still holds with the sub-DFA
        println!("\n--- Verifying mismatch in sub-DFA ---");
        let sub_dfa = {
            let mut states = Vec::new();
            for &old_idx in &reachable_order {
                let st = &dfa.states[old_idx];
                let mut transitions = [u32::MAX; 256];
                for b in 0..256 {
                    let target = st.transitions[b];
                    if target != u32::MAX {
                        transitions[b] = state_map[&(target as usize)] as u32;
                    }
                }
                let finalizers: Vec<usize> =
                    st.finalizers.iter().map(|g| group_map[g]).collect();
                let possible_future_group_ids: Vec<usize> = st
                    .possible_future_group_ids
                    .iter()
                    .map(|g| group_map[g])
                    .collect();
                states.push(FlatDfaState {
                    transitions,
                    finalizers,
                    possible_future_group_ids,
                });
            }
            FlatDfa {
                states,
                start_state: 0, // state 9686 remapped to 0
            }
        };

        // Remap disallowed_follows to BitSet
        let num_new_groups = group_list.len();
        let mut sub_disallowed = BTreeMap::new();
        for (gid, targets) in &new_disallowed {
            let mut bits = BitSet::new(num_new_groups);
            for &t in targets {
                bits.set(t);
            }
            sub_disallowed.insert(*gid, bits);
        }

        let sub_pre = precompute(&sub_dfa, &sub_disallowed);
        let hash_memo = Mutex::new(HashMap::new());

        let left_hash = process_token_for_state(
            &sub_dfa,
            &sub_pre,
            b"!",
            0, // state 9686 remapped to 0
            None,
            &hash_memo,
            &mut Vec::new(),
        );
        let right_hash = process_token_for_state(
            &sub_dfa,
            &sub_pre,
            &[b' ', 0xC2],
            0,
            None,
            &hash_memo,
            &mut Vec::new(),
        );

        println!("Sub-DFA left_hash (\"!\"):     0x{:016X}", left_hash);
        println!("Sub-DFA right_hash (\" \\xC2\"): 0x{:016X}", right_hash);
        println!("Hashes differ: {}", left_hash != right_hash);
        assert_ne!(left_hash, right_hash, "Sub-DFA should still reproduce the mismatch");
    }
}
