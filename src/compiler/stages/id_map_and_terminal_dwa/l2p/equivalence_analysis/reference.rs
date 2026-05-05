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
use std::sync::{Arc, Mutex};

use ahash::{AHasher, RandomState};
use once_cell::sync::Lazy;
use std::hash::Hasher;

use super::compat::{FlatDfa, TokenizerView};
use crate::automata::unweighted_u32::determinize::determinize;
use crate::automata::unweighted_u32::dfa::{Label, DFA};
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic;
use crate::automata::unweighted_u32::nfa::NFA;
use crate::automata::unweighted_u32::subtract::subtract;
use crate::ds::bitset::BitSet;

use super::state::fast::StateEquivalenceResult;
use super::disallowed_follows::{build_disallowed_follow_dfa, normalize_disallowed_follows};
pub type VocabEquivalenceResult = BTreeSet<Vec<usize>>;

const HASH_SEED1: u64 = 0x9e37_79b9_7f4a_7c15;
const HASH_SEED2: u64 = 0xc2b2_ae3d_27d4_eb4f;
const HASH_SEED3: u64 = 0x1656_67b1_9e37_9f9b;
const HASH_SEED4: u64 = 0x85eb_ca6b_27d4_eb2f;
const NONE: u32 = u32::MAX;
const STATE_NONE: usize = usize::MAX;

static HASH_RANDOM_STATE: Lazy<RandomState> =
    Lazy::new(|| RandomState::with_seeds(HASH_SEED1, HASH_SEED2, HASH_SEED3, HASH_SEED4));

#[inline]
fn new_hasher() -> AHasher {
    HASH_RANDOM_STATE.build_hasher()
}

#[inline]
fn terminal_label(gid: usize) -> Label {
    gid as Label
}

#[inline]
fn future_groups_cover_all_terminals(future_groups: &[usize], num_groups: usize) -> bool {
    future_groups.len() == num_groups && future_groups.iter().copied().eq(0..num_groups)
}

struct PrecomputedData {
    num_groups: usize,
    disallowed_detector: Option<DFA>,
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
    let disallowed_detector = Some(build_disallowed_follow_dfa(&normalized));

    PrecomputedData {
        num_groups,
        disallowed_detector,
    }
}

type Edge = (usize, usize); // (group_id, target_position)

#[derive(Debug)]
struct FlatNode {
    end_state: usize,
    edges: Vec<Edge>,
}

fn enqueue_dag_target(
    dag: &mut BTreeMap<usize, FlatNode>,
    queue: &mut VecDeque<usize>,
    target_pos: usize,
    token_len: usize,
) {
    if target_pos < token_len && !dag.contains_key(&target_pos) {
        queue.push_back(target_pos);
        dag.insert(
            target_pos,
            FlatNode {
                end_state: STATE_NONE,
                edges: Vec::new(),
            },
        );
    }
}

fn get_or_create_nfa_state(
    nfa: &mut NFA,
    state_map: &mut HashMap<usize, u32>,
    worklist: &mut VecDeque<usize>,
    pos: usize,
    end_pos: usize,
    accept_sink: u32,
) -> u32 {
    if let Some(&id) = state_map.get(&pos) {
        id
    } else if pos == end_pos {
        accept_sink
    } else {
        let id = nfa.add_state();
        state_map.insert(pos, id);
        worklist.push_back(pos);
        id
    }
}

/// Walk the tokenizer DFA from `start_state` on `slice`, recording the
/// last-match position per group. Returns the end state (or STATE_NONE if
/// the DFA reaches a dead end).
fn walk_tokenizer_dfa(
    dfa: &FlatDfa,
    num_groups: usize,
    slice: &[u8],
    start_state: usize,
    match_positions: &mut [u32],
) -> usize {
    match_positions[..num_groups].fill(NONE);
    let mut current_state = start_state;
    let mut done = dfa.states[current_state].possible_future_group_ids.is_empty();

    for (i, &byte) in slice.iter().enumerate() {
        if done {
            break;
        }
        let next_state = dfa.trans(current_state, byte as usize);
        if next_state == NONE {
            done = true;
            break;
        }
        current_state = next_state as usize;
        let pos = (i + 1) as u32;
        for &gid in &dfa.states[current_state].finalizers {
            if gid < num_groups {
                match_positions[gid] = pos;
            }
        }
        if dfa.states[current_state].possible_future_group_ids.is_empty() {
            done = true;
        }
    }

    if done { STATE_NONE } else { current_state }
}

fn edges_from_match_positions(match_positions: &[u32], num_groups: usize, base_pos: usize) -> Vec<Edge> {
    (0..num_groups)
        .filter_map(|gid| {
            let pv = match_positions[gid];
            (pv != NONE && pv > 0).then(|| (gid, base_pos + pv as usize))
        })
        .collect()
}

fn build_trellis_dag(
    dfa: &FlatDfa,
    num_groups: usize,
    token: &[u8],
    initial_state: usize,
    tmp_match_positions: &mut Vec<u32>,
) -> BTreeMap<usize, FlatNode> {
    let len = token.len();
    tmp_match_positions.resize(num_groups, NONE);

    let root_end = walk_tokenizer_dfa(dfa, num_groups, token, initial_state, tmp_match_positions);
    let root_edges = edges_from_match_positions(tmp_match_positions, num_groups, 0);

    let mut dag: BTreeMap<usize, FlatNode> = BTreeMap::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    dag.insert(0, FlatNode { end_state: root_end, edges: root_edges.clone() });

    for &(_, pos) in &root_edges {
        enqueue_dag_target(&mut dag, &mut queue, pos, len);
    }

    while let Some(pos) = queue.pop_front() {
        let end = walk_tokenizer_dfa(
            dfa,
            num_groups,
            &token[pos..],
            dfa.start_state,
            tmp_match_positions,
        );
        let edges = edges_from_match_positions(tmp_match_positions, num_groups, pos);

        for &(_, target) in &edges {
            enqueue_dag_target(&mut dag, &mut queue, target, len);
        }

        let node = dag.get_mut(&pos).unwrap();
        node.end_state = end;
        node.edges = edges;
    }

    dag
}

fn build_nfa_from_trellis(
    dfa: &FlatDfa,
    dag: &BTreeMap<usize, FlatNode>,
    num_groups: usize,
    ignore_terminal: Option<usize>,
    end_pos: usize,
) -> NFA {
    let mut nfa = NFA::new_empty();
    let mut state_map: HashMap<usize, u32> = HashMap::new();
    let mut worklist: VecDeque<usize> = VecDeque::new();

    let accept_sink = nfa.add_state();
    nfa.set_accepting(accept_sink);

    let root_id = get_or_create_nfa_state(
        &mut nfa,
        &mut state_map,
        &mut worklist,
        0,
        end_pos,
        accept_sink,
    );
    nfa.start_states = vec![root_id];

    while let Some(pos) = worklist.pop_front() {
        let nfa_state = state_map[&pos];

        let node = match dag.get(&pos) {
            Some(n) => n,
            None => continue,
        };

        if node.end_state != STATE_NONE {
            let future_groups = &dfa.states[node.end_state].possible_future_group_ids;
            if future_groups_cover_all_terminals(future_groups, num_groups) {
                nfa.add_epsilon(nfa_state, accept_sink);
            } else {
                for &gid in future_groups {
                    nfa.add_transition(nfa_state, terminal_label(gid), accept_sink);
                }
            }
        }

        for &(gid, target_pos) in &node.edges {
            let is_ignore = ignore_terminal.map_or(false, |ig| gid == ig);

            if is_ignore {
                let child_id = get_or_create_nfa_state(
                    &mut nfa,
                    &mut state_map,
                    &mut worklist,
                    target_pos,
                    end_pos,
                    accept_sink,
                );
                nfa.add_epsilon(nfa_state, child_id);
            } else {
                let child_id = get_or_create_nfa_state(
                    &mut nfa,
                    &mut state_map,
                    &mut worklist,
                    target_pos,
                    end_pos,
                    accept_sink,
                );
                nfa.add_transition(nfa_state, gid as Label, child_id);
            }
        }
    }

    nfa
}

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

fn finalize_reference_dfa(nfa: &NFA, precomputed: &PrecomputedData) -> DFA {
    let determinized = determinize(nfa);
    let minimized = minimize_acyclic(&determinized);
    match &precomputed.disallowed_detector {
        Some(disallowed_detector) => minimize_acyclic(&subtract(&minimized, disallowed_detector)),
        None => minimized,
    }
}

fn process_token_for_state(
    dfa: &FlatDfa,
    precomputed: &PrecomputedData,
    token: &[u8],
    initial_state: usize,
    ignore_terminal: Option<usize>,
    hash_memo: &Mutex<HashMap<u64, u64>>,
    tmp_mp: &mut Vec<u32>,
) -> u64 {
    tmp_mp.resize(precomputed.num_groups, NONE);

    let dag = build_trellis_dag(dfa, precomputed.num_groups, token, initial_state, tmp_mp);

    let nfa = build_nfa_from_trellis(
        dfa,
        &dag,
        precomputed.num_groups,
        ignore_terminal,
        token.len(),
    );
    let nfa_hash = canonical_nfa_hash(&nfa);

    if let Some(&cached) = hash_memo.lock().unwrap().get(&nfa_hash) {
        return cached;
    }

    let final_hash = canonical_hash(&finalize_reference_dfa(&nfa, precomputed));

    hash_memo.lock().unwrap().insert(nfa_hash, final_hash);
    final_hash
}

/// Result of reference equivalence analysis.
pub struct ReferenceEquivalenceResult {
    pub vocab_classes: VocabEquivalenceResult,
    pub state_classes: StateEquivalenceResult,
}

fn empty_reference_result(
    num_tokens: usize,
    initial_states: &[usize],
) -> ReferenceEquivalenceResult {
    let vocab_classes = BTreeSet::from_iter(vec![(0..num_tokens).collect()]);
    let state_classes = initial_states
        .iter()
        .map(|&state| std::iter::once(state).collect())
        .collect();
    ReferenceEquivalenceResult {
        vocab_classes,
        state_classes,
    }
}

fn group_tokens_by_hashes(
    hashes: &[u64],
    num_tokens: usize,
    num_states: usize,
) -> VocabEquivalenceResult {
    let mut vocab_groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    for token_index in 0..num_tokens {
        let signature: Vec<u64> = (0..num_states)
            .map(|state_index| hashes[token_index * num_states + state_index])
            .collect();
        vocab_groups.entry(signature).or_default().push(token_index);
    }
    vocab_groups.into_values().collect()
}

fn group_states_by_hashes(
    hashes: &[u64],
    num_tokens: usize,
    initial_states: &[usize],
) -> StateEquivalenceResult {
    let num_states = initial_states.len();
    let mut state_groups: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
    for (state_index, &state) in initial_states.iter().enumerate() {
        let signature: Vec<u64> = (0..num_tokens)
            .map(|token_index| hashes[token_index * num_states + state_index])
            .collect();
        state_groups.entry(signature).or_default().push(state);
    }
    state_groups
        .into_values()
        .map(|states| states.into_iter().collect::<BTreeSet<usize>>())
        .collect()
}

pub fn find_equivalence_classes<S: AsRef<[u8]> + Sync>(
    tokenizer: &TokenizerView,
    strings: &[S],
    initial_states: &[usize],
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<usize>,
) -> ReferenceEquivalenceResult {
    let dfa = tokenizer.dfa();
    let precomputed = precompute(dfa, disallowed_follows);
    let num_tokens = strings.len();
    let num_states = initial_states.len();

    if num_states == 0 || num_tokens == 0 {
        return empty_reference_result(num_tokens, initial_states);
    }

    let hash_memo = Arc::new(Mutex::new(HashMap::<u64, u64>::new()));

    let hashes: Vec<u64> = {
        use rayon::prelude::*;
        let hash_memo = hash_memo.clone();
        let rows: Vec<Vec<u64>> = strings
            .par_iter()
            .map(|token_ref| {
                let token = token_ref.as_ref();
                let mut tmp_mp = vec![NONE; precomputed.num_groups];
                let row: Vec<u64> = initial_states
                    .iter()
                    .map(|&state| {
                        process_token_for_state(
                            dfa,
                            &precomputed,
                            token,
                            state,
                            ignore_terminal,
                            &hash_memo,
                            &mut tmp_mp,
                        )
                    })
                    .collect();
                row
            })
            .collect();
        rows.into_iter().flatten().collect()
    };

    ReferenceEquivalenceResult {
        vocab_classes: group_tokens_by_hashes(&hashes, num_tokens, num_states),
        state_classes: group_states_by_hashes(&hashes, num_tokens, initial_states),
    }
}
