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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::ast::{bytes, class, seq, star};
    use crate::compiler::compile::{build_tokenizer_from_exprs, compute_disallowed_follows};
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use super::super::compat::{FlatDfaState, TokenizerView};
    use crate::ds::u8set::U8Set;

    #[test]
    fn test_reference_simple_ab_with_disallowed_follow() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"a"), bytes(b"b")]);
        let tokenizer_view = TokenizerView::new(&tokenizer);
        let tokens = vec![b"a".to_vec(), b"b".to_vec()];
        let initial_states = vec![tokenizer_view.initial_state_id()];

        let mut disallowed = BTreeMap::new();
        let mut after_a = BitSet::new(2);
        after_a.set(0);
        disallowed.insert(0u32, after_a);

        let result = find_equivalence_classes(
            &tokenizer_view,
            &tokens,
            &initial_states,
            &disallowed,
            None,
        );

        assert_eq!(result.vocab_classes, BTreeSet::from([vec![0], vec![1]]));
        assert_eq!(
            result.state_classes,
            BTreeSet::from([BTreeSet::from([tokenizer_view.initial_state_id()])]),
        );
    }

    fn build_live_minimal_tokenizer_fixture() -> (TokenizerView, BTreeMap<u32, BitSet>, usize, Vec<Vec<u8>>) {
        let b_or_c = class(U8Set::from_bytes(b"bc"));
        let tokenizer = build_tokenizer_from_exprs(&[
            star(b_or_c.clone()),
            seq(vec![star(b_or_c), bytes(b"b")]),
        ]);
        let tokenizer_view = TokenizerView::new(&tokenizer);

        let mut disallowed_follows = BTreeMap::new();
        let mut all_groups = BitSet::new(2);
        all_groups.set(0);
        all_groups.set(1);
        disallowed_follows.insert(0, all_groups.clone());
        disallowed_follows.insert(1, all_groups);

        let tokens = vec![b"ba".to_vec(), b"bca".to_vec()];
        let initial_state = tokenizer_view.initial_state_id();
        (tokenizer_view, disallowed_follows, initial_state, tokens)
    }

    #[test]
    fn test_live_minimal_tokenizer_fast_reference_agree() {
        let (tokenizer_view, disallowed_follows, initial_state, tokens) =
            build_live_minimal_tokenizer_fixture();

        let fast_classes = super::super::vocab::fast::find_vocab_equivalence_classes_with_follow(
            &tokenizer_view,
            &tokens,
            &[initial_state],
            &disallowed_follows,
        );
        let reference = find_equivalence_classes(&tokenizer_view, &tokens, &[initial_state], &disallowed_follows, None);

        assert_eq!(fast_classes, BTreeSet::from([vec![0, 1]]));
        assert_eq!(reference.vocab_classes, fast_classes);
    }

    #[test]
    fn test_live_minimal_tokenizer_reference_hashes_match() {
        let (tokenizer_view, disallowed_follows, initial_state, tokens) =
            build_live_minimal_tokenizer_fixture();
        let dfa = tokenizer_view.dfa();
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

        assert_eq!(left_hash, right_hash);
    }

    /// Self-contained reproducer: fast vs reference vocab equivalence mismatch.
    ///
    /// Tokenizer (3 groups):
    ///   group 0: `a`
    ///   group 1: `b*`
    ///   group 2: `c`
    ///
    /// Disallowed follows: after group 2, group 1 is forbidden.
    ///
    /// Tokens: `ca` vs `cba`.
    ///
    /// Both tokens start with `c` (group 2). In the suffix after `c`, the
    /// disallowed-follow constraint forbids group 1 (`b*`). Token `cba` has a
    /// `b` byte that could match group 1 in an unconstrained segmentation, but
    /// the constraint blocks it. The fast analysis previously failed to detect
    /// this because multi-segment suffix DAG edges passed through the blocked
    /// group 1 segment without being filtered. The fix detects when all
    /// first-hop edges are disallowed and filters later-hop edges accordingly.
    #[test]
    fn test_self_contained_fast_reference_vocab_mismatch() {
        let exprs = [
            bytes(b"a"),             // group 0
            star(bytes(b"b")),       // group 1
            bytes(b"c"),             // group 2
        ];
        let tokenizer = build_tokenizer_from_exprs(&exprs);
        let tokenizer_view = TokenizerView::new(&tokenizer);

        let mut disallowed = BTreeMap::new();
        let mut bits = BitSet::new(3);
        bits.set(1);
        disallowed.insert(2u32, bits); // after group 2, group 1 forbidden

        let tokens: Vec<Vec<u8>> = vec![b"ca".to_vec(), b"cba".to_vec()];
        let states = vec![tokenizer_view.initial_state_id()];

        let fast = super::super::vocab::fast::find_vocab_equivalence_classes_with_follow(
            &tokenizer_view, &tokens, &states, &disallowed,
        );
        let reference = find_equivalence_classes(&tokenizer_view, &tokens, &states, &disallowed, None);

        // Both should split: "ca" has valid segmentation (group2→group0),
        // while "cba" has none (only path goes through forbidden group1).
        assert_eq!(fast, BTreeSet::from([vec![0], vec![1]]),
            "fast should split the two tokens");
        assert_eq!(reference.vocab_classes, BTreeSet::from([vec![0], vec![1]]),
            "reference should split the two tokens");
    }

    /// Witness-based reproducer for fast vocab equivalence mismatch on
    /// Github_hard/o56012 (Fibaro Home Center RGB Controller schema).
    ///
    /// Constructed from witness.json using only the 6 DFA states actually
    /// visited when processing tokens " a" and " 1" from the distinguishing
    /// state 1065:
    ///
    ///   new 0 (orig 0)    = tokenizer start state
    ///   new 1 (orig 6)    = state reached by byte '1' from start
    ///   new 2 (orig 13)   = state reached by byte 'a' from start
    ///   new 3 (orig 1065) = distinguishing state (initial walk start)
    ///   new 4 (orig 1090) = first hop from distinguishing state
    ///   new 5 (orig 1114) = second hop / self-loop
    ///
    /// Key difference: from the start state (0), byte '1' reaches state 1
    /// (finalizers [1,2,3,6,24]) while byte 'a' reaches state 2 (finalizers
    /// [1,6]). This creates different trellis DAGs which the reference
    /// analysis correctly distinguishes.
    ///
    /// The reference analysis (process_token_for_state) correctly produces
    /// different hashes for these two tokens. The fast analysis incorrectly
    /// merges them — that is the bug this witness demonstrates.
    #[ignore]
    #[test]
    fn test_witness_o56012_space_a_vs_space_1() {
        // Helper: build a FlatDfaState with transitions from (start_byte, end_byte, target) ranges.
        fn s(
            _id: usize,
            ranges: &[(usize, usize, u32)],
            finalizers: &[usize],
            pfg: &[usize],
        ) -> (FlatDfaState, [u32; 256]) {
            let mut transitions = [u32::MAX; 256];
            for &(start, end, target) in ranges {
                for b in start..=end {
                    transitions[b] = target;
                }
            }
            (FlatDfaState {
                finalizers: finalizers.to_vec(),
                possible_future_group_ids: pfg.to_vec(),
            }, transitions)
        }

        // Helper: insert a disallowed_follows entry.
        fn df(map: &mut BTreeMap<u32, BitSet>, gid: u32, num_groups: usize, disallowed: &[usize]) {
            let mut bits = BitSet::new(num_groups);
            for &g in disallowed {
                bits.set(g);
            }
            map.insert(gid, bits);
        }

        let ng: usize = 110; // num_groups from the full grammar

        // 6 states: minimal subset of the full 1167-state DFA that covers both
        // token traces. Transitions to states outside this set map to dead (u32::MAX).
        let state_data: Vec<(FlatDfaState, [u32; 256])> = vec![
            // State 0 (original 0): tokenizer start state. Only bytes 49 ('1') and 97 ('a')
            // have targets in this pruned set; all other bytes are dead.
            s(0, &[(49, 49, 1), (97, 97, 2)], &[], &(0..ng).collect::<Vec<_>>()),
            // State 1 (original 6): reached by byte '1' from start.
            s(1, &[], &[1, 2, 3, 6, 24], &[1, 2, 3, 6, 24]),
            // State 2 (original 13): reached by byte 'a' from start.
            s(2, &[], &[1, 6], &[1, 6, 38, 39, 43, 96]),
            // State 3 (original 1065): distinguishing state, initial walk starts here.
            s(3, &[(32, 33, 4), (35, 91, 4), (93, 127, 4)], &[1, 6], &[1, 6]),
            // State 4 (original 1090): first hop from distinguishing state.
            s(4, &[(32, 33, 5), (35, 91, 5), (93, 127, 5)], &[1, 6], &[1, 6]),
            // State 5 (original 1114): second hop, self-loop on printable ASCII.
            s(5, &[(32, 33, 5), (35, 91, 5), (93, 127, 5)], &[1], &[1]),
        ];
        let (flat_states, trans_arrays): (Vec<_>, Vec<_>) = state_data.into_iter().unzip();
        let flat_trans: Vec<u32> = trans_arrays.into_iter().flat_map(|a| a.into_iter()).collect();
        let dfa = FlatDfa { states: flat_states, transitions: std::sync::Arc::from(flat_trans), start_state: 0 };

        // Disallowed follows from witness.json. 110 groups with 6 unique patterns:
        //   Pattern A: groups {0}         → disallow {2..=5, 8, 11}
        //   Pattern B: 99 groups (1,6,13...) → disallow {1..=109}
        //   Pattern C: groups {2,3,4,5,9,12} → disallow {0..=8, 11, 13..=109}
        //   Pattern D: groups {7,10}      → disallow {1, 6..=7, 9..=10, 12..=109}
        //   Pattern E: group {8}          → disallow {1..=8, 10..=109}
        //   Pattern F: group {11}         → disallow {1, 6..=7, 9..=10, 13..=109}
        let mut disallowed_follows = BTreeMap::new();
        // Pattern A
        df(&mut disallowed_follows, 0, ng, &[2, 3, 4, 5, 8, 11]);
        // Pattern B: groups 1,6,13..109 (and others)
        let pattern_b: Vec<usize> = (1..ng).collect();
        for &gid in &[1u32, 6, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
            49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88,
            89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
            107, 108, 109] {
            df(&mut disallowed_follows, gid, ng, &pattern_b);
        }
        // Pattern C: groups 2,3,4,5,9,12
        let pattern_c: Vec<usize> = (0..=8).chain(std::iter::once(11)).chain(13..ng).collect();
        for &gid in &[2u32, 3, 4, 5, 9, 12] {
            df(&mut disallowed_follows, gid, ng, &pattern_c);
        }
        // Pattern D: groups 7,10
        let pattern_d: Vec<usize> = std::iter::once(1).chain(6..=7).chain(9..=10).chain(12..ng).collect();
        for &gid in &[7u32, 10] {
            df(&mut disallowed_follows, gid, ng, &pattern_d);
        }
        // Pattern E: group 8
        let pattern_e: Vec<usize> = (1..=8).chain(10..ng).collect();
        df(&mut disallowed_follows, 8, ng, &pattern_e);
        // Pattern F: group 11
        let pattern_f: Vec<usize> = std::iter::once(1).chain(6..=7).chain(9..=10).chain(13..ng).collect();
        df(&mut disallowed_follows, 11, ng, &pattern_f);

        let pre = precompute(&dfa, &disallowed_follows);
        let hash_memo = Mutex::new(HashMap::new());

        let token_space_a: &[u8] = &[32, 97];  // " a"
        let token_space_1: &[u8] = &[32, 49];  // " 1"

        // Now run the actual assertions.
        let hash_a = process_token_for_state(
            &dfa, &pre, token_space_a, 3, None, &hash_memo, &mut Vec::new(),
        );
        let hash_1 = process_token_for_state(
            &dfa, &pre, token_space_1, 3, None, &hash_memo, &mut Vec::new(),
        );

        assert_ne!(hash_a, hash_1,
                   "tokens ' a' (hash={hash_a}) and ' 1' (hash={hash_1}) should have different \
             reference hashes from the distinguishing state");
    }

    /// Witness-based reproducer for fast vocab equivalence mismatch on
    /// Github_hard/o56012 (Fibaro Home Center RGB Controller schema).
    ///
    /// The reference analysis correctly separates tokens " a" and " 1" from
    /// the distinguishing state 1065, but the fast analysis incorrectly
    /// merges them. This test asserts that _both_ analyses separate the
    /// tokens, so it should FAIL until the fast analysis bug is fixed.
    #[test]
    fn test_witness_o56012_space_a_vs_space_2() {
        use crate::compiler::grammar::transforms::prepare_grammar_for_compile;
        use crate::import::lark::parse_lark;
        use crate::automata::unweighted_u32::dfa::DFA;
        use crate::automata::unweighted_u32::determinize::determinize;
        use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic;
        use crate::automata::unweighted_u32::subtract::subtract;

        fn format_label(grammar: &crate::grammar::flat::GrammarDef, label: Label) -> String {
            let gid = label as usize;
            format!("{gid}:{}", grammar.terminal_display_name(gid as u32))
        }

        fn pretty_dfa(grammar: &crate::grammar::flat::GrammarDef, dfa: &DFA) -> String {
            use std::fmt::Write;
            let mut out = String::new();
            let _ = writeln!(out, "  {} states, start={}", dfa.states.len(), dfa.start_state);
            for (i, state) in dfa.states.iter().enumerate() {
                let accept_str = if state.is_accepting { " [accept]" } else { "" };
                let trans_parts: Vec<String> = state
                    .transitions
                    .iter()
                    .map(|(&label, &target)| format!("{} -> {}", format_label(grammar, label), target))
                    .collect();
                let trans_str = if trans_parts.is_empty() {
                    String::new()
                } else {
                    format!(" {{ {} }}", trans_parts.join(", "))
                };
                let _ = writeln!(out, "  state {i}{accept_str}{trans_str}");
            }
            out
        }

        fn compute_final_dfa(
            dfa: &FlatDfa,
            pre: &PrecomputedData,
            token: &[u8],
            initial_state: usize,
        ) -> DFA {
            let mut tmp_mp = Vec::new();
            let dag = build_trellis_dag(dfa, pre.num_groups, token, initial_state, &mut tmp_mp);
            let nfa = build_nfa_from_trellis(dfa, &dag, pre.num_groups, None, token.len());
            let det = determinize(&nfa);
            let min = minimize_acyclic(&det);
            match &pre.disallowed_detector {
                Some(dd) => minimize_acyclic(&subtract(&min, dd)),
                None => min,
            }
        }

        let lark_text =
            include_str!("../../../../../../tests/fixtures/github_hard_o56012_split_quotes.lark");
        let grammar = parse_lark(lark_text).expect("fixture grammar should parse");
        let (normalized, tokenizer) = prepare_grammar_for_compile(&grammar);
        let analyzed = AnalyzedGrammar::from_grammar_def(&normalized);
        let disallowed_follows = compute_disallowed_follows(&analyzed);
        let tokenizer_view = TokenizerView::new(&tokenizer);
        let dfa = tokenizer_view.dfa();
        let pre = precompute(dfa, &disallowed_follows);

        let token_space_a: &[u8] = &[32, 97]; // " a"
        let token_space_1: &[u8] = &[32, 49]; // " 1"

        for distinguishing_state in 0..dfa.num_states() {
            let final_a = compute_final_dfa(dfa, &pre, token_space_a, distinguishing_state);
            let final_1 = compute_final_dfa(dfa, &pre, token_space_1, distinguishing_state);

            let hash_a = canonical_hash(&final_a);
            let hash_1 = canonical_hash(&final_1);

            assert_eq!(
                hash_a,
                hash_1,
                "Reference analysis: tokens ' a' and ' 1' differ from state {distinguishing_state}\n\
                 hash_a={hash_a}, hash_1={hash_1}\n\
                 \n\
                 Final DFA for ' a':\n{}\
                 Final DFA for ' 1':\n{}",
                pretty_dfa(&normalized, &final_a),
                pretty_dfa(&normalized, &final_1),
            );
        }
    }

    /// Minimal reproduction of fast-vs-reference vocab equivalence mismatch.
    ///
    /// Derived from automated grammar minimization of Github_hard/o56012:
    ///   grammar `start: "{" "}"` → 2 tokenizer groups
    ///   vocab `["}:", "}}}"]`
    ///
    /// The tokenizer has 2 groups:
    ///   group 0: `{`
    ///   group 1: `}`
    ///
    /// Disallowed follows (from grammar structure):
    ///   after group 0 (`{`): group 0 is disallowed (only `}` can follow `{`)
    ///   after group 1 (`}`): both groups disallowed (nothing follows `}`)
    ///
    /// The reference analysis correctly separates `}:` and `}}}` because they
    /// produce different trellis structures from the initial state. The fast
    /// analysis incorrectly merges them.
    #[test]
    fn test_minimal_equiv_mismatch_o56012() {
        let tokenizer = build_tokenizer_from_exprs(&[bytes(b"{"), bytes(b"}")]);
        let tokenizer_view = TokenizerView::new(&tokenizer);

        // After group 0 (`{`): group 0 disallowed
        let mut disallowed = BTreeMap::new();
        let mut after_0 = BitSet::new(2);
        after_0.set(0);
        disallowed.insert(0u32, after_0);
        // After group 1 (`}`): both groups disallowed
        let mut after_1 = BitSet::new(2);
        after_1.set(0);
        after_1.set(1);
        disallowed.insert(1u32, after_1);

        let tokens: Vec<Vec<u8>> = vec![b"}:".to_vec(), b"}}}".to_vec()];
        let states = vec![tokenizer_view.initial_state_id()];

        let fast = super::super::vocab::fast::find_vocab_equivalence_classes_with_follow(
            &tokenizer_view, &tokens, &states, &disallowed,
        );
        let reference = find_equivalence_classes(&tokenizer_view, &tokens, &states, &disallowed, None);

        // After fixing disallowed-follow subtraction, both tokens produce
        // empty DFAs (no valid segmentation), so both analyses should merge them.
        assert_eq!(
            reference.vocab_classes,
            fast,
            "reference and fast should agree (both merge)"
        );
    }
}
