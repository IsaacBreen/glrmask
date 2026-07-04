//! Exact semantic comparison for partition-local terminal DWA artifacts.
//!
//! A terminal DWA evaluates a terminal-label word by intersecting its transition
//! and final weights. Equivalent artifacts can distribute the same restriction
//! across different edges, so structural edge-weight equality is too strong.

use std::collections::{BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rustc_hash::FxHashMap;
use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::weight::{shared_rangeset, SharedTokenSet, Weight};

/// A concrete witness proving two terminal DWAs disagree on the completed
/// (original-coordinate) terminal language. `word` is the sequence of terminal
/// labels that reaches the disagreeing state pair when evaluating the DWAs
/// restricted to `(original_state, original_token)`.
#[derive(Debug, Clone)]
pub(crate) struct MismatchWitness {
    pub original_state: u32,
    pub original_token: u32,
    pub word: Vec<i32>,
    pub baseline_accepts: bool,
    pub candidate_accepts: bool,
}

impl std::fmt::Display for MismatchWitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "state={} token={} word={:?} baseline_accepts={} candidate_accepts={}",
            self.original_state,
            self.original_token,
            self.word,
            self.baseline_accepts,
            self.candidate_accepts,
        )
    }
}

pub(crate) fn compare(
    baseline: &LocalIdMapTerminalDwa,
    candidate: &LocalIdMapTerminalDwa,
) -> Result<(), String> {
    match find_mismatch(baseline, candidate) {
        None => Ok(()),
        Some(witness) => {
            if dump_witness_enabled() {
                eprintln!("{}", render_witness_dump(baseline, candidate, &witness));
            }
            Err(witness.to_string())
        }
    }
}

/// Exhaustively search for a completed-artifact disagreement between the two
/// terminal DWAs, returning the first concrete witness found (deterministic
/// order over original states, then tokens, then BFS word length).
pub(crate) fn find_mismatch(
    baseline: &LocalIdMapTerminalDwa,
    candidate: &LocalIdMapTerminalDwa,
) -> Option<MismatchWitness> {
    let started_at = Instant::now();
    let differs = symbolic_mismatch_exists(baseline, candidate);
    if comparator_profile_enabled() {
        eprintln!(
            "[glrmask/profile][terminal_dwa_equivalence] symbolic_ms={:.3} differs={}",
            started_at.elapsed().as_secs_f64() * 1000.0,
            differs,
        );
    }
    if !differs {
        return None;
    }
    // The symbolic pass above establishes that a disagreement exists. Re-run
    // the coordinate search only on failure so callers retain the concrete,
    // deterministic witness used by the existing diagnostics.
    find_mismatch_by_coordinate_search(baseline, candidate)
        .or_else(|| panic!("symbolic terminal-DWA comparison found a mismatch without a witness"))
}

fn comparator_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_L2P_TIMING").is_some_and(|value| value == "1")
}

fn find_mismatch_by_coordinate_search(
    baseline: &LocalIdMapTerminalDwa,
    candidate: &LocalIdMapTerminalDwa,
) -> Option<MismatchWitness> {
    // The completed-language predicate only depends on each artifact's
    // internal coordinates.  A many-to-one id map can leave thousands of
    // original states (and hundreds of original tokens) with the exact same
    // pair of internal coordinates in the reference and candidate artifacts.
    // Compare one earliest original representative for every such pair rather
    // than re-running the entire product-DWA search for every duplicate.
    let states = coordinate_domain(
        &baseline.id_map.tokenizer_states.original_to_internal,
        &candidate.id_map.tokenizer_states.original_to_internal,
    );
    let tokens = coordinate_domain(
        &baseline.id_map.vocab_tokens.original_to_internal,
        &candidate.id_map.vocab_tokens.original_to_internal,
    );
    for state in states.iter().copied() {
        for token in tokens.iter().copied() {
            if let Some(witness) = find_mismatch_for_pair(
                &baseline.dwa,
                &candidate.dwa,
                state,
                token,
            ) {
                return Some(witness);
            }
        }
    }
    None
}

fn dump_witness_enabled() -> bool {
    std::env::var_os("GLRMASK_TI_DUMP_WITNESS").is_some_and(|value| value == "1")
}

#[derive(Debug, Clone, Copy)]
struct CoordinateRepresentative {
    original: u32,
    baseline_internal: u32,
    candidate_internal: u32,
}

/// Earliest original representative for every `(baseline_internal,
/// candidate_internal)` coordinate pair. `u32::MAX` is the id-map sentinel for
/// an original coordinate absent from one artifact; it is retained so a
/// one-sided coordinate remains checked exactly once.
fn coordinate_domain(left: &[u32], right: &[u32]) -> Vec<CoordinateRepresentative> {
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for original in 0..left.len().max(right.len()) {
        let baseline_internal = left.get(original).copied().unwrap_or(u32::MAX);
        let candidate_internal = right.get(original).copied().unwrap_or(u32::MAX);
        if baseline_internal == u32::MAX && candidate_internal == u32::MAX {
            continue;
        }
        if seen.insert((baseline_internal, candidate_internal)) {
            result.push(CoordinateRepresentative {
                original: original as u32,
                baseline_internal,
                candidate_internal,
            });
        }
    }
    result
}

/// Dense symbolic coordinates for the exact joint original domain. A class is
/// one distinct pair of baseline/candidate internal ids and carries the first
/// original coordinate that realizes it. These are exactly the coordinate
/// classes enumerated by the slow witness search, but compacted to dense ids so
/// a `Weight` can represent a region of their Cartesian product directly.
struct JointCoordinateDomain {
    representatives: Vec<CoordinateRepresentative>,
    classes_for_baseline_internal: Vec<Vec<u32>>,
    classes_for_candidate_internal: Vec<Vec<u32>>,
}

impl JointCoordinateDomain {
    fn new(baseline: &[u32], candidate: &[u32]) -> Self {
        let representatives = coordinate_domain(baseline, candidate);
        let mut classes_for_baseline_internal = Vec::new();
        let mut classes_for_candidate_internal = Vec::new();
        for (class, representative) in representatives.iter().enumerate() {
            push_joint_class(
                &mut classes_for_baseline_internal,
                representative.baseline_internal,
                class as u32,
            );
            push_joint_class(
                &mut classes_for_candidate_internal,
                representative.candidate_internal,
                class as u32,
            );
        }
        Self {
            representatives,
            classes_for_baseline_internal,
            classes_for_candidate_internal,
        }
    }
}

fn push_joint_class(groups: &mut Vec<Vec<u32>>, internal: u32, class: u32) {
    if internal == u32::MAX {
        return;
    }
    if groups.len() <= internal as usize {
        groups.resize_with(internal as usize + 1, Vec::new);
    }
    groups[internal as usize].push(class);
}

#[derive(Clone, Copy)]
enum CoordinateSide {
    Baseline,
    Candidate,
}

/// Lazily expand one artifact's compact weights into the common joint
/// coordinate domain. Expansion is cached by interned weight/token-set pointer,
/// so each unique compact relation is converted at most once.
struct WeightLift {
    source_state_for_joint_class: Vec<u32>,
    classes_for_source_state: Vec<Vec<u32>>,
    classes_for_source_token: Vec<Vec<u32>>,
    full_weight: Weight,
    lifted_weights: FxHashMap<usize, Weight>,
    lifted_token_sets: FxHashMap<usize, SharedTokenSet>,
}

impl WeightLift {
    fn new(
        state_domain: &JointCoordinateDomain,
        token_domain: &JointCoordinateDomain,
        side: CoordinateSide,
    ) -> Self {
        let source_state_for_joint_class = state_domain
            .representatives
            .iter()
            .map(|representative| match side {
                CoordinateSide::Baseline => representative.baseline_internal,
                CoordinateSide::Candidate => representative.candidate_internal,
            })
            .collect::<Vec<_>>();
        let source_token_for_joint_class = token_domain
            .representatives
            .iter()
            .map(|representative| match side {
                CoordinateSide::Baseline => representative.baseline_internal,
                CoordinateSide::Candidate => representative.candidate_internal,
            })
            .collect::<Vec<_>>();
        let classes_for_source_state = match side {
            CoordinateSide::Baseline => state_domain.classes_for_baseline_internal.clone(),
            CoordinateSide::Candidate => state_domain.classes_for_candidate_internal.clone(),
        };
        let classes_for_source_token = match side {
            CoordinateSide::Baseline => token_domain.classes_for_baseline_internal.clone(),
            CoordinateSide::Candidate => token_domain.classes_for_candidate_internal.clone(),
        };
        let source_defined_tokens: RangeSetBlaze<u32> = source_token_for_joint_class
            .iter()
            .enumerate()
            .filter_map(|(class, &internal)| (internal != u32::MAX).then_some(class as u32..=class as u32))
            .collect();
        let source_defined_tokens = shared_rangeset(source_defined_tokens);
        let full_weight = Weight::from_per_tsid_shared(
            source_state_for_joint_class
                .iter()
                .enumerate()
                .filter(|&(_, &internal)| internal != u32::MAX)
                .map(|(class, _)| (class as u32, Arc::clone(&source_defined_tokens))),
        );
        Self {
            source_state_for_joint_class,
            classes_for_source_state,
            classes_for_source_token,
            full_weight,
            lifted_weights: FxHashMap::default(),
            lifted_token_sets: FxHashMap::default(),
        }
    }

    fn lift_token_set(&mut self, source_tokens: &SharedTokenSet) -> SharedTokenSet {
        let key = Arc::as_ptr(source_tokens) as usize;
        if let Some(mapped) = self.lifted_token_sets.get(&key) {
            return Arc::clone(mapped);
        }
        let mut classes = Vec::new();
        for source_token in source_tokens.iter() {
            if let Some(mapped) = self.classes_for_source_token.get(source_token as usize) {
                classes.extend_from_slice(mapped);
            }
        }
        classes.sort_unstable();
        classes.dedup();
        let mapped: RangeSetBlaze<u32> = classes
            .into_iter()
            .map(|class| class..=class)
            .collect();
        let mapped = shared_rangeset(mapped);
        self.lifted_token_sets.insert(key, Arc::clone(&mapped));
        mapped
    }

    fn lift_weight(&mut self, weight: &Weight) -> Weight {
        if weight.is_empty() {
            return Weight::empty();
        }
        if weight.is_full() {
            return self.full_weight.clone();
        }
        let key = weight.ptr_key();
        if let Some(lifted) = self.lifted_weights.get(&key) {
            return lifted.clone();
        }
        let mut tokens_for_joint_state = vec![None; self.source_state_for_joint_class.len()];
        for (start, end, source_tokens) in weight.range_entries() {
            let lifted_tokens = self.lift_token_set(source_tokens);
            if lifted_tokens.is_empty() {
                continue;
            }
            for source_state in start..=end {
                let Some(classes) = self.classes_for_source_state.get(source_state as usize) else {
                    continue;
                };
                for &class in classes {
                    tokens_for_joint_state[class as usize] = Some(Arc::clone(&lifted_tokens));
                }
            }
        }
        let lifted = Weight::from_per_tsid_shared(
            tokens_for_joint_state
                .into_iter()
                .enumerate()
                .filter_map(|(state, tokens)| tokens.map(|tokens| (state as u32, tokens))),
        );
        self.lifted_weights.insert(key, lifted.clone());
        lifted
    }
}

#[derive(Clone)]
struct SymbolicPendingRegion {
    baseline_state: Option<u32>,
    candidate_state: Option<u32>,
    region: Weight,
}

fn symbolic_product_index(
    baseline_state: Option<u32>,
    candidate_state: Option<u32>,
    candidate_width: usize,
) -> usize {
    baseline_state.map_or(0, |state| state as usize + 1) * candidate_width
        + candidate_state.map_or(0, |state| state as usize + 1)
}

fn symbolic_mismatch_exists(
    baseline: &LocalIdMapTerminalDwa,
    candidate: &LocalIdMapTerminalDwa,
) -> bool {
    let state_domain = JointCoordinateDomain::new(
        &baseline.id_map.tokenizer_states.original_to_internal,
        &candidate.id_map.tokenizer_states.original_to_internal,
    );
    let token_domain = JointCoordinateDomain::new(
        &baseline.id_map.vocab_tokens.original_to_internal,
        &candidate.id_map.vocab_tokens.original_to_internal,
    );
    if state_domain.representatives.is_empty() || token_domain.representatives.is_empty() {
        return false;
    }

    let all_tokens: RangeSetBlaze<u32> = (0..token_domain.representatives.len() as u32)
        .map(|token| token..=token)
        .collect();
    let universe = Weight::from_uniform(
        0..=state_domain.representatives.len() as u32 - 1,
        all_tokens,
    );
    let mut baseline_lift = WeightLift::new(&state_domain, &token_domain, CoordinateSide::Baseline);
    let mut candidate_lift = WeightLift::new(&state_domain, &token_domain, CoordinateSide::Candidate);

    let candidate_width = candidate.dwa.states().len() + 1;
    let mut reached_regions = vec![
        Weight::empty();
        (baseline.dwa.states().len() + 1) * candidate_width
    ];
    let mut pending = VecDeque::new();
    enqueue_symbolic_region(
        Some(baseline.dwa.start_state()),
        Some(candidate.dwa.start_state()),
        universe,
        candidate_width,
        &mut reached_regions,
        &mut pending,
    );

    while let Some(SymbolicPendingRegion {
        baseline_state,
        candidate_state,
        region,
    }) = pending.pop_front()
    {
        let baseline_final = baseline_state
            .and_then(|state| baseline.dwa.states().get(state as usize))
            .and_then(|state| state.final_weight.as_ref())
            .map(|weight| baseline_lift.lift_weight(weight))
            .unwrap_or_else(Weight::empty);
        let candidate_final = candidate_state
            .and_then(|state| candidate.dwa.states().get(state as usize))
            .and_then(|state| state.final_weight.as_ref())
            .map(|weight| candidate_lift.lift_weight(weight))
            .unwrap_or_else(Weight::empty);
        if restricted_weights_differ(&region, &baseline_final, &candidate_final) {
            return true;
        }

        for_each_union_label(
            &baseline.dwa,
            baseline_state,
            &candidate.dwa,
            candidate_state,
            |label| {
                let baseline_edge = baseline_state
                    .and_then(|state| baseline.dwa.states().get(state as usize))
                    .and_then(|state| state.transitions.get(&label));
                let candidate_edge = candidate_state
                    .and_then(|state| candidate.dwa.states().get(state as usize))
                    .and_then(|state| state.transitions.get(&label));
                let baseline_weight = baseline_edge
                    .map(|(_, weight)| baseline_lift.lift_weight(weight))
                    .unwrap_or_else(Weight::empty);
                let candidate_weight = candidate_edge
                    .map(|(_, weight)| candidate_lift.lift_weight(weight))
                    .unwrap_or_else(Weight::empty);
                let baseline_enabled = region.intersection(&baseline_weight);
                let candidate_enabled = region.intersection(&candidate_weight);
                let both_enabled = baseline_enabled.intersection(&candidate_enabled);
                let baseline_only = baseline_enabled.difference(&candidate_enabled);
                let candidate_only = candidate_enabled.difference(&baseline_enabled);
                let baseline_target = baseline_edge.map(|(target, _)| *target);
                let candidate_target = candidate_edge.map(|(target, _)| *target);
                enqueue_symbolic_region(
                    baseline_target,
                    candidate_target,
                    both_enabled,
                    candidate_width,
                    &mut reached_regions,
                    &mut pending,
                );
                enqueue_symbolic_region(
                    baseline_target,
                    None,
                    baseline_only,
                    candidate_width,
                    &mut reached_regions,
                    &mut pending,
                );
                enqueue_symbolic_region(
                    None,
                    candidate_target,
                    candidate_only,
                    candidate_width,
                    &mut reached_regions,
                    &mut pending,
                );
            },
        );
    }
    false
}

fn restricted_weights_differ(region: &Weight, left: &Weight, right: &Weight) -> bool {
    let left_enabled = region.intersection(left);
    let right_enabled = region.intersection(right);
    !left_enabled.difference(&right_enabled).is_empty()
        || !right_enabled.difference(&left_enabled).is_empty()
}

fn enqueue_symbolic_region(
    baseline_state: Option<u32>,
    candidate_state: Option<u32>,
    incoming: Weight,
    candidate_width: usize,
    reached_regions: &mut [Weight],
    pending: &mut VecDeque<SymbolicPendingRegion>,
) {
    if incoming.is_empty() {
        return;
    }
    let index = symbolic_product_index(baseline_state, candidate_state, candidate_width);
    let delta = incoming.difference(&reached_regions[index]);
    if delta.is_empty() {
        return;
    }
    reached_regions[index] = reached_regions[index].union(&delta);
    pending.push_back(SymbolicPendingRegion {
        baseline_state,
        candidate_state,
        region: delta,
    });
}

fn outgoing_labels(dwa: &DWA, state: Option<u32>) -> Vec<i32> {
    state
        .and_then(|state| dwa.states().get(state as usize))
        .map(|state| state.transitions.keys().copied().collect())
        .unwrap_or_default()
}

fn accepts_final(dwa: &DWA, state: Option<u32>, s: u32, t: u32) -> bool {
    state
        .and_then(|id| dwa.states().get(id as usize))
        .and_then(|node| node.final_weight.as_ref())
        .is_some_and(|weight| contains_internal(weight, s, t))
}

fn enabled_target(dwa: &DWA, state: Option<u32>, label: i32, s: u32, t: u32) -> Option<u32> {
    let state = state?;
    let (target, weight) = dwa.states()[state as usize].transitions.get(&label)?;
    contains_internal(weight, s, t).then_some(*target)
}

/// Exact membership without cloning the token set returned by
/// `Weight::tokens_for_tsid`. This comparator can execute the same edge query
/// millions of times on BFCL-512, so the clone dominates otherwise.
fn contains_internal(weight: &Weight, si: u32, ti: u32) -> bool {
    si != u32::MAX
        && ti != u32::MAX
        && (weight.is_full()
            || weight
                .range_entries()
                .any(|(start, end, tokens)| start <= si && si <= end && tokens.contains(ti)))
}

fn find_mismatch_for_pair(
    baseline: &DWA,
    candidate: &DWA,
    state: CoordinateRepresentative,
    token: CoordinateRepresentative,
) -> Option<MismatchWitness> {
    #[derive(Clone, Copy)]
    struct SearchNode {
        baseline_state: Option<u32>,
        candidate_state: Option<u32>,
        parent: Option<usize>,
        incoming_label: Option<i32>,
    }

    let candidate_width = candidate.states().len() + 1;
    let pair_index = |baseline_state: Option<u32>, candidate_state: Option<u32>| {
        baseline_state.map_or(0, |state| state as usize + 1) * candidate_width
            + candidate_state.map_or(0, |state| state as usize + 1)
    };
    let mut nodes = vec![SearchNode {
        baseline_state: Some(baseline.start_state()),
        candidate_state: Some(candidate.start_state()),
        parent: None,
        incoming_label: None,
    }];
    let mut pending = VecDeque::from([0usize]);
    let mut seen = vec![false; (baseline.states().len() + 1) * candidate_width];
    seen[pair_index(Some(baseline.start_state()), Some(candidate.start_state()))] = true;

    while let Some(node_index) = pending.pop_front() {
        let SearchNode {
            baseline_state,
            candidate_state,
            ..
        } = nodes[node_index];
        let baseline_accepts = accepts_final(
            baseline,
            baseline_state,
            state.baseline_internal,
            token.baseline_internal,
        );
        let candidate_accepts = accepts_final(
            candidate,
            candidate_state,
            state.candidate_internal,
            token.candidate_internal,
        );
        if baseline_accepts != candidate_accepts {
            let mut word = Vec::new();
            let mut current = node_index;
            while let Some(parent) = nodes[current].parent {
                word.push(nodes[current]
                    .incoming_label
                    .expect("non-root search node must have an incoming label"));
                current = parent;
            }
            word.reverse();
            return Some(MismatchWitness {
                original_state: state.original,
                original_token: token.original,
                word,
                baseline_accepts,
                candidate_accepts,
            });
        }

        for_each_union_label(baseline, baseline_state, candidate, candidate_state, |label| {
            let next_baseline = enabled_target(
                baseline,
                baseline_state,
                label,
                state.baseline_internal,
                token.baseline_internal,
            );
            let next_candidate = enabled_target(
                candidate,
                candidate_state,
                label,
                state.candidate_internal,
                token.candidate_internal,
            );
            if next_baseline.is_none() && next_candidate.is_none() {
                return;
            }
            let next_index = pair_index(next_baseline, next_candidate);
            if seen[next_index] {
                return;
            }
            seen[next_index] = true;
            nodes.push(SearchNode {
                baseline_state: next_baseline,
                candidate_state: next_candidate,
                parent: Some(node_index),
                incoming_label: Some(label),
            });
            pending.push_back(nodes.len() - 1);
        });
    }
    None
}

/// Visit the sorted union of raw outgoing labels without allocating a
/// `BTreeSet` for every product-DWA node.
fn for_each_union_label(
    baseline: &DWA,
    baseline_state: Option<u32>,
    candidate: &DWA,
    candidate_state: Option<u32>,
    mut visit: impl FnMut(i32),
) {
    let mut baseline_labels = baseline_state
        .and_then(|state| baseline.states().get(state as usize))
        .into_iter()
        .flat_map(|state| state.transitions.keys());
    let mut candidate_labels = candidate_state
        .and_then(|state| candidate.states().get(state as usize))
        .into_iter()
        .flat_map(|state| state.transitions.keys());
    let mut next_baseline = baseline_labels.next().copied();
    let mut next_candidate = candidate_labels.next().copied();

    loop {
        match (next_baseline, next_candidate) {
            (Some(left), Some(right)) if left < right => {
                visit(left);
                next_baseline = baseline_labels.next().copied();
            }
            (Some(left), Some(right)) if right < left => {
                visit(right);
                next_candidate = candidate_labels.next().copied();
            }
            (Some(label), Some(_)) => {
                visit(label);
                next_baseline = baseline_labels.next().copied();
                next_candidate = candidate_labels.next().copied();
            }
            (Some(label), None) => {
                visit(label);
                next_baseline = baseline_labels.next().copied();
            }
            (None, Some(label)) => {
                visit(label);
                next_candidate = candidate_labels.next().copied();
            }
            (None, None) => return,
        }
    }
}

// ---------------------------------------------------------------------------
// Opt-in structured witness dump (GLRMASK_TI_DUMP_WITNESS=1).
//
// This renders every artifact needed to localize a completed-terminal-DWA
// mismatch: both id maps in original<->internal coordinates, both full DWA
// state/edge/final tables decoded into original coordinates, and an exact
// step-by-step trace of the witness word through both DWAs restricted to the
// witness (original_state, original_token). The output is intentionally verbose
// and may be large for real BFCL partitions.
// ---------------------------------------------------------------------------

fn render_witness_dump(
    baseline: &LocalIdMapTerminalDwa,
    candidate: &LocalIdMapTerminalDwa,
    witness: &MismatchWitness,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "===== GLRMASK_TI_DUMP_WITNESS =====");
    let _ = writeln!(out, "witness: {witness}");
    let _ = writeln!(
        out,
        "restriction: original_state={} original_token={}",
        witness.original_state, witness.original_token
    );
    let _ = writeln!(out, "word (terminal labels): {:?}", witness.word);
    let _ = writeln!(out);

    let _ = writeln!(out, "----- baseline id map (TI-off reference) -----");
    render_id_map(&mut out, &baseline.id_map);
    let _ = writeln!(out, "----- candidate id map (TI-on) -----");
    render_id_map(&mut out, &candidate.id_map);

    let _ = writeln!(out, "----- baseline DWA (TI-off reference) -----");
    render_dwa(&mut out, &baseline.dwa, &baseline.id_map);
    let _ = writeln!(out, "----- candidate DWA (TI-on) -----");
    render_dwa(&mut out, &candidate.dwa, &candidate.id_map);

    let _ = writeln!(out, "----- witness trace -----");
    render_trace(&mut out, "baseline", &baseline.dwa, &baseline.id_map, witness);
    render_trace(
        &mut out,
        "candidate",
        &candidate.dwa,
        &candidate.id_map,
        witness,
    );
    let _ = writeln!(out, "===== END GLRMASK_TI_DUMP_WITNESS =====");
    out
}

fn render_many_to_one(out: &mut String, label: &str, map: &ManyToOneIdMap) {
    let _ = writeln!(
        out,
        "  {label}: {} original -> {} internal classes",
        map.original_to_internal.len(),
        map.internal_to_originals.len()
    );
    for (internal, originals) in map.internal_to_originals.iter().enumerate() {
        let _ = writeln!(out, "    internal {internal} <- originals {originals:?}");
    }
}

fn render_id_map(out: &mut String, map: &InternalIdMap) {
    render_many_to_one(out, "tokenizer_states", &map.tokenizer_states);
    render_many_to_one(out, "vocab_tokens", &map.vocab_tokens);
    let _ = writeln!(out);
}

/// Decode a weight into human-readable original-coordinate assertions. Each
/// weight entry maps a range of internal tsids (tokenizer states) to a set of
/// internal vocab-token ids; we expand both back to original ids.
fn render_weight(map: &InternalIdMap, weight: &Weight) -> String {
    if weight.is_full() {
        return "ALL".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    for (tsid_start, tsid_end, tokens) in weight.range_entries() {
        let mut orig_states: BTreeSet<u32> = BTreeSet::new();
        for tsid in tsid_start..=tsid_end {
            if let Some(originals) = map.tokenizer_states.internal_to_originals.get(tsid as usize) {
                orig_states.extend(originals.iter().copied());
            }
        }
        let mut orig_tokens: BTreeSet<u32> = BTreeSet::new();
        for token in tokens.iter() {
            if let Some(originals) = map.vocab_tokens.internal_to_originals.get(token as usize) {
                orig_tokens.extend(originals.iter().copied());
            }
        }
        parts.push(format!(
            "tsid[{tsid_start}..={tsid_end}](states {orig_states:?}) -> tokens {orig_tokens:?}"
        ));
    }
    if parts.is_empty() {
        "EMPTY".to_string()
    } else {
        parts.join("; ")
    }
}

fn render_dwa(out: &mut String, dwa: &DWA, map: &InternalIdMap) {
    let _ = writeln!(
        out,
        "  start_state={} num_states={}",
        dwa.start_state(),
        dwa.states().len()
    );
    for (state_id, state) in dwa.states().iter().enumerate() {
        let final_str = match &state.final_weight {
            None => "none".to_string(),
            Some(weight) => render_weight(map, weight),
        };
        let _ = writeln!(out, "  state {state_id}: final={final_str}");
        for (label, (target, weight)) in state.transitions.iter() {
            let _ = writeln!(
                out,
                "    --{label}--> {target}  weight={}",
                render_weight(map, weight)
            );
        }
    }
    let _ = writeln!(out);
}

fn render_trace(
    out: &mut String,
    name: &str,
    dwa: &DWA,
    map: &InternalIdMap,
    witness: &MismatchWitness,
) {
    let s = witness.original_state;
    let t = witness.original_token;
    let internal_s = map.tokenizer_states.original_to_internal.get(s as usize).copied().unwrap_or(u32::MAX);
    let internal_t = map.vocab_tokens.original_to_internal.get(t as usize).copied().unwrap_or(u32::MAX);
    let mut state: Option<u32> = Some(dwa.start_state());
    let _ = writeln!(out, "  [{name}] start state = {state:?}");
    for (step, &label) in witness.word.iter().enumerate() {
        let next = enabled_target(dwa, state, label, internal_s, internal_t);
        let raw_target = state
            .and_then(|id| dwa.states().get(id as usize))
            .and_then(|node| node.transitions.get(&label).map(|(target, _)| *target));
        let _ = writeln!(
            out,
            "  [{name}] step {step}: label={label} from {state:?} -> raw_target={raw_target:?} enabled_target={next:?} (restricted to state={s} token={t})",
        );
        state = next;
        if state.is_none() {
            break;
        }
    }
    let accepts_final = |dwa: &DWA, map: &InternalIdMap, state: Option<u32>, s: u32, t: u32| {
        let si = map.tokenizer_states.original_to_internal.get(s as usize).copied().unwrap_or(u32::MAX);
        let ti = map.vocab_tokens.original_to_internal.get(t as usize).copied().unwrap_or(u32::MAX);
        state.and_then(|id| dwa.states().get(id as usize)).and_then(|node| node.final_weight.as_ref()).is_some_and(|weight| contains_internal(weight, si, ti))
    };
    let accepts = accepts_final(dwa, map, state, s, t);
    let final_weight = state
        .and_then(|id| dwa.states().get(id as usize))
        .and_then(|node| node.final_weight.as_ref())
        .map(|weight| render_weight(map, weight))
        .unwrap_or_else(|| "none".to_string());
    let _ = writeln!(
        out,
        "  [{name}] final state={state:?} final_weight={final_weight} accepts={accepts}"
    );
    let _ = writeln!(out);
}
