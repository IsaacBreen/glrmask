//! Exact semantic comparison for partition-local terminal DWA artifacts.
//!
//! A terminal DWA evaluates a terminal-label word by intersecting its transition
//! and final weights. Equivalent artifacts can distribute the same restriction
//! across different edges, so structural edge-weight equality is too strong.

use std::collections::{BTreeSet, VecDeque};
use std::fmt::Write as _;

use crate::automata::weighted_u32::dwa::DWA;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::id_map_and_terminal_dwa::types::LocalIdMapTerminalDwa;
use crate::ds::weight::Weight;

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
    let states = original_domain(
        &baseline.id_map.tokenizer_states.original_to_internal,
        &candidate.id_map.tokenizer_states.original_to_internal,
    );
    let tokens = original_domain(
        &baseline.id_map.vocab_tokens.original_to_internal,
        &candidate.id_map.vocab_tokens.original_to_internal,
    );
    for original_state in states {
        for original_token in tokens.iter().copied() {
            if let Some(witness) = find_mismatch_for_pair(
                &baseline.dwa,
                &baseline.id_map,
                &candidate.dwa,
                &candidate.id_map,
                original_state,
                original_token,
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

fn original_domain(left: &[u32], right: &[u32]) -> BTreeSet<u32> {
    left.iter()
        .enumerate()
        .chain(right.iter().enumerate())
        .filter_map(|(original, &internal)| (internal != u32::MAX).then_some(original as u32))
        .collect()
}

fn outgoing_labels(dwa: &DWA, state: Option<u32>) -> Vec<i32> {
    state
        .and_then(|state| dwa.states().get(state as usize))
        .map(|state| state.transitions.keys().copied().collect())
        .unwrap_or_default()
}

fn accepts_final(dwa: &DWA, map: &InternalIdMap, state: Option<u32>, s: u32, t: u32) -> bool {
    state
        .and_then(|id| dwa.states().get(id as usize))
        .and_then(|node| node.final_weight.as_ref())
        .is_some_and(|weight| contains(weight, map, s, t))
}

fn enabled_target(dwa: &DWA, map: &InternalIdMap, state: Option<u32>, label: i32, s: u32, t: u32) -> Option<u32> {
    let state = state?;
    let (target, weight) = dwa.states()[state as usize].transitions.get(&label)?;
    contains(weight, map, s, t).then_some(*target)
}

fn contains(weight: &Weight, map: &InternalIdMap, s: u32, t: u32) -> bool {
    let Some(&si) = map.tokenizer_states.original_to_internal.get(s as usize) else {
        return false;
    };
    let Some(&ti) = map.vocab_tokens.original_to_internal.get(t as usize) else {
        return false;
    };
    si != u32::MAX
        && ti != u32::MAX
        && (weight.is_full() || weight.tokens_for_tsid(si).contains(ti))
}

fn find_mismatch_for_pair(
    baseline: &DWA,
    baseline_map: &InternalIdMap,
    candidate: &DWA,
    candidate_map: &InternalIdMap,
    original_state: u32,
    original_token: u32,
) -> Option<MismatchWitness> {
    let mut pending = VecDeque::from([(
        Some(baseline.start_state()),
        Some(candidate.start_state()),
        Vec::<i32>::new(),
    )]);
    let mut seen = BTreeSet::<(Option<u32>, Option<u32>)>::new();

    while let Some((baseline_state, candidate_state, word)) = pending.pop_front() {
        if !seen.insert((baseline_state, candidate_state)) {
            continue;
        }
        let baseline_accepts = accepts_final(
            baseline,
            baseline_map,
            baseline_state,
            original_state,
            original_token,
        );
        let candidate_accepts = accepts_final(
            candidate,
            candidate_map,
            candidate_state,
            original_state,
            original_token,
        );
        if baseline_accepts != candidate_accepts {
            return Some(MismatchWitness {
                original_state,
                original_token,
                word,
                baseline_accepts,
                candidate_accepts,
            });
        }

        let labels = outgoing_labels(baseline, baseline_state)
            .into_iter()
            .chain(outgoing_labels(candidate, candidate_state))
            .collect::<BTreeSet<_>>();
        for label in labels {
            let next_baseline = enabled_target(
                baseline,
                baseline_map,
                baseline_state,
                label,
                original_state,
                original_token,
            );
            let next_candidate = enabled_target(
                candidate,
                candidate_map,
                candidate_state,
                label,
                original_state,
                original_token,
            );
            if next_baseline.is_none() && next_candidate.is_none() {
                continue;
            }
            let mut next_word = word.clone();
            next_word.push(label);
            pending.push_back((next_baseline, next_candidate, next_word));
        }
    }
    None
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
    let mut state: Option<u32> = Some(dwa.start_state());
    let _ = writeln!(out, "  [{name}] start state = {state:?}");
    for (step, &label) in witness.word.iter().enumerate() {
        let next = enabled_target(dwa, map, state, label, s, t);
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
