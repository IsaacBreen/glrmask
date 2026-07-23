//! Exact-safe vocabulary preclasses for repeated-common-atom terminal families.
//!
//! A family such as `C`, `C{0,63}`, `C{64}`, `" C{64}`, and
//! `C{0,64} "` has a large lexer DFA because every repetition count is a
//! distinct residual.  Token behaviour does not need to be classified on all
//! of those count-expanded states.  When `C` is prefix-free, a token scan has
//! one exact sequence of completed `C` atoms.  The action of a token from a
//! repetition residual is therefore determined by:
//!
//! * its atom-completion byte positions;
//! * its ending residual in `C*`, or whether the scan dies;
//! * a one-byte suffix encountered at an atom boundary; and
//! * the analogous action after each supported one-byte literal prefix;
//! * the recursively interned root-language action of the token suffix after
//!   every position that could be a longest terminal match.
//!
//! Counts are deliberately absent from the signature.  Given any concrete
//! count residual and repetition bounds, the completion positions recover the
//! exact longest accepting width and the ending `(count, atom residual)`.
//! Prefix-freeness is essential: without it, one byte string can admit multiple
//! atom segmentations and the completion trace is not sufficient.  The
//! recursive root action is likewise essential: after a terminal match the
//! vocabulary scanner resumes at the lexer root on the unmatched token suffix.
//!
//! The resulting classes are only a prepartition.  The generic follow-aware
//! vocabulary-equivalence engine still runs on one representative per class
//! and its exact classes are expanded through these aliases.

use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::automata::lexer::Lexer;
use crate::automata::lexer::compile::build_regex_monolithic;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;

use super::fast::VocabEquivalenceResult;

const MIN_TOKENS: usize = 4_096;
const MAX_ACTIVE_TERMINALS: usize = 64;
const MAX_ATOM_STAR_STATES: usize = 64;
const MIN_REDUCTION_FACTOR: usize = 4;

#[derive(Debug)]
pub(crate) struct CommonAtomPreclasses {
    classes: Vec<Vec<usize>>,
    pub(crate) active_terminals: usize,
    pub(crate) atom_states: usize,
    pub(crate) build_ms: f64,
    pub(crate) classify_ms: f64,
}

impl CommonAtomPreclasses {
    pub(crate) fn len(&self) -> usize {
        self.classes.len()
    }

    pub(crate) fn representative_tokens<'a, S: AsRef<[u8]>>(
        &self,
        tokens: &'a [S],
    ) -> Vec<&'a [u8]> {
        self.classes
            .iter()
            .map(|class| tokens[class[0]].as_ref())
            .collect()
    }

    pub(crate) fn expand_exact_classes(
        &self,
        representative_classes: &VocabEquivalenceResult,
    ) -> VocabEquivalenceResult {
        representative_classes
            .iter()
            .map(|representative_class| {
                let total_len = representative_class
                    .iter()
                    .map(|&preclass| self.classes[preclass].len())
                    .sum();
                let mut class = Vec::with_capacity(total_len);
                for &preclass in representative_class {
                    class.extend_from_slice(&self.classes[preclass]);
                }
                class.sort_unstable();
                class
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct TerminalShape {
    terminal: u32,
    prefix: Option<u8>,
    suffix: Option<u8>,
    min_atoms: usize,
    max_atoms: Option<usize>,
}

fn unwrap_shared(mut expr: &Expr) -> &Expr {
    while let Expr::Shared(inner) = expr {
        expr = inner;
    }
    expr
}

fn collect_outer_repeat_atoms(
    expr: &Expr,
    counts: &mut FxHashMap<Expr, usize>,
    first_seen: &mut Vec<Expr>,
) {
    match unwrap_shared(expr) {
        Expr::Seq(parts) => {
            for part in parts {
                collect_outer_repeat_atoms(part, counts, first_seen);
            }
        }
        Expr::Repeat { expr, .. } => {
            let atom = unwrap_shared(expr).clone();
            if !counts.contains_key(&atom) {
                first_seen.push(atom.clone());
            }
            *counts.entry(atom).or_default() += 1;
        }
        _ => {}
    }
}

fn atom_core_bounds(expr: &Expr, atom: &Expr) -> Option<(usize, Option<usize>)> {
    match unwrap_shared(expr) {
        candidate if candidate == atom => Some((1, Some(1))),
        Expr::Repeat { expr, min, max } if unwrap_shared(expr) == atom => Some((*min, *max)),
        _ => None,
    }
}

fn append_literal_bytes(expr: &Expr, bytes: &mut Vec<u8>) -> bool {
    match unwrap_shared(expr) {
        Expr::Epsilon => true,
        Expr::U8Seq(part) => {
            bytes.extend_from_slice(part);
            true
        }
        _ => false,
    }
}

fn terminal_shape(terminal: u32, expr: &Expr, atom: &Expr) -> Option<TerminalShape> {
    if let Some((min_atoms, max_atoms)) = atom_core_bounds(expr, atom) {
        return Some(TerminalShape {
            terminal,
            prefix: None,
            suffix: None,
            min_atoms,
            max_atoms,
        });
    }

    let Expr::Seq(parts) = unwrap_shared(expr) else {
        return None;
    };
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    let mut core_bounds = None;
    for part in parts {
        if let Some(bounds) = atom_core_bounds(part, atom) {
            if core_bounds.is_some() {
                return None;
            }
            core_bounds = Some(bounds);
            continue;
        }
        let target = if core_bounds.is_some() {
            &mut suffix
        } else {
            &mut prefix
        };
        if !append_literal_bytes(part, target) || target.len() > 1 {
            return None;
        }
    }
    let (min_atoms, max_atoms) = core_bounds?;
    Some(TerminalShape {
        terminal,
        prefix: prefix.first().copied(),
        suffix: suffix.first().copied(),
        min_atoms,
        max_atoms,
    })
}

fn find_common_atom_family(active_exprs: &[(u32, Expr)]) -> Option<(Expr, Vec<TerminalShape>)> {
    let mut counts = FxHashMap::<Expr, usize>::default();
    let mut candidates = Vec::new();
    for (_, expr) in active_exprs {
        collect_outer_repeat_atoms(expr, &mut counts, &mut candidates);
    }
    candidates.sort_by(|left, right| counts[right].cmp(&counts[left]));
    for atom in candidates {
        let repeated_uses = counts[&atom];
        if repeated_uses < 2 {
            break;
        }
        let Some(shapes) = active_exprs
            .iter()
            .map(|(terminal, expr)| terminal_shape(*terminal, expr, &atom))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        return Some((atom, shapes));
    }
    None
}

fn atom_is_nonnullable_prefix_free(atom: &Expr) -> bool {
    let tokenizer = build_regex_monolithic(std::slice::from_ref(atom)).into_tokenizer(
        1,
        Some(Arc::from(vec![atom.clone()].into_boxed_slice())),
    );
    if tokenizer.has_epsilon_transitions()
        || tokenizer
            .matched_terminals_iter(tokenizer.initial_state_id())
            .any(|terminal| terminal == 0)
    {
        return false;
    }

    (0..tokenizer.num_states()).all(|state| {
        !tokenizer
            .matched_terminals_iter(state)
            .any(|terminal| terminal == 0)
            || !tokenizer
                .possible_future_terminals_iter(state)
                .any(|terminal| terminal == 0)
    })
}

struct TraceMachine {
    tokenizer: Tokenizer,
    shapes: Vec<TerminalShape>,
    boundary_state: u32,
    prefix_bytes: Vec<u8>,
    suffix_id: [u16; 256],
}

fn build_trace_machine(atom: Expr, shapes: Vec<TerminalShape>) -> Option<TraceMachine> {
    if !atom_is_nonnullable_prefix_free(&atom) {
        return None;
    }
    if shapes.iter().any(|shape| shape.max_atoms.is_none()) {
        // The compact proof below is deliberately limited to finite bounded
        // families.  Unbounded common-atom repetitions can use the generic
        // exact scanner until their residual/count interaction is covered by
        // a separate proof and focused tests.
        return None;
    }

    let star_expr = Expr::Repeat {
        expr: Box::new(atom),
        min: 0,
        max: None,
    };
    let tokenizer = build_regex_monolithic(std::slice::from_ref(&star_expr)).into_tokenizer(
        1,
        Some(Arc::from(vec![star_expr].into_boxed_slice())),
    );
    if tokenizer.has_epsilon_transitions()
        || tokenizer.num_states() as usize > MAX_ATOM_STAR_STATES
    {
        return None;
    }
    let boundary_states = (0..tokenizer.num_states())
        .filter(|&state| {
            tokenizer
                .matched_terminals_iter(state)
                .any(|terminal| terminal == 0)
        })
        .collect::<Vec<_>>();
    let [boundary_state] = boundary_states.as_slice() else {
        return None;
    };
    if *boundary_state != tokenizer.initial_state_id() {
        return None;
    }

    let mut prefix_bytes = shapes
        .iter()
        .filter_map(|shape| shape.prefix)
        .collect::<Vec<_>>();
    prefix_bytes.sort_unstable();
    prefix_bytes.dedup();
    let mut suffix_bytes = shapes
        .iter()
        .filter_map(|shape| shape.suffix)
        .collect::<Vec<_>>();
    suffix_bytes.sort_unstable();
    suffix_bytes.dedup();

    let mut suffix_id = [u16::MAX; 256];
    for (id, byte) in suffix_bytes.into_iter().enumerate() {
        let id = u16::try_from(id).ok()?;
        if tokenizer.step(*boundary_state, byte).is_some() {
            // A suffix that can also begin another atom makes the atom-boundary
            // choice ambiguous.  The compact completion trace does not encode
            // both branches, so keep the generic exact scanner for that family.
            return None;
        }
        suffix_id[byte as usize] = id;
    }

    Some(TraceMachine {
        tokenizer,
        shapes,
        boundary_state: *boundary_state,
        prefix_bytes,
        suffix_id,
    })
}

const TRACE_START: u32 = 1;
const COMPLETION: u32 = 2;
const DEAD: u32 = 3;
const SUFFIX_END: u32 = 4;
const SUFFIX_MORE: u32 = 5;
const ALIVE: u32 = 6;
const PREFIX: u32 = 7;
const PREFIX_MATCH: u32 = 8;
const PREFIX_MISS: u32 = 9;
const ROOT_DEAD: u32 = 10;
const ROOT_LIVE: u32 = 11;
const ROOT_MATCHES: u32 = 12;
const ROOT_CUTS: u32 = 13;

fn append_trace(
    signature: &mut Vec<u32>,
    candidate_cuts: &mut Vec<usize>,
    machine: &TraceMachine,
    input: &[u8],
    start_state: u32,
    width_base: usize,
) {
    signature.push(TRACE_START);
    signature.push(start_state);
    let mut state = start_state;
    for (index, &byte) in input.iter().enumerate() {
        let width = width_base + index + 1;
        let Some(target) = machine.tokenizer.step(state, byte) else {
            let suffix = machine.suffix_id[byte as usize];
            if state == machine.boundary_state && suffix != u16::MAX {
                signature.push(if index + 1 == input.len() {
                    SUFFIX_END
                } else {
                    SUFFIX_MORE
                });
                signature.push(suffix as u32);
                signature.push(width as u32);
                candidate_cuts.push(width);
            } else {
                // Once a prefix-free atom scan dies away from a recognized
                // suffix at an atom boundary, no active family member can
                // recover.  The death byte position is therefore irrelevant;
                // prior completion widths already retain every possible match.
                signature.push(DEAD);
            }
            return;
        };
        state = target;
        if state == machine.boundary_state {
            signature.push(COMPLETION);
            signature.push(width as u32);
            candidate_cuts.push(width);
        }
    }
    signature.push(ALIVE);
    signature.push(state);
}

#[derive(Clone, Copy)]
enum AtomRunEnd {
    Alive { at_boundary: bool },
    Suffix { byte: u8, width: usize },
    Dead,
}

struct AtomRun {
    completion_widths: Vec<usize>,
    end: AtomRunEnd,
}

fn scan_atom_run(machine: &TraceMachine, input: &[u8]) -> AtomRun {
    let mut state = machine.boundary_state;
    let mut completion_widths = Vec::new();
    for (index, &byte) in input.iter().enumerate() {
        let Some(target) = machine.tokenizer.step(state, byte) else {
            let suffix = machine.suffix_id[byte as usize];
            return AtomRun {
                completion_widths,
                end: if state == machine.boundary_state && suffix != u16::MAX {
                    AtomRunEnd::Suffix {
                        byte,
                        width: index + 1,
                    }
                } else {
                    AtomRunEnd::Dead
                },
            };
        };
        state = target;
        if state == machine.boundary_state {
            completion_widths.push(index + 1);
        }
    }
    AtomRun {
        completion_widths,
        end: AtomRunEnd::Alive {
            at_boundary: state == machine.boundary_state,
        },
    }
}

#[inline]
fn count_in_bounds(shape: TerminalShape, count: usize) -> bool {
    count >= shape.min_atoms && shape.max_atoms.is_none_or(|max| count <= max)
}

#[inline]
fn can_add_atom(shape: TerminalShape, count: usize) -> bool {
    shape.max_atoms.is_none_or(|max| count < max)
}

fn shape_root_observation(
    shape: TerminalShape,
    input: &[u8],
    unprefixed_run: &AtomRun,
    matching_prefix_run: Option<&AtomRun>,
) -> (Option<usize>, bool) {
    let (base_width, run) = if let Some(prefix) = shape.prefix {
        let Some(&first) = input.first() else {
            return (None, true);
        };
        if first != prefix {
            return (None, false);
        }
        (
            1,
            matching_prefix_run.expect("matching prefix run must be available"),
        )
    } else {
        (0, unprefixed_run)
    };

    let completed_atoms = run.completion_widths.len();
    let longest_match = if let Some(expected_suffix) = shape.suffix {
        match run.end {
            AtomRunEnd::Suffix { byte, width }
                if byte == expected_suffix && count_in_bounds(shape, completed_atoms) =>
            {
                Some(base_width + width)
            }
            _ => None,
        }
    } else {
        let mut longest = (base_width > 0 && count_in_bounds(shape, 0)).then_some(base_width);
        for (index, &width) in run.completion_widths.iter().enumerate() {
            let count = index + 1;
            if count_in_bounds(shape, count) {
                longest = Some(base_width + width);
            }
        }
        longest
    };

    let can_continue = match run.end {
        AtomRunEnd::Dead | AtomRunEnd::Suffix { .. } => false,
        AtomRunEnd::Alive { at_boundary } => {
            if at_boundary {
                shape.suffix.is_some() && count_in_bounds(shape, completed_atoms)
                    || can_add_atom(shape, completed_atoms)
            } else {
                can_add_atom(shape, completed_atoms)
            }
        }
    };
    (longest_match.filter(|&width| width > 0), can_continue)
}

fn root_observation(machine: &TraceMachine, input: &[u8]) -> (Vec<u32>, Vec<(u32, usize)>) {
    let unprefixed_run = scan_atom_run(machine, input);
    let matching_prefix_run = input.first().and_then(|first| {
        machine
            .prefix_bytes
            .binary_search(first)
            .is_ok()
            .then(|| scan_atom_run(machine, &input[1..]))
    });
    let mut future_terminals = Vec::new();
    let mut matches = Vec::new();
    for &shape in &machine.shapes {
        let (longest_match, can_continue) = shape_root_observation(
            shape,
            input,
            &unprefixed_run,
            matching_prefix_run.as_ref(),
        );
        if can_continue {
            future_terminals.push(shape.terminal);
        }
        if let Some(width) = longest_match {
            matches.push((shape.terminal, width));
        }
    }
    (future_terminals, matches)
}

fn root_semantic_ids_for_token(
    machine: &TraceMachine,
    bytes: &[u8],
    semantic_ids: &mut FxHashMap<Vec<u32>, u32>,
    semantic_by_suffix: &mut FxHashMap<Vec<u8>, u32>,
) -> Vec<u32> {
    let mut token_semantic_ids = vec![0u32; bytes.len() + 1];
    for offset in (0..=bytes.len()).rev() {
        let suffix = &bytes[offset..];
        if let Some(&semantic_id) = semantic_by_suffix.get(suffix) {
            token_semantic_ids[offset] = semantic_id;
            continue;
        }

        let (future_terminals, matches) = root_observation(machine, suffix);

        let mut signature = Vec::new();
        if future_terminals.is_empty() {
            signature.push(ROOT_DEAD);
        } else {
            signature.push(ROOT_LIVE);
            signature.push(future_terminals.len() as u32);
            signature.extend(future_terminals);
        }

        signature.push(ROOT_MATCHES);
        signature.push(matches.len() as u32);
        for (terminal, width) in matches {
            debug_assert!(width > 0 && width <= suffix.len());
            signature.push(terminal);
            signature.push(width as u32);
            signature.push(token_semantic_ids[offset + width]);
        }

        let next_semantic_id = semantic_ids.len() as u32;
        let semantic_id = *semantic_ids.entry(signature).or_insert(next_semantic_id);
        semantic_by_suffix.insert(suffix.to_vec(), semantic_id);
        token_semantic_ids[offset] = semantic_id;
    }
    token_semantic_ids
}

fn classify_tokens<S: AsRef<[u8]>>(
    machine: &TraceMachine,
    tokens: &[S],
) -> Vec<Vec<usize>> {
    let mut root_semantic_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut root_semantic_by_suffix = FxHashMap::<Vec<u8>, u32>::default();
    let mut classes = FxHashMap::<Vec<u32>, Vec<usize>>::default();
    for (token_index, token) in tokens.iter().enumerate() {
        let bytes = token.as_ref();
        let suffix_semantics = root_semantic_ids_for_token(
            machine,
            bytes,
            &mut root_semantic_ids,
            &mut root_semantic_by_suffix,
        );
        let mut signature = Vec::with_capacity(machine.tokenizer.num_states() as usize * 6);
        let mut candidate_cuts = Vec::new();
        for start_state in 0..machine.tokenizer.num_states() {
            append_trace(
                &mut signature,
                &mut candidate_cuts,
                machine,
                bytes,
                start_state,
                0,
            );
        }
        for (prefix_id, &prefix) in machine.prefix_bytes.iter().enumerate() {
            signature.push(PREFIX);
            signature.push(prefix_id as u32);
            if bytes.first() == Some(&prefix) {
                signature.push(PREFIX_MATCH);
                append_trace(
                    &mut signature,
                    &mut candidate_cuts,
                    machine,
                    &bytes[1..],
                    machine.boundary_state,
                    1,
                );
            } else {
                signature.push(PREFIX_MISS);
            }
        }
        candidate_cuts.sort_unstable();
        candidate_cuts.dedup();
        signature.push(ROOT_CUTS);
        signature.push(candidate_cuts.len() as u32);
        for cut in candidate_cuts {
            debug_assert!(cut > 0 && cut <= bytes.len());
            signature.push(cut as u32);
            signature.push(suffix_semantics[cut]);
        }
        classes.entry(signature).or_default().push(token_index);
    }
    let mut classes = classes.into_values().collect::<Vec<_>>();
    classes.sort_unstable();
    classes
}

pub(crate) fn try_find_common_atom_preclasses<S: AsRef<[u8]>>(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    tokens: &[S],
) -> Option<CommonAtomPreclasses> {
    if std::env::var_os("GLRMASK_DISABLE_L2P_COMMON_ATOM_PRECLASS").is_some()
        || tokens.len() < MIN_TOKENS
        || tokens
            .iter()
            .any(|token| token.as_ref().len() > u32::MAX as usize)
    {
        return None;
    }
    let active_groups = active_groups?;
    let active_terminals = active_groups.iter().filter(|&&active| active).count();
    if !(2..=MAX_ACTIVE_TERMINALS).contains(&active_terminals) {
        return None;
    }
    let active_exprs = active_groups
        .iter()
        .enumerate()
        .filter_map(|(terminal, &active)| {
            active.then(|| {
                tokenizer
                    .terminal_expr(terminal as u32)
                    .cloned()
                    .map(|expr| (terminal as u32, expr))
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let build_started_at = Instant::now();
    let (atom, shapes) = find_common_atom_family(&active_exprs)?;
    let machine = build_trace_machine(atom, shapes)?;
    let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;

    let classify_started_at = Instant::now();
    let classes = classify_tokens(&machine, tokens);
    let classify_ms = classify_started_at.elapsed().as_secs_f64() * 1000.0;
    if classes.len().saturating_mul(MIN_REDUCTION_FACTOR) >= tokens.len() {
        return None;
    }

    Some(CommonAtomPreclasses {
        classes,
        active_terminals,
        atom_states: machine.tokenizer.num_states() as usize,
        build_ms,
        classify_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::automata::lexer::compile::build_regex_monolithic;
    use crate::automata::regex::{byte, bytes, choice, repeat, seq};
    use crate::compiler::stages::id_map_and_terminal_dwa::l2p::equivalence_analysis::compat::{
        TokenizerView, compute_byte_classes,
    };

    fn tokenizer_for(exprs: Vec<Expr>) -> Tokenizer {
        build_regex_monolithic(&exprs).into_tokenizer(
            exprs.len() as u32,
            Some(Arc::from(exprs.into_boxed_slice())),
        )
    }

    fn enumerate(alphabet: &[u8], max_len: usize) -> Vec<Vec<u8>> {
        let mut out = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in frontier {
                for &byte in alphabet {
                    let mut token = prefix.clone();
                    token.push(byte);
                    out.push(token.clone());
                    next.push(token);
                }
            }
            frontier = next;
        }
        out
    }

    fn assert_preclasses_refine(
        preclasses: &CommonAtomPreclasses,
        exact_classes: &VocabEquivalenceResult,
        token_count: usize,
    ) {
        let mut exact_class_for_token = vec![usize::MAX; token_count];
        for (class_id, class) in exact_classes.iter().enumerate() {
            for &token in class {
                exact_class_for_token[token] = class_id;
            }
        }
        for preclass in &preclasses.classes {
            let expected = exact_class_for_token[preclass[0]];
            if let Some(&token) = preclass
                .iter()
                .find(|&&token| exact_class_for_token[token] != expected)
            {
                panic!(
                    "common-atom preclass crossed exact vocab classes: first={} first_exact={} other={} other_exact={}",
                    preclass[0], expected, token, exact_class_for_token[token],
                );
            }
        }
    }

    #[test]
    fn common_atom_trace_keeps_exact_actions_separate() {
        let atom = choice(vec![byte(b'a'), bytes(b"bc")]);
        let exprs = vec![
            atom.clone(),
            repeat(atom.clone(), 0, Some(3)),
            repeat(atom.clone(), 2, Some(2)),
            seq(vec![byte(b'"'), repeat(atom.clone(), 2, Some(2))]),
            seq(vec![repeat(atom.clone(), 0, Some(3)), byte(b'"')]),
        ];
        let tokenizer = tokenizer_for(exprs);
        let active = vec![true; 5];
        let mut tokens = enumerate(b"abc\"x", 5);
        while tokens.len() < MIN_TOKENS {
            tokens.push(b"x".to_vec());
        }
        let preclasses = try_find_common_atom_preclasses(&tokenizer, Some(&active), &tokens)
            .expect("prefix-free repeated atom family should be recognized");

        let mut exact_actions = Vec::with_capacity(tokenizer.num_states() as usize * tokens.len());
        for state in 0..tokenizer.num_states() {
            for bytes in &tokens {
                exact_actions.push(tokenizer.execute_from_state(bytes, state));
            }
        }
        for class in &preclasses.classes {
            let first = class[0];
            for &token in &class[1..] {
                for state in 0..tokenizer.num_states() {
                    let first_action = &exact_actions[state as usize * tokens.len() + first];
                    let action = &exact_actions[state as usize * tokens.len() + token];
                    assert_eq!(action, first_action, "preclass merged unequal exact actions");
                }
            }
        }
    }

    #[test]
    fn common_atom_trace_rejects_non_prefix_free_atoms() {
        let atom = choice(vec![byte(b'a'), bytes(b"aa")]);
        let exprs = vec![
            repeat(atom.clone(), 0, Some(3)),
            repeat(atom, 2, Some(2)),
        ];
        let tokenizer = tokenizer_for(exprs);
        let active = vec![true; 2];
        let tokens = vec![b"a".to_vec(); MIN_TOKENS];
        assert!(try_find_common_atom_preclasses(&tokenizer, Some(&active), &tokens).is_none());
    }

    #[test]
    fn common_atom_trace_leaves_unbounded_families_on_generic_exact_path() {
        let atom = choice(vec![byte(b'a'), bytes(b"bc")]);
        let exprs = vec![
            repeat(atom.clone(), 0, None),
            repeat(atom, 2, None),
        ];
        let tokenizer = tokenizer_for(exprs);
        let active = vec![true; 2];
        let tokens = vec![b"abc".to_vec(); MIN_TOKENS];
        assert!(try_find_common_atom_preclasses(&tokenizer, Some(&active), &tokens).is_none());
    }

    #[test]
    fn common_atom_preclasses_refine_filtered_full_lexer_and_expand_exactly() {
        let atom = choice(vec![byte(b'a'), bytes(b"bc")]);
        let mut exprs = vec![
            atom.clone(),
            repeat(atom.clone(), 0, Some(3)),
            repeat(atom.clone(), 2, Some(2)),
            seq(vec![byte(b'"'), repeat(atom.clone(), 2, Some(2))]),
            seq(vec![repeat(atom, 0, Some(3)), byte(b'"')]),
        ];
        exprs.extend([
            bytes(b"other-terminal"),
            repeat(choice(vec![byte(b'a'), byte(b'x')]), 1, Some(5)),
        ]);
        let tokenizer = tokenizer_for(exprs);
        let active = vec![true, true, true, true, true, false, false];
        let mut tokens = enumerate(b"abc\"x", 5);
        while tokens.len() < MIN_TOKENS {
            tokens.push(b"x".to_vec());
        }
        let preclasses = try_find_common_atom_preclasses(&tokenizer, Some(&active), &tokens)
            .expect("active common-atom family should ignore unrelated lexer topology");

        let view = TokenizerView::new_filtered(&tokenizer, &active);
        let byte_to_class = compute_byte_classes(view.dfa());
        let initial_states = (0..view.dfa().states.len()).collect::<Vec<_>>();
        let (exact_classes, _) =
            super::super::fast::find_vocab_equivalence_classes_with_group_filter_profiled(
                &view,
                &tokens,
                &initial_states,
                &BTreeMap::new(),
                Some(&byte_to_class),
                None,
                None,
                None,
            );
        assert_preclasses_refine(&preclasses, &exact_classes, tokens.len());

        let representative_tokens = preclasses.representative_tokens(&tokens);
        let (representative_classes, _) =
            super::super::fast::find_vocab_equivalence_classes_with_group_filter_profiled(
                &view,
                &representative_tokens,
                &initial_states,
                &BTreeMap::new(),
                Some(&byte_to_class),
                None,
                None,
                None,
            );
        assert_eq!(
            preclasses.expand_exact_classes(&representative_classes),
            exact_classes,
        );
    }

    #[test]
    fn common_atom_trace_merges_ordinary_runs_contextually() {
        let ordinary = crate::ds::u8set::U8Set::from_bytes(b"abcdefghijklmnopqrstuvwxyz");
        let hex = crate::ds::u8set::U8Set::from_bytes(b"0123456789abcdef");
        let atom = choice(vec![
            crate::automata::regex::class(ordinary),
            seq(vec![bytes(b"\\u00"), crate::automata::regex::class(hex)]),
        ]);
        let exprs = vec![
            repeat(atom.clone(), 0, Some(64)),
            repeat(atom.clone(), 3, Some(3)),
            seq(vec![repeat(atom.clone(), 0, Some(63)), byte(b'"')]),
            seq(vec![byte(b'"'), repeat(atom, 64, Some(64))]),
        ];
        let tokenizer = tokenizer_for(exprs);
        let active = vec![true; 4];
        let mut tokens = vec![b"zz".to_vec(); MIN_TOKENS - 5];
        tokens.extend([
            b"apple".to_vec(),
            b"grape".to_vec(),
            b"mango".to_vec(),
            b"hello".to_vec(),
            b"world".to_vec(),
        ]);
        let preclasses = try_find_common_atom_preclasses(&tokenizer, Some(&active), &tokens)
            .expect("repeated atom family should be recognized");
        let class_for = |needle: &[u8]| {
            let token = tokens.iter().position(|token| token == needle).unwrap();
            preclasses
                .classes
                .iter()
                .position(|class| class.binary_search(&token).is_ok())
                .unwrap()
        };
        assert_eq!(class_for(b"grape"), class_for(b"mango"));
        assert_eq!(class_for(b"hello"), class_for(b"world"));
        assert_ne!(class_for(b"apple"), class_for(b"grape"));
    }
}
