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

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::lexer::Lexer;
use crate::automata::lexer::compile::build_regex_monolithic;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::regex::Expr;

use super::fast::VocabEquivalenceResult;

const MIN_TOKENS: usize = 4_096;
const MAX_TOKENS: usize = 100_000;

fn max_tokens() -> usize {
    std::env::var("GLRMASK_L2P_COMMON_ATOM_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value >= MIN_TOKENS)
        .unwrap_or(MAX_TOKENS)
}
const MAX_ACTIVE_TERMINALS: usize = 64;
const MAX_ATOM_STAR_STATES: usize = 64;
const MIN_REDUCTION_FACTOR: usize = 4;

#[derive(Debug)]
pub struct CommonAtomPreclasses {
    classes: Vec<Vec<usize>>,
    pub active_terminals: usize,
    pub atom_states: usize,
    pub build_ms: f64,
    pub classify_ms: f64,
}

impl CommonAtomPreclasses {
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    pub fn representative_tokens<'a, S: AsRef<[u8]>>(
        &self,
        tokens: &'a [S],
    ) -> Vec<&'a [u8]> {
        self.classes
            .iter()
            .map(|class| tokens[class[0]].as_ref())
            .collect()
    }

    pub fn expand_exact_classes(
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
    completion_widths: SmallVec<[usize; 8]>,
    end: AtomRunEnd,
}

fn scan_atom_run(machine: &TraceMachine, input: &[u8]) -> AtomRun {
    let mut state = machine.boundary_state;
    let mut completion_widths = SmallVec::<[usize; 8]>::new();
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

fn root_observation(
    machine: &TraceMachine,
    input: &[u8],
) -> (SmallVec<[u32; 8]>, SmallVec<[(u32, usize); 8]>) {
    let unprefixed_run = scan_atom_run(machine, input);
    let matching_prefix_run = input.first().and_then(|first| {
        machine
            .prefix_bytes
            .binary_search(first)
            .is_ok()
            .then(|| scan_atom_run(machine, &input[1..]))
    });
    // The common-atom path is intentionally selected only for a small active
    // terminal family. Keep the root observation inline: allocating two Vecs
    // for every suffix made this phase pay hundreds of thousands of tiny heap
    // allocations on large vocabularies.
    let mut future_terminals = SmallVec::<[u32; 8]>::new();
    let mut matches = SmallVec::<[(u32, usize); 8]>::new();
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

fn root_semantic_id_at(
    machine: &TraceMachine,
    bytes: &[u8],
    offset: usize,
    semantic_ids: &mut FxHashMap<SmallVec<[u32; 32]>, u32>,
    semantic_by_suffix: &mut FxHashMap<Vec<u8>, u32>,
    token_memo: &mut [Option<u32>],
) -> u32 {
    if let Some(id) = token_memo[offset] {
        return id;
    }
    let suffix = &bytes[offset..];
    if let Some(&semantic_id) = semantic_by_suffix.get(suffix) {
        token_memo[offset] = Some(semantic_id);
        return semantic_id;
    }
    let (future_terminals, matches) = root_observation(machine, suffix);
    let mut signature = SmallVec::<[u32; 32]>::new();
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
        debug_assert!(width > 0 && offset + width <= bytes.len());
        let continuation = root_semantic_id_at(
            machine,
            bytes,
            offset + width,
            semantic_ids,
            semantic_by_suffix,
            token_memo,
        );
        signature.push(terminal);
        signature.push(width as u32);
        signature.push(continuation);
    }
    let next_semantic_id = semantic_ids.len() as u32;
    let semantic_id = *semantic_ids.entry(signature).or_insert(next_semantic_id);
    semantic_by_suffix.insert(suffix.to_vec(), semantic_id);
    token_memo[offset] = Some(semantic_id);
    semantic_id
}

fn classify_tokens_serial<S: AsRef<[u8]>>(
    machine: &TraceMachine,
    tokens: &[S],
) -> Vec<Vec<usize>> {
    let mut root_semantic_ids = FxHashMap::<SmallVec<[u32; 32]>, u32>::default();
    let mut root_semantic_by_suffix = FxHashMap::<Vec<u8>, u32>::default();
    let mut classes = FxHashMap::<Vec<u32>, Vec<usize>>::default();
    for (token_index, token) in tokens.iter().enumerate() {
        let bytes = token.as_ref();
        let mut token_semantic_memo = vec![None; bytes.len() + 1];
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
            let semantic_id = root_semantic_id_at(
                machine,
                bytes,
                cut,
                &mut root_semantic_ids,
                &mut root_semantic_by_suffix,
                &mut token_semantic_memo,
            );
            signature.push(cut as u32);
            signature.push(semantic_id);
        }
        classes.entry(signature).or_default().push(token_index);
    }
    let mut classes = classes.into_values().collect::<Vec<_>>();
    classes.sort_unstable();
    classes
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
enum CompactTraceEnd {
    Dead,
    Suffix { suffix: u16, width: u8, at_end: bool },
    Alive { state: u32 },
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct CompactTrace {
    completion_positions: u64,
    end: CompactTraceEnd,
}

#[inline]
fn common_atom_fingerprint_mix(hash: &mut u64, value: u64) {
    *hash ^= value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    *hash = hash.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
}

#[inline]
fn common_atom_fingerprint_trace(hash: &mut u64, trace: CompactTrace) {
    common_atom_fingerprint_mix(hash, trace.completion_positions);
    match trace.end {
        CompactTraceEnd::Dead => common_atom_fingerprint_mix(hash, 0),
        CompactTraceEnd::Suffix { suffix, width, at_end } => {
            common_atom_fingerprint_mix(hash, 1);
            common_atom_fingerprint_mix(
                hash,
                suffix as u64 | ((width as u64) << 16) | ((at_end as u64) << 24),
            );
        }
        CompactTraceEnd::Alive { state } => {
            common_atom_fingerprint_mix(hash, 2);
            common_atom_fingerprint_mix(hash, state as u64);
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct CommonAtomPartialTrace {
    // Vector position is the tokenizer start-state ID, so the explicit
    // TRACE_START/state words from the generic serialization are redundant.
    traces: SmallVec<[CompactTrace; 16]>,
    // Prefix vector position is the prefix ID. `None` is the exact PREFIX_MISS
    // marker; `Some(trace)` is PREFIX_MATCH followed by the boundary trace.
    prefix_traces: SmallVec<[Option<CompactTrace>; 4]>,
    candidate_cuts: u64,
    root_semantics: SmallVec<[(u8, u32); 8]>,
    fingerprint: u64,
}

#[derive(Debug)]
struct CommonAtomSuffixUniverse {
    token_offsets: Vec<usize>,
    suffix_ids: Vec<u32>,
    representatives: Vec<(usize, usize)>,
    lengths: Vec<usize>,
}

#[derive(Debug)]
pub struct PreparedCommonAtomSuffixIndex {
    all_tokens: CommonAtomSuffixUniverse,
    token_bytes: Box<[Vec<u8>]>,
    ids_by_len: Box<[Box<[u32]>]>,
}

impl crate::vocab::VocabDerivedArtifact for PreparedCommonAtomSuffixIndex {}

#[derive(Debug)]
pub struct PreparedCommonAtomSuffixView {
    index: Arc<PreparedCommonAtomSuffixIndex>,
    /// Child-vocabulary entry position -> entry position in `index.token_bytes`.
    parent_entry_indices: Box<[usize]>,
}

impl crate::vocab::VocabDerivedArtifact for PreparedCommonAtomSuffixView {}

/// Propagate a prepared suffix universe through a token-subset vocabulary.
/// `parent_entry_indices` is collected while the subset is materialized, so
/// schema compilation does not need a second token-ID lookup pass.
pub fn inherit_vocab_suffix_index(
    parent: &Vocab,
    child: &Vocab,
    parent_entry_indices: Vec<usize>,
) {
    if let Some(index) = parent.vocab_derived_cache_get::<PreparedCommonAtomSuffixIndex>() {
        debug_assert_eq!(child.len(), parent_entry_indices.len());
        child.vocab_derived_cache_set(Arc::new(PreparedCommonAtomSuffixView {
            index,
            parent_entry_indices: parent_entry_indices.into_boxed_slice(),
        }));
        return;
    }
    if let Some(parent_view) = parent.vocab_derived_cache_get::<PreparedCommonAtomSuffixView>() {
        debug_assert_eq!(child.len(), parent_entry_indices.len());
        let composed = parent_entry_indices
            .into_iter()
            .map(|index| parent_view.parent_entry_indices[index])
            .collect::<Vec<_>>();
        child.vocab_derived_cache_set(Arc::new(PreparedCommonAtomSuffixView {
            index: Arc::clone(&parent_view.index),
            parent_entry_indices: composed.into_boxed_slice(),
        }));
    }
}

/// Precompute grammar-independent suffix identity for one (possibly already
/// partitioned) vocabulary. `prepare_vocab_for_terminal_dwa` calls this on the
/// cached char-type sub-vocabs, so grammar compilation can select its dedup
/// representatives by integer provenance rather than rebuilding suffix IDs.
pub fn prepare_vocab_suffix_index(vocab: &Vocab) {
    if vocab
        .vocab_derived_cache_get::<PreparedCommonAtomSuffixIndex>()
        .is_some()
    {
        return;
    }
    let token_bytes = vocab.entries_map().values().cloned().collect::<Vec<_>>();
    let all_tokens = CommonAtomSuffixUniverse::build(&token_bytes);
    let max_len = all_tokens.lengths.iter().copied().max().unwrap_or(0);
    let mut ids_by_len = vec![Vec::<u32>::new(); max_len + 1];
    for (id, &len) in all_tokens.lengths.iter().enumerate() {
        ids_by_len[len].push(id as u32);
    }
    let prepared = Arc::new(PreparedCommonAtomSuffixIndex {
        all_tokens,
        token_bytes: token_bytes.into_boxed_slice(),
        ids_by_len: ids_by_len
            .into_iter()
            .map(Vec::into_boxed_slice)
            .collect::<Vec<_>>()
            .into_boxed_slice(),
    });
    vocab.vocab_derived_cache_set(prepared);
}

impl CommonAtomSuffixUniverse {
    fn build<S: AsRef<[u8]> + Sync>(tokens: &[S]) -> Self {
        let total_positions = tokens
            .iter()
            .map(|token| token.as_ref().len() + 1)
            .sum::<usize>();
        let mut token_offsets = Vec::with_capacity(tokens.len() + 1);
        let mut suffix_ids = Vec::with_capacity(total_positions);
        let mut representatives = vec![(0usize, tokens.first().map_or(0, |t| t.as_ref().len()))];
        let mut lengths = vec![0usize];

        // Equal suffixes are equal prefixes after reversing the token bytes.
        // Sort those reversed strings once, then reuse exactly the LCP nodes
        // from the previous token. This constructs the same canonical reverse
        // trie as `(byte, tail_id)` hash-consing without any edge dictionary.
        token_offsets.resize(tokens.len() + 1, 0);
        for (index, token) in tokens.iter().enumerate() {
            token_offsets[index + 1] = token_offsets[index] + token.as_ref().len() + 1;
        }
        suffix_ids.resize(*token_offsets.last().unwrap_or(&0), 0);
        let mut order = (0..tokens.len()).collect::<Vec<_>>();
        order.par_sort_unstable_by(|&left, &right| {
            tokens[left]
                .as_ref()
                .iter()
                .rev()
                .cmp(tokens[right].as_ref().iter().rev())
                .then_with(|| left.cmp(&right))
        });
        let mut previous_token = None::<usize>;
        let mut previous_ids = vec![0u32];
        for token_index in order {
            let bytes = tokens[token_index].as_ref();
            let lcp = previous_token.map_or(0usize, |previous_index| {
                tokens[previous_index]
                    .as_ref()
                    .iter()
                    .rev()
                    .zip(bytes.iter().rev())
                    .take_while(|(left, right)| left == right)
                    .count()
            });
            let base = token_offsets[token_index];
            suffix_ids[base + bytes.len()] = 0;
            let mut current_ids = Vec::with_capacity(bytes.len() + 1);
            current_ids.push(0);
            for depth in 1..=bytes.len() {
                let id = if depth <= lcp {
                    previous_ids[depth]
                } else {
                    let offset = bytes.len() - depth;
                    let id = representatives.len() as u32;
                    representatives.push((token_index, offset));
                    lengths.push(depth);
                    id
                };
                current_ids.push(id);
                suffix_ids[base + bytes.len() - depth] = id;
            }
            previous_token = Some(token_index);
            previous_ids = current_ids;
        }
        Self {
            token_offsets,
            suffix_ids,
            representatives,
            lengths,
        }
    }

    #[inline]
    fn suffix_id(&self, token_index: usize, offset: usize) -> u32 {
        self.suffix_ids[self.token_offsets[token_index] + offset]
    }
}

fn compact_trace(
    machine: &TraceMachine,
    input: &[u8],
    start_state: u32,
    width_base: usize,
) -> (CompactTrace, u64) {
    debug_assert!(width_base + input.len() <= 64);
    let mut state = start_state;
    let mut completion_positions = 0u64;
    let mut candidate_cuts = 0u64;
    for (index, &byte) in input.iter().enumerate() {
        let width = width_base + index + 1;
        let bit = 1u64 << (width - 1);
        let Some(target) = machine.tokenizer.step(state, byte) else {
            let suffix = machine.suffix_id[byte as usize];
            let end = if state == machine.boundary_state && suffix != u16::MAX {
                candidate_cuts |= bit;
                CompactTraceEnd::Suffix {
                    suffix,
                    width: width as u8,
                    at_end: index + 1 == input.len(),
                }
            } else {
                CompactTraceEnd::Dead
            };
            return (
                CompactTrace {
                    completion_positions,
                    end,
                },
                candidate_cuts,
            );
        };
        state = target;
        if state == machine.boundary_state {
            completion_positions |= bit;
            candidate_cuts |= bit;
        }
    }
    (
        CompactTrace {
            completion_positions,
            end: CompactTraceEnd::Alive { state },
        },
        candidate_cuts,
    )
}

fn common_atom_partial_trace(machine: &TraceMachine, bytes: &[u8]) -> CommonAtomPartialTrace {
    debug_assert!(bytes.len() <= 64);
    let mut traces = SmallVec::<[CompactTrace; 16]>::new();
    let mut candidate_cuts = 0u64;
    let mut fingerprint = 0x243f_6a88_85a3_08d3u64;
    for start_state in 0..machine.tokenizer.num_states() {
        let (trace, cuts) = compact_trace(machine, bytes, start_state, 0);
        common_atom_fingerprint_trace(&mut fingerprint, trace);
        traces.push(trace);
        candidate_cuts |= cuts;
    }
    let mut prefix_traces = SmallVec::<[Option<CompactTrace>; 4]>::new();
    for &prefix in &machine.prefix_bytes {
        if bytes.first() == Some(&prefix) {
            let (trace, cuts) = compact_trace(machine, &bytes[1..], machine.boundary_state, 1);
            common_atom_fingerprint_mix(&mut fingerprint, 1);
            common_atom_fingerprint_trace(&mut fingerprint, trace);
            prefix_traces.push(Some(trace));
            candidate_cuts |= cuts;
        } else {
            common_atom_fingerprint_mix(&mut fingerprint, 0);
            prefix_traces.push(None);
        }
    }
    common_atom_fingerprint_mix(&mut fingerprint, candidate_cuts);
    CommonAtomPartialTrace {
        traces,
        prefix_traces,
        candidate_cuts,
        root_semantics: SmallVec::new(),
        fingerprint,
    }
}

struct PreparedSuffixContext<'a> {
    index: &'a PreparedCommonAtomSuffixIndex,
    representative_original_indices: &'a [usize],
}

enum CommonAtomSuffixSource<'a> {
    Local(CommonAtomSuffixUniverse),
    Prepared {
        context: PreparedSuffixContext<'a>,
    },
}

fn classify_tokens_phased<S: AsRef<[u8]> + Sync>(
    machine: &TraceMachine,
    tokens: &[S],
    prepared_suffixes: Option<PreparedSuffixContext<'_>>,
) -> Vec<Vec<usize>> {
    let profile = std::env::var_os("GLRMASK_PROFILE_COMMON_ATOM_DETAIL").is_some();
    let total_started = Instant::now();

    let trace_started = Instant::now();
    let mut partials = tokens
        .par_iter()
        .map(|token| common_atom_partial_trace(machine, token.as_ref()))
        .collect::<Vec<_>>();
    let trace_ms = trace_started.elapsed().as_secs_f64() * 1000.0;

    let suffix_build_started = Instant::now();
    let suffix_source = if let Some(context) = prepared_suffixes {
        CommonAtomSuffixSource::Prepared { context }
    } else {
        CommonAtomSuffixSource::Local(CommonAtomSuffixUniverse::build(tokens))
    };
    let suffix_build_ms = suffix_build_started.elapsed().as_secs_f64() * 1000.0;

    let local_ids_by_len;
    let (ids_by_len, semantic_slot_count, suffix_count): (&[Box<[u32]>], usize, usize) =
        match &suffix_source {
            CommonAtomSuffixSource::Local(local) => {
                let max_len = local.lengths.iter().copied().max().unwrap_or(0);
                let mut buckets = vec![Vec::<u32>::new(); max_len + 1];
                for (id, &len) in local.lengths.iter().enumerate() {
                    buckets[len].push(id as u32);
                }
                local_ids_by_len = buckets
                    .into_iter()
                    .map(Vec::into_boxed_slice)
                    .collect::<Vec<_>>();
                (
                    local_ids_by_len.as_slice(),
                    local.representatives.len(),
                    local.representatives.len(),
                )
            }
            CommonAtomSuffixSource::Prepared { context } => (
                context.index.ids_by_len.as_ref(),
                context.index.all_tokens.representatives.len(),
                context.index.all_tokens.representatives.len(),
            ),
        };

    let root_started = Instant::now();
    let cut_filtered_root =
        std::env::var_os("GLRMASK_DISABLE_COMMON_ATOM_CUT_FILTERED_ROOT").is_none();
    let mut semantic_ids = FxHashMap::<SmallVec<[u32; 32]>, u32>::default();
    let mut semantic_by_suffix = vec![u32::MAX; semantic_slot_count];
    let mut cut_filtered_suffixes = 0usize;
    let mut cut_filtered_fallback = false;

    let mut needed = Vec::<bool>::new();
    if cut_filtered_root {
        needed.resize(semantic_slot_count, false);
        for (token_index, partial) in partials.iter().enumerate() {
            let mut cuts = partial.candidate_cuts;
            while cuts != 0 {
                let bit = cuts.trailing_zeros() as usize;
                cuts &= cuts - 1;
                let cut = bit + 1;
                let suffix_id = match &suffix_source {
                    CommonAtomSuffixSource::Local(local) => local.suffix_id(token_index, cut),
                    CommonAtomSuffixSource::Prepared { context, .. } => {
                        let original_index = context.representative_original_indices[token_index];
                        let source = &context.index.all_tokens;
                        source.suffix_ids[source.token_offsets[original_index] + cut]
                    }
                } as usize;
                needed[suffix_id] = true;
            }
        }
        cut_filtered_suffixes = needed.iter().filter(|&&yes| yes).count();
    }

    let run_root_pass = |filter: Option<&[bool]>,
                         semantic_ids: &mut FxHashMap<SmallVec<[u32; 32]>, u32>,
                         semantic_by_suffix: &mut [u32]|
     -> bool {
        for ids in ids_by_len {
            if ids.is_empty() {
                continue;
            }
            let raw_signatures = ids
                .par_iter()
                .filter(|&&id| filter.is_none_or(|needed| needed[id as usize]))
                .map(|&id| {
                    let (bytes, offset, source_token_index) = match &suffix_source {
                        CommonAtomSuffixSource::Local(local) => {
                            let (token_index, offset) = local.representatives[id as usize];
                            (tokens[token_index].as_ref(), offset, token_index)
                        }
                        CommonAtomSuffixSource::Prepared { context } => {
                            let (entry_index, offset) =
                                context.index.all_tokens.representatives[id as usize];
                            (context.index.token_bytes[entry_index].as_slice(), offset, entry_index)
                        }
                    };
                    let suffix = &bytes[offset..];
                    let (future_terminals, matches) = root_observation(machine, suffix);
                    let mut signature = SmallVec::<[u32; 32]>::new();
                    signature.reserve(4 + future_terminals.len() + matches.len() * 3);
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
                        debug_assert!(width > 0 && offset + width <= bytes.len());
                        let continuation_id = match &suffix_source {
                            CommonAtomSuffixSource::Local(local) => {
                                local.suffix_id(source_token_index, offset + width)
                            }
                            CommonAtomSuffixSource::Prepared { context, .. } => {
                                let source = &context.index.all_tokens;
                                source.suffix_ids
                                    [source.token_offsets[source_token_index] + offset + width]
                            }
                        } as usize;
                        let continuation = semantic_by_suffix[continuation_id];
                        if continuation == u32::MAX {
                            return (id, None);
                        }
                        signature.push(terminal);
                        signature.push(width as u32);
                        signature.push(continuation);
                    }
                    (id, Some(signature))
                })
                .collect::<Vec<_>>();
            if raw_signatures.iter().any(|(_, signature)| signature.is_none()) {
                return false;
            }
            for (id, signature) in raw_signatures {
                let signature = signature.expect("root signature checked above");
                let next = semantic_ids.len() as u32;
                let semantic = *semantic_ids.entry(signature).or_insert(next);
                semantic_by_suffix[id as usize] = semantic;
            }
        }
        true
    };

    if cut_filtered_root && !run_root_pass(Some(&needed), &mut semantic_ids, &mut semantic_by_suffix) {
        // The shortcut is only valid when every continuation observed by a
        // candidate-cut suffix is itself candidate-cut-observable and therefore
        // already has a shorter semantic. If not, discard the partial result and
        // run the full exact suffix dynamic program.
        cut_filtered_fallback = true;
        semantic_ids.clear();
        semantic_by_suffix.fill(u32::MAX);
        assert!(run_root_pass(None, &mut semantic_ids, &mut semantic_by_suffix));
    } else if !cut_filtered_root {
        assert!(run_root_pass(None, &mut semantic_ids, &mut semantic_by_suffix));
    }
    let root_ms = root_started.elapsed().as_secs_f64() * 1000.0;

    let finish_started = Instant::now();
    partials
        .par_iter_mut()
        .enumerate()
        .for_each(|(token_index, partial)| {
            let mut cuts = partial.candidate_cuts;
            while cuts != 0 {
                let bit = cuts.trailing_zeros() as usize;
                cuts &= cuts - 1;
                let cut = bit + 1;
                let suffix_id = match &suffix_source {
                    CommonAtomSuffixSource::Local(local) => local.suffix_id(token_index, cut),
                    CommonAtomSuffixSource::Prepared { context, .. } => {
                        let original_index = context.representative_original_indices[token_index];
                        let source = &context.index.all_tokens;
                        source.suffix_ids[source.token_offsets[original_index] + cut]
                    }
                } as usize;
                let semantic = semantic_by_suffix[suffix_id];
                debug_assert_ne!(semantic, u32::MAX);
                partial.root_semantics.push((cut as u8, semantic));
                common_atom_fingerprint_mix(
                    &mut partial.fingerprint,
                    cut as u64 | ((semantic as u64) << 8),
                );
            }
        });
    let finish_ms = finish_started.elapsed().as_secs_f64() * 1000.0;

    let group_started = Instant::now();
    let parallel_group = std::env::var_os("GLRMASK_DISABLE_COMMON_ATOM_PARALLEL_GROUP").is_none()
        && partials.len() >= 8_192
        && rayon::current_num_threads() > 1;
    let mut classes = if parallel_group {
        // Do the expensive exact-key comparisons in independent token chunks.
        // Fingerprints are only routing keys: every local bucket and every
        // cross-chunk merge is verified against the full exact trace key.
        let chunk_size = 2_048usize;
        let local_classes = partials
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let base = chunk_index * chunk_size;
                let mut by_hash = FxHashMap::<u64, Vec<usize>>::default();
                for (offset, partial) in chunk.iter().enumerate() {
                    by_hash
                        .entry(partial.fingerprint)
                        .or_default()
                        .push(base + offset);
                }
                let mut exact_classes = Vec::<(u64, Vec<usize>)>::new();
                for (fingerprint, bucket) in by_hash {
                    let mut collision_classes = Vec::<Vec<usize>>::new();
                    for index in bucket {
                        if let Some(class) = collision_classes.iter_mut().find(|class| {
                            partials[class[0]] == partials[index]
                        }) {
                            class.push(index);
                        } else {
                            collision_classes.push(vec![index]);
                        }
                    }
                    exact_classes.extend(
                        collision_classes
                            .into_iter()
                            .map(|class| (fingerprint, class)),
                    );
                }
                exact_classes
            })
            .collect::<Vec<_>>();

        let mut merged = FxHashMap::<u64, Vec<Vec<usize>>>::default();
        for (fingerprint, class) in local_classes.into_iter().flatten() {
            let candidates = merged.entry(fingerprint).or_default();
            if let Some(existing) = candidates.iter_mut().find(|existing| {
                partials[existing[0]] == partials[class[0]]
            }) {
                existing.extend(class);
            } else {
                candidates.push(class);
            }
        }
        merged
            .into_values()
            .flatten()
            .map(|mut class| {
                class.sort_unstable();
                class
            })
            .collect::<Vec<_>>()
    } else {
        let mut by_hash = FxHashMap::<u64, Vec<usize>>::default();
        by_hash.reserve(tokens.len() / 8);
        for (token_index, partial) in partials.iter().enumerate() {
            by_hash.entry(partial.fingerprint).or_default().push(token_index);
        }
        let mut classes = Vec::<Vec<usize>>::with_capacity(by_hash.len());
        for bucket in by_hash.into_values() {
            let first = bucket[0];
            if bucket
                .iter()
                .skip(1)
                .all(|&index| partials[index] == partials[first])
            {
                classes.push(bucket);
                continue;
            }
            let mut exact = FxHashMap::<CommonAtomPartialTrace, Vec<usize>>::default();
            for index in bucket {
                exact.entry(partials[index].clone()).or_default().push(index);
            }
            classes.extend(exact.into_values());
        }
        classes
    };
    classes.sort_unstable();
    let group_ms = group_started.elapsed().as_secs_f64() * 1000.0;

    if profile {
        eprintln!(
            "[glrmask/profile][common_atom_phased] tokens={} suffixes={} semantics={} classes={} cut_filtered_root={} cut_filtered_suffixes={} cut_filtered_fallback={} parallel_group={} trace_ms={:.3} suffix_build_ms={:.3} root_ms={:.3} finish_ms={:.3} group_ms={:.3} total_ms={:.3}",
            tokens.len(), suffix_count, semantic_ids.len(), classes.len(), cut_filtered_root,
            cut_filtered_suffixes, cut_filtered_fallback, parallel_group,
            trace_ms, suffix_build_ms, root_ms, finish_ms, group_ms,
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    classes
}

fn classify_tokens<S: AsRef<[u8]> + Sync>(
    machine: &TraceMachine,
    tokens: &[S],
    prepared_suffixes: Option<PreparedSuffixContext<'_>>,
) -> Vec<Vec<usize>> {
    let parallel_phased_forced =
        std::env::var_os("GLRMASK_COMMON_ATOM_PARALLEL_PHASED").is_some();
    if std::env::var_os("GLRMASK_DISABLE_COMMON_ATOM_PARALLEL_PHASED").is_none()
        && rayon::current_num_threads() > 1
        && (parallel_phased_forced || tokens.len() >= 32_768)
        && tokens.iter().all(|token| token.as_ref().len() <= 64)
    {
        classify_tokens_phased(machine, tokens, prepared_suffixes)
    } else {
        classify_tokens_serial(machine, tokens)
    }
}

fn try_find_common_atom_preclasses_impl<S: AsRef<[u8]> + Sync>(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    tokens: &[S],
    prepared_suffixes: Option<PreparedSuffixContext<'_>>,
) -> Option<CommonAtomPreclasses> {
    if std::env::var_os("GLRMASK_DISABLE_L2P_COMMON_ATOM_PRECLASS").is_some()
        || tokens.len() < MIN_TOKENS
        || tokens.len() > max_tokens()
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
    let classes = classify_tokens(&machine, tokens, prepared_suffixes);
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

pub fn try_find_common_atom_preclasses<S: AsRef<[u8]> + Sync>(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    tokens: &[S],
) -> Option<CommonAtomPreclasses> {
    try_find_common_atom_preclasses_impl(tokenizer, active_groups, tokens, None)
}

pub fn try_find_common_atom_preclasses_with_vocab<S: AsRef<[u8]> + Sync>(
    tokenizer: &Tokenizer,
    active_groups: Option<&[bool]>,
    tokens: &[S],
    vocab: &Vocab,
    representative_original_indices: &[usize],
) -> Option<CommonAtomPreclasses> {
    if let Some(view) = vocab.vocab_derived_cache_get::<PreparedCommonAtomSuffixView>() {
        let mapped = representative_original_indices
            .iter()
            .map(|&index| view.parent_entry_indices[index])
            .collect::<Vec<_>>();
        return try_find_common_atom_preclasses_impl(
            tokenizer,
            active_groups,
            tokens,
            Some(PreparedSuffixContext {
                index: &view.index,
                representative_original_indices: &mapped,
            }),
        );
    }
    let prepared = vocab.vocab_derived_cache_get::<PreparedCommonAtomSuffixIndex>();
    try_find_common_atom_preclasses_impl(
        tokenizer,
        active_groups,
        tokens,
        prepared.as_deref().map(|index| PreparedSuffixContext {
            index,
            representative_original_indices,
        }),
    )
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
