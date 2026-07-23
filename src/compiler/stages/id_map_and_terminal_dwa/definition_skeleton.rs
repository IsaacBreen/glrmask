//! Analysis-only branch-relative skeletons for fixed terminal definitions.
//!
//! A fixed literal is viewed as an independent logical scanner component. For
//! one vocabulary-partition byte alphabet, every maximal run of bytes outside
//! that alphabet is represented by one abstract opaque edge. The edge is not a
//! real byte transition: partition tokens cannot execute it. It only separates
//! externally-entered residual classes on opposite sides of the opaque run.
//!
//! This module deliberately does not alter the production tokenizer or remove
//! terminal labels. It measures scanner-topology sharing only.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::automata::lexer::compile::compile_terminal_expr_dfa;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::{DFA, Lexer};
use crate::automata::regex::Expr;
use crate::ds::u8set::U8Set;
use crate::Vocab;

use super::l2p::equivalence_analysis::state_equivalence::restricted_observation::hopcroft_refine_sparse_edges;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum SkeletonAtom {
    Visible(u8),
    Opaque,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SkeletonSignature(Vec<SkeletonAtom>);

#[derive(Debug, Clone)]
struct LiteralSkeleton {
    signature: SkeletonSignature,
    source_to_skeleton: Vec<usize>,
}

impl LiteralSkeleton {
    fn build(source: &[u8], alphabet: U8Set) -> Self {
        let mut atoms = Vec::with_capacity(source.len());
        let mut source_to_skeleton = vec![0usize; source.len() + 1];
        let mut source_position = 0usize;
        let mut skeleton_position = 0usize;

        while source_position < source.len() {
            if alphabet.contains(source[source_position]) {
                atoms.push(SkeletonAtom::Visible(source[source_position]));
                source_position += 1;
                skeleton_position += 1;
                source_to_skeleton[source_position] = skeleton_position;
                continue;
            }

            let run_start = source_position;
            while source_position < source.len()
                && !alphabet.contains(source[source_position])
            {
                source_position += 1;
            }

            atoms.push(SkeletonAtom::Opaque);

            // Every proper prefix of the opaque run has identical behaviour
            // over the branch alphabet and maps to the state before the
            // abstract edge. The state after the complete run maps to the
            // opposite side, preserving externally-entered residuals.
            for position in (run_start + 1)..source_position {
                source_to_skeleton[position] = skeleton_position;
            }
            skeleton_position += 1;
            source_to_skeleton[source_position] = skeleton_position;
        }

        Self {
            signature: SkeletonSignature(atoms),
            source_to_skeleton,
        }
    }

    fn source_state_count(&self) -> usize {
        self.source_to_skeleton.len()
    }

    fn skeleton_state_count(&self) -> usize {
        self.signature.0.len() + 1
    }

    fn source_step(source: &[u8], state: usize, byte: u8) -> Option<usize> {
        (state < source.len() && source[state] == byte).then_some(state + 1)
    }

    fn skeleton_step(&self, state: usize, byte: u8) -> Option<usize> {
        match self.signature.0.get(state) {
            Some(SkeletonAtom::Visible(expected)) if *expected == byte => Some(state + 1),
            Some(SkeletonAtom::Visible(_) | SkeletonAtom::Opaque) | None => None,
        }
    }

    fn certify(&self, source: &[u8], alphabet: U8Set) -> Result<(), &'static str> {
        if self.source_to_skeleton.len() != source.len() + 1 {
            return Err("source map does not cover every residual");
        }
        if self.source_to_skeleton.first().copied() != Some(0) {
            return Err("source root does not map to skeleton root");
        }
        if self.source_to_skeleton.iter().any(|&state| state >= self.skeleton_state_count()) {
            return Err("source map target is outside skeleton");
        }

        for source_state in 0..=source.len() {
            let skeleton_state = self.source_to_skeleton[source_state];
            let source_final = source_state == source.len();
            let skeleton_final = skeleton_state == self.signature.0.len();
            if source_final != skeleton_final {
                return Err("finalizer observation differs");
            }

            let source_future = source_state < source.len();
            let skeleton_future = skeleton_state < self.signature.0.len();
            if source_future != skeleton_future {
                return Err("possible-future observation differs");
            }

            for byte in alphabet.iter() {
                let mapped_source_target = Self::source_step(source, source_state, byte)
                    .map(|target| self.source_to_skeleton[target]);
                let skeleton_target = self.skeleton_step(skeleton_state, byte);
                if mapped_source_target != skeleton_target {
                    return Err("restricted transition homomorphism differs");
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct LiteralMember {
    terminal: usize,
    bytes: Vec<u8>,
    skeleton: LiteralSkeleton,
}

#[derive(Debug)]
struct PartitionAnalysis {
    alphabet: U8Set,
    members: Vec<Option<LiteralMember>>,
    unsupported_shapes: Vec<&'static str>,
    construction_time: Duration,
    certification_time: Duration,
    certification_failures: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct RestrictedStateSignature {
    finalizes_terminal: bool,
    terminal_remains_possible: bool,
    transitions: Vec<(u8, u32)>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct CanonicalRestrictedQuotient {
    root_class: u32,
    states: Vec<RestrictedStateSignature>,
}

#[derive(Debug, Clone)]
struct RestrictedQuotient {
    canonical: CanonicalRestrictedQuotient,
    source_to_quotient: Vec<u32>,
}

impl RestrictedQuotient {
    fn build(dfa: &DFA, alphabet: U8Set) -> Result<Self, &'static str> {
        if dfa.has_epsilon_transitions() {
            return Err("epsilon_singleton_dfa");
        }
        if dfa.num_states() == 0 {
            return Err("empty_singleton_dfa");
        }

        let observations = (0..dfa.num_states() as u32)
            .map(|state| {
                (
                    dfa.finalizers(state).contains(0),
                    dfa.possible_future_group_ids(state).contains(0),
                )
            })
            .collect::<Vec<_>>();
        let mut classes = canonical_class_ids(&observations);

        loop {
            let signatures = (0..dfa.num_states() as u32)
                .map(|state| RestrictedStateSignature {
                    finalizes_terminal: observations[state as usize].0,
                    terminal_remains_possible: observations[state as usize].1,
                    transitions: dfa
                        .transitions(state)
                        .filter(|(byte, _)| alphabet.contains(*byte))
                        .map(|(byte, target)| (byte, classes[target as usize]))
                        .collect(),
                })
                .collect::<Vec<_>>();
            let next_classes = canonical_class_ids(&signatures);
            if next_classes == classes {
                let mut states = signatures;
                states.sort_unstable();
                states.dedup();
                return Ok(Self {
                    canonical: CanonicalRestrictedQuotient {
                        root_class: classes[0],
                        states,
                    },
                    source_to_quotient: classes,
                });
            }
            classes = next_classes;
        }
    }

    fn certify(&self, dfa: &DFA, alphabet: &[u8]) -> Result<(), &'static str> {
        if self.source_to_quotient.len() != dfa.num_states() {
            return Err("generic_source_map_does_not_cover_every_residual");
        }
        if self.source_to_quotient.first().copied() != Some(self.canonical.root_class) {
            return Err("generic_root_mapping_differs");
        }

        for state in 0..dfa.num_states() as u32 {
            let quotient_state = self.source_to_quotient[state as usize] as usize;
            let Some(signature) = self.canonical.states.get(quotient_state) else {
                return Err("generic_source_map_target_out_of_range");
            };
            if signature.finalizes_terminal != dfa.finalizers(state).contains(0) {
                return Err("generic_finalizer_observation_differs");
            }
            if signature.terminal_remains_possible
                != dfa.possible_future_group_ids(state).contains(0)
            {
                return Err("generic_future_observation_differs");
            }
            for &byte in alphabet {
                let mapped_source_target = dfa
                    .step(state, byte)
                    .map(|target| self.source_to_quotient[target as usize]);
                let quotient_target = signature
                    .transitions
                    .binary_search_by_key(&byte, |&(transition_byte, _)| transition_byte)
                    .ok()
                    .map(|index| signature.transitions[index].1);
                if quotient_target != mapped_source_target {
                    return Err("generic_restricted_transition_homomorphism_differs");
                }
            }
        }

        Ok(())
    }
}

fn canonical_class_ids<T: Ord + Clone>(values: &[T]) -> Vec<u32> {
    let mut unique = values.to_vec();
    unique.sort_unstable();
    unique.dedup();
    values
        .iter()
        .map(|value| unique.binary_search(value).expect("canonical value") as u32)
        .collect()
}

type SingletonDfaCell = Arc<OnceLock<Arc<DFA>>>;

fn singleton_dfa_cache() -> &'static Mutex<HashMap<Expr, SingletonDfaCell>> {
    static CACHE: OnceLock<Mutex<HashMap<Expr, SingletonDfaCell>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cached_singleton_dfa(expr: &Expr) -> (Arc<DFA>, bool, Duration) {
    let cell = {
        let mut cache = singleton_dfa_cache()
            .lock()
            .expect("definition singleton-DFA cache poisoned");
        Arc::clone(
            cache
                .entry(expr.clone())
                .or_insert_with(|| Arc::new(OnceLock::new())),
        )
    };
    let compiled_here = Cell::new(false);
    let started_at = Instant::now();
    let dfa = Arc::clone(cell.get_or_init(|| {
        compiled_here.set(true);
        Arc::new(compile_terminal_expr_dfa(expr))
    }));
    (dfa, compiled_here.get(), started_at.elapsed())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct GlobalTopologyKey {
    root_class: u32,
    classes: Vec<u32>,
}

#[derive(Debug)]
struct GenericMember {
    terminal: usize,
    example: String,
    source_states: usize,
    topology: GlobalTopologyKey,
    source_to_quotient: Vec<u32>,
}

#[derive(Debug)]
struct SourceMachine {
    terminal: usize,
    example: String,
    dfa: Arc<DFA>,
    offset: usize,
}

#[derive(Debug)]
struct GenericPartitionAnalysis {
    members: Vec<Option<GenericMember>>,
    failures: Vec<Option<&'static str>>,
    singleton_machine_time: Duration,
    quotient_time: Duration,
    certification_time: Duration,
    singleton_compiles: usize,
    singleton_cache_reuses: usize,
    global_states: usize,
    relevant_edges: usize,
    refinement_memory_bytes_estimate: usize,
}

impl GenericPartitionAnalysis {
    fn build(tokenizer: &Tokenizer, alphabet: U8Set, active_mask: &[bool]) -> Self {
        let terminal_count = tokenizer.num_terminals() as usize;
        let mut members = (0..terminal_count).map(|_| None).collect::<Vec<_>>();
        let mut failures = vec![None; terminal_count];
        let mut machines = Vec::<SourceMachine>::new();
        let mut singleton_machine_time = Duration::ZERO;
        let mut singleton_compiles = 0usize;
        let mut singleton_cache_reuses = 0usize;
        let mut global_states = 0usize;

        for terminal in 0..terminal_count {
            if !active_mask.get(terminal).copied().unwrap_or(false) {
                continue;
            }
            let Some(expr) = tokenizer.terminal_expr(terminal as u32) else {
                failures[terminal] = Some("missing_definition");
                continue;
            };

            let singleton_result = catch_unwind(AssertUnwindSafe(|| cached_singleton_dfa(expr)));
            let Ok((dfa, compiled_here, source_time)) = singleton_result else {
                failures[terminal] = Some("singleton_compile_panic");
                continue;
            };
            singleton_machine_time += source_time;
            if compiled_here {
                singleton_compiles += 1;
            } else {
                singleton_cache_reuses += 1;
            }
            if dfa.has_epsilon_transitions() {
                failures[terminal] = Some("epsilon_singleton_dfa");
                continue;
            }
            if dfa.num_states() == 0 {
                failures[terminal] = Some("empty_singleton_dfa");
                continue;
            }

            let mut literal = Vec::new();
            let example = if fixed_literal_bytes(expr, &mut literal) {
                escaped_bytes(&literal)
            } else {
                format!("<{}>", expr_shape(expr))
            };
            machines.push(SourceMachine {
                terminal,
                example,
                dfa,
                offset: global_states,
            });
            global_states += machines.last().expect("source machine").dfa.num_states();
        }

        let mut observations = Vec::<(bool, bool)>::with_capacity(global_states);
        let mut state_machine = Vec::<usize>::with_capacity(global_states);
        let mut offsets = Vec::<u32>::with_capacity(global_states + 1);
        let mut edge_bytes = Vec::<u8>::new();
        let mut edge_targets = Vec::<u32>::new();
        offsets.push(0);
        for (machine_index, machine) in machines.iter().enumerate() {
            for local_state in 0..machine.dfa.num_states() as u32 {
                observations.push((
                    machine.dfa.finalizers(local_state).contains(0),
                    machine
                        .dfa
                        .possible_future_group_ids(local_state)
                        .contains(0),
                ));
                state_machine.push(machine_index);
                for (byte, target) in machine.dfa.transitions(local_state) {
                    if alphabet.contains(byte) {
                        edge_bytes.push(byte);
                        edge_targets.push((machine.offset + target as usize) as u32);
                    }
                }
                offsets.push(edge_bytes.len() as u32);
            }
        }

        let initial_classes = canonical_class_ids(&observations);
        let relevant_edges = edge_bytes.len();
        let refinement_memory_bytes_estimate = offsets
            .len()
            .saturating_mul(std::mem::size_of::<u32>())
            .saturating_add(edge_bytes.len().saturating_mul(std::mem::size_of::<u8>()))
            .saturating_add(edge_targets.len().saturating_mul(std::mem::size_of::<u32>()))
            .saturating_add(
                initial_classes
                    .len()
                    .saturating_mul(std::mem::size_of::<u32>()),
            )
            .saturating_add(
                state_machine
                    .len()
                    .saturating_mul(std::mem::size_of::<usize>()),
            );
        let quotient_started_at = Instant::now();
        let global_classes = hopcroft_refine_sparse_edges(
            &initial_classes,
            offsets,
            edge_bytes,
            edge_targets,
        );
        let quotient_time = quotient_started_at.elapsed();

        let Some(global_classes) = global_classes else {
            for machine in &machines {
                failures[machine.terminal] = Some("invalid_global_restricted_graph");
            }
            return Self {
                members,
                failures,
                singleton_machine_time,
                quotient_time,
                certification_time: Duration::ZERO,
                singleton_compiles,
                singleton_cache_reuses,
                global_states,
                relevant_edges,
                refinement_memory_bytes_estimate,
            };
        };

        let certification_started_at = Instant::now();
        let class_count = global_classes
            .iter()
            .copied()
            .max()
            .map_or(0usize, |class| class as usize + 1);
        let mut representatives = vec![usize::MAX; class_count];
        for (state, &class) in global_classes.iter().enumerate() {
            representatives[class as usize] = representatives[class as usize].min(state);
        }
        let mut certification_failure = None;
        'certify: for (state, &class) in global_classes.iter().enumerate() {
            let representative = representatives[class as usize];
            if representative == usize::MAX || observations[state] != observations[representative] {
                certification_failure = Some("global_observation_certificate_failed");
                break;
            }

            // A deterministic partial byte graph has one common implicit dead
            // target for every absent edge. CharTransitions iterates in byte
            // order, so equality of the complete sparse visible edge lists is
            // exactly equality over every byte in the restricted alphabet,
            // including all jointly absent transitions.
            let current_machine = &machines[state_machine[state]];
            let representative_machine = &machines[state_machine[representative]];
            let state_local = (state - current_machine.offset) as u32;
            let representative_local =
                (representative - representative_machine.offset) as u32;
            let state_edges = current_machine
                .dfa
                .transitions(state_local)
                .filter(|(byte, _)| alphabet.contains(*byte))
                .map(|(byte, target)| {
                    (
                        byte,
                        global_classes[current_machine.offset + target as usize],
                    )
                });
            let representative_edges = representative_machine
                .dfa
                .transitions(representative_local)
                .filter(|(byte, _)| alphabet.contains(*byte))
                .map(|(byte, target)| {
                    (
                        byte,
                        global_classes[representative_machine.offset + target as usize],
                    )
                });
            if !state_edges.eq(representative_edges) {
                certification_failure = Some("global_transition_certificate_failed");
                break 'certify;
            }
        }
        let certification_time = certification_started_at.elapsed();

        if let Some(reason) = certification_failure {
            for machine in &machines {
                failures[machine.terminal] = Some(reason);
            }
        } else {
            for machine in &machines {
                let start = machine.offset;
                let end = start + machine.dfa.num_states();
                let source_to_quotient = global_classes[start..end].to_vec();
                let mut classes = source_to_quotient.clone();
                classes.sort_unstable();
                classes.dedup();
                members[machine.terminal] = Some(GenericMember {
                    terminal: machine.terminal,
                    example: machine.example.clone(),
                    source_states: machine.dfa.num_states(),
                    topology: GlobalTopologyKey {
                        root_class: source_to_quotient[0],
                        classes,
                    },
                    source_to_quotient,
                });
            }
        }

        Self {
            members,
            failures,
            singleton_machine_time,
            quotient_time,
            certification_time,
            singleton_compiles,
            singleton_cache_reuses,
            global_states,
            relevant_edges,
            refinement_memory_bytes_estimate,
        }
    }
}

fn fixed_literal_bytes(expr: &Expr, output: &mut Vec<u8>) -> bool {
    match expr {
        Expr::U8Seq(bytes) => {
            output.extend_from_slice(bytes);
            true
        }
        Expr::U8Class(bytes) if bytes.len() == 1 => {
            output.push(bytes.iter().next().expect("singleton byte class"));
            true
        }
        Expr::Seq(parts) => parts.iter().all(|part| fixed_literal_bytes(part, output)),
        Expr::Shared(inner) => fixed_literal_bytes(inner, output),
        Expr::Epsilon => true,
        _ => false,
    }
}

fn expr_shape(expr: &Expr) -> &'static str {
    match expr {
        Expr::U8Seq(_) => "u8_seq",
        Expr::U8Class(_) => "u8_class",
        Expr::Dfa(_) => "dfa",
        Expr::Intersect { .. } => "intersect",
        Expr::Seq(_) => "seq",
        Expr::Choice(_) => "choice",
        Expr::Exclude { .. } => "exclude",
        Expr::Repeat { .. } => "repeat",
        Expr::Shared(_) => "shared",
        Expr::Epsilon => "epsilon",
    }
}

impl PartitionAnalysis {
    fn build(tokenizer: &Tokenizer, vocab: &Vocab) -> Self {
        let alphabet = U8Set::from_bytes(vocab.relevant_bytes().as_ref());
        let terminal_count = tokenizer.num_terminals() as usize;
        let mut members = Vec::with_capacity(terminal_count);
        let mut unsupported_shapes = Vec::with_capacity(terminal_count);
        let mut construction_time = Duration::ZERO;
        let mut certification_time = Duration::ZERO;
        let mut certification_failures = 0usize;

        for terminal in 0..terminal_count {
            let Some(expr) = tokenizer.terminal_expr(terminal as u32) else {
                unsupported_shapes.push("missing_definition");
                members.push(None);
                continue;
            };

            let construction_started_at = Instant::now();
            let mut bytes = Vec::new();
            let is_literal = fixed_literal_bytes(expr, &mut bytes);
            if !is_literal {
                construction_time += construction_started_at.elapsed();
                unsupported_shapes.push(expr_shape(expr));
                members.push(None);
                continue;
            }
            let skeleton = LiteralSkeleton::build(&bytes, alphabet);
            construction_time += construction_started_at.elapsed();

            let certification_started_at = Instant::now();
            let certified = skeleton.certify(&bytes, alphabet).is_ok();
            certification_time += certification_started_at.elapsed();
            if !certified {
                certification_failures += 1;
                unsupported_shapes.push("literal_certification_failed");
                members.push(None);
                continue;
            }

            unsupported_shapes.push("literal");
            members.push(Some(LiteralMember {
                terminal,
                bytes,
                skeleton,
            }));
        }

        Self {
            alphabet,
            members,
            unsupported_shapes,
            construction_time,
            certification_time,
            certification_failures,
        }
    }
}

fn report_enabled(partition_label: &str) -> bool {
    let enabled = std::env::var("GLRMASK_DEFINITION_SKELETON_REPORT")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false);
    if !enabled {
        return false;
    }

    std::env::var("GLRMASK_DEFINITION_SKELETON_REPORT_FILTER")
        .map(|filter| {
            filter
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .any(|value| partition_label == value)
        })
        .unwrap_or(true)
}

fn escaped_bytes(bytes: &[u8]) -> String {
    bytes
        .iter()
        .flat_map(|byte| std::ascii::escape_default(*byte))
        .map(char::from)
        .collect()
}

fn signature_text(signature: &SkeletonSignature) -> String {
    signature
        .0
        .iter()
        .map(|atom| match atom {
            SkeletonAtom::Visible(byte) => format!("{byte:02x}"),
            SkeletonAtom::Opaque => "O".to_string(),
        })
        .collect::<Vec<_>>()
        .join("-")
}

fn report_scope(
    partition_label: &str,
    scope: &str,
    source_states: usize,
    active_mask: &[bool],
    analysis: &PartitionAnalysis,
) {
    let active_terminals = active_mask.iter().filter(|&&active| active).count();
    let mut classes = BTreeMap::<SkeletonSignature, Vec<&LiteralMember>>::new();
    let mut unsupported = BTreeMap::<&'static str, usize>::new();
    let mut original_singleton_states = 0usize;
    let mut skeleton_states_before_sharing = 0usize;
    let mut map_entries = 0usize;

    for (terminal, &active) in active_mask.iter().enumerate() {
        if !active {
            continue;
        }
        if let Some(member) = analysis.members.get(terminal).and_then(Option::as_ref) {
            original_singleton_states += member.skeleton.source_state_count();
            skeleton_states_before_sharing += member.skeleton.skeleton_state_count();
            map_entries += member.skeleton.source_to_skeleton.len();
            classes
                .entry(member.skeleton.signature.clone())
                .or_default()
                .push(member);
        } else {
            *unsupported
                .entry(
                    analysis
                        .unsupported_shapes
                        .get(terminal)
                        .copied()
                        .unwrap_or("missing_definition"),
                )
                .or_default() += 1;
        }
    }

    let literal_terminals = classes.values().map(Vec::len).sum::<usize>();
    let skeleton_states_after_sharing = classes
        .keys()
        .map(|signature| signature.0.len() + 1)
        .sum::<usize>();
    let reduction_pct = if original_singleton_states == 0 {
        0.0
    } else {
        100.0
            * (original_singleton_states.saturating_sub(skeleton_states_after_sharing)) as f64
            / original_singleton_states as f64
    };
    let unsupported_text = unsupported
        .iter()
        .map(|(shape, count)| format!("{shape}:{count}"))
        .collect::<Vec<_>>()
        .join(",");

    eprintln!(
        "[glrmask/profile][definition_skeleton_scope] partition={} scope={} source_states={} active_terminals={} literal_fast_path={} unsupported={} skeleton_classes={} original_singleton_states={} skeleton_states_before_sharing={} skeleton_states_after_sharing={} literal_coordinate_with_dispatch={} reduction_pct={:.2} map_entries={} map_bytes_estimate={} unsupported_shapes={}",
        partition_label,
        scope,
        source_states,
        active_terminals,
        literal_terminals,
        active_terminals.saturating_sub(literal_terminals),
        classes.len(),
        original_singleton_states,
        skeleton_states_before_sharing,
        skeleton_states_after_sharing,
        skeleton_states_after_sharing.saturating_add(1),
        reduction_pct,
        map_entries,
        map_entries.saturating_mul(std::mem::size_of::<usize>()),
        unsupported_text,
    );

    let mut largest = classes.into_iter().collect::<Vec<_>>();
    largest.sort_by(|(left_signature, left_members), (right_signature, right_members)| {
        right_members
            .len()
            .cmp(&left_members.len())
            .then_with(|| left_signature.cmp(right_signature))
    });
    for (rank, (signature, members)) in largest.into_iter().take(8).enumerate() {
        let terminal_ids = members
            .iter()
            .take(12)
            .map(|member| member.terminal.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let examples = members
            .iter()
            .take(3)
            .map(|member| escaped_bytes(&member.bytes))
            .collect::<Vec<_>>()
            .join("|");
        eprintln!(
            "[glrmask/profile][definition_skeleton_class] partition={} scope={} rank={} members={} skeleton_states={} signature={} terminal_ids={} examples={}",
            partition_label,
            scope,
            rank + 1,
            members.len(),
            signature.0.len() + 1,
            signature_text(&signature),
            terminal_ids,
            examples,
        );
    }
}

fn report_generic_scope(
    partition_label: &str,
    scope: &str,
    source_states: usize,
    active_mask: &[bool],
    analysis: &GenericPartitionAnalysis,
) {
    let active_terminals = active_mask.iter().filter(|&&active| active).count();
    let mut classes = BTreeMap::<GlobalTopologyKey, Vec<&GenericMember>>::new();
    let mut failures = BTreeMap::<&'static str, usize>::new();
    let mut source_singleton_states = 0usize;
    let mut quotient_states_before_sharing = 0usize;
    let mut map_entries = 0usize;

    for (terminal, &active) in active_mask.iter().enumerate() {
        if !active {
            continue;
        }
        if let Some(member) = analysis.members.get(terminal).and_then(Option::as_ref) {
            source_singleton_states += member.source_states;
            quotient_states_before_sharing += member.topology.classes.len();
            map_entries += member.source_to_quotient.len();
            classes
                .entry(member.topology.clone())
                .or_default()
                .push(member);
        } else {
            *failures
                .entry(
                    analysis
                        .failures
                        .get(terminal)
                        .and_then(|failure| *failure)
                        .unwrap_or("missing_generic_result"),
                )
                .or_default() += 1;
        }
    }

    let handled_terminals = classes.values().map(Vec::len).sum::<usize>();
    let quotient_states_after_sharing = classes
        .keys()
        .map(|quotient| quotient.classes.len())
        .sum::<usize>();
    let coordinate_with_dispatch = quotient_states_after_sharing.saturating_add(1);
    let singleton_reduction_pct = if source_singleton_states == 0 {
        0.0
    } else {
        100.0
            * source_singleton_states
                .saturating_sub(quotient_states_after_sharing)
                as f64
            / source_singleton_states as f64
    };
    let source_coordinate_reduction_pct = if source_states == 0 {
        0.0
    } else {
        100.0 * source_states.saturating_sub(coordinate_with_dispatch) as f64
            / source_states as f64
    };
    let failures_text = failures
        .iter()
        .map(|(reason, count)| format!("{reason}:{count}"))
        .collect::<Vec<_>>()
        .join(",");

    eprintln!(
        "[glrmask/profile][definition_quotient_scope] partition={} scope={} source_states={} active_terminals={} handled_terminals={} failures={} quotient_classes={} source_singleton_states={} quotient_states_before_sharing={} quotient_states_after_sharing={} coordinate_with_dispatch={} singleton_reduction_pct={:.2} source_coordinate_reduction_pct={:.2} map_entries={} map_bytes_estimate={} failure_reasons={}",
        partition_label,
        scope,
        source_states,
        active_terminals,
        handled_terminals,
        active_terminals.saturating_sub(handled_terminals),
        classes.len(),
        source_singleton_states,
        quotient_states_before_sharing,
        quotient_states_after_sharing,
        coordinate_with_dispatch,
        singleton_reduction_pct,
        source_coordinate_reduction_pct,
        map_entries,
        map_entries.saturating_mul(std::mem::size_of::<u32>()),
        failures_text,
    );

    let mut largest = classes.into_iter().collect::<Vec<_>>();
    largest.sort_by(|(left_quotient, left_members), (right_quotient, right_members)| {
        right_members
            .len()
            .cmp(&left_members.len())
            .then_with(|| left_quotient.cmp(right_quotient))
    });
    for (rank, (quotient, members)) in largest.into_iter().take(8).enumerate() {
        let terminal_ids = members
            .iter()
            .take(12)
            .map(|member| member.terminal.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let examples = members
            .iter()
            .take(3)
            .map(|member| member.example.as_str())
            .collect::<Vec<_>>()
            .join("|");
        eprintln!(
            "[glrmask/profile][definition_quotient_class] partition={} scope={} rank={} members={} quotient_states={} root_class={} terminal_ids={} examples={}",
            partition_label,
            scope,
            rank + 1,
            members.len(),
            quotient.classes.len(),
            quotient.root_class,
            terminal_ids,
            examples,
        );
    }
}

pub(crate) fn report_partition(
    partition_label: &str,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    source_states: usize,
    l1_mask: &[bool],
    l2p_mask: &[bool],
) {
    if !report_enabled(partition_label) {
        return;
    }

    let total_started_at = Instant::now();
    let analysis = PartitionAnalysis::build(tokenizer, vocab);
    let literal_terminals = analysis.members.iter().filter(|member| member.is_some()).count();
    let map_entries = analysis
        .members
        .iter()
        .filter_map(Option::as_ref)
        .map(|member| member.skeleton.source_to_skeleton.len())
        .sum::<usize>();
    eprintln!(
        "[glrmask/profile][definition_skeleton_partition] partition={} terminal_count={} alphabet_bytes={} literal_fast_path={} unsupported={} certification_failures={} construction_ms={:.3} certification_ms={:.3} map_entries={} map_bytes_estimate={}",
        partition_label,
        analysis.members.len(),
        analysis.alphabet.len(),
        literal_terminals,
        analysis.members.len().saturating_sub(literal_terminals),
        analysis.certification_failures,
        analysis.construction_time.as_secs_f64() * 1000.0,
        analysis.certification_time.as_secs_f64() * 1000.0,
        map_entries,
        map_entries.saturating_mul(std::mem::size_of::<usize>()),
    );

    let partition_mask = l1_mask
        .iter()
        .zip(l2p_mask)
        .map(|(&l1, &l2p)| l1 || l2p)
        .collect::<Vec<_>>();
    report_scope(
        partition_label,
        "partition",
        source_states,
        &partition_mask,
        &analysis,
    );
    report_scope(
        partition_label,
        "l1",
        source_states,
        l1_mask,
        &analysis,
    );
    report_scope(
        partition_label,
        "l2p",
        source_states,
        l2p_mask,
        &analysis,
    );
    eprintln!(
        "[glrmask/profile][definition_skeleton_total] partition={} total_ms={:.3}",
        partition_label,
        total_started_at.elapsed().as_secs_f64() * 1000.0,
    );

    let generic_started_at = Instant::now();
    let generic = GenericPartitionAnalysis::build(tokenizer, analysis.alphabet, &partition_mask);
    let generic_active = partition_mask.iter().filter(|&&active| active).count();
    let generic_handled = partition_mask
        .iter()
        .enumerate()
        .filter(|&(terminal, active)| {
            *active && generic.members.get(terminal).is_some_and(Option::is_some)
        })
        .count();
    let generic_failures = generic_active.saturating_sub(generic_handled);
    eprintln!(
        "[glrmask/profile][definition_quotient_partition] partition={} terminal_count={} active_terminals={} handled_terminals={} failures={} singleton_compiles={} singleton_cache_reuses={} global_states={} relevant_edges={} refinement_memory_bytes_estimate={} singleton_machine_ms={:.3} quotient_ms={:.3} certification_ms={:.3}",
        partition_label,
        generic.members.len(),
        generic_active,
        generic_handled,
        generic_failures,
        generic.singleton_compiles,
        generic.singleton_cache_reuses,
        generic.global_states,
        generic.relevant_edges,
        generic.refinement_memory_bytes_estimate,
        generic.singleton_machine_time.as_secs_f64() * 1000.0,
        generic.quotient_time.as_secs_f64() * 1000.0,
        generic.certification_time.as_secs_f64() * 1000.0,
    );
    report_generic_scope(
        partition_label,
        "partition",
        source_states,
        &partition_mask,
        &generic,
    );
    report_generic_scope(
        partition_label,
        "l1",
        source_states,
        l1_mask,
        &generic,
    );
    report_generic_scope(
        partition_label,
        "l2p",
        source_states,
        l2p_mask,
        &generic,
    );
    eprintln!(
        "[glrmask/profile][definition_quotient_total] partition={} total_ms={:.3}",
        partition_label,
        generic_started_at.elapsed().as_secs_f64() * 1000.0,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::compile::build_regex_partitioned;

    fn punctuation() -> U8Set {
        U8Set::from_bytes(b"\"': -")
    }

    fn tokenizer_for_exprs(exprs: Vec<Expr>) -> Tokenizer {
        let expressions = Arc::<[Expr]>::from(exprs);
        let terminal_count = expressions.len() as u32;
        let partitions = vec![0u32; expressions.len()];
        build_regex_partitioned(&expressions, &partitions)
            .into_tokenizer(terminal_count, Some(Arc::clone(&expressions)))
    }

    fn same_partition(left: &[u32], right: &[u32]) -> bool {
        left.len() == right.len()
            && (0..left.len()).all(|i| {
                (0..left.len()).all(|j| (left[i] == left[j]) == (right[i] == right[j]))
            })
    }

    #[test]
    fn opaque_property_names_share_one_scanner_skeleton() {
        let alphabet = punctuation();
        let foo = LiteralSkeleton::build(b"\"foo\": ", alphabet);
        let bar = LiteralSkeleton::build(b"\"bar\": ", alphabet);
        assert_eq!(foo.signature, bar.signature);
        foo.certify(b"\"foo\": ", alphabet).unwrap();
        bar.certify(b"\"bar\": ", alphabet).unwrap();
    }

    #[test]
    fn visible_hyphen_splits_opaque_runs() {
        let alphabet = punctuation();
        let skeleton = LiteralSkeleton::build(b"\"foo-bar\": ", alphabet);
        assert_eq!(
            skeleton.signature.0,
            vec![
                SkeletonAtom::Visible(b'\"'),
                SkeletonAtom::Opaque,
                SkeletonAtom::Visible(b'-'),
                SkeletonAtom::Opaque,
                SkeletonAtom::Visible(b'\"'),
                SkeletonAtom::Visible(b':'),
                SkeletonAtom::Visible(b' '),
            ]
        );
        skeleton.certify(b"\"foo-bar\": ", alphabet).unwrap();
    }

    #[test]
    fn different_literal_lengths_share_one_skeleton() {
        let alphabet = punctuation();
        let short = LiteralSkeleton::build(b"\"a\": ", alphabet);
        let long = LiteralSkeleton::build(b"\"somethingLong\": ", alphabet);
        assert_eq!(short.signature, long.signature);
        short.certify(b"\"a\": ", alphabet).unwrap();
        long.certify(b"\"somethingLong\": ", alphabet).unwrap();
    }

    #[test]
    fn visible_byte_inside_text_is_preserved() {
        let alphabet = U8Set::from_bytes(b"\"o: ");
        let skeleton = LiteralSkeleton::build(b"\"foo\": ", alphabet);
        assert_eq!(
            skeleton.signature.0,
            vec![
                SkeletonAtom::Visible(b'\"'),
                SkeletonAtom::Opaque,
                SkeletonAtom::Visible(b'o'),
                SkeletonAtom::Visible(b'o'),
                SkeletonAtom::Visible(b'\"'),
                SkeletonAtom::Visible(b':'),
                SkeletonAtom::Visible(b' '),
            ]
        );
        skeleton.certify(b"\"foo\": ", alphabet).unwrap();
    }

    #[test]
    fn terminal_label_is_not_part_of_scanner_signature() {
        let alphabet = punctuation();
        let first = LiteralMember {
            terminal: 7,
            bytes: b"\"foo\": ".to_vec(),
            skeleton: LiteralSkeleton::build(b"\"foo\": ", alphabet),
        };
        let second = LiteralMember {
            terminal: 99,
            bytes: b"\"bar\": ".to_vec(),
            skeleton: LiteralSkeleton::build(b"\"bar\": ", alphabet),
        };
        assert_ne!(first.terminal, second.terminal);
        assert_eq!(first.skeleton.signature, second.skeleton.signature);
    }

    #[test]
    fn residuals_before_and_after_opaque_edge_remain_distinct() {
        let alphabet = punctuation();
        let skeleton = LiteralSkeleton::build(b"\"foo\": ", alphabet);
        let before_run = skeleton.source_to_skeleton[1];
        let after_run = skeleton.source_to_skeleton[4];
        assert_ne!(before_run, after_run);
        assert_eq!(skeleton.source_to_skeleton[2], before_run);
        assert_eq!(skeleton.source_to_skeleton[3], before_run);
        assert_eq!(skeleton.skeleton_step(after_run, b'\"'), Some(after_run + 1));
        skeleton.certify(b"\"foo\": ", alphabet).unwrap();
    }

    #[test]
    fn every_source_residual_is_certified_for_external_entry() {
        let alphabet = punctuation();
        let source = b"\"foo-bar\": ";
        let skeleton = LiteralSkeleton::build(source, alphabet);
        assert_eq!(skeleton.source_to_skeleton.len(), source.len() + 1);
        for state in 0..=source.len() {
            assert!(skeleton.source_to_skeleton[state] < skeleton.skeleton_state_count());
        }
        skeleton.certify(source, alphabet).unwrap();
    }

    #[test]
    fn full_real_byte_alphabet_needs_no_free_placeholder() {
        let alphabet = U8Set::all();
        let source = [0u8, 1, 2, 255];
        let skeleton = LiteralSkeleton::build(&source, alphabet);
        assert!(skeleton
            .signature
            .0
            .iter()
            .all(|atom| matches!(atom, SkeletonAtom::Visible(_))));
        skeleton.certify(&source, alphabet).unwrap();
    }

    #[test]
    fn generic_restricted_quotient_matches_equal_literal_skeletons() {
        let alphabet = punctuation().iter().collect::<Vec<_>>();
        let foo_dfa = compile_terminal_expr_dfa(&Expr::U8Seq(b"\"foo\": ".to_vec()));
        let bar_dfa = compile_terminal_expr_dfa(&Expr::U8Seq(b"\"bar\": ".to_vec()));
        let foo = RestrictedQuotient::build(&foo_dfa, punctuation()).unwrap();
        let bar = RestrictedQuotient::build(&bar_dfa, punctuation()).unwrap();
        foo.certify(&foo_dfa, &alphabet).unwrap();
        bar.certify(&bar_dfa, &alphabet).unwrap();
        assert_eq!(foo.canonical, bar.canonical);
    }

    #[test]
    fn generic_restricted_quotient_certifies_nonliteral_repeat() {
        let alphabet = punctuation().iter().collect::<Vec<_>>();
        let expr = Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
            min: 1,
            max: Some(4),
        };
        let dfa = compile_terminal_expr_dfa(&expr);
        let quotient = RestrictedQuotient::build(&dfa, punctuation()).unwrap();
        quotient.certify(&dfa, &alphabet).unwrap();
        assert_eq!(quotient.source_to_quotient.len(), dfa.num_states());
        assert!(quotient.canonical.states.len() <= dfa.num_states());
    }

    #[test]
    fn generic_full_alphabet_quotient_needs_no_synthetic_byte() {
        let alphabet = U8Set::all().iter().collect::<Vec<_>>();
        let expr = Expr::Choice(vec![
            Expr::U8Seq(b"foo".to_vec()),
            Expr::U8Seq(b"bar".to_vec()),
        ]);
        let dfa = compile_terminal_expr_dfa(&expr);
        let quotient = RestrictedQuotient::build(&dfa, U8Set::all()).unwrap();
        quotient.certify(&dfa, &alphabet).unwrap();
    }

    #[test]
    fn generic_singleton_compiler_lowers_nested_exclude() {
        let alphabet_set = punctuation();
        let alphabet = alphabet_set.iter().collect::<Vec<_>>();
        let expr = Expr::Seq(vec![
            Expr::U8Seq(b"\"".to_vec()),
            Expr::Exclude {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
                exclude: Box::new(Expr::U8Seq(b"b".to_vec())),
            },
            Expr::U8Seq(b"\"".to_vec()),
        ]);
        let dfa = compile_terminal_expr_dfa(&expr);
        let quotient = RestrictedQuotient::build(&dfa, alphabet_set).unwrap();
        quotient.certify(&dfa, &alphabet).unwrap();
    }

    #[test]
    fn generic_singleton_compiler_lowers_nested_intersect() {
        let alphabet_set = punctuation();
        let alphabet = alphabet_set.iter().collect::<Vec<_>>();
        let expr = Expr::Seq(vec![
            Expr::U8Seq(b"\"".to_vec()),
            Expr::Intersect {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
                intersect: Box::new(Expr::U8Class(U8Set::from_bytes(b"bc"))),
            },
            Expr::U8Seq(b"\"".to_vec()),
        ]);
        let dfa = compile_terminal_expr_dfa(&expr);
        let quotient = RestrictedQuotient::build(&dfa, alphabet_set).unwrap();
        quotient.certify(&dfa, &alphabet).unwrap();
    }

    #[test]
    fn global_hopcroft_matches_individual_reference_quotients() {
        let expressions = vec![
            Expr::U8Seq(b"\"foo\": ".to_vec()),
            Expr::U8Seq(b"\"bar\": ".to_vec()),
            Expr::Repeat {
                expr: Box::new(Expr::U8Class(U8Set::from_bytes(b"abc"))),
                min: 1,
                max: Some(4),
            },
        ];
        let tokenizer = tokenizer_for_exprs(expressions.clone());
        let active = vec![true; expressions.len()];
        let global = GenericPartitionAnalysis::build(&tokenizer, punctuation(), &active);
        assert!(global.failures.iter().all(Option::is_none));

        for (terminal, expression) in expressions.iter().enumerate() {
            let dfa = compile_terminal_expr_dfa(expression);
            let reference = RestrictedQuotient::build(&dfa, punctuation()).unwrap();
            let member = global.members[terminal].as_ref().unwrap();
            assert!(same_partition(
                &member.source_to_quotient,
                &reference.source_to_quotient,
            ));
            assert_eq!(
                member.topology.classes.len(),
                reference.canonical.states.len(),
            );
        }
        assert_eq!(
            global.members[0].as_ref().unwrap().topology,
            global.members[1].as_ref().unwrap().topology,
        );
        assert_ne!(
            global.members[0].as_ref().unwrap().topology,
            global.members[2].as_ref().unwrap().topology,
        );
    }

    #[test]
    fn global_hopcroft_shares_equivalent_nonliteral_topologies() {
        let make_repeat = |bytes: &[u8]| Expr::Repeat {
            expr: Box::new(Expr::U8Class(U8Set::from_bytes(bytes))),
            min: 1,
            max: Some(4),
        };
        let expressions = vec![make_repeat(b"abc"), make_repeat(b"xyz")];
        let tokenizer = tokenizer_for_exprs(expressions.clone());
        let active = vec![true; expressions.len()];
        let global = GenericPartitionAnalysis::build(&tokenizer, punctuation(), &active);
        assert!(global.failures.iter().all(Option::is_none));
        assert_eq!(
            global.members[0].as_ref().unwrap().topology,
            global.members[1].as_ref().unwrap().topology,
        );
    }
}
