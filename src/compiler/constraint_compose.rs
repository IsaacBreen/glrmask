//! Composition of already-compiled constraint artifacts.
//!
//! The expensive component parser DWAs are reused exactly. Their private
//! `(tokenizer-state class, vocabulary-token class)` coordinates are first
//! reconciled through the merged raw tokenizer and the shared original
//! vocabulary. Parser-state labels are transported through the table splice's
//! one-to-many relation. Default parser labels retain wildcard semantics during
//! the overlap-local union: explicit positive labels override them, unmatched
//! positive labels fall through to them, and negative labels never do.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use smallvec::SmallVec;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::lexer::tokenizer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::compile_terminal_expr_dfa;
use crate::automata::weighted_u32::determinize::determinize;
use crate::automata::weighted_u32::equivalence::find_difference;
use crate::automata::weighted_u32::dwa::{DWA, DWAState};
use crate::automata::weighted_u32::nwa::NWA;
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::{
    DEFAULT_LABEL, encode_negative_label, encode_positive_label, is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::stages::equiv_types::{
    InternalIdMap, ManyToOneIdMap, MappedArtifact,
};
use crate::compiler::stages::mapped_artifact::{WeightRefs, remap_weights_with_maps};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring;
use crate::compiler::stages::parser_dwa::build_parser_dwa_from_terminal_dwa_with_precomputed_templates;
use crate::compiler::stages::templates::characterize::characterize_selected_terminals;
use crate::compiler::stages::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};
use crate::compiler::stages::templates::Templates;
use crate::compiler::constraint_possible_matches::{
    build_internal_token_bytes_from_groups, runtime_dynamic_vocab_for_vocab,
};
use crate::compiler::glr::table::{
    Action, ComposedTable, SubgrammarTableInput, compose_subgrammar_tables,
};
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::weight::{ScopedWeightOpCache, Weight};
use crate::runtime::{Constraint, ConstraintRuntimeBackend, SpecialTokenTerminal};
use crate::Vocab;

#[inline]
fn compose_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

#[derive(Clone, Copy)]
pub(crate) struct ParserDwaComponent<'a> {
    pub(crate) constraint: &'a Constraint,
    /// Local parser state -> merged parser states.
    pub(crate) parser_state_relation: &'a [Vec<u32>],
    /// Local raw tokenizer state `s` is merged state `tokenizer_state_offset+s`.
    pub(crate) tokenizer_state_offset: u32,
}

pub(crate) struct CompiledSubgrammarInput<'a> {
    pub(crate) placeholder_terminal: u32,
    pub(crate) constraint: &'a Constraint,
}

pub(crate) struct ConstraintComposition {
    pub(crate) constraint: Constraint,
    pub(crate) terminal_offsets: Vec<u32>,
    pub(crate) tokenizer_state_offsets: Vec<u32>,
    pub(crate) parser_state_relations: Vec<Vec<Vec<u32>>>,
}

struct DirectComponentCoordinateMaps {
    local_to_global_tsids: Vec<Vec<u32>>,
    local_to_global_tokens: Vec<Vec<u32>>,
}

struct DirectComponentStateCoordinates {
    tokenizer_states: ManyToOneIdMap,
    local_to_global_tsids: Vec<Vec<Vec<u32>>>,
}

fn build_direct_component_state_coordinates(
    components: &[ParserDwaComponent<'_>],
    merged_tokenizer_state_count: usize,
) -> Result<DirectComponentStateCoordinates, String> {
    let mut state_to_global = vec![u32::MAX; merged_tokenizer_state_count];
    let mut global_to_states = vec![vec![0u32]];
    let mut state_representatives = vec![0u32];
    if let Some(reset) = state_to_global.first_mut() {
        *reset = 0;
    }

    let mut local_to_global_tsids = Vec::with_capacity(components.len());
    for component in components {
        let constraint = component.constraint;
        if constraint.state_to_internal_tsid.len() != constraint.tokenizer.num_states() as usize {
            return Err("component tokenizer-state map does not cover its runtime tokenizer".into());
        }
        if constraint.internal_tsid_to_states.is_empty() {
            return Err("component tokenizer-state map contains no internal TSIDs".into());
        }
        let mut local_map = vec![Vec::<u32>::new(); constraint.internal_tsid_to_states.len()];
        for (local_tsid, local_states) in constraint.internal_tsid_to_states.iter().enumerate() {
            if local_states.is_empty() {
                continue;
            }
            let mut merged_states = Vec::with_capacity(local_states.len());
            for &local_state in local_states {
                let merged_state = component
                    .tokenizer_state_offset
                    .checked_add(local_state)
                    .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
                if merged_state == 0 {
                    local_map[local_tsid].push(0);
                    continue;
                }
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component tokenizer state {local_state} maps outside merged tokenizer"
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to more than one component class"
                    ));
                }
                merged_states.push(merged_state);
            }
            if !merged_states.is_empty() {
                let global_tsid = global_to_states.len() as u32;
                for &merged_state in &merged_states {
                    state_to_global[merged_state as usize] = global_tsid;
                }
                state_representatives.push(merged_states[0]);
                global_to_states.push(merged_states);
                local_map[local_tsid].push(global_tsid);
            }
        }
        let local_start = constraint.tokenizer.initial_state() as usize;
        let start_tsid = constraint
            .state_to_internal_tsid
            .get(local_start)
            .copied()
            .ok_or_else(|| "component start tokenizer state has no internal TSID".to_string())?;
        let Some(start_targets) = local_map.get_mut(start_tsid as usize) else {
            return Err("component start TSID maps outside its internal TSID domain".into());
        };
        start_targets.push(0);
        start_targets.sort_unstable();
        start_targets.dedup();
        local_to_global_tsids.push(local_map);
    }
    if state_to_global.iter().any(|&tsid| tsid == u32::MAX) {
        return Err("direct component state map does not cover the merged tokenizer".into());
    }
    Ok(DirectComponentStateCoordinates {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: state_to_global,
            internal_to_originals: global_to_states,
            representative_original_ids: state_representatives,
        },
        local_to_global_tsids,
    })
}

fn build_direct_component_token_coordinates(
    components: &[ParserDwaComponent<'_>],
    original_token_ids: &[u32],
) -> Result<(ManyToOneIdMap, Vec<Vec<Vec<u32>>>), String> {
    let original_token_count = original_token_ids
        .last()
        .map_or(0, |token| *token as usize + 1);
    let mut selected_original = vec![false; original_token_count];
    for &original in original_token_ids {
        selected_original[original as usize] = true;
    }
    let mut token_to_global = vec![u32::MAX; original_token_count];
    let mut global_to_tokens = Vec::<Vec<u32>>::new();
    let mut token_representatives = Vec::<u32>::new();
    let mut local_to_global_tokens = components
        .iter()
        .map(|component| {
            vec![Vec::<u32>::new(); component.constraint.internal_token_to_tokens.len()]
        })
        .collect::<Vec<_>>();

    let mut add_token_class = |originals: Vec<u32>, tuple: Vec<u32>| -> Result<(), String> {
        if originals.is_empty() || tuple.iter().all(|&local| local == u32::MAX) {
            return Ok(());
        }
        let global = global_to_tokens.len() as u32;
        for &original in &originals {
            token_to_global[original as usize] = global;
        }
        for (component_index, &local) in tuple.iter().enumerate() {
            if local == u32::MAX {
                continue;
            }
            let Some(destinations) = local_to_global_tokens
                .get_mut(component_index)
                .and_then(|classes| classes.get_mut(local as usize))
            else {
                return Err(format!(
                    "component {component_index} token class {local} lies outside its internal token domain"
                ));
            };
            destinations.push(global);
        }
        token_representatives.push(originals[0]);
        global_to_tokens.push(originals);
        Ok(())
    };

    let parent = components[0].constraint;
    for (parent_local, originals) in parent.internal_token_to_tokens.iter().enumerate() {
        let mut groups = BTreeMap::<Vec<u32>, Vec<u32>>::new();
        for &original in originals {
            if !selected_original
                .get(original as usize)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let child_tuple = components[1..]
                .iter()
                .map(|component| {
                    component
                        .constraint
                        .original_token_to_internal
                        .get(original as usize)
                        .copied()
                        .unwrap_or(u32::MAX)
                })
                .collect::<Vec<_>>();
            groups.entry(child_tuple).or_default().push(original);
        }
        for (child_tuple, originals) in groups {
            let mut tuple = Vec::with_capacity(components.len());
            tuple.push(parent_local as u32);
            tuple.extend(child_tuple);
            add_token_class(originals, tuple)?;
        }
    }

    let mut parent_unmapped = BTreeMap::<Vec<u32>, Vec<u32>>::new();
    for &original in original_token_ids {
        let parent_local = parent
            .original_token_to_internal
            .get(original as usize)
            .copied()
            .unwrap_or(u32::MAX);
        if parent_local != u32::MAX {
            continue;
        }
        let child_tuple = components[1..]
            .iter()
            .map(|component| {
                component
                    .constraint
                    .original_token_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
            })
            .collect::<Vec<_>>();
        parent_unmapped.entry(child_tuple).or_default().push(original);
    }
    for (child_tuple, originals) in parent_unmapped {
        let mut tuple = Vec::with_capacity(components.len());
        tuple.push(u32::MAX);
        tuple.extend(child_tuple);
        add_token_class(originals, tuple)?;
    }

    Ok((
        ManyToOneIdMap {
            original_to_internal: token_to_global,
            internal_to_originals: global_to_tokens,
            representative_original_ids: token_representatives,
        },
        local_to_global_tokens,
    ))
}

fn build_direct_component_coordinate_maps(
    components: &[ParserDwaComponent<'_>],
    merged_tokenizer_state_count: usize,
    original_token_ids: &[u32],
) -> Result<(InternalIdMap, Vec<DirectComponentCoordinateMaps>), String> {
    let total_started_at = Instant::now();
    let state_started_at = Instant::now();
    let state_coordinates =
        build_direct_component_state_coordinates(components, merged_tokenizer_state_count)?;
    let state_ms = state_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_started_at = Instant::now();
    let (vocab_tokens, local_to_global_tokens) =
        build_direct_component_token_coordinates(components, original_token_ids)?;
    let token_ms = token_started_at.elapsed().as_secs_f64() * 1000.0;
    let component_maps = state_coordinates
        .local_to_global_tsids
        .into_iter()
        .zip(local_to_global_tokens)
        .map(|(local_to_global_tsids, local_to_global_tokens)| {
            DirectComponentCoordinateMaps {
                local_to_global_tsids,
                local_to_global_tokens,
            }
        })
        .collect::<Vec<_>>();
    let id_map = InternalIdMap {
        tokenizer_states: state_coordinates.tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_component_coordinates] components={} states={} tsids={} tokens={} classes={} state_ms={state_ms:.3} token_ms={token_ms:.3} total_ms={:.3}",
            components.len(),
            merged_tokenizer_state_count,
            id_map.num_tsids(),
            original_token_ids.len(),
            id_map.num_internal_tokens(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((id_map, component_maps))
}

fn mapped_labels(label: i32, relation: &[Vec<u32>]) -> Result<Vec<i32>, String> {
    if label == DEFAULT_LABEL {
        return Err("default labels must be materialized before parser-state transport".into());
    }
    let (local_state, negative) = if is_negative_label(label) {
        (negative_to_positive_label(label) as u32, true)
    } else if label >= 0 {
        (label as u32, false)
    } else {
        return Err(format!("unsupported parser-DWA label {label}"));
    };
    let targets = relation
        .get(local_state as usize)
        .ok_or_else(|| format!("parser-state relation omits local state {local_state}"))?;
    if targets.is_empty() {
        return Err(format!("parser-state relation maps local state {local_state} nowhere"));
    }
    Ok(targets
        .iter()
        .copied()
        .map(|state| {
            if negative {
                encode_negative_label(state)
            } else {
                encode_positive_label(state)
            }
        })
        .collect())
}

fn add_transition_for_mapped_label(
    nwa: &mut NWA,
    from: u32,
    local_label: i32,
    target: u32,
    weight: &Weight,
    relation: &[Vec<u32>],
) -> Result<(), String> {
    for label in mapped_labels(local_label, relation)? {
        nwa.add_transition(from, label, target, weight.clone());
    }
    Ok(())
}

fn materialized_top_acceptance(constraint: &Constraint) -> BTreeMap<i32, Weight> {
    let mut result = BTreeMap::new();
    let default_combined = constraint.parser_top_accept.get(&DEFAULT_LABEL);
    let default_parts = constraint.parser_top_accept_parts.get(&DEFAULT_LABEL);
    for parser_state in 0..constraint.table.num_states {
        let label = encode_positive_label(parser_state);
        let mut parts = Vec::<Weight>::new();
        if let Some(weight) = constraint
            .parser_top_accept
            .get(&label)
            .or(default_combined)
        {
            parts.push(weight.clone());
        }
        if let Some(weights) = constraint
            .parser_top_accept_parts
            .get(&label)
            .or(default_parts)
        {
            parts.extend(weights.iter().cloned());
        }
        if let Some(row) = constraint.table.advance.get(parser_state as usize) {
            for terminal in row.iter_ones() {
                if let Some(weight) = constraint
                    .direct_regular_l1_complete_by_terminal
                    .get(&(terminal as u32))
                {
                    parts.push(weight.clone());
                }
            }
        }
        if !parts.is_empty() {
            let weight = Weight::union_all(parts.iter());
            if !weight.is_empty() {
                result.insert(label, weight);
            }
        }
    }
    result
}

fn component_parser_nwa(component: &ParserDwaComponent<'_>) -> Result<NWA, String> {
    let constraint = component.constraint;
    if component.parser_state_relation.len() != constraint.table.num_states as usize {
        return Err(format!(
            "parser-state relation has {} rows for a {}-state component table",
            component.parser_state_relation.len(),
            constraint.table.num_states,
        ));
    }

    let source = &constraint.parser_dwa;
    let mut nwa = NWA::new(0, 0);
    for _ in source.states() {
        nwa.add_state();
    }
    nwa.set_start_states(vec![source.start_state()]);

    for (state_id, state) in source.states().iter().enumerate() {
        if let Some(final_weight) = &state.final_weight {
            nwa.set_final_weight(state_id as u32, final_weight.clone());
        }
        let explicit_positive = state
            .transitions
            .keys()
            .filter_map(|&label| {
                (label >= 0 && label != DEFAULT_LABEL).then_some(label as u32)
            })
            .collect::<BTreeSet<_>>();
        for (&label, (target, weight)) in &state.transitions {
            if label == DEFAULT_LABEL {
                continue;
            }
            add_transition_for_mapped_label(
                &mut nwa,
                state_id as u32,
                label,
                *target,
                weight,
                component.parser_state_relation,
            )?;
        }
        if let Some((target, weight)) = state.transitions.get(&DEFAULT_LABEL) {
            for local_state in 0..constraint.table.num_states {
                if explicit_positive.contains(&local_state) {
                    continue;
                }
                add_transition_for_mapped_label(
                    &mut nwa,
                    state_id as u32,
                    encode_positive_label(local_state),
                    *target,
                    weight,
                    component.parser_state_relation,
                )?;
            }
        }
    }

    let top_accept = materialized_top_acceptance(constraint);
    if !top_accept.is_empty() {
        let start = nwa.add_state();
        let final_state = nwa.add_state();
        nwa.set_final_weight(final_state, Weight::all());
        for (label, weight) in top_accept {
            add_transition_for_mapped_label(
                &mut nwa,
                start,
                label,
                final_state,
                &weight,
                component.parser_state_relation,
            )?;
        }
        nwa.start_states_mut().push(start);
    }
    Ok(nwa)
}

type PossibleMatches = BTreeMap<u32, Weight>;

struct BoundaryRepair {
    parser_dwa: MappedArtifact<DWA>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    active_terminals: Vec<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct MixedTokenNodeKey {
    offset: usize,
    /// `u32::MAX` means no non-ignore terminal has committed yet.
    last_terminal: u32,
    /// `u32::MAX` means no non-ignore component owner has been observed yet.
    last_owner: u32,
    crossed: bool,
    /// Whether this path has touched a table-splice seed terminal.
    seeded: bool,
    /// False only for the arbitrary-residual first fragment. Once any
    /// terminal commits, subsequent fragments start from lexer reset.
    started: bool,
}

#[derive(Debug)]
struct MixedTokenEdge {
    target: usize,
    terminal: u32,
}

#[derive(Debug)]
struct MixedTokenNode {
    key: MixedTokenNodeKey,
    outgoing: Vec<MixedTokenEdge>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ResidualScanResult {
    /// Per-start-state longest matches. Different residual starts may produce
    /// different valid widths for the same terminal. These collections are
    /// tiny in practice; sorted vectors avoid one tree allocation per scan.
    matches: Vec<(u32, usize)>,
    future_terminals: Vec<u32>,
}

struct MixedOwnerWitness {
    token_id: u32,
    start_states: Vec<u32>,
    nodes: Vec<MixedTokenNode>,
    good: Vec<bool>,
    accepting: Vec<bool>,
}

struct MixedOwnerDiscovery {
    terminals: BitSet,
    token_ids: Vec<u32>,
    witnesses: Vec<MixedOwnerWitness>,
}

fn scan_residual_starts(
    tokenizer: &Tokenizer,
    bytes: &[u8],
    starts: &[u32],
) -> ResidualScanResult {
    let mut result = ResidualScanResult::default();
    for &start in starts {
        let execution = tokenizer.execute_from_state(bytes, start);
        result.matches.extend(
            execution
                .matches
                .into_iter()
                .filter(|matched| matched.width > 0)
                .map(|matched| (matched.id, matched.width)),
        );
        for end_state in execution.end_state {
            result.future_terminals.extend(
                tokenizer
                    .possible_future_terminals_iter(end_state),
            );
        }
    }
    result.matches.sort_unstable();
    result.matches.dedup();
    result.future_terminals.sort_unstable();
    result.future_terminals.dedup();
    result
}


fn component_tokenizer_state_layout(components: &[&Constraint]) -> (Vec<u32>, usize) {
    let mut next_state = 1u32; // Fresh merged epsilon dispatcher.
    let mut offsets = Vec::with_capacity(components.len());
    for component in components {
        offsets.push(next_state);
        next_state = next_state
            .checked_add(component.tokenizer.num_states())
            .expect("composed tokenizer state count overflow");
    }
    (offsets, next_state as usize)
}

fn component_tokenizer_state_layout_owned_parent(
    components: &[&Constraint],
) -> (Vec<u32>, usize) {
    let mut next_state = 0u32;
    let mut offsets = Vec::with_capacity(components.len());
    for component in components {
        offsets.push(next_state);
        next_state = next_state
            .checked_add(component.tokenizer.num_states())
            .expect("composed tokenizer state count overflow");
    }
    (offsets, next_state as usize)
}

fn composite_reset_states(
    _components: &[&Constraint],
    _tokenizer_state_offsets: &[u32],
) -> Vec<u32> {
    // The disjoint-union tokenizer resets to its fresh dispatcher. Executing
    // from state zero epsilon-dispatches to every component start state.
    vec![0]
}

fn expanded_component_reset_states(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
) -> Vec<u32> {
    let mut resets = components
        .iter()
        .zip(tokenizer_state_offsets)
        .map(|(component, offset)| offset + component.tokenizer.start_state())
        .collect::<Vec<_>>();
    resets.sort_unstable();
    resets.dedup();
    resets
}

fn component_reset_live_bytes(components: &[&Constraint]) -> Vec<U8Set> {
    components
        .iter()
        .map(|component| {
            let closures = component.tokenizer.all_singleton_epsilon_closures();
            let mut bytes = U8Set::empty();
            for &state in &closures[component.tokenizer.start_state() as usize] {
                for (byte, _) in component.tokenizer.transitions_from(state) {
                    bytes.insert(byte);
                }
            }
            bytes
        })
        .collect()
}

fn scan_component_residual_starts(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    reset_live_bytes: &[U8Set],
    bytes: &[u8],
    starts: &[u32],
) -> ResidualScanResult {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());
    debug_assert_eq!(components.len(), reset_live_bytes.len());
    let mut result = ResidualScanResult::default();

    let mut scan_local = |component_index: usize, local_start: u32| {
        let component = components[component_index];
        let terminal_offset = terminal_offsets[component_index];
        let (end_states, matches) = component
            .tokenizer
            .execute_summary_from_state(bytes, local_start);
        result.matches.extend(
            matches
                .into_iter()
                .filter(|(_, width)| *width > 0)
                .map(|(terminal, width)| (terminal_offset + terminal, width)),
        );
        for end_state in end_states {
            result.future_terminals.extend(
                component
                    .tokenizer
                    .possible_future_terminals_iter(end_state)
                    .map(|terminal| terminal_offset + terminal),
            );
        }
    };

    for &global_start in starts {
        if global_start == 0 {
            for (component_index, component) in components.iter().enumerate() {
                if bytes.first().is_some_and(|byte| {
                    !reset_live_bytes[component_index].contains(*byte)
                }) {
                    continue;
                }
                scan_local(component_index, component.tokenizer.start_state());
            }
            continue;
        }
        let component_index = tokenizer_state_offsets
            .partition_point(|&offset| offset <= global_start)
            .saturating_sub(1);
        let Some(component) = components.get(component_index) else {
            continue;
        };
        let offset = tokenizer_state_offsets[component_index];
        let local_start = global_start - offset;
        if local_start < component.tokenizer.num_states() {
            scan_local(component_index, local_start);
        }
    }
    result.matches.sort_unstable();
    result.matches.dedup();
    result.future_terminals.sort_unstable();
    result.future_terminals.dedup();
    result
}

fn terminal_owners(num_terminals: usize, terminal_offsets: &[u32]) -> Vec<u32> {
    let mut owners = vec![u32::MAX; num_terminals];
    for (owner, &start) in terminal_offsets.iter().enumerate() {
        let end = terminal_offsets
            .get(owner + 1)
            .copied()
            .unwrap_or(num_terminals as u32);
        for terminal in start..end.min(num_terminals as u32) {
            owners[terminal as usize] = owner as u32;
        }
    }
    owners
}

fn transition_mixed_key(
    key: MixedTokenNodeKey,
    terminal: u32,
    next_offset: usize,
    owners: &[u32],
    seed_terminals: &[bool],
    ignore_terminal: Option<u32>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<MixedTokenNodeKey> {
    if Some(terminal) == ignore_terminal {
        return Some(MixedTokenNodeKey {
            offset: next_offset,
            started: true,
            ..key
        });
    }
    if key.last_terminal != u32::MAX
        && disallowed_follows
            .get(&key.last_terminal)
            .is_some_and(|blocked| blocked.contains(terminal as usize))
    {
        return None;
    }
    let owner = owners
        .get(terminal as usize)
        .copied()
        .unwrap_or(u32::MAX);
    if owner == u32::MAX {
        return None;
    }
    Some(MixedTokenNodeKey {
        offset: next_offset,
        last_terminal: terminal,
        last_owner: owner,
        crossed: key.crossed || (key.last_owner != u32::MAX && key.last_owner != owner),
        seeded: key.seeded
            || seed_terminals
                .get(terminal as usize)
                .copied()
                .unwrap_or(false),
        started: true,
    })
}

fn build_boundary_token_graph(
    bytes: &[u8],
    arbitrary_scan: &ResidualScanResult,
    reset_scans: &[&ResidualScanResult],
    owners: &[u32],
    seed_terminals: &[bool],
    ignore_terminal: Option<u32>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<(Vec<MixedTokenNode>, Vec<bool>, Vec<bool>)> {
    let mut nodes = Vec::<MixedTokenNode>::new();
    let mut node_ids = FxHashMap::<MixedTokenNodeKey, usize>::default();
    let mut queue = std::collections::VecDeque::<usize>::new();
    let start_key = MixedTokenNodeKey {
        offset: 0,
        last_terminal: u32::MAX,
        last_owner: u32::MAX,
        crossed: false,
        seeded: false,
        started: false,
    };
    nodes.push(MixedTokenNode {
        key: start_key,
        outgoing: Vec::new(),
    });
    node_ids.insert(start_key, 0);
    queue.push_back(0);
    let mut accepting = vec![false];

    while let Some(node_id) = queue.pop_front() {
        let key = nodes[node_id].key;
        if key.offset == bytes.len() {
            continue;
        }
        let scan = if key.started {
            reset_scans
                .get(key.offset - 1)
                .copied()
                .expect("reset scan must exist for every positive token offset")
        } else {
            arbitrary_scan
        };

        for &(terminal, width) in &scan.matches {
            let next_offset = key.offset.saturating_add(width);
            if next_offset > bytes.len() {
                continue;
            }
            let Some(target_key) = transition_mixed_key(
                key,
                terminal,
                next_offset,
                owners,
                seed_terminals,
                ignore_terminal,
                disallowed_follows,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len()
                    && (target_key.crossed || target_key.seeded);
                nodes.push(MixedTokenNode {
                    key: target_key,
                    outgoing: Vec::new(),
                });
                node_ids.insert(target_key, target);
                queue.push_back(target);
                accepting.push(is_accepting);
                target
            };
            nodes[node_id]
                .outgoing
                .push(MixedTokenEdge { target, terminal });
        }

        // An unfinished final terminal is a real terminal-DWA label and may
        // itself be the owner-changing or splice-seed edge at token end.
        for &terminal in &scan.future_terminals {
            let Some(target_key) = transition_mixed_key(
                key,
                terminal,
                bytes.len(),
                owners,
                seed_terminals,
                ignore_terminal,
                disallowed_follows,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len()
                    && (target_key.crossed || target_key.seeded);
                nodes.push(MixedTokenNode {
                    key: target_key,
                    outgoing: Vec::new(),
                });
                node_ids.insert(target_key, target);
                queue.push_back(target);
                accepting.push(is_accepting);
                target
            };
            nodes[node_id]
                .outgoing
                .push(MixedTokenEdge { target, terminal });
        }
    }

    if !accepting.iter().any(|&is_accepting| is_accepting) {
        return None;
    }
    let mut good = accepting.clone();
    let mut by_descending_offset = (0..nodes.len()).collect::<Vec<_>>();
    by_descending_offset.sort_unstable_by_key(|&node| std::cmp::Reverse(nodes[node].key.offset));
    for source in by_descending_offset {
        if !good[source]
            && nodes[source]
                .outgoing
                .iter()
                .any(|edge| good[edge.target])
        {
            good[source] = true;
        }
    }
    good[0].then_some((nodes, good, accepting))
}

fn boundary_candidate_state_ranges_by_token(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    vocab: &Vocab,
) -> BTreeMap<u32, Vec<(usize, u32, u32)>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    let mut by_token = BTreeMap::<u32, Vec<(usize, u32, u32)>>::new();
    for (component_index, constraint) in components.iter().enumerate() {
        debug_assert!(constraint.possible_matches_complete);
        for weight in constraint.possible_matches.values() {
            for (start_tsid, end_tsid, internal_tokens) in weight.range_entries() {
                for internal_token in internal_tokens.iter() {
                    if constraint.internal_token_to_tokens.is_empty() {
                        if vocab
                            .entries_map()
                            .get(&internal_token)
                            .is_some_and(|bytes| bytes.len() >= 2)
                        {
                            by_token.entry(internal_token).or_default().push((
                                component_index,
                                start_tsid,
                                end_tsid,
                            ));
                        }
                        continue;
                    }
                    let Some(originals) = constraint
                        .internal_token_to_tokens
                        .get(internal_token as usize)
                    else {
                        continue;
                    };
                    for &original in originals {
                        if vocab
                            .entries_map()
                            .get(&original)
                            .is_some_and(|bytes| bytes.len() >= 2)
                        {
                            by_token.entry(original).or_default().push((
                                component_index,
                                start_tsid,
                                end_tsid,
                            ));
                        }
                    }
                }
            }
        }
    }
    for ranges in by_token.values_mut() {
        ranges.sort_unstable();
        ranges.dedup();
    }
    by_token
}

fn candidate_start_state_groups_for_token(
    token_id: u32,
    candidate_ranges: &BTreeMap<u32, Vec<(usize, u32, u32)>>,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
) -> Vec<(u32, Vec<u32>)> {
    // Global state zero is the retained/fresh merged reset dispatcher. It is a
    // semantic state of its own and must not be conflated with an individual
    // component's local start state.
    let mut support_by_representative = FxHashMap::<u32, Vec<u32>>::default();
    support_by_representative.insert(0, vec![0]);
    if let Some(ranges) = candidate_ranges.get(&token_id) {
        for &(component_index, start_tsid, end_tsid) in ranges {
            let constraint = components[component_index];
            let state_offset = tokenizer_state_offsets[component_index];
            for tsid in start_tsid..=end_tsid {
                let Some(states) = constraint.internal_tsid_to_states.get(tsid as usize) else {
                    continue;
                };
                let mut representative = None;
                for &state in states {
                    let Some(global) = state_offset.checked_add(state) else {
                        continue;
                    };
                    if global == 0 {
                        continue;
                    }
                    representative.get_or_insert(global);
                    support_by_representative
                        .entry(*representative.as_ref().unwrap())
                        .or_default()
                        .push(global);
                }
            }
        }
    }
    let mut groups = support_by_representative.into_iter().collect::<Vec<_>>();
    for (_, support) in &mut groups {
        support.sort_unstable();
        support.dedup();
    }
    groups.sort_unstable_by_key(|(representative, _)| *representative);
    groups
}

#[derive(Clone, Copy)]
struct ExprByteSummary {
    nullable: bool,
    first: U8Set,
    last: U8Set,
    reachable: U8Set,
}

fn expr_byte_summary(expr: &Expr) -> ExprByteSummary {
    match expr {
        Expr::U8Seq(bytes) => ExprByteSummary {
            nullable: bytes.is_empty(),
            first: bytes.first().copied().map_or(U8Set::empty(), U8Set::single),
            last: bytes.last().copied().map_or(U8Set::empty(), U8Set::single),
            reachable: U8Set::from_bytes(bytes),
        },
        Expr::U8Class(bytes) => ExprByteSummary {
            nullable: false,
            first: *bytes,
            last: *bytes,
            reachable: *bytes,
        },
        // Opaque precompiled DFAs are uncommon in imported JSON schemas.  Use
        // a full-byte overapproximation so this filter can only retain extra
        // tokens, never discard a real boundary witness.
        Expr::Dfa(_) => ExprByteSummary {
            nullable: expr.is_nullable(),
            first: U8Set::all(),
            last: U8Set::all(),
            reachable: U8Set::all(),
        },
        Expr::Intersect { expr, intersect } => {
            let left = expr_byte_summary(expr);
            let right = expr_byte_summary(intersect);
            ExprByteSummary {
                nullable: left.nullable && right.nullable,
                first: left.first.intersection(&right.first),
                last: left.last.intersection(&right.last),
                reachable: left.reachable.intersection(&right.reachable),
            }
        }
        Expr::Seq(parts) => {
            let summaries = parts.iter().map(expr_byte_summary).collect::<Vec<_>>();
            let mut first = U8Set::empty();
            for summary in &summaries {
                first |= summary.first;
                if !summary.nullable {
                    break;
                }
            }
            let mut last = U8Set::empty();
            for summary in summaries.iter().rev() {
                last |= summary.last;
                if !summary.nullable {
                    break;
                }
            }
            ExprByteSummary {
                nullable: summaries.iter().all(|summary| summary.nullable),
                first,
                last,
                reachable: summaries
                    .iter()
                    .fold(U8Set::empty(), |bytes, summary| bytes | summary.reachable),
            }
        }
        Expr::Choice(options) => options.iter().map(expr_byte_summary).fold(
            ExprByteSummary {
                nullable: false,
                first: U8Set::empty(),
                last: U8Set::empty(),
                reachable: U8Set::empty(),
            },
            |mut combined, summary| {
                combined.nullable |= summary.nullable;
                combined.first |= summary.first;
                combined.last |= summary.last;
                combined.reachable |= summary.reachable;
                combined
            },
        ),
        Expr::Exclude { expr, .. } => expr_byte_summary(expr),
        Expr::Repeat { expr, min, max } => {
            if *max == Some(0) {
                return ExprByteSummary {
                    nullable: true,
                    first: U8Set::empty(),
                    last: U8Set::empty(),
                    reachable: U8Set::empty(),
                };
            }
            let child = expr_byte_summary(expr);
            ExprByteSummary {
                nullable: *min == 0 || child.nullable,
                ..child
            }
        }
        Expr::Shared(expr) => expr_byte_summary(expr),
        Expr::Epsilon => ExprByteSummary {
            nullable: true,
            first: U8Set::empty(),
            last: U8Set::empty(),
            reachable: U8Set::empty(),
        },
    }
}

fn boundary_token_prefilter(
    vocab: &Vocab,
    components: &[&Constraint],
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    ignore_terminal: Option<u32>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> BTreeSet<u32> {
    let num_terminals = terminal_offsets
        .iter()
        .copied()
        .zip(components)
        .map(|(offset, component)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let owners = terminal_owners(num_terminals, terminal_offsets);
    let mut summaries = vec![None::<ExprByteSummary>; num_terminals];
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            summaries[terminal_offset + local_terminal] = Some(
                component
                    .tokenizer
                    .terminal_expr(local_terminal as u32)
                    .map(expr_byte_summary)
                    .unwrap_or(ExprByteSummary {
                        nullable: false,
                        first: U8Set::all(),
                        last: U8Set::all(),
                        reachable: U8Set::all(),
                    }),
            );
        }
    }

    let mut allowed_pairs = [U8Set::empty(); 256];
    let mut boundary_terminals = vec![false; num_terminals];
    for terminal in 0..num_terminals {
        if Some(terminal as u32) == ignore_terminal {
            continue;
        }
        let Some(left) = summaries[terminal] else {
            continue;
        };
        let blocked = disallowed_follows.get(&(terminal as u32));
        for follow in 0..num_terminals {
            if Some(follow as u32) == ignore_terminal
                || blocked.is_some_and(|blocked| blocked.contains(follow))
            {
                continue;
            }
            let boundary_pair = seed_terminals.get(terminal).copied().unwrap_or(false)
                || seed_terminals.get(follow).copied().unwrap_or(false)
                || owners.get(terminal) != owners.get(follow);
            if !boundary_pair {
                continue;
            }
            let Some(right) = summaries[follow] else {
                continue;
            };
            boundary_terminals[terminal] = true;
            boundary_terminals[follow] = true;
            for byte in left.last.iter() {
                allowed_pairs[byte as usize] |= right.first;
            }
        }
    }


    let mut candidates =
        crate::compiler::stages::id_map_and_terminal_dwa::classify::
            vocab_tokens_with_adjacent_pairs(vocab, &allowed_pairs)
            .into_iter()
            .collect::<BTreeSet<_>>();

    // A component switch can occur exactly at the preceding token boundary.
    // Compile every seed endpoint once, index them by possible first byte, and
    // classify the vocabulary in one pass. The previous terminal-major loop
    // traversed the complete 128k-token vocabulary once per seed terminal.
    let mut endpoint_dfas = Vec::new();
    let mut endpoint_dfas_by_first = (0..256)
        .map(|_| Vec::<usize>::new())
        .collect::<Vec<_>>();
    let mut conservative_first = U8Set::empty();
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            let global_terminal = terminal_offset + local_terminal;
            if !seed_terminals
                .get(global_terminal)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let first = summaries[global_terminal]
                .map(|summary| summary.first)
                .unwrap_or(U8Set::all());
            let Some(expr) = component.tokenizer.terminal_expr(local_terminal as u32) else {
                // Missing retained source is rare and cannot justify an unsafe
                // exclusion. Fall back to the conservative first-byte summary.
                conservative_first |= first;
                continue;
            };
            let dfa_index = endpoint_dfas.len();
            endpoint_dfas.push(compile_terminal_expr_dfa(expr));
            for byte in first.iter() {
                endpoint_dfas_by_first[byte as usize].push(dfa_index);
            }
        }
    }
    for (&token, bytes) in vocab.entries_map().iter() {
        if bytes.len() < 2 {
            continue;
        }
        let first = bytes[0];
        if conservative_first.contains(first) {
            candidates.insert(token);
            continue;
        }
        for &dfa_index in &endpoint_dfas_by_first[first as usize] {
            let dfa = &endpoint_dfas[dfa_index];
            let mut state = 0u32;
            let mut live = true;
            for &byte in bytes {
                let Some(next) = dfa.step(state, byte) else {
                    live = false;
                    break;
                };
                state = next;
            }
            if live
                && (!dfa.finalizers(state).is_empty()
                    || !dfa.possible_future_group_ids(state).is_empty())
            {
                candidates.insert(token);
                break;
            }
        }
    }

    // A boundary endpoint terminal can occupy an entire token immediately
    // before or after a component switch, without the token itself containing
    // the switch. Component possible-matches is exact for those cases; union
    // every grammar-valid boundary endpoint into the byte-pair candidates.
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            if !boundary_terminals
                .get(terminal_offset + local_terminal)
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
            let Some(weight) = component.possible_matches.get(&(local_terminal as u32)) else {
                continue;
            };
            for (_, _, internal_tokens) in weight.range_entries() {
                for internal_token in internal_tokens.iter() {
                    if component.internal_token_to_tokens.is_empty() {
                        if vocab
                            .entries_map()
                            .get(&internal_token)
                            .is_some_and(|bytes| bytes.len() >= 2)
                        {
                            candidates.insert(internal_token);
                        }
                    } else if let Some(originals) =
                        component.internal_token_to_tokens.get(internal_token as usize)
                    {
                        candidates.extend(originals.iter().copied().filter(|token| {
                            vocab
                                .entries_map()
                                .get(token)
                                .is_some_and(|bytes| bytes.len() >= 2)
                        }));
                    }
                }
            }
        }
    }
    candidates
}

fn discover_mixed_owner_terminals(
    vocab: &Vocab,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    ignore_terminal: Option<u32>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> MixedOwnerDiscovery {
    let num_terminals = components
        .iter()
        .zip(terminal_offsets.iter().copied())
        .map(|(component, offset)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let owners = terminal_owners(num_terminals, terminal_offsets);
    let reset_starts = composite_reset_states(components, tokenizer_state_offsets);
    let reset_live_bytes = component_reset_live_bytes(components);
    let candidate_ranges_started_at = Instant::now();
    let candidate_ranges =
        boundary_candidate_state_ranges_by_token(components, tokenizer_state_offsets, vocab);
    let candidate_ranges_ms = candidate_ranges_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_range_rows = candidate_ranges.values().map(Vec::len).sum::<usize>();
    let prefilter_started_at = Instant::now();
    let mut prefilter = boundary_token_prefilter(
        vocab,
        components,
        terminal_offsets,
        seed_terminals,
        ignore_terminal,
        disallowed_follows,
    );
    prefilter.extend(candidate_ranges.keys().copied());
    let prefilter_ms = prefilter_started_at.elapsed().as_secs_f64() * 1000.0;
    let use_prefilter = std::env::var_os("GLRMASK_COMPOSE_DISABLE_BOUNDARY_PREFILTER").is_none();
    let multi_byte_entries = vocab
        .entries_map()
        .iter()
        .filter(|&(&token_id, ref bytes)| {
            bytes.len() >= 2 && (!use_prefilter || prefilter.contains(&token_id))
        })
        .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
        .collect::<Vec<_>>();
    let suffix_cache_started_at = Instant::now();
    let reset_suffix_cache = if std::env::var_os("GLRMASK_COMPOSE_DISABLE_SUFFIX_CACHE")
        .is_some()
    {
        None
    } else {
        let mut suffixes = FxHashSet::<&[u8]>::default();
        for &(_, bytes) in &multi_byte_entries {
            for offset in 1..bytes.len() {
                suffixes.insert(&bytes[offset..]);
            }
        }
        Some(
            suffixes
                .into_par_iter()
                .map(|suffix| {
                    let scan = scan_component_residual_starts(
                        components,
                        tokenizer_state_offsets,
                        terminal_offsets,
                        &reset_live_bytes,
                        suffix,
                        &reset_starts,
                    );
                    (suffix, scan)
                })
                .collect::<FxHashMap<_, _>>(),
        )
    };
    let suffix_cache_ms = suffix_cache_started_at.elapsed().as_secs_f64() * 1000.0;

    // Each model token is an independent acyclic same-token graph. Run those
    // scans in parallel, then merge in vocabulary order for deterministic
    // output and profiling.
    let candidate_start_visits = AtomicUsize::new(0);
    let distinct_scan_groups = AtomicUsize::new(0);
    let max_candidate_starts = AtomicUsize::new(0);
    let results = multi_byte_entries
        .par_iter()
        .filter_map(|&(token_id, bytes)| {
            let owned_reset_scans;
            let reset_scans = if let Some(cache) = reset_suffix_cache.as_ref() {
                (1..bytes.len())
                    .map(|offset| {
                        cache
                            .get(&bytes[offset..])
                            .expect("reset suffix cache must cover every vocabulary suffix")
                    })
                    .collect::<Vec<_>>()
            } else {
                owned_reset_scans = (1..bytes.len())
                    .map(|offset| {
                        scan_component_residual_starts(
                            components,
                            tokenizer_state_offsets,
                            terminal_offsets,
                            &reset_live_bytes,
                            &bytes[offset..],
                            &reset_starts,
                        )
                    })
                    .collect::<Vec<_>>();
                owned_reset_scans.iter().collect::<Vec<_>>()
            };
            let candidate_groups = candidate_start_state_groups_for_token(
                token_id,
                &candidate_ranges,
                components,
                tokenizer_state_offsets,
            );
            candidate_start_visits.fetch_add(candidate_groups.len(), Ordering::Relaxed);
            max_candidate_starts.fetch_max(candidate_groups.len(), Ordering::Relaxed);
            let mut starts_by_scan = FxHashMap::<ResidualScanResult, Vec<u32>>::default();
            for (representative, support_states) in candidate_groups {
                let representative_scan = scan_component_residual_starts(
                    components,
                    tokenizer_state_offsets,
                    terminal_offsets,
                    &reset_live_bytes,
                    bytes,
                    &[representative],
                );
                if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_TSID_REPRESENTATIVE_SCAN")
                    .is_some()
                {
                    for &state in &support_states {
                        let reference = scan_component_residual_starts(
                            components,
                            tokenizer_state_offsets,
                            terminal_offsets,
                            &reset_live_bytes,
                            bytes,
                            &[state],
                        );
                        assert_eq!(
                            representative_scan, reference,
                            "component TSID representative scan differs for token {token_id}, representative {representative}, state {state}",
                        );
                    }
                }
                starts_by_scan
                    .entry(representative_scan)
                    .or_default()
                    .extend(support_states);
            }
            for states in starts_by_scan.values_mut() {
                states.sort_unstable();
                states.dedup();
            }
            distinct_scan_groups.fetch_add(starts_by_scan.len(), Ordering::Relaxed);
            let mut scan_groups = starts_by_scan.into_iter().collect::<Vec<_>>();
            scan_groups.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            let mut local_terminals = FxHashSet::<u32>::default();
            let mut local_witnesses = Vec::new();
            for (arbitrary_scan, start_states) in scan_groups {
                let Some((nodes, good, accepting)) = build_boundary_token_graph(
                    bytes,
                    &arbitrary_scan,
                    &reset_scans,
                    &owners,
                    seed_terminals,
                    ignore_terminal,
                    disallowed_follows,
                ) else {
                    continue;
                };
                for (source, node) in nodes.iter().enumerate() {
                    if !good[source] {
                        continue;
                    }
                    for edge in &node.outgoing {
                        if good[edge.target] {
                            local_terminals.insert(edge.terminal);
                        }
                    }
                }
                local_witnesses.push(MixedOwnerWitness {
                    token_id,
                    start_states,
                    nodes,
                    good,
                    accepting,
                });
            }
            (!local_witnesses.is_empty())
                .then_some((token_id, local_terminals, local_witnesses))
        })
        .collect::<Vec<_>>();

    let mut discovered = BitSet::new(num_terminals);
    let mut mixed_token_ids = Vec::with_capacity(results.len());
    let mut witnesses = Vec::new();
    for (token_id, terminals, mut token_witnesses) in results {
        mixed_token_ids.push(token_id);
        for terminal in terminals {
            discovered.set(terminal as usize);
        }
        witnesses.append(&mut token_witnesses);
    }
    if std::env::var_os("GLRMASK_DUMP_COMPOSE_BOUNDARY_TOKENS").is_some() {
        eprintln!(
            "[glrmask/dump][constraint_boundary_tokens] prefilter={:?} exact={:?}",
            prefilter,
            mixed_token_ids,
        );
    }
    if compose_profile_enabled() {
        let exact = mixed_token_ids.iter().copied().collect::<BTreeSet<_>>();
        let missing = exact.difference(&prefilter).copied().collect::<Vec<_>>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_candidate_fanout] range_tokens={} range_rows={} ranges_ms={candidate_ranges_ms:.3} scanned_tokens={} raw_start_visits={} distinct_scan_groups={} max_starts={}",
            candidate_ranges.len(),
            candidate_range_rows,
            multi_byte_entries.len(),
            candidate_start_visits.load(Ordering::Relaxed),
            distinct_scan_groups.load(Ordering::Relaxed),
            max_candidate_starts.load(Ordering::Relaxed),
        );
        eprintln!(
            "[glrmask/profile][constraint_boundary_prefilter] enabled={} candidates={} scanned={} exact={} missing={} prefilter_ms={prefilter_ms:.3} missing_ids={:?}",
            use_prefilter,
            prefilter.len(),
            multi_byte_entries.len(),
            exact.len(),
            missing.len(),
            missing.iter().take(32).collect::<Vec<_>>(),
        );
        let suffix_occurrences = multi_byte_entries
            .iter()
            .map(|(_, bytes)| bytes.len() - 1)
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_suffix_cache] tokens={} suffix_occurrences={} unique_suffixes={} cache_ms={suffix_cache_ms:.3} enabled={}",
            multi_byte_entries.len(),
            suffix_occurrences,
            reset_suffix_cache.as_ref().map_or(0, FxHashMap::len),
            reset_suffix_cache.is_some(),
        );
    }
    MixedOwnerDiscovery {
        terminals: discovered,
        token_ids: mixed_token_ids,
        witnesses,
    }
}

fn collect_one_byte_seed_relations_serial(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    seed_terminals: &[bool],
    candidate_states: &[u32],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    let mut relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    let mut tokens_by_byte = vec![Vec::<u32>::new(); 256];
    for (&token_id, bytes) in vocab.entries_map().iter().filter(|(_, bytes)| bytes.len() == 1) {
        tokens_by_byte[bytes[0] as usize].push(token_id);
    }
    let closures = tokenizer.all_singleton_epsilon_closures();
    for &raw_state in candidate_states {
        let source_closure = &closures[raw_state as usize];
        let seed_reachable = source_closure.iter().copied().any(|state| {
            tokenizer
                .matched_terminals_iter(state)
                .chain(tokenizer.possible_future_terminals_iter(state))
                .any(|terminal| {
                    seed_terminals
                        .get(terminal as usize)
                        .copied()
                        .unwrap_or(false)
                })
        });
        if !seed_reachable {
            continue;
        }

        let mut targets_by_byte = BTreeMap::<u8, BTreeSet<u32>>::new();
        for &state in source_closure.iter() {
            for (byte, target) in tokenizer.transitions_from(state) {
                if !tokens_by_byte[byte as usize].is_empty() {
                    targets_by_byte.entry(byte).or_default().insert(target);
                }
            }
        }
        for (byte, targets) in targets_by_byte {
            let mut end_states = BTreeSet::<u32>::new();
            for target in targets {
                end_states.extend(closures[target as usize].iter().copied());
            }
            let mut terminals = BTreeSet::<u32>::new();
            for end_state in end_states {
                terminals.extend(
                    tokenizer
                        .matched_terminals_iter(end_state)
                        .chain(tokenizer.possible_future_terminals_iter(end_state))
                        .filter(|terminal| {
                            seed_terminals
                                .get(*terminal as usize)
                                .copied()
                                .unwrap_or(false)
                        }),
                );
            }
            for terminal in terminals {
                relations
                    .entry(vec![terminal])
                    .or_default()
                    .entry(raw_state)
                    .or_default()
                    .extend(tokens_by_byte[byte as usize].iter().copied());
            }
        }
    }
    relations
}

fn collect_one_byte_seed_relations_parallel(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    seed_terminals: &[bool],
    candidate_states: &[u32],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    let mut tokens_by_byte = vec![Vec::<u32>::new(); 256];
    for (&token_id, bytes) in vocab.entries_map().iter().filter(|(_, bytes)| bytes.len() == 1) {
        tokens_by_byte[bytes[0] as usize].push(token_id);
    }
    let seed_terminal_ids = seed_terminals
        .iter()
        .enumerate()
        .filter_map(|(terminal, &selected)| selected.then_some(terminal as u32))
        .collect::<Vec<_>>();
    let closures = tokenizer.all_singleton_epsilon_closures();
    // A relation entry only records existence: for one source state and byte,
    // a seed terminal is supported when any target epsilon-closure state either
    // finishes it or can continue it. Process each transition independently;
    // duplicate `(terminal, state, byte)` entries are harmless and collapse in
    // the final ordered relation. This avoids per-state maps and sets across the
    // million-state common case where every epsilon closure is a singleton.
    let entries = candidate_states
        .par_iter()
        .copied()
        .fold(Vec::<(u32, u32, u8)>::new, |mut output, raw_state| {
            let source_closure = &closures[raw_state as usize];
            let seed_reachable = source_closure.iter().copied().any(|state| {
                seed_terminal_ids.iter().copied().any(|terminal| {
                    tokenizer
                        .matched_terminal_bitset(state)
                        .contains(terminal as usize)
                        || tokenizer
                            .possible_future_terminals(state)
                            .contains(terminal as usize)
                })
            });
            if !seed_reachable {
                return output;
            }

            for &state in source_closure.iter() {
                for (byte, target) in tokenizer.transitions_from(state) {
                    if tokens_by_byte[byte as usize].is_empty() {
                        continue;
                    }
                    let target_closure = &closures[target as usize];
                    for &terminal in &seed_terminal_ids {
                        let supported = target_closure.iter().copied().any(|end_state| {
                            tokenizer
                                .matched_terminal_bitset(end_state)
                                .contains(terminal as usize)
                                || tokenizer
                                    .possible_future_terminals(end_state)
                                    .contains(terminal as usize)
                        });
                        if supported {
                            output.push((terminal, raw_state, byte));
                        }
                    }
                }
            }
            output
        })
        .reduce(Vec::new, |mut left, mut right| {
            left.append(&mut right);
            left
        });

    let mut relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    for (terminal, raw_state, byte) in entries {
        relations
            .entry(vec![terminal])
            .or_default()
            .entry(raw_state)
            .or_default()
            .extend(tokens_by_byte[byte as usize].iter().copied());
    }
    relations
}

fn collect_one_byte_seed_relations(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    seed_terminals: &[bool],
    candidate_states: &[u32],
    relations: &mut BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
) {
    let candidate = if rayon::current_num_threads() == 1
        || std::env::var_os("GLRMASK_COMPOSE_SERIAL_ONE_BYTE_REFERENCE").is_some()
    {
        collect_one_byte_seed_relations_serial(
            tokenizer,
            vocab,
            seed_terminals,
            candidate_states,
        )
    } else {
        collect_one_byte_seed_relations_parallel(
            tokenizer,
            vocab,
            seed_terminals,
            candidate_states,
        )
    };
    if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_ONE_BYTE_PARALLEL").is_some()
        && rayon::current_num_threads() > 1
    {
        let reference = collect_one_byte_seed_relations_serial(
            tokenizer,
            vocab,
            seed_terminals,
            candidate_states,
        );
        assert_eq!(candidate, reference, "parallel one-byte boundary relation differs from serial reference");
        eprintln!(
            "[glrmask/validate][compose_one_byte_parallel] relation_rows={} exact=true",
            candidate.len(),
        );
    }
    for (terminal_path, by_state) in candidate {
        let destination = relations.entry(terminal_path).or_default();
        for (state, tokens) in by_state {
            destination.entry(state).or_default().extend(tokens);
        }
    }
}

fn collect_one_byte_seed_relations_components(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    vocab: &Vocab,
    seed_terminals: &[bool],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());
    let mut relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();

    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        let state_offset = tokenizer_state_offsets[component_index];
        let mut local_seed_terminals =
            vec![false; component.tokenizer.num_terminals() as usize];
        for (local_terminal, selected) in local_seed_terminals.iter_mut().enumerate() {
            *selected = seed_terminals
                .get(terminal_offset as usize + local_terminal)
                .copied()
                .unwrap_or(false);
        }
        if !local_seed_terminals.iter().any(|&selected| selected) {
            continue;
        }

        let mut candidate_states = Vec::<u32>::new();
        if component.terminal_live_states.len() == local_seed_terminals.len() {
            for (terminal, &selected) in local_seed_terminals.iter().enumerate() {
                if selected {
                    candidate_states.extend_from_slice(&component.terminal_live_states[terminal]);
                }
            }
            candidate_states.sort_unstable();
            candidate_states.dedup();
        } else {
            candidate_states.extend(0..component.tokenizer.num_states());
        }
        let mut local_relations =
            BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
        collect_one_byte_seed_relations(
            &component.tokenizer,
            vocab,
            &local_seed_terminals,
            &candidate_states,
            &mut local_relations,
        );
        if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_ONE_BYTE_STATE_INDEX").is_some() {
            let all_states = (0..component.tokenizer.num_states()).collect::<Vec<_>>();
            let reference = collect_one_byte_seed_relations_serial(
                &component.tokenizer,
                vocab,
                &local_seed_terminals,
                &all_states,
            );
            assert_eq!(
                local_relations, reference,
                "terminal-live-state index omitted an exact one-byte boundary relation",
            );
            eprintln!(
                "[glrmask/validate][compose_one_byte_state_index] component={} candidates={} states={} exact=true",
                component_index,
                candidate_states.len(),
                component.tokenizer.num_states(),
            );
        }
        let local_start = component.tokenizer.start_state();
        for (local_path, by_local_state) in local_relations {
            let global_path = local_path
                .into_iter()
                .map(|terminal| terminal_offset + terminal)
                .collect::<Vec<_>>();
            let destination = relations.entry(global_path).or_default();
            for (local_state, tokens) in by_local_state {
                destination
                    .entry(state_offset + local_state)
                    .or_default()
                    .extend(tokens.iter().copied());
                // The fresh merged state zero epsilon-dispatches to every
                // component start state. Its exact one-byte relation is the
                // union of those local start-state relations.
                if local_state == local_start {
                    destination.entry(0).or_default().extend(tokens);
                }
            }
        }
    }
    relations
}

fn component_state_coordinate_map(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    merged_tokenizer_state_count: usize,
) -> Result<ManyToOneIdMap, String> {
    let mut state_to_global = vec![u32::MAX; merged_tokenizer_state_count];
    let mut global_to_states = vec![vec![0u32]];
    let mut representatives = vec![0u32];
    if let Some(reset) = state_to_global.first_mut() {
        *reset = 0;
    }
    for (component_index, component) in components.iter().enumerate() {
        let state_offset = tokenizer_state_offsets[component_index];
        for local_states in &component.internal_tsid_to_states {
            let merged_states = local_states
                .iter()
                .filter_map(|&local_state| {
                    let merged_state = state_offset.checked_add(local_state)?;
                    (merged_state != 0).then_some(merged_state)
                })
                .collect::<Vec<_>>();
            if merged_states.is_empty() {
                continue;
            }
            let global_tsid = global_to_states.len() as u32;
            for &merged_state in &merged_states {
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {merged_state} lies outside merged tokenizer",
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to multiple component TSIDs",
                    ));
                }
                *slot = global_tsid;
            }
            representatives.push(merged_states[0]);
            global_to_states.push(merged_states);
        }
    }
    if state_to_global.iter().any(|&tsid| tsid == u32::MAX) {
        return Err("component TSID map does not cover merged tokenizer".into());
    }
    Ok(ManyToOneIdMap {
        original_to_internal: state_to_global,
        internal_to_originals: global_to_states,
        representative_original_ids: representatives,
    })
}

fn boundary_id_map_for_selected_tokens(
    component_state_map: &ManyToOneIdMap,
    selected_original_tokens: &[u32],
    vocab: &Vocab,
) -> Result<InternalIdMap, String> {
    if selected_original_tokens.is_empty() {
        return Err("boundary witness construction selected no vocabulary tokens".into());
    }
    let max_original_token = vocab.entries_map().keys().next_back().copied().unwrap_or(0);
    let mut original_to_internal = vec![u32::MAX; max_original_token as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(selected_original_tokens.len());
    let mut token_representatives = Vec::with_capacity(selected_original_tokens.len());
    for (internal, &original) in selected_original_tokens.iter().enumerate() {
        let Some(slot) = original_to_internal.get_mut(original as usize) else {
            return Err(format!("boundary token {original} lies outside supplied vocabulary"));
        };
        *slot = internal as u32;
        internal_to_originals.push(vec![original]);
        token_representatives.push(original);
    }
    Ok(InternalIdMap {
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: Vec::new(),
            internal_to_originals: component_state_map
                .representative_original_ids
                .iter()
                .map(|&state| vec![state])
                .collect(),
            representative_original_ids: component_state_map
                .representative_original_ids
                .clone(),
        },
        vocab_tokens: ManyToOneIdMap {
            original_to_internal,
            internal_to_originals,
            representative_original_ids: token_representatives,
        },
        deferred_vocab_singleton_original_ids: None,
    })
}

fn direct_boundary_terminal_automaton(
    num_states: usize,
    component_state_map: Option<&ManyToOneIdMap>,
    vocab: &Vocab,
    seed_relations: BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
    one_byte_ms: f64,
    discovery: &MixedOwnerDiscovery,
    ignore_terminal: Option<u32>,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    let total_started_at = Instant::now();

    let selected_original_tokens = seed_relations
        .values()
        .flat_map(|by_state| by_state.values())
        .flat_map(|tokens| tokens.iter().copied())
        .chain(discovery.token_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if selected_original_tokens.is_empty() {
        return Err("boundary witness construction selected no vocabulary tokens".into());
    }
    let max_original_token = vocab.entries_map().keys().next_back().copied().unwrap_or(0);
    let mut original_to_internal = vec![u32::MAX; max_original_token as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(selected_original_tokens.len());
    let mut token_representatives = Vec::with_capacity(selected_original_tokens.len());
    for (internal, original) in selected_original_tokens.iter().copied().enumerate() {
        original_to_internal[original as usize] = internal as u32;
        internal_to_originals.push(vec![original]);
        token_representatives.push(original);
    }
    let vocab_tokens = ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids: token_representatives,
    };

    // Prefer the already-final component TSID coordinate. This avoids
    // constructing a second million-state quotient only to reconcile it back
    // immediately after boundary parser compilation. The finer coordinate is
    // exact: unsupported component TSIDs simply do not occur in boundary
    // weights.
    let quotient_started_at = Instant::now();
    let tokenizer_states = if let Some(component_state_map) = component_state_map {
        // Boundary weights use the compact component-TSID numbering, but the
        // independently prepared final component map may number equivalent
        // classes differently. Retain one raw representative per TSID so the
        // final reconciliation can translate exactly without cloning the full
        // million-state inverse map.
        ManyToOneIdMap {
            original_to_internal: Vec::new(),
            internal_to_originals: component_state_map
                .representative_original_ids
                .iter()
                .map(|&state| vec![state])
                .collect(),
            representative_original_ids: component_state_map
                .representative_original_ids
                .clone(),
        }
    } else {
        let mut state_signatures = vec![Vec::<(u32, Vec<u32>)>::new(); num_states];
        let mut weight_row = 0u32;
        for by_state in seed_relations.values() {
            for (&state, originals) in by_state {
                state_signatures[state as usize].push((
                    weight_row,
                    originals.iter().copied().collect::<Vec<_>>(),
                ));
            }
            weight_row += 1;
        }
        for witness in &discovery.witnesses {
            for &state in &witness.start_states {
                state_signatures[state as usize].push((weight_row, vec![witness.token_id]));
            }
            weight_row += 1;
        }
        let mut class_by_signature = BTreeMap::<Vec<(u32, Vec<u32>)>, u32>::new();
        let mut state_to_class = vec![u32::MAX; num_states];
        let mut state_representatives = Vec::<u32>::new();
        for (state, signature) in state_signatures.into_iter().enumerate() {
            let class = if let Some(&class) = class_by_signature.get(&signature) {
                class
            } else {
                let class = class_by_signature.len() as u32;
                class_by_signature.insert(signature, class);
                state_representatives.push(state as u32);
                class
            };
            state_to_class[state] = class;
        }
        ManyToOneIdMap::from_original_to_internal_with_representatives(
            state_to_class,
            state_representatives.len() as u32,
            state_representatives,
        )
    };
    let quotient_ms = quotient_started_at.elapsed().as_secs_f64() * 1000.0;
    let id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };

    let state_to_tsid = |state: u32| {
        component_state_map
            .map(|map| map.original_to_internal[state as usize])
            .unwrap_or_else(|| id_map.tokenizer_states.original_to_internal[state as usize])
    };
    let relation_weight = |by_state: BTreeMap<u32, BTreeSet<u32>>| {
        let mut tokens_by_tsid = BTreeMap::<u32, BTreeSet<u32>>::new();
        for (state, originals) in by_state {
            let tsid = state_to_tsid(state);
            let tokens = tokens_by_tsid.entry(tsid).or_default();
            tokens.extend(
                originals
                    .into_iter()
                    .filter_map(|original| id_map.internal_token_for_original(original)),
            );
        }
        Weight::from_per_tsid_token_sets(tokens_by_tsid.into_iter().map(|(tsid, tokens)| {
            (
                tsid,
                tokens.into_iter().collect::<RangeSetBlaze<_>>(),
            )
        }))
    };

    let build_started_at = Instant::now();

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CanonicalNodeKey {
        accepting: bool,
        transitions: Vec<(u32, usize)>,
        epsilons: Vec<usize>,
    }
    #[derive(Debug)]
    struct CanonicalNode {
        accepting: bool,
        transitions: Vec<(u32, usize)>,
        epsilons: Vec<usize>,
    }

    // Put each witness's `(start TSID, token)` support on the epsilon edge
    // entering its graph. The graph itself then denotes only an unweighted
    // terminal suffix language, so structurally equal suffixes can be shared
    // across all residual classes and model tokens without leaking support.
    let mut canonical_by_key = BTreeMap::<CanonicalNodeKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    let mut start_weights_by_canonical = BTreeMap::<usize, Weight>::new();
    for witness in &discovery.witnesses {
        let Some(internal_token) = id_map.internal_token_for_original(witness.token_id) else {
            continue;
        };
        let token_set = RangeSetBlaze::from_iter([internal_token]);
        let witness_tsids = witness
            .start_states
            .iter()
            .map(|&state| state_to_tsid(state))
            .collect::<BTreeSet<_>>();
        let witness_weight = Weight::from_per_tsid_token_sets(
            witness_tsids
                .into_iter()
                .map(|tsid| (tsid, token_set.clone())),
        );
        if witness_weight.is_empty() {
            continue;
        }

        let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
        let mut good_nodes = witness
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
            .collect::<Vec<_>>();
        // Every graph edge consumes positive width, so byte offset is a
        // topological rank. Node allocation order is deliberately irrelevant.
        good_nodes.sort_unstable_by(|left, right| right.1.cmp(&left.1));
        for (local, _) in good_nodes {
            let mut transitions = Vec::new();
            let mut epsilons = Vec::new();
            for edge in witness.nodes[local]
                .outgoing
                .iter()
                .filter(|edge| witness.good[edge.target])
            {
                let target = local_to_canonical[edge.target];
                debug_assert_ne!(target, usize::MAX);
                if Some(edge.terminal) == ignore_terminal {
                    // Ignore bytes advance the token-local lexer path but have
                    // no parser-stack effect. The generic terminal-DWA pipeline
                    // erases ignore labels in exactly this way; exposing the
                    // ignore terminal here would route the path through its
                    // deliberately non-accepting parser template and lose fused
                    // tokens such as `X a!`.
                    epsilons.push(target);
                } else {
                    transitions.push((edge.terminal, target));
                }
            }
            transitions.sort_unstable();
            transitions.dedup();
            epsilons.sort_unstable();
            epsilons.dedup();
            let key = CanonicalNodeKey {
                accepting: witness.accepting[local],
                transitions,
                epsilons,
            };
            let canonical = if let Some(&existing) = canonical_by_key.get(&key) {
                existing
            } else {
                let canonical = canonical_nodes.len();
                canonical_nodes.push(CanonicalNode {
                    accepting: key.accepting,
                    transitions: key.transitions.clone(),
                    epsilons: key.epsilons.clone(),
                });
                canonical_by_key.insert(key, canonical);
                canonical
            };
            local_to_canonical[local] = canonical;
        }
        let start = local_to_canonical[0];
        start_weights_by_canonical
            .entry(start)
            .and_modify(|existing| *existing = existing.union(&witness_weight))
            .or_insert(witness_weight);
    }

    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let global_start = nwa.add_state();
    let seed_final = nwa.add_state();
    nwa.set_final_weight(seed_final, Weight::all());
    let canonical_state_offset = nwa.num_states();
    for _ in &canonical_nodes {
        nwa.add_state();
    }
    nwa.set_start_states(vec![global_start]);

    for (sequence, by_state) in seed_relations {
        debug_assert_eq!(sequence.len(), 1);
        let weight = relation_weight(by_state);
        if !weight.is_empty() {
            nwa.add_transition(global_start, sequence[0] as i32, seed_final, weight);
        }
    }
    for (canonical, weight) in start_weights_by_canonical {
        nwa.add_epsilon(
            global_start,
            canonical_state_offset + canonical as u32,
            weight,
        );
    }
    for (canonical, node) in canonical_nodes.into_iter().enumerate() {
        let source = canonical_state_offset + canonical as u32;
        if node.accepting {
            nwa.set_final_weight(source, Weight::all());
        }
        for (terminal, target) in node.transitions {
            nwa.add_transition(
                source,
                terminal as i32,
                canonical_state_offset + target as u32,
                Weight::all(),
            );
        }
        for target in node.epsilons {
            nwa.add_epsilon(
                source,
                canonical_state_offset + target as u32,
                Weight::all(),
            );
        }
    }
    let raw_states = nwa.num_states();
    let raw_transitions = nwa.num_transitions();
    let canonical_state_count = raw_states.saturating_sub(canonical_state_offset);
    let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;
    let terminal_automaton = TerminalAutomaton::EpsilonNwa(nwa);
    let determinize_ms = 0.0;
    let minimize_ms = 0.0;

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_direct_terminal] witnesses={} selected_tokens={} raw_lexer_states={} boundary_tsids={} canonical_states={} raw_states={} raw_transitions={} one_byte_ms={one_byte_ms:.3} quotient_ms={quotient_ms:.3} build_ms={build_ms:.3} determinize_ms={determinize_ms:.3} minimize_ms={minimize_ms:.3} total_ms={:.3}",
            discovery.witnesses.len(),
            selected_original_tokens.len(),
            num_states,
            id_map.num_tsids(),
            canonical_state_count,
            raw_states,
            raw_transitions,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(MappedArtifact::new(terminal_automaton, id_map))
}

/// Exact union/determinization specialized for epsilon-free acyclic component
/// parser automata. Reachable singleton raw states are copied directly;
/// synthetic subset states
/// are created only for genuinely overlapping branches (normally at the merged
/// root). This remains exact when a future table relation introduces additional
/// label overlap, rather than falling off a disjoint-alphabet fast path.
fn supports_overlap_local_union(automata: &[NWA]) -> bool {
    automata.iter().all(|automaton| {
        automaton.is_acyclic()
            && automaton
                .states()
                .iter()
                .all(|state| state.epsilons.is_empty())
    })
}

fn determinize_epsilon_free_component_union(
    automata: Vec<NWA>,
    default_positive_label_count: Option<u32>,
) -> Option<(DWA, usize)> {
    if !supports_overlap_local_union(&automata) {
        return None;
    }

    let mut raw_states = Vec::new();
    let mut starts = Vec::new();
    for automaton in automata {
        let offset = raw_states.len() as u32;
        let (states, start_states) = automaton.into_parts();
        starts.extend(start_states.into_iter().map(|state| offset + state));
        for mut appended in states {
            for targets in appended.transitions.values_mut() {
                for (target, _) in targets {
                    *target += offset;
                }
            }
            raw_states.push(appended);
        }
    }
    if starts.is_empty() {
        return Some((DWA::new(0, 0), 0));
    }

    type ResidualSubset = SmallVec<[(u32, Weight); 4]>;
    type ResidualSubsetKey = SmallVec<[(u32, usize); 4]>;

    fn normalize_subset(entries: Vec<(u32, Weight)>) -> ResidualSubset {
        let mut by_state = FxHashMap::<u32, Weight>::default();
        for (state, weight) in entries {
            if weight.is_empty() {
                continue;
            }
            by_state
                .entry(state)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
        let mut normalized = by_state.into_iter().collect::<ResidualSubset>();
        normalized.sort_unstable_by_key(|(state, _)| *state);
        normalized
    }

    fn intern_singleton(
        raw_state: u32,
        singleton_states: &mut [u32],
        singleton_count: &mut usize,
        states: &mut Vec<DWAState>,
        queue: &mut VecDeque<(u32, ResidualSubset)>,
    ) -> u32 {
        let slot = &mut singleton_states[raw_state as usize];
        if *slot != u32::MAX {
            return *slot;
        }
        let created = states.len() as u32;
        states.push(DWAState::default());
        *slot = created;
        *singleton_count += 1;
        let mut subset = ResidualSubset::new();
        subset.push((raw_state, Weight::all()));
        queue.push_back((created, subset));
        created
    }

    fn finish_overlap_transition(
        mut contributions: SmallVec<[(u32, Weight); 4]>,
        weight_ops: &mut ScopedWeightOpCache,
        singleton_states: &mut [u32],
        singleton_count: &mut usize,
        states: &mut Vec<DWAState>,
        queue: &mut VecDeque<(u32, ResidualSubset)>,
        subset_states: &mut FxHashMap<ResidualSubsetKey, u32>,
    ) -> Option<(u32, Weight)> {
        let finish_subset = |
            normalized: ResidualSubset,
            edge_weight: Weight,
            singleton_states: &mut [u32],
            singleton_count: &mut usize,
            states: &mut Vec<DWAState>,
            queue: &mut VecDeque<(u32, ResidualSubset)>,
            subset_states: &mut FxHashMap<ResidualSubsetKey, u32>,
        | {
            if normalized.len() == 1 {
                return intern_singleton(
                    normalized[0].0,
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                );
            }
            let key = normalized
                .iter()
                .map(|(state, weight)| (*state, weight.ptr_key()))
                .collect::<ResidualSubsetKey>();
            if let Some(&existing) = subset_states.get(&key) {
                existing
            } else {
                let created = states.len() as u32;
                states.push(DWAState::default());
                subset_states.insert(key, created);
                queue.push_back((created, normalized));
                created
            }
        };

        match contributions.len() {
            0 => None,
            1 => {
                let (target, edge_weight) = contributions.pop().unwrap();
                let target = intern_singleton(
                    target,
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                );
                Some((target, edge_weight))
            }
            2 => {
                let (right_target, right_weight) = contributions.pop().unwrap();
                let (left_target, left_weight) = contributions.pop().unwrap();
                if left_target == right_target {
                    let edge_weight = weight_ops.union(&left_weight, &right_weight);
                    let target = intern_singleton(
                        left_target,
                        singleton_states,
                        singleton_count,
                        states,
                        queue,
                    );
                    return Some((target, edge_weight));
                }
                let edge_weight = weight_ops.union(&left_weight, &right_weight);
                let edge_complement = edge_weight.complement();
                let mut normalized = ResidualSubset::new();
                let left_residual = if edge_complement.is_empty() {
                    left_weight
                } else {
                    weight_ops.union(&left_weight, &edge_complement)
                };
                let right_residual = if edge_complement.is_empty() {
                    right_weight
                } else {
                    weight_ops.union(&right_weight, &edge_complement)
                };
                if left_target < right_target {
                    normalized.push((left_target, left_residual));
                    normalized.push((right_target, right_residual));
                } else {
                    normalized.push((right_target, right_residual));
                    normalized.push((left_target, left_residual));
                }
                let target = finish_subset(
                    normalized,
                    edge_weight.clone(),
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                    subset_states,
                );
                Some((target, edge_weight))
            }
            _ => {
                contributions.sort_unstable_by_key(|(target, _)| *target);
                let mut next_subset = ResidualSubset::new();
                for (target, contribution) in contributions {
                    if let Some((last_target, existing)) = next_subset.last_mut()
                        && *last_target == target
                    {
                        *existing = weight_ops.union(existing, &contribution);
                    } else {
                        next_subset.push((target, contribution));
                    }
                }
                let edge_weight =
                    weight_ops.union_all(next_subset.iter().map(|(_, weight)| weight));
                let edge_complement = edge_weight.complement();
                let normalized = if edge_complement.is_empty() {
                    next_subset
                } else {
                    next_subset
                        .into_iter()
                        .map(|(state, weight)| {
                            let residual = weight_ops.union(&weight, &edge_complement);
                            (state, residual)
                        })
                        .collect::<ResidualSubset>()
                };
                let target = finish_subset(
                    normalized,
                    edge_weight.clone(),
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                    subset_states,
                );
                Some((target, edge_weight))
            }
        }
    }

    let mut states = Vec::<DWAState>::new();
    let mut queue = VecDeque::<(u32, ResidualSubset)>::new();
    let mut singleton_states = vec![u32::MAX; raw_states.len()];
    let mut singleton_count = 0usize;
    let initial_subset = normalize_subset(
        starts
            .into_iter()
            .map(|state| (state, Weight::all()))
            .collect(),
    );
    let start_state = if initial_subset.len() == 1 {
        intern_singleton(
            initial_subset[0].0,
            &mut singleton_states,
            &mut singleton_count,
            &mut states,
            &mut queue,
        )
    } else {
        let state = states.len() as u32;
        states.push(DWAState::default());
        queue.push_back((state, initial_subset.clone()));
        state
    };
    let mut subset_states = FxHashMap::<ResidualSubsetKey, u32>::default();
    if initial_subset.len() > 1 {
        subset_states.insert(
            initial_subset
                .iter()
                .map(|(state, weight)| (*state, weight.ptr_key()))
                .collect::<ResidualSubsetKey>(),
            start_state,
        );
    }

    let mut weight_ops = ScopedWeightOpCache::default();
    let mut profiled_singletons = 0usize;
    let mut profiled_pair_subsets = 0usize;
    let mut profiled_wide_subsets = 0usize;
    let mut profiled_max_subset = 0usize;
    let mut profiled_explicit_labels = 0usize;
    while let Some((output_state, subset)) = queue.pop_front() {
        profiled_max_subset = profiled_max_subset.max(subset.len());
        match subset.len() {
            1 => profiled_singletons += 1,
            2 => profiled_pair_subsets += 1,
            _ => profiled_wide_subsets += 1,
        }
        // Singleton states are always interned with an all-weight prefix. When
        // the raw row is already deterministic, copy it directly instead of
        // rebuilding the same row through label grouping, hash maps, and
        // weight normalization. Only genuine overlap subsets need the general
        // determinization path below.
        if subset.len() == 1 && subset[0].1.is_full() {
            let raw_state = subset[0].0;
            let source = &raw_states[raw_state as usize];
            if source.transitions.values().all(|targets| targets.len() <= 1) {
                let mut output_transitions = Vec::with_capacity(source.transitions.len());
                let final_weight = source
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                for (&label, targets) in &source.transitions {
                    let Some((target, edge_weight)) = targets.first() else {
                        continue;
                    };
                    if edge_weight.is_empty() {
                        continue;
                    }
                    let target = intern_singleton(
                        *target,
                        &mut singleton_states,
                        &mut singleton_count,
                        &mut states,
                        &mut queue,
                    );
                    output_transitions.push((label, (target, edge_weight.clone())));
                }
                states[output_state as usize] = DWAState {
                    transitions: output_transitions.into_iter().collect(),
                    final_weight,
                };
                continue;
            }
        }

        let mut final_parts = SmallVec::<[Weight; 4]>::new();
        for (raw_state, prefix_weight) in &subset {
            let source = &raw_states[*raw_state as usize];
            if let Some(final_weight) = &source.final_weight {
                let contribution = weight_ops.intersection(prefix_weight, final_weight);
                if !contribution.is_empty() {
                    final_parts.push(contribution);
                }
            }
        }
        // Keep wildcard transitions symbolic. Process one label at a time so
        // synthetic rows do not allocate a nested label->target hash table and
        // sort it again afterward. Raw rows are deterministic in the common
        // path; the small target vector also handles genuine NWA overlap.
        let final_weight = if final_parts.is_empty() {
            None
        } else {
            let weight = weight_ops.union_all(final_parts.iter());
            (!weight.is_empty()).then_some(weight)
        };
        let mut output_transitions = Vec::<(i32, (u32, Weight))>::new();

        if subset.len() == 2 {
            let left_source = &raw_states[subset[0].0 as usize];
            let right_source = &raw_states[subset[1].0 as usize];
            let left_prefix = &subset[0].1;
            let right_prefix = &subset[1].1;
            let left_default = left_source.transitions.get(&DEFAULT_LABEL);
            let right_default = right_source.transitions.get(&DEFAULT_LABEL);
            let mut left = left_source
                .transitions
                .iter()
                .filter(|(label, _)| **label != DEFAULT_LABEL)
                .peekable();
            let mut right = right_source
                .transitions
                .iter()
                .filter(|(label, _)| **label != DEFAULT_LABEL)
                .peekable();

            loop {
                let left_label = left.peek().map(|(label, _)| **label);
                let right_label = right.peek().map(|(label, _)| **label);
                let Some(label) = (match (left_label, right_label) {
                    (Some(left), Some(right)) => Some(left.min(right)),
                    (Some(left), None) => Some(left),
                    (None, Some(right)) => Some(right),
                    (None, None) => None,
                }) else {
                    break;
                };
                profiled_explicit_labels += 1;
                let left_targets = if left_label == Some(label) {
                    left.next().map(|(_, targets)| targets)
                } else if label >= 0 {
                    left_default
                } else {
                    None
                };
                let right_targets = if right_label == Some(label) {
                    right.next().map(|(_, targets)| targets)
                } else if label >= 0 {
                    right_default
                } else {
                    None
                };
                let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                for (targets, prefix) in [
                    (left_targets, left_prefix),
                    (right_targets, right_prefix),
                ] {
                    let Some(targets) = targets else {
                        continue;
                    };
                    for (target, edge_weight) in targets {
                        let contribution = weight_ops.intersection(prefix, edge_weight);
                        if !contribution.is_empty() {
                            contributions.push((*target, contribution));
                        }
                    }
                }
                if let Some((target, edge_weight)) = finish_overlap_transition(
                    contributions,
                    &mut weight_ops,
                    &mut singleton_states,
                    &mut singleton_count,
                    &mut states,
                    &mut queue,
                    &mut subset_states,
                ) {
                    output_transitions.push((label, (target, edge_weight)));
                }
            }

            if default_positive_label_count.is_some()
                && (left_default.is_some() || right_default.is_some())
            {
                let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                for (targets, prefix) in [
                    (left_default, left_prefix),
                    (right_default, right_prefix),
                ] {
                    let Some(targets) = targets else {
                        continue;
                    };
                    for (target, edge_weight) in targets {
                        let contribution = weight_ops.intersection(prefix, edge_weight);
                        if !contribution.is_empty() {
                            contributions.push((*target, contribution));
                        }
                    }
                }
                if let Some((target, edge_weight)) = finish_overlap_transition(
                    contributions,
                    &mut weight_ops,
                    &mut singleton_states,
                    &mut singleton_count,
                    &mut states,
                    &mut queue,
                    &mut subset_states,
                ) {
                    // DEFAULT_LABEL sorts before all encoded parser labels.
                    output_transitions.insert(0, (DEFAULT_LABEL, (target, edge_weight)));
                }
            }
            states[output_state as usize] = DWAState {
                transitions: output_transitions.into_iter().collect(),
                final_weight,
            };
            continue;
        }

        let mut explicit_labels = SmallVec::<[i32; 32]>::new();
        for (raw_state, _) in &subset {
            explicit_labels.extend(
                raw_states[*raw_state as usize]
                    .transitions
                    .keys()
                    .copied()
                    .filter(|&label| label != DEFAULT_LABEL),
            );
        }
        explicit_labels.sort_unstable();
        explicit_labels.dedup();
        profiled_explicit_labels += explicit_labels.len();

        let include_default = default_positive_label_count.is_some()
            && subset.iter().any(|(raw_state, _)| {
                raw_states[*raw_state as usize]
                    .transitions
                    .contains_key(&DEFAULT_LABEL)
            });
        let labels = explicit_labels
            .iter()
            .copied()
            .chain(include_default.then_some(DEFAULT_LABEL));
        for label in labels {
            let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
            for (raw_state, prefix_weight) in &subset {
                let source = &raw_states[*raw_state as usize];
                let targets = if label == DEFAULT_LABEL {
                    source.transitions.get(&DEFAULT_LABEL)
                } else {
                    source.transitions.get(&label).or_else(|| {
                        (label >= 0)
                            .then(|| source.transitions.get(&DEFAULT_LABEL))
                            .flatten()
                    })
                };
                let Some(targets) = targets else {
                    continue;
                };
                for (target, edge_weight) in targets {
                    let contribution = weight_ops.intersection(prefix_weight, edge_weight);
                    if !contribution.is_empty() {
                        contributions.push((*target, contribution));
                    }
                }
            }
            if let Some((target, edge_weight)) = finish_overlap_transition(
                contributions,
                &mut weight_ops,
                &mut singleton_states,
                &mut singleton_count,
                &mut states,
                &mut queue,
                &mut subset_states,
            ) {
                output_transitions.push((label, (target, edge_weight)));
            }
        }
        states[output_state as usize] = DWAState {
            transitions: output_transitions.into_iter().collect(),
            final_weight,
        };
    }

    let synthetic_states = states.len().saturating_sub(singleton_count);
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_overlap_local_shape] raw_states={} result_states={} singleton_states={} pair_subsets={} wide_subsets={} max_subset={} explicit_labels={}",
            raw_states.len(),
            states.len(),
            profiled_singletons,
            profiled_pair_subsets,
            profiled_wide_subsets,
            profiled_max_subset,
            profiled_explicit_labels,
        );
    }
    Some((DWA::from_parts(states, start_state), synthetic_states))
}


struct UnmappedComponentParserArtifact {
    automaton: NWA,
    possible_matches: PossibleMatches,
}

fn prepare_unmapped_component_parser_artifacts(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
) -> Result<Vec<UnmappedComponentParserArtifact>, String> {
    if components.len() != terminal_offsets.len() {
        return Err("component/parser terminal-offset count mismatch".into());
    }
    components
        .par_iter()
        .copied()
        .zip(terminal_offsets.par_iter().copied())
        .map(|(component, terminal_offset)| {
            Ok(UnmappedComponentParserArtifact {
                automaton: component_parser_nwa(&component)?,
                possible_matches: component_possible_matches(&component, terminal_offset)?,
            })
        })
        .collect()
}

#[derive(Debug)]
struct BoundaryRefinementPlan {
    common_map: InternalIdMap,
    component_token_map: Vec<Vec<u32>>,
    boundary_tsid_map: Vec<Vec<u32>>,
    boundary_token_map: Vec<Vec<u32>>,
}

struct PreparedOwnedComponentArtifacts {
    automata: Vec<NWA>,
    possible_matches: PossibleMatches,
    id_map: InternalIdMap,
    boundary_tsid_map: Option<Vec<Vec<u32>>>,
    boundary_token_map: Option<Vec<Vec<u32>>>,
    remap_ms: f64,
}

fn build_boundary_refinement_plan(
    component_map: InternalIdMap,
    boundary_map: &InternalIdMap,
) -> Option<BoundaryRefinementPlan> {
    let boundary_tsid_map = boundary_map
        .tokenizer_states
        .internal_to_originals
        .par_iter()
        .map(|originals| {
            let mut mapped = Vec::new();
            for &original in originals {
                let component_tsid = component_map
                    .tokenizer_states
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if component_tsid == u32::MAX {
                    return None;
                }
                mapped.push(component_tsid);
            }
            mapped.sort_unstable();
            mapped.dedup();
            Some(mapped)
        })
        .collect::<Option<Vec<_>>>()?;

    let original_count = component_map
        .vocab_tokens
        .original_to_internal
        .len()
        .max(boundary_map.vocab_tokens.original_to_internal.len());
    let mut original_to_common = vec![u32::MAX; original_count];
    let mut common_to_originals = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<u32>::new();
    let mut component_token_map =
        vec![Vec::<u32>::new(); component_map.num_internal_tokens() as usize];
    let mut boundary_token_map =
        vec![Vec::<u32>::new(); boundary_map.num_internal_tokens() as usize];

    for (component_token, originals) in component_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let mut by_boundary = BTreeMap::<u32, Vec<u32>>::new();
        for &original in originals {
            let boundary_token = boundary_map
                .vocab_tokens
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            by_boundary.entry(boundary_token).or_default().push(original);
        }
        for (boundary_token, grouped_originals) in by_boundary {
            let common = common_to_originals.len() as u32;
            component_token_map[component_token].push(common);
            if boundary_token != u32::MAX {
                boundary_token_map[boundary_token as usize].push(common);
            }
            for &original in &grouped_originals {
                original_to_common[original as usize] = common;
            }
            representatives.push(grouped_originals[0]);
            common_to_originals.push(grouped_originals);
        }
    }
    for (boundary_token, originals) in boundary_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let right_only = originals
            .iter()
            .copied()
            .filter(|&original| {
                component_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
                    == u32::MAX
            })
            .collect::<Vec<_>>();
        if right_only.is_empty() {
            continue;
        }
        let common = common_to_originals.len() as u32;
        boundary_token_map[boundary_token].push(common);
        for &original in &right_only {
            original_to_common[original as usize] = common;
        }
        representatives.push(right_only[0]);
        common_to_originals.push(right_only);
    }
    Some(BoundaryRefinementPlan {
        common_map: InternalIdMap {
            tokenizer_states: component_map.tokenizer_states,
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: original_to_common,
                internal_to_originals: common_to_originals,
                representative_original_ids: representatives,
            },
            deferred_vocab_singleton_original_ids: None,
        },
        component_token_map,
        boundary_tsid_map,
        boundary_token_map,
    })
}

fn compose_local_id_map(
    local_to_base: &[Vec<u32>],
    base_to_common: &[Vec<u32>],
) -> Vec<Vec<u32>> {
    local_to_base
        .iter()
        .map(|base_ids| {
            let mut common = base_ids
                .iter()
                .filter_map(|&base| base_to_common.get(base as usize))
                .flat_map(|ids| ids.iter().copied())
                .collect::<Vec<_>>();
            common.sort_unstable();
            common.dedup();
            common
        })
        .collect()
}

fn remap_unmapped_component_artifacts(
    artifacts: Vec<UnmappedComponentParserArtifact>,
    component_maps: Vec<DirectComponentCoordinateMaps>,
    base_to_common_tokens: Option<&[Vec<u32>]>,
    common_tsid_count: usize,
) -> Result<(Vec<NWA>, PossibleMatches, f64), String> {
    if artifacts.len() != component_maps.len() {
        return Err("component artifact/map count mismatch".into());
    }
    let started_at = Instant::now();
    let remapped = artifacts
        .into_par_iter()
        .zip(component_maps.into_par_iter())
        .map(|(artifact, maps)| {
            let token_map = base_to_common_tokens.map_or(
                maps.local_to_global_tokens.clone(),
                |base_to_common| {
                    compose_local_id_map(&maps.local_to_global_tokens, base_to_common)
                },
            );
            let mut pair = (artifact.automaton, artifact.possible_matches);
            let mut weights = pair.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &maps.local_to_global_tsids,
                &token_map,
                common_tsid_count,
            );
            Ok::<_, String>(pair)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut automata = Vec::with_capacity(remapped.len());
    let mut possible_matches = PossibleMatches::new();
    for (automaton, component_possible_matches) in remapped {
        automata.push(automaton);
        for (terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
    }
    Ok((
        automata,
        possible_matches,
        started_at.elapsed().as_secs_f64() * 1000.0,
    ))
}

fn component_possible_matches(
    component: &ParserDwaComponent<'_>,
    terminal_offset: u32,
) -> Result<PossibleMatches, String> {
    if !component.constraint.possible_matches_complete {
        return Err(
            "cannot compose a constraint with incomplete possible_matches; the dynamic possible-matches fallback is forbidden for constraint composition"
                .into(),
        );
    }
    Ok(component
        .constraint
        .possible_matches
        .iter()
        .map(|(&terminal, weight)| (terminal_offset + terminal, weight.clone()))
        .collect())
}

pub(crate) fn compose_component_parser_dwas_and_possible_matches(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
    merged_tokenizer_state_count: usize,
    original_token_ids: &[u32],
) -> Result<MappedArtifact<(DWA, PossibleMatches)>, String> {
    if components.is_empty() {
        return Err("cannot compose zero parser DWAs".into());
    }
    if terminal_offsets.len() != components.len() {
        return Err(format!(
            "terminal-offset count {} does not match component count {}",
            terminal_offsets.len(),
            components.len(),
        ));
    }
    let total_started_at = Instant::now();
    let coordinate_started_at = Instant::now();
    let (id_map, component_maps) = build_direct_component_coordinate_maps(
        components,
        merged_tokenizer_state_count,
        original_token_ids,
    )?;
    let coordinate_ms = coordinate_started_at.elapsed().as_secs_f64() * 1000.0;
    let global_tsid_count = id_map.num_tsids() as usize;
    struct PreparedComponentArtifact {
        artifact: (NWA, PossibleMatches),
        parser_nwa_ms: f64,
        possible_matches_ms: f64,
        remap_ms: f64,
        component_index: usize,
        local_tsids: usize,
        tsid_fanout: usize,
        local_tokens: usize,
        token_fanout: usize,
    }
    let prepared = components
        .par_iter()
        .copied()
        .zip(terminal_offsets.par_iter().copied())
        .zip(component_maps.into_par_iter())
        .enumerate()
        .map(|(component_index, ((component, terminal_offset), coordinate_maps))| {
            let started_at = Instant::now();
            let parser_nwa = component_parser_nwa(&component)?;
            let parser_nwa_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            let started_at = Instant::now();
            let possible_matches = component_possible_matches(&component, terminal_offset)?;
            let possible_matches_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            let mut artifact = (parser_nwa, possible_matches);
            let started_at = Instant::now();
            let mut weights = artifact.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &coordinate_maps.local_to_global_tsids,
                &coordinate_maps.local_to_global_tokens,
                global_tsid_count,
            );
            drop(weights);
            let remap_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            Ok::<_, String>(PreparedComponentArtifact {
                artifact,
                parser_nwa_ms,
                possible_matches_ms,
                remap_ms,
                component_index,
                local_tsids: coordinate_maps.local_to_global_tsids.len(),
                tsid_fanout: coordinate_maps
                    .local_to_global_tsids
                    .iter()
                    .map(Vec::len)
                    .sum(),
                local_tokens: coordinate_maps.local_to_global_tokens.len(),
                token_fanout: coordinate_maps
                    .local_to_global_tokens
                    .iter()
                    .map(Vec::len)
                    .sum(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let parser_nwa_ms = prepared
        .iter()
        .map(|prepared| prepared.parser_nwa_ms)
        .fold(0.0, f64::max);
    let possible_matches_ms = prepared
        .iter()
        .map(|prepared| prepared.possible_matches_ms)
        .fold(0.0, f64::max);
    let remap_ms = prepared
        .iter()
        .map(|prepared| prepared.remap_ms)
        .fold(0.0, f64::max);
    if compose_profile_enabled() {
        for prepared in &prepared {
            eprintln!(
                "[glrmask/profile][constraint_component_remap] component={} local_tsids={} global_tsids={} tsid_fanout={} local_tokens={} global_tokens={} token_fanout={} parser_nwa_ms={:.3} remap_ms={:.3}",
                prepared.component_index,
                prepared.local_tsids,
                global_tsid_count,
                prepared.tsid_fanout,
                prepared.local_tokens,
                id_map.num_internal_tokens(),
                prepared.token_fanout,
                prepared.parser_nwa_ms,
                prepared.remap_ms,
            );
        }
    }
    let artifacts = prepared
        .into_iter()
        .map(|prepared| prepared.artifact)
        .collect::<Vec<_>>();

    let automata = artifacts
        .iter()
        .map(|(automaton, _)| automaton)
        .collect::<Vec<_>>();
    let union_nwa_states = automata
        .iter()
        .map(|automaton| automaton.num_states())
        .sum::<u32>();
    let mut possible_matches = PossibleMatches::new();
    for (_, component_possible_matches) in &artifacts {
        for (&terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }
    }

    let build_generic = || -> Result<(DWA, f64, f64), String> {
        let append_started_at = Instant::now();
        let mut union = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let mut starts = Vec::new();
        for automaton in &automata {
            let body = union.append_with_body(automaton);
            starts.extend(body.start_states);
        }
        union.set_start_states(starts);
        let append_ms = append_started_at.elapsed().as_secs_f64() * 1000.0;
        let determinize_started_at = Instant::now();
        let dwa = determinize(&union).map_err(|error| error.to_string())?;
        let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
        Ok((dwa, append_ms, determinize_ms))
    };

    let direct_started_at = Instant::now();
    let direct = if std::env::var_os("GLRMASK_COMPOSE_GENERIC_COMPONENT_UNION").is_some() {
        None
    } else {
        determinize_epsilon_free_component_union(
            automata.iter().map(|automaton| (*automaton).clone()).collect(),
            None,
        )
    };
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, union_path, synthetic_states, append_ms, determinize_ms) =
        if let Some((direct_dwa, synthetic_states)) = direct {
            if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_DIRECT_UNION").is_some() {
                let (reference, _, _) = build_generic()?;
                let difference = find_difference(&direct_dwa, &reference)
                    .map_err(|error| error.to_string())?;
                assert_eq!(
                    difference, None,
                    "direct component parser-DWA union differs from generic determinization",
                );
                eprintln!(
                    "[glrmask/validate][compose_component_direct_union] raw_states={} synthetic_states={} exact=true",
                    union_nwa_states,
                    synthetic_states,
                );
            }
            (direct_dwa, "direct", synthetic_states, 0.0, 0.0)
        } else {
            let (generic, append_ms, determinize_ms) = build_generic()?;
            (generic, "generic", 0, append_ms, determinize_ms)
        };
    if compose_profile_enabled() {
        let start_count = automata
            .iter()
            .map(|automaton| automaton.start_states().len())
            .sum::<usize>();
        let epsilon_edges = automata
            .iter()
            .flat_map(|automaton| automaton.states())
            .map(|state| state.epsilons.len())
            .sum::<usize>();
        let nondeterministic_rows = automata
            .iter()
            .flat_map(|automaton| automaton.states())
            .flat_map(|state| state.transitions.values())
            .filter(|targets| targets.len() > 1)
            .count();
        eprintln!(
            "[glrmask/profile][constraint_component_reuse] components={} global_tsids={} global_tokens={} coordinate_ms={coordinate_ms:.3} parser_nwa_ms={parser_nwa_ms:.3} possible_matches_ms={possible_matches_ms:.3} remap_ms={remap_ms:.3} union_path={} direct_ms={direct_ms:.3} append_ms={append_ms:.3} determinize_ms={determinize_ms:.3} starts={} epsilon_edges={} nondeterministic_rows={} union_nwa_states={} synthetic_states={} result_states={} total_ms={:.3}",
            components.len(),
            id_map.num_tsids(),
            id_map.num_internal_tokens(),
            union_path,
            start_count,
            epsilon_edges,
            nondeterministic_rows,
            union_nwa_states,
            synthetic_states,
            dwa.num_states(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(MappedArtifact::new((dwa, possible_matches), id_map))
}

fn explicit_parser_nwa(dwa: &DWA, num_parser_states: u32) -> NWA {
    let mut nwa = NWA::new(0, 0);
    for _ in dwa.states() {
        nwa.add_state();
    }
    nwa.set_start_states(vec![dwa.start_state()]);
    for (source, state) in dwa.states().iter().enumerate() {
        if let Some(final_weight) = &state.final_weight {
            nwa.set_final_weight(source as u32, final_weight.clone());
        }
        let explicit_positive = state
            .transitions
            .keys()
            .filter_map(|&label| {
                (label >= 0 && label != DEFAULT_LABEL).then_some(label as u32)
            })
            .collect::<BTreeSet<_>>();
        for (&label, (target, weight)) in &state.transitions {
            if label != DEFAULT_LABEL {
                nwa.add_transition(source as u32, label, *target, weight.clone());
            }
        }
        if let Some((target, weight)) = state.transitions.get(&DEFAULT_LABEL) {
            for parser_state in 0..num_parser_states {
                if !explicit_positive.contains(&parser_state) {
                    nwa.add_transition(
                        source as u32,
                        encode_positive_label(parser_state),
                        *target,
                        weight.clone(),
                    );
                }
            }
        }
    }
    nwa
}

fn parser_nwa_preserve_defaults(dwa: &DWA) -> NWA {
    let states = dwa
        .states()
        .iter()
        .map(|state| crate::automata::weighted_u32::nwa::NWAState {
            final_weight: state.final_weight.clone(),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| {
                    (label, vec![(*target, weight.clone())])
                })
                .collect(),
            epsilons: Vec::new(),
        })
        .collect::<Vec<_>>();
    NWA::from_parts(states, vec![dwa.start_state()])
}

fn pair_boundary_into_component_refinement(
    component_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    boundary: MappedArtifact<DWA>,
) -> Option<MappedArtifact<((DWA, PossibleMatches), DWA)>> {
    let total_started_at = Instant::now();
    let ((mut component_artifact, component_map), (mut boundary_dwa, boundary_map)) =
        (component_artifacts.into_parts(), boundary.into_parts());

    // The component map covers the full raw tokenizer-state domain. Preserve
    // that TSID partition exactly and map each compact boundary TSID directly
    // to the component classes represented by its raw states.
    if component_map.tokenizer_states.original_to_internal.len()
        < boundary_map.tokenizer_states.original_to_internal.len()
    {
        return None;
    }
    let tsid_started_at = Instant::now();
    let component_tsid_count = component_map.num_tsids() as usize;
    let boundary_tsid_map = boundary_map
        .tokenizer_states
        .internal_to_originals
        .par_iter()
        .map(|originals| {
            let mut seen = vec![false; component_tsid_count];
            let mut mapped = Vec::new();
            for &original in originals {
                let component_tsid = component_map
                    .tokenizer_states
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                if component_tsid == u32::MAX {
                    return None;
                }
                if !std::mem::replace(&mut seen[component_tsid as usize], true) {
                    mapped.push(component_tsid);
                }
            }
            mapped.sort_unstable();
            Some(mapped)
        })
        .collect::<Option<Vec<_>>>()?;
    let component_tsid_map = (0..component_tsid_count as u32)
        .map(|tsid| vec![tsid])
        .collect::<Vec<_>>();
    let tsid_ms = tsid_started_at.elapsed().as_secs_f64() * 1000.0;

    // Preserve component-major token locality. Only component classes touched
    // by a boundary class are split; all untouched originals remain together.
    // This is the exact pair refinement ordered by (component_class,
    // boundary_class), followed by boundary-only classes.
    let token_started_at = Instant::now();
    let original_count = component_map
        .vocab_tokens
        .original_to_internal
        .len()
        .max(boundary_map.vocab_tokens.original_to_internal.len());
    let mut original_to_common = vec![u32::MAX; original_count];
    let mut common_to_originals = Vec::<Vec<u32>>::new();
    let mut representatives = Vec::<u32>::new();
    let mut component_token_map =
        vec![Vec::<u32>::new(); component_map.num_internal_tokens() as usize];
    let mut boundary_token_map =
        vec![Vec::<u32>::new(); boundary_map.num_internal_tokens() as usize];

    for (component_token, originals) in component_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let mut by_boundary = BTreeMap::<u32, Vec<u32>>::new();
        for &original in originals {
            let boundary_token = boundary_map
                .vocab_tokens
                .original_to_internal
                .get(original as usize)
                .copied()
                .unwrap_or(u32::MAX);
            by_boundary.entry(boundary_token).or_default().push(original);
        }
        for (boundary_token, grouped_originals) in by_boundary {
            let common = common_to_originals.len() as u32;
            component_token_map[component_token].push(common);
            if boundary_token != u32::MAX {
                boundary_token_map[boundary_token as usize].push(common);
            }
            for &original in &grouped_originals {
                original_to_common[original as usize] = common;
            }
            representatives.push(grouped_originals[0]);
            common_to_originals.push(grouped_originals);
        }
    }
    for (boundary_token, originals) in boundary_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .enumerate()
    {
        let right_only = originals
            .iter()
            .copied()
            .filter(|&original| {
                component_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX)
                    == u32::MAX
            })
            .collect::<Vec<_>>();
        if right_only.is_empty() {
            continue;
        }
        let common = common_to_originals.len() as u32;
        boundary_token_map[boundary_token].push(common);
        for &original in &right_only {
            original_to_common[original as usize] = common;
        }
        representatives.push(right_only[0]);
        common_to_originals.push(right_only);
    }
    let common_map = InternalIdMap {
        tokenizer_states: component_map.tokenizer_states.clone(),
        vocab_tokens: ManyToOneIdMap {
            original_to_internal: original_to_common,
            internal_to_originals: common_to_originals,
            representative_original_ids: representatives,
        },
        deferred_vocab_singleton_original_ids: None,
    };
    let token_ms = token_started_at.elapsed().as_secs_f64() * 1000.0;

    let component_started_at = Instant::now();
    let mut component_weights = component_artifact.weight_refs_mut();
    remap_weights_with_maps(
        &mut component_weights,
        &component_tsid_map,
        &component_token_map,
        common_map.num_tsids() as usize,
    );
    let component_ms = component_started_at.elapsed().as_secs_f64() * 1000.0;
    let boundary_started_at = Instant::now();
    let mut boundary_weights = boundary_dwa.weight_refs_mut();
    remap_weights_with_maps(
        &mut boundary_weights,
        &boundary_tsid_map,
        &boundary_token_map,
        common_map.num_tsids() as usize,
    );
    let boundary_ms = boundary_started_at.elapsed().as_secs_f64() * 1000.0;

    if compose_profile_enabled() {
        let split_component_classes = component_token_map
            .iter()
            .filter(|destinations| destinations.len() > 1)
            .count();
        eprintln!(
            "[glrmask/profile][constraint_boundary_direct_reconcile] component_tsids={} boundary_tsids={} common_tsids={} component_tokens={} boundary_tokens={} common_tokens={} split_component_classes={} tsid_ms={tsid_ms:.3} token_ms={token_ms:.3} component_remap_ms={component_ms:.3} boundary_remap_ms={boundary_ms:.3} total_ms={:.3}",
            component_map.num_tsids(),
            boundary_map.num_tsids(),
            common_map.num_tsids(),
            component_map.num_internal_tokens(),
            boundary_map.num_internal_tokens(),
            common_map.num_internal_tokens(),
            split_component_classes,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(MappedArtifact::new(
        (component_artifact, boundary_dwa),
        common_map,
    ))
}

fn union_boundary_parser_dwa(
    component_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    boundary: MappedArtifact<DWA>,
    num_parser_states: u32,
) -> Result<MappedArtifact<(DWA, PossibleMatches)>, String> {
    let total_started_at = Instant::now();
    let pair_started_at = Instant::now();
    let paired = if std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_RECONCILE").is_some() {
        component_artifacts.pair_forced_common(boundary)
    } else {
        pair_boundary_into_component_refinement(component_artifacts, boundary)
            .expect("component map must cover the composed boundary coordinate domain")
    };
    let pair_ms = pair_started_at.elapsed().as_secs_f64() * 1000.0;
    let (((component_dwa, possible_matches), boundary_dwa), id_map) = paired.into_parts();
    let explicit_started_at = Instant::now();
    let component_nwa = parser_nwa_preserve_defaults(&component_dwa);
    let boundary_nwa = parser_nwa_preserve_defaults(&boundary_dwa);
    let explicit_ms = explicit_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        let count_defaults = |dwa: &DWA| {
            dwa.states()
                .iter()
                .filter(|state| state.transitions.contains_key(&DEFAULT_LABEL))
                .count()
        };
        eprintln!(
            "[glrmask/profile][constraint_parser_union_defaults] component_default_states={} boundary_default_states={} component_transitions={} boundary_transitions={}",
            count_defaults(&component_dwa),
            count_defaults(&boundary_dwa),
            component_dwa.num_transitions(),
            boundary_dwa.num_transitions(),
        );
    }
    let automata = [&component_nwa, &boundary_nwa];

    let build_generic = || -> Result<(DWA, f64, f64), String> {
        let append_started_at = Instant::now();
        let generic_component_nwa = explicit_parser_nwa(&component_dwa, num_parser_states);
        let generic_boundary_nwa = explicit_parser_nwa(&boundary_dwa, num_parser_states);
        let mut union = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
        let component_body = union.append_with_body(&generic_component_nwa);
        let boundary_body = union.append_with_body(&generic_boundary_nwa);
        union.set_start_states(
            component_body
                .start_states
                .into_iter()
                .chain(boundary_body.start_states)
                .collect(),
        );
        let append_ms = append_started_at.elapsed().as_secs_f64() * 1000.0;
        let determinize_started_at = Instant::now();
        let parser_dwa = determinize(&union).map_err(|error| error.to_string())?;
        let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
        Ok((parser_dwa, append_ms, determinize_ms))
    };

    let direct_started_at = Instant::now();
    let direct = if std::env::var_os("GLRMASK_COMPOSE_DIRECT_BOUNDARY_UNION").is_some() {
        determinize_epsilon_free_component_union(
            automata.iter().map(|automaton| (*automaton).clone()).collect(),
            Some(num_parser_states),
        )
    } else {
        None
    };
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    let (parser_dwa, union_path, synthetic_states, append_ms, determinize_ms) =
        if let Some((direct_dwa, synthetic_states)) = direct {
            if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_BOUNDARY_DIRECT_UNION").is_some() {
                let (reference, _, _) = build_generic()?;
                let difference = find_difference(&direct_dwa, &reference)
                    .map_err(|error| error.to_string())?;
                assert_eq!(
                    difference, None,
                    "direct boundary parser-DWA union differs from generic determinization",
                );
                eprintln!(
                    "[glrmask/validate][compose_boundary_direct_union] raw_states={} synthetic_states={} exact=true",
                    component_nwa.num_states() + boundary_nwa.num_states(),
                    synthetic_states,
                );
            }
            (direct_dwa, "direct", synthetic_states, 0.0, 0.0)
        } else {
            let (generic, append_ms, determinize_ms) = build_generic()?;
            (generic, "generic", 0, append_ms, determinize_ms)
        };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_parser_union] component_states={} boundary_states={} raw_states={} result_states={} pair_ms={pair_ms:.3} explicit_ms={explicit_ms:.3} union_path={} direct_ms={direct_ms:.3} synthetic_states={} append_ms={append_ms:.3} determinize_ms={determinize_ms:.3} total_ms={:.3}",
            component_dwa.num_states(),
            boundary_dwa.num_states(),
            component_nwa.num_states() + boundary_nwa.num_states(),
            parser_dwa.num_states(),
            union_path,
            synthetic_states,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(MappedArtifact::new(
        (parser_dwa, possible_matches),
        id_map,
    ))
}

fn build_composition_templates(
    table: &crate::compiler::glr::table::GLRTable,
    analyzed: &AnalyzedGrammar,
    selected: &[bool],
) -> (
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
) {
    let started_at = Instant::now();
    let characterizations = characterize_selected_terminals(table, analyzed, selected);
    let templates = Templates::from_characterizations(&characterizations);
    let mut template_dfas_by_terminal = vec![None; analyzed.num_terminals as usize];
    for (&terminal, dfa) in &templates.by_terminal {
        let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
        if let Some(split) = try_split_commit_template_dfas(&commit_dfa)
            && let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize)
        {
            *slot = Some(Arc::new(split));
        }
    }
    (
        templates,
        template_dfas_by_terminal,
        started_at.elapsed().as_secs_f64() * 1000.0,
    )
}

fn build_boundary_repair(
    composed_table: &ComposedTable,
    merged_tokenizer: Option<&Tokenizer>,
    merged_tokenizer_state_count: usize,
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<u32>,
    vocab: &Vocab,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    precomputed_component_state_map: Option<&ManyToOneIdMap>,
    deferred_component_state_map: Option<&OnceLock<Result<ManyToOneIdMap, String>>>,
    selected_boundary_tokens: Option<&OnceLock<Result<Option<Vec<u32>>, String>>>,
) -> Result<Option<BoundaryRepair>, String> {
    let total_started_at = Instant::now();
    let mut seed_terminals = vec![false; composed_table.table.num_terminals as usize];
    for &terminal in &composed_table.boundary_seed_terminals {
        if let Some(slot) = seed_terminals.get_mut(terminal as usize) {
            *slot = true;
        }
    }
    if let Some(ignore_terminal) = ignore_terminal
        && let Some(slot) = seed_terminals.get_mut(ignore_terminal as usize)
    {
        *slot = true;
    }

    let augmented_start = composed_table
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let analyzed = AnalyzedGrammar::from_composed_rules(
        composed_table.table.rules.clone(),
        composed_table.table.num_terminals,
        terminal_display_names,
        composed_table.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    // The composed rule graph substitutes every placeholder occurrence with
    // the remapped child root. Its grammar-level ever-follow relation is
    // therefore exact for adjacent terminal matches and can discard lexical
    // component switches that no parser derivation can realize. Retain an
    // opt-out while the composition differential corpus validates this path.
    let disallowed_follows = if std::env::var_os("GLRMASK_COMPOSE_DISABLE_GRAMMAR_FOLLOWS")
        .is_some()
    {
        BTreeMap::<u32, BitSet>::new()
    } else {
        crate::compiler::pipeline::compute_disallowed_follows(&analyzed)
    };
    // Mixed-token discovery and exact one-byte seed analysis are independent
    // read-only passes over the tokenizer.  Running them serially made boundary
    // repair pay two full million-state/vocabulary scans back-to-back.
    let eager_all_templates =
        std::env::var_os("GLRMASK_COMPOSE_SELECTED_TEMPLATES_ONLY").is_none();
    let all_terminals = vec![true; analyzed.num_terminals as usize];
    let (eager_templates, ((mixed_owner, discovery_ms), (seed_relations, one_byte_ms))) =
        rayon::join(
            || {
                eager_all_templates.then(|| {
                    build_composition_templates(
                        &composed_table.table,
                        &analyzed,
                        &all_terminals,
                    )
                })
            },
            || {
                rayon::join(
                    || {
                        let started_at = Instant::now();
                        let mixed_owner = discover_mixed_owner_terminals(
                            vocab,
                            components,
                            tokenizer_state_offsets,
                            &composed_table.terminal_offsets,
                            &seed_terminals,
                            ignore_terminal,
                            &disallowed_follows,
                        );
                        (mixed_owner, started_at.elapsed().as_secs_f64() * 1000.0)
                    },
                    || {
                        let started_at = Instant::now();
                        let relations = collect_one_byte_seed_relations_components(
                            components,
                            tokenizer_state_offsets,
                            &composed_table.terminal_offsets,
                            vocab,
                            &seed_terminals,
                        );
                        if std::env::var_os(
                            "GLRMASK_VALIDATE_COMPOSE_COMPONENT_BOUNDARY_VIEW",
                        )
                        .is_some()
                        {
                            let mut reference = BTreeMap::<
                                Vec<u32>,
                                BTreeMap<u32, BTreeSet<u32>>,
                            >::new();
                            let tokenizer = merged_tokenizer.expect(
                                "component boundary-view validation requires a materialized tokenizer",
                            );
                            let all_states = (0..tokenizer.num_states()).collect::<Vec<_>>();
                            collect_one_byte_seed_relations(
                                tokenizer,
                                vocab,
                                &seed_terminals,
                                &all_states,
                                &mut reference,
                            );
                            assert_eq!(
                                relations, reference,
                                "component-view one-byte relation differs from merged tokenizer"
                            );
                            eprintln!(
                                "[glrmask/validate][compose_component_one_byte_view] relation_rows={} exact=true",
                                relations.len(),
                            );
                        }
                        (relations, started_at.elapsed().as_secs_f64() * 1000.0)
                    },
                )
            },
        );
    let mixed_owner_terminals = mixed_owner.terminals.clone();
    let mut active_terminals = seed_terminals.clone();
    for terminal in mixed_owner_terminals.iter() {
        active_terminals[terminal] = true;
    }
    if !active_terminals.iter().any(|&active| active) {
        if let Some(selected_boundary_tokens) = selected_boundary_tokens {
            let _ = selected_boundary_tokens.set(Ok(None));
        }
        return Ok(None);
    }
    if compose_profile_enabled() {
        let selected = active_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &active)| {
                active.then(|| {
                    format!(
                        "{}:{}",
                        terminal,
                        analyzed.terminal_display_name(terminal as u32),
                    )
                })
            })
            .collect::<Vec<_>>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminals] splice={} mixed={} mixed_tokens={} selected={:?}",
            composed_table.boundary_seed_terminals.len(),
            mixed_owner_terminals.count_ones(),
            mixed_owner.token_ids.len(),
            selected,
        );
    }

    let selected_original_tokens = seed_relations
        .values()
        .flat_map(|by_state| by_state.values())
        .flat_map(|tokens| tokens.iter().copied())
        .chain(mixed_owner.token_ids.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if selected_original_tokens.is_empty() {
        let error = "boundary witness construction selected no vocabulary tokens".to_string();
        if let Some(selected_boundary_tokens) = selected_boundary_tokens {
            let _ = selected_boundary_tokens.set(Err(error.clone()));
        }
        return Err(error);
    }
    if let Some(selected_boundary_tokens) = selected_boundary_tokens {
        let _ = selected_boundary_tokens.set(Ok(Some(selected_original_tokens)));
    }

    let owned_component_state_map = if precomputed_component_state_map.is_none()
        && deferred_component_state_map.is_none()
    {
        Some(component_state_coordinate_map(
            components,
            tokenizer_state_offsets,
            merged_tokenizer_state_count,
        )?)
    } else {
        None
    };
    let deferred_component_state_map = if let Some(deferred) = deferred_component_state_map {
        let prepared = loop {
            if let Some(prepared) = deferred.get() {
                break prepared;
            }
            std::thread::yield_now();
        };
        Some(prepared.as_ref().map_err(Clone::clone)?)
    } else {
        None
    };
    let component_state_map = precomputed_component_state_map
        .or(deferred_component_state_map)
        .or(owned_component_state_map.as_ref())
        .expect("component state map must be available");

    let ((templates, template_dfas_by_terminal, templates_ms), terminal_dwa) =
        rayon::join(
            || {
                let (templates, mut template_dfas_by_terminal, templates_ms) =
                    eager_templates.unwrap_or_else(|| {
                        build_composition_templates(
                            &composed_table.table,
                            &analyzed,
                            &active_terminals,
                        )
                    });
                if eager_all_templates {
                    for (terminal, slot) in template_dfas_by_terminal.iter_mut().enumerate() {
                        if !active_terminals[terminal] {
                            *slot = None;
                        }
                    }
                }
                (templates, template_dfas_by_terminal, templates_ms)
            },
            || {
                let started_at = Instant::now();
                let result = if std::env::var_os(
                    "GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE",
                )
                .is_some()
                {
                    match merged_tokenizer {
                        Some(tokenizer) => {
                            let flat_trans: Arc<[u32]> = Arc::from(
                                crate::compiler::stages::id_map_and_terminal_dwa::l1::
                                    build_flat_transition_table(tokenizer),
                            );
                            let global_max_length_state_map =
                                crate::compiler::stages::id_map_and_terminal_dwa::
                                    build_global_max_length_state_map(tokenizer, vocab, &flat_trans);
                            let coloring =
                                TerminalColoring::identity(analyzed.num_terminals as usize);
                            Ok(
                                crate::compiler::stages::id_map_and_terminal_dwa::
                                    build_restricted_id_map_and_terminal_dwa_with_precomputed_global_max_length(
                                        tokenizer,
                                        vocab,
                                        &coloring,
                                        false,
                                        ignore_terminal,
                                        &analyzed,
                                        &disallowed_follows,
                                        flat_trans,
                                        &global_max_length_state_map,
                                        None,
                                        Some(&active_terminals),
                                    )
                                    .0,
                            )
                        }
                        None => Err(
                            "generic boundary reference requires a materialized tokenizer"
                                .to_string(),
                        ),
                    }
                } else {
                    direct_boundary_terminal_automaton(
                        merged_tokenizer_state_count,
                        Some(component_state_map),
                        vocab,
                        seed_relations,
                        one_byte_ms,
                        &mixed_owner,
                        ignore_terminal,
                    )
                };
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
        );
    let (terminal_dwa, terminal_ms) = terminal_dwa;
    let terminal_dwa = terminal_dwa?;

    let parser_started_at = Instant::now();
    let (terminal_automaton, id_map) = terminal_dwa.into_parts();
    let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
        &composed_table.table,
        &analyzed,
        &terminal_automaton,
        &templates,
        vocab,
        &id_map,
        false,
    );
    let parser_ms = parser_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_build] active={} seed_active={} mixed_active={} mixed_tokens={} discovery_ms={discovery_ms:.3} one_byte_ms={one_byte_ms:.3} terminal_ms={terminal_ms:.3} templates_ms={templates_ms:.3} parser_ms={parser_ms:.3} total_ms={:.3}",
            active_terminals.iter().filter(|&&active| active).count(),
            seed_terminals.iter().filter(|&&active| active).count(),
            mixed_owner_terminals.count_ones(),
            mixed_owner.token_ids.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(Some(BoundaryRepair {
        parser_dwa: MappedArtifact::new(parser_dwa, id_map),
        template_dfas_by_terminal,
        active_terminals,
    }))
}

fn merged_terminal_display_names(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> Vec<String> {
    let mut names = parent.terminal_display_names.clone();
    for (index, child) in children.iter().enumerate() {
        names.extend(
            child
                .constraint
                .terminal_display_names
                .iter()
                .map(|name| format!("subgrammar{index}::{name}")),
        );
    }
    names
}

fn merged_original_token_ids(
    vocab: &Vocab,
    special_token_terminals: &[SpecialTokenTerminal],
) -> Vec<u32> {
    let extras = special_token_terminals
        .iter()
        .map(|special| special.token_id)
        .collect::<BTreeSet<_>>();
    let mut extras = extras.into_iter().peekable();
    let mut merged = Vec::with_capacity(vocab.entries_map().len() + extras.size_hint().0);
    for &token in vocab.entries_map().keys() {
        while extras.peek().is_some_and(|extra| *extra < token) {
            merged.push(extras.next().unwrap());
        }
        if extras.peek().is_some_and(|extra| *extra == token) {
            extras.next();
        }
        merged.push(token);
    }
    merged.extend(extras);
    merged
}

fn merged_special_token_terminals(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    table: &crate::compiler::glr::table::GLRTable,
) -> Vec<SpecialTokenTerminal> {
    fn consumes_terminal(action: &Action) -> bool {
        match action {
            Action::Reduce(_, _) => false,
            Action::Split { shift, accept, .. } => shift.is_some() || *accept,
            Action::Shift(_, _)
            | Action::StackShifts(_)
            | Action::GuardedStackShifts(_)
            | Action::Accept
            | Action::ReplaceShifts(_) => true,
        }
    }

    let mut merged = Vec::new();
    for (component_index, constraint) in std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        let terminal_offset = terminal_offsets[component_index];
        merged.extend(
            constraint
                .special_token_terminals
                .iter()
                .filter_map(|special| {
                    let terminal_id = terminal_offset + special.terminal_id;
                    // Placeholder terminals and placeholders retained inside an
                    // already-composed child are dead after table splicing. Do
                    // not keep their exact-token metadata: runtime special-token
                    // masking is driven by this list rather than the byte lexer.
                    table
                        .action
                        .iter()
                        .any(|row| row.get(&terminal_id).is_some_and(consumes_terminal))
                        .then_some(SpecialTokenTerminal {
                            terminal_id,
                            token_id: special.token_id,
                        })
                }),
        );
    }
    merged.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    merged.dedup_by_key(|special| (special.token_id, special.terminal_id));
    merged
}

fn merged_ignore_terminal(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
) -> Option<u32> {
    let ignores = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
        .filter_map(|(component_index, constraint)| {
            constraint
                .ignore_terminal
                .map(|terminal| terminal_offsets[component_index] + terminal)
        })
        .collect::<Vec<_>>();
    match ignores.as_slice() {
        [] => None,
        [ignore] => Some(*ignore),
        _ => panic!(
            "constraint composition currently supports at most one component ignore terminal; conflicting remapped ignore terminals: {ignores:?}"
        ),
    }
}

fn merged_terminal_live_states(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    num_terminals: usize,
) -> Vec<Vec<u32>> {
    let component_constraints = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    if !component_constraints.iter().all(|component| {
        component.terminal_live_states.len()
            == component.tokenizer.num_terminals() as usize
    }) {
        return Vec::new();
    }
    let mut merged = vec![Vec::<u32>::new(); num_terminals];
    for (component_index, component) in component_constraints.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        let state_offset = tokenizer_state_offsets[component_index];
        let local_start = component.tokenizer.start_state();
        for (local_terminal, states) in component.terminal_live_states.iter().enumerate() {
            let destination = &mut merged[terminal_offset + local_terminal];
            destination.extend(states.iter().map(|&state| state_offset + state));
            if states.binary_search(&local_start).is_ok() {
                destination.push(0);
            }
        }
    }
    for states in &mut merged {
        states.sort_unstable();
        states.dedup();
    }
    merged
}

fn merged_terminal_live_states_owned_parent(
    parent: &mut Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    num_terminals: usize,
) -> Vec<Vec<u32>> {
    if parent.terminal_live_states.len() != parent.tokenizer.num_terminals() as usize
        || children.iter().any(|child| {
            child.constraint.terminal_live_states.len()
                != child.constraint.tokenizer.num_terminals() as usize
        })
    {
        return Vec::new();
    }
    let mut merged = std::mem::take(&mut parent.terminal_live_states);
    merged.resize_with(num_terminals, Vec::new);
    for (child_index, child) in children.iter().enumerate() {
        let component_index = child_index + 1;
        let terminal_offset = terminal_offsets[component_index] as usize;
        let state_offset = tokenizer_state_offsets[component_index];
        let local_start = child.constraint.tokenizer.start_state();
        for (local_terminal, states) in child
            .constraint
            .terminal_live_states
            .iter()
            .enumerate()
        {
            let destination = &mut merged[terminal_offset + local_terminal];
            destination.extend(states.iter().map(|&state| state_offset + state));
            if states.binary_search(&local_start).is_ok() {
                destination.push(0);
            }
            destination.sort_unstable();
            destination.dedup();
        }
    }
    merged
}

fn build_composed_constraint_unfinalized(
    composed_table: ComposedTable,
    tokenizer: Tokenizer,
    tokenizer_state_offsets: Vec<u32>,
    parser_dwa: DWA,
    possible_matches: PossibleMatches,
    internal_ids: InternalIdMap,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    embedded_end_token_ids: Vec<u32>,
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<u32>,
    terminal_live_states: Vec<Vec<u32>>,
    tokenizer_fast_transitions: crate::runtime::FastTokenizerTransitions,
    defer_internal_token_bytes: bool,
    vocab: &Vocab,
) -> ConstraintComposition {
    let terminal_offsets = composed_table.terminal_offsets.clone();
    let parser_state_relations = composed_table.state_relations.clone();
    let InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids,
    } = internal_ids;
    debug_assert!(deferred_vocab_singleton_original_ids.is_none());
    let internal_token_bytes = if defer_internal_token_bytes {
        BTreeMap::new()
    } else {
        build_internal_token_bytes_from_groups(vocab, &vocab_tokens.internal_to_originals)
    };
    let ManyToOneIdMap {
        original_to_internal: state_to_internal_tsid,
        internal_to_originals: internal_tsid_to_states,
        representative_original_ids: _,
    } = tokenizer_states;
    debug_assert!(state_to_internal_tsid.iter().all(|&tsid| tsid != u32::MAX));
    // Direct component coordinates partition the composed raw tokenizer states:
    // each state has exactly one runtime TSID. Materialize the flat relation
    // directly instead of letting generic finalization allocate 1.4 million
    // temporary SmallVec rows and rediscover the same partition.
    // Sentinel `[u32::MAX]` means the relation is exactly the singleton
    // `state_to_internal_tsid` map. Runtime lookup already falls back to that
    // map; the sentinel prevents generic cache finalization from rebuilding two
    // redundant million-entry CSR vectors.
    let state_internal_tsid_offsets = vec![u32::MAX];
    let state_internal_tsids = Vec::new();
    let ManyToOneIdMap {
        original_to_internal: original_token_to_internal,
        internal_to_originals: internal_token_to_tokens,
        representative_original_ids: _,
    } = vocab_tokens;
    let tokenizer_has_epsilon_transitions = tokenizer.has_epsilon_transitions();
    let mut table = composed_table.table;
    table.set_embedded_end_token_ids(&embedded_end_token_ids);
    let num_terminals = table.num_terminals as usize;
    let constraint = Constraint {
        runtime_backend: ConstraintRuntimeBackend::Static,
        parser_dwa,
        parser_top_accept: BTreeMap::new(),
        parser_top_accept_parts: BTreeMap::new(),
        direct_regular_l1_complete_by_terminal: BTreeMap::new(),
        direct_regular_wide_frontier_acceptance: Vec::new(),
        direct_regular_dynamic_hot_frontiers: Vec::new(),
        direct_regular_parser_state_acceptance: Vec::new(),
        direct_regular_automaton: None,
        table,
        terminal_display_names,
        tokenizer,
        tokenizer_has_epsilon_transitions,
        ignore_terminal,
        special_token_terminals,
        dynamic_mask_vocab: runtime_dynamic_vocab_for_vocab(vocab),
        lazy_dynamic_mask_vocab: OnceLock::new(),
        // Constraint composition is not allowed to rely on the legacy dynamic
        // possible-matches fallback. This is the exact transported and
        // reconciled table from every compiled component.
        // DO NOT REMOVE OR WEAKEN THIS COMMENT.
        possible_matches,
        possible_matches_complete: true,
        state_to_internal_tsid,
        internal_tsid_to_states,
        terminal_live_states,
        state_internal_tsid_offsets,
        state_internal_tsids,
        runtime_source_state_offset: None,
        runtime_product_source_offsets: Vec::new(),
        runtime_product_source_states: Vec::new(),
        runtime_product_exact_source_states: Vec::new(),
        runtime_product_state_by_source_subset: FxHashMap::default(),
        template_dfas_by_terminal,
        fast_template_dfas_by_terminal: Vec::new(),
        original_token_to_internal,
        internal_token_to_tokens,
        token_bytes: vocab.entries_arc(),
        internal_token_bytes,
        token_bytes_dense: Vec::new(),
        internal_token_buf_masks: Vec::new(),
        word_group_buf_masks: Vec::new(),
        pair_word_group_buf_masks: Vec::new(),
        quad_word_group_buf_masks: Vec::new(),
        super_word_group_buf_masks: Vec::new(),
        mega_word_group_buf_masks: Vec::new(),
        giga_word_group_buf_masks: Vec::new(),
        word_group_sparse_masks: Vec::new(),
        word_group_prefix_buf_masks: Vec::new(),
        word_group_sparse_prefix_entries: Vec::new(),
        quad_group_sparse_masks: Vec::new(),
        quad_group_dense_masks: Vec::new(),
        byte_group_sparse_masks: Vec::new(),
        byte_group_dense_masks: Vec::new(),
        word_group_sparse_total_entries: 0,
        word_group_sparse_max_entries: 0,
        all_tokens_buf_mask: Box::new([]),
        internal_token_dense_words: 0,
        weight_token_dense_masks: FxHashMap::default(),
        weight_token_buf_masks: FxHashMap::default(),
        weight_token_sparse_buf_masks: FxHashMap::default(),
        direct_sparse_weight_token_sets: FxHashSet::default(),
        seed_terminal_dense: FxHashMap::default(),
        seed_terminal_dense_fallback: Default::default(),
        seed_universe_dense: Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice()),
        dwa_fast_transitions: Vec::new(),
        indexed_dag_dense_transitions: Vec::new(),
        indexed_dag_dense_finals: Vec::new(),
        tokenizer_fast_transitions,
        heavy_token_dense_masks: Vec::new(),
        heavy_token_indices: Vec::new(),
        internal_token_buf_flat: Box::new([]),
        internal_token_buf_offsets: Box::new([]),
        total_internal_buf_cost: 0,
        heavy_total_cost: 0,
        light_avg_cost_x256: 0,
        internal_token_buf_op_costs: Vec::new(),
        word_group_buf_op_costs: Vec::new(),
        final_mask_mapping: crate::runtime::mask_mapping::FinalMaskMapping::default(),
    };
    ConstraintComposition {
        constraint,
        terminal_offsets,
        tokenizer_state_offsets,
        parser_state_relations,
    }
}

fn finalize_composed_constraint(
    composed_table: ComposedTable,
    tokenizer: Tokenizer,
    tokenizer_state_offsets: Vec<u32>,
    parser_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    embedded_end_token_ids: Vec<u32>,
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<u32>,
    terminal_live_states: Vec<Vec<u32>>,
    tokenizer_fast_transitions: crate::runtime::FastTokenizerTransitions,
    vocab: &Vocab,
) -> ConstraintComposition {
    let ((parser_dwa, possible_matches), internal_ids) = parser_artifacts.into_parts();
    let mut composition = build_composed_constraint_unfinalized(
        composed_table,
        tokenizer,
        tokenizer_state_offsets,
        parser_dwa,
        possible_matches,
        internal_ids,
        template_dfas_by_terminal,
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        ignore_terminal,
        terminal_live_states,
        tokenizer_fast_transitions,
        false,
        vocab,
    );
    composition.constraint.rebuild_runtime_caches();
    composition
}

/// Compose already-compiled parent and child constraints. The component
/// lexers, parse tables, parser DWAs, and possible-match tables are transported
/// and reused; only the restricted cross-component boundary repair is compiled
/// from the merged artifacts.
pub(crate) fn compose_constraints(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    let total_started_at = Instant::now();
    if children.is_empty() {
        return Err("constraint composition requires at least one child".into());
    }
    let component_end_token_ids = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .flat_map(|constraint| constraint.table.embedded_end_token_ids())
        .collect::<BTreeSet<_>>();
    let vocab_entries = vocab.entries_arc();
    for (component_index, constraint) in std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        if !Arc::ptr_eq(&constraint.token_bytes, &vocab_entries)
            && constraint.token_bytes.as_ref() != vocab.entries_map()
        {
            return Err(format!(
                "component {component_index} was not compiled for the supplied vocabulary",
            ));
        }
    }
    for (child_index, child) in children.iter().enumerate() {
        let has_byte_token_matches = parent
            .possible_matches
            .get(&child.placeholder_terminal)
            .is_some_and(|weight| !weight.is_empty());
        let has_exact_token_match = parent.special_token_terminals.iter().any(|special| {
            special.terminal_id == child.placeholder_terminal
                && vocab.entries_map().contains_key(&special.token_id)
        });
        if has_byte_token_matches || has_exact_token_match
        {
            return Err(format!(
                "subgrammar placeholder terminal {} for child {child_index} matches one or more model-vocabulary tokens in the compiled parent; placeholders must be non-vocabulary sentinels (for example an @token(...) id outside the supplied vocabulary)",
                child.placeholder_terminal,
            ));
        }
        for special in parent
            .special_token_terminals
            .iter()
            .filter(|special| special.terminal_id == child.placeholder_terminal)
        {
            if component_end_token_ids.contains(&special.token_id) {
                return Err(format!(
                    "subgrammar placeholder terminal {} for child {child_index} uses token ID {}, which is also configured as a grammar-level end token; every replaced placeholder must use a unique sentinel token ID",
                    child.placeholder_terminal, special.token_id,
                ));
            }
        }
    }
    let table_inputs = children
        .iter()
        .map(|child| SubgrammarTableInput {
            placeholder_terminal: child.placeholder_terminal,
            table: &child.constraint.table,
            start_nullable: child.constraint.table.embedded_start_nullable(),
        })
        .collect::<Vec<_>>();
    let table_started_at = Instant::now();
    let composed_table = compose_subgrammar_tables(&parent.table, &table_inputs)?;
    let table_ms = table_started_at.elapsed().as_secs_f64() * 1000.0;

    let component_constraints = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let tokenizer_inputs = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| {
            (&constraint.tokenizer, composed_table.terminal_offsets[index])
        })
        .collect::<Vec<_>>();
    let (expected_tokenizer_state_offsets, merged_tokenizer_state_count) =
        component_tokenizer_state_layout(&component_constraints);

    let parser_components = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| ParserDwaComponent {
            constraint,
            parser_state_relation: &composed_table.state_relations[index],
            tokenizer_state_offset: expected_tokenizer_state_offsets[index],
        })
        .collect::<Vec<_>>();
    let special_token_terminals = merged_special_token_terminals(
        parent,
        children,
        &composed_table.terminal_offsets,
        &composed_table.table,
    );
    let live_special_token_ids = special_token_terminals
        .iter()
        .map(|special| special.token_id)
        .collect::<BTreeSet<_>>();
    let embedded_end_token_ids = component_end_token_ids
        .intersection(&live_special_token_ids)
        .copied()
        .collect::<Vec<_>>();
    for child in children {
        for special in parent
            .special_token_terminals
            .iter()
            .filter(|special| special.terminal_id == child.placeholder_terminal)
        {
            if live_special_token_ids.contains(&special.token_id) {
                return Err(format!(
                    "subgrammar placeholder terminal {} uses token ID {}, which is also used by a live special terminal after composition; every replaced placeholder must use a unique sentinel token ID",
                    child.placeholder_terminal, special.token_id,
                ));
            }
        }
    }
    let original_token_ids = vocab
        .entries_map()
        .keys()
        .copied()
        .chain(
            special_token_terminals
                .iter()
                .map(|special| special.token_id),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let terminal_display_names = merged_terminal_display_names(parent, children);
    let ignore_terminal = merged_ignore_terminal(
        parent,
        children,
        &composed_table.terminal_offsets,
    );

    let needs_materialized_boundary_reference =
        std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE").is_some()
            || std::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_BOUNDARY_VIEW").is_some();

    let (
        (tokenizer, tokenizer_state_offsets),
        tokenizer_ms,
        (parser_artifacts, reuse_ms),
        (boundary_repair, boundary_ms),
    ) = if needs_materialized_boundary_reference {
        let tokenizer_started_at = Instant::now();
        let tokenizer_result =
            Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
        let tokenizer_ms = tokenizer_started_at.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(
            tokenizer_result.1, expected_tokenizer_state_offsets,
            "predicted composed tokenizer state offsets differ from materialized union",
        );
        if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_BOUNDARY_VIEW").is_some() {
            let component_resets = expanded_component_reset_states(
                &component_constraints,
                &tokenizer_result.1,
            );
            let merged_resets = tokenizer_result.0.deterministic_reset_states().to_vec();
            let expanded_exact = component_resets == merged_resets;
            let dispatcher_exact = merged_resets.as_slice()
                == [tokenizer_result.0.initial_state_id()]
                && component_resets.iter().all(|state| {
                    tokenizer_result
                        .0
                        .singleton_epsilon_closure(tokenizer_result.0.initial_state_id())
                        .contains(state)
                });
            assert!(
                expanded_exact || dispatcher_exact,
                "component reset view differs from merged tokenizer: component={component_resets:?} merged={merged_resets:?}",
            );
            eprintln!(
                "[glrmask/validate][compose_component_reset_view] count={} representation={} exact=true",
                component_resets.len(),
                if expanded_exact { "expanded" } else { "dispatcher" },
            );
        }
        let ((parser_artifacts, reuse_ms), (boundary_repair, boundary_ms)) = rayon::join(
            || {
                let started_at = Instant::now();
                let result = compose_component_parser_dwas_and_possible_matches(
                    &parser_components,
                    &composed_table.terminal_offsets,
                    merged_tokenizer_state_count,
                    &original_token_ids,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
            || {
                let started_at = Instant::now();
                let result = build_boundary_repair(
                    &composed_table,
                    Some(&tokenizer_result.0),
                    merged_tokenizer_state_count,
                    terminal_display_names.clone(),
                    ignore_terminal,
                    vocab,
                    &component_constraints,
                    &expected_tokenizer_state_offsets,
                    None,
                    None,
                    None,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
        );
        (
            tokenizer_result,
            tokenizer_ms,
            (parser_artifacts, reuse_ms),
            (boundary_repair, boundary_ms),
        )
    } else {
        // These three jobs are independent when boundary analysis uses the
        // exact component-tokenizer view. Materialize the flattened tokenizer,
        // transport the component parser artifacts, and compile cross-boundary
        // behavior concurrently.
        let ((tokenizer_result, tokenizer_ms), ((parser_artifacts, reuse_ms), (boundary_repair, boundary_ms))) =
            rayon::join(
                || {
                    let started_at = Instant::now();
                    let result =
                        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
                    (result, started_at.elapsed().as_secs_f64() * 1000.0)
                },
                || {
                    rayon::join(
                        || {
                            let started_at = Instant::now();
                            let result = compose_component_parser_dwas_and_possible_matches(
                                &parser_components,
                                &composed_table.terminal_offsets,
                                merged_tokenizer_state_count,
                                &original_token_ids,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                        || {
                            let started_at = Instant::now();
                            let result = build_boundary_repair(
                                &composed_table,
                                None,
                                merged_tokenizer_state_count,
                                terminal_display_names.clone(),
                                ignore_terminal,
                                vocab,
                                &component_constraints,
                                &expected_tokenizer_state_offsets,
                                None,
                                None,
                                None,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                    )
                },
            );
        assert_eq!(
            tokenizer_result.1, expected_tokenizer_state_offsets,
            "predicted composed tokenizer state offsets differ from materialized union",
        );
        (
            tokenizer_result,
            tokenizer_ms,
            (parser_artifacts, reuse_ms),
            (boundary_repair, boundary_ms),
        )
    };
    let parser_artifacts = parser_artifacts?;
    let boundary_repair = boundary_repair?;
    let union_started_at = Instant::now();
    let (parser_union_result, closure_prime_ms) = rayon::join(
        || -> Result<_, String> {
            Ok(match boundary_repair {
                Some(boundary) => {
                    debug_assert!(boundary.active_terminals.iter().any(|&active| active));
                    (
                        union_boundary_parser_dwa(
                            parser_artifacts,
                            boundary.parser_dwa,
                            composed_table.table.num_states,
                        )?,
                        boundary.template_dfas_by_terminal,
                    )
                }
                None => (
                    parser_artifacts,
                    vec![None; composed_table.table.num_terminals as usize],
                ),
            })
        },
        || {
            // The flattened tokenizer's dense singleton-closure cache is
            // independent of parser-DWA union. Prime it here so its allocation
            // latency is hidden by union/determinization instead of being paid
            // serially during runtime-cache finalization.
            let started_at = Instant::now();
            drop(tokenizer.all_singleton_epsilon_closures());
            started_at.elapsed().as_secs_f64() * 1000.0
        },
    );
    let (parser_artifacts, template_dfas_by_terminal) = parser_union_result?;
    let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    let terminal_live_states = merged_terminal_live_states(
        parent,
        children,
        &composed_table.terminal_offsets,
        &tokenizer_state_offsets,
        composed_table.table.num_terminals as usize,
    );
    let result = finalize_composed_constraint(
        composed_table,
        tokenizer,
        tokenizer_state_offsets,
        parser_artifacts,
        template_dfas_by_terminal,
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        ignore_terminal,
        terminal_live_states,
        Default::default(),
        vocab,
    );
    let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition] components={} table_ms={table_ms:.3} tokenizer_ms={tokenizer_ms:.3} reuse_ms={reuse_ms:.3} boundary_ms={boundary_ms:.3} union_ms={union_ms:.3} closure_prime_ms={closure_prime_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
            children.len() + 1,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(result)
}

/// Fast consuming composition path. The parent remains the logical and physical
/// base of the returned ordinary `Constraint`; child tokenizer states are
/// appended to it, so the million-state parent is neither cloned nor rebased.
pub(crate) fn compose_constraints_owned_parent(
    mut parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    if std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE").is_some()
        || std::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_BOUNDARY_VIEW").is_some()
    {
        return compose_constraints(&parent, children, vocab);
    }
    let total_started_at = Instant::now();
    if children.is_empty() {
        return Err("constraint composition requires at least one child".into());
    }
    let component_end_token_ids = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .flat_map(|constraint| constraint.table.embedded_end_token_ids())
        .collect::<BTreeSet<_>>();
    let vocab_entries = vocab.entries_arc();
    for (component_index, constraint) in std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        if !Arc::ptr_eq(&constraint.token_bytes, &vocab_entries)
            && constraint.token_bytes.as_ref() != vocab.entries_map()
        {
            return Err(format!(
                "component {component_index} was not compiled for the supplied vocabulary",
            ));
        }
    }
    for (child_index, child) in children.iter().enumerate() {
        let has_byte_token_matches = parent
            .possible_matches
            .get(&child.placeholder_terminal)
            .is_some_and(|weight| !weight.is_empty());
        let has_exact_token_match = parent.special_token_terminals.iter().any(|special| {
            special.terminal_id == child.placeholder_terminal
                && vocab.entries_map().contains_key(&special.token_id)
        });
        if has_byte_token_matches || has_exact_token_match {
            return Err(format!(
                "subgrammar placeholder terminal {} for child {child_index} matches one or more model-vocabulary tokens in the compiled parent; placeholders must be non-vocabulary sentinels (for example an @token(...) id outside the supplied vocabulary)",
                child.placeholder_terminal,
            ));
        }
        for special in parent
            .special_token_terminals
            .iter()
            .filter(|special| special.terminal_id == child.placeholder_terminal)
        {
            if component_end_token_ids.contains(&special.token_id) {
                return Err(format!(
                    "subgrammar placeholder terminal {} for child {child_index} uses token ID {}, which is also configured as a grammar-level end token; every replaced placeholder must use a unique sentinel token ID",
                    child.placeholder_terminal, special.token_id,
                ));
            }
        }
    }

    let table_inputs = children
        .iter()
        .map(|child| SubgrammarTableInput {
            placeholder_terminal: child.placeholder_terminal,
            table: &child.constraint.table,
            start_nullable: child.constraint.table.embedded_start_nullable(),
        })
        .collect::<Vec<_>>();
    let table_started_at = Instant::now();
    let composed_table = compose_subgrammar_tables(&parent.table, &table_inputs)?;
    let table_ms = table_started_at.elapsed().as_secs_f64() * 1000.0;

    let metadata_started_at = Instant::now();
    let component_constraints = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let (expected_tokenizer_state_offsets, merged_tokenizer_state_count) =
        component_tokenizer_state_layout_owned_parent(&component_constraints);
    let parser_components = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| ParserDwaComponent {
            constraint,
            parser_state_relation: &composed_table.state_relations[index],
            tokenizer_state_offset: expected_tokenizer_state_offsets[index],
        })
        .collect::<Vec<_>>();
    let component_views_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;
    let specials_started_at = Instant::now();
    let special_token_terminals = merged_special_token_terminals(
        &parent,
        children,
        &composed_table.terminal_offsets,
        &composed_table.table,
    );
    let live_special_token_ids = special_token_terminals
        .iter()
        .map(|special| special.token_id)
        .collect::<BTreeSet<_>>();
    let embedded_end_token_ids = component_end_token_ids
        .intersection(&live_special_token_ids)
        .copied()
        .collect::<Vec<_>>();
    for child in children {
        for special in parent
            .special_token_terminals
            .iter()
            .filter(|special| special.terminal_id == child.placeholder_terminal)
        {
            if live_special_token_ids.contains(&special.token_id) {
                return Err(format!(
                    "subgrammar placeholder terminal {} uses token ID {}, which is also used by a live special terminal after composition; every replaced placeholder must use a unique sentinel token ID",
                    child.placeholder_terminal, special.token_id,
                ));
            }
        }
    }
    let specials_ms = specials_started_at.elapsed().as_secs_f64() * 1000.0;
    let token_ids_started_at = Instant::now();
    let original_token_ids = merged_original_token_ids(vocab, &special_token_terminals);
    let token_ids_ms = token_ids_started_at.elapsed().as_secs_f64() * 1000.0;
    let names_started_at = Instant::now();
    let terminal_display_names = merged_terminal_display_names(&parent, children);
    let ignore_terminal = merged_ignore_terminal(
        &parent,
        children,
        &composed_table.terminal_offsets,
    );
    let names_ms = names_started_at.elapsed().as_secs_f64() * 1000.0;
    let metadata_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_metadata] component_views_ms={component_views_ms:.3} specials_ms={specials_ms:.3} token_ids_ms={token_ids_ms:.3} names_ms={names_ms:.3} total_ms={metadata_ms:.3}",
        );
    }
    // Publish state coordinates and the exact boundary-token coordinate as
    // soon as each becomes available. Component weight remapping then overlaps
    // boundary template/terminal/parser construction instead of sitting on the
    // serial parser-union path.
    let state_map_cell = OnceLock::<Result<ManyToOneIdMap, String>>::new();
    let selected_boundary_tokens_cell =
        OnceLock::<Result<Option<Vec<u32>>, String>>::new();
    let preparation_started_at = Instant::now();
    let (prepared_components_result, (boundary_result, boundary_ms)) = rayon::join(
            || {
                let state_started_at = Instant::now();
                let state_result = build_direct_component_state_coordinates(
                    &parser_components,
                    merged_tokenizer_state_count,
                );
                let component_state_ms = state_started_at.elapsed().as_secs_f64() * 1000.0;
                let published_state_map = state_result
                    .as_ref()
                    .map(|coordinates| coordinates.tokenizer_states.clone())
                    .map_err(Clone::clone);
                assert!(
                    state_map_cell.set(published_state_map).is_ok(),
                    "component state map published twice",
                );
                let state_coordinates = state_result?;
                let ((token_coordinate_result, token_coordinate_ms), (unmapped_result, parser_extract_ms)) =
                    rayon::join(
                        || {
                            let started_at = Instant::now();
                            let result = build_direct_component_token_coordinates(
                                &parser_components,
                                &original_token_ids,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                        || {
                            let started_at = Instant::now();
                            let result = prepare_unmapped_component_parser_artifacts(
                                &parser_components,
                                &composed_table.terminal_offsets,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                    );
                let (vocab_tokens, local_to_global_tokens) = token_coordinate_result?;
                let component_maps = state_coordinates
                    .local_to_global_tsids
                    .into_iter()
                    .zip(local_to_global_tokens)
                    .map(|(local_to_global_tsids, local_to_global_tokens)| {
                        DirectComponentCoordinateMaps {
                            local_to_global_tsids,
                            local_to_global_tokens,
                        }
                    })
                    .collect::<Vec<_>>();
                let component_id_map = InternalIdMap {
                    tokenizer_states: state_coordinates.tokenizer_states,
                    vocab_tokens,
                    deferred_vocab_singleton_original_ids: None,
                };
                let selected_boundary_tokens = loop {
                    if let Some(selected) = selected_boundary_tokens_cell.get() {
                        break selected.as_ref().map_err(Clone::clone)?.clone();
                    }
                    std::thread::yield_now();
                };
                let unmapped_components = unmapped_result?;
                let prepared = if let Some(selected_boundary_tokens) = selected_boundary_tokens {
                    let boundary_id_map = boundary_id_map_for_selected_tokens(
                        &component_id_map.tokenizer_states,
                        &selected_boundary_tokens,
                        vocab,
                    )?;
                    let plan = build_boundary_refinement_plan(component_id_map, &boundary_id_map)
                        .ok_or_else(|| {
                            "component coordinate map does not cover boundary repair".to_string()
                        })?;
                    let (automata, possible_matches, remap_ms) =
                        remap_unmapped_component_artifacts(
                            unmapped_components,
                            component_maps,
                            Some(&plan.component_token_map),
                            plan.common_map.num_tsids() as usize,
                        )?;
                    PreparedOwnedComponentArtifacts {
                        automata,
                        possible_matches,
                        id_map: plan.common_map,
                        boundary_tsid_map: Some(plan.boundary_tsid_map),
                        boundary_token_map: Some(plan.boundary_token_map),
                        remap_ms,
                    }
                } else {
                    let (automata, possible_matches, remap_ms) =
                        remap_unmapped_component_artifacts(
                            unmapped_components,
                            component_maps,
                            None,
                            component_id_map.num_tsids() as usize,
                        )?;
                    PreparedOwnedComponentArtifacts {
                        automata,
                        possible_matches,
                        id_map: component_id_map,
                        boundary_tsid_map: None,
                        boundary_token_map: None,
                        remap_ms,
                    }
                };
                Ok::<_, String>((
                    prepared,
                    component_state_ms + token_coordinate_ms,
                    parser_extract_ms,
                ))
            },
            || {
                let started_at = Instant::now();
                let result = build_boundary_repair(
                    &composed_table,
                    None,
                    merged_tokenizer_state_count,
                    terminal_display_names.clone(),
                    ignore_terminal,
                    vocab,
                    &component_constraints,
                    &expected_tokenizer_state_offsets,
                    None,
                    Some(&state_map_cell),
                    Some(&selected_boundary_tokens_cell),
                );
                if selected_boundary_tokens_cell.get().is_none() {
                    let publication = match &result {
                        Err(error) => Err(error.clone()),
                        Ok(None) => Ok(None),
                        Ok(Some(_)) => Err(
                            "boundary repair completed without publishing selected tokens"
                                .to_string(),
                        ),
                    };
                    let _ = selected_boundary_tokens_cell.set(publication);
                }
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
        );
    let (prepared_components, coordinate_ms, parser_extract_ms) = prepared_components_result?;
    let boundary_repair = boundary_result?;
    let preparation_ms = preparation_started_at.elapsed().as_secs_f64() * 1000.0;

    let terminal_live_started_at = Instant::now();
    let terminal_live_states = merged_terminal_live_states_owned_parent(
        &mut parent,
        children,
        &composed_table.terminal_offsets,
        &expected_tokenizer_state_offsets,
        composed_table.table.num_terminals as usize,
    );
    let terminal_live_ms = terminal_live_started_at.elapsed().as_secs_f64() * 1000.0;

    let child_tokenizers = children
        .iter()
        .enumerate()
        .map(|(index, child)| {
            (&child.constraint.tokenizer, composed_table.terminal_offsets[index + 1])
        })
        .collect::<Vec<_>>();
    let tokenizer_started_at = Instant::now();
    let parent_fast_transitions = std::mem::take(&mut parent.tokenizer_fast_transitions);
    let child_fast_transitions = children
        .iter()
        .enumerate()
        .map(|(index, child)| {
            (
                &child.constraint.tokenizer_fast_transitions,
                expected_tokenizer_state_offsets[index + 1],
            )
        })
        .collect::<Vec<_>>();
    let tokenizer_fast_transitions = parent_fast_transitions
        .append_rebased_children(&child_fast_transitions)
        .unwrap_or_default();
    let parent_tokenizer = std::mem::replace(
        &mut parent.tokenizer,
        Tokenizer::disjoint_union_with_terminal_offsets(&[]).0,
    );
    let (tokenizer, tokenizer_state_offsets) =
        Tokenizer::disjoint_union_with_owned_parent(
            parent_tokenizer,
            composed_table.terminal_offsets[0],
            &child_tokenizers,
        );
    let tokenizer_ms = tokenizer_started_at.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        tokenizer_state_offsets, expected_tokenizer_state_offsets,
        "owned-parent tokenizer state offsets differ from predicted layout",
    );


    let num_parser_states = composed_table.table.num_states;
    let num_terminals = composed_table.table.num_terminals as usize;
    let PreparedOwnedComponentArtifacts {
        automata,
        possible_matches,
        id_map,
        boundary_tsid_map,
        boundary_token_map,
        remap_ms: component_remap_ms,
    } = prepared_components;
    let id_num_tsids = id_map.num_tsids();
    let id_max_internal_token = id_map.max_internal_token_id();
    let (boundary_work, template_dfas_by_terminal) = match boundary_repair {
        Some(boundary) => {
            debug_assert!(boundary.active_terminals.iter().any(|&active| active));
            let (boundary_dwa, boundary_id_map) = boundary.parser_dwa.into_parts();
            (
                Some((boundary_dwa, boundary_id_map)),
                boundary.template_dfas_by_terminal,
            )
        }
        None => {
            if boundary_tsid_map.is_some() || boundary_token_map.is_some() {
                return Err(
                    "prepared component artifacts retained boundary maps without a boundary repair"
                        .to_string(),
                );
            }
            (None, vec![None; num_terminals])
        }
    };

    let mut result = build_composed_constraint_unfinalized(
        composed_table,
        tokenizer,
        tokenizer_state_offsets,
        DWA::new(id_num_tsids, id_max_internal_token),
        possible_matches,
        id_map,
        template_dfas_by_terminal,
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        ignore_terminal,
        terminal_live_states,
        tokenizer_fast_transitions,
        true,
        vocab,
    );

    let union_started_at = Instant::now();
    let (parser_union_result, token_cache_prebuild_ms) = rayon::join(
        || -> Result<DWA, String> {
            let final_build_started_at = Instant::now();
            let mut automata = automata;
            match boundary_work {
                Some((mut boundary_dwa, boundary_id_map)) => {
                    let boundary_tsid_map = boundary_tsid_map.ok_or_else(|| {
                        "prepared component artifacts omitted boundary TSID mapping".to_string()
                    })?;
                    let boundary_token_map = boundary_token_map.ok_or_else(|| {
                        "prepared component artifacts omitted boundary token mapping".to_string()
                    })?;
                    if boundary_tsid_map.len() != boundary_id_map.num_tsids() as usize
                        || boundary_token_map.len()
                            != boundary_id_map.num_internal_tokens() as usize
                    {
                        return Err(
                            "published boundary coordinate differs from compiled boundary artifact"
                                .to_string(),
                        );
                    }
                    let mut boundary_weights = boundary_dwa.weight_refs_mut();
                    remap_weights_with_maps(
                        &mut boundary_weights,
                        &boundary_tsid_map,
                        &boundary_token_map,
                        id_num_tsids as usize,
                    );
                    drop(boundary_weights);
                    automata.push(parser_nwa_preserve_defaults(&boundary_dwa));
                    let automata_len = automata.len();
                    let validation_automata = std::env::var_os(
                        "GLRMASK_VALIDATE_COMPOSE_SINGLE_PASS_UNION",
                    )
                    .is_some()
                    .then(|| automata.clone());
                    let direct_started_at = Instant::now();
                    let direct_supported = supports_overlap_local_union(&automata);
                    let (parser_dwa, synthetic_states, union_path) = if direct_supported {
                        let (dwa, synthetic_states) =
                            determinize_epsilon_free_component_union(
                                automata,
                                Some(num_parser_states),
                            )
                            .expect("overlap-local union support was prechecked");
                        (dwa, synthetic_states, "overlap_local")
                    } else {
                        let mut reference =
                            NWA::new(id_num_tsids, id_max_internal_token);
                        let mut starts = Vec::new();
                        let last = automata.len().saturating_sub(1);
                        for (index, automaton) in automata.iter().enumerate() {
                            let explicit = if index == last {
                                explicit_parser_nwa(&boundary_dwa, num_parser_states)
                            } else {
                                automaton.clone()
                            };
                            let body = reference.append_with_body(&explicit);
                            starts.extend(body.start_states);
                        }
                        reference.set_start_states(starts);
                        (
                            determinize(&reference).map_err(|error| error.to_string())?,
                            0,
                            "generic_fallback",
                        )
                    };
                    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
                    if let Some(automata) = validation_automata {
                        let mut reference =
                            NWA::new(id_num_tsids, id_max_internal_token);
                        let mut starts = Vec::new();
                        for (index, automaton) in automata.iter().enumerate() {
                            let explicit = if index + 1 == automata.len() {
                                explicit_parser_nwa(&boundary_dwa, num_parser_states)
                            } else {
                                automaton.clone()
                            };
                            let body = reference.append_with_body(&explicit);
                            starts.extend(body.start_states);
                        }
                        reference.set_start_states(starts);
                        let generic = determinize(&reference).map_err(|error| error.to_string())?;
                        assert_eq!(
                            find_difference(&parser_dwa, &generic)
                                .map_err(|error| error.to_string())?,
                            None,
                            "single-pass component/boundary union differs from generic determinization",
                        );
                    }
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_single_pass_parser_union] path={} automata={} component_remap_ms={component_remap_ms:.3} direct_ms={direct_ms:.3} synthetic_states={} result_states={} total_ms={:.3}",
                            union_path,
                            automata_len,
                            synthetic_states,
                            parser_dwa.num_states(),
                            final_build_started_at.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    Ok(parser_dwa)
                }
                None => {
                    let automata_len = automata.len();
                    let direct_started_at = Instant::now();
                    let direct_supported = supports_overlap_local_union(&automata);
                    let (parser_dwa, synthetic_states, union_path) = if direct_supported {
                        let (dwa, synthetic_states) =
                            determinize_epsilon_free_component_union(automata, None)
                                .expect("overlap-local union support was prechecked");
                        (dwa, synthetic_states, "overlap_local")
                    } else {
                        let mut reference =
                            NWA::new(id_num_tsids, id_max_internal_token);
                        let mut starts = Vec::new();
                        for automaton in &automata {
                            let body = reference.append_with_body(automaton);
                            starts.extend(body.start_states);
                        }
                        reference.set_start_states(starts);
                        (
                            determinize(&reference).map_err(|error| error.to_string())?,
                            0,
                            "generic_fallback",
                        )
                    };
                    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_single_pass_parser_union] path={} automata={} component_remap_ms={component_remap_ms:.3} direct_ms={direct_ms:.3} synthetic_states={} result_states={} total_ms={:.3}",
                            union_path,
                            automata_len,
                            synthetic_states,
                            parser_dwa.num_states(),
                            final_build_started_at.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    Ok(parser_dwa)
                }
            }
        },
        || {
            let started_at = Instant::now();
            result.constraint.internal_token_bytes = build_internal_token_bytes_from_groups(
                vocab,
                &result.constraint.internal_token_to_tokens,
            );
            result.constraint.prebuild_token_mask_caches();
            started_at.elapsed().as_secs_f64() * 1000.0
        },
    );
    result.constraint.parser_dwa = parser_union_result?;
    let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    result.constraint.rebuild_runtime_caches();
    let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_owned_parent] components={} table_ms={table_ms:.3} tokenizer_ms={tokenizer_ms:.3} coordinate_ms={coordinate_ms:.3} parser_extract_ms={parser_extract_ms:.3} boundary_ms={boundary_ms:.3} preparation_ms={preparation_ms:.3} terminal_live_ms={terminal_live_ms:.3} union_ms={union_ms:.3} token_cache_prebuild_ms={token_cache_prebuild_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
            children.len() + 1,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::accumulator::TerminalsDisallowed;
    use crate::compiler::glr::parser::{ParserGSS, advance_stacks, stacks_finished};
    use crate::compiler::glr::table::{
        SubgrammarTableInput, compose_subgrammar_tables,
    };
    use crate::grammar::flat::TerminalID;

    fn byte_vocab() -> Vocab {
        Vocab::new(
            (0u32..=255)
                .map(|byte| (byte, vec![byte as u8]))
                .collect(),
        )
    }

    fn terminal(constraint: &Constraint, name: &str) -> TerminalID {
        constraint
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap() as u32
    }

    #[test]
    fn overlap_local_union_preserves_symbolic_default_semantics() {
        let mut wildcard = NWA::new(1, 0);
        let wildcard_start = wildcard.add_state();
        let wildcard_final = wildcard.add_state();
        wildcard.set_start_states(vec![wildcard_start]);
        wildcard.add_transition(
            wildcard_start,
            DEFAULT_LABEL,
            wildcard_final,
            Weight::all(),
        );
        wildcard.set_final_weight(wildcard_final, Weight::all());

        let mut explicit = NWA::new(1, 0);
        let explicit_start = explicit.add_state();
        let explicit_final = explicit.add_state();
        explicit.set_start_states(vec![explicit_start]);
        explicit.add_transition(
            explicit_start,
            encode_positive_label(3),
            explicit_final,
            Weight::all(),
        );
        explicit.add_transition(
            explicit_start,
            encode_negative_label(5),
            explicit_final,
            Weight::all(),
        );
        explicit.set_final_weight(explicit_final, Weight::all());

        let (union, _) = determinize_epsilon_free_component_union(
            vec![wildcard, explicit],
            Some(8),
        )
        .expect("epsilon-free acyclic union must use overlap-local path");
        let evaluate_runtime_label = |label: i32| {
            let start = &union.states()[union.start_state() as usize];
            let transition = start
                .transitions
                .get(&label)
                .or_else(|| (label >= 0).then(|| start.transitions.get(&DEFAULT_LABEL)).flatten());
            let Some((target, edge_weight)) = transition else {
                return Weight::empty();
            };
            let Some(final_weight) = union.states()[*target as usize].final_weight.as_ref() else {
                return Weight::empty();
            };
            edge_weight.intersection(final_weight)
        };

        assert!(!evaluate_runtime_label(encode_positive_label(3)).is_empty());
        assert!(!evaluate_runtime_label(encode_positive_label(4)).is_empty());
        assert!(!evaluate_runtime_label(encode_negative_label(5)).is_empty());
        assert!(evaluate_runtime_label(encode_negative_label(6)).is_empty());
        assert!(union.states()[union.start_state() as usize]
            .transitions
            .contains_key(&DEFAULT_LABEL));
    }

    #[test]
    fn overlap_local_union_declines_epsilon_inputs_for_generic_fallback() {
        let mut automaton = NWA::new(1, 0);
        let start = automaton.add_state();
        let target = automaton.add_state();
        automaton.set_start_states(vec![start]);
        automaton.add_epsilon(start, target, Weight::all());
        automaton.set_final_weight(target, Weight::all());

        assert!(!supports_overlap_local_union(std::slice::from_ref(&automaton)));
        assert!(determinize_epsilon_free_component_union(vec![automaton.clone()], None).is_none());

        let generic = determinize(&automaton).expect("generic determinization handles epsilon input");
        assert!(generic.num_states() > 0);
        assert!(generic.states().iter().any(|state| state.final_weight.is_some()));
    }

    #[test]
    fn compiled_parser_dwas_reconcile_and_union_with_one_to_many_start_mapping() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_table = compose_subgrammar_tables(
            &parent.table,
            &[SubgrammarTableInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                table: &child.table,
                start_nullable: child.table.embedded_start_nullable(),
            }],
        )
        .unwrap();
        let (merged_tokenizer, tokenizer_offsets) =
            crate::automata::lexer::tokenizer::Tokenizer::disjoint_union_with_terminal_offsets(&[
                (&parent.tokenizer, composed_table.terminal_offsets[0]),
                (&child.tokenizer, composed_table.terminal_offsets[1]),
            ]);
        let merged = compose_component_parser_dwas_and_possible_matches(
            &[
                ParserDwaComponent {
                    constraint: &parent,
                    parser_state_relation: &composed_table.state_relations[0],
                    tokenizer_state_offset: tokenizer_offsets[0],
                },
                ParserDwaComponent {
                    constraint: &child,
                    parser_state_relation: &composed_table.state_relations[1],
                    tokenizer_state_offset: tokenizer_offsets[1],
                },
            ],
            &composed_table.terminal_offsets,
            merged_tokenizer.num_states() as usize,
            &vocab.entries_map().keys().copied().collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(merged.artifact().0.num_states() > 0);
        assert!(merged.artifact().0.num_transitions() > 0);
        assert_eq!(
            merged.id_map().tokenizer_states.original_to_internal.len(),
            merged_tokenizer.num_states() as usize,
        );
        assert_eq!(composed_table.state_relations[1][0].len(), 2);
    }

    fn assert_accepts(constraint: &Constraint, bytes: &[u8]) {
        let mut state = constraint.start();
        state.commit_bytes(bytes).unwrap();
        assert!(state.is_finished(), "expected {:?} to finish", bytes);
    }

    fn assert_rejects(constraint: &Constraint, bytes: &[u8]) {
        let mut state = constraint.start();
        let accepted = state.commit_bytes(bytes).is_ok() && state.is_finished();
        assert!(!accepted, "expected {:?} to reject", bytes);
    }

    fn token_allowed(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    fn assert_constraints_equivalent_on_reachable_prefixes(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
    ) {
        let token_ids = vocab.entries_map().keys().copied().collect::<Vec<_>>();
        let mut frontier = vec![Vec::<u32>::new()];
        for depth in 0..=max_depth {
            let mut next = Vec::new();
            for prefix in frontier {
                let mut actual_state = actual.start();
                let mut expected_state = expected.start();
                for &token in &prefix {
                    actual_state.commit_token(token).unwrap_or_else(|error| {
                        panic!("actual rejected reachable prefix {prefix:?}: {error}")
                    });
                    expected_state.commit_token(token).unwrap_or_else(|error| {
                        panic!("expected rejected its own reachable prefix {prefix:?}: {error}")
                    });
                }

                let actual_mask = actual_state.mask();
                let expected_mask = expected_state.mask();
                assert_eq!(
                    actual_mask, expected_mask,
                    "mask mismatch after reachable prefix {prefix:?}",
                );
                assert_eq!(
                    actual_state.is_finished(),
                    expected_state.is_finished(),
                    "completion mismatch after reachable prefix {prefix:?}",
                );

                if depth == max_depth {
                    continue;
                }
                for &token in &token_ids {
                    if token_allowed(&expected_mask, token) {
                        let mut extended = prefix.clone();
                        extended.push(token);
                        next.push(extended);
                    }
                }
            }
            frontier = next;
        }
    }

    #[test]
    fn owned_parent_composition_matches_borrowed_and_monolithic() {
        let vocab = Vocab::new(vec![
            (0, b"<ab>".to_vec()),
            (1, b"<a".to_vec()),
            (2, b"b>".to_vec()),
            (3, b"<".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b">".to_vec()),
        ]);
        let compile_parent = || {
            Constraint::from_glrm_grammar(
                r#"
                    start document;
                    t SUB ::= @token(999);
                    nt document ::= "<" SUB ">";
                "#,
                &vocab,
            )
            .unwrap()
        };
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" "b";
                nt document ::= "<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let borrowed_parent = compile_parent();
        let borrowed = borrowed_parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();
        let owned = compile_parent()
            .compose_subgrammars_owned(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &owned,
            &monolithic,
            &vocab,
            4,
        );
        assert_constraints_equivalent_on_reachable_prefixes(
            &owned,
            &borrowed,
            &vocab,
            4,
        );
        assert_eq!(owned.tokenizer.start_state(), 0);
        assert!(owned.num_tokenizer_states() <= borrowed.num_tokenizer_states());
        for state in 0..owned.tokenizer.num_states() {
            for byte in u8::MIN..=u8::MAX {
                assert_eq!(
                    owned.tokenizer_fast_transitions.transition(
                        &owned.tokenizer,
                        state,
                        byte,
                    ),
                    owned.tokenizer.get_transition(state, byte),
                    "transported fast tokenizer transition differs at state {state}, byte {byte}",
                );
            }
            let closures = owned.tokenizer.all_singleton_epsilon_closures();
            assert_eq!(
                closures.get(state as usize).expect("closure state is in range"),
                owned.tokenizer.singleton_epsilon_closure(state).as_ref(),
                "transported singleton epsilon closure differs at state {state}",
            );
        }
    }

    #[test]
    fn composition_rejects_placeholder_that_matches_a_real_vocab_token() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            // The placeholder token deliberately exists in the model vocab.
            // Composition must still remove its exact-token path.
            (3, b"<placeholder>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(3);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let error = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .expect_err("real vocabulary tokens cannot be used as composition sentinels");
        assert!(error.to_string().contains("non-vocabulary sentinels"));
        assert!(error.to_string().contains("outside the supplied vocabulary"));
    }

    #[test]
    fn repeated_call_sites_match_monolithic_on_all_small_reachable_prefixes() {
        let vocab = Vocab::new(vec![
            (0, b"<a>b!".to_vec()),
            (1, b"<b>a!".to_vec()),
            (2, b"<".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
            (5, b">".to_vec()),
            (6, b"!".to_vec()),
            (7, b"a>b".to_vec()),
            (8, b"b>a".to_vec()),
            (9, b">b!".to_vec()),
            (10, b">a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" | "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" | "b";
                nt document ::= "<" child ">" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn recursive_child_matches_monolithic_on_all_small_reachable_prefixes() {
        let vocab = Vocab::new(vec![
            (0, b"Xaa!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"Xa".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"a!".to_vec()),
            (5, b"X".to_vec()),
            (6, b"a".to_vec()),
            (7, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" child?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" child?;
                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn parent_ignore_terminal_matches_monolithic_across_child_boundaries() {
        let vocab = Vocab::new(vec![
            (0, b"X a!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"a".to_vec()),
            (5, b"!".to_vec()),
            (6, b" a".to_vec()),
            (7, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                nt document ::= "X" "a" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn child_exact_special_token_survives_composition() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"<END>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t END ::= @token(3);
                nt child ::= "a" END;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                t END ::= @token(3);
                nt document ::= "X" "a" END "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save())
            .expect("composed exact-special-token constraint must survive serialization");

        assert!(composed
            .special_token_terminals
            .iter()
            .any(|special| special.token_id == 3));
        assert!(loaded
            .special_token_terminals
            .iter()
            .any(|special| special.token_id == 3));
        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            4,
        );
        assert_constraints_equivalent_on_reachable_prefixes(
            &loaded,
            &monolithic,
            &vocab,
            4,
        );
    }

    #[test]
    fn child_ignore_terminal_matches_monolithic_across_fused_boundaries() {
        let vocab = Vocab::new(vec![
            (0, b"X a!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"a".to_vec()),
            (5, b"!".to_vec()),
            (6, b" a".to_vec()),
            (7, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore WS;
                t WS ::= " "+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                nt document ::= "X" "a" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn same_compiled_child_can_fill_two_distinct_placeholders() {
        let vocab = Vocab::new(vec![
            (0, b"<a>,<b>".to_vec()),
            (1, b"<b>,<a>".to_vec()),
            (2, b"<".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
            (5, b">,<".to_vec()),
            (6, b">".to_vec()),
            (7, b"<a".to_vec()),
            (8, b"<b".to_vec()),
            (9, b"a>".to_vec()),
            (10, b"b>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "<" LEFT ">,<" RIGHT ">";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" | "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child ::= "a" | "b";
                nt document ::= "<" child ">,<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("LEFT", &child), ("RIGHT", &child)], &vocab)
            .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
    }

    #[test]
    fn three_distinct_children_match_monolithic_across_one_fused_token() {
        let vocab = Vocab::new(vec![
            (0, b"[a|b|c]".to_vec()),
            (1, b"[a|".to_vec()),
            (2, b"b|".to_vec()),
            (3, b"c]".to_vec()),
            (4, b"[".to_vec()),
            (5, b"a".to_vec()),
            (6, b"|".to_vec()),
            (7, b"b".to_vec()),
            (8, b"c".to_vec()),
            (9, b"]".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t FIRST ::= @token(997);
                t SECOND ::= @token(998);
                t THIRD ::= @token(999);
                nt document ::= "[" FIRST "|" SECOND "|" THIRD "]";
            "#,
            &vocab,
        )
        .unwrap();
        let first = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let second = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "b";
            "#,
            &vocab,
        )
        .unwrap();
        let third = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "c";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt first ::= "a";
                nt second ::= "b";
                nt third ::= "c";
                nt document ::= "[" first "|" second "|" third "]";
            "#,
            &vocab,
        )
        .unwrap();
        let composition = compose_constraints(
            &parent,
            &[
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "FIRST"),
                    constraint: &first,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "SECOND"),
                    constraint: &second,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "THIRD"),
                    constraint: &third,
                },
            ],
            &vocab,
        )
        .unwrap();
        let terminal_offsets = composition.terminal_offsets.clone();
        let composed = composition.constraint;

        let mut parser = ParserGSS::from_stacks(&[(
            vec![0],
            TerminalsDisallowed::new(),
        )]);
        for (name, terminal) in [
            ("[", terminal(&parent, "[")),
            ("first:a", terminal_offsets[1] + terminal(&first, "a")),
            ("|1", terminal(&parent, "|")),
            ("second:b", terminal_offsets[2] + terminal(&second, "b")),
            ("|2", terminal(&parent, "|")),
            ("third:c", terminal_offsets[3] + terminal(&third, "c")),
            ("]", terminal(&parent, "]")),
        ] {
            parser = advance_stacks(&composed.table, &parser, terminal);
            assert!(
                !parser.to_stacks(64).unwrap().is_empty(),
                "three-child table lost every stack after {name}",
            );
        }
        assert!(
            stacks_finished(&composed.table, &parser),
            "three-child composed table must accept the terminal sequence",
        );

        let mut composed_bytes = composed.start();
        composed_bytes.commit_bytes(b"[a|b|c]").unwrap();
        assert!(composed_bytes.is_finished());
        let mut monolithic_bytes = monolithic.start();
        monolithic_bytes.commit_bytes(b"[a|b|c]").unwrap();
        assert!(monolithic_bytes.is_finished());

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            7,
        );
        let mut fused = composed.start();
        fused.commit_token(0).unwrap();
        assert!(fused.is_finished());
    }

    #[test]
    fn composition_preserves_out_of_vocab_end_tokens() {
        const PLACEHOLDER_TOKEN: u32 = 999;
        const END_TOKEN: u32 = 1000;
        let vocab = Vocab::new(vec![
            (0, b"Xa!".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
            (3, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start document;
                nt child ::= "a";
                nt document ::= "X" child "!";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();
        assert_eq!(composed.table.embedded_end_token_ids(), vec![END_TOKEN]);

        for content in [vec![0], vec![1, 2, 3]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            assert!(!token_allowed(&actual.mask(), END_TOKEN));
            assert!(!token_allowed(&actual.mask(), PLACEHOLDER_TOKEN));
            for token in content {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.mask(), expected.mask());
            assert!(token_allowed(&actual.mask(), END_TOKEN));
            assert!(!token_allowed(&actual.mask(), PLACEHOLDER_TOKEN));
            assert_eq!(actual.forced(), vec![END_TOKEN]);
            actual.commit_token(END_TOKEN).unwrap();
            expected.commit_token(END_TOKEN).unwrap();
            assert!(actual.is_finished());
            assert!(expected.is_finished());
        }

        let loaded = Constraint::load(&composed.save()).unwrap();
        assert_eq!(loaded.table.embedded_end_token_ids(), vec![END_TOKEN]);
        let mut loaded_state = loaded.start();
        loaded_state.commit_token(0).unwrap();
        assert_eq!(loaded_state.forced(), vec![END_TOKEN]);
        loaded_state.commit_token(END_TOKEN).unwrap();
        assert!(loaded_state.is_finished());
    }

    #[test]
    fn composition_rejects_placeholder_token_id_reused_by_live_end_token() {
        const SHARED_TOKEN: u32 = 999;
        let vocab = Vocab::new(vec![
            (0, b"Xa!".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
            (3, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
            &[SHARED_TOKEN],
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();

        let error = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .expect_err("a placeholder sentinel ID cannot remain live as an end token");
        assert!(
            error
                .to_string()
                .contains("also configured as a grammar-level end token")
        );
        assert!(error.to_string().contains("unique sentinel token ID"));
    }

    #[test]
    fn child_end_token_survives_composition_and_blocks_nested_sentinel_reuse() {
        const END_TOKEN: u32 = 100;
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xa".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start child;
                nt child ::= "a";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                t END ::= @token(100);
                nt document ::= "X" "a" END "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();
        assert_eq!(composed.table.embedded_end_token_ids(), vec![END_TOKEN]);
        let loaded = Constraint::load(&composed.save()).unwrap();
        assert_eq!(loaded.table.embedded_end_token_ids(), vec![END_TOKEN]);

        for sequence in [vec![3, END_TOKEN, 2], vec![0, 1, END_TOKEN, 2]] {
            let mut actual = loaded.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_finished());
            assert!(expected.is_finished());
        }

        let outer = Constraint::from_glrm_grammar(
            r#"
                start outer;
                t INNER ::= @token(100);
                nt outer ::= INNER;
            "#,
            &vocab,
        )
        .unwrap();
        let error = outer
            .compose_subgrammars(&[("INNER", &loaded)], &vocab)
            .expect_err("nested placeholder must not reuse the child's end-token ID");
        assert!(
            error
                .to_string()
                .contains("also configured as a grammar-level end token")
        );
    }

    #[test]
    fn substitution_can_make_the_composed_start_nullable_for_later_embedding() {
        let vocab = Vocab::new(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let nullable_parent = Constraint::from_glrm_grammar(
            r#"
                start middle;
                t CHILD ::= @token(998);
                nt middle ::= CHILD;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(!nullable_parent.table.embedded_start_nullable());
        let nullable_child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        let middle = nullable_parent
            .compose_subgrammars(&[("CHILD", &nullable_child)], &vocab)
            .unwrap();
        assert!(middle.table.embedded_start_nullable());
        let middle = Constraint::load(&middle.save()).unwrap();
        assert!(middle.table.embedded_start_nullable());

        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t MIDDLE ::= @token(999);
                nt document ::= "X" MIDDLE "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = outer_parent
            .compose_subgrammars(&[("MIDDLE", &middle)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item ::= "a";
                nt document ::= "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            4,
        );
    }

    #[test]
    fn composed_constraint_matches_monolithic_when_tokens_do_not_cross_boundaries() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">" inner "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints(
            &parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                constraint: &child,
            }],
            &vocab,
        )
        .unwrap()
        .constraint;

        let valid = b"<ab>ab!";
        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for (offset, &byte) in valid.iter().enumerate() {
            assert_eq!(
                actual.mask(),
                expected.mask(),
                "mask mismatch before offset {offset}, byte {byte:?}",
            );
            actual.commit_bytes(&[byte]).unwrap();
            expected.commit_bytes(&[byte]).unwrap();
        }
        assert_eq!(actual.mask(), expected.mask(), "final mask mismatch");
        assert_eq!(actual.is_finished(), expected.is_finished());
        assert!(actual.is_finished());

        for bytes in [
            b"<ab>ab!".as_slice(),
            b"<a>ab!".as_slice(),
            b"<ab>a!".as_slice(),
            b"<ab>ab".as_slice(),
            b"ab>ab!".as_slice(),
            b"<ab><ab>!".as_slice(),
        ] {
            let mut expected = monolithic.start();
            let expected_accepts = expected.commit_bytes(bytes).is_ok() && expected.is_finished();
            let mut actual = composed.start();
            let actual_accepts = actual.commit_bytes(bytes).is_ok() && actual.is_finished();
            assert_eq!(actual_accepts, expected_accepts, "language mismatch for {bytes:?}");
        }
        assert_accepts(&composed, valid);
        assert_rejects(&composed, b"<ab>a!");
    }

    #[test]
    fn composed_constraint_matches_monolithic_for_fused_entry_and_exit_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"<a".to_vec()),
            (1, b"b>".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<".to_vec()),
            (5, b">".to_vec()),
            (6, b"a".to_vec()),
            (7, b"b".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "<" SUB ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g inner ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                nt document ::= "<" inner ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints(
            &parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                constraint: &child,
            }],
            &vocab,
        )
        .unwrap()
        .constraint;

        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for token in [0, 1, 3] {
            assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }
        assert_eq!(actual.is_finished(), expected.is_finished());
        assert!(actual.is_finished());
    }

    #[test]
    fn composed_constraint_matches_monolithic_for_child_outer_child_token() {
        let vocab = Vocab::new(vec![
            (0, b"<a".to_vec()),
            (1, b"b>,<c".to_vec()),
            (2, b"d>".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<".to_vec()),
            (5, b">,<".to_vec()),
            (6, b">".to_vec()),
            (7, b"a".to_vec()),
            (8, b"b".to_vec()),
            (9, b"c".to_vec()),
            (10, b"d".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "<" LEFT ">,<" RIGHT ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "c" "d";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g left ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                g right ::= {
                    start child;
                    nt child ::= "c" "d";
                };
                nt document ::= "<" left ">,<" right ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints(
            &parent,
            &[
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "LEFT"),
                    constraint: &left,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "RIGHT"),
                    constraint: &right,
                },
            ],
            &vocab,
        )
        .unwrap()
        .constraint;

        let mut expected_separate = monolithic.start();
        let mut actual_separate = composed.start();
        for token in [4, 7, 8, 5, 9, 10, 6, 3] {
            assert_eq!(
                actual_separate.mask(),
                expected_separate.mask(),
                "separate-token mask mismatch before token {token}",
            );
            actual_separate.commit_token(token).unwrap();
            expected_separate.commit_token(token).unwrap();
        }
        assert!(actual_separate.is_finished());
        assert!(expected_separate.is_finished());

        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for token in [0, 1, 2, 3] {
            assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }
        assert_eq!(actual.is_finished(), expected.is_finished());
        assert!(actual.is_finished());
    }

    #[test]
    fn nullable_child_composition_matches_monolithic_across_empty_and_nonempty_paths() {
        let vocab = Vocab::new(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item ::= "a";
                nt child ::= item?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item ::= "a";
                nt document ::= "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();
        assert!(child.table.embedded_start_nullable());
        let loaded_child = Constraint::load(&child.save()).unwrap();
        assert!(loaded_child.table.embedded_start_nullable());

        for child in [&child, &loaded_child] {
            let composed = parent
                .compose_subgrammars(&[("SUB", child)], &vocab)
                .unwrap();
            for sequence in [vec![0], vec![1], vec![2, 4], vec![2, 3, 4]] {
                let mut expected = monolithic.start();
                let mut actual = composed.start();
                for token in sequence {
                    assert_eq!(
                        actual.mask(),
                        expected.mask(),
                        "mask mismatch before token {token}",
                    );
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_finished(), expected.is_finished());
                assert!(actual.is_finished());
            }
        }
    }

    #[test]
    fn nested_composition_matches_flat_monolithic_across_all_boundaries() {
        let vocab = Vocab::new(vec![
            (0, b"X[a]!".to_vec()),
            (1, b"X[".to_vec()),
            (2, b"a]".to_vec()),
            (3, b"!".to_vec()),
            (4, b"X".to_vec()),
            (5, b"[".to_vec()),
            (6, b"a".to_vec()),
            (7, b"]".to_vec()),
        ]);
        let leaf = Constraint::from_glrm_grammar(
            r#"
                start leaf;
                nt leaf ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_grammar(
            r#"
                start middle;
                t LEAF ::= @token(998);
                nt middle ::= "[" LEAF "]";
            "#,
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .compose_subgrammars(&[("LEAF", &leaf)], &vocab)
            .unwrap();
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t MIDDLE ::= @token(999);
                nt document ::= "X" MIDDLE "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = outer_parent
            .compose_subgrammars(&[("MIDDLE", &middle)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document ::= "X" "[" "a" "]" "!";
            "#,
            &vocab,
        )
        .unwrap();

        for sequence in [vec![0], vec![1, 2, 3], vec![4, 5, 6, 7, 3]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_finished(), expected.is_finished());
            assert!(actual.is_finished());
        }
    }

    #[test]
    fn nested_composition_preserves_nullable_start_through_save_load() {
        let vocab = Vocab::new(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let leaf = Constraint::from_glrm_grammar(
            r#"
                start leaf;
                nt leaf ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_grammar(
            r#"
                start middle;
                t LEAF ::= @token(998);
                nt middle ::= LEAF?;
            "#,
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .compose_subgrammars(&[("LEAF", &leaf)], &vocab)
            .unwrap();
        assert!(middle.table.embedded_start_nullable());
        let middle = Constraint::load(&middle.save()).unwrap();
        assert!(middle.table.embedded_start_nullable());

        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t MIDDLE ::= @token(999);
                nt document ::= "X" MIDDLE "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composition = compose_constraints(
            &outer_parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: terminal(&outer_parent, "MIDDLE"),
                constraint: &middle,
            }],
            &vocab,
        )
        .unwrap();
        let middle_terminal_offset = composition.terminal_offsets[1];
        let composed = composition.constraint;
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item ::= "a";
                nt document ::= "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();

        let mut parser = ParserGSS::from_stacks(&[(
            vec![0],
            TerminalsDisallowed::new(),
        )]);
        parser = advance_stacks(&composed.table, &parser, terminal(&outer_parent, "X"));
        parser = advance_stacks(
            &composed.table,
            &parser,
            middle_terminal_offset + terminal(&middle, "subgrammar0::a"),
        );
        parser = advance_stacks(&composed.table, &parser, terminal(&outer_parent, "!"));
        assert!(
            stacks_finished(&composed.table, &parser),
            "nested nullable composed table must accept the nonempty child path",
        );

        for sequence in [vec![0], vec![1], vec![2, 4], vec![2, 3, 4]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_finished(), expected.is_finished());
            assert!(actual.is_finished());
        }
    }

    #[test]
    fn public_composition_matches_monolithic_for_child_alternatives() {
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"b!".to_vec()),
            (2, b"Xc".to_vec()),
            (3, b"d!".to_vec()),
            (4, b"X".to_vec()),
            (5, b"!".to_vec()),
            (6, b"a".to_vec()),
            (7, b"b".to_vec()),
            (8, b"c".to_vec()),
            (9, b"d".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "X" (LEFT | RIGHT) "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "c" "d";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                g left ::= {
                    start child;
                    nt child ::= "a" "b";
                };
                g right ::= {
                    start child;
                    nt child ::= "c" "d";
                };
                nt document ::= "X" (left | right) "!";
            "#,
            &vocab,
        )
        .unwrap();
        let duplicate_error = parent
            .compose_subgrammars(&[("LEFT", &left), ("LEFT", &right)], &vocab)
            .expect_err("duplicate placeholder inputs must be rejected");
        assert!(duplicate_error
            .to_string()
            .contains("was supplied more than once"));
        let composed = parent
            .compose_subgrammars(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save())
            .expect("composed constraints must survive serialization");

        for sequence in [[0, 1], [2, 3]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            let mut roundtripped = loaded.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                assert_eq!(
                    roundtripped.mask(),
                    expected.mask(),
                    "round-tripped mask mismatch before token {token}",
                );
                actual.commit_token(token).unwrap();
                roundtripped.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_finished());
            assert!(roundtripped.is_finished());
            assert!(expected.is_finished());
        }

        let mut crossed = composed.start();
        crossed.commit_token(0).unwrap();
        assert!(crossed.commit_token(3).is_err() || !crossed.is_finished());
    }

    fn sizeable_json_schema(prefix: &str, choices: usize) -> String {
        let choices = (0..choices)
            .map(|index| format!(r#""{prefix}_choice_{index:04}_long_literal_value""#))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"type":"object","additionalProperties":false,"properties":{{"choice":{{"type":"string","enum":[{choices}]}},"payload":{{"type":"string","minLength":1,"maxLength":32}}}},"required":["choice","payload"]}}"#,
        )
    }

    fn composition_benchmark_vocab() -> Vocab {
        let mut entries = (0u32..=255)
            .map(|byte| (byte, vec![byte as u8]))
            .collect::<Vec<_>>();
        entries.extend([
            (256, b"\": {".to_vec()),
            (257, b"}, \"value2\": {".to_vec()),
            (258, b"}}".to_vec()),
            (259, b"{\"value\": {".to_vec()),
        ]);
        Vocab::new(entries)
    }

    fn run_sizeable_json_schema_composition_benchmark(mode: &str, parent_source: &str) {
        let vocab = composition_benchmark_vocab();
        let choices = std::env::var("GLRMASK_COMPOSE_BENCH_CHOICES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(7500);
        let left_schema = sizeable_json_schema("left", choices);
        let right_schema = sizeable_json_schema("right", choices);

        eprintln!("[compose-bench] compile {mode} parent start");
        let parent = Constraint::from_glrm_grammar(parent_source, &vocab).unwrap();
        eprintln!("[compose-bench] compile {mode} parent done");

        let left_started = Instant::now();
        eprintln!("[compose-bench] compile left start");
        let left = Constraint::from_json_schema(&left_schema, &vocab).unwrap();
        let left_ms = left_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[compose-bench] compile left done {left_ms:.3} ms");
        let right_started = Instant::now();
        eprintln!("[compose-bench] compile right start");
        let right = Constraint::from_json_schema(&right_schema, &vocab).unwrap();
        let right_ms = right_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!("[compose-bench] compile right done {right_ms:.3} ms");

        let composition_runs = std::env::var("GLRMASK_COMPOSE_BENCH_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        let mut composition_samples_ms = Vec::with_capacity(composition_runs);
        for run in 0..composition_runs {
            let composition_started = Instant::now();
            eprintln!("[compose-bench] compose {mode} run={} start", run + 1);
            let composed = parent
                .compose_subgrammars(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
                .unwrap();
            let composition_ms = composition_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[compose-bench] compose {mode} run={} done {composition_ms:.3} ms",
                run + 1,
            );
            std::hint::black_box(composed);
            composition_samples_ms.push(composition_ms);
        }
        composition_samples_ms.sort_by(f64::total_cmp);
        let composition_ms = composition_samples_ms[composition_samples_ms.len() / 2];
        let composition_min_ms = composition_samples_ms[0];
        let composition_max_ms = *composition_samples_ms.last().unwrap();

        let child_sum_ms = left_ms + right_ms;
        eprintln!(
            "sizeable JSON composition: mode={mode} left_ms={left_ms:.3} right_ms={right_ms:.3} child_sum_ms={child_sum_ms:.3} composition_runs={} composition_median_ms={composition_ms:.3} composition_min_ms={composition_min_ms:.3} composition_max_ms={composition_max_ms:.3}",
            composition_samples_ms.len(),
        );
        assert!(
            composition_ms * 20.0 < child_sum_ms,
            "composition should be at least 20x faster than rebuilding both children: child_sum_ms={child_sum_ms:.3}, composition_ms={composition_ms:.3}",
        );
        if !cfg!(debug_assertions) {
            assert!(
                composition_ms < 20.0,
                "optimized composition should remain below 20 ms: {composition_ms:.3} ms",
            );
        }
    }

    #[test]
    #[ignore]
    fn sizeable_json_schema_sequential_composition_benchmark_probe() {
        run_sizeable_json_schema_composition_benchmark(
            "sequential",
            r#"
                start document;
                t LEFT ::= @token(1000000);
                t RIGHT ::= @token(1000001);
                nt document ::= "{" "\"value1\": " LEFT ", \"value2\": " RIGHT "}";
            "#,
        );
    }

    #[test]
    #[ignore]
    fn sizeable_json_schema_alternative_composition_benchmark_probe() {
        run_sizeable_json_schema_composition_benchmark(
            "alternative",
            r#"
                start document;
                t LEFT ::= @token(1000000);
                t RIGHT ::= @token(1000001);
                nt document ::= "{" ("\"left\": " LEFT | "\"right\": " RIGHT) "}";
            "#,
        );
    }
}
