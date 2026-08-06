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
use crate::automata::weighted_u32::nwa::{NWA, NWAState};
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::compiler::glr::analysis::{AnalyzedGrammar, EOF};
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
use crate::compiler::stages::templates::characterize::characterize_selected_terminals_profiled;
use crate::compiler::stages::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};
use crate::compiler::stages::templates::{Templates, commit_template_dfas_enabled};
use crate::compiler::constraint_possible_matches::{
    build_internal_token_bytes_from_groups, runtime_dynamic_vocab_for_vocab,
};
use crate::compiler::glr::table::{
    Action, ComposedTable, ControlEliminationReport, SubgrammarTableInput, compose_subgrammar_tables,
    compose_subgrammar_tables_explicit, compose_subgrammar_tables_shared_explicit,
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

fn eliminate_composed_runtime_controls(
    composed: &mut ComposedTable,
) -> Result<Option<ControlEliminationReport>, String> {
    eliminate_runtime_controls_parts(&mut composed.table, &mut composed.control_terminals)
}

fn eliminate_runtime_controls_parts(
    table: &mut crate::compiler::glr::table::GLRTable,
    controls: &mut BTreeSet<u32>,
) -> Result<Option<ControlEliminationReport>, String> {
    if controls.is_empty() {
        debug_assert!(table.control_terminals.is_empty());
        return Ok(None);
    }
    debug_assert_eq!(*controls, table.control_terminals);
    let report = table.eliminate_control_terminals_exact()?;
    controls.clear();
    debug_assert!(table.control_terminals.is_empty());
    Ok(Some(report))
}

fn profile_composed_state_relations(composed: &ComposedTable) {
    if !compose_profile_enabled() {
        return;
    }
    let mut rows = 0usize;
    let mut empty = 0usize;
    let mut singleton = 0usize;
    let mut multi = 0usize;
    let mut max_fanout = 0usize;
    let mut singleton_offset_runs = 0usize;
    let mut singleton_offset_run_rows = 0usize;
    for relation in &composed.state_relations {
        rows += relation.len();
        let mut last_offset = None::<i64>;
        for (local, targets) in relation.iter().enumerate() {
            max_fanout = max_fanout.max(targets.len());
            match targets.as_slice() {
                [] => empty += 1,
                [target] => {
                    singleton += 1;
                    let offset = i64::from(*target) - local as i64;
                    if last_offset != Some(offset) {
                        singleton_offset_runs += 1;
                        last_offset = Some(offset);
                    }
                    singleton_offset_run_rows += 1;
                }
                _ => {
                    multi += 1;
                    last_offset = None;
                }
            }
        }
    }
    eprintln!(
        "[glrmask/profile][constraint_state_relations] components={} rows={} empty={} singleton={} multi={} max_fanout={} singleton_offset_runs={} singleton_offset_run_rows={} boundary_nonterminals={}",
        composed.state_relations.len(),
        rows,
        empty,
        singleton,
        multi,
        max_fanout,
        singleton_offset_runs,
        singleton_offset_run_rows,
        composed.boundary_nonterminals.len(),
    );
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
    merged_reset_state: u32,
) -> Result<DirectComponentStateCoordinates, String> {
    let mut state_to_global = vec![u32::MAX; merged_tokenizer_state_count];
    let Some(reset_slot) = state_to_global.get_mut(merged_reset_state as usize) else {
        return Err(format!(
            "merged tokenizer reset state {merged_reset_state} lies outside {merged_tokenizer_state_count} states",
        ));
    };
    *reset_slot = 0;
    let mut global_to_states = vec![vec![merged_reset_state]];
    let mut state_representatives = vec![merged_reset_state];
    let mut dead_merged_states = Vec::<u32>::new();

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
        let mut merged_states_by_tsid = vec![Vec::<u32>::new(); local_map.len()];
        for (local_state, &local_tsid) in constraint.state_to_internal_tsid.iter().enumerate() {
            let local_state = local_state as u32;
            let merged_state = component
                .tokenizer_state_offset
                .checked_add(local_state)
                .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
            if local_tsid == u32::MAX {
                if merged_state == merged_reset_state {
                    return Err("merged tokenizer reset state has no internal TSID".into());
                }
                dead_merged_states.push(merged_state);
                continue;
            }
            let Some(merged_states) = merged_states_by_tsid.get_mut(local_tsid as usize) else {
                return Err(format!(
                    "component tokenizer state {local_state} maps to TSID {local_tsid} outside {} internal TSIDs",
                    merged_states_by_tsid.len(),
                ));
            };
            if merged_state == merged_reset_state {
                local_map[local_tsid as usize].push(0);
            } else {
                merged_states.push(merged_state);
            }
        }
        for (local_tsid, merged_states) in merged_states_by_tsid.into_iter().enumerate() {
            if !merged_states.is_empty() {
                let global_tsid = global_to_states.len() as u32;
                for &merged_state in &merged_states {
                    let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                        return Err(format!(
                            "component tokenizer state {merged_state} maps outside merged tokenizer"
                        ));
                    };
                    if *slot != u32::MAX {
                        return Err(format!(
                            "merged tokenizer state {merged_state} belongs to more than one component class"
                        ));
                    }
                    *slot = global_tsid;
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
    if !dead_merged_states.is_empty() {
        dead_merged_states.sort_unstable();
        dead_merged_states.dedup();
        let dead_tsid = global_to_states.len() as u32;
        for &merged_state in &dead_merged_states {
            let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                return Err(format!(
                    "dead component tokenizer state {merged_state} lies outside merged tokenizer",
                ));
            };
            if *slot != u32::MAX {
                return Err(format!(
                    "dead component tokenizer state {merged_state} was also assigned a live TSID",
                ));
            }
            *slot = dead_tsid;
        }
        state_representatives.push(dead_merged_states[0]);
        global_to_states.push(dead_merged_states);
    }
    if state_to_global.iter().any(|&tsid| tsid == u32::MAX) {
        return Err(format!(
            "direct component state map does not cover the merged tokenizer (reset={merged_reset_state})",
        ));
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
    merged_reset_state: u32,
    original_token_ids: &[u32],
) -> Result<(InternalIdMap, Vec<DirectComponentCoordinateMaps>), String> {
    let total_started_at = Instant::now();
    let state_started_at = Instant::now();
    let state_coordinates = build_direct_component_state_coordinates(
        components,
        merged_tokenizer_state_count,
        merged_reset_state,
    )?;
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

fn component_parser_nwa_with_top_accept(
    component: &ParserDwaComponent<'_>,
    include_top_accept: bool,
) -> Result<NWA, String> {
    let constraint = component.constraint;
    if component.parser_state_relation.len() != constraint.table.num_states as usize {
        return Err(format!(
            "parser-state relation has {} rows for a {}-state component table",
            component.parser_state_relation.len(),
            constraint.table.num_states,
        ));
    }

    let source = &constraint.parser_dwa;
    // Parser DWAs are acyclic by construction. This conversion only relabels
    // edges and adds a disjoint one-edge top-accept branch, so the resulting
    // component NWA carries the same overlap-local certification without a
    // second whole-graph scan at link time.
    debug_assert!(source.is_acyclic());
    let build_state = |state: &DWAState| -> Result<NWAState, String> {
        let mut explicit_positive = SmallVec::<[u32; 8]>::new();
        let mut entries = Vec::<(i32, u32, Weight)>::new();
        for (&label, (target, weight)) in &state.transitions {
            if label == DEFAULT_LABEL {
                continue;
            }
            if label >= 0 {
                explicit_positive.push(label as u32);
            }
            for mapped in mapped_labels(label, component.parser_state_relation)? {
                entries.push((mapped, *target, weight.clone()));
            }
        }
        explicit_positive.sort_unstable();
        explicit_positive.dedup();
        if let Some((target, weight)) = state.transitions.get(&DEFAULT_LABEL) {
            for local_state in 0..constraint.table.num_states {
                if explicit_positive.binary_search(&local_state).is_ok() {
                    continue;
                }
                let mapped_states = component
                    .parser_state_relation
                    .get(local_state as usize)
                    .ok_or_else(|| {
                        format!("parser-state relation omits local state {local_state}")
                    })?;
                if mapped_states.is_empty() {
                    return Err(format!(
                        "parser-state relation maps local state {local_state} nowhere"
                    ));
                }
                entries.extend(mapped_states.iter().copied().map(|global_state| {
                    (encode_positive_label(global_state), *target, weight.clone())
                }));
            }
        }
        entries.sort_unstable_by_key(|(label, target, weight)| {
            (*label, *target, weight.ptr_key())
        });
        entries.dedup_by(|left, right| {
            left.0 == right.0 && left.1 == right.1 && left.2.ptr_key() == right.2.ptr_key()
        });
        let mut transitions = BTreeMap::<i32, Vec<(u32, Weight)>>::new();
        for (label, target, weight) in entries {
            transitions.entry(label).or_default().push((target, weight));
        }
        Ok(NWAState {
            final_weight: state.final_weight.clone(),
            transitions,
            epsilons: Vec::new(),
        })
    };
    let states = if rayon::current_num_threads() == 1 {
        source.states().iter().map(build_state).collect::<Result<Vec<_>, _>>()?
    } else {
        source
            .states()
            .par_iter()
            .map(build_state)
            .collect::<Result<Vec<_>, _>>()?
    };
    let mut nwa = NWA::from_parts(states, vec![source.start_state()]);

    let top_accept = include_top_accept.then(|| materialized_top_acceptance(constraint));
    if let Some(top_accept) = top_accept.filter(|top_accept| !top_accept.is_empty()) {
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

fn component_parser_nwa(component: &ParserDwaComponent<'_>) -> Result<NWA, String> {
    component_parser_nwa_with_top_accept(component, true)
}

fn component_parser_compressed(
    component: &ParserDwaComponent<'_>,
) -> Result<RawCompressedAutomaton, String> {
    let constraint = component.constraint;
    if component.parser_state_relation.len() != constraint.table.num_states as usize {
        return Err(format!(
            "parser-state relation has {} rows for a {}-state component table",
            component.parser_state_relation.len(),
            constraint.table.num_states,
        ));
    }
    let source = &constraint.parser_dwa;
    debug_assert!(source.is_acyclic());
    let build_state = |state: &DWAState| -> Result<RawCompressedState, String> {
        let mut explicit_positive = SmallVec::<[u32; 8]>::new();
        let mut entries = Vec::<(i32, u32, Weight)>::new();
        for (&label, (target, weight)) in &state.transitions {
            if label == DEFAULT_LABEL {
                continue;
            }
            if label >= 0 {
                explicit_positive.push(label as u32);
            }
            for mapped in mapped_labels(label, component.parser_state_relation)? {
                entries.push((mapped, *target, weight.clone()));
            }
        }
        explicit_positive.sort_unstable();
        explicit_positive.dedup();
        if let Some((target, weight)) = state.transitions.get(&DEFAULT_LABEL) {
            for local_state in 0..constraint.table.num_states {
                if explicit_positive.binary_search(&local_state).is_ok() {
                    continue;
                }
                let mapped_states = component
                    .parser_state_relation
                    .get(local_state as usize)
                    .ok_or_else(|| {
                        format!("parser-state relation omits local state {local_state}")
                    })?;
                if mapped_states.is_empty() {
                    return Err(format!(
                        "parser-state relation maps local state {local_state} nowhere"
                    ));
                }
                entries.extend(mapped_states.iter().copied().map(|global_state| {
                    (encode_positive_label(global_state), *target, weight.clone())
                }));
            }
        }
        entries.sort_unstable_by_key(|(label, target, weight)| {
            (*label, *target, weight.ptr_key())
        });
        entries.dedup_by(|left, right| {
            left.0 == right.0 && left.1 == right.1 && left.2.ptr_key() == right.2.ptr_key()
        });

        let mut runs = Vec::<RawTransitionRun>::new();
        let mut deterministic = true;
        let mut index = 0usize;
        while index < entries.len() {
            let label = entries[index].0;
            let mut targets = Vec::<(u32, Weight)>::new();
            while index < entries.len() && entries[index].0 == label {
                targets.push((entries[index].1, entries[index].2.clone()));
                index += 1;
            }
            deterministic &= targets.len() <= 1;
            if let Some(last) = runs.last_mut()
                && last.end.checked_add(1) == Some(label)
                && same_raw_targets(&last.targets, &targets)
            {
                last.end = label;
            } else {
                runs.push(RawTransitionRun {
                    start: label,
                    end: label,
                    targets,
                });
            }
        }
        Ok(RawCompressedState {
            runs,
            default_targets: None,
            final_weight: state.final_weight.clone(),
            deterministic,
        })
    };
    let states = source
        .states()
        .par_iter()
        .map(build_state)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(RawCompressedAutomaton {
        states,
        start_states: vec![source.start_state()],
    })
}

fn strip_unscoped_ignore_identity_compressed(
    automaton: &mut RawCompressedAutomaton,
    ignore_possible_matches: Option<&Weight>,
) {
    let Some(ignore_weight) = ignore_possible_matches else {
        return;
    };
    for &start in &automaton.start_states {
        let Some(state) = automaton.states.get_mut(start as usize) else {
            continue;
        };
        let Some(final_weight) = state.final_weight.take() else {
            continue;
        };
        let retained = final_weight.difference(ignore_weight);
        if !retained.is_empty() {
            state.final_weight = Some(retained);
        }
    }
}

fn transported_component_top_accept_parts(
    component: &ParserDwaComponent<'_>,
) -> Result<BTreeMap<i32, Vec<Weight>>, String> {
    let constraint = component.constraint;
    if component.parser_state_relation.len() != constraint.table.num_states as usize {
        return Err(format!(
            "parser-state relation has {} rows for a {}-state component table",
            component.parser_state_relation.len(),
            constraint.table.num_states,
        ));
    }

    let default_combined = constraint.parser_top_accept.get(&DEFAULT_LABEL);
    let default_parts = constraint.parser_top_accept_parts.get(&DEFAULT_LABEL);
    let mut transported = BTreeMap::<i32, Vec<Weight>>::new();
    for local_state in 0..constraint.table.num_states {
        let local_label = encode_positive_label(local_state);
        let mut parts = SmallVec::<[Weight; 8]>::new();
        if let Some(weight) = constraint
            .parser_top_accept
            .get(&local_label)
            .or(default_combined)
        {
            parts.push(weight.clone());
        }
        if let Some(weights) = constraint
            .parser_top_accept_parts
            .get(&local_label)
            .or(default_parts)
        {
            parts.extend(weights.iter().cloned());
        }
        if let Some(row) = constraint.table.advance.get(local_state as usize) {
            for terminal in row.iter_ones() {
                if let Some(weight) = constraint
                    .direct_regular_l1_complete_by_terminal
                    .get(&(terminal as u32))
                {
                    parts.push(weight.clone());
                }
            }
        }
        if parts.is_empty() {
            continue;
        }
        parts.sort_unstable_by_key(Weight::ptr_key);
        parts.dedup_by_key(|weight| weight.ptr_key());
        for &global_state in component
            .parser_state_relation
            .get(local_state as usize)
            .ok_or_else(|| format!("parser-state relation omits local state {local_state}"))?
        {
            transported
                .entry(encode_positive_label(global_state))
                .or_default()
                .extend(parts.iter().cloned());
        }
    }
    for parts in transported.values_mut() {
        parts.sort_unstable_by_key(Weight::ptr_key);
        parts.dedup_by_key(|weight| weight.ptr_key());
    }
    Ok(transported)
}

fn collapse_transported_top_accept_parts(
    parts: BTreeMap<i32, Vec<Weight>>,
    num_parser_states: u32,
) -> BTreeMap<i32, Weight> {
    let profile = compose_profile_enabled();
    let started_at = Instant::now();
    let labels = parts.len();
    let total_parts = parts.values().map(Vec::len).sum::<usize>();
    let max_parts = parts.values().map(Vec::len).max().unwrap_or(0);
    let mut collapsed = parts
        .into_par_iter()
        .filter_map(|(label, mut weights)| {
            weights.sort_unstable_by_key(Weight::ptr_key);
            weights.dedup_by_key(|weight| weight.ptr_key());
            let weight = match weights.len() {
                0 => return None,
                1 => weights.pop().unwrap(),
                _ => Weight::union_all(weights.iter()),
            };
            (!weight.is_empty()).then_some((label, weight))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let unique_weights;
    let mut default_multiplicity = 0usize;
    let mut exceptions = collapsed.len();
    let full_positive_domain = collapsed.len() == num_parser_states as usize
        && (0..num_parser_states).all(|state| {
            collapsed.contains_key(&encode_positive_label(state))
        });
    if full_positive_domain && !collapsed.is_empty() {
        let mut by_weight = FxHashMap::<usize, (Weight, usize)>::default();
        for weight in collapsed.values() {
            by_weight
                .entry(weight.ptr_key())
                .and_modify(|(_, count)| *count += 1)
                .or_insert_with(|| (weight.clone(), 1));
        }
        unique_weights = by_weight.len();
        if let Some((default_weight, count)) = by_weight
            .into_values()
            .max_by_key(|(_, count)| *count)
        {
            let default_key = default_weight.ptr_key();
            collapsed.retain(|_, weight| weight.ptr_key() != default_key);
            collapsed.insert(DEFAULT_LABEL, default_weight);
            default_multiplicity = count;
            exceptions = collapsed.len().saturating_sub(1);
        }
    } else {
        unique_weights = collapsed
            .values()
            .map(Weight::ptr_key)
            .collect::<FxHashSet<_>>()
            .len();
    }
    if profile {
        eprintln!(
            "[glrmask/profile][constraint_top_accept_transport] labels={} parts={} max_parts={} collapsed_labels={} full_positive_domain={} unique_weights={} default_multiplicity={} exceptions={} collapse_ms={:.3}",
            labels,
            total_parts,
            max_parts,
            collapsed.len(),
            full_positive_domain,
            unique_weights,
            default_multiplicity,
            exceptions,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    collapsed
}

/// Standalone component parser DWAs erase their one global ignore terminal
/// before parser-state interpretation. Consequently an ignore-only model token
/// appears as an unqualified empty-word final weight at the parser-DWA start.
/// That is correct for a standalone/global-ignore constraint, but would leak a
/// child ignore token into parent states after scoped linking. Remove only that
/// identity branch; boundary repair reintroduces it through the scoped `Skip`
/// actions in the explicit composed table. Other terminalizations of the same
/// model token remain intact.
fn strip_unscoped_ignore_identity(
    automaton: &mut NWA,
    ignore_possible_matches: Option<&Weight>,
) {
    let Some(ignore_weight) = ignore_possible_matches else {
        return;
    };
    let starts = automaton.start_states().to_vec();
    for start in starts {
        let Some(state) = automaton.states_mut().get_mut(start as usize) else {
            continue;
        };
        let Some(final_weight) = state.final_weight.take() else {
            continue;
        };
        let retained = final_weight.difference(ignore_weight);
        if !retained.is_empty() {
            state.final_weight = Some(retained);
        }
    }
}

type PossibleMatches = BTreeMap<u32, Weight>;

struct BoundaryRepair {
    parser_dwa: MappedArtifact<DWA>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    active_terminals: Vec<bool>,
}

type BoundarySeedRelations = BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>;

struct BoundaryLexicalPrepass {
    seed_terminals: Vec<bool>,
    disallowed_follows: BTreeMap<u32, BitSet>,
    boundary_paths: BoundaryTokenDiscovery,
    discovery_ms: f64,
    seed_relations: BoundarySeedRelations,
    one_byte_ms: f64,
    active_terminals: Vec<bool>,
    boundary_special_token_terminals: Vec<SpecialTokenTerminal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BoundaryTokenNodeKey {
    offset: usize,
    /// `u32::MAX` means no non-ignore terminal has committed yet.
    last_terminal: u32,
    /// Whether this path has touched a terminal that can begin a child or a
    /// parent continuation (or a scoped ignore terminal).
    seeded: bool,
    /// False only for the arbitrary-residual first fragment. Once any
    /// terminal commits, subsequent fragments start from lexer reset.
    started: bool,
}

#[derive(Debug)]
struct BoundaryTokenEdge {
    target: usize,
    terminal: u32,
}

#[derive(Debug)]
struct BoundaryTokenNode {
    key: BoundaryTokenNodeKey,
    outgoing: Vec<BoundaryTokenEdge>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct ResidualScanResult {
    /// Per-start-state longest matches. Different residual starts may produce
    /// different valid widths for the same terminal. These collections are
    /// tiny in practice; sorted vectors avoid one tree allocation per scan.
    matches: Vec<(u32, usize)>,
    future_terminals: Vec<u32>,
}

struct BoundaryTokenWitness {
    token_id: u32,
    start_states: Vec<u32>,
    nodes: Vec<BoundaryTokenNode>,
    good: Vec<bool>,
    accepting: Vec<bool>,
}

struct BoundaryTokenDiscovery {
    terminals: BitSet,
    token_ids: Vec<u32>,
    witnesses: Vec<BoundaryTokenWitness>,
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
    merged_reset_state: u32,
) -> Vec<u32> {
    vec![merged_reset_state]
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
    merged_reset_state: u32,
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
        if global_start == merged_reset_state {
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

/// Exact lightweight predicate used before materializing full residual scans.
///
/// Boundary discovery only needs to retain a model token when some suffix can
/// either finish a segment-entry terminal or leave one live at the token end.
/// Testing that predicate directly avoids allocating and storing complete
/// match/future vectors for every vocabulary suffix. Full residual summaries
/// are built later only for tokens that pass this test.
fn seed_dfa_reaches_match_or_future(
    dfa: &crate::automata::lexer::DFA,
    bytes: &[u8],
) -> bool {
    let mut state = 0u32;
    for &byte in bytes {
        let Some(next) = dfa.step(state, byte) else {
            return false;
        };
        state = next;
        if dfa.finalizers(state).contains(0) {
            return true;
        }
    }
    dfa.possible_future_group_ids(state).contains(0)
}

fn scan_component_residual_start_groups(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    reset_live_bytes: &[U8Set],
    merged_reset_state: u32,
    bytes: &[u8],
    candidate_groups: &[(u32, Vec<u32>)],
) -> FxHashMap<ResidualScanResult, Vec<u32>> {
    let validate =
        std::env::var_os("GLRMASK_VALIDATE_COMPOSE_TSID_REPRESENTATIVE_SCAN").is_some();
    let mut starts_by_scan = FxHashMap::<ResidualScanResult, Vec<u32>>::default();
    let mut by_component = vec![Vec::<(u32, &[u32])>::new(); components.len()];

    for (representative, support_states) in candidate_groups {
        if *representative == merged_reset_state {
            let scan = scan_component_residual_starts(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                reset_live_bytes,
                merged_reset_state,
                bytes,
                &[*representative],
            );
            starts_by_scan
                .entry(scan)
                .or_default()
                .extend(support_states);
            continue;
        }
        let component_index = tokenizer_state_offsets
            .partition_point(|&offset| offset <= *representative)
            .saturating_sub(1);
        let Some(component) = components.get(component_index) else {
            continue;
        };
        let offset = tokenizer_state_offsets[component_index];
        let local_start = *representative - offset;
        if local_start < component.tokenizer.num_states() {
            by_component[component_index].push((local_start, support_states));
        }
    }

    for (component_index, starts) in by_component.into_iter().enumerate() {
        if starts.is_empty() {
            continue;
        }
        let component = components[component_index];
        let state_offset = tokenizer_state_offsets[component_index];
        let terminal_offset = terminal_offsets[component_index];
        let local_starts = starts.iter().map(|(start, _)| *start).collect::<Vec<_>>();
        let support_by_start = starts.into_iter().collect::<FxHashMap<_, _>>();

        for (end_states, matches, grouped_starts) in component
            .tokenizer
            .execute_summary_groups_from_states(bytes, &local_starts)
        {
            let mut scan = ResidualScanResult::default();
            scan.matches.extend(
                matches
                    .into_iter()
                    .filter(|(_, width)| *width > 0)
                    .map(|(terminal, width)| (terminal_offset + terminal, width)),
            );
            for end_state in end_states {
                scan.future_terminals.extend(
                    component
                        .tokenizer
                        .possible_future_terminals_iter(end_state)
                        .map(|terminal| terminal_offset + terminal),
                );
            }
            scan.matches.sort_unstable();
            scan.matches.dedup();
            scan.future_terminals.sort_unstable();
            scan.future_terminals.dedup();

            let output_support = starts_by_scan.entry(scan.clone()).or_default();
            for local_start in grouped_starts {
                let Some(support) = support_by_start.get(&local_start) else {
                    continue;
                };
                if validate {
                    for &global_state in *support {
                        let reference = scan_component_residual_starts(
                            components,
                            tokenizer_state_offsets,
                            terminal_offsets,
                            reset_live_bytes,
                            merged_reset_state,
                            bytes,
                            &[global_state],
                        );
                        assert_eq!(
                            scan, reference,
                            "batched component residual scan differs for state {global_state} (component {component_index}, local start {local_start}, offset {state_offset})",
                        );
                    }
                }
                output_support.extend_from_slice(support);
            }
        }
    }

    for states in starts_by_scan.values_mut() {
        states.sort_unstable();
        states.dedup();
    }
    starts_by_scan
}

fn transition_boundary_key(
    key: BoundaryTokenNodeKey,
    terminal: u32,
    next_offset: usize,
    seed_terminals: &[bool],
    ignore_terminals: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<BoundaryTokenNodeKey> {
    if ignore_terminals.contains(terminal as usize) {
        return Some(BoundaryTokenNodeKey {
            offset: next_offset,
            seeded: key.seeded
                || seed_terminals
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false),
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
    Some(BoundaryTokenNodeKey {
        offset: next_offset,
        last_terminal: terminal,
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
    seed_terminals: &[bool],
    ignore_terminals: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<(Vec<BoundaryTokenNode>, Vec<bool>, Vec<bool>)> {
    let mut nodes = Vec::<BoundaryTokenNode>::new();
    let mut node_ids = FxHashMap::<BoundaryTokenNodeKey, usize>::default();
    let mut queue = std::collections::VecDeque::<usize>::new();
    let start_key = BoundaryTokenNodeKey {
        offset: 0,
        last_terminal: u32::MAX,
        seeded: false,
        started: false,
    };
    nodes.push(BoundaryTokenNode {
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
            let Some(target_key) = transition_boundary_key(
                key,
                terminal,
                next_offset,
                seed_terminals,
                ignore_terminals,
                disallowed_follows,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len() && target_key.seeded;
                nodes.push(BoundaryTokenNode {
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
                .push(BoundaryTokenEdge { target, terminal });
        }

        // An unfinished final terminal is a real terminal-DWA label and may
        // itself be the boundary-begin seed at token end.
        for &terminal in &scan.future_terminals {
            let Some(target_key) = transition_boundary_key(
                key,
                terminal,
                bytes.len(),
                seed_terminals,
                ignore_terminals,
                disallowed_follows,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len() && target_key.seeded;
                nodes.push(BoundaryTokenNode {
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
                .push(BoundaryTokenEdge { target, terminal });
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
    _vocab: &Vocab,
    candidate_tokens: &BTreeSet<u32>,
) -> BTreeMap<u32, Vec<(usize, u32, u32)>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    let mut by_token = BTreeMap::<u32, Vec<(usize, u32, u32)>>::new();
    for (component_index, constraint) in components.iter().enumerate() {
        debug_assert!(constraint.possible_matches_complete);

        // Transpose only the candidate vocabulary classes. The previous path
        // enumerated every token in every possible-match bitmap and filtered
        // afterward, even though boundary discovery had already reduced the
        // 128k vocabulary to a few hundred candidates.
        let mut originals_by_internal = FxHashMap::<u32, SmallVec<[u32; 2]>>::default();
        if constraint.internal_token_to_tokens.is_empty() {
            for &original in candidate_tokens {
                originals_by_internal
                    .entry(original)
                    .or_default()
                    .push(original);
            }
        } else {
            for &original in candidate_tokens {
                let Some(&internal) = constraint
                    .original_token_to_internal
                    .get(original as usize)
                else {
                    continue;
                };
                if internal != u32::MAX {
                    originals_by_internal
                        .entry(internal)
                        .or_default()
                        .push(original);
                }
            }
        }
        if originals_by_internal.is_empty() {
            continue;
        }
        let candidate_internal_tokens =
            RangeSetBlaze::from_iter(originals_by_internal.keys().copied());

        for weight in constraint.possible_matches.values() {
            for (start_tsid, end_tsid, internal_tokens) in weight.range_entries() {
                let hits = internal_tokens.as_ref() & &candidate_internal_tokens;
                for internal_token in hits.iter() {
                    let Some(originals) = originals_by_internal.get(&internal_token) else {
                        continue;
                    };
                    for &original in originals {
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
    merged_reset_state: u32,
) -> Vec<(u32, Vec<u32>)> {
    // The retained/fresh merged reset dispatcher is a semantic state of its
    // semantic state of its own and must not be conflated with an individual
    // component's local start state.
    let mut support_by_representative = FxHashMap::<u32, Vec<u32>>::default();
    support_by_representative.insert(merged_reset_state, vec![merged_reset_state]);
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
                    if global == merged_reset_state {
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
) -> BTreeSet<u32> {
    let num_terminals = terminal_offsets
        .iter()
        .copied()
        .zip(components)
        .map(|(offset, component)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
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

    // Boundary repair is needed when a token begins a segment-entry terminal at
    // any byte offset, including offset zero. Compile each seed exactly and scan
    // token suffixes for a completed seed prefix or an unfinished seed at token
    // end. Exact parser discovery below still rejects lexically possible suffixes
    // whose preceding bytes cannot reach the boundary in the grammar.
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
            let Some(expr) = component.tokenizer.terminal_expr(local_terminal as u32) else {
                // Missing retained source is rare and cannot justify an unsafe
                // exclusion. Fall back to the conservative first-byte summary.
                conservative_first |= summaries[global_terminal]
                    .map(|summary| summary.first)
                    .unwrap_or(U8Set::all());
                continue;
            };
            let dfa_index = endpoint_dfas.len();
            let dfa = compile_terminal_expr_dfa(expr);
            for byte in 0u8..=u8::MAX {
                if dfa.step(0, byte).is_some() {
                    endpoint_dfas_by_first[byte as usize].push(dfa_index);
                }
            }
            endpoint_dfas.push(dfa);
        }
    }
    let mut candidates = BTreeSet::new();
    for (&token, bytes) in vocab.entries_map().iter() {
        if bytes.len() < 2 {
            continue;
        }
        'suffixes: for offset in 0..bytes.len() {
            let first = bytes[offset];
            if conservative_first.contains(first) {
                candidates.insert(token);
                break 'suffixes;
            }
            for &dfa_index in &endpoint_dfas_by_first[first as usize] {
                let dfa = &endpoint_dfas[dfa_index];
                let mut state = 0u32;
                let mut reached_token_end = true;
                for &byte in &bytes[offset..] {
                    let Some(next) = dfa.step(state, byte) else {
                        reached_token_end = false;
                        break;
                    };
                    state = next;
                    if dfa.finalizers(state).contains(0) {
                        candidates.insert(token);
                        break 'suffixes;
                    }
                }
                if reached_token_end && dfa.possible_future_group_ids(state).contains(0) {
                    candidates.insert(token);
                    break 'suffixes;
                }
            }
        }
    }
    let seed_dfa_candidates = candidates.len();

    // The first terminal of the entered segment can occupy an entire token.
    // Component possible-matches is exact for that endpoint case. A terminal
    // immediately *before* the boundary needs no repair unless the same token
    // also begins the new segment; the byte-pair candidates cover that case.
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            if !seed_terminals
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
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_prefilter_sources] adjacent_pairs={} after_seed_dfas={} after_seed_possible_matches={} seed_terminals={:?}",
            seed_dfa_candidates,
            seed_dfa_candidates,
            candidates.len(),
            seed_terminals
                .iter()
                .enumerate()
                .filter_map(|(terminal, &seed)| seed.then_some(terminal))
                .collect::<Vec<_>>(),
        );
    }
    candidates
}

fn discover_boundary_token_paths(
    vocab: &Vocab,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    merged_reset_state: u32,
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    ignore_terminals: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> BoundaryTokenDiscovery {
    let num_terminals = components
        .iter()
        .zip(terminal_offsets.iter().copied())
        .map(|(component, offset)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let reset_starts = composite_reset_states(merged_reset_state);
    let reset_live_bytes = component_reset_live_bytes(components);
    let use_prefilter = std::env::var_os("GLRMASK_COMPOSE_DISABLE_BOUNDARY_PREFILTER").is_none();
    let all_multi_byte_entries = vocab
        .entries_map()
        .iter()
        .filter(|(_, bytes)| bytes.len() >= 2)
        .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
        .collect::<Vec<_>>();
    let prefilter_started_at = Instant::now();
    let prefilter = if use_prefilter {
        let seed_compile_started_at = Instant::now();
        let mut seen_seed_exprs = FxHashSet::<&Expr>::default();
        let mut seed_dfas = FxHashSet::<crate::automata::lexer::DFA>::default();
        let mut fallback_seed_terminals_by_component = Vec::with_capacity(components.len());
        for (component_index, component) in components.iter().enumerate() {
            let terminal_offset = terminal_offsets[component_index] as usize;
            let mut fallback = Vec::new();
            for local_terminal in 0..component.tokenizer.num_terminals() as usize {
                if !seed_terminals
                    .get(terminal_offset + local_terminal)
                    .copied()
                    .unwrap_or(false)
                {
                    continue;
                }
                if let Some(expr) = component.tokenizer.terminal_expr(local_terminal as u32) {
                    if seen_seed_exprs.insert(expr) {
                        seed_dfas.insert(compile_terminal_expr_dfa(expr));
                    }
                } else {
                    fallback.push(local_terminal as u32);
                }
            }
            fallback_seed_terminals_by_component.push(fallback);
        }
        let projected_seed_dfas = components
            .par_iter()
            .zip(fallback_seed_terminals_by_component.par_iter())
            .filter_map(|(component, terminals)| {
                (!terminals.is_empty()).then(|| {
                    component
                        .tokenizer
                        .selected_terminal_language_dfa(terminals)
                })
            })
            .collect::<Vec<_>>();
        let projected_seed_dfa_count = projected_seed_dfas.len();
        seed_dfas.extend(projected_seed_dfas);
        let seed_dfas = seed_dfas.into_iter().collect::<Vec<_>>();
        let seed_compile_ms = seed_compile_started_at.elapsed().as_secs_f64() * 1000.0;

        let coarse_started_at = Instant::now();
        let coarse = boundary_token_prefilter(vocab, components, terminal_offsets, seed_terminals);
        let coarse_ms = coarse_started_at.elapsed().as_secs_f64() * 1000.0;

        let suffix_index_started_at = Instant::now();
        let suffix_index = crate::compiler::vocab_suffix_index::get(vocab);
        let suffix_count = suffix_index.entries().len();
        let suffix_index_ms = suffix_index_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_started_at = Instant::now();
        let exact = suffix_index
            .entries()
            .par_iter()
            .filter_map(|entry| {
                let has_coarse_owner = entry
                    .token_ids()
                    .iter()
                    .any(|token_id| coarse.contains(token_id));
                (has_coarse_owner
                    && seed_dfas
                        .iter()
                        .any(|dfa| seed_dfa_reaches_match_or_future(dfa, entry.suffix())))
                .then(|| {
                    entry
                        .token_ids()
                        .iter()
                        .copied()
                        .filter(|token_id| coarse.contains(token_id))
                        .collect::<SmallVec<[u32; 2]>>()
                })
            })
            .flatten_iter()
            .collect::<BTreeSet<_>>();
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_seed_suffix_filter] unique_seed_dfas={} projected_seed_dfas={} coarse_tokens={} suffixes={} exact_tokens={} seed_compile_ms={seed_compile_ms:.3} coarse_ms={coarse_ms:.3} suffix_index_ms={suffix_index_ms:.3} exact_ms={:.3}",
                seed_dfas.len(),
                projected_seed_dfa_count,
                coarse.len(),
                suffix_count,
                exact.len(),
                exact_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        exact
    } else {
        all_multi_byte_entries
            .iter()
            .map(|&(token_id, _)| token_id)
            .collect::<BTreeSet<_>>()
    };
    let prefilter_ms = prefilter_started_at.elapsed().as_secs_f64() * 1000.0;
    let multi_byte_entries = all_multi_byte_entries
        .iter()
        .copied()
        .filter(|(token_id, _)| prefilter.contains(token_id))
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
                        merged_reset_state,
                        suffix,
                        &reset_starts,
                    );
                    (suffix, scan)
                })
                .collect::<FxHashMap<_, _>>(),
        )
    };
    let suffix_cache_ms = suffix_cache_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_ranges_started_at = Instant::now();
    let candidate_ranges = boundary_candidate_state_ranges_by_token(
        components,
        tokenizer_state_offsets,
        vocab,
        &prefilter,
    );
    let candidate_ranges_ms = candidate_ranges_started_at.elapsed().as_secs_f64() * 1000.0;
    let candidate_range_rows = candidate_ranges.values().map(Vec::len).sum::<usize>();
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
                            merged_reset_state,
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
                merged_reset_state,
            );
            candidate_start_visits.fetch_add(candidate_groups.len(), Ordering::Relaxed);
            max_candidate_starts.fetch_max(candidate_groups.len(), Ordering::Relaxed);
            let starts_by_scan = scan_component_residual_start_groups(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                &reset_live_bytes,
                merged_reset_state,
                bytes,
                &candidate_groups,
            );
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
                    seed_terminals,
                    ignore_terminals,
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
                local_witnesses.push(BoundaryTokenWitness {
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
    let mut boundary_token_ids = Vec::with_capacity(results.len());
    let mut witnesses = Vec::new();
    for (token_id, terminals, mut token_witnesses) in results {
        boundary_token_ids.push(token_id);
        for terminal in terminals {
            discovered.set(terminal as usize);
        }
        witnesses.append(&mut token_witnesses);
    }
    if std::env::var_os("GLRMASK_DUMP_COMPOSE_BOUNDARY_TOKENS").is_some() {
        eprintln!(
            "[glrmask/dump][constraint_boundary_tokens] prefilter={:?} exact={:?}",
            prefilter,
            boundary_token_ids,
        );
    }
    if compose_profile_enabled() {
        let exact = boundary_token_ids.iter().copied().collect::<BTreeSet<_>>();
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
    BoundaryTokenDiscovery {
        terminals: discovered,
        token_ids: boundary_token_ids,
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
    merged_reset_state: u32,
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
                // The merged reset epsilon-dispatches to every component start
                // state. Its exact one-byte relation is the union of those
                // local start-state relations.
                if local_state == local_start {
                    destination
                        .entry(merged_reset_state)
                        .or_default()
                        .extend(tokens);
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
    merged_reset_state: u32,
) -> Result<ManyToOneIdMap, String> {
    let mut state_to_global = vec![u32::MAX; merged_tokenizer_state_count];
    let Some(reset_slot) = state_to_global.get_mut(merged_reset_state as usize) else {
        return Err(format!(
            "merged tokenizer reset state {merged_reset_state} lies outside {merged_tokenizer_state_count} states",
        ));
    };
    *reset_slot = 0;
    let mut global_to_states = vec![vec![merged_reset_state]];
    let mut representatives = vec![merged_reset_state];
    let mut dead_merged_states = Vec::<u32>::new();
    for (component_index, component) in components.iter().enumerate() {
        let state_offset = tokenizer_state_offsets[component_index];
        let mut merged_states_by_tsid =
            vec![Vec::<u32>::new(); component.internal_tsid_to_states.len()];
        for (local_state, &local_tsid) in component.state_to_internal_tsid.iter().enumerate() {
            let merged_state = state_offset
                .checked_add(local_state as u32)
                .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
            if local_tsid == u32::MAX {
                if merged_state == merged_reset_state {
                    return Err("merged tokenizer reset state has no internal TSID".into());
                }
                dead_merged_states.push(merged_state);
                continue;
            }
            let Some(states) = merged_states_by_tsid.get_mut(local_tsid as usize) else {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} maps to TSID {local_tsid} outside {} internal TSIDs",
                    merged_states_by_tsid.len(),
                ));
            };
            if merged_state != merged_reset_state {
                states.push(merged_state);
            }
        }
        for merged_states in merged_states_by_tsid {
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
    if !dead_merged_states.is_empty() {
        dead_merged_states.sort_unstable();
        dead_merged_states.dedup();
        let dead_tsid = global_to_states.len() as u32;
        for &merged_state in &dead_merged_states {
            let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                return Err(format!(
                    "dead component tokenizer state {merged_state} lies outside merged tokenizer",
                ));
            };
            if *slot != u32::MAX {
                return Err(format!(
                    "dead component tokenizer state {merged_state} was also assigned a live TSID",
                ));
            }
            *slot = dead_tsid;
        }
        representatives.push(dead_merged_states[0]);
        global_to_states.push(dead_merged_states);
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
) -> Result<InternalIdMap, String> {
    if selected_original_tokens.is_empty() {
        return Err("boundary witness construction selected no model tokens".into());
    }
    let max_original_token = selected_original_tokens.last().copied().unwrap_or(0);
    let mut original_to_internal = vec![u32::MAX; max_original_token as usize + 1];
    let mut internal_to_originals = Vec::with_capacity(selected_original_tokens.len());
    let mut token_representatives = Vec::with_capacity(selected_original_tokens.len());
    for (internal, &original) in selected_original_tokens.iter().enumerate() {
        original_to_internal[original as usize] = internal as u32;
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
    discovery: &BoundaryTokenDiscovery,
    globally_erasable_ignore_terminals: &BitSet,
    control_terminals: &BTreeSet<u32>,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    let total_started_at = Instant::now();

    let selected_original_tokens = seed_relations
        .values()
        .flat_map(|by_state| by_state.values())
        .flat_map(|tokens| tokens.iter().copied())
        .chain(discovery.token_ids.iter().copied())
        .collect::<BTreeSet<_>>();
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
                if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                    // Ignore bytes advance the token-local lexer path but have
                    // no parser-stack effect in every scope. Explicit child
                    // controls do not change that fact: a genuinely global
                    // ignore commutes with entry and return and is erased by the
                    // ordinary terminal-DWA pipeline in exactly this way.
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

    for &control in control_terminals {
        nwa.add_transition(
            global_start,
            control as i32,
            global_start,
            Weight::all(),
        );
        nwa.add_transition(
            seed_final,
            control as i32,
            seed_final,
            Weight::all(),
        );
    }

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
        for &control in control_terminals {
            nwa.add_transition(source, control as i32, source, Weight::all());
        }
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

fn add_control_loops_to_terminal_artifact(
    artifact: MappedArtifact<TerminalAutomaton>,
    control_terminals: &BTreeSet<u32>,
) -> MappedArtifact<TerminalAutomaton> {
    if control_terminals.is_empty() {
        return artifact;
    }
    let (automaton, id_map) = artifact.into_parts();
    let (mut nwa, had_epsilon) = match automaton {
        TerminalAutomaton::Dwa(dwa) => (dwa.to_nwa(), false),
        TerminalAutomaton::TokenDeterministicNwa(nwa) => (nwa, false),
        TerminalAutomaton::EpsilonNwa(nwa) => (nwa, true),
    };
    for state in 0..nwa.num_states() {
        for &control in control_terminals {
            nwa.add_transition(state, control as i32, state, Weight::all());
        }
    }
    MappedArtifact::new(
        if had_epsilon {
            TerminalAutomaton::EpsilonNwa(nwa)
        } else {
            TerminalAutomaton::TokenDeterministicNwa(nwa)
        },
        id_map,
    )
}

/// Add exact out-of-vocabulary special-token branches to the boundary terminal
/// artifact. Byte-token boundary discovery cannot see these IDs because they
/// are deliberately absent from `Vocab`, but parser-DWA construction still
/// needs their terminal effect when a linker control closure precedes or
/// follows the special terminal.
///
/// Existing internal token IDs are preserved and special IDs are appended as
/// singleton classes, so no existing terminal-automaton weight needs remapping.
fn add_boundary_special_token_paths(
    artifact: MappedArtifact<TerminalAutomaton>,
    special_token_terminals: &[SpecialTokenTerminal],
    raw_source_state: u32,
    fallback_state_map: Option<&ManyToOneIdMap>,
    control_terminals: &BTreeSet<u32>,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    if special_token_terminals.is_empty() {
        return Ok(artifact);
    }

    let (automaton, mut id_map) = artifact.into_parts();
    let source_tsid = id_map
        .tokenizer_states
        .original_to_internal
        .get(raw_source_state as usize)
        .copied()
        .filter(|&tsid| tsid != u32::MAX)
        .or_else(|| {
            fallback_state_map
                .and_then(|map| map.original_to_internal.get(raw_source_state as usize))
                .copied()
                .filter(|&tsid| tsid != u32::MAX)
        })
        .ok_or_else(|| {
            format!(
                "boundary special-token source state {raw_source_state} has no tokenizer-state coordinate",
            )
        })?;

    let mut unique = special_token_terminals.to_vec();
    unique.sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    unique.dedup_by_key(|special| (special.token_id, special.terminal_id));

    let max_special_id = unique
        .iter()
        .map(|special| special.token_id)
        .max()
        .unwrap_or(0);
    if id_map.vocab_tokens.original_to_internal.len() <= max_special_id as usize {
        id_map
            .vocab_tokens
            .original_to_internal
            .resize(max_special_id as usize + 1, u32::MAX);
    }
    for special in &unique {
        let slot = &mut id_map.vocab_tokens.original_to_internal[special.token_id as usize];
        if *slot == u32::MAX {
            *slot = id_map.vocab_tokens.internal_to_originals.len() as u32;
            id_map
                .vocab_tokens
                .internal_to_originals
                .push(vec![special.token_id]);
            id_map
                .vocab_tokens
                .representative_original_ids
                .push(special.token_id);
        }
    }

    let mut nwa = match automaton {
        TerminalAutomaton::Dwa(dwa) => dwa.to_nwa(),
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => nwa,
    };
    let starts = nwa.start_states().to_vec();
    if starts.is_empty() {
        return Err("boundary terminal automaton has no start state".into());
    }
    let special_final = nwa.add_state();
    nwa.set_final_weight(special_final, Weight::all());
    for &control in control_terminals {
        nwa.add_transition(
            special_final,
            control as i32,
            special_final,
            Weight::all(),
        );
    }
    for special in unique {
        let internal_token = id_map
            .internal_token_for_original(special.token_id)
            .expect("special token was inserted into the boundary token coordinate");
        let weight = Weight::from_per_tsid_token_sets(std::iter::once((
            source_tsid,
            RangeSetBlaze::from_iter([internal_token]),
        )));
        for &start in &starts {
            nwa.add_transition(
                start,
                special.terminal_id as i32,
                special_final,
                weight.clone(),
            );
        }
    }

    // This may overlap a byte-token branch for an ID that intentionally has
    // both exact-special and byte semantics, so retain the general NWA form.
    Ok(MappedArtifact::new(
        TerminalAutomaton::EpsilonNwa(nwa),
        id_map,
    ))
}

#[derive(Clone)]
struct RawTransitionRun {
    start: i32,
    end: i32,
    targets: Vec<(u32, Weight)>,
}

#[derive(Clone)]
struct RawCompressedState {
    runs: Vec<RawTransitionRun>,
    default_targets: Option<Vec<(u32, Weight)>>,
    final_weight: Option<Weight>,
    deterministic: bool,
}

#[derive(Clone)]
struct RawCompressedAutomaton {
    states: Vec<RawCompressedState>,
    start_states: Vec<u32>,
}

fn same_raw_targets(left: &[(u32, Weight)], right: &[(u32, Weight)]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((left_target, left_weight), (right_target, right_weight))| {
                left_target == right_target && left_weight.ptr_key() == right_weight.ptr_key()
            })
}

impl RawCompressedAutomaton {
    fn from_nwa(automaton: NWA) -> Self {
        let (states, start_states) = automaton.into_parts();
        let states = states
            .into_par_iter()
            .map(|state| {
                debug_assert!(state.epsilons.is_empty());
                let mut runs = Vec::<RawTransitionRun>::new();
                let mut default_targets = None;
                let mut deterministic = true;
                for (label, targets) in state.transitions {
                    deterministic &= targets.len() <= 1;
                    if targets.is_empty() {
                        continue;
                    }
                    if label == DEFAULT_LABEL {
                        default_targets = Some(targets);
                        continue;
                    }
                    if let Some(last) = runs.last_mut()
                        && last.end.checked_add(1) == Some(label)
                        && same_raw_targets(&last.targets, &targets)
                    {
                        last.end = label;
                    } else {
                        runs.push(RawTransitionRun {
                            start: label,
                            end: label,
                            targets,
                        });
                    }
                }
                RawCompressedState {
                    runs,
                    default_targets,
                    final_weight: state.final_weight,
                    deterministic,
                }
            })
            .collect();
        Self {
            states,
            start_states,
        }
    }

    fn from_dwa_preserving_defaults(automaton: &DWA) -> Self {
        let states = automaton
            .states()
            .par_iter()
            .map(|state| {
                let mut runs = Vec::<RawTransitionRun>::new();
                let mut default_targets = None;
                for (&label, (target, weight)) in &state.transitions {
                    let targets = vec![(*target, weight.clone())];
                    if label == DEFAULT_LABEL {
                        default_targets = Some(targets);
                        continue;
                    }
                    if let Some(last) = runs.last_mut()
                        && last.end.checked_add(1) == Some(label)
                        && same_raw_targets(&last.targets, &targets)
                    {
                        last.end = label;
                    } else {
                        runs.push(RawTransitionRun {
                            start: label,
                            end: label,
                            targets,
                        });
                    }
                }
                RawCompressedState {
                    runs,
                    default_targets,
                    final_weight: state.final_weight.clone(),
                    deterministic: true,
                }
            })
            .collect();
        Self {
            states,
            start_states: vec![automaton.start_state()],
        }
    }

    fn to_nwa(&self) -> NWA {
        let states = self
            .states
            .iter()
            .map(|state| {
                let mut transitions = BTreeMap::<i32, Vec<(u32, Weight)>>::new();
                for run in &state.runs {
                    for label in run.start..=run.end {
                        transitions.insert(label, run.targets.clone());
                    }
                }
                if let Some(default_targets) = &state.default_targets {
                    transitions.insert(DEFAULT_LABEL, default_targets.clone());
                }
                NWAState {
                    final_weight: state.final_weight.clone(),
                    transitions,
                    epsilons: Vec::new(),
                }
            })
            .collect();
        NWA::from_parts(states, self.start_states.clone())
    }
}

impl WeightRefs for RawCompressedAutomaton {
    fn weight_refs(&self) -> Vec<&Weight> {
        let mut weights = Vec::new();
        for state in &self.states {
            if let Some(weight) = &state.final_weight {
                weights.push(weight);
            }
            for run in &state.runs {
                weights.extend(run.targets.iter().map(|(_, weight)| weight));
            }
            if let Some(targets) = &state.default_targets {
                weights.extend(targets.iter().map(|(_, weight)| weight));
            }
        }
        weights
    }

    fn weight_refs_mut(&mut self) -> Vec<&mut Weight> {
        let mut weights = Vec::new();
        for state in &mut self.states {
            if let Some(weight) = &mut state.final_weight {
                weights.push(weight);
            }
            for run in &mut state.runs {
                weights.extend(run.targets.iter_mut().map(|(_, weight)| weight));
            }
            if let Some(targets) = &mut state.default_targets {
                weights.extend(targets.iter_mut().map(|(_, weight)| weight));
            }
        }
        weights
    }
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
    Some(determinize_epsilon_free_component_union_prechecked(
        automata,
        default_positive_label_count,
    ))
}

fn determinize_epsilon_free_component_union_prechecked(
    automata: Vec<NWA>,
    default_positive_label_count: Option<u32>,
) -> (DWA, usize) {
    debug_assert!(supports_overlap_local_union(&automata));
    let compress_started_at = Instant::now();
    let compressed = automata
        .into_par_iter()
        .map(RawCompressedAutomaton::from_nwa)
        .collect::<Vec<_>>();
    let raw_compress_ms = compress_started_at.elapsed().as_secs_f64() * 1000.0;
    determinize_compressed_component_union_prechecked(
        compressed,
        default_positive_label_count,
        raw_compress_ms,
    )
}

fn determinize_compressed_component_union_prechecked(
    automata: Vec<RawCompressedAutomaton>,
    default_positive_label_count: Option<u32>,
    raw_compress_ms: f64,
) -> (DWA, usize) {
    let total_started_at = Instant::now();

    struct OutputTransitionRun {
        start: i32,
        end: i32,
        target: u32,
        weight: Weight,
    }

    #[derive(Default)]
    struct PendingDwaState {
        runs: Vec<OutputTransitionRun>,
        default_transition: Option<(u32, Weight)>,
        final_weight: Option<Weight>,
        deferred_final_pairs: SmallVec<[(usize, usize); 4]>,
    }

    let mut next_offset = 0u32;
    let mut starts = Vec::new();
    let components = automata
        .into_iter()
        .map(|mut automaton| {
            let offset = next_offset;
            next_offset = next_offset
                .checked_add(automaton.states.len() as u32)
                .expect("component parser-DWA state count overflow");
            starts.extend(
                automaton
                    .start_states
                    .iter()
                    .map(|state| offset + *state),
            );
            for state in &mut automaton.states {
                for run in &mut state.runs {
                    for (target, _) in &mut run.targets {
                        *target += offset;
                    }
                }
                if let Some(targets) = &mut state.default_targets {
                    for (target, _) in targets {
                        *target += offset;
                    }
                }
            }
            automaton.states
        })
        .collect::<Vec<_>>();
    let mut raw_states = Vec::<RawCompressedState>::with_capacity(next_offset as usize);
    for states in components {
        raw_states.extend(states);
    }
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return (DWA::new(0, 0), 0);
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
        states: &mut Vec<PendingDwaState>,
        queue: &mut VecDeque<(u32, ResidualSubset)>,
    ) -> u32 {
        let slot = &mut singleton_states[raw_state as usize];
        if *slot != u32::MAX {
            return *slot;
        }
        let created = states.len() as u32;
        states.push(PendingDwaState::default());
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
        states: &mut Vec<PendingDwaState>,
        queue: &mut VecDeque<(u32, ResidualSubset)>,
        subset_states: &mut FxHashMap<ResidualSubsetKey, u32>,
    ) -> Option<(u32, Weight)> {
        let finish_subset = |
            normalized: ResidualSubset,
            edge_weight: Weight,
            singleton_states: &mut [u32],
            singleton_count: &mut usize,
            states: &mut Vec<PendingDwaState>,
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
                states.push(PendingDwaState::default());
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

    let mut states = Vec::<PendingDwaState>::new();
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
        states.push(PendingDwaState::default());
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
    let mut profiled_output_transitions = 0usize;
    let mut profiled_contribution_candidates = 0usize;
    let mut profiled_nonempty_contributions = 0usize;
    let profile_union_detail = compose_profile_enabled();
    let mut profiled_singleton_ms = 0.0f64;
    let mut profiled_pair_ms = 0.0f64;
    let mut profiled_wide_ms = 0.0f64;
    let mut profiled_boundary_prep_ms = 0.0f64;
    let mut profiled_interval_ms = 0.0f64;
    let mut profiled_default_ms = 0.0f64;
    let mut profiled_row_finalize_ms = 0.0f64;
    let mut profiled_final_candidates = 0usize;
    let mut profiled_final_nonempty = 0usize;
    let mut profiled_final_full_prefix = 0usize;
    let mut profiled_final_full_source = 0usize;
    let mut profiled_final_ptr_equal = 0usize;
    let mut profiled_final_states_zero = 0usize;
    let mut profiled_final_states_one = 0usize;
    let mut profiled_final_states_two = 0usize;
    let mut profiled_final_states_wide = 0usize;
    let mut profiled_final_prefix_ranges = 0usize;
    let mut profiled_final_source_ranges = 0usize;
    let mut final_intersection_inputs =
        FxHashMap::<(usize, usize), (Weight, Weight)>::default();
    let subset_started_at = Instant::now();
    while let Some((output_state, subset)) = queue.pop_front() {
        let state_started_at = profile_union_detail.then(Instant::now);
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
            if source.deterministic {
                let final_weight = source
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                let mut runs = Vec::with_capacity(source.runs.len());
                for raw_run in &source.runs {
                    let targets = &raw_run.targets;
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
                    profiled_output_transitions +=
                        (i64::from(raw_run.end) - i64::from(raw_run.start) + 1) as usize;
                    runs.push(OutputTransitionRun {
                        start: raw_run.start,
                        end: raw_run.end,
                        target,
                        weight: edge_weight.clone(),
                    });
                }
                let default_transition = source
                    .default_targets
                    .as_ref()
                    .and_then(|targets| targets.first())
                    .filter(|(_, weight)| !weight.is_empty())
                    .map(|(target, weight)| {
                        let target = intern_singleton(
                            *target,
                            &mut singleton_states,
                            &mut singleton_count,
                            &mut states,
                            &mut queue,
                        );
                        profiled_output_transitions += 1;
                        (target, weight.clone())
                    });
                states[output_state as usize] = PendingDwaState {
                    runs,
                    default_transition,
                    final_weight,
                    deferred_final_pairs: SmallVec::new(),
                };
                if let Some(state_started_at) = state_started_at {
                    profiled_singleton_ms +=
                        state_started_at.elapsed().as_secs_f64() * 1000.0;
                }
                continue;
            }
        }

        let mut deferred_final_pairs = SmallVec::<[(usize, usize); 4]>::new();
        for (raw_state, prefix_weight) in &subset {
            let source = &raw_states[*raw_state as usize];
            if let Some(final_weight) = &source.final_weight {
                if profile_union_detail {
                    profiled_final_candidates += 1;
                    profiled_final_full_prefix += usize::from(prefix_weight.is_full());
                    profiled_final_full_source += usize::from(final_weight.is_full());
                    profiled_final_ptr_equal +=
                        usize::from(prefix_weight.ptr_key() == final_weight.ptr_key());
                    profiled_final_prefix_ranges += prefix_weight.num_ranges();
                    profiled_final_source_ranges += final_weight.num_ranges();
                }
                let left = prefix_weight.ptr_key();
                let right = final_weight.ptr_key();
                let key = if left <= right {
                    (left, right)
                } else {
                    (right, left)
                };
                final_intersection_inputs
                    .entry(key)
                    .or_insert_with(|| (prefix_weight.clone(), final_weight.clone()));
                deferred_final_pairs.push(key);
            }
        }
        if deferred_final_pairs.is_empty() {
            profiled_final_states_zero += 1;
        }
        // Keep wildcard transitions symbolic. Process one label at a time so
        // synthetic rows do not allocate a nested label->target hash table and
        // sort it again afterward. Raw rows are deterministic in the common
        // path; the small target vector also handles genuine NWA overlap.
        let mut default_transition = None::<(u32, Weight)>;

        // Sweep maximal intervals on which every constituent row has the same
        // explicit/default transition. The old implementation repeated the
        // same weighted subset construction once per parser-state label; real
        // composed rows contain millions of adjacent labels with identical
        // outcomes. Run-wise evaluation preserves the exact explicit/default
        // semantics and expands to the ordinary DWA map only after each result
        // interval has been solved once.
        let boundary_prep_started_at = profile_union_detail.then(Instant::now);
        let mut boundaries = SmallVec::<[i64; 64]>::new();
        let mut has_default = false;
        for (raw_state, _) in &subset {
            for run in &raw_states[*raw_state as usize].runs {
                boundaries.push(i64::from(run.start));
                boundaries.push(i64::from(run.end) + 1);
            }
            has_default |= raw_states[*raw_state as usize].default_targets.is_some();
        }
        if has_default && default_positive_label_count.is_some() {
            // Explicit negative labels never fall through to DEFAULT_LABEL;
            // positive labels do. Split a run interval at that semantic edge.
            boundaries.push(0);
        }
        boundaries.sort_unstable();
        boundaries.dedup();
        if let Some(boundary_prep_started_at) = boundary_prep_started_at {
            profiled_boundary_prep_ms +=
                boundary_prep_started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let mut run_positions = vec![0usize; subset.len()];
        let mut output_runs = Vec::<OutputTransitionRun>::new();
        let interval_started_at = profile_union_detail.then(Instant::now);
        for interval in boundaries.windows(2) {
            let start = interval[0];
            let end = interval[1] - 1;
            if start > end || start < i64::from(i32::MIN) || end > i64::from(i32::MAX) {
                continue;
            }
            let label = start as i32;
            let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
            let mut has_explicit = false;
            for (subset_index, (raw_state, prefix_weight)) in subset.iter().enumerate() {
                let source = &raw_states[*raw_state as usize];
                let runs = &source.runs;
                let position = &mut run_positions[subset_index];
                while *position < runs.len() && runs[*position].end < label {
                    *position += 1;
                }
                let explicit = runs
                    .get(*position)
                    .filter(|run| run.start <= label && label <= run.end);
                has_explicit |= explicit.is_some();
                let targets = explicit.map(|run| run.targets.as_slice()).or_else(|| {
                    (label >= 0)
                        .then(|| source.default_targets.as_deref())
                        .flatten()
                });
                let Some(targets) = targets else {
                    continue;
                };
                for (target, edge_weight) in targets {
                    profiled_contribution_candidates += 1;
                    let contribution = weight_ops.intersection(prefix_weight, edge_weight);
                    if !contribution.is_empty() {
                        profiled_nonempty_contributions += 1;
                        contributions.push((*target, contribution));
                    }
                }
            }
            // A gap covered only by defaults is represented by the one
            // DEFAULT_LABEL cell below, not by redundant explicit labels.
            if !has_explicit {
                continue;
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
                let start = start as i32;
                let end = end as i32;
                if let Some(previous) = output_runs.last_mut()
                    && previous.end.checked_add(1) == Some(start)
                    && previous.target == target
                    && previous.weight.ptr_key() == edge_weight.ptr_key()
                {
                    previous.end = end;
                } else {
                    output_runs.push(OutputTransitionRun {
                        start,
                        end,
                        target,
                        weight: edge_weight,
                    });
                }
            }
        }
        if let Some(interval_started_at) = interval_started_at {
            profiled_interval_ms += interval_started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let default_started_at = profile_union_detail.then(Instant::now);
        if default_positive_label_count.is_some() && has_default {
            let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
            for (raw_state, prefix_weight) in &subset {
                let Some(targets) = raw_states[*raw_state as usize].default_targets.as_ref()
                else {
                    continue;
                };
                for (target, edge_weight) in targets {
                    profiled_contribution_candidates += 1;
                    let contribution = weight_ops.intersection(prefix_weight, edge_weight);
                    if !contribution.is_empty() {
                        profiled_nonempty_contributions += 1;
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
                default_transition = Some((target, edge_weight));
            }
        }
        if let Some(default_started_at) = default_started_at {
            profiled_default_ms += default_started_at.elapsed().as_secs_f64() * 1000.0;
        }

        let row_finalize_started_at = profile_union_detail.then(Instant::now);
        let explicit_transition_count = output_runs
            .iter()
            .map(|run| (i64::from(run.end) - i64::from(run.start) + 1) as usize)
            .sum::<usize>();
        profiled_explicit_labels += explicit_transition_count;
        profiled_output_transitions +=
            explicit_transition_count + usize::from(default_transition.is_some());
        states[output_state as usize] = PendingDwaState {
            runs: output_runs,
            default_transition,
            final_weight: None,
            deferred_final_pairs,
        };
        if let Some(row_finalize_started_at) = row_finalize_started_at {
            profiled_row_finalize_ms +=
                row_finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        }
        if let Some(state_started_at) = state_started_at {
            let elapsed = state_started_at.elapsed().as_secs_f64() * 1000.0;
            if subset.len() == 2 {
                profiled_pair_ms += elapsed;
            } else {
                profiled_wide_ms += elapsed;
            }
        }
    }
    let subset_ms = subset_started_at.elapsed().as_secs_f64() * 1000.0;

    // Final weights are independent once the residual subset graph is fixed.
    // Deduplicate every `(prefix, raw-final)` intersection globally, evaluate
    // those unique pairs in parallel, then deduplicate the resulting union
    // recipes before assigning them back to states. This preserves the shared
    // work of the serial scoped cache without forcing all expensive range-map
    // operations through one thread.
    let final_weight_started_at = Instant::now();
    let final_intersection_count = final_intersection_inputs.len();
    let final_intersection_results = final_intersection_inputs
        .into_par_iter()
        .map(|(key, (left, right))| (key, left.intersection_uncached(&right)))
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<FxHashMap<_, _>>();

    type FinalUnionKey = SmallVec<[usize; 4]>;
    let mut state_union_keys = vec![None::<FinalUnionKey>; states.len()];
    let mut final_union_inputs = FxHashMap::<FinalUnionKey, SmallVec<[Weight; 4]>>::default();
    for (state_id, state) in states.iter_mut().enumerate() {
        if state.deferred_final_pairs.is_empty() {
            continue;
        }
        let mut contributions = state
            .deferred_final_pairs
            .iter()
            .filter_map(|key| final_intersection_results.get(key))
            .filter(|weight| !weight.is_empty())
            .cloned()
            .collect::<SmallVec<[Weight; 4]>>();
        contributions.sort_unstable_by_key(Weight::ptr_key);
        contributions.dedup_by_key(|weight| weight.ptr_key());
        profiled_final_nonempty += contributions.len();
        match contributions.len() {
            0 => profiled_final_states_zero += 1,
            1 => {
                profiled_final_states_one += 1;
                state.final_weight = contributions.pop();
            }
            2 => {
                profiled_final_states_two += 1;
                let key = contributions
                    .iter()
                    .map(Weight::ptr_key)
                    .collect::<FinalUnionKey>();
                final_union_inputs
                    .entry(key.clone())
                    .or_insert(contributions);
                state_union_keys[state_id] = Some(key);
            }
            _ => {
                profiled_final_states_wide += 1;
                let key = contributions
                    .iter()
                    .map(Weight::ptr_key)
                    .collect::<FinalUnionKey>();
                final_union_inputs
                    .entry(key.clone())
                    .or_insert(contributions);
                state_union_keys[state_id] = Some(key);
            }
        }
        state.deferred_final_pairs.clear();
    }
    let final_union_count = final_union_inputs.len();
    let final_union_results = final_union_inputs
        .into_par_iter()
        .map(|(key, weights)| {
            let mut ops = ScopedWeightOpCache::default();
            let weight = ops.union_all(weights.iter());
            (key, weight)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<FxHashMap<_, _>>();
    for (state, key) in states.iter_mut().zip(state_union_keys) {
        if let Some(key) = key {
            let weight = final_union_results
                .get(&key)
                .expect("final union recipe was evaluated")
                .clone();
            state.final_weight = (!weight.is_empty()).then_some(weight);
        }
    }
    let profiled_final_weight_ms = final_weight_started_at.elapsed().as_secs_f64() * 1000.0;

    let synthetic_states = states.len().saturating_sub(singleton_count);
    let mut outcome_groups = 0usize;
    let mut contiguous_runs = 0usize;
    let mut max_outcome_groups = 0usize;
    let mut max_contiguous_runs = 0usize;
    if compose_profile_enabled() {
        for state in &states {
            let mut outcomes = FxHashSet::<(u32, usize)>::default();
            for run in &state.runs {
                outcomes.insert((run.target, run.weight.ptr_key()));
            }
            if let Some((target, weight)) = &state.default_transition {
                outcomes.insert((*target, weight.ptr_key()));
            }
            let row_runs = state.runs.len() + usize::from(state.default_transition.is_some());
            outcome_groups += outcomes.len();
            contiguous_runs += row_runs;
            max_outcome_groups = max_outcome_groups.max(outcomes.len());
            max_contiguous_runs = max_contiguous_runs.max(row_runs);
        }
    }

    let materialize_started_at = Instant::now();
    let states = states
        .into_par_iter()
        .map(|state| {
            let transition_count = state
                .runs
                .iter()
                .map(|run| (i64::from(run.end) - i64::from(run.start) + 1) as usize)
                .sum::<usize>()
                + usize::from(state.default_transition.is_some());
            let mut entries = Vec::with_capacity(transition_count);
            if let Some((target, weight)) = state.default_transition {
                entries.push((DEFAULT_LABEL, (target, weight)));
            }
            for run in state.runs {
                for label in run.start..=run.end {
                    entries.push((label, (run.target, run.weight.clone())));
                }
            }
            DWAState {
                transitions: entries.into_iter().collect(),
                final_weight: state.final_weight,
            }
        })
        .collect::<Vec<_>>();
    let materialize_ms = materialize_started_at.elapsed().as_secs_f64() * 1000.0;

    if compose_profile_enabled() {
        let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glrmask/profile][constraint_overlap_local_shape] raw_states={} result_states={} singleton_states={} pair_subsets={} wide_subsets={} max_subset={} explicit_labels={} output_transitions={} contribution_candidates={} nonempty_contributions={} outcome_groups={} contiguous_runs={} max_outcome_groups={} max_contiguous_runs={} final_candidates={} final_nonempty={} final_full_prefix={} final_full_source={} final_ptr_equal={} final_states_zero={} final_states_one={} final_states_two={} final_states_wide={} final_unique_intersections={} final_unique_unions={} final_prefix_ranges={} final_source_ranges={} scoped_intersection_cache={} raw_compress_ms={raw_compress_ms:.3} subset_ms={subset_ms:.3} singleton_ms={profiled_singleton_ms:.3} pair_ms={profiled_pair_ms:.3} wide_ms={profiled_wide_ms:.3} final_weight_ms={profiled_final_weight_ms:.3} boundary_prep_ms={profiled_boundary_prep_ms:.3} interval_ms={profiled_interval_ms:.3} default_ms={profiled_default_ms:.3} row_finalize_ms={profiled_row_finalize_ms:.3} materialize_ms={materialize_ms:.3} total_ms={total_ms:.3}",
            raw_states.len(),
            states.len(),
            profiled_singletons,
            profiled_pair_subsets,
            profiled_wide_subsets,
            profiled_max_subset,
            profiled_explicit_labels,
            profiled_output_transitions,
            profiled_contribution_candidates,
            profiled_nonempty_contributions,
            outcome_groups,
            contiguous_runs,
            max_outcome_groups,
            max_contiguous_runs,
            profiled_final_candidates,
            profiled_final_nonempty,
            profiled_final_full_prefix,
            profiled_final_full_source,
            profiled_final_ptr_equal,
            profiled_final_states_zero,
            profiled_final_states_one,
            profiled_final_states_two,
            profiled_final_states_wide,
            final_intersection_count,
            final_union_count,
            profiled_final_prefix_ranges,
            profiled_final_source_ranges,
            weight_ops.intersection_entry_count(),
        );
    }
    (DWA::from_parts(states, start_state), synthetic_states)
}


struct UnmappedComponentParserArtifact {
    automaton: RawCompressedAutomaton,
    possible_matches: PossibleMatches,
    top_accept_parts: BTreeMap<i32, Vec<Weight>>,
}

fn prepare_unmapped_component_parser_artifacts(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
    strip_scoped_ignore_identity: bool,
    transport_top_accept_directly: bool,
) -> Result<Vec<UnmappedComponentParserArtifact>, String> {
    if components.len() != terminal_offsets.len() {
        return Err("component/parser terminal-offset count mismatch".into());
    }
    components
        .par_iter()
        .copied()
        .zip(terminal_offsets.par_iter().copied())
        .map(|(component, terminal_offset)| {
            let possible_matches = component_possible_matches(&component, terminal_offset)?;
            let mut automaton = if transport_top_accept_directly {
                component_parser_compressed(&component)?
            } else {
                RawCompressedAutomaton::from_nwa(component_parser_nwa_with_top_accept(
                    &component,
                    true,
                )?)
            };
            let top_accept_parts = if transport_top_accept_directly {
                transported_component_top_accept_parts(&component)?
            } else {
                BTreeMap::new()
            };
            if strip_scoped_ignore_identity {
                let ignore_weight = component
                    .constraint
                    .ignore_terminal
                    .and_then(|ignore| possible_matches.get(&(terminal_offset + ignore)));
                strip_unscoped_ignore_identity_compressed(&mut automaton, ignore_weight);
            }
            Ok(UnmappedComponentParserArtifact {
                automaton,
                possible_matches,
                top_accept_parts,
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
    automata: Vec<RawCompressedAutomaton>,
    possible_matches: PossibleMatches,
    top_accept_parts: BTreeMap<i32, Vec<Weight>>,
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
) -> Result<(
    Vec<RawCompressedAutomaton>,
    PossibleMatches,
    BTreeMap<i32, Vec<Weight>>,
    f64,
), String> {
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
            let mut triple = (
                artifact.automaton,
                artifact.possible_matches,
                artifact.top_accept_parts,
            );
            let mut weights = triple.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &maps.local_to_global_tsids,
                &token_map,
                common_tsid_count,
            );
            Ok::<_, String>(triple)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut automata = Vec::with_capacity(remapped.len());
    let mut possible_matches = PossibleMatches::new();
    let mut top_accept_parts = BTreeMap::<i32, Vec<Weight>>::new();
    for (automaton, component_possible_matches, component_top_accept_parts) in remapped {
        automata.push(automaton);
        for (terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
        for (label, parts) in component_top_accept_parts {
            top_accept_parts.entry(label).or_default().extend(parts);
        }
    }
    for parts in top_accept_parts.values_mut() {
        parts.sort_unstable_by_key(Weight::ptr_key);
        parts.dedup_by_key(|weight| weight.ptr_key());
    }
    Ok((
        automata,
        possible_matches,
        top_accept_parts,
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
    merged_reset_state: u32,
    original_token_ids: &[u32],
    strip_scoped_ignore_identity: bool,
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
        merged_reset_state,
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
            let mut parser_nwa = component_parser_nwa(&component)?;
            let parser_nwa_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            let started_at = Instant::now();
            let possible_matches = component_possible_matches(&component, terminal_offset)?;
            let possible_matches_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            if strip_scoped_ignore_identity {
                let ignore_weight = component
                    .constraint
                    .ignore_terminal
                    .and_then(|ignore| possible_matches.get(&(terminal_offset + ignore)));
                strip_unscoped_ignore_identity(&mut parser_nwa, ignore_weight);
            }
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
    // This is a graph-preserving DWA -> NWA view: no epsilon edges are added
    // and transition targets are unchanged. Callers may therefore retain the
    // source parser DWA's acyclic certification.
    debug_assert!(dwa.is_acyclic());
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
) -> (Templates, f64) {
    let started_at = Instant::now();
    let direct_started_at = Instant::now();
    let (mut templates, direct_eligible) =
        Templates::from_individually_direct_regular_terminals(table, selected);
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    let generic_selected = selected
        .iter()
        .zip(&direct_eligible)
        .map(|(&selected, &direct)| selected && !direct)
        .collect::<Vec<_>>();
    let (characterizations, characterization_profile) =
        characterize_selected_terminals_profiled(table, analyzed, &generic_selected);
    let (generic_templates, template_profile) =
        Templates::from_characterizations_profiled(&characterizations);
    templates.by_terminal.extend(generic_templates.by_terminal);
    templates
        .by_terminal_nwa
        .extend(generic_templates.by_terminal_nwa);
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_direct_templates] selected={} direct={} generic={} total_ms={direct_ms:.3}",
            selected.iter().filter(|&&selected| selected).count(),
            direct_eligible.iter().filter(|&&direct| direct).count(),
            generic_selected.iter().filter(|&&selected| selected).count(),
        );
        eprintln!(
            "[glrmask/profile][constraint_boundary_characterization] terminals={} unique_action_signatures={} max_signature_multiplicity={} quotient_hits={} signature_ms={:.3} characterize_ms={:.3} fanout_ms={:.3} validation_ms={:.3} total_ms={:.3}",
            characterization_profile.terminals,
            characterization_profile.unique_action_signatures,
            characterization_profile.max_action_signature_multiplicity,
            characterization_profile.quotient_hits,
            characterization_profile.signature_ms,
            characterization_profile.characterize_ms,
            characterization_profile.fanout_ms,
            characterization_profile.validation_ms,
            characterization_profile.total_ms,
        );
        eprintln!(
            "[glrmask/profile][constraint_boundary_template_compile] terminals={} unique={} compiled={} quotient_hits={} build_nfa_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} fanout_ms={:.3} validation_ms={:.3} profile_total_ms={:.3} wall_ms={:.3} nfa_states={} premin_dfa_states={} dfa_states={} max_nfa_states={} max_premin_dfa_states={} max_dfa_states={}",
            template_profile.num_terminals,
            template_profile.unique_characterizations,
            template_profile.compiled_characterizations,
            template_profile.quotient_hits,
            template_profile.build_nfa_ms,
            template_profile.determinize_ms,
            template_profile.minimize_ms,
            template_profile.fanout_ms,
            template_profile.validation_ms,
            template_profile.total_ms,
            template_profile.wall_ms,
            template_profile.total_nfa_states,
            template_profile.total_premin_dfa_states,
            template_profile.total_dfa_states,
            template_profile.max_nfa_states,
            template_profile.max_premin_dfa_states,
            template_profile.max_dfa_states,
        );
    }
    (templates, started_at.elapsed().as_secs_f64() * 1000.0)
}

fn build_composition_commit_templates(
    templates: &Templates,
    num_terminals: usize,
    selected: Option<&[bool]>,
) -> (Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>, f64) {
    let started_at = Instant::now();
    let compiled = templates
        .by_terminal
        .par_iter()
        .filter_map(|(&terminal, dfa)| {
            if selected.is_some_and(|selected| {
                !selected
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)
            }) {
                return None;
            }
            let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
            try_split_commit_template_dfas(&commit_dfa)
                .map(|split| (terminal as usize, Arc::new(split)))
        })
        .collect::<Vec<_>>();
    let mut template_dfas_by_terminal = vec![None; num_terminals];
    for (terminal, split) in compiled {
        if let Some(slot) = template_dfas_by_terminal.get_mut(terminal) {
            *slot = Some(split);
        }
    }
    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_commit_templates] input_templates={} output_templates={} total_ms={elapsed_ms:.3}",
            templates.by_terminal.len(),
            template_dfas_by_terminal.iter().filter(|slot| slot.is_some()).count(),
        );
    }
    (template_dfas_by_terminal, elapsed_ms)
}

fn remap_split_template_dfa(
    dfa: &UnweightedDfa,
    relation: &[Vec<u32>],
) -> Result<UnweightedDfa, String> {
    let mut remapped = dfa.clone();
    for state in &mut remapped.states {
        let mut transitions = BTreeMap::new();
        for (&label, &target) in &state.transitions {
            if label == DEFAULT_LABEL {
                return Err("split commit template retained an unmaterialized default label".into());
            }
            let labels = mapped_labels(label, relation)?;
            if labels.len() != 1 {
                return Err(format!(
                    "split commit template requires one-to-one parser-state transport; label {label} maps to {} states",
                    labels.len(),
                ));
            }
            let mapped = labels[0];
            if let Some(existing) = transitions.insert(mapped, target)
                && existing != target
            {
                return Err(format!(
                    "split commit template label transport collides at {mapped}: {existing} vs {target}",
                ));
            }
        }
        state.transitions = transitions;
    }
    Ok(remapped)
}

fn remap_split_commit_template(
    template: &crate::runtime::CommitTemplateDfas,
    relation: &[Vec<u32>],
) -> Result<crate::runtime::CommitTemplateDfas, String> {
    Ok(crate::runtime::CommitTemplateDfas {
        pop: remap_split_template_dfa(&template.pop, relation)?,
        read: remap_split_template_dfa(&template.read, relation)?,
        push: remap_split_template_dfa(&template.push, relation)?,
        pop_to_read: template.pop_to_read.clone(),
        pop_to_push: template.pop_to_push.clone(),
        read_to_push: template.read_to_push.clone(),
    })
}

fn transport_component_commit_templates(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    rebuild_missing: bool,
) -> (
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    Vec<bool>,
    f64,
) {
    let started_at = Instant::now();
    let mut transported = vec![None; composed_table.table.num_terminals as usize];
    let mut needs_rebuild = vec![false; composed_table.table.num_terminals as usize];
    let mut transported_count = 0usize;
    let mut failed_count = 0usize;
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = composed_table.terminal_offsets[component_index] as usize;
        let relation = &composed_table.state_relations[component_index];
        for (local_terminal, template) in component.template_dfas_by_terminal.iter().enumerate() {
            let global_terminal = terminal_offset + local_terminal;
            let Some(template) = template else {
                if rebuild_missing {
                    needs_rebuild[global_terminal] = true;
                }
                continue;
            };
            match remap_split_commit_template(template, relation) {
                Ok(template) => {
                    transported[global_terminal] = Some(Arc::new(template));
                    transported_count += 1;
                }
                Err(_) => {
                    needs_rebuild[global_terminal] = true;
                    failed_count += 1;
                }
            }
        }
    }
    let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_commit_template_transport] components={} transported={} failed={} missing={} total_ms={elapsed_ms:.3}",
            components.len(),
            transported_count,
            failed_count,
            needs_rebuild.iter().filter(|&&needed| needed).count(),
        );
    }
    (transported, needs_rebuild, elapsed_ms)
}

fn profile_boundary_parser_dwa(dwa: &DWA) {
    if !compose_profile_enabled() {
        return;
    }
    let states = dwa.states();
    let mut positive = 0usize;
    let mut negative = 0usize;
    let mut defaults = 0usize;
    let mut final_states = 0usize;
    let mut indegree = vec![0usize; states.len()];
    for state in states {
        final_states += usize::from(state.final_weight.is_some());
        for (&label, (target, _)) in &state.transitions {
            if label == DEFAULT_LABEL {
                defaults += 1;
            } else if label >= 0 {
                positive += 1;
            } else {
                negative += 1;
            }
            if let Some(degree) = indegree.get_mut(*target as usize) {
                *degree += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state, degree) in indegree.iter().copied().enumerate() {
        if degree == 0 {
            queue.push_back(state);
        }
    }
    let mut topo = Vec::with_capacity(states.len());
    while let Some(source) = queue.pop_front() {
        topo.push(source);
        for &(target, _) in states[source].transitions.values() {
            let degree = &mut indegree[target as usize];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(target as usize);
            }
        }
    }
    let mut height = vec![0usize; states.len()];
    for &source in topo.iter().rev() {
        height[source] = states[source]
            .transitions
            .values()
            .map(|(target, _)| 1 + height[*target as usize])
            .max()
            .unwrap_or(0);
    }
    let mut by_height = BTreeMap::<usize, usize>::new();
    for value in &height {
        *by_height.entry(*value).or_default() += 1;
    }
    eprintln!(
        "[glrmask/profile][constraint_boundary_parser_shape] states={} transitions={} finals={} positive={} negative={} defaults={} acyclic={} max_height={} start_height={} height_hist={:?}",
        states.len(),
        dwa.num_transitions(),
        final_states,
        positive,
        negative,
        defaults,
        topo.len() == states.len(),
        height.iter().copied().max().unwrap_or(0),
        height.get(dwa.start_state() as usize).copied().unwrap_or(0),
        by_height,
    );
}

fn prepare_boundary_lexical_prepass(
    analyzed: &AnalyzedGrammar,
    boundary_nonterminals: &BTreeSet<u32>,
    terminal_offsets: &[u32],
    merged_tokenizer: Option<&Tokenizer>,
    merged_reset_state: u32,
    ignore_terminals: &MergedIgnoreTerminals,
    vocab: &Vocab,
    special_token_terminals: &[SpecialTokenTerminal],
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    selected_boundary_tokens: Option<&OnceLock<Result<Option<Vec<u32>>, String>>>,
) -> Result<Option<BoundaryLexicalPrepass>, String> {
    let mut seed_terminals = vec![false; analyzed.num_terminals as usize];
    for &nonterminal in boundary_nonterminals {
        let Some(first) = analyzed.first.get(nonterminal as usize) else {
            return Err(format!(
                "boundary nonterminal {nonterminal} lies outside composed FIRST analysis",
            ));
        };
        let Some(follow) = analyzed.follow.get(nonterminal as usize) else {
            return Err(format!(
                "boundary nonterminal {nonterminal} lies outside composed FOLLOW analysis",
            ));
        };
        for terminal in first.iter().chain(follow.iter()) {
            if terminal < seed_terminals.len() {
                seed_terminals[terminal] = true;
            }
        }
    }
    for ignore_terminal in ignore_terminals.scoped.iter() {
        if let Some(slot) = seed_terminals.get_mut(ignore_terminal) {
            *slot = true;
        }
    }

    let disallowed_follows = if std::env::var_os("GLRMASK_COMPOSE_DISABLE_GRAMMAR_FOLLOWS")
        .is_some()
    {
        BTreeMap::<u32, BitSet>::new()
    } else {
        crate::compiler::pipeline::compute_disallowed_follows(analyzed)
    };
    let ((boundary_paths, discovery_ms), (seed_relations, one_byte_ms)) = rayon::join(
        || {
            let started_at = Instant::now();
            let boundary_paths = discover_boundary_token_paths(
                vocab,
                components,
                tokenizer_state_offsets,
                merged_reset_state,
                terminal_offsets,
                &seed_terminals,
                &ignore_terminals.all,
                &disallowed_follows,
            );
            (boundary_paths, started_at.elapsed().as_secs_f64() * 1000.0)
        },
        || {
            let started_at = Instant::now();
            let relations = collect_one_byte_seed_relations_components(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                merged_reset_state,
                vocab,
                &seed_terminals,
            );
            if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_BOUNDARY_VIEW").is_some() {
                let mut reference = BoundarySeedRelations::new();
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
    );

    let discovered_boundary_terminals = boundary_paths.terminals.clone();
    let mut active_terminals = seed_terminals.clone();
    for terminal in discovered_boundary_terminals.iter() {
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
            "[glrmask/profile][constraint_boundary_terminals] begin={} discovered={} boundary_tokens={} selected={:?}",
            seed_terminals.iter().filter(|&&selected| selected).count(),
            discovered_boundary_terminals.count_ones(),
            boundary_paths.token_ids.len(),
            selected,
        );
    }

    let mut boundary_special_token_terminals = special_token_terminals
        .iter()
        .copied()
        .filter(|special| {
            active_terminals
                .get(special.terminal_id as usize)
                .copied()
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    boundary_special_token_terminals
        .sort_unstable_by_key(|special| (special.token_id, special.terminal_id));
    boundary_special_token_terminals
        .dedup_by_key(|special| (special.token_id, special.terminal_id));

    let selected_original_tokens = seed_relations
        .values()
        .flat_map(|by_state| by_state.values())
        .flat_map(|tokens| tokens.iter().copied())
        .chain(boundary_paths.token_ids.iter().copied())
        .chain(
            boundary_special_token_terminals
                .iter()
                .map(|special| special.token_id),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if selected_original_tokens.is_empty() {
        let error = "boundary witness construction selected no model tokens".to_string();
        if let Some(selected_boundary_tokens) = selected_boundary_tokens {
            let _ = selected_boundary_tokens.set(Err(error.clone()));
        }
        return Err(error);
    }
    if let Some(selected_boundary_tokens) = selected_boundary_tokens {
        let _ = selected_boundary_tokens.set(Ok(Some(selected_original_tokens)));
    }

    Ok(Some(BoundaryLexicalPrepass {
        seed_terminals,
        disallowed_follows,
        boundary_paths,
        discovery_ms,
        seed_relations,
        one_byte_ms,
        active_terminals,
        boundary_special_token_terminals,
    }))
}

fn build_boundary_repair(
    composed_table: &ComposedTable,
    merged_tokenizer: Option<&Tokenizer>,
    merged_tokenizer_state_count: usize,
    merged_reset_state: u32,
    terminal_display_names: Vec<String>,
    ignore_terminals: &MergedIgnoreTerminals,
    vocab: &Vocab,
    special_token_terminals: &[SpecialTokenTerminal],
    components: &[&Constraint],
    control_changed_terminals: &[u32],
    tokenizer_state_offsets: &[u32],
    precomputed_component_state_map: Option<&ManyToOneIdMap>,
    deferred_component_state_map: Option<&OnceLock<Result<ManyToOneIdMap, String>>>,
    selected_boundary_tokens: Option<&OnceLock<Result<Option<Vec<u32>>, String>>>,
    precomputed_lexical: Option<BoundaryLexicalPrepass>,
    precomputed_analyzed: Option<AnalyzedGrammar>,
) -> Result<Option<BoundaryRepair>, String> {
    let total_started_at = Instant::now();
    let analyzed_started_at = Instant::now();
    let analyzed = if let Some(analyzed) = precomputed_analyzed {
        analyzed
    } else {
        let augmented_start = composed_table
            .table
            .rules
            .first()
            .map(|rule| rule.lhs)
            .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
        AnalyzedGrammar::from_composed_rules(
            composed_table.table.rules.clone(),
            composed_table.table.num_terminals,
            terminal_display_names,
            composed_table.table.nonterminal_display_names.clone(),
            augmented_start,
        )
    };
    let analyzed_ms = analyzed_started_at.elapsed().as_secs_f64() * 1000.0;
    let commit_transport_started_at = Instant::now();
    let (mut transported_commit_templates, mut commit_rebuild, _transport_ms) =
        transport_component_commit_templates(
            composed_table,
            components,
            commit_template_dfas_enabled(),
        );
    let commit_transport_ms = commit_transport_started_at.elapsed().as_secs_f64() * 1000.0;
    if commit_template_dfas_enabled() || transported_commit_templates.iter().any(Option::is_some) {
        for &terminal in control_changed_terminals {
            if let Some(slot) = commit_rebuild.get_mut(terminal as usize) {
                *slot = true;
            }
        }
    }

    let eager_all_templates =
        std::env::var_os("GLRMASK_COMPOSE_SELECTED_TEMPLATES_ONLY").is_none();
    let (eager_templates, lexical) = if let Some(lexical) = precomputed_lexical {
        (None, Ok(Some(lexical)))
    } else {
        rayon::join(
            || {
                eager_all_templates.then(|| {
                    build_composition_templates(
                        &composed_table.table,
                        &analyzed,
                        &vec![true; analyzed.num_terminals as usize],
                    )
                })
            },
            || {
                prepare_boundary_lexical_prepass(
                    &analyzed,
                    &composed_table.boundary_nonterminals,
                    &composed_table.terminal_offsets,
                    merged_tokenizer,
                    merged_reset_state,
                    ignore_terminals,
                    vocab,
                    special_token_terminals,
                    components,
                    tokenizer_state_offsets,
                    selected_boundary_tokens,
                )
            },
        )
    };
    let Some(BoundaryLexicalPrepass {
        seed_terminals,
        disallowed_follows,
        boundary_paths,
        discovery_ms,
        seed_relations,
        one_byte_ms,
        active_terminals,
        boundary_special_token_terminals,
    }) = lexical?
    else {
        return Ok(None);
    };
    let discovered_boundary_terminals = boundary_paths.terminals.clone();

    let state_map_started_at = Instant::now();
    let owned_component_state_map = if precomputed_component_state_map.is_none()
        && deferred_component_state_map.is_none()
    {
        Some(component_state_coordinate_map(
            components,
            tokenizer_state_offsets,
            merged_tokenizer_state_count,
            merged_reset_state,
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
    let state_map_ms = state_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let template_terminal_started_at = Instant::now();
    let ((templates, templates_ms), terminal_dwa) =
        rayon::join(
            || {
                let (mut templates, templates_ms) = eager_templates.unwrap_or_else(|| {
                    build_composition_templates(
                        &composed_table.table,
                        &analyzed,
                        &active_terminals,
                    )
                });
                if eager_all_templates {
                    templates.by_terminal.retain(|terminal, _| {
                        active_terminals.get(*terminal as usize).copied().unwrap_or(false)
                            || commit_rebuild
                                .get(*terminal as usize)
                                .copied()
                                .unwrap_or(false)
                    });
                    templates.by_terminal_nwa.retain(|terminal, _| {
                        active_terminals.get(*terminal as usize).copied().unwrap_or(false)
                            || commit_rebuild
                                .get(*terminal as usize)
                                .copied()
                                .unwrap_or(false)
                    });
                }
                (templates, templates_ms)
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
                            let canonicalized_tokenizer = ignore_terminals.canonical.map(|canonical| {
                                let mut canonicalized = tokenizer.clone();
                                canonicalized.canonicalize_terminal_aliases(
                                    canonical,
                                    &ignore_terminals.aliases,
                                );
                                canonicalized
                            });
                            let tokenizer = canonicalized_tokenizer.as_ref().unwrap_or(tokenizer);
                            let flat_trans: Arc<[u32]> = Arc::from(
                                crate::compiler::stages::id_map_and_terminal_dwa::l1::
                                    build_flat_transition_table(tokenizer),
                            );
                            let global_max_length_state_map =
                                crate::compiler::stages::id_map_and_terminal_dwa::
                                    build_global_max_length_state_map(tokenizer, vocab, &flat_trans);
                            let coloring =
                                TerminalColoring::identity(analyzed.num_terminals as usize);
                            let artifact =
                                crate::compiler::stages::id_map_and_terminal_dwa::
                                    build_restricted_id_map_and_terminal_dwa_with_precomputed_global_max_length(
                                        tokenizer,
                                        vocab,
                                        &coloring,
                                        false,
                                        ignore_terminals.canonical,
                                        &analyzed,
                                        &disallowed_follows,
                                        flat_trans,
                                        &global_max_length_state_map,
                                        None,
                                        Some(&active_terminals),
                                    )
                                    .0;
                            Ok(add_control_loops_to_terminal_artifact(
                                artifact,
                                &composed_table.control_terminals,
                            ))
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
                        &boundary_paths,
                        &ignore_terminals.global,
                        &composed_table.control_terminals,
                    )
                };
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
        );
    let template_terminal_wall_ms =
        template_terminal_started_at.elapsed().as_secs_f64() * 1000.0;
    let (terminal_dwa, terminal_ms) = terminal_dwa;
    let terminal_dwa = terminal_dwa?;
    let special_source_state = merged_tokenizer
        .map(Tokenizer::initial_state_id)
        .unwrap_or(merged_reset_state);
    debug_assert_eq!(special_source_state, merged_reset_state);
    let special_paths_started_at = Instant::now();
    let terminal_dwa = add_boundary_special_token_paths(
        terminal_dwa,
        &boundary_special_token_terminals,
        special_source_state,
        Some(component_state_map),
        &composed_table.control_terminals,
    )?;
    let special_paths_ms = special_paths_started_at.elapsed().as_secs_f64() * 1000.0;

    let (terminal_automaton, id_map) = terminal_dwa.into_parts();
    let parser_commit_started_at = Instant::now();
    let ((rebuilt_commit_templates, commit_ms), (parser_dwa, parser_ms)) = rayon::join(
        || {
            build_composition_commit_templates(
                &templates,
                analyzed.num_terminals as usize,
                Some(&commit_rebuild),
            )
        },
        || {
            let parser_started_at = Instant::now();
            let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &composed_table.table,
                &analyzed,
                &terminal_automaton,
                &templates,
                vocab,
                &id_map,
                false,
            );
            (
                parser_dwa,
                parser_started_at.elapsed().as_secs_f64() * 1000.0,
            )
        },
    );
    let parser_commit_wall_ms = parser_commit_started_at.elapsed().as_secs_f64() * 1000.0;
    for (terminal, rebuilt) in rebuilt_commit_templates.into_iter().enumerate() {
        if rebuilt.is_some() {
            transported_commit_templates[terminal] = rebuilt;
        }
    }
    let template_dfas_by_terminal = transported_commit_templates;
    profile_boundary_parser_dwa(&parser_dwa);
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_build] active={} begin_active={} discovered_active={} boundary_tokens={} boundary_special_tokens={} analyzed_ms={analyzed_ms:.3} commit_transport_ms={commit_transport_ms:.3} state_map_ms={state_map_ms:.3} discovery_ms={discovery_ms:.3} one_byte_ms={one_byte_ms:.3} terminal_ms={terminal_ms:.3} templates_ms={templates_ms:.3} template_terminal_wall_ms={template_terminal_wall_ms:.3} special_paths_ms={special_paths_ms:.3} commit_ms={commit_ms:.3} parser_ms={parser_ms:.3} parser_commit_wall_ms={parser_commit_wall_ms:.3} total_ms={:.3}",
            active_terminals.iter().filter(|&&active| active).count(),
            seed_terminals.iter().filter(|&&active| active).count(),
            discovered_boundary_terminals.count_ones(),
            boundary_paths.token_ids.len(),
            boundary_special_token_terminals.len(),
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
    control_terminals: &BTreeSet<u32>,
) -> Vec<SpecialTokenTerminal> {
    fn consumes_terminal(action: &Action) -> bool {
        match action {
            Action::Reduce(_, _) => false,
            Action::Split { shift, accept, .. } => shift.is_some() || *accept,
            Action::Shift(_, _)
            | Action::StackShifts(_)
            | Action::GuardedStackShifts(_)
            | Action::Accept
            | Action::ReplaceShifts(_)
            | Action::Skip => true,
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
                    if control_terminals.contains(&terminal_id) {
                        return None;
                    }
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

#[derive(Debug, Clone)]
struct MergedIgnoreTerminals {
    canonical: Option<u32>,
    canonical_expr: Option<crate::automata::regex::Expr>,
    all: BitSet,
    /// Ignore terminals whose identity effect depends on the active parser
    /// scope. These remain visible to the boundary terminal/parser DWA.
    scoped: BitSet,
    /// Equivalent component-local ignore terminals which can be erased before
    /// parser interpretation. They are canonicalized to `canonical` in the
    /// final composed tokenizer/artifacts.
    global: BitSet,
    aliases: Vec<u32>,
}

fn constraint_ignore_expr(constraint: &Constraint) -> Option<&crate::automata::regex::Expr> {
    constraint.ignore_expr.as_ref().or_else(|| {
        constraint
            .ignore_terminal
            .and_then(|terminal| constraint.tokenizer.terminal_expr(terminal))
    })
}

/// Whether every parser scope accepts the same ignore language globally.
///
/// Existing scoped `Skip` actions are proof that a component already contains
/// non-global ignore ownership, so such a component cannot be flattened back
/// into one global ignore merely because its public `ignore_terminal` happens
/// to match another component. Explicit control terminals themselves are not a
/// problem: adjacent/nested children can retain explicit calls while sharing a
/// single globally erased ignore.
fn component_ignores_are_globally_erasable(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    let components = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    if components
        .iter()
        .any(|component| !component.table.skip_terminals.is_empty())
    {
        return false;
    }
    let Some(first) = components.first() else {
        return true;
    };
    match first.ignore_terminal {
        None => components
            .iter()
            .all(|component| component.ignore_terminal.is_none()),
        Some(first_ignore) => {
            let Some(expected) = constraint_ignore_expr(first) else {
                return false;
            };
            components.iter().skip(1).all(|component| {
                let Some(_ignore) = component.ignore_terminal else {
                    return false;
                };
                constraint_ignore_expr(component).is_some_and(|actual| actual == expected)
            })
        }
    }
}

fn components_have_no_explicit_controls(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|component| component.table.control_terminals.is_empty())
}

fn components_have_no_compiled_eof_stack_rewrites(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|component| {
            component.table.action.iter().all(|row| {
                !matches!(
                    row.get(&EOF),
                    Some(
                        Action::Shift(..)
                            | Action::StackShifts(_)
                            | Action::GuardedStackShifts(_)
                            | Action::ReplaceShifts(_)
                            | Action::Skip
                            | Action::Split { shift: Some(_), .. }
                    )
                )
            })
        })
}

/// The legacy splice identifies child start/accept states with parent
/// caller/continuation states. That optimization is not equivalent when one
/// subgrammar call can directly follow another without consuming a real parent
/// terminal: the first return and second entry are both erased, so a token
/// containing only the second child's first terminal has no correct transported
/// component effect.
///
/// Use the grammar's exact ever-follow relation to reject the optimization for
/// direct, nullable-mediated, or nonterminal-mediated call adjacency. The
/// explicit-control linker remains the reference path for those cases.
fn legacy_splice_has_only_byte_terminal_continuations(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
) -> bool {
    if children.is_empty() {
        return true;
    }
    let placeholders = children
        .iter()
        .map(|child| child.placeholder_terminal)
        .collect::<BTreeSet<_>>();
    let boundary_controlled_followers = placeholders
        .iter()
        .copied()
        .chain(
            parent
                .special_token_terminals
                .iter()
                .map(|special| special.terminal_id),
        )
        .collect::<BTreeSet<_>>();
    let Some(augmented_start) = parent.table.rules.first().map(|rule| rule.lhs) else {
        return false;
    };
    let analyzed = AnalyzedGrammar::from_composed_rules(
        parent.table.rules.clone(),
        parent.table.num_terminals,
        parent.terminal_display_names.clone(),
        parent.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let disallowed = crate::compiler::pipeline::compute_disallowed_follows(&analyzed);
    for &left in &placeholders {
        for &right in &boundary_controlled_followers {
            let is_disallowed = disallowed
                .get(&left)
                .is_some_and(|blocked| blocked.contains(right as usize));
            if !is_disallowed {
                return false;
            }
        }
    }
    true
}

fn merged_ignore_terminals(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
    globally_erasable: bool,
) -> MergedIgnoreTerminals {
    let ignores = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
        .filter_map(|(component_index, constraint)| {
            constraint
                .ignore_terminal
                .map(|terminal| terminal_offsets[component_index] + terminal)
        })
        .collect::<Vec<_>>();
    let canonical = globally_erasable
        .then(|| ignores.first().copied())
        .flatten();
    let canonical_expr = canonical.and_then(|_| constraint_ignore_expr(parent).cloned());
    let mut all = BitSet::new(
        terminal_offsets
            .iter()
            .copied()
            .zip(std::iter::once(parent).chain(children.iter().map(|child| child.constraint)))
            .map(|(offset, constraint)| offset + constraint.tokenizer.num_terminals())
            .max()
            .unwrap_or(0) as usize,
    );
    for &ignore in &ignores {
        all.set(ignore as usize);
    }
    for (component_index, component) in std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for &skip in &component.table.skip_terminals {
            all.set(terminal_offset + skip as usize);
        }
    }
    let (global, scoped) = if canonical.is_some() {
        (all.clone(), BitSet::new(all.len()))
    } else {
        (BitSet::new(all.len()), all.clone())
    };
    let aliases = canonical
        .map(|canonical| {
            ignores
                .into_iter()
                .filter(|&ignore| ignore != canonical)
                .collect()
        })
        .unwrap_or_default();
    MergedIgnoreTerminals {
        canonical,
        canonical_expr,
        all,
        scoped,
        global,
        aliases,
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

fn canonicalize_terminal_live_states(
    states: &mut [Vec<u32>],
    canonical: Option<u32>,
    aliases: &[u32],
) {
    let Some(canonical) = canonical else {
        return;
    };
    let canonical = canonical as usize;
    for &alias in aliases {
        let alias = alias as usize;
        if alias >= states.len() || canonical >= states.len() {
            continue;
        }
        let alias_states = std::mem::take(&mut states[alias]);
        states[canonical].extend(alias_states);
    }
    states[canonical].sort_unstable();
    states[canonical].dedup();
}

fn canonicalize_possible_matches(
    possible_matches: &mut PossibleMatches,
    canonical: Option<u32>,
    aliases: &[u32],
) {
    let Some(canonical) = canonical else {
        return;
    };
    for &alias in aliases {
        let Some(alias_weight) = possible_matches.remove(&alias) else {
            continue;
        };
        possible_matches
            .entry(canonical)
            .and_modify(|weight| *weight = weight.union(&alias_weight))
            .or_insert(alias_weight);
    }
}

fn canonicalize_parser_artifact_ignore(
    artifact: MappedArtifact<(DWA, PossibleMatches)>,
    canonical: Option<u32>,
    aliases: &[u32],
) -> MappedArtifact<(DWA, PossibleMatches)> {
    let ((dwa, mut possible_matches), id_map) = artifact.into_parts();
    canonicalize_possible_matches(&mut possible_matches, canonical, aliases);
    MappedArtifact::new((dwa, possible_matches), id_map)
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
    ignore_expr: Option<crate::automata::regex::Expr>,
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
        ignore_expr,
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
    ignore_expr: Option<crate::automata::regex::Expr>,
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
        ignore_expr,
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
    let global_ignores = component_ignores_are_globally_erasable(parent, children);
    let table_inputs = children
        .iter()
        .map(|child| SubgrammarTableInput {
            placeholder_terminal: child.placeholder_terminal,
            table: &child.constraint.table,
            ignore_terminal: (!global_ignores)
                .then_some(child.constraint.ignore_terminal)
                .flatten(),
            start_nullable: child.constraint.table.embedded_start_nullable(),
        })
        .collect::<Vec<_>>();
    let use_legacy_splice =
        global_ignores
            && components_have_no_explicit_controls(parent, children)
            && components_have_no_compiled_eof_stack_rewrites(parent, children)
            && legacy_splice_has_only_byte_terminal_continuations(parent, children);
    let table_started_at = Instant::now();
    let mut composed_table = if use_legacy_splice {
        compose_subgrammar_tables(&parent.table, &table_inputs)?
    } else if std::env::var_os("GLRMASK_COMPOSE_COPIED_EXPLICIT_TABLE").is_some() {
        compose_subgrammar_tables_explicit(
            &parent.table,
            (!global_ignores)
                .then_some(parent.ignore_terminal)
                .flatten(),
            &table_inputs,
        )?
    } else {
        compose_subgrammar_tables_shared_explicit(
            &parent.table,
            (!global_ignores)
                .then_some(parent.ignore_terminal)
                .flatten(),
            &table_inputs,
        )?
    };
    let table_ms = table_started_at.elapsed().as_secs_f64() * 1000.0;
    profile_composed_state_relations(&composed_table);

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
    let merged_reset_state = 0;

    let special_token_terminals = merged_special_token_terminals(
        parent,
        children,
        &composed_table.terminal_offsets,
        &composed_table.table,
        &composed_table.control_terminals,
    );
    let control_elimination_report = eliminate_composed_runtime_controls(&mut composed_table)?;
    let control_elimination_ms = control_elimination_report
        .as_ref()
        .map(|report| report.elapsed_ms)
        .unwrap_or(0.0);
    let control_changed_terminals = control_elimination_report
        .as_ref()
        .map(|report| report.changed_terminals.clone())
        .unwrap_or_default();
    let parser_components = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| ParserDwaComponent {
            constraint,
            parser_state_relation: &composed_table.state_relations[index],
            tokenizer_state_offset: expected_tokenizer_state_offsets[index],
        })
        .collect::<Vec<_>>();
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
    let merged_ignores = merged_ignore_terminals(
        parent,
        children,
        &composed_table.terminal_offsets,
        global_ignores,
    );
    let ignore_terminal = merged_ignores.canonical;

    let needs_materialized_boundary_reference =
        std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE").is_some()
            || std::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_BOUNDARY_VIEW").is_some();

    let (
        (mut tokenizer, tokenizer_state_offsets),
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
                    merged_reset_state,
                    &original_token_ids,
                    !global_ignores,
                );
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
            || {
                let started_at = Instant::now();
                let result = build_boundary_repair(
                    &composed_table,
                    Some(&tokenizer_result.0),
                    merged_tokenizer_state_count,
                    merged_reset_state,
                    terminal_display_names.clone(),
                    &merged_ignores,
                    vocab,
                    &special_token_terminals,
                    &component_constraints,
                    &control_changed_terminals,
                    &expected_tokenizer_state_offsets,
                    None,
                    None,
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
                                merged_reset_state,
                                &original_token_ids,
                                !global_ignores,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                        || {
                            let started_at = Instant::now();
                            let result = build_boundary_repair(
                                &composed_table,
                                None,
                                merged_tokenizer_state_count,
                                merged_reset_state,
                                terminal_display_names.clone(),
                                &merged_ignores,
                                vocab,
                                &special_token_terminals,
                                &component_constraints,
                                &control_changed_terminals,
                                &expected_tokenizer_state_offsets,
                                None,
                                None,
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
    let parser_artifacts = canonicalize_parser_artifact_ignore(
        parser_artifacts,
        merged_ignores.canonical,
        &merged_ignores.aliases,
    );
    let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    if let Some(canonical) = merged_ignores.canonical {
        tokenizer.canonicalize_terminal_aliases(canonical, &merged_ignores.aliases);
    }
    let mut terminal_live_states = merged_terminal_live_states(
        parent,
        children,
        &composed_table.terminal_offsets,
        &tokenizer_state_offsets,
        composed_table.table.num_terminals as usize,
    );
    canonicalize_terminal_live_states(
        &mut terminal_live_states,
        merged_ignores.canonical,
        &merged_ignores.aliases,
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
        merged_ignores.canonical_expr.clone(),
        terminal_live_states,
        Default::default(),
        vocab,
    );
    let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition] components={} table_ms={table_ms:.3} control_elimination_ms={control_elimination_ms:.3} tokenizer_ms={tokenizer_ms:.3} reuse_ms={reuse_ms:.3} boundary_ms={boundary_ms:.3} union_ms={union_ms:.3} closure_prime_ms={closure_prime_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
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

    let global_ignores = component_ignores_are_globally_erasable(&parent, children);
    let table_inputs = children
        .iter()
        .map(|child| SubgrammarTableInput {
            placeholder_terminal: child.placeholder_terminal,
            table: &child.constraint.table,
            ignore_terminal: (!global_ignores)
                .then_some(child.constraint.ignore_terminal)
                .flatten(),
            start_nullable: child.constraint.table.embedded_start_nullable(),
        })
        .collect::<Vec<_>>();
    let use_legacy_splice =
        global_ignores
            && components_have_no_explicit_controls(&parent, children)
            && components_have_no_compiled_eof_stack_rewrites(&parent, children)
            && legacy_splice_has_only_byte_terminal_continuations(&parent, children);
    let table_started_at = Instant::now();
    let mut composed_table = if use_legacy_splice {
        compose_subgrammar_tables(&parent.table, &table_inputs)?
    } else if std::env::var_os("GLRMASK_COMPOSE_COPIED_EXPLICIT_TABLE").is_some() {
        compose_subgrammar_tables_explicit(
            &parent.table,
            (!global_ignores)
                .then_some(parent.ignore_terminal)
                .flatten(),
            &table_inputs,
        )?
    } else {
        compose_subgrammar_tables_shared_explicit(
            &parent.table,
            (!global_ignores)
                .then_some(parent.ignore_terminal)
                .flatten(),
            &table_inputs,
        )?
    };
    let table_ms = table_started_at.elapsed().as_secs_f64() * 1000.0;
    profile_composed_state_relations(&composed_table);

    let metadata_started_at = Instant::now();
    let component_constraints = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let (expected_tokenizer_state_offsets, merged_tokenizer_state_count) =
        component_tokenizer_state_layout_owned_parent(&component_constraints);
    let merged_reset_state = expected_tokenizer_state_offsets[0]
        .checked_add(parent.tokenizer.start_state())
        .ok_or_else(|| "owned-parent merged reset state overflow".to_string())?;
    let component_views_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;
    let specials_started_at = Instant::now();
    let special_token_terminals = merged_special_token_terminals(
        &parent,
        children,
        &composed_table.terminal_offsets,
        &composed_table.table,
        &composed_table.control_terminals,
    );
    let parser_components = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| ParserDwaComponent {
            constraint,
            parser_state_relation: &composed_table.state_relations[index],
            tokenizer_state_offset: expected_tokenizer_state_offsets[index],
        })
        .collect::<Vec<_>>();
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
    let merged_ignores = merged_ignore_terminals(
        &parent,
        children,
        &composed_table.terminal_offsets,
        global_ignores,
    );
    let ignore_terminal = merged_ignores.canonical;
    let names_ms = names_started_at.elapsed().as_secs_f64() * 1000.0;
    let metadata_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_metadata] component_views_ms={component_views_ms:.3} specials_ms={specials_ms:.3} token_ids_ms={token_ids_ms:.3} names_ms={names_ms:.3} total_ms={metadata_ms:.3}",
        );
    }
    // Control elimination mutates only action/control rows. Lexical boundary
    // discovery depends on the already-linked rule graph and immutable child
    // tokenizers, so run those two substantial passes together. Component
    // remapping then overlaps the control-free boundary parser finish.
    let augmented_start = composed_table
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let boundary_analyzed = AnalyzedGrammar::from_composed_rules(
        composed_table.table.rules.clone(),
        composed_table.table.num_terminals,
        terminal_display_names.clone(),
        composed_table.table.nonterminal_display_names.clone(),
        augmented_start,
    );
    let boundary_nonterminals = composed_table.boundary_nonterminals.clone();
    let boundary_terminal_offsets = composed_table.terminal_offsets.clone();
    let selected_boundary_tokens_cell =
        OnceLock::<Result<Option<Vec<u32>>, String>>::new();
    let control_overlap_started_at = Instant::now();
    let (control_elimination_result, lexical_result) = {
        let table = &mut composed_table.table;
        let controls = &mut composed_table.control_terminals;
        if std::env::var_os("GLRMASK_COMPOSE_PARALLEL_CONTROL_LEXICAL").is_none() {
            let control = eliminate_runtime_controls_parts(table, controls);
            let lexical = prepare_boundary_lexical_prepass(
                &boundary_analyzed,
                &boundary_nonterminals,
                &boundary_terminal_offsets,
                None,
                merged_reset_state,
                &merged_ignores,
                vocab,
                &special_token_terminals,
                &component_constraints,
                &expected_tokenizer_state_offsets,
                Some(&selected_boundary_tokens_cell),
            );
            (control, lexical)
        } else {
            rayon::join(
                || eliminate_runtime_controls_parts(table, controls),
                || {
                    prepare_boundary_lexical_prepass(
                        &boundary_analyzed,
                        &boundary_nonterminals,
                        &boundary_terminal_offsets,
                        None,
                        merged_reset_state,
                        &merged_ignores,
                        vocab,
                        &special_token_terminals,
                        &component_constraints,
                        &expected_tokenizer_state_offsets,
                        Some(&selected_boundary_tokens_cell),
                    )
                },
            )
        }
    };
    let control_elimination_report = control_elimination_result?;
    let control_elimination_ms = control_elimination_report
        .as_ref()
        .map(|report| report.elapsed_ms)
        .unwrap_or(0.0);
    let control_changed_terminals = control_elimination_report
        .as_ref()
        .map(|report| report.changed_terminals.clone())
        .unwrap_or_default();
    let lexical_prepass = lexical_result?;
    let lexical_discovery_ms = lexical_prepass
        .as_ref()
        .map(|prepass| prepass.discovery_ms.max(prepass.one_byte_ms))
        .unwrap_or(0.0);
    let control_overlap_ms = control_overlap_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_control_lexical_overlap] control_ms={control_elimination_ms:.3} lexical_ms={lexical_discovery_ms:.3} wall_ms={control_overlap_ms:.3}",
        );
    }

    let state_map_cell = OnceLock::<Result<ManyToOneIdMap, String>>::new();
    let transport_top_accept_directly =
        std::env::var_os("GLRMASK_COMPOSE_LEGACY_TOP_ACCEPT_BRANCH").is_none();
    let preparation_started_at = Instant::now();
    let prepared_components_result = {
            let state_started_at = Instant::now();
            let state_result = build_direct_component_state_coordinates(
                &parser_components,
                merged_tokenizer_state_count,
                merged_reset_state,
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
            let (
                (token_coordinate_result, token_coordinate_ms),
                (unmapped_result, parser_extract_ms),
            ) = rayon::join(
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
                        !global_ignores,
                        transport_top_accept_directly,
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
            let selected_boundary_tokens = selected_boundary_tokens_cell
                .get()
                .expect("boundary lexical prepass must publish selected tokens")
                .as_ref()
                .map_err(Clone::clone)?
                .clone();
            let prepared = if let Some(selected_boundary_tokens) = selected_boundary_tokens {
                let boundary_id_map = boundary_id_map_for_selected_tokens(
                    &component_id_map.tokenizer_states,
                    &selected_boundary_tokens,
                )?;
                let plan = build_boundary_refinement_plan(component_id_map, &boundary_id_map)
                    .ok_or_else(|| {
                        "component coordinate map does not cover boundary repair".to_string()
                    })?;
                let (automata, possible_matches, top_accept_parts, remap_ms) =
                    remap_unmapped_component_artifacts(
                        unmapped_result?,
                        component_maps,
                        Some(&plan.component_token_map),
                        plan.common_map.num_tsids() as usize,
                    )?;
                PreparedOwnedComponentArtifacts {
                    automata,
                    possible_matches,
                    top_accept_parts,
                    id_map: plan.common_map,
                    boundary_tsid_map: Some(plan.boundary_tsid_map),
                    boundary_token_map: Some(plan.boundary_token_map),
                    remap_ms,
                }
            } else {
                let (automata, possible_matches, top_accept_parts, remap_ms) =
                    remap_unmapped_component_artifacts(
                        unmapped_result?,
                        component_maps,
                        None,
                        component_id_map.num_tsids() as usize,
                    )?;
                PreparedOwnedComponentArtifacts {
                    automata,
                    possible_matches,
                    top_accept_parts,
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
        };
    let (boundary_result, boundary_ms) = {
        let started_at = Instant::now();
            let result = match lexical_prepass {
                Some(lexical_prepass) => build_boundary_repair(
                    &composed_table,
                    None,
                    merged_tokenizer_state_count,
                    merged_reset_state,
                    terminal_display_names.clone(),
                    &merged_ignores,
                    vocab,
                    &special_token_terminals,
                    &component_constraints,
                    &control_changed_terminals,
                    &expected_tokenizer_state_offsets,
                    None,
                    Some(&state_map_cell),
                    Some(&selected_boundary_tokens_cell),
                    Some(lexical_prepass),
                    Some(boundary_analyzed),
                ),
                None => Ok(None),
            };
            (result, started_at.elapsed().as_secs_f64() * 1000.0)
        };
    let (prepared_components, coordinate_ms, parser_extract_ms) = prepared_components_result?;
    let boundary_repair = boundary_result?;
    let preparation_ms = preparation_started_at.elapsed().as_secs_f64() * 1000.0;

    let terminal_live_started_at = Instant::now();
    let mut terminal_live_states = merged_terminal_live_states_owned_parent(
        &mut parent,
        children,
        &composed_table.terminal_offsets,
        &expected_tokenizer_state_offsets,
        composed_table.table.num_terminals as usize,
    );
    canonicalize_terminal_live_states(
        &mut terminal_live_states,
        merged_ignores.canonical,
        &merged_ignores.aliases,
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
    let (mut tokenizer, tokenizer_state_offsets) =
        Tokenizer::disjoint_union_with_owned_parent(
            parent_tokenizer,
            composed_table.terminal_offsets[0],
            &child_tokenizers,
        );
    if let Some(canonical) = merged_ignores.canonical {
        tokenizer.canonicalize_terminal_aliases(canonical, &merged_ignores.aliases);
    }
    let tokenizer_ms = tokenizer_started_at.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        tokenizer_state_offsets, expected_tokenizer_state_offsets,
        "owned-parent tokenizer state offsets differ from predicted layout",
    );


    let num_parser_states = composed_table.table.num_states;
    let num_terminals = composed_table.table.num_terminals as usize;
    let PreparedOwnedComponentArtifacts {
        automata,
        mut possible_matches,
        top_accept_parts,
        id_map,
        boundary_tsid_map,
        boundary_token_map,
        remap_ms: component_remap_ms,
    } = prepared_components;
    canonicalize_possible_matches(
        &mut possible_matches,
        merged_ignores.canonical,
        &merged_ignores.aliases,
    );
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
        merged_ignores.canonical_expr.clone(),
        terminal_live_states,
        tokenizer_fast_transitions,
        true,
        vocab,
    );
    result.constraint.parser_top_accept =
        collapse_transported_top_accept_parts(top_accept_parts, num_parser_states);

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
                    automata.push(RawCompressedAutomaton::from_dwa_preserving_defaults(
                        &boundary_dwa,
                    ));
                    let automata_len = automata.len();
                    let validation_automata = std::env::var_os(
                        "GLRMASK_VALIDATE_COMPOSE_SINGLE_PASS_UNION",
                    )
                    .is_some()
                    .then(|| automata.clone());
                    let direct_started_at = Instant::now();
                    let (parser_dwa, synthetic_states) =
                        determinize_compressed_component_union_prechecked(
                            automata,
                            Some(num_parser_states),
                            0.0,
                        );
                    let union_path = "overlap_local";
                    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
                    if let Some(automata) = validation_automata {
                        let mut reference =
                            NWA::new(id_num_tsids, id_max_internal_token);
                        let mut starts = Vec::new();
                        for (index, automaton) in automata.iter().enumerate() {
                            let explicit = if index + 1 == automata.len() {
                                explicit_parser_nwa(&boundary_dwa, num_parser_states)
                            } else {
                                automaton.to_nwa()
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
                    let (parser_dwa, synthetic_states) =
                        determinize_compressed_component_union_prechecked(
                            automata,
                            None,
                            0.0,
                        );
                    let union_path = "overlap_local";
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
            "[glrmask/profile][constraint_composition_owned_parent] components={} table_ms={table_ms:.3} control_elimination_ms={control_elimination_ms:.3} tokenizer_ms={tokenizer_ms:.3} coordinate_ms={coordinate_ms:.3} parser_extract_ms={parser_extract_ms:.3} boundary_ms={boundary_ms:.3} preparation_ms={preparation_ms:.3} terminal_live_ms={terminal_live_ms:.3} union_ms={union_ms:.3} token_cache_prebuild_ms={token_cache_prebuild_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
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
                ignore_terminal: child.ignore_terminal,
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
            merged_tokenizer.initial_state_id(),
            &vocab.entries_map().keys().copied().collect::<Vec<_>>(),
            false,
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
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            true,
        );
    }

    fn assert_constraints_mask_equivalent_on_reachable_prefixes(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
    ) {
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            false,
        );
    }

    fn assert_constraints_equivalent_on_reachable_prefixes_inner(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
        compare_completion: bool,
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
                if compare_completion {
                    assert_eq!(
                        actual_state.is_finished(),
                        expected_state.is_finished(),
                        "completion mismatch after reachable prefix {prefix:?}",
                    );
                }

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
    fn child_ignore_terminal_is_scoped_across_fused_boundaries() {
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

                g child ::= {
                    start child;
                    ignore WS;
                    t WS ::= " "+;
                    nt child ::= "a";
                };

                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.ignore_terminal.is_none());
        assert!(!composed.table.skip_terminals.is_empty());
        assert!(composed.table.action.iter().any(|row| {
            row.iter().any(|(terminal, action)| {
                composed.table.skip_terminals.contains(&terminal)
                    && matches!(action, Action::Skip)
            })
        }));
        assert!(composed.table.control_terminals.is_empty());

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );

        // Child trivia is not globally active before the parent has entered
        // the child scope.
        assert!(!token_allowed(&composed.start().mask(), 3));
        assert!(!token_allowed(&monolithic.start().mask(), 3));
    }

    #[test]
    fn distinct_parent_and_child_ignore_terminals_match_inline_scoped_semantics() {
        let vocab = Vocab::new(vec![
            (0, b"X \ta!".to_vec()),
            (1, b"X\ta!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"\t".to_vec()),
            (5, b"a".to_vec()),
            (6, b"!".to_vec()),
            (7, b" \ta".to_vec()),
            (8, b"a!".to_vec()),
            // Distinguish scope ordering at entry and return. Child trivia
            // followed by parent trivia before child syntax is invalid; child
            // trailing trivia followed by parent trivia is valid, while the
            // reverse ordering is invalid after return has begun.
            (9, b"X\t a!".to_vec()),
            (10, b"Xa\t !".to_vec()),
            (11, b"Xa \t!".to_vec()),
            (12, b"\tX a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;

                g child ::= {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt child ::= "a";
                };

                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save()).unwrap();

        for constraint in [&composed, &loaded] {
            assert!(constraint.ignore_terminal.is_none());
            assert_eq!(constraint.table.skip_terminals.len(), 2);
            assert!(constraint.table.control_terminals.is_empty());
        }

        // The current inline scoped-ignore lowering materialises nullable skip
        // productions. Its `is_complete()` predicate can report a
        // trivia-only prefix as complete before the visible root has parsed.
        // Compare exact masks/commit language here and assert completion on the
        // complete boundary strings below, rather than preserving that
        // unrelated inline-lowering artefact in the explicit linker.
        assert_constraints_mask_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            5,
        );
        assert_constraints_equivalent_on_reachable_prefixes(
            &loaded,
            &composed,
            &vocab,
            5,
        );

        for (token, expected) in [(0, true), (1, true), (9, false), (10, true), (11, false), (12, false)] {
            let mut actual = composed.start();
            let mut loaded_state = loaded.start();
            let mut reference = monolithic.start();
            assert_eq!(actual.commit_token(token).is_ok(), expected, "token {token}");
            assert_eq!(
                loaded_state.commit_token(token).is_ok(),
                expected,
                "loaded token {token}",
            );
            assert_eq!(reference.commit_token(token).is_ok(), expected, "reference token {token}");
            assert_eq!(actual.is_complete(), expected, "token {token}");
            assert_eq!(loaded_state.is_complete(), expected, "loaded token {token}");
            assert_eq!(reference.is_complete(), expected, "reference token {token}");
        }
    }

    #[test]
    fn adjacent_precomposed_child_can_begin_with_child_only_token() {
        let vocab = Vocab::new(vec![
            // End the first child at a model-token boundary, then begin the
            // already-composed second child with a token containing no parent
            // or first-child terminal at all.  A mixed-owner-only boundary
            // selector misses token 1; FIRST/FOLLOW boundary beginnings must
            // retain it.
            (0, b"Xa".to_vec()),
            (1, b"b".to_vec()),
            (2, b"!".to_vec()),
            (3, b"\nb".to_vec()),
            (4, b" b".to_vec()),
            (5, b"a".to_vec()),
            (6, b"X".to_vec()),
            (7, b"\t".to_vec()),
            (8, b"\n".to_vec()),
            (9, b" ".to_vec()),
            (10, b"\n\n".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= "X" LEFT RIGHT "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start left;
                ignore LEFT_WS;
                t LEFT_WS ::= "\t"+;
                nt left ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start right;
                ignore RIGHT_WS;
                t RIGHT_WS ::= "\n"+;
                nt right ::= "b";
            "#,
            &vocab,
        )
        .unwrap();

        // Compose RIGHT first.  When LEFT is linked later, LEFT's parent
        // continuation begins by traversing RIGHT's already-existing control
        // edge.  The direct continuation row therefore exposes a control
        // terminal, not lexical terminal "b".
        let parent_with_right = parent
            .compose_subgrammars(&[("RIGHT", &right)], &vocab)
            .unwrap();
        let composed = parent_with_right
            .compose_subgrammars(&[("LEFT", &left)], &vocab)
            .unwrap();

        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;

                g left ::= {
                    start left;
                    ignore LEFT_WS;
                    t LEFT_WS ::= "\t"+;
                    nt left ::= "a";
                };
                g right ::= {
                    start right;
                    ignore RIGHT_WS;
                    t RIGHT_WS ::= "\n"+;
                    nt right ::= "b";
                };

                nt document ::= "X" left right "!";
            "#,
            &vocab,
        )
        .unwrap();

        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 1));
        assert!(token_allowed(&reference.mask(), 1));
        actual.commit_token(1).unwrap();
        reference.commit_token(1).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_complete());
        assert!(reference.is_complete());

        // Leading RIGHT trivia also begins only after return-from-LEFT followed
        // by enter-RIGHT, and must preserve that scope across the token.
        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 3));
        assert!(token_allowed(&reference.mask(), 3));
        actual.commit_token(3).unwrap();
        reference.commit_token(3).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_complete());
        assert!(reference.is_complete());

        // A multi-byte token containing only RIGHT trivia must itself be a
        // boundary-begin path: return from LEFT, enter RIGHT, consume trivia,
        // and persist the RIGHT start state for the next model token. This is
        // not an owner-crossing token and is not covered by one-byte seed
        // relations.
        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 10));
        assert!(token_allowed(&reference.mask(), 10));
        actual.commit_token(10).unwrap();
        reference.commit_token(10).unwrap();
        actual.commit_token(1).unwrap();
        reference.commit_token(1).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_complete());
        assert!(reference.is_complete());
    }

    #[test]
    fn adjacent_calls_to_same_child_accept_child_only_second_token() {
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xaa!".to_vec()),
            (4, b"X".to_vec()),
            (5, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= "X" SUB SUB "!";
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
                nt child ::= "a";
                nt document ::= "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        let mut actual = composed.start();
        let mut reference = monolithic.start();
        actual.commit_token(0).unwrap();
        reference.commit_token(0).unwrap();
        assert!(token_allowed(&actual.mask(), 1));
        assert!(token_allowed(&reference.mask(), 1));
        actual.commit_token(1).unwrap();
        reference.commit_token(1).unwrap();
        actual.commit_token(2).unwrap();
        reference.commit_token(2).unwrap();
        assert!(actual.is_complete());
        assert!(reference.is_complete());

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            4,
        );
    }

    #[test]
    fn adjacent_children_with_uniform_ignore_compile_controls_and_keep_global_ignore() {
        let vocab = Vocab::new(vec![
            (0, b"X a".to_vec()),
            (1, b" a".to_vec()),
            (2, b" !".to_vec()),
            (3, b" X a a ! ".to_vec()),
            (4, b"X".to_vec()),
            (5, b" ".to_vec()),
            (6, b"a".to_vec()),
            (7, b"!".to_vec()),
            (8, b"<".to_vec()),
            (9, b">".to_vec()),
            (10, b"< X a a ! >".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB SUB "!";
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
                nt child ::= "a";
                nt document ::= "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save()).unwrap();

        for constraint in [&composed, &loaded] {
            assert!(constraint.ignore_terminal.is_some());
            assert!(
                constraint.table.control_terminals.is_empty(),
                "linker entry/return controls must be compiled out of the runtime table"
            );
            assert!(
                constraint.table.skip_terminals.is_empty(),
                "a globally erasable ignore must not be materialized in parser rows"
            );
            assert!(constraint.table.action.iter().all(|row| {
                row.iter().all(|(_, action)| !matches!(action, Action::Skip))
            }));
        }

        for sequence in [vec![0, 1, 2], vec![3], vec![4, 5, 6, 5, 6, 5, 7, 5]] {
            let mut actual = composed.start();
            let mut restored = loaded.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask mismatch before token {token}");
                actual.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_complete());
            assert!(restored.is_complete());
            assert!(expected.is_complete());
        }

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            4,
        );
        assert_constraints_equivalent_on_reachable_prefixes(
            &loaded,
            &composed,
            &vocab,
            4,
        );

        // The compiled child no longer carries runtime controls. Reusing it in
        // an outer composition must retain the same globally erased ignore and
        // compile the new call/return controls away again.
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start outer;
                ignore WS;
                t WS ::= " "+;
                t INNER ::= @token(1000);
                nt outer ::= "<" INNER ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer_monolithic = Constraint::from_glrm_grammar(
            r#"
                start outer;
                ignore WS;
                t WS ::= " "+;
                nt child ::= "a";
                nt outer ::= "<" "X" child child "!" ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .clone()
            .compose_subgrammars(&[("INNER", &composed)], &vocab)
            .unwrap();
        let outer_from_loaded = outer_parent
            .compose_subgrammars(&[("INNER", &loaded)], &vocab)
            .unwrap();
        for constraint in [&outer, &outer_from_loaded] {
            assert!(constraint.ignore_terminal.is_some());
            assert!(constraint.ignore_expr.is_some());
            assert!(constraint.table.control_terminals.is_empty());
            assert!(constraint.table.skip_terminals.is_empty());
            assert!(constraint.table.action.iter().all(|row| {
                row.iter().all(|(_, action)| !matches!(action, Action::Skip))
            }));
        }
        for sequence in [vec![10], vec![8, 3, 9]] {
            let mut actual = outer.start();
            let mut restored_actual = outer_from_loaded.start();
            let mut expected = outer_monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "nested mask mismatch before token {token}");
                assert_eq!(
                    restored_actual.mask(),
                    expected.mask(),
                    "loaded-child nested mask mismatch before token {token}"
                );
                actual.commit_token(token).unwrap();
                restored_actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_complete());
            assert!(restored_actual.is_complete());
            assert!(expected.is_complete());
        }
    }

    #[test]
    fn nonterminal_mediated_adjacent_calls_use_explicit_linker() {
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xaa!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt second ::= SUB;
                nt document ::= "X" SUB second "!";
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
                nt child ::= "a";
                nt second ::= child;
                nt document ::= "X" child second "!";
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
            4,
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
    fn composition_compiles_named_special_continuation_into_static_mask() {
        const SPECIAL_TOKEN: u32 = 1000;
        let vocab = Vocab::new(vec![
            (0, b"Xa".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t DONE ::= @token(1000);
                nt document ::= "X" SUB DONE;
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
                t DONE ::= @token(1000);
                nt document ::= "X" "a" DONE;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.table.control_terminals.is_empty());
        for content in [vec![0], vec![1, 2]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            for token in content {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.mask(), expected.mask());
            assert_eq!(actual.forced(), vec![SPECIAL_TOKEN]);
            actual.commit_token(SPECIAL_TOKEN).unwrap();
            expected.commit_token(SPECIAL_TOKEN).unwrap();
            assert!(actual.is_finished());
            assert!(expected.is_finished());
        }
    }

    #[test]
    fn nullable_child_to_named_special_is_static_masked() {
        const SPECIAL_TOKEN: u32 = 1000;
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                t DONE ::= @token(1000);
                nt document ::= SUB DONE;
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
                t DONE ::= @token(1000);
                nt item ::= "a";
                nt document ::= item? DONE;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.table.control_terminals.is_empty());
        for sequence in [vec![SPECIAL_TOKEN], vec![0, SPECIAL_TOKEN]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_finished());
            assert!(expected.is_finished());
        }
    }

    #[test]
    fn external_only_adjacent_parent_preserves_cross_child_tokens() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"aa".to_vec())]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t SUB ::= @token(999);
                nt document ::= SUB+;
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
                nt document ::= "a"+;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.table.control_terminals.is_empty());
        for sequence in [vec![0], vec![1], vec![0, 0], vec![0, 1]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask());
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.mask(), expected.mask());
            assert_eq!(actual.is_finished(), expected.is_finished());
        }
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
    fn explicit_scoped_control_runtime_benchmark_probe() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" ".to_vec()),
            (2, b"\t".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
            (5, b"X \ta\t !".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                t SUB ::= @token(999);
                nt document ::= "X" SUB "!";
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS ::= "\t"+;
                nt child ::= "a";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_subgrammars(&[("SUB", &child)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= " "+;
                g child ::= {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt child ::= "a";
                };
                nt document ::= "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();

        let runs = std::env::var("GLRMASK_COMPOSE_RUNTIME_BENCH_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(10_000)
            .max(1);
        let split = [0u32, 1, 2, 3, 2, 1, 4];
        let measure = |constraint: &Constraint, sequence: &[u32]| {
            let mut samples = Vec::with_capacity(runs);
            for _ in 0..runs {
                let mut state = constraint.start();
                let mut total = 0u64;
                for &token in sequence {
                    total += state.commit_token_timed_ns(token).unwrap();
                }
                assert!(state.is_complete());
                samples.push(total);
            }
            samples.sort_unstable();
            samples[samples.len() / 2]
        };
        let composed_split = measure(&composed, &split);
        let monolithic_split = measure(&monolithic, &split);
        let composed_fused = measure(&composed, &[5]);
        let monolithic_fused = measure(&monolithic, &[5]);
        let mut profiled = composed.start();
        let fused_profile = profiled.commit_token_profiled(5).unwrap();
        eprintln!(
            "explicit scoped runtime: runs={runs} split_composed_ns={composed_split} split_monolithic_ns={monolithic_split} fused_composed_ns={composed_fused} fused_monolithic_ns={monolithic_fused} fused_profile={fused_profile:?}",
        );
        std::hint::black_box((
            composed_split,
            monolithic_split,
            composed_fused,
            monolithic_fused,
        ));
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
