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

use crate::automata::lexer::tokenizer::{
    Lexer, SingletonEpsilonClosures, Tokenizer, TokenizerStateSet,
};
use crate::automata::lexer::ast::Expr;
use crate::automata::lexer::compile::compile_terminal_expr_dfa;
use crate::automata::weighted_u32::determinize::determinize;
use crate::automata::weighted_u32::equivalence::find_difference;
use crate::automata::weighted_u32::dwa::{DWA, DWAState};
use crate::automata::weighted_u32::nwa::{NWA, NWAState};
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::minimize_acyclic::minimize_acyclic as minimize_unweighted_dfa;
use crate::compiler::glr::analysis::{AnalyzedGrammar, EOF};
use crate::compiler::glr::labels::{
    DEFAULT_LABEL, encode_negative_label, encode_positive_label, is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::stages::equiv_types::{
    InternalIdMap, ManyToOneIdMap, MappedArtifact,
};
use crate::compiler::stages::mapped_artifact::{
    WeightRefs, remap_weights_with_maps, remap_weights_with_maps_serial,
};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring;
use crate::compiler::stages::parser_dwa::{
    build_boolean_terminal_bundle_nwa, build_finite_terminal_word_domain_dwa,
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled,
    build_prebuilt_terminal_bundle_preimage_domain_dwa_profiled,
    build_terminal_bundle_preimage_domain_nwa,
    normalize_parser_stack_domain_nwa, normalize_parser_stack_domain_nwa_preserving_explicit,
    union_parser_stack_domain_dwas, union_parser_stack_domain_nwas,
    universal_parser_stack_domain_dwa, universal_parser_stack_domain_nwa,
    LazyBooleanParserDomains, SharedBooleanParserDomains,
};
use crate::compiler::stages::templates::characterize::{
    InitialEscape, InitialReduce, NtEscape, NtRereduce, StackMatcher,
    TerminalCharacterization, characterize_selected_terminals_profiled,
};
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
use crate::runtime::{
    Constraint, ConstraintRuntimeBackend, PrebuiltParserWeightTokenSets, SpecialTokenTerminal,
};
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
) -> (BTreeMap<i32, Weight>, Vec<Arc<RangeSetBlaze<u32>>>) {
    let profile = compose_profile_enabled();
    let started_at = Instant::now();
    let labels = parts.len();
    let total_parts = parts.values().map(Vec::len).sum::<usize>();
    let max_parts = parts.values().map(Vec::len).max().unwrap_or(0);
    let collapse_one = |(label, mut weights): (i32, Vec<Weight>)| {
        weights.sort_unstable_by_key(Weight::ptr_key);
        weights.dedup_by_key(|weight| weight.ptr_key());
        let weight = match weights.len() {
            0 => return None,
            1 => weights.pop().unwrap(),
            _ => Weight::union_all(weights.iter()),
        };
        (!weight.is_empty()).then_some((label, weight))
    };
    let mut collapsed = if max_parts <= 1 {
        parts
            .into_iter()
            .filter_map(collapse_one)
            .collect::<BTreeMap<_, _>>()
    } else {
        parts
            .into_par_iter()
            .filter_map(collapse_one)
            .collect::<Vec<_>>()
            .into_iter()
            .collect::<BTreeMap<_, _>>()
    };
    let mut token_sets = FxHashMap::<usize, Arc<RangeSetBlaze<u32>>>::default();
    let mut by_weight = FxHashMap::<usize, (Weight, usize)>::default();
    for weight in collapsed.values() {
        by_weight
            .entry(weight.ptr_key())
            .and_modify(|(_, count)| *count += 1)
            .or_insert_with(|| (weight.clone(), 1));
    }
    let unique_weights = by_weight.len();
    for (weight, _) in by_weight.values() {
        if !weight.is_full() && !weight.is_empty() {
            for (_, token_set) in weight.raw_range_values() {
                token_sets
                    .entry(Arc::as_ptr(token_set) as usize)
                    .or_insert_with(|| Arc::clone(token_set));
            }
        }
    }
    let mut default_multiplicity = 0usize;
    let mut exceptions = collapsed.len();
    let full_positive_domain = collapsed.len() == num_parser_states as usize
        && (0..num_parser_states)
            .all(|state| collapsed.contains_key(&encode_positive_label(state)));
    if full_positive_domain
        && !collapsed.is_empty()
        && let Some((default_weight, count)) = by_weight
            .into_values()
            .max_by_key(|(_, count)| *count)
    {
        let default_key = default_weight.ptr_key();
        collapsed.retain(|_, weight| weight.ptr_key() != default_key);
        collapsed.insert(DEFAULT_LABEL, default_weight);
        default_multiplicity = count;
        exceptions = collapsed.len().saturating_sub(1);
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
    (collapsed, token_sets.into_values().collect())
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
    /// Exact multi-byte boundary lanes `(TSID, model-token)` represented by
    /// the witness graph. Kept in the boundary artifact coordinate so parser
    /// union experiments can partition ownership without inspecting parser
    /// topology.
    claimed_weight: Weight,
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
    /// Component owning the lexical terminal fragments seen so far.
    /// `u32::MAX` means no terminal yet; `u32::MAX - 1` means fragments from
    /// more than one component. Keeping this provenance prevents a genuinely
    /// cross-component token path from being conflated with an ordinary
    /// component-local token path that merely touches a boundary seed.
    lexical_component: u32,
    /// Whether this path has touched a terminal that can begin a child or a
    /// parent continuation (or a scoped ignore terminal).
    seeded: bool,
    /// False only for the arbitrary-residual first fragment. Once any
    /// terminal commits, subsequent fragments start from lexer reset.
    started: bool,
}

#[derive(Debug, Clone)]
struct BoundaryTokenEdge {
    target: usize,
    terminal: u32,
}

#[derive(Debug, Clone)]
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

#[derive(Clone)]
struct BoundaryTokenWitness {
    token_id: u32,
    start_states: Vec<u32>,
    nodes: Vec<BoundaryTokenNode>,
    good: Vec<bool>,
    accepting: Vec<bool>,
}

#[derive(Clone)]
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

fn seed_suffix_tokens_sorted_shared_prefixes(
    seed_dfas: &[crate::automata::lexer::DFA],
    entries: &[crate::compiler::vocab_suffix_index::VocabSuffixOwners],
) -> BTreeSet<u32> {
    struct Work {
        lo: usize,
        hi: usize,
        depth: usize,
        dfa_states: SmallVec<[u32; 12]>,
    }

    let mut exact = BTreeSet::<u32>::new();
    if entries.is_empty() || seed_dfas.is_empty() {
        return exact;
    }

    let mut stack = vec![Work {
        lo: 0,
        hi: entries.len(),
        depth: 0,
        dfa_states: std::iter::repeat_n(0u32, seed_dfas.len()).collect(),
    }];
    while let Some(work) = stack.pop() {
        if work.lo >= work.hi {
            continue;
        }

        // The independent predicate checks finalizers immediately after every
        // consumed byte. Once any seed DFA has finalized, every longer suffix
        // in this lexicographic prefix range is accepted too.
        if work.depth != 0
            && seed_dfas
                .iter()
                .zip(&work.dfa_states)
                .any(|(dfa, &state)| state != u32::MAX && dfa.finalizers(state).contains(0))
        {
            for entry in &entries[work.lo..work.hi] {
                exact.extend(entry.token_ids().iter().copied());
            }
            continue;
        }

        let mut child_lo = work.lo;
        if entries[child_lo].suffix().len() == work.depth {
            if seed_dfas
                .iter()
                .zip(&work.dfa_states)
                .any(|(dfa, &state)| {
                    state != u32::MAX && dfa.possible_future_group_ids(state).contains(0)
                })
            {
                exact.extend(entries[child_lo].token_ids().iter().copied());
            }
            child_lo += 1;
        }
        if child_lo >= work.hi {
            continue;
        }

        // All entries in this work item share `depth` bytes and are sorted.
        // Partition the remaining range by the next byte, and only descend
        // into byte classes reachable by at least one seed DFA.
        let mut lo = child_lo;
        while lo < work.hi {
            let byte = entries[lo].suffix()[work.depth];
            let mut hi = lo + 1;
            while hi < work.hi && entries[hi].suffix()[work.depth] == byte {
                hi += 1;
            }
            let mut next_states = SmallVec::<[u32; 12]>::with_capacity(seed_dfas.len());
            let mut any_live = false;
            for (dfa, &state) in seed_dfas.iter().zip(&work.dfa_states) {
                let next = if state == u32::MAX {
                    u32::MAX
                } else {
                    dfa.step(state, byte).unwrap_or(u32::MAX)
                };
                any_live |= next != u32::MAX;
                next_states.push(next);
            }
            if any_live {
                stack.push(Work {
                    lo,
                    hi,
                    depth: work.depth + 1,
                    dfa_states: next_states,
                });
            }
            lo = hi;
        }
    }
    exact
}

fn execute_component_summary_groups_from_states_fast(
    component: &Constraint,
    input: &[u8],
    starts: &[u32],
) -> Vec<(
    TokenizerStateSet,
    SmallVec<[(u32, usize); 4]>,
    Vec<u32>,
)> {
    type ScanKey = (TokenizerStateSet, SmallVec<[(u32, usize); 4]>);

    if component.tokenizer_fast_transitions.len() != component.tokenizer.num_states() as usize {
        return component
            .tokenizer
            .execute_summary_groups_from_states(input, starts);
    }
    let closures = component.tokenizer.all_singleton_epsilon_closures();
    let mut active = FxHashMap::<ScanKey, Vec<u32>>::default();
    for &start in starts {
        let Some(closure) = closures.get(start as usize) else {
            continue;
        };
        active
            .entry((TokenizerStateSet::from_slice(closure), SmallVec::new()))
            .or_default()
            .push(start);
    }
    let mut finished = FxHashMap::<ScanKey, Vec<u32>>::default();
    let mut targets = TokenizerStateSet::new();

    for (index, &byte) in input.iter().enumerate() {
        let width = index + 1;
        let mut next = FxHashMap::<ScanKey, Vec<u32>>::default();
        for ((states, mut matches), support) in active {
            targets.clear();
            for &state in &states {
                let target = component.tokenizer_fast_transitions.transition(
                    &component.tokenizer,
                    state,
                    byte,
                );
                if target == u32::MAX {
                    continue;
                }
                let Some(target_closure) = closures.get(target as usize) else {
                    continue;
                };
                targets.extend_from_slice(target_closure);
            }
            if targets.len() > 1 {
                targets.sort_unstable();
                targets.dedup();
            }
            if targets.is_empty() {
                finished
                    .entry((TokenizerStateSet::new(), matches))
                    .or_default()
                    .extend(support);
                continue;
            }
            for &state in &targets {
                for terminal in component.tokenizer.matched_terminals_iter(state) {
                    if let Some((_, longest)) = matches
                        .iter_mut()
                        .find(|(candidate, _)| *candidate == terminal)
                    {
                        *longest = (*longest).max(width);
                    } else {
                        matches.push((terminal, width));
                    }
                }
            }
            matches.sort_unstable_by_key(|(terminal, _)| *terminal);
            next.entry((targets.clone(), matches))
                .or_default()
                .extend(support);
        }
        active = next;
        if active.is_empty() {
            break;
        }
    }
    for (key, support) in active {
        finished.entry(key).or_default().extend(support);
    }
    let mut groups = finished
        .into_iter()
        .map(|((states, matches), mut support)| {
            support.sort_unstable();
            support.dedup();
            (states, matches, support)
        })
        .collect::<Vec<_>>();
    groups.sort_unstable_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    groups
}

/// Cheap one-sided boundary prefilter over the union of candidate residual starts.
///
/// This deliberately records every terminal/width pair observed on the merged
/// frontier, rather than the per-start longest-match summary used by the exact
/// witness builder.  It can therefore retain extra edges, but it cannot lose an
/// edge present in any exact per-start summary.  Since `build_boundary_token_graph`
/// is monotone in its arbitrary-start scan, rejection here proves that every
/// exact candidate start rejects.  Survivors still go through the exact grouped
/// residual scan below.
fn scan_component_residual_union_frontier(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    reset_live_bytes: &[U8Set],
    merged_reset_state: u32,
    bytes: &[u8],
    candidate_groups: &[(u32, Vec<u32>)],
) -> ResidualScanResult {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());
    debug_assert_eq!(components.len(), reset_live_bytes.len());

    let mut starts_by_component = vec![Vec::<u32>::new(); components.len()];
    let mut include_reset = false;
    for (representative, _) in candidate_groups {
        if *representative == merged_reset_state {
            include_reset = true;
            continue;
        }
        let component_index = tokenizer_state_offsets
            .partition_point(|&offset| offset <= *representative)
            .saturating_sub(1);
        let Some(component) = components.get(component_index) else {
            continue;
        };
        let local_start = *representative - tokenizer_state_offsets[component_index];
        if local_start < component.tokenizer.num_states() {
            starts_by_component[component_index].push(local_start);
        }
    }
    if include_reset {
        for (component_index, component) in components.iter().enumerate() {
            if bytes.first().is_some_and(|byte| !reset_live_bytes[component_index].contains(*byte)) {
                continue;
            }
            starts_by_component[component_index].push(component.tokenizer.start_state());
        }
    }

    let mut result = ResidualScanResult::default();
    for (component_index, mut local_starts) in starts_by_component.into_iter().enumerate() {
        if local_starts.is_empty() {
            continue;
        }
        local_starts.sort_unstable();
        local_starts.dedup();
        let component = components[component_index];
        let terminal_offset = terminal_offsets[component_index];
        let closures = component.tokenizer.all_singleton_epsilon_closures();
        let mut states = TokenizerStateSet::new();
        for start in local_starts {
            if let Some(closure) = closures.get(start as usize) {
                states.extend_from_slice(closure);
            }
        }
        if states.len() > 1 {
            states.sort_unstable();
            states.dedup();
        }
        if states.is_empty() {
            continue;
        }

        for (index, &byte) in bytes.iter().enumerate() {
            let mut targets = TokenizerStateSet::new();
            if component.tokenizer_fast_transitions.len() == component.tokenizer.num_states() as usize {
                for &state in &states {
                    let target = component.tokenizer_fast_transitions.transition(
                        &component.tokenizer,
                        state,
                        byte,
                    );
                    if target == u32::MAX {
                        continue;
                    }
                    if let Some(target_closure) = closures.get(target as usize) {
                        targets.extend_from_slice(target_closure);
                    }
                }
                if targets.len() > 1 {
                    targets.sort_unstable();
                    targets.dedup();
                }
            } else {
                targets = component.tokenizer.step_all(&states, byte);
            }
            states = targets;
            if states.is_empty() {
                break;
            }
            let width = index + 1;
            for &state in &states {
                result.matches.extend(
                    component
                        .tokenizer
                        .matched_terminals_iter(state)
                        .map(|terminal| (terminal_offset + terminal, width)),
                );
            }
        }
        for state in states {
            result.future_terminals.extend(
                component
                    .tokenizer
                    .possible_future_terminals_iter(state)
                    .map(|terminal| terminal_offset + terminal),
            );
        }
    }
    result.matches.sort_unstable();
    result.matches.dedup();
    result.future_terminals.sort_unstable();
    result.future_terminals.dedup();
    result
}

#[derive(Default)]
struct BoundaryCoarseTrieNode {
    parent: Option<usize>,
    depth: usize,
    children: BTreeMap<u8, usize>,
    token_ids: Vec<u32>,
}

struct BoundaryCoarseComponentObservations {
    matches_at_node: Vec<Vec<u32>>,
    future_at_node: Vec<Vec<u32>>,
}

fn scan_component_union_frontier_trie_component(
    component: &Constraint,
    component_index: usize,
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    merged_reset_state: u32,
    candidate_groups: &[(u32, Vec<u32>)],
    trie: &[BoundaryCoarseTrieNode],
) -> BoundaryCoarseComponentObservations {
    let state_offset = tokenizer_state_offsets[component_index];
    let terminal_offset = terminal_offsets[component_index];
    let mut local_starts = Vec::<u32>::new();
    let mut include_reset = false;
    for (representative, _) in candidate_groups {
        if *representative == merged_reset_state {
            include_reset = true;
            continue;
        }
        if *representative < state_offset {
            continue;
        }
        let local = *representative - state_offset;
        if local < component.tokenizer.num_states() {
            local_starts.push(local);
        }
    }
    if include_reset {
        // The old per-token path consults `reset_live_bytes` before inserting
        // this state. Inserting it unconditionally is equivalent: a dead first
        // byte simply has no outgoing transition from the reset closure.
        local_starts.push(component.tokenizer.start_state());
    }
    local_starts.sort_unstable();
    local_starts.dedup();

    let mut observations = BoundaryCoarseComponentObservations {
        matches_at_node: vec![Vec::new(); trie.len()],
        future_at_node: vec![Vec::new(); trie.len()],
    };
    if local_starts.is_empty() || trie.len() <= 1 {
        return observations;
    }

    let closures = component.tokenizer.all_singleton_epsilon_closures();
    let mut initial = TokenizerStateSet::new();
    for start in local_starts {
        if let Some(closure) = closures.get(start as usize) {
            initial.extend_from_slice(closure);
        }
    }
    if initial.len() > 1 {
        initial.sort_unstable();
        initial.dedup();
    }
    if initial.is_empty() {
        return observations;
    }

    let mut state_sets = vec![initial.clone()];
    let mut state_set_ids = FxHashMap::<TokenizerStateSet, usize>::default();
    state_set_ids.insert(initial, 0);
    let mut transition_cache = FxHashMap::<(usize, u8), Option<usize>>::default();
    let mut matched_cache = vec![None::<Vec<u32>>];
    let mut future_cache = vec![None::<Vec<u32>>];

    fn visit(
        component: &Constraint,
        terminal_offset: u32,
        closures: &SingletonEpsilonClosures,
        trie: &[BoundaryCoarseTrieNode],
        node_id: usize,
        state_id: usize,
        state_sets: &mut Vec<TokenizerStateSet>,
        state_set_ids: &mut FxHashMap<TokenizerStateSet, usize>,
        transition_cache: &mut FxHashMap<(usize, u8), Option<usize>>,
        matched_cache: &mut Vec<Option<Vec<u32>>>,
        future_cache: &mut Vec<Option<Vec<u32>>>,
        observations: &mut BoundaryCoarseComponentObservations,
    ) {
        for (&byte, &child) in &trie[node_id].children {
            let target_id = if let Some(&cached) = transition_cache.get(&(state_id, byte)) {
                cached
            } else {
                let mut targets = TokenizerStateSet::new();
                {
                    let states = &state_sets[state_id];
                    if component.tokenizer_fast_transitions.len()
                        == component.tokenizer.num_states() as usize
                    {
                        for &state in states {
                            let target = component.tokenizer_fast_transitions.transition(
                                &component.tokenizer,
                                state,
                                byte,
                            );
                            if target == u32::MAX {
                                continue;
                            }
                            if let Some(target_closure) = closures.get(target as usize) {
                                targets.extend_from_slice(target_closure);
                            }
                        }
                        if targets.len() > 1 {
                            targets.sort_unstable();
                            targets.dedup();
                        }
                    } else {
                        targets = component.tokenizer.step_all(states, byte);
                    }
                }
                let target_id = if targets.is_empty() {
                    None
                } else if let Some(&existing) = state_set_ids.get(&targets) {
                    Some(existing)
                } else {
                    let id = state_sets.len();
                    state_set_ids.insert(targets.clone(), id);
                    state_sets.push(targets);
                    matched_cache.push(None);
                    future_cache.push(None);
                    Some(id)
                };
                transition_cache.insert((state_id, byte), target_id);
                target_id
            };
            let Some(target_id) = target_id else {
                continue;
            };

            if matched_cache[target_id].is_none() {
                let mut matched = Vec::<u32>::new();
                for &state in &state_sets[target_id] {
                    matched.extend(
                        component
                            .tokenizer
                            .matched_terminals_iter(state)
                            .map(|terminal| terminal_offset + terminal),
                    );
                }
                matched.sort_unstable();
                matched.dedup();
                matched_cache[target_id] = Some(matched);
            }
            let row = &mut observations.matches_at_node[child];
            row.extend(
                matched_cache[target_id]
                    .as_ref()
                    .expect("matched-terminal cache must be populated")
                    .iter()
                    .copied(),
            );
            if !trie[child].token_ids.is_empty() {
                if future_cache[target_id].is_none() {
                    let mut future_terminals = Vec::<u32>::new();
                    for &state in &state_sets[target_id] {
                        future_terminals.extend(
                            component
                                .tokenizer
                                .possible_future_terminals_iter(state)
                                .map(|terminal| terminal_offset + terminal),
                        );
                    }
                    future_terminals.sort_unstable();
                    future_terminals.dedup();
                    future_cache[target_id] = Some(future_terminals);
                }
                let future = &mut observations.future_at_node[child];
                future.extend(
                    future_cache[target_id]
                        .as_ref()
                        .expect("future-terminal cache must be populated")
                        .iter()
                        .copied(),
                );
            }
            visit(
                component,
                terminal_offset,
                closures,
                trie,
                child,
                target_id,
                state_sets,
                state_set_ids,
                transition_cache,
                matched_cache,
                future_cache,
                observations,
            );
        }
    }

    visit(
        component,
        terminal_offset,
        closures.as_ref(),
        trie,
        0,
        0,
        &mut state_sets,
        &mut state_set_ids,
        &mut transition_cache,
        &mut matched_cache,
        &mut future_cache,
        &mut observations,
    );
    observations
}

fn scan_component_residual_union_frontier_batch(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    merged_reset_state: u32,
    entries: &[(u32, &[u8])],
    candidate_groups: &[(u32, Vec<u32>)],
) -> FxHashMap<u32, ResidualScanResult> {
    let mut trie = vec![BoundaryCoarseTrieNode::default()];
    for &(token_id, bytes) in entries {
        let mut node = 0usize;
        for &byte in bytes {
            let next = if let Some(&next) = trie[node].children.get(&byte) {
                next
            } else {
                let next = trie.len();
                let depth = trie[node].depth + 1;
                trie.push(BoundaryCoarseTrieNode {
                    parent: Some(node),
                    depth,
                    children: BTreeMap::new(),
                    token_ids: Vec::new(),
                });
                trie[node].children.insert(byte, next);
                next
            };
            node = next;
        }
        trie[node].token_ids.push(token_id);
    }

    let observations = components
        .par_iter()
        .enumerate()
        .map(|(component_index, component)| {
            scan_component_union_frontier_trie_component(
                component,
                component_index,
                tokenizer_state_offsets,
                terminal_offsets,
                merged_reset_state,
                candidate_groups,
                &trie,
            )
        })
        .collect::<Vec<_>>();

    let mut output = FxHashMap::<u32, ResidualScanResult>::default();
    for (leaf, node) in trie.iter().enumerate() {
        if node.token_ids.is_empty() {
            continue;
        }
        let mut result = ResidualScanResult::default();
        let mut cursor = Some(leaf);
        while let Some(node_id) = cursor {
            let depth = trie[node_id].depth;
            if depth != 0 {
                for component in &observations {
                    result.matches.extend(
                        component.matches_at_node[node_id]
                            .iter()
                            .copied()
                            .map(|terminal| (terminal, depth)),
                    );
                }
            }
            cursor = trie[node_id].parent;
        }
        for component in &observations {
            result
                .future_terminals
                .extend(component.future_at_node[leaf].iter().copied());
        }
        result.matches.sort_unstable();
        result.matches.dedup();
        result.future_terminals.sort_unstable();
        result.future_terminals.dedup();
        for &token_id in &node.token_ids {
            output.insert(token_id, result.clone());
        }
    }
    output
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

        let grouped_summaries = if std::env::var_os("GLRMASK_COMPOSE_DISABLE_FAST_RESIDUAL_SCAN")
            .is_some()
        {
            component
                .tokenizer
                .execute_summary_groups_from_states(bytes, &local_starts)
        } else {
            execute_component_summary_groups_from_states_fast(component, bytes, &local_starts)
        };
        if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_FAST_RESIDUAL_SCAN").is_some() {
            assert_eq!(
                grouped_summaries,
                component
                    .tokenizer
                    .execute_summary_groups_from_states(bytes, &local_starts),
                "fast component residual summary differs for component {component_index}",
            );
        }
        for (end_states, matches, grouped_starts) in grouped_summaries {
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

const BOUNDARY_COMPONENT_NONE: u32 = u32::MAX;
const BOUNDARY_COMPONENT_CROSSED: u32 = u32::MAX - 1;

fn boundary_terminal_component(terminal_offsets: &[u32], terminal: u32) -> u32 {
    terminal_offsets
        .partition_point(|&offset| offset <= terminal)
        .saturating_sub(1) as u32
}

fn extend_boundary_component_provenance(current: u32, next: u32) -> u32 {
    match current {
        BOUNDARY_COMPONENT_NONE => next,
        BOUNDARY_COMPONENT_CROSSED => BOUNDARY_COMPONENT_CROSSED,
        current if current == next => current,
        _ => BOUNDARY_COMPONENT_CROSSED,
    }
}

fn transition_boundary_key(
    key: BoundaryTokenNodeKey,
    terminal: u32,
    next_offset: usize,
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    ignore_terminals: &BitSet,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> Option<BoundaryTokenNodeKey> {
    if ignore_terminals.contains(terminal as usize) {
        return Some(BoundaryTokenNodeKey {
            offset: next_offset,
            lexical_component: extend_boundary_component_provenance(
                key.lexical_component,
                boundary_terminal_component(terminal_offsets, terminal),
            ),
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
        lexical_component: extend_boundary_component_provenance(
            key.lexical_component,
            boundary_terminal_component(terminal_offsets, terminal),
        ),
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
    terminal_offsets: &[u32],
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
        lexical_component: BOUNDARY_COMPONENT_NONE,
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
                terminal_offsets,
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
                terminal_offsets,
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

fn candidate_start_state_groups_for_ranges(
    ranges: &[(usize, u32, u32)],
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    merged_reset_state: u32,
) -> Vec<(u32, Vec<u32>)> {
    // The retained/fresh merged reset dispatcher is a semantic state of its
    // semantic state of its own and must not be conflated with an individual
    // component's local start state.
    let mut support_by_representative = FxHashMap::<u32, Vec<u32>>::default();
    support_by_representative.insert(merged_reset_state, vec![merged_reset_state]);
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

fn extend_seed_possible_match_candidates(
    candidates: &mut BTreeSet<u32>,
    vocab: &Vocab,
    components: &[&Constraint],
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
) {
    // A segment-entry terminal can occupy an entire model token. Component
    // possible-matches is already exact for that endpoint case, including
    // terminals whose source expression was not retained.
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

        let suffix_index_started_at = Instant::now();
        let suffix_index = crate::compiler::vocab_suffix_index::get(vocab);
        let suffix_count = suffix_index.entries().len();
        let suffix_index_ms = suffix_index_started_at.elapsed().as_secs_f64() * 1000.0;
        let exact_started_at = Instant::now();
        let mut exact = if std::env::var_os(
            "GLRMASK_EXPERIMENT_BOUNDARY_SORTED_SUFFIX_SCAN",
        )
        .is_some()
        {
            let shared = seed_suffix_tokens_sorted_shared_prefixes(&seed_dfas, suffix_index.entries());
            if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_SORTED_SUFFIX_SCAN").is_some() {
                let reference = suffix_index
                    .entries()
                    .par_iter()
                    .filter_map(|entry| {
                        seed_dfas
                            .iter()
                            .any(|dfa| seed_dfa_reaches_match_or_future(dfa, entry.suffix()))
                            .then(|| {
                                entry
                                    .token_ids()
                                    .iter()
                                    .copied()
                                    .collect::<SmallVec<[u32; 2]>>()
                            })
                    })
                    .flatten_iter()
                    .collect::<BTreeSet<_>>();
                assert_eq!(
                    shared, reference,
                    "shared-prefix seed suffix scan differs from independent suffix scan",
                );
            }
            shared
        } else {
            suffix_index
                .entries()
                .par_iter()
                .filter_map(|entry| {
                    seed_dfas
                        .iter()
                        .any(|dfa| seed_dfa_reaches_match_or_future(dfa, entry.suffix()))
                        .then(|| {
                            entry
                                .token_ids()
                                .iter()
                                .copied()
                                .collect::<SmallVec<[u32; 2]>>()
                        })
                })
                .flatten_iter()
                .collect::<BTreeSet<_>>()
        };
        let suffix_language_tokens = exact.len();
        extend_seed_possible_match_candidates(
            &mut exact,
            vocab,
            components,
            terminal_offsets,
            seed_terminals,
        );
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_seed_suffix_filter] unique_seed_dfas={} projected_seed_dfas={} suffixes={} suffix_language_tokens={} exact_tokens={} seed_compile_ms={seed_compile_ms:.3} suffix_index_ms={suffix_index_ms:.3} exact_ms={:.3}",
                seed_dfas.len(),
                projected_seed_dfa_count,
                suffix_count,
                suffix_language_tokens,
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
    let candidate_groups_started_at = Instant::now();
    let mut groups_by_ranges = FxHashMap::<Vec<(usize, u32, u32)>, Arc<Vec<(u32, Vec<u32>)>>>::default();
    let mut candidate_groups_by_token = FxHashMap::<u32, Arc<Vec<(u32, Vec<u32>)>>>::default();
    for &(token_id, _) in &multi_byte_entries {
        let ranges = candidate_ranges.get(&token_id).map(Vec::as_slice).unwrap_or(&[]);
        let groups = if let Some(existing) = groups_by_ranges.get(ranges) {
            Arc::clone(existing)
        } else {
            let groups = Arc::new(candidate_start_state_groups_for_ranges(
                ranges,
                components,
                tokenizer_state_offsets,
                merged_reset_state,
            ));
            groups_by_ranges.insert(ranges.to_vec(), Arc::clone(&groups));
            groups
        };
        candidate_groups_by_token.insert(token_id, groups);
    }
    let candidate_group_signatures = groups_by_ranges.len();
    let candidate_groups_ms = candidate_groups_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        let mut prefixes_by_signature = FxHashMap::<usize, FxHashSet<Vec<u8>>>::default();
        let mut tokens_by_signature = FxHashMap::<usize, usize>::default();
        for &(token_id, bytes) in &multi_byte_entries {
            let groups = candidate_groups_by_token
                .get(&token_id)
                .expect("candidate groups must be precomputed for every scanned token");
            let signature = Arc::as_ptr(groups) as usize;
            *tokens_by_signature.entry(signature).or_default() += 1;
            let prefixes = prefixes_by_signature.entry(signature).or_default();
            for end in 1..=bytes.len() {
                prefixes.insert(bytes[..end].to_vec());
            }
        }
        let prefix_occurrences = multi_byte_entries
            .iter()
            .map(|(_, bytes)| bytes.len())
            .sum::<usize>();
        let unique_prefixes = prefixes_by_signature
            .values()
            .map(FxHashSet::len)
            .sum::<usize>();
        let mut signature_shapes = tokens_by_signature
            .into_iter()
            .map(|(signature, tokens)| {
                let prefixes = prefixes_by_signature
                    .get(&signature)
                    .map_or(0, FxHashSet::len);
                (tokens, prefixes)
            })
            .collect::<Vec<_>>();
        signature_shapes.sort_unstable();
        eprintln!(
            "[glrmask/profile][constraint_boundary_candidate_prefixes] occurrences={prefix_occurrences} unique_by_signature={unique_prefixes} signatures={signature_shapes:?}",
        );
    }
    let batched_coarse_started_at = Instant::now();
    let batched_coarse_scans = if std::env::var_os(
        "GLRMASK_COMPOSE_DISABLE_BOUNDARY_UNION_PREFILTER",
    )
    .is_some()
    {
        None
    } else {
        let mut batches = FxHashMap::<
            usize,
            (Arc<Vec<(u32, Vec<u32>)>>, Vec<(u32, &[u8])>),
        >::default();
        for &(token_id, bytes) in &multi_byte_entries {
            let groups = candidate_groups_by_token
                .get(&token_id)
                .expect("candidate groups must be precomputed for every scanned token");
            let signature = Arc::as_ptr(groups) as usize;
            let batch = batches
                .entry(signature)
                .or_insert_with(|| (Arc::clone(groups), Vec::new()));
            batch.1.push((token_id, bytes));
        }
        let batch_scans = batches
            .into_par_iter()
            .map(|(_, (groups, entries))| {
                scan_component_residual_union_frontier_batch(
                    components,
                    tokenizer_state_offsets,
                    terminal_offsets,
                    merged_reset_state,
                    &entries,
                    groups.as_slice(),
                )
            })
            .collect::<Vec<_>>();
        let mut scans = FxHashMap::<u32, ResidualScanResult>::default();
        for batch in batch_scans {
            scans.extend(batch);
        }
        Some(scans)
    };
    let batched_coarse_ms = batched_coarse_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_batched_coarse] scans={} wall_ms={batched_coarse_ms:.3}",
            batched_coarse_scans.as_ref().map_or(0, FxHashMap::len),
        );
    }
    // Each model token is an independent acyclic same-token graph. Run those
    // scans in parallel, then merge in vocabulary order for deterministic
    // output and profiling.
    let candidate_start_visits = AtomicUsize::new(0);
    let distinct_scan_groups = AtomicUsize::new(0);
    let max_candidate_starts = AtomicUsize::new(0);
    let coarse_scan_ns = AtomicUsize::new(0);
    let coarse_rejected = AtomicUsize::new(0);
    let coarse_survived = AtomicUsize::new(0);
    let residual_scan_ns = AtomicUsize::new(0);
    let graph_ns = AtomicUsize::new(0);
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
            let candidate_groups = candidate_groups_by_token
                .get(&token_id)
                .expect("candidate groups must be precomputed for every scanned token");
            candidate_start_visits.fetch_add(candidate_groups.len(), Ordering::Relaxed);
            max_candidate_starts.fetch_max(candidate_groups.len(), Ordering::Relaxed);

            let coarse_started_at = Instant::now();
            let coarse_accepts = if batched_coarse_scans.is_none() {
                true
            } else {
                let coarse_scan = batched_coarse_scans
                    .as_ref()
                    .and_then(|scans| scans.get(&token_id))
                    .expect("batched coarse scan must cover every candidate token");
                if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_BATCHED_BOUNDARY_UNION_PREFILTER")
                    .is_some()
                {
                    let reference = scan_component_residual_union_frontier(
                        components,
                        tokenizer_state_offsets,
                        terminal_offsets,
                        &reset_live_bytes,
                        merged_reset_state,
                        bytes,
                        candidate_groups.as_slice(),
                    );
                    assert_eq!(
                        coarse_scan, &reference,
                        "batched coarse residual scan differs for token {token_id}",
                    );
                }
                build_boundary_token_graph(
                    bytes,
                    coarse_scan,
                    &reset_scans,
                    terminal_offsets,
                    seed_terminals,
                    ignore_terminals,
                    disallowed_follows,
                )
                .is_some()
            };
            coarse_scan_ns.fetch_add(
                coarse_started_at.elapsed().as_nanos() as usize,
                Ordering::Relaxed,
            );
            if coarse_accepts {
                coarse_survived.fetch_add(1, Ordering::Relaxed);
            } else {
                coarse_rejected.fetch_add(1, Ordering::Relaxed);
                if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_BOUNDARY_UNION_PREFILTER").is_some() {
                    let exact = scan_component_residual_start_groups(
                        components,
                        tokenizer_state_offsets,
                        terminal_offsets,
                        &reset_live_bytes,
                        merged_reset_state,
                        bytes,
                        candidate_groups.as_slice(),
                    );
                    for arbitrary_scan in exact.keys() {
                        assert!(
                            build_boundary_token_graph(
                                bytes,
                                arbitrary_scan,
                                &reset_scans,
                                terminal_offsets,
                                seed_terminals,
                                ignore_terminals,
                                disallowed_follows,
                            )
                            .is_none(),
                            "union-frontier prefilter rejected an exact boundary witness for token {token_id}",
                        );
                    }
                }
                return None;
            }

            let residual_scan_started_at = Instant::now();
            let starts_by_scan = scan_component_residual_start_groups(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                &reset_live_bytes,
                merged_reset_state,
                bytes,
                candidate_groups.as_slice(),
            );
            residual_scan_ns.fetch_add(
                residual_scan_started_at.elapsed().as_nanos() as usize,
                Ordering::Relaxed,
            );
            distinct_scan_groups.fetch_add(starts_by_scan.len(), Ordering::Relaxed);
            let mut scan_groups = starts_by_scan.into_iter().collect::<Vec<_>>();
            scan_groups.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            let mut local_terminals = FxHashSet::<u32>::default();
            let mut local_witnesses = Vec::new();
            let graph_started_at = Instant::now();
            for (arbitrary_scan, start_states) in scan_groups {
                let Some((nodes, good, accepting)) = build_boundary_token_graph(
                    bytes,
                    &arbitrary_scan,
                    &reset_scans,
                    terminal_offsets,
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
            graph_ns.fetch_add(
                graph_started_at.elapsed().as_nanos() as usize,
                Ordering::Relaxed,
            );
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
            "[glrmask/profile][constraint_boundary_candidate_cpu] coarse_cpu_ms={:.3} coarse_rejected={} coarse_survived={} scan_cpu_ms={:.3} graph_cpu_ms={:.3} rayon_workers={}",
            coarse_scan_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            coarse_rejected.load(Ordering::Relaxed),
            coarse_survived.load(Ordering::Relaxed),
            residual_scan_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            graph_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            rayon::current_num_threads(),
        );
        eprintln!(
            "[glrmask/profile][constraint_boundary_candidate_fanout] range_tokens={} range_rows={} ranges_ms={candidate_ranges_ms:.3} candidate_group_signatures={} candidate_groups_ms={candidate_groups_ms:.3} scanned_tokens={} raw_start_visits={} distinct_scan_groups={} max_starts={}",
            candidate_ranges.len(),
            candidate_range_rows,
            candidate_group_signatures,
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

fn collect_boundary_witness_terminal_sequences(
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
    sequence_cap: usize,
) -> (BTreeSet<Vec<u32>>, usize, usize, bool) {
    fn collect_from(
        witness: &BoundaryTokenWitness,
        node: usize,
        prefix: &mut Vec<u32>,
        output: &mut BTreeSet<Vec<u32>>,
        ignored_terminals: &BitSet,
        sequence_cap: usize,
        capped: &mut bool,
    ) {
        if *capped {
            return;
        }
        if witness.accepting[node] {
            output.insert(prefix.clone());
            if output.len() >= sequence_cap {
                *capped = true;
                return;
            }
        }
        for edge in &witness.nodes[node].outgoing {
            if !witness.good[edge.target] {
                continue;
            }
            let erased = ignored_terminals.contains(edge.terminal as usize);
            if !erased {
                prefix.push(edge.terminal);
            }
            collect_from(
                witness,
                edge.target,
                prefix,
                output,
                ignored_terminals,
                sequence_cap,
                capped,
            );
            if !erased {
                prefix.pop();
            }
            if *capped {
                return;
            }
        }
    }

    let mut unique = BTreeSet::<Vec<u32>>::new();
    let mut capped = false;
    let mut per_witness_path_sum = 0usize;
    let mut max_paths = 0usize;
    for witness in &discovery.witnesses {
        let mut local = BTreeSet::new();
        collect_from(
            witness,
            0,
            &mut Vec::new(),
            &mut local,
            ignored_terminals,
            sequence_cap,
            &mut capped,
        );
        let local_len = local.len();
        max_paths = max_paths.max(local_len);
        per_witness_path_sum = per_witness_path_sum.saturating_add(local_len);
        unique.extend(local);
        if unique.len() >= sequence_cap {
            capped = true;
            break;
        }
    }
    (unique, per_witness_path_sum, max_paths, capped)
}

fn profile_boundary_witness_terminal_language(
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
) {
    if !compose_profile_enabled() {
        return;
    }
    const SEQUENCE_CAP: usize = 200_000;
    let (unique, per_witness_path_sum, max_paths, capped) =
        collect_boundary_witness_terminal_sequences(discovery, ignored_terminals, SEQUENCE_CAP);
    let total_symbols = unique.iter().map(Vec::len).sum::<usize>();
    let max_len = unique.iter().map(Vec::len).max().unwrap_or(0);
    let avg_len = if unique.is_empty() {
        0.0
    } else {
        total_symbols as f64 / unique.len() as f64
    };
    eprintln!(
        "[glrmask/profile][constraint_boundary_terminal_language] witnesses={} unique_sequences={} per_witness_path_sum={} max_paths_per_witness={} max_sequence_len={} avg_sequence_len={avg_len:.3} capped={capped}",
        discovery.witnesses.len(), unique.len(), per_witness_path_sum, max_paths, max_len,
    );
}

fn profile_boundary_component_locality(
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
    control_changed_terminals: &[u32],
) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_REDUNDANCY").is_none() {
        return;
    }

    fn collect(
        witness: &BoundaryTokenWitness,
        node: usize,
        ignored_terminals: &BitSet,
        prefix: &mut Vec<u32>,
        output: &mut BTreeSet<(Vec<u32>, u32)>,
    ) {
        if witness.accepting[node] {
            output.insert((prefix.clone(), witness.nodes[node].key.lexical_component));
        }
        for edge in &witness.nodes[node].outgoing {
            if !witness.good[edge.target] {
                continue;
            }
            let erased = ignored_terminals.contains(edge.terminal as usize);
            if !erased {
                prefix.push(edge.terminal);
            }
            collect(witness, edge.target, ignored_terminals, prefix, output);
            if !erased {
                prefix.pop();
            }
        }
    }

    #[derive(Default)]
    struct TokenKinds {
        local_components: BTreeSet<u32>,
        crossed: bool,
    }

    let changed = control_changed_terminals.iter().copied().collect::<BTreeSet<_>>();
    let mut unique = BTreeSet::<(Vec<u32>, u32)>::new();
    let mut unique_local_changed = BTreeSet::<Vec<u32>>::new();
    let mut unique_local_unchanged = BTreeSet::<Vec<u32>>::new();
    let mut by_token = BTreeMap::<u32, TokenKinds>::new();
    let mut witness_local_only = 0usize;
    let mut witness_cross_only = 0usize;
    let mut witness_mixed = 0usize;
    let mut local_path_sum = 0usize;
    let mut local_changed_path_sum = 0usize;
    let mut local_unchanged_path_sum = 0usize;
    let mut local_changed_first = 0usize;
    let mut local_changed_last = 0usize;
    let mut local_changed_middle = 0usize;
    let mut local_changed_count_hist = BTreeMap::<usize, usize>::new();
    let mut unique_local_changed_count_hist = BTreeMap::<usize, BTreeSet<Vec<u32>>>::new();
    let mut cross_path_sum = 0usize;
    let mut local_by_component = BTreeMap::<u32, usize>::new();
    let mut local_terminals = BTreeSet::<u32>::new();

    for witness in &discovery.witnesses {
        let mut paths = BTreeSet::<(Vec<u32>, u32)>::new();
        collect(
            witness,
            0,
            ignored_terminals,
            &mut Vec::new(),
            &mut paths,
        );
        let mut has_local = false;
        let mut has_cross = false;
        for (sequence, provenance) in &paths {
            unique.insert((sequence.clone(), *provenance));
            let kinds = by_token.entry(witness.token_id).or_default();
            if *provenance == BOUNDARY_COMPONENT_CROSSED {
                has_cross = true;
                cross_path_sum += 1;
                kinds.crossed = true;
            } else if *provenance != BOUNDARY_COMPONENT_NONE {
                has_local = true;
                local_path_sum += 1;
                kinds.local_components.insert(*provenance);
                *local_by_component.entry(*provenance).or_default() += 1;
                local_terminals.extend(sequence.iter().copied());
                let changed_positions = sequence
                    .iter()
                    .enumerate()
                    .filter_map(|(index, terminal)| changed.contains(terminal).then_some(index))
                    .collect::<Vec<_>>();
                if changed_positions.is_empty() {
                    local_unchanged_path_sum += 1;
                    unique_local_unchanged.insert(sequence.clone());
                } else {
                    local_changed_path_sum += 1;
                    unique_local_changed.insert(sequence.clone());
                    *local_changed_count_hist.entry(changed_positions.len()).or_default() += 1;
                    unique_local_changed_count_hist
                        .entry(changed_positions.len())
                        .or_default()
                        .insert(sequence.clone());
                    local_changed_first += usize::from(changed_positions.contains(&0));
                    local_changed_last += usize::from(
                        !sequence.is_empty()
                            && changed_positions.contains(&(sequence.len() - 1)),
                    );
                    local_changed_middle += usize::from(changed_positions.iter().any(|&index| {
                        index != 0 && index + 1 != sequence.len()
                    }));
                }
            }
        }
        match (has_local, has_cross) {
            (true, true) => witness_mixed += 1,
            (true, false) => witness_local_only += 1,
            (false, true) => witness_cross_only += 1,
            (false, false) => {}
        }
    }

    let unique_local = unique
        .iter()
        .filter(|(_, provenance)| {
            *provenance != BOUNDARY_COMPONENT_NONE
                && *provenance != BOUNDARY_COMPONENT_CROSSED
        })
        .count();
    let unique_cross = unique
        .iter()
        .filter(|(_, provenance)| *provenance == BOUNDARY_COMPONENT_CROSSED)
        .count();
    let token_local_only = by_token
        .values()
        .filter(|kinds| !kinds.local_components.is_empty() && !kinds.crossed)
        .count();
    let token_cross_only = by_token
        .values()
        .filter(|kinds| kinds.local_components.is_empty() && kinds.crossed)
        .count();
    let token_mixed = by_token
        .values()
        .filter(|kinds| !kinds.local_components.is_empty() && kinds.crossed)
        .count();

    eprintln!(
        "[glrmask/profile][constraint_boundary_redundancy] tokens={} witnesses={} unique_paths={} unique_local={} unique_cross={} unique_local_terminals={} unique_local_changed={} unique_local_unchanged={} local_path_sum={} local_changed_path_sum={} local_unchanged_path_sum={} changed_count_1={} changed_count_2={} changed_count_3={} changed_count_4={} unique_changed_count_1={} unique_changed_count_2={} unique_changed_count_3={} unique_changed_count_4={} local_changed_first={} local_changed_last={} local_changed_middle={} cross_path_sum={} witness_local_only={} witness_cross_only={} witness_mixed={} token_local_only={} token_cross_only={} token_mixed={} local_by_component={:?} local_terminals={:?}",
        by_token.len(),
        discovery.witnesses.len(),
        unique.len(),
        unique_local,
        unique_cross,
        local_terminals.len(),
        unique_local_changed.len(),
        unique_local_unchanged.len(),
        local_path_sum,
        local_changed_path_sum,
        local_unchanged_path_sum,
        local_changed_count_hist.get(&1).copied().unwrap_or(0),
        local_changed_count_hist.get(&2).copied().unwrap_or(0),
        local_changed_count_hist.get(&3).copied().unwrap_or(0),
        local_changed_count_hist.get(&4).copied().unwrap_or(0),
        unique_local_changed_count_hist.get(&1).map(BTreeSet::len).unwrap_or(0),
        unique_local_changed_count_hist.get(&2).map(BTreeSet::len).unwrap_or(0),
        unique_local_changed_count_hist.get(&3).map(BTreeSet::len).unwrap_or(0),
        unique_local_changed_count_hist.get(&4).map(BTreeSet::len).unwrap_or(0),
        local_changed_first,
        local_changed_last,
        local_changed_middle,
        cross_path_sum,
        witness_local_only,
        witness_cross_only,
        witness_mixed,
        token_local_only,
        token_cross_only,
        token_mixed,
        local_by_component,
        local_terminals,
    );
}

fn tokenizer_state_component(
    tokenizer_state_offsets: &[u32],
    state: u32,
) -> Option<u32> {
    if tokenizer_state_offsets.is_empty() || state < tokenizer_state_offsets[0] {
        return None;
    }
    Some(
        tokenizer_state_offsets
            .partition_point(|&offset| offset <= state)
            .saturating_sub(1) as u32,
    )
}

fn profile_boundary_entry_return_shape(
    discovery: &BoundaryTokenDiscovery,
    tokenizer_state_offsets: &[u32],
    merged_reset_state: u32,
) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_REDUNDANCY").is_none() {
        return;
    }
    #[derive(Default)]
    struct Counts {
        witnesses: usize,
        start_states: usize,
    }
    let mut categories = BTreeMap::<(u32, u32), Counts>::new();
    let mut reset_by_lexical = BTreeMap::<u32, Counts>::new();
    let mut crossed_witnesses = 0usize;
    let mut mixed_start_components = 0usize;

    for witness in &discovery.witnesses {
        let mut lexical_components = BTreeSet::new();
        for (node_id, node) in witness.nodes.iter().enumerate() {
            if witness.good[node_id]
                && witness.accepting[node_id]
                && node.key.lexical_component != BOUNDARY_COMPONENT_NONE
            {
                lexical_components.insert(node.key.lexical_component);
            }
        }
        if lexical_components.contains(&BOUNDARY_COMPONENT_CROSSED) {
            crossed_witnesses += 1;
        }
        let local_lexical = lexical_components
            .iter()
            .copied()
            .filter(|&component| component != BOUNDARY_COMPONENT_CROSSED)
            .collect::<Vec<_>>();
        if local_lexical.len() != 1 {
            continue;
        }
        let lexical = local_lexical[0];
        let mut starts = BTreeSet::new();
        let mut reset_count = 0usize;
        for &state in &witness.start_states {
            if state == merged_reset_state {
                reset_count += 1;
            } else if let Some(component) = tokenizer_state_component(tokenizer_state_offsets, state) {
                starts.insert(component);
            }
        }
        if starts.len() > 1 {
            mixed_start_components += 1;
        }
        for start in starts {
            let counts = categories.entry((start, lexical)).or_default();
            counts.witnesses += 1;
            counts.start_states += witness
                .start_states
                .iter()
                .filter(|&&state| {
                    state != merged_reset_state
                        && tokenizer_state_component(tokenizer_state_offsets, state) == Some(start)
                })
                .count();
        }
        if reset_count != 0 {
            let counts = reset_by_lexical.entry(lexical).or_default();
            counts.witnesses += 1;
            counts.start_states += reset_count;
        }
    }

    let category_summary = categories
        .into_iter()
        .map(|((start, lexical), counts)| (start, lexical, counts.witnesses, counts.start_states))
        .collect::<Vec<_>>();
    let reset_summary = reset_by_lexical
        .into_iter()
        .map(|(lexical, counts)| (lexical, counts.witnesses, counts.start_states))
        .collect::<Vec<_>>();
    eprintln!(
        "[glrmask/profile][constraint_boundary_entry_return_shape] crossed_witnesses={} mixed_start_components={} component_pairs={:?} reset_by_lexical={:?}",
        crossed_witnesses,
        mixed_start_components,
        category_summary,
        reset_summary,
    );
}

fn canonical_unweighted_dwa_signature(dwa: &DWA) -> Vec<(bool, Vec<(i32, u32)>)> {
    if dwa.states().is_empty() {
        return Vec::new();
    }
    let mut canonical = FxHashMap::<u32, u32>::default();
    let mut order = Vec::<u32>::new();
    let mut queue = VecDeque::new();
    canonical.insert(dwa.start_state(), 0);
    order.push(dwa.start_state());
    queue.push_back(dwa.start_state());
    while let Some(source) = queue.pop_front() {
        for (_, (target, weight)) in &dwa.states()[source as usize].transitions {
            debug_assert!(weight.is_full());
            if !canonical.contains_key(target) {
                let next = canonical.len() as u32;
                canonical.insert(*target, next);
                order.push(*target);
                queue.push_back(*target);
            }
        }
    }
    order
        .into_iter()
        .map(|source| {
            let state = &dwa.states()[source as usize];
            if let Some(weight) = &state.final_weight {
                debug_assert!(weight.is_full());
            }
            let row = state
                .transitions
                .iter()
                .map(|(&label, (target, weight))| {
                    debug_assert!(weight.is_full());
                    (label, canonical[target])
                })
                .collect::<Vec<_>>();
            (state.final_weight.as_ref().is_some_and(Weight::is_full), row)
        })
        .collect()
}

fn profile_finite_boundary_word_domains(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_WORD_DOMAINS").is_none() {
        return;
    }
    let (sequences, _, _, capped) =
        collect_boundary_witness_terminal_sequences(discovery, ignored_terminals, 200_000);
    assert!(!capped, "finite boundary-word domain experiment exceeded sequence cap");
    let sequences = sequences.into_iter().collect::<Vec<_>>();
    let sequence_terminals = sequences
        .iter()
        .flatten()
        .copied()
        .collect::<BTreeSet<_>>();
    let missing_terminals = sequence_terminals
        .iter()
        .copied()
        .filter(|terminal| !templates.by_terminal_nwa.contains_key(terminal))
        .collect::<Vec<_>>();
    eprintln!(
        "[glrmask/profile][constraint_boundary_word_domain_inputs] sequence_terminals={} missing_terminals={} missing={missing_terminals:?}",
        sequence_terminals.len(), missing_terminals.len(),
    );
    let started_at = Instant::now();
    let domains = sequences
        .par_iter()
        .map(|sequence| {
            build_finite_terminal_word_domain_dwa(table, templates, sequence)
                .map(|dwa| {
                    let states = dwa.states().len();
                    let transitions = dwa.num_transitions();
                    (canonical_unweighted_dwa_signature(&dwa), states, transitions)
                })
        })
        .collect::<Vec<_>>();
    let wall_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    let built = domains.iter().filter(|domain| domain.is_some()).count();
    let total_states = domains
        .iter()
        .filter_map(|domain| domain.as_ref().map(|(_, states, _)| *states))
        .sum::<usize>();
    let total_transitions = domains
        .iter()
        .filter_map(|domain| domain.as_ref().map(|(_, _, transitions)| *transitions))
        .sum::<usize>();
    let max_states = domains
        .iter()
        .filter_map(|domain| domain.as_ref().map(|(_, states, _)| *states))
        .max()
        .unwrap_or(0);
    let mut frequencies = FxHashMap::<Vec<(bool, Vec<(i32, u32)>)>, usize>::default();
    for (signature, _, _) in domains.into_iter().flatten() {
        *frequencies.entry(signature).or_default() += 1;
    }
    let unique_domains = frequencies.len();
    let max_domain_multiplicity = frequencies.values().copied().max().unwrap_or(0);
    eprintln!(
        "[glrmask/profile][constraint_boundary_word_domains] sequences={} built={} unique_domains={} max_domain_multiplicity={} total_states={} total_transitions={} max_states={} wall_ms={wall_ms:.3}",
        sequences.len(), built, unique_domains, max_domain_multiplicity, total_states, total_transitions, max_states,
    );
}

/// Exact weighted union of already-normalized boolean parser-stack prefix
/// languages.
///
/// Each root carries a fixed `(TSID, token)` support weight. A support lane is
/// removed from every still-live root as soon as any root accepts it, matching
/// the parser DWA's forced-prefix semantics. Root DWAs have already normalized
/// parser DEFAULT behavior, so DEFAULT is just another symbolic label here and
/// no support provenance is reconstructed or discarded.
fn combine_weighted_prefix_root_dwas(roots: &[(DWA, Weight)]) -> DWA {
    #[derive(Clone)]
    struct Lane {
        root: usize,
        state: u32,
        weight: Weight,
    }

    fn canonicalize_lanes(lanes: impl IntoIterator<Item = Lane>) -> Vec<Lane> {
        let mut grouped = BTreeMap::<(usize, u32), Weight>::new();
        for lane in lanes {
            if lane.weight.is_empty() {
                continue;
            }
            grouped
                .entry((lane.root, lane.state))
                .and_modify(|existing| *existing = existing.union(&lane.weight))
                .or_insert(lane.weight);
        }
        grouped
            .into_iter()
            .map(|((root, state), weight)| Lane { root, state, weight })
            .collect()
    }

    fn lane_key(lanes: &[Lane]) -> Vec<(usize, u32, usize)> {
        lanes
            .iter()
            .map(|lane| (lane.root, lane.state, lane.weight.ptr_key()))
            .collect()
    }

    let initial = canonicalize_lanes(roots.iter().enumerate().filter_map(
        |(root, (dwa, weight))| {
            (!weight.is_empty()).then(|| Lane {
                root,
                state: dwa.start_state(),
                weight: weight.clone(),
            })
        },
    ));
    if initial.is_empty() {
        return DWA::new(0, 0);
    }

    let mut states = vec![DWAState::default()];
    let mut lanes_by_state = vec![initial.clone()];
    let mut ids = FxHashMap::<Vec<(usize, u32, usize)>, u32>::default();
    ids.insert(lane_key(&initial), 0);
    let mut queue = VecDeque::from([0u32]);

    while let Some(state_id) = queue.pop_front() {
        let lanes = lanes_by_state[state_id as usize].clone();
        let mut final_weight = Weight::empty();
        for lane in &lanes {
            let Some(root_state) = roots[lane.root].0.states().get(lane.state as usize) else {
                continue;
            };
            let Some(root_final) = root_state.final_weight.as_ref() else {
                continue;
            };
            let contribution = lane.weight.intersection(root_final);
            if !contribution.is_empty() {
                final_weight = final_weight.union(&contribution);
            }
        }
        if !final_weight.is_empty() {
            states[state_id as usize].final_weight = Some(final_weight.clone());
        }

        let mut live = Vec::<Lane>::new();
        let mut labels = BTreeSet::<i32>::new();
        for lane in lanes {
            let residual = lane.weight.difference(&final_weight);
            if residual.is_empty() {
                continue;
            }
            let Some(root_state) = roots[lane.root].0.states().get(lane.state as usize) else {
                continue;
            };
            labels.extend(root_state.transitions.keys().copied());
            live.push(Lane {
                root: lane.root,
                state: lane.state,
                weight: residual,
            });
        }

        for label in labels {
            let mut next = Vec::<Lane>::new();
            for lane in &live {
                let root_state = &roots[lane.root].0.states()[lane.state as usize];
                let Some((target, edge_weight)) = root_state.transitions.get(&label) else {
                    continue;
                };
                let weight = lane.weight.intersection(edge_weight);
                if !weight.is_empty() {
                    next.push(Lane {
                        root: lane.root,
                        state: *target,
                        weight,
                    });
                }
            }
            let next = canonicalize_lanes(next);
            if next.is_empty() {
                continue;
            }
            let mut edge_weight = Weight::empty();
            for lane in &next {
                edge_weight = edge_weight.union(&lane.weight);
            }
            let key = lane_key(&next);
            let target = if let Some(&target) = ids.get(&key) {
                target
            } else {
                let target = states.len() as u32;
                states.push(DWAState::default());
                lanes_by_state.push(next);
                ids.insert(key, target);
                queue.push_back(target);
                target
            };
            states[state_id as usize]
                .transitions
                .insert(label, (target, edge_weight));
        }
    }

    DWA::from_parts(states, 0)
}

fn profile_boundary_lazy_domain_dp(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
    delta_plan: Option<&BoundaryTemplateDeltaPlan>,
    direct_context: Option<(&ManyToOneIdMap, &InternalIdMap, &BoundarySeedRelations)>,
) -> Option<DWA> {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DOMAINS").is_none() {
        return None;
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CanonicalKey {
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

    let canonicalize_started_at = Instant::now();
    let mut canonical_by_key = BTreeMap::<CanonicalKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    let intern_canonical = |key: CanonicalKey,
                            canonical_by_key: &mut BTreeMap<CanonicalKey, usize>,
                            canonical_nodes: &mut Vec<CanonicalNode>| {
        if let Some(&existing) = canonical_by_key.get(&key) {
            existing
        } else {
            let canonical = canonical_nodes.len();
            debug_assert!(key.transitions.iter().all(|(_, target)| *target < canonical));
            debug_assert!(key.epsilons.iter().all(|target| *target < canonical));
            canonical_nodes.push(CanonicalNode {
                accepting: key.accepting,
                transitions: key.transitions.clone(),
                epsilons: key.epsilons.clone(),
            });
            canonical_by_key.insert(key, canonical);
            canonical
        }
    };
    let mut witness_starts = Vec::<usize>::with_capacity(discovery.witnesses.len());
    for witness in &discovery.witnesses {
        let mut good_nodes = witness
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
            .collect::<Vec<_>>();
        good_nodes.sort_unstable_by(|left, right| right.1.cmp(&left.1));
        if let Some(delta_plan) = delta_plan {
            let mut good_local = vec![false; witness.nodes.len()];
            let mut good_cross = vec![false; witness.nodes.len()];
            for &(local, _) in &good_nodes {
                if witness.accepting[local] {
                    match witness.nodes[local].key.lexical_component {
                        BOUNDARY_COMPONENT_CROSSED => good_cross[local] = true,
                        BOUNDARY_COMPONENT_NONE => {}
                        _ => good_local[local] = true,
                    }
                }
                for edge in &witness.nodes[local].outgoing {
                    if !witness.good[edge.target] {
                        continue;
                    }
                    good_local[local] |= good_local[edge.target];
                    good_cross[local] |= good_cross[edge.target];
                }
            }

            let mut starts = Vec::<usize>::new();
            if good_cross[0] {
                let mut cross_map = vec![usize::MAX; witness.nodes.len()];
                for &(local, _) in &good_nodes {
                    if !good_cross[local] {
                        continue;
                    }
                    let mut transitions = Vec::new();
                    let mut epsilons = Vec::new();
                    for edge in witness.nodes[local]
                        .outgoing
                        .iter()
                        .filter(|edge| good_cross[edge.target])
                    {
                        let target = cross_map[edge.target];
                        debug_assert_ne!(target, usize::MAX);
                        if ignored_terminals.contains(edge.terminal as usize) {
                            epsilons.push(target);
                        } else {
                            transitions.push((edge.terminal, target));
                        }
                    }
                    transitions.sort_unstable();
                    transitions.dedup();
                    epsilons.sort_unstable();
                    epsilons.dedup();
                    cross_map[local] = intern_canonical(
                        CanonicalKey {
                            accepting: witness.accepting[local]
                                && witness.nodes[local].key.lexical_component
                                    == BOUNDARY_COMPONENT_CROSSED,
                            transitions,
                            epsilons,
                        },
                        &mut canonical_by_key,
                        &mut canonical_nodes,
                    );
                }
                starts.push(cross_map[0]);
            }

            if good_local[0] {
                let mut local_changed_counts = vec![BTreeSet::<u8>::new(); witness.nodes.len()];
                for &(local, _) in &good_nodes {
                    if witness.accepting[local]
                        && witness.nodes[local].key.lexical_component
                            != BOUNDARY_COMPONENT_CROSSED
                        && witness.nodes[local].key.lexical_component
                            != BOUNDARY_COMPONENT_NONE
                    {
                        local_changed_counts[local].insert(0);
                    }
                    for edge in witness.nodes[local]
                        .outgoing
                        .iter()
                        .filter(|edge| good_local[edge.target])
                    {
                        let changed = u8::from(
                            !ignored_terminals.contains(edge.terminal as usize)
                                && delta_plan.changed_terminals.contains(&edge.terminal),
                        );
                        let child_counts = local_changed_counts[edge.target]
                            .iter()
                            .copied()
                            .collect::<Vec<_>>();
                        for count in child_counts {
                            local_changed_counts[local]
                                .insert(count.saturating_add(changed).min(2));
                        }
                    }
                }
                let local_unique_delta = local_changed_counts[0]
                    .iter()
                    .copied()
                    .eq(std::iter::once(1));
                let mut productive_local = good_local.clone();
                if local_unique_delta {
                    productive_local.fill(false);
                    for &(local, _) in &good_nodes {
                        if witness.accepting[local]
                            && witness.nodes[local].key.lexical_component
                                != BOUNDARY_COMPONENT_CROSSED
                            && witness.nodes[local].key.lexical_component
                                != BOUNDARY_COMPONENT_NONE
                        {
                            productive_local[local] = true;
                        }
                        for edge in witness.nodes[local]
                            .outgoing
                            .iter()
                            .filter(|edge| good_local[edge.target] && productive_local[edge.target])
                        {
                            let productive_edge = if ignored_terminals
                                .contains(edge.terminal as usize)
                            {
                                true
                            } else if let Some(entry) =
                                delta_plan.by_global_terminal.get(&edge.terminal)
                            {
                                templates
                                    .by_terminal
                                    .get(&entry.delta_terminal)
                                    .is_some_and(|delta| !unweighted_dfa_language_is_empty(delta))
                            } else {
                                true
                            };
                            if productive_edge {
                                productive_local[local] = true;
                                break;
                            }
                        }
                    }
                }
                let mut local_map = vec![usize::MAX; witness.nodes.len()];
                for &(local, _) in &good_nodes {
                    if !productive_local[local] {
                        continue;
                    }
                    let mut transitions = Vec::new();
                    let mut epsilons = Vec::new();
                    for edge in witness.nodes[local]
                        .outgoing
                        .iter()
                        .filter(|edge| productive_local[edge.target])
                    {
                        let target = local_map[edge.target];
                        debug_assert_ne!(target, usize::MAX);
                        if ignored_terminals.contains(edge.terminal as usize) {
                            epsilons.push(target);
                        } else if local_unique_delta {
                            if let Some(entry) = delta_plan.by_global_terminal.get(&edge.terminal) {
                                if templates
                                    .by_terminal
                                    .get(&entry.delta_terminal)
                                    .is_some_and(|delta| !unweighted_dfa_language_is_empty(delta))
                                {
                                    transitions.push((entry.delta_terminal, target));
                                }
                            } else {
                                transitions.push((edge.terminal, target));
                            }
                        } else {
                            transitions.push((edge.terminal, target));
                        }
                    }
                    transitions.sort_unstable();
                    transitions.dedup();
                    epsilons.sort_unstable();
                    epsilons.dedup();
                    local_map[local] = intern_canonical(
                        CanonicalKey {
                            accepting: witness.accepting[local]
                                && witness.nodes[local].key.lexical_component
                                    != BOUNDARY_COMPONENT_CROSSED
                                && witness.nodes[local].key.lexical_component
                                    != BOUNDARY_COMPONENT_NONE,
                            transitions,
                            epsilons,
                        },
                        &mut canonical_by_key,
                        &mut canonical_nodes,
                    );
                }
                if productive_local[0] {
                    starts.push(local_map[0]);
                }
            }

            starts.sort_unstable();
            starts.dedup();
            let start = match starts.as_slice() {
                [] => intern_canonical(
                    CanonicalKey {
                        accepting: false,
                        transitions: Vec::new(),
                        epsilons: Vec::new(),
                    },
                    &mut canonical_by_key,
                    &mut canonical_nodes,
                ),
                [start] => *start,
                _ => intern_canonical(
                    CanonicalKey {
                        accepting: false,
                        transitions: Vec::new(),
                        epsilons: starts,
                    },
                    &mut canonical_by_key,
                    &mut canonical_nodes,
                ),
            };
            witness_starts.push(start);
        } else {
            let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
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
                    if ignored_terminals.contains(edge.terminal as usize) {
                        epsilons.push(target);
                    } else {
                        transitions.push((edge.terminal, target));
                    }
                }
                transitions.sort_unstable();
                transitions.dedup();
                epsilons.sort_unstable();
                epsilons.dedup();
                local_to_canonical[local] = intern_canonical(
                    CanonicalKey {
                        accepting: witness.accepting[local],
                        transitions,
                        epsilons,
                    },
                    &mut canonical_by_key,
                    &mut canonical_nodes,
                );
            }
            witness_starts.push(local_to_canonical[0]);
        }
    }
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for node in &canonical_nodes {
        let mut by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            by_target.entry(target).or_default().push(terminal);
        }
        for (_, mut terminals) in by_target {
            terminals.sort_unstable();
            terminals.dedup();
            terminal_sets.insert(terminals);
        }
    }
    let bundle_started_at = Instant::now();
    let prebuilt_bundles = terminal_sets
        .par_iter()
        .filter_map(|terminals| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
        })
        .collect::<FxHashMap<_, _>>();
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;

    let dp_started_at = Instant::now();
    let mut arena = LazyBooleanParserDomains::new();
    let mut roots = vec![LazyBooleanParserDomains::EMPTY; canonical_nodes.len()];
    let mut preimage_calls = 0usize;
    let mut preimage_cache_hits = 0usize;
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
    for (node_id, node) in canonical_nodes.iter().enumerate() {
        if node.accepting {
            roots[node_id] = LazyBooleanParserDomains::UNIVERSAL;
            continue;
        }
        let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            terminals_by_target.entry(target).or_default().push(terminal);
        }
        let mut branches = Vec::<u32>::new();
        for (target, mut terminals) in terminals_by_target {
            terminals.sort_unstable();
            terminals.dedup();
            let bundle = prebuilt_bundles
                .get(&terminals)
                .expect("every lazy-domain terminal bundle must be prebuilt");
            if std::env::var_os("GLRMASK_DEBUG_BOUNDARY_LAZY_TERMINAL_19").is_some()
                && terminals.as_slice() == [19]
            {
                eprintln!(
                    "[glrmask/debug][lazy_terminal_19_bundle] target={} target_root={} bundle={:?}",
                    target,
                    roots[target],
                    bundle,
                );
            }
            let target_root = roots[target];
            let cache_key = (terminals.clone(), target_root);
            let root = if let Some(&cached) = preimage_cache.get(&cache_key) {
                preimage_cache_hits += 1;
                cached
            } else {
                let root = arena
                    .preimage_bundle(bundle, target_root)
                    .expect("lazy-domain preimage requires read-then-push templates");
                preimage_cache.insert(cache_key, root);
                root
            };
            if std::env::var_os("GLRMASK_DEBUG_BOUNDARY_LAZY_TERMINAL_19").is_some()
                && terminals.as_slice() == [19]
            {
                eprintln!(
                    "[glrmask/debug][lazy_terminal_19_root] root={} nwa={:?}",
                    root,
                    arena.to_nwa(root),
                );
            }
            preimage_calls += 1;
            branches.push(root);
        }
        branches.extend(node.epsilons.iter().map(|&target| roots[target]));
        roots[node_id] = arena.union_all(branches);
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;

    let export_started_at = Instant::now();
    let mut normalized_root_states = 0usize;
    let mut normalized_root_transitions = 0usize;
    let mut unique_witness_roots = FxHashSet::<u32>::default();
    for &start in &witness_starts {
        unique_witness_roots.insert(roots[start]);
    }
    if compose_profile_enabled() {
        let mut productive_tokens = BTreeSet::<u32>::new();
        let mut empty_tokens = BTreeSet::<u32>::new();
        let mut productive_witnesses = 0usize;
        let mut empty_witnesses = 0usize;
        for (witness, &start) in discovery.witnesses.iter().zip(&witness_starts) {
            if roots[start] == LazyBooleanParserDomains::EMPTY {
                empty_witnesses += 1;
                empty_tokens.insert(witness.token_id);
            } else {
                productive_witnesses += 1;
                productive_tokens.insert(witness.token_id);
            }
        }
        eprintln!(
            "[glrmask/profile][constraint_boundary_lazy_productivity] productive_witnesses={} empty_witnesses={} productive_tokens={} empty_tokens={} productive_ids={:?} empty_ids={:?}",
            productive_witnesses,
            empty_witnesses,
            productive_tokens.len(),
            empty_tokens.len(),
            productive_tokens,
            empty_tokens,
        );
    }
    // Normalize only one representative per distinct witness root. This is the
    // cost shape production will have before all roots are exported together.
    for &root in &unique_witness_roots {
        let nwa = arena.to_nwa(root);
        let dwa = normalize_parser_stack_domain_nwa(table, &nwa);
        normalized_root_states += dwa.states().len();
        normalized_root_transitions += dwa.num_transitions();
    }
    let export_ms = export_started_at.elapsed().as_secs_f64() * 1000.0;

    if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_LAZY_DOMAINS").is_some() {
        // Keep the validation oracle in positive-NWA form all the way through
        // the suffix recurrence.  Separately normalized parser DWAs may contain
        // support-local DEFAULT edges; converting those DWAs back to NWAs and
        // unioning them loses the support provenance that makes those DEFAULTs
        // valid and can create spurious cross-branch acceptance.  The positive
        // NWA recurrence preserves the exact branch-local semantics and only
        // normalizes after each complete suffix-node union is assembled.
        let mut reference = vec![None::<NWA>; canonical_nodes.len()];
        for (node_id, node) in canonical_nodes.iter().enumerate() {
            let domain_nwa = if node.accepting {
                universal_parser_stack_domain_nwa()
            } else {
                let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
                for &(terminal, target) in &node.transitions {
                    terminals_by_target.entry(target).or_default().push(terminal);
                }
                let mut branches = Vec::<NWA>::new();
                for (target, mut terminals) in terminals_by_target {
                    terminals.sort_unstable();
                    terminals.dedup();
                    let target_domain = reference[target].as_ref().unwrap();
                    if let Some(branch) = build_terminal_bundle_preimage_domain_nwa(
                        table,
                        templates,
                        &terminals,
                        target_domain,
                    ) {
                        branches.push(branch);
                    }
                }
                for &target in &node.epsilons {
                    branches.push(reference[target].as_ref().unwrap().clone());
                }
                let refs = branches.iter().collect::<Vec<_>>();
                union_parser_stack_domain_nwas(&refs)
            };
            let domain = normalize_parser_stack_domain_nwa(table, &domain_nwa);
            let lazy_nwa = arena.to_nwa(roots[node_id]);
            let lazy = normalize_parser_stack_domain_nwa(table, &lazy_nwa);
            let difference = find_difference(&lazy, &domain)
                .expect("lazy parser domains are acyclic after normalization");
            if let Some(witness) = difference {
                eprintln!(
                    "[glrmask/debug][lazy_domain_mismatch] node={} root={} node_spec={:?} witness={:?} lazy_accept={} reference_accept={} lazy_states={} lazy_transitions={} reference_nwa_states={} reference_states={} reference_transitions={}",
                    node_id,
                    roots[node_id],
                    node,
                    witness,
                    !lazy.eval_word(&witness).is_empty(),
                    !domain.eval_word(&witness).is_empty(),
                    lazy.states().len(),
                    lazy.num_transitions(),
                    domain_nwa.states().len(),
                    domain.states().len(),
                    domain.num_transitions(),
                );
                panic!("lazy parser-domain DP differs at canonical suffix node {node_id}");
            }
            reference[node_id] = Some(domain_nwa);
        }
        eprintln!(
            "[glrmask/validate][constraint_boundary_lazy_domains] nodes={} exact=true",
            canonical_nodes.len(),
        );
    }

    eprintln!(
        "[glrmask/profile][constraint_boundary_lazy_domains] canonical_nodes={} witnesses={} unique_suffix_roots={} unique_witness_roots={} expr_nodes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} unique_bundles={} normalized_root_states={} normalized_root_transitions={} canonicalize_ms={canonicalize_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3} export_ms={export_ms:.3}",
        canonical_nodes.len(),
        witness_starts.len(),
        roots.iter().copied().collect::<FxHashSet<_>>().len(),
        unique_witness_roots.len(),
        arena.node_count(),
        preimage_calls,
        preimage_cache_hits,
        preimage_cache.len(),
        prebuilt_bundles.len(),
        normalized_root_states,
        normalized_root_transitions,
    );

    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DIRECT_PARSER").is_none() {
        return None;
    }
    let Some((component_state_map, id_map, seed_relations)) = direct_context else {
        eprintln!(
            "[glrmask/profile][constraint_boundary_lazy_direct_parser] skipped=true reason=unsupported_boundary_side_lane",
        );
        return None;
    };
    let direct_started_at = Instant::now();
    let mut root_weights = BTreeMap::<u32, Weight>::new();
    let mut merge_root_weight = |root: u32, weight: Weight| {
        if root == LazyBooleanParserDomains::EMPTY || weight.is_empty() {
            return;
        }
        root_weights
            .entry(root)
            .and_modify(|existing| *existing = existing.union(&weight))
            .or_insert(weight);
    };

    for (witness_index, (witness, &start)) in discovery
        .witnesses
        .iter()
        .zip(&witness_starts)
        .enumerate()
    {
        let Some(internal_token) = id_map.internal_token_for_original(witness.token_id) else {
            continue;
        };
        let token_set = RangeSetBlaze::from_iter([internal_token]);
        let mut tsids = BTreeSet::<u32>::new();
        for &raw_state in &witness.start_states {
            let Some(&tsid) = component_state_map.original_to_internal.get(raw_state as usize) else {
                continue;
            };
            if tsid != u32::MAX {
                tsids.insert(tsid);
            }
        }
        let weight = Weight::from_per_tsid_token_sets(
            tsids
                .into_iter()
                .map(|tsid| (tsid, token_set.clone())),
        );
        if std::env::var_os("GLRMASK_DEBUG_BOUNDARY_LAZY_LANE_17122").is_some()
            && (witness.token_id == 17122 || witness.token_id == 534)
            && !weight.tokens_for_tsid(0).is_empty()
        {
            let root = roots[start];
            let probe = if witness.token_id == 17122 {
                vec![540i32, 444, 295, 26, 1]
            } else {
                vec![8810i32, DEFAULT_LABEL, 8723, 1]
            };
            let root_dwa = normalize_parser_stack_domain_nwa(table, &arena.to_nwa(root));
            eprintln!(
                "[glrmask/debug][lazy_lane_witness] witness_index={witness_index} root={root} start_states={:?} probe_accept={} nodes={} token={}",
                witness.start_states,
                !root_dwa.eval_word(&probe).is_empty(),
                witness.nodes.len(),
                witness.token_id,
            );
        }
        merge_root_weight(roots[start], weight);
    }

    // One-byte seed relations are a separate lane in the direct terminal
    // compiler. Attach their exact support to the corresponding one-terminal
    // stack preimage here as well, using the installed delta template when the
    // splice delta plan rewrites that seed.
    for (sequence, by_state) in seed_relations {
        debug_assert_eq!(sequence.len(), 1);
        let terminal = sequence[0];
        let effective_terminal = delta_plan
            .and_then(|plan| plan.by_global_terminal.get(&terminal))
            .map(|entry| entry.delta_terminal)
            .unwrap_or(terminal);
        let Some(bundle) = build_boolean_terminal_bundle_nwa(templates, &[effective_terminal]) else {
            continue;
        };
        let Some(root) = arena.preimage_bundle(&bundle, LazyBooleanParserDomains::UNIVERSAL)
        else {
            continue;
        };
        let mut tokens_by_tsid = BTreeMap::<u32, BTreeSet<u32>>::new();
        for (&raw_state, originals) in by_state {
            let Some(&tsid) = component_state_map.original_to_internal.get(raw_state as usize) else {
                continue;
            };
            if tsid == u32::MAX {
                continue;
            }
            let tokens = tokens_by_tsid.entry(tsid).or_default();
            tokens.extend(
                originals
                    .iter()
                    .filter_map(|&original| id_map.internal_token_for_original(original)),
            );
        }
        let weight = Weight::from_per_tsid_token_sets(tokens_by_tsid.into_iter().map(
            |(tsid, tokens)| (tsid, tokens.into_iter().collect::<RangeSetBlaze<_>>()),
        ));
        if std::env::var_os("GLRMASK_DEBUG_BOUNDARY_LAZY_LANE_17122").is_some()
            && by_state
                .values()
                .any(|tokens| tokens.contains(&17122) || tokens.contains(&534))
            && !weight.tokens_for_tsid(0).is_empty()
        {
            let probe_17122 = [444i32, DEFAULT_LABEL, 26, 1];
            let probe_534 = [8810i32, DEFAULT_LABEL, 8723, 1];
            let root_dwa = normalize_parser_stack_domain_nwa(table, &arena.to_nwa(root));
            eprintln!(
                "[glrmask/debug][lazy_lane_seed] sequence={sequence:?} effective_terminal={effective_terminal} root={root} probe_17122_accept={} probe_534_accept={} contains_17122={} contains_534={} by_state_rows={}",
                !root_dwa.eval_word(&probe_17122).is_empty(),
                !root_dwa.eval_word(&probe_534).is_empty(),
                by_state.values().any(|tokens| tokens.contains(&17122)),
                by_state.values().any(|tokens| tokens.contains(&534)),
                by_state.len(),
            );
        }
        merge_root_weight(root, weight);
    }

    let distinct_weighted_roots = root_weights.len();
    let root_dwas = root_weights
        .into_iter()
        .map(|(root, weight)| {
            (
                normalize_parser_stack_domain_nwa_preserving_explicit(table, &arena.to_nwa(root)),
                weight,
            )
        })
        .collect::<Vec<_>>();
    let root_dwa_states = root_dwas
        .iter()
        .map(|(dwa, _)| dwa.states().len())
        .sum::<usize>();
    let root_dwa_transitions = root_dwas
        .iter()
        .map(|(dwa, _)| dwa.num_transitions())
        .sum::<usize>();
    let candidate = combine_weighted_prefix_root_dwas(&root_dwas);
    eprintln!(
        "[glrmask/profile][constraint_boundary_lazy_direct_parser] weighted_roots={} root_dwa_states={} root_dwa_transitions={} dwa_states={} dwa_transitions={} total_ms={:.3}",
        distinct_weighted_roots,
        root_dwa_states,
        root_dwa_transitions,
        candidate.states().len(),
        candidate.num_transitions(),
        direct_started_at.elapsed().as_secs_f64() * 1000.0,
    );
    Some(candidate)
}

fn profile_boundary_shared_domain_dp(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_SHARED_DOMAINS").is_none() {
        return;
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CanonicalKey {
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

    let canonicalize_started_at = Instant::now();
    let mut canonical_by_key = BTreeMap::<CanonicalKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    let mut witness_starts = Vec::<usize>::with_capacity(discovery.witnesses.len());
    for witness in &discovery.witnesses {
        let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
        let mut good_nodes = witness
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
            .collect::<Vec<_>>();
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
                if ignored_terminals.contains(edge.terminal as usize) {
                    epsilons.push(target);
                } else {
                    transitions.push((edge.terminal, target));
                }
            }
            transitions.sort_unstable();
            transitions.dedup();
            epsilons.sort_unstable();
            epsilons.dedup();
            let key = CanonicalKey {
                accepting: witness.accepting[local],
                transitions,
                epsilons,
            };
            let canonical = if let Some(&existing) = canonical_by_key.get(&key) {
                existing
            } else {
                let canonical = canonical_nodes.len();
                debug_assert!(key.transitions.iter().all(|(_, target)| *target < canonical));
                debug_assert!(key.epsilons.iter().all(|target| *target < canonical));
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
        witness_starts.push(local_to_canonical[0]);
    }
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for node in &canonical_nodes {
        let mut by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            by_target.entry(target).or_default().push(terminal);
        }
        for (_, mut terminals) in by_target {
            terminals.sort_unstable();
            terminals.dedup();
            terminal_sets.insert(terminals);
        }
    }
    let bundle_started_at = Instant::now();
    let prebuilt_bundles = terminal_sets
        .par_iter()
        .filter_map(|terminals| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
        })
        .collect::<FxHashMap<_, _>>();
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;

    let dp_started_at = Instant::now();
    let mut arena = SharedBooleanParserDomains::new();
    let mut roots = vec![SharedBooleanParserDomains::EMPTY; canonical_nodes.len()];
    let mut preimage_calls = 0usize;
    let mut preimage_cache_hits = 0usize;
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
    let mut branch_unions = 0usize;
    for (node_id, node) in canonical_nodes.iter().enumerate() {
        if node.accepting {
            roots[node_id] = SharedBooleanParserDomains::UNIVERSAL;
            continue;
        }
        let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            terminals_by_target.entry(target).or_default().push(terminal);
        }
        let mut branches = Vec::<u32>::new();
        for (target, mut terminals) in terminals_by_target {
            terminals.sort_unstable();
            terminals.dedup();
            let bundle = prebuilt_bundles
                .get(&terminals)
                .expect("every shared-domain terminal bundle must be prebuilt");
            let target_root = roots[target];
            let cache_key = (terminals.clone(), target_root);
            let root = if let Some(&cached) = preimage_cache.get(&cache_key) {
                preimage_cache_hits += 1;
                cached
            } else {
                let root = arena
                    .preimage_bundle(bundle, target_root)
                    .expect("shared-domain preimage requires read-then-push templates");
                preimage_cache.insert(cache_key, root);
                root
            };
            preimage_calls += 1;
            if root != SharedBooleanParserDomains::EMPTY {
                branches.push(root);
            }
        }
        branches.extend(node.epsilons.iter().map(|&target| roots[target]));
        branch_unions += usize::from(branches.len() > 1);
        roots[node_id] = arena.union_all(branches);
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;

    if std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_SHARED_DOMAINS").is_some() {
        let mut reference = vec![None::<DWA>; canonical_nodes.len()];
        for (node_id, node) in canonical_nodes.iter().enumerate() {
            let domain = if node.accepting {
                universal_parser_stack_domain_dwa()
            } else {
                let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
                for &(terminal, target) in &node.transitions {
                    terminals_by_target.entry(target).or_default().push(terminal);
                }
                let mut branches = Vec::<DWA>::new();
                for (target, mut terminals) in terminals_by_target {
                    terminals.sort_unstable();
                    terminals.dedup();
                    let bundle = prebuilt_bundles
                        .get(&terminals)
                        .expect("validation terminal bundle must be prebuilt");
                    let target_domain = reference[target]
                        .as_ref()
                        .expect("validation dependency must precede source");
                    let (branch, _) =
                        build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled(
                            table,
                            bundle,
                            target_domain,
                        );
                    if let Some(branch) = branch
                        && !branch.states().is_empty()
                    {
                        branches.push(branch);
                    }
                }
                for &target in &node.epsilons {
                    branches.push(
                        reference[target]
                            .as_ref()
                            .expect("validation epsilon dependency must precede source")
                            .clone(),
                    );
                }
                let refs = branches.iter().collect::<Vec<_>>();
                union_parser_stack_domain_dwas(table, &refs)
            };

            let shared = arena.to_dwa(roots[node_id]);
            let difference = find_difference(&shared, &domain)
                .expect("shared parser domains are acyclic during validation");
            assert_eq!(
                difference,
                None,
                "shared parser-domain DP differs at canonical suffix node {node_id}",
            );
            reference[node_id] = Some(domain);
        }
        eprintln!(
            "[glrmask/validate][constraint_boundary_shared_domains] nodes={} exact=true",
            canonical_nodes.len(),
        );
    }

    let unique_suffix_roots = roots.iter().copied().collect::<FxHashSet<_>>().len();
    let unique_witness_roots = witness_starts
        .iter()
        .map(|&start| roots[start])
        .collect::<FxHashSet<_>>()
        .len();
    eprintln!(
        "[glrmask/profile][constraint_boundary_shared_domains] canonical_nodes={} witnesses={} unique_suffix_roots={} unique_witness_roots={} graph_nodes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} branch_unions={} unique_bundles={} canonicalize_ms={canonicalize_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3}",
        canonical_nodes.len(),
        witness_starts.len(),
        unique_suffix_roots,
        unique_witness_roots,
        arena.node_count(),
        preimage_calls,
        preimage_cache_hits,
        preimage_cache.len(),
        branch_unions,
        prebuilt_bundles.len(),
    );
}

fn profile_boundary_suffix_domain_dp(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_SUFFIX_DOMAINS").is_none() {
        return;
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CanonicalKey {
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

    let canonicalize_started_at = Instant::now();
    let mut canonical_by_key = BTreeMap::<CanonicalKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    let mut witness_starts = Vec::<usize>::with_capacity(discovery.witnesses.len());
    for witness in &discovery.witnesses {
        let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
        let mut good_nodes = witness
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
            .collect::<Vec<_>>();
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
                if ignored_terminals.contains(edge.terminal as usize) {
                    epsilons.push(target);
                } else {
                    transitions.push((edge.terminal, target));
                }
            }
            transitions.sort_unstable();
            transitions.dedup();
            epsilons.sort_unstable();
            epsilons.dedup();
            let key = CanonicalKey {
                accepting: witness.accepting[local],
                transitions,
                epsilons,
            };
            let canonical = if let Some(&existing) = canonical_by_key.get(&key) {
                existing
            } else {
                let canonical = canonical_nodes.len();
                debug_assert!(key
                    .transitions
                    .iter()
                    .all(|(_, target)| *target < canonical));
                debug_assert!(key.epsilons.iter().all(|target| *target < canonical));
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
        witness_starts.push(local_to_canonical[0]);
    }
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut levels = vec![0usize; canonical_nodes.len()];
    for (node_id, node) in canonical_nodes.iter().enumerate() {
        let dependency_level = node
            .transitions
            .iter()
            .map(|(_, target)| levels[*target])
            .chain(node.epsilons.iter().map(|target| levels[*target]))
            .max()
            .unwrap_or(0);
        levels[node_id] = if node.accepting { 0 } else { dependency_level + 1 };
    }
    let max_level = levels.iter().copied().max().unwrap_or(0);
    let mut nodes_by_level = vec![Vec::<usize>::new(); max_level + 1];
    for (node, &level) in levels.iter().enumerate() {
        nodes_by_level[level].push(node);
    }

    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for node in &canonical_nodes {
        let mut by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            by_target.entry(target).or_default().push(terminal);
        }
        for (_, mut terminals) in by_target {
            terminals.sort_unstable();
            terminals.dedup();
            terminal_sets.insert(terminals);
        }
    }
    let bundle_prebuild_started_at = Instant::now();
    let prebuilt_bundles = terminal_sets
        .par_iter()
        .filter_map(|terminals| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
        })
        .collect::<FxHashMap<_, _>>();
    let bundle_prebuild_ms = bundle_prebuild_started_at.elapsed().as_secs_f64() * 1000.0;

    let dp_started_at = Instant::now();
    let mut domains = vec![None::<DWA>; canonical_nodes.len()];
    let preimage_calls = AtomicUsize::new(0);
    let bundle_terminal_sum = AtomicUsize::new(0);
    let branch_unions = AtomicUsize::new(0);
    let preimage_bundle_ns = AtomicUsize::new(0);
    let preimage_concat_ns = AtomicUsize::new(0);
    let preimage_resolve_ns = AtomicUsize::new(0);
    let preimage_normalize_ns = AtomicUsize::new(0);
    let preimage_total_ns = AtomicUsize::new(0);
    let preimage_bundle_states = AtomicUsize::new(0);
    let preimage_concat_states = AtomicUsize::new(0);
    for (level, nodes) in nodes_by_level.iter().enumerate() {
        let computed = nodes
            .par_iter()
            .map(|&node_id| {
                let node = &canonical_nodes[node_id];
                if node.accepting {
                    return (node_id, universal_parser_stack_domain_dwa());
                }
                let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
                for &(terminal, target) in &node.transitions {
                    terminals_by_target.entry(target).or_default().push(terminal);
                }
                let mut branch_domains = Vec::<DWA>::new();
                for (target, mut terminals) in terminals_by_target {
                    terminals.sort_unstable();
                    terminals.dedup();
                    bundle_terminal_sum.fetch_add(terminals.len(), Ordering::Relaxed);
                    preimage_calls.fetch_add(1, Ordering::Relaxed);
                    let target_domain = domains[target]
                        .as_ref()
                        .expect("suffix-domain dependency must be in an earlier level");
                    let bundle = prebuilt_bundles
                        .get(&terminals)
                        .expect("every suffix terminal bundle must be prebuilt");
                    let (domain, profile) = build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled(
                        table,
                        bundle,
                        target_domain,
                    );
                    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_BOOLEAN_PREIMAGE").is_some() {
                        let direct = domain.as_ref().expect("direct preimage must exist");
                        let (reference, _) = build_prebuilt_terminal_bundle_preimage_domain_dwa_profiled(
                            table, bundle, target_domain,
                        );
                        let reference = reference.expect("reference preimage must exist");
                        let difference = find_difference(direct, &reference)
                            .expect("preimage domains must be acyclic");
                        if let Some(witness) = difference {
                            let direct_weight = direct.eval_word(&witness);
                            let reference_weight = reference.eval_word(&witness);
                            panic!(
                                "direct preimage mismatch terminals={terminals:?} target_states={} bundle_states={} witness={witness:?} direct_accept={} reference_accept={} target={target_domain:?}",
                                target_domain.states().len(),
                                bundle.states().len(),
                                !direct_weight.is_empty(),
                                !reference_weight.is_empty(),
                            );
                        }
                    }
                    preimage_bundle_ns.fetch_add((profile.bundle_ms * 1_000_000.0) as usize, Ordering::Relaxed);
                    preimage_concat_ns.fetch_add((profile.concatenate_ms * 1_000_000.0) as usize, Ordering::Relaxed);
                    preimage_resolve_ns.fetch_add((profile.resolve_ms * 1_000_000.0) as usize, Ordering::Relaxed);
                    preimage_normalize_ns.fetch_add((profile.normalize_ms * 1_000_000.0) as usize, Ordering::Relaxed);
                    preimage_total_ns.fetch_add((profile.total_ms * 1_000_000.0) as usize, Ordering::Relaxed);
                    preimage_bundle_states.fetch_add(profile.bundle_states, Ordering::Relaxed);
                    preimage_concat_states.fetch_add(profile.concatenated_states, Ordering::Relaxed);
                    let Some(domain) = domain else {
                        continue;
                    };
                    if !domain.states().is_empty() {
                        branch_domains.push(domain);
                    }
                }
                for &target in &node.epsilons {
                    branch_domains.push(
                        domains[target]
                            .as_ref()
                            .expect("epsilon suffix-domain dependency must be earlier")
                            .clone(),
                    );
                }
                let refs = branch_domains.iter().collect::<Vec<_>>();
                if refs.len() > 1 {
                    branch_unions.fetch_add(1, Ordering::Relaxed);
                }
                (node_id, union_parser_stack_domain_dwas(table, &refs))
            })
            .collect::<Vec<_>>();
        for (node_id, domain) in computed {
            debug_assert_eq!(levels[node_id], level);
            domains[node_id] = Some(domain);
        }
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;
    let domains = domains
        .into_iter()
        .map(|domain| domain.expect("every canonical suffix node must have a domain"))
        .collect::<Vec<_>>();

    let mut preimage_keys = FxHashSet::<(Vec<u32>, Vec<(bool, Vec<(i32, u32)>)>)>::default();
    let mut unique_terminal_bundles = FxHashSet::<Vec<u32>>::default();
    for node in &canonical_nodes {
        let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
        for &(terminal, target) in &node.transitions {
            terminals_by_target.entry(target).or_default().push(terminal);
        }
        for (target, mut terminals) in terminals_by_target {
            terminals.sort_unstable();
            terminals.dedup();
            unique_terminal_bundles.insert(terminals.clone());
            preimage_keys.insert((
                terminals,
                canonical_unweighted_dwa_signature(&domains[target]),
            ));
        }
    }
    let unique_preimage_keys = preimage_keys.len();
    let unique_signatures = domains
        .iter()
        .map(canonical_unweighted_dwa_signature)
        .collect::<FxHashSet<_>>()
        .len();
    let root_signatures = witness_starts
        .iter()
        .map(|&start| canonical_unweighted_dwa_signature(&domains[start]))
        .collect::<FxHashSet<_>>()
        .len();
    let total_states = domains.iter().map(|domain| domain.states().len()).sum::<usize>();
    let total_transitions = domains.iter().map(DWA::num_transitions).sum::<usize>();
    let max_states = domains.iter().map(|domain| domain.states().len()).max().unwrap_or(0);
    eprintln!(
        "[glrmask/profile][constraint_boundary_suffix_domains] canonical_nodes={} witness_roots={} levels={} level_sizes={:?} unique_node_domains={} unique_root_domains={} preimage_calls={} unique_preimage_keys={} unique_terminal_bundles={} bundle_terminal_sum={} branch_unions={} total_states={} total_transitions={} max_states={} preimage_bundle_cpu_ms={:.3} preimage_concat_cpu_ms={:.3} preimage_resolve_cpu_ms={:.3} preimage_normalize_cpu_ms={:.3} preimage_total_cpu_ms={:.3} preimage_bundle_states={} preimage_concat_states={} bundle_prebuild_ms={bundle_prebuild_ms:.3} canonicalize_ms={canonicalize_ms:.3} dp_ms={dp_ms:.3}",
        canonical_nodes.len(), witness_starts.len(), nodes_by_level.len(), nodes_by_level.iter().map(Vec::len).collect::<Vec<_>>(), unique_signatures, root_signatures,
        preimage_calls.load(Ordering::Relaxed), unique_preimage_keys, unique_terminal_bundles.len(), bundle_terminal_sum.load(Ordering::Relaxed), branch_unions.load(Ordering::Relaxed), total_states, total_transitions, max_states,
        preimage_bundle_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        preimage_concat_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        preimage_resolve_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        preimage_normalize_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        preimage_total_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0,
        preimage_bundle_states.load(Ordering::Relaxed),
        preimage_concat_states.load(Ordering::Relaxed),
    );
}

fn profile_boundary_suffix_nwa_dp(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    ignored_terminals: &BitSet,
) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_SUFFIX_NWA_DOMAINS").is_none() {
        return;
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct CanonicalKey {
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

    let canonicalize_started_at = Instant::now();
    let mut canonical_by_key = BTreeMap::<CanonicalKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    let mut witness_starts = Vec::<usize>::with_capacity(discovery.witnesses.len());
    for witness in &discovery.witnesses {
        let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
        let mut good_nodes = witness
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
            .collect::<Vec<_>>();
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
                if ignored_terminals.contains(edge.terminal as usize) {
                    epsilons.push(target);
                } else {
                    transitions.push((edge.terminal, target));
                }
            }
            transitions.sort_unstable();
            transitions.dedup();
            epsilons.sort_unstable();
            epsilons.dedup();
            let key = CanonicalKey {
                accepting: witness.accepting[local],
                transitions,
                epsilons,
            };
            let canonical = if let Some(&existing) = canonical_by_key.get(&key) {
                existing
            } else {
                let canonical = canonical_nodes.len();
                debug_assert!(key.transitions.iter().all(|(_, target)| *target < canonical));
                debug_assert!(key.epsilons.iter().all(|target| *target < canonical));
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
        witness_starts.push(local_to_canonical[0]);
    }
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut levels = vec![0usize; canonical_nodes.len()];
    for (node_id, node) in canonical_nodes.iter().enumerate() {
        let dependency_level = node
            .transitions
            .iter()
            .map(|(_, target)| levels[*target])
            .chain(node.epsilons.iter().map(|target| levels[*target]))
            .max()
            .unwrap_or(0);
        levels[node_id] = if node.accepting { 0 } else { dependency_level + 1 };
    }
    let max_level = levels.iter().copied().max().unwrap_or(0);
    let mut nodes_by_level = vec![Vec::<usize>::new(); max_level + 1];
    for (node, &level) in levels.iter().enumerate() {
        nodes_by_level[level].push(node);
    }

    let dp_started_at = Instant::now();
    let mut domains = vec![None::<NWA>; canonical_nodes.len()];
    let preimage_calls = AtomicUsize::new(0);
    let bundle_terminal_sum = AtomicUsize::new(0);
    let branch_unions = AtomicUsize::new(0);
    for nodes in &nodes_by_level {
        let computed = nodes
            .par_iter()
            .map(|&node_id| {
                let node = &canonical_nodes[node_id];
                if node.accepting {
                    return (node_id, universal_parser_stack_domain_nwa());
                }
                let mut terminals_by_target = BTreeMap::<usize, Vec<u32>>::new();
                for &(terminal, target) in &node.transitions {
                    terminals_by_target.entry(target).or_default().push(terminal);
                }
                let mut branches = Vec::<NWA>::new();
                for (target, mut terminals) in terminals_by_target {
                    terminals.sort_unstable();
                    terminals.dedup();
                    bundle_terminal_sum.fetch_add(terminals.len(), Ordering::Relaxed);
                    preimage_calls.fetch_add(1, Ordering::Relaxed);
                    let target_domain = domains[target]
                        .as_ref()
                        .expect("suffix NWA dependency must be earlier");
                    if let Some(domain) = build_terminal_bundle_preimage_domain_nwa(
                        table,
                        templates,
                        &terminals,
                        target_domain,
                    ) {
                        if !domain.states().is_empty() {
                            branches.push(domain);
                        }
                    }
                }
                for &target in &node.epsilons {
                    branches.push(
                        domains[target]
                            .as_ref()
                            .expect("epsilon suffix NWA dependency must be earlier")
                            .clone(),
                    );
                }
                let refs = branches.iter().collect::<Vec<_>>();
                if refs.len() > 1 {
                    branch_unions.fetch_add(1, Ordering::Relaxed);
                }
                (node_id, union_parser_stack_domain_nwas(&refs))
            })
            .collect::<Vec<_>>();
        for (node_id, domain) in computed {
            domains[node_id] = Some(domain);
        }
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;
    let domains = domains
        .into_iter()
        .map(|domain| domain.expect("every canonical suffix node must have an NWA domain"))
        .collect::<Vec<_>>();
    let total_states = domains.iter().map(|domain| domain.states().len()).sum::<usize>();
    let total_transitions = domains.iter().map(NWA::num_transitions).sum::<usize>();
    let max_states = domains.iter().map(|domain| domain.states().len()).max().unwrap_or(0);
    let root_state_sum = witness_starts
        .iter()
        .map(|&root| domains[root].states().len())
        .sum::<usize>();

    let normalize_roots = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_SUFFIX_NWA_NORMALIZE_ROOTS").is_some();
    let normalize_started_at = Instant::now();
    let normalized_root_states = if normalize_roots {
        witness_starts
            .par_iter()
            .map(|&root| normalize_parser_stack_domain_nwa(table, &domains[root]).states().len())
            .sum::<usize>()
    } else {
        0
    };
    let normalize_root_ms = if normalize_roots {
        normalize_started_at.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };

    eprintln!(
        "[glrmask/profile][constraint_boundary_suffix_nwa_domains] canonical_nodes={} witness_roots={} levels={} level_sizes={:?} preimage_calls={} bundle_terminal_sum={} branch_unions={} total_states={} total_transitions={} max_states={} root_state_sum={} normalized_root_states={} canonicalize_ms={canonicalize_ms:.3} dp_ms={dp_ms:.3} normalize_root_ms={normalize_root_ms:.3}",
        canonical_nodes.len(), witness_starts.len(), nodes_by_level.len(),
        nodes_by_level.iter().map(Vec::len).collect::<Vec<_>>(),
        preimage_calls.load(Ordering::Relaxed), bundle_terminal_sum.load(Ordering::Relaxed),
        branch_unions.load(Ordering::Relaxed), total_states, total_transitions, max_states,
        root_state_sum, normalized_root_states,
    );
}

fn profile_token_sharded_boundary_parser(
    composed_table: &ComposedTable,
    analyzed: &AnalyzedGrammar,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    component_state_map: &ManyToOneIdMap,
    merged_tokenizer_state_count: usize,
    vocab: &Vocab,
    globally_erasable_ignore_terminals: &BitSet,
) -> Result<(), String> {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TOKEN_SHARDS").is_none() {
        return Ok(());
    }
    let mut by_token = BTreeMap::<u32, Vec<BoundaryTokenWitness>>::new();
    for witness in &discovery.witnesses {
        by_token
            .entry(witness.token_id)
            .or_default()
            .push(witness.clone());
    }
    let shards = by_token
        .into_iter()
        .map(|(token_id, witnesses)| BoundaryTokenDiscovery {
            terminals: BitSet::new(analyzed.num_terminals as usize),
            token_ids: vec![token_id],
            witnesses,
        })
        .collect::<Vec<_>>();
    let wall_started_at = Instant::now();
    let results = shards
        .par_iter()
        .map(|shard| -> Result<(u32, f64, usize, usize), String> {
            let terminal = direct_boundary_terminal_automaton(
                merged_tokenizer_state_count,
                Some(component_state_map),
                vocab,
                BTreeMap::new(),
                0.0,
                shard,
                globally_erasable_ignore_terminals,
                &composed_table.control_terminals,
                None,
            )?;
            let (terminal_automaton, id_map) = terminal.into_parts();
            let parser_started_at = Instant::now();
            let parser = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &composed_table.table,
                analyzed,
                &terminal_automaton,
                templates,
                vocab,
                &id_map,
                false,
            );
            Ok((
                shard.token_ids[0],
                parser_started_at.elapsed().as_secs_f64() * 1000.0,
                parser.states().len(),
                parser.num_transitions(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let wall_ms = wall_started_at.elapsed().as_secs_f64() * 1000.0;
    let parser_cpu_ms = results.iter().map(|(_, ms, _, _)| *ms).sum::<f64>();
    let max_parser_ms = results
        .iter()
        .map(|(_, ms, _, _)| *ms)
        .fold(0.0f64, f64::max);
    let total_states = results.iter().map(|(_, _, states, _)| *states).sum::<usize>();
    let total_transitions = results
        .iter()
        .map(|(_, _, _, transitions)| *transitions)
        .sum::<usize>();
    let max_states = results
        .iter()
        .map(|(_, _, states, _)| *states)
        .max()
        .unwrap_or(0);
    eprintln!(
        "[glrmask/profile][constraint_boundary_token_shards] shards={} witnesses={} parser_cpu_ms={parser_cpu_ms:.3} max_parser_ms={max_parser_ms:.3} total_states={} total_transitions={} max_states={} wall_ms={wall_ms:.3}",
        shards.len(), discovery.witnesses.len(), total_states, total_transitions, max_states,
    );
    Ok(())
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
    delta_plan: Option<&BoundaryTemplateDeltaPlan>,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    let total_started_at = Instant::now();
    profile_boundary_witness_terminal_language(discovery, globally_erasable_ignore_terminals);

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
    // entering its graph. In delta mode, genuinely cross-component paths keep
    // the full composed terminal templates. Component-local paths instead
    // branch between the transported old template and the exact template
    // delta, and only paths that use at least one delta are accepted. The
    // all-old path is already present in the reused component parser DWA.
    let mut canonical_by_key = BTreeMap::<CanonicalNodeKey, usize>::new();
    let mut canonical_nodes = Vec::<CanonicalNode>::new();
    let mut start_weights_by_canonical = BTreeMap::<usize, Weight>::new();
    let intern_canonical = |key: CanonicalNodeKey,
                            canonical_by_key: &mut BTreeMap<CanonicalNodeKey, usize>,
                            canonical_nodes: &mut Vec<CanonicalNode>| {
        if let Some(&existing) = canonical_by_key.get(&key) {
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
        }
    };

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

        let mut good_nodes = witness
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(local, node)| witness.good[local].then_some((local, node.key.offset)))
            .collect::<Vec<_>>();
        // Every graph edge consumes positive width, so byte offset is a
        // topological rank. Node allocation order is deliberately irrelevant.
        good_nodes.sort_unstable_by(|left, right| right.1.cmp(&left.1));

        if let Some(delta_plan) = delta_plan {
            let mut good_local = vec![false; witness.nodes.len()];
            let mut good_cross = vec![false; witness.nodes.len()];
            for &(local, _) in &good_nodes {
                if witness.accepting[local] {
                    match witness.nodes[local].key.lexical_component {
                        BOUNDARY_COMPONENT_CROSSED => good_cross[local] = true,
                        BOUNDARY_COMPONENT_NONE => {}
                        _ => good_local[local] = true,
                    }
                }
                for edge in &witness.nodes[local].outgoing {
                    if !witness.good[edge.target] {
                        continue;
                    }
                    good_local[local] |= good_local[edge.target];
                    good_cross[local] |= good_cross[edge.target];
                }
            }

            // Full composed semantics for genuinely cross-component lexical
            // paths. These token/terminal sequences do not exist in any one
            // reused component artifact.
            let mut cross_map = vec![usize::MAX; witness.nodes.len()];
            for &(local, _) in &good_nodes {
                if !good_cross[local] {
                    continue;
                }
                let mut transitions = Vec::new();
                let mut epsilons = Vec::new();
                for edge in witness.nodes[local]
                    .outgoing
                    .iter()
                    .filter(|edge| good_cross[edge.target])
                {
                    let target = cross_map[edge.target];
                    debug_assert_ne!(target, usize::MAX);
                    if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                        epsilons.push(target);
                    } else {
                        transitions.push((edge.terminal, target));
                    }
                }
                transitions.sort_unstable();
                transitions.dedup();
                epsilons.sort_unstable();
                epsilons.dedup();
                cross_map[local] = intern_canonical(
                    CanonicalNodeKey {
                        accepting: witness.accepting[local]
                            && witness.nodes[local].key.lexical_component
                                == BOUNDARY_COMPONENT_CROSSED,
                        transitions,
                        epsilons,
                    },
                    &mut canonical_by_key,
                    &mut canonical_nodes,
                );
            }
            if good_cross[0] {
                let start = cross_map[0];
                start_weights_by_canonical
                    .entry(start)
                    .and_modify(|existing| *existing = existing.union(&witness_weight))
                    .or_insert_with(|| witness_weight.clone());
            }

            // In the common reusable case every component-local accepted
            // terminal path contains exactly one linker-changed terminal. Prove
            // that property on the tiny witness DAG before specializing. Then
            // the exact boundary delta needs no old/new tracking state at all:
            // relabel that unique changed terminal with its `new \ old`
            // template and leave every unchanged terminal untouched.
            let mut local_changed_counts = vec![BTreeSet::<u8>::new(); witness.nodes.len()];
            for &(local, _) in &good_nodes {
                if witness.accepting[local]
                    && witness.nodes[local].key.lexical_component
                        != BOUNDARY_COMPONENT_CROSSED
                    && witness.nodes[local].key.lexical_component
                        != BOUNDARY_COMPONENT_NONE
                {
                    local_changed_counts[local].insert(0);
                }
                for edge in witness.nodes[local]
                    .outgoing
                    .iter()
                    .filter(|edge| good_local[edge.target])
                {
                    let changed = u8::from(
                        !globally_erasable_ignore_terminals.contains(edge.terminal as usize)
                            && delta_plan.changed_terminals.contains(&edge.terminal),
                    );
                    let child_counts = local_changed_counts[edge.target]
                        .iter()
                        .copied()
                        .collect::<Vec<_>>();
                    for count in child_counts {
                        local_changed_counts[local].insert(count.saturating_add(changed).min(2));
                    }
                }
            }
            let local_unique_delta = !good_local[0]
                || local_changed_counts[0].iter().copied().eq(std::iter::once(1));

            if local_unique_delta {
                let mut local_map = vec![usize::MAX; witness.nodes.len()];
                for &(local, _) in &good_nodes {
                    if !good_local[local] {
                        continue;
                    }
                    let mut transitions = Vec::new();
                    let mut epsilons = Vec::new();
                    for edge in witness.nodes[local]
                        .outgoing
                        .iter()
                        .filter(|edge| good_local[edge.target])
                    {
                        let target = local_map[edge.target];
                        debug_assert_ne!(target, usize::MAX);
                        if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                            epsilons.push(target);
                        } else if let Some(entry) = delta_plan.by_global_terminal.get(&edge.terminal) {
                            transitions.push((entry.delta_terminal, target));
                        } else {
                            // If the one changed terminal lacks a retained old
                            // template, preserve the full composed template. It
                            // may retain duplicate old behavior but cannot drop
                            // any valid boundary behavior.
                            transitions.push((edge.terminal, target));
                        }
                    }
                    transitions.sort_unstable();
                    transitions.dedup();
                    epsilons.sort_unstable();
                    epsilons.dedup();
                    local_map[local] = intern_canonical(
                        CanonicalNodeKey {
                            accepting: witness.accepting[local]
                                && witness.nodes[local].key.lexical_component
                                    != BOUNDARY_COMPONENT_CROSSED
                                && witness.nodes[local].key.lexical_component
                                    != BOUNDARY_COMPONENT_NONE,
                            transitions,
                            epsilons,
                        },
                        &mut canonical_by_key,
                        &mut canonical_nodes,
                    );
                }
                if good_local[0] {
                    let start = local_map[0];
                    start_weights_by_canonical
                        .entry(start)
                        .and_modify(|existing| *existing = existing.union(&witness_weight))
                        .or_insert(witness_weight);
                }
            } else {
                // The certification is structural, not a heuristic. If a future
                // grammar has zero or multiple changed terminals on a local
                // boundary path, keep the full composed path rather than apply
                // an invalid subtraction shortcut.
                let mut local_map = vec![usize::MAX; witness.nodes.len()];
                for &(local, _) in &good_nodes {
                    if !good_local[local] {
                        continue;
                    }
                    let mut transitions = Vec::new();
                    let mut epsilons = Vec::new();
                    for edge in witness.nodes[local]
                        .outgoing
                        .iter()
                        .filter(|edge| good_local[edge.target])
                    {
                        let target = local_map[edge.target];
                        debug_assert_ne!(target, usize::MAX);
                        if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                            epsilons.push(target);
                        } else {
                            transitions.push((edge.terminal, target));
                        }
                    }
                    transitions.sort_unstable();
                    transitions.dedup();
                    epsilons.sort_unstable();
                    epsilons.dedup();
                    local_map[local] = intern_canonical(
                        CanonicalNodeKey {
                            accepting: witness.accepting[local]
                                && witness.nodes[local].key.lexical_component
                                    != BOUNDARY_COMPONENT_CROSSED
                                && witness.nodes[local].key.lexical_component
                                    != BOUNDARY_COMPONENT_NONE,
                            transitions,
                            epsilons,
                        },
                        &mut canonical_by_key,
                        &mut canonical_nodes,
                    );
                }
                if good_local[0] {
                    let start = local_map[0];
                    start_weights_by_canonical
                        .entry(start)
                        .and_modify(|existing| *existing = existing.union(&witness_weight))
                        .or_insert(witness_weight);
                }
            }
        } else {
            let mut local_to_canonical = vec![usize::MAX; witness.nodes.len()];
            for &(local, _) in &good_nodes {
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
                        epsilons.push(target);
                    } else {
                        transitions.push((edge.terminal, target));
                    }
                }
                transitions.sort_unstable();
                transitions.dedup();
                epsilons.sort_unstable();
                epsilons.dedup();
                local_to_canonical[local] = intern_canonical(
                    CanonicalNodeKey {
                        accepting: witness.accepting[local],
                        transitions,
                        epsilons,
                    },
                    &mut canonical_by_key,
                    &mut canonical_nodes,
                );
            }
            let start = local_to_canonical[0];
            start_weights_by_canonical
                .entry(start)
                .and_modify(|existing| *existing = existing.union(&witness_weight))
                .or_insert(witness_weight);
        }
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
        if weight.is_empty() {
            continue;
        }
        let terminal = sequence[0];
        if let Some(entry) = delta_plan.and_then(|plan| plan.by_global_terminal.get(&terminal)) {
            // One-byte component-local seed tokens already have their old
            // parser behavior in the component artifact. Only the new linker
            // template delta belongs in boundary repair.
            nwa.add_transition(
                global_start,
                entry.delta_terminal as i32,
                seed_final,
                weight,
            );
        } else {
            nwa.add_transition(global_start, terminal as i32, seed_final, weight);
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

fn subtract_weight_support_from_raw_automata(
    automata: &mut [RawCompressedAutomaton],
    claimed: &Weight,
) {
    if claimed.is_empty() {
        return;
    }
    for automaton in automata {
        for state in &mut automaton.states {
            for run in &mut state.runs {
                for (_, weight) in &mut run.targets {
                    *weight = weight.difference(claimed);
                }
                run.targets.retain(|(_, weight)| !weight.is_empty());
            }
            state.runs.retain(|run| !run.targets.is_empty());
            if let Some(targets) = &mut state.default_targets {
                for (_, weight) in targets.iter_mut() {
                    *weight = weight.difference(claimed);
                }
                targets.retain(|(_, weight)| !weight.is_empty());
                if targets.is_empty() {
                    state.default_targets = None;
                }
            }
            if let Some(final_weight) = state.final_weight.take() {
                let remaining = final_weight.difference(claimed);
                state.final_weight = (!remaining.is_empty()).then_some(remaining);
            }
        }
    }
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

fn acyclic_dwa_topology_classes(
    dwa: &DWA,
    include_final: bool,
    interner: &mut BTreeMap<(bool, Vec<(i32, u32)>), u32>,
) -> Vec<u32> {
    let n = dwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in dwa.states() {
        for (target, weight) in state.transitions.values() {
            if !weight.is_empty() && (*target as usize) < n {
                indegree[*target as usize] += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::with_capacity(n);
    while let Some(state) = queue.pop_front() {
        topo.push(state);
        for (target, weight) in dwa.states()[state as usize].transitions.values() {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            indegree[*target as usize] -= 1;
            if indegree[*target as usize] == 0 {
                queue.push_back(*target);
            }
        }
    }
    assert_eq!(topo.len(), n, "topology overlap profile requires acyclic DWA");

    let mut classes = vec![u32::MAX; n];
    for &state_id in topo.iter().rev() {
        let state = &dwa.states()[state_id as usize];
        let row = state
            .transitions
            .iter()
            .filter(|(_, (_, weight))| !weight.is_empty())
            .map(|(&label, (target, _))| (label, classes[*target as usize]))
            .collect::<Vec<_>>();
        let final_present = include_final
            && state
                .final_weight
                .as_ref()
                .is_some_and(|weight| !weight.is_empty());
        let key = (final_present, row);
        let next = interner.len() as u32;
        let class = *interner.entry(key).or_insert(next);
        classes[state_id as usize] = class;
    }
    classes
}

fn profile_boundary_component_topology_overlap(component: &DWA, boundary: &DWA) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TOPOLOGY_OVERLAP").is_none() {
        return;
    }
    let mut strict_interner = BTreeMap::new();
    let component_strict = acyclic_dwa_topology_classes(component, true, &mut strict_interner);
    let component_strict_set = component_strict.iter().copied().collect::<FxHashSet<_>>();
    let boundary_strict = acyclic_dwa_topology_classes(boundary, true, &mut strict_interner);
    let strict_matches = boundary_strict
        .iter()
        .filter(|class| component_strict_set.contains(class))
        .count();

    let mut relaxed_interner = BTreeMap::new();
    let component_relaxed = acyclic_dwa_topology_classes(component, false, &mut relaxed_interner);
    let component_relaxed_set = component_relaxed.iter().copied().collect::<FxHashSet<_>>();
    let boundary_relaxed = acyclic_dwa_topology_classes(boundary, false, &mut relaxed_interner);
    let relaxed_matches = boundary_relaxed
        .iter()
        .filter(|class| component_relaxed_set.contains(class))
        .count();

    eprintln!(
        "[glrmask/profile][constraint_boundary_topology_overlap] component_states={} boundary_states={} strict_matching_boundary_states={} relaxed_matching_boundary_states={} strict_fraction={:.3} relaxed_fraction={:.3} strict_start_match={} relaxed_start_match={}",
        component.states().len(),
        boundary.states().len(),
        strict_matches,
        relaxed_matches,
        strict_matches as f64 / boundary.states().len().max(1) as f64,
        relaxed_matches as f64 / boundary.states().len().max(1) as f64,
        component_strict_set.contains(&boundary_strict[boundary.start_state() as usize]),
        component_relaxed_set.contains(&boundary_relaxed[boundary.start_state() as usize]),
    );
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
    let (dwa, synthetic_states, _) = determinize_compressed_component_union_prechecked(
        compressed,
        default_positive_label_count,
        raw_compress_ms,
    );
    (dwa, synthetic_states)
}

fn raw_deterministic_topology_classes(states: &[RawCompressedState]) -> Vec<u32> {
    let n = states.len();
    let mut indegree = vec![0usize; n];
    for state in states {
        if !state.deterministic {
            continue;
        }
        for run in &state.runs {
            if let Some((target, weight)) = run.targets.first()
                && !weight.is_empty()
                && (*target as usize) < n
            {
                indegree[*target as usize] += 1;
            }
        }
        if let Some(targets) = &state.default_targets
            && let Some((target, weight)) = targets.first()
            && !weight.is_empty()
            && (*target as usize) < n
        {
            indegree[*target as usize] += 1;
        }
    }
    let mut queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::with_capacity(n);
    while let Some(state_id) = queue.pop_front() {
        topo.push(state_id);
        let state = &states[state_id as usize];
        if !state.deterministic {
            continue;
        }
        for run in &state.runs {
            if let Some((target, weight)) = run.targets.first()
                && !weight.is_empty()
                && (*target as usize) < n
            {
                indegree[*target as usize] -= 1;
                if indegree[*target as usize] == 0 {
                    queue.push_back(*target);
                }
            }
        }
        if let Some(targets) = &state.default_targets
            && let Some((target, weight)) = targets.first()
            && !weight.is_empty()
            && (*target as usize) < n
        {
            indegree[*target as usize] -= 1;
            if indegree[*target as usize] == 0 {
                queue.push_back(*target);
            }
        }
    }
    if topo.len() != n {
        return vec![u32::MAX; n];
    }

    type Signature = (bool, Vec<(i32, i32, u32)>, Option<u32>);
    let mut interner = BTreeMap::<Signature, u32>::new();
    let mut classes = vec![u32::MAX; n];
    for &state_id in topo.iter().rev() {
        let state = &states[state_id as usize];
        if !state.deterministic {
            continue;
        }
        let mut row = Vec::with_capacity(state.runs.len());
        let mut valid = true;
        for run in &state.runs {
            let Some((target, weight)) = run.targets.first() else {
                continue;
            };
            if weight.is_empty() {
                continue;
            }
            let target_class = classes[*target as usize];
            if target_class == u32::MAX {
                valid = false;
                break;
            }
            row.push((run.start, run.end, target_class));
        }
        if !valid {
            continue;
        }
        let default = match state.default_targets.as_deref() {
            Some([(target, weight)]) if !weight.is_empty() => {
                let class = classes[*target as usize];
                if class == u32::MAX {
                    continue;
                }
                Some(class)
            }
            Some(targets) if targets.iter().any(|(_, weight)| !weight.is_empty()) => continue,
            _ => None,
        };
        let key = (
            state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()),
            row,
            default,
        );
        let next = interner.len() as u32;
        classes[state_id as usize] = *interner.entry(key).or_insert(next);
    }
    classes
}

#[derive(Default)]
struct UnionDominanceProfile {
    two_target_overlaps: usize,
    same_topology: usize,
    left_subset_right: usize,
    right_subset_left: usize,
    either_subset: usize,
    equal_weight: usize,
}

struct RawResidualDominance<'a> {
    states: &'a [RawCompressedState],
    topology_classes: &'a [u32],
    memo: FxHashMap<(u32, usize, u32, usize), bool>,
    calls: usize,
    cache_hits: usize,
    proven: usize,
}

impl<'a> RawResidualDominance<'a> {
    fn new(states: &'a [RawCompressedState], topology_classes: &'a [u32]) -> Self {
        Self {
            states,
            topology_classes,
            memo: FxHashMap::default(),
            calls: 0,
            cache_hits: 0,
            proven: 0,
        }
    }

    fn is_subset(
        &mut self,
        left_state: u32,
        left_prefix: &Weight,
        right_state: u32,
        right_prefix: &Weight,
        weight_ops: &mut ScopedWeightOpCache,
    ) -> bool {
        self.calls += 1;
        if left_prefix.is_empty() {
            self.proven += 1;
            return true;
        }
        if right_prefix.is_empty() {
            return false;
        }
        let left_class = self.topology_classes[left_state as usize];
        let right_class = self.topology_classes[right_state as usize];
        if left_class == u32::MAX || left_class != right_class {
            return false;
        }
        let key = (
            left_state,
            left_prefix.ptr_key(),
            right_state,
            right_prefix.ptr_key(),
        );
        if let Some(&cached) = self.memo.get(&key) {
            self.cache_hits += 1;
            return cached;
        }

        // The topology graph is acyclic, so recursive queries always descend.
        // Insert pessimistically only after evaluation; no recursion cycle can
        // revisit this key before completion.
        let left = &self.states[left_state as usize];
        let right = &self.states[right_state as usize];
        if !left.deterministic || !right.deterministic {
            self.memo.insert(key, false);
            return false;
        }

        let final_ok = match (&left.final_weight, &right.final_weight) {
            (Some(left_final), Some(right_final)) => {
                let left_final = weight_ops.intersection(left_prefix, left_final);
                if left_final.is_empty() {
                    true
                } else {
                    let right_final = weight_ops.intersection(right_prefix, right_final);
                    left_final.is_subset(&right_final)
                }
            }
            (Some(left_final), None) => {
                weight_ops.intersection(left_prefix, left_final).is_empty()
            }
            (None, _) => true,
        };
        if !final_ok {
            self.memo.insert(key, false);
            return false;
        }

        let mut left_runs = left.runs.iter().filter(|run| {
            run.targets
                .first()
                .is_some_and(|(_, weight)| !weight.is_empty())
        });
        let mut right_runs = right.runs.iter().filter(|run| {
            run.targets
                .first()
                .is_some_and(|(_, weight)| !weight.is_empty())
        });
        loop {
            match (left_runs.next(), right_runs.next()) {
                (None, None) => break,
                (Some(left_run), Some(right_run)) => {
                    if left_run.start != right_run.start || left_run.end != right_run.end {
                        self.memo.insert(key, false);
                        return false;
                    }
                    let (left_target, left_edge) = &left_run.targets[0];
                    let (right_target, right_edge) = &right_run.targets[0];
                    let next_left = weight_ops.intersection(left_prefix, left_edge);
                    if next_left.is_empty() {
                        continue;
                    }
                    let next_right = weight_ops.intersection(right_prefix, right_edge);
                    if !self.is_subset(
                        *left_target,
                        &next_left,
                        *right_target,
                        &next_right,
                        weight_ops,
                    ) {
                        self.memo.insert(key, false);
                        return false;
                    }
                }
                _ => {
                    self.memo.insert(key, false);
                    return false;
                }
            }
        }

        match (&left.default_targets, &right.default_targets) {
            (Some(left_targets), Some(right_targets)) => {
                let left_live = left_targets
                    .iter()
                    .find(|(_, weight)| !weight.is_empty());
                let right_live = right_targets
                    .iter()
                    .find(|(_, weight)| !weight.is_empty());
                match (left_live, right_live) {
                    (Some((left_target, left_edge)), Some((right_target, right_edge))) => {
                        let next_left = weight_ops.intersection(left_prefix, left_edge);
                        if !next_left.is_empty() {
                            let next_right = weight_ops.intersection(right_prefix, right_edge);
                            if !self.is_subset(
                                *left_target,
                                &next_left,
                                *right_target,
                                &next_right,
                                weight_ops,
                            ) {
                                self.memo.insert(key, false);
                                return false;
                            }
                        }
                    }
                    (Some((_, left_edge)), None) => {
                        if !weight_ops.intersection(left_prefix, left_edge).is_empty() {
                            self.memo.insert(key, false);
                            return false;
                        }
                    }
                    (None, _) => {}
                }
            }
            (Some(left_targets), None) => {
                if let Some((_, left_edge)) = left_targets
                    .iter()
                    .find(|(_, weight)| !weight.is_empty())
                    && !weight_ops.intersection(left_prefix, left_edge).is_empty()
                {
                    self.memo.insert(key, false);
                    return false;
                }
            }
            (None, _) => {}
        }

        self.memo.insert(key, true);
        self.proven += 1;
        true
    }
}

fn determinize_compressed_component_union_prechecked(
    automata: Vec<RawCompressedAutomaton>,
    default_positive_label_count: Option<u32>,
    raw_compress_ms: f64,
) -> (DWA, usize, PrebuiltParserWeightTokenSets) {
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
    let topology_classes = (std::env::var_os("GLRMASK_EXPERIMENT_UNION_DOMINANCE_PROFILE").is_some()
        || std::env::var_os("GLRMASK_EXPERIMENT_UNION_RESIDUAL_DOMINANCE").is_some())
        .then(|| raw_deterministic_topology_classes(&raw_states));
    let mut dominance_profile = UnionDominanceProfile::default();
    if starts.is_empty() {
        return (DWA::new(0, 0), 0, PrebuiltParserWeightTokenSets::default());
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
        topology_classes: Option<&[u32]>,
        dominance_profile: &mut UnionDominanceProfile,
        residual_dominance: Option<&mut RawResidualDominance<'_>>,
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
                dominance_profile.two_target_overlaps += 1;
                if let Some(classes) = topology_classes {
                    let left_class = classes[left_target as usize];
                    let right_class = classes[right_target as usize];
                    if left_class != u32::MAX && left_class == right_class {
                        dominance_profile.same_topology += 1;
                        let left_subset = left_weight.is_subset(&right_weight);
                        let right_subset = right_weight.is_subset(&left_weight);
                        dominance_profile.left_subset_right += usize::from(left_subset);
                        dominance_profile.right_subset_left += usize::from(right_subset);
                        dominance_profile.either_subset += usize::from(left_subset || right_subset);
                        dominance_profile.equal_weight += usize::from(left_subset && right_subset);
                    }
                }
                if let Some(residual_dominance) = residual_dominance {
                    if residual_dominance.is_subset(
                        left_target,
                        &left_weight,
                        right_target,
                        &right_weight,
                        weight_ops,
                    ) {
                        let target = intern_singleton(
                            right_target,
                            singleton_states,
                            singleton_count,
                            states,
                            queue,
                        );
                        return Some((target, right_weight));
                    }
                    if residual_dominance.is_subset(
                        right_target,
                        &right_weight,
                        left_target,
                        &left_weight,
                        weight_ops,
                    ) {
                        let target = intern_singleton(
                            left_target,
                            singleton_states,
                            singleton_count,
                            states,
                            queue,
                        );
                        return Some((target, left_weight));
                    }
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
    let enable_residual_dominance =
        std::env::var_os("GLRMASK_EXPERIMENT_UNION_RESIDUAL_DOMINANCE").is_some();
    let mut residual_dominance = if enable_residual_dominance {
        topology_classes
            .as_deref()
            .map(|classes| RawResidualDominance::new(&raw_states, classes))
    } else {
        None
    };
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
                topology_classes.as_deref(),
                &mut dominance_profile,
                residual_dominance.as_mut(),
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
                topology_classes.as_deref(),
                &mut dominance_profile,
                residual_dominance.as_mut(),
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

    if let Some(residual_dominance) = &residual_dominance {
        eprintln!(
            "[glrmask/profile][constraint_union_residual_dominance] calls={} cache_hits={} proven={} memo_entries={}",
            residual_dominance.calls,
            residual_dominance.cache_hits,
            residual_dominance.proven,
            residual_dominance.memo.len(),
        );
    }
    if topology_classes.is_some() {
        eprintln!(
            "[glrmask/profile][constraint_union_dominance_potential] two_target_overlaps={} same_topology={} either_subset={} equal_weight={} left_subset_right={} right_subset_left={}",
            dominance_profile.two_target_overlaps,
            dominance_profile.same_topology,
            dominance_profile.either_subset,
            dominance_profile.equal_weight,
            dominance_profile.left_subset_right,
            dominance_profile.right_subset_left,
        );
    }
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
            let mut final_sets = FxHashMap::<usize, Arc<RangeSetBlaze<u32>>>::default();
            let mut transition_sets = FxHashMap::<usize, Arc<RangeSetBlaze<u32>>>::default();
            if let Some(final_weight) = state.final_weight.as_ref()
                && !final_weight.is_full()
                && !final_weight.is_empty()
            {
                for (_, token_set) in final_weight.raw_range_values() {
                    final_sets
                        .entry(Arc::as_ptr(token_set) as usize)
                        .or_insert_with(|| Arc::clone(token_set));
                }
            }
            for run in &state.runs {
                for (_, token_set) in run.weight.raw_range_values() {
                    transition_sets
                        .entry(Arc::as_ptr(token_set) as usize)
                        .or_insert_with(|| Arc::clone(token_set));
                }
            }
            if let Some((_, weight)) = state.default_transition.as_ref() {
                for (_, token_set) in weight.raw_range_values() {
                    transition_sets
                        .entry(Arc::as_ptr(token_set) as usize)
                        .or_insert_with(|| Arc::clone(token_set));
                }
            }
            let mut entries = Vec::with_capacity(transition_count);
            if let Some((target, weight)) = state.default_transition {
                entries.push((DEFAULT_LABEL, (target, weight)));
            }
            for run in state.runs {
                for label in run.start..=run.end {
                    entries.push((label, (run.target, run.weight.clone())));
                }
            }
            let dwa_state = DWAState {
                transitions: entries.into_iter().collect(),
                final_weight: state.final_weight,
            };
            (
                dwa_state,
                final_sets.into_values().collect::<Vec<_>>(),
                transition_sets.into_values().collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let mut materialized_states = Vec::with_capacity(states.len());
    let mut final_sets = FxHashMap::<usize, Arc<RangeSetBlaze<u32>>>::default();
    let mut transition_sets = FxHashMap::<usize, Arc<RangeSetBlaze<u32>>>::default();
    for (state, local_final_sets, local_transition_sets) in states {
        materialized_states.push(state);
        for token_set in local_final_sets {
            final_sets
                .entry(Arc::as_ptr(&token_set) as usize)
                .or_insert(token_set);
        }
        for token_set in local_transition_sets {
            transition_sets
                .entry(Arc::as_ptr(&token_set) as usize)
                .or_insert(token_set);
        }
    }
    let states = materialized_states;
    let prebuilt_weight_token_sets = PrebuiltParserWeightTokenSets {
        final_sets: final_sets.into_values().collect(),
        transition_sets: transition_sets.into_values().collect(),
        includes_parser_top_accept: false,
    };
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
    (
        DWA::from_parts(states, start_state),
        synthetic_states,
        prebuilt_weight_token_sets,
    )
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
            remap_weights_with_maps_serial(
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


fn remap_template_label(label: i32, state_relation: &[Vec<u32>]) -> Option<i32> {
    if label == DEFAULT_LABEL {
        return Some(label);
    }
    let (local_state, negative) = if is_negative_label(label) {
        (negative_to_positive_label(label) as u32, true)
    } else if label >= 0 {
        (label as u32, false)
    } else {
        return None;
    };
    let mapped = state_relation.get(local_state as usize)?;
    if mapped.len() != 1 {
        return None;
    }
    Some(if negative {
        encode_negative_label(mapped[0])
    } else {
        encode_positive_label(mapped[0])
    })
}

fn unweighted_dfa_difference(left: &UnweightedDfa, right: &UnweightedDfa) -> UnweightedDfa {
    type Pair = (u32, Option<u32>);
    let start: Pair = (left.start_state, Some(right.start_state));
    let mut output = UnweightedDfa::new();
    let mut state_by_pair = FxHashMap::<Pair, u32>::default();
    state_by_pair.insert(start, output.start_state);
    let mut queue = VecDeque::from([start]);

    while let Some((left_state, right_state)) = queue.pop_front() {
        let output_state = state_by_pair[&(left_state, right_state)];
        let left_node = &left.states[left_state as usize];
        let right_accepting = right_state
            .and_then(|state| right.states.get(state as usize))
            .is_some_and(|state| state.is_accepting);
        output.set_accepting(output_state, left_node.is_accepting && !right_accepting);

        for (&label, &left_target) in &left_node.transitions {
            let right_target = right_state
                .and_then(|state| right.states.get(state as usize))
                .and_then(|state| state.transitions.get(&label).copied());
            let pair = (left_target, right_target);
            let target = if let Some(&target) = state_by_pair.get(&pair) {
                target
            } else {
                let target = output.add_state();
                state_by_pair.insert(pair, target);
                queue.push_back(pair);
                target
            };
            output.add_transition(output_state, label, target);
        }
    }
    output
}

fn unweighted_dfa_language_is_empty(dfa: &UnweightedDfa) -> bool {
    let mut seen = vec![false; dfa.states.len()];
    let mut stack = vec![dfa.start_state];
    while let Some(state) = stack.pop() {
        let Some(seen_state) = seen.get_mut(state as usize) else {
            continue;
        };
        if *seen_state {
            continue;
        }
        *seen_state = true;
        let node = &dfa.states[state as usize];
        if node.is_accepting {
            return false;
        }
        stack.extend(node.transitions.values().copied());
    }
    true
}

fn trim_unweighted_dfa_productive(dfa: UnweightedDfa) -> UnweightedDfa {
    if dfa.states.is_empty() {
        return dfa;
    }
    let mut predecessors = vec![Vec::<u32>::new(); dfa.states.len()];
    let mut productive = vec![false; dfa.states.len()];
    let mut queue = VecDeque::<u32>::new();
    for (state_id, state) in dfa.states.iter().enumerate() {
        if state.is_accepting {
            productive[state_id] = true;
            queue.push_back(state_id as u32);
        }
        for &target in state.transitions.values() {
            if let Some(row) = predecessors.get_mut(target as usize) {
                row.push(state_id as u32);
            }
        }
    }
    while let Some(target) = queue.pop_front() {
        for &source in &predecessors[target as usize] {
            if !productive[source as usize] {
                productive[source as usize] = true;
                queue.push_back(source);
            }
        }
    }
    if !productive
        .get(dfa.start_state as usize)
        .copied()
        .unwrap_or(false)
    {
        return UnweightedDfa::new();
    }

    let mut remap = vec![u32::MAX; dfa.states.len()];
    let mut states = Vec::with_capacity(productive.iter().filter(|&&value| value).count());
    for (old, state) in dfa.states.iter().enumerate() {
        if productive[old] {
            remap[old] = states.len() as u32;
            states.push(crate::automata::unweighted_u32::dfa::DFAState {
                is_accepting: state.is_accepting,
                transitions: BTreeMap::new(),
            });
        }
    }
    for (old, state) in dfa.states.iter().enumerate() {
        if !productive[old] {
            continue;
        }
        let new = remap[old] as usize;
        for (&label, &target) in &state.transitions {
            let mapped = remap[target as usize];
            if mapped != u32::MAX {
                states[new].transitions.insert(label, mapped);
            }
        }
    }
    UnweightedDfa {
        states,
        start_state: remap[dfa.start_state as usize],
    }
}

fn unweighted_dfa_to_template_nwa(dfa: &UnweightedDfa) -> NWA {
    let states = dfa
        .states
        .iter()
        .map(|state| NWAState {
            final_weight: state.is_accepting.then(Weight::empty),
            transitions: state
                .transitions
                .iter()
                .map(|(&label, &target)| (label, vec![(target, Weight::empty())]))
                .collect(),
            epsilons: Vec::new(),
        })
        .collect();
    NWA::from_parts(states, vec![dfa.start_state])
}

#[derive(Debug, Clone)]
struct BoundaryTemplateDeltaEntry {
    global_terminal: u32,
    base_terminal: u32,
    delta_terminal: u32,
    old_template: UnweightedDfa,
}

#[derive(Debug, Clone)]
struct BoundaryTemplateDeltaPlan {
    original_num_terminals: u32,
    synthetic_num_terminals: u32,
    changed_terminals: BTreeSet<u32>,
    by_global_terminal: BTreeMap<u32, BoundaryTemplateDeltaEntry>,
}

fn prepare_boundary_template_delta_plan(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    candidate_terminals: &[u32],
    original_num_terminals: u32,
) -> BoundaryTemplateDeltaPlan {
    let mut entries = Vec::<(u32, UnweightedDfa)>::new();
    for &global_terminal in candidate_terminals {
        let component_index = composed_table
            .terminal_offsets
            .partition_point(|&offset| offset <= global_terminal)
            .saturating_sub(1);
        let Some(component) = components.get(component_index).copied() else {
            continue;
        };
        let terminal_offset = composed_table.terminal_offsets[component_index];
        let local_terminal = global_terminal.saturating_sub(terminal_offset);
        let Some(local_old) = component.composition_templates.by_terminal.get(&local_terminal) else {
            continue;
        };
        let Some(old_template) = transport_template_dfa(
            local_old.clone(),
            &composed_table.state_relations[component_index],
        ) else {
            continue;
        };
        entries.push((global_terminal, old_template));
    }
    entries.sort_unstable_by_key(|(terminal, _)| *terminal);
    entries.dedup_by_key(|(terminal, _)| *terminal);

    let mut by_global_terminal = BTreeMap::new();
    let mut next = original_num_terminals;
    for (global_terminal, old_template) in entries {
        let base_terminal = next;
        let delta_terminal = next + 1;
        next += 2;
        by_global_terminal.insert(
            global_terminal,
            BoundaryTemplateDeltaEntry {
                global_terminal,
                base_terminal,
                delta_terminal,
                old_template,
            },
        );
    }
    let changed_terminals = by_global_terminal.keys().copied().collect();
    BoundaryTemplateDeltaPlan {
        original_num_terminals,
        synthetic_num_terminals: next,
        changed_terminals,
        by_global_terminal,
    }
}

fn install_boundary_template_deltas(
    templates: &mut Templates,
    plan: &BoundaryTemplateDeltaPlan,
) -> Result<(), String> {
    for entry in plan.by_global_terminal.values() {
        let new_template = templates
            .by_terminal
            .get(&entry.global_terminal)
            .ok_or_else(|| format!("missing composed template for boundary terminal {}", entry.global_terminal))?
            .clone();
        let removed = unweighted_dfa_difference(&entry.old_template, &new_template);
        if !unweighted_dfa_language_is_empty(&removed) {
            return Err(format!(
                "component template for terminal {} is not a subset of the composed template",
                entry.global_terminal,
            ));
        }
        let delta = trim_unweighted_dfa_productive(unweighted_dfa_difference(
            &new_template,
            &entry.old_template,
        ));
        templates
            .by_terminal
            .insert(entry.base_terminal, entry.old_template.clone());
        templates.by_terminal_nwa.insert(
            entry.base_terminal,
            unweighted_dfa_to_template_nwa(&entry.old_template),
        );
        templates.by_terminal.insert(entry.delta_terminal, delta.clone());
        templates
            .by_terminal_nwa
            .insert(entry.delta_terminal, unweighted_dfa_to_template_nwa(&delta));
    }
    Ok(())
}

fn transport_template_dfa(
    mut dfa: UnweightedDfa,
    state_relation: &[Vec<u32>],
) -> Option<UnweightedDfa> {
    for state in &mut dfa.states {
        let old = std::mem::take(&mut state.transitions);
        let mut mapped = BTreeMap::new();
        for (label, target) in old {
            let label = remap_template_label(label, state_relation)?;
            if let Some(previous) = mapped.insert(label, target)
                && previous != target
            {
                return None;
            }
        }
        state.transitions = mapped;
    }
    Some(dfa)
}

fn transport_template_nwa(mut nwa: NWA, state_relation: &[Vec<u32>]) -> Option<NWA> {
    for state in nwa.states_mut() {
        let old = std::mem::take(&mut state.transitions);
        let mut mapped = BTreeMap::<i32, Vec<(u32, Weight)>>::new();
        for (label, targets) in old {
            let label = remap_template_label(label, state_relation)?;
            mapped.entry(label).or_default().extend(targets);
        }
        for targets in mapped.values_mut() {
            targets.sort_unstable_by_key(|(target, weight)| (*target, weight.ptr_key()));
            targets.dedup_by(|left, right| {
                left.0 == right.0 && left.1.ptr_key() == right.1.ptr_key()
            });
        }
        state.transitions = mapped;
    }
    Some(nwa)
}

fn profile_control_template_deltas(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    composed_templates: &Templates,
    control_changed_terminals: &[u32],
) {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_REDUNDANCY").is_none() {
        return;
    }

    let mut compared = 0usize;
    let mut old_not_subset = Vec::<u32>::new();
    let mut empty_delta = Vec::<u32>::new();
    let mut old_states = 0usize;
    let mut new_states = 0usize;
    let mut delta_states = 0usize;
    let mut old_transitions = 0usize;
    let mut new_transitions = 0usize;
    let mut delta_transitions = 0usize;
    let mut trimmed_delta_states = 0usize;
    let mut trimmed_delta_transitions = 0usize;
    let mut minimized_delta_states = 0usize;
    let mut minimized_delta_transitions = 0usize;
    let mut per_terminal = Vec::<(u32, usize, usize, usize, usize)>::new();

    for &global_terminal in control_changed_terminals {
        let component_index = composed_table
            .terminal_offsets
            .partition_point(|&offset| offset <= global_terminal)
            .saturating_sub(1);
        let Some(component) = components.get(component_index).copied() else {
            continue;
        };
        let terminal_offset = composed_table.terminal_offsets[component_index];
        let local_terminal = global_terminal.saturating_sub(terminal_offset);
        let Some(local_old) = component.composition_templates.by_terminal.get(&local_terminal) else {
            continue;
        };
        let Some(old) = transport_template_dfa(
            local_old.clone(),
            &composed_table.state_relations[component_index],
        ) else {
            continue;
        };
        let Some(new) = composed_templates.by_terminal.get(&global_terminal) else {
            continue;
        };

        compared += 1;
        let removed = unweighted_dfa_difference(&old, new);
        if !unweighted_dfa_language_is_empty(&removed) {
            old_not_subset.push(global_terminal);
        }
        let delta = unweighted_dfa_difference(new, &old);
        if unweighted_dfa_language_is_empty(&delta) {
            empty_delta.push(global_terminal);
        }
        let trimmed_delta = trim_unweighted_dfa_productive(delta.clone());
        let minimized_delta = minimize_unweighted_dfa(&trimmed_delta);
        let trimmed_transitions = trimmed_delta
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        let minimized_transitions = minimized_delta
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        trimmed_delta_states += trimmed_delta.states.len();
        trimmed_delta_transitions += trimmed_transitions;
        minimized_delta_states += minimized_delta.states.len();
        minimized_delta_transitions += minimized_transitions;
        per_terminal.push((
            global_terminal,
            old.states.len(),
            new.states.len(),
            trimmed_delta.states.len(),
            minimized_delta.states.len(),
        ));
        old_states += old.states.len();
        new_states += new.states.len();
        delta_states += delta.states.len();
        old_transitions += old.states.iter().map(|state| state.transitions.len()).sum::<usize>();
        new_transitions += new.states.iter().map(|state| state.transitions.len()).sum::<usize>();
        delta_transitions += delta.states.iter().map(|state| state.transitions.len()).sum::<usize>();
    }

    eprintln!(
        "[glrmask/profile][constraint_boundary_template_delta] changed={} compared={} old_not_subset={} empty_delta={} old_states={} new_states={} delta_states={} trimmed_delta_states={} minimized_delta_states={} old_transitions={} new_transitions={} delta_transitions={} trimmed_delta_transitions={} minimized_delta_transitions={} old_not_subset_ids={:?} empty_delta_ids={:?} per_terminal={:?}",
        control_changed_terminals.len(),
        compared,
        old_not_subset.len(),
        empty_delta.len(),
        old_states,
        new_states,
        delta_states,
        trimmed_delta_states,
        minimized_delta_states,
        old_transitions,
        new_transitions,
        delta_transitions,
        trimmed_delta_transitions,
        minimized_delta_transitions,
        old_not_subset,
        empty_delta,
        per_terminal,
    );
}

fn map_characterization_state(state: u32, state_relation: &[Vec<u32>]) -> Option<u32> {
    let mapped = state_relation.get(state as usize)?;
    (mapped.len() == 1).then_some(mapped[0])
}

fn map_stack_matcher(
    matcher: &StackMatcher,
    state_relation: &[Vec<u32>],
) -> Option<StackMatcher> {
    Some(match matcher {
        StackMatcher::Any => StackMatcher::Any,
        StackMatcher::State(state) => {
            StackMatcher::State(map_characterization_state(*state, state_relation)?)
        }
        StackMatcher::States(states) => {
            let mut mapped = states
                .iter()
                .map(|&state| map_characterization_state(state, state_relation))
                .collect::<Option<Vec<_>>>()?;
            mapped.sort_unstable();
            mapped.dedup();
            match mapped.as_slice() {
                [] => return None,
                [state] => StackMatcher::State(*state),
                _ => StackMatcher::States(mapped),
            }
        }
    })
}

fn map_stack_matchers(
    matchers: &[StackMatcher],
    state_relation: &[Vec<u32>],
) -> Option<Vec<StackMatcher>> {
    matchers
        .iter()
        .map(|matcher| map_stack_matcher(matcher, state_relation))
        .collect()
}

fn map_characterization_pushes(
    pushes: &[u32],
    state_relation: &[Vec<u32>],
) -> Option<Vec<u32>> {
    pushes
        .iter()
        .map(|&state| map_characterization_state(state, state_relation))
        .collect()
}

fn transport_terminal_characterization(
    characterization: &TerminalCharacterization,
    state_relation: &[Vec<u32>],
    nonterminal_offset: u32,
) -> Option<TerminalCharacterization> {
    Some(TerminalCharacterization {
        escapes: characterization
            .escapes
            .iter()
            .map(|escape| {
                Some(InitialEscape {
                    pop: map_stack_matchers(&escape.pop, state_relation)?,
                    pushes: map_characterization_pushes(&escape.pushes, state_relation)?,
                })
            })
            .collect::<Option<Vec<_>>>()?,
        reduces: characterization
            .reduces
            .iter()
            .map(|reduce| {
                Some(InitialReduce {
                    pop: map_stack_matchers(&reduce.pop, state_relation)?,
                    nonterminal: reduce.nonterminal + nonterminal_offset,
                })
            })
            .collect::<Option<Vec<_>>>()?,
        nt_escapes: characterization
            .nt_escapes
            .iter()
            .map(|escape| {
                Some(NtEscape {
                    source_nonterminal: escape.source_nonterminal + nonterminal_offset,
                    pop: map_stack_matchers(&escape.pop, state_relation)?,
                    pushes: map_characterization_pushes(&escape.pushes, state_relation)?,
                })
            })
            .collect::<Option<Vec<_>>>()?,
        nt_rereduces: characterization
            .nt_rereduces
            .iter()
            .map(|reduce| {
                Some(NtRereduce {
                    source_nonterminal: reduce.source_nonterminal + nonterminal_offset,
                    pop: map_stack_matchers(&reduce.pop, state_relation)?,
                    target_nonterminal: reduce.target_nonterminal + nonterminal_offset,
                })
            })
            .collect::<Option<Vec<_>>>()?,
        all_nts: characterization
            .all_nts
            .iter()
            .map(|nonterminal| nonterminal + nonterminal_offset)
            .collect(),
    })
}


fn action_reduced_nonterminals(action: &Action, output: &mut BTreeSet<u32>) {
    match action {
        Action::Reduce(nonterminal, _) => {
            output.insert(*nonterminal);
        }
        Action::Split { reduces, .. } => {
            output.extend(reduces.iter().map(|(nonterminal, _)| *nonterminal));
        }
        _ => {}
    }
}

/// Conservative exact invalidation for rebased terminal characterizations.
///
/// A terminal characterization depends on its action column, on every goto
/// predecessor of a state where that terminal acts, and on goto entries used
/// by its nonconsuming reductions. Table composition preserves component rows
/// under the state/nonterminal rebasing relation, except where linker/control
/// construction adds one of those contexts. Mark every terminal that can
/// observe such an addition; all others may reuse the rebased component
/// characterization exactly.
fn characterization_context_fallbacks(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    nonterminal_offsets: &[u32],
    active: &[bool],
) -> Vec<bool> {
    let table = &composed_table.table;
    let mut composed_predecessors = vec![Vec::<(u32, u32, bool)>::new(); table.num_states as usize];
    for (revealed_state, row) in table.goto.iter().enumerate() {
        for (&nonterminal, &(target, replace)) in row {
            if let Some(predecessors) = composed_predecessors.get_mut(target as usize) {
                predecessors.push((revealed_state as u32, nonterminal, replace));
            }
        }
    }
    for predecessors in &mut composed_predecessors {
        predecessors.sort_unstable();
        predecessors.dedup();
    }

    let mut changed_targets = BTreeSet::<u32>::new();
    let mut changed_nonterminals = BTreeSet::<u32>::new();
    let mut fallback = vec![false; active.len()];

    for (component_index, component) in components.iter().enumerate() {
        let relation = &composed_table.state_relations[component_index];
        let terminal_offset = composed_table.terminal_offsets[component_index];
        let nonterminal_offset = nonterminal_offsets[component_index];
        let singleton_states = relation
            .iter()
            .map(|states| (states.len() == 1).then_some(states[0]))
            .collect::<Option<Vec<_>>>();
        let Some(singleton_states) = singleton_states else {
            for local_terminal in 0..component.table.num_terminals {
                let global = terminal_offset + local_terminal;
                if active.get(global as usize).copied().unwrap_or(false) {
                    fallback[global as usize] = true;
                }
            }
            continue;
        };

        let mut expected_predecessors =
            vec![Vec::<(u32, u32, bool)>::new(); component.table.num_states as usize];
        for (revealed_state, row) in component.table.goto.iter().enumerate() {
            for (&local_nonterminal, &(local_target, replace)) in row {
                let Some(&global_revealed) = singleton_states.get(revealed_state) else {
                    continue;
                };
                let Some(&global_target) = singleton_states.get(local_target as usize) else {
                    continue;
                };
                expected_predecessors[local_target as usize].push((
                    global_revealed,
                    nonterminal_offset + local_nonterminal,
                    replace,
                ));
                let expected = (global_target, replace);
                let global_nonterminal = nonterminal_offset + local_nonterminal;
                let actual = table
                    .goto
                    .get(global_revealed as usize)
                    .and_then(|global_row| global_row.get(&global_nonterminal))
                    .copied();
                if actual != Some(expected) {
                    changed_nonterminals.insert(global_nonterminal);
                }
            }
        }
        for predecessors in &mut expected_predecessors {
            predecessors.sort_unstable();
            predecessors.dedup();
        }
        for (local_target, expected) in expected_predecessors.iter().enumerate() {
            let global_target = singleton_states[local_target];
            let actual = composed_predecessors
                .get(global_target as usize)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            if actual != expected.as_slice() {
                changed_targets.insert(global_target);
            }
        }

        // Detect additional composed goto entries absent from the component row.
        for (local_revealed, &global_revealed) in singleton_states.iter().enumerate() {
            let local_row = component.table.goto.get(local_revealed);
            if let Some(global_row) = table.goto.get(global_revealed as usize) {
                for &global_nonterminal in global_row.keys() {
                    let local_nonterminal = global_nonterminal.checked_sub(nonterminal_offset);
                    let represented = local_nonterminal.is_some_and(|local_nonterminal| {
                        local_nonterminal < component.table.nonterminal_display_names.len() as u32
                            && local_row
                                .is_some_and(|row| row.contains_key(&local_nonterminal))
                    });
                    if !represented {
                        changed_nonterminals.insert(global_nonterminal);
                    }
                }
            }
        }
    }

    for &state in &changed_targets {
        if let Some(row) = table.action.get(state as usize) {
            for (terminal, _) in row.iter() {
                if active.get(terminal as usize).copied().unwrap_or(false) {
                    fallback[terminal as usize] = true;
                }
            }
        }
    }
    for row in &table.action {
        for (terminal, action) in row.iter() {
            if !active.get(terminal as usize).copied().unwrap_or(false) {
                continue;
            }
            let mut reduced = BTreeSet::new();
            action_reduced_nonterminals(action, &mut reduced);
            if reduced.iter().any(|nonterminal| changed_nonterminals.contains(nonterminal)) {
                fallback[terminal as usize] = true;
            }
        }
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_characterization_context] changed_targets={} changed_nonterminals={} fallback_terminals={}",
            changed_targets.len(),
            changed_nonterminals.len(),
            fallback.iter().filter(|&&selected| selected).count(),
        );
    }
    fallback
}

fn build_transported_composition_templates(
    composed_table: &ComposedTable,
    analyzed: &AnalyzedGrammar,
    active: &[bool],
    control_changed_terminals: &[u32],
    components: &[&Constraint],
) -> (Templates, f64) {
    let started_at = Instant::now();
    let changed = control_changed_terminals
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut nonterminal_offsets = Vec::with_capacity(components.len());
    let mut next_nonterminal = 0u32;
    for component in components {
        nonterminal_offsets.push(next_nonterminal);
        next_nonterminal += component.table.nonterminal_display_names.len() as u32;
    }

    struct LocalTemplatePlan {
        terminal_offset: u32,
        state_relation_index: usize,
        nonterminal_offset: u32,
        selected_terminals: Vec<u32>,
        owned_characterizations: Option<BTreeMap<u32, TerminalCharacterization>>,
        owned_templates: Option<Templates>,
        cached: bool,
        wall_ms: f64,
    }

    let local_started_all = Instant::now();
    let mut local_results = components
        .iter()
        .enumerate()
        .filter_map(|(component_index, component)| {
            let terminal_offset = composed_table.terminal_offsets[component_index];
            let mut selected = vec![false; component.table.num_terminals as usize];
            for (local_terminal, slot) in selected.iter_mut().enumerate() {
                let global = terminal_offset + local_terminal as u32;
                *slot = active.get(global as usize).copied().unwrap_or(false)
                    && !changed.contains(&global);
            }
            if !selected.iter().any(|&value| value) {
                return None;
            }
            let local_started_at = Instant::now();
            let selected_terminals = selected
                .iter()
                .enumerate()
                .filter_map(|(terminal, &selected)| selected.then_some(terminal as u32))
                .collect::<Vec<_>>();
            let cached_characterizations =
                !component.composition_terminal_characterizations.is_empty()
                    && selected_terminals.iter().all(|terminal| {
                        component
                            .composition_terminal_characterizations
                            .contains_key(terminal)
                    });
            let cached_templates = selected_terminals.iter().all(|terminal| {
                component.composition_templates.by_terminal.contains_key(terminal)
                    && component
                        .composition_templates
                        .by_terminal_nwa
                        .contains_key(terminal)
            });
            let cached = cached_characterizations && cached_templates;

            if cached {
                if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_CACHED_CHARACTERIZATIONS")
                    .is_some()
                    || std::env::var_os("GLRMASK_VALIDATE_COMPOSE_CACHED_TEMPLATES").is_some()
                {
                    let augmented_start = component.table.rules.first()?.lhs;
                    let local_analyzed = AnalyzedGrammar::from_composed_rules(
                        component.table.rules.clone(),
                        component.table.num_terminals,
                        component.terminal_display_names.clone(),
                        component.table.nonterminal_display_names.clone(),
                        augmented_start,
                    );
                    let reference_characterizations = characterize_selected_terminals_profiled(
                        &component.table,
                        &local_analyzed,
                        &selected,
                    )
                    .0;
                    if std::env::var_os(
                        "GLRMASK_VALIDATE_COMPOSE_CACHED_CHARACTERIZATIONS",
                    )
                    .is_some()
                    {
                        for terminal in &selected_terminals {
                            assert_eq!(
                                component.composition_terminal_characterizations.get(terminal),
                                reference_characterizations.get(terminal),
                            );
                        }
                    }
                    if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_CACHED_TEMPLATES")
                        .is_some()
                    {
                        let reference =
                            Templates::from_characterizations_profiled(&reference_characterizations).0;
                        for terminal in &selected_terminals {
                            assert_eq!(
                                component.composition_templates.by_terminal.get(terminal),
                                reference.by_terminal.get(terminal),
                            );
                            let candidate = &component.composition_templates.by_terminal_nwa[terminal];
                            let expected = &reference.by_terminal_nwa[terminal];
                            assert_eq!(candidate.states(), expected.states());
                            assert_eq!(candidate.start_states(), expected.start_states());
                        }
                    }
                }
                return Some(LocalTemplatePlan {
                    terminal_offset,
                    state_relation_index: component_index,
                    nonterminal_offset: nonterminal_offsets[component_index],
                    selected_terminals,
                    owned_characterizations: None,
                    owned_templates: None,
                    cached: true,
                    wall_ms: local_started_at.elapsed().as_secs_f64() * 1000.0,
                });
            }

            let characterizations = if cached_characterizations {
                selected_terminals
                    .iter()
                    .map(|terminal| {
                        (
                            *terminal,
                            component.composition_terminal_characterizations[terminal].clone(),
                        )
                    })
                    .collect::<BTreeMap<_, _>>()
            } else {
                let augmented_start = component.table.rules.first()?.lhs;
                let local_analyzed = AnalyzedGrammar::from_composed_rules(
                    component.table.rules.clone(),
                    component.table.num_terminals,
                    component.terminal_display_names.clone(),
                    component.table.nonterminal_display_names.clone(),
                    augmented_start,
                );
                characterize_selected_terminals_profiled(
                    &component.table,
                    &local_analyzed,
                    &selected,
                )
                .0
            };
            let templates = if cached_templates {
                Templates {
                    by_terminal: selected_terminals
                        .iter()
                        .map(|terminal| {
                            (
                                *terminal,
                                component.composition_templates.by_terminal[terminal].clone(),
                            )
                        })
                        .collect(),
                    by_terminal_nwa: selected_terminals
                        .iter()
                        .map(|terminal| {
                            (
                                *terminal,
                                component.composition_templates.by_terminal_nwa[terminal]
                                    .clone(),
                            )
                        })
                        .collect(),
                }
            } else {
                Templates::from_characterizations_profiled(&characterizations).0
            };
            Some(LocalTemplatePlan {
                terminal_offset,
                state_relation_index: component_index,
                nonterminal_offset: nonterminal_offsets[component_index],
                selected_terminals,
                owned_characterizations: Some(characterizations),
                owned_templates: Some(templates),
                cached: false,
                wall_ms: local_started_at.elapsed().as_secs_f64() * 1000.0,
            })
        })
        .collect::<Vec<_>>();
    let local_collect_ms = local_started_all.elapsed().as_secs_f64() * 1000.0;

    // Table composition preserves each component's parser rows under the exact
    // state/nonterminal rebasing relation. Control-rewritten columns are known
    // fallbacks immediately. Context certification, cached-template transport,
    // and compilation of those known fallbacks are otherwise independent, so
    // run all three lanes together. If context certification discovers any
    // additional fallback terminals, discard their speculative transports and
    // compile only that incremental set afterward.
    let mut known_fallback_selected = vec![false; active.len()];
    for &terminal in control_changed_terminals {
        if active.get(terminal as usize).copied().unwrap_or(false) {
            known_fallback_selected[terminal as usize] = true;
        }
    }
    for local in &local_results {
        if local.owned_characterizations.is_none()
            && local.selected_terminals.iter().any(|terminal| {
                !components[local.state_relation_index]
                    .composition_terminal_characterizations
                    .contains_key(terminal)
            })
        {
            for &local_terminal in &local.selected_terminals {
                known_fallback_selected[(local.terminal_offset + local_terminal) as usize] = true;
            }
        }
    }

    struct TemplateTransportResult {
        templates: Templates,
        late_fallback_selected: Vec<bool>,
        transported_count: usize,
        cached_components: usize,
        local_wall_ms: f64,
        dfa_transport_ms: f64,
        nwa_transport_ms: f64,
        template_transport_ms: f64,
    }
    struct FreshFallbackResult {
        characterizations: BTreeMap<u32, TerminalCharacterization>,
        templates: Templates,
        characterize_ms: f64,
        compile_ms: f64,
    }

    let parallel_core_started_at = Instant::now();
    let ((context_fallback_selected, context_ms), (transported, mut known_fallback)) =
        rayon::join(
            || {
                let context_started_at = Instant::now();
                let fallback = characterization_context_fallbacks(
                    composed_table,
                    components,
                    &nonterminal_offsets,
                    active,
                );
                (
                    fallback,
                    context_started_at.elapsed().as_secs_f64() * 1000.0,
                )
            },
            || {
                rayon::join(
                    || {
                        let template_transport_started_at = Instant::now();
                        let mut templates = Templates::default();
                        let mut transported_count = 0usize;
                        let mut cached_components = 0usize;
                        let mut local_wall_ms = 0.0f64;
                        let mut dfa_transport_ms = 0.0f64;
                        let mut nwa_transport_ms = 0.0f64;
                        let mut late_fallback_selected = vec![false; active.len()];
                        for local in &mut local_results {
                            cached_components += usize::from(local.cached);
                            local_wall_ms = local_wall_ms.max(local.wall_ms);
                            let relation =
                                &composed_table.state_relations[local.state_relation_index];
                            let component = components[local.state_relation_index];
                            for &local_terminal in &local.selected_terminals {
                                let global_terminal = local.terminal_offset + local_terminal;
                                if known_fallback_selected[global_terminal as usize] {
                                    continue;
                                }
                                let dfa = if let Some(local_templates) =
                                    local.owned_templates.as_mut()
                                {
                                    local_templates.by_terminal.remove(&local_terminal)
                                } else {
                                    component
                                        .composition_templates
                                        .by_terminal
                                        .get(&local_terminal)
                                        .cloned()
                                };
                                let Some(dfa) = dfa else {
                                    late_fallback_selected[global_terminal as usize] = true;
                                    continue;
                                };
                                let nwa = if let Some(local_templates) =
                                    local.owned_templates.as_mut()
                                {
                                    local_templates.by_terminal_nwa.remove(&local_terminal)
                                } else {
                                    component
                                        .composition_templates
                                        .by_terminal_nwa
                                        .get(&local_terminal)
                                        .cloned()
                                };
                                let Some(nwa) = nwa else {
                                    late_fallback_selected[global_terminal as usize] = true;
                                    continue;
                                };
                                let dfa_started_at = Instant::now();
                                let transported_dfa = transport_template_dfa(dfa, relation);
                                dfa_transport_ms +=
                                    dfa_started_at.elapsed().as_secs_f64() * 1000.0;
                                let nwa_started_at = Instant::now();
                                let transported_nwa = transport_template_nwa(nwa, relation);
                                nwa_transport_ms +=
                                    nwa_started_at.elapsed().as_secs_f64() * 1000.0;
                                let (Some(dfa), Some(nwa)) =
                                    (transported_dfa, transported_nwa)
                                else {
                                    late_fallback_selected[global_terminal as usize] = true;
                                    continue;
                                };
                                templates.by_terminal.insert(global_terminal, dfa);
                                templates.by_terminal_nwa.insert(global_terminal, nwa);
                                transported_count += 1;
                            }
                        }
                        TemplateTransportResult {
                            templates,
                            late_fallback_selected,
                            transported_count,
                            cached_components,
                            local_wall_ms,
                            dfa_transport_ms,
                            nwa_transport_ms,
                            template_transport_ms: template_transport_started_at
                                .elapsed()
                                .as_secs_f64()
                                * 1000.0,
                        }
                    },
                    || {
                        let fresh_started_at = Instant::now();
                        let (characterizations, _) = characterize_selected_terminals_profiled(
                            &composed_table.table,
                            analyzed,
                            &known_fallback_selected,
                        );
                        let characterize_ms =
                            fresh_started_at.elapsed().as_secs_f64() * 1000.0;
                        let compile_started_at = Instant::now();
                        let (templates, _) =
                            Templates::from_characterizations_profiled(&characterizations);
                        FreshFallbackResult {
                            characterizations,
                            templates,
                            characterize_ms,
                            compile_ms: compile_started_at.elapsed().as_secs_f64() * 1000.0,
                        }
                    },
                )
            },
        );
    let parallel_core_ms = parallel_core_started_at.elapsed().as_secs_f64() * 1000.0;

    let TemplateTransportResult {
        mut templates,
        late_fallback_selected,
        transported_count,
        cached_components,
        local_wall_ms,
        dfa_transport_ms,
        nwa_transport_ms,
        template_transport_ms,
    } = transported;

    let mut fallback_selected = known_fallback_selected;
    let mut additional_fallback_selected = vec![false; active.len()];
    for terminal in 0..active.len() {
        if (context_fallback_selected[terminal] || late_fallback_selected[terminal])
            && !fallback_selected[terminal]
        {
            additional_fallback_selected[terminal] = true;
            fallback_selected[terminal] = true;
            templates.by_terminal.remove(&(terminal as u32));
            templates.by_terminal_nwa.remove(&(terminal as u32));
        }
    }

    let additional_fresh_started_at = Instant::now();
    let additional_characterizations = if additional_fallback_selected
        .iter()
        .any(|&selected| selected)
    {
        characterize_selected_terminals_profiled(
            &composed_table.table,
            analyzed,
            &additional_fallback_selected,
        )
        .0
    } else {
        BTreeMap::new()
    };
    let additional_fresh_ms = additional_fresh_started_at.elapsed().as_secs_f64() * 1000.0;
    let additional_compile_started_at = Instant::now();
    let additional_templates = if additional_characterizations.is_empty() {
        Templates::default()
    } else {
        Templates::from_characterizations_profiled(&additional_characterizations).0
    };
    let additional_compile_ms = additional_compile_started_at.elapsed().as_secs_f64() * 1000.0;

    known_fallback
        .characterizations
        .extend(additional_characterizations);
    known_fallback
        .templates
        .by_terminal
        .extend(additional_templates.by_terminal);
    known_fallback
        .templates
        .by_terminal_nwa
        .extend(additional_templates.by_terminal_nwa);
    let fallback_characterizations = known_fallback.characterizations;
    let fallback_count = fallback_characterizations.len();
    templates
        .by_terminal
        .extend(known_fallback.templates.by_terminal);
    templates
        .by_terminal_nwa
        .extend(known_fallback.templates.by_terminal_nwa);
    let fresh_ms = known_fallback.characterize_ms + additional_fresh_ms;
    let fallback_ms = known_fallback.compile_ms + additional_compile_ms;
    let composed_characterize_ms = context_ms + fresh_ms;
    let characterization_compare_ms = 0.0f64;
    let mut characterization_transport_ms = 0.0f64;

    if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_CHARACTERIZATION_TRANSPORT").is_some() {
        let validation_transport_started_at = Instant::now();
        let mut candidate = fallback_characterizations.clone();
        for local in &local_results {
            let relation = &composed_table.state_relations[local.state_relation_index];
            let component = components[local.state_relation_index];
            for &local_terminal in &local.selected_terminals {
                let global_terminal = local.terminal_offset + local_terminal;
                if fallback_selected[global_terminal as usize] {
                    continue;
                }
                let local_characterization = local
                    .owned_characterizations
                    .as_ref()
                    .and_then(|characterizations| characterizations.get(&local_terminal))
                    .unwrap_or_else(|| {
                        &component.composition_terminal_characterizations[&local_terminal]
                    });
                let transported = transport_terminal_characterization(
                    local_characterization,
                    relation,
                    local.nonterminal_offset,
                )
                .expect("certified singleton relation must transport characterization");
                candidate.insert(global_terminal, transported);
            }
        }
        characterization_transport_ms =
            validation_transport_started_at.elapsed().as_secs_f64() * 1000.0;
        let reference = characterize_selected_terminals_profiled(
            &composed_table.table,
            analyzed,
            active,
        )
        .0;
        if candidate != reference {
            for (&terminal, expected) in &reference {
                let actual = candidate.get(&terminal);
                if actual != Some(expected) {
                    eprintln!(
                        "[glrmask/validate][compose_characterization_transport_mismatch] terminal={} fallback={} actual={:?} expected={:?}",
                        terminal,
                        fallback_selected.get(terminal as usize).copied().unwrap_or(false),
                        actual,
                        expected,
                    );
                    break;
                }
            }
        }
        assert_eq!(candidate, reference);
        eprintln!(
            "[glrmask/validate][compose_characterization_transport] transported={} fresh={} exact=true",
            candidate.len().saturating_sub(fallback_count),
            fallback_count,
        );
    }

    if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_TEMPLATE_TRANSPORT").is_some() {
        let reference_characterizations = characterize_selected_terminals_profiled(
            &composed_table.table,
            analyzed,
            active,
        )
        .0;
        let (reference, _) = Templates::from_characterizations_profiled(&reference_characterizations);
        assert_eq!(templates.by_terminal, reference.by_terminal);
        assert_eq!(
            templates.by_terminal_nwa.keys().collect::<Vec<_>>(),
            reference.by_terminal_nwa.keys().collect::<Vec<_>>(),
        );
        for (terminal, candidate) in &templates.by_terminal_nwa {
            let expected = &reference.by_terminal_nwa[terminal];
            assert_eq!(candidate.states(), expected.states());
            assert_eq!(candidate.start_states(), expected.start_states());
        }
        eprintln!(
            "[glrmask/validate][compose_template_transport] transported={} fallback={} exact=true",
            transported_count,
            fallback_count,
        );
    }
    let wall_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_template_transport] transported={} fallback={} cached_components={} local_collect_ms={local_collect_ms:.3} local_ms={local_wall_ms:.3} context_ms={context_ms:.3} parallel_core_ms={parallel_core_ms:.3} characterization_transport_ms={characterization_transport_ms:.3} fresh_ms={fresh_ms:.3} composed_characterize_ms={composed_characterize_ms:.3} characterization_compare_ms={characterization_compare_ms:.3} dfa_transport_ms={dfa_transport_ms:.3} nwa_transport_ms={nwa_transport_ms:.3} template_transport_ms={template_transport_ms:.3} fallback_ms={fallback_ms:.3} total_ms={wall_ms:.3}",
            transported_count,
            fallback_count,
            cached_components,
        );
    }
    (templates, wall_ms)
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
        if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_REDUNDANCY").is_some() {
            let seed_ids = seed_terminals
                .iter()
                .enumerate()
                .filter_map(|(terminal, &selected)| selected.then_some(terminal as u32))
                .collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][constraint_boundary_seed_ids] ids={seed_ids:?}",
            );
        }
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
    let lazy_direct_seed_relations = std::env::var_os(
        "GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DIRECT_PARSER",
    )
    .is_some()
    .then(|| seed_relations.clone());
    profile_boundary_component_locality(
        &boundary_paths,
        &ignore_terminals.all,
        control_changed_terminals,
    );
    profile_boundary_entry_return_shape(
        &boundary_paths,
        tokenizer_state_offsets,
        merged_reset_state,
    );

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

    let seed_delta_terminals = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_SEED_DELTAS")
        .is_some()
        .then(|| {
            seed_terminals
                .iter()
                .enumerate()
                .filter_map(|(terminal, &selected)| selected.then_some(terminal as u32))
                .collect::<Vec<_>>()
        });
    let boundary_delta_plan = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TEMPLATE_DELTA")
        .is_some()
        .then(|| {
            prepare_boundary_template_delta_plan(
                composed_table,
                components,
                seed_delta_terminals
                    .as_deref()
                    .unwrap_or(control_changed_terminals),
                analyzed.num_terminals,
            )
        });

    let template_terminal_started_at = Instant::now();
    let ((templates, templates_ms), terminal_dwa) =
        rayon::join(
            || {
                let (mut templates, templates_ms) = eager_templates.unwrap_or_else(|| {
                    if std::env::var_os("GLRMASK_COMPOSE_DISABLE_TEMPLATE_TRANSPORT").is_some() {
                        build_composition_templates(
                            &composed_table.table,
                            &analyzed,
                            &active_terminals,
                        )
                    } else {
                        build_transported_composition_templates(
                            composed_table,
                            &analyzed,
                            &active_terminals,
                            control_changed_terminals,
                            components,
                        )
                    }
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
                        boundary_delta_plan.as_ref(),
                    )
                };
                (result, started_at.elapsed().as_secs_f64() * 1000.0)
            },
        );
    let template_terminal_wall_ms =
        template_terminal_started_at.elapsed().as_secs_f64() * 1000.0;
    let (mut templates, templates_ms) = (templates, templates_ms);
    if let Some(plan) = boundary_delta_plan.as_ref() {
        install_boundary_template_deltas(&mut templates, plan)?;
    }
    let (terminal_dwa, terminal_ms) = terminal_dwa;
    let terminal_dwa = terminal_dwa?;
    let seed_delta_profile_terminals = std::env::var_os(
        "GLRMASK_EXPERIMENT_BOUNDARY_SEED_DELTAS",
    )
    .is_some()
    .then(|| {
        seed_terminals
            .iter()
            .enumerate()
            .filter_map(|(terminal, &selected)| selected.then_some(terminal as u32))
            .collect::<Vec<_>>()
    });
    profile_control_template_deltas(
        composed_table,
        components,
        &templates,
        seed_delta_profile_terminals
            .as_deref()
            .unwrap_or(control_changed_terminals),
    );
    profile_finite_boundary_word_domains(
        &composed_table.table,
        &templates,
        &boundary_paths,
        &ignore_terminals.global,
    );
    profile_boundary_shared_domain_dp(
        &composed_table.table,
        &templates,
        &boundary_paths,
        &ignore_terminals.global,
    );
    profile_boundary_suffix_domain_dp(
        &composed_table.table,
        &templates,
        &boundary_paths,
        &ignore_terminals.global,
    );
    profile_boundary_suffix_nwa_dp(
        &composed_table.table,
        &templates,
        &boundary_paths,
        &ignore_terminals.global,
    );
    profile_token_sharded_boundary_parser(
        composed_table,
        &analyzed,
        &templates,
        &boundary_paths,
        component_state_map,
        merged_tokenizer_state_count,
        vocab,
        &ignore_terminals.global,
    )?;
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
    let lazy_direct_context = if boundary_special_token_terminals.is_empty()
        && composed_table.control_terminals.is_empty()
    {
        lazy_direct_seed_relations
            .as_ref()
            .map(|seed_relations| (component_state_map, &id_map, seed_relations))
    } else {
        None
    };
    let lazy_direct_parser = profile_boundary_lazy_domain_dp(
        &composed_table.table,
        &templates,
        &boundary_paths,
        &ignore_terminals.global,
        boundary_delta_plan.as_ref(),
        lazy_direct_context,
    );
    let mut claimed_tokens_by_tsid = BTreeMap::<u32, BTreeSet<u32>>::new();
    for witness in &boundary_paths.witnesses {
        let Some(internal_token) = id_map.internal_token_for_original(witness.token_id) else {
            continue;
        };
        for &raw_state in &witness.start_states {
            let Some(&tsid) = component_state_map.original_to_internal.get(raw_state as usize) else {
                continue;
            };
            if tsid != u32::MAX {
                claimed_tokens_by_tsid.entry(tsid).or_default().insert(internal_token);
            }
        }
    }
    let claimed_weight = Weight::from_per_tsid_token_sets(
        claimed_tokens_by_tsid.into_iter().map(|(tsid, tokens)| {
            (tsid, tokens.into_iter().collect::<RangeSetBlaze<_>>())
        }),
    );
    let mut parser_analyzed = analyzed.clone();
    if let Some(plan) = boundary_delta_plan.as_ref() {
        debug_assert_eq!(plan.original_num_terminals, analyzed.num_terminals);
        parser_analyzed.num_terminals = plan.synthetic_num_terminals;
        parser_analyzed
            .terminal_display_names
            .resize(plan.synthetic_num_terminals as usize, "<boundary-delta>".to_string());
    }
    if std::env::var_os("GLRMASK_DEBUG_BOUNDARY_LAZY_LANE_17122").is_some() {
        for (original_token, probe) in [
            (534u32, vec![8810i32, DEFAULT_LABEL, 8723, 1]),
            (17122u32, vec![540i32, 444, 295, 26, 1]),
        ] {
            let Some(internal_token) = id_map.internal_token_for_original(original_token) else {
                continue;
            };
            let lane = Weight::from_token_set_for_tsid(
                0,
                RangeSetBlaze::from_iter([internal_token]),
            );
            let mut isolated = terminal_automaton.clone();
            match &mut isolated {
                TerminalAutomaton::Dwa(dwa) => {
                    for state in dwa.states_mut() {
                        if let Some(final_weight) = state.final_weight.as_mut() {
                            *final_weight = final_weight.intersection(&lane);
                        }
                        state.transitions.retain(|_, (_, weight)| {
                            *weight = weight.intersection(&lane);
                            !weight.is_empty()
                        });
                    }
                }
                TerminalAutomaton::TokenDeterministicNwa(nwa)
                | TerminalAutomaton::EpsilonNwa(nwa) => {
                    for state in nwa.states_mut() {
                        if let Some(final_weight) = state.final_weight.as_mut() {
                            *final_weight = final_weight.intersection(&lane);
                        }
                        state.transitions.retain(|_, branches| {
                            branches.retain_mut(|(_, weight)| {
                                *weight = weight.intersection(&lane);
                                !weight.is_empty()
                            });
                            !branches.is_empty()
                        });
                        state.epsilons.retain_mut(|(_, weight)| {
                            *weight = weight.intersection(&lane);
                            !weight.is_empty()
                        });
                    }
                }
            }
            let isolated_parser = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &composed_table.table,
                &parser_analyzed,
                &isolated,
                &templates,
                vocab,
                &id_map,
                false,
            );
            eprintln!(
                "[glrmask/debug][lazy_lane_isolated_generic] original_token={original_token} states={} transitions={} probe={probe:?} probe_accept={} weight={:?}",
                isolated_parser.states().len(),
                isolated_parser.num_transitions(),
                !isolated_parser.eval_word(&probe).is_empty(),
                isolated_parser.eval_word(&probe),
            );
        }
    }
    let use_lazy_direct_parser = std::env::var_os(
        "GLRMASK_EXPERIMENT_USE_LAZY_DIRECT_BOUNDARY_PARSER",
    )
    .is_some();
    let validate_lazy_direct_parser = std::env::var_os(
        "GLRMASK_VALIDATE_BOUNDARY_LAZY_DIRECT_PARSER",
    )
    .is_some();
    let parser_commit_started_at = Instant::now();
    let (rebuilt_commit_templates, commit_ms, generic_parser_dwa, parser_ms) =
        if use_lazy_direct_parser && !validate_lazy_direct_parser {
            let (rebuilt_commit_templates, commit_ms) = build_composition_commit_templates(
                &templates,
                analyzed.num_terminals as usize,
                Some(&commit_rebuild),
            );
            (rebuilt_commit_templates, commit_ms, None, 0.0)
        } else {
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
                        &parser_analyzed,
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
            (rebuilt_commit_templates, commit_ms, Some(parser_dwa), parser_ms)
        };
    let parser_commit_wall_ms = parser_commit_started_at.elapsed().as_secs_f64() * 1000.0;
    if let (Some(lazy_direct_parser), Some(parser_dwa)) =
        (lazy_direct_parser.as_ref(), generic_parser_dwa.as_ref())
    {
        let compare_started_at = Instant::now();
        let difference = find_difference(lazy_direct_parser, &parser_dwa)
            .expect("boundary parser DWAs should be acyclic for direct-root validation");
        let difference_detail = difference.as_ref().map(|witness| {
            let candidate_weight = lazy_direct_parser.eval_word(witness);
            let reference_weight = parser_dwa.eval_word(witness);
            let missing = reference_weight.difference(&candidate_weight);
            let extra = candidate_weight.difference(&reference_weight);
            let first_lane = |weight: &Weight| {
                (0..id_map.num_tsids()).find_map(|tsid| {
                    let tokens = weight.tokens_for_tsid(tsid);
                    let token = tokens.iter().next()?;
                    let originals = id_map
                        .vocab_tokens
                        .internal_to_originals
                        .get(token as usize)
                        .cloned()
                        .unwrap_or_default();
                    Some((tsid, token, originals))
                })
            };
            (first_lane(&missing), first_lane(&extra))
        });
        eprintln!(
            "[glrmask/validate][constraint_boundary_lazy_direct_parser] candidate_states={} candidate_transitions={} reference_states={} reference_transitions={} difference={difference:?} difference_detail={difference_detail:?} compare_ms={:.3}",
            lazy_direct_parser.states().len(),
            lazy_direct_parser.num_transitions(),
            parser_dwa.states().len(),
            parser_dwa.num_transitions(),
            compare_started_at.elapsed().as_secs_f64() * 1000.0,
        );
        if validate_lazy_direct_parser {
            assert!(
                difference.is_none(),
                "lazy direct boundary parser differs from generic boundary parser on labels {difference:?}",
            );
        }
    }
    let parser_dwa = if use_lazy_direct_parser {
        lazy_direct_parser.expect(
            "using lazy direct boundary parser requires GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DIRECT_PARSER",
        )
    } else {
        generic_parser_dwa.expect("generic boundary parser must be built when lazy direct is disabled")
    };
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
        claimed_weight,
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
        composition_terminal_characterizations: BTreeMap::new(),
        composition_templates: Templates::default(),
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
        prebuilt_parser_weight_token_sets: None,
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
    let boundary_repair = if std::env::var_os("GLRMASK_EXPERIMENT_SKIP_BOUNDARY_UNION").is_some() {
        None
    } else {
        boundary_repair
    };
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
        std::env::var_os("GLRMASK_EXPERIMENT_FORCE_EXPLICIT_LINKER").is_none()
            && global_ignores
            && components_have_no_explicit_controls(&parent, children)
            && components_have_no_compiled_eof_stack_rewrites(&parent, children)
            && legacy_splice_has_only_byte_terminal_continuations(&parent, children);
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_linker_path] owned_parent=true legacy_splice={use_legacy_splice} global_ignores={global_ignores}",
        );
    }
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
    let state_map_cell = OnceLock::<Result<ManyToOneIdMap, String>>::new();
    let transport_top_accept_directly =
        std::env::var_os("GLRMASK_COMPOSE_LEGACY_TOP_ACCEPT_BRANCH").is_none();

    struct PreparedComponentBase {
        component_id_map: InternalIdMap,
        component_maps: Vec<DirectComponentCoordinateMaps>,
        unmapped: Vec<UnmappedComponentParserArtifact>,
        coordinate_ms: f64,
        parser_extract_ms: f64,
    }

    let component_terminal_offsets = composed_table.terminal_offsets.clone();
    let prepare_component_base = || -> Result<PreparedComponentBase, String> {
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
                    &component_terminal_offsets,
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
        Ok(PreparedComponentBase {
            component_id_map: InternalIdMap {
                tokenizer_states: state_coordinates.tokenizer_states,
                vocab_tokens,
                deferred_vocab_singleton_original_ids: None,
            },
            component_maps,
            unmapped: unmapped_result?,
            coordinate_ms: component_state_ms + token_coordinate_ms,
            parser_extract_ms,
        })
    };

    // Component parser extraction depends only on immutable component artifacts
    // and the already-linked state relations. It does not depend on either
    // control elimination or boundary-token discovery, so start it in the same
    // Rayon phase. This hides the ~50 ms parser-artifact extraction lane behind
    // lexical boundary discovery instead of serializing the two.
    let preparation_started_at = Instant::now();
    let control_overlap_started_at = Instant::now();
    let ((control_elimination_result, lexical_result), component_base_result) = rayon::join(
        || {
            let table = &mut composed_table.table;
            let controls = &mut composed_table.control_terminals;
            if std::env::var_os("GLRMASK_COMPOSE_SERIAL_CONTROL_LEXICAL").is_some() {
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
        },
        prepare_component_base,
    );
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
    let component_base = component_base_result?;
    let coordinate_ms = component_base.coordinate_ms;
    let parser_extract_ms = component_base.parser_extract_ms;
    let control_overlap_ms = control_overlap_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_control_lexical_overlap] control_ms={control_elimination_ms:.3} lexical_ms={lexical_discovery_ms:.3} changed_terminals={} coordinate_ms={coordinate_ms:.3} parser_extract_ms={parser_extract_ms:.3} wall_ms={control_overlap_ms:.3}",
            control_changed_terminals.len(),
        );
        if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_REDUNDANCY").is_some() {
            eprintln!(
                "[glrmask/profile][constraint_control_changed_ids] ids={control_changed_terminals:?}",
            );
        }
    }

    let remap_components = || -> Result<PreparedOwnedComponentArtifacts, String> {
        let PreparedComponentBase {
            component_id_map,
            component_maps,
            unmapped,
            ..
        } = component_base;
        let selected_boundary_tokens = selected_boundary_tokens_cell
            .get()
            .expect("boundary lexical prepass must publish selected tokens")
            .as_ref()
            .map_err(Clone::clone)?
            .clone();
        if let Some(selected_boundary_tokens) = selected_boundary_tokens {
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
                    unmapped,
                    component_maps,
                    Some(&plan.component_token_map),
                    plan.common_map.num_tsids() as usize,
                )?;
            Ok(PreparedOwnedComponentArtifacts {
                automata,
                possible_matches,
                top_accept_parts,
                id_map: plan.common_map,
                boundary_tsid_map: Some(plan.boundary_tsid_map),
                boundary_token_map: Some(plan.boundary_token_map),
                remap_ms,
            })
        } else {
            let common_tsids = component_id_map.num_tsids() as usize;
            let (automata, possible_matches, top_accept_parts, remap_ms) =
                remap_unmapped_component_artifacts(
                    unmapped,
                    component_maps,
                    None,
                    common_tsids,
                )?;
            Ok(PreparedOwnedComponentArtifacts {
                automata,
                possible_matches,
                top_accept_parts,
                id_map: component_id_map,
                boundary_tsid_map: None,
                boundary_token_map: None,
                remap_ms,
            })
        }
    };
    let finish_boundary = || {
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
    let (prepared_components_result, (boundary_result, boundary_ms)) =
        if std::env::var_os("GLRMASK_COMPOSE_SERIAL_COMPONENT_BOUNDARY").is_none() {
            rayon::join(remap_components, finish_boundary)
        } else {
            (remap_components(), finish_boundary())
        };
    let prepared_components = prepared_components_result?;
    let boundary_repair = boundary_result?;
    let preparation_ms = preparation_started_at.elapsed().as_secs_f64() * 1000.0;

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
                Some((boundary_dwa, boundary_id_map, boundary.claimed_weight)),
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

    let boundary_work = if std::env::var_os("GLRMASK_EXPERIMENT_SKIP_BOUNDARY_UNION").is_some() {
        None
    } else {
        boundary_work
    };

    // Parser-DWA union is independent of tokenizer materialization, terminal-live
    // assembly, and token-mask prebuilding. The old owned-parent path serialized
    // those stages even though both sides consume disjoint artifacts produced by
    // preparation. Run them together so the ~30-40 ms parser union hides the
    // ~20 ms tokenizer/result side instead of paying both on the critical path.
    let post_prepare_overlap_started_at = Instant::now();
    let (parser_union_result, result_side) = rayon::join(
        || -> Result<(DWA, PrebuiltParserWeightTokenSets, f64), String> {
            let union_started_at = Instant::now();
            let final_build_started_at = Instant::now();
            let mut automata = automata;
            let (parser_dwa, weight_token_sets) = match boundary_work {
                Some((mut boundary_dwa, boundary_id_map, mut claimed_weight)) => {
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
                    remap_weights_with_maps(
                        &mut [&mut claimed_weight],
                        &boundary_tsid_map,
                        &boundary_token_map,
                        id_num_tsids as usize,
                    );
                    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TOPOLOGY_OVERLAP").is_some() {
                        let (component_only, _, _) = determinize_compressed_component_union_prechecked(
                            automata.clone(),
                            None,
                            0.0,
                        );
                        profile_boundary_component_topology_overlap(&component_only, &boundary_dwa);
                    }
                    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_LANE_OWNERSHIP").is_some() {
                        subtract_weight_support_from_raw_automata(&mut automata, &claimed_weight);
                    }
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
                    let (parser_dwa, synthetic_states, weight_token_sets) =
                        determinize_compressed_component_union_prechecked(
                            automata,
                            Some(num_parser_states),
                            0.0,
                        );
                    let union_path = "overlap_local";
                    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
                    if let Some(automata) = validation_automata {
                        let mut reference = NWA::new(id_num_tsids, id_max_internal_token);
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
                    (parser_dwa, weight_token_sets)
                }
                None => {
                    let automata_len = automata.len();
                    let direct_started_at = Instant::now();
                    let (parser_dwa, synthetic_states, weight_token_sets) =
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
                    (parser_dwa, weight_token_sets)
                }
            };
            Ok((
                parser_dwa,
                weight_token_sets,
                union_started_at.elapsed().as_secs_f64() * 1000.0,
            ))
        },
        || {
            let terminal_live_started_at = Instant::now();
            let mut terminal_live_states = merged_terminal_live_states_owned_parent(
                &mut parent,
                children,
                &composed_table.terminal_offsets,
                &expected_tokenizer_state_offsets,
                num_terminals,
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
            let (parser_top_accept, parser_top_accept_token_sets) =
                collapse_transported_top_accept_parts(top_accept_parts, num_parser_states);
            result.constraint.parser_top_accept = parser_top_accept;

            let token_cache_started_at = Instant::now();
            result.constraint.internal_token_bytes = build_internal_token_bytes_from_groups(
                vocab,
                &result.constraint.internal_token_to_tokens,
            );
            result.constraint.prebuild_token_mask_caches();
            let token_cache_prebuild_ms =
                token_cache_started_at.elapsed().as_secs_f64() * 1000.0;
            (
                result,
                parser_top_accept_token_sets,
                terminal_live_ms,
                tokenizer_ms,
                token_cache_prebuild_ms,
            )
        },
    );
    let post_prepare_overlap_ms =
        post_prepare_overlap_started_at.elapsed().as_secs_f64() * 1000.0;
    let (parser_dwa, mut prebuilt_parser_weight_token_sets, union_ms) = parser_union_result?;
    let (
        mut result,
        parser_top_accept_token_sets,
        terminal_live_ms,
        tokenizer_ms,
        token_cache_prebuild_ms,
    ) = result_side;
    result.constraint.parser_dwa = parser_dwa;
    if std::env::var_os("GLRMASK_COMPOSE_DISABLE_PREBUILT_WEIGHT_INVENTORY").is_none() {
        let mut final_sets = prebuilt_parser_weight_token_sets
            .final_sets
            .into_iter()
            .map(|tokens| (Arc::as_ptr(&tokens) as usize, tokens))
            .collect::<FxHashMap<_, _>>();
        for tokens in parser_top_accept_token_sets {
            final_sets
                .entry(Arc::as_ptr(&tokens) as usize)
                .or_insert(tokens);
        }
        prebuilt_parser_weight_token_sets.final_sets = final_sets.into_values().collect();
        prebuilt_parser_weight_token_sets.includes_parser_top_accept = true;
        result.constraint.prebuilt_parser_weight_token_sets =
            Some(prebuilt_parser_weight_token_sets);
    }
    let finalize_started_at = Instant::now();
    result.constraint.rebuild_runtime_caches();
    let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_owned_parent] components={} table_ms={table_ms:.3} control_elimination_ms={control_elimination_ms:.3} tokenizer_ms={tokenizer_ms:.3} coordinate_ms={coordinate_ms:.3} parser_extract_ms={parser_extract_ms:.3} boundary_ms={boundary_ms:.3} preparation_ms={preparation_ms:.3} terminal_live_ms={terminal_live_ms:.3} union_ms={union_ms:.3} token_cache_prebuild_ms={token_cache_prebuild_ms:.3} post_prepare_overlap_ms={post_prepare_overlap_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
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
