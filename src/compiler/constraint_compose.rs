//! Composition of already-compiled constraint artifacts.
//!
//! Leaf providers retain their own parser and lexer coordinates. Composition
//! links those providers and publishes only the cross-boundary relations.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use smallvec::SmallVec;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
#[cfg(test)]
use crate::automata::lexer::ast::Expr;
use crate::automata::weighted_u32::determinize::determinize;
use crate::automata::weighted_u32::dwa::DWA;
use crate::automata::weighted_u32::nwa::{NWA, NWAState};
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::compiler::glr::labels::{
    DEFAULT_LABEL, encode_negative_label, encode_positive_label, is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::stages::equiv_types::{
    InternalIdMap, ManyToOneIdMap, MappedArtifact,
};
use crate::compiler::stages::mapped_artifact::{WeightRefs, remap_weights_with_maps};
use crate::compiler::stages::parser_dwa::{
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles,
    normalize_weighted_parser_stack_nwa,
    normalize_signed_weighted_parser_stack_nwa_small_boundary,
    resolve_negative_codes_small_boundary,
};
use glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::templates::characterize::{
    try_characterize_selected_terminals_for_terminal_count,
};
use crate::compiler::stages::templates::Templates;
use crate::compiler::glr::table::{
    Action,
    ComposedTable,
    ControlEliminationReport,
};
use crate::ds::bitset::BitSet;
use crate::ds::weight::{
    ScopedWeightOpCache,
    Weight,
};
use crate::runtime::{
    Constraint, ConstraintRuntimeBackend, SpecialTokenTerminal,
};
use crate::Vocab;

#[inline]
pub(crate) fn compose_profile_enabled() -> bool {
    crate::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
        || crate::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()

pub(crate) fn eliminate_composed_runtime_controls(
    composed: &mut ComposedTable,
) -> Result<Option<ControlEliminationReport>, String> {
    if composed.control_terminals.is_empty() {
        debug_assert!(composed.table.control_terminals.is_empty());
        return Ok(None);
    }
    debug_assert_eq!(composed.control_terminals, composed.table.control_terminals);
    let report = composed.table.eliminate_control_terminals_exact()?;
    composed.control_terminals.clear();
    debug_assert!(composed.table.control_terminals.is_empty());
    Ok(Some(report))
}

#[derive(Clone, Copy)]
pub(crate) struct PossibleMatchComponent<'a> {
    pub(crate) constraint: &'a Constraint,
    /// Local raw tokenizer state `s` is merged state `tokenizer_state_offset+s`.
    pub(crate) tokenizer_state_offset: u32,
    /// Global terminal offset of this component in the composed table.
    pub(crate) terminal_offset: u32,
}


pub(crate) struct CompiledSubgrammarInput<'a> {
    pub(crate) placeholder_terminal: u32,
    /// Additional parent terminals that call the same compiled child. The
    /// child parser/tokenizer artifacts are incorporated once and shared by
    /// every listed call site.
    pub(crate) additional_placeholder_terminals: &'a [u32],
    pub(crate) constraint: &'a Constraint,
}

impl<'a> CompiledSubgrammarInput<'a> {
    #[inline]
    fn placeholder_terminals(&self) -> impl Iterator<Item = u32> + '_ {
        std::iter::once(self.placeholder_terminal)
            .chain(self.additional_placeholder_terminals.iter().copied())
    }
}

pub(crate) fn build_segmented_parser_links(
    children: &[CompiledSubgrammarInput<'_>],
) -> Result<Vec<crate::runtime::SegmentedParserLink>, String> {
    let link_count = children
        .iter()
        .map(|child| 1 + child.additional_placeholder_terminals.len())
        .sum();
    let mut links = Vec::with_capacity(link_count);
    for (child_index, child) in children.iter().enumerate() {
        let child_component = u32::try_from(child_index + 1)
            .map_err(|_| "segmented parser component index overflow".to_owned())?;
        let return_pop = child.constraint.composition_child_return_pop()?;
        let child_start_nullable = child.constraint.composition_start_nullable()?;
        for slot_terminal in child.placeholder_terminals() {
            links.push(crate::runtime::SegmentedParserLink {
                parent_component: 0,
                slot_terminal,
                child_component,
                child_start: 0,
                return_pop,
                child_start_nullable,
            });
        }
    }
    Ok(links)
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


pub(crate) fn install_published_static_boundary_shards(
    overlay: &mut crate::runtime::StaticDynamicOverlayMetadata,
    shards: Vec<PublishedStaticBoundaryShard>,
) -> Result<(), String> {
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for shard in shards {
        let component_index = shard.start_component as usize;
        let component = overlay
            .segmented_parser_components
            .get_mut(component_index)
            .ok_or_else(|| {
                format!(
                    "static boundary shard references unknown start component {}",
                    shard.start_component,
                )
            })?;
        let runtime_shard = crate::runtime::SegmentedBoundaryShard {
            mask_vocabulary: Default::default(),
            start_component: shard.start_component,
            accepts_empty_stack: component_index == 0,
            candidate_tokens: Some(shard.candidate_tokens),
            backend: crate::runtime::SegmentedBoundaryShardBackend::StaticParser(shard.boundary),
        };
        component.boundary = Some(runtime_shard);
    }
    Ok(())
}

fn install_dynamic_direct_boundary_shards(
    overlay: &mut crate::runtime::StaticDynamicOverlayMetadata,
    boundary_tokens_by_start_component: Option<&[Vec<u32>]>,
) {
    for component in &mut overlay.segmented_parser_components {
        component.boundary = None;
    }
    for component_index in 0..overlay.segmented_parser_components.len() {
        let shard = crate::runtime::SegmentedBoundaryShard {
            mask_vocabulary: Default::default(),
            start_component: component_index as u32,
            accepts_empty_stack: component_index == 0,
            candidate_tokens: boundary_tokens_by_start_component
                .and_then(|rows| rows.get(component_index))
                .map(|tokens| Arc::<[u32]>::from(tokens.clone())),
            backend: crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect,
        };
        overlay.segmented_parser_components[component_index].boundary = Some(shard);
    }
}


fn build_direct_component_token_coordinates(
    components: &[PossibleMatchComponent<'_>],
    original_token_ids: &[u32],
) -> Result<(ManyToOneIdMap, Vec<Vec<Vec<u32>>>, bool), String> {
    let original_token_count = original_token_ids
        .last()
        .map_or(0, |token| *token as usize + 1);
    let token_table_error = |error: crate::Error| error.to_string();
    let mut selected_original =
        crate::error::try_token_table(original_token_count, false).map_err(token_table_error)?;
    for &original in original_token_ids {
        selected_original[original as usize] = true;
    }
    let mut token_to_global =
        crate::error::try_token_table(original_token_count, u32::MAX).map_err(token_table_error)?;
    let mut global_to_tokens = Vec::<Vec<u32>>::new();
    let mut token_representatives = Vec::<u32>::new();
    let mut local_to_global_tokens = components
        .iter()
        .map(|component| {
            let constraint = component.constraint;
            let local_count = if constraint.has_original_token_map() {
                constraint.internal_token_count()
            } else {
                original_token_count
            };
            vec![Vec::<u32>::new(); local_count]
        })
        .collect::<Vec<_>>();

    // A retained dynamic component has no static internal-token quotient: its
    // exact private coordinate is the original model-token ID. That coordinate
    // is injective, so once it participates in a product refinement no two
    // distinct original tokens can ever share a global class. Construct that
    // singleton refinement directly instead of inserting one unique Vec-key per
    // token into a BTreeMap merely to rediscover the identity partition.
    if components
        .iter()
        .any(|component| !component.constraint.has_original_token_map())
    {
        for &original in original_token_ids {
            let global = global_to_tokens.len() as u32;
            token_to_global[original as usize] = global;
            for (component_index, component) in components.iter().enumerate() {
                let constraint = component.constraint;
                let local = if constraint.has_original_token_map() {
                    constraint
                        .original_token_internal_at(original)
                        .unwrap_or(u32::MAX)
                } else {
                    // `original_token_ids` is already the composed public token
                    // domain: supplied vocabulary IDs plus live merged exact
                    // specials. Compiler-only late-grammar sentinels are removed
                    // before this point and are therefore not identity members.
                    original
                };
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
            token_representatives.push(original);
            global_to_tokens.push(vec![original]);
        }
        return Ok((
            ManyToOneIdMap {
                original_to_internal: token_to_global,
                internal_to_originals: global_to_tokens,
                representative_original_ids: token_representatives,
            },
            local_to_global_tokens,
            true,
        ));
    }

    // The overwhelmingly common subgrammar link is parent + one composed
    // child. Its exact common vocabulary partition is keyed by one child class
    // inside each parent class; avoid allocating a temporary Vec tuple for
    // every original token and using that Vec as a tree key.
    if components.len() == 2 {
        let parent = components[0].constraint;
        let child = components[1].constraint;
        let mut add_pair_class =
            |originals: Vec<u32>, parent_local: u32, child_local: u32| -> Result<(), String> {
                if originals.is_empty()
                    || (parent_local == u32::MAX && child_local == u32::MAX)
                {
                    return Ok(());
                }
                let global = global_to_tokens.len() as u32;
                for &original in &originals {
                    token_to_global[original as usize] = global;
                }
                for (component_index, local) in [parent_local, child_local].into_iter().enumerate() {
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

        let singleton_fast =
            crate::env::var_os("GLRMASK_DISABLE_TOKEN_COORD_SINGLETON_FAST").is_none();
        let dense_child_buckets =
            crate::env::var_os("GLRMASK_DISABLE_TOKEN_COORD_DENSE_CHILD_BUCKETS").is_none();
        if dense_child_buckets {
            let child_class_count = child.internal_token_count();
            let mut buckets = (0..child_class_count)
                .map(|_| Vec::<u32>::new())
                .collect::<Vec<_>>();
            let mut touched = Vec::<usize>::new();
            let mut touched_flags = vec![false; child_class_count];
            let mut unmapped = Vec::<u32>::new();

            if let Some(parent_token_groups) = parent.internal_token_groups() {
                for (parent_local, originals) in parent_token_groups.iter().enumerate() {
                    if singleton_fast && originals.len() == 1 {
                        let original = originals[0];
                        if selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            let child_local = child
                                .original_token_internal_at(original)
                                .unwrap_or(u32::MAX);
                            add_pair_class(vec![original], parent_local as u32, child_local)?;
                        }
                        continue;
                    }
                    for &original in originals {
                        if !selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let child_local = child
                            .original_token_internal_at(original)
                            .unwrap_or(u32::MAX);
                        if child_local == u32::MAX {
                            unmapped.push(original);
                            continue;
                        }
                        let child_index = child_local as usize;
                        if child_index >= child_class_count {
                            return Err(format!(
                                "component 1 token class {child_local} lies outside its internal token domain"
                            ));
                        }
                        if !touched_flags[child_index] {
                            touched_flags[child_index] = true;
                            touched.push(child_index);
                        }
                        buckets[child_index].push(original);
                    }
                    touched.sort_unstable();
                    for child_index in touched.drain(..) {
                        touched_flags[child_index] = false;
                        let originals = std::mem::take(&mut buckets[child_index]);
                        add_pair_class(originals, parent_local as u32, child_index as u32)?;
                    }
                    if !unmapped.is_empty() {
                        add_pair_class(
                            std::mem::take(&mut unmapped),
                            parent_local as u32,
                            u32::MAX,
                        )?;
                    }
                }
            }

            for &original in original_token_ids {
                let parent_local = parent
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                if parent_local != u32::MAX {
                    continue;
                }
                let child_local = child
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                if child_local == u32::MAX {
                    unmapped.push(original);
                    continue;
                }
                let child_index = child_local as usize;
                if child_index >= child_class_count {
                    return Err(format!(
                        "component 1 token class {child_local} lies outside its internal token domain"
                    ));
                }
                if !touched_flags[child_index] {
                    touched_flags[child_index] = true;
                    touched.push(child_index);
                }
                buckets[child_index].push(original);
            }
            touched.sort_unstable();
            for child_index in touched.drain(..) {
                touched_flags[child_index] = false;
                let originals = std::mem::take(&mut buckets[child_index]);
                add_pair_class(originals, u32::MAX, child_index as u32)?;
            }
            if !unmapped.is_empty() {
                add_pair_class(std::mem::take(&mut unmapped), u32::MAX, u32::MAX)?;
            }
        } else {
            if let Some(parent_token_groups) = parent.internal_token_groups() {
                for (parent_local, originals) in parent_token_groups.iter().enumerate() {
                    if singleton_fast && originals.len() == 1 {
                        let original = originals[0];
                        if selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            let child_local = child
                                .original_token_internal_at(original)
                                .unwrap_or(u32::MAX);
                            add_pair_class(vec![original], parent_local as u32, child_local)?;
                        }
                        continue;
                    }
                    let mut groups = BTreeMap::<u32, Vec<u32>>::new();
                    for &original in originals {
                        if !selected_original
                            .get(original as usize)
                            .copied()
                            .unwrap_or(false)
                        {
                            continue;
                        }
                        let child_local = child
                            .original_token_internal_at(original)
                            .unwrap_or(u32::MAX);
                        groups.entry(child_local).or_default().push(original);
                    }
                    for (child_local, originals) in groups {
                        add_pair_class(originals, parent_local as u32, child_local)?;
                    }
                }
            }

            let mut parent_unmapped = BTreeMap::<u32, Vec<u32>>::new();
            for &original in original_token_ids {
                let parent_local = parent
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                if parent_local != u32::MAX {
                    continue;
                }
                let child_local = child
                    .original_token_internal_at(original)
                    .unwrap_or(u32::MAX);
                parent_unmapped.entry(child_local).or_default().push(original);
            }
            for (child_local, originals) in parent_unmapped {
                add_pair_class(originals, u32::MAX, child_local)?;
            }
        }

        return Ok((
            ManyToOneIdMap {
                original_to_internal: token_to_global,
                internal_to_originals: global_to_tokens,
                representative_original_ids: token_representatives,
            },
            local_to_global_tokens,
            false,
        ));
    }
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
    let parent_token_groups = parent
        .internal_token_groups()
        .ok_or_else(|| "composed parent has no explicit internal-token partition".to_owned())?;
    for (parent_local, originals) in parent_token_groups.iter().enumerate() {
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
                        .original_token_internal_at(original)
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
        let parent_local = parent.original_token_internal_at(original).unwrap_or(u32::MAX);
        if parent_local != u32::MAX {
            continue;
        }
        let child_tuple = components[1..]
            .iter()
            .map(|component| {
                component
                    .constraint
                    .original_token_internal_at(original)
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
        false,
    ))
}


type PossibleMatches = BTreeMap<u32, Weight>;

#[derive(Clone)]
pub(crate) struct PublishedStaticBoundaryShard {
    pub(crate) start_component: u32,
    pub(crate) candidate_tokens: Arc<[u32]>,
    pub(crate) boundary: Arc<crate::runtime::SegmentedBoundaryParser>,
}


/// Publication metadata for a completed boundary parser.
pub(crate) struct BoundaryShardMetadata {
    pub(crate) start_component: u32,
    pub(crate) id_map: InternalIdMap,
    pub(crate) candidate_tokens: Arc<[u32]>,
}

/// Per-shard publish profile: template characterization + parser-DWA
/// materialization + runtime normalization, with output sizes.
pub(crate) struct WalkShardPublishProfile {
    pub(crate) templates_ms: f64,
    pub(crate) materialize_ms: f64,
    pub(crate) normalize_ms: f64,
    pub(crate) terms: usize,
    pub(crate) parser_states: usize,
    pub(crate) parser_trans: usize,
}

/// Publish one walk-built shard as a `StaticParser` boundary shard with
/// shard-local TSIDs (`uses_composed_tsid_coordinate = false` + the walk
/// id_map's private raw-state and token maps).
///
/// Templates are characterized fresh over the provider-materialized exact
/// boundary table (live leaf coordinate, unbound slots emptied — never the
/// dynamic path's control-bearing recursive table) for exactly the terminals
/// the crossing DWA emits (the same construction the discovery-built path
/// applies via `set_parser_table_override`); the parser is built with the
/// standard count-only constructor + runtime normalization.
///
/// Precondition: live state keys stay within the link-time union ranges (no
/// lazily-allocated virtual-residual tokenizer states — the private map only
/// covers link-time states). The caller falls back to dynamic shards when a
/// component tokenizer has a virtual residual runtime.
///
/// Coordinate contract: the installing runtime is compact segmented, whose
/// live tokenizer keys are leaf-packed scoped states (leaves back-to-back
/// from 0 with NO reset state) — not merged-union states (which insert a
/// fresh reset fan-out at 0). The private map is therefore indexed by the
/// scoped coordinate: `scoped(leaf i, local l)` maps to the TSID of
/// `merged[tokenizer_offsets[i] + l]`, with leaves in link-component order.
/// Indexing it by merged states instead shifts every lookup by the reset
/// state and silently misroutes queries (inner-6 over-admission +
/// inner-7 under-admission were both this bug).

pub(crate) fn publish_walk_boundary_shard_work(
    work: BoundaryShardMetadata,
    crossing: DWA,
    boundary_table: &Arc<crate::compiler::glr::table::GLRTable>,
    tokenizer_offsets: &[u32],
    component_state_counts: &[u32],
) -> Result<(PublishedStaticBoundaryShard, WalkShardPublishProfile), String> {
    let num_terminals = boundary_table.num_terminals;
    let mut selected = vec![false; num_terminals as usize];
    for state in crossing.states() {
        for &label in state.transitions.keys() {
            if label >= 0
                && let Some(slot) = selected.get_mut(label as usize)
            {
                *slot = true;
            }
        }
    }
    let templates_started_at = Instant::now();
    let characterizations = try_characterize_selected_terminals_for_terminal_count(
        boundary_table,
        num_terminals,
        &selected,
    )
    .map_err(|error| error.to_string())?;
    let templates = Templates::from_characterizations(&characterizations);
    let templates_ms = templates_started_at.elapsed().as_secs_f64() * 1000.0;

    let materialize_started_at = Instant::now();
    let mut id_map = work.id_map;
    let mut crossing = crossing;
    if crate::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_TSID_QUOTIENT").is_some() {
        let old_tsids = id_map.num_tsids() as usize;
        let token_classes = id_map.num_internal_tokens() as usize;
        let started_at = Instant::now();
        if let Some((old_to_new, unique_weights)) =
            compute_boundary_dwa_tsid_behavior_quotient(&crossing, old_tsids, token_classes)
        {
            let new_tsids = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
            if new_tsids < old_tsids {
                apply_boundary_tsid_quotient_to_dwa_and_id_map(&mut crossing, &mut id_map, &old_to_new)
                    .expect("early boundary TSID quotient must apply to matching DWA/id-map coordinate");
                if compose_profile_enabled() {
                    eprintln!("[glrmask/profile][constraint_boundary_early_tsid_quotient] old_tsids={} new_tsids={} unique_terminal_weights={} total_ms={:.3}",
                        old_tsids, new_tsids, unique_weights, started_at.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
    }
    let terminal_automaton = TerminalAutomaton::Dwa(crossing);
    let nondeterministic_bundles = crate::env::var_os("GLRMASK_EXPERIMENT_COMPILE_NONDETERMINISTIC_BUNDLES").is_some()
        && id_map.num_internal_tokens() >= crate::env::var("GLRMASK_COMPILE_NONDETERMINISTIC_BUNDLES_MIN_TOKENS")
            .ok().and_then(|value| value.parse::<u32>().ok()).unwrap_or(64);
    let parser_nwa = if nondeterministic_bundles {
        build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles(
            &terminal_automaton, num_terminals, &templates, boundary_table)
    } else {
        build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count(
            &terminal_automaton, num_terminals, &templates, boundary_table)
    };
    let small = id_map.num_tsids() <= 16 && id_map.num_internal_tokens() <= 64;
    let (parser_dwa, materialize_ms, normalize_ms) = match parser_nwa {
        None => (DWA::new(id_map.num_tsids(), id_map.max_internal_token_id()),
            materialize_started_at.elapsed().as_secs_f64() * 1000.0, 0.0),
        Some(nwa) if small && crate::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_FUSED").is_some() => {
            let dwa = normalize_signed_weighted_parser_stack_nwa_small_boundary(
                boundary_table, &nwa, id_map.num_tsids() as usize, id_map.num_internal_tokens() as usize)
                .ok_or_else(|| "small-boundary fused signed publication rejected boundary coordinate".to_string())?;
            (dwa, materialize_started_at.elapsed().as_secs_f64() * 1000.0, 0.0)
        }
        Some(mut nwa) => {
            let grouped = boundary_table.construction
                == crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged;
            let reverse_topo = if small
                && crate::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_RESOLUTION").is_some()
            {
                nwa = resolve_negative_codes_small_boundary(&nwa,
                    id_map.num_tsids() as usize, id_map.num_internal_tokens() as usize)
                    .ok_or_else(|| "small-boundary signed resolver rejected boundary coordinate".to_string())?;
                None
            } else {
                resolve_negative_codes_in_nwa(&mut nwa, grouped)
            };
            if crate::env::var_os("GLRMASK_EXPERIMENT_FAST_HASHCONS_RESOLVED_BOUNDARY_NWA").is_some() {
                nwa = reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(nwa, reverse_topo);
            } else if crate::env::var_os("GLRMASK_EXPERIMENT_HASHCONS_RESOLVED_BOUNDARY_NWA").is_some() {
                nwa = reverse_hashcons_positive_acyclic_nwa(nwa);
            }
            for (source, state) in nwa.states().iter().enumerate() {
                if let Some(&label) = state.transitions.keys().find(|&&label| is_negative_label(label)) {
                    return Err(format!("runtime boundary parser NWA contains negative/PUSH label {label} at state {source}"));
                }
            }
            let materialize_ms = materialize_started_at.elapsed().as_secs_f64() * 1000.0;
            let normalize_started_at = Instant::now();
            let dwa = normalize_weighted_parser_stack_nwa(boundary_table, &nwa);
            (dwa, materialize_ms, normalize_started_at.elapsed().as_secs_f64() * 1000.0)
        }
    };
            "[glrmask/profile][constraint_walk_shard_publish] start_component={} terms={} parser_states={} parser_trans={} templates_ms={templates_ms:.3} materialize_ms={materialize_ms:.3} normalize_ms={normalize_ms:.3}",
            work.start_component,
            selected.iter().filter(|slot| **slot).count(),
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
        );
    }
    finish_walk_shard_publication(
    finish_walk_shard_publication(
        work.start_component,
        work.candidate_tokens,
        parser_dwa,
        id_map,
        WalkShardPublishProfile {
            templates_ms,
            materialize_ms,
            normalize_ms,
            terms: selected.iter().filter(|slot| **slot).count(),
            parser_states: 0,
            parser_trans: 0,
        },
        BoundaryTsidLayout::Scoped { offsets: tokenizer_offsets, counts: component_state_counts },
    )
}

/// Publish one signed-transfer boundary shard built WITHOUT any composed or
/// provider table: the caller supplies the already positively-normalized
/// parser DWA (control-aware signed-NWA composition + exact negative
/// resolution + table-free normalization). Publication into the runtime
/// `StaticParser` format (shard-local TSID map in the scoped tokenizer
/// coordinate) is shared with the table-built path.
pub(crate) fn publish_signed_boundary_shard_work(
    work: BoundaryShardMetadata,
    parser_dwa: DWA,
    profile: WalkShardPublishProfile,
    coordinate: BoundaryTsidLayout<'_>,
) -> Result<(PublishedStaticBoundaryShard, WalkShardPublishProfile), String> {
    ensure_positive_runtime_parser_dwa(&parser_dwa)?;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_signed_shard_publish] start_component={} terms={} parser_states={} parser_trans={} templates_ms={:.3} materialize_ms={:.3} normalize_ms={:.3}",
            work.start_component,
            profile.terms,
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
            profile.templates_ms,
            profile.materialize_ms,
            profile.normalize_ms,
        );
    }
    finish_walk_shard_publication(
        work.start_component,
        work.candidate_tokens,
        parser_dwa,
        work.id_map,
        profile,
        coordinate,
    )
}

/// The selected parser-weight coordinate determines which scanner mapping is needed.
#[derive(Clone, Copy)]
pub(crate) enum BoundaryTsidLayout<'a> {
    Composed,
    Scoped {
        offsets: &'a [u32],
        counts: &'a [u32],
    },
}

fn finish_walk_shard_publication(
    start_component: u32,
    candidate_tokens: Arc<[u32]>,
    parser_dwa: DWA,
    id_map: InternalIdMap,
    mut profile: WalkShardPublishProfile,
    coordinate: BoundaryTsidLayout<'_>,
) -> Result<(PublishedStaticBoundaryShard, WalkShardPublishProfile), String> {
    let coordinate = match coordinate {
        BoundaryTsidLayout::Composed => crate::runtime::BoundaryTsidCoordinate::Composed,
        BoundaryTsidLayout::Scoped { offsets, counts } => {
            if offsets.len() != counts.len() {
                return Err(format!(
                    "walk boundary shard {start_component} tokenizer layout mismatch: {} offsets vs {} components",
                    offsets.len(), counts.len(),
                ));
            }
            let mut map = Vec::with_capacity(counts.iter().map(|&count| count as usize).sum());
            for (component, (&base, &count)) in offsets.iter().zip(counts).enumerate() {
                let start = base as usize;
                let end = start.checked_add(count as usize).ok_or_else(|| {
                    format!("walk boundary shard {start_component} component {component} tokenizer range overflows")
                })?;
                let row = id_map.tokenizer_states.original_to_internal.get(start..end).ok_or_else(|| {
                    format!("walk boundary shard {start_component} component {component} tokenizer range lies outside map")
                })?;
                if let Some(local) = row.iter().position(|&tsid| tsid == u32::MAX) {
                    return Err(format!(
                        "walk boundary shard {start_component} scoped state (component {component}, local {local}) has no TSID (merged {})",
                        start + local,
                    ));
                }
                map.extend_from_slice(row);
            }
            crate::runtime::BoundaryTsidCoordinate::ShardLocal(map)
        }
    };
    profile.parser_states = parser_dwa.num_states() as usize;
    profile.parser_trans = parser_dwa.num_transitions();
    Ok((
        PublishedStaticBoundaryShard {
            start_component,
            candidate_tokens,
            boundary: Arc::new(crate::runtime::SegmentedBoundaryParser {
                parser_dwa,
                coordinate,
                internal_token_to_originals: id_map.vocab_tokens.internal_to_originals,
            }),
        },
        profile,
    ))
}






fn compute_boundary_dwa_tsid_behavior_quotient(
    dwa: &DWA,
    old_tsid_count: usize,
    token_classes: usize,
) -> Option<(Vec<u32>, usize)> {
    if old_tsid_count <= 1 || token_classes == 0 || token_classes > 64 {
        return None;
    }
    let mut unique_ids = FxHashMap::<usize, u16>::default();
    let mut unique_weights = Vec::<Weight>::new();
    for weight in dwa.weight_refs() {
        if weight.is_full() || weight.is_empty() {
            continue;
        }
        let key = weight.ptr_key();
        if unique_ids.contains_key(&key) {
            continue;
        }
        let Ok(index) = u16::try_from(unique_weights.len()) else {
            return None;
        };
        unique_ids.insert(key, index);
        unique_weights.push(weight.clone());
    }
    type Signature = SmallVec<[(u16, u64); 4]>;
    let mut signatures = vec![Signature::new(); old_tsid_count];
    for (weight_index, weight) in unique_weights.iter().enumerate() {
        for (start, end, tokens) in weight.range_entries() {
            if end as usize >= old_tsid_count {
                return None;
            }
            let mut mask = 0u64;
            for range in tokens.ranges() {
                for token in range {
                    if token as usize >= token_classes || token >= 64 {
                        return None;
                    }
                    mask |= 1u64 << token;
                }
            }
            if mask == 0 {
                continue;
            }
            for tsid in start..=end {
                signatures[tsid as usize].push((weight_index as u16, mask));
            }
        }
    }
    let mut classes = FxHashMap::<Signature, u32>::default();
    let mut old_to_new = vec![0u32; old_tsid_count];
    for (old_tsid, signature) in signatures.into_iter().enumerate() {
        let next = classes.len() as u32;
        let class = *classes.entry(signature).or_insert(next);
        old_to_new[old_tsid] = class;
    }
    Some((old_to_new, unique_weights.len()))
}

fn apply_boundary_tsid_quotient_to_dwa_and_id_map(
    dwa: &mut DWA,
    id_map: &mut InternalIdMap,
    old_to_new: &[u32],
) -> Option<()> {
    let old_tsid_count = id_map.num_tsids() as usize;
    if old_to_new.len() != old_tsid_count {
        return None;
    }
    let new_tsid_count = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
    if new_tsid_count == 0 || new_tsid_count >= old_tsid_count {
        return None;
    }
    let token_count = id_map.num_internal_tokens() as usize;
    let tsid_map = old_to_new.iter().map(|&class| vec![class]).collect::<Vec<_>>();
    let token_map = (0..token_count as u32).map(|token| vec![token]).collect::<Vec<_>>();
    {
        let mut weights = dwa.weight_refs_mut();
        remap_weights_with_maps(&mut weights, &tsid_map, &token_map, new_tsid_count);
    }
    let old_groups = std::mem::take(&mut id_map.tokenizer_states.internal_to_originals);
    let old_representatives = std::mem::take(&mut id_map.tokenizer_states.representative_original_ids);
    let mut new_groups = vec![Vec::<u32>::new(); new_tsid_count];
    let mut new_representatives = vec![u32::MAX; new_tsid_count];
    for old_tsid in 0..old_tsid_count {
        let class = old_to_new[old_tsid] as usize;
        if let Some(group) = old_groups.get(old_tsid) {
            new_groups[class].extend(group.iter().copied());
        }
        if new_representatives[class] == u32::MAX {
            new_representatives[class] = old_representatives
                .get(old_tsid)
                .copied()
                .or_else(|| new_groups[class].first().copied())
                .unwrap_or(old_tsid as u32);
        }
    }
    for group in &mut new_groups {
        group.sort_unstable();
        group.dedup();
    }
    for internal in &mut id_map.tokenizer_states.original_to_internal {
        if *internal != u32::MAX {
            *internal = *old_to_new.get(*internal as usize)?;
        }
    }
    id_map.tokenizer_states.internal_to_originals = new_groups;
    id_map.tokenizer_states.representative_original_ids = new_representatives;
    Some(())
}

/// Runtime parser segments are allowed to remain nondeterministic, but their
/// alphabet is strictly the positive parser-stack read alphabet (plus DEFAULT).
/// Negative labels encode compile-time PUSH effects and must never cross this
/// boundary. Keep this as a release-mode check so future compiler fast paths
/// cannot accidentally reintroduce runtime signed-stack interpretation.
fn ensure_positive_runtime_parser_dwa(dwa: &DWA) -> Result<(), String> {
    for (source, state) in dwa.states().iter().enumerate() {
        if let Some(&label) = state.transitions.keys().find(|&&label| is_negative_label(label)) {
            return Err(format!(
                "runtime boundary parser DWA contains negative/PUSH label {label} at state {source}"
            ));
        }
    }
    Ok(())
}



/// Build the exact reusable boundary-trigger Parser DWA for one component.
///
/// The trigger's weight coordinate is deliberately private and unquotiented:
/// raw local tokenizer-state IDs x original model-token IDs.  Ordinary
/// component TSID/token quotients are whole-token equivalences and need not
/// preserve proper-prefix boundary observations.
///
/// `candidate_tokens` may be any conservative superset. The current caller
/// feeds the cheaper Tokens trigger, which is built by scanning every raw lexer
/// state and therefore cannot remove a real proper-prefix event.
pub(crate) fn build_exact_component_boundary_trigger(
    constraint: &Constraint,
    candidate_tokens: &[u32],
) -> Result<Option<DWA>, String> {
    use crate::runtime::parser_backend::scoped_program::ScopedProgram;
    let parser = constraint.template_parser()
        .expect("LR-BACKED CONSTRAINT TRIGGER CONSTRUCTION IS FORBIDDEN");
    let merged;
    let tokenizer = if let Some(layout) = constraint.recursive_parser_layout()? {
        let leaves = layout.leaves.iter().map(|leaf| constraint
            .constraint_at_recursive_component_path(&leaf.component_path)
            .ok_or("exact trigger leaf does not resolve"))
            .collect::<Result<Vec<_>, _>>()?;
        let inputs = leaves.iter().zip(&layout.leaf_terminal_offsets)
            .map(|(leaf, &offset)| (leaf.composition_tokenizer(), offset)).collect::<Vec<_>>();
        merged = Tokenizer::disjoint_union_with_terminal_offsets(&inputs).0;
        &merged
    } else { constraint.composition_tokenizer() };
    let mut placeholders = constraint
        .late_grammar_slots.iter().map(|slot| slot.terminal_id)
        .collect::<Vec<_>>();
    placeholders.sort_unstable();
    placeholders.dedup();
    placeholders.retain(|&terminal| terminal < constraint.parser_terminal_count());

    let finish_terminal = constraint.parser_terminal_count();
    let mut event_terminals = placeholders.clone();
    event_terminals.push(finish_terminal);

    let mut terminal_nwa = NWA::new(
        tokenizer.num_states(),
        constraint.max_original_token_id().unwrap_or(0),
    );
    let global_start = terminal_nwa.add_state();
    let accept = terminal_nwa.add_state();
    terminal_nwa.set_start_states(vec![global_start]);
    terminal_nwa.set_final_weight(accept, Weight::all());

    let mut used_terminals = BTreeSet::<u32>::new();
    used_terminals.extend(event_terminals.iter().copied());
    let reset = tokenizer.start_state();
    let mut accepted_pairs = 0usize;

    for &token_id in candidate_tokens {
        let Some(bytes) = constraint.token_bytes_for_id(token_id) else {
            continue;
        };
        if bytes.len() < 2 {
            continue;
        }

        // Every raw tokenizer state is semantically distinct in this trigger
        // coordinate. Equal lexical DAGs may be hash-consed later; correctness
        // does not rely on the ordinary component TSID quotient.
        for raw_start in 0..tokenizer.num_states() {
            // Discover the tiny byte-offset DAG first. Do not allocate NWA
            // states for the overwhelmingly common `(raw state, token)` pairs
            // that have no proper internal lexeme boundary at all.
            let mut edges_by_offset = BTreeMap::<usize, Vec<(u32, usize)>>::new();
            let mut seen_offsets = BTreeSet::from([0usize]);
            let mut queue = VecDeque::from([0usize]);
            while let Some(offset) = queue.pop_front() {
                let lexer_start = if offset == 0 { raw_start } else { reset };
                let execution = tokenizer
                    .execute_from_state(&bytes[offset..], lexer_start);
                for matched in execution.matches {
                    if matched.width == 0 {
                        continue;
                    }
                    let Some(next_offset) = offset.checked_add(matched.width) else {
                        continue;
                    };
                    // An event at model-token end is handled before the next
                    // mask call and must not appear in the internal trigger.
                    if next_offset >= bytes.len() {
                        continue;
                    }
                    edges_by_offset
                        .entry(offset)
                        .or_default()
                        .push((matched.id, next_offset));
                    if seen_offsets.insert(next_offset) {
                        queue.push_back(next_offset);
                    }
                }
            }
            if edges_by_offset.is_empty() {
                continue;
            }
            for edges in edges_by_offset.values_mut() {
                edges.sort_unstable();
                edges.dedup();
            }

            let mut state_by_offset = BTreeMap::<usize, u32>::new();
            for &offset in &seen_offsets {
                state_by_offset.insert(offset, terminal_nwa.add_state());
            }
            let root = state_by_offset[&0];
            let support = Weight::from_per_tsid_token_sets(std::iter::once((
                raw_start,
                std::iter::once(token_id).collect::<RangeSetBlaze<u32>>(),
            )));
            terminal_nwa.add_epsilon(global_start, root, support);
            for (&offset, &state) in &state_by_offset {
                if offset != 0 {
                    for &event in &event_terminals {
                        terminal_nwa.add_transition(state, event as i32, accept, Weight::all());
                    }
                }
            }
            for (offset, edges) in edges_by_offset {
                let source = state_by_offset[&offset];
                for (terminal, next_offset) in edges {
                    terminal_nwa.add_transition(
                        source,
                        terminal as i32,
                        state_by_offset[&next_offset],
                        Weight::all(),
                    );
                    used_terminals.insert(terminal);
                }
            }
            accepted_pairs += 1;
        }
    }

    if accepted_pairs == 0 {
        return Ok(Some(DWA::new(
            tokenizer.num_states(),
            constraint.max_original_token_id().unwrap_or(0),
        )));
    }

    // The same immutable scoped relations and exact control-star assembler
    // used by static boundaries also decide readiness at the proper prefix.
    let (mut programs, completion, source_controls) = if let Some(composition) = parser.composition() {
        (composition.views[..composition.control_start as usize].to_vec(),
            composition.completion_view.clone(),
            composition.views[composition.control_start as usize..].to_vec())
    } else {
        ((0..parser.terminal_count as usize).map(|terminal|
            ScopedProgram::terminal(constraint, terminal)).collect(),
            ScopedProgram::completion(constraint), vec![])
    };
    programs.push(completion);
    let controls = (programs.len() as u32..programs.len() as u32 + source_controls.len() as u32)
        .collect::<Vec<_>>();
    programs.extend(source_controls);
    used_terminals.extend(controls.iter().copied());
    let (templates, classes) = crate::template_parser::static_compile::prepare_scoped_boundary_programs(
        &programs, parser.state_count, &used_terminals).map_err(|error| error.to_string())?;
    let lexical = determinize(&terminal_nwa)
        .map_err(|error| format!("exact trigger lexical determinization: {error:?}"))?;
    let output = crate::compiler::boundary_transfer::template_program::compile_classed(
        &templates, &controls, None, &lexical, &classes)?;
    Ok(Some(output.parser_dwa))
}




/// Reconstruct the merged terminal-expression sidecar when one or more loaded
/// components kept source expressions deferred outside their tokenizers.
/// Fresh components normally let `Tokenizer` merge these directly, so callers
/// only need this when the merged tokenizer otherwise has no expression list.
pub(crate) fn merged_retained_terminal_exprs(
    components: &[&Constraint],
    terminal_offsets: &[u32],
    total_terminals: u32,
) -> Option<Vec<crate::automata::regex::Expr>> {
    if components.len() != terminal_offsets.len() {
        return None;
    }
    let mut merged = vec![None; total_terminals as usize];
    for (component, &offset) in components.iter().zip(terminal_offsets) {
        let exprs = component.retained_terminal_exprs()?;
        if exprs.len() != component.tokenizer.num_terminals() as usize {
            return None;
        }
        for (local_terminal, expr) in exprs.iter().enumerate() {
            let global = offset as usize + local_terminal;
            let slot = merged.get_mut(global)?;
            if slot.is_some() {
                return None;
            }
            *slot = Some(expr.clone());
        }
    }
    merged.into_iter().collect()
}

fn full_nwa_topological_order(nwa: &NWA) -> Option<Vec<u32>> {
    let state_count = nwa.num_states() as usize;
    let mut indegree = vec![0usize; state_count];
    for state in nwa.states() {
        for (target, _) in &state.epsilons {
            *indegree.get_mut(*target as usize)? += 1;
        }
        for targets in state.transitions.values() {
            for (target, _) in targets {
                *indegree.get_mut(*target as usize)? += 1;
            }
        }
    }
    let mut queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut order = Vec::with_capacity(state_count);
    while let Some(source) = queue.pop_front() {
        order.push(source);
        let state = &nwa.states()[source as usize];
        for target in state
            .epsilons
            .iter()
            .map(|(target, _)| *target)
            .chain(
                state
                    .transitions
                    .values()
                    .flat_map(|targets| targets.iter().map(|(target, _)| *target)),
            )
        {
            let degree = &mut indegree[target as usize];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(target);
            }
        }
    }
    (order.len() == state_count).then_some(order)
}


fn reverse_hashcons_positive_acyclic_nwa_fast(nwa: NWA) -> NWA {
    reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(nwa, None, false)
}

fn reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
    nwa: NWA,
    precomputed_reverse_topo: Option<Vec<u32>>,
) -> NWA {
    reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(nwa, precomputed_reverse_topo, false)
}


fn reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(
    nwa: NWA,
    precomputed_reverse_topo: Option<Vec<u32>>,
    allow_signed_labels: bool,
) -> NWA {
    use rayon::prelude::*;
    use rustc_hash::FxHasher;
    use smallvec::SmallVec;
    use std::hash::{Hash, Hasher};

    fn fingerprint_state(source: &NWAState, old_to_new: &[u32]) -> u64 {
        let mut hasher = FxHasher::default();
        source
            .final_weight
            .as_ref()
            .map(Weight::ptr_key)
            .hash(&mut hasher);
        source.transitions.len().hash(&mut hasher);
        for (&label, targets) in &source.transitions {
            label.hash(&mut hasher);
            targets.len().hash(&mut hasher);
            for (target, weight) in targets {
                old_to_new[*target as usize].hash(&mut hasher);
                weight.ptr_key().hash(&mut hasher);
            }
        }
        source.epsilons.len().hash(&mut hasher);
        for (target, weight) in &source.epsilons {
            old_to_new[*target as usize].hash(&mut hasher);
            weight.ptr_key().hash(&mut hasher);
        }
        hasher.finish()
    }

    // Fingerprints are only a candidate index. Exact structural comparison is
    // the equivalence proof, so hash collisions cannot merge distinct states.
    fn equivalent_old_states(
        source: &NWAState,
        representative: &NWAState,
        old_to_new: &[u32],
    ) -> bool {
        if source.final_weight.as_ref().map(Weight::ptr_key)
            != representative.final_weight.as_ref().map(Weight::ptr_key)
            || source.transitions.len() != representative.transitions.len()
            || source.epsilons.len() != representative.epsilons.len()
        {
            return false;
        }
        for ((&source_label, source_targets), (&rep_label, rep_targets)) in
            source.transitions.iter().zip(representative.transitions.iter())
        {
            if source_label != rep_label || source_targets.len() != rep_targets.len() {
                return false;
            }
            for ((source_target, source_weight), (rep_target, rep_weight)) in
                source_targets.iter().zip(rep_targets.iter())
            {
                if old_to_new[*source_target as usize] != old_to_new[*rep_target as usize]
                    || source_weight.ptr_key() != rep_weight.ptr_key()
                {
                    return false;
                }
            }
        }
        source
            .epsilons
            .iter()
            .zip(representative.epsilons.iter())
            .all(
                |((source_target, source_weight), (rep_target, rep_weight))| {
                    old_to_new[*source_target as usize] == old_to_new[*rep_target as usize]
                        && source_weight.ptr_key() == rep_weight.ptr_key()
                },
            )
    }

    let profile = compose_profile_enabled();
    let total_started = profile.then(Instant::now);
    let topo_started = profile.then(Instant::now);
    let old_state_count = nwa.num_states() as usize;
    let reverse_order = if let Some(order) = precomputed_reverse_topo
        && order.len() == old_state_count
        && order.iter().all(|&state| (state as usize) < old_state_count)
    {
        order
    } else {
        let Some(order) = full_nwa_topological_order(&nwa) else {
            return nwa;
        };
        order.into_iter().rev().collect()
    };
    let topo_ms = topo_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);
    let (old_states, old_starts) = nwa.into_parts();
    let mut old_to_new = vec![u32::MAX; old_state_count];
    let mut representatives = Vec::<u32>::new();
    let mut buckets = FxHashMap::<u64, SmallVec<[u32; 2]>>::default();
    let classify_started = profile.then(Instant::now);

    for old_id in reverse_order {
        let source = &old_states[old_id as usize];
        debug_assert!(allow_signed_labels
            || source
                .transitions
                .keys()
                .all(|&label| !is_negative_label(label)));
        let fingerprint = fingerprint_state(source, &old_to_new);
        let existing = buckets.get(&fingerprint).and_then(|candidates| {
            candidates.iter().copied().find(|&candidate| {
                equivalent_old_states(
                    source,
                    &old_states[representatives[candidate as usize] as usize],
                    &old_to_new,
                )
            })
        });
        let new_id = if let Some(existing) = existing {
            existing
        } else {
            let created = representatives.len() as u32;
            representatives.push(old_id);
            buckets.entry(fingerprint).or_default().push(created);
            created
        };
        old_to_new[old_id as usize] = new_id;
    }
    let classify_ms = classify_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);

    let renumber_by_old_id = crate::env::var_os(
        "GLRMASK_EXPERIMENT_HASHCONS_RENUMBER_SKIP_SORT",
    )
    .is_some();
    if renumber_by_old_id {
        let mut class_by_old_id = representatives
            .iter()
            .copied()
            .enumerate()
            .map(|(class_id, old_id)| (old_id, class_id as u32))
            .collect::<Vec<_>>();
        class_by_old_id.sort_unstable_by_key(|(old_id, _)| *old_id);
        let mut class_to_sorted = vec![u32::MAX; representatives.len()];
        for (sorted_id, &(_, class_id)) in class_by_old_id.iter().enumerate() {
            class_to_sorted[class_id as usize] = sorted_id as u32;
        }
        for mapped in &mut old_to_new {
            *mapped = class_to_sorted[*mapped as usize];
        }
        representatives = class_by_old_id
            .into_iter()
            .map(|(old_id, _)| old_id)
            .collect();
    }

    // Move the representative states into the quotient and drop all duplicate
    // states on the Rayon workers. This avoids paying the discarded graph's
    // large serial destructor tail on the composition critical path.
    let move_started = profile.then(Instant::now);
    let mut representative_new_id = vec![u32::MAX; old_state_count];
    for (new_id, &old_id) in representatives.iter().enumerate() {
        representative_new_id[old_id as usize] = new_id as u32;
    }
    let mut moved = old_states
        .into_par_iter()
        .enumerate()
        .filter_map(|(old_id, mut state)| {
            let new_id = representative_new_id[old_id];
            if new_id == u32::MAX {
                return None;
            }
            for targets in state.transitions.values_mut() {
                for (target, _) in targets {
                    *target = old_to_new[*target as usize];
                }
            }
            for (target, _) in &mut state.epsilons {
                *target = old_to_new[*target as usize];
            }
            Some((new_id, state))
        })
        .collect::<Vec<_>>();
    if !renumber_by_old_id
        || moved.windows(2).any(|pair| pair[0].0 >= pair[1].0)
    {
        moved.par_sort_unstable_by_key(|(new_id, _)| *new_id);
    }
    let states = moved.into_iter().map(|(_, state)| state).collect::<Vec<_>>();
    let move_ms = move_started.map(|started| started.elapsed().as_secs_f64() * 1000.0);

    let mut starts = old_starts
        .into_iter()
        .map(|state| old_to_new[state as usize])
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    if let (Some(topo_ms), Some(classify_ms), Some(move_ms), Some(total_started)) =
        (topo_ms, classify_ms, move_ms, total_started)
    {
        eprintln!(
            "[glrmask/profile][fast_reverse_hashcons_nwa] signed={} input_states={} output_states={} topo_ms={topo_ms:.3} classify_ms={classify_ms:.3} move_drop_ms={move_ms:.3} total_ms={:.3}",
            allow_signed_labels,
            old_state_count,
            states.len(),
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    NWA::from_parts(states, starts)
}

fn reverse_hashcons_positive_acyclic_nwa(nwa: NWA) -> NWA {
    #[derive(Hash, PartialEq, Eq)]
    struct Key {
        final_weight: Option<usize>,
        transitions: Vec<(i32, Vec<(u32, usize)>)>,
        epsilons: Vec<(u32, usize)>,
    }

    let Some(order) = full_nwa_topological_order(&nwa) else {
        return nwa;
    };
    let old_state_count = nwa.num_states() as usize;
    let (old_states, old_starts) = nwa.into_parts();
    let mut old_to_new = vec![u32::MAX; old_state_count];
    let mut states = Vec::new();
    let mut ids = FxHashMap::<Key, u32>::default();

    for old_id in order.into_iter().rev() {
        let source = &old_states[old_id as usize];
        debug_assert!(source.transitions.keys().all(|&label| !is_negative_label(label)));
        let transitions = source
            .transitions
            .iter()
            .map(|(&label, targets)| {
                (
                    label,
                    targets
                        .iter()
                        .map(|(target, weight)| {
                            let mapped = old_to_new[*target as usize];
                            debug_assert_ne!(mapped, u32::MAX);
                            (mapped, weight.ptr_key())
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let epsilons = source
            .epsilons
            .iter()
            .map(|(target, weight)| {
                let mapped = old_to_new[*target as usize];
                debug_assert_ne!(mapped, u32::MAX);
                (mapped, weight.ptr_key())
            })
            .collect::<Vec<_>>();
        let key = Key {
            final_weight: source.final_weight.as_ref().map(Weight::ptr_key),
            transitions,
            epsilons,
        };
        let new_id = if let Some(&existing) = ids.get(&key) {
            existing
        } else {
            let created = states.len() as u32;
            let mut state = source.clone();
            for targets in state.transitions.values_mut() {
                for (target, _) in targets {
                    *target = old_to_new[*target as usize];
                }
            }
            for (target, _) in &mut state.epsilons {
                *target = old_to_new[*target as usize];
            }
            states.push(state);
            ids.insert(key, created);
            created
        };
        old_to_new[old_id as usize] = new_id;
    }

    let mut starts = old_starts
        .into_iter()
        .map(|state| old_to_new[state as usize])
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    NWA::from_parts(states, starts)
}



/// Transport complete PM relations across the same intact disjoint leaf
/// lexer union used by native static boundaries. The component coordinate
/// carries PM as well as parser observations; terminal-DWA equivalence alone
/// is never a certificate of possible-match equivalence.
pub(super) fn prepared_leaf_possible_matches(
    leaves: &[&Constraint],
    tokenizer_offsets: &[u32],
    terminal_offsets: &[u32],
    merged_tokenizer_states: u32,
    vocab: &Vocab,
) -> Result<Option<MappedArtifact<PossibleMatches>>, String> {
    if leaves.len()!=tokenizer_offsets.len() || leaves.len()!=terminal_offsets.len() {
        return Err("prepared PM leaf coordinate count disagrees".into());
    }
    if leaves.iter().any(|leaf| !leaf.possible_matches_complete
        || leaf.tokenizer.has_any_virtual_runtime()
        || leaf.tokenizer_tsids.has_csr()
        || leaf.tokenizer_tsids.primary().len()!=leaf.tokenizer.num_states() as usize) {
        return Ok(None);
    }
    let started=Instant::now();
    let components=leaves.iter().zip(tokenizer_offsets).zip(terminal_offsets)
        .map(|((&constraint,&tokenizer_state_offset),&terminal_offset)|PossibleMatchComponent {
            constraint,tokenizer_state_offset,terminal_offset,
        }).collect::<Vec<_>>();
    let mut original_tokens=vocab.entries_map().keys().copied().collect::<Vec<_>>();
    original_tokens.sort_unstable();
    // Native leaf PM deliberately leaves non-query lexer states unmapped.
    // Parser coordinates require total state coverage, so do not use their
    // total-state builder here. Lift precisely the retained PM classes and
    // the synthetic root's epsilon membership, preserving the empty domain.
    let mut state_map=vec![u32::MAX;merged_tokenizer_states as usize];
    let mut state_groups=vec![vec![0]];
    if let Some(root)=state_map.first_mut() {*root=0;}
    let mut local_state_maps=Vec::with_capacity(leaves.len());
    for component in &components {
        let leaf=component.constraint;
        let groups=leaf.internal_tsid_groups();
        let mut local=vec![Vec::new();groups.len()];
        for (tsid,group) in groups.iter().enumerate() {
            if group.is_empty() {continue;}
            let global=state_groups.len() as u32;
            let mut translated=Vec::with_capacity(group.len());
            for &state in group {
                let state=component.tokenizer_state_offset.checked_add(state)
                    .ok_or_else(||"prepared PM tokenizer state overflow".to_string())?;
                let slot=state_map.get_mut(state as usize)
                    .ok_or_else(||"prepared PM state lies outside lexical union".to_string())?;
                if *slot!=u32::MAX {return Err("prepared PM leaf state coordinates overlap".into());}
                *slot=global;translated.push(state);
            }
            local[tsid].push(global);state_groups.push(translated);
        }
        let start=leaf.tokenizer_tsids.primary()[leaf.tokenizer.initial_state() as usize];
        if start!=u32::MAX {
            local.get_mut(start as usize).ok_or_else(||"prepared PM initial class is invalid".to_string())?.push(0);
        }
        local_state_maps.push(local);
    }
    let tokenizer_states=ManyToOneIdMap::from_original_to_internal_allowing_unmapped(state_map,state_groups.len() as u32);
    let (vocab_tokens,local_token_maps,_)=build_direct_component_token_coordinates(&components,&original_tokens)?;
    let maps=local_state_maps.into_iter().zip(local_token_maps)
        .map(|(local_to_global_tsids,local_to_global_tokens)|DirectComponentCoordinateMaps {
            local_to_global_tsids,local_to_global_tokens,
        }).collect::<Vec<_>>();
    let id_map=InternalIdMap {tokenizer_states,vocab_tokens,deferred_vocab_singleton_original_ids:None};
    let coordinate_ms=started.elapsed().as_secs_f64()*1000.0;
    let remap_started=Instant::now();
    let mut possible=PossibleMatches::new();
    for (component,maps) in components.iter().zip(maps) {
        let mut local=component_possible_matches(component,component.terminal_offset)?;
        remap_weights_with_maps(&mut local.weight_refs_mut(),&maps.local_to_global_tsids,
            &maps.local_to_global_tokens,id_map.num_tsids() as usize);
        for (terminal,weight) in local {
            if weight.is_empty() {continue;}
            possible.entry(terminal).and_modify(|existing|*existing=existing.union(&weight)).or_insert(weight);
        }
    }
    let remap_ms=remap_started.elapsed().as_secs_f64()*1000.0;
    let compact_started=Instant::now();
    let mut mapped=MappedArtifact::new(possible,id_map);
    // Remove component parser-only distinctions before B is reconciled. This
    // exact PM observation quotient avoids inflating the boundary artifact.
    mapped.compact_dimensions_fast();
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][prepared_leaf_possible_matches] leaves={} rows={} tsids={} tokens={} coordinate_ms={coordinate_ms:.3} remap_ms={remap_ms:.3} compact_ms={:.3} total_ms={:.3}",
            leaves.len(),mapped.artifact().len(),mapped.id_map().num_tsids(),mapped.id_map().num_internal_tokens(),
            compact_started.elapsed().as_secs_f64()*1000.0,started.elapsed().as_secs_f64()*1000.0);
    }
    Ok(Some(mapped))
}

fn component_possible_matches(
    component: &PossibleMatchComponent<'_>,
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
        .runtime_possible_match_terminals()
        .filter_map(|terminal| {
            component
                .constraint
                .runtime_possible_match_weight(terminal)
                .map(|weight| (terminal_offset + terminal, weight.to_weight()))
        })
        .collect())
}


fn remap_composition_template_label(label: i32, state_relation: &[Vec<u32>]) -> Option<i32> {
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


/// in the same transition traversal.  The eager composition fast path needs
/// both representations immediately, so rebuilding the skeleton afterward
/// would just walk and allocate every transported transition a second time.
pub(crate) fn transport_composition_template_dfa_with_skeleton(
    mut dfa: UnweightedDfa,
    state_relation: &[Vec<u32>],
) -> Option<(UnweightedDfa, NWA)> {
    let mut nwa_states = Vec::with_capacity(dfa.states.len());
    for state in &mut dfa.states {
        let old = std::mem::take(&mut state.transitions);
        let mut mapped = BTreeMap::new();
        let mut nwa_transitions = BTreeMap::new();
        for (label, target) in old {
            let mapped_label = remap_composition_template_label(label, state_relation)?;
            if let Some(previous) = mapped.insert(mapped_label, target)
                && previous != target
            {
                return None;
            }
            if let Some(previous) = nwa_transitions.insert(
                mapped_label,
                vec![(target, Weight::empty())],
            ) && previous[0].0 != target
            {
                return None;
            }
        }
        state.transitions = mapped;
        nwa_states.push(NWAState {
            final_weight: state.is_accepting.then(Weight::empty),
            transitions: nwa_transitions,
            epsilons: Vec::new(),
        });
    }
    let start_state = dfa.start_state;
    let nwa = NWA::from_parts(nwa_states, vec![start_state]);
    Some((dfa, nwa))
}



pub(crate) fn merged_terminal_display_names(
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

pub(crate) fn merged_leaf_terminal_display_names(leaves: &[&Constraint]) -> Vec<String> {
    let mut names = Vec::new();
    for (index, leaf) in leaves.iter().enumerate() {
        if index == 0 {
            names.extend(leaf.terminal_display_names.iter().cloned());
        } else {
            names.extend(
                leaf.terminal_display_names
                    .iter()
                    .map(|name| format!("subgrammar{index}::{name}")),
            );
        }
    }
    names
}

/// Leaf-slice version of `component_ignores_are_globally_erasable` for nested
/// static links, where the link spans recursively expanded intact leaves
/// rather than direct components.
pub(crate) fn leaf_ignores_are_globally_erasable(leaves: &[&Constraint]) -> bool {
    if leaves.iter().any(|leaf| !leaf.parser_skip_terminals().is_empty()) {
        return false;
    }
    let Some(first) = leaves.first() else {
        return true;
    };
    match first.ignore_terminal {
        None => leaves.iter().all(|leaf| leaf.ignore_terminal.is_none()),
        Some(_) => {
            let Some(expected) = constraint_ignore_expr(first) else {
                return false;
            };
            leaves.iter().skip(1).all(|leaf| {
                let Some(_ignore) = leaf.ignore_terminal else {
                    return false;
                };
                constraint_ignore_expr(leaf).is_some_and(|actual| actual == expected)
            })
        }
    }
}

/// Leaf-slice version of `merged_ignore_terminals` for nested static links.
pub(crate) fn merged_leaf_ignore_terminals(
    leaves: &[&Constraint],
    leaf_offsets: &[u32],
    globally_erasable: bool,
) -> MergedIgnoreTerminals {
    let ignores = leaves
        .iter()
        .enumerate()
        .filter_map(|(leaf_index, leaf)| {
            leaf.ignore_terminal
                .map(|terminal| leaf_offsets[leaf_index] + terminal)
        })
        .collect::<Vec<_>>();
    let canonical = globally_erasable
        .then(|| ignores.first().copied())
        .flatten();
    let canonical_expr = canonical.and_then(|_| constraint_ignore_expr(leaves[0]).cloned());
    let mut all = BitSet::new(
        leaf_offsets
            .iter()
            .copied()
            .zip(leaves.iter())
            .map(|(offset, leaf)| offset + leaf.tokenizer.num_terminals())
            .max()
            .unwrap_or(0) as usize,
    );
    for &ignore in &ignores {
        all.set(ignore as usize);
    }
    for (leaf_index, leaf) in leaves.iter().enumerate() {
        let terminal_offset = leaf_offsets[leaf_index] as usize;
        for &skip in leaf.parser_skip_terminals() {
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

/// Recursively clear boundary shards inside a nested component selected for
/// the conservative legacy replacement lane.
///
/// That lane builds the outer shard over full vocabulary with per-leaf
/// ownership, so it explicitly subsumes the replaced block's inner repair.
/// Certified-static blocks do not use this helper: their inner static shards
/// remain load-bearing and coexist with the summary-restricted outer block
/// shard. Operates on this composition's own overlay copies (`Arc::make_mut`
/// detaches shared children), never on the caller's input constraints.
pub(crate) fn clear_nested_segmented_boundary_shards(constraint: &mut Constraint) {
    let Some(overlay) = constraint.static_dynamic_overlay.as_mut() else {
        return;
    };
    for component in &mut overlay.segmented_parser_components {
        let inner = std::sync::Arc::make_mut(&mut component.constraint);
        if inner.static_dynamic_overlay.is_none() {
            continue;
        }
        if let Err(error) =
            install_published_static_boundary_shards(inner.static_dynamic_overlay.as_mut().expect(
                "nested segmented component requires overlay for shard clearing",
            ), Vec::new())
        {
            debug_assert!(false, "nested shard clearing must not fail: {error}");
        }
        clear_nested_segmented_boundary_shards(inner);
    }
}


fn merged_original_token_ids(
    vocab: &Vocab,
    special_token_terminals: &[SpecialTokenTerminal],
) -> Vec<u32> {
    // The output is the existing dense original-token coordinate. Normalize
    // interval inputs first, rather than inserting their IDs into a tree.
    let ordinary = RangeSetBlaze::from_iter(vocab.entries_map().keys().copied());
    let exact = RangeSetBlaze::from_iter(special_token_terminals.iter().map(|special| special.token_ids()));
    (ordinary | exact).into_iter().collect()
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
                            token_id: special.token_id, token_id_end: special.token_id_end,
                        })
                }),
        );
    }
    merged.sort_unstable_by_key(|special| (special.token_id, special.token_id_end, special.terminal_id));
    merged.dedup_by_key(|special| (special.token_id, special.token_id_end, special.terminal_id));
    merged
}

#[derive(Debug, Clone)]
pub(crate) struct MergedIgnoreTerminals {
    pub(crate) canonical: Option<u32>,
    pub(crate) canonical_expr: Option<crate::automata::regex::Expr>,
    pub(crate) all: BitSet,
    /// Ignore terminals whose identity effect depends on the active parser
    /// scope. These remain visible to the boundary terminal/parser DWA.
    pub(crate) scoped: BitSet,
    /// Equivalent component-local ignore terminals which can be erased before
    /// parser interpretation. They are canonicalized to `canonical` in the
    /// final composed tokenizer/artifacts.
    pub(crate) global: BitSet,
    pub(crate) aliases: Vec<u32>,
}



fn constraint_ignore_expr(constraint: &Constraint) -> Option<&crate::automata::regex::Expr> {
    constraint.ignore_expr.as_ref().or_else(|| {
        constraint
            .ignore_terminal
            .and_then(|terminal| constraint.retained_terminal_expr(terminal))
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
pub(crate) fn component_ignores_are_globally_erasable(
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




pub(crate) fn merged_ignore_terminals(
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






/// A provider-native coordinator owns the shared vocabulary and the global
/// terminal coordinate, but does not compile a second lexer/parser mask graph.
fn build_recursive_coordinator(
    mut table: crate::compiler::glr::table::GLRTable,
    tokenizer: Arc<Tokenizer>,
    tokenizer_state_offsets: Vec<u32>,
    terminal_offsets: Vec<u32>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    embedded_end_token_ids: Vec<u32>,
    terminal_display_names: Vec<String>,
    tokenizer_fast_transitions: crate::runtime::FastTokenizerTransitions,
    vocab: &Vocab,
) -> ConstraintComposition {
    let parser_state_relations = vec![Vec::new(); terminal_offsets.len()];
    let tokenizer_has_epsilon_transitions = tokenizer.has_epsilon_transitions();
    assert!(table.num_states == 0 && table.action.is_empty() && table.goto.is_empty());
    table.set_embedded_end_token_ids(&embedded_end_token_ids);
    let token_bytes = crate::runtime::TokenByteDomain::from_entries(vocab.entries_arc());
    let constraint = Constraint::from_data(crate::runtime::ConstraintData {
            end_tokens: std::sync::Arc::from([]),
            marker_queries: None,
        runtime_backend: ConstraintRuntimeBackend::Static,
        static_dynamic_overlay: None,
        boundary_trigger: crate::runtime::BoundaryTrigger::None,
        boundary_candidate_summary: std::sync::OnceLock::new(),
        late_grammar_slots: Vec::new(),
        late_bind_vocab: OnceLock::from(vocab.clone()),
        pending_vocab_fingerprint: None,
        pending_special_token_ids: None,
        parser_dwa: crate::runtime::ParserDwaStorage::Absent,
        non_dwa_weights: Default::default(),
        direct_regular_wide_frontier_acceptance: Vec::new(),
        direct_regular_dynamic_hot_frontiers: Vec::new(),
        direct_regular_parser_state_acceptance: Vec::new(),
        direct_regular_automaton: None,
        table: crate::runtime::parser_backend::ParserExecutable::explicit(table),
        terminal_display_names,
        tokenizer,
        boundary_completion_index: None,
        tokenizer_has_epsilon_transitions,
        ignore_terminal: None,
        special_token_terminals,
        dynamic_mask_vocab: Default::default(),
        lazy_dynamic_mask_vocab: OnceLock::new(),
        empty_byte_token_ids: crate::compiler::compile::vocab_empty_byte_token_ids(vocab),
        possible_matches_complete: true,
        // No outer TSID quotient is needed by the provider runtime.
        tokenizer_tsids: crate::runtime::TokenizerTsidMap::partition(vec![0], vec![vec![0]]),
        composition_reset_tokens_by_terminal: Vec::new(),
        composition_parser_templates_by_terminal: Vec::new(),
        composition_parser_characterizations_by_terminal: Vec::new(),
        composition_grammar_summary: None,
        terminal_live_states: Vec::new(),
        token_classes: crate::runtime::TokenClasses::mapped(Vec::new(), Vec::new()),
        token_bytes,
        internal_token_buf_masks: Vec::new(),
        sliding_word_group_buf_masks: Default::default(),
        word_group_sparse_masks: Vec::new(),
        word_group_prefix_buf_masks: Default::default(),
        word_group_sparse_prefix_entries: Vec::new(),
        quad_group_sparse_masks: Vec::new(),
        quad_group_dense_masks: Vec::new(),
        byte_group_sparse_masks: Vec::new(),
        byte_group_dense_masks: Vec::new(),
        word_group_sparse_total_entries: 0,
        word_group_sparse_max_entries: 0,
        all_tokens_buf_mask: Box::new([]),
        internal_token_dense_words: 0,
        weight_token_masks: Default::default(),
        packed_dwa_token_dense_masks: Default::default(),
        seed_terminal_dense: FxHashMap::default(),
        seed_terminal_dense_fallback: Default::default(),
        seed_universe_dense: Arc::<[u64]>::from(Vec::<u64>::new().into_boxed_slice()),
        dwa_fast_transitions: Default::default(),
        parser_runtime_caches_prebuilt: false,
        indexed_dag_dense_transitions: Vec::new(),
        indexed_dag_dense_finals: Vec::new(),
        tokenizer_fast_transitions,
        heavy_token_dense_masks: Vec::new(),
        heavy_token_indices: Vec::new(),
            internal_token_buf_flat: Box::new([]),
            backed_internal_token_buf_flat: None,
        internal_token_buf_offsets: Box::new([]),
        total_internal_buf_cost: 0,
        heavy_total_cost: 0,
        light_avg_cost_x256: 0,
        internal_token_buf_op_costs: Vec::new(),
        word_group_buf_op_costs: Vec::new(),
        parser_state_domain_labels: Vec::new(),
        ignore_expr: None,
        serialized_artifact_cache: None,
        deferred_terminal_exprs_blob: None,
        deferred_terminal_exprs: Default::default(),
        deferred_composition_metadata_blob: None,
        composition_link_metadata_materialized: true,
        deferred_table_rules_blob: None,
        deferred_table_rules: Default::default(),
    });
    ConstraintComposition {
        constraint,
        terminal_offsets,
        tokenizer_state_offsets,
        parser_state_relations,
    }
}







fn recursive_component_tokenizer_span(constraint: &Constraint) -> Result<u32, String> {
    Ok(constraint
        .recursive_parser_layout()?
        .map_or(constraint.tokenizer.num_states(), |layout| layout.total_tokenizer_states))
}

fn merged_special_token_terminals_recursive_fast(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    terminal_offsets: &[u32],
) -> Vec<SpecialTokenTerminal> {
    let bound_parent_slots = children
        .iter()
        .flat_map(CompiledSubgrammarInput::placeholder_terminals)
        .collect::<BTreeSet<_>>();
    let mut merged = parent
        .special_token_terminals
        .iter()
        .filter(|special| {
            !bound_parent_slots.contains(&special.terminal_id)
                && !parent.is_late_grammar_placeholder_terminal(special.terminal_id)
        })
        .cloned()
        .collect::<Vec<_>>();
    for (child_index, child) in children.iter().enumerate() {
        let offset = terminal_offsets[child_index + 1];
        merged.extend(
            child
                .constraint
                .special_token_terminals
                .iter()
                .filter(|special| {
                    !child
                        .constraint
                        .is_late_grammar_placeholder_terminal(special.terminal_id)
                })
                .map(|special| SpecialTokenTerminal {
                    terminal_id: offset + special.terminal_id,
                    token_id: special.token_id, token_id_end: special.token_id_end,
                }),
        );
    }
    merged.sort_unstable_by_key(|special| (special.token_id, special.token_id_end, special.terminal_id));
    merged.dedup_by_key(|special| (special.token_id, special.token_id_end, special.terminal_id));
    merged
}

fn compose_dynamic_recursive_shared_fast(
    mut parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    shared_children: &[Arc<Constraint>],
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    let started_at = Instant::now();
    if children.len() != shared_children.len() {
        return Err("dynamic recursive shared child/component count mismatch".into());
    }
    for (index, (input, shared)) in children.iter().zip(shared_children).enumerate() {
        if !std::ptr::eq(input.constraint, shared.as_ref()) {
            return Err(format!(
                "dynamic recursive shared child {index} does not match borrowed composition input"
            ));
        }
    }
    if children.is_empty() {
        return Err("constraint composition requires at least one child".into());
    }
    parent.materialize_composition_link_metadata_for_compilation()?;
    let components = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let component_count = components.len();
    let vocab_check_started_at = Instant::now();
    for (component_index, constraint) in components.iter().enumerate() {
        if !constraint.token_bytes_match_vocab(vocab) {
            return Err(format!(
                "component {component_index} was not compiled for the supplied vocabulary",
            ));
        }
    }
    let vocab_check_ms = vocab_check_started_at.elapsed().as_secs_f64() * 1000.0;
    let placeholder_started_at = Instant::now();
    let component_end_token_ids = components
        .iter()
        .flat_map(|constraint| constraint.table.embedded_end_token_ids())
        .collect::<BTreeSet<_>>();
    validate_compiled_subgrammar_placeholders(
        &parent,
        children,
        vocab,
        &component_end_token_ids,
    )?;
    let placeholder_ms = placeholder_started_at.elapsed().as_secs_f64() * 1000.0;

    let terminal_checks_started_at = Instant::now();
    let mut terminal_offsets = Vec::with_capacity(components.len());
    let mut next_terminal = 0u32;
    for component in &components {
        terminal_offsets.push(next_terminal);
        next_terminal = next_terminal
            .checked_add(component.table.num_terminals)
            .ok_or_else(|| "dynamic recursive terminal coordinate overflow".to_owned())?;
    }
    let global_ignores = component_ignores_are_globally_erasable(&parent, children);
    let merged_ignores = merged_ignore_terminals(
        &parent,
        children,
        &terminal_offsets,
        global_ignores,
    );
    if merged_ignores.canonical.is_some() {
        return Err(
            "dynamic recursive fast path does not yet support globally-erased ignore aliases"
                .to_owned(),
        );
    }
    if children
        .iter()
        .any(|child| child.constraint.composition_start_nullable().unwrap_or(true))
    {
        return Err("dynamic recursive fast path does not yet support nullable children".into());
    }
    let terminal_checks_ms = terminal_checks_started_at.elapsed().as_secs_f64() * 1000.0;

    // Reuse checked component-prepared proper-prefix vocabularies. Unknown
    // metadata stays unknown: never run the static boundary compiler here and
    // never treat an unavailable proof as an empty candidate set.
    let candidate_started = Instant::now();
    let mut dynamic_candidate_ids = components.iter().map(|component| {
        if component.boundary_candidate_summary.get().is_some_and(|summary| summary.is_known()) {
            crate::compiler::boundary_candidates::boundary_candidate_ids(component, vocab).0
        } else { None }
    }).collect::<Vec<_>>();
    let no_ignores = components.iter().all(|component| component.ignore_terminal.is_none()
        && component.table.skip_terminals.is_empty());
    if (!global_ignores || no_ignores) && dynamic_candidate_ids[0].is_some() {
        let entered = children.iter().map(|child| child.constraint).collect::<Vec<_>>();
        let calls = children.iter().flat_map(|child| std::iter::once(child.placeholder_terminal)
            .chain(child.additional_placeholder_terminals.iter().copied())).collect::<Vec<_>>();
        if let Ok(refined) = crate::compiler::boundary_tail::build_root_call_candidates(
            &parent, &entered, &calls, vocab,
        ) {
            dynamic_candidate_ids[0].as_mut().expect("checked above")
                .retain(|id| refined.candidate_ids.binary_search(id).is_ok());
        }
    }
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][dynamic_prepared_boundary_vocabularies] counts={:?} total_ms={:.3}",
            dynamic_candidate_ids.iter().map(|ids| ids.as_ref().map(Vec::len)).collect::<Vec<_>>(),
            candidate_started.elapsed().as_secs_f64()*1000.0);
    }

    let tokenizer_span_started_at = Instant::now();
    let mut tokenizer_state_offsets = Vec::with_capacity(components.len());
    let mut next_tokenizer_state = 0u32;
    for component in &components {
        tokenizer_state_offsets.push(next_tokenizer_state);
        next_tokenizer_state = next_tokenizer_state
            .checked_add(recursive_component_tokenizer_span(component)?)
            .ok_or_else(|| "dynamic recursive tokenizer coordinate overflow".to_owned())?;
    }
    let tokenizer_span_ms = tokenizer_span_started_at.elapsed().as_secs_f64() * 1000.0;
    let links_started_at = Instant::now();
    let segmented_parser_links = build_segmented_parser_links(children)?;
    let links_ms = links_started_at.elapsed().as_secs_f64() * 1000.0;
    let validate_layout_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if segmented_parser_links.is_empty() {
        return Err("dynamic recursive fast path requires at least one linker control".into());
    }

    let metadata_started_at = Instant::now();
    let terminal_display_names = merged_terminal_display_names(&parent, children);
    debug_assert_eq!(terminal_display_names.len(), next_terminal as usize);
    let special_token_terminals = merged_special_token_terminals_recursive_fast(
        &parent,
        children,
        &terminal_offsets,
    );
    let live_special_token_ids = special_token_terminals.iter()
        .map(|special| special.token_ids()).collect::<RangeSetBlaze<u32>>();
    let embedded_end_token_ids = component_end_token_ids.iter().copied()
        .filter(|&id| live_special_token_ids.contains(id)).collect::<Vec<_>>();
    let metadata_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;

    let shell_started_at = Instant::now();
    // The live recursive runtime never executes this outer table. Keep only
    // enough root grammar metadata for nullable/end-token facts and the global
    // terminal coordinate; exact LR behavior remains in the retained leaves.
    let shell_rules = parent.table.rules.first().cloned().into_iter().collect::<Vec<_>>();
    let shell_table = crate::compiler::glr::table::GLRTable {
        action: Vec::new(),
        goto: Vec::new(),
        num_states: 0,
        num_terminals: next_terminal,
        num_rules: shell_rules.len() as u32,
        rules: shell_rules,
        nonterminal_display_names: parent.table.nonterminal_display_names.clone(),
        construction: parent.table.construction,
        admission_policy: parent.table.admission_policy,
        advance: Vec::new(),
        unconditional_advance: Vec::new(),
        forwarded_shifts: FxHashSet::default(),
        control_terminals: BTreeSet::new(),
        skip_terminals: BTreeSet::new(),
        guarded_shift_index: Vec::new(),
        direct_regular_wide_frontiers: Vec::new(),
    };
    let shell_ms = shell_started_at.elapsed().as_secs_f64() * 1000.0;

    let coordinator_started_at = Instant::now();
    let root_tokenizer = parent.tokenizer.clone();
    let root_fast_transitions = parent.tokenizer_fast_transitions.clone();
    let mut result = build_recursive_coordinator(
        shell_table,
        root_tokenizer,
        tokenizer_state_offsets.clone(),
        terminal_offsets.clone(),
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        root_fast_transitions,
        vocab,
    );
    let coordinator_ms = coordinator_started_at.elapsed().as_secs_f64() * 1000.0;

    let publish_started_at = Instant::now();
    drop(components);
    let mut segmented_components = Vec::with_capacity(component_count);
    segmented_components.push(crate::runtime::SegmentedParserComponent {
        constraint: Arc::new(parent),
        boundary: None,
        tokenizer_state_offset: tokenizer_state_offsets[0],
        terminal_offset: terminal_offsets[0],
        global_terminal_aliases: Vec::new(),
        local_tsid_to_global_tsids: Vec::new(),
    });
    for (child_index, child) in shared_children.iter().enumerate() {
        let component_index = child_index + 1;
        segmented_components.push(crate::runtime::SegmentedParserComponent {
            constraint: Arc::clone(child),
            boundary: None,
            tokenizer_state_offset: tokenizer_state_offsets[component_index],
            terminal_offset: terminal_offsets[component_index],
            global_terminal_aliases: Vec::new(),
            local_tsid_to_global_tsids: Vec::new(),
        });
    }
    let overlay = result
        .constraint
        .static_dynamic_overlay
        .get_or_insert_with(Default::default);
    overlay.segmented_parser_components = segmented_components;
    overlay.segmented_parser_links = segmented_parser_links;
    install_dynamic_direct_boundary_shards(overlay, None);
    for (component, ids) in overlay.segmented_parser_components.iter_mut().zip(dynamic_candidate_ids) {
        if let Some(shard) = component.boundary.as_mut() {
            shard.candidate_tokens = ids.map(Arc::<[u32]>::from);
        }
    }

    let publish_ms = publish_started_at.elapsed().as_secs_f64() * 1000.0;

    let layout_started_at = Instant::now();
    let layout = result
        .constraint
        .recursive_parser_layout_for_pending_root()?
        .ok_or_else(|| "dynamic recursive fast path failed to derive recursive layout".to_owned())?;
    let layout_ms = layout_started_at.elapsed().as_secs_f64() * 1000.0;
    let relation_started_at = Instant::now();
    // Authoritative DynamicDirect execution never consumes an outer recursive
    // TSID quotient. Leave it entirely absent in the live coordinator; the
    // serializer emits legacy compatibility rows only if/when save() is called.
    let relation_ms = relation_started_at.elapsed().as_secs_f64() * 1000.0;
    result.constraint.serialized_artifact_cache = None;

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_dynamic_recursive_shared_fast] components={} terminals={} scoped_tokenizer_states={} vocab_check_ms={vocab_check_ms:.3} placeholder_ms={placeholder_ms:.3} terminal_checks_ms={terminal_checks_ms:.3} tokenizer_span_ms={tokenizer_span_ms:.3} links_ms={links_ms:.3} validate_layout_ms={validate_layout_ms:.3} metadata_ms={metadata_ms:.3} shell_ms={shell_ms:.3} coordinator_ms={coordinator_ms:.3} publish_ms={publish_ms:.3} layout_ms={layout_ms:.3} relation_ms={relation_ms:.3} total_ms={:.3}",
            component_count,
            next_terminal,
            layout.total_tokenizer_states,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(result)
}



/// Compose already-compiled parent and child constraints. The component
/// lexers, parse tables, parser DWAs, and possible-match tables are transported
/// and reused; only the restricted cross-component boundary repair is compiled
/// from the merged artifacts.
fn validate_compiled_subgrammar_placeholders(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
    component_end_token_ids: &BTreeSet<u32>,
) -> Result<(), String> {
    for (child_index, child) in children.iter().enumerate() {
        for placeholder_terminal in child.placeholder_terminals() {
            let has_byte_token_matches = parent.runtime_possible_match_weight(placeholder_terminal)
                .is_some_and(|weight| !weight.is_empty());
            let has_exact_token_match = parent.special_token_terminals.iter().any(|special| {
                special.terminal_id == placeholder_terminal
                    && vocab.intersects_token_range(special.token_id, special.token_id_end)
            });
            if has_byte_token_matches || has_exact_token_match {
                return Err(format!(
                    "subgrammar placeholder terminal {placeholder_terminal} for child {child_index} matches one or more model-vocabulary tokens in the compiled parent; placeholders must be non-vocabulary sentinels (for example an @token(...) id outside the supplied vocabulary)",
                ));
            }
            for special in parent
                .special_token_terminals
                .iter()
                .filter(|special| special.terminal_id == placeholder_terminal)
            {
                if component_end_token_ids.range(special.token_ids()).next().is_some() {
                    return Err(format!(
                        "subgrammar placeholder terminal {placeholder_terminal} for child {child_index} uses token ID {}, which is also configured as a grammar-level end token; every replaced placeholder must use a unique sentinel token ID",
                        special.token_id,
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Boundary policy for the exact segmented composition runtime. Component
/// masks remain independently backed by their source constraints in both
/// modes. `StaticParserDwa` compiles crossing shards; `Dynamic` stores no
/// required B artifact and uses the exact composed full-vocabulary walker.
/// Boundary triggers remain available as dormant acceleration metadata rather
/// than restricting correctness traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SegmentedBoundaryBackend {
    StaticParserDwa,
    Dynamic,
}

pub(crate) fn compose_constraints_owned_parent_segmented(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<ConstraintComposition, String> {
    compose_constraints_owned_parent_impl(
        parent,
        children,
        None,
        boundary_backend,
        None,
        vocab,
    )
}

/// Compose with a per-starting-component boundary policy. `static_components`
/// is indexed in component order (parent first, then supplied children). A set
/// bit compiles and uses that component's static boundary shard; an unset bit
/// uses the zero-data exact dynamic boundary walker, gated by the component's
/// reusable trigger metadata.
pub(crate) fn compose_constraints_owned_parent_segmented_hybrid(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    vocab: &Vocab,
    static_components: &BitSet,
) -> Result<ConstraintComposition, String> {
    if static_components.len() != children.len() + 1 {
        return Err(format!(
            "hybrid boundary policy has {} components for a {}-component composition",
            static_components.len(),
            children.len() + 1,
        ));
    }
    compose_constraints_owned_parent_impl(
        parent,
        children,
        None,
        SegmentedBoundaryBackend::StaticParserDwa,
        Some(static_components),
        vocab,
    )
}

pub(crate) fn compose_constraints_owned_parent_segmented_shared(
    parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    shared_children: &[Arc<Constraint>],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<ConstraintComposition, String> {
    if shared_children.len() != children.len() {
        return Err("shared child/component count mismatch".into());
    }
    for (index, (input, shared)) in children.iter().zip(shared_children).enumerate() {
        if !std::ptr::eq(input.constraint, shared.as_ref()) {
            return Err(format!("shared child {index} does not match borrowed composition input"));
        }
    }
    compose_constraints_owned_parent_impl(
        parent,
        children,
        Some(shared_children),
        boundary_backend,
        None,
        vocab,
    )
}

fn compose_constraints_owned_parent_impl(
    mut parent: Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    shared_children: Option<&[Arc<Constraint>]>,
    boundary_backend: SegmentedBoundaryBackend,
    static_boundary_components: Option<&BitSet>,
    vocab: &Vocab,
) -> Result<ConstraintComposition, String> {
    if parent.has_template_parser() {
        let component_end_token_ids = std::iter::once(&parent)
            .chain(children.iter().map(|input| input.constraint))
            .flat_map(|constraint| constraint.embedded_end_token_ids())
            .collect::<BTreeSet<_>>();
        validate_compiled_subgrammar_placeholders(&parent, children, vocab, &component_end_token_ids)?;
        let mut named_children = Vec::with_capacity(children.len());
        let mut source_state_offsets = vec![0u32];
        let mut next_state = parent.parser_symbol_count();
        for (index, input) in children.iter().enumerate() {
            assert!(input.constraint.has_template_parser() && !input.constraint.table.is_present(),
                "LR-BACKED CONSTRAINT COMPOSITION IS FORBIDDEN");
            let name = format!("__compiled_component_{index}");
            for terminal in input.placeholder_terminals() {
                if terminal >= parent.parser_terminal_count() {
                    return Err("compiled placeholder terminal lies outside parent".into());
                }
                if let Some(slot) = parent.late_grammar_slots.iter_mut().find(|slot| slot.terminal_id == terminal) {
                    slot.name = name.clone();
                } else {
                    parent.late_grammar_slots.push(crate::runtime::LateGrammarSlot { name: name.clone(), terminal_id: terminal });
                }
            }
            let child = shared_children.map_or_else(|| Arc::new(input.constraint.clone()),
                |shared| Arc::clone(&shared[index]));
            source_state_offsets.push(next_state);
            next_state = next_state.checked_add(child.parser_symbol_count()).ok_or("parser scope overflow")?;
            named_children.push((name, child));
        }
        let source_state_counts = std::iter::once(parent.parser_symbol_count())
            .chain(children.iter().map(|input| input.constraint.parser_symbol_count())).collect::<Vec<_>>();
        let mut constraint = crate::runtime::parser_backend::link::compose(parent, &named_children, vocab)
            .map_err(|error| error.to_string())?;
        let dynamic_boundaries = static_boundary_components.map(|_| constraint.static_dynamic_overlay.as_ref()
            .unwrap().segmented_parser_components.iter().map(|component| component.boundary.clone()).collect::<Vec<_>>());
        if boundary_backend != SegmentedBoundaryBackend::Dynamic {
            crate::compiler::template_boundary::install(&mut constraint, vocab).map_err(|error| error.to_string())?;
        }
        if let (Some(selected), Some(dynamic)) = (static_boundary_components, dynamic_boundaries) {
            let overlay = constraint.static_dynamic_overlay.as_mut().unwrap();
            for (index, component) in overlay.segmented_parser_components.iter_mut().enumerate() {
                if !selected.contains(index) { component.boundary = dynamic[index].clone(); }
            }
            constraint.serialized_artifact_cache = None;
        }
        let (terminal_offsets, tokenizer_state_offsets) = constraint.static_dynamic_overlay.as_ref()
            .map(|overlay| overlay.segmented_parser_components.iter()
                .map(|component| (component.terminal_offset, component.tokenizer_state_offset))
                .unzip())
            .unwrap_or_else(|| (vec![0], vec![0]));
        let parser_state_relations = source_state_counts.iter().zip(source_state_offsets)
            .map(|(&count, offset)| (0..count).map(|local| vec![offset + local]).collect()).collect();
        return Ok(ConstraintComposition { constraint, terminal_offsets, tokenizer_state_offsets, parser_state_relations });
    }
    // Ordinary Dynamic has a real executable table in every leaf. Reuse the
    // existing recursive linker, never a native-to-LR reconstruction or a
    // flattened source/compiler view. Its nullable/global-ignore boundary is
    // checked by that linker and returns a capability error.
    if boundary_backend != SegmentedBoundaryBackend::Dynamic
        || !parent.table.is_present()
        || children.iter().any(|input|input.constraint.has_template_parser() || !input.constraint.table.is_present()) {
        return Err("a composition can't mix parts precompiled with the LrTable and TemplateDfa parser backends".into());
    }
    let shared=shared_children.ok_or("retained-LR composition requires owned shared children")?;
    let owned=shared.iter().map(|child| {
        let mut child=child.clone();
        if child.deferred_composition_metadata_blob.is_some() && !child.composition_link_metadata_materialized {
            Arc::make_mut(&mut child).materialize_composition_link_metadata_for_compilation()?;
        }
        Ok(child)
    }).collect::<Result<Vec<_>,String>>()?;
    let inputs=children.iter().zip(&owned).map(|(input,child)|CompiledSubgrammarInput {
        placeholder_terminal:input.placeholder_terminal,
        additional_placeholder_terminals:input.additional_placeholder_terminals,
        constraint:child.as_ref(),
    }).collect::<Vec<_>>();
    compose_dynamic_recursive_shared_fast(parent,&inputs,&owned,vocab)
}

/// Original model tokens accepted anywhere in an acyclic terminal DWA.
///
/// Fixpoint-free single topological pass; matches the fixpoint propagation
/// on acyclic inputs. Used for shard candidate-token triggers and gates.
/// Exact union of the correlated weights of all accepting paths.
/// This is the ordinary accepted-token summary before projecting away TSIDs.
pub(crate) fn accepted_weight_support(dwa: &DWA) -> Weight {
    assert!(dwa.is_acyclic(), "accepted-token summary expects acyclic DWA");
    let n = dwa.num_states() as usize;
    let mut indegree = vec![0usize; n];
    for state in dwa.states() {
        for &(target, _) in state.transitions.values() {
            indegree[target as usize] += 1;
        }
    }
    let mut queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::with_capacity(n);
    while let Some(source) = queue.pop_front() {
        topo.push(source);
        for &(target, _) in dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    assert_eq!(topo.len(), n);

    let mut reach = vec![Weight::empty(); n];
    reach[dwa.start_state() as usize] = Weight::all();
    let mut accepted = Weight::empty();
    let mut ops = ScopedWeightOpCache::default();
    for source in topo {
        let source_support = reach[source as usize].clone();
        if source_support.is_empty() {
            continue;
        }
        let state = &dwa.states()[source as usize];
        if let Some(final_weight) = state.final_weight.as_ref() {
            let support = ops.intersection(&source_support, final_weight);
            accepted = ops.union(&accepted, &support);
        }
        for &(target, ref edge_weight) in state.transitions.values() {
            let support = ops.intersection(&source_support, edge_weight);
            if support.is_empty() {
                continue;
            }
            reach[target as usize] = ops.union(&reach[target as usize], &support);
        }
    }

    accepted
}

pub(crate) fn accepted_original_tokens(
    dwa: &DWA,
    id_map: &InternalIdMap,
) -> BTreeSet<u32> {
    let profile = crate::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some();
    let started = profile.then(Instant::now);
    let accepted = accepted_weight_support(dwa);
    let support_ms = started.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
    let mut originals = BTreeSet::new();
    for (_, internal_tokens) in accepted.raw_range_values() {
        for range in internal_tokens.ranges() {
            for internal_token in range {
                if let Some(ids) = id_map
                    .vocab_tokens
                    .internal_to_originals
                    .get(internal_token as usize)
                {
                    originals.extend(ids.iter().copied());
                }
            }
        }
    }
    if let Some(started) = started {
        eprintln!("[glrmask/profile][accepted_token_support] states={} edges={} tokens={} support_ms={support_ms:.3} total_ms={:.3}",
            dwa.num_states(), dwa.num_transitions(), originals.len(), started.elapsed().as_secs_f64() * 1000.0);
    }
    originals
}

/// Load a `vocab_dump.bin` cache file (test/bench helper).
pub(crate) fn load_vocab(path: &str) -> Vocab {
    use std::fs;
    let bytes = fs::read(path).expect("read vocab dump");
    fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {
        let end = *offset + 4;
        let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
        *offset = end;
        value
    }
    let mut offset = 0usize;
    let count = read_u32(&bytes, &mut offset) as usize;
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let id = read_u32(&bytes, &mut offset);
        let len = read_u32(&bytes, &mut offset) as usize;
        let end = offset + len;
        entries.push((id, bytes[offset..end].to_vec()));
        offset = end;
    }
    assert_eq!(offset, bytes.len());
    Vocab::from_entries_unchecked(entries)
}

#[cfg(test)]
mod tests {
    use crate::ErrorExt as _;
    use super::*;
    use crate::compiler::glr::table::{
        GlrTableConstruction, SubgrammarTableInput, compose_subgrammar_tables,
    };
    use crate::grammar::flat::TerminalID;
    include!("minbound_scoped_ignore.rs");

    fn byte_vocab() -> Vocab {
        Vocab::from_entries_unchecked(
            (0u32..=255)
                .map(|byte| (byte, vec![byte as u8]))
                .collect(),
        )
    }

    // Compiler-algorithm fixtures own temporary LR tables directly. These
    // tables are never stored in a Constraint or used by its runtime.
    fn compiler_table_fixture(constraint: &Constraint) -> crate::compiler::glr::table::GLRTable {
        let grammar = constraint.template_parser().unwrap().link_grammar.as_ref().unwrap();
        let analyzed = grammar.analyze(constraint.terminal_display_names.clone());
        let mut table = crate::compiler::glr::table::GLRTable::build_with_default_construction(
            &analyzed, GlrTableConstruction::ExperimentalCoreMerged);
        table.set_embedded_start_nullable(grammar.root_nullable);
        table.set_embedded_end_token_ids(&grammar.embedded_end_token_ids);
        table
    }

    fn terminal(constraint: &Constraint, name: &str) -> TerminalID {
        constraint
            .terminal_display_names
            .iter()
            .position(|candidate| candidate == name)
            .unwrap() as u32
    }

    /// Ordinary (non-composition) compile of a grammar-level inline reference:
    /// structurally rewrite `extern grammar` occurrences to inline subgrammars
    /// and compile through the normal named-grammar path, using the ordinary
    /// GLRM default table construction (`ExperimentalCoreMerged`).
    fn compile_inline_source_oracle(
        parent: &str,
        children: &[(&str, &str)],
        vocab: &Vocab,
    ) -> Constraint {
        let named = crate::grammar::glrm::from_glrm_with_inline_subgrammars(parent, children)
            .expect("grammar-level inline subgrammar rewrite");
        crate::import::compile_from_named_grammar(
            named,
            vocab,
            "inline_source_oracle",
            GlrTableConstruction::ExperimentalCoreMerged,
            &[],
        )
        .expect("ordinary compile of grammar-level inline reference")
    }

    /// Differential check: the grammar-level inline reference (ordinary
    /// compile) must agree with explicit segmented Static and Dynamic built
    /// from the same source/vocabulary, including fused model tokens that cross
    /// the child boundary.
    #[cfg(feature = "internal-api")]
    #[test]
    fn grammar_level_inline_reference_matches_segmented_static_and_dynamic() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"!".to_vec()),
            (4, b"Xa".to_vec()),
            (5, b"b!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();

        let inline_parent = r#"
            start document;
            extern grammar SUB;
            nt document = "X" SUB "!";
        "#;
        let inline_children = [("SUB", "start child; nt child = \"a\" \"b\";")];
        let named = crate::grammar::glrm::from_glrm_with_inline_subgrammars(
            inline_parent,
            &inline_children,
        )
        .unwrap();
        let inline = crate::__private::compile_named_grammar(named.clone(), &vocab).unwrap();
        let compiled = crate::__private::compile_named_grammar(named, &vocab).unwrap();
        let segmented_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let segmented_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();

        for (name, constraint) in [
            ("inline", &inline),
            ("compile_named_grammar", &compiled),
            ("segmented_static", &segmented_static),
            ("segmented_dynamic", &segmented_dynamic),
        ] {
            let mut state = constraint.start();
            assert!(state.commit_bytes(b"Xa"), "{name} rejected fused Xa");
            assert!(state.commit_bytes(b"b"), "{name} rejected b");
            assert!(state.commit_bytes(b"!"), "{name} rejected !");
            assert!(state.is_accepting(), "{name} did not accept Xab!");
        }
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &inline,
            &segmented_static,
            &vocab,
            4,
            "inline-vs-segmented-static",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &segmented_dynamic,
            &segmented_static,
            &vocab,
            4,
            "segmented-dynamic-vs-segmented-static",
        );
    }


    #[test]
    fn dynamic_recursive_unprepared_inputs_use_minimal_shared_linker() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
            (3, b"!".to_vec()), (4, b"Xa".to_vec()), (5, b"b!".to_vec()),
            (6, b"Xab!".to_vec()), (7, b"XX".to_vec()), (8, b"c".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"start document; extern token SUB; nt document = "X" SUB "!";"#,
            &vocab, &[("SUB", 999)], &[]).unwrap();
        let mut child = Constraint::from_glrm_grammar(
            r#"start child; t A = "a"; t B = "b"; nt child = A B;"#, &vocab,
        ).unwrap();
        child.ensure_composition_reset_tokens_by_terminal();
        child.serialized_artifact_cache = None;
        let inner = parent.compose_linked_children_for_test_dynamic(
            &[("SUB", &child)], &vocab,
        ).unwrap();
        let parent_bytes = parent.save();
        for (expected_text, child_bytes) in [("Xab!", child.save()), ("XXab!!", inner.save())] {
            let loaded_child = Arc::new(Constraint::load(&child_bytes, &vocab).unwrap());
            let was_materialized = loaded_child.composition_link_metadata_materialized;
            if expected_text == "Xab!" {
                assert!(loaded_child.deferred_composition_metadata_blob.is_some());
                assert!(!was_materialized, "ordinary loaded child must exercise deferred link preparation");
            }
            let reference = Constraint::from_glrm_grammar(
                &format!("start document; nt document = {expected_text:?};"), &vocab,
            ).unwrap();
            for shared in [false, true] {
                let loaded_parent = Constraint::load(&parent_bytes, &vocab).unwrap();
                let inputs = [CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&loaded_parent, "SUB"),
                    additional_placeholder_terminals: &[],
                    constraint: loaded_child.as_ref(),
                }];
                let bound = if shared {
                    compose_constraints_owned_parent_segmented_shared(
                        loaded_parent, &inputs, &[Arc::clone(&loaded_child)], &vocab,
                        SegmentedBoundaryBackend::Dynamic,
                    )
                } else {
                    compose_constraints_owned_parent_segmented(
                        loaded_parent, &inputs, &vocab, SegmentedBoundaryBackend::Dynamic,
                    )
                }.unwrap().constraint;
                assert!(bound.uses_compact_segmented_parser_runtime());
                assert!(!bound.table.is_present(), "binding must publish a table-free coordinator");
                assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
                    &bound, &reference, &vocab, 6, "unprepared dynamic link vs monolithic",
                );
                let restored = Constraint::load(bound.save(), &vocab).unwrap();
                assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
                    &restored, &reference, &vocab, 6, "unprepared dynamic link reload",
                );
                assert_eq!(loaded_child.composition_link_metadata_materialized, was_materialized,
                    "preparing a retained child must not mutate the caller's shared/borrowed input");
                assert_eq!(loaded_child.save(), child_bytes);
            }
        }
    }

    #[test]
    fn dynamic_recursive_shared_fast_matches_borrowed_and_reload() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"!".to_vec()),
            (4, b"Xa".to_vec()),
            (5, b"b!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let shared_child = Arc::new(child.clone());
        let placeholder_terminal = terminal(&parent, "SUB");
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal,
            additional_placeholder_terminals: &[],
            constraint: shared_child.as_ref(),
        }];
        let fast = compose_constraints_owned_parent_segmented_shared(
            parent.clone(),
            &inputs,
            &[Arc::clone(&shared_child)],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        let borrowed = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(fast.uses_compact_segmented_parser_runtime());
        let overlay = fast.static_dynamic_overlay.as_ref().unwrap();
        assert!(
            overlay
                .recursive_tokenizer_internal_tsids
                .get()
                .is_none(),
            "all-DynamicDirect fast coordinator should omit the redundant recursive TSID relation",
        );

        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &fast,
            &borrowed,
            &vocab,
            4,
            "dynamic-shared-fast-vs-borrowed",
        );

        let bytes = fast.save();
        let reloaded = Constraint::load(&bytes, &vocab).unwrap();
        assert!(reloaded.uses_compact_segmented_parser_runtime());
        for constraint in [&fast, &reloaded] {
            assert!(
                constraint.static_dynamic_overlay.as_ref().unwrap()
                    .recursive_tokenizer_internal_tsids.get().is_none(),
                "saving and loading DynamicDirect must not invent a coordinator TSID relation",
            );
        }
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &fast,
            &reloaded,
            &vocab,
            4,
            "dynamic-shared-fast-vs-reload",
        );

        // A later static-boundary link is allowed to reconstruct compiler-only
        // flattened views on demand from the retained recursive component tree.
        // The fast dynamic build itself must never pay this cost.
        let outer_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token OUTER;
                nt document = "X" OUTER "!";
            "#,
            &vocab, &[("OUTER", 998)], &[])
        .unwrap();
        let static_outer = outer_parent
            .compose_linked_children_for_test(&[("OUTER", &fast)], &vocab)
            .unwrap();
        let dynamic_outer = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token OUTER;
                nt document = "X" OUTER "!";
            "#,
            &vocab, &[("OUTER", 998)], &[])
        .unwrap()
        .compose_linked_children_for_test_dynamic(&[("OUTER", &fast)], &vocab)
        .unwrap();
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &static_outer,
            &dynamic_outer,
            &vocab,
            4,
            "dynamic-shared-fast-static-upgrade-vs-dynamic",
        );
    }

    /// Differential check for a nested external child+grandchild through the
    /// grammar-level inline reference versus explicit segmented Static.
    #[cfg(feature = "internal-api")]
    #[test]
    fn grammar_level_inline_reference_nested_matches_segmented_static() {
        let vocab = byte_vocab();
        let inner = Constraint::from_glrm_grammar(
            r#"
                start inner;
                nt inner = "b";
            "#,
            &vocab,
        )
        .unwrap();
        let outer_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start outer;
                extern token INNER;
                nt outer = "a" INNER;
            "#,
            &vocab, &[("INNER", 998)], &[])
        .unwrap();
        let outer = outer_parent
            .compose_linked_children_for_test(&[("INNER", &inner)], &vocab)
            .unwrap();
        let document_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token OUTER;
                nt document = "<" OUTER ">";
            "#,
            &vocab, &[("OUTER", 999)], &[])
        .unwrap();
        let segmented = document_parent
            .compose_linked_children_for_test(&[("OUTER", &outer)], &vocab)
            .unwrap();

        let named = crate::grammar::glrm::from_glrm_with_inline_subgrammars(
            r#"
                start document;
                extern grammar OUTER;
                nt document = "<" OUTER ">";
            "#,
            &[
                (
                    "OUTER",
                    "start outer; extern grammar INNER; nt outer = \"a\" INNER;",
                ),
                ("OUTER::INNER", "start inner; nt inner = \"b\";"),
            ],
        )
        .unwrap();
        let inline = crate::__private::compile_named_grammar(named, &vocab).unwrap();

        let mut inline_state = inline.start();
        let mut segmented_state = segmented.start();
        assert!(inline_state.commit_bytes(b"<ab>"));
        assert!(segmented_state.commit_bytes(b"<ab>"));
        assert!(inline_state.is_accepting());
        assert!(segmented_state.is_accepting());
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &inline,
            &segmented,
            &vocab,
            4,
            "nested-inline-vs-segmented-static",
        );
    }

    trait ComposeLinkedChildrenForTest {
        fn compose_linked_children_for_test(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint>;

        fn compose_linked_children_for_test_owned(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint>;

        /// Exact dynamic segmented composition. Reference route for nested
        /// links (nullable nested links decline loudly on the static route;
        /// this backend accepts those shapes).
        fn compose_linked_children_for_test_dynamic(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint>;
    }

    impl ComposeLinkedChildrenForTest for Constraint {
        fn compose_linked_children_for_test(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint> {
            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = terminal(self, name);
                if !seen.insert(placeholder_terminal) {
                    return Err(crate::Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self.clone(),
                &inputs,
                vocab,
                SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::Error::Compilation)
        }

        fn compose_linked_children_for_test_owned(
            self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint> {
            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = terminal(&self, name);
                if !seen.insert(placeholder_terminal) {
                    return Err(crate::Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self,
                &inputs,
                vocab,
                SegmentedBoundaryBackend::StaticParserDwa,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::Error::Compilation)
        }

        fn compose_linked_children_for_test_dynamic(
            &self,
            children: &[(&str, &Constraint)],
            vocab: &Vocab,
        ) -> crate::Result<Constraint> {
            let mut inputs = Vec::with_capacity(children.len());
            let mut seen = BTreeSet::new();
            for &(name, child) in children {
                let placeholder_terminal = terminal(self, name);
                if !seen.insert(placeholder_terminal) {
                    return Err(crate::Error::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    additional_placeholder_terminals: &[],
                    constraint: child,
                });
            }
            compose_constraints_owned_parent_segmented(
                self.clone(),
                &inputs,
                vocab,
                SegmentedBoundaryBackend::Dynamic,
            )
            .map(|composition| composition.constraint)
            .map_err(crate::Error::Compilation)
        }
    }

    #[test]
    fn duplicate_children_from_independent_artifacts_compose() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"<abc>,<abc>".to_vec()),
            (2, b"<".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
            (7, b"c".to_vec()),
            (5, b">,<".to_vec()),
            (6, b">".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token LEFT;
                extern token RIGHT;
                nt document = "<" LEFT ">,<" RIGHT ">";
            "#,
            &vocab, &[("LEFT", 998), ("RIGHT", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt value = "a" "b" "c";
                nt child = value;
            "#,
            &vocab,
        )
        .unwrap();
        let loaded_child = Constraint::load(child.save(), &vocab).unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt value = "a" "b" "c";
                nt child = value;
                nt document = "<" child ">,<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let runtime_composed = parent
            .compose_linked_children_for_test(&[("LEFT", &child), ("RIGHT", &loaded_child)], &vocab)
            .unwrap();
        assert_constraints_equivalent_on_reachable_prefixes(
            &runtime_composed,
            &monolithic,
            &vocab,
            8,
        );
        let mut actual = runtime_composed.start();
        assert!(actual.commit_token(0));
        assert!(actual.is_accepting());

    }


    #[test]
    fn fast_reverse_hashcons_matches_exact_structural_quotient_on_weighted_dags() {
        fn weight(start: u32, end: u32) -> Weight {
            Weight::from_token_set_for_tsid(
                0,
                RangeSetBlaze::from_iter(std::iter::once(start..=end)),
            )
        }

        fn generated(seed: u32) -> NWA {
            // Construct duplicate sub-DAGs deliberately.  Every edge points to
            // a larger old state id, so the graph is acyclic; paired states use
            // distinct old targets whose suffixes are nevertheless equivalent.
            let mut nwa = NWA::new(1, 63);
            for _ in 0..14 {
                nwa.add_state();
            }
            let full = Weight::all();
            let narrow = weight(seed % 16, (seed % 16) + 7);
            nwa.set_final_weight(12, narrow.clone());
            nwa.set_final_weight(13, narrow.clone());
            for (left, right, left_target, right_target, label) in [
                (10, 11, 12, 13, 1),
                (8, 9, 10, 11, 2),
                (6, 7, 8, 9, DEFAULT_LABEL),
                (4, 5, 6, 7, 3),
                (2, 3, 4, 5, 4),
                (0, 1, 2, 3, 5),
            ] {
                nwa.add_transition(left, label, left_target, narrow.clone());
                nwa.add_transition(right, label, right_target, narrow.clone());
                if label != DEFAULT_LABEL {
                    nwa.add_epsilon(left, left_target, full.clone());
                    nwa.add_epsilon(right, right_target, full.clone());
                }
                if (left + seed).is_multiple_of(3) {
                    nwa.set_final_weight(left, full.clone());
                    nwa.set_final_weight(right, full.clone());
                }
            }
            nwa.set_start_states(vec![0, 1]);
            nwa
        }

        for seed in 0..64 {
            let input = generated(seed);
            let reference = reverse_hashcons_positive_acyclic_nwa(input.clone());
            let reverse_topo = full_nwa_topological_order(&input)
                .expect("generated quotient graph must be acyclic")
                .into_iter()
                .rev()
                .collect::<Vec<_>>();
            let fast = reverse_hashcons_positive_acyclic_nwa_fast(input.clone());
            let reused = reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
                input,
                Some(reverse_topo),
            );
            assert_eq!(
                fast.start_states(),
                reference.start_states(),
                "fast quotient start states differ for seed {seed}",
            );
            assert_eq!(
                fast.states(),
                reference.states(),
                "fast quotient structure differs from exact structural quotient for seed {seed}",
            );
            assert_eq!(
                reused.start_states(),
                reference.start_states(),
                "reused-topology quotient start states differ for seed {seed}",
            );
            assert_eq!(
                reused.states(),
                reference.states(),
                "reused-topology quotient differs from exact structural quotient for seed {seed}",
            );
        }
    }


    fn assert_accepts(constraint: &Constraint, bytes: &[u8]) {
        let mut state = constraint.start();
        assert!(state.commit_bytes(bytes));
        assert!(state.is_accepting(), "expected {:?} to finish", bytes);
    }

    fn assert_rejects(constraint: &Constraint, bytes: &[u8]) {
        let mut state = constraint.start();
        let accepted = state.commit_bytes(bytes) && state.is_accepting();
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
            "",
        );
    }

    /// Labeled variant: prefixes every failure message with `label` so a
    /// multi-backend comparison identifies which side (oracle/static/dynamic)
    /// diverged.
    fn assert_constraints_equivalent_on_reachable_prefixes_labeled(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
        label: &str,
    ) {
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            true,
            label,
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
            "",
        );
    }

    /// Labeled variant of the mask-only comparison.
    fn assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
        label: &str,
    ) {
        assert_constraints_equivalent_on_reachable_prefixes_inner(
            actual,
            expected,
            vocab,
            max_depth,
            false,
            label,
        );
    }

    fn assert_constraints_equivalent_on_reachable_prefixes_inner(
        actual: &Constraint,
        expected: &Constraint,
        vocab: &Vocab,
        max_depth: usize,
        compare_completion: bool,
        label: &str,
    ) {
        let tag = if label.is_empty() {
            String::new()
        } else {
            format!("{label}: ")
        };
        let token_ids = vocab.entries_map().keys().copied().collect::<Vec<_>>();
        let mut frontier = vec![Vec::<u32>::new()];
        for depth in 0..=max_depth {
            let mut next = Vec::new();
            for prefix in frontier {
                let mut actual_state = actual.start();
                let mut expected_state = expected.start();
                for &token in &prefix {
                    assert!(actual_state.commit_token(token));
                    assert!(expected_state.commit_token(token));
                }

                let actual_mask = actual_state.mask();
                let expected_mask = expected_state.mask();
                if crate::env::var_os("GLRMASK_DEBUG_PREFIX10_STACKS").is_some()
                    && prefix == [10]
                {
                    eprintln!("PREFIX10_ACTUAL {:?}", actual_state.debug_parser_stacks());
                    eprintln!("PREFIX10_EXPECTED {:?}", expected_state.debug_parser_stacks());
                }
                assert_eq!(
                    actual_mask, expected_mask,
                    "{tag}mask mismatch after reachable prefix {prefix:?}",
                );
                if compare_completion {
                    assert_eq!(
                        actual_state.is_accepting(),
                        expected_state.is_accepting(),
                        "{tag}completion mismatch after reachable prefix {prefix:?}",
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
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"<ab>".to_vec()),
            (1, b"<a".to_vec()),
            (2, b"b>".to_vec()),
            (3, b"<".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b">".to_vec()),
        ]);
        let compile_parent = || {
            Constraint::from_glrm_with_token_ids(
                r#"
                    start document;
                    extern token SUB;
                    nt document = "<" SUB ">";
                "#,
                &vocab, &[("SUB", 999)], &[])
            .unwrap()
        };
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child = "a" "b";
                nt document = "<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let borrowed_parent = compile_parent();
        let borrowed = borrowed_parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let owned = compile_parent()
            .compose_linked_children_for_test_owned(&[("SUB", &child)], &vocab)
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
    fn explicit_segmented_backends_agree() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"d".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"cd".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "a" SUB "d";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "b" "c";
            "#,
            &vocab,
        )
        .unwrap();
        let placeholder = terminal(&parent, "SUB");
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: placeholder,
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];

        // Explicit segmented StaticParserDwa and Dynamic stay supported and agree.
        let static_owned = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .map(|composition| composition.constraint)
        .expect("explicit StaticParserDwa segmented composition must remain supported");
        let dynamic_owned = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .map(|composition| composition.constraint)
        .expect("explicit Dynamic segmented composition must remain supported");

        let mut static_state = static_owned.start();
        let mut dynamic_state = dynamic_owned.start();
        assert_eq!(static_state.mask(), dynamic_state.mask(), "start masks must agree");
        assert!(static_state.commit_bytes(b"ab"), "static must accept the cross-boundary token");
        assert!(dynamic_state.commit_bytes(b"ab"), "dynamic must accept the cross-boundary token");
        assert_eq!(
            static_state.mask(),
            dynamic_state.mask(),
            "masks after the cross-boundary token must agree",
        );
        assert!(static_state.commit_bytes(b"cd"));
        assert!(dynamic_state.commit_bytes(b"cd"));
        assert!(static_state.is_accepting() && dynamic_state.is_accepting());
    }

    #[test]
    fn prepared_loaded_components_match_fresh_owned_composition() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"<ab>".to_vec()),
            (1, b"<a".to_vec()),
            (2, b"b>".to_vec()),
            (3, b"<".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b">".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "<" SUB ">";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let fresh = parent
            .clone()
            .compose_linked_children_for_test_owned(&[("SUB", &child)], &vocab)
            .unwrap();

        let mut loaded_parent = Constraint::load(parent.save(), &vocab).unwrap();
        let mut loaded_child = Constraint::load(child.save(), &vocab).unwrap();
        loaded_parent.prepare_for_composition_internal(&vocab).unwrap();
        loaded_child.prepare_for_composition_internal(&vocab).unwrap();
        assert!(loaded_parent.parser_dwa.packed().is_some());
        assert!(loaded_child.parser_dwa.packed().is_some());
        assert!(loaded_parent.deferred_composition_metadata_blob.is_none());
        assert!(loaded_child.deferred_composition_metadata_blob.is_none());

        let prepared = loaded_parent
            .compose_linked_children_for_test_owned(&[("SUB", &loaded_child)], &vocab)
            .unwrap();
        assert_constraints_equivalent_on_reachable_prefixes(
            &prepared,
            &fresh,
            &vocab,
            4,
        );
    }

    #[test]
    fn composition_rejects_placeholder_that_matches_a_real_vocab_token() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            // The placeholder token deliberately exists in the model vocab.
            // Composition must still remove its exact-token path.
            (3, b"<placeholder>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 3)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let error = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .expect_err("real vocabulary tokens cannot be used as composition sentinels");
        assert!(error.to_string().contains("non-vocabulary sentinels"));
        assert!(error.to_string().contains("outside the supplied vocabulary"));
    }

    #[test]
    fn repeated_call_sites_match_monolithic_on_all_small_reachable_prefixes() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "<" SUB ">" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" | "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child = "a" | "b";
                nt document = "<" child ">" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"Xaa!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"Xa".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"a!".to_vec()),
            (5, b"X".to_vec()),
            (6, b"a".to_vec()),
            (7, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" child?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child = "a" child?;
                nt document = "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X a!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"a".to_vec()),
            (5, b"!".to_vec()),
            (6, b" a".to_vec()),
            (7, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore WS;
                t WS = " "+;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS = " "+;
                nt document = "X" "a" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let source_oracle = compile_inline_source_oracle(
            r#"
                start document;
                ignore WS;
                t WS = " "+;
                extern grammar SUB;
                nt document = "X" SUB "!";
            "#,
            &[("SUB", "start child; nt child = \"a\";")],
            &vocab,
        );

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &source_oracle,
            &monolithic,
            &vocab,
            5,
            "oracle-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_static,
            &source_oracle,
            &vocab,
            5,
            "static-vs-oracle",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &source_oracle,
            &vocab,
            5,
            "dynamic-vs-oracle",
        );
    }

    #[test]
    fn child_exact_special_token_survives_composition() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"<END>".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_with_token_ids(
            r#"
                start child;
                extern token END;
                nt child = "a" END;
            "#,
            &vocab, &[("END", 3)], &[])
        .unwrap();
        let monolithic = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token END;
                nt document = "X" "a" END "!";
            "#,
            &vocab, &[("END", 3)], &[])
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(composed.save(), &vocab)
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
    fn scoped_visible_terminals_survive_model_token_boundaries() {
        // Entry orientation:
        //   token 0 = "Xa" leaves parent IGNORE="ab" in progress;
        //   token 1 = "bt" completes that IGNORE and enters the child.
        // Exit orientation:
        //   token 3 = "X" enters the child at reset;
        //   token 4 = "ta" completes child T and starts parent IGNORE;
        //   token 5 = "b!" completes IGNORE in the following model token.
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"Xa".to_vec()),
            (1, b"bt".to_vec()),
            (2, b"!".to_vec()),
            (3, b"X".to_vec()),
            (4, b"ta".to_vec()),
            (5, b"b!".to_vec()),
            (6, b"a".to_vec()),
            (7, b"b".to_vec()),
            (8, b"t".to_vec()),
            (9, b"ab".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = "ab";
                t X = "X";
                t BANG = "!";
                extern token SUB;
                nt document = X SUB BANG;
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t CHILD_T = "t";
                nt child = CHILD_T;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = "ab";
                t X = "X";
                t BANG = "!";
                grammar child {
                    start child;
                    t CHILD_T = "t";
                    nt child = CHILD_T;
                };
                nt document = X child BANG;
            "#,
            &vocab,
        )
        .unwrap();

        // Scoped trivia is represented as a real terminal in the composed LR
        // table. Its parser semantics is state-dependent identity (`Skip`), not
        // a lexer-side boundary special case.
        let parent_table = compiler_table_fixture(&parent);
        let child_table = compiler_table_fixture(&child);
        let composed_table = compose_subgrammar_tables(
            &parent_table,
            Some(terminal(&parent, "PARENT_WS")),
            &[SubgrammarTableInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
table: &child_table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.composition_start_nullable().unwrap(),
            }],
        )
        .unwrap();
        let parent_ignore = terminal(&parent, "PARENT_WS");
        assert!(composed_table.table.skip_terminals.contains(&parent_ignore));
        assert!(composed_table.table.action.iter().any(|row| {
            matches!(row.get(&parent_ignore), Some(Action::Skip))
        }));

        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let source_oracle = compile_inline_source_oracle(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = "ab";
                t X = "X";
                t BANG = "!";
                extern grammar SUB;
                nt document = X SUB BANG;
            "#,
            &[(
                "SUB",
                "start child; t CHILD_T = \"t\"; nt child = CHILD_T;",
            )],
            &vocab,
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &source_oracle,
            &monolithic,
            &vocab,
            4,
            "oracle-vs-monolithic",
        );
        let mut prepared_parent = parent.clone();
        let mut prepared_child = child.clone();
        prepared_parent.ensure_composition_reset_tokens_by_terminal();
        prepared_child.ensure_composition_reset_tokens_by_terminal();
        let mut prepared_parent = Constraint::load(prepared_parent.save(), &vocab).unwrap();
        let mut prepared_child = Constraint::load(prepared_child.save(), &vocab).unwrap();
        prepared_parent
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        prepared_child
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            prepared_parent.composition_reset_tokens_by_terminal.len(),
            prepared_parent.tokenizer.num_terminals() as usize,
        );
        assert_eq!(
            prepared_child.composition_reset_tokens_by_terminal.len(),
            prepared_child.tokenizer.num_terminals() as usize,
        );
        let cached_composed = prepared_parent
            .compose_linked_children_for_test(&[("SUB", &prepared_child)], &vocab)
            .unwrap();
        for sequence in [[0u32, 1, 2].as_slice(), [3u32, 4, 5].as_slice()] {
            let mut actual = composed_static.start();
            let mut dynamic = composed_dynamic.start();
            let mut cached = cached_composed.start();
            let mut expected = monolithic.start();
            for &token in sequence {
                let actual_mask = actual.mask();
                let dynamic_mask = dynamic.mask();
                let cached_mask = cached.mask();
                let expected_mask = expected.mask();
                assert_eq!(
                    cached_mask, expected_mask,
                    "prepared-cache mask mismatch before token {token} in sequence {sequence:?}",
                );
                assert_eq!(
                    dynamic_mask, expected_mask,
                    "dynamic mask mismatch before token {token} in sequence {sequence:?}",
                );
                if actual_mask != expected_mask {
                    eprintln!(
                        "IGNORE_FUSION_STATE sequence={sequence:?} before_token={token} actual={:?} expected={:?}",
                        actual.debug_parser_stacks(),
                        expected.debug_parser_stacks(),
                    );
                    let differing = vocab
                        .entries_map()
                        .iter()
                        .filter_map(|(&candidate, bytes)| {
                            (token_allowed(&actual_mask, candidate)
                                != token_allowed(&expected_mask, candidate))
                                .then_some((candidate, bytes.clone(), token_allowed(&actual_mask, candidate), token_allowed(&expected_mask, candidate)))
                        })
                        .collect::<Vec<_>>();
                    eprintln!("IGNORE_FUSION_DIFF {differing:?}");
                }
                assert_eq!(
                    actual_mask,
                    expected_mask,
                    "mask mismatch before token {token} in sequence {sequence:?}",
                );
                assert!(actual.commit_token(token));
                assert!(dynamic.commit_token(token));
                assert!(cached.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert_eq!(actual.mask(), expected.mask());
            assert_eq!(dynamic.mask(), expected.mask());
            assert_eq!(cached.mask(), expected.mask());
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert_eq!(dynamic.is_accepting(), expected.is_accepting());
            assert_eq!(cached.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting(), "sequence {sequence:?} should finish");
        }
    }

    #[test]
    fn child_ignore_terminal_is_scoped_across_fused_boundaries() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X a!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b" ".to_vec()),
            (4, b"a".to_vec()),
            (5, b"!".to_vec()),
            (6, b" a".to_vec()),
            (7, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore WS;
                t WS = " "+;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;

                grammar child {
                    start child;
                    ignore WS;
                    t WS = " "+;
                    nt child = "a";
                };

                nt document = "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let source_oracle = compile_inline_source_oracle(
            r#"
                start document;
                extern grammar SUB;
                nt document = "X" SUB "!";
            "#,
            &[(
                "SUB",
                "start child; ignore WS; t WS = \" \"+; nt child = \"a\";",
            )],
            &vocab,
        );

        // Fused-token integration: the scoped child ignore is transparent to
        // within-token follow pruning, so the single fused model token "X a!"
        // (parent X -> child scoped-ignore WS -> child a -> parent !) must be
        // admitted at start by the static shard, matching the oracle and the
        // dynamic backend.
        assert!(
            token_allowed(&composed_static.start().mask(), 0),
            "static shard must admit the fused parent-X + scoped-child-ignore token"
        );
        assert!(token_allowed(&source_oracle.start().mask(), 0));
        assert!(token_allowed(&composed_dynamic.start().mask(), 0));

        // Inside the child scope the child ignore is allowed, but the child
        // cannot complete the document alone (acceptance stays false until the
        // parent's "!").
        let mut static_inside_child = composed_static.start();
        let mut oracle_inside_child = source_oracle.start();
        assert!(static_inside_child.commit_token(2));
        assert!(oracle_inside_child.commit_token(2));
        assert!(token_allowed(&static_inside_child.mask(), 3));
        assert!(token_allowed(&oracle_inside_child.mask(), 3));
        assert!(!static_inside_child.is_accepting());
        assert!(!oracle_inside_child.is_accepting());

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &source_oracle,
            &monolithic,
            &vocab,
            5,
            "oracle-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_static,
            &source_oracle,
            &vocab,
            5,
            "static-vs-oracle",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &source_oracle,
            &vocab,
            5,
            "dynamic-vs-oracle",
        );

        // Child trivia is not globally active before the parent has entered
        // the child scope.
        assert!(!token_allowed(&composed_static.start().mask(), 3));
        assert!(!token_allowed(&monolithic.start().mask(), 3));
    }

    #[test]
    fn compact_dynamic_scoped_ignore_parser_preserves_both_entry_scopes() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS = "\t"+;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = terminal(&parent, "SUB");
        let x = terminal(&parent, "X");
        let parent_ws = terminal(&parent, "PARENT_WS");
        let child_ws_local = terminal(&child, "CHILD_WS");
        let composed = compose_constraints_owned_parent_segmented(
            parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: slot,
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        assert!(composed.uses_compact_segmented_parser_runtime());
        let overlay = composed.static_dynamic_overlay.as_ref().unwrap();
        let child_ws = overlay.segmented_parser_components[1].terminal_offset + child_ws_local;
        let state = composed.initial_state_map();
        let start = state.values().next().unwrap();
        let after_x = composed
            .advance_compact_segmented_parser(start, x)
            .expect("compact parser runtime");
        let mut scopes = after_x
            .peek_values()
            .into_iter()
            .map(|state| composed.compact_segmented_parser_component(state).unwrap().0)
            .collect::<Vec<_>>();
        scopes.sort_unstable();
        scopes.dedup();
        assert_eq!(scopes, vec![0, 1], "CALL closure must retain parent and child choices");

        let after_parent_ws = composed
            .advance_compact_segmented_parser(&after_x, parent_ws)
            .expect("compact parser runtime");
        assert!(!after_parent_ws.is_empty(), "parent trivia must remain legal before CALL");
        assert!(after_parent_ws.peek_values().into_iter().any(|state| {
            composed
                .compact_segmented_parser_component(state)
                .is_some_and(|(component, _)| component == 1)
        }));

        let after_child_ws = composed
            .advance_compact_segmented_parser(&after_x, child_ws)
            .expect("compact parser runtime");
        assert!(!after_child_ws.is_empty(), "child trivia must be legal after CALL");
        assert!(after_child_ws.peek_values().into_iter().all(|state| {
            composed
                .compact_segmented_parser_component(state)
                .is_some_and(|(component, _)| component == 1)
        }));
    }

    #[test]
    fn distinct_parent_and_child_ignore_terminals_match_inline_scoped_semantics() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS = "\t"+;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;

                grammar child {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS = "\t"+;
                    nt child = "a";
                };

                nt document = "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed_static = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(composed_static.save(), &vocab).unwrap();

        for constraint in [&composed_static, &loaded] {
            assert!(constraint.ignore_terminal.is_none());
        }

        // The inline scoped-ignore lowering materialises nullable skip
        // productions. Its `is_complete()` predicate can report a trivia-only
        // prefix as complete before the visible root has parsed. Compare exact
        // masks/commit language here and assert completion on the complete
        // boundary strings below, rather than preserving that unrelated
        // inline-lowering artefact in the explicit linker. Per-stage mask
        // attribution for this witness lives in `runtime::mask`
        // (`GLRMASK_DEBUG_MASK_STAGES`).
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_static,
            &monolithic,
            &vocab,
            5,
            "static-vs-monolithic",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            5,
            "dynamic-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed_static,
            &vocab,
            5,
            "loaded-vs-static",
        );

        for (token, expected) in [(0, true), (1, true), (9, false), (10, true), (11, false), (12, false)] {
            let mut actual = composed_static.start();
            let mut dynamic = composed_dynamic.start();
            let mut loaded_state = loaded.start();
            let mut reference = monolithic.start();
            assert_eq!(actual.commit_token(token), expected, "token {token}");
            assert_eq!(dynamic.commit_token(token), expected, "dynamic token {token}");
            assert_eq!(
                loaded_state.commit_token(token),
                expected,
                "loaded token {token}",
            );
            assert_eq!(reference.commit_token(token), expected, "reference token {token}");
            assert_eq!(actual.is_accepting(), expected, "token {token}");
            assert_eq!(dynamic.is_accepting(), expected, "dynamic token {token}");
            assert_eq!(loaded_state.is_accepting(), expected, "loaded token {token}");
            assert_eq!(reference.is_accepting(), expected, "reference token {token}");
        }
    }

    // Keep large by-value construction temporaries out of the test runner's
    // frame. No stack-size override, new thread, or reduced oracle coverage.
    #[inline(never)]
    fn scoped_oracle_test_source(
        grammar: &str,
        token_ids: &[(&str, u32)],
        vocab: &Vocab,
    ) -> Box<Constraint> {
        Box::new(Constraint::from_glrm_with_token_ids(grammar, vocab, token_ids, &[]).unwrap())
    }

    #[inline(never)]
    fn scoped_oracle_test_link(
        parent: &Constraint, slot: &str, child: &Constraint, vocab: &Vocab, dynamic: bool,
    ) -> Box<Constraint> {
        let composition = if dynamic {
            parent.compose_linked_children_for_test_dynamic(&[(slot, child)], vocab)
        } else {
            parent.compose_linked_children_for_test(&[(slot, child)], vocab)
        };
        Box::new(composition.unwrap())
    }

    #[inline(never)]
    fn scoped_oracle_test_reload(source: &Constraint) -> Box<Constraint> {
        Box::new(Constraint::load(source.save(), &crate::public_api::constraint_vocab(source)).unwrap())
    }

    #[inline(never)]
    fn scoped_oracle_test_start(source: &Constraint) -> Box<crate::ConstraintState> {
        Box::new(source.start())
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~21s); run with --features slow-tests"
    )]
    fn scoped_ignore_oracle_survives_reload_and_nested_recomposition() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"<".to_vec()),
            (1, b">".to_vec()),
            (2, b"X".to_vec()),
            (3, b"!".to_vec()),
            (4, b" ".to_vec()),
            (5, b"\t".to_vec()),
            (6, b"a".to_vec()),
            (7, b"\t\t".to_vec()),
            (8, b"\ta".to_vec()),
            (9, b"a\t".to_vec()),
            (10, b"X\t".to_vec()),
            (11, b"a ".to_vec()),
            (12, b"a\t ".to_vec()),
            (13, b"a \t".to_vec()),
            (14, b"<X\t".to_vec()),
            (15, b"!>".to_vec()),
            (16, b"X\ta\t !".to_vec()),
            (17, b"<X\ta\t !>".to_vec()),
            (18, b"X \ta!".to_vec()),
            (19, b"X\t a!".to_vec()),
            (20, b"Xa\t !".to_vec()),
            (21, b"Xa \t!".to_vec()),
            (22, b"X a!".to_vec()),
            (23, b"<X\t a!>".to_vec()),
        ]);
        let parent = scoped_oracle_test_source(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern token SUB;
                nt document = "X" SUB "!";
            "#, &[("SUB", 999)],
            &vocab);
        let child = scoped_oracle_test_source(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS = "\t"+;
                nt child = "a";
            "#, &[],
            &vocab);
        let monolithic = scoped_oracle_test_source(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                grammar child {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS = "\t"+;
                    nt child = "a";
                };
                nt document = "X" child "!";
            "#, &[],
            &vocab);
        let composed = scoped_oracle_test_link(&parent, "SUB", &child, &vocab, false);
        let composed_dynamic = scoped_oracle_test_link(&parent, "SUB", &child, &vocab, true);
        let loaded = scoped_oracle_test_reload(&composed);
        let make_states = || {
            vec![
                composed.start(),
                composed_dynamic.start(),
                loaded.start(),
                monolithic.start(),
            ]
        };

        // Exhaust the small reachable token-prefix graph.  We compare masks
        // rather than the inline lowering's trivia-only completion artifact;
        // successful complete strings are checked explicitly below.
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "static-vs-monolithic",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "dynamic-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "loaded-vs-static",
        );

        // Cover each scoped-ignore boundary shape explicitly:
        // - parent ignore before child;
        // - child ignore as the first child token;
        // - repeated / ignore-only child tokens;
        // - ignore+real and real+ignore fused model tokens;
        // - child return followed by parent ignore;
        // - entry/exit fusion in one model token.
        let valid_sequences: &[&[u32]] = &[
            &[2, 4, 5, 6, 3],
            &[2, 5, 6, 3],
            &[2, 7, 6, 3],
            &[2, 8, 3],
            &[2, 9, 3],
            &[2, 11, 3],
            &[2, 12, 3],
            &[10, 6, 3],
            &[16],
            &[18],
            &[20],
            &[22],
        ];
        for &sequence in valid_sequences {
            let mut actual = scoped_oracle_test_start(&composed);
            let mut dynamic = scoped_oracle_test_start(&composed_dynamic);
            let mut restored = scoped_oracle_test_start(&loaded);
            let mut expected = scoped_oracle_test_start(&monolithic);
            for &token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask before {sequence:?} token {token}");
                assert_eq!(dynamic.mask(), expected.mask(), "dynamic mask before {sequence:?} token {token}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask before {sequence:?} token {token}");
                assert!(actual.commit_token(token));
                assert!(dynamic.commit_token(token));
                assert!(restored.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert!(actual.is_accepting(), "composed incomplete for {sequence:?}");
            assert!(dynamic.is_accepting(), "dynamic composed incomplete for {sequence:?}");
            assert!(restored.is_accepting(), "loaded incomplete for {sequence:?}");
            assert!(expected.is_accepting(), "reference incomplete for {sequence:?}");
        }

        // Parent trivia must not become active while the child scope is still
        // parsing, and child trivia must not leak back into the parent after
        // the child has returned.
        for sequence in [&[2u32, 13][..], &[19][..], &[21][..]] {
            let mut actual = scoped_oracle_test_start(&composed);
            let mut dynamic = scoped_oracle_test_start(&composed_dynamic);
            let mut restored = scoped_oracle_test_start(&loaded);
            let mut expected = scoped_oracle_test_start(&monolithic);
            for &token in &sequence[..sequence.len() - 1] {
                assert!(actual.commit_token(token));
                assert!(dynamic.commit_token(token));
                assert!(restored.commit_token(token));
                assert!(expected.commit_token(token));
            }
            let token = *sequence.last().unwrap();
            let expected_result = expected.commit_token(token);
            assert_eq!(actual.commit_token(token), expected_result);
            assert_eq!(dynamic.commit_token(token), expected_result);
            assert_eq!(restored.commit_token(token), expected_result);
            assert!(!expected_result, "invalid scoped sequence {sequence:?} was accepted");
        }

        // Reuse the serialized, already-composed constraint as a child again.
        // This is the inherited-skip case: CHILD_WS is no longer a top-level
        // ignore of `loaded`, but its scoped phase behavior must survive the
        // next call/return boundary exactly.
        let outer_parent = scoped_oracle_test_source(
            r#"
                start outer;
                extern token INNER;
                nt outer = "<" INNER ">";
            "#, &[("INNER", 1000)],
            &vocab);
        let outer = scoped_oracle_test_link(&outer_parent, "INNER", &loaded, &vocab, false);
        let outer_dynamic = scoped_oracle_test_link(&outer_parent, "INNER", &loaded, &vocab, true);
        let outer_monolithic = scoped_oracle_test_source(
            r#"
                start outer;
                grammar inner {
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS = " "+;
                    grammar child {
                        start child;
                        ignore CHILD_WS;
                        t CHILD_WS = "\t"+;
                        nt child = "a";
                    };
                    nt document = "X" child "!";
                };
                nt outer = "<" inner ">";
            "#, &[],
            &vocab);

        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &outer,
            &outer_monolithic,
            &vocab,
            4,
            "outer-static-vs-outer-monolithic",
        );
        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &outer_dynamic,
            &outer_monolithic,
            &vocab,
            4,
            "outer-dynamic-vs-outer-monolithic",
        );
        for sequence in [&[0u32, 16, 1][..], &[14, 6, 15][..], &[17][..]] {
            let mut actual = scoped_oracle_test_start(&outer);
            let mut dynamic = scoped_oracle_test_start(&outer_dynamic);
            let mut expected = scoped_oracle_test_start(&outer_monolithic);
            for &token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "outer mask before {sequence:?} token {token}");
                assert_eq!(dynamic.mask(), expected.mask(), "outer dynamic mask before {sequence:?} token {token}");
                assert!(actual.commit_token(token));
                assert!(dynamic.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert!(actual.is_accepting(), "outer incomplete for {sequence:?}");
            assert!(dynamic.is_accepting(), "outer dynamic incomplete for {sequence:?}");
            assert!(expected.is_accepting(), "outer reference incomplete for {sequence:?}");
        }

        let mut actual = scoped_oracle_test_start(&outer);
        let mut dynamic = scoped_oracle_test_start(&outer_dynamic);
        let mut expected = scoped_oracle_test_start(&outer_monolithic);
        assert_eq!(actual.commit_token(23), expected.commit_token(23));
        assert_eq!(dynamic.commit_token(23), expected.commit_token(23));
        assert!(!expected.is_accepting());
    }

    /// Replay a token sequence through one route, returning the index of the
    /// first rejected token (`None` if the whole sequence commits).
    fn replay_first_rejection(
        commit: &mut dyn FnMut(u32) -> bool,
        sequence: &[u32],
    ) -> Option<usize> {
        sequence.iter().position(|&token| !commit(token))
    }

    #[test]
    fn integration_scoped_ignore_follow_boundary_masks() {
        // Parent `X SUB !` with a parent-space ignore; child `a` then `b`
        // with a child-tab ignore. Completion-aware mask parity (depth 4)
        // across static/dynamic/loaded/inline; explicit valid completions;
        // rejections with first-rejection-index agreement; save/load parity;
        // real StaticParser shards installed.
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"!".to_vec()),
            (2, b"a".to_vec()),
            (3, b"b".to_vec()),
            (4, b"c".to_vec()),
            (5, b" ".to_vec()),
            (6, b"\t".to_vec()),
            (7, b"X\ta".to_vec()),
            (8, b"X\tab!".to_vec()),
            (9, b"a\tb".to_vec()),
            (10, b"b\t !".to_vec()),
            (11, b"X\tb!".to_vec()),
            (12, b"Xa\tc!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS = "\t"+;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                grammar child {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS = "\t"+;
                    nt child = "a" "b";
                };
                nt document = "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(composed.save(), &vocab).unwrap();
        let make_states = || {
            vec![
                composed.start(),
                composed_dynamic.start(),
                loaded.start(),
                monolithic.start(),
            ]
        };
        // Real StaticParser shards installed (not fallback-only).
        let overlay = composed
            .static_dynamic_overlay
            .as_ref()
            .expect("static_dynamic_overlay must be present");
        assert!(
            !overlay.segmented_parser_components.is_empty(),
            "static link must install parser components",
        );
        for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
            match &component.boundary.as_ref().expect("component must have a boundary shard").backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_) => {}
                other => panic!("component {index} must be StaticParser, got {other:?}"),
            }
        }
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "integration-ignore static-vs-inline",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "integration-ignore dynamic-vs-inline",
        );
        // Valid strings complete on all four builds (incl. ignore ownership:
        // [0,5,6,2,6,3,6,5,1] spaces the parent parts, tabs the child parts;
        // [0,2,10] fuses child-b + parent-ignore + `!`).
        for sequence in [
            &[0u32, 2, 3, 1][..],
            &[7, 3, 1][..],
            &[0, 9, 1][..],
            &[8][..],
            &[0, 5, 6, 2, 6, 3, 6, 5, 1][..],
            &[0, 2, 10][..],
        ] {
            let mut states = make_states();
            for &token in sequence {
                let masks: Vec<Vec<u32>> =
                    states.iter().map(|state| state.mask()).collect();
                assert_eq!(masks[0], masks[3], "static mask before {sequence:?} token {token}");
                assert_eq!(masks[1], masks[3], "dynamic mask before {sequence:?} token {token}");
                assert_eq!(masks[2], masks[3], "loaded mask before {sequence:?} token {token}");
                for state in states.iter_mut() {
                    assert!(state.commit_token(token));
                }
            }
            for (label, state) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(states.iter())
            {
                assert!(state.is_accepting(), "{label} incomplete for {sequence:?}");
            }
        }
        // Invalid: missing `a` ([0,1]), unexpected `c` ([12]), wrong-scope
        // space inside the child ([0,2,5,3,1]), stray fused token ([11]).
        // Every route must reject at least one token; first-rejection index
        // must agree across routes.
        for sequence in [&[0u32, 1][..], &[0, 2, 4, 1][..], &[12][..], &[11][..], &[0, 2, 5, 3, 1][..]] {
            let mut rejects = Vec::new();
            for mut state in make_states() {
                rejects.push(replay_first_rejection(
                    &mut |token| state.commit_token(token),
                    sequence,
                ));
            }
            for (label, rejection) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(rejects.iter())
            {
                assert!(
                    rejection.is_some(),
                    "{label} accepted invalid scoped sequence {sequence:?}",
                );
            }
            assert_eq!(rejects[0], rejects[3], "static/inline rejection index on {sequence:?}");
            assert_eq!(rejects[1], rejects[3], "dynamic/inline rejection index on {sequence:?}");
            assert_eq!(rejects[2], rejects[3], "loaded/inline rejection index on {sequence:?}");
        }
        // Save/load preserves masks + completion.
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "integration-ignore loaded-vs-static",
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow (~22s); run with --features slow-tests"
    )]
    fn integration_finite_exclusion_boundary_masks() {
        // Parent `X SUB !` (no ignores); child `OPEN = [ab]+ - "ab"`.
        // Tests exclusion-operator composition compatibility (static vs
        // dynamic vs inline, save/load). NOT a certificate/master-activation
        // proof: the tiny fixture does not trigger dynamic-virtual
        // optimization (see the dedicated lexer certificate unit test).
        // Prefix `Xab` stays extendible; only completed `Xab!` is forbidden.
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"!".to_vec()),
            (2, b"a".to_vec()),
            (3, b"b".to_vec()),
            (4, b"Xa".to_vec()),
            (5, b"Xb".to_vec()),
            (6, b"Xab".to_vec()),
            (7, b"Xaa".to_vec()),
            (8, b"ab".to_vec()),
            (9, b"aa".to_vec()),
            (10, b"ab!".to_vec()),
            (11, b"aa!".to_vec()),
            (12, b"Xa!".to_vec()),
            (13, b"Xb!".to_vec()),
            (14, b"Xab!".to_vec()),
            (15, b"Xaa!".to_vec()),
            (16, b"Xaba!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t OPEN = [ab]+ - "ab";
                nt child = OPEN;
            "#,
            &vocab,
        )
        .unwrap();
        // The child lowers `OPEN` to an exclusion operator (AST-level proof
        // the sidecar-eligible shape reaches composition; independent of any
        // tokenizer certificate location).
        let open_terminal = child
            .terminal_display_names()
            .iter()
            .position(|name| name == "OPEN")
            .expect("child must have OPEN terminal") as u32;
        assert!(
            matches!(
                child.retained_terminal_expr(open_terminal),
                Some(Expr::Exclude { .. })
            ),
            "child OPEN terminal must lower to Expr::Exclude",
        );
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                grammar child {
                    start child;
                    t OPEN = [ab]+ - "ab";
                    nt child = OPEN;
                };
                nt document = "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(composed.save(), &vocab).unwrap();
        // Keep the large ConstraintState objects off this test function's
        // stack. Several four-state inline arrays were enough to exceed the
        // default Rust test-thread stack on macOS before the first statement
        // executed, despite the runtime code itself being fine.
        let make_states = || {
            vec![
                composed.start(),
                composed_dynamic.start(),
                loaded.start(),
                monolithic.start(),
            ]
        };
        // Real StaticParser shards installed (not fallback-only).
        let overlay = composed
            .static_dynamic_overlay
            .as_ref()
            .expect("static_dynamic_overlay must be present");
        assert!(
            !overlay.segmented_parser_components.is_empty(),
            "static link must install parser components",
        );
        for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
            match &component.boundary.as_ref().expect("component must have a boundary shard").backend {
                crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_) => {}
                other => panic!("component {index} must be StaticParser, got {other:?}"),
            }
        }
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &composed_dynamic,
            &vocab,
            4,
            "integration-exclusion static-vs-dynamic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "integration-exclusion static-vs-inline",
        );
        // Valid completions, incl. legitimate Xabab!/Xaab! and char-by-char.
        for sequence in [
            &[12u32][..],
            &[13][..],
            &[15][..],
            &[16][..],
            &[0, 2, 1][..],
            &[0, 3, 1][..],
            &[0, 2, 2, 1][..],
            &[0, 2, 3, 2, 1][..],
            &[6, 8, 1][..],
            &[4, 8, 1][..],
            &[4, 2, 1][..],
            &[4, 9, 1][..],
        ] {
            let mut states = make_states();
            for &token in sequence {
                let masks: Vec<Vec<u32>> =
                    states.iter().map(|state| state.mask()).collect();
                assert_eq!(masks[0], masks[1], "static/dynamic mask before {sequence:?} token {token}");
                assert_eq!(masks[0], masks[3], "static/inline mask before {sequence:?} token {token}");
                assert_eq!(masks[2], masks[3], "loaded/inline mask before {sequence:?} token {token}");
                for state in states.iter_mut() {
                    assert!(state.commit_token(token));
                }
            }
            for (label, state) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(states.iter())
            {
                assert!(state.is_accepting(), "{label} incomplete for {sequence:?}");
            }
        }
        // `Xab` remains extendible on every route: commit 6, then 2, then 1,
        // and assert acceptance (Xaba! is valid).
        for mut state in make_states() {
            for &token in &[6u32, 2, 1] {
                assert!(state.commit_token(token));
            }
            assert!(state.is_accepting(), "Xaba! must complete");
        }
        // Rejected: completed `Xab!` only.
        for sequence in [&[14u32][..], &[0, 2, 3, 1][..], &[6, 1][..], &[4, 3, 1][..], &[0, 8, 1][..]] {
            let mut rejects = Vec::new();
            for mut state in make_states() {
                rejects.push(replay_first_rejection(
                    &mut |token| state.commit_token(token),
                    sequence,
                ));
            }
            for (label, rejection) in
                ["static", "dynamic", "loaded", "inline"].iter().zip(rejects.iter())
            {
                assert!(
                    rejection.is_some(),
                    "{label} accepted forbidden exclusion sequence {sequence:?}",
                );
            }
            assert_eq!(rejects[0], rejects[3], "static/inline rejection index on {sequence:?}");
            assert_eq!(rejects[1], rejects[3], "dynamic/inline rejection index on {sequence:?}");
            assert_eq!(rejects[2], rejects[3], "loaded/inline rejection index on {sequence:?}");
        }
        // Save/load preserves masks + completion.
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "integration-exclusion loaded-vs-static",
        );
    }

    #[test]
    fn adjacent_precomposed_child_can_begin_with_child_only_token() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern token LEFT;
                extern token RIGHT;
                nt document = "X" LEFT RIGHT "!";
            "#,
            &vocab, &[("LEFT", 998), ("RIGHT", 999)], &[])
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start left;
                ignore LEFT_WS;
                t LEFT_WS = "\t"+;
                nt left = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start right;
                ignore RIGHT_WS;
                t RIGHT_WS = "\n"+;
                nt right = "b";
            "#,
            &vocab,
        )
        .unwrap();

        // Compose RIGHT first.  When LEFT is linked later, LEFT's parent
        // continuation begins by traversing RIGHT's already-existing control
        // edge.  The direct continuation row therefore exposes a control
        // terminal, not lexical terminal "b".
        let parent_with_right = parent
            .compose_linked_children_for_test(&[("RIGHT", &right)], &vocab)
            .unwrap();
        let composed = parent_with_right
            .compose_linked_children_for_test(&[("LEFT", &left)], &vocab)
            .unwrap();
        let parent_with_right_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("RIGHT", &right)], &vocab)
            .unwrap();
        let composed_dynamic = parent_with_right_dynamic
            .compose_linked_children_for_test_dynamic(&[("LEFT", &left)], &vocab)
            .unwrap();

        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;

                grammar left {
                    start left;
                    ignore LEFT_WS;
                    t LEFT_WS = "\t"+;
                    nt left = "a";
                };
                grammar right {
                    start right;
                    ignore RIGHT_WS;
                    t RIGHT_WS = "\n"+;
                    nt right = "b";
                };

                nt document = "X" left right "!";
            "#,
            &vocab,
        )
        .unwrap();

        let mut actual = composed.start();
        let mut reference = monolithic.start();
        assert!(actual.commit_token(0));
        assert!(reference.commit_token(0));
        assert!(token_allowed(&actual.mask(), 1));
        assert!(token_allowed(&reference.mask(), 1));
        assert!(actual.commit_token(1));
        assert!(reference.commit_token(1));
        assert!(actual.commit_token(2));
        assert!(reference.commit_token(2));
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        // Leading RIGHT trivia also begins only after return-from-LEFT followed
        // by enter-RIGHT, and must preserve that scope across the token.
        let mut actual = composed.start();
        let mut reference = monolithic.start();
        assert!(actual.commit_token(0));
        assert!(reference.commit_token(0));
        assert!(token_allowed(&actual.mask(), 3));
        assert!(token_allowed(&reference.mask(), 3));
        assert!(actual.commit_token(3));
        assert!(reference.commit_token(3));
        assert!(actual.commit_token(2));
        assert!(reference.commit_token(2));
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        // A multi-byte token containing only RIGHT trivia must itself be a
        // boundary-begin path: return from LEFT, enter RIGHT, consume trivia,
        // and persist the RIGHT start state for the next model token. This is
        // not an owner-crossing token and is not covered by one-byte seed
        // relations.
        let mut actual = composed.start();
        let mut reference = monolithic.start();
        assert!(actual.commit_token(0));
        assert!(reference.commit_token(0));
        assert!(token_allowed(&actual.mask(), 10));
        assert!(token_allowed(&reference.mask(), 10));
        assert!(actual.commit_token(10));
        assert!(reference.commit_token(10));
        assert!(actual.commit_token(1));
        assert!(reference.commit_token(1));
        assert!(actual.commit_token(2));
        assert!(reference.commit_token(2));
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        assert_constraints_mask_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "dynamic-vs-monolithic",
        );
    }

    #[test]
    fn adjacent_calls_to_same_child_accept_child_only_second_token() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"Xa".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xaa!".to_vec()),
            (4, b"X".to_vec()),
            (5, b"a!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child = "a";
                nt document = "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();

        let mut actual = composed.start();
        let mut reference = monolithic.start();
        assert!(actual.commit_token(0));
        assert!(reference.commit_token(0));
        assert!(token_allowed(&actual.mask(), 1));
        assert!(token_allowed(&reference.mask(), 1));
        assert!(actual.commit_token(1));
        assert!(reference.commit_token(1));
        assert!(actual.commit_token(2));
        assert!(reference.commit_token(2));
        assert!(actual.is_accepting());
        assert!(reference.is_accepting());

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "polluter-static-vs-monolithic",
        );
    }

    #[test]
    fn adjacent_children_with_uniform_ignore_compile_controls_and_keep_global_ignore() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore WS;
                t WS = " "+;
                extern token SUB;
                nt document = "X" SUB SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore WS;
                t WS = " "+;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore WS;
                t WS = " "+;
                nt child = "a";
                nt document = "X" child child "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let composed_dynamic = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(composed.save(), &vocab).unwrap();

        for constraint in [&composed, &loaded] {
            assert!(!constraint.table.is_present());
            let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
            assert!(overlay.segmented_parser_components.iter().all(|component|
                component.constraint.ignore_terminal.is_some()));
            assert!(constraint.parser_control_terminals().is_empty());
        }

        for sequence in [vec![0, 1, 2], vec![3], vec![4, 5, 6, 5, 6, 5, 7, 5]] {
            let mut actual = composed.start();
            let mut dynamic = composed_dynamic.start();
            let mut restored = loaded.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                assert_eq!(dynamic.mask(), expected.mask(), "dynamic mask mismatch before token {token}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask mismatch before token {token}");
                assert!(actual.commit_token(token));
                assert!(dynamic.commit_token(token));
                assert!(restored.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert!(actual.is_accepting());
            assert!(dynamic.is_accepting());
            assert!(restored.is_accepting());
            assert!(expected.is_accepting());
        }

        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "static-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "dynamic-vs-monolithic",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &loaded,
            &composed,
            &vocab,
            4,
            "loaded-vs-static",
        );

        // The compiled child no longer carries runtime controls. Reusing it in
        // an outer composition must retain the same globally erased ignore and
        // compile the new call/return controls away again.
        let outer_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start outer;
                ignore WS;
                t WS = " "+;
                extern token INNER;
                nt outer = "<" INNER ">";
            "#,
            &vocab, &[("INNER", 1000)], &[])
        .unwrap();
        let outer_monolithic = Constraint::from_glrm_grammar(
            r#"
                start outer;
                ignore WS;
                t WS = " "+;
                nt child = "a";
                nt outer = "<" "X" child child "!" ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .clone()
            .compose_linked_children_for_test(&[("INNER", &composed)], &vocab)
            .unwrap();
        let outer_from_loaded = outer_parent
            .compose_linked_children_for_test(&[("INNER", &loaded)], &vocab)
            .unwrap();
        for constraint in [&outer, &outer_from_loaded] {
            assert!(!constraint.table.is_present());
            assert!(constraint.parser_control_terminals().is_empty());
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
                assert!(actual.commit_token(token));
                assert!(restored_actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert!(actual.is_accepting());
            assert!(restored_actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn nonterminal_mediated_adjacent_calls_use_explicit_linker() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"Xa".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xaa!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt second = SUB;
                nt document = "X" SUB second "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child = "a";
                nt second = child;
                nt document = "X" child second "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token LEFT;
                extern token RIGHT;
                nt document = "<" LEFT ">,<" RIGHT ">";
            "#,
            &vocab, &[("LEFT", 998), ("RIGHT", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" | "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt child = "a" | "b";
                nt document = "<" child ">,<" child ">";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("LEFT", &child), ("RIGHT", &child)], &vocab)
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
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token FIRST;
                extern token SECOND;
                extern token THIRD;
                nt document = "[" FIRST "|" SECOND "|" THIRD "]";
            "#,
            &vocab, &[("FIRST", 997), ("SECOND", 998), ("THIRD", 999)], &[])
        .unwrap();
        let first = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let second = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "b";
            "#,
            &vocab,
        )
        .unwrap();
        let third = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "c";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt first = "a";
                nt second = "b";
                nt third = "c";
                nt document = "[" first "|" second "|" third "]";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "FIRST"),
                    additional_placeholder_terminals: &[],
                    constraint: &first,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "SECOND"),
                    additional_placeholder_terminals: &[],
                    constraint: &second,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "THIRD"),
                    additional_placeholder_terminals: &[],
                    constraint: &third,
                },
            ],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;

        let mut composed_bytes = composed.start();
        assert!(composed_bytes.commit_bytes(b"[a|b|c]"));
        assert!(composed_bytes.is_accepting());
        let mut monolithic_bytes = monolithic.start();
        assert!(monolithic_bytes.commit_bytes(b"[a|b|c]"));
        assert!(monolithic_bytes.is_accepting());

        assert_constraints_equivalent_on_reachable_prefixes(
            &composed,
            &monolithic,
            &vocab,
            7,
        );
        let mut fused = composed.start();
        assert!(fused.commit_token(0));
        assert!(fused.is_accepting());
    }

    #[test]
    fn composition_preserves_out_of_vocab_end_tokens() {
        const PLACEHOLDER_TOKEN: u32 = 999;
        const END_TOKEN: u32 = 1000;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"Xa!".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
            (3, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)],
            &[END_TOKEN])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start document;
                nt child = "a";
                nt document = "X" child "!";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        assert_eq!(composed.embedded_end_token_ids(), vec![END_TOKEN]);

        for content in [vec![0], vec![1, 2, 3]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            assert!(!token_allowed(&actual.mask(), END_TOKEN));
            assert!(!token_allowed(&actual.mask(), PLACEHOLDER_TOKEN));
            for token in content {
                assert_eq!(actual.mask(), expected.mask());
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert_eq!(actual.mask(), expected.mask());
            assert!(token_allowed(&actual.mask(), END_TOKEN));
            assert!(!token_allowed(&actual.mask(), PLACEHOLDER_TOKEN));
            assert_eq!(actual.forced_canonical_reference(), vec![END_TOKEN]);
            assert!(actual.commit_token(END_TOKEN));
            assert!(expected.commit_token(END_TOKEN));
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }

        let loaded = Constraint::load(composed.save(), &vocab).unwrap();
        assert_eq!(loaded.embedded_end_token_ids(), vec![END_TOKEN]);
        let mut loaded_state = loaded.start();
        assert!(loaded_state.commit_token(0));
        assert_eq!(loaded_state.forced_canonical_reference(), vec![END_TOKEN]);
        assert!(loaded_state.commit_token(END_TOKEN));
        assert!(loaded_state.is_accepting());
    }

    #[test]

    fn composition_compiles_named_special_continuation_into_static_mask() {
        const SPECIAL_TOKEN: u32 = 1000;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"Xa".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token DONE;
                nt document = "X" SUB DONE;
            "#,
            &vocab, &[("SUB", 999), ("DONE", 1000)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token DONE;
                nt document = "X" "a" DONE;
            "#,
            &vocab, &[("DONE", 1000)], &[])
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.parser_control_terminals().is_empty());
        for content in [vec![0], vec![1, 2]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            for token in content {
                assert_eq!(actual.mask(), expected.mask());
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert_eq!(actual.mask(), expected.mask());
            assert_eq!(actual.forced_canonical_reference(), vec![SPECIAL_TOKEN]);
            assert!(actual.commit_token(SPECIAL_TOKEN));
            assert!(expected.commit_token(SPECIAL_TOKEN));
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn nullable_child_to_named_special_is_masked() {
        const SPECIAL_TOKEN: u32 = 1000;
        let vocab = Vocab::from_entries_unchecked(vec![(0, b"a".to_vec())]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token DONE;
                nt document = SUB DONE;
            "#,
            &vocab, &[("SUB", 999), ("DONE", 1000)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item = "a";
                nt child = item?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token DONE;
                nt item = "a";
                nt document = item? DONE;
            "#,
            &vocab, &[("DONE", 1000)], &[])
        .unwrap();
        // Nullable bound children are outside the signed-transfer static
        // linker's supported class (unbounded silent Entry/Return episodes);
        // static links decline them loudly, so this differential runs on the
        // exact dynamic backend.
        let composed = parent
            .compose_linked_children_for_test_dynamic(&[("SUB", &child)], &vocab)
            .unwrap();

        assert!(composed.parser_control_terminals().is_empty());
        for sequence in [vec![SPECIAL_TOKEN], vec![0, SPECIAL_TOKEN]] {
            let mut actual = composed.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask());
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]

    fn composition_rejects_placeholder_token_id_reused_by_live_end_token() {
        const SHARED_TOKEN: u32 = 999;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"Xa!".to_vec()),
            (1, b"X".to_vec()),
            (2, b"a".to_vec()),
            (3, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)],
            &[SHARED_TOKEN])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();

        let error = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"Xa".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar_with_end_tokens(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
            &[END_TOKEN],
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token END;
                nt document = "X" "a" END "!";
            "#,
            &vocab, &[("END", 100)], &[])
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        assert_eq!(composed.embedded_end_token_ids(), vec![END_TOKEN]);
        let loaded = Constraint::load(composed.save(), &vocab).unwrap();
        assert_eq!(loaded.embedded_end_token_ids(), vec![END_TOKEN]);

        for sequence in [vec![3, END_TOKEN, 2], vec![0, 1, END_TOKEN, 2]] {
            let mut actual = loaded.start();
            let mut expected = monolithic.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask());
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }

        let outer = Constraint::from_glrm_with_token_ids(
            r#"
                start outer;
                extern token INNER;
                nt outer = INNER;
            "#,
            &vocab, &[("INNER", 100)], &[])
        .unwrap();
        let error = outer
            .compose_linked_children_for_test(&[("INNER", &loaded)], &vocab)
            .expect_err("nested placeholder must not reuse the child's end-token ID");
        assert!(
            error
                .to_string()
                .contains("also configured as a grammar-level end token")
        );
    }

    #[test]
    fn substitution_can_make_the_composed_start_nullable_for_later_embedding() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let nullable_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start middle;
                extern token CHILD;
                nt middle = CHILD;
            "#,
            &vocab, &[("CHILD", 998)], &[])
        .unwrap();
        assert!(!nullable_parent.composition_start_nullable().unwrap());
        let nullable_child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item = "a";
                nt child = item?;
            "#,
            &vocab,
        )
        .unwrap();
        // Nullable bound children are outside the signed-transfer static
        // linker's supported class; compose the nullable middle dynamically.
        // Table nullability metadata is backend-independent.
        let middle = nullable_parent
            .compose_linked_children_for_test_dynamic(&[("CHILD", &nullable_child)], &vocab)
            .unwrap();
        assert!(middle.composition_start_nullable().unwrap());
        let middle = Constraint::load(middle.save(), &vocab).unwrap();
        assert!(middle.composition_start_nullable().unwrap());

        let outer_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token MIDDLE;
                nt document = "X" MIDDLE "!";
            "#,
            &vocab, &[("MIDDLE", 999)], &[])
        .unwrap();
        let composed = outer_parent
            .compose_linked_children_for_test_dynamic(&[("MIDDLE", &middle)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item = "a";
                nt document = "X" item? "!";
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "<" SUB ">" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                grammar inner {
                    start child;
                    nt child = "a" "b";
                };
                nt document = "<" inner ">" inner "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
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
            assert!(actual.commit_bytes(&[byte]));
            assert!(expected.commit_bytes(&[byte]));
        }
        assert_eq!(actual.mask(), expected.mask(), "final mask mismatch");
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());

        for bytes in [
            b"<ab>ab!".as_slice(),
            b"<a>ab!".as_slice(),
            b"<ab>a!".as_slice(),
            b"<ab>ab".as_slice(),
            b"ab>ab!".as_slice(),
            b"<ab><ab>!".as_slice(),
        ] {
            let mut expected = monolithic.start();
            let expected_accepts = expected.commit_bytes(bytes) && expected.is_accepting();
            let mut actual = composed.start();
            let actual_accepts = actual.commit_bytes(bytes) && actual.is_accepting();
            assert_eq!(actual_accepts, expected_accepts, "language mismatch for {bytes:?}");
        }
        assert_accepts(&composed, valid);
        assert_rejects(&composed, b"<ab>a!");
    }

    #[test]
    fn composed_constraint_reuses_one_child_across_distinct_placeholders() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token LEFT;
                extern token RIGHT;
                nt document = "<" LEFT ">" RIGHT "!";
            "#,
            &vocab, &[("LEFT", 998), ("RIGHT", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                grammar inner {
                    start child;
                    nt child = "a" "b";
                };
                nt document = "<" inner ">" inner "!";
            "#,
            &vocab,
        )
        .unwrap();
        let left = terminal(&parent, "LEFT");
        let right = terminal(&parent, "RIGHT");
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: left,
                additional_placeholder_terminals: &[right],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;
        let segmented = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: left,
                additional_placeholder_terminals: &[right],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;

        let overlay = segmented
            .static_dynamic_overlay
            .as_ref()
            .expect("segmented composition should retain component wrapper metadata");
        assert_eq!(overlay.segmented_parser_links.len(), 2);
        assert_eq!(
            overlay
                .segmented_parser_links
                .iter()
                .map(|link| link.slot_terminal)
                .collect::<Vec<_>>(),
            vec![left, right],
        );
        for link in &overlay.segmented_parser_links {
            assert_eq!(link.parent_component, 0);
            assert_eq!(link.child_component, 1);
            assert_eq!(link.child_start, 0);
            assert!(!link.child_start_nullable);
            assert!(matches!(link.return_pop, 1 | 2));
        }

        let valid = b"<ab>ab!";
        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for (offset, &byte) in valid.iter().enumerate() {
            assert_eq!(
                actual.mask(),
                expected.mask(),
                "mask mismatch before offset {offset}, byte {byte:?}",
            );
            assert!(actual.commit_bytes(&[byte]));
            assert!(expected.commit_bytes(&[byte]));
        }
        assert_eq!(actual.mask(), expected.mask(), "final mask mismatch");
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
        assert_accepts(&composed, valid);
        assert_rejects(&composed, b"<ab>a!");

        assert!(
            segmented.uses_compact_segmented_parser_runtime(),
            "source-built dynamic composition should use compact parser coordinates",
        );
        let loaded_segmented = Constraint::load(segmented.save(), &vocab).unwrap();
        assert!(
            loaded_segmented.uses_compact_segmented_parser_runtime(),
            "v24 dynamic composition should restore compact parser coordinates",
        );
        for constraint in [&segmented, &loaded_segmented] {
            let mut expected = monolithic.start();
            let mut actual = constraint.start();
            for (offset, &byte) in valid.iter().enumerate() {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "segmented dynamic mask mismatch before offset {offset}, byte {byte:?}",
                );
                assert!(actual.commit_bytes(&[byte]));
                assert!(expected.commit_bytes(&[byte]));
            }
            assert_eq!(actual.mask(), expected.mask(), "segmented dynamic final mask mismatch");
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting());
        }
    }

    #[test]
    fn hybrid_boundary_policy_is_per_component_and_roundtrips() {
        // This fixture deliberately executes DynamicDirect shards. Other
        // tests temporarily arm the process-wide strict-static trap, so its
        // readers must share the writers' lock as well.
        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "x" SUB "z";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let reference = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document = "x" "y" "z";
            "#,
            &vocab,
        )
        .unwrap();
        let slot = terminal(&parent, "SUB");
        let child_input = [CompiledSubgrammarInput {
            placeholder_terminal: slot,
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];

        for static_component in [0usize, 1usize] {
            let mut static_components = BitSet::new(2);
            static_components.set(static_component);
            let hybrid = compose_constraints_owned_parent_segmented_hybrid(
                parent.clone(),
                &child_input,
                &vocab,
                &static_components,
            )
            .unwrap()
            .constraint;

            let assert_policy = |constraint: &Constraint| {
                let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
                assert_eq!(overlay.segmented_parser_components.len(), 2);
                for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                    let shard = component
                        .boundary
                        .as_ref()
                        .expect("each component in this fixture has crossing tokens");
                    if index == static_component {
                        assert!(matches!(
                            shard.backend,
                            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                        ));
                    } else {
                        assert!(matches!(
                            shard.backend,
                            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                        ));
                    }
                }
            };
            assert_policy(&hybrid);
            let loaded = Constraint::load(hybrid.save(), &vocab).unwrap();
            assert_policy(&loaded);

            for constraint in [&hybrid, &loaded] {
                for tokens in [&[5][..], &[0, 4][..], &[3, 2][..], &[0, 1, 2][..]] {
                    let mut actual = constraint.start();
                    let mut expected = reference.start();
                    for &token in tokens {
                        assert_eq!(
                            actual.mask(),
                            expected.mask(),
                            "hybrid component {static_component} mismatch before {tokens:?} token {token}",
                        );
                        assert!(actual.commit_token(token));
                        assert!(expected.commit_token(token));
                    }
                    assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                    assert!(actual.is_accepting(), "{tokens:?}");
                }
            }
        }
    }

    #[test]
    fn segmented_parser_links_preserve_nullable_child_local_coordinates() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "<" SUB ">";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item = "a";
                nt child = item?;
            "#,
            &vocab,
        )
        .unwrap();
        let slot = terminal(&parent, "SUB");
        let composed = compose_constraints_owned_parent_segmented(
            parent,
            &[CompiledSubgrammarInput {
                placeholder_terminal: slot,
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        let overlay = composed.static_dynamic_overlay.as_ref().unwrap();
        let [link] = overlay.segmented_parser_links.as_slice() else {
            panic!("expected one local parser link");
        };
        assert_eq!(link.parent_component, 0);
        assert_eq!(link.slot_terminal, slot);
        assert_eq!(link.child_component, 1);
        assert_eq!(link.child_start, 0);
        assert!(link.child_start_nullable);
        assert_eq!(
            link.return_pop,
            child.composition_child_return_pop().unwrap(),
        );
        let layout = composed
            .recursive_parser_layout()
            .unwrap()
            .expect("segmented runtime must derive the recursive parser coordinate");
        assert_eq!(layout.links.len(), 1);
        assert_eq!(layout.links[0].parent_component, 0);
        assert_eq!(layout.links[0].child_component, 1);
        assert_eq!(layout.links[0].child_start, 0);
        assert!(layout.links[0].child_start_nullable);
        assert_eq!(layout.leaf_state_offsets[0], 0);
        assert_eq!(
            layout.leaf_state_offsets[1],
            layout.leaves[0].state_count,
            "child root must be expressed in the contiguous recursive leaf coordinate",
        );
    }

    #[test]
    fn composed_constraint_matches_monolithic_for_fused_entry_and_exit_tokens() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"<a".to_vec()),
            (1, b"b>".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<".to_vec()),
            (5, b">".to_vec()),
            (6, b"a".to_vec()),
            (7, b"b".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "<" SUB ">" "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                grammar inner {
                    start child;
                    nt child = "a" "b";
                };
                nt document = "<" inner ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
                constraint: &child,
            }],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;

        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for token in [0, 1, 3] {
            assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
            assert!(actual.commit_token(token));
            assert!(expected.commit_token(token));
        }
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
    }

    #[test]
    fn composed_constraint_matches_monolithic_for_child_outer_child_token() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token LEFT;
                extern token RIGHT;
                nt document = "<" LEFT ">,<" RIGHT ">" "!";
            "#,
            &vocab, &[("LEFT", 998), ("RIGHT", 999)], &[])
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "c" "d";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                grammar left {
                    start child;
                    nt child = "a" "b";
                };
                grammar right {
                    start child;
                    nt child = "c" "d";
                };
                nt document = "<" left ">,<" right ">" "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &[
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "LEFT"),
                    additional_placeholder_terminals: &[],
                    constraint: &left,
                },
                CompiledSubgrammarInput {
                    placeholder_terminal: terminal(&parent, "RIGHT"),
                    additional_placeholder_terminals: &[],
                    constraint: &right,
                },
            ],
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
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
            assert!(actual_separate.commit_token(token));
            assert!(expected_separate.commit_token(token));
        }
        assert!(actual_separate.is_accepting());
        assert!(expected_separate.is_accepting());

        let mut expected = monolithic.start();
        let mut actual = composed.start();
        for token in [0, 1, 2, 3] {
            assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
            assert!(actual.commit_token(token));
            assert!(expected.commit_token(token));
        }
        assert_eq!(actual.is_accepting(), expected.is_accepting());
        assert!(actual.is_accepting());
    }

    #[test]
    fn same_child_divergent_continuations_match_dynamic_and_monolithic() {
        // One placeholder called from two parent states with divergent
        // continuations (L SUB x | R SUB y). The fused exit tokens ax/ay
        // discriminate the return linkage: after L only ax (not ay) may
        // complete the child, and after R only ay (not ax). A shared
        // child-start row that resolves the return to a single call site
        // under-admits the other site's fused token here.
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"ax".to_vec()),
            (1, b"ay".to_vec()),
            (2, b"L".to_vec()),
            (3, b"R".to_vec()),
            (4, b"a".to_vec()),
            (5, b"x".to_vec()),
            (6, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "L" SUB "x" | "R" SUB "y";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt inner = "a";
                nt document = "L" inner "x" | "R" inner "y";
            "#,
            &vocab,
        )
        .unwrap();
        let sub = terminal(&parent, "SUB");
        assert!(
            (0..parent.parser_symbol_count())
                .filter(|&state| parent.parser_advance_row(state).is_some_and(|row| row.contains(sub as usize)))
                .count()
                >= 2,
            "parent must shift SUB from two divergent call-site states",
        );
        let inputs = [CompiledSubgrammarInput {
            placeholder_terminal: sub,
            additional_placeholder_terminals: &[],
            constraint: &child,
        }];
        let static_composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .unwrap()
        .constraint;
        let dynamic_composed = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;

        for sequence in [vec![2, 0], vec![3, 1], vec![2, 4, 5], vec![3, 4, 6]] {
            let mut mono = monolithic.start();
            let mut static_state = static_composed.start();
            let mut dynamic_state = dynamic_composed.start();
            for token in sequence.clone() {
                assert_eq!(
                    static_state.mask(),
                    mono.mask(),
                    "static mask mismatch before token {token} in {sequence:?}",
                );
                assert_eq!(
                    dynamic_state.mask(),
                    mono.mask(),
                    "dynamic mask mismatch before token {token} in {sequence:?}",
                );
                assert!(static_state.commit_token(token));
                assert!(dynamic_state.commit_token(token));
                assert!(mono.commit_token(token));
            }
            assert_eq!(static_state.is_accepting(), mono.is_accepting());
            assert!(static_state.is_accepting());
            assert_eq!(dynamic_state.is_accepting(), mono.is_accepting());
        }
        for sequence in [vec![2, 1], vec![3, 0]] {
            for (name, constraint) in [
                ("monolithic", &monolithic),
                ("static", &static_composed),
                ("dynamic", &dynamic_composed),
            ] {
                let mut state = constraint.start();
                let mut committed = true;
                for token in &sequence {
                    if !state.commit_token(*token) {
                        committed = false;
                        break;
                    }
                }
                assert!(
                    !(committed && state.is_accepting()),
                    "{name} wrongly accepts cross-site {sequence:?}",
                );
            }
        }
    }

    #[test]
    fn nullable_child_composition_matches_monolithic_across_empty_and_nonempty_paths() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item = "a";
                nt child = item?;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item = "a";
                nt document = "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();
        assert!(child.composition_start_nullable().unwrap());
        let loaded_child = Constraint::load(child.save(), &vocab).unwrap();
        assert!(loaded_child.composition_start_nullable().unwrap());

        // Nullable bound children are outside the signed-transfer static
        // linker's supported class (unbounded silent Entry/Return episodes);
        // static links decline them loudly, so this X! differential runs on
        // the exact dynamic backend.
        for child in [&child, &loaded_child] {
            let composed = parent
                .compose_linked_children_for_test_dynamic(&[("SUB", child)], &vocab)
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
                    assert!(actual.commit_token(token));
                    assert!(expected.commit_token(token));
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting());
                assert!(actual.is_accepting());
            }
        }
    }

    #[test]
    fn nested_composition_preserves_overlapping_child_start_alternatives() {
        let vocab = byte_vocab();

        let expr = Constraint::from_glrm_grammar(
            r#"
                start expr;
                nt expr = "x";
            "#,
            &vocab,
        )
        .unwrap();

        let arg_a_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start args;
                extern token EXPR;
                nt args = "{" "a" ":" EXPR "}" | EXPR;
            "#,
            &vocab, &[("EXPR", 996)], &[])
        .unwrap();
        let arg_a = arg_a_parent
            .compose_linked_children_for_test(&[("EXPR", &expr)], &vocab)
            .unwrap();
        let arg_a_dynamic = arg_a_parent
            .compose_linked_children_for_test_dynamic(&[("EXPR", &expr)], &vocab)
            .unwrap();

        let arg_b_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start args;
                extern token EXPR;
                nt args = "{" "b" ":" EXPR "}" | EXPR;
            "#,
            &vocab, &[("EXPR", 996)], &[])
        .unwrap();
        let arg_b = arg_b_parent
            .compose_linked_children_for_test(&[("EXPR", &expr)], &vocab)
            .unwrap();
        let arg_b_dynamic = arg_b_parent
            .compose_linked_children_for_test_dynamic(&[("EXPR", &expr)], &vocab)
            .unwrap();

        let dispatch_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start call;
                extern token ARGA;
                extern token ARGB;
                nt call = "t" "." "ta" "(" ARGA ")"
                          | "t" "." "tb" "(" ARGB ")";
            "#,
            &vocab, &[("ARGA", 997), ("ARGB", 998)], &[])
        .unwrap();
        let dispatch = dispatch_parent
            .compose_linked_children_for_test(&[("ARGA", &arg_a), ("ARGB", &arg_b)], &vocab)
            .unwrap();
        let dispatch_dynamic = dispatch_parent
            .compose_linked_children_for_test_dynamic(
                &[("ARGA", &arg_a_dynamic), ("ARGB", &arg_b_dynamic)],
                &vocab,
            )
            .unwrap();

        // The two argument children both expose the same nested `expr` as a
        // whole-start alternative. The resulting composition must remain a
        // valid child for a later composition level.
        let outer_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token CALL;
                nt document = "X" CALL "!";
            "#,
            &vocab, &[("CALL", 999)], &[])
        .unwrap();
        let composed = outer_parent
            .clone()
            .compose_linked_children_for_test(&[("CALL", &dispatch)], &vocab)
            .unwrap();
        let composed_dynamic = outer_parent
            .clone()
            .compose_linked_children_for_test_dynamic(&[("CALL", &dispatch_dynamic)], &vocab)
            .unwrap();
        let loaded_dispatch = Constraint::load(dispatch.save(), &vocab).unwrap();
        let composed_from_loaded = outer_parent
            .compose_linked_children_for_test(&[("CALL", &loaded_dispatch)], &vocab)
            .unwrap();

        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt expr = "x";
                nt arg_a = "{" "a" ":" expr "}" | expr;
                nt arg_b = "{" "b" ":" expr "}" | expr;
                nt call = "t" "." "ta" "(" arg_a ")"
                          | "t" "." "tb" "(" arg_b ")";
                nt document = "X" call "!";
            "#,
            &vocab,
        )
        .unwrap();

        for bytes in [b"Xt.ta(x)!".as_slice(), b"Xt.tb(x)!", b"Xt.ta({a:x})!", b"Xt.tb({b:x})!"] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            let mut dynamic = composed_dynamic.start();
            let mut restored = composed_from_loaded.start();
            for &byte in bytes {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before byte {byte:?}");
                assert_eq!(dynamic.mask(), expected.mask(), "dynamic mask mismatch before byte {byte:?}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask mismatch before byte {byte:?}");
                assert!(actual.commit_token(byte as u32));
                assert!(dynamic.commit_token(byte as u32));
                assert!(restored.commit_token(byte as u32));
                assert!(expected.commit_token(byte as u32));
            }
            assert!(actual.is_accepting());
            assert!(dynamic.is_accepting());
            assert!(restored.is_accepting());
            assert!(expected.is_accepting());
        }

        // Structural sharing is a table-size optimization only. With sharing at
        // its default (no env opt-out), both explicit segmented backends must
        // still agree with the fully source-inlined grammar on masks *and*
        // acceptance, which requires the quotient to be skipped whenever an
        // explicit segmented boundary is requested (both build the same
        // global->local state_relations inverse).
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed,
            &monolithic,
            &vocab,
            4,
            "contextual-static-vs-source-inline",
        );
        assert_constraints_equivalent_on_reachable_prefixes_labeled(
            &composed_dynamic,
            &monolithic,
            &vocab,
            4,
            "contextual-dynamic-vs-source-inline",
        );
    }

    #[test]
    fn nested_composition_matches_flat_monolithic_across_all_boundaries() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
                nt leaf = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start middle;
                extern token LEAF;
                nt middle = "[" LEAF "]";
            "#,
            &vocab, &[("LEAF", 998)], &[])
        .unwrap();
        let middle = middle_parent
            .compose_linked_children_for_test(&[("LEAF", &leaf)], &vocab)
            .unwrap();
        let outer_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token MIDDLE;
                nt document = "X" MIDDLE "!";
            "#,
            &vocab, &[("MIDDLE", 999)], &[])
        .unwrap();
        let composed = outer_parent
            .compose_linked_children_for_test_dynamic(&[("MIDDLE", &middle)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt document = "X" "[" "a" "]" "!";
            "#,
            &vocab,
        )
        .unwrap();

        for sequence in [vec![0], vec![1, 2, 3], vec![4, 5, 6, 7, 3]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting());
        }
    }

    #[test]
    fn nested_composition_preserves_nullable_start_through_save_load() {
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X!".to_vec()),
            (1, b"Xa!".to_vec()),
            (2, b"X".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let leaf = Constraint::from_glrm_grammar(
            r#"
                start leaf;
                nt leaf = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start middle;
                extern token LEAF;
                nt middle = LEAF?;
            "#,
            &vocab, &[("LEAF", 998)], &[])
        .unwrap();
        let middle = middle_parent
            .compose_linked_children_for_test(&[("LEAF", &leaf)], &vocab)
            .unwrap();
        assert!(middle.composition_start_nullable().unwrap());
        let middle = Constraint::load(middle.save(), &vocab).unwrap();
        assert!(middle.composition_start_nullable().unwrap());

        let outer_parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token MIDDLE;
                nt document = "X" MIDDLE "!";
            "#,
            &vocab, &[("MIDDLE", 999)], &[])
        .unwrap();
        // Nested link: the middle component is itself a segmented
        // composition whose start is nullable (deferred class), so the outer
        // link uses the exact Dynamic boundary backend. Behavior must still
        // match the monolithic constraint exactly, including through
        // save/load.
        let nested_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&outer_parent, "MIDDLE"),
            additional_placeholder_terminals: &[],
            constraint: &middle,
        }];
        let static_composed = compose_constraints_owned_parent_segmented(
            outer_parent.clone(), &nested_inputs, &vocab, SegmentedBoundaryBackend::StaticParserDwa,
        ).expect("native nullable static closure").constraint;
        let composed = compose_constraints_owned_parent_segmented(
            outer_parent.clone(),
            &nested_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .unwrap()
        .constraint;
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt item = "a";
                nt document = "X" item? "!";
            "#,
            &vocab,
        )
        .unwrap();

        assert_constraints_equivalent_on_reachable_prefixes(
            &static_composed, &monolithic, &vocab, 4,
        );
        let reloaded_static = Constraint::load(static_composed.save(), &vocab).unwrap();
        assert_constraints_equivalent_on_reachable_prefixes(
            &reloaded_static, &monolithic, &vocab, 4,
        );
        for sequence in [vec![0], vec![1], vec![2, 4], vec![2, 3, 4]] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            for token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before token {token}");
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting());
            assert!(actual.is_accepting());
        }
    }

    #[test]
    fn nested_static_link_matches_dynamic_through_public_compose() {
        if crate::isolate_environment_test(false) { return; }
        // Production nested static link (acyclic, effectively nonnullable,
        // depth 2) through the PUBLIC compose route: inner Dynamic reference
        // composition, outer StaticParserDwa link, mask-gated corpus
        // differential with the strict-static trap armed on the static side,
        // plus backend pins and a shard ablation so the test cannot pass
        // vacuously.
        struct TrapGuard(bool);
        impl TrapGuard {
            fn set() -> Self {
                unsafe {
                    crate::__private::set_env_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", Some("1"));
                }
                Self(true)
            }
            fn clear(&mut self) {
                if self.0 {
                    unsafe {
                        crate::__private::set_env_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", None);
                    }
                    self.0 = false;
                }
            }
        }
        impl Drop for TrapGuard {
            fn drop(&mut self) {
                self.clear();
            }
        }
        fn admits(mask: &[u32], token: u32) -> bool {
            mask.get(token as usize / 32)
                .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
        }
        fn is_subset(a: &[u32], b: &[u32]) -> bool {
            a.iter().zip(b.iter()).all(|(x, y)| x & !y == 0) && a.len() <= b.len()
        }

        let _env_lock = crate::TEST_ENV_LOCK.lock().unwrap();
        // Token ids: 0 L, 1 R, 2 x, 3 y, 4 m, 5 g, 6 Lm, 7 Rm, 8 mg, 9 gm,
        // 10 gx, 11 gy (same fixture as the lower-level nested pin).
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const M: u32 = 4;
        const G: u32 = 5;
        const LM: u32 = 6;
        const RM: u32 = 7;
        const MG: u32 = 8;
        const GM: u32 = 9;
        const GX: u32 = 10;
        const GY: u32 = 11;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"m".to_vec()),
            (5, b"g".to_vec()),
            (6, b"Lm".to_vec()),
            (7, b"Rm".to_vec()),
            (8, b"mg".to_vec()),
            (9, b"gm".to_vec()),
            (10, b"gx".to_vec()),
            (11, b"gy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let mid = Constraint::from_glrm_with_token_ids(
            r#"
                start m;
                extern token SUB2;
                nt m = "m" SUB2;
            "#,
            &vocab, &[("SUB2", 998)], &[])
        .unwrap();
        let grandchild = Constraint::from_glrm_grammar(
            r#"
                start g;
                nt g = "g";
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            !mid.composition_start_nullable().unwrap(),
            "nested fixture mid must stay effectively nonnullable",
        );
        assert!(
            !grandchild.composition_start_nullable().unwrap(),
            "nested fixture grandchild must stay effectively nonnullable",
        );
        // Inner reference composition through the production dynamic route.
        let mid_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&mid, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &grandchild,
        }];
        let mid_dyn = compose_constraints_owned_parent_segmented(
            mid.clone(),
            &mid_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("inner dynamic compose")
        .constraint;
        let outer_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        let outer_dyn = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("outer dynamic compose")
        .constraint;
        // The supported nested class links statically through the public
        // route (no loud decline).
        let outer_static = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("outer nested static compose")
        .constraint;
        // Backend pins: both top-level components hold StaticParser shards and
        // the nested middle overlay carries no shards of its own.
        {
            let overlay = outer_static
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            let mid_inner = overlay.segmented_parser_components[1]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("mid overlay");
            assert!(
                mid_inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.as_ref().is_none_or(|shard|
                        matches!(shard.backend, crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)))),
                "nested source boundaries must also execute statically",
            );
            assert!(mid_inner.boundary_shards().all(|shard|
                matches!(shard.backend, crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_))));
        }
        // Mask-gated corpus from the dynamic reference (no trap): prefixes of
        // the complete L/R strings plus fused-spelling entry points.
        struct Node {
            path: Vec<u32>,
            mask: Vec<u32>,
        }
        let candidates: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![L, M],
            vec![R, M],
            vec![LM],
            vec![RM],
            vec![L, M, G],
            vec![R, M, G],
            vec![LM, G],
            vec![RM, G],
            vec![L, M, G, M],
            vec![R, M, G, M],
            vec![L, M, G, M, G],
            vec![R, M, G, M, G],
            vec![L, M, G, M, G, X],
            vec![R, M, G, M, G, Y],
        ];
        let mut nodes = Vec::new();
        for path in &candidates {
            let mut st = outer_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits(&st.mask(), token) {
                    ok = false;
                    break;
                }
                if !st.commit_token(token) {
                    ok = false;
                    break;
                }
            }
            if !ok {
                continue;
            }
            let mask = st.mask();
            nodes.push(Node { path: path.clone(), mask });
        }
        assert!(
            nodes.len() >= 10,
            "nested corpus must cover real positions, got {}",
            nodes.len()
        );
        // Trap-armed replay: every recorded path matches identically on the
        // public static composition. Any hidden dynamic fallback panics here.
        let mut trap = TrapGuard::set();
        let mut mismatches = 0usize;
        for node in &nodes {
            let mut st = outer_static.start();
            for &token in &node.path {
                assert!(st.commit_token(token), "static replay");
            }
            let actual = st.mask();
            if actual != node.mask {
                eprintln!(
                    "[glrmask/test][public_nested_static_mismatch] path={:?} dynamic={:?} static={:?}",
                    node.path, node.mask, actual,
                );
                mismatches += 1;
            }
        }
        trap.clear();
        assert_eq!(mismatches, 0, "nested static must match dynamic on every path");
        // Readable call-site discrimination through two nesting levels.
        for (prefix, fused, sibling, site) in [
            (vec![L, M, G, M], GX, GY, "L"),
            (vec![R, M, G, M], GY, GX, "R"),
        ] {
            let mut st = outer_static.start();
            for &token in &prefix {
                assert!(st.commit_token(token));
            }
            let mask = st.mask();
            assert!(admits(&mask, fused), "nested admits {fused} after {site}mgm");
            assert!(!admits(&mask, sibling), "nested rejects {sibling} after {site}mgm");
        }
        {
            let mut st = outer_static.start();
            assert!(st.commit_token(L));
            assert!(admits(&st.mask(), M), "nested admits m after L");
            assert!(admits(&st.mask(), MG), "nested admits mg after L");
            assert!(!admits(&st.mask(), RM), "nested rejects Rm after L");
        }
        // Ablation: clearing the installed top-level shards must lose
        // admissions (the shards are genuinely used, not redundant).
        let mut ablated = outer_static.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &candidates {
            let mut dyn_st = outer_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits(&dyn_st.mask(), token) {
                    break;
                }
                assert!(dyn_st.commit_token(token));
                if !admits(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                assert!(abl_st.commit_token(token));
                assert!(
                    is_subset(&abl_st.mask(), &dyn_st.mask()),
                    "shard-less mask must stay a subset of the reference",
                );
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    /// Mask-for-mask differential over byte-prefix scenarios: static and dynamic
    /// compositions must agree on masks, commit outcomes, and acceptance.
    /// Oracle-driven (no hardcoded masks).
    fn assert_composed_masks_equal(
        name: &str,
        static_comp: &Constraint,
        dynamic: &Constraint,
        prefixes: &[Vec<u8>],
    ) {
        for (scenario, prefix) in prefixes.iter().enumerate() {
            let mut st_static = static_comp.start();
            let mut st_dyn = dynamic.start();
            let mut consumed = 0usize;
            loop {
                assert_eq!(
                    st_static.mask(),
                    st_dyn.mask(),
                    "{name}: mask mismatch scenario={scenario} consumed={consumed}",
                );
                if consumed == prefix.len() {
                    break;
                }
                let r_static = st_static.commit_bytes(&prefix[consumed..consumed + 1]);
                let r_dyn = st_dyn.commit_bytes(&prefix[consumed..consumed + 1]);
                assert_eq!(
                    r_static,
                    r_dyn,
                    "{name}: commit divergence scenario={scenario} consumed={consumed}",
                );
                if !r_static {
                    break;
                }
                consumed += 1;
            }
            assert_eq!(
                st_static.is_accepting(),
                st_dyn.is_accepting(),
                "{name}: acceptance mismatch scenario={scenario}",
            );
        }
    }

    /// Token-level differential over fused/multibyte scenarios: commit whole
    /// vocabulary tokens (not byte slices) on both backends, comparing masks,
    /// commit outcomes, and acceptance. Oracle-driven (no hardcoded masks).
    /// Multibyte tokens that span a component boundary cross CALL/RETURN inside
    /// the boundary shards; byte-prefix differentials alone cannot see them.
    fn assert_composed_token_masks_equal(
        name: &str,
        static_comp: &Constraint,
        dynamic: &Constraint,
        paths: &[Vec<u32>],
    ) {
        for (scenario, path) in paths.iter().enumerate() {
            let mut st_static = static_comp.start();
            let mut st_dyn = dynamic.start();
            let mut step = 0usize;
            loop {
                assert_eq!(
                    st_static.mask(),
                    st_dyn.mask(),
                    "{name}: mask mismatch scenario={scenario} step={step}",
                );
                if step == path.len() {
                    break;
                }
                let r_static = st_static.commit_token(path[step]);
                let r_dyn = st_dyn.commit_token(path[step]);
                assert_eq!(
                    r_static,
                    r_dyn,
                    "{name}: commit divergence scenario={scenario} step={step} token={}",
                    path[step],
                );
                if !r_static {
                    break;
                }
                step += 1;
            }
            assert_eq!(
                st_static.is_accepting(),
                st_dyn.is_accepting(),
                "{name}: acceptance mismatch scenario={scenario}",
            );
        }
    }

    /// Process-wide strict-static trap: any hidden dynamic fallback on a
    /// claimed-static path panics while armed. Tests arm it only around the
    /// static side (the dynamic oracle is built unarmed) under `TEST_ENV_LOCK`.
    struct StrictStaticTrapGuard(bool);
    impl StrictStaticTrapGuard {
        fn arm() -> Self {
            unsafe {
                crate::__private::set_env_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", Some("1"));
            }
            Self(true)
        }
        fn disarm(&mut self) {
            if self.0 {
                unsafe {
                    crate::__private::set_env_var("GLRMASK_STRICT_STATIC_TRAP_DYNAMIC", None);
                }
                self.0 = false;
            }
        }
    }
    impl Drop for StrictStaticTrapGuard {
        fn drop(&mut self) {
            self.disarm();
        }
    }

    fn admits_token(mask: &[u32], token: u32) -> bool {
        mask.get(token as usize / 32)
            .is_some_and(|word| word & (1u32 << (token % 32)) != 0)
    }

    #[test]
    fn floor_flag_does_not_suppress_explicit_static_boundary_shards() {
        // The legacy `GLRMASK_EXPERIMENT_OWNED_COMPONENTS_ONLY_STATIC` floor
        // flag must not suppress boundary shards on modern explicit Static /
        // Dynamic links: the env read is gated on
        // `explicit_segmented_boundary.is_none()`. Uses the public nested
        // fixture (12-token vocab; parent `"L" SUB SUB "x" | "R" SUB SUB "y"`;
        // mid `"m" SUB2`; grandchild `"g"`): inner Dynamic reference, outer
        // Static + Dynamic. Unflagged baseline first; then flag set (RAII)
        // with flagged static + dynamic; then all four compared mask-for-mask
        // over [LM,G,M,GX] and [RM,G,M,GY] with end acceptance.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        struct FloorFlagGuard {
            previous: Option<std::ffi::OsString>,
        }
        impl FloorFlagGuard {
            fn set(value: Option<&str>) -> Self {
                const KEY: &str = "GLRMASK_EXPERIMENT_OWNED_COMPONENTS_ONLY_STATIC";
                let previous = crate::env::var_os(KEY);
                unsafe {
                    crate::__private::set_env_var(KEY, value);
                }
                Self { previous }
            }
        }
        impl Drop for FloorFlagGuard {
            fn drop(&mut self) {
                const KEY: &str = "GLRMASK_EXPERIMENT_OWNED_COMPONENTS_ONLY_STATIC";
                unsafe {
                    crate::__private::set_env_var(
                        KEY,
                        self.previous.as_deref().and_then(std::ffi::OsStr::to_str),
                    );
                }
            }
        }
        fn has_static_shard(constraint: &Constraint) -> bool {
            constraint
                .static_dynamic_overlay
                .as_ref()
                .is_some_and(|overlay| {
                    overlay
                        .segmented_parser_components
                        .iter()
                        .any(|component| {
                            matches!(
                                component.boundary.as_ref().map(|shard| &shard.backend),
                                Some(crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_))
                            )
                        })
                })
        }
        const M: u32 = 4;
        const G: u32 = 5;
        const LM: u32 = 6;
        const RM: u32 = 7;
        const GX: u32 = 10;
        const GY: u32 = 11;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"m".to_vec()),
            (5, b"g".to_vec()),
            (6, b"Lm".to_vec()),
            (7, b"Rm".to_vec()),
            (8, b"mg".to_vec()),
            (9, b"gm".to_vec()),
            (10, b"gx".to_vec()),
            (11, b"gy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                nt document = "L" SUB SUB "x" | "R" SUB SUB "y";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let mid = Constraint::from_glrm_with_token_ids(
            r#"
                start m;
                extern token SUB2;
                nt m = "m" SUB2;
            "#,
            &vocab, &[("SUB2", 998)], &[])
        .unwrap();
        let grandchild = Constraint::from_glrm_grammar(
            r#"
                start g;
                nt g = "g";
            "#,
            &vocab,
        )
        .unwrap();
        let mid_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&mid, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &grandchild,
        }];
        let mid_dyn = compose_constraints_owned_parent_segmented(
            mid.clone(),
            &mid_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("inner dynamic compose")
        .constraint;
        let outer_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &mid_dyn,
        }];
        // Unflagged baseline static + dynamic FIRST (flag explicitly absent).
        let _baseline_unset = FloorFlagGuard::set(None);
        let base_static = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("baseline static compose")
        .constraint;
        let base_dynamic = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &outer_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("baseline dynamic compose")
        .constraint;
        assert!(
            has_static_shard(&base_static),
            "baseline static link must publish a static shard",
        );
        // Flagged static + dynamic under RAII flag (restores prior value).
        let (flag_static, flag_dynamic) = {
            let _flag = FloorFlagGuard::set(Some("1"));
            (
                compose_constraints_owned_parent_segmented(
                    parent.clone(),
                    &outer_inputs,
                    &vocab,
                    SegmentedBoundaryBackend::StaticParserDwa,
                )
                .expect("flagged static compose")
                .constraint,
                compose_constraints_owned_parent_segmented(
                    parent.clone(),
                    &outer_inputs,
                    &vocab,
                    SegmentedBoundaryBackend::Dynamic,
                )
                .expect("flagged dynamic compose")
                .constraint,
            )
        };
        assert!(
            has_static_shard(&flag_static),
            "floor flag must not suppress static boundary shards on an explicit static link",
        );
        // All four compared mask-for-mask: INITIAL masks first, then the
        // commit loop; fresh states (reset) per sequence.
        for sequence in [&[LM, G, M, GX][..], &[RM, G, M, GY][..]] {
            let mut states = [
                base_static.start(),
                base_dynamic.start(),
                flag_static.start(),
                flag_dynamic.start(),
            ];
            let initial = [
                states[0].mask(),
                states[1].mask(),
                states[2].mask(),
                states[3].mask(),
            ];
            assert_eq!(initial[0], initial[1], "baseline initial static==dynamic");
            assert_eq!(initial[2], initial[3], "flagged initial static==dynamic");
            assert_eq!(initial[0], initial[2], "flagged initial==baseline initial");
            for &token in sequence {
                for st in &mut states {
                    assert!(
                        admits_token(&st.mask(), token),
                        "floor-flag token {token} admitted",
                    );
                    assert!(st.commit_token(token), "floor-flag commit");
                }
                let masks = [
                    states[0].mask(),
                    states[1].mask(),
                    states[2].mask(),
                    states[3].mask(),
                ];
                assert_eq!(masks[0], masks[1], "baseline static==dynamic");
                assert_eq!(masks[2], masks[3], "flagged static==dynamic");
                assert_eq!(masks[0], masks[2], "flagged==baseline");
            }
            for st in &states {
                assert!(st.is_accepting(), "floor-flag sequence must accept");
            }
        }
    }

    #[test]
    #[ignore = "diagnostic: requires GLRMASK_MINIMIZE_EQ_LEFT/RIGHT prepared static.bin paths (A1/B1); run explicitly"]
    fn prepared_static_minimize_orders_are_weighted_equivalent() {
        // Diagnostic weighted-language equivalence between two prepared
        // static compositions differing ONLY in signed-shard minimize order
        // (Stable vs DescendingDomain). Both must be built by the SAME binary
        // from the SAME inputs (selected10 cache, sequential-3 prepare flow).
        // For each StaticParser shard present in BOTH (matched by
        // start_component, nonempty on both sides, identical component set),
        // `find_difference` (symmetric checker) must return None on the
        // recursive-parser-coordinate DWAs. Coordinate comparability is
        // verified from loaded metadata (`uses_composed_tsid_coordinate`,
        // `tokenizer_state_to_tsid`, `internal_token_to_originals` equal);
        // any mismatch stops the test instead of comparing incomparable
        // weights. State counts are NOT compared (representation may differ).
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let left_path = crate::env::var("GLRMASK_MINIMIZE_EQ_LEFT")
            .expect("GLRMASK_MINIMIZE_EQ_LEFT must name the Stable prepared static.bin");
        let right_path = crate::env::var("GLRMASK_MINIMIZE_EQ_RIGHT")
            .expect("GLRMASK_MINIMIZE_EQ_RIGHT must name the Descending prepared static.bin");
        let phase1 = crate::env::var("PHASE1_DIR")
            .expect("PHASE1_DIR must name the selected10 cache root");
        let vocab = super::load_vocab(
            &std::path::Path::new(&phase1).join("vocab_dump.bin").display().to_string(),
        );
        let load = |path: &str| {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            Constraint::load(bytes, &vocab)
                .unwrap_or_else(|e| panic!("load {path}: {e:?}"))
        };
        let left = load(&left_path);
        let right = load(&right_path);
        let left_overlay = left
            .static_dynamic_overlay
            .as_ref()
            .expect("LEFT prepared static must retain a segmented overlay");
        let right_overlay = right
            .static_dynamic_overlay
            .as_ref()
            .expect("RIGHT prepared static must retain a segmented overlay");
        let collect = |overlay: &crate::runtime::StaticDynamicOverlayMetadata| {
            let mut shards = std::collections::BTreeMap::new();
            for component in &overlay.segmented_parser_components {
                let Some(shard) = component.boundary.as_ref() else {
                    continue;
                };
                let crate::runtime::SegmentedBoundaryShardBackend::StaticParser(parser) =
                    &shard.backend
                else {
                    continue;
                };
                assert!(
                    shards.insert(shard.start_component, parser.clone()).is_none(),
                    "duplicate start_component shard",
                );
            }
            shards
        };
        let left_shards = collect(left_overlay);
        let right_shards = collect(right_overlay);
        assert!(!left_shards.is_empty(), "LEFT must carry >=1 static shard");
        assert!(!right_shards.is_empty(), "RIGHT must carry >=1 static shard");
        let left_set: Vec<u32> = left_shards.keys().copied().collect();
        let right_set: Vec<u32> = right_shards.keys().copied().collect();
        assert_eq!(
            left_set, right_set,
            "static shard component sets must be identical (LEFT {left_set:?} vs RIGHT {right_set:?})"
        );
        let mut checked = 0usize;
        for (component, left_parser) in &left_shards {
            let right_parser = &right_shards[component];
            eprintln!("[minimize-eq] start_component={component}: checking");
            assert_eq!(
                left_parser.coordinate, right_parser.coordinate,
                "component {component}: scanner coordinate differs; incomparable weights",
            );
            assert_eq!(
                left_parser.internal_token_to_originals,
                right_parser.internal_token_to_originals,
                "component {component}: internal_token_to_originals differs; incomparable weights",
            );
            let left_dwa = &left_parser.parser_dwa;
            let right_dwa = &right_parser.parser_dwa;
            assert!(left_dwa.is_acyclic(), "component {component}: LEFT DWA must be acyclic");
            assert!(right_dwa.is_acyclic(), "component {component}: RIGHT DWA must be acyclic");
            match crate::automata::weighted_u32::equivalence::find_difference(left_dwa, right_dwa) {
                Ok(None) => {
                    eprintln!(
                        "[minimize-eq] start_component={component}: done equivalent (states {} vs {})",
                        left_dwa.num_states(),
                        right_dwa.num_states(),
                    );
                    checked += 1;
                }
                Ok(Some(word)) => panic!(
                    "component {component}: weighted-language difference at word {word:?}"
                ),
                Err(e) => panic!("component {component}: checker error: {e:?}"),
            }
        }
        assert!(checked > 0, "must check >=1 shard non-vacuously");
        eprintln!("[minimize-eq] all {checked} static shards weighted-equivalent");
    }

    #[test]
    fn already_composed_parent_static_link_extends_and_matches_dynamic() {
        // Chained static binds: reusing an already-composed constraint as the
        // *parent* of a static link expands the parent block to intact leaves
        // exactly like nested children (no dynamic fallback, no silent
        // substitution). First link fills SUB through the dynamic route,
        // leaving SUB2 free; the static second link must succeed with
        // installed static shards and dynamic-identical masks.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token SUB2;
                nt document = "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab, &[("SUB", 999), ("SUB2", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a = "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let first_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let composed_parent = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &first_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("first dynamic compose")
        .constraint;
        assert!(
            composed_parent.uses_compact_segmented_parser_runtime(),
            "first composition must leave a segmented parent",
        );
        // Coverage boundary: this shape carries no terminal aliases, so slot
        // resolution runs the interval path. (`global_terminal_aliases` are
        // ignore-fold metadata produced only by
        // `build_segmented_runtime_metadata` from `merged_ignore_terminals`
        // (which sees only per-component `ignore_terminal`s) — additional
        // placeholder terminals produce links, never aliases. A slot naming a
        // folded terminal therefore declines loudly as ambiguous, by design.)
        for component in composed_parent
            .static_dynamic_overlay
            .as_ref()
            .expect("segmented overlay")
            .segmented_parser_components
            .iter()
        {
            assert!(
                component.global_terminal_aliases.is_empty(),
                "pin fixture shape must stay alias-free",
            );
        }
        let second_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&composed_parent, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let extended = compose_constraints_owned_parent_segmented(
            composed_parent.clone(),
            &second_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static link over a composed parent must extend")
        .constraint;
        let overlay = extended
            .static_dynamic_overlay
            .as_ref()
            .expect("extended composition must retain segmented metadata");
        assert!(
            overlay.boundary_shards().next().is_some(),
            "extended composition must publish static shards",
        );
        let reference = compose_constraints_owned_parent_segmented(
            composed_parent,
            &second_inputs,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic second link")
        .constraint;
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'x'],
        ];
        assert_composed_masks_equal("composed-parent-extend", &extended, &reference, &prefixes);
    }

    #[test]
    fn composed_parent_three_bind_chain_with_nonroot_slot_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the branching fixture (measured
        // there): 3-level multibyte static prime/mask exceeds the default 2MB.
        std::thread::Builder::new()
            .name("three-bind".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_three_bind_chain_body)
            .expect("spawn three-bind fixture worker")
            .join()
            .expect("three-bind fixture worker panicked");
    }

    /// Deep-stack worker body for the three-bind test (see the wrapper above).
    fn composed_parent_three_bind_chain_body() {
        // Three chained static binds where the second bind's slot is owned by
        // a NON-ROOT leaf of the parent block (A's open SUB_A), and the third
        // bind fills two placeholders (additional terminals) with a composed
        // child: composed-as-parent and composed-as-child in one chain.
        // Multibyte tokens cross CALL (Lx, xy, Ry) and RETURN (yx, yy, yz) over
        // three nesting levels.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const LX: u32 = 5;
        const XY: u32 = 6;
        const YX: u32 = 7;
        const RY: u32 = 8;
        const YY: u32 = 9;
        const YZ: u32 = 10;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b"Lx".to_vec()),
            (6, b"xy".to_vec()),
            (7, b"yx".to_vec()),
            (8, b"Ry".to_vec()),
            (9, b"yy".to_vec()),
            (10, b"yz".to_vec()),
        ]);
        let root = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token SUB2;
                extern token SUB3;
                nt document = "L" SUB "x" | "R" SUB2 "y" | "R" SUB3 "z";
            "#,
            &vocab, &[("SUB", 999), ("SUB2", 998), ("SUB3", 997)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_with_token_ids(
            r#"
                start a;
                extern token SUB_A;
                nt a = "x" SUB_A;
            "#,
            &vocab, &[("SUB_A", 996)], &[])
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                nt c = "y";
            "#,
            &vocab,
        )
        .unwrap();
        // Composed child for the third bind (two slots, one filler).
        let parent_b = Constraint::from_glrm_with_token_ids(
            r#"
                start bp;
                extern token SB1;
                extern token SB2;
                nt bp = SB1 | SB2;
            "#,
            &vocab, &[("SB1", 995), ("SB2", 994)], &[])
        .unwrap();
        let child_e = Constraint::from_glrm_grammar(
            r#"
                start e;
                nt e = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let b_inputs = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent_b, "SB1"),
            additional_placeholder_terminals: &[terminal(&parent_b, "SB2")],
            constraint: &child_e,
        }];
        let child_b = compose_constraints_owned_parent_segmented(
            parent_b,
            &b_inputs,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind0: composed child")
        .constraint;
        // Placeholder/link contract: additional placeholder terminals produce
        // one link per slot sharing the child (`build_segmented_parser_links`),
        // never `global_terminal_aliases` entries (those are ignore-fold
        // metadata from `build_segmented_runtime_metadata` over
        // `merged_ignore_terminals`, which sees only `ignore_terminal`s).
        {
            let overlay = child_b
                .static_dynamic_overlay
                .as_ref()
                .expect("bind0 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_links.len(),
                2,
                "two slots (SB1, SB2) must produce two links",
            );
            for link in &overlay.segmented_parser_links {
                assert_eq!(link.parent_component, 0);
                assert_eq!(link.child_component, 1);
            }
            assert_eq!(
                overlay
                    .segmented_parser_links
                    .iter()
                    .map(|link| link.slot_terminal)
                    .collect::<Vec<_>>(),
                vec![b_inputs[0].placeholder_terminal]
                    .into_iter()
                    .chain(b_inputs[0].additional_placeholder_terminals.iter().copied())
                    .collect::<Vec<_>>(),
                "one link per placeholder slot, in slot order",
            );
            assert!(
                overlay
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.global_terminal_aliases.is_empty()),
                "multi-slot binds must not produce terminal aliases",
            );
        }
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1 dynamic")
        .constraint;
        // SUB_A is owned by A's leaf (non-root): its composed id is A's overlay
        // offset plus its A-local id (the public bind path exposes the same id
        // as a qualified late-grammar slot).
        let sub_a = stage1
            .static_dynamic_overlay
            .as_ref()
            .expect("segmented overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_a, "SUB_A");
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_a,
            additional_placeholder_terminals: &[],
            constraint: &child_c,
        }];
        // White-box pin: the SUB_A slot lives in A's leaf (non-root), so the
        // outer link must target a nonzero parent leaf with a leaf-local slot.
        let expansion = crate::compiler::boundary_walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        assert_eq!(expansion.top_leaf_ranges.len(), 2, "two tops");
        assert_eq!(expansion.top_leaf_ranges[0].len(), 2, "parent block spans two leaves");
        assert_eq!(expansion.links.len(), 2, "one inner plus one outer link");
        let outer = expansion
            .links
            .iter()
            .find(|link| link.child_component == expansion.top_root_leaves[1])
            .expect("outer link to the new child");
        assert_ne!(
            outer.parent_component, 0,
            "non-root-owned slot must not resolve to leaf 0",
        );
        assert!(
            (outer.slot_terminal as usize)
                < expansion.leaves[outer.parent_component as usize].parser_terminal_count() as usize,
            "resolved slot must be leaf-local",
        );
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        let bind3 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage2, "SUB2"),
            additional_placeholder_terminals: &[terminal(&stage2, "SUB3")],
            constraint: &child_b,
        }];
        let stage3 = compose_constraints_owned_parent_segmented(
            stage2.clone(),
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind3")
        .constraint;
        let stage3_dyn = compose_constraints_owned_parent_segmented(
            stage2_dyn,
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind3 dynamic")
        .constraint;
        // bind3 fills two placeholders with one composed child: two links, no
        // aliases (same placeholder/link contract as bind0).
        {
            let overlay = stage3
                .static_dynamic_overlay
                .as_ref()
                .expect("bind3 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_links.len(),
                2,
                "two slots (SUB2, SUB3) must produce two links",
            );
            for link in &overlay.segmented_parser_links {
                assert_eq!(link.parent_component, 0);
                assert_eq!(link.child_component, 1);
            }
            assert!(
                overlay
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.global_terminal_aliases.is_empty()),
                "multi-slot binds must not produce terminal aliases",
            );
        }
        // Static shard authority: both tops hold StaticParser shards and no
        // redundant global boundary parser. The already-composed parent block
        // retains its own static inner repair: under immediate block ownership
        // the outer shard omits block-internal crossings by construction.
        {
            let overlay = stage3
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            let inner = overlay.segmented_parser_components[0]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("parent-block overlay");
            assert!(
                inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ))),
                "parent-block overlay must retain only static inner shards",
            );
            assert!(
                inner.boundary_shards().next().is_some()
                    && inner.boundary_shards().all(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    )),
                "parent-block overlay static shard list must survive the extension",
            );
        }
        // Byte-prefix differential (single-byte commits).
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'L', b'x', b'y', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'R', b'y', b'z'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("bind3-bytes", &stage3, &stage3_dyn, &prefixes);
        // Token-level differential (whole multibyte commits across CALL/RETURN).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, X, Y],
            vec![L, XY],
            vec![LX, Y],
            vec![LX, YX],
            vec![L, X, Y, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![RY, Z],
            vec![R, YZ],
            vec![R, Y, Z],
            vec![L, Y],
            vec![R, X],
            vec![L, X, Z],
            vec![Z],
        ];
        assert_composed_token_masks_equal("bind3-tokens", &stage3, &stage3_dyn, &paths);
        // Fused/single equivalence: the same bytes through fused crossings
        // reach the same masks and acceptance as single-byte tokens, proving
        // the multibyte crossings resolve through the static shards.
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XY], vec![L, X, Y]),
            (vec![LX, YX], vec![L, X, Y, X]),
            (vec![R, YY], vec![R, Y, Y]),
            (vec![R, YZ], vec![R, Y, Z]),
        ] {
            let mut st_fused = stage3.start();
            for &token in &fused {
                assert!(st_fused.commit_token(token));
            }
            let mut st_single = stage3.start();
            for &token in &single {
                assert!(st_single.commit_token(token));
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        assert!(admits_token(&stage3.start().mask(), LX), "Lx crossing must be live");
        assert!(admits_token(&stage3.start().mask(), RY), "Ry crossing must be live");
        // Trap-armed replay of the oracle-gated corpus: any hidden dynamic
        // fallback on the static side panics here.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage3_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || !st.commit_token(token) {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "three-bind corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage3.start();
            for &token in path {
                assert!(st.commit_token(token), "static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "three-bind static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions (the shards are genuinely authoritative, not redundant).
        let mut ablated = stage3.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage3_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                assert!(dyn_st.commit_token(token));
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                assert!(abl_st.commit_token(token));
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_branching_then_nonroot_extension_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Static prime/mask evaluation recurses per nesting level with large
        // debug scratch frames (pre-existing runtime characteristic: the
        // existing deep tests need >1MB too, measured); this 3-level multibyte
        // shape needs more than the default 2MB, so the whole body runs on a
        // worker with headroom. The panic hook still prints assertion details;
        // a join failure fails the test.
        std::thread::Builder::new()
            .name("branching-extension".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_branching_then_nonroot_extension_body)
            .expect("spawn branching fixture worker")
            .join()
            .expect("branching fixture worker panicked");
    }

    /// Deep-stack worker body for the branching test (see the wrapper above).
    fn composed_parent_branching_then_nonroot_extension_body() {
        // Branching-then-nonroot-extension: bind1a/bind1b fill sibling slots A
        // and B; bind2 fills a slot owned by A's leaf (non-root) with C. Leaf
        // expansion follows overlay storage ([root, A, B] plus the appended
        // C), while the analysis splice packs link-tree DFS ([root, A, C, B]);
        // the splice must remap explicitly into leaf coordinate instead of
        // assuming the orders coincide. Multibyte tokens cross CALL (Lx, xz,
        // Ry) and RETURN (zx, yy) over two nesting levels.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const LX: u32 = 5;
        const XZ: u32 = 6;
        const ZX: u32 = 7;
        const RY: u32 = 8;
        const YY: u32 = 9;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b"Lx".to_vec()),
            (6, b"xz".to_vec()),
            (7, b"zx".to_vec()),
            (8, b"Ry".to_vec()),
            (9, b"yy".to_vec()),
        ]);
        let root = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB_A;
                extern token SUB_B;
                nt document = "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab, &[("SUB_A", 999), ("SUB_B", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_with_token_ids(
            r#"
                start a;
                extern token SUB_INNER;
                nt a = "x" SUB_INNER;
            "#,
            &vocab, &[("SUB_INNER", 996)], &[])
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                nt c = "z";
            "#,
            &vocab,
        )
        .unwrap();
        // Two sequential sibling binds (multi-child single links are a known
        // dynamic-oracle gap: the structural quotient breaks the functional
        // LR-state relation there, so the oracle cannot be built that way).
        // bind1a fills SUB_A, bind1b fills SUB_B; the existing root then has
        // siblings A and B with overlay nesting [[root, A], B].
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        let bind1b = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1a, "SUB_B"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1b")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1b dynamic")
        .constraint;
        {
            let overlay = stage1
                .static_dynamic_overlay
                .as_ref()
                .expect("stage1 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_components.len(),
                2,
                "bind1b stores [[root, A], B]",
            );
            assert!(
                overlay.segmented_parser_components[0]
                    .constraint
                    .uses_compact_segmented_parser_runtime(),
                "bind1b component 0 must stay a segmented subtree",
            );
        }
        // SUB_INNER is owned by A's leaf: component 0's offset plus A's offset
        // inside the nested overlay plus its A-local id (the same id the
        // public bind path exposes as a qualified slot).
        let sub_inner = {
            let overlay = stage1
                .static_dynamic_overlay
                .as_ref()
                .expect("segmented overlay");
            let nested = &overlay.segmented_parser_components[0];
            let nested_overlay = nested
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("nested segmented overlay");
            nested.terminal_offset
                + nested_overlay.segmented_parser_components[1].terminal_offset
                + terminal(&child_a, "SUB_INNER")
        };
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_inner,
            additional_placeholder_terminals: &[],
            constraint: &child_c,
        }];
        // White-box pin: leaves are storage order [root, A, B, C] with the
        // parent block spanning three leaves, and the outer link targets A's
        // (non-root) leaf with a leaf-local slot.
        let expansion = crate::compiler::boundary_walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        assert_eq!(expansion.leaves.len(), 4, "root, A, B, C");
        assert_eq!(
            expansion.top_leaf_ranges,
            vec![vec![0usize, 1usize, 2usize], vec![3usize]],
            "parent block spans three leaves; C is its own top",
        );
        let outer = expansion
            .links
            .iter()
            .find(|link| link.child_component == expansion.top_root_leaves[1])
            .expect("outer link to the new child");
        assert_eq!(
            outer.parent_component, 1,
            "slot owned by A must resolve to A's leaf",
        );
        assert!(
            (outer.slot_terminal as usize)
                < expansion.leaves[1].parser_terminal_count() as usize,
            "resolved slot must be leaf-local",
        );
        // The fixture must genuinely interleave: leaves stay storage order
        // [root, A, B, C] while the splice packs DFS [root, A, C, B].
        let dfs_order = crate::compiler::boundary_walk::nested_splice_leaf_order(&expansion)
            .expect("bind2 DFS order");
        assert_eq!(
            dfs_order,
            vec![0usize, 1, 3, 2],
            "splice DFS order must interleave with leaf order",
        );
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2: late non-root extension must link statically")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Static shard authority: both tops hold StaticParser shards and no
        // redundant global boundary parser. The already-composed parent block
        // retains its own static inner repair: under immediate block ownership
        // the new outer shard deliberately omits block-internal crossings, so
        // those retained shards are load-bearing coverage rather than duplicate
        // work.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            let inner = overlay.segmented_parser_components[0]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("parent-block overlay");
            assert!(
                inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ))),
                "parent-block overlay must retain only static inner shards",
            );
            assert!(
                inner.boundary_shards().next().is_some()
                    && inner.boundary_shards().all(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    )),
                "parent-block overlay static shard list must survive the extension",
            );
        }
        // Byte-prefix differential (single-byte commits).
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'z'],
            vec![b'L', b'x', b'z', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("branch-extend-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (whole multibyte commits across CALL/RETURN).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, X, Z],
            vec![L, XZ],
            vec![LX, Z],
            vec![LX, ZX],
            vec![L, X, Z, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![L, X, Y],
            vec![Z],
        ];
        assert_composed_token_masks_equal("branch-extend-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence: the same bytes through fused crossings
        // reach the same masks and acceptance as single-byte tokens, proving
        // the multibyte crossings resolve through the static shards.
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XZ], vec![L, X, Z]),
            (vec![LX, ZX], vec![L, X, Z, X]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                assert!(st_fused.commit_token(token));
            }
            let mut st_single = stage2.start();
            for &token in &single {
                assert!(st_single.commit_token(token));
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        assert!(admits_token(&stage2.start().mask(), LX), "Lx crossing must be live");
        assert!(admits_token(&stage2.start().mask(), RY), "Ry crossing must be live");
        // Trap-armed replay of the oracle-gated corpus: any hidden dynamic
        // fallback on the static side panics here.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || !st.commit_token(token) {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "branching corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                assert!(st.commit_token(token), "static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "branching static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions (the shards are genuinely authoritative, not redundant).
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                assert!(dyn_st.commit_token(token));
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                assert!(abl_st.commit_token(token));
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_cross_parent_multislot_static_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the other 3-level multibyte
        // static prime/mask fixtures: exceeds the default 2MB.
        std::thread::Builder::new()
            .name("cross-parent-multislot".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_cross_parent_multislot_body)
            .expect("spawn cross-parent fixture worker")
            .join()
            .expect("cross-parent fixture worker panicked");
    }

    /// Deep-stack worker body (see above): a cross-parent multi-slot bind is
    /// a VALID public bind case ("incorporated once and shared by every
    /// listed call site") supported by the Dynamic backend, and the static
    /// route supports it too via occurrence mapping. ONE child input fills
    /// slots owned by DIFFERENT leaves (root leaf + A's leaf), so the packed
    /// analysis visits the shared child once per parent path ([0,1,2,2])
    /// while the runtime keeps ONE shared child. Static-vs-dynamic
    /// differentials over multibyte CALL/RETURN crossings from both parents,
    /// fused/single equivalence, a trap-armed replay, and shard ablation
    /// prove the acceptance is genuine. This is NOT the same case as DAG
    /// reuse across inputs (distinct leaves; see the reused-sibling test).
    fn composed_parent_cross_parent_multislot_body() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const LX: u32 = 4;
        const RY: u32 = 5;
        const XY: u32 = 6;
        const YX: u32 = 7;
        const YY: u32 = 8;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"Lx".to_vec()),
            (5, b"Ry".to_vec()),
            (6, b"xy".to_vec()),
            (7, b"yx".to_vec()),
            (8, b"yy".to_vec()),
        ]);
        let root = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB_A;
                extern token SUB_B;
                nt document = "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab, &[("SUB_A", 999), ("SUB_B", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_with_token_ids(
            r#"
                start a;
                extern token SUB_INNER;
                nt a = "x" SUB_INNER;
            "#,
            &vocab, &[("SUB_INNER", 996)], &[])
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                nt c = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        let sub_b = terminal(&stage1a, "SUB_B");
        let sub_inner = stage1a
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_a, "SUB_INNER");
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_b,
            additional_placeholder_terminals: &[sub_inner],
            constraint: &child_c,
        }];
        // White-box pin: the expansion really is the DAG shape (child root
        // with two distinct parents), and the occurrence oracle visits the
        // shared child once per parent path.
        let expansion =
            crate::compiler::boundary_walk::expand_nested_link_leaves(&stage1a, &bind2)
                .expect("expansion");
        assert_eq!(expansion.leaves.len(), 3, "root, A, C");
        let child_root = expansion.top_root_leaves[1] as usize;
        let mut parents: Vec<u32> = expansion
            .links
            .iter()
            .filter(|link| link.child_component as usize == child_root)
            .map(|link| link.parent_component)
            .collect();
        parents.sort_unstable();
        parents.dedup();
        assert_eq!(parents, vec![0, 1], "one child, two distinct parents");
        let dfs_order =
            crate::compiler::boundary_walk::nested_splice_leaf_order(&expansion)
                .expect("occurrence oracle accepts the shared child");
        assert_eq!(
            dfs_order,
            vec![0, 1, 2, 2],
            "shared child occurs once per parent path"
        );
        // The static route supports the valid bind (no deferral).
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static cross-parent multi-slot bind is supported")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Static shard authority on the extended artifact.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
        }
        // Byte-prefix differential: crossings into/out of the shared child
        // from BOTH parents.
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'L', b'x', b'y', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'y'],
        ];
        assert_composed_masks_equal("xparent-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (multibyte CALL/RETURN crossings).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, XY],
            vec![LX, Y],
            vec![L, X, Y],
            vec![L, X, Y, X],
            vec![LX, Y, X],
            vec![L, XY, X],
            vec![L, X, YX],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![Y],
        ];
        assert_composed_token_masks_equal("xparent-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence across both parents' CALL/RETURN sites.
        for (fused, single) in [
            (vec![LX, Y, X], vec![L, X, Y, X]),
            (vec![L, XY, X], vec![L, X, Y, X]),
            (vec![L, X, YX], vec![L, X, Y, X]),
            (vec![RY, Y], vec![R, Y, Y]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                assert!(st_fused.commit_token(token));
            }
            let mut st_single = stage2.start();
            for &token in &single {
                assert!(st_single.commit_token(token));
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        // Both parents genuinely reach the shared child: each call site
        // accepts completions on both backends.
        for path in [vec![L, X, Y, X], vec![R, Y, Y]] {
            let mut st = stage2_dyn.start();
            for &token in &path {
                assert!(st.commit_token(token));
            }
            assert!(st.is_accepting(), "oracle must accept {path:?}");
            let mut st = stage2.start();
            for &token in &path {
                assert!(st.commit_token(token));
            }
            assert!(st.is_accepting(), "static must accept {path:?}");
        }
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || !st.commit_token(token) {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "cross-parent corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                assert!(st.commit_token(token), "static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(
            mismatches, 0,
            "cross-parent static must match dynamic on every path"
        );
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                assert!(dyn_st.commit_token(token));
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                assert!(abl_st.commit_token(token));
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_cross_parent_shared_subtree_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the sibling 3-level multibyte
        // static prime/mask fixtures: exceeds the default 2MB.
        std::thread::Builder::new()
            .name("cross-parent-subtree".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_cross_parent_shared_subtree_body)
            .expect("spawn cross-parent subtree fixture worker")
            .join()
            .expect("cross-parent subtree fixture worker panicked");
    }

    /// Deep-stack worker body (see above): repeated-subtree authority. The
    /// shared child C has its own bound descendant D (late bind into the
    /// shared leaf AFTER the cross-parent bind), so the packed analysis
    /// repeats the whole C+D subtree once per parent path ([0,1,2,3,2,3])
    /// while the runtime still shares one C and one D. Same fused
    /// multibyte CALL/RETURN/trap/ablation authority as the sibling test,
    /// extended across the C-to-D call site.
    fn composed_parent_cross_parent_shared_subtree_body() {
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const LX: u32 = 5;
        const RY: u32 = 6;
        const XY: u32 = 7;
        const YX: u32 = 8;
        const YY: u32 = 9;
        const YZ: u32 = 10;
        const ZX: u32 = 11;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b"Lx".to_vec()),
            (6, b"Ry".to_vec()),
            (7, b"xy".to_vec()),
            (8, b"yx".to_vec()),
            (9, b"yy".to_vec()),
            (10, b"yz".to_vec()),
            (11, b"zx".to_vec()),
        ]);
        let root = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB_A;
                extern token SUB_B;
                nt document = "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab, &[("SUB_A", 999), ("SUB_B", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_with_token_ids(
            r#"
                start a;
                extern token SUB_INNER;
                nt a = "x" SUB_INNER;
            "#,
            &vocab, &[("SUB_INNER", 996)], &[])
        .unwrap();
        let child_c = Constraint::from_glrm_with_token_ids(
            r#"
                start c;
                extern token SUB_D;
                nt c = "y" SUB_D;
            "#,
            &vocab, &[("SUB_D", 995)], &[])
        .unwrap();
        let child_d = Constraint::from_glrm_grammar(
            r#"
                start d;
                nt d = "z";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        let sub_b = terminal(&stage1a, "SUB_B");
        let sub_inner = stage1a
            .static_dynamic_overlay
            .as_ref()
            .expect("overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_a, "SUB_INNER");
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_b,
            additional_placeholder_terminals: &[sub_inner],
            constraint: &child_c,
        }];
        // The slotted C binds cross-parent exactly like the atomic sibling.
        let expansion =
            crate::compiler::boundary_walk::expand_nested_link_leaves(&stage1a, &bind2)
                .expect("expansion");
        assert_eq!(expansion.leaves.len(), 3, "root, A, C");
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static cross-parent bind of the slotted child")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Late bind D into the SHARED leaf: C is a top-level component of
        // stage2, so its slot addresses directly off its terminal offset.
        let sub_d = stage2
            .static_dynamic_overlay
            .as_ref()
            .expect("stage2 overlay")
            .segmented_parser_components[1]
            .terminal_offset
            + terminal(&child_c, "SUB_D");
        let bind3 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_d,
            additional_placeholder_terminals: &[],
            constraint: &child_d,
        }];
        // White-box pin: the shared child keeps both parents, D hangs under
        // the shared child alone (shape pin, numbering-independent), and the
        // occurrence oracle repeats the whole shared subtree once per parent
        // path (leaves append in bind order: root, A, C, D).
        let expansion3 =
            crate::compiler::boundary_walk::expand_nested_link_leaves(&stage2, &bind3)
                .expect("expansion3");
        assert_eq!(expansion3.leaves.len(), 4, "root, A, C, D");
        let parents_of = |leaf: usize| -> Vec<u32> {
            let mut parents: Vec<u32> = expansion3
                .links
                .iter()
                .filter(|link| link.child_component as usize == leaf)
                .map(|link| link.parent_component)
                .collect();
            parents.sort_unstable();
            parents.dedup();
            parents
        };
        let mut shared = None;
        for leaf in 0..expansion3.leaves.len() {
            if parents_of(leaf).len() == 2 {
                assert!(shared.is_none(), "exactly one shared child");
                shared = Some(leaf);
            }
        }
        let shared = shared.expect("shared child keeps two parents");
        let mut under_shared = Vec::new();
        for leaf in 0..expansion3.leaves.len() {
            if parents_of(leaf) == vec![shared as u32] {
                under_shared.push(leaf);
            }
        }
        assert_eq!(
            under_shared.len(),
            1,
            "one descendant hangs under the shared child alone"
        );
        let dfs_order3 =
            crate::compiler::boundary_walk::nested_splice_leaf_order(&expansion3)
                .expect("occurrence oracle accepts the shared subtree");
        assert_eq!(
            dfs_order3,
            vec![0, 1, 2, 3, 2, 3],
            "shared subtree occurs once per parent path"
        );
        let stage3 = compose_constraints_owned_parent_segmented(
            stage2.clone(),
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static late bind into the shared leaf")
        .constraint;
        let stage3_dyn = compose_constraints_owned_parent_segmented(
            stage2_dyn,
            &bind3,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind3 dynamic")
        .constraint;
        // Static shard authority on the twice-extended artifact.
        {
            let overlay = stage3
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
        }
        // Byte-prefix differential across the C-to-D call site.
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'y'],
            vec![b'L', b'x', b'y', b'z'],
            vec![b'L', b'x', b'y', b'z', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'z'],
            vec![b'R', b'y', b'z', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'x'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("xsubtree-bytes", &stage3, &stage3_dyn, &prefixes);
        // Token-level differential (multibyte CALL/RETURN crossings).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, XY],
            vec![LX, Y],
            vec![L, X, Y],
            vec![L, X, Y, Z],
            vec![L, X, Y, Z, X],
            vec![LX, Y, Z, X],
            vec![L, XY, Z, X],
            vec![L, X, YZ, X],
            vec![L, X, Y, ZX],
            vec![R, Y],
            vec![RY, Y],
            vec![R, Y, Z],
            vec![RY, Z],
            vec![R, YZ],
            vec![R, Y, Z, Y],
            vec![RY, Z, Y],
            vec![R, YZ, Y],
            vec![L, Y],
            vec![R, X],
            vec![Z],
        ];
        assert_composed_token_masks_equal("xsubtree-tokens", &stage3, &stage3_dyn, &paths);
        // Fused/single equivalence across every CALL/RETURN site.
        for (fused, single) in [
            (vec![LX, Y, Z, X], vec![L, X, Y, Z, X]),
            (vec![L, XY, Z, X], vec![L, X, Y, Z, X]),
            (vec![L, X, YZ, X], vec![L, X, Y, Z, X]),
            (vec![L, X, Y, ZX], vec![L, X, Y, Z, X]),
            (vec![RY, Z, Y], vec![R, Y, Z, Y]),
            (vec![R, YZ, Y], vec![R, Y, Z, Y]),
        ] {
            let mut st_fused = stage3.start();
            for &token in &fused {
                assert!(st_fused.commit_token(token));
            }
            let mut st_single = stage3.start();
            for &token in &single {
                assert!(st_single.commit_token(token));
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        // Both parents genuinely reach the shared subtree on both backends.
        for path in [vec![L, X, Y, Z, X], vec![R, Y, Z, Y]] {
            let mut st = stage3_dyn.start();
            for &token in &path {
                assert!(st.commit_token(token));
            }
            assert!(st.is_accepting(), "oracle must accept {path:?}");
            let mut st = stage3.start();
            for &token in &path {
                assert!(st.commit_token(token));
            }
            assert!(st.is_accepting(), "static must accept {path:?}");
        }
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage3_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || !st.commit_token(token) {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "shared-subtree corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage3.start();
            for &token in path {
                assert!(st.commit_token(token), "static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(
            mismatches, 0,
            "shared-subtree static must match dynamic on every path"
        );
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage3.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage3_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                assert!(dyn_st.commit_token(token));
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                assert!(abl_st.commit_token(token));
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_extend_with_whitespace_aliases_matches_dynamic() {
        if crate::isolate_environment_test(false) { return; }
        // Same deep-debug-stack headroom as the branching/three-bind fixtures
        // (measured there): 3-level multibyte static prime/mask exceeds the
        // default 2MB.
        std::thread::Builder::new()
            .name("whitespace-aliases".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(composed_parent_whitespace_aliases_body)
            .expect("spawn whitespace fixture worker")
            .join()
            .expect("whitespace fixture worker panicked");
    }

    /// Deep-stack worker body for the whitespace-alias test (see above).
    fn composed_parent_whitespace_aliases_body() {
        // Late bind over alias-bearing overlays: every grammar declares an
        // identical whitespace ignore, so each composed overlay carries REAL
        // `global_terminal_aliases` (child ignores folded into the canonical
        // parent ignore). The static extension must still succeed with
        // dynamic-identical behavior: slots are non-vocabulary placeholder
        // sentinels while alias keys are vocabulary-matching ignore terminals,
        // so no legitimate slot can name an alias key (placeholder validation
        // rejects vocabulary-matching terminals) and resolution takes the
        // interval path. Success alongside provably non-empty aliases proves
        // the interval path narrowly (any alias hit would decline loudly).
        // Multibyte tokens cross CALL/RETURN with whitespace flowing across
        // component boundaries on both sides of the late bind.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const Z: u32 = 4;
        const SP: u32 = 5;
        const LX: u32 = 6;
        const XZ: u32 = 7;
        const ZX: u32 = 8;
        const RY: u32 = 9;
        const YY: u32 = 10;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"z".to_vec()),
            (5, b" ".to_vec()),
            (6, b"Lx".to_vec()),
            (7, b"xz".to_vec()),
            (8, b"zx".to_vec()),
            (9, b"Ry".to_vec()),
            (10, b"yy".to_vec()),
        ]);
        let root = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore WS;
                extern token SUB_A;
                extern token SUB_B;
                t WS = " "+;
                nt document = "L" SUB_A "x" | "R" SUB_B "y";
            "#,
            &vocab, &[("SUB_A", 999), ("SUB_B", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_with_token_ids(
            r#"
                start a;
                ignore WS;
                extern token SUB_INNER;
                t WS = " "+;
                nt a = "x" SUB_INNER;
            "#,
            &vocab, &[("SUB_INNER", 996)], &[])
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                ignore WS;
                t WS = " "+;
                nt b = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let child_c = Constraint::from_glrm_grammar(
            r#"
                start c;
                ignore WS;
                t WS = " "+;
                nt c = "z";
            "#,
            &vocab,
        )
        .unwrap();
        for (name, constraint) in
            [("root", &root), ("A", &child_a), ("B", &child_b), ("C", &child_c)]
        {
            assert!(
                constraint.ignore_terminal.is_some(),
                "{name} must carry a real ignore terminal",
            );
        }
        let bind1a = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&root, "SUB_A"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1a = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1a")
        .constraint;
        let stage1a_dyn = compose_constraints_owned_parent_segmented(
            root.clone(),
            &bind1a,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1a dynamic")
        .constraint;
        // bind1a's overlay carries a REAL alias: A's whitespace folded into
        // the canonical root whitespace (exact producer math).
        let canonical_a = terminal(&root, "WS");
        {
            let overlay = stage1a
                .static_dynamic_overlay
                .as_ref()
                .expect("stage1a segmented overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            assert!(
                overlay.segmented_parser_components[0]
                    .global_terminal_aliases
                    .is_empty(),
                "canonical component carries no alias entries",
            );
            assert_eq!(
                overlay.segmented_parser_components[1].global_terminal_aliases,
                Vec::<(u32, u32)>::new(),
                "native child keeps its local ignore coordinate",
            );
            // The alias key matches vocabulary (whitespace) while the bound
            // slot is a non-vocabulary sentinel: no overlap by construction.
            assert_eq!(root.ignore_terminal, Some(canonical_a));
            assert_eq!(root.ignore_expr, child_a.ignore_expr,
                "component-local ignores retain the same whitespace language");
            assert_ne!(
                bind1a[0].placeholder_terminal, canonical_a,
                "bound slot must not name the alias key",
            );
        }
        let bind1b = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1a, "SUB_B"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            stage1a.clone(),
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1b")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            stage1a_dyn,
            &bind1b,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1b dynamic")
        .constraint;
        // bind1b's overlay likewise folds B's whitespace into the same
        // canonical ignore (component 0 starts at terminal 0).
        {
            let overlay = stage1
                .static_dynamic_overlay
                .as_ref()
                .expect("stage1 segmented overlay");
            assert_eq!(
                overlay.segmented_parser_components[1].global_terminal_aliases,
                Vec::<(u32, u32)>::new(),
                "native second child keeps its local ignore coordinate",
            );
            assert_ne!(
                bind1b[0].placeholder_terminal, canonical_a,
                "bound slot must not name the alias key",
            );
        }
        let sub_inner = {
            let overlay = stage1.static_dynamic_overlay.as_ref().expect("overlay");
            let nested = &overlay.segmented_parser_components[0];
            let nested_overlay = nested
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("nested overlay");
            nested.terminal_offset
                + nested_overlay.segmented_parser_components[1].terminal_offset
                + terminal(&child_a, "SUB_INNER")
        };
        // The late-bind slot (in stage1's composed coordinate) also cannot
        // name the alias key: non-vocabulary sentinel vs whitespace.
        assert_ne!(
            sub_inner, canonical_a,
            "late-bind slot must not name the alias key",
        );
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: sub_inner,
            additional_placeholder_terminals: &[],
            constraint: &child_c,
        }];
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2 over alias-bearing parent must extend statically")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        // Static shard authority on the extended artifact.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
        }
        // Byte-prefix differential with whitespace flowing across boundaries.
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'z'],
            vec![b'L', b'x', b'z', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b' '],
            vec![b'L', b' '],
            vec![b'L', b' ', b'x'],
            vec![b'L', b'x', b' ', b'z', b'x'],
            vec![b'L', b'x', b'z', b' ', b'x'],
            vec![b'R', b' ', b'y', b'y'],
            vec![b'L', b'y'],
            vec![b'z'],
        ];
        assert_composed_masks_equal("ws-alias-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (multibyte crossings + whitespace tokens).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![SP],
            vec![L, X],
            vec![L, SP, X],
            vec![LX, SP],
            vec![L, X, Z],
            vec![L, XZ],
            vec![LX, Z],
            vec![LX, ZX],
            vec![L, X, Z, X],
            vec![LX, SP, ZX],
            vec![L, SP, X, SP, Z, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![R, SP, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![Z],
        ];
        assert_composed_token_masks_equal("ws-alias-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence (whitespace-free pairs, proven pattern).
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XZ], vec![L, X, Z]),
            (vec![LX, ZX], vec![L, X, Z, X]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                assert!(st_fused.commit_token(token));
            }
            let mut st_single = stage2.start();
            for &token in &single {
                assert!(st_single.commit_token(token));
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        // Whitespace genuinely flows: the oracle accepts spaced completions.
        for path in [vec![L, SP, X, Z, X], vec![R, SP, Y, Y]] {
            let mut st = stage2_dyn.start();
            for &token in &path {
                assert!(st.commit_token(token));
            }
            assert!(st.is_accepting(), "oracle must accept spaced {path:?}");
        }
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || !st.commit_token(token) {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "whitespace corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                assert!(st.commit_token(token), "static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "whitespace static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                assert!(dyn_st.commit_token(token));
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                assert!(abl_st.commit_token(token));
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
    }

    #[test]
    fn composed_parent_static_extend_with_reused_sibling() {
        // DAG reuse: the same child constraint fills slots on both binds. Leaf
        // expansion keys visits by position (never a global seen set), so the
        // shared child expands once per use with distinct leaves.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token SUB2;
                nt document = "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab, &[("SUB", 999), ("SUB2", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a = "x";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent.clone(),
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let expansion = crate::compiler::boundary_walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        assert_eq!(expansion.leaves.len(), 3, "root plus two uses of the shared child");
        assert_eq!(
            expansion.top_leaf_ranges,
            vec![vec![0usize, 1usize], vec![2usize]],
            "each use of the shared child must occupy its own leaf (no identity dedup)",
        );
        assert_eq!(
            expansion.leaves[1].parser_terminal_count(),
            expansion.leaves[2].parser_terminal_count(),
            "both uses expand the same reassembled child table",
        );
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2")
        .constraint;
        let stage1_dyn = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind1 dynamic")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            stage1_dyn,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("bind2 dynamic")
        .constraint;
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'x'],
            vec![b'R'],
            vec![b'R', b'x'],
            vec![b'R', b'x', b'y'],
            vec![b'L', b'y'],
            vec![b'R', b'y'],
        ];
        assert_composed_masks_equal("reused-sibling", &stage2, &stage2_dyn, &prefixes);
    }

    #[test]
    fn composed_parent_static_extend_after_serialization_round_trip() {
        if crate::isolate_environment_test(false) { return; }
        // Components and links survive save/load, so a loaded composed parent
        // extends statically with identical results. Multibyte tokens cross
        // CALL (Lx, Ry) and RETURN (xx, yy) on both the pre- and post-serde
        // links.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        const L: u32 = 0;
        const R: u32 = 1;
        const X: u32 = 2;
        const Y: u32 = 3;
        const LX: u32 = 4;
        const XX: u32 = 5;
        const RY: u32 = 6;
        const YY: u32 = 7;
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"Lx".to_vec()),
            (5, b"xx".to_vec()),
            (6, b"Ry".to_vec()),
            (7, b"yy".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token SUB2;
                nt document = "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab, &[("SUB", 999), ("SUB2", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a = "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bytes = stage1.save();
        let loaded = Constraint::load(bytes.as_slice(), &vocab).expect("load composed parent");
        assert!(
            loaded.uses_compact_segmented_parser_runtime(),
            "loaded parent must retain its segmented tree",
        );
        // The loaded overlay must carry bind1's link intact (slot, child, and
        // alias-free components), since the static extend resolves slots
        // through it.
        {
            let overlay = loaded
                .static_dynamic_overlay
                .as_ref()
                .expect("loaded segmented overlay");
            assert_eq!(
                overlay.segmented_parser_components.len(),
                2,
                "loaded parent keeps [root, A]",
            );
            let [link] = overlay.segmented_parser_links.as_slice() else {
                panic!(
                    "loaded parent must retain bind1's single link, got {:?}",
                    overlay.segmented_parser_links.len(),
                );
            };
            assert_eq!(link.parent_component, 0);
            assert_eq!(link.slot_terminal, bind1[0].placeholder_terminal);
            assert_eq!(link.child_component, 1);
            assert!(
                overlay
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.global_terminal_aliases.is_empty()),
                "loaded components must stay alias-free",
            );
        }
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&loaded, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage2 = compose_constraints_owned_parent_segmented(
            loaded.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("static extend after load")
        .constraint;
        let stage2_dyn = compose_constraints_owned_parent_segmented(
            loaded,
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::Dynamic,
        )
        .expect("dynamic extend after load")
        .constraint;
        // Static shard authority on the post-serde extend: both tops hold
        // StaticParser shards and no redundant global boundary parser. The
        // loaded parent block retains its own static inner repair.
        {
            let overlay = stage2
                .static_dynamic_overlay
                .as_ref()
                .expect("static overlay");
            assert_eq!(overlay.segmented_parser_components.len(), 2);
            for (index, component) in overlay.segmented_parser_components.iter().enumerate() {
                let shard = component.boundary.as_ref().unwrap_or_else(|| {
                    panic!("installed top component {index} must carry a static shard")
                });
                assert!(
                    matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ),
                    "installed component {index} must be a StaticParser shard",
                );
            }
            let inner = overlay.segmented_parser_components[0]
                .constraint
                .static_dynamic_overlay
                .as_ref()
                .expect("parent-block overlay");
            assert!(
                inner
                    .segmented_parser_components
                    .iter()
                    .all(|component| component.boundary.as_ref().is_none_or(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    ))),
                "parent-block overlay must retain only static inner shards",
            );
            assert!(
                inner.boundary_shards().next().is_some()
                    && inner.boundary_shards().all(|shard| matches!(
                        shard.backend,
                        crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
                    )),
                "parent-block overlay static shard list must survive the post-serde extension",
            );
        }
        // Byte-prefix differential (single-byte commits).
        let prefixes: Vec<Vec<u8>> = vec![
            vec![],
            vec![b'L'],
            vec![b'L', b'x'],
            vec![b'L', b'x', b'x'],
            vec![b'R'],
            vec![b'R', b'y'],
            vec![b'R', b'y', b'y'],
            vec![b'L', b'y'],
        ];
        assert_composed_masks_equal("post-serde-extend-bytes", &stage2, &stage2_dyn, &prefixes);
        // Token-level differential (whole multibyte commits across CALL/RETURN).
        let paths: Vec<Vec<u32>> = vec![
            vec![],
            vec![L],
            vec![R],
            vec![LX],
            vec![RY],
            vec![L, X],
            vec![L, XX],
            vec![LX, X],
            vec![L, X, X],
            vec![R, Y],
            vec![RY, Y],
            vec![R, YY],
            vec![R, Y, Y],
            vec![L, Y],
            vec![R, X],
            vec![X],
        ];
        assert_composed_token_masks_equal("post-serde-extend-tokens", &stage2, &stage2_dyn, &paths);
        // Fused/single equivalence over the extended (deeper) artifact.
        for (fused, single) in [
            (vec![LX], vec![L, X]),
            (vec![RY], vec![R, Y]),
            (vec![L, XX], vec![L, X, X]),
            (vec![LX, X], vec![L, X, X]),
            (vec![R, YY], vec![R, Y, Y]),
        ] {
            let mut st_fused = stage2.start();
            for &token in &fused {
                assert!(st_fused.commit_token(token));
            }
            let mut st_single = stage2.start();
            for &token in &single {
                assert!(st_single.commit_token(token));
            }
            assert_eq!(
                st_fused.mask(),
                st_single.mask(),
                "fused/single mask equivalence for {fused:?} vs {single:?}",
            );
            assert_eq!(
                st_fused.is_accepting(),
                st_single.is_accepting(),
                "fused/single acceptance equivalence for {fused:?} vs {single:?}",
            );
        }
        assert!(admits_token(&stage2.start().mask(), LX), "Lx crossing must be live");
        assert!(admits_token(&stage2.start().mask(), RY), "Ry crossing must be live");
        // Trap-armed replay of the oracle-gated corpus.
        let mut nodes: Vec<(Vec<u32>, Vec<u32>)> = Vec::new();
        for path in &paths {
            let mut st = stage2_dyn.start();
            let mut ok = true;
            for &token in path {
                if !admits_token(&st.mask(), token) || !st.commit_token(token) {
                    ok = false;
                    break;
                }
            }
            if ok {
                nodes.push((path.clone(), st.mask()));
            }
        }
        assert!(
            nodes.len() >= 10,
            "post-serde corpus must cover real positions, got {}",
            nodes.len()
        );
        let mut trap = StrictStaticTrapGuard::arm();
        let mut mismatches = 0usize;
        for (path, mask) in &nodes {
            let mut st = stage2.start();
            for &token in path {
                assert!(st.commit_token(token), "static replay");
            }
            if st.mask() != *mask {
                mismatches += 1;
            }
        }
        trap.disarm();
        assert_eq!(mismatches, 0, "post-serde static must match dynamic on every path");
        // Ablation: clearing the installed top-level shards must lose
        // admissions.
        let mut ablated = stage2.clone();
        install_published_static_boundary_shards(
            ablated.static_dynamic_overlay.as_mut().expect("overlay"),
            Vec::new(),
        )
        .expect("clear shards");
        let mut strict = 0usize;
        for path in &paths {
            let mut dyn_st = stage2_dyn.start();
            let mut abl_st = ablated.start();
            for &token in path {
                if !admits_token(&dyn_st.mask(), token) {
                    break;
                }
                assert!(dyn_st.commit_token(token));
                if !admits_token(&abl_st.mask(), token) {
                    strict += 1;
                    break;
                }
                assert!(abl_st.commit_token(token));
            }
        }
        assert!(strict >= 1, "ablation must lose at least one admission");
        // The extended artifact itself round-trips with identical behavior.
        let reloaded =
            Constraint::load(stage2.save().as_slice(), &vocab).expect("reload extended composition");
        assert_composed_masks_equal("post-serde-reload-bytes", &reloaded, &stage2_dyn, &prefixes);
        assert_composed_token_masks_equal(
            "post-serde-reload-tokens",
            &reloaded,
            &stage2_dyn,
            &paths,
        );
    }

    #[test]
    fn composed_parent_static_extend_pins_runtime_leaf_layout() {
        // The installing runtime must visit the extended link's leaves in the
        // walk's preorder (parent block first): pin terminal offsets exactly
        // and tokenizer offsets structurally against the leaf expansion.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token SUB2;
                nt document = "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab, &[("SUB", 999), ("SUB2", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a = "x";
            "#,
            &vocab,
        )
        .unwrap();
        let child_b = Constraint::from_glrm_grammar(
            r#"
                start b;
                nt b = "y";
            "#,
            &vocab,
        )
        .unwrap();
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &child_b,
        }];
        let stage2 = compose_constraints_owned_parent_segmented(
            stage1.clone(),
            &bind2,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind2")
        .constraint;
        let expansion = crate::compiler::boundary_walk::expand_nested_link_leaves(&stage1, &bind2)
            .expect("bind2 expansion");
        let layout = stage2
            .recursive_parser_layout()
            .expect("layout result")
            .expect("recursive layout present");
        assert_eq!(
            layout.leaf_terminal_offsets, expansion.leaf_terminal_offsets,
            "runtime terminal preorder must match the walk leaf order",
        );
        assert_eq!(
            layout.total_leaf_terminals, expansion.num_terminals,
            "runtime terminal total must match the walk domain",
        );
        assert_eq!(
            layout.leaf_tokenizer_state_offsets.len(),
            expansion.leaves.len(),
            "one tokenizer block per leaf",
        );
        assert!(
            layout
                .leaf_tokenizer_state_offsets
                .windows(2)
                .all(|pair| pair[0] < pair[1]),
            "tokenizer blocks must pack back-to-back in leaf order: {:?}",
            layout.leaf_tokenizer_state_offsets,
        );
        assert!(
            layout.total_tokenizer_states
                > *layout.leaf_tokenizer_state_offsets.last().unwrap_or(&0),
            "tokenizer total must cover the packed blocks",
        );
    }

    #[test]
    fn composed_parent_static_extend_with_nullable_child_matches_monolithic() {
        // Native finite control closure supports a nullable later binding.
        // Compare its masks and commits with an independently compiled grammar.
        let _env_lock = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"L".to_vec()),
            (1, b"R".to_vec()),
            (2, b"x".to_vec()),
            (3, b"y".to_vec()),
            (4, b"a".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token SUB;
                extern token SUB2;
                nt document = "L" SUB "x" | "R" SUB2 "y";
            "#,
            &vocab, &[("SUB", 999), ("SUB2", 998)], &[])
        .unwrap();
        let child_a = Constraint::from_glrm_grammar(
            r#"
                start a;
                nt a = "x";
            "#,
            &vocab,
        )
        .unwrap();
        let nullable_b = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt item = "a";
                nt child = item?;
            "#,
            &vocab,
        )
        .unwrap();
        assert!(
            nullable_b.composition_start_nullable().unwrap(),
            "pin fixture must stay effectively nullable",
        );
        let bind1 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&parent, "SUB"),
            additional_placeholder_terminals: &[],
            constraint: &child_a,
        }];
        let stage1 = compose_constraints_owned_parent_segmented(
            parent,
            &bind1,
            &vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
        .expect("bind1")
        .constraint;
        let bind2 = [CompiledSubgrammarInput {
            placeholder_terminal: terminal(&stage1, "SUB2"),
            additional_placeholder_terminals: &[],
            constraint: &nullable_b,
        }];
        let composed = compose_constraints_owned_parent_segmented(
            stage1, &bind2, &vocab, SegmentedBoundaryBackend::StaticParserDwa,
        ).expect("nullable second binding").constraint;
        let monolithic = Constraint::from_glrm_grammar(
            r#"start document; nt document = "L" "x" "x" | "R" "a"? "y";"#,
            &vocab,
        ).unwrap();
        assert_constraints_equivalent_on_reachable_prefixes(&composed, &monolithic, &vocab, 4);
    }

    #[test]
    fn internal_composition_matches_monolithic_for_child_alternatives() {
        let vocab = Vocab::from_entries_unchecked(vec![
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
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                extern token LEFT;
                extern token RIGHT;
                nt document = "X" (LEFT | RIGHT) "!";
            "#,
            &vocab, &[("LEFT", 998), ("RIGHT", 999)], &[])
        .unwrap();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "a" "b";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child = "c" "d";
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                grammar left {
                    start child;
                    nt child = "a" "b";
                };
                grammar right {
                    start child;
                    nt child = "c" "d";
                };
                nt document = "X" (left | right) "!";
            "#,
            &vocab,
        )
        .unwrap();
        let duplicate_error = parent
            .compose_linked_children_for_test(&[("LEFT", &left), ("LEFT", &right)], &vocab)
            .expect_err("duplicate placeholder inputs must be rejected");
        assert!(duplicate_error
            .to_string()
            .contains("was supplied more than once"));
        let composed = parent
            .compose_linked_children_for_test(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
            .unwrap();
        let loaded = Constraint::load(composed.save(), &vocab)
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
                assert!(actual.commit_token(token));
                assert!(roundtripped.commit_token(token));
                assert!(expected.commit_token(token));
            }
            assert!(actual.is_accepting());
            assert!(roundtripped.is_accepting());
            assert!(expected.is_accepting());
        }

        let mut crossed = composed.start();
        assert!(crossed.commit_token(0));
        assert!(!crossed.commit_token(3) || !crossed.is_accepting());
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
        Vocab::from_entries_unchecked(entries)
    }

    fn run_sizeable_json_schema_composition_benchmark(
        mode: &str,
        parent_source: &str,
        token_ids: &[(&str, u32)],
    ) {
        let vocab = composition_benchmark_vocab();
        let choices = crate::env::var("GLRMASK_COMPOSE_BENCH_CHOICES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(7500);
        let left_schema = sizeable_json_schema("left", choices);
        let right_schema = sizeable_json_schema("right", choices);

        eprintln!("[compose-bench] compile {mode} parent start");
        let parent =
            Constraint::from_glrm_with_token_ids(parent_source, &vocab, token_ids, &[]).unwrap();
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

        let composition_runs = crate::env::var("GLRMASK_COMPOSE_BENCH_RUNS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(3)
            .max(1);
        let mut composition_samples_ms = Vec::with_capacity(composition_runs);
        for run in 0..composition_runs {
            let composition_started = Instant::now();
            eprintln!("[compose-bench] compose {mode} run={} start", run + 1);
            let composed = parent
                .compose_linked_children_for_test(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
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
        let vocab = Vocab::from_entries_unchecked(vec![
            (0, b"X".to_vec()),
            (1, b" ".to_vec()),
            (2, b"\t".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
            (5, b"X \ta\t !".to_vec()),
        ]);
        let parent = Constraint::from_glrm_with_token_ids(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                extern token SUB;
                nt document = "X" SUB "!";
            "#,
            &vocab, &[("SUB", 999)], &[])
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                ignore CHILD_WS;
                t CHILD_WS = "\t"+;
                nt child = "a";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS = " "+;
                grammar child {
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS = "\t"+;
                    nt child = "a";
                };
                nt document = "X" child "!";
            "#,
            &vocab,
        )
        .unwrap();

        let runs = crate::env::var("GLRMASK_COMPOSE_RUNTIME_BENCH_RUNS")
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
                assert!(state.is_accepting());
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
                extern token LEFT;
                extern token RIGHT;
                nt document = "{" "\"value1\": " LEFT ", \"value2\": " RIGHT "}";
            "#,
            &[("LEFT", 1000000), ("RIGHT", 1000001)],
        );
    }

    #[test]
    #[ignore]
    fn sizeable_json_schema_alternative_composition_benchmark_probe() {
        run_sizeable_json_schema_composition_benchmark(
            "alternative",
            r#"
                start document;
                extern token LEFT;
                extern token RIGHT;
                nt document = "{" ("\"left\": " LEFT | "\"right\": " RIGHT) "}";
            "#,
            &[("LEFT", 1000000), ("RIGHT", 1000001)],
        );
    }
}
