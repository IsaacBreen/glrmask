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
use std::time::Instant;
use smallvec::SmallVec;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::lexer::{Lexer, tokenizer::Tokenizer};
use crate::automata::lexer::ast::Expr;
use crate::automata::weighted_u32::determinize::determinize;
use crate::automata::weighted_u32::minimize::{minimize_owned, reverse_hashcons_owned};
use crate::automata::weighted_u32::equivalence::find_difference;
use crate::automata::weighted_u32::dwa::{DWA, DWAState};
use crate::automata::weighted_u32::nwa::{NWA, NWAState};
use crate::automata::weighted_u32::terminal_automaton::TerminalAutomaton;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::{
    DEFAULT_LABEL, encode_negative_label, encode_positive_label, is_negative_label,
    negative_to_positive_label,
};
use crate::compiler::stages::equiv_types::{
    InternalIdMap, ManyToOneIdMap, MappedArtifact,
};
use crate::compiler::stages::mapped_artifact::{WeightRefs, remap_weights_with_maps};
use crate::compiler::stages::id_map_and_terminal_dwa::classify::vocab_tokens_with_adjacent_pairs;
use crate::compiler::stages::parser_dwa::{
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table_with_bundle_cache,
    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles,
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    PrebuiltParserBundleCache,
    normalize_weighted_parser_stack_nwa,
    normalize_weighted_parser_stack_nwa_small_boundary,
    normalize_weighted_parser_stack_nwa_small_boundary_for_parser_state_count,
    normalize_weighted_parser_stack_nwa_small_boundary_with_tsid_map,
    normalize_signed_weighted_parser_stack_nwa_small_boundary,
    normalize_signed_weighted_parser_stack_nwa_small_boundary_for_parser_state_count,
    resolve_negative_codes_small_boundary,
};
use glrmask_parser_dwa::__private::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::templates::characterize::{
    characterize_selected_terminals,
    try_characterize_selected_terminals_for_terminal_count,
    try_characterize_terminal_action_state_seeds_for_terminal_count,
    try_characterize_terminal_nt_predecessor_seeds_for_terminal_count,
};
use crate::compiler::stages::templates::compile_dfa::{
    specialize_template_dfa_defaults_for_commit_split_input,
    try_split_commit_template_dfas,
};
use crate::compiler::stages::templates::Templates;
use crate::compiler::constraint_possible_matches::{
    runtime_dynamic_vocab_for_vocab,
};
use crate::compiler::glr::table::{
    Action,
    ComposedTable,
    ControlEliminationReport,
};
use crate::grammar::flat::Symbol;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::weight::{
    ScopedWeightOpCache,
    Weight,
};
use crate::runtime::{
    CompositionGrammarSummary, Constraint, ConstraintRuntimeBackend, SpecialTokenTerminal,
};
use crate::Vocab;
use super::{macro_parallelism_disabled, report_macro_item_timings};

#[inline]
pub(crate) fn compose_profile_enabled() -> bool {
    crate::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
        || crate::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

/// The generic parser-DWA builder currently skips weighted minimization unless
/// explicitly overridden. Boundary overlays are different: leaving their large
/// intermediate DWA unminimized makes the subsequent component union explode.
fn parser_builder_skips_internal_minimization() -> bool {
    crate::env::var("GLRMASK_SKIP_PARSER_DWA_MINIMIZE")
        .ok()
        .map(|value| {
            let trimmed = value.trim();
            !(trimmed.is_empty()
                || trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(true)
}

fn boundary_parser_minimize_min_states() -> u32 {
    crate::env::var("GLRMASK_BOUNDARY_PARSER_MINIMIZE_MIN_STATES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(64)
}

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
pub(crate) struct ParserDwaComponent<'a> {
    pub(crate) constraint: &'a Constraint,
    /// Local parser state -> merged parser states.
    pub(crate) parser_state_relation: &'a [Vec<u32>],
    /// Local raw tokenizer state `s` is merged state `tokenizer_state_offset+s`.
    pub(crate) tokenizer_state_offset: u32,
    /// Global terminal offset of this component in the composed table.
    pub(crate) terminal_offset: u32,
}

const NO_PARSER_DOMAIN_LABEL: i32 = i32::MAX;

#[derive(Debug, Clone)]
struct ParserDefaultDomain {
    /// Fallback label for exact component-owned states that had no pre-existing
    /// symbolic parser-domain label in the cached component.
    label: i32,
    base_has_states: bool,
    /// Cached/nested symbolic fallback label -> refined outer fallback label.
    ///
    /// A nested constraint already uses lookup order
    /// `concrete -> nested-domain -> DEFAULT`.  When it is linked again we do
    /// not need to expand that nested domain back to thousands of concrete LR
    /// labels.  Instead, split this component's outer DEFAULT domain by the
    /// cached nested-domain partition.  Every composed parser state still has
    /// exactly one runtime fallback label; row construction resolves
    /// nested-domain-over-DEFAULT precedence onto that refined label.
    nested_labels: BTreeMap<i32, i32>,
    states: BitSet,
    predicted_saved_edges: usize,
}

impl ParserDefaultDomain {
    fn output_labels(&self) -> impl Iterator<Item = i32> + '_ {
        self.base_has_states
            .then_some(self.label)
            .into_iter()
            .chain(self.nested_labels.values().copied())
    }
}

#[derive(Debug, Clone)]
struct ParserDefaultDomainPlan {
    /// Composed parser state -> synthetic fallback label. The sentinel means
    /// that state has no component-local wildcard domain.
    parser_state_labels: Vec<i32>,
    /// One optional exact wildcard domain per component, including the parent.
    component_domains: Vec<Option<ParserDefaultDomain>>,
    predicted_saved_edges: usize,
}

/// Build an exact quotient of the positive parser-state input alphabet for
/// component-local DEFAULT transitions.
///
/// A composed LR state is eligible for child `i` iff it has exactly one
/// preimage `(i, local_state)` across *all* component state relations. This
/// excludes caller/return aliases, same-child many-to-one quotients, and
/// cross-child structural sharing. Eligible domains are therefore pairwise
/// disjoint, and every state in a domain has one unambiguous local-state
/// interpretation. A child DEFAULT can then be represented once by a synthetic
/// label: runtime lookup tries the concrete LR-state label first and the domain
/// label only on a miss. States outside the domain retain ordinary explicit
/// DEFAULT materialization, making this representation exactly equivalent to
/// full materialization on every parser stack, not merely reachable stacks.
fn build_parser_default_domain_plan(
    components: &[ParserDwaComponent<'_>],
    num_parser_states: u32,
) -> ParserDefaultDomainPlan {
    let n = num_parser_states as usize;
    let mut preimage_count = vec![0u32; n];
    let mut owner_component = vec![u32::MAX; n];
    let mut owner_local = vec![u32::MAX; n];

    for (component_index, component) in components.iter().enumerate() {
        for (local_state, targets) in component.parser_state_relation.iter().enumerate() {
            for &target in targets {
                let Some(count) = preimage_count.get_mut(target as usize) else {
                    continue;
                };
                *count = count.saturating_add(1);
                if *count == 1 {
                    owner_component[target as usize] = component_index as u32;
                    owner_local[target as usize] = local_state as u32;
                } else {
                    owner_component[target as usize] = u32::MAX;
                    owner_local[target as usize] = u32::MAX;
                }
            }
        }
    }

    let mut local_multiplicities = components
        .iter()
        .map(|component| vec![0usize; component.parser_state_relation.len()])
        .collect::<Vec<_>>();
    for state in 0..n {
        if preimage_count[state] != 1 {
            continue;
        }
        let component = owner_component[state] as usize;
        let local = owner_local[state] as usize;
        if component >= components.len() {
            continue;
        }
        if let Some(slot) = local_multiplicities
            .get_mut(component)
            .and_then(|rows| rows.get_mut(local))
        {
            *slot += 1;
        }
    }

    const MIN_SAVED_EDGES: usize = 4_096;
    let mut component_predicted = vec![0usize; components.len()];
    for component_index in 0..components.len() {
        let component = components[component_index];
        let domain_total = local_multiplicities[component_index].iter().sum::<usize>();
        if domain_total == 0 {
            continue;
        }
        let source = component.constraint.parser_dwa.for_compilation().expect("validated component DWA");
        for state in source.states() {
            if !state.transitions.contains_key(&DEFAULT_LABEL) {
                continue;
            }
            let explicit_domain = state
                .transitions
                .keys()
                .filter_map(|&label| {
                    (label >= 0 && label != DEFAULT_LABEL).then_some(label as usize)
                })
                .filter_map(|local| local_multiplicities[component_index].get(local))
                .copied()
                .sum::<usize>();
            component_predicted[component_index] = component_predicted[component_index]
                .saturating_add(domain_total.saturating_sub(explicit_domain));
        }
    }

    // A symbolic child domain already pays the one runtime state->domain map
    // lookup. Once that map is selected, representing the parent's exact
    // unambiguous domain as one additional label is purely a graph-size win:
    // it replaces concrete DEFAULT expansion without adding lookup depth.
    // Keep parent-only/small compositions on the historical zero-overhead
    // path.
    let child_feature_selected = component_predicted.iter().skip(1).any(|&n| n >= MIN_SAVED_EDGES);
    let required_domain_labels = (0..components.len())
        .map(|component_index| {
            let component = components[component_index];
            let mut nested = BTreeSet::<i32>::new();
            let mut has_base = false;
            for (local, &multiplicity) in local_multiplicities[component_index].iter().enumerate() {
                if multiplicity == 0 {
                    continue;
                }
                let source_domain = component
                    .constraint
                    .parser_state_domain_labels
                    .get(local)
                    .copied()
                    .unwrap_or(NO_PARSER_DOMAIN_LABEL);
                if source_domain == NO_PARSER_DOMAIN_LABEL {
                    has_base = true;
                } else {
                    nested.insert(source_domain);
                }
            }
            nested.len() + usize::from(has_base)
        })
        .sum::<usize>();
    let labels_fit = (num_parser_states as i64)
        .saturating_add(required_domain_labels as i64)
        < DEFAULT_LABEL as i64;
    let mut next_label = num_parser_states as i32;
    let mut component_domains = vec![None; components.len()];
    let mut parser_state_labels = vec![NO_PARSER_DOMAIN_LABEL; n];
    let mut predicted_saved_edges = 0usize;

    let feature_selected = labels_fit && child_feature_selected;
    if feature_selected {
        // Once one child amortizes the table-sized runtime map, every further
        // exact child domain is a marginal win: one synthetic label replaces a
        // positive number of concrete edges without increasing lookup depth.
        for component_index in 0..components.len() {
            let predicted = component_predicted[component_index];
            if predicted == 0 {
                continue;
            }
            let mut states = BitSet::new(n);
            let mut nested_source_labels = BTreeSet::<i32>::new();
            let mut base_has_states = false;
            for (local, &multiplicity) in local_multiplicities[component_index].iter().enumerate() {
                if multiplicity == 0 {
                    continue;
                }
                let source_domain = components[component_index]
                    .constraint
                    .parser_state_domain_labels
                    .get(local)
                    .copied()
                    .unwrap_or(NO_PARSER_DOMAIN_LABEL);
                if source_domain == NO_PARSER_DOMAIN_LABEL {
                    base_has_states = true;
                } else {
                    nested_source_labels.insert(source_domain);
                }
            }
            // Keep one stable base label even when every uniquely-owned state
            // belongs to a nested domain; it simplifies the transport plan and
            // is never emitted/installed when `base_has_states` is false.
            let label = next_label;
            if base_has_states {
                next_label += 1;
            }
            let mut nested_labels = BTreeMap::<i32, i32>::new();
            for source_label in nested_source_labels {
                let output_label = next_label;
                next_label += 1;
                nested_labels.insert(source_label, output_label);
            }
            for state in 0..n {
                if preimage_count[state] == 1
                    && owner_component[state] == component_index as u32
                {
                    states.set(state);
                    let local = owner_local[state] as usize;
                    let source_domain = components[component_index]
                        .constraint
                        .parser_state_domain_labels
                        .get(local)
                        .copied()
                        .unwrap_or(NO_PARSER_DOMAIN_LABEL);
                    parser_state_labels[state] = if source_domain == NO_PARSER_DOMAIN_LABEL {
                        label
                    } else {
                        *nested_labels
                            .get(&source_domain)
                            .expect("nested source domain was inventoried above")
                    };
                }
            }
            if states.is_empty() {
                continue;
            }
            predicted_saved_edges = predicted_saved_edges.saturating_add(predicted);
            component_domains[component_index] = Some(ParserDefaultDomain {
                label,
                base_has_states,
                nested_labels,
                states,
                predicted_saved_edges: predicted,
            });
        }
    }

    if predicted_saved_edges == 0 {
        // Keep ordinary/small compositions on the zero-overhead runtime path:
        // an empty map makes the exact-label -> DEFAULT lookup identical to
        // pre-v12 constraints.
        parser_state_labels.clear();
    }
    ParserDefaultDomainPlan {
        parser_state_labels,
        component_domains,
        predicted_saved_edges,
    }
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

struct DirectComponentStateCoordinates {
    tokenizer_states: ManyToOneIdMap,
    local_to_global_tsids: Vec<Vec<Vec<u32>>>,
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


#[inline]
fn tokenizer_tsid_relation_is_singleton(constraint: &Constraint) -> bool {
    constraint.tokenizer_tsids.is_singleton()
}

fn build_direct_component_state_coordinates_from_constraints(
    components: &[(&Constraint, u32)],
    merged_tokenizer_state_count: usize,
) -> Result<DirectComponentStateCoordinates, String> {
    let mut state_to_global = vec![u32::MAX; merged_tokenizer_state_count];
    let mut global_to_states = vec![vec![0u32]];
    let mut state_representatives = vec![0u32];
    if let Some(reset) = state_to_global.first_mut() {
        *reset = 0;
    }

    let mut local_to_global_tsids = Vec::with_capacity(components.len());
    for (component_index, &(constraint, tokenizer_state_offset)) in components.iter().enumerate() {
        let component_tokenizer = constraint.composition_tokenizer();
        // Dynamic constraints deliberately omit the static parser-DWA TSID
        // quotient. Preserve that backend and use the exact raw tokenizer
        // coordinate as this component's private TSID coordinate instead of
        // compiling/materializing a quotient solely for linking.
        if constraint.tokenizer_tsids.is_unmapped()
        {
            let local_tsid_count = component_tokenizer.num_states() as usize;
            if local_tsid_count == 0 {
                return Err("component tokenizer contains no states".into());
            }
            let mut local_map = vec![Vec::<u32>::new(); local_tsid_count];
            for local_state in 0..component_tokenizer.num_states() {
                let merged_state = tokenizer_state_offset
                    .checked_add(local_state)
                    .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
                if merged_state == 0 {
                    local_map[local_state as usize].push(0);
                    continue;
                }
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to more than one component class"
                    ));
                }
                let global_tsid = global_to_states.len() as u32;
                *slot = global_tsid;
                state_representatives.push(merged_state);
                global_to_states.push(vec![merged_state]);
                local_map[local_state as usize].push(global_tsid);
            }
            let local_start = component_tokenizer.initial_state() as usize;
            let Some(start_targets) = local_map.get_mut(local_start) else {
                return Err("component start tokenizer state lies outside its raw-state domain".into());
            };
            start_targets.push(0);
            start_targets.sort_unstable();
            start_targets.dedup();
            local_to_global_tsids.push(local_map);
            continue;
        }
        if constraint.tokenizer_tsids.primary().len() != component_tokenizer.num_states() as usize {
            return Err(format!(
                "component {component_index} tokenizer-state map does not cover its runtime tokenizer: mapped_states={} tokenizer_states={} dynamic={}",
                constraint.tokenizer_tsids.primary().len(),
                component_tokenizer.num_states(),
                constraint.uses_dynamic_runtime(),
            ));
        }
        let internal_tsid_to_states = constraint.internal_tsid_groups();
        let local_tsid_count = internal_tsid_to_states.len();
        if local_tsid_count == 0 {
            return Err("component tokenizer-state map contains no internal TSIDs".into());
        }
        let mut local_map = vec![Vec::<u32>::new(); local_tsid_count];

        if tokenizer_tsid_relation_is_singleton(constraint) {
            for (local_tsid, local_states) in internal_tsid_to_states.iter().enumerate() {
                if local_states.is_empty() {
                    continue;
                }
                let mut merged_states = Vec::with_capacity(local_states.len());
                for &local_state in local_states {
                    let merged_state = tokenizer_state_offset.checked_add(local_state)
                        .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
                    if merged_state == 0 {
                        local_map[local_tsid].push(0);
                        continue;
                    }
                    let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                        return Err(format!(
                            "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
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
                .tokenizer_tsids.primary()
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
            continue;
        }

        // A runtime-product tokenizer state can intentionally represent several
        // pre-product TSIDs.  A many-to-one global coordinate cannot identify
        // such a state with *each* constituent TSID independently.  Instead,
        // quotient raw states by their exact local-TSID membership signature.
        // A local TSID then maps to every global signature-class containing it.
        // This is the exact powerset lifting of the local TSID relation and
        // degenerates to the historical one-class-per-local-TSID mapping when
        // every raw state has singleton membership.
        let mut states_by_signature = BTreeMap::<Vec<u32>, Vec<u32>>::new();
        for local_state in 0..component_tokenizer.num_states() {
            let mut signature = constraint.internal_tsids_for_state(local_state).to_vec();
            signature.sort_unstable();
            signature.dedup();
            if signature.is_empty() {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} has no internal TSID"
                ));
            }
            if let Some(&bad) = signature
                .iter()
                .find(|&&tsid| tsid as usize >= local_tsid_count)
            {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} references out-of-range internal TSID {bad}"
                ));
            }
            let merged_state = tokenizer_state_offset.checked_add(local_state)
                .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
            if merged_state as usize >= merged_tokenizer_state_count {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
                ));
            }
            // State zero is the merged reset coordinate. In the owned-parent
            // layout the parent's local reset can physically be state zero;
            // its complete TSID signature is attached to global reset below.
            if merged_state != 0 {
                states_by_signature
                    .entry(signature)
                    .or_default()
                    .push(merged_state);
            }
        }

        for (signature, mut merged_states) in states_by_signature {
            merged_states.sort_unstable();
            merged_states.dedup();
            let global_tsid = global_to_states.len() as u32;
            for &merged_state in &merged_states {
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {merged_state} lies outside merged tokenizer"
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to more than one exact membership class"
                    ));
                }
                *slot = global_tsid;
            }
            state_representatives.push(merged_states[0]);
            global_to_states.push(merged_states);
            for local_tsid in signature {
                local_map[local_tsid as usize].push(global_tsid);
            }
        }

        let local_start = constraint.tokenizer.initial_state();
        let mut start_signature = constraint.internal_tsids_for_state(local_start).to_vec();
        start_signature.sort_unstable();
        start_signature.dedup();
        if start_signature.is_empty() {
            return Err("component start tokenizer state has no internal TSID".into());
        }
        for start_tsid in start_signature {
            let Some(start_targets) = local_map.get_mut(start_tsid as usize) else {
                return Err("component start TSID maps outside its internal TSID domain".into());
            };
            start_targets.push(0);
        }
        for targets in &mut local_map {
            targets.sort_unstable();
            targets.dedup();
        }
        local_to_global_tsids.push(local_map);
    }
    if state_to_global.contains(&u32::MAX) {
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


fn build_direct_component_state_coordinates(
    components: &[ParserDwaComponent<'_>],
    merged_tokenizer_state_count: usize,
) -> Result<DirectComponentStateCoordinates, String> {
    let inputs = components
        .iter()
        .map(|component| (component.constraint, component.tokenizer_state_offset))
        .collect::<Vec<_>>();
    build_direct_component_state_coordinates_from_constraints(&inputs, merged_tokenizer_state_count)
}

fn build_direct_component_token_coordinates(
    components: &[ParserDwaComponent<'_>],
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
    let (vocab_tokens, local_to_global_tokens, _token_coordinate_is_singleton) =
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

fn concrete_local_parser_states_for_label(
    local_state: u32,
    relation: &[Vec<u32>],
    parser_state_domain_labels: &[i32],
) -> Result<Vec<u32>, String> {
    if (local_state as usize) < relation.len() {
        return Ok(vec![local_state]);
    }
    if parser_state_domain_labels.is_empty() {
        return Err(format!("parser-state relation omits local state {local_state}"));
    }
    let synthetic = i32::try_from(local_state)
        .map_err(|_| format!("parser-state label {local_state} exceeds i32 range"))?;
    let states = parser_state_domain_labels
        .iter()
        .enumerate()
        .filter_map(|(state, &label)| (label == synthetic).then_some(state as u32))
        .collect::<Vec<_>>();
    if states.is_empty() {
        return Err(format!(
            "parser-state relation omits local state {local_state} and no stored domain expands it"
        ));
    }
    Ok(states)
}

fn mapped_labels(
    label: i32,
    relation: &[Vec<u32>],
    parser_state_domain_labels: &[i32],
) -> Result<Vec<i32>, String> {
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
    let local_states = concrete_local_parser_states_for_label(
        local_state,
        relation,
        parser_state_domain_labels,
    )?;
    let mut mapped = Vec::new();
    for local_state in local_states {
        let targets = relation
            .get(local_state as usize)
            .ok_or_else(|| format!("parser-state relation omits local state {local_state}"))?;
        if targets.is_empty() {
            return Err(format!("parser-state relation maps local state {local_state} nowhere"));
        }
        mapped.extend(targets.iter().copied());
    }
    mapped.sort_unstable();
    mapped.dedup();
    Ok(mapped
        .into_iter()
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
    parser_state_domain_labels: &[i32],
) -> Result<(), String> {
    for label in mapped_labels(local_label, relation, parser_state_domain_labels)? {
        nwa.add_transition(from, label, target, weight.clone());
    }
    Ok(())
}

fn materialized_top_acceptance(constraint: &Constraint, weights: &crate::runtime::NonDwaWeights<Weight>) -> BTreeMap<i32, Weight> {
    let mut result = BTreeMap::new();
    let default_combined = weights.parser_top_accept.get(&DEFAULT_LABEL);
    let default_parts = weights.parser_top_accept_parts.get(&DEFAULT_LABEL);
    for parser_state in 0..constraint.parser_symbol_count() {
        let label = encode_positive_label(parser_state);
        let mut parts = Vec::<Weight>::new();
        if let Some(weight) = weights.parser_top_accept
            .get(&label)
            .or_else(|| {
                constraint
                    .parser_state_domain_label(parser_state)
                    .and_then(|domain| weights.parser_top_accept.get(&domain))
            })
            .or(default_combined)
        {
            parts.push(weight.clone());
        }
        if let Some(weights) = weights.parser_top_accept_parts
            .get(&label)
            .or_else(|| {
                constraint
                    .parser_state_domain_label(parser_state)
                    .and_then(|domain| weights.parser_top_accept_parts.get(&domain))
            })
            .or(default_parts)
        {
            parts.extend(weights.iter().cloned());
        }
        if let Some(row) = constraint.parser_advance_row(parser_state) {
            for terminal in row.iter_ones() {
                if let Some(weight) = weights.direct_regular_l1_complete_by_terminal
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

fn add_component_parser_state_transitions(
    nwa: &mut NWA,
    source_state: u32,
    source: &DWAState,
    parser_state_relation: &[Vec<u32>],
    parser_state_domain_labels: &[i32],
    num_local_parser_states: u32,
    default_domain: Option<&ParserDefaultDomain>,
) -> Result<(), String> {
    let mut explicit_positive = BTreeSet::new();
    for &label in source.transitions.keys() {
        if label < 0 || label == DEFAULT_LABEL {
            continue;
        }
        for local_state in concrete_local_parser_states_for_label(
            label as u32,
            parser_state_relation,
            parser_state_domain_labels,
        )? {
            explicit_positive.insert(local_state);
        }
    }

    for (&label, (target, weight)) in &source.transitions {
        if label == DEFAULT_LABEL {
            continue;
        }
        if let Some(domain) = default_domain
            && let Some(&refined_label) = domain.nested_labels.get(&label)
        {
            // Preserve the cached inner symbolic domain for all local states
            // that remain uniquely owned after this outer composition.
            nwa.add_transition(
                source_state,
                refined_label,
                *target,
                weight.clone(),
            );
            // Any local member of the inner domain that is not in the exact
            // outer symbolic domain still needs its ordinary concrete edge.
            for local_state in concrete_local_parser_states_for_label(
                label as u32,
                parser_state_relation,
                parser_state_domain_labels,
            )? {
                for &mapped_state in parser_state_relation
                    .get(local_state as usize)
                    .ok_or_else(|| {
                        format!("parser-state relation omits local state {local_state}")
                    })?
                {
                    if !domain.states.contains(mapped_state as usize) {
                        nwa.add_transition(
                            source_state,
                            encode_positive_label(mapped_state),
                            *target,
                            weight.clone(),
                        );
                    }
                }
            }
            continue;
        }
        add_transition_for_mapped_label(
            nwa,
            source_state,
            label,
            *target,
            weight,
            parser_state_relation,
            parser_state_domain_labels,
        )?;
    }

    let Some((target, weight)) = source.transitions.get(&DEFAULT_LABEL) else {
        return Ok(());
    };
    if let Some(domain) = default_domain {
        // One refined symbolic label denotes each exact nested-domain class,
        // plus an optional base class for states with no cached inner domain.
        // If the source row already has an explicit nested-domain transition,
        // that transition wins; otherwise the source DEFAULT is installed on
        // the refined nested label. This compiles the lookup chain
        // `concrete -> inner domain -> DEFAULT` into a single outer fallback.
        if domain.base_has_states {
            nwa.add_transition(source_state, domain.label, *target, weight.clone());
        }
        for (&source_domain, &refined_label) in &domain.nested_labels {
            if !source.transitions.contains_key(&source_domain) {
                nwa.add_transition(
                    source_state,
                    refined_label,
                    *target,
                    weight.clone(),
                );
            }
        }
    }
    for local_state in 0..num_local_parser_states {
        if explicit_positive.contains(&local_state) {
            continue;
        }
        let targets = parser_state_relation
            .get(local_state as usize)
            .ok_or_else(|| format!("parser-state relation omits local state {local_state}"))?;
        for &mapped_state in targets {
            if default_domain
                .is_some_and(|domain| domain.states.contains(mapped_state as usize))
            {
                continue;
            }
            nwa.add_transition(
                source_state,
                encode_positive_label(mapped_state),
                *target,
                weight.clone(),
            );
        }
    }
    Ok(())
}

fn component_parser_nwa(
    component: &ParserDwaComponent<'_>,
    default_domain: Option<&ParserDefaultDomain>,
) -> Result<NWA, String> {
    let constraint = component.constraint;
    if component.parser_state_relation.len() != constraint.parser_symbol_count() as usize {
        return Err(format!(
            "parser-state relation has {} rows for a {}-state component table",
            component.parser_state_relation.len(),
            constraint.parser_symbol_count(),
        ));
    }

    let source = constraint.parser_dwa.for_compilation()?;
    let mut nwa = NWA::new(0, 0);
    for _ in source.states() {
        nwa.add_state();
    }
    nwa.set_start_states(vec![source.start_state()]);

    for (state_id, state) in source.states().iter().enumerate() {
        if let Some(final_weight) = &state.final_weight {
            nwa.set_final_weight(state_id as u32, final_weight.clone());
        }
        add_component_parser_state_transitions(
            &mut nwa,
            state_id as u32,
            state,
            component.parser_state_relation,
            &constraint.parser_state_domain_labels,
            constraint.parser_symbol_count(),
            default_domain,
        )?;
    }

    let weights = constraint.non_dwa_weights.materialized_for_compilation()?;
    let top_accept = materialized_top_acceptance(constraint, &weights);
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
                &constraint.parser_state_domain_labels,
            )?;
        }
        nwa.start_states_mut().push(start);
    }
    if compose_profile_enabled() {
        let stored_domain_counts = constraint
            .parser_state_domain_labels
            .iter()
            .copied()
            .filter(|&label| label != NO_PARSER_DOMAIN_LABEL)
            .fold(BTreeMap::<i32, usize>::new(), |mut counts, label| {
                *counts.entry(label).or_default() += 1;
                counts
            });
        eprintln!(
            "[glrmask/profile][constraint_component_parser_transport_shape] source_states={} source_transitions={} source_default_states={} stored_domain_states={} stored_domains={:?} mapped_states={} mapped_transitions={} symbolic_default={} relation_singleton={}",
            source.num_states(),
            source.num_transitions(),
            source
                .states()
                .iter()
                .filter(|state| state.transitions.contains_key(&DEFAULT_LABEL))
                .count(),
            stored_domain_counts.values().sum::<usize>(),
            stored_domain_counts,
            nwa.num_states(),
            nwa.num_transitions(),
            default_domain.is_some(),
            component.parser_state_relation.iter().all(|targets| targets.len() == 1),
        );
    }
    Ok(nwa)
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
    parser: BoundaryParserWork,
    /// Complete exact static partition of B by the component in which model-
    /// token execution starts. `Some` means the partition is authoritative;
    /// `None` means this compiler path did not construct one and runtime must
    /// retain the established global/fallback boundary representation.
    static_boundary_shards: Option<Vec<BoundaryShardWork>>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    commit_templates_deferred: bool,
    composition_parser_templates_by_terminal: Vec<Option<UnweightedDfa>>,
    active_terminals: Vec<bool>,
    /// Conservative first-internal-crossing token summaries indexed by the
    /// component in which model-token execution starts.  These are derived
    /// from the same exact boundary witness discovery used to build B; they
    /// are accelerator metadata only and are never required for correctness.
    boundary_tokens_by_start_component: Option<Vec<Vec<u32>>>,
}

struct BoundaryShardWork {
    start_component: u32,
    candidate_tokens: Arc<[u32]>,
    parser: BoundaryParserWork,
}

fn build_commit_templates_from_raw_templates(
    raw_templates: &[Option<UnweightedDfa>],
) -> Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>> {
    let build = |(terminal, dfa): (usize, &Option<UnweightedDfa>)| {
            let dfa = dfa.as_ref()?;
            let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
            try_split_commit_template_dfas(&commit_dfa)
                .map(|split| (terminal, Arc::new(split)))
    };
    let split_templates = if macro_parallelism_disabled() {
        let mut timings = Vec::new();
        let result = raw_templates
            .iter()
            .enumerate()
            .filter_map(|item| {
                let started = Instant::now();
                let result = build(item);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<Vec<_>>();
        report_macro_item_timings("compose_commit_template_splits", &timings);
        result
    } else {
        raw_templates
            .par_iter()
            .enumerate()
            .filter_map(build)
            .collect::<Vec<_>>()
    };
    let mut result = vec![None; raw_templates.len()];
    for (terminal, split) in split_templates {
        result[terminal] = Some(split);
    }
    result
}

fn defer_boundary_commit_templates() -> bool {
    crate::env::var_os("GLRMASK_DISABLE_DEFER_BOUNDARY_COMMIT_TEMPLATES").is_none()
}

enum BoundaryParserWork {
    Materialized(MappedArtifact<DWA>),
    Deferred {
        terminal_automaton: TerminalAutomaton,
        id_map: InternalIdMap,
        analyzed: AnalyzedGrammar,
        templates: Templates,
    },
    /// Exact capture/precomputed boundary path after the composed GLR table is
    /// already available. Parser-NWA construction needs only the terminal
    /// domain size, not nullable/FIRST/FOLLOW analysis.
    DeferredTerminalCount {
        terminal_automaton: TerminalAutomaton,
        id_map: InternalIdMap,
        num_terminals: u32,
        templates: Templates,
        prebuilt_bundle_cache: Option<PrebuiltParserBundleCache>,
        /// Optional exact parser-table view used only to compile this static
        /// boundary accelerator.  Authoritative compositions keep linker
        /// controls in their live table, but a static B can compile against an
        /// exact control-eliminated clone without changing the runtime parser
        /// coordinate or semantics.
        parser_table_override: Option<Arc<crate::compiler::glr::table::GLRTable>>,
    },
}

enum PositiveBoundaryParser {
    Dwa(DWA),
    Nwa(NWA),
}


enum BoundaryRuntimeCandidate {
    Parser {
        positive: PositiveBoundaryParser,
        id_map: InternalIdMap,
        template_cache: Option<Vec<Option<UnweightedDfa>>>,
        build_ms: f64,
    },
}

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
    let parser_work = BoundaryParserWork::DeferredTerminalCount {
        terminal_automaton: TerminalAutomaton::Dwa(crossing),
        id_map: work.id_map,
        num_terminals,
        templates,
        prebuilt_bundle_cache: None,
        parser_table_override: Some(Arc::clone(boundary_table)),
    };
    let materialize_started_at = Instant::now();
    let (positive, id_map, _template_cache) =
        parser_work.materialize_positive_parser(boundary_table)?;
    let materialize_ms = materialize_started_at.elapsed().as_secs_f64() * 1000.0;
    positive.ensure_positive()?;
    let normalize_started_at = Instant::now();
    let parser_dwa = positive.into_runtime_dwa(boundary_table);
    let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
    ensure_positive_runtime_parser_dwa(&parser_dwa)?;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_walk_shard_publish] start_component={} terms={} parser_states={} parser_trans={} templates_ms={templates_ms:.3} materialize_ms={materialize_ms:.3} normalize_ms={normalize_ms:.3}",
            work.start_component,
            selected.iter().filter(|slot| **slot).count(),
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
        );
    }
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





fn compute_boundary_tsid_behavior_quotient(
    nwa: &NWA,
    old_tsid_count: usize,
    token_classes: usize,
) -> Option<(Vec<u32>, usize)> {
    if old_tsid_count <= 1 || token_classes == 0 || token_classes > 64 {
        return None;
    }
    let mut unique_ids = FxHashMap::<usize, u16>::default();
    let mut unique_weights = Vec::<Weight>::new();
    let mut register = |weight: &Weight| {
        if weight.is_full() || weight.is_empty() {
            return;
        }
        let key = weight.ptr_key();
        if unique_ids.contains_key(&key) {
            return;
        }
        let Ok(index) = u16::try_from(unique_weights.len()) else {
            return;
        };
        unique_ids.insert(key, index);
        unique_weights.push(weight.clone());
    };
    for state in nwa.states() {
        if let Some(weight) = state.final_weight.as_ref() {
            register(weight);
        }
        for (_, weight) in &state.epsilons {
            register(weight);
        }
        for branches in state.transitions.values() {
            for (_, weight) in branches {
                register(weight);
            }
        }
    }
    if unique_weights.len() > u16::MAX as usize {
        return None;
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


impl PositiveBoundaryParser {
    fn into_runtime_dwa(self, table: &crate::compiler::glr::table::GLRTable) -> DWA {
        match self {
            Self::Dwa(dwa) => dwa,
            Self::Nwa(nwa) => normalize_weighted_parser_stack_nwa(table, &nwa),
        }
    }

    /// Quotient the private boundary TSID coordinate by its exact
    /// behavior on every weight in the already-positive boundary NWA.
    ///
    /// Two TSIDs are equivalent iff every source weight assigns them the same
    /// boundary-token set. All subsequent parser-DWA normalization operations
    /// are pointwise union/intersection/difference over weights, so this is a
    /// congruence: normalizing after the quotient is exactly the quotient of
    /// normalizing before it.
    fn quotient_boundary_tsids(
        &mut self,
        id_map: &mut InternalIdMap,
    ) -> Option<Vec<u32>> {
        crate::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TSID_QUOTIENT")?;
        let Self::Nwa(nwa) = self else {
            return None;
        };
        let old_tsid_count = id_map.num_tsids() as usize;
        let token_classes = id_map.num_internal_tokens() as usize;
        if old_tsid_count <= 1 || token_classes == 0 || token_classes > 64 {
            return None;
        }
        let started_at = Instant::now();

        let signature_started_at = Instant::now();
        let (old_to_new, unique_source_weight_count) =
            compute_boundary_tsid_behavior_quotient(nwa, old_tsid_count, token_classes)?;
        let signature_ms = signature_started_at.elapsed().as_secs_f64() * 1000.0;
        let grouping_ms = 0.0;
        let new_tsid_count = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
        if new_tsid_count >= old_tsid_count {
            return None;
        }

        let remap_started_at = Instant::now();
        let defer_weight_remap = crate::env::var_os(
            "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_WEIGHT_DETERMINIZER",
        )
        .is_some();
        if !defer_weight_remap {
            let tsid_map = old_to_new
                .iter()
                .map(|&class| vec![class])
                .collect::<Vec<_>>();
            let token_map = (0..token_classes as u32)
                .map(|token| vec![token])
                .collect::<Vec<_>>();
            let mut weights = nwa.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &tsid_map,
                &token_map,
                new_tsid_count,
            );
        }
        let remap_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

        // The segmented runtime carries the exact raw-state -> private-TSID map
        // separately. The boundary InternalIdMap therefore needs only one raw
        // representative per quotient class here, not the union of all 16k
        // source representatives.
        let old_representatives =
            std::mem::take(&mut id_map.tokenizer_states.representative_original_ids);
        let mut new_representatives = vec![u32::MAX; new_tsid_count];
        for old_tsid in 0..old_tsid_count {
            let class = old_to_new[old_tsid] as usize;
            if new_representatives[class] == u32::MAX {
                new_representatives[class] = old_representatives
                    .get(old_tsid)
                    .copied()
                    .unwrap_or(old_tsid as u32);
            }
        }
        for internal in &mut id_map.tokenizer_states.original_to_internal {
            if *internal != u32::MAX {
                let &class = old_to_new.get(*internal as usize)?;
                *internal = class;
            }
        }
        id_map.tokenizer_states.internal_to_originals = new_representatives
            .iter()
            .map(|&representative| vec![representative])
            .collect();
        id_map.tokenizer_states.representative_original_ids = new_representatives;

        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_tsid_quotient] old_tsids={} new_tsids={} unique_source_weights={} signature_ms={signature_ms:.3} grouping_ms={grouping_ms:.3} remap_ms={remap_ms:.3} total_ms={:.3}",
                old_tsid_count,
                new_tsid_count,
                unique_source_weight_count,
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        Some(old_to_new)
    }

    fn into_runtime_dwa_small_boundary(
        self,
        table: &crate::compiler::glr::table::GLRTable,
        num_tsids: usize,
        num_tokens: usize,
        source_tsid_map: Option<&[u32]>,
    ) -> DWA {
        match self {
            Self::Dwa(dwa) => dwa,
            Self::Nwa(nwa) => {
                if let Some(source_tsid_map) = source_tsid_map {
                    normalize_weighted_parser_stack_nwa_small_boundary_with_tsid_map(
                        table,
                        &nwa,
                        num_tsids,
                        num_tokens,
                        source_tsid_map,
                    )
                } else {
                    normalize_weighted_parser_stack_nwa_small_boundary(
                        table,
                        &nwa,
                        num_tsids,
                        num_tokens,
                    )
                }
            }
        }
    }

    fn into_runtime_dwa_small_boundary_for_parser_state_count(
        self,
        num_parser_states: u32,
        num_tsids: usize,
        num_tokens: usize,
        source_tsid_map: Option<&[u32]>,
    ) -> DWA {
        match self {
            Self::Dwa(dwa) => dwa,
            Self::Nwa(nwa) => normalize_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
                num_parser_states,
                &nwa,
                num_tsids,
                num_tokens,
                source_tsid_map,
            ),
        }
    }

    fn ensure_positive(&self) -> Result<(), String> {
        match self {
            Self::Dwa(dwa) => ensure_positive_runtime_parser_dwa(dwa),
            Self::Nwa(nwa) => {
                for (source, state) in nwa.states().iter().enumerate() {
                    if let Some(&label) = state
                        .transitions
                        .keys()
                        .find(|&&label| is_negative_label(label))
                    {
                        return Err(format!(
                            "runtime boundary parser NWA contains negative/PUSH label {label} at state {source}"
                        ));
                    }
                }
                Ok(())
            }
        }
    }

}


impl BoundaryParserWork {
    fn set_parser_table_override(
        &mut self,
        table: Arc<crate::compiler::glr::table::GLRTable>,
    ) -> Result<(), String> {
        match self {
            Self::DeferredTerminalCount {
                num_terminals,
                templates,
                parser_table_override,
                ..
            } => {
                if table.num_terminals != *num_terminals {
                    return Err(format!(
                        "parser-table override terminal domain mismatch: table={} boundary={}",
                        table.num_terminals, *num_terminals,
                    ));
                }
                // Parser templates encode stack effects, so they belong to the
                // parser-state coordinate just as much as the table itself.
                // Re-characterize exactly the terminal subset already selected
                // by boundary discovery; reusing materialized-table templates
                // with a recursive provider table would mix two stack alphabets.
                let mut selected = vec![false; *num_terminals as usize];
                for &terminal in templates.by_terminal.keys() {
                    let Some(slot) = selected.get_mut(terminal as usize) else {
                        return Err(format!(
                            "boundary template terminal {terminal} lies outside parser domain {num_terminals}"
                        ));
                    };
                    *slot = true;
                }
                let characterizations = try_characterize_selected_terminals_for_terminal_count(
                    &table,
                    *num_terminals,
                    &selected,
                )
                .map_err(|error| error.to_string())?;
                *templates = Templates::from_characterizations(&characterizations);
                *parser_table_override = Some(table);
                Ok(())
            }
            _ => Err(
                "parser-table override requires deferred count-only boundary work".to_owned(),
            ),
        }
    }

    fn parser_table_override(
        &self,
    ) -> Option<&Arc<crate::compiler::glr::table::GLRTable>> {
        match self {
            Self::DeferredTerminalCount {
                parser_table_override,
                ..
            } => parser_table_override.as_ref(),
            _ => None,
        }
    }

    fn materialize_positive_parser(
        self,
        table: &crate::compiler::glr::table::GLRTable,
    ) -> Result<(
        PositiveBoundaryParser,
        InternalIdMap,
        Option<Vec<Option<UnweightedDfa>>>,
    ), String> {
        match self {
            Self::Materialized(parser) => {
                let (dwa, id_map) = parser.into_parts();
                Ok((PositiveBoundaryParser::Dwa(dwa), id_map, None))
            }
            Self::Deferred {
                terminal_automaton,
                id_map,
                analyzed,
                templates,
            } => {
                let Some(mut parser_nwa) = build_parser_nwa_from_terminal_dwa_with_precomputed_templates(
                    &terminal_automaton,
                    &analyzed,
                    &templates,
                    table,
                ) else {
                    // No parser stack can realize any discovered terminal path:
                    // the exact boundary repair language is empty, not invalid.
                    return Ok((
                        PositiveBoundaryParser::Dwa(DWA::new(
                            id_map.num_tsids(),
                            id_map.max_internal_token_id(),
                        )),
                        id_map,
                        None,
                    ));
                };
                resolve_negative_codes_in_nwa(
                    &mut parser_nwa,
                    table.construction
                        == crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
                );
                let parser_dwa = normalize_weighted_parser_stack_nwa(table, &parser_nwa);
                Ok((PositiveBoundaryParser::Dwa(parser_dwa), id_map, None))
            }
            Self::DeferredTerminalCount {
                mut terminal_automaton,
                mut id_map,
                num_terminals,
                templates,
                prebuilt_bundle_cache: _,
                parser_table_override,
            } => {
                let table = parser_table_override.as_deref().unwrap_or(table);
                if crate::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_TSID_QUOTIENT").is_some()
                    && let TerminalAutomaton::Dwa(dwa) = &mut terminal_automaton
                {
                    let old_tsids = id_map.num_tsids() as usize;
                    let token_classes = id_map.num_internal_tokens() as usize;
                    let started_at = Instant::now();
                    if let Some((old_to_new, unique_weights)) =
                        compute_boundary_dwa_tsid_behavior_quotient(dwa, old_tsids, token_classes)
                    {
                        let new_tsids = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
                        if new_tsids < old_tsids {
                            apply_boundary_tsid_quotient_to_dwa_and_id_map(dwa, &mut id_map, &old_to_new)
                                .expect("early boundary TSID quotient must apply to matching DWA/id-map coordinate");
                            if compose_profile_enabled() {
                                eprintln!(
                                    "[glrmask/profile][constraint_boundary_early_tsid_quotient] old_tsids={} new_tsids={} unique_terminal_weights={} total_ms={:.3}",
                                    old_tsids, new_tsids, unique_weights, started_at.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                        }
                    }
                }
                let build_started_at = Instant::now();
                let nondeterministic_bundles_requested = crate::env::var_os(
                    "GLRMASK_EXPERIMENT_COMPILE_NONDETERMINISTIC_BUNDLES",
                )
                .is_some();
                // Preserving bundle nondeterminism avoids repeated local DFA
                // union work on wide boundary alphabets, but on small boundary
                // coordinates it creates more NWA support for the final
                // publication pass than it saves during bundle construction.
                // Key this on the boundary token coordinate itself: it is a
                // representation-size policy, not a grammar/layout heuristic.
                let nondeterministic_bundle_min_tokens = crate::env::var(
                    "GLRMASK_COMPILE_NONDETERMINISTIC_BUNDLES_MIN_TOKENS",
                )
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(64);
                let compile_nondeterministic_bundles = nondeterministic_bundles_requested
                    && id_map.num_internal_tokens() >= nondeterministic_bundle_min_tokens;
                let parser_nwa = if compile_nondeterministic_bundles {
                    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_nondeterministic_bundles(
                        &terminal_automaton,
                        num_terminals,
                        &templates,
                        table,
                    )
                } else {
                    build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count(
                        &terminal_automaton,
                        num_terminals,
                        &templates,
                        table,
                    )
                };
                let Some(mut parser_nwa) = parser_nwa else {
                    let mut template_cache = vec![None; num_terminals as usize];
                    for (terminal, dfa) in templates.by_terminal {
                        if let Some(slot) = template_cache.get_mut(terminal as usize) {
                            *slot = Some(dfa);
                        }
                    }
                    return Ok((
                        PositiveBoundaryParser::Dwa(DWA::new(
                            id_map.num_tsids(),
                            id_map.max_internal_token_id(),
                        )),
                        id_map,
                        Some(template_cache),
                    ));
                };
                let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;
                if crate::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_FUSED").is_some()
                    && id_map.num_tsids() as usize <= 16
                    && id_map.num_internal_tokens() as usize <= 64
                {
                    let fused_started_at = Instant::now();
                    let parser_dwa = normalize_signed_weighted_parser_stack_nwa_small_boundary(
                        table,
                        &parser_nwa,
                        id_map.num_tsids() as usize,
                        id_map.num_internal_tokens() as usize,
                    )
                    .ok_or_else(|| "small-boundary fused signed publication rejected boundary coordinate".to_string())?;
                    let fused_ms = fused_started_at.elapsed().as_secs_f64() * 1000.0;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_segmented_boundary_parser_phases] build_nwa_ms={build_ms:.3} fused_signed_publish_ms={fused_ms:.3} deterministic_runtime=true compile_nondeterministic_bundles={compile_nondeterministic_bundles}",
                        );
                    }
                    let mut template_cache = vec![None; num_terminals as usize];
                    for (terminal, dfa) in templates.by_terminal {
                        if let Some(slot) = template_cache.get_mut(terminal as usize) {
                            *slot = Some(dfa);
                        }
                    }
                    return Ok((
                        PositiveBoundaryParser::Dwa(parser_dwa),
                        id_map,
                        Some(template_cache),
                    ));
                }
                let resolve_started_at = Instant::now();
                let allow_grouped_cancellation = table.construction
                    == crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged;
                let resolved_reverse_topo = if crate::env::var_os(
                    "GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_RESOLUTION",
                )
                .is_some()
                    && id_map.num_tsids() as usize <= 16
                    && id_map.num_internal_tokens() as usize <= 64
                {
                    parser_nwa = resolve_negative_codes_small_boundary(
                        &parser_nwa,
                        id_map.num_tsids() as usize,
                        id_map.num_internal_tokens() as usize,
                    )
                    .ok_or_else(|| {
                        "small-boundary signed resolver rejected boundary coordinate".to_string()
                    })?;
                    None
                } else {
                    resolve_negative_codes_in_nwa(&mut parser_nwa, allow_grouped_cancellation)
                };
                let resolve_negative_ms =
                    resolve_started_at.elapsed().as_secs_f64() * 1000.0;
                let hashcons_started_at = Instant::now();
                let before_hashcons_states = parser_nwa.num_states();
                let before_hashcons_transitions = parser_nwa.num_transitions();
                if crate::env::var_os("GLRMASK_EXPERIMENT_FAST_HASHCONS_RESOLVED_BOUNDARY_NWA")
                    .is_some()
                {
                    parser_nwa = reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
                        parser_nwa,
                        resolved_reverse_topo,
                    );
                } else if crate::env::var_os("GLRMASK_EXPERIMENT_HASHCONS_RESOLVED_BOUNDARY_NWA")
                    .is_some()
                {
                    parser_nwa = reverse_hashcons_positive_acyclic_nwa(parser_nwa);
                }
                let hashcons_ms = hashcons_started_at.elapsed().as_secs_f64() * 1000.0;
                if compose_profile_enabled() {
                    eprintln!(
                        "[glrmask/profile][constraint_segmented_boundary_parser_phases] build_nwa_ms={build_ms:.3} resolve_negative_ms={resolve_negative_ms:.3} hashcons_ms={hashcons_ms:.3} hashcons_states={}=>{} hashcons_transitions={}=>{} determinize_dwa_ms=deferred deterministic_runtime=false positive_nwa=true compile_nondeterministic_bundles={compile_nondeterministic_bundles}",
                        before_hashcons_states,
                        parser_nwa.num_states(),
                        before_hashcons_transitions,
                        parser_nwa.num_transitions(),
                    );
                }
                let mut template_cache = vec![None; num_terminals as usize];
                for (terminal, dfa) in templates.by_terminal {
                    if let Some(slot) = template_cache.get_mut(terminal as usize) {
                        *slot = Some(dfa);
                    }
                }
                Ok((
                    PositiveBoundaryParser::Nwa(parser_nwa),
                    id_map,
                    Some(template_cache),
                ))
            }
        }
    }

    fn materialize_positive_parser_without_table(
        self,
        allow_grouped_cancellation: bool,
        num_parser_states: u32,
    ) -> Result<BoundaryRuntimeCandidate, String> {
        let Self::DeferredTerminalCount {
            mut terminal_automaton,
            mut id_map,
            num_terminals,
            templates,
            prebuilt_bundle_cache,
            parser_table_override,
        } = self
        else {
            return Err(
                "table-free boundary parser construction requires deferred count-only work".into(),
            );
        };
        let started_at = Instant::now();
        if let Some(table) = parser_table_override {
            let work = Self::DeferredTerminalCount {
                terminal_automaton,
                id_map,
                num_terminals,
                templates,
                prebuilt_bundle_cache,
                parser_table_override: None,
            };
            let (positive, id_map, template_cache) =
                work.materialize_positive_parser(table.as_ref())?;
            return Ok(BoundaryRuntimeCandidate::Parser {
                positive,
                id_map,
                template_cache,
                build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            });
        }
        let fused_signed_requested =
            crate::env::var_os("GLRMASK_EXPERIMENT_SMALL_BOUNDARY_SIGNED_FUSED").is_some();
        if fused_signed_requested
            && crate::env::var_os("GLRMASK_EXPERIMENT_EARLY_BOUNDARY_TSID_QUOTIENT").is_some()
            && let TerminalAutomaton::Dwa(dwa) = &mut terminal_automaton
        {
            let old_tsids = id_map.num_tsids() as usize;
            let token_classes = id_map.num_internal_tokens() as usize;
            let quotient_started = Instant::now();
            if let Some((old_to_new, unique_weights)) =
                compute_boundary_dwa_tsid_behavior_quotient(dwa, old_tsids, token_classes)
            {
                let new_tsids = old_to_new.iter().copied().max().map_or(0, |v| v as usize + 1);
                if new_tsids < old_tsids {
                    apply_boundary_tsid_quotient_to_dwa_and_id_map(
                        dwa,
                        &mut id_map,
                        &old_to_new,
                    )
                    .ok_or_else(|| "early table-free boundary TSID quotient failed".to_string())?;
                    if compose_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][constraint_boundary_early_tsid_quotient_no_table] old_tsids={} new_tsids={} unique_terminal_weights={} total_ms={:.3}",
                            old_tsids,
                            new_tsids,
                            unique_weights,
                            quotient_started.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }
            }
        }
        let nondeterministic_bundles_requested = crate::env::var_os(
            "GLRMASK_EXPERIMENT_COMPILE_NONDETERMINISTIC_BUNDLES",
        )
        .is_some();
        let nondeterministic_bundle_min_tokens = crate::env::var(
            "GLRMASK_COMPILE_NONDETERMINISTIC_BUNDLES_MIN_TOKENS",
        )
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(64);
        let compile_nondeterministic_bundles = nondeterministic_bundles_requested
            && id_map.num_internal_tokens() >= nondeterministic_bundle_min_tokens;
        let parser_nwa = if !compile_nondeterministic_bundles
            && let Some(cache) = prebuilt_bundle_cache.as_ref()
        {
            build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table_with_bundle_cache(
                &terminal_automaton,
                num_terminals,
                &templates,
                cache,
            )
        } else {
            build_parser_nwa_from_terminal_dwa_with_precomputed_templates_for_terminal_count_no_table(
                &terminal_automaton,
                num_terminals,
                &templates,
                compile_nondeterministic_bundles,
            )
        };
        let Some(mut parser_nwa) = parser_nwa else {
            let mut template_cache = vec![None; num_terminals as usize];
            for (terminal, dfa) in templates.by_terminal {
                if let Some(slot) = template_cache.get_mut(terminal as usize) {
                    *slot = Some(dfa);
                }
            }
            return Ok(BoundaryRuntimeCandidate::Parser {
                positive: PositiveBoundaryParser::Dwa(DWA::new(
                    id_map.num_tsids(),
                    id_map.max_internal_token_id(),
                )),
                id_map,
                template_cache: Some(template_cache),
                build_ms: started_at.elapsed().as_secs_f64() * 1000.0,
            });
        };
        if fused_signed_requested
            && id_map.num_tsids() as usize <= 16
            && id_map.num_internal_tokens() as usize <= 64
        {
            let fused_started = Instant::now();
            let parser_dwa =
                normalize_signed_weighted_parser_stack_nwa_small_boundary_for_parser_state_count(
                    num_parser_states,
                    &parser_nwa,
                    id_map.num_tsids() as usize,
                    id_map.num_internal_tokens() as usize,
                )
                .ok_or_else(|| {
                    "table-free small-boundary fused signed publication rejected coordinate"
                        .to_string()
                })?;
            let mut template_cache = vec![None; num_terminals as usize];
            for (terminal, dfa) in templates.by_terminal {
                if let Some(slot) = template_cache.get_mut(terminal as usize) {
                    *slot = Some(dfa);
                }
            }
            let build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_boundary_fused_signed_no_table] signed_states={} output_states={} output_transitions={} fused_ms={:.3} total_ms={build_ms:.3}",
                    parser_nwa.num_states(),
                    parser_dwa.num_states(),
                    parser_dwa.num_transitions(),
                    fused_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            return Ok(BoundaryRuntimeCandidate::Parser {
                positive: PositiveBoundaryParser::Dwa(parser_dwa),
                id_map,
                template_cache: Some(template_cache),
                build_ms,
            });
        }
        if crate::env::var_os("GLRMASK_EXPERIMENT_HASHCONS_SIGNED_BOUNDARY_NWA").is_some() {
            parser_nwa = reverse_hashcons_signed_acyclic_nwa_fast(parser_nwa);
        }
        let resolved_reverse_topo =
            resolve_negative_codes_in_nwa(&mut parser_nwa, allow_grouped_cancellation);
        if crate::env::var_os("GLRMASK_EXPERIMENT_FAST_HASHCONS_RESOLVED_BOUNDARY_NWA").is_some() {
            parser_nwa = reverse_hashcons_positive_acyclic_nwa_fast_with_reverse_topo(
                parser_nwa,
                resolved_reverse_topo,
            );
        } else if crate::env::var_os("GLRMASK_EXPERIMENT_HASHCONS_RESOLVED_BOUNDARY_NWA").is_some() {
            parser_nwa = reverse_hashcons_positive_acyclic_nwa(parser_nwa);
        }
        let mut template_cache = vec![None; num_terminals as usize];
        for (terminal, dfa) in templates.by_terminal {
            if let Some(slot) = template_cache.get_mut(terminal as usize) {
                *slot = Some(dfa);
            }
        }
        let build_ms = started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_positive_no_table] states={} transitions={} build_ms={build_ms:.3} compile_nondeterministic_bundles={compile_nondeterministic_bundles}",
                parser_nwa.num_states(),
                parser_nwa.num_transitions(),
            );
        }
        Ok(BoundaryRuntimeCandidate::Parser {
            positive: PositiveBoundaryParser::Nwa(parser_nwa),
            id_map,
            template_cache: Some(template_cache),
            build_ms,
        })
    }

    fn materialize_unminimized(
        self,
        table: &crate::compiler::glr::table::GLRTable,
        vocab: &Vocab,
    ) -> MappedArtifact<DWA> {
        match self {
            Self::Materialized(parser) => parser,
            Self::Deferred {
                terminal_automaton,
                id_map,
                analyzed,
                templates,
            } => MappedArtifact::new(
                build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                    table,
                    &analyzed,
                    &terminal_automaton,
                    &templates,
                    vocab,
                    &id_map,
                    false,
                ),
                id_map,
            ),
            Self::DeferredTerminalCount { .. } => panic!(
                "count-only boundary parser work is supported only by the segmented NWA runtime"
            ),
        }
    }

    fn materialize(self, table: &crate::compiler::glr::table::GLRTable, vocab: &Vocab) -> MappedArtifact<DWA> {
        match self {
            Self::Materialized(parser) => parser,
            Self::Deferred {
                terminal_automaton,
                id_map,
                analyzed,
                templates,
            } => {
                let parser_dwa = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                    table,
                    &analyzed,
                    &terminal_automaton,
                    &templates,
                    vocab,
                    &id_map,
                    false,
                );
                let parser_dwa = if parser_dwa.num_states() >= boundary_parser_minimize_min_states()
                    && parser_builder_skips_internal_minimization()
                    && crate::env::var_os("GLRMASK_DISABLE_BOUNDARY_PARSER_MINIMIZE").is_none()
                {
                    minimize_owned(reverse_hashcons_owned(parser_dwa))
                } else {
                    parser_dwa
                };
                MappedArtifact::new(parser_dwa, id_map)
            }
            Self::DeferredTerminalCount { .. } => panic!(
                "count-only boundary parser work is supported only by the segmented NWA runtime"
            ),
        }
    }
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


#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct BoundaryTokenNodeKey {
    offset: usize,
    /// `u32::MAX` means no parser-visible terminal has committed yet.
    /// Globally erased trivia does not update this field.
    last_terminal: u32,
    /// Whether the first non-globally-erased terminal of this model token is
    /// a FIRST/FOLLOW boundary seed. Later seed terminals do not establish a
    /// seed-only boundary witness; they require an actual interface crossing.
    seeded: bool,
    /// Whether two concrete parser-visible terminals on this path witness an
    /// actual boundary interface pair. This is independently sufficient
    /// boundary evidence; the LR table later decides the full parser language.
    interface_witnessed: bool,
    /// Whether this token path has consumed at least one globally erased
    /// terminal before the current point. This is the only multi-byte seed-only
    /// case not already represented by the reset one-terminal relation.
    erased_seen: bool,
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

impl ResidualScanResult {
    fn union_from(&mut self, other: &Self) {
        self.matches.extend_from_slice(&other.matches);
        self.future_terminals.extend_from_slice(&other.future_terminals);
    }

    fn canonicalize(&mut self) {
        self.matches.sort_unstable();
        self.matches.dedup();
        self.future_terminals.sort_unstable();
        self.future_terminals.dedup();
    }
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

fn boundary_tokens_by_start_component(
    discovery: &BoundaryTokenDiscovery,
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    component_count: usize,
) -> Vec<Vec<u32>> {
    let terminal_owner = |terminal: u32| -> Option<usize> {
        let component = terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .checked_sub(1)?;
        (component < component_count).then_some(component)
    };
    let tokenizer_state_owner = |state: u32| -> Option<usize> {
        if state == 0 {
            return None;
        }
        let component = tokenizer_state_offsets
            .partition_point(|&offset| offset <= state)
            .checked_sub(1)?;
        (component < component_count).then_some(component)
    };

    let mut by_component = vec![BTreeSet::<u32>::new(); component_count];
    for witness in &discovery.witnesses {
        let mut owners = BTreeSet::<usize>::new();
        for &state in &witness.start_states {
            if let Some(owner) = tokenizer_state_owner(state) {
                owners.insert(owner);
                continue;
            }

            // Raw state zero is the merged reset dispatcher rather than a
            // component-local tokenizer state.  At that coordinate the first
            // productive terminal in the exact witness chooses the component
            // in which token execution starts.  A witness may have more than
            // one such first owner; publish the token conservatively to each.
            if state == 0
                && let Some(root) = witness.nodes.first()
            {
                for edge in root
                    .outgoing
                    .iter()
                    .filter(|edge| witness.good.get(edge.target).copied().unwrap_or(false))
                {
                    if let Some(owner) = terminal_owner(edge.terminal) {
                        owners.insert(owner);
                    }
                }
            }
        }
        for owner in owners {
            by_component[owner].insert(witness.token_id);
        }
    }

    by_component
        .into_iter()
        .map(|tokens| tokens.into_iter().collect())
        .collect()
}

#[derive(Clone)]
struct PreTableBoundaryBaseDiscovery {
    terminal_offsets: Vec<u32>,
    prefer_cross_interface_only: bool,
    tokenizer_state_offsets: Vec<u32>,
    summary: CompositionGrammarSummary,
    base_interface_pairs: BTreeSet<(u32, u32)>,
    discovery_interface_pairs: BTreeSet<(u32, u32)>,
    discovery: BoundaryTokenDiscovery,
    component_state_map: Option<ManyToOneIdMap>,
    terminal_artifact: Option<MappedArtifact<TerminalAutomaton>>,
    elapsed_ms: f64,
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
    let parser = constraint.template_parser.as_ref()
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








fn visible_boundary_interface_pairs(
    analyzed: &AnalyzedGrammar,
    boundary_nonterminals: &BTreeSet<u32>,
    control_terminals: &BTreeSet<u32>,
) -> BTreeSet<(u32, u32)> {
    let set_len = analyzed.num_terminals as usize + 1;
    let mut last = vec![BitSet::new(set_len); analyzed.num_nonterminals as usize];
    loop {
        let mut changed = false;
        for rule in &analyzed.rules {
            let lhs = rule.lhs as usize;
            let mut additions = BitSet::new(set_len);
            for symbol in rule.rhs.iter().rev() {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        if *terminal < analyzed.num_terminals {
                            additions.set(*terminal as usize);
                        }
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        if let Some(row) = last.get(*nonterminal as usize) {
                            additions.union_with(row);
                        }
                        if !analyzed.nullable.contains(nonterminal) {
                            break;
                        }
                    }
                }
            }
            let before = last[lhs].count_ones();
            last[lhs].union_with(&additions);
            changed |= last[lhs].count_ones() != before;
        }
        if !changed {
            break;
        }
    }

    // PRECEDE is FOLLOW on the reversed grammar: the visible terminals that
    // may occur immediately before a nonterminal, propagating through nullable
    // prefixes and callers.
    let mut precede = vec![BitSet::new(set_len); analyzed.num_nonterminals as usize];
    loop {
        let mut changed = false;
        for rule in &analyzed.rules {
            for (position, symbol) in rule.rhs.iter().enumerate() {
                let Symbol::Nonterminal(nonterminal) = symbol else {
                    continue;
                };
                let mut additions = BitSet::new(set_len);
                let mut prefix_nullable = true;
                for prefix in rule.rhs[..position].iter().rev() {
                    match prefix {
                        Symbol::Terminal(terminal) => {
                            if *terminal < analyzed.num_terminals {
                                additions.set(*terminal as usize);
                            }
                            prefix_nullable = false;
                            break;
                        }
                        Symbol::Nonterminal(previous) => {
                            if let Some(row) = last.get(*previous as usize) {
                                additions.union_with(row);
                            }
                            if !analyzed.nullable.contains(previous) {
                                prefix_nullable = false;
                                break;
                            }
                        }
                    }
                }
                if prefix_nullable
                    && let Some(lhs_precede) = precede.get(rule.lhs as usize) {
                        additions.union_with(lhs_precede);
                    }
                let target = &mut precede[*nonterminal as usize];
                let before = target.count_ones();
                target.union_with(&additions);
                changed |= target.count_ones() != before;
            }
        }
        if !changed {
            break;
        }
    }

    let lexical = |terminal: usize| {
        terminal < analyzed.num_terminals as usize
            && !control_terminals.contains(&(terminal as u32))
    };
    let mut pairs = BTreeSet::new();
    for &nonterminal in boundary_nonterminals {
        let Some(before) = precede.get(nonterminal as usize) else {
            continue;
        };
        let Some(first) = analyzed.first.get(nonterminal as usize) else {
            continue;
        };
        for left in before.iter().filter(|&terminal| lexical(terminal)) {
            for right in first.iter().filter(|&terminal| lexical(terminal)) {
                pairs.insert((left as u32, right as u32));
            }
        }
        let Some(last_row) = last.get(nonterminal as usize) else {
            continue;
        };
        let Some(after) = analyzed.follow.get(nonterminal as usize) else {
            continue;
        };
        for left in last_row.iter().filter(|&terminal| lexical(terminal)) {
            for right in after.iter().filter(|&terminal| lexical(terminal)) {
                pairs.insert((left as u32, right as u32));
            }
        }
        if analyzed.nullable.contains(&nonterminal) {
            for left in before.iter().filter(|&terminal| lexical(terminal)) {
                for right in after.iter().filter(|&terminal| lexical(terminal)) {
                    pairs.insert((left as u32, right as u32));
                }
            }
        }
    }
    pairs
}





/// Extend grammar-derived boundary adjacency through parser-visible terminals
/// whose LR action preserves stack depth. These terminals need not occur in a
/// grammar RHS (scoped trivia is inserted directly into LR rows), so FIRST /
/// FOLLOW alone cannot see them.
///
/// For an actual interface pair `a -> b`, if a stack-neutral terminal `n` can
/// be consumed at a parser top from which `b` is admissible (or can replace the
/// top with a state from which `b` is admissible), then the concrete lexical
/// boundary may be `a -> n -> b`. Record exactly those two adjacencies.
/// `n` is deliberately *not* promoted to a global boundary seed: its relevance
/// is contextual to this interface, and arbitrary residual starts inside `n`
/// are recovered from the left-hand side of the resulting `(n, b)` pair.
fn extend_boundary_interfaces_through_stack_neutral_lr_actions(
    table: &crate::compiler::glr::table::GLRTable,
    base_pairs: &BTreeSet<(u32, u32)>,
) -> BTreeSet<(u32, u32)> {
    let mut pairs = base_pairs.clone();
    if table.skip_terminals.is_empty() || base_pairs.is_empty() {
        return pairs;
    }

    let target_is_admissible = |state: u32, terminal: u32| {
        table
            .advance
            .get(state as usize)
            .is_some_and(|row| row.contains(terminal as usize))
            || table
                .action
                .get(state as usize)
                .and_then(|row| row.get(&terminal))
                .is_some()
    };

    for &(left, right) in base_pairs {
        for &neutral in &table.skip_terminals {
            let mut bridges = false;
            for (state, row) in table.action.iter().enumerate() {
                let state = state as u32;
                let Some(action) = row.get(&neutral) else {
                    continue;
                };
                match action {
                    Action::Skip => {
                        bridges |= target_is_admissible(state, right);
                    }
                    Action::Shift(target, true) => {
                        bridges |= target_is_admissible(*target, right);
                    }
                    Action::ReplaceShifts(targets) => {
                        bridges |= targets
                            .iter()
                            .copied()
                            .any(|target| target_is_admissible(target, right));
                    }
                    Action::StackShifts(shifts)
                        if shifts
                            .iter()
                            .all(|shift| shift.pop == 1 && shift.pushes.len() == 1) =>
                    {
                        bridges |= shifts.iter().any(|shift| {
                            target_is_admissible(shift.pushes[0], right)
                        });
                    }
                    Action::Split {
                        shift: Some((target, true)),
                        reduces,
                        accept: false,
                    } if reduces.is_empty() => {
                        bridges |= target_is_admissible(*target, right);
                    }
                    _ => {}
                }
                if bridges {
                    break;
                }
            }
            if bridges {
                pairs.insert((left, neutral));
                pairs.insert((neutral, right));
            }
        }
    }
    pairs
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






/// Necessary byte-level filter for a visible terminal interface realized inside
/// one model token. If terminal `a` is followed by terminal `b`, then at the
/// split where `a` finishes and `b` begins the token contains an adjacent byte
/// pair `(last(a), first(b))`. This says nothing about parser legality by itself;
/// it only supplies cheap candidate tokens for the exact graph below.
fn boundary_interface_adjacent_pair_candidates(
    vocab: &Vocab,
    components: &[&Constraint],
    terminal_offsets: &[u32],
    interface_pairs: &BTreeSet<(u32, u32)>,
) -> Vec<u32> {
    let num_terminals = components
        .iter()
        .zip(terminal_offsets.iter().copied())
        .map(|(component, offset)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let mut summaries = vec![None::<ExprByteSummary>; num_terminals];
    let sparse_summaries = crate::env::var_os(
        "GLRMASK_DISABLE_SPARSE_INTERFACE_BYTE_SUMMARIES",
    )
    .is_none();
    let needed_terminals = sparse_summaries.then(|| {
        interface_pairs
            .iter()
            .flat_map(|&(left, right)| [left, right])
            .collect::<BTreeSet<_>>()
    });
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index] as usize;
        for local_terminal in 0..component.tokenizer.num_terminals() as usize {
            let global_terminal = terminal_offset + local_terminal;
            if needed_terminals
                .as_ref()
                .is_some_and(|needed| !needed.contains(&(global_terminal as u32)))
            {
                continue;
            }
            summaries[global_terminal] = Some(
                component
                    .retained_terminal_expr(local_terminal as u32)
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
    for &(left, right) in interface_pairs {
        let Some(left) = summaries.get(left as usize).and_then(|summary| *summary) else {
            continue;
        };
        let Some(right) = summaries.get(right as usize).and_then(|summary| *summary) else {
            continue;
        };
        for last in left.last.iter() {
            allowed_pairs[last as usize] |= right.first;
        }
    }
    vocab_tokens_with_adjacent_pairs(vocab, &allowed_pairs)
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
    if compose_profile_enabled() {
        let shapes = automata
            .iter()
            .map(|automaton| (automaton.num_states(), automaton.num_transitions(), automaton.start_states().len()))
            .collect::<Vec<_>>();
        eprintln!("[glrmask/profile][constraint_overlap_local_inputs] shapes={shapes:?}");
    }

    let mut raw_states = Vec::new();
    let mut raw_owners = Vec::<usize>::new();
    let mut raw_locals = Vec::<u32>::new();
    let mut starts = Vec::new();
    for (automaton_index, automaton) in automata.into_iter().enumerate() {
        let offset = raw_states.len() as u32;
        let (states, start_states) = automaton.into_parts();
        raw_owners.extend(std::iter::repeat_n(automaton_index, states.len()));
        raw_locals.extend(0..states.len() as u32);
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
    // Semantic key, not a storage-address key.  Weight has structural Eq and a
    // cached structural Hash, so equal residual languages intern to the same
    // determinized state regardless of allocator/interner lifetime ordering.
    type ResidualSubsetKey = SmallVec<[(u32, Weight); 4]>;

    #[derive(Clone)]
    enum PendingUnionWeight {
        Immediate(Weight),
        Deferred(usize),
    }

    #[derive(Clone, Copy)]
    struct DeferredWeightPatchRun {
        state: u32,
        start: i32,
        end: i32,
        job: usize,
    }

    fn append_pending_transition_run(
        output_state: u32,
        start: i32,
        end: i32,
        target: u32,
        pending_weight: PendingUnionWeight,
        output_transitions: &mut Vec<(i32, (u32, Weight))>,
        deferred_patches: &mut Vec<DeferredWeightPatchRun>,
        insert_default_first: bool,
    ) {
        debug_assert!(start <= end);
        let mut append_entries = |weight: &Weight| {
            if insert_default_first {
                debug_assert_eq!(start, end);
                output_transitions.insert(0, (start, (target, weight.clone())));
            } else {
                for label in start..=end {
                    output_transitions.push((label, (target, weight.clone())));
                }
            }
        };
        match pending_weight {
            PendingUnionWeight::Immediate(weight) => append_entries(&weight),
            PendingUnionWeight::Deferred(job) => {
                append_entries(&Weight::empty());
                if let Some(previous) = deferred_patches.last_mut()
                    && previous.state == output_state
                    && previous.job == job
                    && previous.end.checked_add(1) == Some(start)
                {
                    previous.end = end;
                } else {
                    deferred_patches.push(DeferredWeightPatchRun {
                        state: output_state,
                        start,
                        end,
                        job,
                    });
                }
            }
        }
    }


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

    fn explicit_transition_runs<'a>(
        source: &'a NWAState,
    ) -> Vec<(i32, i32, &'a Vec<(u32, Weight)>)> {
        let mut runs = Vec::<(i32, i32, &'a Vec<(u32, Weight)>)>::new();
        for (&label, targets) in source
            .transitions
            .iter()
            .filter(|(label, _)| **label != DEFAULT_LABEL)
        {
            if let Some((_, end, previous_targets)) = runs.last_mut()
                && end.checked_add(1) == Some(label)
                && same_raw_targets(previous_targets, targets)
            {
                *end = label;
            } else {
                runs.push((label, label, targets));
            }
        }
        runs
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
        defer_support_unions: bool,
        deferred_union_ids: &mut FxHashMap<(usize, usize), usize>,
        deferred_union_jobs: &mut Vec<(Weight, Weight)>,
        singleton_states: &mut [u32],
        singleton_count: &mut usize,
        states: &mut Vec<DWAState>,
        queue: &mut VecDeque<(u32, ResidualSubset)>,
        subset_states: &mut FxHashMap<ResidualSubsetKey, u32>,
    ) -> Option<(u32, PendingUnionWeight)> {
        let finish_subset = |
            normalized: ResidualSubset,
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
                .map(|(state, weight)| (*state, weight.clone()))
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
                Some((target, PendingUnionWeight::Immediate(edge_weight)))
            }
            2 => {
                let (right_target, right_weight) = contributions.pop().unwrap();
                let (left_target, left_weight) = contributions.pop().unwrap();

                // The target residual subset is determined entirely by the
                // non-empty contributions.  Support outside this edge is
                // unobservable after the edge is taken, so constructing the
                // expensive union is not required to discover graph topology.
                // This matches the eager normal form used here today, where
                // Weight::complement() deliberately contributes no finite
                // outside-support normalization.
                let pending_weight = if left_weight.is_full() || right_weight.is_full() {
                    PendingUnionWeight::Immediate(Weight::all())
                } else if left_weight.ptr_key() == right_weight.ptr_key() {
                    PendingUnionWeight::Immediate(left_weight.clone())
                } else if defer_support_unions {
                    let left_key = left_weight.ptr_key();
                    let right_key = right_weight.ptr_key();
                    let key = if left_key <= right_key {
                        (left_key, right_key)
                    } else {
                        (right_key, left_key)
                    };
                    let job = if let Some(&job) = deferred_union_ids.get(&key) {
                        job
                    } else {
                        let job = deferred_union_jobs.len();
                        deferred_union_ids.insert(key, job);
                        deferred_union_jobs.push((left_weight.clone(), right_weight.clone()));
                        job
                    };
                    PendingUnionWeight::Deferred(job)
                } else {
                    PendingUnionWeight::Immediate(weight_ops.union(&left_weight, &right_weight))
                };

                if left_target == right_target {
                    let target = intern_singleton(
                        left_target,
                        singleton_states,
                        singleton_count,
                        states,
                        queue,
                    );
                    return Some((target, pending_weight));
                }

                let mut normalized = ResidualSubset::new();
                if left_target < right_target {
                    normalized.push((left_target, left_weight));
                    normalized.push((right_target, right_weight));
                } else {
                    normalized.push((right_target, right_weight));
                    normalized.push((left_target, left_weight));
                }
                let target = finish_subset(
                    normalized,
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                    subset_states,
                );
                Some((target, pending_weight))
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
                    singleton_states,
                    singleton_count,
                    states,
                    queue,
                    subset_states,
                );
                Some((target, PendingUnionWeight::Immediate(edge_weight)))
            }
        }
    }

    let preallocate_raw_singletons =
        crate::env::var_os("GLRMASK_EXPERIMENT_PREALLOCATE_RAW_SINGLETONS").is_some();
    let mut preallocated_nondeterministic = Vec::<u32>::new();
    let (mut states, mut singleton_states, mut singleton_count) = if preallocate_raw_singletons {
        let states = raw_states
            .iter()
            .enumerate()
            .map(|(raw_state, source)| {
                if source.transitions.values().any(|targets| targets.len() > 1) {
                    preallocated_nondeterministic.push(raw_state as u32);
                    return DWAState::default();
                }
                let source_has_default = source.transitions.contains_key(&DEFAULT_LABEL);
                let transitions = source
                    .transitions
                    .iter()
                    .filter_map(|(&label, targets)| {
                        let (target, edge_weight) = targets.first()?;
                        if edge_weight.is_empty()
                            && !(source_has_default && label >= 0 && label != DEFAULT_LABEL)
                        {
                            return None;
                        }
                        Some((label, (*target, edge_weight.clone())))
                    })
                    .collect();
                let final_weight = source
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                DWAState {
                    transitions,
                    final_weight,
                }
            })
            .collect::<Vec<_>>();
        let singleton_states = (0..raw_states.len() as u32).collect::<Vec<_>>();
        let singleton_count = raw_states.len();
        (states, singleton_states, singleton_count)
    } else {
        (
            Vec::<DWAState>::new(),
            vec![u32::MAX; raw_states.len()],
            0usize,
        )
    };
    let mut dead_shadow_state = None::<u32>;
    let mut queue = VecDeque::<(u32, ResidualSubset)>::new();
    if preallocate_raw_singletons {
        for raw_state in preallocated_nondeterministic.iter().copied() {
            let mut subset = ResidualSubset::new();
            subset.push((raw_state, Weight::all()));
            queue.push_back((raw_state, subset));
        }
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_raw_singleton_preallocation] raw_states={} nondeterministic_rows={}",
                raw_states.len(),
                preallocated_nondeterministic.len(),
            );
        }
    }
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
                .map(|(state, weight)| (*state, weight.clone()))
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
    let mut profiled_pair_left_only_labels = 0usize;
    let mut profiled_pair_right_only_labels = 0usize;
    let mut profiled_pair_both_labels = 0usize;
    let mut profiled_pair_left_prefix_full = 0usize;
    let mut profiled_pair_right_prefix_full = 0usize;
    let mut profiled_pair_cached_default_uses = 0usize;
    let mut profiled_pair_owner_counts = BTreeMap::<(usize, usize), usize>::new();
    let mut profiled_boundary_pair_states = FxHashMap::<u32, usize>::default();
    let mut profiled_pair_left_residuals = FxHashSet::<(u32, usize)>::default();
    let mut profiled_pair_right_residuals = FxHashSet::<(u32, usize)>::default();
    let mut profiled_pair_left_raw_states = FxHashSet::<u32>::default();
    let mut profiled_pair_right_raw_states = FxHashSet::<u32>::default();
    type ResidualEdgeIntersections = SmallVec<[(usize, Weight); 8]>;
    let mut cached_residual_rows =
        FxHashMap::<(u32, usize), Arc<ResidualEdgeIntersections>>::default();
    let use_residual_row_cache =
        crate::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_RESIDUAL_ROW_CACHE").is_none();
    let use_interval_pair_runs =
        crate::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_INTERVAL_RUNS").is_none();
    let defer_support_unions =
        crate::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_DEFER_SUPPORT_UNIONS").is_none()
            && rayon::current_num_threads() > 1;
    let mut deferred_union_ids = FxHashMap::<(usize, usize), usize>::default();
    let mut deferred_union_jobs = Vec::<(Weight, Weight)>::new();
    let mut deferred_weight_patches = Vec::<DeferredWeightPatchRun>::new();
    let mut deferred_final_union_ids = FxHashMap::<SmallVec<[usize; 4]>, usize>::default();
    let mut deferred_final_union_jobs = Vec::<SmallVec<[Weight; 4]>>::new();
    let mut deferred_final_weight_patches = Vec::<(u32, usize)>::new();
    let profiled_local_intersection_lookups = std::cell::Cell::new(0usize);
    let profiled_global_intersection_lookups = std::cell::Cell::new(0usize);
    let profiled_row_cache_build_intersections = std::cell::Cell::new(0usize);
    let residual_prefetch_min_raw_states = crate::env::var(
        "GLRMASK_DIRECT_UNION_RESIDUAL_PREFETCH_MIN_RAW_STATES",
    )
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .unwrap_or(65_536);
    let use_parallel_residual_row_prefetch = use_residual_row_cache
        && raw_states.len() >= residual_prefetch_min_raw_states
        && crate::env::var_os("GLRMASK_DISABLE_PARALLEL_RESIDUAL_ROW_PREFETCH").is_none()
        && rayon::current_num_threads() > 1;
    let residual_row_prefetch_batch = crate::env::var("GLRMASK_RESIDUAL_ROW_PREFETCH_BATCH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8192)
        .max(1);
    let mut residual_row_prefetch_remaining = 0usize;
    let mut profiled_residual_row_prefetch_batches = 0usize;
    let mut profiled_residual_row_prefetch_rows = 0usize;
    let mut profiled_residual_row_prefetch_intersection_jobs = 0usize;
    let mut profiled_residual_row_prefetch_ms = 0.0f64;
    while let Some((output_state, subset)) = queue.pop_front() {
        if use_parallel_residual_row_prefetch && residual_row_prefetch_remaining == 0 {
            let started_at = Instant::now();
            let covered_states = (queue.len() + 1).min(residual_row_prefetch_batch);
            let mut missing = FxHashMap::<(u32, usize), (u32, Weight)>::default();
            for candidate in std::iter::once(&subset)
                .chain(queue.iter().map(|(_, subset)| subset))
                .take(covered_states)
            {
                if candidate.len() != 2 {
                    continue;
                }
                for (raw_state, prefix) in candidate {
                    let key = (*raw_state, prefix.ptr_key());
                    if !cached_residual_rows.contains_key(&key) {
                        missing
                            .entry(key)
                            .or_insert_with(|| (*raw_state, prefix.clone()));
                    }
                }
            }
            let jobs = missing.into_iter().collect::<Vec<_>>();
            let dedup_intersection_jobs =
                crate::env::var_os("GLRMASK_DISABLE_DEDUP_RESIDUAL_PREFETCH_JOBS").is_none();
            let prepared = if dedup_intersection_jobs {
                // Read source rows in parallel, then intern exact
                // (residual-prefix, source-edge-weight) algebra jobs across the
                // whole frontier batch.  The row cache itself is keyed by raw
                // state because different rows can expose different labels,
                // but their Weight intersections are often identical.
                let row_specs = jobs
                    .par_iter()
                    .map(|(key, (raw_state, prefix))| {
                        let source = &raw_states[*raw_state as usize];
                        let mut edges = SmallVec::<[(usize, Weight); 8]>::new();
                        for targets in source.transitions.values() {
                            for (_, edge_weight) in targets {
                                let edge_key = edge_weight.ptr_key();
                                if edges.iter().any(|(key, _)| *key == edge_key) {
                                    continue;
                                }
                                edges.push((edge_key, edge_weight.clone()));
                            }
                        }
                        edges.sort_unstable_by_key(|(key, _)| *key);
                        (*key, prefix.clone(), edges)
                    })
                    .collect::<Vec<_>>();
                let mut intersection_job_ids = FxHashMap::<(usize, usize), usize>::default();
                let mut intersection_jobs = Vec::<(Weight, Weight)>::new();
                let mut indexed_rows = Vec::<((u32, usize), SmallVec<[(usize, usize); 8]>)>::with_capacity(row_specs.len());
                for (row_key, prefix, edges) in row_specs {
                    let prefix_key = prefix.ptr_key();
                    let mut indexed = SmallVec::<[(usize, usize); 8]>::new();
                    for (edge_key, edge_weight) in edges {
                        let pair_key = if prefix_key <= edge_key {
                            (prefix_key, edge_key)
                        } else {
                            (edge_key, prefix_key)
                        };
                        let job = if let Some(&job) = intersection_job_ids.get(&pair_key) {
                            job
                        } else {
                            let job = intersection_jobs.len();
                            intersection_job_ids.insert(pair_key, job);
                            intersection_jobs.push((prefix.clone(), edge_weight));
                            job
                        };
                        indexed.push((edge_key, job));
                    }
                    indexed_rows.push((row_key, indexed));
                }
                profiled_residual_row_prefetch_intersection_jobs += intersection_jobs.len();
                let intersections = intersection_jobs
                    .par_iter()
                    .map(|(left, right)| left.intersection_uncached(right))
                    .collect::<Vec<_>>();
                indexed_rows
                    .into_iter()
                    .map(|(row_key, indexed)| {
                        let row = indexed
                            .into_iter()
                            .map(|(edge_key, job)| (edge_key, intersections[job].clone()))
                            .collect::<ResidualEdgeIntersections>();
                        (row_key, Arc::new(row))
                    })
                    .collect::<Vec<_>>()
            } else {
                jobs
                    .par_iter()
                    .map(|(key, (raw_state, prefix))| {
                        let source = &raw_states[*raw_state as usize];
                        let mut weight_ops = ScopedWeightOpCache::default();
                        let mut edge_weights = ResidualEdgeIntersections::new();
                        for targets in source.transitions.values() {
                            for (_, edge_weight) in targets {
                                let edge_key = edge_weight.ptr_key();
                                if edge_weights.iter().any(|(key, _)| *key == edge_key) {
                                    continue;
                                }
                                edge_weights.push((
                                    edge_key,
                                    weight_ops.intersection(prefix, edge_weight),
                                ));
                            }
                        }
                        edge_weights.sort_unstable_by_key(|(key, _)| *key);
                        (*key, Arc::new(edge_weights))
                    })
                    .collect::<Vec<_>>()
            };
            profiled_row_cache_build_intersections.set(
                profiled_row_cache_build_intersections.get()
                    + prepared.iter().map(|(_, row)| row.len()).sum::<usize>(),
            );
            profiled_residual_row_prefetch_rows += prepared.len();
            for (key, row) in prepared {
                cached_residual_rows.entry(key).or_insert(row);
            }
            profiled_residual_row_prefetch_batches += 1;
            profiled_residual_row_prefetch_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            residual_row_prefetch_remaining = covered_states;
        }
        if use_parallel_residual_row_prefetch {
            residual_row_prefetch_remaining = residual_row_prefetch_remaining.saturating_sub(1);
        }
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
                let source_has_default = source.transitions.contains_key(&DEFAULT_LABEL);
                for (&label, targets) in &source.transitions {
                    let Some((target, edge_weight)) = targets.first() else {
                        continue;
                    };
                    if edge_weight.is_empty()
                        && !(source_has_default && label >= 0 && label != DEFAULT_LABEL)
                    {
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
        let final_weight = match final_parts.len() {
            0 => None,
            1 => final_parts.pop(),
            _ if defer_support_unions => {
                final_parts.sort_unstable_by_key(Weight::ptr_key);
                let key = final_parts
                    .iter()
                    .map(Weight::ptr_key)
                    .collect::<SmallVec<[usize; 4]>>();
                let job = if let Some(&job) = deferred_final_union_ids.get(&key) {
                    job
                } else {
                    let job = deferred_final_union_jobs.len();
                    deferred_final_union_ids.insert(key, job);
                    deferred_final_union_jobs.push(final_parts);
                    job
                };
                deferred_final_weight_patches.push((output_state, job));
                None
            }
            _ => {
                let weight = weight_ops.union_all(final_parts.iter());
                (!weight.is_empty()).then_some(weight)
            }
        };
        let mut output_transitions = Vec::<(i32, (u32, Weight))>::new();

        if subset.len() == 2 {
            let owner_pair = (
                raw_owners[subset[0].0 as usize],
                raw_owners[subset[1].0 as usize],
            );
            *profiled_pair_owner_counts.entry(owner_pair).or_default() += 1;
            if owner_pair.1 == 2 {
                *profiled_boundary_pair_states
                    .entry(raw_locals[subset[1].0 as usize])
                    .or_default() += 1;
            } else if owner_pair.0 == 2 {
                *profiled_boundary_pair_states
                    .entry(raw_locals[subset[0].0 as usize])
                    .or_default() += 1;
            }
            let left_source = &raw_states[subset[0].0 as usize];
            let right_source = &raw_states[subset[1].0 as usize];
            let left_prefix = &subset[0].1;
            let right_prefix = &subset[1].1;
            profiled_pair_left_raw_states.insert(subset[0].0);
            profiled_pair_right_raw_states.insert(subset[1].0);
            profiled_pair_left_residuals.insert((subset[0].0, left_prefix.ptr_key()));
            profiled_pair_right_residuals.insert((subset[1].0, right_prefix.ptr_key()));
            let mut prepare_residual_row = |
                raw_state: u32,
                prefix: &Weight,
                source: &crate::automata::weighted_u32::nwa::NWAState,
                weight_ops: &mut ScopedWeightOpCache,
            | -> Option<Arc<ResidualEdgeIntersections>> {
                if !use_residual_row_cache {
                    return None;
                }
                let key = (raw_state, prefix.ptr_key());
                if let Some(cached) = cached_residual_rows.get(&key) {
                    return Some(Arc::clone(cached));
                }
                let mut edge_weights = ResidualEdgeIntersections::new();
                for targets in source.transitions.values() {
                    for (_, edge_weight) in targets {
                        let edge_key = edge_weight.ptr_key();
                        if edge_weights.iter().any(|(key, _)| *key == edge_key) {
                            continue;
                        }
                        profiled_row_cache_build_intersections.set(
                            profiled_row_cache_build_intersections.get() + 1,
                        );
                        edge_weights.push((
                            edge_key,
                            weight_ops.intersection(prefix, edge_weight),
                        ));
                    }
                }
                edge_weights.sort_unstable_by_key(|(key, _)| *key);
                let cached = Arc::new(edge_weights);
                cached_residual_rows.insert(key, Arc::clone(&cached));
                Some(cached)
            };
            let left_residual_row = prepare_residual_row(
                subset[0].0,
                left_prefix,
                left_source,
                &mut weight_ops,
            );
            let right_residual_row = prepare_residual_row(
                subset[1].0,
                right_prefix,
                right_source,
                &mut weight_ops,
            );
            let restricted_intersection = |
                cached: Option<&ResidualEdgeIntersections>,
                prefix: &Weight,
                edge_weight: &Weight,
                weight_ops: &mut ScopedWeightOpCache,
            | -> Weight {
                if let Some(cached) = cached {
                    profiled_local_intersection_lookups.set(
                        profiled_local_intersection_lookups.get() + 1,
                    );
                    let key = edge_weight.ptr_key();
                    let index = cached
                        .binary_search_by_key(&key, |(cached_key, _)| *cached_key)
                        .expect("residual row cache must cover every source edge weight");
                    cached[index].1.clone()
                } else {
                    profiled_global_intersection_lookups.set(
                        profiled_global_intersection_lookups.get() + 1,
                    );
                    weight_ops.intersection(prefix, edge_weight)
                }
            };
            if left_prefix.is_full() {
                profiled_pair_left_prefix_full += 1;
            }
            if right_prefix.is_full() {
                profiled_pair_right_prefix_full += 1;
            }
            let left_default = left_source.transitions.get(&DEFAULT_LABEL);
            let right_default = right_source.transitions.get(&DEFAULT_LABEL);
            let has_symbolic_default = default_positive_label_count.is_some()
                && (left_default.is_some() || right_default.is_some());
            let build_default_contributions = |
                targets: Option<&Vec<(u32, Weight)>>,
                cached: Option<&ResidualEdgeIntersections>,
                prefix: &Weight,
                weight_ops: &mut ScopedWeightOpCache,
            | {
                let mut contributions = SmallVec::<[(u32, Weight); 2]>::new();
                if let Some(targets) = targets {
                    for (target, edge_weight) in targets {
                        let contribution = restricted_intersection(
                            cached,
                            prefix,
                            edge_weight,
                            weight_ops,
                        );
                        if !contribution.is_empty() {
                            contributions.push((*target, contribution));
                        }
                    }
                }
                contributions
            };
            let left_default_contributions = build_default_contributions(
                left_default,
                left_residual_row.as_deref(),
                left_prefix,
                &mut weight_ops,
            );
            let right_default_contributions = build_default_contributions(
                right_default,
                right_residual_row.as_deref(),
                right_prefix,
                &mut weight_ops,
            );
            if use_interval_pair_runs {
                // Exact run-wise version of the label merge below.  Every
                // boundary is a start/end+1 of a maximal source run (plus 0,
                // where DEFAULT begins to apply).  Therefore both raw target
                // vectors and both residual-prefix intersections are constant
                // throughout each interval.  Solve the weighted successor once
                // per interval, then expand that already-computed ordinary DWA
                // transition across its explicit labels.
                let left_runs = explicit_transition_runs(left_source);
                let right_runs = explicit_transition_runs(right_source);
                let mut boundaries = Vec::<i64>::with_capacity(
                    2 * (left_runs.len() + right_runs.len()) + 1,
                );
                for (start, end, _) in left_runs.iter().chain(right_runs.iter()) {
                    boundaries.push(i64::from(*start));
                    boundaries.push(i64::from(*end) + 1);
                }
                if has_symbolic_default {
                    // Negative explicit labels never fall through to DEFAULT;
                    // non-negative labels do.  No source run can cross the
                    // excluded DEFAULT_LABEL itself, but 0 must still be an
                    // interval boundary for the fallback rule.
                    boundaries.push(0);
                }
                boundaries.sort_unstable();
                boundaries.dedup();

                let mut left_position = 0usize;
                let mut right_position = 0usize;
                for interval in boundaries.windows(2) {
                    let interval_start = interval[0];
                    let interval_end = interval[1] - 1;
                    if interval_start > interval_end
                        || interval_start < i64::from(i32::MIN)
                        || interval_end > i64::from(i32::MAX)
                    {
                        continue;
                    }
                    let label = interval_start as i32;
                    while left_position < left_runs.len()
                        && left_runs[left_position].1 < label
                    {
                        left_position += 1;
                    }
                    while right_position < right_runs.len()
                        && right_runs[right_position].1 < label
                    {
                        right_position += 1;
                    }
                    let left_explicit = left_runs.get(left_position).and_then(|run| {
                        (run.0 <= label && label <= run.1).then_some(run.2)
                    });
                    let right_explicit = right_runs.get(right_position).and_then(|run| {
                        (run.0 <= label && label <= run.1).then_some(run.2)
                    });
                    if left_explicit.is_none() && right_explicit.is_none() {
                        // A gap covered only by DEFAULT is represented by the
                        // one symbolic DEFAULT transition below, never by a
                        // redundant explicit row.
                        continue;
                    }
                    let span = (interval_end - interval_start + 1) as usize;
                    profiled_explicit_labels += span;
                    let left_targets = left_explicit.or_else(|| {
                        (label >= 0).then_some(left_default).flatten()
                    });
                    let right_targets = right_explicit.or_else(|| {
                        (label >= 0).then_some(right_default).flatten()
                    });
                    match (left_targets.is_some(), right_targets.is_some()) {
                        (true, true) => profiled_pair_both_labels += span,
                        (true, false) => profiled_pair_left_only_labels += span,
                        (false, true) => profiled_pair_right_only_labels += span,
                        (false, false) => unreachable!(
                            "run interval has an explicit label on at least one side"
                        ),
                    }

                    let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                    if let Some(targets) = left_explicit {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                left_residual_row.as_deref(),
                                left_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    } else if left_targets.is_some() {
                        profiled_pair_cached_default_uses += span;
                        contributions.extend(left_default_contributions.iter().cloned());
                    }
                    if let Some(targets) = right_explicit {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                right_residual_row.as_deref(),
                                right_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    } else if right_targets.is_some() {
                        profiled_pair_cached_default_uses += span;
                        contributions.extend(right_default_contributions.iter().cloned());
                    }

                    if let Some((target, edge_weight)) = finish_overlap_transition(
                        contributions,
                        &mut weight_ops,
                        defer_support_unions,
                        &mut deferred_union_ids,
                        &mut deferred_union_jobs,
                        &mut singleton_states,
                        &mut singleton_count,
                        &mut states,
                        &mut queue,
                        &mut subset_states,
                    ) {
                        append_pending_transition_run(
                            output_state,
                            interval_start as i32,
                            interval_end as i32,
                            target,
                            edge_weight,
                            &mut output_transitions,
                            &mut deferred_weight_patches,
                            false,
                        );
                    } else if label >= 0 && has_symbolic_default {
                        let dead = *dead_shadow_state.get_or_insert_with(|| {
                            let dead = states.len() as u32;
                            states.push(DWAState::default());
                            dead
                        });
                        for explicit_label in (interval_start as i32)..=(interval_end as i32) {
                            output_transitions.push((
                                explicit_label,
                                (dead, Weight::empty()),
                            ));
                        }
                    }
                }
            } else {
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
                match (left_targets.is_some(), right_targets.is_some()) {
                    (true, true) => profiled_pair_both_labels += 1,
                    (true, false) => profiled_pair_left_only_labels += 1,
                    (false, true) => profiled_pair_right_only_labels += 1,
                    (false, false) => unreachable!("merged explicit label must exist on at least one side"),
                }
                let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                if left_label == Some(label) {
                    if let Some(targets) = left_targets {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                left_residual_row.as_deref(),
                                left_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    }
                } else if left_targets.is_some() {
                    profiled_pair_cached_default_uses += 1;
                    contributions.extend(left_default_contributions.iter().cloned());
                }
                if right_label == Some(label) {
                    if let Some(targets) = right_targets {
                        for (target, edge_weight) in targets {
                            let contribution = restricted_intersection(
                                right_residual_row.as_deref(),
                                right_prefix,
                                edge_weight,
                                &mut weight_ops,
                            );
                            if !contribution.is_empty() {
                                contributions.push((*target, contribution));
                            }
                        }
                    }
                } else if right_targets.is_some() {
                    profiled_pair_cached_default_uses += 1;
                    contributions.extend(right_default_contributions.iter().cloned());
                }
                if let Some((target, edge_weight)) = finish_overlap_transition(
                    contributions,
                    &mut weight_ops,
                    defer_support_unions,
                    &mut deferred_union_ids,
                    &mut deferred_union_jobs,
                    &mut singleton_states,
                    &mut singleton_count,
                    &mut states,
                    &mut queue,
                    &mut subset_states,
                ) {
                    append_pending_transition_run(
                        output_state,
                        label,
                        label,
                        target,
                        edge_weight,
                        &mut output_transitions,
                        &mut deferred_weight_patches,
                        false,
                    );
                } else if label >= 0 && has_symbolic_default {
                    let dead = *dead_shadow_state.get_or_insert_with(|| {
                        let dead = states.len() as u32;
                        states.push(DWAState::default());
                        dead
                    });
                    output_transitions.push((label, (dead, Weight::empty())));
                }
            }
            }

            if has_symbolic_default {
                let mut contributions = SmallVec::<[(u32, Weight); 4]>::new();
                contributions.extend(left_default_contributions.iter().cloned());
                contributions.extend(right_default_contributions.iter().cloned());
                if let Some((target, edge_weight)) = finish_overlap_transition(
                    contributions,
                    &mut weight_ops,
                    defer_support_unions,
                    &mut deferred_union_ids,
                    &mut deferred_union_jobs,
                    &mut singleton_states,
                    &mut singleton_count,
                    &mut states,
                    &mut queue,
                    &mut subset_states,
                ) {
                    append_pending_transition_run(
                        output_state,
                        DEFAULT_LABEL,
                        DEFAULT_LABEL,
                        target,
                        edge_weight,
                        &mut output_transitions,
                        &mut deferred_weight_patches,
                        true,
                    );
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
                defer_support_unions,
                &mut deferred_union_ids,
                &mut deferred_union_jobs,
                &mut singleton_states,
                &mut singleton_count,
                &mut states,
                &mut queue,
                &mut subset_states,
            ) {
                append_pending_transition_run(
                    output_state,
                    label,
                    label,
                    target,
                    edge_weight,
                    &mut output_transitions,
                    &mut deferred_weight_patches,
                    label == DEFAULT_LABEL,
                );
            } else if label != DEFAULT_LABEL && label >= 0 && include_default {
                let dead = *dead_shadow_state.get_or_insert_with(|| {
                    let dead = states.len() as u32;
                    states.push(DWAState::default());
                    dead
                });
                output_transitions.push((label, (dead, Weight::empty())));
            }
        }
        states[output_state as usize] = DWAState {
            transitions: output_transitions.into_iter().collect(),
            final_weight,
        };
    }

    let deferred_union_started_at = Instant::now();
    let deferred_union_results = if defer_support_unions {
        deferred_union_jobs
            .par_iter()
            .map(|(left, right)| Weight::union_all_direct([left, right]))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let deferred_union_ms = deferred_union_started_at.elapsed().as_secs_f64() * 1000.0;
    let deferred_final_union_started_at = Instant::now();
    let deferred_final_union_results = if defer_support_unions {
        deferred_final_union_jobs
            .par_iter()
            .map(|weights| Weight::union_all_direct(weights.iter()))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let deferred_final_union_ms =
        deferred_final_union_started_at.elapsed().as_secs_f64() * 1000.0;

    let deferred_patch_started_at = Instant::now();
    if defer_support_unions {
        for patch in &deferred_weight_patches {
            let weight = &deferred_union_results[patch.job];
            debug_assert!(!weight.is_empty());
            for (_, (_, edge_weight)) in states[patch.state as usize]
                .transitions
                .range_mut(patch.start..=patch.end)
            {
                *edge_weight = weight.clone();
            }
        }
        for &(state, job) in &deferred_final_weight_patches {
            let weight = deferred_final_union_results[job].clone();
            debug_assert!(!weight.is_empty());
            states[state as usize].final_weight = Some(weight);
        }
    }
    let deferred_patch_ms = deferred_patch_started_at.elapsed().as_secs_f64() * 1000.0;

    let synthetic_states = states.len().saturating_sub(singleton_count);
    if compose_profile_enabled() {
        let left_residual_row_labels = profiled_pair_left_residuals
            .iter()
            .map(|(state, _)| raw_states[*state as usize].transitions.len())
            .sum::<usize>();
        let right_residual_row_labels = profiled_pair_right_residuals
            .iter()
            .map(|(state, _)| raw_states[*state as usize].transitions.len())
            .sum::<usize>();
        let count_distinct_row_weights = |state: u32| {
            raw_states[state as usize]
                .transitions
                .values()
                .flat_map(|targets| targets.iter().map(|(_, weight)| weight.ptr_key()))
                .collect::<FxHashSet<_>>()
                .len()
        };
        let count_explicit_row_runs = |state: u32| {
            let source = &raw_states[state as usize];
            let mut labels = 0usize;
            let mut runs = 0usize;
            let mut previous: Option<(i32, &Vec<(u32, Weight)>)> = None;
            for (&label, targets) in source
                .transitions
                .iter()
                .filter(|(label, _)| **label != DEFAULT_LABEL)
            {
                labels += 1;
                let continues = previous.is_some_and(|(previous_label, previous_targets)| {
                    previous_label.checked_add(1) == Some(label)
                        && same_raw_targets(previous_targets, targets)
                });
                if !continues {
                    runs += 1;
                }
                previous = Some((label, targets));
            }
            (labels, runs)
        };
        let (left_residual_explicit_labels, left_residual_runs) = profiled_pair_left_residuals
            .iter()
            .map(|(state, _)| count_explicit_row_runs(*state))
            .fold((0usize, 0usize), |(labels, runs), next| {
                (labels + next.0, runs + next.1)
            });
        let (right_residual_explicit_labels, right_residual_runs) = profiled_pair_right_residuals
            .iter()
            .map(|(state, _)| count_explicit_row_runs(*state))
            .fold((0usize, 0usize), |(labels, runs), next| {
                (labels + next.0, runs + next.1)
            });
        let left_residual_row_weights = profiled_pair_left_residuals
            .iter()
            .map(|(state, _)| count_distinct_row_weights(*state))
            .sum::<usize>();
        let right_residual_row_weights = profiled_pair_right_residuals
            .iter()
            .map(|(state, _)| count_distinct_row_weights(*state))
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_overlap_pair_owners] counts={profiled_pair_owner_counts:?}"
        );
        let mut boundary_pair_states = profiled_boundary_pair_states.into_iter().collect::<Vec<_>>();
        boundary_pair_states.sort_unstable_by_key(|&(state, count)| (std::cmp::Reverse(count), state));
        boundary_pair_states.truncate(24);
        eprintln!(
            "[glrmask/profile][constraint_overlap_boundary_state_pairs] top={boundary_pair_states:?}"
        );
        eprintln!(
            "[glrmask/profile][constraint_overlap_local_shape] raw_states={} result_states={} singleton_states={} raw_singletons_preallocated={} pair_subsets={} wide_subsets={} max_subset={} explicit_labels={} pair_left_only_labels={} pair_right_only_labels={} pair_both_labels={} pair_left_prefix_full={} pair_right_prefix_full={} pair_cached_default_uses={} left_raw_states={} right_raw_states={} left_residual_rows={} right_residual_rows={} left_residual_row_labels={} right_residual_row_labels={} left_residual_explicit_labels={} right_residual_explicit_labels={} left_residual_runs={} right_residual_runs={} left_residual_row_weights={} right_residual_row_weights={} cached_residual_rows={} cached_residual_edge_weights={} local_intersection_lookups={} global_intersection_lookups={} row_cache_build_intersections={} residual_prefetch_batches={} residual_prefetch_rows={} residual_prefetch_intersection_jobs={} residual_prefetch_ms={:.3} union_cache_entries={} intersection_cache_entries={} deferred_union_jobs={} deferred_patch_runs={} deferred_final_union_jobs={} deferred_final_patches={} deferred_union_ms={deferred_union_ms:.3} deferred_final_union_ms={deferred_final_union_ms:.3} deferred_patch_ms={deferred_patch_ms:.3}",
            raw_states.len(),
            states.len(),
            profiled_singletons,
            preallocate_raw_singletons,
            profiled_pair_subsets,
            profiled_wide_subsets,
            profiled_max_subset,
            profiled_explicit_labels,
            profiled_pair_left_only_labels,
            profiled_pair_right_only_labels,
            profiled_pair_both_labels,
            profiled_pair_left_prefix_full,
            profiled_pair_right_prefix_full,
            profiled_pair_cached_default_uses,
            profiled_pair_left_raw_states.len(),
            profiled_pair_right_raw_states.len(),
            profiled_pair_left_residuals.len(),
            profiled_pair_right_residuals.len(),
            left_residual_row_labels,
            right_residual_row_labels,
            left_residual_explicit_labels,
            right_residual_explicit_labels,
            left_residual_runs,
            right_residual_runs,
            left_residual_row_weights,
            right_residual_row_weights,
            cached_residual_rows.len(),
            cached_residual_rows.values().map(|row| row.len()).sum::<usize>(),
            profiled_local_intersection_lookups.get(),
            profiled_global_intersection_lookups.get(),
            profiled_row_cache_build_intersections.get(),
            profiled_residual_row_prefetch_batches,
            profiled_residual_row_prefetch_rows,
            profiled_residual_row_prefetch_intersection_jobs,
            profiled_residual_row_prefetch_ms,
            weight_ops.union_entry_count(),
            weight_ops.intersection_entry_count(),
            deferred_union_jobs.len(),
            deferred_weight_patches.len(),
            deferred_final_union_jobs.len(),
            deferred_final_weight_patches.len(),
        );
    }
    Some((DWA::from_parts(states, start_state), synthetic_states))
}


struct UnmappedComponentParserArtifact {
    automaton: NWA,
    possible_matches: PossibleMatches,
}



/// Exact deferred union of already-compiled component parser DWAs.
///
/// The semantic union is represented by multiple NWA start components; no
/// parser row, parser weight, or mapped transition is copied while constructing
/// the view.  In the overwhelmingly common composition case every local LR
/// state has exactly one composed-state image, so retain that relation as one
/// flat `u32` per local state.  The eventual component+boundary parser consumer
/// may materialize the view and determinize the combined NWA in one pass.
///
/// This intentionally makes the *component union itself* the cheap operation:
/// expensive parser-state label transport and weight-coordinate publication are
/// properties of later realization, not of forming the union language.
enum DeferredParserStateRelation {
    Singleton(Vec<u32>),
    General(Vec<Vec<u32>>),
}

impl DeferredParserStateRelation {
    fn from_relation(relation: &[Vec<u32>]) -> Self {
        if relation.iter().all(|targets| targets.len() == 1) {
            Self::Singleton(relation.iter().map(|targets| targets[0]).collect())
        } else {
            Self::General(relation.to_vec())
        }
    }

    fn materialize(&self) -> std::borrow::Cow<'_, [Vec<u32>]> {
        match self {
            Self::Singleton(targets) => std::borrow::Cow::Owned(
                targets.iter().map(|&target| vec![target]).collect(),
            ),
            Self::General(relation) => std::borrow::Cow::Borrowed(relation),
        }
    }

    fn is_singleton(&self) -> bool {
        matches!(self, Self::Singleton(_))
    }
}

struct DeferredComponentParserUnionComponent<'a> {
    constraint: &'a Constraint,
    parser_state_relation: DeferredParserStateRelation,
    tokenizer_state_offset: u32,
    terminal_offset: u32,
    default_domain: Option<ParserDefaultDomain>,
}

struct DeferredComponentParserUnionView<'a> {
    components: Vec<DeferredComponentParserUnionComponent<'a>>,
    strip_scoped_ignore_identity: bool,
}

enum ComponentParserUnionWork<'a> {
    Deferred(DeferredComponentParserUnionView<'a>),
    Materialized(Vec<NWA>),
}

impl ComponentParserUnionWork<'_> {
    fn materialize_automata(self) -> Result<Vec<NWA>, String> {
        match self {
            Self::Deferred(view) => view.materialize_automata(),
            Self::Materialized(automata) => Ok(automata),
        }
    }

    fn view_shape(&self) -> Option<(usize, usize)> {
        match self {
            Self::Deferred(view) => Some((
                view.component_count(),
                view.singleton_relation_count(),
            )),
            Self::Materialized(_) => None,
        }
    }
}

impl<'a> DeferredComponentParserUnionView<'a> {
    fn new(
        constraints: &[&'a Constraint],
        parser_state_relations: &[Vec<Vec<u32>>],
        tokenizer_state_offsets: &[u32],
        terminal_offsets: &[u32],
        default_domains: &[Option<ParserDefaultDomain>],
        strip_scoped_ignore_identity: bool,
    ) -> Option<Self> {
        if constraints.len() != parser_state_relations.len()
            || constraints.len() != tokenizer_state_offsets.len()
            || constraints.len() != terminal_offsets.len()
            || constraints.len() != default_domains.len()
        {
            return None;
        }
        Some(Self {
            components: constraints
                .iter()
                .copied()
                .zip(parser_state_relations)
                .zip(tokenizer_state_offsets.iter().copied())
                .zip(terminal_offsets.iter().copied())
                .zip(default_domains.iter().cloned())
                .map(
                    |((((constraint, relation), tokenizer_state_offset), terminal_offset), default_domain)| DeferredComponentParserUnionComponent {
                    constraint,
                    parser_state_relation: DeferredParserStateRelation::from_relation(
                        relation,
                    ),
                    tokenizer_state_offset,
                    terminal_offset,
                    default_domain,
                },
                )
                .collect(),
            strip_scoped_ignore_identity,
        })
    }

    fn materialize_automata(self) -> Result<Vec<NWA>, String> {
        let strip_scoped_ignore_identity = self.strip_scoped_ignore_identity;
        let build = |component: DeferredComponentParserUnionComponent<'a>| {
                let relation = component.parser_state_relation.materialize();
                let parser_component = ParserDwaComponent {
                    constraint: component.constraint,
                    parser_state_relation: relation.as_ref(),
                    tokenizer_state_offset: component.tokenizer_state_offset,
                    terminal_offset: component.terminal_offset,
                };
                let mut automaton =
                    component_parser_nwa(&parser_component, component.default_domain.as_ref())?;
                if strip_scoped_ignore_identity {
                    let ignore_weight = component
                        .constraint
                        .ignore_terminal
                        .and_then(|ignore| component.constraint.runtime_possible_match_weight(ignore)).map(|w| w.to_weight());
                    strip_unscoped_ignore_identity(&mut automaton, ignore_weight.as_ref());
                }
                Ok(automaton)
        };
        if macro_parallelism_disabled() {
            let mut timings = Vec::with_capacity(self.components.len());
            let result = self
                .components
                .into_iter()
                .map(|component| {
                    let started = Instant::now();
                    let result = build(component);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect();
            report_macro_item_timings("compose_deferred_component_parser_materialize", &timings);
            result
        } else {
            self.components.into_par_iter().map(build).collect()
        }
    }

    fn component_count(&self) -> usize {
        self.components.len()
    }

    fn singleton_relation_count(&self) -> usize {
        self.components
            .iter()
            .filter(|component| component.parser_state_relation.is_singleton())
            .count()
    }
}


#[derive(Debug)]
struct BoundaryRefinementPlan {
    common_map: InternalIdMap,
    component_token_map: Option<Vec<Vec<u32>>>,
    boundary_tsid_map: Vec<Vec<u32>>,
    boundary_token_map: Vec<Vec<u32>>,
}

struct PreparedOwnedComponentArtifacts {
    /// Per-component maps into `id_map`. Parser automata are materialized only
    /// at the final parser-union boundary.
    automata_maps: Vec<DirectComponentCoordinateMaps>,
    possible_matches: PossibleMatches,
    id_map: InternalIdMap,
    boundary_tsid_map: Option<Vec<Vec<u32>>>,
    boundary_token_map: Option<Vec<Vec<u32>>>,
    token_mask_caches: Option<crate::runtime::InternalTokenMaskPrebuild>,
    token_mask_cache_ms: f64,
    remap_ms: f64,
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

fn reverse_hashcons_signed_acyclic_nwa_fast(nwa: NWA) -> NWA {
    reverse_hashcons_acyclic_nwa_fast_with_reverse_topo(nwa, None, true)
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
        .map(|((&constraint,&tokenizer_state_offset),&terminal_offset)|ParserDwaComponent {
            constraint,parser_state_relation:&[],tokenizer_state_offset,terminal_offset,
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
        .runtime_possible_match_terminals()
        .filter_map(|terminal| {
            component
                .constraint
                .runtime_possible_match_weight(terminal)
                .map(|weight| (terminal_offset + terminal, weight.to_weight()))
        })
        .collect())
}

fn compose_component_parser_dwas_and_possible_matches(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
    default_domains: &[Option<ParserDefaultDomain>],
    merged_tokenizer_state_count: usize,
    original_token_ids: &[u32],
    strip_scoped_ignore_identity: bool,
) -> Result<(MappedArtifact<(DWA, PossibleMatches)>, BTreeMap<i32, Weight>), String> {
    if components.is_empty() {
        return Err("cannot compose zero parser DWAs".into());
    }
    if terminal_offsets.len() != components.len() || default_domains.len() != components.len() {
        return Err(format!(
            "terminal-offset/default-domain count ({}/{}) does not match component count {}",
            terminal_offsets.len(),
            default_domains.len(),
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
    let jobs = components
        .iter()
        .copied()
        .zip(terminal_offsets.iter().copied())
        .zip(default_domains.iter())
        .zip(component_maps)
        .enumerate()
        .collect::<Vec<_>>();
    let prepare = |(component_index, (((component, terminal_offset), default_domain), coordinate_maps)): (
        usize,
        (((ParserDwaComponent<'_>, u32), &Option<ParserDefaultDomain>), DirectComponentCoordinateMaps),
    )| {
            let started_at = Instant::now();
            let mut parser_nwa = component_parser_nwa(&component, default_domain.as_ref())?;
            let parser_nwa_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            let started_at = Instant::now();
            let possible_matches = component_possible_matches(&component, terminal_offset)?;
            let possible_matches_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            if strip_scoped_ignore_identity {
                let ignore_weight = component
                    .constraint
                    .ignore_terminal
                    .and_then(|ignore| possible_matches.get(&(terminal_offset + ignore)));
                // The standalone parser's globally-erased trivia identity must
                // not leak into other scopes. The composed boundary parser
                // reintroduces any state-dependent visible terminal behavior
                // directly from the composed LR table.
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
    };
    let prepared = if macro_parallelism_disabled() {
        let mut timings = Vec::with_capacity(jobs.len());
        let result = jobs
            .into_iter()
            .map(|item| {
                let started = Instant::now();
                let result = prepare(item);
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
                result
            })
            .collect::<Result<Vec<_>, String>>()?;
        report_macro_item_timings("compose_component_parser_prepare", &timings);
        result
    } else {
        jobs.into_par_iter()
            .map(prepare)
            .collect::<Result<Vec<_>, String>>()?
    };
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
    let direct = if crate::env::var_os("GLRMASK_COMPOSE_GENERIC_COMPONENT_UNION").is_some() {
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
            if crate::env::var_os("GLRMASK_VALIDATE_COMPOSE_COMPONENT_DIRECT_UNION").is_some() {
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
    Ok((MappedArtifact::new((dwa, possible_matches), id_map), BTreeMap::new()))
}

fn explicit_parser_nwa(
    dwa: &DWA,
    num_parser_states: u32,
    extra_positive_labels: &[i32],
) -> NWA {
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
            for &label in extra_positive_labels {
                if label >= 0 && !state.transitions.contains_key(&label) {
                    nwa.add_transition(source as u32, label, *target, weight.clone());
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

    // If the component token partition already refines the boundary token
    // partition, their exact common refinement is the component partition
    // itself. Preserve that coordinate verbatim: the component parser DWA and
    // possible-matches need no token remap at all; only the boundary artifact
    // is lifted into the finer component classes.
    //
    // Treat "absent from the boundary map" as a partition cell of its own. A
    // component class containing both selected and unselected originals would
    // therefore be split by the boundary view and must use the generic meet
    // construction below.
    let direct_token_refinement_started_at = Instant::now();
    let component_refines_boundary = component_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .all(|originals| {
            let mut boundary_class = None::<u32>;
            for &original in originals {
                let class = boundary_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                match boundary_class {
                    None => boundary_class = Some(class),
                    Some(existing) if existing == class => {}
                    Some(_) => return false,
                }
            }
            true
        });
    let boundary_covered_by_component = boundary_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .flatten()
        .all(|&original| {
            component_map
                .vocab_tokens
                .original_to_internal
                .get(original as usize)
                .copied()
                .is_some_and(|token| token != u32::MAX)
        });
    if component_refines_boundary && boundary_covered_by_component {
        let mut boundary_token_map =
            vec![Vec::<u32>::new(); boundary_map.num_internal_tokens() as usize];
        for (boundary_token, originals) in boundary_map
            .vocab_tokens
            .internal_to_originals
            .iter()
            .enumerate()
        {
            let destinations = &mut boundary_token_map[boundary_token];
            for &original in originals {
                let component_token = component_map
                    .vocab_tokens
                    .original_to_internal
                    .get(original as usize)
                    .copied()
                    .unwrap_or(u32::MAX);
                debug_assert_ne!(component_token, u32::MAX);
                destinations.push(component_token);
            }
            destinations.sort_unstable();
            destinations.dedup();
        }
        let token_ms = direct_token_refinement_started_at.elapsed().as_secs_f64() * 1000.0;
        let boundary_started_at = Instant::now();
        let mut boundary_weights = boundary_dwa.weight_refs_mut();
        remap_weights_with_maps(
            &mut boundary_weights,
            &boundary_tsid_map,
            &boundary_token_map,
            component_map.num_tsids() as usize,
        );
        let boundary_ms = boundary_started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_direct_reconcile] mode=component_token_refinement component_tsids={} boundary_tsids={} common_tsids={} component_tokens={} boundary_tokens={} common_tokens={} tsid_ms={tsid_ms:.3} token_ms={token_ms:.3} component_remap_ms=0.000 boundary_remap_ms={boundary_ms:.3} total_ms={:.3}",
                component_map.num_tsids(),
                boundary_map.num_tsids(),
                component_map.num_tsids(),
                component_map.num_internal_tokens(),
                boundary_map.num_internal_tokens(),
                component_map.num_internal_tokens(),
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Some(MappedArtifact::new(
            (component_artifact, boundary_dwa),
            component_map,
        ));
    }

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


fn profile_exact_boundary_parser_delta(
    component_dwa: &DWA,
    boundary_dwa: &DWA,
    num_parser_states: u32,
) {
    if crate::env::var_os("GLRMASK_PROFILE_EXACT_BOUNDARY_PARSER_DELTA").is_none() {
        return;
    }
    let total_started = Instant::now();
    let extra_positive_labels = component_dwa
        .states()
        .iter()
        .chain(boundary_dwa.states())
        .flat_map(|state| state.transitions.keys().copied())
        .filter(|&label| label >= num_parser_states as i32 && label != DEFAULT_LABEL)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let explicit_started = Instant::now();
    let component_explicit = determinize(&explicit_parser_nwa(
        component_dwa,
        num_parser_states,
        &extra_positive_labels,
    ));
    let boundary_explicit = determinize(&explicit_parser_nwa(
        boundary_dwa,
        num_parser_states,
        &extra_positive_labels,
    ));
    let explicit_ms = explicit_started.elapsed().as_secs_f64() * 1000.0;
    let (Ok(component_explicit), Ok(boundary_explicit)) = (component_explicit, boundary_explicit) else {
        eprintln!("[glrmask/profile][constraint_exact_boundary_parser_delta] failed=explicitize explicit_ms={explicit_ms:.3}");
        return;
    };
    if !component_explicit.is_acyclic() || !boundary_explicit.is_acyclic() {
        eprintln!(
            "[glrmask/profile][constraint_exact_boundary_parser_delta] failed=cyclic component_states={} boundary_states={} explicit_ms={explicit_ms:.3}",
            component_explicit.num_states(),
            boundary_explicit.num_states(),
        );
        return;
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct Key {
        boundary: u32,
        component: u32,
        boundary_weight: usize,
        component_weight: usize,
    }
    let difference_started = Instant::now();
    let mut ops = ScopedWeightOpCache::default();
    let all = Weight::all();
    let empty = Weight::empty();
    let mut states = vec![DWAState::default()];
    let mut payloads = vec![(
        boundary_explicit.start_state(),
        Some(component_explicit.start_state()),
        all.clone(),
        all.clone(),
    )];
    let mut ids = FxHashMap::<Key, u32>::default();
    ids.insert(
        Key {
            boundary: boundary_explicit.start_state(),
            component: component_explicit.start_state(),
            boundary_weight: all.ptr_key(),
            component_weight: all.ptr_key(),
        },
        0,
    );
    let mut queue = VecDeque::from([0u32]);
    while let Some(out_state) = queue.pop_front() {
        let (boundary_state, component_state, boundary_prefix, component_prefix) =
            payloads[out_state as usize].clone();
        let boundary_row = &boundary_explicit.states()[boundary_state as usize];
        if let Some(boundary_final) = boundary_row.final_weight.as_ref() {
            let boundary_accept = ops.intersection(&boundary_prefix, boundary_final);
            let component_accept = component_state
                .and_then(|state| component_explicit.states()[state as usize].final_weight.as_ref())
                .map(|final_weight| ops.intersection(&component_prefix, final_weight))
                .unwrap_or_else(Weight::empty);
            let residual = ops.difference(&boundary_accept, &component_accept);
            if !residual.is_empty() {
                states[out_state as usize].final_weight = Some(residual);
            }
        }
        for (&label, (boundary_target, boundary_edge)) in &boundary_row.transitions {
            let next_boundary_weight = ops.intersection(&boundary_prefix, boundary_edge);
            if next_boundary_weight.is_empty() {
                continue;
            }
            let (next_component, next_component_weight) = if let Some(component_state) = component_state {
                if let Some((target, edge)) = component_explicit.states()[component_state as usize]
                    .transitions
                    .get(&label)
                {
                    let support = ops.intersection(&component_prefix, edge);
                    if support.is_empty() {
                        (None, empty.clone())
                    } else {
                        (Some(*target), support)
                    }
                } else {
                    (None, empty.clone())
                }
            } else {
                (None, empty.clone())
            };
            let key = Key {
                boundary: *boundary_target,
                component: next_component.unwrap_or(u32::MAX),
                boundary_weight: next_boundary_weight.ptr_key(),
                component_weight: next_component_weight.ptr_key(),
            };
            let target = if let Some(&target) = ids.get(&key) {
                target
            } else {
                let target = states.len() as u32;
                ids.insert(key, target);
                states.push(DWAState::default());
                payloads.push((
                    *boundary_target,
                    next_component,
                    next_boundary_weight,
                    next_component_weight,
                ));
                queue.push_back(target);
                target
            };
            states[out_state as usize]
                .transitions
                .insert(label, (target, Weight::all()));
        }
    }
    let raw = DWA::from_parts(states, 0);
    let raw_states = raw.num_states();
    let raw_transitions = raw.num_transitions();
    let raw_finals = raw
        .states()
        .iter()
        .filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()))
        .count();
    let minimized = reverse_hashcons_owned(raw);
    let minimized_states = minimized.num_states();
    let minimized_transitions = minimized.num_transitions();
    let minimized_finals = minimized
        .states()
        .iter()
        .filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()))
        .count();
    eprintln!(
        "[glrmask/profile][constraint_exact_boundary_parser_delta] component_input_states={} component_input_transitions={} boundary_input_states={} boundary_input_transitions={} explicit_component_states={} explicit_component_transitions={} explicit_boundary_states={} explicit_boundary_transitions={} extra_labels={} explicit_ms={explicit_ms:.3} raw_states={} raw_transitions={} raw_finals={} minimized_states={} minimized_transitions={} minimized_finals={} difference_ms={:.3} total_ms={:.3}",
        component_dwa.num_states(),
        component_dwa.num_transitions(),
        boundary_dwa.num_states(),
        boundary_dwa.num_transitions(),
        component_explicit.num_states(),
        component_explicit.num_transitions(),
        boundary_explicit.num_states(),
        boundary_explicit.num_transitions(),
        extra_positive_labels.len(),
        raw_states,
        raw_transitions,
        raw_finals,
        minimized_states,
        minimized_transitions,
        minimized_finals,
        difference_started.elapsed().as_secs_f64() * 1000.0,
        total_started.elapsed().as_secs_f64() * 1000.0,
    );
}

fn union_boundary_parser_dwa(
    component_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    boundary: MappedArtifact<DWA>,
    num_parser_states: u32,
) -> Result<MappedArtifact<(DWA, PossibleMatches)>, String> {
    let total_started_at = Instant::now();
    let pair_started_at = Instant::now();
    let paired = if crate::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_RECONCILE").is_some() {
        component_artifacts.pair_forced_common(boundary)
    } else {
        pair_boundary_into_component_refinement(component_artifacts, boundary)
            .expect("component map must cover the composed boundary coordinate domain")
    };
    let pair_ms = pair_started_at.elapsed().as_secs_f64() * 1000.0;
    let (((component_dwa, possible_matches), boundary_dwa), id_map) = paired.into_parts();
    profile_exact_boundary_parser_delta(&component_dwa, &boundary_dwa, num_parser_states);
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
        let extra_positive_labels = component_dwa
            .states()
            .iter()
            .chain(boundary_dwa.states())
            .flat_map(|state| state.transitions.keys().copied())
            .filter(|&label| label >= num_parser_states as i32 && label != DEFAULT_LABEL)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let generic_component_nwa = explicit_parser_nwa(
            &component_dwa,
            num_parser_states,
            &extra_positive_labels,
        );
        let generic_boundary_nwa = explicit_parser_nwa(
            &boundary_dwa,
            num_parser_states,
            &extra_positive_labels,
        );
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
    let direct = if crate::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_UNION").is_some() {
        None
    } else {
        determinize_epsilon_free_component_union(
            automata.iter().map(|automaton| (*automaton).clone()).collect(),
            Some(num_parser_states),
        )
    };
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    let (parser_dwa, union_path, synthetic_states, append_ms, determinize_ms) =
        if let Some((direct_dwa, synthetic_states)) = direct {
            if crate::env::var_os("GLRMASK_VALIDATE_COMPOSE_BOUNDARY_DIRECT_UNION").is_some() {
                let (reference, _, _) = build_generic()?;
                let extra_positive_labels = component_dwa
                    .states()
                    .iter()
                    .chain(boundary_dwa.states())
                    .flat_map(|state| state.transitions.keys().copied())
                    .filter(|&label| label >= num_parser_states as i32 && label != DEFAULT_LABEL)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let direct_explicit_nwa =
                    explicit_parser_nwa(&direct_dwa, num_parser_states, &extra_positive_labels);
                let direct_explicit =
                    determinize(&direct_explicit_nwa).map_err(|error| error.to_string())?;
                let difference = find_difference(&direct_explicit, &reference)
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
            "[glrmask/profile][constraint_parser_union] component_states={} boundary_states={} raw_states={} result_states={} result_transitions={} pair_ms={pair_ms:.3} explicit_ms={explicit_ms:.3} union_path={} direct_ms={direct_ms:.3} synthetic_states={} append_ms={append_ms:.3} determinize_ms={determinize_ms:.3} total_ms={:.3}",
            component_dwa.num_states(),
            boundary_dwa.num_states(),
            component_nwa.num_states() + boundary_nwa.num_states(),
            parser_dwa.num_states(),
            parser_dwa.num_transitions(),
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

fn transport_composition_template_dfa(
    mut dfa: UnweightedDfa,
    state_relation: &[Vec<u32>],
) -> Option<UnweightedDfa> {
    for state in &mut dfa.states {
        let old = std::mem::take(&mut state.transitions);
        let mut mapped = BTreeMap::new();
        for (label, target) in old {
            let mapped_label = remap_composition_template_label(label, state_relation)?;
            if let Some(previous) = mapped.insert(mapped_label, target)
                && previous != target
            {
                return None;
            }
        }
        state.transitions = mapped;
    }
    Some(dfa)
}

/// Transport a cached composition-template DFA and construct its NWA skeleton
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

fn unweighted_dfa_shortest_word(dfa: &UnweightedDfa) -> Option<Vec<i32>> {
    if dfa.states.is_empty() {
        return None;
    }
    let mut queue = VecDeque::from([dfa.start_state]);
    let mut previous = vec![None::<(u32, i32)>; dfa.states.len()];
    let mut seen = vec![false; dfa.states.len()];
    seen[dfa.start_state as usize] = true;
    while let Some(state) = queue.pop_front() {
        if dfa.states[state as usize].is_accepting {
            let mut word = Vec::new();
            let mut current = state;
            while current != dfa.start_state {
                let (parent, label) = previous[current as usize]?;
                word.push(label);
                current = parent;
            }
            word.reverse();
            return Some(word);
        }
        for (&label, &target) in &dfa.states[state as usize].transitions {
            if !seen[target as usize] {
                seen[target as usize] = true;
                previous[target as usize] = Some((state, label));
                queue.push_back(target);
            }
        }
    }
    None
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
            predecessors[target as usize].push(state_id as u32);
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
    if !productive.get(dfa.start_state as usize).copied().unwrap_or(false) {
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
    UnweightedDfa { states, start_state: remap[dfa.start_state as usize] }
}

fn rebuild_transported_component_templates(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
) -> BTreeMap<u32, UnweightedDfa> {
    let profile_transport =
        crate::env::var_os("GLRMASK_PROFILE_TEMPLATE_TRANSPORT").is_some();
    let mut result = BTreeMap::new();
    for (component_index, component) in components.iter().copied().enumerate() {
        let component_started = Instant::now();
        let terminal_offset = composed_table.terminal_offsets[component_index];
        let mut selected = vec![false; component.table.num_terminals as usize];
        for (local, selected_slot) in selected.iter_mut().enumerate() {
            let global = terminal_offset as usize + local;
            *selected_slot = active_terminals.get(global).copied().unwrap_or(false);
        }
        if !selected.iter().any(|&value| value) {
            continue;
        }
        let mut missing = vec![false; selected.len()];
        let relation = &composed_table.state_relations[component_index];
        let cached_started = Instant::now();
        let transport_cached = |(local_terminal, &is_selected): (usize, &bool)| {
                is_selected.then(|| {
                    let transported = component
                        .composition_parser_templates_by_terminal
                        .get(local_terminal)
                        .and_then(Option::as_ref)
                        .cloned()
                        .and_then(|dfa| transport_composition_template_dfa(dfa, relation));
                    (local_terminal, transported)
                })
        };
        let cached_results = if macro_parallelism_disabled() {
            let mut timings = Vec::new();
            let result = selected
                .iter()
                .enumerate()
                .filter_map(|item| {
                    let started = Instant::now();
                    let result = transport_cached(item);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect::<Vec<_>>();
            report_macro_item_timings("compose_cached_template_transport_selected", &timings);
            result
        } else {
            selected
                .par_iter()
                .enumerate()
                .filter_map(transport_cached)
                .collect::<Vec<_>>()
        };
        let cached_ms = cached_started.elapsed().as_secs_f64() * 1000.0;
        let insert_started = Instant::now();
        let mut cache_hits = 0usize;
        let mut cache_states = 0usize;
        let mut cache_transitions = 0usize;
        for (local_terminal, transported) in cached_results {
            if let Some(transported) = transported {
                cache_hits += 1;
                cache_states += transported.states.len();
                cache_transitions += transported
                    .states
                    .iter()
                    .map(|state| state.transitions.len())
                    .sum::<usize>();
                let global_terminal = terminal_offset + local_terminal as u32;
                result.insert(global_terminal, transported);
            } else {
                missing[local_terminal] = true;
            }
        }
        let insert_ms = insert_started.elapsed().as_secs_f64() * 1000.0;
        if !missing.iter().any(|&value| value) {
            if profile_transport {
                eprintln!(
                    "[glrmask/profile][composition_template_transport] component={} selected={} cache_hits={} cache_misses=0 cache_states={} cache_transitions={} cached_ms={cached_ms:.3} insert_ms={insert_ms:.3} fallback_ms=0.000 total_ms={:.3}",
                    component_index,
                    selected.iter().filter(|&&value| value).count(),
                    cache_hits,
                    cache_states,
                    cache_transitions,
                    component_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            continue;
        }
        let fallback_started = Instant::now();
        let Some(augmented_start) = component.table.rules.first().map(|rule| rule.lhs) else {
            continue;
        };
        let analyzed = AnalyzedGrammar::from_composed_rules(
            component.table.rules.clone(),
            component.table.num_terminals,
            component.terminal_display_names().to_vec(),
            component.table.nonterminal_display_names.clone(),
            augmented_start,
        );
        let characterizations = characterize_selected_terminals(&component.table, &analyzed, &missing);
        let templates = Templates::from_characterizations(&characterizations);
        for (local_terminal, old_template) in templates.by_terminal {
            let global_terminal = terminal_offset + local_terminal;
            let Some(transported) = transport_composition_template_dfa(
                old_template,
                &composed_table.state_relations[component_index],
            ) else {
                continue;
            };
            result.insert(global_terminal, transported);
        }
        if profile_transport {
            eprintln!(
                "[glrmask/profile][composition_template_transport] component={} selected={} cache_hits={} cache_misses={} cache_states={} cache_transitions={} cached_ms={cached_ms:.3} insert_ms={insert_ms:.3} fallback_ms={:.3} total_ms={:.3}",
                component_index,
                selected.iter().filter(|&&value| value).count(),
                cache_hits,
                missing.iter().filter(|&&value| value).count(),
                cache_states,
                cache_transitions,
                fallback_started.elapsed().as_secs_f64() * 1000.0,
                component_started.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }
    result
}



#[derive(Debug, Clone)]
struct ConcreteBoundaryDeltaEntry {
    old_terminal: u32,
    old_template: UnweightedDfa,
    delta_terminal: u32,
    delta_template: UnweightedDfa,
}

#[derive(Debug, Clone)]
struct ConcreteBoundaryDeltaPlan {
    original_num_terminals: u32,
    synthetic_num_terminals: u32,
    by_global_terminal: BTreeMap<u32, ConcreteBoundaryDeltaEntry>,
    /// Active terminals whose transported standalone template and composed
    /// template were compared exactly. An empty standalone language is represented
    /// by an ordinary empty DFA; a missing transported template is unknown and
    /// therefore conservative, never interpreted as the empty language.
    compared_terminals: BTreeSet<u32>,
    /// Active terminals for which we could not prove `Old ⊆ New` after
    /// transporting the cached component template into composed parser
    /// coordinates. Any local path touching one of these stays on the full
    /// composed-template lane.
    unsafe_terminals: BTreeSet<u32>,
}

fn prepare_concrete_boundary_delta_plan(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
    composed_templates: &Templates,
    original_num_terminals: u32,
) -> ConcreteBoundaryDeltaPlan {
    let old_templates = rebuild_transported_component_templates(
        composed_table,
        components,
        active_terminals,
    );
    let mut by_global_terminal = BTreeMap::new();
    let mut compared_terminals = BTreeSet::new();
    let mut unsafe_terminals = BTreeSet::new();
    let mut next_terminal = original_num_terminals;

    for (terminal, &active) in active_terminals.iter().enumerate() {
        if !active {
            continue;
        }
        let terminal = terminal as u32;
        let Some(new) = composed_templates.by_terminal.get(&terminal) else {
            unsafe_terminals.insert(terminal);
            continue;
        };
        let Some(old) = old_templates.get(&terminal) else {
            // Characterization produces an explicit empty DFA when the standalone
            // terminal has no parser action. Missing here therefore means template
            // transport failed (for example because the state relation is not
            // functionally representable), not Old=∅. Keep the full conservative lane.
            unsafe_terminals.insert(terminal);
            continue;
        };
        compared_terminals.insert(terminal);

        let removed = unweighted_dfa_difference(old, new);
        if !unweighted_dfa_language_is_empty(&removed) {
            unsafe_terminals.insert(terminal);
            continue;
        }
        let delta = trim_unweighted_dfa_productive(unweighted_dfa_difference(new, old));
        if unweighted_dfa_language_is_empty(&delta) {
            continue;
        }
        let old_terminal = next_terminal;
        let delta_terminal = next_terminal + 1;
        next_terminal += 2;
        by_global_terminal.insert(
            terminal,
            ConcreteBoundaryDeltaEntry {
                old_terminal,
                old_template: old.clone(),
                delta_terminal,
                delta_template: delta,
            },
        );
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_delta_plan] deltas={} unsafe={} unsafe_ids={:?}",
            by_global_terminal.len(),
            unsafe_terminals.len(),
            unsafe_terminals.iter().take(64).copied().collect::<Vec<_>>(),
        );
    }

    ConcreteBoundaryDeltaPlan {
        original_num_terminals,
        synthetic_num_terminals: next_terminal,
        by_global_terminal,
        compared_terminals,
        unsafe_terminals,
    }
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
    let characterize_started_at = Instant::now();
    let characterizations = characterize_selected_terminals(table, analyzed, selected);
    let characterize_ms = characterize_started_at.elapsed().as_secs_f64() * 1000.0;
    let templates_started_at = Instant::now();
    let templates = Templates::from_characterizations(&characterizations);
    let from_characterizations_ms = templates_started_at.elapsed().as_secs_f64() * 1000.0;
    let commit_started_at = Instant::now();
    let mut template_dfas_by_terminal = vec![None; analyzed.num_terminals as usize];
    if !defer_boundary_commit_templates() {
        let split = |(&terminal, dfa): (&u32, &UnweightedDfa)| {
                let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
                try_split_commit_template_dfas(&commit_dfa)
                    .map(|split| (terminal, Arc::new(split)))
        };
        let split_templates = if macro_parallelism_disabled() {
            let mut timings = Vec::with_capacity(templates.by_terminal.len());
            let result = templates
                .by_terminal
                .iter()
                .filter_map(|item| {
                    let started = Instant::now();
                    let result = split(item);
                    timings.push(started.elapsed().as_secs_f64() * 1000.0);
                    result
                })
                .collect::<Vec<_>>();
            report_macro_item_timings("compose_template_commit_splits", &timings);
            result
        } else {
            templates
                .by_terminal
                .par_iter()
                .filter_map(split)
                .collect::<Vec<_>>()
        };
        for (terminal, split) in split_templates {
            if let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize) {
                *slot = Some(split);
            }
        }
    }
    let commit_ms = commit_started_at.elapsed().as_secs_f64() * 1000.0;
    let total_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition_templates] selected={} characterized={} characterize_ms={characterize_ms:.3} from_characterizations_ms={from_characterizations_ms:.3} commit_ms={commit_ms:.3} total_ms={total_ms:.3}",
            selected.iter().filter(|&&value| value).count(),
            characterizations.len(),
        );
    }
    (templates, template_dfas_by_terminal, total_ms)
}

/// Reuse cached component parser-template characterizations and apply only the
/// stack effects introduced by the current subgrammar splice.
///
/// This fast path is deliberately conservative.  It is enabled only when the
/// parent LR-state coordinate is unchanged, every selected parent terminal has
/// a cached standalone characterization, and composition does not rewrite an
/// existing parent action/forwarded-shift cell.  New actions in appended child
/// states and new boundary-nonterminal goto predecessors are additive and can
/// therefore be characterized independently, unioned into the cached parent
/// characterization, and recompiled only for genuinely changed terminals.
struct CachedCompositionTemplatePlan {
    reused_dfas: BTreeMap<u32, UnweightedDfa>,
    characterization_deltas: BTreeMap<
        u32,
        crate::compiler::stages::templates::characterize::TerminalCharacterization,
    >,
    fresh_characterizations: BTreeMap<
        u32,
        crate::compiler::stages::templates::characterize::TerminalCharacterization,
    >,
    num_terminals: usize,
}

struct EagerChangedParentTemplates {
    templates: Templates,
    changed_parent: Vec<bool>,
    delta_templates: BTreeMap<u32, (UnweightedDfa, UnweightedDfa)>,
    unsafe_parent: BTreeSet<u32>,
    build_ms: f64,
}

struct EagerConcreteDeltaPrecompute {
    delta_templates: BTreeMap<u32, (UnweightedDfa, UnweightedDfa)>,
    unsafe_terminals: BTreeSet<u32>,
}

impl CachedCompositionTemplatePlan {
    fn materialize(self, parent: &Constraint) -> Templates {
        let total_started_at = Instant::now();
        let reused_started_at = Instant::now();
        let mut templates = Templates::from_terminal_dfas(self.reused_dfas);
        let reused_ms = reused_started_at.elapsed().as_secs_f64() * 1000.0;
        let patch_assembly_started_at = Instant::now();
        let mut patched_characterizations = BTreeMap::new();
        for (terminal, delta) in self.characterization_deltas {
            let mut patched = parent
                .composition_parser_characterizations_by_terminal
                .get(terminal as usize)
                .and_then(Option::as_ref)
                .expect("cached composition plan requires its parent characterization")
                .clone();
            patched.escapes.extend(delta.escapes);
            patched.reduces.extend(delta.reduces);
            patched.nt_escapes.extend(delta.nt_escapes);
            patched.nt_rereduces.extend(delta.nt_rereduces);
            patched.all_nts.extend(delta.all_nts);
            patched.escapes.sort();
            patched.escapes.dedup();
            patched.reduces.sort();
            patched.reduces.dedup();
            patched.nt_escapes.sort();
            patched.nt_escapes.dedup();
            patched.nt_rereduces.sort();
            patched.nt_rereduces.dedup();
            patched_characterizations.insert(terminal, patched);
        }
        let patch_assembly_ms = patch_assembly_started_at.elapsed().as_secs_f64() * 1000.0;
        let patch_compile_started_at = Instant::now();
        let patched = Templates::from_characterizations(&patched_characterizations);
        let patch_compile_ms = patch_compile_started_at.elapsed().as_secs_f64() * 1000.0;
        templates.by_terminal.extend(patched.by_terminal);
        templates.by_terminal_nwa.extend(patched.by_terminal_nwa);
        let fresh_started_at = Instant::now();
        let fresh = Templates::from_characterizations(&self.fresh_characterizations);
        let fresh_ms = fresh_started_at.elapsed().as_secs_f64() * 1000.0;
        templates.by_terminal.extend(fresh.by_terminal);
        templates.by_terminal_nwa.extend(fresh.by_terminal_nwa);
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_cached_template_materialize] reused_ms={reused_ms:.3} patch_assembly_ms={patch_assembly_ms:.3} patch_compile_ms={patch_compile_ms:.3} fresh_ms={fresh_ms:.3} total_ms={:.3}",
                total_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        templates
    }

    fn raw_template_cache(templates: &Templates, num_terminals: usize) -> Vec<Option<UnweightedDfa>> {
        let mut raw = vec![None; num_terminals];
        for (&terminal, dfa) in &templates.by_terminal {
            if let Some(slot) = raw.get_mut(terminal as usize) {
                *slot = Some(dfa.clone());
            }
        }
        raw
    }
}




fn try_prepare_cached_composition_template_plan(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    analyzed: &AnalyzedGrammar,
    selected: &[bool],
    scoped_ignore_terminals: &BitSet,
) -> Option<CachedCompositionTemplatePlan> {
    try_prepare_cached_composition_template_plan_for_terminal_count(
        composed_table,
        components,
        analyzed.num_terminals,
        Some(analyzed),
        selected,
        scoped_ignore_terminals,
    )
}

fn try_prepare_cached_composition_template_plan_for_terminal_count(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    num_terminals: u32,
    analyzed_for_fresh: Option<&AnalyzedGrammar>,
    selected: &[bool],
    scoped_ignore_terminals: &BitSet,
) -> Option<CachedCompositionTemplatePlan> {
    let total_started_at = Instant::now();
    if crate::env::var_os("GLRMASK_DISABLE_CACHED_BOUNDARY_TEMPLATES").is_some() {
        return None;
    }
    let parent = *components.first()?;
    let parent_relation = composed_table.state_relations.first()?;
    if parent_relation
        .iter()
        .enumerate()
        .any(|(local, targets)| targets.as_slice() != [local as u32])
    {
        return None;
    }
    if selected.len() != num_terminals as usize {
        return None;
    }
    let parent_terminal_end = composed_table
        .terminal_offsets
        .get(1)
        .copied()
        .unwrap_or(num_terminals)
        .min(num_terminals);
    if parent_terminal_end > parent.table.num_terminals {
        return None;
    }

    let action_seed_started_at = Instant::now();
    let mut action_seeds = vec![Vec::<u32>::new(); num_terminals as usize];
    for terminal in 0..parent_terminal_end {
        if !selected[terminal as usize] {
            continue;
        }
        parent
            .composition_parser_characterizations_by_terminal
            .get(terminal as usize)
            .and_then(Option::as_ref)?;
        for state in 0..parent.table.num_states {
            if composed_table.table.action(state, terminal) != parent.table.action(state, terminal)
                || composed_table.table.forwarded_shifts.contains(&(state, terminal))
                    != parent.table.forwarded_shifts.contains(&(state, terminal))
            {
                return None;
            }
        }
        for state in parent.table.num_states..composed_table.table.num_states {
            let action = composed_table.table.action(state, terminal);
            let forwarded = composed_table.table.forwarded_shifts.contains(&(state, terminal));
            if forwarded && action.is_none() {
                return None;
            }
            if action.is_some() {
                action_seeds[terminal as usize].push(state);
            }
        }
    }
    let action_seed_ms = action_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let action_characterize_started_at = Instant::now();
    let action_deltas = try_characterize_terminal_action_state_seeds_for_terminal_count(
        &composed_table.table,
        num_terminals,
        &action_seeds,
    )
    .ok()?;
    let action_characterize_ms =
        action_characterize_started_at.elapsed().as_secs_f64() * 1000.0;

    let nt_seed_started_at = Instant::now();
    let mut nt_predecessor_seeds =
        vec![Vec::<(u32, u32, u32, bool)>::new(); num_terminals as usize];
    for (revealed_state, row) in composed_table.table.goto.iter().enumerate() {
        for &boundary_nonterminal in &composed_table.boundary_nonterminals {
            let Some(&(top_state, goto_replace)) = row.get(&boundary_nonterminal) else {
                continue;
            };
            for terminal in 0..parent_terminal_end {
                if selected[terminal as usize]
                    && composed_table.table.action(top_state, terminal).is_some()
                {
                    nt_predecessor_seeds[terminal as usize].push((
                        top_state,
                        revealed_state as u32,
                        boundary_nonterminal,
                        goto_replace,
                    ));
                }
            }
        }
    }
    for seeds in &mut nt_predecessor_seeds {
        seeds.sort_unstable();
        seeds.dedup();
    }
    let nt_seed_ms = nt_seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let nt_characterize_started_at = Instant::now();
    let nt_deltas = try_characterize_terminal_nt_predecessor_seeds_for_terminal_count(
        &composed_table.table,
        num_terminals,
        &nt_predecessor_seeds,
    )
    .ok()?;
    let nt_characterize_ms =
        nt_characterize_started_at.elapsed().as_secs_f64() * 1000.0;

    let patch_started_at = Instant::now();
    let mut characterization_deltas = BTreeMap::new();
    let mut changed_parent = vec![false; selected.len()];
    for terminal in 0..parent_terminal_end {
        if !selected[terminal as usize] {
            continue;
        }
        let action_delta = action_deltas.get(&terminal);
        let nt_delta = nt_deltas.get(&terminal);
        let has_delta = [action_delta, nt_delta].into_iter().flatten().any(|delta| {
            !delta.escapes.is_empty()
                || !delta.reduces.is_empty()
                || !delta.nt_escapes.is_empty()
                || !delta.nt_rereduces.is_empty()
        });
        if !has_delta {
            continue;
        }
        let mut delta = crate::compiler::stages::templates::characterize::TerminalCharacterization {
            escapes: Vec::new(),
            reduces: Vec::new(),
            nt_escapes: Vec::new(),
            nt_rereduces: Vec::new(),
            all_nts: BTreeSet::new(),
        };
        for part in [action_delta, nt_delta].into_iter().flatten() {
            // These seeded characterizations are additive by construction.
            // Keep only the tiny delta here; cloning/merging the cached parent
            // characterization belongs to parser-template materialization.
            delta.escapes.extend(part.escapes.iter().cloned());
            delta.reduces.extend(part.reduces.iter().cloned());
            delta.nt_escapes.extend(part.nt_escapes.iter().cloned());
            delta
                .nt_rereduces
                .extend(part.nt_rereduces.iter().cloned());
            delta.all_nts.extend(part.all_nts.iter().copied());
        }
        delta.escapes.sort();
        delta.escapes.dedup();
        delta.reduces.sort();
        delta.reduces.dedup();
        delta.nt_escapes.sort();
        delta.nt_escapes.dedup();
        delta.nt_rereduces.sort();
        delta.nt_rereduces.dedup();
        changed_parent[terminal as usize] = true;
        characterization_deltas.insert(terminal, delta);
    }
    let patch_ms = patch_started_at.elapsed().as_secs_f64() * 1000.0;

    let mut reuse_selected = selected.to_vec();
    for (terminal, changed) in changed_parent.iter().copied().enumerate() {
        if changed {
            reuse_selected[terminal] = false;
        }
    }
    let mut fresh_selected = vec![false; selected.len()];
    for terminal in scoped_ignore_terminals.iter() {
        if terminal < selected.len() && selected[terminal] {
            reuse_selected[terminal] = false;
            fresh_selected[terminal] = true;
        }
    }
    for &terminal in &composed_table.control_terminals {
        if let Some(&true) = selected.get(terminal as usize) {
            reuse_selected[terminal as usize] = false;
            fresh_selected[terminal as usize] = true;
        }
    }

    let transport_started_at = Instant::now();
    let reused_dfas = rebuild_transported_component_templates(
        composed_table,
        components,
        &reuse_selected,
    );
    let transport_ms = transport_started_at.elapsed().as_secs_f64() * 1000.0;
    for (terminal, &active) in selected.iter().enumerate() {
        if active
            && !changed_parent[terminal]
            && !reused_dfas.contains_key(&(terminal as u32))
        {
            fresh_selected[terminal] = true;
        }
    }
    let fresh_started_at = Instant::now();
    let fresh_characterizations = if fresh_selected.iter().any(|&fresh| fresh) {
        characterize_selected_terminals(
            &composed_table.table,
            analyzed_for_fresh?,
            &fresh_selected,
        )
    } else {
        BTreeMap::new()
    };
    let fresh_ms = fresh_started_at.elapsed().as_secs_f64() * 1000.0;
    if selected.iter().enumerate().any(|(terminal, &active)| {
        active
            && !reused_dfas.contains_key(&(terminal as u32))
            && !characterization_deltas.contains_key(&(terminal as u32))
            && !fresh_characterizations.contains_key(&(terminal as u32))
    }) {
        return None;
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_cached_template_plan] selected={} action_seed_ms={action_seed_ms:.3} action_characterize_ms={action_characterize_ms:.3} nt_seed_ms={nt_seed_ms:.3} nt_characterize_ms={nt_characterize_ms:.3} patch_ms={patch_ms:.3} transport_ms={transport_ms:.3} fresh_ms={fresh_ms:.3} reused={} patched={} fresh={} total_ms={:.3}",
            selected.iter().filter(|&&value| value).count(),
            reused_dfas.len(),
            characterization_deltas.len(),
            fresh_characterizations.len(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(CachedCompositionTemplatePlan {
        reused_dfas,
        characterization_deltas,
        fresh_characterizations,
        num_terminals: num_terminals as usize,
    })
}


fn try_build_cached_composition_templates(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    analyzed: &AnalyzedGrammar,
    selected: &[bool],
    scoped_ignore_terminals: &BitSet,
) -> Option<(
    Templates,
    Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    f64,
)> {
    let started_at = Instant::now();
    let plan = try_prepare_cached_composition_template_plan(
        composed_table,
        components,
        analyzed,
        selected,
        scoped_ignore_terminals,
    )?;
    let changed_parent = plan.characterization_deltas.len();
    let fresh = plan.fresh_characterizations.len();
    let reused = plan.reused_dfas.len();
    debug_assert_eq!(plan.num_terminals, analyzed.num_terminals as usize);

    let templates = plan.materialize(components[0]);
    let template_dfas_by_terminal = if defer_boundary_commit_templates() {
        vec![None; analyzed.num_terminals as usize]
    } else {
        let raw_templates = CachedCompositionTemplatePlan::raw_template_cache(
            &templates,
            analyzed.num_terminals as usize,
        );
        build_commit_templates_from_raw_templates(&raw_templates)
    };

    if crate::env::var_os("GLRMASK_VALIDATE_CACHED_BOUNDARY_TEMPLATES").is_some() {
        let (reference, _, _) =
            build_composition_templates(&composed_table.table, analyzed, selected);
        for (terminal, &active) in selected.iter().enumerate() {
            if !active {
                continue;
            }
            let terminal = terminal as u32;
            let candidate = templates
                .by_terminal
                .get(&terminal)
                .expect("cached-template fast path must cover selected terminal");
            let expected = reference
                .by_terminal
                .get(&terminal)
                .expect("reference template build must cover selected terminal");
            let candidate_only = unweighted_dfa_difference(candidate, expected);
            let expected_only = unweighted_dfa_difference(expected, candidate);
            assert!(
                unweighted_dfa_language_is_empty(&candidate_only)
                    && unweighted_dfa_language_is_empty(&expected_only),
                "cached composition template differs from full characterization for terminal {terminal}: candidate_only={:?} expected_only={:?}",
                unweighted_dfa_shortest_word(&candidate_only),
                unweighted_dfa_shortest_word(&expected_only),
            );
        }
    }
    let total_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_cached_composition_templates] selected={} reused={} changed_parent={} fresh={} raw_templates={} commit_templates={} total_ms={total_ms:.3}",
            selected.iter().filter(|&&value| value).count(),
            reused,
            changed_parent,
            fresh,
            templates.by_terminal.len(),
            template_dfas_by_terminal.iter().flatten().count(),
        );
    }
    Some((templates, template_dfas_by_terminal, total_ms))
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






fn build_composed_constraint_unfinalized(
    composed_table: ComposedTable,
    tokenizer: impl Into<Arc<Tokenizer>>,
    tokenizer_state_offsets: Vec<u32>,
    parser_dwa: DWA,
    parser_state_domain_labels: Vec<i32>,
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
    defer_dynamic_mask_vocab: bool,
    vocab: &Vocab,
    retained_lr_coordinator: bool,
) -> ConstraintComposition {
    let tokenizer = tokenizer.into();
    let total_started_at = Instant::now();
    let phase_started_at = Instant::now();
    let terminal_offsets = composed_table.terminal_offsets.clone();
    let parser_state_relations = composed_table.state_relations.clone();
    let relation_clone_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids,
    } = internal_ids;
    debug_assert!(deferred_vocab_singleton_original_ids.is_none());
    let ManyToOneIdMap {
        original_to_internal: state_to_internal_tsid,
        internal_to_originals: internal_tsid_to_states,
        representative_original_ids: _,
    } = tokenizer_states;
    debug_assert!(state_to_internal_tsid.iter().all(|&tsid| tsid != u32::MAX));
    let ManyToOneIdMap {
        original_to_internal: original_token_to_internal,
        internal_to_originals: internal_token_to_tokens,
        representative_original_ids: _,
    } = vocab_tokens;
    let tokenizer_has_epsilon_transitions = tokenizer.has_epsilon_transitions();
    let mut table = composed_table.table;
    table.set_embedded_end_token_ids(&embedded_end_token_ids);
    let num_terminals = table.num_terminals as usize;
    let phase_started_at = Instant::now();
    let dynamic_mask_vocab = if defer_dynamic_mask_vocab {
        // The explicit segmented result is a static coordinator: local masks
        // are authoritative in retained A components and cross-component masks
        // are authoritative in B. Its own direct-dynamic vocabulary is not on
        // the normal mask path. Keep it lazy just like a loaded static
        // constraint; if a future fallback ever requests it,
        // `dynamic_mask_vocab_for_runtime()` can reconstruct it from
        // `token_bytes` on demand.
        crate::runtime::DynamicMaskVocab::default()
    } else {
        runtime_dynamic_vocab_for_vocab(vocab)
    };
    let dynamic_vocab_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
    let phase_started_at = Instant::now();
    let token_bytes = crate::runtime::TokenByteDomain::from_entries(vocab.entries_arc());
    let token_bytes_ms = phase_started_at.elapsed().as_secs_f64() * 1000.0;
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
        parser_dwa: crate::runtime::ParserDwaStorage::Owned(parser_dwa),
        non_dwa_weights: crate::runtime::NonDwaWeightStorage::Materialized(
            crate::runtime::NonDwaWeights {
                // Constraint composition is not allowed to rely on the legacy dynamic
                // possible-matches fallback. This is the exact transported and
                // reconciled table from every compiled component.
                // DO NOT REMOVE OR WEAKEN THIS COMMENT.
                possible_matches,
                ..Default::default()
            },
        ),
        direct_regular_wide_frontier_acceptance: Vec::new(),
        direct_regular_dynamic_hot_frontiers: Vec::new(),
        direct_regular_parser_state_acceptance: Vec::new(),
        direct_regular_automaton: None,
        table: if retained_lr_coordinator {
            // Only a nonexecutable coordinate shell. Actual LR execution stays
            // in the retained Dynamic leaves; native constructors never use it.
            assert!(table.num_states==0 && table.action.is_empty() && table.goto.is_empty());
            crate::runtime::parser_backend::ParserTableStorage::explicit(table)
        } else {table.into()},
        terminal_display_names,
        tokenizer,
        boundary_completion_index: None,
        tokenizer_has_epsilon_transitions,
        ignore_terminal,
        special_token_terminals,
        dynamic_mask_vocab,
        lazy_dynamic_mask_vocab: OnceLock::new(),
        empty_byte_token_ids: crate::compiler::compile::vocab_empty_byte_token_ids(vocab),
        possible_matches_complete: true,
        tokenizer_tsids: crate::runtime::TokenizerTsidMap::partition(
            state_to_internal_tsid, internal_tsid_to_states,
        ),
        composition_reset_tokens_by_terminal: Vec::new(),
        composition_parser_templates_by_terminal: Vec::new(),
        composition_parser_characterizations_by_terminal: Vec::new(),
        composition_grammar_summary: None,
        terminal_live_states,
        lr_terminal_templates: template_dfas_by_terminal,
        lr_fast_terminal_templates: Vec::new(),
            template_parser: None,
        token_classes: crate::runtime::TokenClasses::mapped(original_token_to_internal, internal_token_to_tokens),
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
        parser_state_domain_labels,
        ignore_expr,
        serialized_artifact_cache: None,
        deferred_terminal_exprs_blob: None,
        deferred_terminal_exprs: Default::default(),
        deferred_composition_metadata_blob: None,
        composition_link_metadata_materialized: true,
        deferred_table_rules_blob: None,
        deferred_table_rules: Default::default(),
    });
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_unfinalized_build] relation_clone_ms={relation_clone_ms:.3} dynamic_vocab_ms={dynamic_vocab_ms:.3} token_bytes_ms={token_bytes_ms:.3} total_ms={:.3}",
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
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
    let composed_table = ComposedTable {
        table: shell_table,
        terminal_offsets: terminal_offsets.clone(),
        placeholder_terminals: children
            .iter()
            .flat_map(CompiledSubgrammarInput::placeholder_terminals)
            .collect(),
        placeholder_component_indices: children
            .iter()
            .enumerate()
            .flat_map(|(index, child)| {
                std::iter::repeat_n(index + 1, 1 + child.additional_placeholder_terminals.len())
            })
            .collect(),
        state_relations: vec![Vec::new(); component_count],
        boundary_nonterminals: BTreeSet::new(),
        control_terminals: BTreeSet::new(),
        appended_parent_action_terminals: BTreeSet::new(),
    };
    let shell_ms = shell_started_at.elapsed().as_secs_f64() * 1000.0;

    let coordinator_started_at = Instant::now();
    let root_tokenizer = parent.tokenizer.clone();
    let root_fast_transitions = parent.tokenizer_fast_transitions.clone();
    let id_map = InternalIdMap {
        // DynamicDirect never consumes the coordinator TSID quotient. Keep a
        // one-class placeholder without building the global state/token partition.
        tokenizer_states: ManyToOneIdMap {
            original_to_internal: vec![0],
            internal_to_originals: vec![vec![0]],
            representative_original_ids: vec![0],
        },
        vocab_tokens: ManyToOneIdMap::empty(),
        deferred_vocab_singleton_original_ids: None,
    };
    let mut result = build_composed_constraint_unfinalized(
        composed_table,
        root_tokenizer,
        tokenizer_state_offsets.clone(),
        DWA::new(1, 0),
        Vec::new(),
        Default::default(),
        id_map,
        vec![None; next_terminal as usize],
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        None,
        None,
        Vec::new(),
        root_fast_transitions,
        true,
        vocab,
        true,
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

    // Empty bytes are an explicit "provider-native only" marker. Dynamic
    // recomposition consumes the retained component tree directly. Static or
    // legacy compiler views may reconstruct from the provider in a later path.
    let _ = overlay
        .recursive_compiler_table
        .set(Arc::<[u8]>::from(Vec::<u8>::new().into_boxed_slice()));
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
        let grammar = constraint.template_parser.as_ref().unwrap().link_grammar.as_ref().unwrap();
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
    fn stored_parser_domain_label_expands_before_nested_transport() {
        let relation = vec![vec![10], vec![11, 12], vec![20], vec![30]];
        let domain = 4i32;
        let domain_labels = vec![NO_PARSER_DOMAIN_LABEL, domain, domain, NO_PARSER_DOMAIN_LABEL];

        assert_eq!(
            mapped_labels(domain, &relation, &domain_labels).unwrap(),
            vec![
                encode_positive_label(11),
                encode_positive_label(12),
                encode_positive_label(20),
            ],
            "a stored synthetic parser-domain label must expand to every concrete local state in that domain before the outer state relation is applied",
        );
    }

    #[test]
    fn symbolic_child_default_domain_expands_exactly_to_materialized_transport() {
        let relation = vec![vec![10], vec![11], vec![12, 13], vec![20]];
        let mut source = DWAState::default();
        source.transitions.insert(
            encode_positive_label(0),
            (1, Weight::all()),
        );
        // This explicit one-to-many local label must override the synthetic
        // DEFAULT on both of its mapped concrete parser states.
        source.transitions.insert(
            encode_positive_label(2),
            (1, Weight::all()),
        );
        source
            .transitions
            .insert(DEFAULT_LABEL, (2, Weight::all()));

        let mut baseline = NWA::new(1, 0);
        for _ in 0..3 {
            baseline.add_state();
        }
        add_component_parser_state_transitions(
            &mut baseline,
            0,
            &source,
            &relation,
            &[],
            relation.len() as u32,
            None,
        )
        .unwrap();

        let mut domain_states = BitSet::new(32);
        for state in [11usize, 12, 13] {
            domain_states.set(state);
        }
        let domain = ParserDefaultDomain {
            label: 30,
            base_has_states: true,
            nested_labels: BTreeMap::new(),
            states: domain_states,
            predicted_saved_edges: 3,
        };
        let mut compressed = NWA::new(1, 0);
        for _ in 0..3 {
            compressed.add_state();
        }
        add_component_parser_state_transitions(
            &mut compressed,
            0,
            &source,
            &relation,
            &[],
            relation.len() as u32,
            Some(&domain),
        )
        .unwrap();

        // Expand the runtime lookup semantics `concrete -> domain` back onto
        // concrete labels. The resulting transition relation must equal the old
        // fully-materialized transport exactly.
        let domain_targets = compressed.states()[0]
            .transitions
            .get(&domain.label)
            .cloned()
            .expect("compressed row must carry one domain fallback");
        let mut expanded = compressed.states()[0].transitions.clone();
        expanded.remove(&domain.label);
        for parser_state in domain.states.iter_ones() {
            expanded
                .entry(encode_positive_label(parser_state as u32))
                .or_insert_with(|| domain_targets.clone());
        }
        assert_eq!(expanded, baseline.states()[0].transitions);
        assert_eq!(
            expanded[&encode_positive_label(12)][0].0,
            1,
            "explicit local transitions must retain precedence over domain DEFAULT",
        );
        assert_eq!(expanded[&encode_positive_label(11)][0].0, 2);
        assert_eq!(expanded[&encode_positive_label(20)][0].0, 2);
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

    #[test]
    fn overlap_local_union_matches_generic_reference_on_generated_acyclic_inputs() {
        fn next_u32(state: &mut u64) -> u32 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 32) as u32
        }

        fn generated_dwa(random: &mut u64, salt: u32) -> DWA {
            let mut dwa = DWA::new(1, 63);
            for _ in 0..3 {
                dwa.add_state();
            }
            for state in 0..4u32 {
                if !next_u32(random).is_multiple_of(3) {
                    let start = (next_u32(random) + salt) % 48;
                    let end = (start + 3 + next_u32(random) % 12).min(63);
                    dwa.set_final_weight(
                        state,
                        Weight::from_token_set_for_tsid(
                            0,
                            RangeSetBlaze::from_iter(std::iter::once(start..=end)),
                        ),
                    );
                }
                if state == 3 {
                    continue;
                }
                let mut labels = BTreeSet::<i32>::new();
                for concrete in 0..6u32 {
                    if !next_u32(random).is_multiple_of(3) {
                        labels.insert(encode_positive_label(concrete));
                    }
                }
                if next_u32(random).is_multiple_of(2) {
                    labels.insert(encode_negative_label(next_u32(random) % 6));
                }
                if next_u32(random).is_multiple_of(2) {
                    labels.insert(20 + (next_u32(random) % 3) as i32);
                }
                if next_u32(random).is_multiple_of(3) {
                    labels.insert(DEFAULT_LABEL);
                }
                for label in labels {
                    let target = state + 1 + next_u32(random) % (3 - state);
                    let start = (next_u32(random) + salt * 3) % 48;
                    let end = (start + 2 + next_u32(random) % 14).min(63);
                    dwa.add_transition(
                        state,
                        label,
                        target,
                        Weight::from_token_set_for_tsid(
                            0,
                            RangeSetBlaze::from_iter(std::iter::once(start..=end)),
                        ),
                    );
                }
            }
            dwa
        }

        let mut random = 0xd15c_a11c_5eed_2026u64;
        for arity in 2..=3usize {
            for case in 0..64u32 {
                let dwas = (0..arity)
                    .map(|index| generated_dwa(&mut random, case * 4 + index as u32 + 1))
                    .collect::<Vec<_>>();
                let direct_inputs = dwas
                    .iter()
                    .map(parser_nwa_preserve_defaults)
                    .collect::<Vec<_>>();
                let (direct, _) =
                    determinize_epsilon_free_component_union(direct_inputs, Some(6))
                        .expect("generated inputs are epsilon-free and acyclic");

                let extra_positive_labels = dwas
                    .iter()
                    .flat_map(|dwa| dwa.states())
                    .flat_map(|state| state.transitions.keys().copied())
                    .filter(|&label| label >= 6 && label != DEFAULT_LABEL)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                let mut union = NWA::new(1, 63);
                let mut starts = Vec::new();
                for dwa in &dwas {
                    let explicit = explicit_parser_nwa(dwa, 6, &extra_positive_labels);
                    let body = union.append_with_body(&explicit);
                    starts.extend(body.start_states);
                }
                union.set_start_states(starts);
                let generic = determinize(&union).expect("generic reference must determinize");
                let direct_explicit_nwa = explicit_parser_nwa(&direct, 6, &extra_positive_labels);
                let direct_explicit = determinize(&direct_explicit_nwa)
                    .expect("expanded direct result must determinize");
                let difference = find_difference(&direct_explicit, &generic)
                    .expect("generated direct/reference equivalence must be decidable");
                assert_eq!(
                    difference, None,
                    "overlap-local union differs from generic reference for arity {arity} case {case}",
                );

                if arity == 3 {
                    let (first_pair, _) = determinize_epsilon_free_component_union(
                        vec![
                            parser_nwa_preserve_defaults(&dwas[0]),
                            parser_nwa_preserve_defaults(&dwas[1]),
                        ],
                        Some(6),
                    )
                    .expect("first generated pair is epsilon-free and acyclic");
                    let (nested_direct, _) = determinize_epsilon_free_component_union(
                        vec![
                            parser_nwa_preserve_defaults(&first_pair),
                            parser_nwa_preserve_defaults(&dwas[2]),
                        ],
                        Some(6),
                    )
                    .expect("nested generated direct union stays epsilon-free and acyclic");
                    let nested_explicit_nwa =
                        explicit_parser_nwa(&nested_direct, 6, &extra_positive_labels);
                    let nested_explicit = determinize(&nested_explicit_nwa)
                        .expect("expanded nested direct result must determinize");
                    let nested_difference = find_difference(&nested_explicit, &generic)
                        .expect("nested direct/reference equivalence must be decidable");
                    assert_eq!(
                        nested_difference, None,
                        "nested overlap-local union differs from generic reference in case {case}",
                    );
                }
            }
        }
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
        // Synthetic child-domain labels are ordinary nonnegative symbols above
        // the concrete parser-state range. A global DEFAULT from another
        // automaton must still contribute on them.
        explicit.add_transition(
            explicit_start,
            100,
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
        assert!(!evaluate_runtime_label(100).is_empty());
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
        let parent_table = compiler_table_fixture(&parent);
        let child_table = compiler_table_fixture(&child);
        let composed_table = compose_subgrammar_tables(
            &parent_table,
            None,
            &[SubgrammarTableInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                additional_placeholder_terminals: &[],
table: &child_table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.composition_start_nullable().unwrap(),
            }],
        )
        .unwrap();
        let (merged_tokenizer, tokenizer_offsets) =
            crate::automata::lexer::tokenizer::Tokenizer::disjoint_union_with_terminal_offsets(&[
                (&parent.tokenizer, composed_table.terminal_offsets[0]),
                (&child.tokenizer, composed_table.terminal_offsets[1]),
            ]);
        let parser_components = [
            ParserDwaComponent {
                constraint: &parent,
                parser_state_relation: &composed_table.state_relations[0],
                tokenizer_state_offset: tokenizer_offsets[0],
                terminal_offset: 0,
            },
            ParserDwaComponent {
                constraint: &child,
                parser_state_relation: &composed_table.state_relations[1],
                tokenizer_state_offset: tokenizer_offsets[1],
                terminal_offset: 0,
            },
        ];
        let default_domains = build_parser_default_domain_plan(
            &parser_components,
            composed_table.table.num_states,
        );
        let merged = compose_component_parser_dwas_and_possible_matches(
            &parser_components,
            &composed_table.terminal_offsets,
            &default_domains.component_domains,
            merged_tokenizer.num_states() as usize,
            &vocab.entries_map().keys().copied().collect::<Vec<_>>(),
            false,
        )
        .unwrap();
        assert!(merged.0.artifact().0.num_states() > 0);
        assert!(merged.0.artifact().0.num_transitions() > 0);
        assert_eq!(
            merged.0.id_map().tokenizer_states.original_to_internal.len(),
            merged_tokenizer.num_states() as usize,
        );
        assert_eq!(composed_table.state_relations[1][0].len(), 2);
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
