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
use crate::automata::weighted_u32::minimize::{minimize_owned, reverse_hashcons_owned};
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
use crate::compiler::stages::mapped_artifact::{WeightRefs, remap_weights_with_maps};
use crate::compiler::stages::id_map_and_terminal_dwa::classify::vocab_tokens_with_adjacent_pairs;
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalColoring;
use crate::compiler::stages::parser_dwa::{
    LazyBooleanParserDomains, SharedBooleanParserDomains, build_boolean_terminal_bundle_nwa,
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates,
    build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled,
    build_terminal_bundle_preimage_domain_nwa,
    universal_parser_stack_domain_dwa,
    normalize_parser_stack_domain_nwa_preserving_explicit,
    normalize_weighted_parser_stack_nwa,
};
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
    Action, ComposedTable, ControlEliminationReport, SubgrammarTableInput, compose_subgrammar_tables,
    compose_subgrammar_tables_explicit,
};
use crate::grammar::flat::Symbol;
use crate::ds::bitset::BitSet;
use crate::ds::u8set::U8Set;
use crate::ds::weight::{ScopedWeightOpCache, SharedTokenSet, Weight};
use crate::runtime::{Constraint, ConstraintRuntimeBackend, SpecialTokenTerminal};
use crate::Vocab;

mod structural_sharing;
mod runtime_lexer_product;
use runtime_lexer_product::maybe_install_runtime_lexer_product;
use structural_sharing::{
    StructuralSharingReport, composition_terminal_classes, contextually_share_composed_states,
    quotient_composed_table_structurally, structural_nonterminal_classes,
    structural_sharing_enabled,
};

#[inline]
fn compose_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

/// The generic parser-DWA builder currently skips weighted minimization unless
/// explicitly overridden. Boundary overlays are different: leaving their large
/// intermediate DWA unminimized makes the subsequent component union explode.
fn parser_builder_skips_internal_minimization() -> bool {
    std::env::var("GLRMASK_SKIP_PARSER_DWA_MINIMIZE")
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
    std::env::var("GLRMASK_BOUNDARY_PARSER_MINIMIZE_MIN_STATES")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(64)
}

fn eliminate_composed_runtime_controls(
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
    /// Composed execution table used to scope a standalone global-ignore
    /// identity to parser tops where that terminal is actually `Skip`.
    pub(crate) composed_table: Option<&'a crate::compiler::glr::table::GLRTable>,
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
    /// One optional exact wildcard domain per component. Component zero is the
    /// parent and is deliberately never assigned a domain.
    component_domains: Vec<Option<ParserDefaultDomain>>,
    predicted_saved_edges: usize,
}

fn symbolic_child_defaults_env_override() -> Option<bool> {
    std::env::var("GLRMASK_COMPOSE_SYMBOLIC_CHILD_DEFAULTS")
        .ok()
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
}

fn symbolic_child_default_min_saved_edges() -> usize {
    std::env::var("GLRMASK_COMPOSE_SYMBOLIC_CHILD_DEFAULT_MIN_SAVED_EDGES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4_096)
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
    let force_parent_defaults =
        std::env::var_os("GLRMASK_EXPERIMENT_SYMBOLIC_PARENT_DEFAULTS").is_some();
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

    let force = symbolic_child_defaults_env_override();
    let min_saved = symbolic_child_default_min_saved_edges();
    let mut component_predicted = vec![0usize; components.len()];
    for component_index in 0..components.len() {
        let component = components[component_index];
        let domain_total = local_multiplicities[component_index].iter().sum::<usize>();
        if domain_total == 0 {
            continue;
        }
        for state in component.constraint.parser_dwa.states() {
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
    // path unless explicitly forced.
    let child_feature_selected = match force {
        Some(false) => false,
        Some(true) => true,
        None => component_predicted
            .iter()
            .skip(1)
            .any(|&predicted| predicted >= min_saved),
    };
    let symbolic_parent_defaults = force_parent_defaults || child_feature_selected;
    let first_domain_component = usize::from(!symbolic_parent_defaults);
    let required_domain_labels = (first_domain_component..components.len())
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

    let feature_selected = labels_fit && (child_feature_selected || force_parent_defaults);
    if feature_selected {
        // Once one child amortizes the table-sized runtime map, every further
        // exact child domain is a marginal win: one synthetic label replaces a
        // positive number of concrete edges without increasing lookup depth.
        for component_index in first_domain_component..components.len() {
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

#[inline]
fn tokenizer_tsid_relation_is_singleton(constraint: &Constraint) -> bool {
    constraint.state_internal_tsid_offsets.as_slice() == [u32::MAX]
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
    for (component_index, component) in components.iter().enumerate() {
        let constraint = component.constraint;
        if constraint.state_to_internal_tsid.len() != constraint.tokenizer.num_states() as usize {
            return Err("component tokenizer-state map does not cover its runtime tokenizer".into());
        }
        let local_tsid_count = constraint.internal_tsid_to_states.len();
        if local_tsid_count == 0 {
            return Err("component tokenizer-state map contains no internal TSIDs".into());
        }
        let mut local_map = vec![Vec::<u32>::new(); local_tsid_count];

        if tokenizer_tsid_relation_is_singleton(constraint) {
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
        for local_state in 0..constraint.tokenizer.num_states() {
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
            let merged_state = component
                .tokenizer_state_offset
                .checked_add(local_state)
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
            .or_else(|| {
                constraint
                    .parser_state_domain_label(parser_state)
                    .and_then(|domain| constraint.parser_top_accept.get(&domain))
            })
            .or(default_combined)
        {
            parts.push(weight.clone());
        }
        if let Some(weights) = constraint
            .parser_top_accept_parts
            .get(&label)
            .or_else(|| {
                constraint
                    .parser_state_domain_label(parser_state)
                    .and_then(|domain| constraint.parser_top_accept_parts.get(&domain))
            })
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
    if component.parser_state_relation.len() != constraint.table.num_states as usize {
        return Err(format!(
            "parser-state relation has {} rows for a {}-state component table",
            component.parser_state_relation.len(),
            constraint.table.num_states,
        ));
    }

    let prepared_source;
    let source = if std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_HASHCONS_PREP").is_some()
        && constraint.parser_dwa.num_states() >= 10_000
    {
        let started_at = Instant::now();
        let before_states = constraint.parser_dwa.num_states();
        let before_transitions = constraint.parser_dwa.num_transitions();
        prepared_source = reverse_hashcons_owned(constraint.parser_dwa.clone());
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_component_hashcons_prep] before_states={} before_transitions={} after_states={} after_transitions={} ms={:.3}",
                before_states,
                before_transitions,
                prepared_source.num_states(),
                prepared_source.num_transitions(),
                started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        &prepared_source
    } else {
        &constraint.parser_dwa
    };
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
            constraint.table.num_states,
            default_domain,
        )?;
    }

    let top_accept = materialized_top_acceptance(constraint);
    if std::env::var_os("GLRMASK_EXPERIMENT_EXTERNALIZE_COMPONENT_TOP_ACCEPT").is_none()
        && !top_accept.is_empty()
    {
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
    if std::env::var_os("GLRMASK_EXPERIMENT_EXTERNALIZE_COMPONENT_TOP_ACCEPT").is_none()
        && std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_SCOPED_IGNORE_TOP_ACCEPT").is_some()
        && let (Some(table), Some(local_ignore)) =
            (component.composed_table, constraint.ignore_terminal)
        && let Some(ignore_possible) = constraint.possible_matches.get(&local_ignore)
        && let Some(parser_empty) = constraint
            .parser_dwa
            .states()
            .get(constraint.parser_dwa.start_state() as usize)
            .and_then(|state| state.final_weight.as_ref())
    {
        let identity_weight = parser_empty.intersection(ignore_possible);
        let global_ignore = component.terminal_offset + local_ignore;
        let identity_states = table
            .action
            .iter()
            .enumerate()
            .filter_map(|(state, row)| {
                let action = row.get(&global_ignore)?;
                let stack_neutral = matches!(action, Action::Skip | Action::Shift(_, true))
                    || matches!(action, Action::ReplaceShifts(_))
                    || matches!(action, Action::StackShifts(shifts) if shifts.iter().all(|shift| shift.pop == 1 && shift.pushes.len() == 1))
                    || matches!(action, Action::Split { shift: Some((_, true)), reduces, accept: false } if reduces.is_empty());
                stack_neutral.then_some(state as u32)
            })
            .collect::<Vec<_>>();
        if !identity_states.is_empty() && !identity_weight.is_empty() {
            let start = nwa.add_state();
            let final_state = nwa.add_state();
            nwa.set_final_weight(final_state, Weight::all());
            for state in identity_states {
                nwa.add_transition(
                    start,
                    encode_positive_label(state),
                    final_state,
                    identity_weight.clone(),
                );
            }
            nwa.start_states_mut().push(start);
        }
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
    parser_dwa: MappedArtifact<DWA>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    active_terminals: Vec<bool>,
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

fn scan_component_residual_start_groups(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    reset_live_bytes: &[U8Set],
    bytes: &[u8],
    candidate_groups: &[(u32, Vec<u32>)],
) -> FxHashMap<ResidualScanResult, Vec<u32>> {
    let validate =
        std::env::var_os("GLRMASK_VALIDATE_COMPOSE_TSID_REPRESENTATIVE_SCAN").is_some();
    let mut starts_by_scan = FxHashMap::<ResidualScanResult, Vec<u32>>::default();
    let mut by_component = vec![Vec::<(u32, &[u32])>::new(); components.len()];

    for (representative, support_states) in candidate_groups {
        if *representative == 0 {
            let scan = scan_component_residual_starts(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                reset_live_bytes,
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
                if prefix_nullable {
                    if let Some(lhs_precede) = precede.get(rule.lhs as usize) {
                        additions.union_with(lhs_precede);
                    }
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

fn transition_boundary_key(
    key: BoundaryTokenNodeKey,
    terminal: u32,
    next_offset: usize,
    seed_terminals: &[bool],
    globally_erased_terminals: &BitSet,
    interface_pairs: &BTreeSet<(u32, u32)>,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
    follow_transparent_terminals: &BitSet,
) -> Option<BoundaryTokenNodeKey> {
    if globally_erased_terminals.contains(terminal as usize) {
        return Some(BoundaryTokenNodeKey {
            offset: next_offset,
            started: true,
            ..key
        });
    }

    if key.last_terminal != u32::MAX
        && !follow_transparent_terminals.contains(key.last_terminal as usize)
        && !follow_transparent_terminals.contains(terminal as usize)
        && disallowed_follows
            .and_then(|rows| rows.get(&key.last_terminal))
            .is_some_and(|blocked| blocked.contains(terminal as usize))
    {
        return None;
    }
    let interface_witnessed = key.last_terminal != u32::MAX
        && interface_pairs.contains(&(key.last_terminal, terminal));
    Some(BoundaryTokenNodeKey {
        offset: next_offset,
        last_terminal: terminal,
        seeded: key.seeded
            || (key.last_terminal == u32::MAX
                && seed_terminals
                    .get(terminal as usize)
                    .copied()
                    .unwrap_or(false)),
        interface_witnessed: key.interface_witnessed || interface_witnessed,
        erased_seen: key.erased_seen || globally_erased_terminals.contains(terminal as usize),
        started: true,
    })
}

fn build_boundary_token_graph(
    bytes: &[u8],
    arbitrary_scan: &ResidualScanResult,
    reset_scans: &[&ResidualScanResult],
    seed_terminals: &[bool],
    globally_erased_terminals: &BitSet,
    interface_pairs: &BTreeSet<(u32, u32)>,
    initial_interface_witnessed: bool,
    allow_seed_only: bool,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
    follow_transparent_terminals: &BitSet,
) -> Option<(Vec<BoundaryTokenNode>, Vec<bool>, Vec<bool>)> {
    let accept_complete_cross_candidates =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_COMPLETE_PATH_DISCOVERY").is_some();
    let mut nodes = Vec::<BoundaryTokenNode>::new();
    let mut node_ids = FxHashMap::<BoundaryTokenNodeKey, usize>::default();
    let mut queue = std::collections::VecDeque::<usize>::new();
    let start_key = BoundaryTokenNodeKey {
        offset: 0,
        last_terminal: u32::MAX,
        seeded: false,
        interface_witnessed: initial_interface_witnessed,
        erased_seen: false,
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
                globally_erased_terminals,
                interface_pairs,
                disallowed_follows,
                follow_transparent_terminals,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len()
                    && (accept_complete_cross_candidates
                        || target_key.interface_witnessed
                        || (allow_seed_only && target_key.seeded));
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
                globally_erased_terminals,
                interface_pairs,
                disallowed_follows,
                follow_transparent_terminals,
            ) else {
                continue;
            };
            let target = if let Some(&target) = node_ids.get(&target_key) {
                target
            } else {
                let target = nodes.len();
                let is_accepting = target_key.offset == bytes.len()
                    && (accept_complete_cross_candidates
                        || target_key.interface_witnessed
                        || (allow_seed_only && target_key.seeded));
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
    vocab: &Vocab,
    candidate_tokens: &BTreeSet<u32>,
) -> BTreeMap<u32, Vec<(usize, u32, u32)>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    let union_possible_matches =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_UNION_POSSIBLE_MATCHES").is_some();
    let mut by_token = BTreeMap::<u32, Vec<(usize, u32, u32)>>::new();
    for (component_index, constraint) in components.iter().enumerate() {
        debug_assert!(constraint.possible_matches_complete);
        let union;
        let weights: Box<dyn Iterator<Item = &Weight> + '_> = if union_possible_matches {
            union = Weight::union_all(constraint.possible_matches.values());
            Box::new(std::iter::once(&union))
        } else {
            Box::new(constraint.possible_matches.values())
        };
        for weight in weights {
            for (start_tsid, end_tsid, internal_tokens) in weight.range_entries() {
                for internal_token in internal_tokens.iter() {
                    if constraint.internal_token_to_tokens.is_empty() {
                        if candidate_tokens.contains(&internal_token)
                            && vocab
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
                        if candidate_tokens.contains(&original)
                            && vocab
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
    extra_start_states_by_token: &BTreeMap<u32, Vec<u32>>,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
) -> Vec<(u32, Vec<u32>)> {
    // Global state zero is the retained/fresh merged reset dispatcher. It is a
    // semantic state of its own and must not be conflated with an individual
    // component's local start state.
    let mut support_by_representative = FxHashMap::<u32, Vec<u32>>::default();
    support_by_representative.insert(0, vec![0]);
    if let Some(extra_states) = extra_start_states_by_token.get(&token_id) {
        // These states are an exact support subset, not necessarily a whole
        // existing TSID class, so preserve them as singleton representatives.
        // The residual scanner will still merge equal lexical scans afterward.
        for &state in extra_states {
            if state != 0 {
                support_by_representative.entry(state).or_default().push(state);
            }
        }
    }
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
    allow_suffix_seed: bool,
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

    // Seed-only boundary evidence begins at model-token offset zero. When a
    // globally erased terminal exists, keep the older conservative suffix scan
    // because erased trivia may precede the first parser-visible seed inside the
    // same token. Otherwise later seed occurrences require an actual interface
    // witness and are supplied by the adjacent-pair candidate set.
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
        let offset_count = if allow_suffix_seed { bytes.len() } else { 1 };
        'suffixes: for offset in 0..offset_count {
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

    // Legacy/global-erased-trivia fallback. `possible_matches` deliberately
    // includes terminals reached later inside a model token, so it is too broad
    // for the ordinary reset-origin seed rule. Retain it only when erased trivia
    // may legally precede the parser-visible seed.
    if allow_suffix_seed {
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
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_prefilter_sources] allow_suffix_seed={} after_seed_dfas={} after_seed_possible_matches={} seed_terminals={:?}",
            allow_suffix_seed,
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


/// Extra arbitrary lexer starts needed by boundary discovery.
///
/// Static `possible_matches` is intentionally sparse: it indexes delayed-terminal
/// queries used by runtime masking, not every raw lexer residual from which a
/// model token may begin. Boundary composition has a stronger requirement: a
/// token can start in the middle of any parser-visible terminal that participates
/// in a boundary witness. Preserve those residual starts directly from the
/// component `terminal_live_states` inverse index.
///
/// This is deliberately terminal-generic. Scoped IGNORE participates only
/// because it is a visible terminal in `relevant_terminals`; globally erased
/// IGNORE is excluded by the caller and remains lexical epsilon.
fn boundary_visible_residual_starts_by_first_byte(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    relevant_terminals: &BitSet,
) -> Vec<Vec<u32>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());

    // Necessary first-byte filter: if the epsilon closure of a residual raw
    // state has no transition on the model token's first byte, that token
    // cannot begin from that state. The inverse is useful both for selecting
    // candidate vocabulary tokens and for recovering their exact raw starts.
    let mut starts_by_first_byte = (0..256).map(|_| Vec::<u32>::new()).collect::<Vec<_>>();
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        let state_offset = tokenizer_state_offsets[component_index];
        let closures = component.tokenizer.all_singleton_epsilon_closures();
        let mut relevant_states = Vec::<u32>::new();
        for local_terminal in 0..component.tokenizer.num_terminals() {
            let global_terminal = terminal_offset + local_terminal;
            if !relevant_terminals.contains(global_terminal as usize) {
                continue;
            }
            let Some(live_states) = component.terminal_live_states.get(local_terminal as usize) else {
                continue;
            };
            relevant_states.extend(
                live_states
                    .iter()
                    .copied()
                    .filter(|&state| state != component.tokenizer.start_state()),
            );
        }
        relevant_states.sort_unstable();
        relevant_states.dedup();

        for local_state in relevant_states {
            let Some(global_state) = state_offset.checked_add(local_state) else {
                continue;
            };
            let Some(closure) = closures.get(local_state as usize) else {
                continue;
            };
            let mut bytes = U8Set::empty();
            for &closure_state in closure.iter() {
                for (byte, _) in component.tokenizer.transitions_from(closure_state) {
                    bytes.insert(byte);
                }
            }
            for byte in bytes.iter() {
                starts_by_first_byte[byte as usize].push(global_state);
            }
        }
    }
    for starts in &mut starts_by_first_byte {
        starts.sort_unstable();
        starts.dedup();
    }
    starts_by_first_byte
}

fn boundary_visible_residual_starts_by_token(
    vocab: &Vocab,
    candidate_tokens: &BTreeSet<u32>,
    starts_by_first_byte: &[Vec<u32>],
) -> BTreeMap<u32, Vec<u32>> {
    candidate_tokens
        .iter()
        .filter_map(|&token| {
            let bytes = vocab.entries_map().get(&token)?;
            (bytes.len() >= 2)
                .then(|| starts_by_first_byte[bytes[0] as usize].clone())
                .filter(|starts| !starts.is_empty())
                .map(|starts| (token, starts))
        })
        .collect()
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



fn boundary_terminal_residual_continuation_candidates(
    vocab: &Vocab,
    components: &[&Constraint],
    terminal_offsets: &[u32],
    terminals: &BitSet,
) -> Vec<u32> {
    let mut selected = BTreeSet::new();
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        for local_terminal in 0..component.tokenizer.num_terminals() {
            let global_terminal = terminal_offset + local_terminal;
            if !terminals.contains(global_terminal as usize) {
                continue;
            }
            let Some(expr) = component.tokenizer.terminal_expr(local_terminal) else {
                // Retained Expr metadata is expected for modern artifacts. If it
                // is absent, fall back conservatively to the exact component
                // possible-matches relation rather than dropping support.
                if let Some(weight) = component.possible_matches.get(&local_terminal) {
                    for (_, _, internal_tokens) in weight.range_entries() {
                        for internal_token in internal_tokens.iter() {
                            if component.internal_token_to_tokens.is_empty() {
                                if vocab
                                    .entries_map()
                                    .get(&internal_token)
                                    .is_some_and(|bytes| bytes.len() >= 2)
                                {
                                    selected.insert(internal_token);
                                }
                            } else if let Some(originals) =
                                component.internal_token_to_tokens.get(internal_token as usize)
                            {
                                selected.extend(originals.iter().copied().filter(|token| {
                                    vocab
                                        .entries_map()
                                        .get(token)
                                        .is_some_and(|bytes| bytes.len() >= 2)
                                }));
                            }
                        }
                    }
                }
                continue;
            };

            let dfa = compile_terminal_expr_dfa(expr);
            // States reachable after at least one byte are exactly the possible
            // within-terminal residual positions. Keep the start state too if a
            // nonempty cycle reaches it.
            let mut residual = vec![false; dfa.num_states()];
            let mut queue = VecDeque::new();
            for (_, target) in dfa.transitions(0) {
                if !residual[target as usize] {
                    residual[target as usize] = true;
                    queue.push_back(target);
                }
            }
            while let Some(state) = queue.pop_front() {
                for (_, target) in dfa.transitions(state) {
                    if !residual[target as usize] {
                        residual[target as usize] = true;
                        queue.push_back(target);
                    }
                }
            }
            let residual_states = residual
                .iter()
                .enumerate()
                .filter_map(|(state, &reachable)| reachable.then_some(state as u32))
                .collect::<Vec<_>>();
            if residual_states.is_empty() {
                continue;
            }

            for (&token, bytes) in vocab.entries_map() {
                if bytes.len() < 2 {
                    continue;
                }
                let mut current = residual_states.clone();
                for &byte in bytes {
                    let mut next = Vec::with_capacity(current.len());
                    for state in current {
                        if let Some(target) = dfa.step(state, byte) {
                            next.push(target);
                        }
                    }
                    if next.is_empty() {
                        current = next;
                        break;
                    }
                    next.sort_unstable();
                    next.dedup();
                    current = next;
                }
                if current.iter().copied().any(|state| {
                    dfa.finalizers(state).contains(0)
                        || dfa.possible_future_group_ids(state).contains(0)
                }) {
                    selected.insert(token);
                }
            }
        }
    }
    selected.into_iter().collect()
}


fn boundary_context_residual_states(
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    context_terminals: &BitSet,
) -> FxHashSet<u32> {
    let mut states = FxHashSet::default();
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        let state_offset = tokenizer_state_offsets[component_index];
        for local_terminal in 0..component.tokenizer.num_terminals() {
            let global_terminal = terminal_offset + local_terminal;
            if !context_terminals.contains(global_terminal as usize) {
                continue;
            }
            let Some(live_states) = component.terminal_live_states.get(local_terminal as usize) else {
                continue;
            };
            for &local_state in live_states {
                if local_state == component.tokenizer.start_state() {
                    continue;
                }
                if let Some(global_state) = state_offset.checked_add(local_state) {
                    states.insert(global_state);
                }
            }
        }
    }
    states
}

fn discover_boundary_token_paths(
    vocab: &Vocab,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    ignore_terminals: &BitSet,
    interface_pairs: &BTreeSet<(u32, u32)>,
    context_terminals: &BitSet,
    follow_transparent_terminals: &BitSet,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
) -> BoundaryTokenDiscovery {
    let num_terminals = components
        .iter()
        .zip(terminal_offsets.iter().copied())
        .map(|(component, offset)| offset + component.tokenizer.num_terminals())
        .max()
        .unwrap_or(0) as usize;
    let reset_starts = composite_reset_states(components, tokenizer_state_offsets);
    let reset_state_set = reset_starts.iter().copied().collect::<FxHashSet<_>>();
    let reset_live_bytes = component_reset_live_bytes(components);
    let mut residual_start_terminals = BitSet::new(num_terminals);
    for (terminal, &seed) in seed_terminals.iter().enumerate() {
        if seed && !ignore_terminals.contains(terminal) {
            residual_start_terminals.set(terminal);
        }
    }
    for &(left, _) in interface_pairs {
        if !ignore_terminals.contains(left as usize) {
            residual_start_terminals.set(left as usize);
        }
    }
    // Only parser-visible stack-neutral terminals need an explicit carry bit
    // across model-token boundaries. Ordinary component terminals keep their
    // parser support through the transported component DWA; the neutral LR
    // terminals are newly state-dependent behavior in the composed table.
    for terminal in context_terminals.iter() {
        if !ignore_terminals.contains(terminal) {
            residual_start_terminals.set(terminal);
        }
    }
    let use_component_scoped_identity =
        std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_SCOPED_IGNORE_TOP_ACCEPT").is_some();
    let boundary_context_states = if use_component_scoped_identity {
        FxHashSet::default()
    } else {
        boundary_context_residual_states(
            components,
            tokenizer_state_offsets,
            terminal_offsets,
            context_terminals,
        )
    };
    let use_prefilter = std::env::var_os("GLRMASK_COMPOSE_DISABLE_BOUNDARY_PREFILTER").is_none();
    let all_multi_byte_entries = vocab
        .entries_map()
        .iter()
        .filter(|(_, bytes)| bytes.len() >= 2)
        .map(|(&token_id, bytes)| (token_id, bytes.as_slice()))
        .collect::<Vec<_>>();
    let residual_starts_by_first_byte = boundary_visible_residual_starts_by_first_byte(
        components,
        tokenizer_state_offsets,
        terminal_offsets,
        &residual_start_terminals,
    );
    // Cheap candidate selection precedes all suffix execution. In the ordinary
    // case a seed-only boundary starts at byte zero; later boundaries inside a
    // token are supplied by exact grammar/LR interface byte pairs. Only a
    // globally erased terminal requires the conservative old suffix-seed rule.
    let prefilter_started_at = Instant::now();
    let allow_suffix_seed = !ignore_terminals.is_empty();
    let mut prefilter = if use_prefilter {
        boundary_token_prefilter(
            vocab,
            components,
            terminal_offsets,
            seed_terminals,
            allow_suffix_seed,
        )
    } else {
        all_multi_byte_entries
            .iter()
            .map(|&(token_id, _)| token_id)
            .collect::<BTreeSet<_>>()
    };
    let interface_pair_candidates = if use_prefilter {
        boundary_interface_adjacent_pair_candidates(
            vocab,
            components,
            terminal_offsets,
            interface_pairs,
        )
    } else {
        Vec::new()
    };
    prefilter.extend(interface_pair_candidates.iter().copied());
    let context_residual_candidates = if use_prefilter && !use_component_scoped_identity {
        boundary_terminal_residual_continuation_candidates(
            vocab,
            components,
            terminal_offsets,
            context_terminals,
        )
    } else {
        Vec::new()
    };
    prefilter.extend(context_residual_candidates.iter().copied());
    if let Some(path) = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TOKEN_ALLOWLIST") {
        let allowed = std::fs::read_to_string(path)
            .expect("read experimental boundary token allowlist")
            .lines()
            .filter_map(|line| line.trim().parse::<u32>().ok())
            .collect::<BTreeSet<_>>();
        prefilter.retain(|token| allowed.contains(token));
        if compose_profile_enabled() {
            eprintln!("[glrmask/profile][constraint_boundary_oracle_allowlist] allowed={} surviving_prefilter={}", allowed.len(), prefilter.len());
        }
    }
    let extra_residual_starts = boundary_visible_residual_starts_by_token(
        vocab,
        &prefilter,
        &residual_starts_by_first_byte,
    );
    let prefilter_ms = prefilter_started_at.elapsed().as_secs_f64() * 1000.0;
    let multi_byte_entries = all_multi_byte_entries
        .iter()
        .copied()
        .filter(|(token_id, _)| prefilter.contains(token_id))
        .collect::<Vec<_>>();

    // Exact graph expansion needs reset scans only for suffixes of surviving
    // candidates. Building this cache before coarse selection previously scanned
    // ~271k unique Llama-3 suffixes even when only a few thousand tokens could
    // witness a boundary.
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
                &extra_residual_starts,
                components,
                tokenizer_state_offsets,
            );
            candidate_start_visits.fetch_add(candidate_groups.len(), Ordering::Relaxed);
            max_candidate_starts.fetch_max(candidate_groups.len(), Ordering::Relaxed);
            let starts_by_scan = scan_component_residual_start_groups(
                components,
                tokenizer_state_offsets,
                terminal_offsets,
                &reset_live_bytes,
                bytes,
                &candidate_groups,
            );
            distinct_scan_groups.fetch_add(starts_by_scan.len(), Ordering::Relaxed);
            let mut scan_groups = starts_by_scan.into_iter().collect::<Vec<_>>();
            scan_groups.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            let mut local_terminals = FxHashSet::<u32>::default();
            let mut local_witnesses = Vec::new();
            for (arbitrary_scan, start_states) in scan_groups {
                let mut residual_starts = Vec::new();
                let mut reset_starts_for_scan = Vec::new();
                let mut contextual_starts = Vec::new();
                for start_state in start_states {
                    if boundary_context_states.contains(&start_state) {
                        contextual_starts.push(start_state);
                    } else if reset_state_set.contains(&start_state) {
                        reset_starts_for_scan.push(start_state);
                    } else {
                        residual_starts.push(start_state);
                    }
                }
                for (start_states, initial_interface_witnessed, allow_seed_only) in [
                    (residual_starts, false, false),
                    (reset_starts_for_scan, false, !ignore_terminals.is_empty()),
                    (contextual_starts, true, false),
                ] {
                    if start_states.is_empty() {
                        continue;
                    }
                    let Some((nodes, good, accepting)) = build_boundary_token_graph(
                        bytes,
                        &arbitrary_scan,
                        &reset_scans,
                        seed_terminals,
                        ignore_terminals,
                        interface_pairs,
                        initial_interface_witnessed,
                        allow_seed_only,
                        disallowed_follows,
                        follow_transparent_terminals,
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
            "[glrmask/profile][constraint_boundary_prefilter] enabled={} interface_pair_candidates={} context_residual_candidates={} candidates={} scanned={} exact={} missing={} prefilter_ms={prefilter_ms:.3} missing_ids={:?}",
            use_prefilter,
            interface_pair_candidates.len(),
            context_residual_candidates.len(),
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
        let local_tsid_count = component.internal_tsid_to_states.len();
        if local_tsid_count == 0 {
            return Err(format!("component {component_index} has no internal TSIDs"));
        }
        if tokenizer_tsid_relation_is_singleton(component) {
            for local_states in &component.internal_tsid_to_states {
                let mut merged_states = Vec::with_capacity(local_states.len());
                for &local_state in local_states {
                    let merged_state = state_offset
                        .checked_add(local_state)
                        .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
                    if merged_state != 0 {
                        merged_states.push(merged_state);
                    }
                }
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
            continue;
        }
        let mut states_by_signature = BTreeMap::<Vec<u32>, Vec<u32>>::new();
        for local_state in 0..component.tokenizer.num_states() {
            let mut signature = component.internal_tsids_for_state(local_state).to_vec();
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
            let merged_state = state_offset
                .checked_add(local_state)
                .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
            if merged_state as usize >= merged_tokenizer_state_count {
                return Err(format!(
                    "component {component_index} tokenizer state {local_state} maps outside merged tokenizer"
                ));
            }
            if merged_state != 0 {
                states_by_signature
                    .entry(signature)
                    .or_default()
                    .push(merged_state);
            }
        }
        for (_signature, mut merged_states) in states_by_signature {
            merged_states.sort_unstable();
            merged_states.dedup();
            let global_tsid = global_to_states.len() as u32;
            for &merged_state in &merged_states {
                let Some(slot) = state_to_global.get_mut(merged_state as usize) else {
                    return Err(format!(
                        "component {component_index} tokenizer state {merged_state} lies outside merged tokenizer",
                    ));
                };
                if *slot != u32::MAX {
                    return Err(format!(
                        "merged tokenizer state {merged_state} belongs to multiple exact membership classes",
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
    coordinate_original_tokens: &[u32],
    seed_relations: BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
    one_byte_ms: f64,
    discovery: &BoundaryTokenDiscovery,
    globally_erasable_ignore_terminals: &BitSet,
    control_terminals: &BTreeSet<u32>,
    terminal_offsets: &[u32],
    tokenizer_state_offsets: &[u32],
    delta_plan: Option<&ConcreteBoundaryDeltaPlan>,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    let total_started_at = Instant::now();

    // Keep the token coordinate published to the owned component-preparation
    // lane exactly, even when later semantic factoring proves some candidates
    // redundant.  The coordinate may contain unused token classes; changing it
    // after publication would invalidate the concurrently prepared remap.
    let selected_original_tokens = coordinate_original_tokens
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let max_original_token = vocab
        .entries_map()
        .keys()
        .next_back()
        .copied()
        .into_iter()
        .chain(selected_original_tokens.iter().copied())
        .max()
        .unwrap_or(0);
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
    // Do not build and persistently union one singleton Weight per boundary
    // witness. Large composed children routinely have tens of thousands of
    // fused vocabulary witnesses that collapse onto the same canonical suffix
    // graph; repeated Weight::union then dominates composition. Accumulate the
    // exact support relation first and materialize one Weight per canonical
    // start only after all witnesses have been canonicalized.
    let mut start_tokens_by_canonical =
        BTreeMap::<usize, BTreeMap<u32, BTreeSet<u32>>>::new();
    let mut intern_canonical = |key: CanonicalNodeKey| -> usize {
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

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct ExpandedKey {
        local: usize,
        last_component: Option<usize>,
        crossed: bool,
        changed_count: u8,
        unsafe_path: bool,
    }
    #[derive(Debug, Clone)]
    struct ExpandedEdge {
        target: usize,
        terminal: u32,
    }
    #[derive(Debug, Clone)]
    struct ExpandedNode {
        key: ExpandedKey,
        outgoing: Vec<ExpandedEdge>,
    }
    #[derive(Debug, Clone, Copy)]
    enum DeltaLane {
        CrossedFull,
        LocalComplexFull,
        LocalDeltaNovelty,
    }

    let terminal_component = |terminal: u32| -> usize {
        terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .saturating_sub(1)
    };
    let tokenizer_state_component = |state: u32| -> Option<usize> {
        if state == 0 {
            None
        } else {
            Some(
                tokenizer_state_offsets
                    .partition_point(|&offset| offset <= state)
                    .saturating_sub(1),
            )
        }
    };
    let mut delta_cross_lane_starts = 0usize;
    let mut delta_cross_tokens = BTreeSet::<u32>::new();
    let mut delta_complex_lane_starts = 0usize;
    let mut delta_single_lane_starts = 0usize;
    let mut delta_start_groups = 0usize;
    let cross_lane_only =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_CROSS_LANE_ONLY").is_some();

    for witness in &discovery.witnesses {
        let Some(internal_token) = id_map.internal_token_for_original(witness.token_id) else {
            continue;
        };

        let Some(delta_plan) = delta_plan else {
            let witness_tsids = witness
                .start_states
                .iter()
                .map(|&state| state_to_tsid(state))
                .collect::<BTreeSet<_>>();
            if witness_tsids.is_empty() {
                continue;
            }

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
                local_to_canonical[local] = intern_canonical(CanonicalNodeKey {
                    accepting: witness.accepting[local],
                    transitions,
                    epsilons,
                });
            }
            let start = local_to_canonical[0];
            let tokens_by_tsid = start_tokens_by_canonical.entry(start).or_default();
            for tsid in witness_tsids {
                tokens_by_tsid.entry(tsid).or_default().insert(internal_token);
            }
            continue;
        };

        // The same model token may be valid from tokenizer states owned by
        // different cached components. Keep those support classes separate:
        // "component-local" is a statement about a particular start component,
        // not about the token globally.
        let mut starts_by_component = BTreeMap::<Option<usize>, Vec<u32>>::new();
        for &state in &witness.start_states {
            starts_by_component
                .entry(tokenizer_state_component(state))
                .or_default()
                .push(state);
        }
        for (initial_component, start_states) in starts_by_component {
            delta_start_groups += 1;
            let witness_tsids = start_states
                .iter()
                .map(|&state| state_to_tsid(state))
                .collect::<BTreeSet<_>>();
            if witness_tsids.is_empty() {
                continue;
            }

            let start_key = ExpandedKey {
                local: 0,
                last_component: initial_component,
                crossed: false,
                changed_count: 0,
                // Merged tokenizer state 0 epsilon-dispatches to every
                // component start, and component parser-DWA transport maps each
                // local start TSID onto this same global TSID 0. Therefore a
                // reset-origin path becomes component-local as soon as its
                // first committed terminal chooses a component; no extra
                // conservatism is required merely because the dispatcher was
                // the lexical start state.
                unsafe_path: false,
            };
            let mut expanded_by_key = FxHashMap::<ExpandedKey, usize>::default();
            let mut expanded_nodes = vec![ExpandedNode {
                key: start_key,
                outgoing: Vec::new(),
            }];
            expanded_by_key.insert(start_key, 0);
            let mut queue = VecDeque::from([0usize]);
            while let Some(source) = queue.pop_front() {
                let source_key = expanded_nodes[source].key;
                if !witness.good[source_key.local] {
                    continue;
                }
                let source_edges = witness.nodes[source_key.local]
                    .outgoing
                    .iter()
                    .filter(|edge| witness.good[edge.target])
                    .cloned()
                    .collect::<Vec<_>>();
                for edge in source_edges {
                    let next_component = terminal_component(edge.terminal);
                    let parser_relevant = !globally_erasable_ignore_terminals
                        .contains(edge.terminal as usize);
                    let changed = parser_relevant
                        && delta_plan.by_global_terminal.contains_key(&edge.terminal);
                    let unsafe_terminal = parser_relevant
                        && delta_plan.unsafe_terminals.contains(&edge.terminal);
                    let switched = source_key
                        .last_component
                        .is_some_and(|component| component != next_component);
                    let next_key = ExpandedKey {
                        local: edge.target,
                        last_component: Some(next_component),
                        crossed: source_key.crossed || switched,
                        changed_count: source_key
                            .changed_count
                            .saturating_add(u8::from(changed))
                            .min(2),
                        unsafe_path: source_key.unsafe_path || unsafe_terminal,
                    };
                    let target = if let Some(&target) = expanded_by_key.get(&next_key) {
                        target
                    } else {
                        let target = expanded_nodes.len();
                        expanded_by_key.insert(next_key, target);
                        expanded_nodes.push(ExpandedNode {
                            key: next_key,
                            outgoing: Vec::new(),
                        });
                        queue.push_back(target);
                        target
                    };
                    expanded_nodes[source].outgoing.push(ExpandedEdge {
                        target,
                        terminal: edge.terminal,
                    });
                }
            }

            // Every edge consumes positive byte width in the witness DAG, so
            // descending byte offset is a reverse topological order even after
            // the finite metadata expansion above.
            let mut reverse_order = (0..expanded_nodes.len()).collect::<Vec<_>>();
            reverse_order.sort_unstable_by_key(|&node| {
                std::cmp::Reverse(witness.nodes[expanded_nodes[node].key.local].key.offset)
            });

            for lane in [
                DeltaLane::CrossedFull,
                DeltaLane::LocalComplexFull,
                DeltaLane::LocalDeltaNovelty,
            ] {
                if cross_lane_only && !matches!(lane, DeltaLane::CrossedFull) {
                    continue;
                }
                // For a safe component-local terminal word with one or more
                // ordinary changed terminals, do not rebuild the whole
                // composed word.  For every changed terminal t we have proved
                // Old_t âŠ† New_t and materialized the disjoint remainder
                // Delta_t = New_t \\ Old_t.  Therefore, for a word t1..tn,
                //
                //   New_1..New_n \\ Old_1..Old_n
                //
                // is the disjoint union obtained by choosing the *first*
                // changed occurrence that takes Delta: all earlier changed
                // occurrences take Old, that occurrence takes Delta, and all
                // later occurrences take New.  The cached component parser
                // artifact supplies the omitted all-Old branch.  The boolean
                // product below is exactly that first-delta decomposition and
                // works for arbitrarily many changed terminals without a full
                // local repair lane.
                if matches!(lane, DeltaLane::LocalDeltaNovelty) {
                    let lexical_accepts = |key: ExpandedKey| {
                        witness.accepting[key.local]
                            && !key.crossed
                            && !key.unsafe_path
                            && key.changed_count != 0
                    };
                    let mut productive = vec![[false; 2]; expanded_nodes.len()];
                    let mut expanded_to_canonical =
                        vec![[usize::MAX; 2]; expanded_nodes.len()];

                    for &source in &reverse_order {
                        for novelty_seen in [true, false] {
                            let seen_index = usize::from(novelty_seen);
                            let accepting = novelty_seen && lexical_accepts(expanded_nodes[source].key);
                            let mut transitions = Vec::new();
                            let mut epsilons = Vec::new();

                            for edge in &expanded_nodes[source].outgoing {
                                if globally_erasable_ignore_terminals
                                    .contains(edge.terminal as usize)
                                {
                                    if productive[edge.target][seen_index] {
                                        let target =
                                            expanded_to_canonical[edge.target][seen_index];
                                        debug_assert_ne!(target, usize::MAX);
                                        epsilons.push(target);
                                    }
                                    continue;
                                }

                                if let Some(entry) = delta_plan.by_global_terminal.get(&edge.terminal) {
                                    if novelty_seen {
                                        if productive[edge.target][1] {
                                            let target = expanded_to_canonical[edge.target][1];
                                            debug_assert_ne!(target, usize::MAX);
                                            // Once novelty has occurred, later
                                            // changed terminals may use all of
                                            // New = Old âˆª Delta.
                                            transitions.push((edge.terminal, target));
                                        }
                                    } else {
                                        if productive[edge.target][0] {
                                            let target = expanded_to_canonical[edge.target][0];
                                            debug_assert_ne!(target, usize::MAX);
                                            transitions.push((entry.old_terminal, target));
                                        }
                                        if productive[edge.target][1] {
                                            let target = expanded_to_canonical[edge.target][1];
                                            debug_assert_ne!(target, usize::MAX);
                                            transitions.push((entry.delta_terminal, target));
                                        }
                                    }
                                } else if productive[edge.target][seen_index] {
                                    let target = expanded_to_canonical[edge.target][seen_index];
                                    debug_assert_ne!(target, usize::MAX);
                                    transitions.push((edge.terminal, target));
                                }
                            }

                            transitions.sort_unstable();
                            transitions.dedup();
                            epsilons.sort_unstable();
                            epsilons.dedup();
                            if !accepting && transitions.is_empty() && epsilons.is_empty() {
                                continue;
                            }
                            productive[source][seen_index] = true;
                            expanded_to_canonical[source][seen_index] =
                                intern_canonical(CanonicalNodeKey {
                                    accepting,
                                    transitions,
                                    epsilons,
                                });
                        }
                    }

                    if !productive[0][0] {
                        continue;
                    }
                    let start = expanded_to_canonical[0][0];
                    let tokens_by_tsid = start_tokens_by_canonical.entry(start).or_default();
                    for &tsid in &witness_tsids {
                        tokens_by_tsid.entry(tsid).or_default().insert(internal_token);
                    }
                    delta_single_lane_starts += 1;
                    continue;
                }

                let accepts = |key: ExpandedKey| {
                    if !witness.accepting[key.local] {
                        return false;
                    }
                    match lane {
                        DeltaLane::CrossedFull => {
                            if std::env::var_os("GLRMASK_EXPERIMENT_STRICT_INTERFACE_CROSS_LANE").is_some() {
                                key.crossed && witness.nodes[key.local].key.interface_witnessed
                            } else {
                                key.crossed
                            }
                        }
                        DeltaLane::LocalComplexFull => !key.crossed && key.unsafe_path,
                        DeltaLane::LocalDeltaNovelty => unreachable!(),
                    }
                };
                let mut productive = expanded_nodes
                    .iter()
                    .map(|node| accepts(node.key))
                    .collect::<Vec<_>>();
                for &source in &reverse_order {
                    if !productive[source]
                        && expanded_nodes[source]
                            .outgoing
                            .iter()
                            .any(|edge| productive[edge.target])
                    {
                        productive[source] = true;
                    }
                }
                if !productive[0] {
                    continue;
                }

                let mut expanded_to_canonical = vec![usize::MAX; expanded_nodes.len()];
                for &source in &reverse_order {
                    if !productive[source] {
                        continue;
                    }
                    let mut transitions = Vec::new();
                    let mut epsilons = Vec::new();
                    for edge in expanded_nodes[source]
                        .outgoing
                        .iter()
                        .filter(|edge| productive[edge.target])
                    {
                        let target = expanded_to_canonical[edge.target];
                        debug_assert_ne!(target, usize::MAX);
                        if globally_erasable_ignore_terminals.contains(edge.terminal as usize) {
                            epsilons.push(target);
                        } else {
                            let terminal = match lane {
                                DeltaLane::LocalDeltaNovelty => unreachable!(),
                                DeltaLane::CrossedFull | DeltaLane::LocalComplexFull => edge.terminal,
                            };
                            transitions.push((terminal, target));
                        }
                    }
                    transitions.sort_unstable();
                    transitions.dedup();
                    epsilons.sort_unstable();
                    epsilons.dedup();
                    expanded_to_canonical[source] = intern_canonical(CanonicalNodeKey {
                        accepting: accepts(expanded_nodes[source].key),
                        transitions,
                        epsilons,
                    });
                }
                let start = expanded_to_canonical[0];
                let tokens_by_tsid = start_tokens_by_canonical.entry(start).or_default();
                for &tsid in &witness_tsids {
                    tokens_by_tsid.entry(tsid).or_default().insert(internal_token);
                }
                match lane {
                    DeltaLane::CrossedFull => {
                        delta_cross_lane_starts += 1;
                        delta_cross_tokens.insert(witness.token_id);
                    },
                    DeltaLane::LocalComplexFull => delta_complex_lane_starts += 1,
                    DeltaLane::LocalDeltaNovelty => unreachable!(),
                }
            }
            // Safe, non-crossing accepting paths with zero changed terminals
            // are intentionally absent: their parser behavior is already in
            // the transported cached component parser DWA.
        }
    }

    let start_weights_by_canonical = start_tokens_by_canonical
        .into_iter()
        .map(|(canonical, tokens_by_tsid)| {
            let weight = Weight::from_per_tsid_token_sets(
                tokens_by_tsid.into_iter().map(|(tsid, tokens)| {
                    (tsid, tokens.into_iter().collect::<RangeSetBlaze<_>>())
                }),
            );
            (canonical, weight)
        })
        .collect::<BTreeMap<_, _>>();

    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let global_start = nwa.add_state();
    let seed_final = nwa.add_state();
    nwa.set_final_weight(seed_final, Weight::all());
    let canonical_state_offset = nwa.num_states();
    for _ in &canonical_nodes {
        nwa.add_state();
    }
    nwa.set_start_states(vec![global_start]);

    if !cross_lane_only {
        for (sequence, by_state) in seed_relations {
            debug_assert_eq!(sequence.len(), 1);
            let weight = relation_weight(by_state);
            if !weight.is_empty() {
                nwa.add_transition(global_start, sequence[0] as i32, seed_final, weight);
            }
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
    if compose_profile_enabled() {
        let mut support_tsids = BTreeSet::<u32>::new();
        let mut support_raw_states = BTreeSet::<u32>::new();
        let mut support_internal_tokens = BTreeSet::<u32>::new();
        let mut support_original_tokens = BTreeSet::<u32>::new();
        let start_node = &nwa.states()[global_start as usize];
        let start_weights = start_node
            .transitions
            .values()
            .flatten()
            .map(|(_, weight)| weight)
            .chain(start_node.epsilons.iter().map(|(_, weight)| weight));
        for weight in start_weights {
            for (range, tokens) in weight.raw_range_values() {
                for tsid in *range.start()..=*range.end() {
                    support_tsids.insert(tsid);
                    if let Some(raws) = id_map.tokenizer_states.internal_to_originals.get(tsid as usize) {
                        support_raw_states.extend(raws.iter().copied());
                    }
                }
                for token_range in tokens.ranges() {
                    for token in token_range {
                        support_internal_tokens.insert(token);
                        if let Some(originals) = id_map.vocab_tokens.internal_to_originals.get(token as usize) {
                            support_original_tokens.extend(originals.iter().copied());
                        }
                    }
                }
            }
        }
        eprintln!(
            "[glrmask/profile][constraint_boundary_start_support] tsids={} raw_states={} internal_tokens={} original_tokens={}",
            support_tsids.len(),
            support_raw_states.len(),
            support_internal_tokens.len(),
            support_original_tokens.len(),
        );
    }
    let raw_states = nwa.num_states();
    let raw_transitions = nwa.num_transitions();
    let canonical_state_count = raw_states.saturating_sub(canonical_state_offset);
    let build_ms = build_started_at.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let dwa = determinize(&nwa).map_err(|error| error.to_string())?;
    let determinize_ms = started.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let mut dwa = minimize_owned(dwa);
    let minimize_ms = started.elapsed().as_secs_f64() * 1000.0;
    for state in 0..dwa.num_states() {
        for &control in control_terminals {
            dwa.add_transition(state, control as i32, state, Weight::all());
        }
    }
    let final_states = dwa.num_states();
    let final_transitions = dwa.num_transitions();
    let terminal_automaton = TerminalAutomaton::Dwa(dwa);

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_direct_terminal] witnesses={} selected_tokens={} raw_lexer_states={} boundary_tsids={} canonical_states={} raw_states={} raw_transitions={} final_states={} final_transitions={} controls={} delta_cross_lane_starts={} delta_cross_tokens={} delta_complex_lane_starts={} delta_single_lane_starts={} delta_start_groups={} one_byte_ms={one_byte_ms:.3} quotient_ms={quotient_ms:.3} build_ms={build_ms:.3} determinize_ms={determinize_ms:.3} minimize_ms={minimize_ms:.3} total_ms={:.3}",
            discovery.witnesses.len(),
            selected_original_tokens.len(),
            num_states,
            id_map.num_tsids(),
            canonical_state_count,
            raw_states,
            raw_transitions,
            final_states,
            final_transitions,
            control_terminals.len(),
            delta_cross_lane_starts,
            delta_cross_tokens.len(),
            delta_complex_lane_starts,
            delta_single_lane_starts,
            delta_start_groups,
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
        std::env::var_os("GLRMASK_EXPERIMENT_PREALLOCATE_RAW_SINGLETONS").is_some();
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
        std::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_RESIDUAL_ROW_CACHE").is_none();
    let use_interval_pair_runs =
        std::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_INTERVAL_RUNS").is_none();
    let defer_support_unions =
        std::env::var_os("GLRMASK_DISABLE_DIRECT_UNION_DEFER_SUPPORT_UNIONS").is_none()
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
    let residual_prefetch_min_raw_states = std::env::var(
        "GLRMASK_DIRECT_UNION_RESIDUAL_PREFETCH_MIN_RAW_STATES",
    )
    .ok()
    .and_then(|value| value.parse::<usize>().ok())
    .unwrap_or(65_536);
    let use_parallel_residual_row_prefetch = use_residual_row_cache
        && raw_states.len() >= residual_prefetch_min_raw_states
        && std::env::var_os("GLRMASK_DISABLE_PARALLEL_RESIDUAL_ROW_PREFETCH").is_none()
        && rayon::current_num_threads() > 1;
    let residual_row_prefetch_batch = std::env::var("GLRMASK_RESIDUAL_ROW_PREFETCH_BATCH")
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
                std::env::var_os("GLRMASK_DISABLE_DEDUP_RESIDUAL_PREFETCH_JOBS").is_none();
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

fn prepare_unmapped_component_parser_artifacts(
    components: &[ParserDwaComponent<'_>],
    terminal_offsets: &[u32],
    default_domains: &[Option<ParserDefaultDomain>],
    strip_scoped_ignore_identity: bool,
) -> Result<Vec<UnmappedComponentParserArtifact>, String> {
    if components.len() != terminal_offsets.len() || components.len() != default_domains.len() {
        return Err("component/parser terminal-offset/default-domain count mismatch".into());
    }
    components
        .par_iter()
        .copied()
        .zip(terminal_offsets.par_iter().copied())
        .zip(default_domains.par_iter())
        .map(|((component, terminal_offset), default_domain)| {
            let possible_matches = component_possible_matches(&component, terminal_offset)?;
            let mut automaton = component_parser_nwa(&component, default_domain.as_ref())?;
            if strip_scoped_ignore_identity {
                let ignore_weight = component
                    .constraint
                    .ignore_terminal
                    .and_then(|ignore| possible_matches.get(&(terminal_offset + ignore)));
                strip_unscoped_ignore_identity(&mut automaton, ignore_weight);
            }
            Ok(UnmappedComponentParserArtifact {
                automaton,
                possible_matches,
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
    /// Per-automaton maps into `id_map`. Parser weights deliberately remain in
    /// their cached component-local coordinates until the final
    /// component+boundary union consumes them. This keeps component linking a
    /// structural operation; only the tiny PossibleMatches table is eagerly
    /// published in the final coordinate space.
    automata_maps: Vec<DirectComponentCoordinateMaps>,
    possible_matches: PossibleMatches,
    id_map: InternalIdMap,
    boundary_tsid_map: Option<Vec<Vec<u32>>>,
    boundary_token_map: Option<Vec<Vec<u32>>>,
    remap_ms: f64,
}

fn prepare_deferred_component_artifacts(
    artifacts: Vec<UnmappedComponentParserArtifact>,
    component_maps: Vec<DirectComponentCoordinateMaps>,
    base_to_common_tokens: Option<&[Vec<u32>]>,
    common_tsid_count: usize,
) -> Result<(Vec<NWA>, Vec<DirectComponentCoordinateMaps>, PossibleMatches, f64), String> {
    if artifacts.len() != component_maps.len() {
        return Err("component artifact/map count mismatch".into());
    }
    let started_at = Instant::now();
    let prepared = artifacts
        .into_par_iter()
        .zip(component_maps.into_par_iter())
        .map(|(artifact, mut maps)| {
            if let Some(base_to_common) = base_to_common_tokens {
                maps.local_to_global_tokens =
                    compose_local_id_map(&maps.local_to_global_tokens, base_to_common);
            }
            let mut possible_matches = artifact.possible_matches;
            let mut weights = possible_matches.weight_refs_mut();
            remap_weights_with_maps(
                &mut weights,
                &maps.local_to_global_tsids,
                &maps.local_to_global_tokens,
                common_tsid_count,
            );
            drop(weights);
            Ok::<_, String>((artifact.automaton, maps, possible_matches))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut automata = Vec::with_capacity(prepared.len());
    let mut maps = Vec::with_capacity(prepared.len());
    let mut possible_matches = PossibleMatches::new();
    for (automaton, map, component_possible_matches) in prepared {
        automata.push(automaton);
        maps.push(map);
        for (terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
    }
    Ok((
        automata,
        maps,
        possible_matches,
        started_at.elapsed().as_secs_f64() * 1000.0,
    ))
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
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][constraint_component_weight_ref_shape] parser_refs={} possible_match_refs={}",
                    pair.0.weight_refs().len(),
                    pair.1.weight_refs().len(),
                );
            }
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
    let prepared = components
        .par_iter()
        .copied()
        .zip(terminal_offsets.par_iter().copied())
        .zip(default_domains.par_iter())
        .zip(component_maps.into_par_iter())
        .enumerate()
        .map(|(component_index, (((component, terminal_offset), default_domain), coordinate_maps))| {
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
    if std::env::var_os("GLRMASK_PROFILE_EXACT_BOUNDARY_PARSER_DELTA").is_none() {
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
    let paired = if std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_RECONCILE").is_some() {
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
    let direct = if std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_UNION").is_some() {
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
            if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_BOUNDARY_DIRECT_UNION").is_some() {
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
    let mut result = BTreeMap::new();
    for (component_index, component) in components.iter().copied().enumerate() {
        let terminal_offset = composed_table.terminal_offsets[component_index];
        let mut selected = vec![false; component.table.num_terminals as usize];
        for (local, selected_slot) in selected.iter_mut().enumerate() {
            let global = terminal_offset as usize + local;
            *selected_slot = active_terminals.get(global).copied().unwrap_or(false);
        }
        if !selected.iter().any(|&value| value) {
            continue;
        }
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
        let characterizations = characterize_selected_terminals(&component.table, &analyzed, &selected);
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


/// Exact reset/base-case support for a one-terminal parser-template delta.
///
/// `collect_one_byte_seed_relations*` covers arbitrary lexer states but only
/// one-byte model tokens.  This companion relation covers arbitrary-length
/// model tokens that can complete one selected grammar terminal exactly at the
/// lexer reset.  The parser word is still length one, so it is governed by the
/// same Old/New/Delta factorization.
fn boundary_delta_reset_relations(
    components: &[&Constraint],
    terminal_offsets: &[u32],
    vocab: &Vocab,
    selected_terminals: &[bool],
    control_terminals: &BTreeSet<u32>,
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    debug_assert_eq!(components.len(), terminal_offsets.len());
    let started_at = Instant::now();

    let mut selected_by_component = Vec::<BitSet>::with_capacity(components.len());
    for (component_index, component) in components.iter().enumerate() {
        let terminal_offset = terminal_offsets[component_index];
        let mut selected = BitSet::new(component.tokenizer.num_terminals() as usize);
        for local in 0..component.tokenizer.num_terminals() {
            let global = terminal_offset + local;
            if selected_terminals
                .get(global as usize)
                .copied()
                .unwrap_or(false)
                && !control_terminals.contains(&global)
            {
                selected.set(local as usize);
            }
        }
        selected_by_component.push(selected);
    }

    let mut pairs = Vec::<(u32, u32)>::new();
    let mut fallback_components = Vec::<usize>::new();
    let mut cached_components = 0usize;
    for (component_index, component) in components.iter().enumerate() {
        let selected = &selected_by_component[component_index];
        if selected.is_empty() {
            continue;
        }
        if component.composition_reset_tokens_by_terminal.len()
            == component.tokenizer.num_terminals() as usize
        {
            cached_components += 1;
            let terminal_offset = terminal_offsets[component_index];
            for local_terminal in selected.iter() {
                if let Some(tokens) = component
                    .composition_reset_tokens_by_terminal
                    .get(local_terminal)
                {
                    pairs.extend(
                        tokens
                            .iter()
                            .copied()
                            .map(|token| (terminal_offset + local_terminal as u32, token)),
                    );
                }
            }
        } else {
            fallback_components.push(component_index);
        }
    }

    // Old/unprepared artifacts remain exact. Scan only those components; a
    // current composition-ready cache never touches the vocabulary here.
    if !fallback_components.is_empty() {
        let mut fallback_pairs = vocab
            .entries_map()
            .par_iter()
            .filter(|(_, bytes)| !bytes.is_empty())
            .fold(Vec::<(u32, u32)>::new, |mut output, (&token_id, bytes)| {
                for &component_index in &fallback_components {
                    let component = components[component_index];
                    let selected = &selected_by_component[component_index];
                    let (_, matches) = component
                        .tokenizer
                        .execute_summary_from_state(bytes, component.tokenizer.start_state());
                    for (local_terminal, width) in matches {
                        if width == bytes.len() && selected.contains(local_terminal as usize) {
                            output.push((
                                terminal_offsets[component_index] + local_terminal,
                                token_id,
                            ));
                        }
                    }
                }
                output
            })
            .reduce(Vec::new, |mut left, mut right| {
                left.append(&mut right);
                left
            });
        pairs.append(&mut fallback_pairs);
    }

    let mut result = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    for (terminal, token) in pairs {
        result
            .entry(vec![terminal])
            .or_default()
            .entry(0)
            .or_default()
            .insert(token);
    }
    if compose_profile_enabled() {
        let token_cells = result
            .values()
            .flat_map(|by_state| by_state.values())
            .map(BTreeSet::len)
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_delta_reset_relations] terminals={} token_cells={} cached_components={} fallback_components={} ms={:.3}",
            result.len(),
            token_cells,
            cached_components,
            fallback_components.len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    result
}

fn merge_one_terminal_relations(
    into: &mut BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
    from: BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
) {
    for (sequence, by_state) in from {
        let target_by_state = into.entry(sequence).or_default();
        for (state, tokens) in by_state {
            target_by_state.entry(state).or_default().extend(tokens);
        }
    }
}

/// Rewrite one-terminal boundary seeds into the exact additive parser-template
/// novelty supplied by composition.  A transported component normally already
/// supplies Old_t, so a safe changed terminal contributes only
/// Delta_t = New_t \\ Old_t and an unchanged compared terminal contributes
/// nothing.  The exception is support shadowed by the normalized component
/// parser's start-final acceptance: prefix-final subtraction removed Old_t's
/// outgoing branch for that exact (lexer-state, model-token) support cell, so
/// the boundary must retain full New_t there.
fn factor_one_terminal_seed_relations(
    relations: BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
    plan: &ConcreteBoundaryDeltaPlan,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    debug_assert_eq!(components.len(), tokenizer_state_offsets.len());
    debug_assert_eq!(components.len(), terminal_offsets.len());

    let owner_for_terminal = |terminal: u32| -> Option<usize> {
        let component = terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .checked_sub(1)?;
        let local = terminal.checked_sub(*terminal_offsets.get(component)?)?;
        (local < components.get(component)?.tokenizer.num_terminals()).then_some(component)
    };

    let scoped_ignore_shadow_is_separately_supplied = |terminal: u32| -> bool {
        if std::env::var_os("GLRMASK_EXPERIMENT_SCOPED_IGNORE_SHADOW_REUSE").is_none()
            || std::env::var_os("GLRMASK_EXPERIMENT_COMPONENT_SCOPED_IGNORE_TOP_ACCEPT").is_none()
        {
            return false;
        }
        let Some(component_index) = owner_for_terminal(terminal) else {
            return false;
        };
        let Some(ignore) = components[component_index].ignore_terminal else {
            return false;
        };
        terminal_offsets[component_index] + ignore == terminal
    };

    let support_shadowed_at_component_start =
        |terminal: u32, raw_state: u32, original_token: u32| -> Option<bool> {
            let component_index = owner_for_terminal(terminal)?;
            let component = *components.get(component_index)?;
            let state_offset = *tokenizer_state_offsets.get(component_index)?;
            let local_state = if raw_state == 0 {
                component.tokenizer.start_state()
            } else {
                let local = raw_state.checked_sub(state_offset)?;
                (local < component.tokenizer.num_states()).then_some(local)?
            };
            let internal_token = component
                .original_token_to_internal
                .get(original_token as usize)
                .copied()?;
            if internal_token == u32::MAX {
                return None;
            }
            let start_final = component
                .parser_dwa
                .states()
                .get(component.parser_dwa.start_state() as usize)?
                .final_weight
                .as_ref();
            let Some(start_final) = start_final else {
                return Some(false);
            };
            Some(
                component
                    .internal_tsids_for_state(local_state)
                    .iter()
                    .copied()
                    .any(|tsid| start_final.tokens_for_tsid(tsid).contains(internal_token)),
            )
        };

    fn insert_relation(
        factored: &mut BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
        label: u32,
        state: u32,
        token: u32,
    ) {
        factored
            .entry(vec![label])
            .or_default()
            .entry(state)
            .or_default()
            .insert(token);
    }

    let mut factored = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    let mut promoted_shadow_cells = 0usize;
    let mut promoted_shadow_tokens = BTreeSet::<u32>::new();
    let mut promoted_shadow_by_terminal = BTreeMap::<u32, BTreeSet<u32>>::new();
    let mut reused_scoped_ignore_shadow_cells = 0usize;
    let mut dropped_cells = 0usize;
    let mut delta_cells = 0usize;
    let mut conservative_cells = 0usize;

    for (sequence, by_state) in relations {
        if sequence.len() != 1 {
            // This helper is deliberately only the n=1 base case of the
            // first-delta decomposition.  Preserve any future non-unit caller
            // conservatively rather than silently changing its language.
            factored.insert(sequence, by_state);
            continue;
        }
        let terminal = sequence[0];
        let entry = plan.by_global_terminal.get(&terminal);
        let unsafe_or_unclassified = plan.unsafe_terminals.contains(&terminal)
            || (!plan.compared_terminals.contains(&terminal) && entry.is_none());

        for (state, tokens) in by_state {
            for token in tokens {
                if unsafe_or_unclassified {
                    conservative_cells += 1;
                    insert_relation(&mut factored, terminal, state, token);
                    continue;
                }
                match support_shadowed_at_component_start(terminal, state, token) {
                    Some(true) => {
                        if scoped_ignore_shadow_is_separately_supplied(terminal) {
                            reused_scoped_ignore_shadow_cells += 1;
                            continue;
                        }
                        promoted_shadow_cells += 1;
                        promoted_shadow_tokens.insert(token);
                        promoted_shadow_by_terminal.entry(terminal).or_default().insert(token);
                        insert_relation(&mut factored, terminal, state, token);
                    }
                    Some(false) => {
                        if let Some(entry) = entry {
                            delta_cells += 1;
                            insert_relation(&mut factored, entry.delta_terminal, state, token);
                        } else {
                            // Compared and absent from the delta map means
                            // Old_t == New_t; the component already supplies it.
                            dropped_cells += 1;
                        }
                    }
                    None => {
                        // If ownership/coordinate provenance cannot be proved,
                        // keep full New_t.  Correctness beats the optimization.
                        conservative_cells += 1;
                        insert_relation(&mut factored, terminal, state, token);
                    }
                }
            }
        }
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_factor_one_terminal] rows={} promoted_shadow_cells={} promoted_shadow_tokens={} promoted_shadow_by_terminal={:?} reused_scoped_ignore_shadow_cells={} delta_cells={} dropped_cells={} conservative_cells={}",
            factored.len(),
            promoted_shadow_cells,
            promoted_shadow_tokens.len(),
            promoted_shadow_by_terminal.iter().map(|(&terminal, tokens)| (terminal, tokens.len())).collect::<Vec<_>>(),
            reused_scoped_ignore_shadow_cells,
            delta_cells,
            dropped_cells,
            conservative_cells,
        );
    }
    factored
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
        .collect::<Vec<_>>();
    NWA::from_parts(states, vec![dfa.start_state])
}

fn install_concrete_boundary_delta_templates(
    templates: &mut Templates,
    plan: &ConcreteBoundaryDeltaPlan,
) {
    for entry in plan.by_global_terminal.values() {
        templates
            .by_terminal
            .insert(entry.old_terminal, entry.old_template.clone());
        templates.by_terminal_nwa.insert(
            entry.old_terminal,
            unweighted_dfa_to_template_nwa(&entry.old_template),
        );
        templates
            .by_terminal
            .insert(entry.delta_terminal, entry.delta_template.clone());
        templates.by_terminal_nwa.insert(
            entry.delta_terminal,
            unweighted_dfa_to_template_nwa(&entry.delta_template),
        );
    }
}

fn profile_current_boundary_template_delta(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    active_terminals: &[bool],
    composed_templates: &Templates,
    boundary_paths: &BoundaryTokenDiscovery,
    tokenizer_state_offsets: &[u32],
) {
    if std::env::var_os("GLRMASK_PROFILE_CURRENT_BOUNDARY_TEMPLATE_DELTA").is_none() {
        return;
    }
    let started = Instant::now();
    let old_templates = rebuild_transported_component_templates(
        composed_table,
        components,
        active_terminals,
    );
    let mut changed = BTreeSet::<u32>::new();
    let mut incomparable = Vec::<u32>::new();
    let mut compared = 0usize;
    let mut old_states = 0usize;
    let mut new_states = 0usize;
    let mut delta_states = 0usize;
    let mut delta_transitions = 0usize;
    let mut minimized_delta_states = 0usize;
    let mut minimized_delta_transitions = 0usize;
    for (&terminal, old) in &old_templates {
        let Some(new) = composed_templates.by_terminal.get(&terminal) else {
            continue;
        };
        compared += 1;
        let removed = unweighted_dfa_difference(old, new);
        if !unweighted_dfa_language_is_empty(&removed) {
            if compose_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][current_boundary_template_removed] terminal={} name={} old_states={} new_states={} witness={:?}",
                    terminal,
                    composed_templates
                        .by_terminal
                        .contains_key(&terminal)
                        .then(|| terminal.to_string())
                        .unwrap_or_default(),
                    old.states.len(),
                    new.states.len(),
                    unweighted_dfa_shortest_word(&removed),
                );
            }
            incomparable.push(terminal);
            continue;
        }
        let delta = trim_unweighted_dfa_productive(unweighted_dfa_difference(new, old));
        if !unweighted_dfa_language_is_empty(&delta) {
            changed.insert(terminal);
            delta_states += delta.states.len();
            delta_transitions += delta.states.iter().map(|state| state.transitions.len()).sum::<usize>();
            let minimized = minimize_unweighted_dfa(&delta);
            minimized_delta_states += minimized.states.len();
            minimized_delta_transitions += minimized
                .states
                .iter()
                .map(|state| state.transitions.len())
                .sum::<usize>();
        }
        old_states += old.states.len();
        new_states += new.states.len();
    }

    let terminal_component = |terminal: u32| -> usize {
        composed_table
            .terminal_offsets
            .partition_point(|&offset| offset <= terminal)
            .saturating_sub(1)
    };
    let state_component = |state: u32| -> Option<usize> {
        if state == 0 {
            None
        } else {
            Some(
                tokenizer_state_offsets
                    .partition_point(|&offset| offset <= state)
                    .saturating_sub(1),
            )
        }
    };
    let mut witnesses_with_cross = 0usize;
    let mut witnesses_with_local = 0usize;
    let mut local_all_one = 0usize;
    let mut local_with_zero = 0usize;
    let mut local_with_one = 0usize;
    let mut local_with_multi = 0usize;
    let mut local_only_all_one = 0usize;
    for witness in &boundary_paths.witnesses {
        let mut reach = vec![BTreeSet::<(Option<usize>, bool, u8)>::new(); witness.nodes.len()];
        for component in witness.start_states.iter().copied().map(state_component) {
            reach[0].insert((component, false, 0));
        }
        let mut order = (0..witness.nodes.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&node| witness.nodes[node].key.offset);
        for source in order {
            if !witness.good[source] || reach[source].is_empty() {
                continue;
            }
            let source_reach = reach[source].clone();
            for edge in witness.nodes[source]
                .outgoing
                .iter()
                .filter(|edge| witness.good[edge.target])
            {
                let next_component = terminal_component(edge.terminal);
                let add = u8::from(changed.contains(&edge.terminal));
                for &(last_component, crossed, count) in &source_reach {
                    let next_crossed = crossed
                        || last_component.is_some_and(|component| component != next_component);
                    reach[edge.target].insert((
                        Some(next_component),
                        next_crossed,
                        count.saturating_add(add).min(2),
                    ));
                }
            }
        }
        let mut local_counts = BTreeSet::<u8>::new();
        let mut has_cross = false;
        for (node, &accepting) in witness.accepting.iter().enumerate() {
            if !accepting {
                continue;
            }
            for &(_, crossed, count) in &reach[node] {
                if crossed {
                    has_cross = true;
                } else {
                    local_counts.insert(count);
                }
            }
        }
        if has_cross {
            witnesses_with_cross += 1;
        }
        if !local_counts.is_empty() {
            witnesses_with_local += 1;
            local_with_zero += usize::from(local_counts.contains(&0));
            local_with_one += usize::from(local_counts.contains(&1));
            local_with_multi += usize::from(local_counts.contains(&2));
            if local_counts.len() == 1 && local_counts.contains(&1) {
                local_all_one += 1;
                if !has_cross {
                    local_only_all_one += 1;
                }
            }
        }
    }
    eprintln!(
        "[glrmask/profile][current_boundary_template_delta] compared={} old_templates={} changed={} incomparable={} old_states={} new_states={} delta_states={} delta_transitions={} minimized_delta_states={} minimized_delta_transitions={} witnesses={} witnesses_with_cross={} witnesses_with_local={} local_all_one={} local_only_all_one={} local_with_zero={} local_with_one={} local_with_multi={} changed_ids={:?} incomparable_ids={:?} total_ms={:.3}",
        compared,
        old_templates.len(),
        changed.len(),
        incomparable.len(),
        old_states,
        new_states,
        delta_states,
        delta_transitions,
        minimized_delta_states,
        minimized_delta_transitions,
        boundary_paths.witnesses.len(),
        witnesses_with_cross,
        witnesses_with_local,
        local_all_one,
        local_only_all_one,
        local_with_zero,
        local_with_one,
        local_with_multi,
        changed,
        incomparable,
        started.elapsed().as_secs_f64() * 1000.0,
    );
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
    for (&terminal, dfa) in &templates.by_terminal {
        let commit_dfa = specialize_template_dfa_defaults_for_commit_split_input(dfa);
        if let Some(split) = try_split_commit_template_dfas(&commit_dfa)
            && let Some(slot) = template_dfas_by_terminal.get_mut(terminal as usize)
        {
            *slot = Some(Arc::new(split));
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


/// Exact weighted determinization of shared lazy parser-stack predicates.
///
/// The lazy arena's DEFAULT read is an additive wildcard. For each output row
/// we therefore compute every concrete explicit derivative with wildcard
/// branches included, and a separate wildcard-only derivative used as the
/// runtime DWA DEFAULT fallback. This converts symbolic additive DEFAULT
/// semantics to ordinary DWA fallback semantics without materializing a parser
/// NWA or normalizing each supported root separately.
/// Partition weighted parser-domain roots into exact disjoint support atoms.
///
/// Each input Weight is a set of `(TSID, token)` points. For every TSID we
/// sweep token-range endpoints and intern the exact set of parser-domain roots
/// active on each token interval. Points with the same membership signature
/// are accumulated into one Weight. The resulting Weights are pairwise
/// disjoint, so later prefix-finality subtraction never has to split an atom.
fn atomize_weighted_parser_roots(
    arena: &mut SharedBooleanParserDomains,
    roots: &[(u32, Weight)],
) -> Vec<(u32, Weight)> {
    let started_at = Instant::now();
    let mut by_tsid = BTreeMap::<u32, Vec<(u32, SharedTokenSet)>>::new();
    for (root, weight) in roots {
        if weight.is_empty() {
            continue;
        }
        if weight.is_full() {
            // Boundary support is vocabulary/TSID bounded. A full sentinel
            // would require an explicit finite universe to atomize, so leave
            // that rare shape to the ordinary weighted combiner.
            return roots.to_vec();
        }
        for (range, tokens) in weight.raw_range_values() {
            for tsid in *range.start()..=*range.end() {
                by_tsid
                    .entry(tsid)
                    .or_default()
                    .push((*root, Arc::clone(tokens)));
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct TokenEvent {
        pos: u32,
        root: u32,
        add: bool,
    }

    let mut signature_ids = FxHashMap::<Vec<u32>, usize>::default();
    let mut signatures = Vec::<Vec<u32>>::new();
    let mut entries_by_signature = Vec::<Vec<(u32, RangeSetBlaze<u32>)>>::new();
    let mut token_events_total = 0usize;
    let mut token_segments_total = 0usize;

    for (tsid, root_sets) in by_tsid {
        let mut events = Vec::<TokenEvent>::new();
        for (root, tokens) in root_sets {
            for range in tokens.ranges() {
                events.push(TokenEvent {
                    pos: *range.start(),
                    root,
                    add: true,
                });
                if let Some(pos) = range.end().checked_add(1) {
                    events.push(TokenEvent {
                        pos,
                        root,
                        add: false,
                    });
                }
            }
        }
        token_events_total += events.len();
        events.sort_unstable_by_key(|event| (event.pos, event.add, event.root));
        if events.is_empty() {
            continue;
        }

        let mut active = BTreeSet::<u32>::new();
        let mut ranges_by_signature = BTreeMap::<usize, Vec<std::ops::RangeInclusive<u32>>>::new();
        let mut cursor = events[0].pos;
        let mut index = 0usize;
        while index < events.len() {
            let pos = events[index].pos;
            if cursor < pos && !active.is_empty() {
                let signature = active.iter().copied().collect::<Vec<_>>();
                let signature_id = if let Some(&existing) = signature_ids.get(&signature) {
                    existing
                } else {
                    let id = signatures.len();
                    signatures.push(signature.clone());
                    signature_ids.insert(signature, id);
                    entries_by_signature.push(Vec::new());
                    id
                };
                ranges_by_signature
                    .entry(signature_id)
                    .or_default()
                    .push(cursor..=pos - 1);
                token_segments_total += 1;
            }

            let bucket_start = index;
            while index < events.len() && events[index].pos == pos {
                index += 1;
            }
            for event in &events[bucket_start..index] {
                if !event.add {
                    active.remove(&event.root);
                }
            }
            for event in &events[bucket_start..index] {
                if event.add {
                    active.insert(event.root);
                }
            }
            cursor = pos;
        }
        debug_assert!(active.is_empty());

        for (signature_id, ranges) in ranges_by_signature {
            let tokens = RangeSetBlaze::from_iter(ranges);
            if !tokens.is_empty() {
                entries_by_signature[signature_id].push((tsid, tokens));
            }
        }
    }

    let raw_atoms = signatures.len();
    let atoms = signatures
        .into_iter()
        .zip(entries_by_signature)
        .filter_map(|(signature, entries)| {
            let domain = arena.union_all(signature);
            if domain == SharedBooleanParserDomains::EMPTY {
                return None;
            }
            let weight = Weight::from_per_tsid_token_sets(entries);
            (!weight.is_empty()).then_some((domain, weight))
        })
        .collect::<Vec<_>>();
    let unique_domains = atoms
        .iter()
        .map(|(domain, _)| *domain)
        .collect::<FxHashSet<_>>()
        .len();
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_domain_atomize] input_roots={} raw_atoms={} output_atoms={} unique_domains={} token_events={} token_segments={} total_ms={:.3}",
            roots.len(),
            raw_atoms,
            atoms.len(),
            unique_domains,
            token_events_total,
            token_segments_total,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    atoms
}

/// Determinize parser-domain support after exact `(TSID, token)` atomization.
/// Atom weights are pairwise disjoint and never change. A state therefore needs
/// only `(atom_id, parser-domain-root)` pairs; prefix finality drops complete
/// atoms instead of computing Weight differences. Concrete edge/final weights
/// are materialized once per distinct atom-id support set.
fn combine_disjoint_weighted_shared_parser_atoms(
    arena: &mut SharedBooleanParserDomains,
    atoms: &[(u32, Weight)],
) -> DWA {
    let started_at = Instant::now();
    let initial = atoms
        .iter()
        .enumerate()
        .filter_map(|(atom, (root, weight))| {
            (!weight.is_empty() && !arena.is_empty_root(*root))
                .then_some((atom as u32, *root))
        })
        .collect::<Vec<_>>();
    if initial.is_empty() {
        return DWA::new(0, 0);
    }

    let mut states = vec![DWAState::default()];
    let mut lanes_by_state = vec![initial.clone()];
    let mut ids = FxHashMap::<Vec<(u32, u32)>, u32>::default();
    ids.insert(initial, 0);
    let mut queue = VecDeque::from([0u32]);
    let mut support_weights = FxHashMap::<Vec<u32>, Weight>::default();
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut support_hits = 0usize;
    let mut support_misses = 0usize;
    let mut max_lanes = 0usize;

    fn materialize_support(
        atom_ids: Vec<u32>,
        atoms: &[(u32, Weight)],
        cache: &mut FxHashMap<Vec<u32>, Weight>,
        weight_ops: &mut ScopedWeightOpCache,
        hits: &mut usize,
        misses: &mut usize,
    ) -> Weight {
        if atom_ids.is_empty() {
            return Weight::empty();
        }
        if let Some(existing) = cache.get(&atom_ids) {
            *hits += 1;
            return existing.clone();
        }
        *misses += 1;
        let weights = atom_ids
            .iter()
            .map(|&atom| &atoms[atom as usize].1)
            .collect::<Vec<_>>();
        let weight = weight_ops.union_all(weights);
        cache.insert(atom_ids, weight.clone());
        weight
    }

    while let Some(state_id) = queue.pop_front() {
        let lanes = lanes_by_state[state_id as usize].clone();
        max_lanes = max_lanes.max(lanes.len());

        let final_atoms = lanes
            .iter()
            .filter_map(|&(atom, root)| arena.is_universal_root(root).then_some(atom))
            .collect::<Vec<_>>();
        let final_weight = materialize_support(
            final_atoms,
            atoms,
            &mut support_weights,
            &mut weight_ops,
            &mut support_hits,
            &mut support_misses,
        );
        if !final_weight.is_empty() {
            states[state_id as usize].final_weight = Some(final_weight);
        }

        // Atomic support makes prefix normalization exact without any Weight
        // subtraction: an atom is either accepted by its whole current domain
        // or not accepted at all.
        let live = lanes
            .into_iter()
            .filter(|&(_, root)| !arena.is_universal_root(root))
            .collect::<Vec<_>>();
        if live.is_empty() {
            continue;
        }

        let mut default_next = Vec::<(u32, u32)>::new();
        let mut overrides_by_label = BTreeMap::<i32, Vec<(u32, u32)>>::new();
        for &(atom, root) in &live {
            let default = arena.advance_default(root);
            if !arena.is_empty_root(default) {
                default_next.push((atom, default));
            }
            for &(label, derivative) in arena.explicit_derivatives(root).iter() {
                debug_assert!(label >= 0 && label != DEFAULT_LABEL);
                overrides_by_label
                    .entry(label)
                    .or_default()
                    .push((atom, derivative));
            }
        }

        let mut emit = |label: i32,
                        next: Vec<(u32, u32)>,
                        states: &mut Vec<DWAState>,
                        lanes_by_state: &mut Vec<Vec<(u32, u32)>>,
                        ids: &mut FxHashMap<Vec<(u32, u32)>, u32>,
                        queue: &mut VecDeque<u32>| {
            if next.is_empty() {
                return;
            }
            let atom_ids = next.iter().map(|&(atom, _)| atom).collect::<Vec<_>>();
            let edge_weight = materialize_support(
                atom_ids,
                atoms,
                &mut support_weights,
                &mut weight_ops,
                &mut support_hits,
                &mut support_misses,
            );
            if edge_weight.is_empty() {
                return;
            }
            let target = if let Some(&target) = ids.get(&next) {
                target
            } else {
                let target = states.len() as u32;
                states.push(DWAState::default());
                lanes_by_state.push(next.clone());
                ids.insert(next, target);
                queue.push_back(target);
                target
            };
            states[state_id as usize]
                .transitions
                .insert(label, (target, edge_weight));
        };

        emit(
            DEFAULT_LABEL,
            default_next.clone(),
            &mut states,
            &mut lanes_by_state,
            &mut ids,
            &mut queue,
        );

        // A concrete derivative differs from DEFAULT only for atoms listed in
        // this label's sparse override row. Merge the sorted override atoms into
        // the sorted DEFAULT lane vector instead of rescanning every live atom.
        for (label, overrides) in overrides_by_label {
            let mut next = Vec::with_capacity(default_next.len() + overrides.len());
            let mut default_index = 0usize;
            let mut override_index = 0usize;
            while default_index < default_next.len() || override_index < overrides.len() {
                match (
                    default_next.get(default_index),
                    overrides.get(override_index),
                ) {
                    (Some(&(default_atom, default_root)), Some(&(override_atom, override_root))) => {
                        if default_atom < override_atom {
                            next.push((default_atom, default_root));
                            default_index += 1;
                        } else if override_atom < default_atom {
                            next.push((override_atom, override_root));
                            override_index += 1;
                        } else {
                            next.push((override_atom, override_root));
                            default_index += 1;
                            override_index += 1;
                        }
                    }
                    (Some(&lane), None) => {
                        next.push(lane);
                        default_index += 1;
                    }
                    (None, Some(&lane)) => {
                        next.push(lane);
                        override_index += 1;
                    }
                    (None, None) => break,
                }
            }
            emit(
                label,
                next,
                &mut states,
                &mut lanes_by_state,
                &mut ids,
                &mut queue,
            );
        }
    }

    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_domain_atom_determinize] atoms={} states={} transitions={} max_lanes={} derivative_rows_cached={} support_cache_entries={} support_hits={} support_misses={} total_ms={:.3}",
            atoms.len(),
            states.len(),
            states.iter().map(|state| state.transitions.len()).sum::<usize>(),
            max_lanes,
            arena.derivative_row_cache_len(),
            support_weights.len(),
            support_hits,
            support_misses,
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    DWA::from_parts(states, 0)
}

fn combine_weighted_shared_parser_roots(
    arena: &mut SharedBooleanParserDomains,
    roots: &[(u32, Weight)],
) -> DWA {
    #[derive(Clone)]
    struct Lane {
        root: u32,
        weight: Weight,
    }

    fn canonicalize_lanes(
        lanes: impl IntoIterator<Item = Lane>,
        weight_ops: &mut ScopedWeightOpCache,
    ) -> Vec<Lane> {
        let mut grouped = BTreeMap::<u32, Vec<Weight>>::new();
        for lane in lanes {
            if !lane.weight.is_empty() {
                grouped.entry(lane.root).or_default().push(lane.weight);
            }
        }
        grouped
            .into_iter()
            .filter_map(|(root, weights)| {
                let weight = weight_ops.union_all(weights.iter());
                (!weight.is_empty()).then_some(Lane { root, weight })
            })
            .collect()
    }

    fn lane_key(lanes: &[Lane]) -> Vec<(u32, usize)> {
        lanes
            .iter()
            .map(|lane| (lane.root, lane.weight.ptr_key()))
            .collect()
    }

    fn union_lane_weights(
        lanes: &[Lane],
        weight_ops: &mut ScopedWeightOpCache,
    ) -> Weight {
        weight_ops.union_all(lanes.iter().map(|lane| &lane.weight))
    }

    let mut weight_ops = ScopedWeightOpCache::default();
    let keep_final_lanes =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DOMAIN_KEEP_FINAL_LANES").is_some();
    let initial = canonicalize_lanes(roots.iter().filter_map(|(root, weight)| {
        (!weight.is_empty() && !arena.is_empty_root(*root)).then(|| Lane {
            root: *root,
            weight: weight.clone(),
        })
    }), &mut weight_ops);
    if initial.is_empty() {
        return DWA::new(0, 0);
    }

    let mut states = vec![DWAState::default()];
    let mut lanes_by_state = vec![initial.clone()];
    let mut ids = FxHashMap::<Vec<(u32, usize)>, u32>::default();
    ids.insert(lane_key(&initial), 0);
    let mut queue = VecDeque::from([0u32]);
    let mut labels_by_root = FxHashMap::<u32, Arc<Vec<i32>>>::default();

    while let Some(state_id) = queue.pop_front() {
        let lanes = lanes_by_state[state_id as usize].clone();
        let mut final_parts = Vec::<Weight>::new();
        for lane in &lanes {
            if arena.is_universal_root(lane.root) {
                final_parts.push(lane.weight.clone());
            }
        }
        let final_weight = weight_ops.union_all(final_parts.iter());
        if !final_weight.is_empty() {
            states[state_id as usize].final_weight = Some(final_weight.clone());
        }

        let live = if keep_final_lanes {
            lanes
        } else {
            lanes
                .into_iter()
                .filter_map(|lane| {
                    let residual = weight_ops.difference(&lane.weight, &final_weight);
                    (!residual.is_empty()).then_some(Lane {
                        root: lane.root,
                        weight: residual,
                    })
                })
                .collect::<Vec<_>>()
        };
        if live.is_empty() {
            continue;
        }

        let mut explicit_labels = BTreeSet::<i32>::new();
        for lane in &live {
            let labels = labels_by_root
                .entry(lane.root)
                .or_insert_with(|| Arc::new(arena.explicit_labels(lane.root)));
            explicit_labels.extend(labels.iter().copied());
        }

        let mut emit = |label: i32,
                        next: Vec<Lane>,
                        states: &mut Vec<DWAState>,
                        lanes_by_state: &mut Vec<Vec<Lane>>,
                        ids: &mut FxHashMap<Vec<(u32, usize)>, u32>,
                        queue: &mut VecDeque<u32>| {
            let next = canonicalize_lanes(next, &mut weight_ops);
            if next.is_empty() {
                return;
            }
            let edge_weight = union_lane_weights(&next, &mut weight_ops);
            if edge_weight.is_empty() {
                return;
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
        };

        // Runtime DEFAULT fallback: derivative contributed only by symbolic
        // wildcard reads, for every positive parser-state label not explicitly
        // represented below.
        let default_next = live
            .iter()
            .filter_map(|lane| {
                let root = arena.advance_default(lane.root);
                (!arena.is_empty_root(root)).then(|| Lane {
                    root,
                    weight: lane.weight.clone(),
                })
            })
            .collect::<Vec<_>>();
        emit(
            DEFAULT_LABEL,
            default_next,
            &mut states,
            &mut lanes_by_state,
            &mut ids,
            &mut queue,
        );

        for label in explicit_labels {
            debug_assert!(label >= 0 && label != DEFAULT_LABEL);
            let next = live
                .iter()
                .filter_map(|lane| {
                    let root = arena.advance(lane.root, label as u32);
                    (!arena.is_empty_root(root)).then(|| Lane {
                        root,
                        weight: lane.weight.clone(),
                    })
                })
                .collect::<Vec<_>>();
            emit(
                label,
                next,
                &mut states,
                &mut lanes_by_state,
                &mut ids,
                &mut queue,
            );
        }
    }

    DWA::from_parts(states, 0)
}

fn profile_direct_boundary_terminal_dwa_domain_dp(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    terminal_dwa: &DWA,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_BOUNDARY_DOMAIN_DP").is_none()
        || !terminal_dwa.is_acyclic()
    {
        return;
    }
    let total_started_at = Instant::now();
    let n = terminal_dwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in terminal_dwa.states() {
        for &(target, _) in state.transitions.values() {
            indegree[target as usize] += 1;
        }
    }
    let mut topo_queue = VecDeque::new();
    for (state, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            topo_queue.push_back(state as u32);
        }
    }
    let mut topo = Vec::with_capacity(n);
    while let Some(source) = topo_queue.pop_front() {
        topo.push(source);
        for &(target, _) in terminal_dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                topo_queue.push_back(target);
            }
        }
    }
    assert_eq!(topo.len(), n);

    #[derive(Clone)]
    struct Group {
        target: u32,
        weight: Weight,
        terminals: Vec<u32>,
    }
    let mut groups_by_state = Vec::<Vec<Group>>::with_capacity(n);
    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for state in terminal_dwa.states() {
        let mut groups = BTreeMap::<(u32, usize), (Weight, Vec<u32>)>::new();
        for (&label, (target, weight)) in &state.transitions {
            if label < 0 {
                eprintln!("[glrmask/profile][direct_boundary_domain_dp] skipped=true reason=negative_terminal_label");
                return;
            }
            groups
                .entry((*target, weight.ptr_key()))
                .or_insert_with(|| (weight.clone(), Vec::new()))
                .1
                .push(label as u32);
        }
        let groups = groups
            .into_iter()
            .map(|((target, _), (weight, mut terminals))| {
                terminals.sort_unstable();
                terminals.dedup();
                terminal_sets.insert(terminals.clone());
                Group { target, weight, terminals }
            })
            .collect::<Vec<_>>();
        groups_by_state.push(groups);
    }
    let bundle_started_at = Instant::now();
    let bundles = terminal_sets
        .par_iter()
        .filter_map(|terminals| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
        })
        .collect::<FxHashMap<_, _>>();
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;
    if bundles.len() != terminal_sets.len() {
        eprintln!(
            "[glrmask/profile][direct_boundary_domain_dp] skipped=true reason=missing_bundle built={} requested={}",
            bundles.len(),
            terminal_sets.len(),
        );
        return;
    }

    type BooleanDomainKey = (u32, Vec<(bool, Vec<(i32, u32, bool)>)>);
    let boolean_domain_key = |domain: &DWA| -> BooleanDomainKey {
        let states = domain
            .states()
            .iter()
            .map(|state| {
                let final_accept = state
                    .final_weight
                    .as_ref()
                    .is_some_and(|weight| !weight.is_empty());
                let transitions = state
                    .transitions
                    .iter()
                    .map(|(&label, &(target, ref weight))| {
                        debug_assert!(weight.is_full() || weight.is_empty());
                        (label, target, !weight.is_empty())
                    })
                    .collect::<Vec<_>>();
                (final_accept, transitions)
            })
            .collect::<Vec<_>>();
        (domain.start_state(), states)
    };

    #[derive(Clone)]
    struct Lane {
        domain: Arc<DWA>,
        support: Weight,
    }
    let universal = Arc::new(universal_parser_stack_domain_dwa());
    let mut domain_interner = FxHashMap::<BooleanDomainKey, Arc<DWA>>::default();
    domain_interner.insert(boolean_domain_key(&universal), Arc::clone(&universal));
    let mut structural_hits = 0usize;
    let mut structural_misses = 1usize;
    let mut domains = vec![FxHashMap::<usize, Lane>::default(); n];
    let mut preimage_cache = FxHashMap::<(Vec<u32>, usize), Arc<DWA>>::default();
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut cache_hits = 0usize;
    let mut cache_misses = 0usize;
    let mut direct_none = 0usize;
    let mut preimage_ms = 0.0f64;
    let mut normalize_ms = 0.0f64;
    let mut concat_ms = 0.0f64;
    let mut max_lanes = 0usize;
    let mut unique_domain_ptrs = FxHashSet::<usize>::default();
    unique_domain_ptrs.insert(Arc::as_ptr(&universal) as usize);

    for &source in topo.iter().rev() {
        let mut lanes = FxHashMap::<usize, Lane>::default();
        if let Some(final_weight) = &terminal_dwa.states()[source as usize].final_weight {
            let ptr = Arc::as_ptr(&universal) as usize;
            lanes.insert(ptr, Lane {
                domain: Arc::clone(&universal),
                support: final_weight.clone(),
            });
        }
        for group in &groups_by_state[source as usize] {
            let bundle = &bundles[&group.terminals];
            for target_lane in domains[group.target as usize].values() {
                let support = weight_ops.intersection(&group.weight, &target_lane.support);
                if support.is_empty() {
                    continue;
                }
                let target_ptr = Arc::as_ptr(&target_lane.domain) as usize;
                let key = (group.terminals.clone(), target_ptr);
                let domain = if let Some(existing) = preimage_cache.get(&key) {
                    cache_hits += 1;
                    Arc::clone(existing)
                } else {
                    cache_misses += 1;
                    let (result, profile) =
                        build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled(
                            table,
                            bundle,
                            &target_lane.domain,
                        );
                    preimage_ms += profile.total_ms;
                    normalize_ms += profile.normalize_ms;
                    concat_ms += profile.concatenate_ms;
                    let Some(result) = result else {
                        direct_none += 1;
                        continue;
                    };
                    let structural_key = boolean_domain_key(&result);
                    let result = if let Some(existing) = domain_interner.get(&structural_key) {
                        structural_hits += 1;
                        Arc::clone(existing)
                    } else {
                        structural_misses += 1;
                        let result = Arc::new(result);
                        domain_interner.insert(structural_key, Arc::clone(&result));
                        result
                    };
                    unique_domain_ptrs.insert(Arc::as_ptr(&result) as usize);
                    preimage_cache.insert(key, Arc::clone(&result));
                    result
                };
                let ptr = Arc::as_ptr(&domain) as usize;
                if let Some(existing) = lanes.get_mut(&ptr) {
                    existing.support = weight_ops.union(&existing.support, &support);
                } else {
                    lanes.insert(ptr, Lane { domain, support });
                }
            }
        }
        max_lanes = max_lanes.max(lanes.len());
        domains[source as usize] = lanes;
    }

    let start = &domains[terminal_dwa.start_state() as usize];
    let unique_domain_states = preimage_cache
        .values()
        .map(|domain| domain.num_states() as usize)
        .sum::<usize>()
        + universal.num_states() as usize;
    let unique_domain_transitions = preimage_cache
        .values()
        .map(|domain| domain.num_transitions())
        .sum::<usize>()
        + universal.num_transitions();
    let start_domain_states = start
        .values()
        .map(|lane| lane.domain.num_states() as usize)
        .sum::<usize>();
    let start_domain_transitions = start
        .values()
        .map(|lane| lane.domain.num_transitions())
        .sum::<usize>();

    if std::env::var_os("GLRMASK_PROFILE_DIRECT_BOUNDARY_DOMAIN_FLATTEN").is_some() {
        let flatten_started_at = Instant::now();
        let mut arena = NWA::new(0, 0);
        let global_start = arena.add_state();
        arena.set_start_states(vec![global_start]);
        let mut appended_states = 1usize;
        for lane in start.values() {
            let body = arena.append_with_body(&lane.domain.to_nwa());
            appended_states += lane.domain.num_states() as usize;
            for target in body.start_states {
                arena.add_epsilon(global_start, target, lane.support.clone());
            }
        }
        debug_assert_eq!(arena.num_states() as usize, appended_states);
        let append_ms = flatten_started_at.elapsed().as_secs_f64() * 1000.0;
        let normalize_started_at = Instant::now();
        let flattened = normalize_weighted_parser_stack_nwa(table, &arena);
        let normalize_ms = normalize_started_at.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[glrmask/profile][direct_boundary_domain_flatten] start_domains={} input_states={} input_transitions={} output_states={} output_transitions={} append_ms={append_ms:.3} normalize_ms={normalize_ms:.3} total_ms={:.3}",
            start.len(),
            arena.num_states(),
            arena.num_transitions(),
            flattened.num_states(),
            flattened.num_transitions(),
            flatten_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    eprintln!(
        "[glrmask/profile][direct_boundary_domain_dp] terminal_states={} terminal_transitions={} groups={} terminal_sets={} start_domains={} max_lanes={} preimage_cache_entries={} cache_hits={} cache_misses={} direct_none={} structural_hits={} structural_misses={} interned_domains={} unique_domain_ptrs={} aggregate_domain_states={} aggregate_domain_transitions={} start_domain_states={} start_domain_transitions={} bundle_ms={bundle_ms:.3} preimage_cpu_ms={preimage_ms:.3} concatenate_cpu_ms={concat_ms:.3} normalize_cpu_ms={normalize_ms:.3} total_ms={:.3}",
        terminal_dwa.num_states(),
        terminal_dwa.num_transitions(),
        groups_by_state.iter().map(Vec::len).sum::<usize>(),
        terminal_sets.len(),
        start.len(),
        max_lanes,
        preimage_cache.len(),
        cache_hits,
        cache_misses,
        direct_none,
        structural_hits,
        structural_misses,
        domain_interner.len(),
        unique_domain_ptrs.len(),
        unique_domain_states,
        unique_domain_transitions,
        start_domain_states,
        start_domain_transitions,
        total_started_at.elapsed().as_secs_f64() * 1000.0,
    );
}

fn validate_lazy_boundary_terminal_dwa_preimages(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    terminal_dwa: &DWA,
) {
    if std::env::var_os("GLRMASK_VALIDATE_LAZY_BOUNDARY_PREIMAGE_DP").is_none()
        || !terminal_dwa.is_acyclic()
    {
        return;
    }
    let started_at = Instant::now();
    let n = terminal_dwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in terminal_dwa.states() {
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
        for &(target, _) in terminal_dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    assert_eq!(topo.len(), n);

    #[derive(Clone)]
    struct Group {
        target: u32,
        weight: Weight,
        terminals: Vec<u32>,
    }
    let mut groups_by_state = Vec::<Vec<Group>>::with_capacity(n);
    let mut terminal_sets = BTreeSet::<Vec<u32>>::new();
    for state in terminal_dwa.states() {
        let mut groups = BTreeMap::<(u32, usize), (Weight, Vec<u32>)>::new();
        for (&label, (target, weight)) in &state.transitions {
            assert!(label >= 0);
            groups
                .entry((*target, weight.ptr_key()))
                .or_insert_with(|| (weight.clone(), Vec::new()))
                .1
                .push(label as u32);
        }
        let groups = groups
            .into_iter()
            .map(|((target, _), (weight, mut terminals))| {
                terminals.sort_unstable();
                terminals.dedup();
                terminal_sets.insert(terminals.clone());
                Group { target, weight, terminals }
            })
            .collect::<Vec<_>>();
        groups_by_state.push(groups);
    }
    let bundles = terminal_sets
        .par_iter()
        .map(|terminals| {
            let bundle = build_boolean_terminal_bundle_nwa(templates, terminals)
                .expect("lazy validation bundle must build");
            (terminals.clone(), Arc::new(bundle))
        })
        .collect::<FxHashMap<_, _>>();

    let mut arena = LazyBooleanParserDomains::new();
    let mut domains = vec![BTreeMap::<u32, Weight>::new(); n];
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
    let mut checked = 0usize;
    let mut cache_hits = 0usize;
    for &source in topo.iter().rev() {
        let mut lanes = BTreeMap::<u32, Weight>::new();
        if let Some(final_weight) = &terminal_dwa.states()[source as usize].final_weight {
            lanes.insert(LazyBooleanParserDomains::UNIVERSAL, final_weight.clone());
        }
        for group in &groups_by_state[source as usize] {
            let bundle = &bundles[&group.terminals];
            for (&target_root, target_weight) in &domains[group.target as usize] {
                let support = weight_ops.intersection(&group.weight, target_weight);
                if support.is_empty() {
                    continue;
                }
                let key = (group.terminals.clone(), target_root);
                let root = if let Some(&root) = preimage_cache.get(&key) {
                    cache_hits += 1;
                    root
                } else {
                    let root = arena
                        .preimage_bundle(bundle, target_root)
                        .expect("lazy preimage must exist");
                    let target_nwa = arena.to_nwa(target_root);
                    let oracle_nwa = build_terminal_bundle_preimage_domain_nwa(
                        table,
                        templates,
                        &group.terminals,
                        &target_nwa,
                    )
                    .expect("oracle preimage must exist");
                    let oracle = normalize_parser_stack_domain_nwa_preserving_explicit(
                        table,
                        &oracle_nwa,
                    );
                    let lazy_nwa = arena.to_nwa(root);
                    let lazy = normalize_parser_stack_domain_nwa_preserving_explicit(
                        table,
                        &lazy_nwa,
                    );
                    let difference = find_difference(&lazy, &oracle)
                        .expect("lazy/oracle parser domains should be acyclic");
                    if let Some(word) = difference.as_ref() {
                        eprintln!(
                            "[glrmask/validate][lazy_boundary_preimage_mismatch] terminals={:?} target_root={} root={} witness={:?} lazy={} oracle={}",
                            group.terminals,
                            target_root,
                            root,
                            word,
                            lazy.eval_word(word),
                            oracle.eval_word(word),
                        );
                        panic!("lazy parser-domain preimage differs from exact NWA oracle");
                    }
                    checked += 1;
                    preimage_cache.insert(key, root);
                    root
                };
                if root == LazyBooleanParserDomains::EMPTY {
                    continue;
                }
                if let Some(existing) = lanes.get_mut(&root) {
                    *existing = weight_ops.union(existing, &support);
                } else {
                    lanes.insert(root, support);
                }
            }
        }
        domains[source as usize] = lanes;
    }
    eprintln!(
        "[glrmask/validate][lazy_boundary_preimage_dp] exact=true checked={} cache_hits={} expr_nodes={} start_roots={} total_ms={:.3}",
        checked,
        cache_hits,
        arena.node_count(),
        domains[terminal_dwa.start_state() as usize].len(),
        started_at.elapsed().as_secs_f64() * 1000.0,
    );
}

fn build_boundary_parser_from_weighted_terminal_dwa(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    terminal_dwa: &DWA,
) -> Option<DWA> {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_TERMINAL_DWA_DOMAIN_DP").is_none() {
        return None;
    }
    if !terminal_dwa.is_acyclic() {
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_boundary_terminal_domain_dp] skipped=true reason=cyclic terminal_states={}",
                terminal_dwa.num_states(),
            );
        }
        return None;
    }

    let total_started_at = Instant::now();
    let n = terminal_dwa.states().len();
    let mut indegree = vec![0usize; n];
    for state in terminal_dwa.states() {
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
        for &(target, _) in terminal_dwa.states()[source as usize].transitions.values() {
            indegree[target as usize] -= 1;
            if indegree[target as usize] == 0 {
                queue.push_back(target);
            }
        }
    }
    debug_assert_eq!(topo.len(), n);

    #[derive(Clone)]
    struct TerminalGroup {
        target: u32,
        edge_weight: Weight,
        terminals: Vec<u32>,
        skip_allowed_states: Option<Arc<Vec<u32>>>,
    }
    let use_direct_skip =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DIRECT_SKIP_PREIMAGE").is_some();
    let mut seen_terminal = vec![false; table.num_terminals as usize];
    let mut non_skip_terminal = vec![false; table.num_terminals as usize];
    let mut skip_states_by_terminal = vec![Vec::<u32>::new(); table.num_terminals as usize];
    if use_direct_skip {
        for (source, row) in table.action.iter().enumerate() {
            for (terminal, action) in row {
                let Some(seen) = seen_terminal.get_mut(terminal as usize) else {
                    continue;
                };
                *seen = true;
                match action {
                    Action::Skip => skip_states_by_terminal[terminal as usize].push(source as u32),
                    _ => non_skip_terminal[terminal as usize] = true,
                }
            }
        }
    }
    let pure_skip = (0..table.num_terminals as usize)
        .map(|terminal| {
            use_direct_skip && seen_terminal[terminal] && !non_skip_terminal[terminal]
        })
        .collect::<Vec<_>>();

    let group_started_at = Instant::now();
    let mut groups_by_state = Vec::<Vec<TerminalGroup>>::with_capacity(n);
    let mut all_terminal_sets = BTreeSet::<Vec<u32>>::new();
    let mut skip_group_count = 0usize;
    for state in terminal_dwa.states() {
        let mut groups = BTreeMap::<(u32, usize, bool), (Weight, Vec<u32>)>::new();
        for (&label, (target, weight)) in &state.transitions {
            if label < 0 {
                return None;
            }
            let terminal = label as u32;
            let is_skip = pure_skip.get(terminal as usize).copied().unwrap_or(false);
            let entry = groups
                .entry((*target, weight.ptr_key(), is_skip))
                .or_insert_with(|| (weight.clone(), Vec::new()));
            entry.1.push(terminal);
        }
        let groups = groups
            .into_iter()
            .map(|((target, _, is_skip), (edge_weight, mut terminals))| {
                terminals.sort_unstable();
                terminals.dedup();
                let skip_allowed_states = if is_skip {
                    skip_group_count += 1;
                    let mut allowed = terminals
                        .iter()
                        .flat_map(|&terminal| {
                            skip_states_by_terminal[terminal as usize].iter().copied()
                        })
                        .collect::<Vec<_>>();
                    allowed.sort_unstable();
                    allowed.dedup();
                    Some(Arc::new(allowed))
                } else {
                    all_terminal_sets.insert(terminals.clone());
                    None
                };
                TerminalGroup {
                    target,
                    edge_weight,
                    terminals,
                    skip_allowed_states,
                }
            })
            .collect::<Vec<_>>();
        groups_by_state.push(groups);
    }
    let groups_ms = group_started_at.elapsed().as_secs_f64() * 1000.0;

    let bundle_started_at = Instant::now();
    let bundles = all_terminal_sets
        .par_iter()
        .filter_map(|terminals| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
        })
        .collect::<FxHashMap<_, _>>();
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;
    if bundles.len() != all_terminal_sets.len() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminal_domain_dp] skipped=true reason=missing_bundle built={} requested={}",
            bundles.len(),
            all_terminal_sets.len(),
        );
        return None;
    }

    let dp_started_at = Instant::now();
    let mut arena = SharedBooleanParserDomains::new();
    let mut domains = vec![BTreeMap::<u32, Weight>::new(); n];
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
    let mut preimage_calls = 0usize;
    let mut preimage_cache_hits = 0usize;
    let mut max_lanes = 0usize;
    let mut weight_ops = ScopedWeightOpCache::default();

    fn merge_lane(
        lanes: &mut BTreeMap<u32, Weight>,
        root: u32,
        weight: Weight,
        weight_ops: &mut ScopedWeightOpCache,
    ) {
        if weight.is_empty() {
            return;
        }
        if let Some(existing) = lanes.get_mut(&root) {
            *existing = weight_ops.union(existing, &weight);
        } else {
            lanes.insert(root, weight);
        }
    }

    for &source in topo.iter().rev() {
        let source_index = source as usize;
        let state = &terminal_dwa.states()[source_index];
        let mut lanes = BTreeMap::<u32, Weight>::new();
        if let Some(final_weight) = state.final_weight.as_ref() {
            merge_lane(
                &mut lanes,
                SharedBooleanParserDomains::UNIVERSAL,
                final_weight.clone(),
                &mut weight_ops,
            );
        }
        for group in &groups_by_state[source_index] {
            let target_lanes = &domains[group.target as usize];
            if target_lanes.is_empty() {
                continue;
            }
            let bundle = group
                .skip_allowed_states
                .is_none()
                .then(|| {
                    bundles
                        .get(&group.terminals)
                        .expect("every non-skip terminal-DWA group bundle must be prebuilt")
                });
            for (&target_root, target_support) in target_lanes {
                let support = weight_ops.intersection(&group.edge_weight, target_support);
                if support.is_empty() {
                    continue;
                }
                let key = (group.terminals.clone(), target_root);
                let source_root = if let Some(&cached) = preimage_cache.get(&key) {
                    preimage_cache_hits += 1;
                    cached
                } else {
                    let root = if let Some(allowed) = group.skip_allowed_states.as_deref() {
                        arena.preimage_identity_skip(target_root, allowed)
                    } else {
                        arena.preimage_bundle(bundle.expect("non-skip bundle must exist"), target_root)?
                    };
                    if std::env::var_os("GLRMASK_VALIDATE_SHARED_BOUNDARY_PREIMAGE").is_some()
                        && group.skip_allowed_states.is_none()
                    {
                        let target_nwa = arena.to_nwa(target_root);
                        let oracle_nwa = build_terminal_bundle_preimage_domain_nwa(
                            table,
                            templates,
                            &group.terminals,
                            &target_nwa,
                        )
                        .expect("oracle preimage must exist when shared preimage exists");
                        let oracle = normalize_parser_stack_domain_nwa_preserving_explicit(
                            table,
                            &oracle_nwa,
                        );
                        let shared_nwa = arena.to_nwa(root);
                        let shared = normalize_parser_stack_domain_nwa_preserving_explicit(
                            table,
                            &shared_nwa,
                        );
                        let difference = find_difference(&shared, &oracle)
                            .expect("parser-domain preimages should be acyclic");
                        if let Some(word) = difference.as_ref() {
                            let first_derivative = word.first().and_then(|&label| {
                                (label >= 0).then(|| {
                                    let derivative = arena.advance(root, label as u32);
                                    (
                                        derivative,
                                        arena.is_universal_root(derivative),
                                        arena.advance_default(root),
                                        arena.explicit_derivatives(root)
                                            .iter()
                                            .find(|(candidate, _)| *candidate == label)
                                            .copied(),
                                    )
                                })
                            });
                            let bundle_debug = bundle.map(|bundle| {
                                bundle
                                    .states()
                                    .iter()
                                    .enumerate()
                                    .map(|(state, node)| {
                                        (
                                            state,
                                            node.final_weight.as_ref().is_some_and(|w| !w.is_empty()),
                                            node.transitions
                                                .iter()
                                                .map(|(&label, targets)| {
                                                    (label, targets.iter().map(|(target, _)| *target).collect::<Vec<_>>())
                                                })
                                                .collect::<Vec<_>>(),
                                            node.epsilons.iter().map(|(target, _)| *target).collect::<Vec<_>>(),
                                        )
                                    })
                                    .collect::<Vec<_>>()
                            });
                            eprintln!(
                                "[glrmask/validate][shared_boundary_preimage_mismatch] terminals={:?} target_root={} shared_root={} witness={:?} shared={} oracle={} first_derivative={:?} bundle={:?}",
                                group.terminals,
                                target_root,
                                root,
                                word,
                                shared.eval_word(word),
                                oracle.eval_word(word),
                                first_derivative,
                                bundle_debug,
                            );
                            panic!("shared parser-domain preimage differs from exact NWA oracle");
                        }
                    }
                    preimage_cache.insert(key, root);
                    root
                };
                preimage_calls += 1;
                if source_root != SharedBooleanParserDomains::EMPTY {
                    merge_lane(&mut lanes, source_root, support, &mut weight_ops);
                }
            }
        }
        max_lanes = max_lanes.max(lanes.len());
        domains[source_index] = lanes;
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;

    if compose_profile_enabled() {
        let start_outer_ranges = domains[terminal_dwa.start_state() as usize]
            .values()
            .map(|weight| weight.raw_range_values().count())
            .sum::<usize>();
        let start_inner_ranges = domains[terminal_dwa.start_state() as usize]
            .values()
            .flat_map(|weight| weight.raw_range_values().map(|(_, tokens)| tokens.ranges().count()))
            .sum::<usize>();
        let start_tsid_cells = domains[terminal_dwa.start_state() as usize]
            .values()
            .flat_map(|weight| weight.raw_range_values().map(|(range, _)| {
                (*range.end() as usize + 1).saturating_sub(*range.start() as usize)
            }))
            .sum::<usize>();
        let start_inner_range_tsid_cells = domains[terminal_dwa.start_state() as usize]
            .values()
            .flat_map(|weight| weight.raw_range_values().map(|(range, tokens)| {
                let width = (*range.end() as usize + 1).saturating_sub(*range.start() as usize);
                width.saturating_mul(tokens.ranges().count())
            }))
            .sum::<usize>();
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminal_domain_dp_phase] phase=dp terminal_states={} groups={} skip_groups={} terminal_sets={} graph_nodes={} start_roots={} start_outer_ranges={} start_inner_ranges={} start_tsid_cells={} start_inner_range_tsid_cells={} max_state_lanes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} groups_ms={groups_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3}",
            terminal_dwa.num_states(),
            groups_by_state.iter().map(Vec::len).sum::<usize>(),
            skip_group_count,
            all_terminal_sets.len(),
            arena.node_count(),
            domains[terminal_dwa.start_state() as usize].len(),
            start_outer_ranges,
            start_inner_ranges,
            start_tsid_cells,
            start_inner_range_tsid_cells,
            max_lanes,
            preimage_calls,
            preimage_cache_hits,
            preimage_cache.len(),
        );
    }
    if let Ok(spec) = std::env::var("GLRMASK_DEBUG_BOUNDARY_DOMAIN_LANE") {
        let parts = spec
            .split(':')
            .filter_map(|part| part.parse::<u32>().ok())
            .collect::<Vec<_>>();
        if let [tsid, token, parser_state] = parts.as_slice() {
            let mut carrying = Vec::new();
            for (&root, weight) in &domains[terminal_dwa.start_state() as usize] {
                if weight.tokens_for_tsid(*tsid).contains(*token) {
                    let derivative = arena.advance(root, *parser_state);
                    carrying.push((
                        root,
                        derivative,
                        arena.is_universal_root(derivative),
                        arena.advance_default(root),
                        arena.explicit_derivatives(root)
                            .iter()
                            .find(|(label, _)| *label == *parser_state as i32)
                            .copied(),
                    ));
                }
            }
            eprintln!(
                "[glrmask/debug][constraint_boundary_domain_lane] tsid={} token={} parser_state={} carrying_roots={} rows={:?}",
                tsid,
                token,
                parser_state,
                carrying.len(),
                carrying,
            );
        }
    }
    let direct_started_at = Instant::now();
    let root_weights = domains[terminal_dwa.start_state() as usize]
        .iter()
        .map(|(&root, weight)| (root, weight.clone()))
        .collect::<Vec<_>>();
    let use_atoms = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DOMAIN_DISJOINT_ATOMS")
        .is_some();
    let root_weights = if use_atoms
        || std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_DOMAIN_ATOMIZE").is_some()
    {
        atomize_weighted_parser_roots(&mut arena, &root_weights)
    } else {
        root_weights
    };
    let candidate = if use_atoms {
        combine_disjoint_weighted_shared_parser_atoms(&mut arena, &root_weights)
    } else {
        combine_weighted_shared_parser_roots(&mut arena, &root_weights)
    };
    let direct_ms = direct_started_at.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[glrmask/profile][constraint_boundary_terminal_domain_dp] terminal_states={} terminal_transitions={} groups={} skip_groups={} terminal_sets={} graph_nodes={} weighted_start_roots={} max_state_lanes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} output_states={} output_transitions={} groups_ms={groups_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3} direct_ms={direct_ms:.3} total_ms={:.3}",
        terminal_dwa.num_states(),
        terminal_dwa.num_transitions(),
        groups_by_state.iter().map(Vec::len).sum::<usize>(),
        skip_group_count,
        all_terminal_sets.len(),
        arena.node_count(),
        root_weights.len(),
        max_lanes,
        preimage_calls,
        preimage_cache_hits,
        preimage_cache.len(),
        candidate.num_states(),
        candidate.num_transitions(),
        total_started_at.elapsed().as_secs_f64() * 1000.0,
    );
    Some(candidate)
}

fn build_full_boundary_lazy_direct_parser(
    table: &crate::compiler::glr::table::GLRTable,
    templates: &Templates,
    discovery: &BoundaryTokenDiscovery,
    globally_erasable_ignore_terminals: &BitSet,
    component_state_map: &ManyToOneIdMap,
    id_map: &InternalIdMap,
    seed_relations: &BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
) -> Option<DWA> {
    if std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DIRECT_PARSER").is_none() {
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

    let total_started_at = Instant::now();
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
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=canonicalize canonical_nodes={} witnesses={} ms={canonicalize_ms:.3}", canonical_nodes.len(), witness_starts.len());
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
    let bundle_started_at = Instant::now();
    let prebuilt_bundles = terminal_sets
        .par_iter()
        .filter_map(|terminals| {
            build_boolean_terminal_bundle_nwa(templates, terminals)
                .map(|bundle| (terminals.clone(), Arc::new(bundle)))
        })
        .collect::<FxHashMap<_, _>>();
    let bundle_ms = bundle_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=bundles terminal_sets={} built={} ms={bundle_ms:.3}", terminal_sets.len(), prebuilt_bundles.len());
    }
    if prebuilt_bundles.len() != terminal_sets.len() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_lazy_direct_parser] skipped=true reason=missing_bundle built={} requested={}",
            prebuilt_bundles.len(),
            terminal_sets.len(),
        );
        return None;
    }

    let dp_started_at = Instant::now();
    let mut arena = SharedBooleanParserDomains::new();
    let mut roots = vec![SharedBooleanParserDomains::EMPTY; canonical_nodes.len()];
    let mut preimage_calls = 0usize;
    let mut preimage_cache_hits = 0usize;
    let mut preimage_cache = FxHashMap::<(Vec<u32>, u32), u32>::default();
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
                .expect("every lazy-domain terminal bundle must be prebuilt");
            let target_root = roots[target];
            let cache_key = (terminals, target_root);
            let root = if let Some(&cached) = preimage_cache.get(&cache_key) {
                preimage_cache_hits += 1;
                cached
            } else {
                let root = arena.preimage_bundle(bundle, target_root)?;
                preimage_cache.insert(cache_key, root);
                root
            };
            preimage_calls += 1;
            branches.push(root);
        }
        branches.extend(node.epsilons.iter().map(|&target| roots[target]));
        roots[node_id] = arena.union_all(branches);
    }
    let dp_ms = dp_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=dp expr_nodes={} preimage_calls={} cache_hits={} cache_entries={} ms={dp_ms:.3}", arena.node_count(), preimage_calls, preimage_cache_hits, preimage_cache.len());
    }

    // Batch support by lazy root before constructing Weight objects. Thousands
    // of model-token witnesses routinely share one parser predicate; repeated
    // singleton Weight unions would recreate the old boundary-weight pathology.
    let support_started_at = Instant::now();
    let mut tokens_by_root_tsid = BTreeMap::<u32, BTreeMap<u32, BTreeSet<u32>>>::new();
    for (witness, &start) in discovery.witnesses.iter().zip(&witness_starts) {
        let root = roots[start];
        if root == SharedBooleanParserDomains::EMPTY {
            continue;
        }
        let Some(internal_token) = id_map.internal_token_for_original(witness.token_id) else {
            continue;
        };
        let by_tsid = tokens_by_root_tsid.entry(root).or_default();
        for &raw_state in &witness.start_states {
            let Some(&tsid) = component_state_map.original_to_internal.get(raw_state as usize) else {
                continue;
            };
            if tsid != u32::MAX {
                by_tsid.entry(tsid).or_default().insert(internal_token);
            }
        }
    }

    let mut seed_bundle_cache = FxHashMap::<u32, Arc<NWA>>::default();
    for (sequence, by_state) in seed_relations {
        debug_assert_eq!(sequence.len(), 1);
        let terminal = sequence[0];
        let bundle = if let Some(bundle) = seed_bundle_cache.get(&terminal) {
            Arc::clone(bundle)
        } else {
            let bundle = Arc::new(build_boolean_terminal_bundle_nwa(templates, &[terminal])?);
            seed_bundle_cache.insert(terminal, Arc::clone(&bundle));
            bundle
        };
        let root = arena.preimage_bundle(&bundle, SharedBooleanParserDomains::UNIVERSAL)?;
        if root == SharedBooleanParserDomains::EMPTY {
            continue;
        }
        let by_tsid = tokens_by_root_tsid.entry(root).or_default();
        for (&raw_state, originals) in by_state {
            let Some(&tsid) = component_state_map.original_to_internal.get(raw_state as usize) else {
                continue;
            };
            if tsid == u32::MAX {
                continue;
            }
            let tokens = by_tsid.entry(tsid).or_default();
            tokens.extend(
                originals
                    .iter()
                    .filter_map(|&original| id_map.internal_token_for_original(original)),
            );
        }
    }
    let support_ms = support_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_boundary_lazy_phase] phase=support roots={} ms={support_ms:.3}", tokens_by_root_tsid.len());
    }

    let root_started_at = Instant::now();
    let root_weights = tokens_by_root_tsid
        .into_iter()
        .filter_map(|(root, by_tsid)| {
            let weight = Weight::from_per_tsid_token_sets(by_tsid.into_iter().map(
                |(tsid, tokens)| (tsid, tokens.into_iter().collect::<RangeSetBlaze<_>>()),
            ));
            (!weight.is_empty()).then_some((root, weight))
        })
        .collect::<Vec<_>>();
    let candidate = combine_weighted_shared_parser_roots(&mut arena, &root_weights);
    let root_ms = root_started_at.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "[glrmask/profile][constraint_boundary_lazy_direct_parser] canonical_nodes={} witnesses={} terminal_sets={} expr_nodes={} preimage_calls={} preimage_cache_hits={} preimage_cache_entries={} weighted_roots={} dwa_states={} dwa_transitions={} canonicalize_ms={canonicalize_ms:.3} bundle_ms={bundle_ms:.3} dp_ms={dp_ms:.3} support_ms={support_ms:.3} direct_ms={root_ms:.3} total_ms={:.3}",
        canonical_nodes.len(),
        witness_starts.len(),
        terminal_sets.len(),
        arena.node_count(),
        preimage_calls,
        preimage_cache_hits,
        preimage_cache.len(),
        root_weights.len(),
        candidate.num_states(),
        candidate.num_transitions(),
        total_started_at.elapsed().as_secs_f64() * 1000.0,
    );
    Some(candidate)
}

fn build_boundary_repair(
    composed_table: &ComposedTable,
    merged_tokenizer: Option<&Tokenizer>,
    merged_tokenizer_state_count: usize,
    terminal_display_names: Vec<String>,
    ignore_terminals: &MergedIgnoreTerminals,
    vocab: &Vocab,
    special_token_terminals: &[SpecialTokenTerminal],
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    precomputed_component_state_map: Option<&ManyToOneIdMap>,
    deferred_component_state_map: Option<&OnceLock<Result<ManyToOneIdMap, String>>>,
    selected_boundary_tokens: Option<&OnceLock<Result<Option<Vec<u32>>, String>>>,
) -> Result<Option<BoundaryRepair>, String> {
    let total_started_at = Instant::now();
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

    // A transported component parser DWA already covers paths wholly inside
    // that component. Boundary repair is required whenever the explicit
    // linker can take a zero-width call/return before the next lexical
    // terminal. For a child root N, those lexical beginnings are exactly:
    //
    //   FIRST(N)  â€” paths which begin the child; and
    //   FOLLOW(N) â€” paths which begin the parent continuation.
    //
    // Compute this over the fully composed rule graph, rather than reading the
    // direct child-start/continuation rows. FIRST/FOLLOW propagates through
    // nullable suffixes, adjacent children, and children whose first visible
    // syntax belongs to another nested child.
    let mut seed_terminals = vec![false; composed_table.table.num_terminals as usize];
    for &nonterminal in &composed_table.boundary_nonterminals {
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
    let base_interface_pairs = visible_boundary_interface_pairs(
        &analyzed,
        &composed_table.boundary_nonterminals,
        &composed_table.control_terminals,
    );
    let interface_pairs = if std::env::var_os("GLRMASK_EXPERIMENT_BASE_BOUNDARY_INTERFACES_ONLY")
        .is_some()
    {
        base_interface_pairs.clone()
    } else {
        extend_boundary_interfaces_through_stack_neutral_lr_actions(
            &composed_table.table,
            &base_interface_pairs,
        )
    };
    let mut follow_transparent_terminals = BitSet::new(seed_terminals.len());
    if std::env::var_os("GLRMASK_EXPERIMENT_STRICT_BOUNDARY_FOLLOWS").is_none() {
        for &terminal in &composed_table.table.skip_terminals {
            follow_transparent_terminals.set(terminal as usize);
        }
    }
    let mut boundary_context_terminals = BitSet::new(seed_terminals.len());
    for &terminal in &composed_table.table.skip_terminals {
        let participates = seed_terminals
            .get(terminal as usize)
            .copied()
            .unwrap_or(false)
            || interface_pairs
                .iter()
                .any(|&(left, right)| left == terminal || right == terminal);
        if participates {
            boundary_context_terminals.set(terminal as usize);
        }
    }
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_interface_pairs] base_pairs={} extended_pairs={} stack_neutral_terminals={}",
            base_interface_pairs.len(),
            interface_pairs.len(),
            composed_table.table.skip_terminals.len(),
        );
    }
    if compose_profile_enabled() {
        let owner = |terminal: u32| {
            composed_table
                .terminal_offsets
                .partition_point(|&offset| offset <= terminal)
                .saturating_sub(1)
        };
        let mut base_owner_counts = BTreeMap::<(usize, usize), usize>::new();
        for &(left, right) in &base_interface_pairs {
            *base_owner_counts.entry((owner(left), owner(right))).or_default() += 1;
        }
        let mut extended_owner_counts = BTreeMap::<(usize, usize), usize>::new();
        for &(left, right) in &interface_pairs {
            *extended_owner_counts.entry((owner(left), owner(right))).or_default() += 1;
        }
        eprintln!("[glrmask/profile][constraint_boundary_interface_owners] base={base_owner_counts:?} extended={extended_owner_counts:?}");
    }
    // LR-inserted stack-neutral terminals are not global boundary-discovery
    // seeds: that would make every token containing trivia look like a boundary
    // token. They still need their ordinary one-terminal language compiled so a
    // token consisting solely of such a terminal receives the LR `Skip` /
    // state-refining template. Keep that support set separate from FIRST/FOLLOW.
    if compose_profile_enabled() {
        let context = boundary_context_terminals
            .iter()
            .map(|terminal| format!("{}:{}", terminal, analyzed.terminal_display_name(terminal as u32)))
            .collect::<Vec<_>>();
        eprintln!("[glrmask/profile][constraint_boundary_context_terminals] {context:?}");
    }
    let mut one_terminal_support_terminals = seed_terminals.clone();
    for &terminal in &composed_table.table.skip_terminals {
        if let Some(selected) = one_terminal_support_terminals.get_mut(terminal as usize) {
            *selected = true;
        }
    }
    // Lexical discovery does not impose grammar-RHS follow legality. The
    // composed LR table is the authority for parser-visible terminal sequences,
    // including state-dependent identity/scope-refining terminals.
    // Boundary-path discovery and exact one-byte seed analysis are independent
    // read-only passes over the tokenizer.  Running them serially made boundary
    // repair pay two full million-state/vocabulary scans back-to-back.
    let boundary_disallowed_follows =
        std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_GRAMMAR_FOLLOWS")
            .is_some()
            .then(|| crate::compiler::pipeline::compute_disallowed_follows(&analyzed));
    let eager_all_templates =
        std::env::var_os("GLRMASK_COMPOSE_SELECTED_TEMPLATES_ONLY").is_none();
    let all_terminals = vec![true; analyzed.num_terminals as usize];
    let (eager_templates, ((boundary_paths, discovery_ms), (seed_relations, one_byte_ms))) =
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
                        let boundary_paths = discover_boundary_token_paths(
                            vocab,
                            components,
                            tokenizer_state_offsets,
                            &composed_table.terminal_offsets,
                            &seed_terminals,
                            &ignore_terminals.global,
                            &interface_pairs,
                            &boundary_context_terminals,
                            &follow_transparent_terminals,
                            boundary_disallowed_follows.as_ref(),
                        );
                        (boundary_paths, started_at.elapsed().as_secs_f64() * 1000.0)
                    },
                    || {
                        let started_at = Instant::now();
                        let relations = collect_one_byte_seed_relations_components(
                            components,
                            tokenizer_state_offsets,
                            &composed_table.terminal_offsets,
                            vocab,
                            &one_terminal_support_terminals,
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
                                &one_terminal_support_terminals,
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
    let discovered_boundary_terminals = boundary_paths.terminals.clone();
    let mut active_terminals = one_terminal_support_terminals.clone();
    for terminal in discovered_boundary_terminals.iter() {
        active_terminals[terminal] = true;
    }
    for &terminal in &composed_table.control_terminals {
        if let Some(active) = active_terminals.get_mut(terminal as usize) {
            *active = true;
        }
    }
    // Reset-complete one-terminal model tokens must be part of the published
    // boundary token coordinate before the owned path prepares its token map.
    // Discover them against the active terminal superset here; the concrete
    // delta factor below will later drop unchanged compared terminals.
    let mut seed_relations = seed_relations;
    merge_one_terminal_relations(
        &mut seed_relations,
        boundary_delta_reset_relations(
            components,
            &composed_table.terminal_offsets,
            vocab,
            &active_terminals,
            &composed_table.control_terminals,
        ),
    );

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
    if !active_terminals.iter().any(|&active| active) {
        if let Some(selected_boundary_tokens) = selected_boundary_tokens {
            let _ = selected_boundary_tokens.set(Ok(None));
        }
        return Ok(None);
    }
    if compose_profile_enabled() {
        let selected_count = active_terminals.iter().filter(|&&active| active).count();
        eprintln!(
            "[glrmask/profile][constraint_boundary_terminals] begin={} one_terminal_support={} discovered={} boundary_tokens={} active={}",
            seed_terminals.iter().filter(|&&selected| selected).count(),
            one_terminal_support_terminals.iter().filter(|&&selected| selected).count(),
            discovered_boundary_terminals.count_ones(),
            boundary_paths.token_ids.len(),
            selected_count,
        );
        if std::env::var_os("GLRMASK_PROFILE_COMPOSE_VERBOSE").is_some() {
            let selected = active_terminals
                .iter()
                .enumerate()
                .filter_map(|(terminal, &active)| {
                    active.then(|| format!("{}:{}", terminal, analyzed.terminal_display_name(terminal as u32)))
                })
                .collect::<Vec<_>>();
            eprintln!("[glrmask/profile][constraint_boundary_terminal_names] selected={selected:?}");
        }
    }

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
        let _ = selected_boundary_tokens.set(Ok(Some(selected_original_tokens.clone())));
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

    let use_concrete_delta =
        std::env::var_os("GLRMASK_DISABLE_CONCRETE_BOUNDARY_TEMPLATE_DELTA").is_none()
            && std::env::var_os("GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE").is_none();
    let lazy_seed_relations = std::env::var_os("GLRMASK_EXPERIMENT_BOUNDARY_LAZY_DIRECT_PARSER")
        .is_some()
        .then(|| seed_relations.clone());

    let ((templates, template_dfas_by_terminal, templates_ms), terminal_dwa, concrete_delta_plan) =
        if use_concrete_delta {
            // Delta construction depends on the composed templates, so this
            // concrete-delta path intentionally serializes template construction
            // before the boundary terminal graph. The ordinary path retains
            // the existing rayon overlap below.
            let (mut templates, mut template_dfas_by_terminal, templates_ms) =
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
            let plan = prepare_concrete_boundary_delta_plan(
                composed_table,
                components,
                &active_terminals,
                &templates,
                analyzed.num_terminals,
            );
            install_concrete_boundary_delta_templates(&mut templates, &plan);
            let seed_relations = if std::env::var_os(
                "GLRMASK_DISABLE_FACTOR_ONE_TERMINAL_SEEDS",
            )
            .is_some()
            {
                seed_relations
            } else {
                factor_one_terminal_seed_relations(
                    seed_relations,
                    &plan,
                    components,
                    tokenizer_state_offsets,
                    &composed_table.terminal_offsets,
                )
            };
            let seed_relations = if std::env::var_os("GLRMASK_EXPERIMENT_DROP_ONE_TERMINAL_BOUNDARY").is_some() {
                BTreeMap::new()
            } else {
                seed_relations
            };
            let started_at = Instant::now();
            let result = direct_boundary_terminal_automaton(
                merged_tokenizer_state_count,
                Some(component_state_map),
                vocab,
                &selected_original_tokens,
                seed_relations,
                one_byte_ms,
                &boundary_paths,
                &ignore_terminals.global,
                &composed_table.control_terminals,
                &composed_table.terminal_offsets,
                tokenizer_state_offsets,
                Some(&plan),
            );
            (
                (templates, template_dfas_by_terminal, templates_ms),
                (result, started_at.elapsed().as_secs_f64() * 1000.0),
                Some(plan),
            )
        } else {
            let (template_result, terminal_result) = rayon::join(
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
                                            &BTreeMap::new(),
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
                            &selected_original_tokens,
                            seed_relations,
                            one_byte_ms,
                            &boundary_paths,
                            &ignore_terminals.global,
                            &composed_table.control_terminals,
                            &composed_table.terminal_offsets,
                            tokenizer_state_offsets,
                            None,
                        )
                    };
                    (result, started_at.elapsed().as_secs_f64() * 1000.0)
                },
            );
            (template_result, terminal_result, None)
        };
    let (terminal_dwa, terminal_ms) = terminal_dwa;
    let profile_delta_selected = if std::env::var_os(
        "GLRMASK_PROFILE_BOUNDARY_SEED_TEMPLATE_DELTA",
    )
    .is_some()
    {
        &seed_terminals
    } else {
        &active_terminals
    };
    profile_current_boundary_template_delta(
        composed_table,
        components,
        profile_delta_selected,
        &templates,
        &boundary_paths,
        tokenizer_state_offsets,
    );
    let terminal_dwa = terminal_dwa?;
    let special_source_state = merged_tokenizer
        .map(Tokenizer::initial_state_id)
        .unwrap_or(0);
    let terminal_dwa = add_boundary_special_token_paths(
        terminal_dwa,
        &boundary_special_token_terminals,
        special_source_state,
        Some(component_state_map),
        &composed_table.control_terminals,
    )?;

    let parser_started_at = Instant::now();
    let (terminal_automaton, id_map) = terminal_dwa.into_parts();
    if let TerminalAutomaton::Dwa(dwa) = &terminal_automaton {
        profile_direct_boundary_terminal_dwa_domain_dp(
            &composed_table.table,
            &templates,
            dwa,
        );
        validate_lazy_boundary_terminal_dwa_preimages(
            &composed_table.table,
            &templates,
            dwa,
        );
    }
    let terminal_domain_candidate = match &terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => build_boundary_parser_from_weighted_terminal_dwa(
            &composed_table.table,
            &templates,
            dwa,
        ),
        _ => None,
    };
    let lazy_parser_candidate = if concrete_delta_plan.is_none()
        && boundary_special_token_terminals.is_empty()
        && composed_table.control_terminals.is_empty()
    {
        lazy_seed_relations.as_ref().and_then(|seed_relations| {
            build_full_boundary_lazy_direct_parser(
                &composed_table.table,
                &templates,
                &boundary_paths,
                &ignore_terminals.global,
                component_state_map,
                &id_map,
                seed_relations,
            )
        })
    } else {
        None
    };
    let direct_parser_candidate = terminal_domain_candidate.or(lazy_parser_candidate);
    let mut parser_analyzed = analyzed.clone();
    if let Some(plan) = concrete_delta_plan.as_ref() {
        debug_assert_eq!(plan.original_num_terminals, analyzed.num_terminals);
        parser_analyzed.num_terminals = plan.synthetic_num_terminals;
        parser_analyzed
            .terminal_display_names
            .resize(plan.synthetic_num_terminals as usize, "<boundary-delta>".to_string());
    }
    let use_direct_parser =
        std::env::var_os("GLRMASK_EXPERIMENT_USE_BOUNDARY_LAZY_DIRECT_PARSER").is_some();
    let validate_direct_parser =
        std::env::var_os("GLRMASK_VALIDATE_BOUNDARY_LAZY_DIRECT_PARSER").is_some();
    let mut generic_parser_dwa = if !use_direct_parser
        || validate_direct_parser
        || direct_parser_candidate.is_none()
    {
        Some(build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
            &composed_table.table,
            &parser_analyzed,
            &terminal_automaton,
            &templates,
            vocab,
            &id_map,
            false,
        ))
    } else {
        None
    };
    if validate_direct_parser {
        if let Some(candidate) = direct_parser_candidate.as_ref() {
            let generic = generic_parser_dwa
                .as_ref()
                .expect("direct-parser validation requires generic reference");
            let mut extra_positive_labels = candidate
                .states()
                .iter()
                .chain(generic.states())
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| {
                    label >= composed_table.table.num_states as i32 && label != DEFAULT_LABEL
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            extra_positive_labels.sort_unstable();
            let candidate_explicit = determinize(&explicit_parser_nwa(
                candidate,
                composed_table.table.num_states,
                &extra_positive_labels,
            ))
            .map_err(|error| error.to_string())?;
            let generic_explicit = determinize(&explicit_parser_nwa(
                generic,
                composed_table.table.num_states,
                &extra_positive_labels,
            ))
            .map_err(|error| error.to_string())?;
            let difference = find_difference(&candidate_explicit, &generic_explicit)
                .map_err(|error| error.to_string())?;
            if let Some(word) = difference.as_ref() {
                let candidate_weight = candidate_explicit.eval_word(word);
                let generic_weight = generic_explicit.eval_word(word);
                let candidate_only = candidate_weight.difference(&generic_weight);
                let generic_only = generic_weight.difference(&candidate_weight);
                let summarize = |weight: &Weight| {
                    weight
                        .raw_range_values()
                        .take(6)
                        .map(|(range, tokens)| {
                            let ranges = tokens.ranges().take(8).collect::<Vec<_>>();
                            ((*range.start(), *range.end()), ranges)
                        })
                        .collect::<Vec<_>>()
                };
                eprintln!(
                    "[glrmask/validate][constraint_boundary_lazy_direct_parser_mismatch] word={word:?} candidate_only={:?} generic_only={:?}",
                    summarize(&candidate_only),
                    summarize(&generic_only),
                );
            }
            assert_eq!(
                difference, None,
                "lazy direct boundary parser differs from generic parser DWA",
            );
            eprintln!(
                "[glrmask/validate][constraint_boundary_lazy_direct_parser] exact=true candidate_states={} generic_states={}",
                candidate.num_states(),
                generic.num_states(),
            );
        }
    }
    let parser_dwa = if use_direct_parser {
        direct_parser_candidate
            .or_else(|| generic_parser_dwa.take())
            .expect("boundary parser construction produced no parser DWA")
    } else {
        generic_parser_dwa
            .take()
            .expect("generic boundary parser must be built when direct path is disabled")
    };
    let pre_hashcons_states = parser_dwa.num_states();
    let pre_hashcons_transitions = parser_dwa.num_transitions();
    let mut hashcons_ms = 0.0;
    let mut boundary_minimize_ms = 0.0;
    let parser_dwa = if pre_hashcons_states >= boundary_parser_minimize_min_states()
        && parser_builder_skips_internal_minimization()
        && std::env::var_os("GLRMASK_DISABLE_BOUNDARY_PARSER_MINIMIZE").is_none()
    {
        let hashcons_started_at = Instant::now();
        let hashconsed = reverse_hashcons_owned(parser_dwa);
        hashcons_ms = hashcons_started_at.elapsed().as_secs_f64() * 1000.0;
        let minimize_started_at = Instant::now();
        let minimized = minimize_owned(hashconsed);
        boundary_minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;
        minimized
    } else {
        parser_dwa
    };
    let post_minimize_states = parser_dwa.num_states();
    let post_minimize_transitions = parser_dwa.num_transitions();
    let parser_ms = parser_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_boundary_build] active={} begin_active={} discovered_active={} boundary_tokens={} boundary_special_tokens={} discovery_ms={discovery_ms:.3} one_byte_ms={one_byte_ms:.3} terminal_ms={terminal_ms:.3} templates_ms={templates_ms:.3} parser_pre_states={} parser_pre_transitions={} hashcons_ms={hashcons_ms:.3} boundary_minimize_ms={boundary_minimize_ms:.3} parser_post_states={} parser_post_transitions={} parser_ms={parser_ms:.3} total_ms={:.3}",
            active_terminals.iter().filter(|&&active| active).count(),
            seed_terminals.iter().filter(|&&active| active).count(),
            discovered_boundary_terminals.count_ones(),
            boundary_paths.token_ids.len(),
            boundary_special_token_terminals.len(),
            pre_hashcons_states,
            pre_hashcons_transitions,
            post_minimize_states,
            post_minimize_transitions,
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


fn build_static_dynamic_overlay_metadata(
    composed_table: &ComposedTable,
    components: &[&Constraint],
    merged_ignores: &MergedIgnoreTerminals,
    terminal_display_names: &[String],
    tokenizer_state_offsets: &[u32],
) -> Result<
    (
        crate::runtime::StaticDynamicOverlayMetadata,
        Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    ),
    String,
> {
    let delta_started_at = Instant::now();
    let augmented_start = composed_table
        .table
        .rules
        .first()
        .map(|rule| rule.lhs)
        .ok_or_else(|| "composed table contains no augmented-start rule".to_string())?;
    let analyzed = AnalyzedGrammar::from_composed_rules(
        composed_table.table.rules.clone(),
        composed_table.table.num_terminals,
        terminal_display_names.to_vec(),
        composed_table.table.nonterminal_display_names.clone(),
        augmented_start,
    );

    // Only the parent grammar is rewritten at this link. Child rule graphs are
    // embedded unchanged, so ordinary child terminal templates transport
    // identically. Scoped ignores are the sole child-side exception.
    let parent_terminal_end = composed_table
        .terminal_offsets
        .get(1)
        .copied()
        .unwrap_or(analyzed.num_terminals) as usize;
    let mut delta_terminals = vec![false; analyzed.num_terminals as usize];
    let selected_parent_end = parent_terminal_end.min(delta_terminals.len());
    delta_terminals[..selected_parent_end].fill(true);
    for terminal in merged_ignores.scoped.iter() {
        if let Some(selected) = delta_terminals.get_mut(terminal) {
            *selected = true;
        }
    }

    let (templates, template_dfas_by_terminal, templates_ms) =
        build_composition_templates(&composed_table.table, &analyzed, &delta_terminals);
    let plan_started_at = Instant::now();
    let plan = prepare_concrete_boundary_delta_plan(
        composed_table,
        components,
        &delta_terminals,
        &templates,
        analyzed.num_terminals,
    );
    let plan_ms = plan_started_at.elapsed().as_secs_f64() * 1000.0;
    let mut repair_terminals = vec![false; analyzed.num_terminals as usize];
    for &terminal in plan.by_global_terminal.keys() {
        if let Some(repair) = repair_terminals.get_mut(terminal as usize) {
            *repair = true;
        }
    }
    for &terminal in &plan.unsafe_terminals {
        if let Some(repair) = repair_terminals.get_mut(terminal as usize) {
            *repair = true;
        }
    }
    let mut parent_states = vec![false; composed_table.table.num_states as usize];
    let mut child_states = vec![false; composed_table.table.num_states as usize];
    for (component_index, relation) in composed_table.state_relations.iter().enumerate() {
        for targets in relation {
            for &state in targets {
                let slot = if component_index == 0 {
                    parent_states.get_mut(state as usize)
                } else {
                    child_states.get_mut(state as usize)
                };
                if let Some(slot) = slot {
                    *slot = true;
                }
            }
        }
    }
    let non_parent_only_parser_states = parent_states
        .iter()
        .zip(&child_states)
        .map(|(&parent, &child)| child && !parent)
        .collect::<Vec<_>>();
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_component_only_delta_metadata] changed={} unsafe={} templates_ms={templates_ms:.3} plan_ms={plan_ms:.3} total_ms={:.3}",
            plan.by_global_terminal.len(),
            plan.unsafe_terminals.len(),
            delta_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok((
        crate::runtime::StaticDynamicOverlayMetadata {
            terminal_offsets: composed_table.terminal_offsets.clone(),
            tokenizer_state_offsets: tokenizer_state_offsets.to_vec(),
            repair_terminals,
            non_parent_only_parser_states,
        },
        template_dfas_by_terminal,
    ))
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
        static_dynamic_overlay: None,
        scoped_ignore_only_tokens: Vec::new(),
        scoped_ignore_prefix_fusions: Vec::new(),
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
        composition_reset_tokens_by_terminal: Vec::new(),
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
        parser_state_domain_labels,
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
    parser_top_accept: BTreeMap<i32, Weight>,
    parser_state_domain_labels: Vec<i32>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    embedded_end_token_ids: Vec<u32>,
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<u32>,
    ignore_expr: Option<crate::automata::regex::Expr>,
    terminal_live_states: Vec<Vec<u32>>,
    tokenizer_fast_transitions: crate::runtime::FastTokenizerTransitions,
    structural_terminal_aliases: usize,
    components_have_no_runtime_product: bool,
    vocab: &Vocab,
) -> ConstraintComposition {
    let ((parser_dwa, possible_matches), internal_ids) = parser_artifacts.into_parts();
    let mut composition = build_composed_constraint_unfinalized(
        composed_table,
        tokenizer,
        tokenizer_state_offsets,
        parser_dwa,
        parser_state_domain_labels,
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
    composition.constraint.parser_top_accept = parser_top_accept;
    let lexer_product_started_at = Instant::now();
    let lexer_product_report = maybe_install_runtime_lexer_product(
        &mut composition.constraint,
        structural_terminal_aliases,
        components_have_no_runtime_product,
    );
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_runtime_lexer_product] attempted={} selected={} parser_overlap={} terminal_aliases={} source_states={} product_states={} source_transitions={} product_transitions={} multi_tsid_product_states={} total_ms={:.3}",
            lexer_product_report.attempted,
            lexer_product_report.selected,
            lexer_product_report.parser_overlap,
            structural_terminal_aliases,
            lexer_product_report.source_states,
            lexer_product_report.product_states,
            lexer_product_report.source_transitions,
            lexer_product_report.product_transitions,
            lexer_product_report.multi_tsid_product_states,
            lexer_product_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
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
    let components_have_no_runtime_product = std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|constraint| constraint.runtime_source_state_offset().is_none());
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
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_linker_inputs] global_ignores={} parent_controls={} child_controls={:?} eof_rewrites_clean={} all_children_nonnullable={} legacy_follow_safe={} child_count={}",
            global_ignores,
            parent.table.control_terminals.len(),
            children.iter().map(|child| child.constraint.table.control_terminals.len()).collect::<Vec<_>>(),
            components_have_no_compiled_eof_stack_rewrites(parent, children),
            children.iter().all(|child| !child.constraint.table.embedded_start_nullable()),
            legacy_splice_has_only_byte_terminal_continuations(parent, children),
            children.len());
    }
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
    let all_children_nonnullable = children
        .iter()
        .all(|child| !child.constraint.table.embedded_start_nullable());
    let use_legacy_splice = std::env::var_os("GLRMASK_EXPERIMENT_FORCE_EXPLICIT_SUBGRAMMAR_CONTROLS").is_none()
        && components_have_no_explicit_controls(parent, children)
        && components_have_no_compiled_eof_stack_rewrites(parent, children)
        && (all_children_nonnullable
            || legacy_splice_has_only_byte_terminal_continuations(parent, children));
    let table_started_at = Instant::now();
    let mut composed_table = if use_legacy_splice {
        compose_subgrammar_tables(
            &parent.table,
            (!global_ignores).then_some(parent.ignore_terminal).flatten(),
            &table_inputs,
        )?
    } else {
        compose_subgrammar_tables_explicit(
            &parent.table,
            (!global_ignores)
                .then_some(parent.ignore_terminal)
                .flatten(),
            &table_inputs,
        )?
    };
    let structural_started_at = Instant::now();
    let structural_states_before = composed_table.table.num_states as usize;
    // Structural sharing is an optional exact quotient. With only one child
    // there are no duplicate sibling parser regions to coalesce, while the
    // full terminal/nonterminal bisimulation scan is material on large links.
    // Skip that pure optimization unless at least two child components exist.
    let attempt_structural_sharing = structural_sharing_enabled() && children.len() > 1;
    let structural_report = if attempt_structural_sharing {
        let terminal_analysis = composition_terminal_classes(parent, children, &composed_table);
        let nonterminal_classes = structural_nonterminal_classes(
            &composed_table.table,
            &terminal_analysis.classes,
            &composed_table.boundary_nonterminals,
        );
        let (candidate_groups, contextual_states_saved) =
            contextually_share_composed_states(
                &mut composed_table,
                parent,
                children,
                &terminal_analysis.classes,
                &nonterminal_classes,
            );
        let mut report = quotient_composed_table_structurally(
            &mut composed_table,
            &terminal_analysis,
            &nonterminal_classes,
        )?;
        report.contextual_candidate_groups = candidate_groups;
        report.contextual_states_saved = contextual_states_saved;
        report.states_before = structural_states_before;
        report.states_after = composed_table.table.num_states as usize;
        report
    } else {
        StructuralSharingReport {
            nonterminals_before: composed_table.table.nonterminal_display_names.len(),
            nonterminal_classes: composed_table.table.nonterminal_display_names.len(),
            states_before: composed_table.table.num_states as usize,
            states_after: composed_table.table.num_states as usize,
            ..StructuralSharingReport::default()
        }
    };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_structural_sharing] enabled={} terminal_aliases={} terminal_structural_matches={} terminal_exact_checks={} terminal_exact_unknown={} nonterminals_before={} nonterminal_classes={} contextual_candidate_groups={} contextual_saved_states={} states_before={} states_after={} saved_states={} total_ms={:.3}",
            attempt_structural_sharing,
            structural_report.terminal_aliases,
            structural_report.terminal_structural_matches,
            structural_report.terminal_exact_checks,
            structural_report.terminal_exact_unknown,
            structural_report.nonterminals_before,
            structural_report.nonterminal_classes,
            structural_report.contextual_candidate_groups,
            structural_report.contextual_states_saved,
            structural_report.states_before,
            structural_report.states_after,
            structural_report.states_before.saturating_sub(structural_report.states_after),
            structural_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let table_ms = table_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_component_terminal_offsets] offsets={:?}", composed_table.terminal_offsets);
    }

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

    let control_elimination_report = eliminate_composed_runtime_controls(&mut composed_table)?;
    let control_elimination_ms = control_elimination_report
        .as_ref()
        .map(|report| report.elapsed_ms)
        .unwrap_or(0.0);
    let special_token_terminals = merged_special_token_terminals(
        parent,
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
            terminal_offset: composed_table.terminal_offsets[index],
            composed_table: Some(&composed_table.table),
        })
        .collect::<Vec<_>>();
    if compose_profile_enabled() {
        for (component_index, relation) in composed_table.state_relations.iter().enumerate() {
            let empty = relation.iter().filter(|targets| targets.is_empty()).count();
            let singleton = relation.iter().filter(|targets| targets.len() == 1).count();
            let multi = relation.len().saturating_sub(empty + singleton);
            let mut affine_delta = None::<i64>;
            let mut affine_singletons = 0usize;
            let mut non_affine_singletons = 0usize;
            for (local, targets) in relation.iter().enumerate() {
                if let [target] = targets.as_slice() {
                    let delta = *target as i64 - local as i64;
                    match affine_delta {
                        None => {
                            affine_delta = Some(delta);
                            affine_singletons += 1;
                        }
                        Some(expected) if expected == delta => affine_singletons += 1,
                        Some(_) => non_affine_singletons += 1,
                    }
                }
            }
            eprintln!(
                "[glrmask/profile][constraint_state_relation_shape] component={} local_states={} empty={} singleton={} multi={} affine_delta={:?} affine_singletons={} non_affine_singletons={} total_targets={}",
                component_index,
                relation.len(),
                empty,
                singleton,
                multi,
                affine_delta,
                affine_singletons,
                non_affine_singletons,
                relation.iter().map(Vec::len).sum::<usize>(),
            );
        }
    }
    let parser_default_domains = build_parser_default_domain_plan(
        &parser_components,
        composed_table.table.num_states,
    );
    if compose_profile_enabled() {
        let selected = parser_default_domains
            .component_domains
            .iter()
            .flatten()
            .count();
        let domain_states = parser_default_domains
            .component_domains
            .iter()
            .flatten()
            .map(|domain| domain.states.count_ones())
            .sum::<usize>();
        let per_component = parser_default_domains
            .component_domains
            .iter()
            .enumerate()
            .filter_map(|(index, domain)| {
                domain.as_ref().map(|domain| {
                    format!("{index}:{}:{}", domain.states.count_ones(), domain.predicted_saved_edges)
                })
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[glrmask/profile][constraint_parser_default_domains] selected={} domain_states={} predicted_saved_edges={} per_component=[{}]",
            selected,
            domain_states,
            parser_default_domains.predicted_saved_edges,
            per_component,
        );
    }

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

    // Experimental lower-bound / hybrid-backend path: build the exact composed
    // grammar/table/tokenizer but intentionally skip parser-DWA transport,
    // boundary parser compilation, possible-match reconciliation, and dense
    // static-mask finalization. DynamicConstraint is the existing exact runtime
    // backend for this representation. This establishes the link-time floor
    // that a structurally shared static/hybrid parser backend must approach.
    if std::env::var_os("GLRMASK_EXPERIMENT_COMPOSE_DYNAMIC_RUNTIME").is_some() {
        let dynamic_started_at = Instant::now();
        let tokenizer_started_at = Instant::now();
        let (mut tokenizer, tokenizer_state_offsets) =
            Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
        let tokenizer_ms = tokenizer_started_at.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(
            tokenizer_state_offsets, expected_tokenizer_state_offsets,
            "predicted composed tokenizer state offsets differ from materialized union",
        );
        if let Some(canonical) = merged_ignores.canonical {
            tokenizer.canonicalize_terminal_aliases(canonical, &merged_ignores.aliases);
        }

        let terminal_offsets = composed_table.terminal_offsets.clone();
        let parser_state_relations = composed_table.state_relations.clone();
        composed_table
            .table
            .set_embedded_end_token_ids(&embedded_end_token_ids);
        let mut dynamic = crate::DynamicConstraint::from_parts_with_dynamic_vocab_unfinalized(
            composed_table.table,
            terminal_display_names,
            tokenizer,
            None,
            ignore_terminal,
            special_token_terminals,
            vocab,
            runtime_dynamic_vocab_for_vocab(vocab),
        );
        let finalize_started_at = Instant::now();
        dynamic.inner.rebuild_dynamic_runtime_caches();
        let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_composition_dynamic] components={} table_ms={table_ms:.3} control_elimination_ms={control_elimination_ms:.3} tokenizer_ms={tokenizer_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
                children.len() + 1,
                dynamic_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(ConstraintComposition {
            constraint: dynamic.inner,
            terminal_offsets,
            tokenizer_state_offsets,
            parser_state_relations,
        });
    }

    if std::env::var_os("GLRMASK_EXPERIMENT_COMPOSE_COMPONENTS_ONLY_STATIC").is_some() {
        let fast_started_at = Instant::now();
        let ((tokenizer_result, tokenizer_ms), ((parser_artifacts, reuse_ms), static_dynamic_overlay)) =
            rayon::join(
                || {
                    let started_at = Instant::now();
                    let result = Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
                    (result, started_at.elapsed().as_secs_f64() * 1000.0)
                },
                || {
                    rayon::join(
                        || {
                            let started_at = Instant::now();
                            let result = compose_component_parser_dwas_and_possible_matches(
                                &parser_components,
                                &composed_table.terminal_offsets,
                                &parser_default_domains.component_domains,
                                merged_tokenizer_state_count,
                                &original_token_ids,
                                !global_ignores,
                            );
                            (result, started_at.elapsed().as_secs_f64() * 1000.0)
                        },
                        || {
                            if std::env::var_os(
                                "GLRMASK_EXPERIMENT_COMPONENT_ONLY_BUILD_DELTA_METADATA",
                            )
                            .is_some()
                            {
                                build_static_dynamic_overlay_metadata(
                                    &composed_table,
                                    &component_constraints,
                                    &merged_ignores,
                                    &terminal_display_names,
                                    &expected_tokenizer_state_offsets,
                                )
                                .map(Some)
                            } else {
                                Ok(None)
                            }
                        },
                    )
                },
            );
        let (mut tokenizer, tokenizer_state_offsets) = tokenizer_result;
        assert_eq!(
            tokenizer_state_offsets, expected_tokenizer_state_offsets,
            "predicted composed tokenizer state offsets differ from materialized union",
        );
        let (parser_artifacts, parser_top_accept) = parser_artifacts?;
        let repair_bundle = static_dynamic_overlay?;
        let (static_dynamic_overlay, template_dfas_by_terminal) = match repair_bundle {
            Some((metadata, template_dfas)) => (Some(metadata), template_dfas),
            None => (None, vec![None; terminal_display_names.len()]),
        };
        let parser_artifacts = canonicalize_parser_artifact_ignore(
            parser_artifacts,
            merged_ignores.canonical,
            &merged_ignores.aliases,
        );
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
        let finalize_started_at = Instant::now();
        let mut result = finalize_composed_constraint(
            composed_table,
            tokenizer,
            tokenizer_state_offsets,
            parser_artifacts,
            parser_top_accept,
            parser_default_domains.parser_state_labels.clone(),
            template_dfas_by_terminal,
            special_token_terminals,
            embedded_end_token_ids,
            terminal_display_names,
            ignore_terminal,
            merged_ignores.canonical_expr.clone(),
            terminal_live_states,
            Default::default(),
            structural_report.terminal_aliases,
            components_have_no_runtime_product,
            vocab,
        );
        result.constraint.static_dynamic_overlay = static_dynamic_overlay;
        result.constraint.rebuild_scoped_ignore_runtime_tokens();
        let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            eprintln!(
                "[glrmask/profile][constraint_composition_components_only] components={} table_ms={table_ms:.3} control_elimination_ms={control_elimination_ms:.3} tokenizer_ms={tokenizer_ms:.3} reuse_ms={reuse_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
                children.len() + 1,
                fast_started_at.elapsed().as_secs_f64() * 1000.0,
            );
        }
        return Ok(result);
    }

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
                    &parser_default_domains.component_domains,
                    merged_tokenizer_state_count,
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
                    terminal_display_names.clone(),
                    &merged_ignores,
                    vocab,
                    special_token_terminals.as_slice(),
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
                                &parser_default_domains.component_domains,
                                merged_tokenizer_state_count,
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
                                terminal_display_names.clone(),
                                &merged_ignores,
                                vocab,
                                special_token_terminals.as_slice(),
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
    let (parser_artifacts, _component_top_accept) = parser_artifacts?;
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
        BTreeMap::new(),
        parser_default_domains.parser_state_labels.clone(),
        template_dfas_by_terminal,
        special_token_terminals,
        embedded_end_token_ids,
        terminal_display_names,
        ignore_terminal,
        merged_ignores.canonical_expr.clone(),
        terminal_live_states,
        Default::default(),
        structural_report.terminal_aliases,
        components_have_no_runtime_product,
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
    let components_have_no_runtime_product = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .all(|constraint| constraint.runtime_source_state_offset().is_none());
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
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_linker_inputs] global_ignores={} parent_controls={} child_controls={:?} eof_rewrites_clean={} all_children_nonnullable={} legacy_follow_safe={} child_count={}",
            global_ignores,
            parent.table.control_terminals.len(),
            children.iter().map(|child| child.constraint.table.control_terminals.len()).collect::<Vec<_>>(),
            components_have_no_compiled_eof_stack_rewrites(&parent, children),
            children.iter().all(|child| !child.constraint.table.embedded_start_nullable()),
            legacy_splice_has_only_byte_terminal_continuations(&parent, children),
            children.len());
    }
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
    let all_children_nonnullable = children
        .iter()
        .all(|child| !child.constraint.table.embedded_start_nullable());
    let use_legacy_splice = std::env::var_os("GLRMASK_EXPERIMENT_FORCE_EXPLICIT_SUBGRAMMAR_CONTROLS").is_none()
        && components_have_no_explicit_controls(&parent, children)
        && components_have_no_compiled_eof_stack_rewrites(&parent, children)
        && (all_children_nonnullable
            || legacy_splice_has_only_byte_terminal_continuations(&parent, children));
    let table_started_at = Instant::now();
    let mut composed_table = if use_legacy_splice {
        compose_subgrammar_tables(
            &parent.table,
            (!global_ignores).then_some(parent.ignore_terminal).flatten(),
            &table_inputs,
        )?
    } else {
        compose_subgrammar_tables_explicit(
            &parent.table,
            (!global_ignores)
                .then_some(parent.ignore_terminal)
                .flatten(),
            &table_inputs,
        )?
    };
    let structural_started_at = Instant::now();
    let structural_states_before = composed_table.table.num_states as usize;
    let attempt_structural_sharing = structural_sharing_enabled() && children.len() > 1;
    let structural_report = if attempt_structural_sharing {
        let terminal_analysis = composition_terminal_classes(&parent, children, &composed_table);
        let nonterminal_classes = structural_nonterminal_classes(
            &composed_table.table,
            &terminal_analysis.classes,
            &composed_table.boundary_nonterminals,
        );
        let (candidate_groups, contextual_states_saved) =
            contextually_share_composed_states(
                &mut composed_table,
                &parent,
                children,
                &terminal_analysis.classes,
                &nonterminal_classes,
            );
        let mut report = quotient_composed_table_structurally(
            &mut composed_table,
            &terminal_analysis,
            &nonterminal_classes,
        )?;
        report.contextual_candidate_groups = candidate_groups;
        report.contextual_states_saved = contextual_states_saved;
        report.states_before = structural_states_before;
        report.states_after = composed_table.table.num_states as usize;
        report
    } else {
        StructuralSharingReport {
            nonterminals_before: composed_table.table.nonterminal_display_names.len(),
            nonterminal_classes: composed_table.table.nonterminal_display_names.len(),
            states_before: composed_table.table.num_states as usize,
            states_after: composed_table.table.num_states as usize,
            ..StructuralSharingReport::default()
        }
    };
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_structural_sharing] enabled={} terminal_aliases={} terminal_structural_matches={} terminal_exact_checks={} terminal_exact_unknown={} nonterminals_before={} nonterminal_classes={} contextual_candidate_groups={} contextual_saved_states={} states_before={} states_after={} saved_states={} total_ms={:.3}",
            attempt_structural_sharing,
            structural_report.terminal_aliases,
            structural_report.terminal_structural_matches,
            structural_report.terminal_exact_checks,
            structural_report.terminal_exact_unknown,
            structural_report.nonterminals_before,
            structural_report.nonterminal_classes,
            structural_report.contextual_candidate_groups,
            structural_report.contextual_states_saved,
            structural_report.states_before,
            structural_report.states_after,
            structural_report.states_before.saturating_sub(structural_report.states_after),
            structural_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let table_ms = table_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!("[glrmask/profile][constraint_component_terminal_offsets] offsets={:?}", composed_table.terminal_offsets);
    }

    let metadata_started_at = Instant::now();
    let component_constraints = std::iter::once(&parent)
        .chain(children.iter().map(|child| child.constraint))
        .collect::<Vec<_>>();
    let (expected_tokenizer_state_offsets, merged_tokenizer_state_count) =
        component_tokenizer_state_layout_owned_parent(&component_constraints);

    let component_views_ms = metadata_started_at.elapsed().as_secs_f64() * 1000.0;
    let specials_started_at = Instant::now();
    let control_elimination_report = eliminate_composed_runtime_controls(&mut composed_table)?;
    let control_elimination_ms = control_elimination_report
        .as_ref()
        .map(|report| report.elapsed_ms)
        .unwrap_or(0.0);
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
            terminal_offset: composed_table.terminal_offsets[index],
            composed_table: Some(&composed_table.table),
        })
        .collect::<Vec<_>>();
    if compose_profile_enabled() {
        for (component_index, relation) in composed_table.state_relations.iter().enumerate() {
            let empty = relation.iter().filter(|targets| targets.is_empty()).count();
            let singleton = relation.iter().filter(|targets| targets.len() == 1).count();
            let multi = relation.len().saturating_sub(empty + singleton);
            let mut deltas = BTreeMap::<i64, usize>::new();
            for (local, targets) in relation.iter().enumerate() {
                if let [target] = targets.as_slice() {
                    *deltas.entry(*target as i64 - local as i64).or_default() += 1;
                }
            }
            let dominant = deltas
                .iter()
                .max_by_key(|(_, count)| **count)
                .map(|(delta, count)| (*delta, *count));
            eprintln!(
                "[glrmask/profile][constraint_state_relation_shape] component={} local_states={} empty={} singleton={} multi={} dominant_affine={:?} distinct_deltas={} total_targets={}",
                component_index,
                relation.len(),
                empty,
                singleton,
                multi,
                dominant,
                deltas.len(),
                relation.iter().map(Vec::len).sum::<usize>(),
            );
        }
    }
    let parser_default_domains = build_parser_default_domain_plan(
        &parser_components,
        composed_table.table.num_states,
    );
    if compose_profile_enabled() {
        let selected = parser_default_domains
            .component_domains
            .iter()
            .flatten()
            .count();
        let domain_states = parser_default_domains
            .component_domains
            .iter()
            .flatten()
            .map(|domain| domain.states.count_ones())
            .sum::<usize>();
        let per_component = parser_default_domains
            .component_domains
            .iter()
            .enumerate()
            .filter_map(|(index, domain)| {
                domain.as_ref().map(|domain| {
                    format!("{index}:{}:{}", domain.states.count_ones(), domain.predicted_saved_edges)
                })
            })
            .collect::<Vec<_>>()
            .join(",");
        eprintln!(
            "[glrmask/profile][constraint_parser_default_domains] selected={} domain_states={} predicted_saved_edges={} per_component=[{}]",
            selected,
            domain_states,
            parser_default_domains.predicted_saved_edges,
            per_component,
        );
    }
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
    // Publish state coordinates and the exact boundary-token coordinate as
    // soon as each becomes available. Component weight remapping then overlaps
    // boundary template/terminal/parser construction instead of sitting on the
    // serial parser-union path.
    let state_map_cell = OnceLock::<Result<ManyToOneIdMap, String>>::new();
    let selected_boundary_tokens_cell =
        OnceLock::<Result<Option<Vec<u32>>, String>>::new();
    let skip_boundary_for_floor =
        std::env::var_os("GLRMASK_EXPERIMENT_OWNED_COMPONENTS_ONLY_STATIC").is_some();
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
                                &parser_default_domains.component_domains,
                                !global_ignores,
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
                    )?;
                    let plan = build_boundary_refinement_plan(component_id_map, &boundary_id_map)
                        .ok_or_else(|| {
                            "component coordinate map does not cover boundary repair".to_string()
                        })?;
                    let (automata, automata_maps, possible_matches, remap_ms) =
                        prepare_deferred_component_artifacts(
                            unmapped_components,
                            component_maps,
                            Some(&plan.component_token_map),
                            plan.common_map.num_tsids() as usize,
                        )?;
                    PreparedOwnedComponentArtifacts {
                        automata,
                        automata_maps,
                        possible_matches,
                        id_map: plan.common_map,
                        boundary_tsid_map: Some(plan.boundary_tsid_map),
                        boundary_token_map: Some(plan.boundary_token_map),
                        remap_ms,
                    }
                } else {
                    let (automata, automata_maps, possible_matches, remap_ms) =
                        prepare_deferred_component_artifacts(
                            unmapped_components,
                            component_maps,
                            None,
                            component_id_map.num_tsids() as usize,
                        )?;
                    PreparedOwnedComponentArtifacts {
                        automata,
                        automata_maps,
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
                if skip_boundary_for_floor {
                    let _ = selected_boundary_tokens_cell.set(Ok(None));
                    return (Ok(None), 0.0);
                }
                let started_at = Instant::now();
                let result = build_boundary_repair(
                    &composed_table,
                    None,
                    merged_tokenizer_state_count,
                    terminal_display_names.clone(),
                    &merged_ignores,
                    vocab,
                    special_token_terminals.as_slice(),
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
        automata_maps,
        mut possible_matches,
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
        parser_default_domains.parser_state_labels.clone(),
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

    let parser_default_domain_labels = parser_default_domains
        .component_domains
        .iter()
        .flatten()
        .flat_map(ParserDefaultDomain::output_labels)
        .collect::<Vec<_>>();
    let union_started_at = Instant::now();
    let (parser_union_result, token_cache_prebuild_ms) = rayon::join(
        || -> Result<DWA, String> {
            let final_build_started_at = Instant::now();
            let deferred_remap_started_at = Instant::now();
            let mut automata = automata
                .into_par_iter()
                .zip(automata_maps.into_par_iter())
                .map(|(mut automaton, maps)| {
                    let mut weights = automaton.weight_refs_mut();
                    remap_weights_with_maps(
                        &mut weights,
                        &maps.local_to_global_tsids,
                        &maps.local_to_global_tokens,
                        id_num_tsids as usize,
                    );
                    drop(weights);
                    automaton
                })
                .collect::<Vec<_>>();
            let deferred_component_remap_ms =
                deferred_remap_started_at.elapsed().as_secs_f64() * 1000.0;
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
                                explicit_parser_nwa(
                                    &boundary_dwa,
                                    num_parser_states,
                                    &parser_default_domain_labels,
                                )
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
                                explicit_parser_nwa(
                                    &boundary_dwa,
                                    num_parser_states,
                                    &parser_default_domain_labels,
                                )
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
                            "[glrmask/profile][constraint_single_pass_parser_union] path={} automata={} eager_component_remap_ms={component_remap_ms:.3} deferred_component_remap_ms={deferred_component_remap_ms:.3} direct_ms={direct_ms:.3} synthetic_states={} result_states={} result_transitions={} total_ms={:.3}",
                            union_path,
                            automata_len,
                            synthetic_states,
                            parser_dwa.num_states(),
                            parser_dwa.num_transitions(),
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
                            "[glrmask/profile][constraint_single_pass_parser_union] path={} automata={} eager_component_remap_ms={component_remap_ms:.3} deferred_component_remap_ms={deferred_component_remap_ms:.3} direct_ms={direct_ms:.3} synthetic_states={} result_states={} result_transitions={} total_ms={:.3}",
                            union_path,
                            automata_len,
                            synthetic_states,
                            parser_dwa.num_states(),
                            parser_dwa.num_transitions(),
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
    let lexer_product_started_at = Instant::now();
    let lexer_product_report = maybe_install_runtime_lexer_product(
        &mut result.constraint,
        structural_report.terminal_aliases,
        components_have_no_runtime_product,
    );
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_runtime_lexer_product] attempted={} selected={} parser_overlap={} terminal_aliases={} source_states={} product_states={} source_transitions={} product_transitions={} multi_tsid_product_states={} total_ms={:.3}",
            lexer_product_report.attempted,
            lexer_product_report.selected,
            lexer_product_report.parser_overlap,
            structural_report.terminal_aliases,
            lexer_product_report.source_states,
            lexer_product_report.product_states,
            lexer_product_report.source_transitions,
            lexer_product_report.product_transitions,
            lexer_product_report.multi_tsid_product_states,
            lexer_product_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
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
    include!("minbound_trigram.rs");
    include!("minbound_factor.rs");

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
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    constraint: child,
                });
            }
            compose_constraints(self, &inputs, vocab)
                .map(|composition| composition.constraint)
                .map_err(crate::GlrMaskError::Compilation)
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
                    return Err(crate::GlrMaskError::Compilation(format!(
                        "parent placeholder terminal {name:?} was supplied more than once",
                    )));
                }
                inputs.push(CompiledSubgrammarInput {
                    placeholder_terminal,
                    constraint: child,
                });
            }
            compose_constraints_owned_parent(self, &inputs, vocab)
                .map(|composition| composition.constraint)
                .map_err(crate::GlrMaskError::Compilation)
        }
    }

    #[test]
    fn structural_sharing_quotients_duplicate_child_lr_regions() {
        let vocab = Vocab::new(vec![
            (0, b"<abc>,<abc>".to_vec()),
            (2, b"<".to_vec()),
            (3, b"a".to_vec()),
            (4, b"b".to_vec()),
            (7, b"c".to_vec()),
            (5, b">,<".to_vec()),
            (6, b">".to_vec()),
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
                nt value ::= "a" "b" "c";
                nt child ::= value;
            "#,
            &vocab,
        )
        .unwrap();
        let loaded_child = Constraint::load(&child.save()).unwrap();
        let children = [
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "LEFT"),
                constraint: &child,
            },
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "RIGHT"),
                constraint: &loaded_child,
            },
        ];
        let table_inputs = children
            .iter()
            .map(|child| SubgrammarTableInput {
                placeholder_terminal: child.placeholder_terminal,
                table: &child.constraint.table,
                ignore_terminal: child.constraint.ignore_terminal,
                start_nullable: child.constraint.table.embedded_start_nullable(),
            })
            .collect::<Vec<_>>();
        let mut composed = compose_subgrammar_tables(&parent.table, None, &table_inputs).unwrap();
        let before = composed.table.num_states;
        let terminal_analysis = composition_terminal_classes(&parent, &children, &composed);
        assert!(
            loaded_child.tokenizer.terminal_exprs().is_some(),
            "current serialized artifacts should retain terminal proof expressions",
        );
        assert!(
            terminal_analysis
                .classes
                .iter()
                .enumerate()
                .any(|(terminal, &class)| terminal as u32 != class),
            "the separately loaded child should still form terminal aliases",
        );
        let nonterminal_classes = structural_nonterminal_classes(
            &composed.table,
            &terminal_analysis.classes,
            &composed.boundary_nonterminals,
        );
        let (candidate_groups, contextual_saved) =
            contextually_share_composed_states(
                &mut composed,
                &parent,
                &children,
                &terminal_analysis.classes,
                &nonterminal_classes,
            );
        let _ = quotient_composed_table_structurally(
            &mut composed,
            &terminal_analysis,
            &nonterminal_classes,
        )
        .unwrap();

        assert!(candidate_groups > 0);
        assert!(
            contextual_saved > 0 && composed.table.num_states < before,
            "duplicate independently compiled children should share at least one LR state",
        );

        let shared_local_states = composed.state_relations[1]
            .iter()
            .zip(&composed.state_relations[2])
            .filter(|(left, right)| left == right)
            .count();
        assert!(
            shared_local_states > 0,
            "at least one corresponding child-local LR state should map to the same quotient state",
        );

        // Exercise the real artifact-reuse path as well: component parser DWAs
        // are transported through the many-to-one LR-state relation rather
        // than rebuilt from the quotient table.
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt value ::= "a" "b" "c";
                nt child ::= value;
                nt document ::= "<" child ">,<" child ">";
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
        actual.commit_token(0).unwrap();
        assert!(actual.is_finished());

    }

    #[test]
    fn contextual_sharing_rejects_ambiguity_with_no_stack_provenance() {
        let vocab = byte_vocab();
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t P ::= "p";
                t Q ::= "p";
                t BANG ::= "!";
                t QUESTION ::= "?";
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt left_prefix ::= P;
                nt right_prefix ::= Q;
                nt left_branch ::= left_prefix LEFT;
                nt right_branch ::= right_prefix RIGHT;
                nt document ::= left_branch BANG | right_branch QUESTION;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t CA ::= "a";
                t CB ::= "b";
                t CC ::= "c";
                nt child ::= CA CB CC;
            "#,
            &vocab,
        )
        .unwrap();
        let children = [
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "LEFT"),
                constraint: &child,
            },
            CompiledSubgrammarInput {
                placeholder_terminal: terminal(&parent, "RIGHT"),
                constraint: &child,
            },
        ];
        let table_inputs = children
            .iter()
            .map(|child| SubgrammarTableInput {
                placeholder_terminal: child.placeholder_terminal,
                table: &child.constraint.table,
                ignore_terminal: child.constraint.ignore_terminal,
                start_nullable: child.constraint.table.embedded_start_nullable(),
            })
            .collect::<Vec<_>>();
        let mut shared = compose_subgrammar_tables(&parent.table, None, &table_inputs).unwrap();
        let terminal_analysis = composition_terminal_classes(&parent, &children, &shared);
        let nonterminal_classes = structural_nonterminal_classes(
            &shared.table,
            &terminal_analysis.classes,
            &shared.boundary_nonterminals,
        );
        let (candidate_count, saved) =
            contextually_share_composed_states(
                &mut shared,
                &parent,
                &children,
                &terminal_analysis.classes,
                &nonterminal_classes,
            );
        assert!(candidate_count > 0, "the duplicate child states should be detected structurally");
        assert_eq!(
            saved, 0,
            "when two linked copies have the exact same lower stack context, a table-only quotient must preserve their distinct LR states",
        );
    }

    #[test]
    fn runtime_lexer_product_coalesces_equivalent_ambiguous_child_lanes() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"!".to_vec()),
            (4, b"?".to_vec()),
        ]);
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t BANG ::= "!";
                t QUESTION ::= "?";
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt document ::= LEFT BANG | RIGHT QUESTION;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t WORD ::= "abc";
                nt child ::= WORD;
            "#,
            &vocab,
        )
        .unwrap();
        let composed = parent
            .compose_linked_children_for_test(&[("LEFT", &child), ("RIGHT", &child)], &vocab)
            .unwrap();

        let mut state = composed.start();
        state.commit_token(0).unwrap();
        let explicitly_disabled = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "no" | "off"
                )
            });
        if explicitly_disabled {
            assert!(composed.runtime_source_state_offset().is_none());
            assert_eq!(
                state.state.len(),
                2,
                "without the runtime product, the partial token 'a' should leave the two equivalent child lexer lanes split",
            );
            return;
        }

        let source_offset = composed
            .runtime_source_state_offset()
            .expect("duplicate ambiguous child lexer lanes should select the exact runtime product");
        assert!(source_offset > 0);
        assert_eq!(
            state.state.len(),
            1,
            "the persistent lexer frontier should have one product key after the partial token 'a'",
        );
        let product_state = *state.state.keys().next().unwrap();
        assert!(product_state < source_offset);
        assert!(
            composed
                .runtime_product_source_states(product_state)
                .is_some_and(|sources| sources.len() >= 2),
            "the single product key must represent at least two exact source lexer lanes",
        );

        for suffix in [[1, 2, 3], [1, 2, 4]] {
            let mut cursor = composed.start();
            cursor.commit_token(0).unwrap();
            for token in suffix {
                cursor.commit_token(token).unwrap();
            }
            assert!(cursor.is_finished());
        }
    }

    #[test]
    fn multi_tsid_runtime_lexer_product_remains_recomposable() {
        let vocab = byte_vocab();
        let left = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "abc" | "z";
            "#,
            &vocab,
        )
        .unwrap();
        let right = Constraint::from_glrm_grammar(
            r#"
                start child;
                nt child ::= "abd" | "z";
            "#,
            &vocab,
        )
        .unwrap();
        let middle_parent = Constraint::from_glrm_grammar(
            r#"
                start call;
                t LEFT ::= @token(998);
                t RIGHT ::= @token(999);
                nt call ::= LEFT | RIGHT;
            "#,
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .compose_linked_children_for_test(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
            .unwrap();

        let explicitly_disabled = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "no" | "off"
                )
            });
        if !explicitly_disabled {
            let source_offset = middle
                .runtime_source_state_offset()
                .expect("the overlapping abc/abd child lanes should select a runtime product");
            assert!(
                (0..source_offset).any(|state| middle.internal_tsids_for_state(state).len() > 1),
                "the regression requires at least one product state with multiple TSID memberships",
            );
        }

        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t CALL ::= @token(1000);
                nt document ::= "<" CALL ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .compose_linked_children_for_test(&[("CALL", &middle)], &vocab)
            .unwrap();
        let loaded_middle = Constraint::load(&middle.save()).unwrap();
        let outer_loaded = Constraint::from_glrm_grammar(
            r#"
                start document;
                t CALL ::= @token(1000);
                nt document ::= "<" CALL ">";
            "#,
            &vocab,
        )
        .unwrap()
        .compose_linked_children_for_test(&[("CALL", &loaded_middle)], &vocab)
        .unwrap();

        for bytes in [b"<abc>".as_slice(), b"<abd>", b"<z>"] {
            for constraint in [&outer, &outer_loaded] {
                let mut state = constraint.start();
                for &byte in bytes {
                    state.commit_token(byte as u32).unwrap();
                }
                assert!(state.is_finished(), "recomposed runtime-product child rejected {bytes:?}");
            }
        }
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
                if next_u32(random) % 3 != 0 {
                    let start = (next_u32(random) + salt) % 48;
                    let end = (start + 3 + next_u32(random) % 12).min(63);
                    dwa.set_final_weight(
                        state,
                        Weight::from_token_set_for_tsid(
                            0,
                            RangeSetBlaze::from_iter([start..=end]),
                        ),
                    );
                }
                if state == 3 {
                    continue;
                }
                let mut labels = BTreeSet::<i32>::new();
                for concrete in 0..6u32 {
                    if next_u32(random) % 3 != 0 {
                        labels.insert(encode_positive_label(concrete));
                    }
                }
                if next_u32(random) % 2 == 0 {
                    labels.insert(encode_negative_label(next_u32(random) % 6));
                }
                if next_u32(random) % 2 == 0 {
                    labels.insert(20 + (next_u32(random) % 3) as i32);
                }
                if next_u32(random) % 3 == 0 {
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
                            RangeSetBlaze::from_iter([start..=end]),
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
            None,
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
        let parser_components = [
            ParserDwaComponent {
                constraint: &parent,
                parser_state_relation: &composed_table.state_relations[0],
                tokenizer_state_offset: tokenizer_offsets[0],
                terminal_offset: 0,
                composed_table: None,
            },
            ParserDwaComponent {
                constraint: &child,
                parser_state_relation: &composed_table.state_relations[1],
                tokenizer_state_offset: tokenizer_offsets[1],
                terminal_offset: 0,
                composed_table: None,
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
                if std::env::var_os("GLRMASK_DEBUG_PREFIX10_STACKS").is_some()
                    && prefix == [10]
                {
                    eprintln!("PREFIX10_ACTUAL {:?}", actual_state.debug_parser_stacks());
                    eprintln!("PREFIX10_EXPECTED {:?}", expected_state.debug_parser_stacks());
                }
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
    fn scoped_visible_terminals_survive_model_token_boundaries() {
        // Entry orientation:
        //   token 0 = "Xa" leaves parent IGNORE="ab" in progress;
        //   token 1 = "bt" completes that IGNORE and enters the child.
        // Exit orientation:
        //   token 3 = "X" enters the child at reset;
        //   token 4 = "ta" completes child T and starts parent IGNORE;
        //   token 5 = "b!" completes IGNORE in the following model token.
        let vocab = Vocab::new(vec![
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
        let parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= "ab";
                t X ::= "X";
                t BANG ::= "!";
                t SUB ::= @token(999);
                nt document ::= X SUB BANG;
            "#,
            &vocab,
        )
        .unwrap();
        let child = Constraint::from_glrm_grammar(
            r#"
                start child;
                t CHILD_T ::= "t";
                nt child ::= CHILD_T;
            "#,
            &vocab,
        )
        .unwrap();
        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                ignore PARENT_WS;
                t PARENT_WS ::= "ab";
                t X ::= "X";
                t BANG ::= "!";
                g child ::= {
                    start child;
                    t CHILD_T ::= "t";
                    nt child ::= CHILD_T;
                };
                nt document ::= X child BANG;
            "#,
            &vocab,
        )
        .unwrap();

        // Scoped trivia is represented as a real terminal in the composed LR
        // table. Its parser semantics is state-dependent identity (`Skip`), not
        // a lexer-side boundary special case.
        let composed_table = compose_subgrammar_tables(
            &parent.table,
            Some(terminal(&parent, "PARENT_WS")),
            &[SubgrammarTableInput {
                placeholder_terminal: terminal(&parent, "SUB"),
                table: &child.table,
                ignore_terminal: child.ignore_terminal,
                start_nullable: child.table.embedded_start_nullable(),
            }],
        )
        .unwrap();
        let parent_ignore = terminal(&parent, "PARENT_WS");
        assert!(composed_table.table.skip_terminals.contains(&parent_ignore));
        assert!(composed_table.table.action.iter().any(|row| {
            matches!(row.get(&parent_ignore), Some(Action::Skip))
        }));

        let composed = parent
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let mut prepared_parent = parent.clone();
        let mut prepared_child = child.clone();
        prepared_parent.ensure_composition_reset_tokens_by_terminal();
        prepared_child.ensure_composition_reset_tokens_by_terminal();
        let prepared_parent = Constraint::load(&prepared_parent.save()).unwrap();
        let prepared_child = Constraint::load(&prepared_child.save()).unwrap();
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
            let mut actual = composed.start();
            let mut cached = cached_composed.start();
            let mut expected = monolithic.start();
            for &token in sequence {
                let actual_mask = actual.mask();
                let cached_mask = cached.mask();
                let expected_mask = expected.mask();
                assert_eq!(
                    cached_mask, expected_mask,
                    "prepared-cache mask mismatch before token {token} in sequence {sequence:?}",
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
                actual.commit_token(token).unwrap_or_else(|error| {
                    panic!("composed rejected token {token} in {sequence:?}: {error}")
                });
                cached.commit_token(token).unwrap_or_else(|error| {
                    panic!("prepared-cache composed rejected token {token} in {sequence:?}: {error}")
                });
                expected.commit_token(token).unwrap_or_else(|error| {
                    panic!("monolithic rejected token {token} in {sequence:?}: {error}")
                });
            }
            assert_eq!(actual.mask(), expected.mask());
            assert_eq!(cached.mask(), expected.mask());
            assert_eq!(actual.is_finished(), expected.is_finished());
            assert_eq!(cached.is_finished(), expected.is_finished());
            assert!(actual.is_finished(), "sequence {sequence:?} should finish");
        }
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
    fn scoped_ignore_oracle_survives_reload_and_nested_recomposition() {
        let vocab = Vocab::new(vec![
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
            .unwrap();
        let loaded = Constraint::load(&composed.save()).unwrap();

        // Exhaust the small reachable token-prefix graph.  We compare masks
        // rather than the inline lowering's trivia-only completion artifact;
        // successful complete strings are checked explicitly below.
        assert_constraints_mask_equivalent_on_reachable_prefixes(
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
            let mut actual = composed.start();
            let mut restored = loaded.start();
            let mut expected = monolithic.start();
            for &token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "mask before {sequence:?} token {token}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask before {sequence:?} token {token}");
                actual.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_complete(), "composed incomplete for {sequence:?}");
            assert!(restored.is_complete(), "loaded incomplete for {sequence:?}");
            assert!(expected.is_complete(), "reference incomplete for {sequence:?}");
        }

        // Parent trivia must not become active while the child scope is still
        // parsing, and child trivia must not leak back into the parent after
        // the child has returned.
        for sequence in [&[2u32, 13][..], &[19][..], &[21][..]] {
            let mut actual = composed.start();
            let mut restored = loaded.start();
            let mut expected = monolithic.start();
            for &token in &sequence[..sequence.len() - 1] {
                actual.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            let token = *sequence.last().unwrap();
            let expected_result = expected.commit_token(token).is_ok();
            assert_eq!(actual.commit_token(token).is_ok(), expected_result);
            assert_eq!(restored.commit_token(token).is_ok(), expected_result);
            assert!(!expected_result, "invalid scoped sequence {sequence:?} was accepted");
        }

        // Reuse the serialized, already-composed constraint as a child again.
        // This is the inherited-skip case: CHILD_WS is no longer a top-level
        // ignore of `loaded`, but its scoped phase behavior must survive the
        // next call/return boundary exactly.
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start outer;
                t INNER ::= @token(1000);
                nt outer ::= "<" INNER ">";
            "#,
            &vocab,
        )
        .unwrap();
        let outer = outer_parent
            .compose_linked_children_for_test(&[("INNER", &loaded)], &vocab)
            .unwrap();
        let outer_monolithic = Constraint::from_glrm_grammar(
            r#"
                start outer;
                g inner ::= {
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
                };
                nt outer ::= "<" inner ">";
            "#,
            &vocab,
        )
        .unwrap();

        assert_constraints_mask_equivalent_on_reachable_prefixes(
            &outer,
            &outer_monolithic,
            &vocab,
            4,
        );
        for sequence in [&[0u32, 16, 1][..], &[14, 6, 15][..], &[17][..]] {
            let mut actual = outer.start();
            let mut expected = outer_monolithic.start();
            for &token in sequence {
                assert_eq!(actual.mask(), expected.mask(), "outer mask before {sequence:?} token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert!(actual.is_complete(), "outer incomplete for {sequence:?}");
            assert!(expected.is_complete(), "outer reference incomplete for {sequence:?}");
        }

        let mut actual = outer.start();
        let mut expected = outer_monolithic.start();
        assert_eq!(actual.commit_token(23).is_ok(), expected.commit_token(23).is_ok());
        assert!(!expected.is_complete());
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
            .compose_linked_children_for_test(&[("RIGHT", &right)], &vocab)
            .unwrap();
        let composed = parent_with_right
            .compose_linked_children_for_test(&[("LEFT", &left)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("INNER", &composed)], &vocab)
            .unwrap();
        let outer_from_loaded = outer_parent
            .compose_linked_children_for_test(&[("INNER", &loaded)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
            .compose_linked_children_for_test(&[("CHILD", &nullable_child)], &vocab)
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
            .compose_linked_children_for_test(&[("MIDDLE", &middle)], &vocab)
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
                .compose_linked_children_for_test(&[("SUB", child)], &vocab)
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
    fn contextual_structural_sharing_remains_recomposable_when_children_share_whole_start_alternative() {
        let vocab = byte_vocab();

        let expr = Constraint::from_glrm_grammar(
            r#"
                start expr;
                nt expr ::= "x";
            "#,
            &vocab,
        )
        .unwrap();

        let arg_a_parent = Constraint::from_glrm_grammar(
            r#"
                start args;
                t EXPR ::= @token(996);
                nt args ::= "{" "a" ":" EXPR "}" | EXPR;
            "#,
            &vocab,
        )
        .unwrap();
        let arg_a = arg_a_parent
            .compose_linked_children_for_test(&[("EXPR", &expr)], &vocab)
            .unwrap();

        let arg_b_parent = Constraint::from_glrm_grammar(
            r#"
                start args;
                t EXPR ::= @token(996);
                nt args ::= "{" "b" ":" EXPR "}" | EXPR;
            "#,
            &vocab,
        )
        .unwrap();
        let arg_b = arg_b_parent
            .compose_linked_children_for_test(&[("EXPR", &expr)], &vocab)
            .unwrap();

        let dispatch_parent = Constraint::from_glrm_grammar(
            r#"
                start call;
                t ARGA ::= @token(997);
                t ARGB ::= @token(998);
                nt call ::= "t" "." "ta" "(" ARGA ")"
                          | "t" "." "tb" "(" ARGB ")";
            "#,
            &vocab,
        )
        .unwrap();
        let dispatch = dispatch_parent
            .compose_linked_children_for_test(&[("ARGA", &arg_a), ("ARGB", &arg_b)], &vocab)
            .unwrap();

        // The two argument children both expose the same nested `expr` as a
        // whole-start alternative. Contextual structural sharing can prove an
        // interior quotient for that overlap. The resulting table must still
        // remain a valid child for a *later* composition level.
        let outer_parent = Constraint::from_glrm_grammar(
            r#"
                start document;
                t CALL ::= @token(999);
                nt document ::= "X" CALL "!";
            "#,
            &vocab,
        )
        .unwrap();
        let composed = outer_parent
            .clone()
            .compose_linked_children_for_test(&[("CALL", &dispatch)], &vocab)
            .unwrap();
        let loaded_dispatch = Constraint::load(&dispatch.save()).unwrap();
        let composed_from_loaded = outer_parent
            .compose_linked_children_for_test(&[("CALL", &loaded_dispatch)], &vocab)
            .unwrap();

        let monolithic = Constraint::from_glrm_grammar(
            r#"
                start document;
                nt expr ::= "x";
                nt arg_a ::= "{" "a" ":" expr "}" | expr;
                nt arg_b ::= "{" "b" ":" expr "}" | expr;
                nt call ::= "t" "." "ta" "(" arg_a ")"
                          | "t" "." "tb" "(" arg_b ")";
                nt document ::= "X" call "!";
            "#,
            &vocab,
        )
        .unwrap();

        for bytes in [b"Xt.ta(x)!".as_slice(), b"Xt.tb(x)!", b"Xt.ta({a:x})!", b"Xt.tb({b:x})!"] {
            let mut expected = monolithic.start();
            let mut actual = composed.start();
            let mut restored = composed_from_loaded.start();
            for &byte in bytes {
                assert_eq!(actual.mask(), expected.mask(), "mask mismatch before byte {byte:?}");
                assert_eq!(restored.mask(), expected.mask(), "loaded mask mismatch before byte {byte:?}");
                actual.commit_token(byte as u32).unwrap();
                restored.commit_token(byte as u32).unwrap();
                expected.commit_token(byte as u32).unwrap();
            }
            assert!(actual.is_finished());
            assert!(restored.is_finished());
            assert!(expected.is_finished());
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
            .compose_linked_children_for_test(&[("LEAF", &leaf)], &vocab)
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
            .compose_linked_children_for_test(&[("MIDDLE", &middle)], &vocab)
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
            .compose_linked_children_for_test(&[("LEAF", &leaf)], &vocab)
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
    fn internal_composition_matches_monolithic_for_child_alternatives() {
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
            .compose_linked_children_for_test(&[("LEFT", &left), ("LEFT", &right)], &vocab)
            .expect_err("duplicate placeholder inputs must be rejected");
        assert!(duplicate_error
            .to_string()
            .contains("was supplied more than once"));
        let composed = parent
            .compose_linked_children_for_test(&[("LEFT", &left), ("RIGHT", &right)], &vocab)
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
            .compose_linked_children_for_test(&[("SUB", &child)], &vocab)
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
    #[test]
    #[ignore]
    fn debug_minimal_terminal_boundary_subtraction_selected10() {
        use std::fs;
        use std::sync::Arc;
        use std::time::Instant;

        fn load_vocab(path: &str) -> Vocab {
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
            Vocab::new(entries)
        }



        fn save_terminal_capture(path: &str, artifact: &MappedArtifact<DWA>, names: &[String]) {
            use std::io::Write;
            fn write_u32(out: &mut Vec<u8>, value: u32) { out.extend_from_slice(&value.to_le_bytes()); }
            fn write_u64(out: &mut Vec<u8>, value: u64) { out.extend_from_slice(&value.to_le_bytes()); }
            fn write_vec(out: &mut Vec<u8>, values: &[u32]) {
                write_u32(out, values.len() as u32);
                for &value in values { write_u32(out, value); }
            }
            fn write_vec_vec(out: &mut Vec<u8>, values: &[Vec<u32>]) {
                write_u32(out, values.len() as u32);
                for value in values { write_vec(out, value); }
            }
            fn write_map(out: &mut Vec<u8>, map: &ManyToOneIdMap) {
                write_vec(out, &map.original_to_internal);
                write_vec_vec(out, &map.internal_to_originals);
                write_vec(out, &map.representative_original_ids);
            }
            let encoded = bincode::serialize(artifact.artifact()).unwrap();
            let mut out = Vec::new();
            out.extend_from_slice(b"GLRMTD1\0");
            write_u32(&mut out, names.len() as u32);
            for name in names {
                write_u32(&mut out, name.len() as u32);
                out.extend_from_slice(name.as_bytes());
            }
            write_map(&mut out, &artifact.id_map().tokenizer_states);
            write_map(&mut out, &artifact.id_map().vocab_tokens);
            write_u64(&mut out, encoded.len() as u64);
            out.extend_from_slice(&encoded);
            let mut file = fs::File::create(path).unwrap();
            file.write_all(&out).unwrap();
            eprintln!("MINBOUND saved_capture path={path} states={} transitions={} bytes={}", artifact.artifact().num_states(), artifact.artifact().num_transitions(), out.len());
        }

        fn load_terminal_capture(path: &str) -> (MappedArtifact<DWA>, Vec<String>) {
            let bytes = fs::read(path).expect("read terminal-DWA capture");
            let mut offset = 0usize;
            assert_eq!(&bytes[..8], b"GLRMTD1\0");
            offset += 8;
            fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {
                let end = *offset + 4;
                let value = u32::from_le_bytes(bytes[*offset..end].try_into().unwrap());
                *offset = end;
                value
            }
            fn read_u64(bytes: &[u8], offset: &mut usize) -> u64 {
                let end = *offset + 8;
                let value = u64::from_le_bytes(bytes[*offset..end].try_into().unwrap());
                *offset = end;
                value
            }
            fn read_vec(bytes: &[u8], offset: &mut usize) -> Vec<u32> {
                let len = read_u32(bytes, offset) as usize;
                (0..len).map(|_| read_u32(bytes, offset)).collect()
            }
            fn read_vec_vec(bytes: &[u8], offset: &mut usize) -> Vec<Vec<u32>> {
                let len = read_u32(bytes, offset) as usize;
                (0..len).map(|_| read_vec(bytes, offset)).collect()
            }
            fn read_map(bytes: &[u8], offset: &mut usize) -> ManyToOneIdMap {
                ManyToOneIdMap {
                    original_to_internal: read_vec(bytes, offset),
                    internal_to_originals: read_vec_vec(bytes, offset),
                    representative_original_ids: read_vec(bytes, offset),
                }
            }
            let name_count = read_u32(&bytes, &mut offset) as usize;
            let mut names = Vec::with_capacity(name_count);
            for _ in 0..name_count {
                let len = read_u32(&bytes, &mut offset) as usize;
                let end = offset + len;
                names.push(String::from_utf8(bytes[offset..end].to_vec()).unwrap());
                offset = end;
            }
            let tokenizer_states = read_map(&bytes, &mut offset);
            let vocab_tokens = read_map(&bytes, &mut offset);
            let dwa_len = read_u64(&bytes, &mut offset) as usize;
            let end = offset + dwa_len;
            let dwa: DWA = bincode::deserialize(&bytes[offset..end]).expect("deserialize terminal DWA");
            offset = end;
            assert_eq!(offset, bytes.len());
            let id_map = InternalIdMap {
                tokenizer_states,
                vocab_tokens,
                deferred_vocab_singleton_original_ids: None,
            };
            (MappedArtifact::new(dwa, id_map), names)
        }

        fn analyzed(constraint: &Constraint) -> AnalyzedGrammar {
            let augmented_start = constraint
                .table
                .rules
                .first()
                .expect("constraint table has augmented start")
                .lhs;
            AnalyzedGrammar::from_composed_rules(
                constraint.table.rules.clone(),
                constraint.table.num_terminals,
                constraint.terminal_display_names.clone(),
                constraint.table.nonterminal_display_names.clone(),
                augmented_start,
            )
        }

        fn build_terminal_dwa_parts(
            name: &str,
            tokenizer: &Tokenizer,
            table: &crate::compiler::glr::table::GLRTable,
            terminal_display_names: &[String],
            ignore_terminal: Option<u32>,
            vocab: &Vocab,
        ) -> MappedArtifact<DWA> {
            let started = Instant::now();
            let augmented_start = table
                .rules
                .first()
                .expect("terminal-DWA oracle table has augmented start")
                .lhs;
            let grammar = AnalyzedGrammar::from_composed_rules(
                table.rules.clone(),
                table.num_terminals,
                terminal_display_names.to_vec(),
                table.nonterminal_display_names.clone(),
                augmented_start,
            );
            let disallowed = crate::compiler::pipeline::compute_disallowed_follows(&grammar);
            let flat: Arc<[u32]> = Arc::from(
                crate::compiler::stages::id_map_and_terminal_dwa::l1::build_flat_transition_table(
                    tokenizer,
                ),
            );
            // This experiment wants language, not the production global tokenizer-state
            // quotient. Keep the raw tokenizer-state coordinate exact and singleton so we
            // do not pay the large max-length/global-equivalence preparation just to compare
            // terminal languages. Reconciliation below will still put every DWA into one
            // exact common TSID refinement.
            let raw_ids = (0..tokenizer.num_states()).collect::<Vec<_>>();
            let state_map = ManyToOneIdMap::from_singleton_original_to_internal_with_representatives(
                raw_ids.clone(),
                raw_ids,
            );
            let coloring = TerminalColoring::identity(grammar.num_terminals as usize);
            let (artifact, profile) =
                crate::compiler::stages::id_map_and_terminal_dwa::
                    build_restricted_id_map_and_terminal_dwa_with_precomputed_global_max_length(
                        tokenizer,
                        vocab,
                        &coloring,
                        false,
                        ignore_terminal,
                        &grammar,
                        &disallowed,
                        flat,
                        &state_map,
                        None,
                        None,
                    );
            let (automaton, id_map) = artifact.into_parts();
            let dwa = match automaton {
                TerminalAutomaton::Dwa(dwa) => dwa,
                TerminalAutomaton::TokenDeterministicNwa(nwa)
                | TerminalAutomaton::EpsilonNwa(nwa) => determinize(&nwa).unwrap(),
            };
            eprintln!(
                "MINBOUND terminal_build name={name} states={} transitions={} tsids={} tokens={} disallowed_pairs={} ms={:.3} profile_total_ms={:.3}",
                dwa.num_states(),
                dwa.num_transitions(),
                id_map.num_tsids(),
                id_map.num_internal_tokens(),
                disallowed.values().map(BitSet::count_ones).sum::<usize>(),
                started.elapsed().as_secs_f64() * 1000.0,
                profile.total_ms(),
            );
            MappedArtifact::new(dwa, id_map)
        }

        fn build_terminal_dwa(
            name: &str,
            constraint: &Constraint,
            vocab: &Vocab,
        ) -> MappedArtifact<DWA> {
            build_terminal_dwa_parts(
                name,
                &constraint.tokenizer,
                &constraint.table,
                &constraint.terminal_display_names,
                constraint.ignore_terminal,
                vocab,
            )
        }


        fn exact_terminal_language_classes(tokenizer: &Tokenizer) -> Vec<u32> {
            let mut representative_by_expr = FxHashMap::<crate::automata::regex::Expr, u32>::default();
            let mut classes = Vec::with_capacity(tokenizer.num_terminals() as usize);
            for terminal in 0..tokenizer.num_terminals() {
                let representative = tokenizer
                    .terminal_expr(terminal)
                    .map(|expr| {
                        *representative_by_expr.entry(expr.clone()).or_insert(terminal)
                    })
                    .unwrap_or(terminal);
                classes.push(representative);
            }
            classes
        }

        fn quotient_terminal_language_labels(name: &str, dwa: &DWA, classes: &[u32]) -> DWA {
            let started = Instant::now();
            let mut nwa = dwa.to_nwa();
            for state in nwa.states_mut() {
                let old = std::mem::take(&mut state.transitions);
                for (label, targets) in old {
                    let mapped = if label == DEFAULT_LABEL {
                        DEFAULT_LABEL
                    } else {
                        assert!(label >= 0, "{name}: unexpected negative terminal label {label}");
                        classes.get(label as usize).copied().unwrap_or(label as u32) as i32
                    };
                    state.transitions.entry(mapped).or_default().extend(targets);
                }
            }
            let determinized = determinize(&nwa).expect("terminal-language quotient determinization");
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(determinized);
            eprintln!(
                "MINBOUND language_quotient name={name} input_states={} input_transitions={} states={} transitions={} ms={:.3}",
                dwa.num_states(), dwa.num_transitions(), minimized.num_states(), minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }

        fn offset_terminal_labels(
            name: &str,
            mut artifact: MappedArtifact<DWA>,
            terminal_offset: u32,
        ) -> MappedArtifact<DWA> {
            for state in artifact.artifact_mut().states_mut() {
                let old = std::mem::take(&mut state.transitions);
                for (label, edge) in old {
                    let mapped = if label == DEFAULT_LABEL {
                        DEFAULT_LABEL
                    } else {
                        assert!(label >= 0, "{name}: unexpected negative terminal label {label}");
                        label + terminal_offset as i32
                    };
                    assert!(state.transitions.insert(mapped, edge).is_none());
                }
            }
            artifact
        }

        fn remap_terminal_labels(
            name: &str,
            mut artifact: MappedArtifact<DWA>,
            local_names: &[String],
            monolithic_names: &[String],
            prefix: &str,
        ) -> MappedArtifact<DWA> {
            let mut ids_by_name = BTreeMap::<&str, Vec<u32>>::new();
            for (id, display) in monolithic_names.iter().enumerate() {
                ids_by_name.entry(display.as_str()).or_default().push(id as u32);
            }
            let used = artifact
                .artifact()
                .states()
                .iter()
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| label != DEFAULT_LABEL)
                .collect::<BTreeSet<_>>();
            // Display names are not unique: generated JSON terminals often reuse
            // names such as `__terminal_expr_16` for several distinct IDs. Composition
            // preserves terminal order within each embedded grammar, so map the Nth local
            // occurrence of a display name to the Nth prefixed monolithic occurrence.
            let mut local_occurrence = vec![0usize; local_names.len()];
            let mut seen = BTreeMap::<&str, usize>::new();
            for (local, display) in local_names.iter().enumerate() {
                let slot = seen.entry(display.as_str()).or_default();
                local_occurrence[local] = *slot;
                *slot += 1;
            }
            let mut mapping = BTreeMap::<i32, i32>::new();
            for label in used {
                assert!(label >= 0, "terminal DWA unexpectedly contains negative label {label}");
                let local = label as usize;
                let display = local_names
                    .get(local)
                    .unwrap_or_else(|| panic!("{name}: terminal label {label} outside display names"));
                let expected = format!("{prefix}{display}");
                let candidates = ids_by_name
                    .get(expected.as_str())
                    .unwrap_or_else(|| panic!("{name}: monolithic terminal not found for local {label} {display:?}, expected {expected:?}"));
                let occurrence = local_occurrence[local];
                let mapped = *candidates.get(occurrence).unwrap_or_else(|| {
                    panic!("{name}: local occurrence {occurrence} of {display:?} has no corresponding monolithic ID; candidates={candidates:?}")
                });
                mapping.insert(label, mapped as i32);
            }
            let mut default_rows = 0usize;
            for state in artifact.artifact_mut().states_mut() {
                if state.transitions.contains_key(&DEFAULT_LABEL) {
                    default_rows += 1;
                }
                let old = std::mem::take(&mut state.transitions);
                for (label, edge) in old {
                    let mapped = if label == DEFAULT_LABEL {
                        DEFAULT_LABEL
                    } else {
                        *mapping.get(&label).unwrap_or_else(|| panic!("{name}: no terminal mapping for explicit label {label}"))
                    };
                    assert!(
                        state.transitions.insert(mapped, edge).is_none(),
                        "{name}: terminal label remap merged distinct transitions onto {mapped}",
                    );
                }
            }
            eprintln!("MINBOUND label_map name={name} used={} default_rows={} prefix={prefix:?}", mapping.len(), default_rows);
            artifact
        }


        fn rebase_tokenizer_state_universe(
            name: &str,
            artifact: MappedArtifact<DWA>,
            global_offset: u32,
            enclosing_reset_states: &[u32],
            monolithic_state_count: usize,
        ) -> MappedArtifact<DWA> {
            let (dwa, mut id_map) = artifact.into_parts();
            let local = id_map.tokenizer_states;
            let local_state_count = local.original_to_internal.len();
            let mut original_to_internal = vec![u32::MAX; monolithic_state_count];
            let mut internal_to_originals = vec![Vec::<u32>::new(); local.internal_to_originals.len()];

            for (local_state, &tsid) in local.original_to_internal.iter().enumerate() {
                if tsid == u32::MAX {
                    continue;
                }
                let global_state = global_offset
                    .checked_add(local_state as u32)
                    .expect("terminal-DWA experiment tokenizer-state offset overflow");
                assert!(
                    (global_state as usize) < monolithic_state_count,
                    "{name}: rebased state {global_state} lies outside monolithic tokenizer ({monolithic_state_count})",
                );
                assert_eq!(
                    original_to_internal[global_state as usize],
                    u32::MAX,
                    "{name}: duplicate local tokenizer-state embedding at global state {global_state}",
                );
                original_to_internal[global_state as usize] = tsid;
                internal_to_originals[tsid as usize].push(global_state);
            }

            // Tokenizers always start at local state zero. A composed reset epsilon-dispatches
            // into each child start state, so the child's start TSID is observable at every
            // enclosing reset coordinate as well as at its physically rebased local state.
            let start_tsid = local.original_to_internal.first().copied().unwrap_or(u32::MAX);
            assert_ne!(start_tsid, u32::MAX, "{name}: tokenizer start state is unmapped");
            for &reset in enclosing_reset_states {
                assert!((reset as usize) < monolithic_state_count);
                match original_to_internal[reset as usize] {
                    u32::MAX => {
                        original_to_internal[reset as usize] = start_tsid;
                        internal_to_originals[start_tsid as usize].push(reset);
                    }
                    existing if existing == start_tsid => {}
                    existing => panic!(
                        "{name}: reset state {reset} is already assigned to local TSID {existing}, cannot also assign start TSID {start_tsid}"
                    ),
                }
            }
            for states in &mut internal_to_originals {
                states.sort_unstable();
                states.dedup();
            }
            let representative_original_ids = internal_to_originals
                .iter()
                .map(|states| states.first().copied().unwrap_or(u32::MAX))
                .collect::<Vec<_>>();
            id_map.tokenizer_states = ManyToOneIdMap {
                original_to_internal,
                internal_to_originals,
                representative_original_ids,
            };
            eprintln!(
                "MINBOUND state_rebase name={name} local_states={local_state_count} offset={global_offset} resets={enclosing_reset_states:?} monolithic_states={monolithic_state_count}",
            );
            MappedArtifact::new(dwa, id_map)
        }

        fn union_dwas(dwas: &[DWA], id_map: &InternalIdMap) -> DWA {
            let started = Instant::now();
            let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
            let mut starts = Vec::new();
            for dwa in dwas {
                let body = nwa.append_with_body(&dwa.to_nwa());
                starts.extend(body.start_states);
            }
            nwa.set_start_states(starts);
            let raw_states = nwa.num_states();
            let raw_transitions = nwa.num_transitions();
            let dwa = determinize(&nwa).expect("component terminal DWA union determinization");
            eprintln!(
                "MINBOUND component_union inputs={} raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
                dwas.len(), raw_states, raw_transitions, dwa.num_states(), dwa.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            dwa
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        struct ResidualKey {
            mono: u32,
            component: u32,
            mono_weight: usize,
            component_weight: usize,
        }

        fn exact_weighted_difference(monolithic: &DWA, components: &DWA) -> DWA {
            let started = Instant::now();
            assert!(monolithic.is_acyclic(), "monolithic terminal DWA must be acyclic");
            assert!(components.is_acyclic(), "component terminal DWA union must be acyclic");
            assert!(
                monolithic.states().iter().all(|s| !s.transitions.contains_key(&DEFAULT_LABEL)),
                "experiment residual builder expects no DEFAULT label in monolithic terminal DWA",
            );
            let mut ops = ScopedWeightOpCache::default();
            let all = Weight::all();
            let empty = Weight::empty();
            let start_key = ResidualKey {
                mono: monolithic.start_state(),
                component: components.start_state(),
                mono_weight: all.ptr_key(),
                component_weight: all.ptr_key(),
            };
            let mut states = vec![DWAState::default()];
            let mut payloads = vec![(
                monolithic.start_state(),
                Some(components.start_state()),
                all.clone(),
                all,
            )];
            let mut ids = FxHashMap::<ResidualKey, u32>::default();
            ids.insert(start_key, 0);
            let mut queue = VecDeque::from([0u32]);
            let mut nonempty_finals = 0usize;
            let mut total_final_outer_ranges = 0usize;
            let mut total_final_token_ranges = 0usize;

            while let Some(out_state) = queue.pop_front() {
                let (mono_state, component_state, mono_prefix, component_prefix) =
                    payloads[out_state as usize].clone();
                let mono_row = &monolithic.states()[mono_state as usize];
                if let Some(mono_final) = mono_row.final_weight.as_ref() {
                    let mono_accept = ops.intersection(&mono_prefix, mono_final);
                    let component_accept = component_state
                        .and_then(|state| components.states()[state as usize].final_weight.as_ref())
                        .map(|final_weight| ops.intersection(&component_prefix, final_weight))
                        .unwrap_or_else(Weight::empty);
                    let residual = ops.difference(&mono_accept, &component_accept);
                    if !residual.is_empty() {
                        nonempty_finals += 1;
                        total_final_outer_ranges += residual.raw_range_values().count();
                        total_final_token_ranges += residual
                            .raw_range_values()
                            .map(|(_, tokens)| tokens.ranges().count())
                            .sum::<usize>();
                        states[out_state as usize].final_weight = Some(residual);
                    }
                }

                for (&label, (mono_target, mono_edge_weight)) in &mono_row.transitions {
                    let next_mono_weight = ops.intersection(&mono_prefix, mono_edge_weight);
                    if next_mono_weight.is_empty() {
                        continue;
                    }
                    let (next_component, next_component_weight) = if let Some(component_state) = component_state {
                        let row = &components.states()[component_state as usize];
                        if let Some((target, edge_weight)) = row
                            .transitions
                            .get(&label)
                            .or_else(|| row.transitions.get(&DEFAULT_LABEL))
                        {
                            let support = ops.intersection(&component_prefix, edge_weight);
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
                    let key = ResidualKey {
                        mono: *mono_target,
                        component: next_component.unwrap_or(u32::MAX),
                        mono_weight: next_mono_weight.ptr_key(),
                        component_weight: next_component_weight.ptr_key(),
                    };
                    let target = if let Some(&target) = ids.get(&key) {
                        target
                    } else {
                        let target = states.len() as u32;
                        ids.insert(key, target);
                        states.push(DWAState::default());
                        payloads.push((
                            *mono_target,
                            next_component,
                            next_mono_weight,
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
            let minimize_started = Instant::now();
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
            let minimize_ms = minimize_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "MINBOUND residual raw_states={} raw_transitions={} raw_nonempty_finals={} final_outer_ranges={} final_token_ranges={} minimized_states={} minimized_transitions={} minimize_ms={minimize_ms:.3} total_ms={:.3}",
                raw_states,
                raw_transitions,
                nonempty_finals,
                total_final_outer_ranges,
                total_final_token_ranges,
                minimized.num_states(),
                minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }




        fn acyclic_path_stats(name: &str, dwa: &DWA) {
            fn dfs(state: u32, dwa: &DWA, memo: &mut [Option<(usize, u128)>]) -> (usize, u128) {
                if let Some(value) = memo[state as usize] { return value; }
                let row = &dwa.states()[state as usize];
                let mut max_len = if row.final_weight.as_ref().is_some_and(|w| !w.is_empty()) { 0 } else { 0 };
                let mut paths = if row.final_weight.as_ref().is_some_and(|w| !w.is_empty()) { 1u128 } else { 0 };
                for (_label, (target, _weight)) in &row.transitions {
                    let (tail_len, tail_paths) = dfs(*target, dwa, memo);
                    if tail_paths != 0 { max_len = max_len.max(1 + tail_len); }
                    paths = paths.saturating_add(tail_paths);
                }
                let value = (max_len, paths);
                memo[state as usize] = Some(value);
                value
            }
            let mut memo = vec![None; dwa.num_states() as usize];
            let (max_len, paths) = dfs(dwa.start_state(), dwa, &mut memo);
            eprintln!("MINBOUND path_stats name={name} max_terminal_length={max_len} accepting_label_paths={paths}");
        }

        fn acyclic_parser_path_stats(name: &str, dwa: &DWA) {
            assert!(dwa.is_acyclic(), "{name}: parser DWA unexpectedly cyclic");
            fn dfs(state: u32, dwa: &DWA, memo: &mut [Option<(usize, u128)>]) -> (usize, u128) {
                if let Some(value) = memo[state as usize] { return value; }
                let row = &dwa.states()[state as usize];
                let mut max_len = 0usize;
                let mut paths = if row.final_weight.as_ref().is_some_and(|w| !w.is_empty()) { 1u128 } else { 0 };
                for (_label, (target, _weight)) in &row.transitions {
                    let (tail_len, tail_paths) = dfs(*target, dwa, memo);
                    if tail_paths != 0 { max_len = max_len.max(1 + tail_len); }
                    paths = paths.saturating_add(tail_paths);
                }
                let value = (max_len, paths);
                memo[state as usize] = Some(value);
                value
            }
            let mut memo = vec![None; dwa.num_states() as usize];
            let (max_len, paths) = dfs(dwa.start_state(), dwa, &mut memo);
            let mut positive = 0usize;
            let mut negative = 0usize;
            let mut defaults = 0usize;
            for state in dwa.states() {
                for &label in state.transitions.keys() {
                    if label == DEFAULT_LABEL { defaults += 1; }
                    else if is_negative_label(label) { negative += 1; }
                    else { positive += 1; }
                }
            }
            let finals = dwa.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|w| !w.is_empty())).count();
            eprintln!(
                "MINBOUND parser_path_stats name={name} max_stack_effect_labels={max_len} accepting_label_paths={paths} final_states={finals} positive_edges={positive} negative_edges={negative} default_edges={defaults}"
            );
        }

        fn terminal_component_id(display: &str) -> u16 {
            const CHILD_PREFIX: &str = "subgrammar0::subgrammar";
            if let Some(rest) = display.strip_prefix(CHILD_PREFIX) {
                let digits = rest.chars().take_while(|ch| ch.is_ascii_digit()).collect::<String>();
                if !digits.is_empty() && rest[digits.len()..].starts_with("::") {
                    return 2 + digits.parse::<u16>().expect("schema component id");
                }
            }
            if display.starts_with("subgrammar0::") { 1 } else { 0 }
        }

        fn filter_residual_min_terminals(residual: &DWA, minimum: u8) -> DWA {
            let started = Instant::now();
            let mut states = vec![DWAState::default()];
            let mut ids = FxHashMap::<(u32, u8), u32>::default();
            let mut payloads = vec![(residual.start_state(), 0u8)];
            ids.insert((residual.start_state(), 0), 0);
            let mut queue = VecDeque::from([0u32]);
            while let Some(out) = queue.pop_front() {
                let (source, depth) = payloads[out as usize];
                let source_state = &residual.states()[source as usize];
                if depth >= minimum {
                    states[out as usize].final_weight = source_state.final_weight.clone();
                }
                for (&label, (target, weight)) in &source_state.transitions {
                    let next_depth = depth.saturating_add(1).min(minimum);
                    let key = (*target, next_depth);
                    let next = if let Some(&id) = ids.get(&key) {
                        id
                    } else {
                        let id = states.len() as u32;
                        ids.insert(key, id);
                        states.push(DWAState::default());
                        payloads.push((*target, next_depth));
                        queue.push_back(id);
                        id
                    };
                    states[out as usize].transitions.insert(label, (next, weight.clone()));
                }
            }
            let raw = DWA::from_parts(states, 0);
            let raw_states = raw.num_states();
            let raw_transitions = raw.num_transitions();
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
            eprintln!(
                "MINBOUND ideal_filter kind=min_terminals minimum={} raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
                minimum, raw_states, raw_transitions, minimized.num_states(), minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }

        fn filter_residual_cross_component(residual: &DWA, terminal_names: &[String]) -> DWA {
            const NONE: u16 = u16::MAX;
            const CROSS: u16 = u16::MAX - 1;
            let started = Instant::now();
            let mut states = vec![DWAState::default()];
            let mut ids = FxHashMap::<(u32, u16), u32>::default();
            let mut payloads = vec![(residual.start_state(), NONE)];
            ids.insert((residual.start_state(), NONE), 0);
            let mut queue = VecDeque::from([0u32]);
            while let Some(out) = queue.pop_front() {
                let (source, seen) = payloads[out as usize];
                let source_state = &residual.states()[source as usize];
                if seen == CROSS {
                    states[out as usize].final_weight = source_state.final_weight.clone();
                }
                for (&label, (target, weight)) in &source_state.transitions {
                    assert!(label >= 0 && label != DEFAULT_LABEL, "terminal residual has non-terminal label {label}");
                    let component = terminal_component_id(
                        terminal_names.get(label as usize).expect("terminal name for residual label"),
                    );
                    let next_seen = match seen {
                        NONE => component,
                        CROSS => CROSS,
                        existing if existing == component => existing,
                        _ => CROSS,
                    };
                    let key = (*target, next_seen);
                    let next = if let Some(&id) = ids.get(&key) {
                        id
                    } else {
                        let id = states.len() as u32;
                        ids.insert(key, id);
                        states.push(DWAState::default());
                        payloads.push((*target, next_seen));
                        queue.push_back(id);
                        id
                    };
                    states[out as usize].transitions.insert(label, (next, weight.clone()));
                }
            }
            let raw = DWA::from_parts(states, 0);
            let raw_states = raw.num_states();
            let raw_transitions = raw.num_transitions();
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
            eprintln!(
                "MINBOUND ideal_filter kind=cross_component raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
                raw_states, raw_transitions, minimized.num_states(), minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }

        fn filter_residual_cross_outer_component(
            residual: &DWA,
            terminal_offsets: &[u32],
        ) -> DWA {
            const NONE: u16 = u16::MAX;
            const CROSS: u16 = u16::MAX - 1;
            let started = Instant::now();
            let mut states = vec![DWAState::default()];
            let mut ids = FxHashMap::<(u32, u16), u32>::default();
            let mut payloads = vec![(residual.start_state(), NONE)];
            ids.insert((residual.start_state(), NONE), 0);
            let mut queue = VecDeque::from([0u32]);
            while let Some(out) = queue.pop_front() {
                let (source, seen) = payloads[out as usize];
                let source_state = &residual.states()[source as usize];
                if seen == CROSS {
                    states[out as usize].final_weight = source_state.final_weight.clone();
                }
                for (&label, (target, weight)) in &source_state.transitions {
                    assert!(
                        label >= 0 && label != DEFAULT_LABEL,
                        "terminal residual has non-terminal label {label}"
                    );
                    let component = terminal_offsets
                        .partition_point(|&offset| offset <= label as u32)
                        .saturating_sub(1) as u16;
                    let next_seen = match seen {
                        NONE => component,
                        CROSS => CROSS,
                        existing if existing == component => existing,
                        _ => CROSS,
                    };
                    let key = (*target, next_seen);
                    let next = if let Some(&id) = ids.get(&key) {
                        id
                    } else {
                        let id = states.len() as u32;
                        ids.insert(key, id);
                        states.push(DWAState::default());
                        payloads.push((*target, next_seen));
                        queue.push_back(id);
                        id
                    };
                    states[out as usize]
                        .transitions
                        .insert(label, (next, weight.clone()));
                }
            }
            let raw = DWA::from_parts(states, 0);
            let raw_states = raw.num_states();
            let raw_transitions = raw.num_transitions();
            let minimized =
                crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(raw);
            eprintln!(
                "MINBOUND ideal_filter kind=cross_outer_component raw_states={} raw_transitions={} states={} transitions={} ms={:.3}",
                raw_states,
                raw_transitions,
                minimized.num_states(),
                minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }

        fn residual_diagnostics(
            name: &str,
            residual: &DWA,
            id_map: &InternalIdMap,
            terminal_names: &[String],
        ) {
            fn support_counts(weight: &Weight, id_map: &InternalIdMap) -> (usize, usize, usize, usize) {
                if weight.is_empty() {
                    return (0, 0, 0, 0);
                }
                let mut tsids = BTreeSet::<u32>::new();
                let mut raw_states = BTreeSet::<u32>::new();
                let mut internal_tokens = BTreeSet::<u32>::new();
                let mut original_tokens = BTreeSet::<u32>::new();
                for (tsid_range, tokens) in weight.raw_range_values() {
                    for tsid in tsid_range {
                        tsids.insert(tsid);
                        if let Some(states) = id_map.tokenizer_states.internal_to_originals.get(tsid as usize) {
                            raw_states.extend(states.iter().copied());
                        }
                    }
                    for token_range in tokens.ranges() {
                        for token in token_range {
                            internal_tokens.insert(token);
                            if let Some(originals) = id_map.vocab_tokens.internal_to_originals.get(token as usize) {
                                original_tokens.extend(originals.iter().copied());
                            }
                        }
                    }
                }
                (tsids.len(), raw_states.len(), internal_tokens.len(), original_tokens.len())
            }

            let total_weight = Weight::union_all(
                residual.states().iter().filter_map(|state| state.final_weight.as_ref()),
            );
            let total = support_counts(&total_weight, id_map);
            let start = &residual.states()[residual.start_state() as usize];
            let mut one_step = Vec::<(i32, Weight)>::new();
            for (&label, (target, edge_weight)) in &start.transitions {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if let Some(final_weight) = residual.states()[*target as usize].final_weight.as_ref() {
                    let support = edge_weight.intersection(final_weight);
                    if !support.is_empty() {
                        one_step.push((label, support));
                    }
                }
            }
            let one_weight = Weight::union_all(one_step.iter().map(|(_, weight)| weight));
            let one = support_counts(&one_weight, id_map);
            eprintln!(
                "MINBOUND support name={name} total_tsids={} total_raw_states={} total_internal_tokens={} total_original_tokens={} one_terminal_labels={} one_terminal_tsids={} one_terminal_raw_states={} one_terminal_internal_tokens={} one_terminal_original_tokens={}",
                total.0, total.1, total.2, total.3,
                one_step.len(), one.0, one.1, one.2, one.3,
            );
            let mut one_details = one_step
                .iter()
                .map(|(label, weight)| {
                    let counts = support_counts(weight, id_map);
                    let display = terminal_names
                        .get(*label as usize)
                        .map(String::as_str)
                        .unwrap_or("<unknown>");
                    (*label, display.to_string(), counts)
                })
                .collect::<Vec<_>>();
            one_details.sort_by_key(|(_, _, counts)| std::cmp::Reverse(counts.3));
            for (label, display, counts) in one_details.into_iter().take(30) {
                eprintln!(
                    "MINBOUND one_terminal name={name} terminal={} display={:?} tsids={} raw_states={} internal_tokens={} original_tokens={}",
                    label, display, counts.0, counts.1, counts.2, counts.3,
                );
            }
        }

        fn parser_from_residual(
            name: &str,
            residual: &DWA,
            full: &Constraint,
            common_id_map: &InternalIdMap,
            vocab: &Vocab,
        ) {
            let mut selected = vec![false; full.table.num_terminals as usize];
            for state in residual.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < selected.len() {
                        selected[label as usize] = true;
                    }
                }
            }
            let active = selected.iter().filter(|&&value| value).count();
            let grammar = analyzed(full);
            let template_started = Instant::now();
            let (templates, _, _) = build_composition_templates(&full.table, &grammar, &selected);
            let template_ms = template_started.elapsed().as_secs_f64() * 1000.0;
            let parser_started = Instant::now();
            let parser = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &full.table,
                &grammar,
                &TerminalAutomaton::Dwa(residual.clone()),
                &templates,
                vocab,
                common_id_map,
                false,
            );
            let parser_ms = parser_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "MINBOUND parser name={name} active_terminals={} residual_states={} residual_transitions={} template_ms={template_ms:.3} parser_ms={parser_ms:.3} parser_states={} parser_transitions={}",
                active,
                residual.num_states(),
                residual.num_transitions(),
                parser.num_states(),
                parser.num_transitions(),
            );
        }

        /// Keep exactly those weighted terminal paths whose composed LR stack-effect
        /// relation is nonempty.  This is stronger than the pair/trigram CFG filters:
        /// it asks the same parser-template semantics used to build the boundary parser.
        ///
        /// The residual is acyclic, so compute parser-domain lanes backwards.  A lane
        /// `(terminal_state, root)` denotes suffixes from `terminal_state` whose exact
        /// parser-stack preimage is `root`.  Reifying those lanes as an NWA preserves
        /// path distinctions that would be lost by merely pruning individual edges.
        fn parser_domain_tighten_residual(
            name: &str,
            residual: &DWA,
            table: &crate::compiler::glr::table::GLRTable,
            templates: &Templates,
        ) -> DWA {
            assert!(residual.is_acyclic(), "{name}: grammar tightening expects acyclic residual");
            let started = Instant::now();
            let n = residual.num_states() as usize;

            let mut indegree = vec![0usize; n];
            for state in residual.states() {
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
                for &(target, _) in residual.states()[source as usize].transitions.values() {
                    indegree[target as usize] -= 1;
                    if indegree[target as usize] == 0 {
                        queue.push_back(target);
                    }
                }
            }
            assert_eq!(topo.len(), n, "{name}: residual unexpectedly cyclic");

            let mut seen_terminal = vec![false; table.num_terminals as usize];
            let mut non_skip_terminal = vec![false; table.num_terminals as usize];
            let mut skip_states_by_terminal = vec![Vec::<u32>::new(); table.num_terminals as usize];
            for (source, row) in table.action.iter().enumerate() {
                for (terminal, action) in row {
                    let Some(seen) = seen_terminal.get_mut(terminal as usize) else { continue };
                    *seen = true;
                    match action {
                        Action::Skip => skip_states_by_terminal[terminal as usize].push(source as u32),
                        _ => non_skip_terminal[terminal as usize] = true,
                    }
                }
            }
            let pure_skip = (0..table.num_terminals as usize)
                .map(|terminal| seen_terminal[terminal] && !non_skip_terminal[terminal])
                .collect::<Vec<_>>();

            #[derive(Clone)]
            struct Edge {
                source_root: u32,
                label: u32,
                target_state: u32,
                target_root: u32,
                support: Weight,
            }

            let mut arena = SharedBooleanParserDomains::new();
            let mut domains = vec![BTreeMap::<u32, Weight>::new(); n];
            let mut edges_by_state = vec![Vec::<Edge>::new(); n];
            let mut bundles = FxHashMap::<u32, Arc<NWA>>::default();
            let mut preimage_cache = FxHashMap::<(u32, u32), u32>::default();
            let mut weight_ops = ScopedWeightOpCache::default();
            let mut preimage_calls = 0usize;

            for &source in topo.iter().rev() {
                let mut lanes = BTreeMap::<u32, Weight>::new();
                if let Some(final_weight) = residual.states()[source as usize].final_weight.as_ref() {
                    lanes.insert(SharedBooleanParserDomains::UNIVERSAL, final_weight.clone());
                }
                for (&label, (target, edge_weight)) in &residual.states()[source as usize].transitions {
                    assert!(label >= 0 && label != DEFAULT_LABEL, "{name}: non-terminal residual label {label}");
                    let terminal = label as u32;
                    for (&target_root, target_support) in &domains[*target as usize] {
                        let support = weight_ops.intersection(edge_weight, target_support);
                        if support.is_empty() {
                            continue;
                        }
                        let source_root = if let Some(&root) = preimage_cache.get(&(terminal, target_root)) {
                            root
                        } else {
                            let root = if pure_skip.get(terminal as usize).copied().unwrap_or(false) {
                                arena.preimage_identity_skip(
                                    target_root,
                                    &skip_states_by_terminal[terminal as usize],
                                )
                            } else {
                                let bundle = if let Some(bundle) = bundles.get(&terminal) {
                                    Arc::clone(bundle)
                                } else {
                                    let bundle = build_boolean_terminal_bundle_nwa(templates, &[terminal])
                                        .unwrap_or_else(|| NWA::new(0, 0));
                                    let bundle = Arc::new(bundle);
                                    bundles.insert(terminal, Arc::clone(&bundle));
                                    bundle
                                };
                                arena.preimage_bundle(&bundle, target_root).unwrap_or(SharedBooleanParserDomains::EMPTY)
                            };
                            preimage_calls += 1;
                            preimage_cache.insert((terminal, target_root), root);
                            root
                        };
                        if source_root == SharedBooleanParserDomains::EMPTY {
                            continue;
                        }
                        if let Some(existing) = lanes.get_mut(&source_root) {
                            *existing = weight_ops.union(existing, &support);
                        } else {
                            lanes.insert(source_root, support.clone());
                        }
                        edges_by_state[source as usize].push(Edge {
                            source_root,
                            label: terminal,
                            target_state: *target,
                            target_root,
                            support,
                        });
                    }
                }
                domains[source as usize] = lanes;
            }

            let mut lane_id = FxHashMap::<(u32, u32), u32>::default();
            let mut nwa = NWA::new(0, 0);
            for (state, lanes) in domains.iter().enumerate() {
                for &root in lanes.keys() {
                    let id = nwa.add_state();
                    lane_id.insert((state as u32, root), id);
                }
            }
            let starts = domains[residual.start_state() as usize]
                .keys()
                .map(|&root| lane_id[&(residual.start_state(), root)])
                .collect::<Vec<_>>();
            nwa.set_start_states(starts);
            for (state, lanes) in domains.iter().enumerate() {
                if let Some(final_weight) = residual.states()[state].final_weight.as_ref() {
                    if lanes.contains_key(&SharedBooleanParserDomains::UNIVERSAL) {
                        let id = lane_id[&(state as u32, SharedBooleanParserDomains::UNIVERSAL)];
                        nwa.set_final_weight(id, final_weight.clone());
                    }
                }
                for edge in &edges_by_state[state] {
                    let from = lane_id[&(state as u32, edge.source_root)];
                    let to = lane_id[&(edge.target_state, edge.target_root)];
                    nwa.add_transition(from, edge.label as i32, to, edge.support.clone());
                }
            }
            let lane_states = nwa.num_states();
            let lane_transitions = nwa.num_transitions();
            let deterministic = determinize(&nwa).expect("parser-domain-tight residual determinization");
            let premin_states = deterministic.num_states();
            let premin_transitions = deterministic.num_transitions();
            let minimized = crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(deterministic);
            eprintln!(
                "MINBOUND parser_domain_tight name={name} input_states={} input_transitions={} lane_states={} lane_transitions={} start_lanes={} parser_roots={} preimage_calls={} premin_states={} premin_transitions={} states={} transitions={} ms={:.3}",
                residual.num_states(),
                residual.num_transitions(),
                lane_states,
                lane_transitions,
                domains[residual.start_state() as usize].len(),
                arena.node_count(),
                preimage_calls,
                premin_states,
                premin_transitions,
                minimized.num_states(),
                minimized.num_transitions(),
                started.elapsed().as_secs_f64() * 1000.0,
            );
            minimized
        }

        fn accepted_original_tokens(
            dwa: &DWA,
            id_map: &InternalIdMap,
        ) -> BTreeSet<u32> {
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
            originals
        }

        if std::env::var_os("GLRMASK_MINBOUND_LIVE_OUTER").is_some() {
            let root = std::env::var("GLRMASK_MINBOUND_DIR").expect("GLRMASK_MINBOUND_DIR");
            let vocab_path = std::env::var("GLRMASK_MINBOUND_VOCAB").expect("GLRMASK_MINBOUND_VOCAB");
            let mut vocab = load_vocab(&vocab_path);
            if let Ok(filter_path) = std::env::var("GLRMASK_MINBOUND_VOCAB_FILTER") {
                let allowed = fs::read_to_string(&filter_path)
                    .expect("read minimum-boundary vocab filter")
                    .split_whitespace()
                    .map(|value| value.parse::<u32>().expect("parse filtered token id"))
                    .collect::<BTreeSet<_>>();
                vocab = Vocab::new(
                    vocab
                        .entries_map()
                        .iter()
                        .filter(|(token, _)| allowed.contains(token))
                        .map(|(&token, bytes)| (token, bytes.clone()))
                        .collect(),
                );
                eprintln!("MINBOUND filtered_vocab tokens={}", vocab.entries_map().len());
            }
            let load = |name: &str| -> Constraint {
                Constraint::load(&fs::read(std::path::Path::new(&root).join(name)).unwrap()).unwrap()
            };
            let core = load("core.bin");
            let dispatch_name = std::env::var("GLRMASK_MINBOUND_DISPATCH")
                .unwrap_or_else(|_| "dispatch-literal.bin".to_string());
            let dispatch = load(&dispatch_name);
            let placeholder = terminal(&core, "PROGRAMMATIC_TOOL_SUFFIX");
            let child = CompiledSubgrammarInput {
                placeholder_terminal: placeholder,
                constraint: &dispatch,
            };
            let children = [child];
            let global_ignores = component_ignores_are_globally_erasable(&core, &children);
            let table_inputs = [SubgrammarTableInput {
                placeholder_terminal: placeholder,
                table: &dispatch.table,
                ignore_terminal: (!global_ignores).then_some(dispatch.ignore_terminal).flatten(),
                start_nullable: dispatch.table.embedded_start_nullable(),
            }];
            let outer_table_started = Instant::now();
            let mut composed_table = compose_subgrammar_tables(
                &core.table,
                (!global_ignores).then_some(core.ignore_terminal).flatten(),
                &table_inputs,
            )
            .expect("compose outer table for minimum-boundary oracle");
            eliminate_composed_runtime_controls(&mut composed_table)
                .expect("eliminate outer controls for minimum-boundary oracle");
            let outer_table_ms = outer_table_started.elapsed().as_secs_f64() * 1000.0;
            let terminal_names = merged_terminal_display_names(&core, &children);
            let tokenizer_inputs = [
                (&core.tokenizer, composed_table.terminal_offsets[0]),
                (&dispatch.tokenizer, composed_table.terminal_offsets[1]),
            ];
            let outer_tokenizer_started = Instant::now();
            let (merged_tokenizer, tokenizer_offsets) =
                Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
            let outer_tokenizer_ms = outer_tokenizer_started.elapsed().as_secs_f64() * 1000.0;
            let merged_ignores = merged_ignore_terminals(
                &core,
                &children,
                &composed_table.terminal_offsets,
                global_ignores,
            );
            eprintln!(
                "MINBOUND outer_setup global_ignores={} lr_states={} terminals={} tokenizer_states={} offsets={:?} terminal_offsets={:?}",
                global_ignores,
                composed_table.table.num_states,
                composed_table.table.num_terminals,
                merged_tokenizer.num_states(),
                tokenizer_offsets,
                composed_table.terminal_offsets,
            );
            eprintln!(
                "MINBOUND OUTER_BASE_TIMES table_ms={outer_table_ms:.3} tokenizer_ms={outer_tokenizer_ms:.3}"
            );

            let monolithic = build_terminal_dwa_parts(
                "outer_B_composed",
                &merged_tokenizer,
                &composed_table.table,
                &terminal_names,
                merged_ignores.canonical,
                &vocab,
            );
            // A and B must use the same visible-terminal semantics. When the
            // component ignores differ, the composed lexer does not erase them;
            // therefore the standalone component terminal DWAs used in A must
            // also expose IGNORE as an ordinary terminal for this oracle.
            let core_dwa = offset_terminal_labels(
                "outer_A_core",
                build_terminal_dwa_parts(
                    "outer_A_core",
                    &core.tokenizer,
                    &core.table,
                    &core.terminal_display_names,
                    global_ignores.then_some(core.ignore_terminal).flatten(),
                    &vocab,
                ),
                composed_table.terminal_offsets[0],
            );
            let dispatch_dwa = offset_terminal_labels(
                "outer_A_dispatch",
                build_terminal_dwa_parts(
                    "outer_A_dispatch",
                    &dispatch.tokenizer,
                    &dispatch.table,
                    &dispatch.terminal_display_names,
                    global_ignores.then_some(dispatch.ignore_terminal).flatten(),
                    &vocab,
                ),
                composed_table.terminal_offsets[1],
            );
            let state_count = merged_tokenizer.num_states() as usize;
            let core_dwa = rebase_tokenizer_state_universe(
                "outer_A_core",
                core_dwa,
                tokenizer_offsets[0],
                &[0],
                state_count,
            );
            let dispatch_dwa = rebase_tokenizer_state_universe(
                "outer_A_dispatch",
                dispatch_dwa,
                tokenizer_offsets[1],
                &[0],
                state_count,
            );
            let reconcile_started = Instant::now();
            let reconciled = MappedArtifact::reconcile_vec(vec![monolithic, core_dwa, dispatch_dwa]);
            let (all, common_id_map) = reconciled.into_parts();
            eprintln!(
                "MINBOUND outer_reconcile tsids={} tokens={} ms={:.3}",
                common_id_map.num_tsids(),
                common_id_map.num_internal_tokens(),
                reconcile_started.elapsed().as_secs_f64() * 1000.0,
            );
            let a = union_dwas(&all[1..], &common_id_map);
            let concrete_residual = exact_weighted_difference(&all[0], &a);
            acyclic_path_stats("outer_exact_concrete_B_minus_A", &concrete_residual);
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A",
                &concrete_residual,
                &common_id_map,
                &terminal_names,
            );
            let concrete_depth2 = filter_residual_min_terminals(&concrete_residual, 2);
            let concrete_crossed = filter_residual_cross_outer_component(
                &concrete_residual,
                &composed_table.terminal_offsets,
            );
            let concrete_non_cross =
                exact_weighted_difference(&concrete_residual, &concrete_crossed);
            let concrete_non_cross_nonempty = concrete_non_cross
                .states()
                .iter()
                .filter(|state| {
                    state
                        .final_weight
                        .as_ref()
                        .is_some_and(|weight| !weight.is_empty())
                })
                .count();
            eprintln!(
                "MINBOUND OUTER_CONCRETE_DECOMP depth_ge_2_states={} depth_ge_2_transitions={} crossed_states={} crossed_transitions={} non_cross_states={} non_cross_transitions={} non_cross_nonempty_finals={}",
                concrete_depth2.num_states(),
                concrete_depth2.num_transitions(),
                concrete_crossed.num_states(),
                concrete_crossed.num_transitions(),
                concrete_non_cross.num_states(),
                concrete_non_cross.num_transitions(),
                concrete_non_cross_nonempty,
            );
            eprintln!(
                "MINBOUND OUTER_CONCRETE_RESULT states={} transitions={} labels={}",
                concrete_residual.num_states(),
                concrete_residual.num_transitions(),
                concrete_residual.states().iter().flat_map(|state| state.transitions.keys().copied()).filter(|&label| label >= 0).collect::<BTreeSet<_>>().len(),
            );

            let concrete_grammar = AnalyzedGrammar::from_composed_rules(
                composed_table.table.rules.clone(),
                composed_table.table.num_terminals,
                terminal_names.clone(),
                composed_table.table.nonterminal_display_names.clone(),
                composed_table
                    .table
                    .rules
                    .first()
                    .expect("outer composed table has augmented start")
                    .lhs,
            );

            // Exact bounded grammar-factor oracle.  Unlike the trigram proxy
            // below, this recognizes arbitrary-length terminal factors lazily
            // and is explored only along the finite concrete B-A residual.
            // Parser controls and scoped/global skip terminals are invisible
            // grammar symbols for this necessary-condition filter.
            let mut factor_zero_width = composed_table.table.control_terminals.clone();
            factor_zero_width.extend(composed_table.table.skip_terminals.iter().copied());
            factor_zero_width.extend(merged_ignores.scoped.iter().map(|terminal| terminal as u32));
            let concrete_factor_tight = mb_filter_exact_factor_lazy(
                &concrete_residual,
                &concrete_grammar,
                &factor_zero_width,
            );
            acyclic_path_stats(
                "outer_exact_concrete_B_minus_A_lazy_grammar_factor",
                &concrete_factor_tight,
            );
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A_lazy_grammar_factor",
                &concrete_factor_tight,
                &common_id_map,
                &terminal_names,
            );
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A_lazy_grammar_factor.cap")
                    .to_str()
                    .unwrap(),
                &MappedArtifact::new(concrete_factor_tight.clone(), common_id_map.clone()),
                &terminal_names,
            );

            let concrete_weight = Weight::union_all(
                concrete_residual
                    .states()
                    .iter()
                    .filter_map(|state| state.final_weight.as_ref()),
            );
            let mut concrete_original_tokens = BTreeSet::<u32>::new();
            for (_, internal_tokens) in concrete_weight.raw_range_values() {
                for range in internal_tokens.ranges() {
                    for internal_token in range {
                        if let Some(originals) = common_id_map
                            .vocab_tokens
                            .internal_to_originals
                            .get(internal_token as usize)
                        {
                            concrete_original_tokens.extend(originals.iter().copied());
                        }
                    }
                }
            }
            eprintln!("MINBOUND OUTER_CONCRETE_TOKENS {:?}", concrete_original_tokens);
            let concrete_capture = MappedArtifact::new(concrete_residual.clone(), common_id_map.clone());
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A.cap")
                    .to_str()
                    .unwrap(),
                &concrete_capture,
                &terminal_names,
            );

            // Exact grammar/parser-domain tightening on the concrete terminal labels.
            // Unlike the trigram diagnostic below, this uses the same LR stack-effect
            // templates as boundary parser construction and preserves path distinctions.
            let mut concrete_selected = vec![false; composed_table.table.num_terminals as usize];
            for state in concrete_residual.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < concrete_selected.len() {
                        concrete_selected[label as usize] = true;
                    }
                }
            }
            let concrete_template_started = Instant::now();
            let (concrete_templates, _, _) = build_composition_templates(
                &composed_table.table,
                &concrete_grammar,
                &concrete_selected,
            );
            let concrete_template_ms = concrete_template_started.elapsed().as_secs_f64() * 1000.0;
            let concrete_grammar_tight = parser_domain_tighten_residual(
                "outer_exact_concrete_B_minus_A",
                &concrete_residual,
                &composed_table.table,
                &concrete_templates,
            );
            acyclic_path_stats(
                "outer_exact_concrete_B_minus_A_parser_domain_tight",
                &concrete_grammar_tight,
            );
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A_parser_domain_tight",
                &concrete_grammar_tight,
                &common_id_map,
                &terminal_names,
            );

            // The grammar-factor and parser-stack-domain filters are distinct
            // necessary conditions.  Measure their exact weighted overlap
            // rather than assuming either subsumes the other.
            let factor_minus_parser =
                exact_weighted_difference(&concrete_factor_tight, &concrete_grammar_tight);
            let parser_minus_factor =
                exact_weighted_difference(&concrete_grammar_tight, &concrete_factor_tight);
            let concrete_ideal =
                exact_weighted_difference(&concrete_factor_tight, &factor_minus_parser);
            let factor_general_minimized = minimize_owned(concrete_factor_tight.clone());
            let concrete_ideal_minimized = minimize_owned(concrete_ideal.clone());
            let factor_accepted_tokens =
                accepted_original_tokens(&factor_general_minimized, &common_id_map);
            let ideal_accepted_tokens =
                accepted_original_tokens(&concrete_ideal_minimized, &common_id_map);
            let nonempty_finals = |dwa: &DWA| {
                dwa.states()
                    .iter()
                    .filter(|state| {
                        state
                            .final_weight
                            .as_ref()
                            .is_some_and(|weight| !weight.is_empty())
                    })
                    .count()
            };
            eprintln!(
                "MINBOUND OUTER_CONCRETE_FILTER_RELATION factor_minus_parser_states={} factor_minus_parser_transitions={} factor_minus_parser_finals={} parser_minus_factor_states={} parser_minus_factor_transitions={} parser_minus_factor_finals={} intersection_states={} intersection_transitions={} factor_general_states={} factor_general_transitions={} factor_accepted_tokens={} ideal_general_states={} ideal_general_transitions={} ideal_accepted_tokens={}",
                factor_minus_parser.num_states(),
                factor_minus_parser.num_transitions(),
                nonempty_finals(&factor_minus_parser),
                parser_minus_factor.num_states(),
                parser_minus_factor.num_transitions(),
                nonempty_finals(&parser_minus_factor),
                concrete_ideal.num_states(),
                concrete_ideal.num_transitions(),
                factor_general_minimized.num_states(),
                factor_general_minimized.num_transitions(),
                factor_accepted_tokens.len(),
                concrete_ideal_minimized.num_states(),
                concrete_ideal_minimized.num_transitions(),
                ideal_accepted_tokens.len(),
            );
            acyclic_path_stats(
                "outer_exact_concrete_B_minus_A_ideal_intersection",
                &concrete_ideal_minimized,
            );
            residual_diagnostics(
                "outer_exact_concrete_B_minus_A_ideal_intersection",
                &concrete_ideal_minimized,
                &common_id_map,
                &terminal_names,
            );
            eprintln!(
                "MINBOUND OUTER_CONCRETE_IDEAL_ACCEPTED_TOKENS {:?}",
                ideal_accepted_tokens,
            );
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A_ideal_intersection.cap")
                    .to_str()
                    .unwrap(),
                &MappedArtifact::new(concrete_ideal_minimized.clone(), common_id_map.clone()),
                &terminal_names,
            );
            if std::env::var_os("GLRMASK_MINBOUND_STOP_AFTER_FACTOR_COMPARE").is_some() {
                return;
            }
            let concrete_tight_weight = Weight::union_all(
                concrete_grammar_tight
                    .states()
                    .iter()
                    .filter_map(|state| state.final_weight.as_ref()),
            );
            let mut concrete_tight_original_tokens = BTreeSet::<u32>::new();
            for (_, internal_tokens) in concrete_tight_weight.raw_range_values() {
                for range in internal_tokens.ranges() {
                    for internal_token in range {
                        if let Some(originals) = common_id_map
                            .vocab_tokens
                            .internal_to_originals
                            .get(internal_token as usize)
                        {
                            concrete_tight_original_tokens.extend(originals.iter().copied());
                        }
                    }
                }
            }
            eprintln!(
                "MINBOUND OUTER_CONCRETE_GRAMMAR_TIGHT template_ms={:.3} states={} transitions={} original_tokens={} removed_tokens={} tokens={:?}",
                concrete_template_ms,
                concrete_grammar_tight.num_states(),
                concrete_grammar_tight.num_transitions(),
                concrete_tight_original_tokens.len(),
                concrete_original_tokens.difference(&concrete_tight_original_tokens).count(),
                concrete_tight_original_tokens,
            );
            let concrete_tight_capture = MappedArtifact::new(
                concrete_grammar_tight.clone(),
                common_id_map.clone(),
            );
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_concrete_B_minus_A_parser_domain_tight.cap")
                    .to_str()
                    .unwrap(),
                &concrete_tight_capture,
                &terminal_names,
            );

            let mut tight_selected = vec![false; composed_table.table.num_terminals as usize];
            for state in concrete_grammar_tight.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < tight_selected.len() {
                        tight_selected[label as usize] = true;
                    }
                }
            }
            let tight_active_terminals = tight_selected.iter().filter(|&&yes| yes).count();
            let tight_template_started = Instant::now();
            let (tight_templates, _, _) = build_composition_templates(
                &composed_table.table,
                &concrete_grammar,
                &tight_selected,
            );
            let tight_template_ms = tight_template_started.elapsed().as_secs_f64() * 1000.0;
            let tight_parser_started = Instant::now();
            let tight_parser_raw = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &composed_table.table,
                &concrete_grammar,
                &TerminalAutomaton::Dwa(concrete_grammar_tight.clone()),
                &tight_templates,
                &vocab,
                &common_id_map,
                false,
            );
            let tight_parser_build_ms = tight_parser_started.elapsed().as_secs_f64() * 1000.0;
            let tight_parser_raw_states = tight_parser_raw.num_states();
            let tight_parser_raw_transitions = tight_parser_raw.num_transitions();
            acyclic_parser_path_stats("outer_ideal_boundary_parser_raw", &tight_parser_raw);
            let tight_parser_raw_for_union = tight_parser_raw.clone();
            let tight_hashcons_started = Instant::now();
            let tight_parser_hashconsed = reverse_hashcons_owned(tight_parser_raw);
            let tight_hashcons_ms = tight_hashcons_started.elapsed().as_secs_f64() * 1000.0;
            let tight_parser_hashcons_states = tight_parser_hashconsed.num_states();
            let tight_parser_hashcons_transitions = tight_parser_hashconsed.num_transitions();
            let tight_parser_hashconsed_for_union = tight_parser_hashconsed.clone();
            let tight_parser_hashconsed_for_acyclic = tight_parser_hashconsed.clone();
            let tight_acyclic_started = Instant::now();
            let tight_parser_acyclic =
                crate::automata::weighted_u32::minimize_acyclic::minimize_acyclic_owned(
                    tight_parser_hashconsed_for_acyclic,
                );
            let tight_acyclic_ms = tight_acyclic_started.elapsed().as_secs_f64() * 1000.0;
            let tight_minimize_started = Instant::now();
            let tight_parser_minimized = minimize_owned(tight_parser_hashconsed);
            let tight_minimize_ms = tight_minimize_started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                find_difference(&tight_parser_minimized, &tight_parser_acyclic)
                    .expect("compare parser minimizers"),
                None,
                "acyclic and general parser minimizers differ",
            );
            acyclic_parser_path_stats("outer_ideal_boundary_parser_minimized", &tight_parser_minimized);
            let tight_parser_tokens = accepted_original_tokens(&tight_parser_minimized, &common_id_map);
            eprintln!(
                "MINBOUND OUTER_CONCRETE_GRAMMAR_TIGHT_PARSER active_terminals={} template_ms={:.3} build_ms={:.3} raw_states={} raw_transitions={} hashcons_ms={:.3} hashcons_states={} hashcons_transitions={} acyclic_minimize_ms={:.3} acyclic_states={} acyclic_transitions={} minimize_ms={:.3} minimized_states={} minimized_transitions={} original_tokens={} tokens={:?}",
                tight_active_terminals,
                tight_template_ms,
                tight_parser_build_ms,
                tight_parser_raw_states,
                tight_parser_raw_transitions,
                tight_hashcons_ms,
                tight_parser_hashcons_states,
                tight_parser_hashcons_transitions,
                tight_acyclic_ms,
                tight_parser_acyclic.num_states(),
                tight_parser_acyclic.num_transitions(),
                tight_minimize_ms,
                tight_parser_minimized.num_states(),
                tight_parser_minimized.num_transitions(),
                tight_parser_tokens.len(),
                tight_parser_tokens,
            );

            // Time the two parser-union stages separately, assuming the ideal
            // terminal boundary has already been supplied.  First transport and
            // union only the cached component parser DWAs; then union that result
            // with the minimized ideal boundary parser DWA.
            let oracle_specials = merged_special_token_terminals(
                &core,
                &children,
                &composed_table.terminal_offsets,
                &composed_table.table,
                &composed_table.control_terminals,
            );
            let oracle_original_token_ids = merged_original_token_ids(&vocab, &oracle_specials);
            let oracle_parser_components = vec![
                ParserDwaComponent {
                    constraint: &core,
                    parser_state_relation: &composed_table.state_relations[0],
                    tokenizer_state_offset: tokenizer_offsets[0],
                    terminal_offset: composed_table.terminal_offsets[0],
                    composed_table: Some(&composed_table.table),
                },
                ParserDwaComponent {
                    constraint: &dispatch,
                    parser_state_relation: &composed_table.state_relations[1],
                    tokenizer_state_offset: tokenizer_offsets[1],
                    terminal_offset: composed_table.terminal_offsets[1],
                    composed_table: Some(&composed_table.table),
                },
            ];
            let oracle_default_domains = build_parser_default_domain_plan(
                &oracle_parser_components,
                composed_table.table.num_states,
            );
            let component_union_started = Instant::now();
            let (component_artifacts, _component_top_accept) =
                compose_component_parser_dwas_and_possible_matches(
                    &oracle_parser_components,
                    &composed_table.terminal_offsets,
                    &oracle_default_domains.component_domains,
                    merged_tokenizer.num_states() as usize,
                    &oracle_original_token_ids,
                    !global_ignores,
                )
                .expect("compose component parser DWAs for ideal-boundary timing");
            let component_union_ms = component_union_started.elapsed().as_secs_f64() * 1000.0;
            let component_parser_states = component_artifacts.artifact().0.num_states();
            let component_parser_transitions = component_artifacts.artifact().0.num_transitions();

            let run_final_union = |kind: &str, boundary_dwa: DWA| {
                let boundary_states = boundary_dwa.num_states();
                let boundary_transitions = boundary_dwa.num_transitions();
                let started = Instant::now();
                let union = union_boundary_parser_dwa(
                    component_artifacts.clone(),
                    MappedArtifact::new(boundary_dwa, common_id_map.clone()),
                    composed_table.table.num_states,
                )
                .unwrap_or_else(|error| panic!("union {kind} ideal boundary parser: {error}"));
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                eprintln!(
                    "MINBOUND OUTER_IDEAL_FINAL_UNION_VARIANT kind={} boundary_states={} boundary_transitions={} ms={:.3} final_states={} final_transitions={}",
                    kind,
                    boundary_states,
                    boundary_transitions,
                    elapsed_ms,
                    union.artifact().0.num_states(),
                    union.artifact().0.num_transitions(),
                );
                (elapsed_ms, union.artifact().0.num_states(), union.artifact().0.num_transitions())
            };
            if std::env::var_os("GLRMASK_MINBOUND_MINIMIZED_UNION_ONLY").is_none() {
                let _raw_union = run_final_union("raw", tight_parser_raw_for_union);
                let _hashcons_union = run_final_union("hashcons", tight_parser_hashconsed_for_union);
            }
            let (final_union_ms, final_parser_states, final_parser_transitions) =
                run_final_union("minimized", tight_parser_minimized.clone());
            eprintln!(
                "MINBOUND OUTER_IDEAL_UNION_TIMES component_union_ms={component_union_ms:.3} component_states={} component_transitions={} boundary_states={} boundary_transitions={} final_union_ms={final_union_ms:.3} final_states={} final_transitions={}",
                component_parser_states,
                component_parser_transitions,
                tight_parser_minimized.num_states(),
                tight_parser_minimized.num_transitions(),
                final_parser_states,
                final_parser_transitions,
            );

            let classes = exact_terminal_language_classes(&merged_tokenizer);
            let alias_count = classes.iter().enumerate().filter(|&(terminal, &class)| terminal as u32 != class).count();
            let b_quotient = quotient_terminal_language_labels("outer_B_language_quotient", &all[0], &classes);
            let a_quotient = quotient_terminal_language_labels("outer_A_language_quotient", &a, &classes);
            eprintln!("MINBOUND language_classes terminals={} aliases={} classes={}", classes.len(), alias_count, classes.iter().copied().collect::<BTreeSet<_>>().len());
            let reverse = exact_weighted_difference(&a_quotient, &b_quotient);
            let reverse_nonempty = reverse.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())).count();
            eprintln!(
                "MINBOUND OUTER_SUBSET_CHECK A_minus_B_states={} A_minus_B_transitions={} A_minus_B_nonempty_finals={}",
                reverse.num_states(), reverse.num_transitions(), reverse_nonempty,
            );
            let residual = exact_weighted_difference(&b_quotient, &a_quotient);
            acyclic_path_stats("outer_exact_B_minus_A", &residual);
            residual_diagnostics(
                "outer_exact_B_minus_A",
                &residual,
                &common_id_map,
                &terminal_names,
            );

            // Optional grammar-tightening pass after exact language quotient.
            // Replace each terminal occurrence in the composed CFG by its exact
            // lexer-language representative, so validity is existential over
            // aliases even when their LR actions differ.
            let mut quotient_rules = composed_table.table.rules.clone();
            for rule in &mut quotient_rules {
                for symbol in &mut rule.rhs {
                    if let Symbol::Terminal(terminal) = symbol
                        && let Some(&class) = classes.get(*terminal as usize)
                    {
                        *terminal = class;
                    }
                }
            }
            let quotient_augmented_start = quotient_rules
                .first()
                .expect("quotient grammar has augmented start")
                .lhs;
            let quotient_grammar = AnalyzedGrammar::from_composed_rules(
                quotient_rules,
                composed_table.table.num_terminals,
                terminal_names.clone(),
                composed_table.table.nonterminal_display_names.clone(),
                quotient_augmented_start,
            );
            let quotient_controls = composed_table
                .control_terminals
                .iter()
                .map(|terminal| classes.get(*terminal as usize).copied().unwrap_or(*terminal))
                .collect::<BTreeSet<_>>();
            let candidate_trigrams = mb_candidate_trigrams(&residual);
            let valid_trigrams = mb_valid_candidate_trigrams(
                &quotient_grammar,
                &quotient_controls,
                &candidate_trigrams,
            );
            let grammar_tight = mb_filter_valid_trigrams(&residual, &valid_trigrams);
            acyclic_path_stats("outer_exact_B_minus_A_grammar_tight", &grammar_tight);
            let grammar_tight_labels = grammar_tight
                .states()
                .iter()
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| label >= 0)
                .collect::<BTreeSet<_>>();
            eprintln!(
                "MINBOUND OUTER_GRAMMAR_TIGHT states={} transitions={} labels={} candidate_trigrams={} valid_trigrams={}",
                grammar_tight.num_states(), grammar_tight.num_transitions(), grammar_tight_labels.len(), candidate_trigrams.len(), valid_trigrams.len(),
            );
            let labels = residual
                .states()
                .iter()
                .flat_map(|state| state.transitions.keys().copied())
                .filter(|&label| label >= 0)
                .collect::<BTreeSet<_>>();
            eprintln!(
                "MINBOUND OUTER_EXACT_RESULT A_states={} A_transitions={} B_states={} B_transitions={} residual_states={} residual_transitions={} residual_labels={}",
                a_quotient.num_states(),
                a_quotient.num_transitions(),
                b_quotient.num_states(),
                b_quotient.num_transitions(),
                residual.num_states(),
                residual.num_transitions(),
                labels.len(),
            );
            let augmented_start = composed_table
                .table
                .rules
                .first()
                .expect("outer composed table has augmented start")
                .lhs;
            let grammar = AnalyzedGrammar::from_composed_rules(
                composed_table.table.rules.clone(),
                composed_table.table.num_terminals,
                terminal_names.clone(),
                composed_table.table.nonterminal_display_names.clone(),
                augmented_start,
            );
            let mut selected = vec![false; composed_table.table.num_terminals as usize];
            for state in residual.states() {
                for &label in state.transitions.keys() {
                    if label >= 0 && (label as usize) < selected.len() {
                        selected[label as usize] = true;
                    }
                }
            }
            let active = selected.iter().filter(|&&yes| yes).count();
            let template_started = Instant::now();
            let (templates, _, _) = build_composition_templates(
                &composed_table.table,
                &grammar,
                &selected,
            );
            let template_ms = template_started.elapsed().as_secs_f64() * 1000.0;
            let parser_started = Instant::now();
            let parser = build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
                &composed_table.table,
                &grammar,
                &TerminalAutomaton::Dwa(residual.clone()),
                &templates,
                &vocab,
                &common_id_map,
                false,
            );
            let parser_ms = parser_started.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "MINBOUND OUTER_EXACT_PARSER active_terminals={} template_ms={:.3} parser_ms={:.3} parser_states={} parser_transitions={}",
                active,
                template_ms,
                parser_ms,
                parser.num_states(),
                parser.num_transitions(),
            );
            let depth2 = filter_residual_min_terminals(&residual, 2);
            let crossed = filter_residual_cross_component(&residual, &terminal_names);
            eprintln!(
                "MINBOUND OUTER_EXACT_DECOMP depth_ge_2_states={} depth_ge_2_transitions={} crossed_states={} crossed_transitions={}",
                depth2.num_states(),
                depth2.num_transitions(),
                crossed.num_states(),
                crossed.num_transitions(),
            );
            let capture = MappedArtifact::new(residual, common_id_map);
            save_terminal_capture(
                std::path::Path::new(&root)
                    .join("tdwa_outer_exact_B_minus_A.cap")
                    .to_str()
                    .unwrap(),
                &capture,
                &terminal_names,
            );
            return;
        }

        let root = std::env::var("GLRMASK_MINBOUND_DIR").expect("GLRMASK_MINBOUND_DIR");
        let vocab_path = std::env::var("GLRMASK_MINBOUND_VOCAB").expect("GLRMASK_MINBOUND_VOCAB");
        let vocab = load_vocab(&vocab_path);
        let load = |name: &str| -> Constraint {
            Constraint::load(&fs::read(format!("{root}\\{name}")).unwrap()).unwrap()
        };
        // All terminal DWAs are now captured. The subtraction phase itself does
        // not invoke the terminal compiler at all.
        let (monolithic_artifact, full_terminal_names) =
            load_terminal_capture(&format!("{root}\\tdwa_monolithic.cap"));
        acyclic_path_stats("monolithic", monolithic_artifact.artifact());
        let monolithic_state_count = monolithic_artifact.id_map().tokenizer_states.original_to_internal.len();
        eprintln!(
            "MINBOUND capture monolithic terminals={} states={} transitions={} raw_tokenizer_states={}",
            full_terminal_names.len(),
            monolithic_artifact.artifact().num_states(),
            monolithic_artifact.artifact().num_transitions(),
            monolithic_state_count,
        );

        // Read all primitive state counts first. The owned-parent composition layout is
        // cumulative, so this completely determines the exact embedding without loading
        // any Constraint or reconstructing a tokenizer.
        let (core_raw, core_names) = load_terminal_capture(&format!("{root}\\tdwa_core.cap"));
        let core_state_count = core_raw.id_map().tokenizer_states.original_to_internal.len();
        let (dispatch_raw, dispatch_names) = load_terminal_capture(&format!("{root}\\tdwa_dispatch.cap"));
        let dispatch_state_count = dispatch_raw.id_map().tokenizer_states.original_to_internal.len();
        let mut schema_raw = Vec::new();
        let mut schema_state_counts = Vec::new();
        for index in 0..10 {
            let pair = load_terminal_capture(&format!("{root}\\tdwa_schema_{index}.cap"));
            schema_state_counts.push(pair.0.id_map().tokenizer_states.original_to_internal.len());
            schema_raw.push(pair);
        }
        let schema_states_sum = schema_state_counts.iter().sum::<usize>();
        assert_eq!(
            core_state_count + 1 + dispatch_state_count + schema_states_sum,
            monolithic_state_count,
            "primitive tokenizer states plus the dispatcher's fresh reset must exactly cover the monolithic tokenizer",
        );
        // Outer owned-parent composition keeps the core at offset zero. The nested
        // dispatcher composition contributes one fresh dispatcher-reset state followed
        // by its primitive parent tokenizer and then schema tokenizers in order.
        let dispatch_reset = core_state_count as u32;
        let dispatch_parent_offset = dispatch_reset + 1;
        eprintln!(
            "MINBOUND tokenizer_layout core={} dispatch_reset={} dispatch_parent={} schemas={:?} monolithic={}",
            core_state_count, dispatch_reset, dispatch_state_count, schema_state_counts, monolithic_state_count,
        );

        let mut artifacts = Vec::<MappedArtifact<DWA>>::new();
        artifacts.push(monolithic_artifact);

        let core = remap_terminal_labels("core", core_raw, &core_names, &full_terminal_names, "");
        artifacts.push(rebase_tokenizer_state_universe(
            "core", core, 0, &[0], monolithic_state_count,
        ));

        // `tdwa_dispatch.cap` is the primitive dispatcher-parent compile, captured
        // before its subgrammars are composed. It therefore begins immediately after
        // the nested dispatcher's fresh reset.
        let dispatch = remap_terminal_labels(
            "dispatch_parent", dispatch_raw, &dispatch_names, &full_terminal_names, "subgrammar0::",
        );
        artifacts.push(rebase_tokenizer_state_universe(
            "dispatch_parent",
            dispatch,
            dispatch_parent_offset,
            &[0, dispatch_reset],
            monolithic_state_count,
        ));

        let mut schema_offset = dispatch_parent_offset + dispatch_state_count as u32;
        for (index, ((artifact, names), &state_count)) in schema_raw
            .into_iter()
            .zip(schema_state_counts.iter())
            .enumerate()
        {
            let artifact = remap_terminal_labels(
                &format!("schema_{index}"),
                artifact,
                &names,
                &full_terminal_names,
                &format!("subgrammar0::subgrammar{index}::"),
            );
            artifacts.push(rebase_tokenizer_state_universe(
                &format!("schema_{index}"),
                artifact,
                schema_offset,
                &[0, dispatch_reset],
                monolithic_state_count,
            ));
            schema_offset += state_count as u32;
        }
        assert_eq!(schema_offset as usize, monolithic_state_count);

        let reconcile_started = Instant::now();
        let reconciled = MappedArtifact::reconcile_vec(artifacts);
        let (all, common_id_map) = reconciled.into_parts();
        eprintln!(
            "MINBOUND reconcile artifacts={} common_tsids={} common_tokens={} ms={:.3}",
            all.len(), common_id_map.num_tsids(), common_id_map.num_internal_tokens(),
            reconcile_started.elapsed().as_secs_f64() * 1000.0,
        );
        assert_eq!(all.len(), 13);
        let monolithic = &all[0];
        let build_parser = std::env::var_os("GLRMASK_MINBOUND_BUILD_PARSER").is_some();
        let all_components_residual = {
            let monolithic = &all[0];
            let run_variant = |name: &str, indices: &[usize]| -> DWA {
                eprintln!("MINBOUND variant={name} indices={indices:?}");
                let components = indices.iter().map(|&index| all[index].clone()).collect::<Vec<_>>();
                let union = union_dwas(&components, &common_id_map);
                let residual = exact_weighted_difference(monolithic, &union);
                let nonempty_final_states = residual
                    .states()
                    .iter()
                    .filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()))
                    .count();
                let labels = residual
                    .states()
                    .iter()
                    .flat_map(|state| state.transitions.keys().copied())
                    .filter(|&label| label >= 0)
                    .collect::<BTreeSet<_>>();
                eprintln!(
                    "MINBOUND variant_result name={name} component_union_states={} component_union_transitions={} residual_states={} residual_transitions={} residual_final_states={} residual_labels={}",
                    union.num_states(), union.num_transitions(), residual.num_states(), residual.num_transitions(), nonempty_final_states, labels.len(),
                );
                if std::env::var_os("GLRMASK_MINBOUND_VERBOSE_DIAG").is_some() { residual_diagnostics(name, &residual, &common_id_map, &full_terminal_names); }
                residual
            };

            // Partial controls are useful diagnostics; the last variant is the exact
            // primitive-component subtraction requested by the experiment.
            let _ = run_variant("core_plus_dispatch_parent", &[1, 2]);
            let _ = run_variant("core_plus_individual_schemas", &[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
            run_variant("all_primitive_components", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        };

        drop(all);
        let full_for_factor = load("full.bin");
        let full_grammar = analyzed(&full_for_factor);
        let trigram_candidates = mb_candidate_trigrams(&all_components_residual);
        let valid_trigrams = mb_valid_candidate_trigrams(
            &full_grammar,
            &full_for_factor.table.control_terminals,
            &trigram_candidates,
        );
        let trigram_filtered = mb_filter_valid_trigrams(&all_components_residual, &valid_trigrams);
        acyclic_path_stats("trigram_filtered", &trigram_filtered);
        eprintln!(
            "MINBOUND substring_result name=valid_trigrams states={} transitions={} final_states={} labels={}",
            trigram_filtered.num_states(), trigram_filtered.num_transitions(),
            trigram_filtered.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())).count(),
            trigram_filtered.states().iter().flat_map(|state| state.transitions.keys().copied()).filter(|&label| label >= 0).collect::<BTreeSet<_>>().len(),
        );

        let depth_ge_2 = filter_residual_min_terminals(&all_components_residual, 2);
        let cross_component_only = filter_residual_cross_component(&all_components_residual, &full_terminal_names);
        for (name, dwa) in [("depth_ge_2_substring_proxy", &depth_ge_2), ("cross_component_only", &cross_component_only)] {
            let labels = dwa.states().iter().flat_map(|state| state.transitions.keys().copied()).filter(|&label| label >= 0).collect::<BTreeSet<_>>();
            let finals = dwa.states().iter().filter(|state| state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty())).count();
            eprintln!("MINBOUND ideal_result name={name} states={} transitions={} final_states={} labels={}", dwa.num_states(), dwa.num_transitions(), finals, labels.len());
        }

        if build_parser {
            drop(trigram_candidates);
            drop(valid_trigrams);
            drop(depth_ge_2);
            drop(cross_component_only);
            drop(all_components_residual);
            parser_from_residual("valid_trigrams", &trigram_filtered, &full_for_factor, &common_id_map, &vocab);
        }
    }

}
