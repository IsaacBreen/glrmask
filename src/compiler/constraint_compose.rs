//! Composition of already-compiled constraint artifacts.
//!
//! The expensive component parser DWAs are reused exactly. Their private
//! `(tokenizer-state class, vocabulary-token class)` coordinates are first
//! reconciled through the merged raw tokenizer and the shared original
//! vocabulary. Parser-state labels are transported through the table splice's
//! one-to-many relation. Default parser labels are materialized over the finite
//! component parser-state alphabet before transport, because a component
//! default must not match unrelated states from another grammar.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::lexer::tokenizer::Lexer;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted_u32::determinize::determinize;
use crate::automata::weighted_u32::dwa::DWA;
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
use crate::ds::weight::Weight;
use crate::runtime::{Constraint, ConstraintRuntimeBackend, SpecialTokenTerminal};
use crate::Vocab;

#[inline]
fn compose_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPOSE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

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

fn build_direct_component_coordinate_maps(
    components: &[ParserDwaComponent<'_>],
    merged_tokenizer_state_count: usize,
    original_token_ids: &[u32],
) -> Result<(InternalIdMap, Vec<DirectComponentCoordinateMaps>), String> {
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
                // Compaction may leave an unreferenced TSID slot. Generic
                // reconciliation maps it nowhere; preserve that behavior.
                continue;
            }
            let global_tsid = global_to_states.len() as u32;
            let mut merged_states = Vec::with_capacity(local_states.len());
            for &local_state in local_states {
                let merged_state = component
                    .tokenizer_state_offset
                    .checked_add(local_state)
                    .ok_or_else(|| "component tokenizer-state offset overflow".to_string())?;
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
                *slot = global_tsid;
                merged_states.push(merged_state);
            }
            state_representatives.push(merged_states[0]);
            global_to_states.push(merged_states);
            local_map[local_tsid].push(global_tsid);
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
        // Merged raw state zero epsilon-dispatches to every component start.
        // Its common-refinement tuple therefore contains each local start TSID.
        start_targets.push(0);
        start_targets.sort_unstable();
        start_targets.dedup();
        local_to_global_tsids.push(local_map);
    }
    if state_to_global.iter().any(|&tsid| tsid == u32::MAX) {
        return Err("direct component state map does not cover the merged tokenizer".into());
    }

    let original_token_count = original_token_ids
        .last()
        .map_or(0, |token| *token as usize + 1);
    let mut token_to_global = vec![u32::MAX; original_token_count];
    let mut global_to_tokens = Vec::<Vec<u32>>::new();
    let mut token_representatives = Vec::<u32>::new();
    let mut tuple_to_global = FxHashMap::<Vec<u32>, u32>::default();
    for &original in original_token_ids {
        let tuple = components
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
        if tuple.iter().all(|&local| local == u32::MAX) {
            continue;
        }
        let next = global_to_tokens.len() as u32;
        let global = *tuple_to_global.entry(tuple).or_insert_with(|| {
            global_to_tokens.push(Vec::new());
            token_representatives.push(original);
            next
        });
        token_to_global[original as usize] = global;
        global_to_tokens[global as usize].push(original);
    }

    let mut component_maps = Vec::with_capacity(components.len());
    for (component_index, component) in components.iter().enumerate() {
        let local_token_count = component.constraint.internal_token_to_tokens.len();
        let mut local_to_global_tokens = vec![Vec::<u32>::new(); local_token_count];
        for (global, originals) in global_to_tokens.iter().enumerate() {
            let Some(&representative) = originals.first() else {
                continue;
            };
            let local = component
                .constraint
                .original_token_to_internal
                .get(representative as usize)
                .copied()
                .unwrap_or(u32::MAX);
            if local == u32::MAX {
                continue;
            }
            let Some(destinations) = local_to_global_tokens.get_mut(local as usize) else {
                return Err(format!(
                    "component {component_index} token class {local} lies outside its internal token domain"
                ));
            };
            destinations.push(global as u32);
        }
        component_maps.push(DirectComponentCoordinateMaps {
            local_to_global_tsids: std::mem::take(&mut local_to_global_tsids[component_index]),
            local_to_global_tokens,
        });
    }

    Ok((
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap {
                original_to_internal: state_to_global,
                internal_to_originals: global_to_states,
                representative_original_ids: state_representatives,
            },
            vocab_tokens: ManyToOneIdMap {
                original_to_internal: token_to_global,
                internal_to_originals: global_to_tokens,
                representative_original_ids: token_representatives,
            },
            deferred_vocab_singleton_original_ids: None,
        },
        component_maps,
    ))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ResidualScanResult {
    /// Per-start-state longest matches. Different residual starts may produce
    /// different valid widths for the same terminal.
    matches: BTreeSet<(u32, usize)>,
    future_terminals: BTreeSet<u32>,
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
    let mut node_ids = BTreeMap::<MixedTokenNodeKey, usize>::new();
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

    let mut good = accepting.clone();
    let mut changed = true;
    while changed {
        changed = false;
        for source in (0..nodes.len()).rev() {
            if !good[source]
                && nodes[source]
                    .outgoing
                    .iter()
                    .any(|edge| good[edge.target])
            {
                good[source] = true;
                changed = true;
            }
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
                            .entries
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
                            .entries
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

fn candidate_start_states_for_token(
    token_id: u32,
    candidate_ranges: &BTreeMap<u32, Vec<(usize, u32, u32)>>,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    reset_starts: &[u32],
) -> Vec<u32> {
    // State zero is the fresh disjoint-union reset dispatcher. Keep its
    // aggregate execution as well as the individual deterministic roots.
    let mut starts = Vec::with_capacity(reset_starts.len() + 1);
    starts.push(0);
    starts.extend_from_slice(reset_starts);
    if let Some(ranges) = candidate_ranges.get(&token_id) {
        for &(component_index, start_tsid, end_tsid) in ranges {
            let constraint = components[component_index];
            let state_offset = tokenizer_state_offsets[component_index];
            for tsid in start_tsid..=end_tsid {
                if let Some(states) = constraint.internal_tsid_to_states.get(tsid as usize) {
                    starts.extend(states.iter().map(|&state| state_offset + state));
                }
            }
        }
    }
    starts.sort_unstable();
    starts.dedup();
    starts
}

fn discover_mixed_owner_terminals(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
    terminal_offsets: &[u32],
    seed_terminals: &[bool],
    ignore_terminal: Option<u32>,
    disallowed_follows: &BTreeMap<u32, BitSet>,
) -> MixedOwnerDiscovery {
    let num_terminals = tokenizer.num_terminals() as usize;
    let owners = terminal_owners(num_terminals, terminal_offsets);
    let reset_starts = tokenizer.deterministic_reset_states().to_vec();
    let candidate_ranges =
        boundary_candidate_state_ranges_by_token(components, tokenizer_state_offsets, vocab);
    let multi_byte_entries = vocab
        .entries
        .iter()
        .filter(|(_, bytes)| bytes.len() >= 2)
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
                    let scan = scan_residual_starts(tokenizer, suffix, &reset_starts);
                    (suffix, scan)
                })
                .collect::<FxHashMap<_, _>>(),
        )
    };
    let suffix_cache_ms = suffix_cache_started_at.elapsed().as_secs_f64() * 1000.0;

    // Each model token is an independent acyclic same-token graph. Run those
    // scans in parallel, then merge in vocabulary order for deterministic
    // output and profiling.
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
                        scan_residual_starts(tokenizer, &bytes[offset..], &reset_starts)
                    })
                    .collect::<Vec<_>>();
                owned_reset_scans.iter().collect::<Vec<_>>()
            };
            let candidate_starts = candidate_start_states_for_token(
                token_id,
                &candidate_ranges,
                components,
                tokenizer_state_offsets,
                &reset_starts,
            );
            let mut starts_by_scan = BTreeMap::<ResidualScanResult, Vec<u32>>::new();
            for start in candidate_starts {
                starts_by_scan
                    .entry(scan_residual_starts(tokenizer, bytes, &[start]))
                    .or_default()
                    .push(start);
            }
            let mut local_terminals = BTreeSet::<u32>::new();
            let mut local_witnesses = Vec::new();
            for (arbitrary_scan, start_states) in starts_by_scan {
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
    if compose_profile_enabled() {
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
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    let mut relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    let mut tokens_by_byte = vec![Vec::<u32>::new(); 256];
    for (&token_id, bytes) in vocab.entries.iter().filter(|(_, bytes)| bytes.len() == 1) {
        tokens_by_byte[bytes[0] as usize].push(token_id);
    }
    let closures = tokenizer.all_singleton_epsilon_closures();
    for raw_state in 0..tokenizer.num_states() {
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
) -> BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>> {
    let mut tokens_by_byte = vec![Vec::<u32>::new(); 256];
    for (&token_id, bytes) in vocab.entries.iter().filter(|(_, bytes)| bytes.len() == 1) {
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
    let entries = (0..tokenizer.num_states() as usize)
        .into_par_iter()
        .fold(Vec::<(u32, u32, u8)>::new, |mut output, raw_state| {
            let source_closure = &closures[raw_state];
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
                            output.push((terminal, raw_state as u32, byte));
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
    relations: &mut BTreeMap<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>,
) {
    let candidate = if rayon::current_num_threads() == 1
        || std::env::var_os("GLRMASK_COMPOSE_SERIAL_ONE_BYTE_REFERENCE").is_some()
    {
        collect_one_byte_seed_relations_serial(tokenizer, vocab, seed_terminals)
    } else {
        collect_one_byte_seed_relations_parallel(tokenizer, vocab, seed_terminals)
    };
    if std::env::var_os("GLRMASK_VALIDATE_COMPOSE_ONE_BYTE_PARALLEL").is_some()
        && rayon::current_num_threads() > 1
    {
        let reference = collect_one_byte_seed_relations_serial(tokenizer, vocab, seed_terminals);
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

fn direct_boundary_terminal_automaton(
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    seed_terminals: &[bool],
    discovery: &MixedOwnerDiscovery,
    ignore_terminal: Option<u32>,
) -> Result<MappedArtifact<TerminalAutomaton>, String> {
    let total_started_at = Instant::now();
    let num_states = tokenizer.num_states() as usize;
    let mut seed_relations = BTreeMap::<Vec<u32>, BTreeMap<u32, BTreeSet<u32>>>::new();
    let one_byte_started_at = Instant::now();
    collect_one_byte_seed_relations(tokenizer, vocab, seed_terminals, &mut seed_relations);
    let one_byte_ms = one_byte_started_at.elapsed().as_secs_f64() * 1000.0;

    let selected_original_tokens = seed_relations
        .values()
        .flat_map(|by_state| by_state.values())
        .flat_map(|tokens| tokens.iter().copied())
        .chain(discovery.token_ids.iter().copied())
        .collect::<BTreeSet<_>>();
    if selected_original_tokens.is_empty() {
        return Err("boundary witness construction selected no vocabulary tokens".into());
    }
    let max_original_token = vocab.entries.keys().next_back().copied().unwrap_or(0);
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

    // Build the exact coarsest raw-state coordinate needed by this boundary
    // automaton. Two lexer states are interchangeable here iff every final
    // weight in the witness NWA assigns them the same selected-token set.
    // This avoids both unsafe reuse of parser-compacted runtime TSIDs and the
    // catastrophic one-TSID-per-raw-state fallback.
    let quotient_started_at = Instant::now();
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
    let tokenizer_states = ManyToOneIdMap::from_original_to_internal_with_representatives(
        state_to_class,
        state_representatives.len() as u32,
        state_representatives,
    );
    let quotient_ms = quotient_started_at.elapsed().as_secs_f64() * 1000.0;
    let id_map = InternalIdMap {
        tokenizer_states,
        vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };

    let relation_weight = |by_state: BTreeMap<u32, BTreeSet<u32>>| {
        let mut tokens_by_tsid = BTreeMap::<u32, BTreeSet<u32>>::new();
        for (state, originals) in by_state {
            let tsid = id_map.tokenizer_states.original_to_internal[state as usize];
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
            .map(|&state| id_map.tokenizer_states.original_to_internal[state as usize])
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
    let mut artifacts = Vec::with_capacity(components.len());
    let mut parser_nwa_ms = 0.0;
    let mut possible_matches_ms = 0.0;
    let mut remap_ms = 0.0;
    for (((component, terminal_offset), coordinate_maps), component_index) in components
        .iter()
        .zip(terminal_offsets.iter().copied())
        .zip(component_maps)
        .zip(0usize..)
    {
        let started_at = Instant::now();
        let parser_nwa = component_parser_nwa(component)?;
        parser_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        let started_at = Instant::now();
        let possible_matches = component_possible_matches(component, terminal_offset)?;
        possible_matches_ms += started_at.elapsed().as_secs_f64() * 1000.0;
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
        remap_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        if compose_profile_enabled() {
            let tsid_fanout = coordinate_maps
                .local_to_global_tsids
                .iter()
                .map(Vec::len)
                .sum::<usize>();
            let token_fanout = coordinate_maps
                .local_to_global_tokens
                .iter()
                .map(Vec::len)
                .sum::<usize>();
            eprintln!(
                "[glrmask/profile][constraint_component_remap] component={} local_tsids={} global_tsids={} tsid_fanout={} local_tokens={} global_tokens={} token_fanout={}",
                component_index,
                coordinate_maps.local_to_global_tsids.len(),
                global_tsid_count,
                tsid_fanout,
                coordinate_maps.local_to_global_tokens.len(),
                id_map.num_internal_tokens(),
                token_fanout,
            );
        }
        artifacts.push(artifact);
    }

    let union_started_at = Instant::now();
    let mut union = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let mut starts = Vec::new();
    let mut possible_matches = PossibleMatches::new();
    for (automaton, component_possible_matches) in artifacts {
        let body = union.append_with_body(&automaton);
        starts.extend(body.start_states);
        for (terminal, weight) in component_possible_matches {
            possible_matches
                .entry(terminal)
                .and_modify(|existing| *existing = existing.union(&weight))
                .or_insert(weight);
        }
    }
    union.set_start_states(starts);
    let append_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;
    let determinize_started_at = Instant::now();
    let dwa = determinize(&union).map_err(|error| error.to_string())?;
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_component_reuse] components={} global_tsids={} global_tokens={} coordinate_ms={coordinate_ms:.3} parser_nwa_ms={parser_nwa_ms:.3} possible_matches_ms={possible_matches_ms:.3} remap_ms={remap_ms:.3} append_ms={append_ms:.3} determinize_ms={determinize_ms:.3} union_nwa_states={} result_states={} total_ms={:.3}",
            components.len(),
            id_map.num_tsids(),
            id_map.num_internal_tokens(),
            union.num_states(),
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

fn union_boundary_parser_dwa(
    component_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    boundary: MappedArtifact<DWA>,
    num_parser_states: u32,
) -> Result<MappedArtifact<(DWA, PossibleMatches)>, String> {
    let total_started_at = Instant::now();
    let pair_started_at = Instant::now();
    let paired = component_artifacts.pair_forced_common(boundary);
    let pair_ms = pair_started_at.elapsed().as_secs_f64() * 1000.0;
    let (((component_dwa, possible_matches), boundary_dwa), id_map) = paired.into_parts();
    let explicit_started_at = Instant::now();
    let component_nwa = explicit_parser_nwa(&component_dwa, num_parser_states);
    let boundary_nwa = explicit_parser_nwa(&boundary_dwa, num_parser_states);
    let explicit_ms = explicit_started_at.elapsed().as_secs_f64() * 1000.0;
    let append_started_at = Instant::now();
    let mut union = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let component_body = union.append_with_body(&component_nwa);
    let boundary_body = union.append_with_body(&boundary_nwa);
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
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_parser_union] component_states={} boundary_states={} union_nwa_states={} result_states={} pair_ms={pair_ms:.3} explicit_ms={explicit_ms:.3} append_ms={append_ms:.3} determinize_ms={determinize_ms:.3} total_ms={:.3}",
            component_dwa.num_states(),
            boundary_dwa.num_states(),
            union.num_states(),
            parser_dwa.num_states(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(MappedArtifact::new(
        (parser_dwa, possible_matches),
        id_map,
    ))
}

fn build_boundary_repair(
    composed_table: &ComposedTable,
    tokenizer: &Tokenizer,
    terminal_display_names: Vec<String>,
    ignore_terminal: Option<u32>,
    vocab: &Vocab,
    components: &[&Constraint],
    tokenizer_state_offsets: &[u32],
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
    let discovery_started_at = Instant::now();
    let mixed_owner = discover_mixed_owner_terminals(
        tokenizer,
        vocab,
        components,
        tokenizer_state_offsets,
        &composed_table.terminal_offsets,
        &seed_terminals,
        ignore_terminal,
        &disallowed_follows,
    );
    let discovery_ms = discovery_started_at.elapsed().as_secs_f64() * 1000.0;
    let mixed_owner_terminals = mixed_owner.terminals.clone();
    let mut active_terminals = seed_terminals.clone();
    for terminal in mixed_owner_terminals.iter() {
        active_terminals[terminal] = true;
    }
    if !active_terminals.iter().any(|&active| active) {
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

    let ((templates, template_dfas_by_terminal, templates_ms), (terminal_dwa, terminal_ms)) =
        rayon::join(
            || {
                let started_at = Instant::now();
                let characterizations = characterize_selected_terminals(
                    &composed_table.table,
                    &analyzed,
                    &active_terminals,
                );
                let templates = Templates::from_characterizations(&characterizations);
                let mut template_dfas_by_terminal =
                    vec![None; analyzed.num_terminals as usize];
                for (&terminal, dfa) in &templates.by_terminal {
                    let commit_dfa =
                        specialize_template_dfa_defaults_for_commit_split_input(dfa);
                    if let Some(split) = try_split_commit_template_dfas(&commit_dfa)
                        && let Some(slot) =
                            template_dfas_by_terminal.get_mut(terminal as usize)
                    {
                        *slot = Some(Arc::new(split));
                    }
                }
                (
                    templates,
                    template_dfas_by_terminal,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                )
            },
            || {
                let started_at = Instant::now();
                let result = if std::env::var_os(
                    "GLRMASK_COMPOSE_GENERIC_BOUNDARY_REFERENCE",
                )
                .is_some()
                {
                    let flat_trans: Arc<[u32]> = Arc::from(
                        crate::compiler::stages::id_map_and_terminal_dwa::l1::
                            build_flat_transition_table(tokenizer),
                    );
                    let global_max_length_state_map =
                        crate::compiler::stages::id_map_and_terminal_dwa::
                            build_global_max_length_state_map(tokenizer, vocab, &flat_trans);
                    let coloring = TerminalColoring::identity(analyzed.num_terminals as usize);
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
                } else {
                    direct_boundary_terminal_automaton(
                        tokenizer,
                        vocab,
                        &seed_terminals,
                        &mixed_owner,
                        ignore_terminal,
                    )
                };
                (
                    result,
                    started_at.elapsed().as_secs_f64() * 1000.0,
                )
            },
        );
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
            "[glrmask/profile][constraint_boundary_build] active={} seed_active={} mixed_active={} mixed_tokens={} discovery_ms={discovery_ms:.3} terminal_ms={terminal_ms:.3} templates_ms={templates_ms:.3} parser_ms={parser_ms:.3} total_ms={:.3}",
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

fn finalize_composed_constraint(
    parent: &Constraint,
    children: &[CompiledSubgrammarInput<'_>],
    composed_table: ComposedTable,
    tokenizer: Tokenizer,
    tokenizer_state_offsets: Vec<u32>,
    parser_artifacts: MappedArtifact<(DWA, PossibleMatches)>,
    template_dfas_by_terminal: Vec<Option<Arc<crate::runtime::CommitTemplateDfas>>>,
    special_token_terminals: Vec<SpecialTokenTerminal>,
    embedded_end_token_ids: Vec<u32>,
    vocab: &Vocab,
) -> ConstraintComposition {
    let terminal_offsets = composed_table.terminal_offsets.clone();
    let parser_state_relations = composed_table.state_relations.clone();
    let ((parser_dwa, possible_matches), internal_ids) = parser_artifacts.into_parts();
    let internal_token_bytes = build_internal_token_bytes_from_groups(
        vocab,
        &internal_ids.vocab_tokens.internal_to_originals,
    );
    let state_to_internal_tsid = internal_ids.tokenizer_states.original_to_internal.clone();
    let internal_tsid_to_states = internal_ids.tokenizer_states.internal_to_originals_vecs();
    let original_token_to_internal = internal_ids.vocab_tokens.original_to_internal.clone();
    let internal_token_to_tokens = internal_ids.vocab_tokens.internal_to_originals_vecs();
    let ignore_terminal = merged_ignore_terminal(parent, children, &terminal_offsets);
    let terminal_display_names = merged_terminal_display_names(parent, children);
    let tokenizer_has_epsilon_transitions = tokenizer.has_epsilon_transitions();
    let mut table = composed_table.table;
    table.set_embedded_end_token_ids(&embedded_end_token_ids);
    let num_terminals = table.num_terminals as usize;
    let mut constraint = Constraint {
        runtime_backend: ConstraintRuntimeBackend::Static,
        parser_dwa,
        parser_top_accept: BTreeMap::new(),
        parser_top_accept_parts: BTreeMap::new(),
        direct_regular_l1_complete_by_terminal: BTreeMap::new(),
        direct_regular_wide_frontier_acceptance: Vec::new(),
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
        state_internal_tsid_offsets: Vec::new(),
        state_internal_tsids: Vec::new(),
        runtime_source_state_offset: None,
        runtime_product_source_offsets: Vec::new(),
        runtime_product_source_states: Vec::new(),
        runtime_product_exact_source_states: Vec::new(),
        runtime_product_state_by_source_subset: FxHashMap::default(),
        template_dfas_by_terminal,
        fast_template_dfas_by_terminal: Vec::new(),
        original_token_to_internal,
        internal_token_to_tokens,
        token_bytes: Arc::clone(&vocab.entries),
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
        tokenizer_fast_transitions: Default::default(),
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
    constraint.rebuild_runtime_caches();
    ConstraintComposition {
        constraint,
        terminal_offsets,
        tokenizer_state_offsets,
        parser_state_relations,
    }
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
    for (component_index, constraint) in std::iter::once(parent)
        .chain(children.iter().map(|child| child.constraint))
        .enumerate()
    {
        if constraint.token_bytes.as_ref() != vocab.entries.as_ref() {
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
                && vocab.entries.contains_key(&special.token_id)
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
    let tokenizer_started_at = Instant::now();
    let (tokenizer, tokenizer_state_offsets) =
        Tokenizer::disjoint_union_with_terminal_offsets(&tokenizer_inputs);
    let tokenizer_ms = tokenizer_started_at.elapsed().as_secs_f64() * 1000.0;

    let parser_components = component_constraints
        .iter()
        .enumerate()
        .map(|(index, constraint)| ParserDwaComponent {
            constraint,
            parser_state_relation: &composed_table.state_relations[index],
            tokenizer_state_offset: tokenizer_state_offsets[index],
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
        .entries
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
    let ((parser_artifacts, reuse_ms), (boundary_repair, boundary_ms)) = rayon::join(
        || {
            let started_at = Instant::now();
            let result = compose_component_parser_dwas_and_possible_matches(
                &parser_components,
                &composed_table.terminal_offsets,
                tokenizer.num_states() as usize,
                &original_token_ids,
            );
            (result, started_at.elapsed().as_secs_f64() * 1000.0)
        },
        || {
            let started_at = Instant::now();
            let result = build_boundary_repair(
                &composed_table,
                &tokenizer,
                terminal_display_names,
                ignore_terminal,
                vocab,
                &component_constraints,
                &tokenizer_state_offsets,
            );
            (result, started_at.elapsed().as_secs_f64() * 1000.0)
        },
    );
    let parser_artifacts = parser_artifacts?;
    let boundary_repair = boundary_repair?;
    let union_started_at = Instant::now();
    let (parser_artifacts, template_dfas_by_terminal) = match boundary_repair {
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
    };
    let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;

    let finalize_started_at = Instant::now();
    let result = finalize_composed_constraint(
        parent,
        children,
        composed_table,
        tokenizer,
        tokenizer_state_offsets,
        parser_artifacts,
        template_dfas_by_terminal,
        special_token_terminals,
        embedded_end_token_ids,
        vocab,
    );
    let finalize_ms = finalize_started_at.elapsed().as_secs_f64() * 1000.0;
    if compose_profile_enabled() {
        eprintln!(
            "[glrmask/profile][constraint_composition] components={} table_ms={table_ms:.3} tokenizer_ms={tokenizer_ms:.3} reuse_ms={reuse_ms:.3} boundary_ms={boundary_ms:.3} union_ms={union_ms:.3} finalize_ms={finalize_ms:.3} total_ms={:.3}",
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
            &vocab.entries.keys().copied().collect::<Vec<_>>(),
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
        let token_ids = vocab.entries.keys().copied().collect::<Vec<_>>();
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
