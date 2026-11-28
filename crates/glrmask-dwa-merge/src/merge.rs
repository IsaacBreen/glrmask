//! Merge multiple (InternalIdMap, DWA) pairs into one.
//!
//! Handles both overlapping vocabs (e.g., L1 + L2+ from the same partition)
//! and disjoint vocabs (e.g., different character-type partitions) uniformly
//! via composite-key refinement.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::Instant;

use glrmask_invariant::__private::charge_template_build_budget as charge_build_budget;
use range_set_blaze::RangeSetBlaze;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::automata::weighted::determinize::{determinize, normalize_direct_singletons_enabled};
use crate::automata::weighted::equivalence::find_difference;
use crate::automata::weighted::dwa::{DWA, DWAState};
use crate::automata::weighted::minimize::minimize_owned;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::compiler::stages::mapped_artifact::{MappedArtifact, refine_observation_classes, refine_sparse_memberships, local_to_refined_classes, WeightRefs, TokenAxisMap, remap_weights_with_token_axis};
use crate::ds::weight::{ScopedWeightOpCache, Weight};

use super::types::{LocalIdMapTerminalDwa, TerminalDwaPhaseProfile, compile_profile_enabled};

enum FastGlobalTokenMaps {
    Direct(Vec<Vec<u32>>),
    General(Vec<Vec<Vec<u32>>>),
}

fn merged_terminal_dwa_phase_enabled(variable: &str, default: bool) -> bool {
    std::env::var(variable)
        .map(|value| {
            let trimmed = value.trim();
            !trimmed.is_empty() && trimmed != "0" && !trimmed.eq_ignore_ascii_case("false")
        })
        .unwrap_or(default)
}

fn minimize_merged_terminal_dwa_enabled(default: bool) -> bool {
    merged_terminal_dwa_phase_enabled("GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA", default)
}

fn compact_merged_terminal_dwa_enabled(default: bool) -> bool {
    merged_terminal_dwa_phase_enabled("GLRMASK_COMPACT_MERGED_TERMINAL_DWA", default)
}

fn assert_direct_global_terminal_dwa_merge_equivalence_enabled() -> bool {
    std::env::var_os("GLRMASK_ASSERT_DIRECT_GLOBAL_TERMINAL_DWA_MERGE_EQUIVALENCE").is_some()
}

fn select_primary_token_nwa_tsid_source(inputs: &[LocalIdMapTerminalDwa]) -> Option<usize> {
    inputs
        .iter()
        .enumerate()
        .max_by_key(|(_, input)| input.id_map.num_tsids())
        .map(|(index, _)| index)
}

#[derive(Default)]
struct TokenNwaRemapDetail {
    unique_weights: usize,
    full_weights: usize,
    compact_entries: usize,
    local_tsid_visits: usize,
    global_tsid_expansions: usize,
    max_local_entry_span: usize,
    max_global_entry_expansion: usize,
}

fn profile_token_nwa_remap_detail(
    dwa: &DWA,
    local_to_global_tsids: &[Vec<u32>],
) -> TokenNwaRemapDetail {
    let mut detail = TokenNwaRemapDetail::default();
    let mut seen = FxHashSet::<usize>::default();
    let mut inspect = |weight: &Weight| {
        if weight.is_empty() || !seen.insert(weight.ptr_key()) {
            return;
        }
        detail.unique_weights += 1;
        if weight.is_full() {
            detail.full_weights += 1;
            detail.global_tsid_expansions += local_to_global_tsids.iter().map(Vec::len).sum::<usize>();
            return;
        }
        let Some(entries) = weight.compact_entries() else {
            return;
        };
        detail.compact_entries += entries.len();
        for (start, end, _) in entries {
            let span = end.saturating_sub(start) as usize + 1;
            detail.local_tsid_visits += span;
            detail.max_local_entry_span = detail.max_local_entry_span.max(span);
            let mut entry_expansion = 0usize;
            for local_tsid in start..=end {
                let expansion = local_to_global_tsids
                    .get(local_tsid as usize)
                    .map_or(0, Vec::len);
                detail.global_tsid_expansions += expansion;
                entry_expansion += expansion;
            }
            detail.max_global_entry_expansion = detail.max_global_entry_expansion.max(entry_expansion);
        }
    };
    for state in dwa.states() {
        if let Some(final_weight) = &state.final_weight {
            inspect(final_weight);
        }
        for (_, weight) in state.transitions.values() {
            inspect(weight);
        }
    }
    detail
}

/// Return whether each source owns a disjoint set of original vocabulary tokens.
///
/// After remapping, this is stronger than a representation detail: every
/// non-empty weight from a source is contained in that source's global token
/// domain.  Consequently intersections of contributions from different
/// sources are empty, which lets `direct_union_disjoint_token_domain_dwas`
/// distribute the weighted path intersection over the branch union exactly.
fn token_domains_disjoint<'a>(
    maps: impl IntoIterator<Item = &'a InternalIdMap>,
    original_token_count: usize,
) -> bool {
    let mut occupied = vec![false; original_token_count];
    for map in maps {
        for (original, &local) in map.vocab_tokens.original_to_internal.iter().enumerate() {
            if local != u32::MAX && std::mem::replace(&mut occupied[original], true) {
                return false;
            }
        }
    }
    true
}

fn inputs_have_disjoint_token_domains(inputs: &[LocalIdMapTerminalDwa], max_token_id: u32) -> bool {
    token_domains_disjoint(inputs.iter().map(|input| &input.id_map), max_token_id as usize + 1)
}

fn remap_dwa_with_maps(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    local_to_global_tokens: &[Vec<u32>],
    global: &InternalIdMap,
) {
    remap_dwa(dwa, local_to_global_tsids, TokenAxisMap::Explicit(local_to_global_tokens), global);
}

fn remap_dwa_with_token_offset(
    dwa: &mut DWA,
    local_to_global_tsids: &[Vec<u32>],
    token_offset: u32,
    local_token_count: u32,
    global: &InternalIdMap,
) {
    remap_dwa(dwa, local_to_global_tsids, TokenAxisMap::Offset {
        offset: token_offset, local_token_count,
    }, global);
}

fn remap_dwa(
    dwa: &mut DWA,
    tsids: &[Vec<u32>],
    tokens: TokenAxisMap<'_>,
    global: &InternalIdMap,
) {
    remap_weights_with_token_axis(
        &mut dwa.weight_refs_mut(), tsids, tokens,
        global.num_tsids() as usize, global.num_internal_tokens() as usize,
    );
    for state in dwa.states_mut() {
        if state.final_weight.as_ref().is_some_and(Weight::is_empty) {
            state.final_weight = None;
        }
        state.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
}

/// Exact disjoint-token union; equivalent to
/// ordinary NWA-union determinization, before minimization.
fn direct_union_disjoint_token_domain_dwas(inputs: &[DWA]) -> DWA {
    assert!(!inputs.is_empty());

    const DEAD: u32 = u32::MAX;

    let mut output = DWA::new(0, 0);
    let start_tuple: Vec<u32> = inputs.iter().map(DWA::start_state).collect();
    let mut state_ids = FxHashMap::<Vec<u32>, u32>::default();
    let mut worklist = VecDeque::<Vec<u32>>::new();
    state_ids.insert(start_tuple.clone(), output.start_state());
    worklist.push_back(start_tuple);

    while let Some(component_states) = worklist.pop_front() {
        let from_state = state_ids[&component_states];

        let final_weight = Weight::union_all(
            component_states
                .iter()
                .enumerate()
                .filter_map(|(input_index, &state_id)| {
                    (state_id != DEAD)
                        .then(|| inputs[input_index].states()[state_id as usize].final_weight.as_ref())
                        .flatten()
                }),
        );
        if !final_weight.is_empty() {
            output.set_final_weight(from_state, final_weight);
        }

        let mut by_label = FxHashMap::<i32, Vec<(usize, u32, Weight)>>::default();
        for (input_index, &state_id) in component_states.iter().enumerate() {
            if state_id == DEAD {
                continue;
            }
            for (&label, &(target, ref weight)) in &inputs[input_index].states()[state_id as usize].transitions {
                by_label
                    .entry(label)
                    .or_default()
                    .push((input_index, target, weight.clone()));
            }
        }

        for (label, transitions) in by_label {
            let edge_weight = Weight::union_all(transitions.iter().map(|(_, _, weight)| weight));
            if edge_weight.is_empty() {
                continue;
            }

            let mut target_tuple = vec![DEAD; inputs.len()];
            for (input_index, target, _) in transitions {
                target_tuple[input_index] = target;
            }

            let to_state = if let Some(&existing) = state_ids.get(&target_tuple) {
                existing
            } else {
                let new_state = output.add_state();
                state_ids.insert(target_tuple.clone(), new_state);
                worklist.push_back(target_tuple);
                new_state
            };
            output.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    debug_assert!(output.is_acyclic());
    output
}

fn contiguous_token_offset(local_to_global: &[u32]) -> Option<u32> {
    let offset = *local_to_global.first()?;
    if offset == u32::MAX {
        return None;
    }
    local_to_global
        .iter()
        .enumerate()
        .all(|(local, &global)| {
            global != u32::MAX && offset.checked_add(local as u32) == Some(global)
        })
        .then_some(offset)
}

/// Build the exact deterministic union without subset determinization whenever
/// the source vocabularies are disjoint.  The generic NWA route remains the
/// fallback for overlapping domains.
fn try_merge_disjoint_token_domain_dwas(
    inputs: &[LocalIdMapTerminalDwa],
    global_id_map: &InternalIdMap,
    direct_local_to_global_token_maps: Option<&Vec<Vec<u32>>>,
    max_token_id: u32,
) -> Option<DWA> {
    if !inputs_have_disjoint_token_domains(inputs, max_token_id) {
        return None;
    }

    let profiling = compile_profile_enabled();
    let mut build_maps_ms = 0.0;
    let mut clone_ms = 0.0;
    let mut remap_ms = 0.0;
    let mut remapped = Vec::with_capacity(inputs.len());
    for (input_index, input) in inputs.iter().enumerate() {
        let started_at = profiling.then(Instant::now);
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, global_id_map);
        let direct_token_map = direct_local_to_global_token_maps
            .and_then(|maps| maps.get(input_index));
        let token_offset = direct_token_map.and_then(|map| contiguous_token_offset(map));
        if profiling {
            let tsid_pairs = tsid_map.iter().map(Vec::len).sum::<usize>();
            let tsid_one_to_one = tsid_map.iter().filter(|targets| targets.len() == 1).count();
            let tsid_max_fanout = tsid_map.iter().map(Vec::len).max().unwrap_or(0);
            eprintln!(
                "[glrmask/profile][direct_disjoint_terminal_dwa_maps] input={} local_tsids={} tsid_pairs={} tsid_one_to_one={} tsid_max_fanout={} local_tokens={} token_offset={:?}",
                input_index,
                tsid_map.len(),
                tsid_pairs,
                tsid_one_to_one,
                tsid_max_fanout,
                input.id_map.num_internal_tokens(),
                token_offset,
            );
        }
        if let Some(started_at) = started_at {
            build_maps_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let mut dwa = input.dwa.clone();
        if let Some(started_at) = started_at {
            clone_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        if let Some(token_offset) = token_offset {
            remap_dwa_with_token_offset(
                &mut dwa,
                &tsid_map,
                token_offset,
                input.id_map.num_internal_tokens(),
                &global_id_map,
            );
        } else {
            let token_map = direct_token_map
                .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, global_id_map));
            remap_dwa_with_maps(
                &mut dwa,
                &tsid_map,
                &token_map,
                &global_id_map,
            );
        }
        if let Some(started_at) = started_at {
            remap_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        remapped.push(dwa);
    }
    let union_started_at = profiling.then(Instant::now);
    let result = direct_union_disjoint_token_domain_dwas(&remapped);
    if let Some(union_started_at) = union_started_at {
        eprintln!(
            "[glrmask/profile][direct_disjoint_terminal_dwa_union] inputs={} build_maps_ms={:.3} clone_ms={:.3} remap_ms={:.3} union_ms={:.3} states={} transitions={}",
            inputs.len(),
            build_maps_ms,
            clone_ms,
            remap_ms,
            union_started_at.elapsed().as_secs_f64() * 1000.0,
            result.num_states(),
            result.num_transitions(),
        );
    }

    Some(result)
}

/// Merge local branch outputs for a single partition into one compacted DWA.
pub fn merge_local_id_maps_and_terminal_dwas(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_local_id_maps_and_terminal_dwas called with empty inputs");

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        input.profile = TerminalDwaPhaseProfile::default();
        return input;
    }

    let input_refs: Vec<&LocalIdMapTerminalDwa> = inputs.iter().collect();
    let id_map_refs: Vec<&InternalIdMap> = input_refs.iter().map(|input| &input.id_map).collect();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);

    let pre_minimize = try_merge_disjoint_token_domain_dwas(
        &inputs,
        &global_id_map,
        direct_local_to_global_token_maps.as_ref(),
        max_token_id,
    )
    .unwrap_or_else(|| {
        let mut global_nwa = NWA::new(
            global_id_map.num_tsids(),
            global_id_map.max_internal_token_id(),
        );
        let mut global_body = global_nwa.body();

        for local in &inputs {
            let mut nwa = local.dwa.to_nwa();
            let tsid_map = build_local_to_global_tsid_map(&local.id_map, &global_id_map);
            let token_map = build_local_to_global_token_map(&local.id_map, &global_id_map);
            remap_nwa_with_maps(
                &mut nwa,
                &tsid_map,
                &token_map,
                &global_id_map,
            );
            global_body = global_nwa.union_in_place(&nwa, &global_body);
        }
        global_nwa.set_start_states(global_body.start_states);
        determinize(&global_nwa).expect("merge terminal NWA determinization failed")
    });
    let compact_started_at = Instant::now();
    let mut mapped_dwa = MappedArtifact::new(minimize_owned(pre_minimize), global_id_map);
    mapped_dwa.compact_dimensions_fast();
    let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, id_map) = mapped_dwa.into_parts();

    LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            compact_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    }
}

/// Union mapped deterministic weighted automata in a common ID space.
///
/// The merge machinery is independent of how the input DWAs were produced;
/// parser DWAs use the same TSID/token-labelled weights as terminal DWAs.  This
/// wrapper keeps parser construction from depending on the terminal-specific
/// transport struct while reusing the exact remap/union/determinize path.
pub fn merge_mapped_dwas(
    inputs: Vec<MappedArtifact<DWA>>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> MappedArtifact<DWA> {
    assert!(!inputs.is_empty(), "merge_mapped_dwas called with empty inputs");
    let locals = inputs
        .into_iter()
        .map(|input| {
            let (dwa, id_map) = input.into_parts();
            LocalIdMapTerminalDwa {
                id_map,
                dwa,
                profile: TerminalDwaPhaseProfile::default(),
            }
        })
        .collect();
    let merged = merge_local_id_maps_and_terminal_dwas(
        locals,
        num_tokenizer_states,
        max_token_id,
    );
    MappedArtifact::new(merged.dwa, merged.id_map)
}

fn immediate_dwa_accepting_edges(dwa: &DWA) -> Option<Vec<(i32, Weight)>> {
    if dwa.states().len() != 2 {
        return None;
    }
    let start_id = dwa.start_state() as usize;
    let final_id = 1usize.checked_sub(start_id)?;
    let start = &dwa.states()[start_id];
    let final_state = &dwa.states()[final_id];
    if start
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
        || !final_state.transitions.is_empty()
    {
        return None;
    }
    let final_weight = final_state.final_weight.as_ref()?;
    if final_weight.is_empty() || start.transitions.is_empty() {
        return None;
    }

    let mut accepting = Vec::with_capacity(start.transitions.len());
    for (&label, (target, edge_weight)) in &start.transitions {
        if label < 0 || label == DEFAULT_LABEL || *target as usize != final_id || edge_weight.is_empty() {
            return None;
        }
        let weight = edge_weight.intersection(final_weight);
        if weight.is_empty() {
            return None;
        }
        accepting.push((label, weight));
    }
    Some(accepting)
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct ExactWeightedStateSignature {
    final_weight: Option<usize>,
    transitions: Vec<(i32, usize, usize)>,
}

/// Merge only states with byte-for-byte identical weighted behavior modulo
/// already-merged children. This is linear-ish hash-consing for an acyclic DWA,
/// not the compatibility search performed by the full weighted minimizer.
fn exact_quotient_acyclic_dwa(dwa: DWA) -> DWA {
    if dwa.states().is_empty() || !dwa.is_acyclic() {
        return dwa;
    }

    fn visit(state: usize, dwa: &DWA, seen: &mut [bool], order: &mut Vec<usize>) {
        if seen[state] {
            return;
        }
        seen[state] = true;
        for (target, weight) in dwa.states()[state].transitions.values() {
            let target = *target as usize;
            if !weight.is_empty() && target < dwa.states().len() {
                visit(target, dwa, seen, order);
            }
        }
        order.push(state);
    }

    let mut seen = vec![false; dwa.states().len()];
    let mut order = Vec::with_capacity(dwa.states().len());
    visit(dwa.start_state() as usize, &dwa, &mut seen, &mut order);

    let mut class_of = vec![usize::MAX; dwa.states().len()];
    let mut signature_to_class = FxHashMap::<ExactWeightedStateSignature, usize>::default();
    let mut representatives = Vec::<usize>::new();
    for &state_id in &order {
        let state = &dwa.states()[state_id];
        let signature = ExactWeightedStateSignature {
            final_weight: state.final_weight.as_ref().map(Weight::ptr_key),
            transitions: state
                .transitions
                .iter()
                .filter_map(|(&label, (target, weight))| {
                    (!weight.is_empty()).then_some((
                        label,
                        class_of[*target as usize],
                        weight.ptr_key(),
                    ))
                })
                .collect(),
        };
        let class = if let Some(&class) = signature_to_class.get(&signature) {
            class
        } else {
            let class = representatives.len();
            signature_to_class.insert(signature, class);
            representatives.push(state_id);
            class
        };
        class_of[state_id] = class;
    }

    let mut states = vec![DWAState::default(); representatives.len()];
    for (class, &representative) in representatives.iter().enumerate() {
        let source = &dwa.states()[representative];
        states[class].final_weight = source.final_weight.clone();
        states[class].transitions = source
            .transitions
            .iter()
            .filter_map(|(&label, (target, weight))| {
                (!weight.is_empty()).then_some((
                    label,
                    (class_of[*target as usize] as u32, weight.clone()),
                ))
            })
            .collect();
    }
    DWA::from_parts(states, class_of[dwa.start_state() as usize] as u32)
}

/// Fold a depth-one parser language into another deterministic parser DWA.
///
/// For a label accepted by both inputs, the merged start edge reaches a
/// decorated copy of the other destination. The copy admits the immediate
/// branch in its final weight, while every continuation edge is intersected
/// with the other branch's incoming weight. This preserves the token-path
/// correlation exactly, even when the two source weight domains overlap.
fn try_graft_immediate_parser_dwa(immediate: &DWA, mut other: DWA) -> Option<DWA> {
    let accepting = immediate_dwa_accepting_edges(immediate)?;
    let start_id = other.start_state();
    let original_start = other.states().get(start_id as usize)?.clone();
    let mut decorated_by_key = FxHashMap::<(u32, usize, usize), u32>::default();
    let mut missing = Vec::<(i32, Weight)>::new();
    for (label, add_weight) in accepting {
        let existing = original_start
            .transitions
            .get(&label)
            .or_else(|| original_start.transitions.get(&DEFAULT_LABEL))
            .cloned();
        if let Some((target, edge_weight)) = existing {
            let key = (target, edge_weight.ptr_key(), add_weight.ptr_key());
            let decorated = if let Some(&existing) = decorated_by_key.get(&key) {
                existing
            } else {
                let source_state = if target == start_id {
                    &original_start
                } else {
                    other.states().get(target as usize)?
                };
                let mut decorated_state = source_state.clone();
                decorated_state.final_weight = Some(match decorated_state.final_weight.take() {
                    Some(final_weight) => final_weight.union(&add_weight),
                    None => add_weight.clone(),
                });
                decorated_state.transitions.retain(|_, (_, continuation_weight)| {
                    *continuation_weight = continuation_weight.intersection(&edge_weight);
                    !continuation_weight.is_empty()
                });
                let new_state = other.add_state();
                other.states_mut()[new_state as usize] = decorated_state;
                decorated_by_key.insert(key, new_state);
                new_state
            };
            other.add_transition(
                start_id,
                label,
                decorated,
                edge_weight.union(&add_weight),
            );
        } else {
            missing.push((label, add_weight));
        }
    }

    if !missing.is_empty() {
        let sink = other.add_state();
        other.set_final_weight(sink, Weight::all());
        for (label, weight) in missing {
            other.add_transition(start_id, label, sink, weight);
        }
    }
    Some(other)
}


/// Exact weighted union specialized for two deterministic, epsilon-free DWAs.
///
/// When generic determinization normalizes singleton subsets `(q, w)` to
/// `(q, all)`, their source rows can be moved directly. Otherwise singleton
/// residuals remain explicit. This distinction preserves structural fallback
/// row selection, not just raw weighted language. Paired subsets always retain
/// at most one state from each input.
fn union_two_parser_dwas_direct(left: DWA, right: DWA) -> DWA {
    union_two_parser_dwas_impl::<false>(left, right)
}

/// Union the languages actually consumed by the parser-stack evaluator.
/// Each operand selects its explicit edge, or its own default when absent,
/// BEFORE the two contributions are combined. A raw alphabet union is not
/// equivalent: an explicit edge contributed by one family would shadow the
/// other family's fallback and silently lose valid token/lexer coordinates.
fn union_two_parser_dwas_with_fallback(left: DWA, right: DWA) -> DWA {
    union_two_parser_dwas_impl::<true>(left, right)
}

fn has_parser_fallback(dwa: &DWA) -> bool {
    dwa.states().iter().any(|row| row.transitions.contains_key(&DEFAULT_LABEL))
}

fn union_two_parser_dwas_impl<const FALLBACK: bool>(mut left: DWA, mut right: DWA) -> DWA {
    type PairKey = (u32, u32, Weight, Weight);
    const ABSENT: u32 = u32::MAX;

    fn intern_pair(
        left_state: u32,
        right_state: u32,
        left_weight: Weight,
        right_weight: Weight,
        source_state_count: u32,
        pair_states: &mut Vec<DWAState>,
        pairs: &mut FxHashMap<PairKey, u32>,
        worklist: &mut VecDeque<(u32, u32, u32, Weight, Weight)>,
    ) -> u32 {
        let key = (
            left_state,
            right_state,
            left_weight.clone(),
            right_weight.clone(),
        );
        if let Some(&state) = pairs.get(&key) {
            return state;
        }
        let state = source_state_count + pair_states.len() as u32;
        pair_states.push(DWAState::default());
        pairs.insert(key, state);
        worklist.push_back((
            state,
            left_state,
            right_state,
            left_weight,
            right_weight,
        ));
        state
    }

    let left_start = left.start_state();
    let right_start = right.start_state();
    let mut left_states = std::mem::take(left.states_mut());
    let mut right_states = std::mem::take(right.states_mut());
    if FALLBACK {
        // An explicit empty edge suppresses an operand's fallback. Preserve
        // that structural barrier with a nonaccepting sink, rather than
        // deleting it as raw-alphabet normalization legitimately would.
        for states in [&mut left_states, &mut right_states] {
            let dead = states.len() as u32;
            let mut needed = false;
            for row in states.iter_mut() {
                for (target, weight) in row.transitions.values_mut() {
                    if weight.is_empty() {
                        *target = dead;
                        *weight = Weight::all();
                        needed = true;
                    }
                }
            }
            if needed { states.push(DWAState::default()); }
        }
    }
    let right_offset = left_states.len() as u32;
    let source_state_count = right_offset + right_states.len() as u32;
    // NWA::union_in_place appends each source without adding states. Use the
    // same size and override policy as the reference path.
    let normalize_singletons = normalize_direct_singletons_enabled(source_state_count as usize);
    // Generic expansion omits empty-weight edges. Copied singleton rows must
    // do the same, since an explicit empty edge would shadow DEFAULT_LABEL.
    for row in left_states.iter_mut().chain(right_states.iter_mut()) {
        row.transitions.retain(|_, (_, weight)| !weight.is_empty());
    }
    // A zero contribution on a concrete label still needs to shadow a
    // nonzero default in the merged row. Its explicit dead edge must survive
    // ordinary empty-weight cleanup and singleton-row reuse.
    let mut pair_states = if FALLBACK { vec![DWAState::default()] } else { Vec::new() };
    let dead_output = source_state_count;

    let mut pairs = FxHashMap::<PairKey, u32>::default();
    let mut worklist = VecDeque::<(u32, u32, u32, Weight, Weight)>::new();
    let start = intern_pair(
        left_start,
        right_start,
        Weight::all(),
        Weight::all(),
        source_state_count,
        &mut pair_states,
        &mut pairs,
        &mut worklist,
    );
    let mut weight_ops = ScopedWeightOpCache::default();

    while let Some((out_state, left_id, right_id, left_residual, right_residual)) =
        worklist.pop_front()
    {
        let left_state = (left_id != ABSENT).then(|| &left_states[left_id as usize]);
        let right_state = (right_id != ABSENT).then(|| &right_states[right_id as usize]);
        let mut output = DWAState::default();

        if let Some(final_weight) = left_state.and_then(|state| state.final_weight.as_ref()) {
            let contribution = weight_ops.intersection(&left_residual, final_weight);
            if !contribution.is_empty() {
                output.final_weight = Some(contribution);
            }
        }
        if let Some(final_weight) = right_state.and_then(|state| state.final_weight.as_ref()) {
            let contribution = weight_ops.intersection(&right_residual, final_weight);
            if !contribution.is_empty() {
                output.final_weight = Some(match output.final_weight.take() {
                    Some(existing) => weight_ops.union(&existing, &contribution),
                    None => contribution,
                });
            }
        }

        let mut left_transitions = left_state.into_iter().flat_map(|state| state.transitions.iter()).peekable();
        let mut right_transitions = right_state.into_iter().flat_map(|state| state.transitions.iter()).peekable();
        loop {
            let label = match (left_transitions.peek(), right_transitions.peek()) {
                (Some((left_label, _)), Some((right_label, _))) => {
                    Some((**left_label).min(**right_label))
                }
                (Some((left_label, _)), None) => Some(**left_label),
                (None, Some((right_label, _))) => Some(**right_label),
                (None, None) => None,
            };
            let Some(label) = label else { break };

            let left_edge = if left_transitions
                .peek()
                .is_some_and(|(candidate, _)| **candidate == label)
            {
                left_transitions.next().map(|(_, edge)| edge)
            } else if FALLBACK {
                left_state.and_then(|row| row.transitions.get(&DEFAULT_LABEL))
            } else {
                None
            };
            let right_edge = if right_transitions
                .peek()
                .is_some_and(|(candidate, _)| **candidate == label)
            {
                right_transitions.next().map(|(_, edge)| edge)
            } else if FALLBACK {
                right_state.and_then(|row| row.transitions.get(&DEFAULT_LABEL))
            } else {
                None
            };

            let left_contribution = left_edge.and_then(|(target, weight)| {
                let contribution = weight_ops.intersection(&left_residual, weight);
                (!contribution.is_empty()).then_some((*target, contribution))
            });
            let right_contribution = right_edge.and_then(|(target, weight)| {
                let contribution = weight_ops.intersection(&right_residual, weight);
                (!contribution.is_empty()).then_some((*target, contribution))
            });

            let transition = match (left_contribution, right_contribution) {
                (None, None) if FALLBACK && label != DEFAULT_LABEL =>
                    Some((dead_output, Weight::all())),
                (None, None) => None,
                (Some((target, weight)), None) => {
                    let out_target = if normalize_singletons { target } else {
                        intern_pair(target, ABSENT, weight.clone(), Weight::empty(),
                            source_state_count, &mut pair_states, &mut pairs, &mut worklist)
                    };
                    Some((out_target, weight))
                }
                (None, Some((target, weight))) => {
                    let out_target = if normalize_singletons { right_offset + target } else {
                        intern_pair(ABSENT, target, Weight::empty(), weight.clone(),
                            source_state_count, &mut pair_states, &mut pairs, &mut worklist)
                    };
                    Some((out_target, weight))
                }
                (Some((left_target, left_weight)), Some((right_target, right_weight))) => {
                    let edge_weight = weight_ops.union(&left_weight, &right_weight);
                    let target = intern_pair(
                        left_target,
                        right_target,
                        left_weight,
                        right_weight,
                        source_state_count,
                        &mut pair_states,
                        &mut pairs,
                        &mut worklist,
                    );
                    Some((target, edge_weight))
                }
            };
            if let Some((target, weight)) = transition {
                output.transitions.insert(label, (target, weight));
            }
        }
        pair_states[(out_state - source_state_count) as usize] = output;
    }

    // Move source rows into the union with stable singleton IDs. Only right-side
    // source targets need the fixed offset; left-side IDs already match.
    for state in &mut right_states {
        for (target, _) in state.transitions.values_mut() {
            *target += right_offset;
        }
    }
    let mut states = left_states;
    states.reserve(right_states.len() + pair_states.len());
    states.extend(right_states);
    states.extend(pair_states);

    // Not every copied singleton row is reachable from the paired start. Compact
    // by moving the live rows; no BTreeMap/Weight cloning is needed.
    let mut reachable = vec![false; states.len()];
    let mut reachable_queue = VecDeque::new();
    reachable[start as usize] = true;
    reachable_queue.push_back(start);
    while let Some(state_id) = reachable_queue.pop_front() {
        for (target, _) in states[state_id as usize].transitions.values() {
            let target_idx = *target as usize;
            if !reachable[target_idx] {
                reachable[target_idx] = true;
                reachable_queue.push_back(*target);
            }
        }
    }

    let mut remap = vec![u32::MAX; states.len()];
    let mut next_state = 0u32;
    for (state_id, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            remap[state_id] = next_state;
            next_state += 1;
        }
    }
    let new_start = remap[start as usize];
    let mut compact = Vec::with_capacity(next_state as usize);
    for (state_id, mut state) in states.into_iter().enumerate() {
        if !reachable[state_id] {
            continue;
        }
        for (target, _) in state.transitions.values_mut() {
            let mapped = remap[*target as usize];
            debug_assert_ne!(mapped, u32::MAX);
            *target = mapped;
        }
        compact.push(state);
    }

    DWA::from_parts(compact, new_start)
}

/// Merge parser-family DWAs without paying for an unnecessary second global
/// weighted minimization.
///
/// A depth-one family is grafted exactly into the other family, including the
/// overlapping-weight case. Other shapes retain the ordinary generic merge as
/// a fallback. Parser-family artifacts are immediately reconciled with possible
/// matches and finalized, so globally minimizing them here can dominate compile
/// time without being required for correctness.
/// Publish parser families directly in the already-planned joint observation
/// coordinate. Only primary stack languages enter the weighted union;
/// one-symbol accepting languages can remain independent top overlays.
pub fn merge_parser_dwas_in_common(
    inputs: Vec<MappedArtifact<DWA>>,
    common: &InternalIdMap,
    overlay_immediate: bool,
) -> (DWA, BTreeMap<i32, Weight>) {
    let started = Instant::now();
    let count = inputs.len();
    let mut overlay = BTreeMap::<i32, Weight>::new();
    let mut primaries = Vec::with_capacity(count);
    // Classify before remapping: the local graph establishes whether the
    // first-symbol acceptance projection is an exact language replacement.
    let immediate_count = if overlay_immediate && count > 1 {
        inputs.iter().filter(|input| immediate_dwa_accepting_edges(input.artifact()).is_some()).count()
    } else {
        0
    };
    let mut overlays_left = if count == 2 && immediate_count != 0 {
        1
    } else if immediate_count < count {
        immediate_count
    } else {
        0
    };
    for input in inputs {
        if overlays_left != 0
            && let Some(edges) = immediate_dwa_accepting_edges(input.artifact())
        {
            let (_, local) = input.into_parts();
            let weights = MappedArtifact::new(edges.into_iter().collect::<BTreeMap<_, _>>(), local)
                .into_common_artifact(common);
            for (label, weight) in weights {
                overlay.entry(label).and_modify(|existing| *existing = existing.union(&weight))
                    .or_insert(weight);
            }
            overlays_left -= 1;
        } else {
            primaries.push(input);
        }
    }

    let primary_count = primaries.len();
    if primary_count > 1 {
        charge_build_budget("template_determinization", primaries.iter()
            .map(|input| input.artifact().num_states() as u64
                + input.artifact().num_transitions() as u64)
            .sum());
    }
    // Preserve the disjoint-domain proof from the *local* coordinates:
    // it cannot be recovered from the now-shared destination map.
    let original_count = primaries.iter()
        .map(|input| input.id_map().vocab_tokens.original_to_internal.len())
        .max().unwrap_or(0);
    let disjoint = primary_count > 2 && token_domains_disjoint(
        primaries.iter().map(MappedArtifact::id_map), original_count,
    );
    let dwas = primaries.into_iter()
        .map(|input| input.into_common_artifact(common))
        .collect::<Vec<_>>();
    let (merged, mode) = union_parser_dwas_in_common(dwas, common, disjoint);
    if compile_profile_enabled() && count > 1 {
        eprintln!(
            "[glrmask/profile][parser_dwa_merge] inputs={} mode={} primary_families={} overlay_labels={} states={} transitions={} total_ms={:.3}",
            count, mode, primary_count, overlay.len(), merged.num_states(), merged.num_transitions(),
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    (merged, overlay)
}

fn union_parser_dwas_in_common(
    dwas: Vec<DWA>,
    common: &InternalIdMap,
    disjoint: bool,
) -> (DWA, &'static str) {
    if dwas.is_empty() {
        return (DWA::new(common.num_tsids(), common.max_internal_token_id()), "empty");
    }
    if dwas.len() == 1 {
        return (dwas.into_iter().next().unwrap(), "single");
    }
    // DEFAULT is a per-operand fallback rather than a shared alphabet
    // letter. Its semantic union must not enter the generic minimizer.
    if dwas.iter().any(has_parser_fallback) {
        return (dwas.into_iter().reduce(union_two_parser_dwas_with_fallback).unwrap(),
            "fallback_semantic_union");
    }
    if disjoint {
        return (direct_union_disjoint_token_domain_dwas(&dwas), "direct_disjoint");
    }

    let source_states = dwas.iter().map(|dwa| dwa.states().len()).sum::<usize>();
    let direct = dwas.len() == 2 && source_states >= 16_384
        && normalize_direct_singletons_enabled(source_states)
        && std::env::var_os("GLRMASK_DISABLE_DIRECT_TWO_PARSER_UNION").is_none();
    let generic_union = |sources: &[DWA]| {
        let mut union = NWA::new(common.num_tsids(), common.max_internal_token_id());
        let mut body = union.body();
        for source in sources {
            body = union.union_in_place(&source.to_nwa(), &body);
        }
        union.set_start_states(body.start_states);
        determinize(&union).expect("parser-family NWA union must determinize")
    };
    if direct {
        let reference = std::env::var_os("GLRMASK_VALIDATE_DIRECT_TWO_PARSER_UNION").is_some()
            .then(|| generic_union(&dwas));
        let merged = dwas.into_iter().reduce(union_two_parser_dwas_direct).unwrap();
        if let Some(reference) = reference {
            let difference = find_difference(&merged, &reference)
                .expect("direct parser-family union equivalence check failed");
            assert!(difference.is_none(),
                "direct parser-family union differs from generic union on labels {difference:?}");
        }
        return (merged, "direct_two_dwa");
    }
    let merged = generic_union(&dwas);
    if dwas.len() > 2 {
        (minimize_owned(merged), "minimized_generic_union")
    } else {
        (merged, "exact_raw_union")
    }
}

// Keep the existing independent mapped-union regression fixtures exercising
// the same production merge, including source-specific default labels.
#[cfg(test)]
fn merge_mapped_parser_dwas(
    inputs: Vec<MappedArtifact<DWA>>,
    _num_tokenizer_states: usize,
    _max_token_id: u32,
) -> MappedArtifact<DWA> {
    let maps = inputs.iter().map(MappedArtifact::id_map).collect::<Vec<_>>();
    let common = crate::compiler::stages::mapped_artifact::joint_observation_id_map(&maps);
    let (dwa, _) = merge_parser_dwas_in_common(inputs, &common, false);
    MappedArtifact::new(dwa, common)
}

#[cfg(test)]
fn merge_mapped_parser_dwas_with_top_accept(
    inputs: Vec<MappedArtifact<DWA>>,
    _num_tokenizer_states: usize,
    _max_token_id: u32,
) -> (MappedArtifact<DWA>, BTreeMap<i32, Weight>) {
    let maps = inputs.iter().map(MappedArtifact::id_map).collect::<Vec<_>>();
    let common = crate::compiler::stages::mapped_artifact::joint_observation_id_map(&maps);
    let (dwa, overlay) = merge_parser_dwas_in_common(inputs, &common, true);
    (MappedArtifact::new(dwa, common), overlay)
}

/// Build the ordinary global NWA union and determinize it. This is used only
/// as an opt-in semantic oracle for the specialized disjoint-domain merge.
fn generic_global_union_determinization(
    inputs: &[LocalIdMapTerminalDwa],
    global_id_map: &InternalIdMap,
    direct_local_to_global_token_maps: Option<&Vec<Vec<u32>>>,
) -> DWA {
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();
    for (input_index, input) in inputs.iter().enumerate() {
        let mut nwa = input.dwa.to_nwa();
        let tsid_map = build_local_to_global_tsid_map(&input.id_map, global_id_map);
        let token_map = direct_local_to_global_token_maps
            .and_then(|maps| maps.get(input_index))
            .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
            .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, global_id_map));
        remap_nwa_with_maps(
            &mut nwa,
            &tsid_map,
            &token_map,
            &global_id_map,
        );
        global_body = global_nwa.union_in_place(&nwa, &global_body);
    }
    global_nwa.set_start_states(global_body.start_states);
    determinize(&global_nwa).expect("global terminal NWA reference determinization failed")
}

fn prune_unreachable_nwa(nwa: NWA) -> NWA {
    let mut reachable = vec![false; nwa.states().len()];
    let mut stack = nwa.start_states().to_vec();
    while let Some(state_id) = stack.pop() {
        let index = state_id as usize;
        if index >= reachable.len() || reachable[index] {
            continue;
        }
        reachable[index] = true;
        let state = &nwa.states()[index];
        for branches in state.transitions.values() {
            stack.extend(branches.iter().map(|(target, _)| *target));
        }
        stack.extend(state.epsilons.iter().map(|(target, _)| *target));
    }

    let mut old_to_new = vec![u32::MAX; reachable.len()];
    let mut states = Vec::new();
    for (old, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            old_to_new[old] = states.len() as u32;
            states.push(nwa.states()[old].clone());
        }
    }
    for state in &mut states {
        for branches in state.transitions.values_mut() {
            for (target, _) in branches {
                *target = old_to_new[*target as usize];
            }
        }
        for (target, _) in &mut state.epsilons {
            *target = old_to_new[*target as usize];
        }
    }
    let starts = nwa
        .start_states()
        .iter()
        .map(|start| old_to_new[*start as usize])
        .collect();
    NWA::from_parts(states, starts)
}


/// Build a global token-deterministic NWA when every source owns a disjoint
/// vocabulary domain. Multiple branches for one label are then deterministic
/// with respect to the source token, so parser construction can consume them
/// directly without materializing a product DWA.
pub fn try_merge_id_maps_and_token_deterministic_nwa(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> Option<(NWA, InternalIdMap, TerminalDwaPhaseProfile)> {
    try_merge_id_maps_and_token_deterministic_nwa_impl(
        inputs,
        num_tokenizer_states,
        max_token_id,
        false,
    )
}

/// Build the same exact token-deterministic NWA union when the caller already
/// proved that every input owns a disjoint original-vocabulary domain.
///
/// Generic callers must use the checked entry point above. This entry point
/// avoids rediscovering the partition invariant by scanning the whole vocab.
pub fn try_merge_id_maps_and_token_deterministic_nwa_proven_disjoint(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> Option<(NWA, InternalIdMap, TerminalDwaPhaseProfile)> {
    try_merge_id_maps_and_token_deterministic_nwa_impl(
        inputs,
        num_tokenizer_states,
        max_token_id,
        true,
    )
}

fn try_merge_id_maps_and_token_deterministic_nwa_impl(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
    token_domains_proven_disjoint: bool,
) -> Option<(NWA, InternalIdMap, TerminalDwaPhaseProfile)> {
    if inputs.len() < 2 {
        return None;
    }
    if token_domains_proven_disjoint {
        if std::env::var_os("GLRMASK_ASSERT_PROVEN_DISJOINT_TOKEN_NWA_DOMAINS").is_some() {
            assert!(
                inputs_have_disjoint_token_domains(inputs, max_token_id),
                "caller-proven token-NWA merge received overlapping vocabulary domains",
            );
        }
    } else if !inputs_have_disjoint_token_domains(inputs, max_token_id) {
        return None;
    }

    let total_started_at = Instant::now();
    let primary_tsid_source = select_primary_token_nwa_tsid_source(inputs)
        .filter(|source| *source < inputs.len());
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let id_map_started_at = Instant::now();
    let (global_id_map, direct_local_to_global_token_maps, precomputed_tsid_maps) =
        if token_domains_proven_disjoint {
            let (id_map, direct_maps, tsid_maps) = build_unified_global_id_map_disjoint_fast(
                &id_map_refs,
                num_tokenizer_states,
                max_token_id,
                primary_tsid_source,
            );
            (id_map, Some(direct_maps), Some(tsid_maps))
        } else {
            let (id_map, direct_maps) =
                build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
            (id_map, direct_maps, None)
        };
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;
    let all_tsid_maps: Vec<Vec<Vec<u32>>> = precomputed_tsid_maps.unwrap_or_else(|| {
        inputs
            .iter()
            .map(|input| build_local_to_global_tsid_map(&input.id_map, &global_id_map))
            .collect()
    });

    let remap_started_at = Instant::now();
    let profiling = compile_profile_enabled();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut combined_start: Option<u32> = None;
    let tsid_map_ms = 0.0;
    let mut token_map_ms = 0.0;
    let mut clone_ms = 0.0;
    let mut remap_ms = 0.0;
    let mut to_nwa_ms = 0.0;
    let mut append_ms = 0.0;
    let mut fuse_start_ms = 0.0;
    for (input_index, input) in inputs.iter().enumerate() {
        let tsid_map = &all_tsid_maps[input_index];
        let remap_detail = std::env::var_os("GLRMASK_PROFILE_TOKEN_NWA_REMAP_DETAIL")
            .is_some()
            .then(|| profile_token_nwa_remap_detail(&input.dwa, tsid_map));
        let started_at = profiling.then(Instant::now);
        let direct_token_map = direct_local_to_global_token_maps
            .as_ref()
            .and_then(|maps| maps.get(input_index));
        let token_offset = direct_token_map.and_then(|map| contiguous_token_offset(map));
        if let Some(started_at) = started_at {
            token_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let mut dwa = input.dwa.clone();
        if let Some(started_at) = started_at {
            clone_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        if let Some(token_offset) = token_offset {
            remap_dwa_with_token_offset(
                &mut dwa, tsid_map, token_offset,
                input.id_map.num_internal_tokens(), &global_id_map,
            );
        } else {
            let token_map = direct_token_map
                .map(|map| build_direct_local_to_global_token_map(map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
            remap_dwa_with_maps(&mut dwa, tsid_map, &token_map, &global_id_map);
        }
        if let Some(started_at) = started_at {
            let elapsed_ms = started_at.elapsed().as_secs_f64() * 1000.0;
            remap_ms += elapsed_ms;
            let tsid_pairs = tsid_map.iter().map(Vec::len).sum::<usize>();
            let tsid_max_fanout = tsid_map.iter().map(Vec::len).max().unwrap_or(0);
            let tsid_runs = tsid_map
                .iter()
                .map(|globals| {
                    globals
                        .iter()
                        .enumerate()
                        .filter(|(index, global)| {
                            *index == 0 || **global != globals[*index - 1].saturating_add(1)
                        })
                        .count()
                })
                .sum::<usize>();
            eprintln!(
                "[glrmask/profile][token_deterministic_terminal_nwa_input] input={} states={} transitions={} local_tsids={} tsid_pairs={} tsid_runs={} tsid_max_fanout={} remap_ms={:.3}",
                input_index,
                input.dwa.num_states(),
                input.dwa.num_transitions(),
                input.id_map.num_tsids(),
                tsid_pairs,
                tsid_runs,
                tsid_max_fanout,
                elapsed_ms,
            );
        }
        if let Some(detail) = remap_detail {
            eprintln!(
                "[glrmask/profile][token_nwa_remap_detail] input={} unique_weights={} full_weights={} compact_entries={} local_tsid_visits={} global_tsid_expansions={} max_local_entry_span={} max_global_entry_expansion={}",
                input_index,
                detail.unique_weights,
                detail.full_weights,
                detail.compact_entries,
                detail.local_tsid_visits,
                detail.global_tsid_expansions,
                detail.max_local_entry_span,
                detail.max_global_entry_expansion,
            );
        }
        let started_at = profiling.then(Instant::now);
        let source_nwa = dwa.to_nwa();
        if let Some(started_at) = started_at {
            to_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        let started_at = profiling.then(Instant::now);
        let body = global_nwa.append_with_body(&source_nwa);
        if let Some(started_at) = started_at {
            append_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
        debug_assert_eq!(body.start_states.len(), 1);
        let source_start = body.start_states[0];
        let started_at = profiling.then(Instant::now);
        if let Some(combined_start) = combined_start {
            let source = global_nwa.states()[source_start as usize].clone();
            if let Some(final_weight) = source.final_weight {
                let target = &mut global_nwa.states_mut()[combined_start as usize].final_weight;
                *target = Some(match target.take() {
                    Some(existing) => existing.union(&final_weight),
                    None => final_weight,
                });
            }
            for (label, branches) in source.transitions {
                for (target, weight) in branches {
                    global_nwa.add_transition(combined_start, label, target, weight);
                }
            }
            assert!(source.epsilons.is_empty(), "source DWA conversion must not create epsilon edges");
        } else {
            combined_start = Some(source_start);
        }
        if let Some(started_at) = started_at {
            fuse_start_ms += started_at.elapsed().as_secs_f64() * 1000.0;
        }
    }
    let started_at = profiling.then(Instant::now);
    global_nwa.set_start_states(vec![combined_start.expect("at least one input")]);
    let global_nwa = prune_unreachable_nwa(global_nwa);
    let prune_ms = started_at.map(|started_at| started_at.elapsed().as_secs_f64() * 1000.0).unwrap_or(0.0);
    debug_assert!(global_nwa.is_acyclic());
    debug_assert!(global_nwa.states().iter().all(|state| state.epsilons.is_empty()));
    let remap_and_union_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    if assert_direct_global_terminal_dwa_merge_equivalence_enabled() {
        let reference = generic_global_union_determinization(
            inputs,
            &global_id_map,
            direct_local_to_global_token_maps.as_ref(),
        );
        let determinized = determinize(&global_nwa)
            .expect("token-deterministic terminal NWA assertion determinization failed");
        let difference = find_difference(&determinized, &reference)
            .expect("token-deterministic terminal NWA equivalence requires acyclic inputs");
        assert!(
            difference.is_none(),
            "token-deterministic terminal NWA differs from generic NWA union on labels {:?}",
            difference,
        );
        if profiling {
            eprintln!("[glrmask/profile][token_deterministic_terminal_nwa_equivalence] result=equivalent");
        }
    }

    if profiling {
        eprintln!(
            "[glrmask/profile][token_deterministic_terminal_nwa_merge] inputs={} id_map_ms={:.3} remap_and_union_ms={:.3} tsid_map_ms={:.3} token_map_ms={:.3} clone_ms={:.3} remap_ms={:.3} to_nwa_ms={:.3} append_ms={:.3} fuse_start_ms={:.3} prune_ms={:.3} states={} transitions={} total_ms={:.3}",
            inputs.len(),
            id_map_ms,
            remap_and_union_ms,
            tsid_map_ms,
            token_map_ms,
            clone_ms,
            remap_ms,
            to_nwa_ms,
            append_ms,
            fuse_start_ms,
            prune_ms,
            global_nwa.num_states(),
            global_nwa.num_transitions(),
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    Some((
        global_nwa,
        global_id_map,
        TerminalDwaPhaseProfile {
            id_map_ms,
            global_merge_ms: total_started_at.elapsed().as_secs_f64() * 1000.0,
            ..TerminalDwaPhaseProfile::default()
        },
    ))
}


/// Merge already-compacted partition outputs into one global DWA.
///
/// Partition-local merges have already minimized their local outputs. When
/// global token domains are disjoint, an exact deterministic source-product is
/// used and remains unminimized/uncompacted by default; this avoids constructing
/// a large subset DWA only to erase it immediately. Overlapping domains retain
/// the generic determinize-minimize-compact route. `GLRMASK_MINIMIZE_MERGED_TERMINAL_DWA`
/// and `GLRMASK_COMPACT_MERGED_TERMINAL_DWA` explicitly override either default.
pub fn merge_id_maps_and_terminal_dwas(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    merge_id_maps_and_terminal_dwas_impl(inputs, num_tokenizer_states, max_token_id, false)
}

/// Merge terminal-DWA pieces whose vocabulary domains are proven disjoint by
/// their caller. Generic callers should use `merge_id_maps_and_terminal_dwas`.
pub fn merge_id_maps_and_terminal_dwas_proven_disjoint(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> LocalIdMapTerminalDwa {
    merge_id_maps_and_terminal_dwas_impl(inputs, num_tokenizer_states, max_token_id, true)
}

fn merge_id_maps_and_terminal_dwas_impl(
    inputs: Vec<LocalIdMapTerminalDwa>,
    num_tokenizer_states: usize,
    max_token_id: u32,
    token_domains_proven_disjoint: bool,
) -> LocalIdMapTerminalDwa {
    assert!(!inputs.is_empty(), "merge_id_maps_and_terminal_dwas called with empty inputs");

    if inputs.len() == 1 {
        let mut input = inputs.into_iter().next().unwrap();
        input.profile = TerminalDwaPhaseProfile::default();
        return input;
    }

    if let Some(merged) = try_merge_immediate_terminal_dwas(
        &inputs,
        num_tokenizer_states,
        max_token_id,
        token_domains_proven_disjoint,
    ) {
        return merged;
    }

    let total_started_at = Instant::now();

    let build_unified_global_id_map_started_at = Instant::now();
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    let (global_id_map, direct_local_to_global_token_maps) =
        build_unified_global_id_map(&id_map_refs, num_tokenizer_states, max_token_id);
    let build_unified_global_id_map_ms =
        build_unified_global_id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let remap_and_union_started_at = Instant::now();
    // The direct product is exact when token domains are disjoint. Unlike the
    // generic NWA route it keeps source-local deterministic contexts separate,
    // avoiding a large subset expansion that global minimization later erases.
    let direct_global = try_merge_disjoint_token_domain_dwas(
        &inputs,
        &global_id_map,
        direct_local_to_global_token_maps.as_ref(),
        max_token_id,
);
    let used_direct_global = direct_global.is_some();
    let mut global_nwa = NWA::new(
        global_id_map.num_tsids(),
        global_id_map.max_internal_token_id(),
    );
    let mut global_body = global_nwa.body();
    let profiling = compile_profile_enabled();
    let mut to_nwa_ms = 0.0;
    let mut build_tsid_map_ms = 0.0;
    let mut build_token_map_ms = 0.0;
    let mut remap_nwa_ms = 0.0;
    let mut union_ms = 0.0;

    if direct_global.is_none() {
        for (input_idx, input) in inputs.iter().enumerate() {
            let started_at = profiling.then(Instant::now);
            let mut nwa = input.dwa.to_nwa();
            if let Some(started_at) = started_at {
                to_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            let tsid_map = build_local_to_global_tsid_map(&input.id_map, &global_id_map);
            if let Some(started_at) = started_at {
                build_tsid_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            let token_map = direct_local_to_global_token_maps
                .as_ref()
                .and_then(|maps| maps.get(input_idx))
                .map(|direct_map| build_direct_local_to_global_token_map(direct_map))
                .unwrap_or_else(|| build_local_to_global_token_map(&input.id_map, &global_id_map));
            if let Some(started_at) = started_at {
                build_token_map_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            remap_nwa_with_maps(
                &mut nwa,
                &tsid_map,
                &token_map,
                &global_id_map,
            );
            if let Some(started_at) = started_at {
                remap_nwa_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }

            let started_at = profiling.then(Instant::now);
            global_body = global_nwa.union_in_place(&nwa, &global_body);
            if let Some(started_at) = started_at {
                union_ms += started_at.elapsed().as_secs_f64() * 1000.0;
            }
        }
        global_nwa.set_start_states(global_body.start_states);
    }
    let remap_and_union_ms = remap_and_union_started_at.elapsed().as_secs_f64() * 1000.0;
    let nwa_states_before_determinize = global_nwa.num_states();
    let nwa_transitions_before_determinize = global_nwa.num_transitions();

    let determinize_started_at = Instant::now();
    let det = direct_global.unwrap_or_else(|| {
        determinize(&global_nwa).expect("merge terminal NWA determinization failed")
    });
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let det_states = det.num_states();
    let det_transitions = det.num_transitions();

    if used_direct_global && assert_direct_global_terminal_dwa_merge_equivalence_enabled() {
        let reference = generic_global_union_determinization(
            &inputs,
            &global_id_map,
            direct_local_to_global_token_maps.as_ref(),
        );
        let difference = find_difference(&det, &reference)
            .expect("direct global terminal DWA equivalence requires acyclic inputs");
        assert!(
            difference.is_none(),
            "direct global terminal DWA differs from generic NWA union on labels {:?}",
            difference,
        );
        if profiling {
            eprintln!("[glrmask/profile][direct_global_terminal_dwa_equivalence] result=equivalent");
        }
    }

    // A direct product is already exact. Its slightly larger topology is much
    // cheaper to construct than generic subset expansion and remains suitable
    // for the downstream joint reconciliation pass, so do not minimize it by
    // default. Explicit environment values still override this strategy.
    let minimize_enabled = minimize_merged_terminal_dwa_enabled(!used_direct_global);
    let minimize_started_at = Instant::now();
    let dwa = if minimize_enabled {
        minimize_owned(det)
    } else {
        det
    };
    let minimize_ms = if minimize_enabled {
        minimize_started_at.elapsed().as_secs_f64() * 1000.0
    } else {
        0.0
    };
    let mut mapped_dwa = MappedArtifact::new(dwa, global_id_map);
    let before_compact_stats = profiling.then(|| mapped_dwa.artifact().stats());
    let before_compact_range_counts = profiling.then(|| mapped_dwa.interned_range_counts());
    let before_num_tsids = profiling.then(|| mapped_dwa.id_map().num_tsids());
    let before_num_tokens = profiling.then(|| mapped_dwa.id_map().num_internal_tokens());
    let compact_enabled = compact_merged_terminal_dwa_enabled(!used_direct_global);
    let (compact_report, compact_ms) = if compact_enabled {
        let compact_started_at = Instant::now();
        let compact_report = if profiling {
            mapped_dwa.compact_dimensions_fast_with_stats()
        } else {
            mapped_dwa.compact_dimensions_fast()
        };
        let compact_ms = compact_started_at.elapsed().as_secs_f64() * 1000.0;
        (Some(compact_report), compact_ms)
    } else {
        (None, 0.0)
    };
    let after_compact_stats = profiling.then(|| mapped_dwa.artifact().stats());
    let after_compact_range_counts = profiling.then(|| mapped_dwa.interned_range_counts());
    let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;
    let (dwa, id_map) = mapped_dwa.into_parts();

    if profiling {
        let before_compact_stats = before_compact_stats.unwrap();
        let before_compact_range_counts = before_compact_range_counts.unwrap();
        let before_num_tsids = before_num_tsids.unwrap();
        let before_num_tokens = before_num_tokens.unwrap();
        let after_compact_stats = after_compact_stats.unwrap();
        let after_compact_range_counts = after_compact_range_counts.unwrap();
        let profile_stats = compact_report.and_then(|report| report.profile_stats);
        eprintln!(
            "[glrmask/profile][terminal_dwa_global_merge] inputs={} build_unified_global_id_map_ms={:.3} remap_and_union_ms={:.3} to_nwa_ms={:.3} build_tsid_map_ms={:.3} build_token_map_ms={:.3} remap_nwa_ms={:.3} union_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} compact_ms={:.3} total_ms={:.3} nwa_states_before_determinize={} nwa_transitions_before_determinize={} det_states={} det_transitions={} dwa_states_before_compact={} dwa_transitions_before_compact={} dwa_states_after_compact={} dwa_transitions_after_compact={}",
            inputs.len(),
            build_unified_global_id_map_ms,
            remap_and_union_ms,
            to_nwa_ms,
            build_tsid_map_ms,
            build_token_map_ms,
            remap_nwa_ms,
            union_ms,
            determinize_ms,
            minimize_ms,
            compact_ms,
            total_ms,
            nwa_states_before_determinize,
            nwa_transitions_before_determinize,
            det_states,
            det_transitions,
            before_compact_stats.states,
            before_compact_stats.transitions,
            after_compact_stats.states,
            after_compact_stats.transitions,
        );
        eprintln!(
            "[glrmask/profile][merged_terminal_dwa] minimize_enabled={} compact_enabled={} states_before_compact={} transitions_before_compact={} interned_ranges_before_compact={} token_ranges_before_compact={} states_after_compact={} transitions_after_compact={} interned_ranges_after_compact={} token_ranges_after_compact={} tsids_before_compact={} tsids_after_compact={} tokens_before_compact={} tokens_after_compact={} compact_ms={:.3}",
            minimize_enabled,
            compact_enabled,
            before_compact_stats.states,
            before_compact_stats.transitions,
            before_compact_stats.interned_ranges,
            before_compact_range_counts.token_ranges,
            after_compact_stats.states,
            after_compact_stats.transitions,
            after_compact_stats.interned_ranges,
            after_compact_range_counts.token_ranges,
            before_num_tsids,
            profile_stats.map(|stats| stats.tsids_after).unwrap_or(before_num_tsids as usize),
            before_num_tokens,
            profile_stats.map(|stats| stats.tokens_after).unwrap_or(before_num_tokens as usize),
            compact_ms,
        );
    }

    LocalIdMapTerminalDwa {
        id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            compact_ms,
            global_merge_ms: total_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    }
}

/// Merge a family of depth-one terminal DWAs directly.
///
/// The generic global merge converts the inputs to an NWA, determinizes a
/// large set of one-edge alternatives, then minimizes the result back to two
/// states. For an immediate family the exact union is simply the union of the
/// remapped acceptance weights on each start label.
fn try_merge_immediate_terminal_dwas(
    inputs: &[LocalIdMapTerminalDwa],
    num_tokenizer_states: usize,
    max_token_id: u32,
    token_domains_proven_disjoint: bool,
) -> Option<LocalIdMapTerminalDwa> {
    if inputs.len() < 2
        || inputs
            .iter()
            .any(|input| immediate_dwa_accepting_edges(&input.dwa).is_none())
    {
        return None;
    }

    let total_started_at = Instant::now();
    let id_map_started_at = Instant::now();
    let id_map_refs: Vec<&InternalIdMap> = inputs.iter().map(|input| &input.id_map).collect();
    // These inputs are the compiler's disjoint vocabulary partitions. Preserve
    // that proof instead of scanning every original-token map again, and retain
    // the local→global TSID maps produced while constructing the global classes.
    let (global_id_map, token_maps, local_to_global_tsid_maps) =
        build_unified_global_id_map_fast(
            &id_map_refs,
            num_tokenizer_states,
            max_token_id,
            None,
            token_domains_proven_disjoint,
        );
    let id_map_ms = id_map_started_at.elapsed().as_secs_f64() * 1000.0;

    let remap_started_at = Instant::now();
    let mut weights_by_label = BTreeMap::<i32, Vec<Weight>>::new();
    for (input_index, input) in inputs.iter().enumerate() {
        let mut accepting = immediate_dwa_accepting_edges(&input.dwa)?;
        let tsid_map = &local_to_global_tsid_maps[input_index];
        let direct_storage;
        let token_axis = match &token_maps {
            FastGlobalTokenMaps::Direct(maps) => {
                let direct = &maps[input_index];
                if let Some(offset) = contiguous_token_offset(direct) {
                    TokenAxisMap::Offset {
                        offset, local_token_count: input.id_map.num_internal_tokens(),
                    }
                } else {
                    direct_storage = build_direct_local_to_global_token_map(direct);
                    TokenAxisMap::Explicit(&direct_storage)
                }
            }
            FastGlobalTokenMaps::General(maps) => TokenAxisMap::Explicit(&maps[input_index]),
        };
        let mut weights = accepting.iter_mut().map(|(_, weight)| weight).collect::<Vec<_>>();
        remap_weights_with_token_axis(
            &mut weights, tsid_map, token_axis,
            global_id_map.num_tsids() as usize, global_id_map.num_internal_tokens() as usize,
        );
        for (label, weight) in accepting {
            if !weight.is_empty() {
                weights_by_label.entry(label).or_default().push(weight);
            }
        }
    }
    let remap_ms = remap_started_at.elapsed().as_secs_f64() * 1000.0;

    let union_started_at = Instant::now();
    let mut dwa = DWA::new(global_id_map.num_tsids(), global_id_map.max_internal_token_id());
    let final_state = dwa.add_state();
    dwa.set_final_weight(final_state, Weight::all());
    for (label, weights) in weights_by_label {
        let weight = Weight::union_all(weights.iter());
        if !weight.is_empty() {
            dwa.add_transition(dwa.start_state(), label, final_state, weight);
        }
    }
    let union_ms = union_started_at.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_started_at.elapsed().as_secs_f64() * 1000.0;

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][immediate_terminal_dwa_merge] inputs={} id_map_ms={:.3} remap_ms={:.3} union_ms={:.3} states={} transitions={} tsids={} tokens={} total_ms={:.3}",
            inputs.len(),
            id_map_ms,
            remap_ms,
            union_ms,
            dwa.num_states(),
            dwa.num_transitions(),
            global_id_map.num_tsids(),
            global_id_map.num_internal_tokens(),
            total_ms,
        );
    }

    Some(LocalIdMapTerminalDwa {
        id_map: global_id_map,
        dwa,
        profile: TerminalDwaPhaseProfile {
            global_merge_ms: total_ms,
            ..TerminalDwaPhaseProfile::default()
        },
    })
}

/// Fast exact global map construction for an already-proven disjoint token
/// partition. The input order and original-state order are deterministic, so
/// first-seen class IDs are stable for this merge and avoid a costly sort and
/// relabel pass used only for cross-order canonicalization.
fn build_unified_global_id_map_fast(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
    primary_source: Option<usize>,
    require_direct_token_maps: bool,
) -> (InternalIdMap, FastGlobalTokenMaps, Vec<Vec<Vec<u32>>>) {
    let mut state_maps = inputs.iter().map(|input| &input.tokenizer_states).collect::<Vec<_>>();
    let primary = primary_source.filter(|&source| source < inputs.len());
    if let Some(source) = primary {
        let anchor = state_maps.remove(source);
        state_maps.insert(0, anchor);
    }
    let tokenizer_states = refine_observation_classes(
        &state_maps, num_tokenizer_states, false, primary.is_some(),
    );
    let (vocab_tokens, token_maps) = if require_direct_token_maps {
        let (map, direct) = build_unified_global_token_id_map_assume_disjoint(inputs, max_token_id);
        (map, FastGlobalTokenMaps::Direct(direct))
    } else {
        let (map, fanout) = build_unified_global_token_id_map_sparse_generic(inputs, max_token_id);
        (map, FastGlobalTokenMaps::General(fanout))
    };
    let common = InternalIdMap {
        tokenizer_states, vocab_tokens,
        deferred_vocab_singleton_original_ids: None,
    };
    let tsid_maps = inputs.iter().map(|local| build_local_to_global_tsid_map(local, &common))
        .collect();
    (common, token_maps, tsid_maps)
}

fn build_unified_global_id_map_disjoint_fast(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
    primary_source: Option<usize>,
) -> (InternalIdMap, Vec<Vec<u32>>, Vec<Vec<Vec<u32>>>) {
    let (id_map, token_maps, tsid_maps) = build_unified_global_id_map_fast(
        inputs,
        num_tokenizer_states,
        max_token_id,
        primary_source,
        true,
    );
    let FastGlobalTokenMaps::Direct(direct_maps) = token_maps else {
        panic!("disjoint token domains must produce direct token maps");
    };
    (id_map, direct_maps, tsid_maps)
}

fn build_unified_global_token_id_map_assume_disjoint(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> (ManyToOneIdMap, Vec<Vec<u32>>) {
    let mut token_o2i = vec![u32::MAX; max_token_id as usize + 1];
    let token_class_count: usize = inputs
        .iter()
        .map(|input| input.vocab_tokens.internal_to_originals.len())
        .sum();
    let mut token_i2o = Vec::with_capacity(token_class_count);
    let mut token_reps = Vec::with_capacity(token_class_count);
    let mut direct_local_to_global_token_maps = Vec::with_capacity(inputs.len());

    for input in inputs {
        let mut local_to_global = vec![u32::MAX; input.num_internal_tokens() as usize];
        for (local_class, originals) in input.vocab_tokens.internal_to_originals.iter().enumerate() {
            if originals.is_empty() {
                continue;
            }
            let global_class = token_i2o.len() as u32;
            local_to_global[local_class] = global_class;
            token_reps.push(input.vocab_tokens.representative_original_ids[local_class]);
            token_i2o.push(originals.clone());
            for &token_id in originals {
                token_o2i[token_id as usize] = global_class;
            }
        }
        direct_local_to_global_token_maps.push(local_to_global);
    }

    (
        ManyToOneIdMap {
            original_to_internal: token_o2i,
            internal_to_originals: token_i2o,
            representative_original_ids: token_reps,
        },
        direct_local_to_global_token_maps,
    )
}

/// Fast exact global map construction for an already-proven disjoint token
/// partition. The input order and original-state order are deterministic, so
/// first-seen class IDs are stable for this merge and avoid a costly sort and
/// relabel pass used only for cross-order canonicalization.
fn build_unified_global_id_map(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> (InternalIdMap, Option<Vec<Vec<u32>>>) {
    let state_maps = inputs.iter().map(|input| &input.tokenizer_states).collect::<Vec<_>>();
    let tokenizer_states = refine_observation_classes(
        &state_maps, num_tokenizer_states, false, true,
    );
    let (vocab_tokens, direct_local_to_global_token_maps) =
        build_unified_global_token_id_map_disjoint(inputs, max_token_id)
            .unwrap_or_else(|| (build_unified_global_token_id_map_generic(inputs, max_token_id), None));
    (
        InternalIdMap {
            tokenizer_states,
            vocab_tokens,
            deferred_vocab_singleton_original_ids: None,
        },
        direct_local_to_global_token_maps,
    )
}

/// Build only the unified exact ID map used by terminal-DWA merging, without
/// remapping or constructing any automata.
pub fn merge_internal_id_maps_only(
    inputs: &[&InternalIdMap],
    num_tokenizer_states: usize,
    max_token_id: u32,
) -> InternalIdMap {
    assert!(!inputs.is_empty(), "merge_internal_id_maps_only called with empty inputs");
    if inputs.len() == 1 {
        return inputs[0].clone();
    }
    build_unified_global_id_map(inputs, num_tokenizer_states, max_token_id).0
}

fn build_unified_global_token_id_map_sparse_generic(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> (ManyToOneIdMap, Vec<Vec<Vec<u32>>>) {
    let maps = inputs.iter().map(|input| &input.vocab_tokens).collect::<Vec<_>>();
    let (map, fanout) = refine_sparse_memberships(&maps, max_token_id as usize + 1, false, true);
    (map, fanout.expect("requested sparse membership fanout"))
}

fn build_unified_global_token_id_map_generic(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> ManyToOneIdMap {
    let maps = inputs.iter().map(|input| &input.vocab_tokens).collect::<Vec<_>>();
    refine_observation_classes(&maps, max_token_id as usize + 1, true, true)
}

fn build_unified_global_token_id_map_disjoint(
    inputs: &[&InternalIdMap],
    max_token_id: u32,
) -> Option<(ManyToOneIdMap, Option<Vec<Vec<u32>>>)> {
    if !token_domains_disjoint(inputs.iter().copied(), max_token_id as usize + 1) {
        return None;
    }
    let (map, direct) = build_unified_global_token_id_map_assume_disjoint(inputs, max_token_id);
    Some((map, Some(direct)))
}

fn build_direct_local_to_global_token_map(local_to_global: &[u32]) -> Vec<Vec<u32>> {
    local_to_global
        .iter()
        .map(|&global_class| {
            if global_class == u32::MAX {
                Vec::new()
            } else {
                vec![global_class]
            }
        })
        .collect()
}

/// Both dimensions transport weights along the exact common partition.
fn build_local_to_global_tsid_map(local: &InternalIdMap, global: &InternalIdMap) -> Vec<Vec<u32>> {
    local_to_refined_classes(&local.tokenizer_states, &global.tokenizer_states)
}

fn build_local_to_global_token_map(local: &InternalIdMap, global: &InternalIdMap) -> Vec<Vec<u32>> {
    local_to_refined_classes(&local.vocab_tokens, &global.vocab_tokens)
}

/// Remap all weights in an NWA from local TSID/token space to global space.
fn remap_nwa_with_maps(
    nwa: &mut NWA,
    tsids: &[Vec<u32>],
    tokens: &[Vec<u32>],
    global: &InternalIdMap,
) {
    remap_weights_with_token_axis(
        &mut nwa.weight_refs_mut(), tsids, TokenAxisMap::Explicit(tokens),
        global.num_tsids() as usize, global.num_internal_tokens() as usize,
    );
    for state in nwa.states_mut() {
        if state.final_weight.as_ref().is_some_and(Weight::is_empty) {
            state.final_weight = None;
        }
        for targets in state.transitions.values_mut() {
            targets.retain(|(_, weight)| !weight.is_empty());
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
        state.epsilons.retain(|(_, weight)| !weight.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singleton_weight(token: u32) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(std::iter::once(token..=token)),
        )))
    }

    fn token_range_weight(start: u32, end: u32) -> Weight {
        Weight::from_per_tsid_token_sets(std::iter::once((
            0,
            RangeSetBlaze::from_iter(std::iter::once(start..=end)),
        )))
    }

    fn generic_union(inputs: &[DWA]) -> DWA {
        let mut nwa = NWA::new(1, 2);
        let mut body = nwa.body();
        for input in inputs {
            let branch = input.to_nwa();
            body = nwa.union_in_place(&branch, &body);
        }
        nwa.set_start_states(body.start_states);
        determinize(&nwa).expect("test NWA should determinize")
    }

    #[test]
    fn direct_two_parser_union_keeps_correlated_residuals_and_singleton_rows() {
        let mut left = DWA::new(1, 2);
        let l1 = left.add_state();
        let l2 = left.add_state();
        left.add_transition(0, 7, l1, token_range_weight(0, 1));
        left.add_transition(l1, 8, l2, token_range_weight(0, 2));
        left.set_final_weight(l1, singleton_weight(0));
        left.set_final_weight(l2, singleton_weight(0));
        let mut right = DWA::new(1, 2);
        let r1 = right.add_state();
        let r2 = right.add_state();
        right.add_transition(0, 7, r1, token_range_weight(1, 2));
        right.add_transition(r1, 9, r2, token_range_weight(0, 2));
        right.set_final_weight(0, singleton_weight(1));
        right.set_final_weight(r1, singleton_weight(2));
        right.set_final_weight(r2, singleton_weight(2));

        let direct = union_two_parser_dwas_direct(left.clone(), right.clone());
        let generic = generic_union(&[left.clone(), right.clone()]);
        for word in all_words(&[7, 8, 9, DEFAULT_LABEL], 5) {
            let expected = left.eval_word(&word).union(&right.eval_word(&word));
            assert_eq!(direct.eval_word(&word), expected, "direct word={word:?}");
            assert_eq!(generic.eval_word(&word), expected, "generic word={word:?}");
        }
        assert!(find_difference(&direct, &generic).unwrap().is_none());
    }

    #[test]
    fn direct_two_parser_union_matches_generated_weighted_automata() {
        // All operations are exact finite-set union/intersection. Include
        // start finality, empty/full weights, and the raw fallback label used
        // by the existing generic parser-family union. Its input contract is
        // acyclic: the generic determinizer explicitly rejects cyclic NWAs.
        fn next(seed: &mut u64) -> u64 {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed
        }
        fn weight(seed: &mut u64) -> Weight {
            let bits = next(seed);
            if bits.is_multiple_of(11) { return Weight::all(); }
            Weight::from_per_tsid_token_sets((0..2).map(|tsid| {
                let tokens = (0..3).filter(|token| bits & (1 << (tsid * 3 + token)) != 0);
                (tsid, RangeSetBlaze::from_iter(tokens))
            }))
        }
        fn machine(seed: &mut u64) -> DWA {
            let mut result = DWA::new(2, 2);
            for _ in 1..4 { result.add_state(); }
            for state in 0..4 {
                if next(seed) & 1 != 0 { result.set_final_weight(state, weight(seed)); }
                for label in [0, 7, 8, DEFAULT_LABEL] {
                    if state == 3 { continue; }
                    if next(seed).is_multiple_of(3) { continue; }
                    let target = state + 1 + (next(seed) % u64::from(3 - state)) as u32;
                    result.add_transition(state, label, target, weight(seed));
                }
            }
            result
        }
        let words = all_words(&[0, 7, 8, DEFAULT_LABEL], 4);
        for case in 1..=96 {
            let mut seed = 0x9e3779b97f4a7c15_u64.wrapping_mul(case);
            let left = machine(&mut seed);
            let right = machine(&mut seed);
            let direct = union_two_parser_dwas_direct(left.clone(), right.clone());
            let mut nwa = NWA::new(2, 2);
            let mut body = nwa.body();
            for source in [&left, &right] {
                body = nwa.union_in_place(&source.to_nwa(), &body);
            }
            nwa.set_start_states(body.start_states);
            let generic = determinize(&nwa).expect("finite test weights must determinize");
            for word in &words {
                let expected = left.eval_word(word).union(&right.eval_word(word));
                assert_eq!(direct.eval_word(word), expected, "case={case} word={word:?}");
                assert_eq!(generic.eval_word(word), expected, "generic case={case} word={word:?}");
            }
            assert!(find_difference(&direct, &generic).unwrap().is_none(), "case={case}");
        }
    }

    #[test]
    fn direct_two_parser_union_preserves_generic_fallback_row_selection() {
        fn fallback_eval(dwa: &DWA, word: &[i32]) -> Weight {
            let mut state = dwa.start_state();
            let mut path = Weight::all();
            for label in word {
                let row = &dwa.states()[state as usize];
                let Some((target, weight)) = row.transitions.get(label)
                    .or_else(|| row.transitions.get(&DEFAULT_LABEL)) else {
                        return Weight::empty();
                    };
                path = path.intersection(weight);
                state = *target;
            }
            dwa.states()[state as usize].final_weight.as_ref()
                .map(|weight| path.intersection(weight)).unwrap_or_else(Weight::empty)
        }
        for padded in [false, true] {
        for boundary in 0..32 {
            let mut left = DWA::new(1, 2);
            let l1 = left.add_state();
            let l2 = left.add_state();
            left.add_transition(0, 7, l1, token_range_weight(0, 1));
            left.add_transition(l1, DEFAULT_LABEL, l2, token_range_weight(0, 1));
            if boundary & 1 != 0 { left.add_transition(l1, 8, l2, singleton_weight(2)); }
            if boundary & 16 != 0 { left.add_transition(l1, 8, l2, Weight::empty()); }
            left.set_final_weight(l2, token_range_weight(0, 2));
            let mut right = DWA::new(1, 2);
            let r1 = right.add_state();
            let r2 = right.add_state();
            right.add_transition(0, if boundary & 8 == 0 { 7 } else { 10 }, r1,
                token_range_weight(1, 2));
            if boundary & 2 != 0 { right.add_transition(r1, 8, r2, singleton_weight(1)); }
            if boundary & 4 != 0 { right.add_transition(r1, DEFAULT_LABEL, r2, singleton_weight(2)); }
            right.set_final_weight(r2, Weight::all());
            // Exercise both sides of the generic singleton normalization
            // threshold without mutating process-wide environment variables.
            if padded {
                while left.num_states() < 16_384 { left.add_state(); }
            }
            let direct = union_two_parser_dwas_direct(left.clone(), right.clone());
            let generic = generic_union(&[left, right]);
            for word in all_words(&[7, 8, 9], 3) {
                assert_eq!(fallback_eval(&direct, &word), fallback_eval(&generic, &word),
                    "padded={padded} boundary={boundary} word={word:?}");
            }
        }
        }
    }

    fn parser_fallback_eval(dwa: &DWA, word: &[i32]) -> Weight {
        let mut state = dwa.start_state();
        let mut path = Weight::all();
        for label in word {
            let row = &dwa.states()[state as usize];
            let Some((target, weight)) = row.transitions.get(label)
                .or_else(|| row.transitions.get(&DEFAULT_LABEL)) else {
                    return Weight::empty();
                };
            path = path.intersection(weight);
            state = *target;
        }
        dwa.states()[state as usize].final_weight.as_ref()
            .map(|weight| path.intersection(weight)).unwrap_or_else(Weight::empty)
    }

    #[test]
    fn parser_family_union_preserves_each_sources_fallback_contribution() {
        let mut left = DWA::new(1, 1);
        let left_middle = left.add_state();
        let left_final = left.add_state();
        left.add_transition(0, 7, left_middle, singleton_weight(0));
        left.add_transition(left_middle, DEFAULT_LABEL, left_final, singleton_weight(0));
        left.set_final_weight(left_final, Weight::all());
        let mut right = DWA::new(1, 1);
        let right_middle = right.add_state();
        let right_final = right.add_state();
        right.add_transition(0, 7, right_middle, singleton_weight(1));
        right.add_transition(right_middle, 8, right_final, singleton_weight(1));
        right.set_final_weight(right_final, Weight::all());
        let id_map = InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(vec![0], 1),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(vec![0, 1], 2),
            deferred_vocab_singleton_original_ids: None,
        };
        let expected = parser_fallback_eval(&left, &[7, 8])
            .union(&parser_fallback_eval(&right, &[7, 8]));
        assert_eq!(expected, token_range_weight(0, 1));
        let merged = merge_mapped_parser_dwas(vec![
            MappedArtifact::new(left, id_map.clone()),
            MappedArtifact::new(right, id_map),
        ], 1, 1);
        assert_eq!(parser_fallback_eval(merged.artifact(), &[7, 8]), expected,
            "one family's explicit label must not suppress the other family's default edge");
    }

    #[test]
    fn fallback_semantic_union_matches_independent_operands_with_empty_edges_and_cycles() {
        let words = all_words(&[7, 8, 9, 10], 4);
        for variant in 0..64u32 {
            let mut left = DWA::new(1, 2);
            let l1 = left.add_state(); let l2 = left.add_state();
            left.add_transition(0, 7, l1, token_range_weight(0, 1));
            left.add_transition(l1, DEFAULT_LABEL, l2, token_range_weight(0, 2));
            if variant & 1 != 0 { left.add_transition(l1, 8, l2, singleton_weight(2)); }
            if variant & 2 != 0 { left.add_transition(l1, 9, l2, Weight::empty()); }
            if variant & 4 != 0 { left.add_transition(l2, DEFAULT_LABEL, l1, singleton_weight(0)); }
            left.set_final_weight(l2, token_range_weight(0, 2));
            let mut right = DWA::new(1, 2);
            let r1 = right.add_state(); let r2 = right.add_state();
            right.add_transition(0, if variant & 8 != 0 { 10 } else { 7 }, r1, token_range_weight(1, 2));
            right.add_transition(r1, 8, r2, singleton_weight(2));
            if variant & 16 != 0 { right.add_transition(r1, DEFAULT_LABEL, r2, singleton_weight(1)); }
            if variant & 32 != 0 { right.add_transition(r1, 9, r2, Weight::empty()); }
            right.add_transition(r2, 7, r1, token_range_weight(1, 2));
            right.set_final_weight(r2, token_range_weight(0, 2));
            let merged = union_two_parser_dwas_with_fallback(left.clone(), right.clone());
            for word in &words {
                let expected = parser_fallback_eval(&left, word).union(&parser_fallback_eval(&right, word));
                assert_eq!(parser_fallback_eval(&merged, word), expected, "variant={variant} word={word:?}");
            }
        }
    }

    #[test]
    fn fallback_semantic_union_preserves_singleton_normalization_and_rejecting_barriers() {
        for padded in [false, true] {
            let mut left = DWA::new(1, 2);
            let l1 = left.add_state(); let l2 = left.add_state();
            left.add_transition(0, 7, l1, singleton_weight(0));
            left.add_transition(l1, DEFAULT_LABEL, l2, token_range_weight(0, 2));
            // Structurally present, but empty after the incoming residual.
            left.add_transition(l1, 8, l2, singleton_weight(1));
            left.add_transition(l1, 9, l2, Weight::empty());
            left.set_final_weight(l2, Weight::all());
            let mut right = DWA::new(1, 2);
            let r1 = right.add_state(); let r2 = right.add_state();
            right.add_transition(0, 10, r1, singleton_weight(2));
            right.add_transition(r1, DEFAULT_LABEL, r2, singleton_weight(2));
            right.set_final_weight(r2, Weight::all());
            if padded { while left.num_states() < 16_384 { left.add_state(); } }
            let merged = union_two_parser_dwas_with_fallback(left.clone(), right.clone());
            for word in all_words(&[7, 8, 9, 10, 11], 3) {
                let expected = parser_fallback_eval(&left, &word).union(&parser_fallback_eval(&right, &word));
                assert_eq!(parser_fallback_eval(&merged, &word), expected, "padded={padded} word={word:?}");
            }
        }
    }

    #[test]
    fn mapped_three_family_union_and_immediate_default_overlay_remain_exact() {
        let identity = InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(vec![0], 1),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(vec![0, 1, 2], 3),
            deferred_vocab_singleton_original_ids: None,
        };
        let mut dwas = Vec::new();
        for token in 0..3 {
            let mut dwa = DWA::new(1, 2);
            let final_state = dwa.add_state();
            dwa.add_transition(0, if token == 0 { DEFAULT_LABEL } else { 7 + token as i32 }, final_state, singleton_weight(token));
            dwa.set_final_weight(final_state, Weight::all());
            dwas.push(dwa);
        }
        let inputs = dwas.iter().cloned().map(|dwa| MappedArtifact::new(dwa, identity.clone())).collect();
        let merged = merge_mapped_parser_dwas(inputs, 1, 2);
        for word in all_words(&[7, 8, 9, 10], 2) {
            let expected = dwas.iter().fold(Weight::empty(), |acc, dwa| acc.union(&parser_fallback_eval(dwa, &word)));
            assert_eq!(parser_fallback_eval(merged.artifact(), &word), expected, "word={word:?}");
        }
        // A default-labelled immediate input must not be reduced to an
        // explicit-key overlay; explicit-only overlays still compose safely.
        assert!(immediate_dwa_accepting_edges(&dwas[0]).is_none());
        assert!(immediate_dwa_accepting_edges(&dwas[1]).is_some());
        let inputs = dwas.iter().cloned().map(|dwa| MappedArtifact::new(dwa, identity.clone())).collect();
        let (body, overlay) = merge_mapped_parser_dwas_with_top_accept(inputs, 1, 2);
        for label in [7, 8, 9, 10] {
            let expected = dwas.iter().fold(Weight::empty(), |acc, dwa| acc.union(&parser_fallback_eval(dwa, &[label])));
            let actual = parser_fallback_eval(body.artifact(), &[label])
                .union(overlay.get(&label).unwrap_or(&Weight::empty()));
            assert_eq!(actual, expected, "overlay label={label}");
        }
    }

    fn all_words(labels: &[i32], max_len: usize) -> Vec<Vec<i32>> {
        let mut words = vec![Vec::new()];
        let mut frontier = vec![Vec::new()];
        for _ in 0..max_len {
            let mut next = Vec::new();
            for prefix in frontier {
                for &label in labels {
                    let mut word = prefix.clone();
                    word.push(label);
                    words.push(word.clone());
                    next.push(word);
                }
            }
            frontier = next;
        }
        words
    }

    fn id_map_with_tsid_partition(
        original_to_internal: Vec<u32>,
        representative_original_ids: Vec<u32>,
    ) -> InternalIdMap {
        let num_tsids = representative_original_ids.len() as u32;
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                original_to_internal,
                num_tsids,
                representative_original_ids,
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                Vec::new(),
                0,
            ),
            deferred_vocab_singleton_original_ids: None,
        }
    }

    fn reference_local_to_global_tsid_map(
        local_id_map: &InternalIdMap,
        global_id_map: &InternalIdMap,
    ) -> Vec<Vec<u32>> {
        let mut result = vec![std::collections::BTreeSet::new(); local_id_map.num_tsids() as usize];
        for (original_state, &local_tsid) in local_id_map
            .tokenizer_states
            .original_to_internal
            .iter()
            .enumerate()
        {
            if local_tsid == u32::MAX {
                continue;
            }
            result[local_tsid as usize]
                .insert(global_id_map.tokenizer_states.original_to_internal[original_state]);
        }
        result
            .into_iter()
            .map(|targets| targets.into_iter().collect())
            .collect()
    }

    fn id_map_with_vocab_partition(
        original_to_internal: Vec<u32>,
        num_classes: u32,
    ) -> InternalIdMap {
        InternalIdMap {
            tokenizer_states: ManyToOneIdMap::from_original_to_internal_with_representatives(
                vec![0],
                1,
                vec![0],
            ),
            vocab_tokens: ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
                original_to_internal,
                num_classes,
            ),
            deferred_vocab_singleton_original_ids: None,
        }
    }

    #[test]
    fn sparse_overlapping_token_map_matches_generic_partition() {
        let first = id_map_with_vocab_partition(
            vec![0, 0, u32::MAX, 1, 1],
            2,
        );
        let second = id_map_with_vocab_partition(
            vec![u32::MAX, 0, 0, 1, u32::MAX],
            2,
        );
        let inputs = [&first, &second];
        let generic = build_unified_global_token_id_map_generic(&inputs, 4);
        let (sparse, local_to_global) =
            build_unified_global_token_id_map_sparse_generic(&inputs, 4);
        let mut generic_groups = generic.internal_to_originals.clone();
        let mut sparse_groups = sparse.internal_to_originals.clone();
        generic_groups.sort_unstable();
        sparse_groups.sort_unstable();
        assert_eq!(sparse_groups, generic_groups);
        for (input_index, input) in inputs.iter().enumerate() {
            for (local_class, originals) in
                input.vocab_tokens.internal_to_originals.iter().enumerate()
            {
                for &token_id in originals {
                    let global_class = sparse.original_to_internal[token_id as usize];
                    assert!(
                        local_to_global[input_index][local_class].contains(&global_class),
                        "input={input_index} local_class={local_class} token={token_id} global={global_class}",
                    );
                }
            }
        }
    }

    #[test]
    fn assume_disjoint_token_map_matches_checked_disjoint() {
        let first = id_map_with_vocab_partition(
            vec![0, 0, u32::MAX, u32::MAX, 1, 1],
            2,
        );
        let second = id_map_with_vocab_partition(
            vec![u32::MAX, u32::MAX, 0, 0, u32::MAX, u32::MAX],
            1,
        );
        let inputs = [&first, &second];
        let (assumed, assumed_maps) =
            build_unified_global_token_id_map_assume_disjoint(&inputs, 5);
        let (checked, checked_maps) = build_unified_global_token_id_map_disjoint(&inputs, 5)
            .expect("fixture token domains are disjoint");

        assert_eq!(assumed.original_to_internal, checked.original_to_internal);
        assert_eq!(assumed.internal_to_originals, checked.internal_to_originals);
        assert_eq!(
            assumed.representative_original_ids,
            checked.representative_original_ids
        );
        assert_eq!(assumed_maps, checked_maps.expect("checked path returns direct maps"));
    }

    #[test]
    fn sparse_membership_refinement_keeps_arbitrary_overlaps() {
        let first = id_map_with_vocab_partition(vec![0, 0, 0, u32::MAX], 1);
        let second = id_map_with_vocab_partition(vec![0, 0, u32::MAX, 0], 1);
        let third = id_map_with_vocab_partition(vec![0, u32::MAX, 0, 0], 1);
        let inputs = [&first, &second, &third];
        let (map, fanout) = build_unified_global_token_id_map_sparse_generic(&inputs, 3);
        assert_eq!(map.internal_to_originals, vec![vec![0], vec![1], vec![2], vec![3]]);
        assert_eq!(fanout, vec![vec![vec![0, 1, 2]], vec![vec![0, 1, 3]], vec![vec![0, 2, 3]]]);
    }

    #[test]
    fn representative_tsid_map_matches_set_reference() {
        let global = id_map_with_tsid_partition(vec![0, 0, 1, 1, 2, 2], vec![0, 2, 4]);
        let local = id_map_with_tsid_partition(vec![0, 0, 1, 1, 0, 0], vec![0, 2]);

        assert_eq!(
            build_local_to_global_tsid_map(&local, &global),
            reference_local_to_global_tsid_map(&local, &global),
        );
    }

    #[test]
    fn representative_tsid_map_skips_global_classes_unmapped_locally() {
        let global = id_map_with_tsid_partition(vec![0, 0, 1, 1, 2, 2], vec![0, 2, 4]);
        let local = id_map_with_tsid_partition(
            vec![0, 0, u32::MAX, u32::MAX, 1, 1],
            vec![0, 4],
        );

        assert_eq!(
            build_local_to_global_tsid_map(&local, &global),
            vec![vec![0], vec![2]],
        );
        assert_eq!(
            build_local_to_global_tsid_map(&local, &global),
            reference_local_to_global_tsid_map(&local, &global),
        );
    }

    #[test]
    fn contiguous_token_offset_rejects_unmapped_classes() {
        assert_eq!(contiguous_token_offset(&[9, 10, 11]), Some(9));
        assert_eq!(contiguous_token_offset(&[u32::MAX]), None);
        assert_eq!(contiguous_token_offset(&[9, u32::MAX]), None);
    }

    #[test]
    fn affine_token_remap_matches_explicit_token_map() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=2, 5..=5])),
            (1, RangeSetBlaze::from_iter([1..=4])),
        ]);
        let tsids = vec![vec![1, 3], vec![2]];
        let explicit = vec![vec![7], vec![8], vec![9], vec![10], vec![11], vec![12]];
        let mut explicit_result = weight.clone();
        let mut offset_result = weight;
        remap_weights_with_token_axis(&mut [&mut explicit_result], &tsids,
            TokenAxisMap::Explicit(&explicit), 4, 13);
        remap_weights_with_token_axis(&mut [&mut offset_result], &tsids,
            TokenAxisMap::Offset { offset: 7, local_token_count: 6 }, 4, 13);
        assert_eq!(explicit_result, offset_result);
        let expected = Weight::from_per_tsid_token_sets([
            (1, RangeSetBlaze::from_iter([7..=9, 12..=12])),
            (2, RangeSetBlaze::from_iter([8..=11])),
            (3, RangeSetBlaze::from_iter([7..=9, 12..=12])),
        ]);
        assert_eq!(offset_result, expected);
    }

    #[test]
    fn interval_remap_matches_overlapping_fallback() {
        let weight = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([0..=1])),
            (1, RangeSetBlaze::from_iter([2..=3])),
        ]);
        let mut disjoint = weight.clone();
        remap_weights_with_token_axis(&mut [&mut disjoint], &[vec![2, 4], vec![0, 1, 3]],
            TokenAxisMap::Offset { offset: 10, local_token_count: 4 }, 5, 14);
        let expected = Weight::from_per_tsid_token_sets([
            (0, RangeSetBlaze::from_iter([12..=13])),
            (1, RangeSetBlaze::from_iter([12..=13])),
            (2, RangeSetBlaze::from_iter([10..=11])),
            (3, RangeSetBlaze::from_iter([12..=13])),
            (4, RangeSetBlaze::from_iter([10..=11])),
        ]);
        assert_eq!(disjoint, expected);
        let mut overlapping = weight;
        remap_weights_with_token_axis(&mut [&mut overlapping], &[vec![0], vec![0]],
            TokenAxisMap::Offset { offset: 10, local_token_count: 4 }, 5, 14);
        assert_eq!(overlapping, Weight::from_uniform(0..=0, RangeSetBlaze::from_iter([10..=13])));
    }

    #[test]
    fn full_weight_remap_covers_exact_state_and_token_image() {
        let mut weight = Weight::all();
        remap_weights_with_token_axis(&mut [&mut weight], &[vec![2, 4], vec![0, 1, 3]],
            TokenAxisMap::Offset { offset: 10, local_token_count: 3 }, 5, 13);
        assert_eq!(weight, Weight::from_uniform(0..=4, RangeSetBlaze::from_iter([10..=12])));
        let mut partial = Weight::all();
        remap_weights_with_token_axis(&mut [&mut partial], &[vec![1], vec![3]],
            TokenAxisMap::Offset { offset: 0, local_token_count: 2 }, 5, 13);
        assert_eq!(partial, Weight::from_per_tsid_token_sets([
            (1, RangeSetBlaze::from_iter([0..=1])),
            (3, RangeSetBlaze::from_iter([0..=1])),
        ]));
    }

    #[test]
    fn direct_disjoint_union_matches_generic_subset_determinization() {
        let left_weight = singleton_weight(0);
        let mut left = DWA::new(1, 1);
        let left_mid = left.add_state();
        let left_end = left.add_state();
        left.add_transition(left.start_state(), 7, left_mid, left_weight.clone());
        left.add_transition(left_mid, 8, left_end, left_weight.clone());
        left.set_final_weight(left_mid, left_weight.clone());
        left.set_final_weight(left_end, left_weight);

        let right_weight = singleton_weight(1);
        let mut right = DWA::new(1, 1);
        let right_mid = right.add_state();
        let right_end = right.add_state();
        right.add_transition(right.start_state(), 7, right_mid, right_weight.clone());
        right.add_transition(right_mid, 9, right_end, right_weight.clone());
        right.set_final_weight(right_mid, right_weight.clone());
        right.set_final_weight(right_end, right_weight);

        let inputs = vec![left, right];
        let direct = direct_union_disjoint_token_domain_dwas(&inputs);
        let generic = generic_union(&inputs);

        for word in all_words(&[7, 8, 9], 3) {
            assert_eq!(
                direct.eval_word(&word),
                generic.eval_word(&word),
                "word={word:?}"
            );
        }
    }

    #[test]
    fn immediate_parser_graft_preserves_overlapping_path_correlation() {
        let mut immediate = DWA::new(1, 2);
        let immediate_final = immediate.add_state();
        immediate.add_transition(
            immediate.start_state(),
            7,
            immediate_final,
            token_range_weight(0, 1),
        );
        immediate.set_final_weight(immediate_final, Weight::all());

        let mut other = DWA::new(1, 2);
        let middle = other.add_state();
        let end = other.add_state();
        other.add_transition(
            other.start_state(),
            7,
            middle,
            token_range_weight(1, 2),
        );
        // Token 0 occurs on the continuation edge, but it did not take the
        // other automaton's first edge. A naive merged edge/final update would
        // incorrectly allow token 0 to continue through label 8.
        other.add_transition(middle, 8, end, token_range_weight(0, 2));
        other.set_final_weight(middle, singleton_weight(1));
        other.set_final_weight(end, Weight::all());

        let generic = generic_union(&[immediate.clone(), other.clone()]);
        let grafted = try_graft_immediate_parser_dwa(&immediate, other.clone())
            .expect("depth-one parser DWA should graft");
        let quotient = exact_quotient_acyclic_dwa(grafted);
        let overlay: BTreeMap<i32, Weight> = immediate_dwa_accepting_edges(&immediate)
            .expect("depth-one parser DWA should form an overlay")
            .into_iter()
            .collect();

        for word in all_words(&[7, 8], 3) {
            let overlay_result = match word.as_slice() {
                [label] => other
                    .eval_word(&word)
                    .union(overlay.get(label).unwrap_or(&Weight::empty())),
                _ => other.eval_word(&word),
            };
            assert_eq!(
                quotient.eval_word(&word),
                generic.eval_word(&word),
                "word={word:?}"
            );
            assert_eq!(overlay_result, generic.eval_word(&word), "overlay word={word:?}");
        }
        assert_eq!(quotient.eval_word(&[7]), token_range_weight(0, 1));
        assert_eq!(quotient.eval_word(&[7, 8]), token_range_weight(1, 2));
    }

    #[test]
    fn immediate_parser_family_is_overlayed_before_two_deep_families_merge() {
        let mut left = DWA::new(1, 2);
        let left_mid = left.add_state();
        let left_end = left.add_state();
        left.add_transition(left.start_state(), 7, left_mid, token_range_weight(0, 1));
        left.add_transition(left_mid, 8, left_end, token_range_weight(0, 1));
        left.set_final_weight(left_end, Weight::all());

        let mut right = DWA::new(1, 2);
        let right_mid = right.add_state();
        let right_end = right.add_state();
        right.add_transition(right.start_state(), 7, right_mid, token_range_weight(1, 2));
        right.add_transition(right_mid, 9, right_end, token_range_weight(1, 2));
        right.set_final_weight(right_end, Weight::all());

        let mut immediate = DWA::new(1, 2);
        let immediate_final = immediate.add_state();
        immediate.add_transition(immediate.start_state(), 10, immediate_final, singleton_weight(2));
        immediate.set_final_weight(immediate_final, Weight::all());

        let generic = generic_union(&[left.clone(), right.clone(), immediate.clone()]);
        let id_map = id_map_with_vocab_partition(vec![0, 1, 2], 3);
        let (merged, overlay) = merge_mapped_parser_dwas_with_top_accept(
            vec![
                MappedArtifact::new(left, id_map.clone()),
                MappedArtifact::new(right, id_map.clone()),
                MappedArtifact::new(immediate, id_map),
            ],
            1,
            2,
        );

        for word in all_words(&[7, 8, 9, 10], 3) {
            let mut actual = merged.artifact().eval_word(&word);
            if let [label] = word.as_slice()
                && let Some(weight) = overlay.get(label) {
                    actual = actual.union(weight);
                }
            assert_eq!(actual, generic.eval_word(&word), "word={word:?}");
        }
    }

    #[test]
    fn multiple_immediate_parser_families_stay_as_top_accept_overlay() {
        let mut deep = DWA::new(1, 2);
        let deep_mid = deep.add_state();
        let deep_end = deep.add_state();
        deep.add_transition(deep.start_state(), 7, deep_mid, token_range_weight(1, 2));
        deep.add_transition(deep_mid, 8, deep_end, token_range_weight(0, 2));
        deep.set_final_weight(deep_mid, singleton_weight(1));
        deep.set_final_weight(deep_end, Weight::all());

        let mut first = DWA::new(1, 2);
        let first_final = first.add_state();
        first.add_transition(first.start_state(), 7, first_final, singleton_weight(0));
        first.set_final_weight(first_final, Weight::all());

        let mut second = DWA::new(1, 2);
        let second_final = second.add_state();
        second.add_transition(second.start_state(), 9, second_final, singleton_weight(2));
        second.set_final_weight(second_final, Weight::all());

        let generic = generic_union(&[deep.clone(), first.clone(), second.clone()]);
        let id_map = id_map_with_vocab_partition(vec![0, 1, 2], 3);
        let (merged, overlay) = merge_mapped_parser_dwas_with_top_accept(
            vec![
                MappedArtifact::new(deep, id_map.clone()),
                MappedArtifact::new(first, id_map.clone()),
                MappedArtifact::new(second, id_map),
            ],
            1,
            2,
        );

        for word in all_words(&[7, 8, 9], 3) {
            let mut actual = merged.artifact().eval_word(&word);
            if let [label] = word.as_slice()
                && let Some(weight) = overlay.get(label) {
                    actual = actual.union(weight);
                }
            assert_eq!(actual, generic.eval_word(&word), "word={word:?}");
        }
    }
}
