use std::collections::{hash_map::Entry, BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::equivalence::find_difference;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::{DEFAULT_LABEL, is_negative_label, negative_to_positive_label};
use crate::compiler::glr::table::{
    Action, AdmissionPolicy, GLRTable, GlrTableConstruction,
};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::resolve_negatives::{
    apply_finality_fixpoint, resolve_negative_codes_in_nwa,
};
use crate::compiler::stages::templates::Templates;
use crate::ds::bitset::BitSet;
use crate::ds::weight::{ScopedWeightOpCache, Weight};

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

type TerminalBundle = BTreeMap<TerminalID, Weight>;
type BundleSignature = Vec<(TerminalID, Weight)>;
type TargetContribs = SmallVec<[(u32, Weight); 4]>;
type DeferredFinalEntries = SmallVec<[(u32, Weight); 4]>;
type FinalPathWeights = SmallVec<[Weight; 4]>;
type FinalGroups = SmallVec<[(Weight, FinalPathWeights); 4]>;

const PROFILE_PARSER_DWA_DETERMINIZE_DETAIL_ENV: &str =
    "GLRMASK_PROFILE_PARSER_DWA_DETERMINIZE_DETAIL";
const LARGE_PARSER_NWA_ALGEBRA_OPT_MIN_STATES: usize = 65_536;

#[inline]
fn add_target_contribution(contribs: &mut TargetContribs, target: u32, add: Weight) {
    if add.is_empty() {
        return;
    }

    if let Some((_, existing)) = contribs.iter_mut().find(|(existing_target, _)| *existing_target == target) {
        *existing = existing.union(&add);
    } else {
        contribs.push((target, add));
    }
}

#[derive(Default)]
struct ParserDwaDeterminizeDetail {
    states_processed: usize,
    outgoing_transitions_scanned: usize,
    intersection_calls: usize,
    nonempty_intersections: usize,
    target_contribution_pushes: usize,
    target_contribution_merges: usize,
    target_contrib_len_before_sum: usize,
    target_contrib_len_after_sum: usize,
    target_contrib_len_before_max: usize,
    target_contrib_len_after_max: usize,
    subset_key_constructions: usize,
    subset_intern_hits: usize,
    subset_intern_misses: usize,
    closure_cache_hits: usize,
    closure_cache_misses: usize,
    intersection_scan_ms: f64,
    label_processing_ms: f64,
    labels_processed: usize,
    label_contribs_sum: usize,
    label_contribs_max: usize,
    contribution_sort_ms: f64,
    edge_weight_union_ms: f64,
    closure_key_ms: f64,
    closure_lookup_ms: f64,
    local_epsilon_closure_miss_ms: f64,
    post_closure_subset_key_ms: f64,
    subset_map_lookup_ms: f64,
    add_transition_ms: f64,
    final_weight_states: usize,
    final_weight_entries: usize,
    final_weight_entries_max: usize,
    final_weight_signature_distinct: usize,
    final_weight_signature_hit_potential: usize,
    final_grouping_ms: f64,
    final_path_union_ms: f64,
    final_intersection_ms: f64,
    final_output_union_ms: f64,
    union_cache_hits: usize,
    union_cache_misses: usize,
    union_cache_key_len_sum: usize,
    union_cache_key_len_max: usize,
    union_cache_ms: f64,
    fallback_labels_expanded: usize,
    fallback_contrib_entries_duplicated: usize,
}

impl ParserDwaDeterminizeDetail {
    fn enabled() -> bool {
        std::env::var(PROFILE_PARSER_DWA_DETERMINIZE_DETAIL_ENV)
            .map(|value| {
                let normalized = value.trim().to_ascii_lowercase();
                matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    }

    fn record_target_contrib_len(&mut self, before: usize, after: usize) {
        self.target_contrib_len_before_sum += before;
        self.target_contrib_len_after_sum += after;
        self.target_contrib_len_before_max = self.target_contrib_len_before_max.max(before);
        self.target_contrib_len_after_max = self.target_contrib_len_after_max.max(after);
    }

    fn emit(&self, name: &str) {
        eprintln!(
            "[glrmask/profile][parser_dwa_determinize_detail] name={} states_processed={} outgoing_transitions_scanned={} intersection_calls={} nonempty_intersections={} target_contribution_pushes={} target_contribution_merges={} target_contrib_len_before_sum={} target_contrib_len_after_sum={} target_contrib_len_before_max={} target_contrib_len_after_max={} subset_key_constructions={} subset_intern_hits={} subset_intern_misses={} closure_cache_hits={} closure_cache_misses={} intersection_scan_ms={:.3} label_processing_ms={:.3} final_weight_states={} final_weight_entries={} final_weight_entries_max={} final_weight_signature_distinct={} final_weight_signature_hit_potential={} fallback_labels_expanded={} fallback_contrib_entries_duplicated={}",
            name,
            self.states_processed,
            self.outgoing_transitions_scanned,
            self.intersection_calls,
            self.nonempty_intersections,
            self.target_contribution_pushes,
            self.target_contribution_merges,
            self.target_contrib_len_before_sum,
            self.target_contrib_len_after_sum,
            self.target_contrib_len_before_max,
            self.target_contrib_len_after_max,
            self.subset_key_constructions,
            self.subset_intern_hits,
            self.subset_intern_misses,
            self.closure_cache_hits,
            self.closure_cache_misses,
            self.intersection_scan_ms,
            self.label_processing_ms,
            self.final_weight_states,
            self.final_weight_entries,
            self.final_weight_entries_max,
            self.final_weight_signature_distinct,
            self.final_weight_signature_hit_potential,
            self.fallback_labels_expanded,
            self.fallback_contrib_entries_duplicated,
        );
        eprintln!(
            "[glrmask/profile][parser_dwa_determinize_fine] name={} labels_processed={} label_contribs_sum={} label_contribs_max={} contribution_sort_ms={:.3} edge_weight_union_ms={:.3} closure_key_ms={:.3} closure_lookup_ms={:.3} local_epsilon_closure_miss_ms={:.3} post_closure_subset_key_ms={:.3} subset_map_lookup_ms={:.3} add_transition_ms={:.3} final_grouping_ms={:.3} final_path_union_ms={:.3} final_intersection_ms={:.3} final_output_union_ms={:.3} union_cache_hits={} union_cache_misses={} union_cache_key_len_sum={} union_cache_key_len_max={} union_cache_ms={:.3}",
            name,
            self.labels_processed,
            self.label_contribs_sum,
            self.label_contribs_max,
            self.contribution_sort_ms,
            self.edge_weight_union_ms,
            self.closure_key_ms,
            self.closure_lookup_ms,
            self.local_epsilon_closure_miss_ms,
            self.post_closure_subset_key_ms,
            self.subset_map_lookup_ms,
            self.add_transition_ms,
            self.final_grouping_ms,
            self.final_path_union_ms,
            self.final_intersection_ms,
            self.final_output_union_ms,
            self.union_cache_hits,
            self.union_cache_misses,
            self.union_cache_key_len_sum,
            self.union_cache_key_len_max,
            self.union_cache_ms,
        );
    }
}

#[inline]
fn add_target_contribution_profiled(
    contribs: &mut TargetContribs,
    target: u32,
    add: Weight,
    mut detail: Option<&mut ParserDwaDeterminizeDetail>,
) {
    if detail.is_none() {
        add_target_contribution(contribs, target, add);
        return;
    }

    if add.is_empty() {
        return;
    }

    let before = contribs.len();
    if let Some((_, existing)) = contribs
        .iter_mut()
        .find(|(existing_target, _)| *existing_target == target)
    {
        *existing = existing.union(&add);
        if let Some(detail) = detail.as_mut() {
            detail.target_contribution_merges += 1;
        }
    } else {
        contribs.push((target, add));
        if let Some(detail) = detail.as_mut() {
            detail.target_contribution_pushes += 1;
        }
    }
    if let Some(detail) = detail {
        detail.record_target_contrib_len(before, contribs.len());
    }
}

#[inline]
fn push_target_contribution_profiled(
    contribs: &mut TargetContribs,
    target: u32,
    add: Weight,
    detail: Option<&mut ParserDwaDeterminizeDetail>,
) {
    if add.is_empty() {
        return;
    }
    let before = contribs.len();
    contribs.push((target, add));
    if let Some(detail) = detail {
        detail.target_contribution_pushes += 1;
        detail.record_target_contrib_len(before, contribs.len());
    }
}

#[inline]
fn merge_sorted_target_contributions(
    contribs: &mut TargetContribs,
    mut detail: Option<&mut ParserDwaDeterminizeDetail>,
) {
    if contribs.len() < 2 {
        return;
    }
    let mut write = 0usize;
    for read in 1..contribs.len() {
        if contribs[write].0 == contribs[read].0 {
            let merged = contribs[write].1.union(&contribs[read].1);
            contribs[write].1 = merged;
            if let Some(detail) = detail.as_mut() {
                detail.target_contribution_merges += 1;
            }
        } else {
            write += 1;
            if write != read {
                contribs[write] = contribs[read].clone();
            }
        }
    }
    contribs.truncate(write + 1);
}

fn extend_target_contribs(dst: &mut TargetContribs, src: &TargetContribs) {
    for (target, weight) in src {
        add_target_contribution(dst, *target, weight.clone());
    }
}

#[derive(Debug, Clone, Copy)]
struct Branch {
    target: u32,
    bundle_id: usize,
}

#[derive(Debug, Clone)]
struct StateSummary {
    final_weight: Option<Weight>,
    epsilon_branches: Vec<(u32, Weight)>,
    branches: Vec<Branch>,
}

#[derive(Debug, Clone)]
struct StateSummaries {
    states: Vec<StateSummary>,
    start_states: Vec<u32>,
    unique_bundles: Vec<TerminalBundle>,
    bundle_accepts: Vec<bool>,
}

#[derive(Debug, Clone)]
struct DeterminizedDwaWithSupports {
    dwa: DWA,
    supports: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
enum PendingEdgeWeight {
    Immediate(Weight),
    Deferred(usize),
}

#[derive(Debug, Clone)]
struct CachedClosure {
    to_state: u32,
    edge_weight: PendingEdgeWeight,
}

#[derive(Debug, Clone, Copy)]
struct DeferredEdgeWeightPatch {
    state: u32,
    label: i32,
    job: usize,
}

fn elapsed_ms(started_at: Instant) -> f64 {
    started_at.elapsed().as_secs_f64() * 1000.0
}

fn skip_parser_dwa_minimization_env_override() -> Option<bool> {
    std::env::var("GLRMASK_SKIP_PARSER_DWA_MINIMIZE")
        .ok()
        .map(|value| {
            let trimmed = value.trim();
            !(trimmed.is_empty()
                || trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false"))
        })
}

#[inline]
fn should_skip_parser_dwa_minimization(
    _pre_minimize_states: usize,
    _pre_minimize_transitions: usize,
) -> bool {
    // Parser-DWA minimization is behavior-preserving but comparatively expensive
    // on the large-schema tail path.  The preceding construction already shares
    // continuation subgraphs and applies default/fallback normalization, so the
    // unminimized DWA is small enough for the runtime fast-transition cache.
    // Keep an escape hatch for size-sensitive experiments.
    skip_parser_dwa_minimization_env_override().unwrap_or(true)
}

#[derive(Default)]
struct ParserNwaBuildProfile {
    state_prep_ms: f64,
    compose_state_ms: f64,
    parser_nwa_build_ms: f64,
}

#[derive(Default)]
struct ParserDwaComposeDetailProfile {
    total_states: usize,
    productive_states: usize,
    total_branches: usize,
    productive_branches: usize,
    unique_bundles: usize,
    accepting_bundles: usize,
    state_init_ms: f64,
    branch_walk_ms: f64,
    memo_hit_clone_ms: f64,
    fragment_build_ms: f64,
    epsilon_link_ms: f64,
    bundle_profile_total_ms: f64,
    bundle_profile_build_group_dfas_ms: f64,
    bundle_profile_union_groups_ms: f64,
    bundle_profile_determinize_ms: f64,
    bundle_profile_minimize_ms: f64,
    bundle_profile_dwa_to_nwa_ms: f64,
    memo_hits: usize,
    memo_misses: usize,
    bundle_cache_builds: usize,
    bundle_profile_result_dwa_states: usize,
    bundle_profile_result_dwa_transitions: usize,
    bundle_profile_result_nwa_states: usize,
    bundle_profile_result_nwa_transitions: usize,
    epsilon_edges_added: usize,
    fragment_start_states_total: usize,
}

fn parser_dwa_compose_detail_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_PARSER_DWA_COMPOSE_DETAIL")
        .map(|value| value == "1")
        .unwrap_or(false)
}

fn group_terminal_edges_by_target(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, TerminalBundle> {
    let mut bundles_by_target = BTreeMap::<u32, TerminalBundle>::new();
    let mut add = |target: u32, label: i32, weight: &Weight| {
        if label < 0 || label as u32 >= grammar.num_terminals || weight.is_empty() {
            return;
        }
        bundles_by_target
            .entry(target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    };

    match terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => {
            let Some(state) = dwa.states().get(state_id as usize) else {
                return bundles_by_target;
            };
            for (&label, (target, weight)) in &state.transitions {
                add(*target, label, weight);
            }
        }
        TerminalAutomaton::TokenDeterministicNwa(nwa) => {
            let Some(state) = nwa.states().get(state_id as usize) else {
                return bundles_by_target;
            };
            assert!(
                state.epsilons.is_empty(),
                "token-deterministic terminal NWA must not contain epsilon edges",
            );
            for (&label, branches) in &state.transitions {
                for (target, weight) in branches {
                    add(*target, label, weight);
                }
            }
        }
        TerminalAutomaton::EpsilonNwa(nwa) => {
            let Some(state) = nwa.states().get(state_id as usize) else {
                return bundles_by_target;
            };
            for (&label, branches) in &state.transitions {
                for (target, weight) in branches {
                    add(*target, label, weight);
                }
            }
        }
    }

    bundles_by_target
}

fn terminal_state_final_weight(
    terminal_automaton: &TerminalAutomaton,
    state_id: usize,
) -> Option<Weight> {
    match terminal_automaton {
        TerminalAutomaton::Dwa(dwa) => dwa
            .states()
            .get(state_id)
            .and_then(|state| state.final_weight.clone()),
        TerminalAutomaton::TokenDeterministicNwa(nwa)
        | TerminalAutomaton::EpsilonNwa(nwa) => nwa
            .states()
            .get(state_id)
            .and_then(|state| state.final_weight.clone()),
    }
}

fn terminal_state_epsilon_branches(
    terminal_automaton: &TerminalAutomaton,
    state_id: usize,
) -> Vec<(u32, Weight)> {
    match terminal_automaton {
        TerminalAutomaton::EpsilonNwa(nwa) => nwa
            .states()
            .get(state_id)
            .map(|state| state.epsilons.clone())
            .unwrap_or_default(),
        TerminalAutomaton::Dwa(_) | TerminalAutomaton::TokenDeterministicNwa(_) => Vec::new(),
    }
}

fn bundle_signature(bundle: &TerminalBundle) -> BundleSignature {
    bundle
        .iter()
        .map(|(&terminal, weight)| (terminal, weight.clone()))
        .collect()
}

fn terminal_template_has_acceptance(template: &NWA) -> bool {
    template.states().iter().any(|state| state.final_weight.is_some())
}

fn terminal_bundle_has_acceptance(bundle: &TerminalBundle, templates: &Templates) -> bool {
    bundle.iter().any(|(&terminal, weight)| {
        !weight.is_empty()
            && templates
                .by_terminal_nwa
                .get(&terminal)
                .is_some_and(terminal_template_has_acceptance)
    })
}


fn build_state_summaries(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> StateSummaries {
    let state_count = terminal_automaton.num_states();
    let mut branches_by_state: Vec<Vec<Branch>> = Vec::with_capacity(state_count);
    let mut bundle_ids_by_signature: FxHashMap<BundleSignature, usize> = FxHashMap::default();
    let mut unique_bundles: Vec<TerminalBundle> = Vec::new();

    for state_id in 0..state_count {
        let bundles_by_target =
            group_terminal_edges_by_target(terminal_automaton, grammar, state_id as u32);
        let mut branches = Vec::with_capacity(bundles_by_target.len());
        for (target, bundle) in bundles_by_target {
            let signature = bundle_signature(&bundle);
            let bundle_id = if let Some(&bundle_id) = bundle_ids_by_signature.get(&signature) {
                bundle_id
            } else {
                let bundle_id = unique_bundles.len();
                bundle_ids_by_signature.insert(signature, bundle_id);
                unique_bundles.push(bundle);
                bundle_id
            };
            branches.push(Branch { target, bundle_id });
        }
        branches_by_state.push(branches);
    }

    let bundle_accepts: Vec<bool> = unique_bundles
        .iter()
        .map(|bundle| terminal_bundle_has_acceptance(bundle, templates))
        .collect();

    let states = (0..state_count)
        .map(|state_id| StateSummary {
            final_weight: terminal_state_final_weight(terminal_automaton, state_id),
            epsilon_branches: terminal_state_epsilon_branches(terminal_automaton, state_id),
            branches: std::mem::take(&mut branches_by_state[state_id]),
        })
        .collect();

    StateSummaries {
        states,
        start_states: terminal_automaton.start_states(),
        unique_bundles,
        bundle_accepts,
    }
}

fn union_final_weight(slot: &mut Option<Weight>, add: Weight) -> bool {
    if add.is_empty() {
        return false;
    }

    match slot {
        Some(existing) => {
            let updated = existing.union(&add);
            if updated != *existing {
                *existing = updated;
                true
            } else {
                false
            }
        }
        None => {
            *slot = Some(add);
            true
        }
    }
}

fn parser_state_label(label: i32, num_parser_states: u32) -> Option<u32> {
    if label >= 0 && (label as u32) < num_parser_states {
        Some(label as u32)
    } else {
        None
    }
}

fn top_row_action_is_unconditionally_applicable(_action: &Action) -> bool {
    // RowPresenceExact actions originate from a parser row whose terminal
    // domain is an exact admission set. Guarded stack shifts are a lowered
    // representation of that row's already-valid action, not a weaker
    // admission predicate. Their guards select the exact stack effect; they do
    // not make the terminal cease to be admissible from the row's top state.
    true
}

fn immediate_completion_weights_by_terminal(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
) -> BTreeMap<TerminalID, Weight> {
    let mut complete_by_terminal = BTreeMap::<TerminalID, Weight>::new();
    for start_state in terminal_automaton.start_states() {
        for (target, bundle) in
            group_terminal_edges_by_target(terminal_automaton, grammar, start_state)
        {
            let Some(target_final) = terminal_state_final_weight(terminal_automaton, target as usize)
            else {
                continue;
            };
            for (terminal, edge_weight) in bundle {
                let complete = edge_weight.intersection(&target_final);
                if complete.is_empty() {
                    continue;
                }
                complete_by_terminal
                    .entry(terminal)
                    .and_modify(|existing| *existing = existing.union(&complete))
                    .or_insert(complete);
            }
        }
    }
    complete_by_terminal
}

fn immediate_acceptance_certificate_parts(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> BTreeMap<i32, Vec<Weight>> {
    if table.admission_policy != AdmissionPolicy::RowPresenceExact {
        return BTreeMap::new();
    }
    let complete_by_terminal = immediate_completion_weights_by_terminal(terminal_automaton, grammar);
    table
        .action
        .iter()
        .enumerate()
        .filter_map(|(parser_top, row)| {
            let parts = row
                .iter()
                .filter_map(|(terminal, action)| {
                    (top_row_action_is_unconditionally_applicable(action))
                        .then(|| complete_by_terminal.get(&terminal).cloned())
                        .flatten()
                })
                .collect::<Vec<_>>();
            (!parts.is_empty()).then_some((parser_top as i32, parts))
        })
        .collect()
}

fn immediate_acceptance_certificates(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Vec<Weight> {
    let parts = immediate_acceptance_certificate_parts(terminal_automaton, grammar, table);
    let use_signature_cache =
        std::env::var_os("GLRMASK_DISABLE_IMMEDIATE_ACCEPT_SIGNATURE_CACHE").is_none();
    if !use_signature_cache {
        return (0..table.num_states)
            .map(|parser_top| {
                parts
                    .get(&(parser_top as i32))
                    .map(|weights| Weight::union_all(weights.iter()))
                    .unwrap_or_else(Weight::empty)
            })
            .collect();
    }

    // Parser rows frequently repeat the same set of immediately-accepted
    // terminals.  The old path recomputed the identical multi-way token-set
    // union once per parser state.  Intern the exact pointer signature and
    // materialize each union once; terminal completion weights remain alive for
    // this whole scope, so pointer identity is stable and exact.
    let mut union_by_signature = FxHashMap::<Vec<usize>, Weight>::default();
    let mut nonempty_rows = 0usize;
    let mut part_refs = 0usize;
    let result = (0..table.num_states)
        .map(|parser_top| {
            let Some(weights) = parts.get(&(parser_top as i32)) else {
                return Weight::empty();
            };
            nonempty_rows += 1;
            part_refs += weights.len();
            let mut signature = weights.iter().map(Weight::ptr_key).collect::<Vec<_>>();
            signature.sort_unstable();
            signature.dedup();
            if let Some(cached) = union_by_signature.get(&signature) {
                return cached.clone();
            }
            let union = Weight::union_all(weights.iter());
            union_by_signature.insert(signature, union.clone());
            union
        })
        .collect();
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][immediate_accept_signature_cache] nonempty_rows={} part_refs={} distinct_signatures={} reused_rows={}",
            nonempty_rows,
            part_refs,
            union_by_signature.len(),
            nonempty_rows.saturating_sub(union_by_signature.len()),
        );
    }
    result
}

pub fn try_build_immediate_parser_top_accept_parts(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<i32, Vec<Weight>>> {
    terminal_automaton_is_immediate_completion(terminal_automaton, grammar, table)
        .then(|| immediate_acceptance_certificate_parts(terminal_automaton, grammar, table))
}

pub fn try_build_immediate_terminal_completion_weights(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<TerminalID, Weight>> {
    terminal_automaton_is_immediate_completion(terminal_automaton, grammar, table)
        .then(|| immediate_completion_weights_by_terminal(terminal_automaton, grammar))
}

#[cfg(test)]
fn direct_regular_action_targets(action: &Action) -> Option<SmallVec<[u32; 4]>> {
    match action {
        Action::Shift(target, true) => Some(SmallVec::from_slice(&[*target])),
        Action::StackShifts(shifts) => {
            let mut targets = SmallVec::<[u32; 4]>::new();
            for shift in shifts {
                if shift.pop != 1 || shift.pushes.len() != 1 {
                    return None;
                }
                targets.push(shift.pushes[0]);
            }
            targets.sort_unstable();
            targets.dedup();
            (!targets.is_empty()).then_some(targets)
        }
        _ => None,
    }
}

/// Compute exact token acceptance for a constant-depth direct-regular parser by
/// solving the weighted product of terminal-automaton states and parser-top
/// states. All product edges are epsilon edges because the sole parser-stack
/// symbol has already selected the product's parser-top coordinate.
pub fn try_build_direct_regular_parser_top_accept_parts(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<i32, Vec<Weight>>> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    if table.admission_policy != AdmissionPolicy::RowPresenceExact
        || table.num_states == 0
        || automaton.states.is_empty()
        || automaton.start_states.is_empty()
    {
        return None;
    }
    let started_at = Instant::now();
    let terminal_state_count = terminal_automaton.num_states();
    if terminal_state_count == 0 {
        return None;
    }

    // Runtime parser state 0 is synthetic. Direct-regular NFA state N maps to
    // parser state N+1, matching direct GLR table construction. Working from
    // the sparse NFA rather than its epsilon-closed action rows avoids copying
    // a suffix frontier into every preceding state of skip-chain grammars.
    let parser_state_count = automaton.states.len().checked_add(1)?;
    if parser_state_count != table.num_states as usize {
        return None;
    }
    let product_state_count = terminal_state_count.checked_mul(parser_state_count)?;
    if product_state_count > u32::MAX as usize {
        return None;
    }
    let product_state = |terminal_state: usize, parser_state: usize| -> u32 {
        (terminal_state * parser_state_count + parser_state) as u32
    };

    let mut product = NWA::new(0, 0);
    for _ in 0..product_state_count {
        product.add_state();
    }

    // A final terminal-automaton state means one grammar terminal has
    // completed. Any live direct-regular parser state is a valid post-commit
    // continuation, exactly as in the previous table-product construction.
    for terminal_state in 0..terminal_state_count {
        if let Some(final_weight) = terminal_state_final_weight(terminal_automaton, terminal_state)
            && !final_weight.is_empty()
        {
            for parser_state in 0..parser_state_count {
                product.set_final_weight(
                    product_state(terminal_state, parser_state),
                    final_weight.clone(),
                );
            }
        }

        // Terminal-automaton epsilon edges do not change parser position.
        for (target, weight) in
            terminal_state_epsilon_branches(terminal_automaton, terminal_state)
        {
            if weight.is_empty() || target as usize >= terminal_state_count {
                continue;
            }
            for parser_state in 0..parser_state_count {
                product.add_epsilon(
                    product_state(terminal_state, parser_state),
                    product_state(target as usize, parser_state),
                    weight.clone(),
                );
            }
        }
    }

    // Parser epsilon edges do not change terminal-automaton state. State 0 has
    // epsilon edges to every direct-regular start state; NFA state N is N+1.
    let all = Weight::all();
    for terminal_state in 0..terminal_state_count {
        let terminal_product_base = terminal_state * parser_state_count;
        for &start in &automaton.start_states {
            if start as usize >= automaton.states.len() {
                return None;
            }
            product.add_epsilon(
                (terminal_product_base) as u32,
                (terminal_product_base + start as usize + 1) as u32,
                all.clone(),
            );
        }
        for (source, state) in automaton.states.iter().enumerate() {
            for &target in &state.epsilons {
                if target as usize >= automaton.states.len() {
                    return None;
                }
                product.add_epsilon(
                    (terminal_product_base + source + 1) as u32,
                    (terminal_product_base + target as usize + 1) as u32,
                    all.clone(),
                );
            }
        }
    }

    // Index the sparse direct-regular terminal edges by grammar terminal.
    let mut parser_sources_by_terminal =
        vec![Vec::<(u32, SmallVec<[u32; 4]>)>::new(); grammar.num_terminals as usize];
    for (source, state) in automaton.states.iter().enumerate() {
        for (&terminal, targets) in &state.transitions {
            if terminal >= grammar.num_terminals || targets.is_empty() {
                return None;
            }
            let mut mapped = SmallVec::<[u32; 4]>::new();
            for &target in targets {
                if target as usize >= automaton.states.len() {
                    return None;
                }
                mapped.push(target + 1);
            }
            mapped.sort_unstable();
            mapped.dedup();
            parser_sources_by_terminal[terminal as usize].push((source as u32 + 1, mapped));
        }
    }

    // Match terminal-automaton labelled edges with sparse parser-NFA edges.
    for terminal_state in 0..terminal_state_count {
        for (target, bundle) in
            group_terminal_edges_by_target(terminal_automaton, grammar, terminal_state as u32)
        {
            if target as usize >= terminal_state_count {
                return None;
            }
            for (terminal, edge_weight) in bundle {
                if edge_weight.is_empty() {
                    continue;
                }
                for (source, parser_targets) in
                    &parser_sources_by_terminal[terminal as usize]
                {
                    let from = product_state(terminal_state, *source as usize);
                    for &parser_target in parser_targets {
                        product.add_epsilon(
                            from,
                            product_state(target as usize, parser_target as usize),
                            edge_weight.clone(),
                        );
                    }
                }
            }
        }
    }

    apply_finality_fixpoint(&mut product);
    let terminal_starts = terminal_automaton.start_states();
    let mut parts = BTreeMap::new();
    for parser_top in 0..parser_state_count {
        let weights = terminal_starts
            .iter()
            .filter_map(|&terminal_start| {
                product
                    .states()
                    .get(product_state(terminal_start as usize, parser_top) as usize)
                    .and_then(|state| state.final_weight.clone())
                    .filter(|weight| !weight.is_empty())
            })
            .collect::<Vec<_>>();
        if !weights.is_empty() {
            parts.insert(parser_top as i32, weights);
        }
    }

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][direct_regular_parser_sparse_product] terminal_states={} parser_states={} product_states={} product_edges={} labels={} part_refs={} unique_weights={} total_ms={:.3}",
            terminal_state_count,
            parser_state_count,
            product_state_count,
            product.num_transitions(),
            parts.len(),
            parts.values().map(Vec::len).sum::<usize>(),
            parts
                .values()
                .flatten()
                .map(Weight::ptr_key)
                .collect::<rustc_hash::FxHashSet<_>>()
                .len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(parts)
}

#[cfg(test)]
fn try_build_direct_regular_parser_top_accept_parts_table_product_reference(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<BTreeMap<i32, Vec<Weight>>> {
    if grammar.direct_regular_automaton.is_none()
        || table.admission_policy != AdmissionPolicy::RowPresenceExact
        || table.num_states == 0
    {
        return None;
    }
    let started_at = Instant::now();
    let parser_state_count = table.num_states as usize;
    let terminal_state_count = terminal_automaton.num_states();
    if terminal_state_count == 0 {
        return None;
    }

    let mut parser_sources_by_terminal =
        vec![Vec::<(u32, SmallVec<[u32; 4]>)>::new(); grammar.num_terminals as usize];
    for (source, row) in table.action.iter().enumerate() {
        for (terminal, action) in row {
            if terminal >= grammar.num_terminals {
                continue;
            }
            let targets = direct_regular_action_targets(action)?;
            if targets.iter().any(|&target| target >= table.num_states) {
                return None;
            }
            parser_sources_by_terminal[terminal as usize].push((source as u32, targets));
        }
    }

    let product_state = |terminal_state: usize, parser_state: u32| -> u32 {
        (terminal_state * parser_state_count + parser_state as usize) as u32
    };
    let product_state_count = terminal_state_count.checked_mul(parser_state_count)?;
    let mut product = NWA::new(0, 0);
    for _ in 0..product_state_count {
        product.add_state();
    }

    for terminal_state in 0..terminal_state_count {
        if let Some(final_weight) = terminal_state_final_weight(terminal_automaton, terminal_state)
            && !final_weight.is_empty()
        {
            for parser_state in 0..table.num_states {
                product.set_final_weight(
                    product_state(terminal_state, parser_state),
                    final_weight.clone(),
                );
            }
        }

        for (target, weight) in
            terminal_state_epsilon_branches(terminal_automaton, terminal_state)
        {
            if weight.is_empty() || target as usize >= terminal_state_count {
                continue;
            }
            for parser_state in 0..table.num_states {
                product.add_epsilon(
                    product_state(terminal_state, parser_state),
                    product_state(target as usize, parser_state),
                    weight.clone(),
                );
            }
        }

        for (target, bundle) in
            group_terminal_edges_by_target(terminal_automaton, grammar, terminal_state as u32)
        {
            if target as usize >= terminal_state_count {
                return None;
            }
            for (terminal, edge_weight) in bundle {
                if edge_weight.is_empty() {
                    continue;
                }
                for (source, parser_targets) in
                    &parser_sources_by_terminal[terminal as usize]
                {
                    let from = product_state(terminal_state, *source);
                    for &parser_target in parser_targets {
                        product.add_epsilon(
                            from,
                            product_state(target as usize, parser_target),
                            edge_weight.clone(),
                        );
                    }
                }
            }
        }
    }

    apply_finality_fixpoint(&mut product);
    let terminal_starts = terminal_automaton.start_states();
    let mut parts = BTreeMap::new();
    for parser_top in 0..table.num_states {
        let weights = terminal_starts
            .iter()
            .filter_map(|&terminal_start| {
                product
                    .states()
                    .get(product_state(terminal_start as usize, parser_top) as usize)
                    .and_then(|state| state.final_weight.clone())
                    .filter(|weight| !weight.is_empty())
            })
            .collect::<Vec<_>>();
        if !weights.is_empty() {
            parts.insert(parser_top as i32, weights);
        }
    }

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][direct_regular_parser_product] terminal_states={} parser_states={} product_states={} product_edges={} labels={} part_refs={} unique_weights={} total_ms={:.3}",
            terminal_state_count,
            parser_state_count,
            product_state_count,
            product.num_transitions(),
            parts.len(),
            parts.values().map(Vec::len).sum::<usize>(),
            parts
                .values()
                .flatten()
                .map(Weight::ptr_key)
                .collect::<rustc_hash::FxHashSet<_>>()
                .len(),
            started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Some(parts)
}

fn terminal_automaton_is_immediate_completion(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> bool {
    if table.admission_policy != AdmissionPolicy::RowPresenceExact {
        return false;
    }
    let TerminalAutomaton::Dwa(dwa) = terminal_automaton else {
        return false;
    };
    let Some(start) = dwa.states().get(dwa.start_state() as usize) else {
        return false;
    };

    if start
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
    {
        return false;
    }

    let mut saw_edge = false;
    for (&label, (target, edge_weight)) in &start.transitions {
        if edge_weight.is_empty() {
            continue;
        }
        if label < 0 || label as u32 >= grammar.num_terminals {
            return false;
        }
        let Some(target_state) = dwa.states().get(*target as usize) else {
            return false;
        };
        if target_state
            .transitions
            .values()
            .any(|(_, weight)| !weight.is_empty())
        {
            return false;
        }
        let Some(target_final) = target_state.final_weight.as_ref() else {
            return false;
        };
        if !edge_weight.is_subset(target_final) {
            return false;
        }
        saw_edge = true;
    }
    saw_edge
}

pub fn try_build_immediate_parser_dwa(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Option<DWA> {
    if !terminal_automaton_is_immediate_completion(terminal_automaton, grammar, table) {
        return None;
    }
    let certificates = immediate_acceptance_certificates(terminal_automaton, grammar, table);
    let mut parser_dwa = DWA::new(0, 0);
    let final_state = parser_dwa.add_state();
    parser_dwa.set_final_weight(final_state, Weight::all());
    for (parser_top, weight) in certificates.into_iter().enumerate() {
        if !weight.is_empty() {
            parser_dwa.add_transition(0, parser_top as i32, final_state, weight);
        }
    }
    Some(parser_dwa)
}

fn collapse_immediate_acceptance_certificates(
    parser_dwa: &mut DWA,
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> usize {
    if parser_dwa.states().is_empty() {
        return 0;
    }
    let certificates = immediate_acceptance_certificates(terminal_automaton, grammar, table);
    let start_state = parser_dwa.start_state();
    let mut rewrites = Vec::<(i32, Weight)>::new();
    for (&label, (_target, edge_weight)) in
        &parser_dwa.states()[start_state as usize].transitions
    {
        let Some(parser_top) = parser_state_label(label, table.num_states) else {
            continue;
        };
        if edge_weight.is_subset(&certificates[parser_top as usize]) {
            rewrites.push((label, edge_weight.clone()));
        }
    }
    if rewrites.is_empty() {
        return 0;
    }

    let sink = parser_dwa.add_state();
    parser_dwa.set_final_weight(sink, Weight::union_all(rewrites.iter().map(|(_, w)| w)));
    for (label, _) in &rewrites {
        let (target, _) = parser_dwa.states_mut()[start_state as usize]
            .transitions
            .get_mut(label)
            .expect("certified start transition disappeared");
        *target = sink;
    }
    rewrites.len()
}

fn trim_unreachable_dwa(dwa: DWA) -> DWA {
    if dwa.states().is_empty() {
        return dwa;
    }
    let old_states = dwa.states().to_vec();
    let old_start = dwa.start_state() as usize;
    let mut reachable = vec![false; old_states.len()];
    let mut queue = VecDeque::from([old_start]);
    reachable[old_start] = true;
    while let Some(state_id) = queue.pop_front() {
        for (target, weight) in old_states[state_id].transitions.values() {
            let target = *target as usize;
            if weight.is_empty() || target >= old_states.len() || reachable[target] {
                continue;
            }
            reachable[target] = true;
            queue.push_back(target);
        }
    }

    let mut remap = vec![u32::MAX; old_states.len()];
    let mut new_states = Vec::with_capacity(reachable.iter().filter(|&&live| live).count());
    for (old_id, state) in old_states.iter().enumerate() {
        if reachable[old_id] {
            remap[old_id] = new_states.len() as u32;
            new_states.push(state.clone());
        }
    }
    for state in &mut new_states {
        state.transitions.retain(|_, (target, weight)| {
            if weight.is_empty() || (*target as usize) >= remap.len() {
                return false;
            }
            let mapped = remap[*target as usize];
            if mapped == u32::MAX {
                return false;
            }
            *target = mapped;
            true
        });
    }
    DWA::from_parts(new_states, remap[old_start])
}

/// Push final weights from transition-free leaf states into their incoming
/// edges, then share one `final = all` sink. Runtime evaluation already
/// intersects the accumulated path weight with the destination final weight,
/// so this is an exact weighted normalization rather than a language change.
fn collapse_final_leaf_targets(mut dwa: DWA) -> DWA {
    if dwa.states().is_empty() {
        return dwa;
    }
    let leaf_finals: Vec<Option<Weight>> = dwa
        .states()
        .iter()
        .map(|state| {
            (state.transitions.is_empty())
                .then(|| state.final_weight.clone())
                .flatten()
                .filter(|weight| !weight.is_empty())
        })
        .collect();
    if leaf_finals.iter().all(Option::is_none) {
        return dwa;
    }

    let sink = dwa.add_state();
    dwa.set_final_weight(sink, Weight::all());
    let mut changed = false;
    for state_id in 0..sink as usize {
        let mut remove = Vec::new();
        for (&label, (target, edge_weight)) in &mut dwa.states_mut()[state_id].transitions {
            let Some(final_weight) = leaf_finals
                .get(*target as usize)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let pushed = edge_weight.intersection(final_weight);
            if pushed.is_empty() {
                remove.push(label);
            } else {
                *target = sink;
                *edge_weight = pushed;
            }
            changed = true;
        }
        for label in remove {
            dwa.states_mut()[state_id].transitions.remove(&label);
        }
    }
    if !changed {
        dwa.states_mut().pop();
        return dwa;
    }
    trim_unreachable_dwa(dwa)
}

enum PossibleOutgoingIds {
    Empty,
    All,
    Some(BitSet),
}

fn build_possible_outgoing_ids_by_state(
    parser_nwa: &NWA,
    state_supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<PossibleOutgoingIds> {
    enum OutgoingIds {
        Empty,
        All,
        Some(Vec<u32>),
    }

    let num_parser_states = num_parser_states as usize;
    let all_parser_states = BitSet::all(num_parser_states);
    let state_outgoing_ids: Vec<OutgoingIds> = parser_nwa
        .states()
        .iter()
        .map(|state| {
            let mut ids = Vec::new();
            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    return OutgoingIds::All;
                }
                if let Some(parser_state_id) = parser_state_label(label, num_parser_states as u32) {
                    ids.push(parser_state_id);
                }
            }
            if ids.is_empty() {
                OutgoingIds::Empty
            } else {
                OutgoingIds::Some(ids)
            }
        })
        .collect();

    state_supports
        .iter()
        .map(|support| {
            if support.len() == 1 {
                let state_id = support[0] as usize;
                return match state_outgoing_ids.get(state_id) {
                    Some(OutgoingIds::Empty) => PossibleOutgoingIds::Empty,
                    Some(OutgoingIds::All) => PossibleOutgoingIds::All,
                    Some(OutgoingIds::Some(ids)) => {
                        let mut bitset = BitSet::new(num_parser_states);
                        for &parser_state_id in ids {
                            bitset.set(parser_state_id as usize);
                        }
                        if bitset == all_parser_states {
                            PossibleOutgoingIds::All
                        } else {
                            PossibleOutgoingIds::Some(bitset)
                        }
                    }
                    None => PossibleOutgoingIds::Empty,
                };
            }

            let mut ids = BitSet::new(num_parser_states);
            for &state_id in support {
                let Some(state_ids) = state_outgoing_ids.get(state_id as usize) else {
                    continue;
                };
                match state_ids {
                    OutgoingIds::Empty => {}
                    OutgoingIds::All => return PossibleOutgoingIds::All,
                    OutgoingIds::Some(state_ids) => {
                        for &parser_state_id in state_ids {
                            ids.set(parser_state_id as usize);
                        }
                        if ids == all_parser_states {
                            break;
                        }
                    }
                }
            }
            if ids.is_empty() {
                PossibleOutgoingIds::Empty
            } else if ids == all_parser_states {
                PossibleOutgoingIds::All
            } else {
                PossibleOutgoingIds::Some(ids)
            }
        })
        .collect()
}

fn local_epsilon_closure(
    nwa: &NWA,
    weight_by_state: &mut Vec<Option<Weight>>,
    closure_queue: &mut VecDeque<u32>,
    seed: &mut FxHashMap<u32, Weight>,
) {
    let mut seed_states: Vec<u32> = Vec::new();
    for (&state_id, weight) in seed.iter() {
        weight_by_state[state_id as usize] = Some(weight.clone());
        closure_queue.push_back(state_id);
        seed_states.push(state_id);
    }
    if seed.len() == 1 {
        let state_id = seed_states[0];
        if let Some(state) = nwa.states().get(state_id as usize) {
            if state.epsilons.is_empty() {
                closure_queue.clear();
                for &s in &seed_states {
                    weight_by_state[s as usize] = None;
                }
                return;
            }
        }
    }
    while let Some(state_id) = closure_queue.pop_front() {
        let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
            continue;
        };
        let Some(state) = nwa.states().get(state_id as usize) else {
            continue;
        };
        for (target, edge_weight) in &state.epsilons {
            let contribution = current_weight.intersection(edge_weight);
            if contribution.is_empty() {
                continue;
            }
            let target_idx = *target as usize;
            if let Some(existing) = &weight_by_state[target_idx] {
                if !contribution.is_subset(existing) {
                    weight_by_state[target_idx] = Some(existing.union(&contribution));
                    closure_queue.push_back(*target);
                    if !seed_states.contains(target) {
                        seed_states.push(*target);
                    }
                }
            } else {
                weight_by_state[target_idx] = Some(contribution);
                closure_queue.push_back(*target);
                seed_states.push(*target);
            }
        }
    }
    seed.clear();
    for &s in &seed_states {
        if let Some(w) = weight_by_state[s as usize].take() {
            seed.insert(s, w);
        }
    }
}

fn local_epsilon_closure_canonical(
    nwa: &NWA,
    weight_by_state: &mut [Option<Weight>],
    closure_queue: &mut VecDeque<u32>,
    seeds: &[(u32, Weight)],
    touched_states: &mut Vec<u32>,
    canonical: &mut Vec<(u32, Weight)>,
    weight_ops: &mut ScopedWeightOpCache,
    use_weight_cache: bool,
) {
    debug_assert!(closure_queue.is_empty());
    touched_states.clear();
    canonical.clear();

    for (state_id, weight) in seeds {
        let slot = &mut weight_by_state[*state_id as usize];
        debug_assert!(slot.is_none());
        *slot = Some(weight.clone());
        closure_queue.push_back(*state_id);
        touched_states.push(*state_id);
    }

    while let Some(state_id) = closure_queue.pop_front() {
        let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
            continue;
        };
        let Some(state) = nwa.states().get(state_id as usize) else {
            continue;
        };
        for (target, edge_weight) in &state.epsilons {
            let contribution = if use_weight_cache {
                weight_ops.intersection(&current_weight, edge_weight)
            } else {
                current_weight.intersection(edge_weight)
            };
            if contribution.is_empty() {
                continue;
            }
            let target_idx = *target as usize;
            if let Some(existing) = &weight_by_state[target_idx] {
                if !contribution.is_subset(existing) {
                    weight_by_state[target_idx] = Some(if use_weight_cache {
                        weight_ops.union(existing, &contribution)
                    } else {
                        existing.union(&contribution)
                    });
                    closure_queue.push_back(*target);
                }
            } else {
                weight_by_state[target_idx] = Some(contribution);
                closure_queue.push_back(*target);
                touched_states.push(*target);
            }
        }
    }

    touched_states.sort_unstable();
    for &state_id in touched_states.iter() {
        if let Some(weight) = weight_by_state[state_id as usize].take()
            && !weight.is_empty()
        {
            canonical.push((state_id, weight));
        }
    }
}

fn determinize_with_supports(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
) -> DeterminizedDwaWithSupports {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    #[derive(Default)]
    struct UnionAllCache {
        entries: FxHashMap<SmallVec<[usize; 16]>, Weight>,
        bulk_ops: ScopedWeightOpCache,
        ordered_keys: bool,
        profile_enabled: bool,
        hits: usize,
        misses: usize,
        key_len_sum: usize,
        key_len_max: usize,
        total_ms: f64,
    }

    impl UnionAllCache {
        fn record_elapsed(&mut self, started: Option<Instant>) {
            if let Some(started) = started {
                self.total_ms += elapsed_ms(started);
            }
        }

        fn union_all<'a>(&mut self, weights: impl IntoIterator<Item = &'a Weight>) -> Weight {
            let started = self.profile_enabled.then(Instant::now);
            let mut meaningful = SmallVec::<[&Weight; 8]>::new();
            for weight in weights {
                if weight.is_full() {
                    self.record_elapsed(started);
                    return Weight::all();
                }
                if !weight.is_empty() {
                    meaningful.push(weight);
                }
            }

            if meaningful.is_empty() {
                self.record_elapsed(started);
                return Weight::empty();
            }
            if meaningful.len() == 1 {
                self.record_elapsed(started);
                return meaningful[0].clone();
            }

            let mut key: SmallVec<[usize; 16]> =
                meaningful.iter().map(|weight| weight.ptr_key()).collect();
            // Contributions are already in deterministic target-state order.
            // Using that exact sequence as the cache key preserves correctness:
            // a different order merely misses the cache and recomputes the exact
            // union. Canonical sorting only increases sharing, while costing more
            // than the rare extra miss on parser-DWA workloads.
            if !self.ordered_keys {
                key.sort_unstable();
                key.dedup();
            }
            self.key_len_sum += key.len();
            self.key_len_max = self.key_len_max.max(key.len());

            if key.len() == 1 {
                self.record_elapsed(started);
                return meaningful[0].clone();
            }

            if let Some(weight) = self.entries.get(&key) {
                let weight = weight.clone();
                self.hits += 1;
                self.record_elapsed(started);
                return weight;
            }

            self.misses += 1;
            let weight = self.bulk_ops.union_all(meaningful.into_iter());
            self.entries.insert(key, weight.clone());
            self.record_elapsed(started);
            weight
        }
    }

    let num_nwa_states = nwa.states().len();

    // Use flat arrays for epsilon closure when NWA is small enough.
    // weight_by_state[i] = Some(weight) means state i is in the closure.
    let mut weight_by_state: Vec<Option<Weight>> = vec![None; num_nwa_states];
    let mut closure_queue: VecDeque<u32> = VecDeque::new();
    // Reusable buffer for canonicalized entries.
    let mut canon_buf: Vec<(u32, Weight)> = Vec::new();

    // Epsilon closure using flat arrays instead of FxHashMap.
    let epsilon_closure = |weight_by_state: &mut Vec<Option<Weight>>,
                           closure_queue: &mut VecDeque<u32>,
                           seed: &mut FxHashMap<u32, Weight>| {
        // Initialize flat array from seed.
        let mut seed_states: Vec<u32> = Vec::new();
        for (&state_id, weight) in seed.iter() {
            weight_by_state[state_id as usize] = Some(weight.clone());
            closure_queue.push_back(state_id);
            seed_states.push(state_id);
        }

        // Fast path: single seed with no epsilons.
        if seed.len() == 1 {
            let state_id = seed_states[0];
            if let Some(state) = nwa.states().get(state_id as usize) {
                if state.epsilons.is_empty() {
                    // Clean up and return early — seed is already populated.
                    closure_queue.clear();
                    for &s in &seed_states {
                        weight_by_state[s as usize] = None;
                    }
                    return;
                }
            }
        }

        while let Some(state_id) = closure_queue.pop_front() {
            let Some(current_weight) = weight_by_state[state_id as usize].clone() else {
                continue;
            };
            let Some(state) = nwa.states().get(state_id as usize) else {
                continue;
            };
            for (target, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let target_idx = *target as usize;
                if let Some(existing) = &weight_by_state[target_idx] {
                    if !contribution.is_subset(existing) {
                        weight_by_state[target_idx] = Some(existing.union(&contribution));
                        closure_queue.push_back(*target);
                    }
                } else {
                    weight_by_state[target_idx] = Some(contribution);
                    closure_queue.push_back(*target);
                    seed_states.push(*target);
                }
            }
        }

        // Write results back to seed map.
        seed.clear();
        for &s in &seed_states {
            if let Some(w) = weight_by_state[s as usize].take() {
                seed.insert(s, w);
            }
        }
    };

    // Canonicalize from FxHashMap into reusable buffer.
    let canonicalize_into =
        |map: &FxHashMap<u32, Weight>, buf: &mut Vec<(u32, Weight)>| {
            buf.clear();
            for (&state_id, weight) in map.iter() {
                if !weight.is_empty() {
                    buf.push((state_id, weight.clone()));
                }
            }
            buf.sort_unstable_by_key(|(state_id, _)| *state_id);
        };

    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![Vec::new()];

    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        let existing = start_subset.get(&state_id).cloned().unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    epsilon_closure(&mut weight_by_state, &mut closure_queue, &mut start_subset);
    if start_subset.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    canonicalize_into(&start_subset, &mut canon_buf);
    supports[0] = canon_buf.iter().map(|(state_id, _)| *state_id).collect();

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    let mut singleton_subsets: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    let start_key = subset_key(&canon_buf);
    subset_map.insert(start_key, dwa.start_state());
    if let [(state_id, weight)] = canon_buf.as_slice() {
        singleton_subsets.insert((*state_id, weight.ptr_key()), dwa.start_state());
    }
    let mut worklist: VecDeque<(u32, Vec<(u32, Weight)>)> = VecDeque::new();
    worklist.push_back((dwa.start_state(), canon_buf.clone()));

    let dense_label_limit = dense_positive_label_limit.map(|n| n as usize).unwrap_or(0);
    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    let mut intersection_cache = ScopedWeightOpCache::default();
    let use_epsilon_closure_weight_cache =
        std::env::var_os("GLRMASK_DISABLE_PARSER_EPSILON_CLOSURE_WEIGHT_CACHE").is_none();
    // Memoize local epsilon-closure outputs keyed by pre-closure weighted subsets.
    let mut closure_cache: FxHashMap<Vec<(u32, usize)>, CachedClosure> = FxHashMap::default();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut closure_touched_states: Vec<u32> = Vec::new();
    let mut closure_canon: Vec<(u32, Weight)> = Vec::new();
    let use_flat_canonical_closure = std::env::var("GLRMASK_DISABLE_FLAT_CANONICAL_EPSILON_CLOSURE")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || value == "0" || value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(true);
    let mut detail =
        ParserDwaDeterminizeDetail::enabled().then(ParserDwaDeterminizeDetail::default);
    let mut profile_transition_weight_edges = 0usize;
    let mut profile_transition_weight_runs = 0usize;
    let mut union_cache = UnionAllCache {
        ordered_keys: std::env::var_os("GLRMASK_DISABLE_ORDERED_UNION_CACHE_KEY").is_none(),
        profile_enabled: detail.is_some(),
        ..UnionAllCache::default()
    };
    let large_parser_nwa = num_nwa_states >= LARGE_PARSER_NWA_ALGEBRA_OPT_MIN_STATES;
    let use_local_transition_weight_cache = large_parser_nwa
        && std::env::var_os("GLRMASK_DISABLE_PARSER_LOCAL_TRANSITION_WEIGHT_CACHE").is_none();
    let defer_edge_weight_unions = large_parser_nwa
        && std::env::var_os("GLRMASK_DISABLE_PARSER_DEFER_EDGE_WEIGHT_UNIONS").is_none()
        && rayon::current_num_threads() > 1;
    let mut deferred_edge_union_ids = FxHashMap::<SmallVec<[usize; 16]>, usize>::default();
    let mut deferred_edge_union_jobs = Vec::<SmallVec<[Weight; 8]>>::new();
    let mut deferred_edge_weight_patches = Vec::<DeferredEdgeWeightPatch>::new();

    // Deferred final weight computation: store subset entries for each DWA state
    // and compute final weights in parallel after the main loop.
    let mut deferred_final_entries: Vec<(u32, DeferredFinalEntries)> = Vec::new();

    while let Some((from_state, subset_entries)) = worklist.pop_front() {
        if let Some(detail) = detail.as_mut() {
            detail.states_processed += 1;
        }

        // Save subset entries for deferred parallel final weight computation.
        // Only save entries whose NWA states have final weights.
        let has_finals: DeferredFinalEntries = subset_entries.iter()
            .filter(|(nwa_state_id, _)| nwa.states()[*nwa_state_id as usize].final_weight.is_some())
            .map(|(id, w)| (*id, w.clone()))
            .collect();
        if !has_finals.is_empty() {
            deferred_final_entries.push((from_state, has_finals));
        }

        let scan_started = detail.as_ref().map(|_| Instant::now());
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states()[*nwa_state_id as usize];
            let mut previous_transition_weight = None::<usize>;
            let mut previous_intersection = None::<Weight>;
            let mut local_transition_intersections = SmallVec::<[(usize, Weight); 8]>::new();
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
                    let transition_key = transition_weight.ptr_key();
                    if detail.is_some() {
                        profile_transition_weight_edges += 1;
                        if previous_transition_weight != Some(transition_key) {
                            profile_transition_weight_runs += 1;
                        }
                    }
                    if let Some(detail) = detail.as_mut() {
                        detail.outgoing_transitions_scanned += 1;
                    }
                    let next_weight = if previous_transition_weight == Some(transition_key) {
                        previous_intersection
                            .as_ref()
                            .expect("cached transition intersection must exist")
                            .clone()
                    } else if use_local_transition_weight_cache {
                        if let Some((_, cached)) = local_transition_intersections
                            .iter()
                            .find(|(key, _)| *key == transition_key)
                        {
                            cached.clone()
                        } else {
                            if let Some(detail) = detail.as_mut() {
                                detail.intersection_calls += 1;
                            }
                            let weight = intersection_cache.intersection(path_weight, transition_weight);
                            local_transition_intersections.push((transition_key, weight.clone()));
                            weight
                        }
                    } else {
                        if let Some(detail) = detail.as_mut() {
                            detail.intersection_calls += 1;
                        }
                        intersection_cache.intersection(path_weight, transition_weight)
                    };
                    previous_intersection = Some(next_weight.clone());
                    previous_transition_weight = Some(transition_key);
                    if next_weight.is_empty() {
                        continue;
                    }
                    if let Some(detail) = detail.as_mut() {
                        detail.nonempty_intersections += 1;
                    }

                    let target_weights = if label >= 0 && (label as usize) < dense_label_limit {
                        let label_idx = label as usize;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        &mut dense_raw_targets[label_idx]
                    } else if label == DEFAULT_LABEL {
                        default_touched = true;
                        &mut default_raw_targets
                    } else {
                        sparse_raw_targets.entry(label).or_default()
                    };
                    push_target_contribution_profiled(
                        target_weights,
                        *target,
                        next_weight,
                        detail.as_mut(),
                    );
                }
            }
        }
        if let (Some(detail), Some(started_at)) = (detail.as_mut(), scan_started) {
            detail.intersection_scan_ms += elapsed_ms(started_at);
        }

        let mut pre_closure_key: Vec<(u32, usize)> = Vec::new();
        let label_started = detail.as_ref().map(|_| Instant::now());

        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));

            if let Some(detail) = detail.as_mut() {
                detail.labels_processed += 1;
                detail.label_contribs_sum += contribs.len();
                detail.label_contribs_max = detail.label_contribs_max.max(contribs.len());
            }
            let sort_started = detail.as_ref().map(|_| Instant::now());
            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);
            merge_sorted_target_contributions(&mut contribs, detail.as_mut());
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), sort_started) {
                detail.contribution_sort_ms += elapsed_ms(started_at);
            }

            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                if nwa.states()[*only_state as usize].epsilons.is_empty() {
                    let singleton_key = (*only_state, only_weight.ptr_key());
                    let subset_lookup_started = detail.as_ref().map(|_| Instant::now());
                    let to_state = if let Some(existing) = singleton_subsets.get(&singleton_key).copied() {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_hits += 1;
                        }
                        existing
                    } else {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_misses += 1;
                        }
                        let new_state = dwa.add_state();
                        subset_map.insert(vec![singleton_key], new_state);
                        singleton_subsets.insert(singleton_key, new_state);
                        worklist.push_back((new_state, vec![(*only_state, only_weight.clone())]));
                        supports.push(vec![*only_state]);
                        new_state
                    };
                    if let (Some(detail), Some(started_at)) =
                        (detail.as_mut(), subset_lookup_started)
                    {
                        detail.subset_map_lookup_ms += elapsed_ms(started_at);
                    }
                    let add_transition_started = detail.as_ref().map(|_| Instant::now());
                    dwa.add_transition(from_state, label, to_state, only_weight.clone());
                    if let (Some(detail), Some(started_at)) =
                        (detail.as_mut(), add_transition_started)
                    {
                        detail.add_transition_ms += elapsed_ms(started_at);
                    }
                    return;
                }
            }

            let closure_key_started = detail.as_ref().map(|_| Instant::now());
            pre_closure_key.clear();
            pre_closure_key.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            if let Some(detail) = detail.as_mut() {
                detail.subset_key_constructions += 1;
            }
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_key_started) {
                detail.closure_key_ms += elapsed_ms(started_at);
            }

            let closure_lookup_started = detail.as_ref().map(|_| Instant::now());
            let cached = closure_cache.get(&pre_closure_key).cloned();
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_lookup_started) {
                detail.closure_lookup_ms += elapsed_ms(started_at);
            }
            if let Some(cached) = cached {
                if let Some(detail) = detail.as_mut() {
                    detail.closure_cache_hits += 1;
                }
                let add_transition_started = detail.as_ref().map(|_| Instant::now());
                match cached.edge_weight {
                    PendingEdgeWeight::Immediate(weight) => {
                        dwa.add_transition(from_state, label, cached.to_state, weight);
                    }
                    PendingEdgeWeight::Deferred(job) => {
                        dwa.add_transition(from_state, label, cached.to_state, Weight::empty());
                        deferred_edge_weight_patches.push(DeferredEdgeWeightPatch {
                            state: from_state,
                            label,
                            job,
                        });
                    }
                }
                if let (Some(detail), Some(started_at)) =
                    (detail.as_mut(), add_transition_started)
                {
                    detail.add_transition_ms += elapsed_ms(started_at);
                }
                return;
            }

            if let Some(detail) = detail.as_mut() {
                detail.closure_cache_misses += 1;
            }
            let edge_weight_started = detail.as_ref().map(|_| Instant::now());
            let edge_weight = if defer_edge_weight_unions && contribs.len() > 1 {
                if let Some((_, full_weight)) = contribs.iter().find(|(_, weight)| weight.is_full()) {
                    PendingEdgeWeight::Immediate(full_weight.clone())
                } else {
                    let mut key = contribs
                        .iter()
                        .map(|(_, weight)| weight.ptr_key())
                        .collect::<SmallVec<[usize; 16]>>();
                    if !union_cache.ordered_keys {
                        key.sort_unstable();
                        key.dedup();
                    }
                    let job = if let Some(&job) = deferred_edge_union_ids.get(&key) {
                        job
                    } else {
                        let job = deferred_edge_union_jobs.len();
                        deferred_edge_union_ids.insert(key, job);
                        deferred_edge_union_jobs.push(
                            contribs
                                .iter()
                                .map(|(_, weight)| weight.clone())
                                .collect::<SmallVec<[Weight; 8]>>(),
                        );
                        job
                    };
                    PendingEdgeWeight::Deferred(job)
                }
            } else {
                let weight = union_cache.union_all(contribs.iter().map(|(_, weight)| weight));
                if weight.is_empty() {
                    return;
                }
                PendingEdgeWeight::Immediate(weight)
            };
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), edge_weight_started) {
                detail.edge_weight_union_ms += elapsed_ms(started_at);
            }
            let closure_started = detail.as_ref().map(|_| Instant::now());
            let mut owned_canon = Vec::new();
            if use_flat_canonical_closure {
                local_epsilon_closure_canonical(
                    nwa,
                    &mut weight_by_state,
                    &mut closure_queue,
                    &contribs,
                    &mut closure_touched_states,
                    &mut closure_canon,
                    &mut intersection_cache,
                    use_epsilon_closure_weight_cache,
                );
            } else {
                let mut target_subset: FxHashMap<u32, Weight> = contribs
                    .iter()
                    .map(|(state_id, weight)| (*state_id, weight.clone()))
                    .collect();
                local_epsilon_closure(
                    nwa,
                    &mut weight_by_state,
                    &mut closure_queue,
                    &mut target_subset,
                );
                owned_canon = target_subset
                    .iter()
                    .filter(|(_, weight)| !weight.is_empty())
                    .map(|(state_id, weight)| (*state_id, weight.clone()))
                    .collect();
                owned_canon.sort_unstable_by_key(|(state_id, _)| *state_id);
            }
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_started) {
                detail.local_epsilon_closure_miss_ms += elapsed_ms(started_at);
            }
            let canon = if use_flat_canonical_closure {
                closure_canon.as_slice()
            } else {
                owned_canon.as_slice()
            };
            if canon.is_empty() {
                return;
            }

            let subset_lookup_started = detail.as_ref().map(|_| Instant::now());
            let to_state = if let [(only_state, only_weight)] = canon {
                let singleton_key = (*only_state, only_weight.ptr_key());
                if let Some(existing) = singleton_subsets.get(&singleton_key).copied() {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    let new_state = dwa.add_state();
                    subset_map.insert(vec![singleton_key], new_state);
                    singleton_subsets.insert(singleton_key, new_state);
                    worklist.push_back((new_state, canon.to_vec()));
                    supports.push(vec![*only_state]);
                    new_state
                }
            } else {
                let subset_key_started = detail.as_ref().map(|_| Instant::now());
                key_buf.clear();
                key_buf.extend(canon.iter().map(|(sid, w)| (*sid, w.ptr_key())));
                if let Some(detail) = detail.as_mut() {
                    detail.subset_key_constructions += 1;
                }
                if let (Some(detail), Some(started_at)) = (detail.as_mut(), subset_key_started) {
                    detail.post_closure_subset_key_ms += elapsed_ms(started_at);
                }
                if let Some(existing) = subset_map.get(&key_buf).copied() {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    let new_state = dwa.add_state();
                    subset_map.insert(key_buf.clone(), new_state);
                    worklist.push_back((new_state, canon.to_vec()));
                    supports.push(canon.iter().map(|(sid, _)| *sid).collect());
                    new_state
                }
            };
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), subset_lookup_started) {
                detail.subset_map_lookup_ms += elapsed_ms(started_at);
            }
            closure_cache.insert(
                pre_closure_key.clone(),
                CachedClosure {
                    to_state,
                    edge_weight: edge_weight.clone(),
                },
            );
            let add_transition_started = detail.as_ref().map(|_| Instant::now());
            match edge_weight {
                PendingEdgeWeight::Immediate(weight) => {
                    dwa.add_transition(from_state, label, to_state, weight);
                }
                PendingEdgeWeight::Deferred(job) => {
                    dwa.add_transition(from_state, label, to_state, Weight::empty());
                    deferred_edge_weight_patches.push(DeferredEdgeWeightPatch {
                        state: from_state,
                        label,
                        job,
                    });
                }
            }
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), add_transition_started) {
                detail.add_transition_ms += elapsed_ms(started_at);
            }
        };

        for label_idx in touched_dense_labels.drain(..) {
            dense_label_touched[label_idx] = false;
            process_label(label_idx as i32, std::mem::take(&mut dense_raw_targets[label_idx]));
        }

        if default_touched {
            default_touched = false;
            process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
        }

        for (label, contribs) in sparse_raw_targets.drain() {
            process_label(label, contribs);
        }
        if let (Some(detail), Some(started_at)) = (detail.as_mut(), label_started) {
            detail.label_processing_ms += elapsed_ms(started_at);
        }
    }

    if !deferred_edge_union_jobs.is_empty() {
        use rayon::prelude::*;
        let deferred_started_at = Instant::now();
        let patch_count = deferred_edge_weight_patches.len();
        let results = deferred_edge_union_jobs
            .par_iter()
            .map_init(ScopedWeightOpCache::default, |weight_ops, weights| {
                weight_ops.union_all(weights.iter())
            })
            .collect::<Vec<_>>();
        let union_ms = elapsed_ms(deferred_started_at);
        let patch_started_at = Instant::now();
        for patch in deferred_edge_weight_patches {
            let weight = results[patch.job].clone();
            debug_assert!(!weight.is_empty());
            let row = &mut dwa.states_mut()[patch.state as usize].transitions;
            let (_, edge_weight) = row
                .get_mut(&patch.label)
                .expect("deferred parser edge-weight patch must find its transition");
            *edge_weight = weight;
        }
        if compile_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_support_deferred_edge_unions] jobs={} patches={} union_ms={union_ms:.3} patch_ms={:.3}",
                results.len(),
                patch_count,
                elapsed_ms(patch_started_at),
            );
        }
    }

    let mut final_signature_ids: FxHashMap<Vec<(usize, Vec<usize>)>, usize> = FxHashMap::default();
    let mut final_signature_groups: Vec<FinalGroups> = Vec::new();
    let mut final_jobs: Vec<(u32, usize)> = Vec::with_capacity(deferred_final_entries.len());
    let final_grouping_started = detail.as_ref().map(|_| Instant::now());
    for (state_id, entries) in &deferred_final_entries {
        if let Some(detail) = detail.as_mut() {
            detail.final_weight_entries += entries.len();
            detail.final_weight_entries_max = detail.final_weight_entries_max.max(entries.len());
        }

        let mut groups: SmallVec<[(usize, Weight, FinalPathWeights); 4]> = SmallVec::new();
        for (nwa_state_id, path_weight) in entries {
            if let Some(state_final) = nwa.states()[*nwa_state_id as usize].final_weight.as_ref() {
                let final_key = state_final.ptr_key();
                if let Some((_, _, path_weights)) = groups
                    .iter_mut()
                    .find(|(existing_final_key, _, _)| *existing_final_key == final_key)
                {
                    path_weights.push(path_weight.clone());
                } else {
                    let mut path_weights = SmallVec::new();
                    path_weights.push(path_weight.clone());
                    groups.push((final_key, state_final.clone(), path_weights));
                }
            }
        }
        groups.sort_unstable_by_key(|(final_key, _, _)| *final_key);
        let mut signature: Vec<(usize, Vec<usize>)> = Vec::with_capacity(groups.len());
        for (final_key, _, path_weights) in &mut groups {
            path_weights.sort_unstable_by_key(|weight| weight.ptr_key());
            path_weights.dedup_by_key(|weight| weight.ptr_key());
            signature.push((
                *final_key,
                path_weights.iter().map(|weight| weight.ptr_key()).collect(),
            ));
        }
        let signature_id = match final_signature_ids.entry(signature) {
            Entry::Occupied(entry) => *entry.get(),
            Entry::Vacant(entry) => {
                let signature_id = final_signature_groups.len();
                let owned_groups: FinalGroups = groups
                    .into_iter()
                    .map(|(_, state_final, path_weights)| (state_final, path_weights))
                    .collect();
                final_signature_groups.push(owned_groups);
                entry.insert(signature_id);
                signature_id
            }
        };
        final_jobs.push((*state_id, signature_id));
    }
    if let (Some(detail), Some(started_at)) = (detail.as_mut(), final_grouping_started) {
        detail.final_grouping_ms += elapsed_ms(started_at);
    }
    if let Some(detail) = detail.as_mut() {
        detail.final_weight_states = final_jobs.len();
        detail.final_weight_signature_distinct = final_signature_groups.len();
        detail.final_weight_signature_hit_potential =
            final_jobs.len().saturating_sub(final_signature_groups.len());
    }

    // Compute final weights in parallel once per distinct final-weight signature.
    {
        use rayon::prelude::*;
        let detail_enabled = detail.is_some();
        let final_weights_by_signature: Vec<Option<Weight>> = {
            let intern_started_at = Instant::now();
            let mut component_ids = FxHashMap::<(usize, Vec<usize>), usize>::default();
            let mut components = Vec::<(Weight, SmallVec<[Weight; 4]>)>::new();
            let signature_components: Vec<SmallVec<[usize; 8]>> = final_signature_groups
                .iter()
                .map(|groups| {
                    groups
                        .iter()
                        .map(|(final_w, path_weights)| {
                            let key = (
                                final_w.ptr_key(),
                                path_weights.iter().map(Weight::ptr_key).collect::<Vec<_>>(),
                            );
                            if let Some(&component_id) = component_ids.get(&key) {
                                component_id
                            } else {
                                let component_id = components.len();
                                component_ids.insert(key, component_id);
                                components.push((final_w.clone(), path_weights.clone()));
                                component_id
                            }
                        })
                        .collect::<SmallVec<[usize; 8]>>()
                })
                .collect::<Vec<_>>();
            let intern_ms = elapsed_ms(intern_started_at);
            let parallel_min_components = std::env::var("GLRMASK_PARSER_FINAL_PARALLEL_MIN_COMPONENTS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .unwrap_or(512);
            let rayon_workers = rayon::current_num_threads();
            let parallel_min_signatures = std::env::var("GLRMASK_PARSER_FINAL_PARALLEL_MIN_SIGNATURES")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                // Above 48 workers, continuing to scale the threshold makes
                // larger machines *less* likely to parallelize the same exact
                // signature workload. Cap at the 48-worker crossover instead.
                .unwrap_or_else(|| 128usize.max(rayon_workers.saturating_mul(4).min(192)));
            let has_parallel_workers = rayon_workers > 1;
            let use_parallel_final_components = has_parallel_workers
                && components.len() >= parallel_min_components;
            let use_parallel_final_signatures = has_parallel_workers
                && signature_components.len() >= parallel_min_signatures;
            let compute_component = |weight_ops: &mut ScopedWeightOpCache,
                                     (final_w, path_weights): &(Weight, SmallVec<[Weight; 4]>)| {
                let path_started = detail_enabled.then(Instant::now);
                let path_union = weight_ops.union_all(path_weights.iter());
                let path_ms = path_started.map(elapsed_ms).unwrap_or(0.0);
                let intersection_started = detail_enabled.then(Instant::now);
                let contribution = weight_ops.intersection(&path_union, final_w);
                let intersection_ms = intersection_started.map(elapsed_ms).unwrap_or(0.0);
                (
                    (!contribution.is_empty()).then_some(contribution),
                    path_ms,
                    intersection_ms,
                )
            };
            let component_results: Vec<(Option<Weight>, f64, f64)> =
                if use_parallel_final_components {
                    components
                        .par_iter()
                        .map_init(ScopedWeightOpCache::default, compute_component)
                        .collect()
                } else {
                    let mut weight_ops = ScopedWeightOpCache::default();
                    components
                        .iter()
                        .map(|component| compute_component(&mut weight_ops, component))
                        .collect()
                };
            if let Some(detail) = detail.as_mut() {
                detail.final_path_union_ms +=
                    component_results.iter().map(|(_, ms, _)| *ms).sum::<f64>();
                detail.final_intersection_ms +=
                    component_results.iter().map(|(_, _, ms)| *ms).sum::<f64>();
            }
            let output_started_at = Instant::now();
            let mut output_signature_ids = FxHashMap::<SmallVec<[usize; 8]>, usize>::default();
            let mut unique_component_ids = Vec::<SmallVec<[usize; 8]>>::new();
            let mut jobs = Vec::<usize>::with_capacity(signature_components.len());
            for component_ids in &signature_components {
                let mut keyed = component_ids
                    .iter()
                    .filter_map(|&component_id| {
                        component_results[component_id]
                            .0
                            .as_ref()
                            .map(|weight| (weight.ptr_key(), component_id))
                    })
                    .collect::<SmallVec<[(usize, usize); 8]>>();
                keyed.sort_unstable_by_key(|(key, _)| *key);
                keyed.dedup_by_key(|(key, _)| *key);
                let key = keyed
                    .iter()
                    .map(|(key, _)| *key)
                    .collect::<SmallVec<[usize; 8]>>();
                let output_id = if let Some(&existing) = output_signature_ids.get(&key) {
                    existing
                } else {
                    let output_id = unique_component_ids.len();
                    unique_component_ids.push(
                        keyed.iter().map(|(_, component_id)| *component_id).collect(),
                    );
                    output_signature_ids.insert(key, output_id);
                    output_id
                };
                jobs.push(output_id);
            }
            if compile_profile_enabled() {
                eprintln!(
                    "[glrmask/profile][parser_final_output_intern] input_signatures={} output_signatures={} reused={}",
                    signature_components.len(),
                    unique_component_ids.len(),
                    signature_components.len().saturating_sub(unique_component_ids.len()),
                );
            }
            let compute_unique = |weight_ops: &mut ScopedWeightOpCache,
                                  component_ids: &SmallVec<[usize; 8]>| {
                let weights = component_ids
                    .iter()
                    .filter_map(|&component_id| component_results[component_id].0.as_ref());
                let weight = weight_ops.union_all(weights);
                (!weight.is_empty()).then_some(weight)
            };
            let unique_results = if use_parallel_final_signatures {
                unique_component_ids
                    .par_iter()
                    .map_init(ScopedWeightOpCache::default, compute_unique)
                    .collect::<Vec<_>>()
            } else {
                let mut weight_ops = ScopedWeightOpCache::default();
                unique_component_ids
                    .iter()
                    .map(|component_ids| compute_unique(&mut weight_ops, component_ids))
                    .collect::<Vec<_>>()
            };
            let results = jobs
                .into_iter()
                .map(|output_id| unique_results[output_id].clone())
                .collect::<Vec<_>>();
            if std::env::var_os("GLRMASK_VALIDATE_INTERNED_FINAL_GROUPS").is_some() {
                let reference = final_signature_groups
                    .iter()
                    .map(|final_groups| {
                        let contributions = final_groups
                            .iter()
                            .filter_map(|(final_w, path_weights)| {
                                let path_union = Weight::union_all(path_weights.iter());
                                let contribution = path_union.intersection(final_w);
                                (!contribution.is_empty()).then_some(contribution)
                            })
                            .collect::<SmallVec<[Weight; 4]>>();
                        let result = Weight::union_all(contributions.iter());
                        (!result.is_empty()).then_some(result)
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    results, reference,
                    "interned parser final-weight components changed the weighted language",
                );
            }
            if let Some(detail) = detail.as_mut() {
                detail.final_output_union_ms += elapsed_ms(output_started_at);
            }
            if compile_profile_enabled() {
                let total_components = signature_components.iter().map(|ids| ids.len()).sum::<usize>();
                eprintln!(
                    "[glrmask/profile][parser_final_group_intern] signatures={} total_components={} unique_components={} intern_ms={:.3}",
                    signature_components.len(),
                    total_components,
                    components.len(),
                    intern_ms,
                );
            }
            results
        };
        for (state_id, signature_id) in final_jobs {
            if let Some(weight) = &final_weights_by_signature[signature_id] {
                dwa.set_final_weight(state_id, weight.clone());
            }
        }
    }

    if let Some(detail) = detail.as_mut() {
        detail.union_cache_hits = union_cache.hits;
        detail.union_cache_misses = union_cache.misses;
        detail.union_cache_key_len_sum = union_cache.key_len_sum;
        detail.union_cache_key_len_max = union_cache.key_len_max;
        detail.union_cache_ms = union_cache.total_ms;
    }

    if detail.is_some() {
        eprintln!(
            "[glrmask/profile][parser_support_transition_weight_runs] edges={} runs={} adjacent_reuse={} reuse_factor={:.3} intersection_cache_entries={}",
            profile_transition_weight_edges,
            profile_transition_weight_runs,
            profile_transition_weight_edges.saturating_sub(profile_transition_weight_runs),
            if profile_transition_weight_runs == 0 { 0.0 } else { profile_transition_weight_edges as f64 / profile_transition_weight_runs as f64 },
            intersection_cache.intersection_entry_count(),
        );
    }
    if detail.is_some() {
        eprintln!(
            "[glrmask/profile][parser_support_bulk_token_union_cache] entries={}",
            union_cache.bulk_ops.bulk_token_union_entry_count(),
        );
    }
    if let Some(detail) = detail {
        detail.emit("support");
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

fn determinize_parser_dwa_with_fallbacks_impl(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
    normalize_singletons: bool,
) -> DWA {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    let dense_label_limit = num_parser_states as usize;
    let mut result = DWA::new(0, 0);

    let mut start_subset = FxHashMap::default();
    start_subset.insert(dwa.start_state(), Weight::all());

    let mut canon_buf: Vec<(u32, Weight)> = start_subset
        .iter()
        .map(|(state_id, weight)| (*state_id, weight.clone()))
        .collect();
    canon_buf.sort_unstable_by_key(|(state_id, _)| *state_id);

    let mut subset_map: FxHashMap<Vec<(u32, usize)>, u32> = FxHashMap::default();
    // A singleton weighted subset `(q, w)` denotes `w ∩ L(q)`. The incoming
    // transition already carries `w`, so the destination can always be the
    // same canonical state for `L(q)` with residual `all`. Keeping `w` in the
    // state key duplicates rows for different accumulated token sets and was
    // the dominant cost of fallback determinization on large grammars.
    let mut normalized_singleton_subsets: FxHashMap<u32, u32> = FxHashMap::default();
    let mut weighted_singleton_subsets: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    let normalized_singleton_weight = Weight::all();
    let normalized_singleton_key = normalized_singleton_weight.ptr_key();
    let start_key = subset_key(&canon_buf);
    subset_map.insert(start_key, result.start_state());
    if let [(state_id, weight)] = canon_buf.as_slice() {
        if normalize_singletons && weight.is_full() {
            normalized_singleton_subsets.insert(*state_id, result.start_state());
        } else {
            weighted_singleton_subsets
                .insert((*state_id, weight.ptr_key()), result.start_state());
        }
    }
    let mut worklist: VecDeque<(u32, Vec<(u32, Weight)>)> = VecDeque::new();
    worklist.push_back((result.start_state(), canon_buf.clone()));

    let mut dense_raw_targets: Vec<TargetContribs> =
        (0..dense_label_limit).map(|_| TargetContribs::new()).collect();
    let mut default_raw_targets: TargetContribs = TargetContribs::new();
    let mut sparse_raw_targets: FxHashMap<i32, TargetContribs> = FxHashMap::default();
    let mut touched_dense_labels: Vec<usize> = Vec::new();
    let mut dense_label_touched: Vec<bool> = vec![false; dense_label_limit];
    let mut default_touched = false;
    let mut dense_default_all_raw_targets: TargetContribs = TargetContribs::new();
    let mut intersection_cache = ScopedWeightOpCache::default();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut final_contributions: Vec<Weight> = Vec::new();
    let mut detail =
        ParserDwaDeterminizeDetail::enabled().then(ParserDwaDeterminizeDetail::default);

    while let Some((from_state, subset_entries)) = worklist.pop_front() {
        dense_default_all_raw_targets.clear();
        if let Some(detail) = detail.as_mut() {
            detail.states_processed += 1;
        }

        final_contributions.clear();
        let scan_started = detail.as_ref().map(|_| Instant::now());
        for (state_id, path_weight) in &subset_entries {
            let Some(state_final) = dwa.states()[*state_id as usize].final_weight.as_ref() else {
                continue;
            };
            if let Some(detail) = detail.as_mut() {
                detail.intersection_calls += 1;
            }
            let contribution = intersection_cache.intersection(path_weight, state_final);
            if !contribution.is_empty() {
                if let Some(detail) = detail.as_mut() {
                    detail.nonempty_intersections += 1;
                }
                final_contributions.push(contribution);
            }
        }
        let final_weight = Weight::union_all(final_contributions.iter());
        if !final_weight.is_empty() {
            result.set_final_weight(from_state, final_weight);
        }

        // The fallback pass overwhelmingly visits singleton subsets whose input
        // state has no DEFAULT edge. In that case every output transition stays
        // singleton, so staging contributions by label and running the generic
        // subset path is pure overhead.
        if normalize_singletons
            && let [(dwa_state_id, path_weight)] = subset_entries.as_slice()
            && path_weight.is_full()
            && let Some(state) = dwa.states().get(*dwa_state_id as usize)
            && !state.transitions.contains_key(&DEFAULT_LABEL)
        {
            // Preserve the already-sorted row allocation and rewrite only its
            // targets/weights. This avoids millions of individual BTreeMap
            // insertions in the dominant singleton fallback path.
            let mut rewritten = state.transitions.clone();
            let mut remove = Vec::new();
            for (&label, (target, transition_weight)) in &mut rewritten {
                if let Some(detail) = detail.as_mut() {
                    detail.outgoing_transitions_scanned += 1;
                }
                let input_target = *target;
                if transition_weight.is_empty() {
                    remove.push(label);
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.nonempty_intersections += 1;
                }
                let to_state = if let Some(existing) =
                    normalized_singleton_subsets.get(&input_target).copied()
                {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    let new_state = result.add_state();
                    subset_map.insert(vec![(input_target, normalized_singleton_key)], new_state);
                    normalized_singleton_subsets.insert(input_target, new_state);
                    worklist.push_back((
                        new_state,
                        vec![(input_target, normalized_singleton_weight.clone())],
                    ));
                    new_state
                };
                *target = to_state;
            }
            for label in remove {
                rewritten.remove(&label);
            }
            result.states_mut()[from_state as usize].transitions = rewritten;
            continue;
        }

        for (dwa_state_id, path_weight) in &subset_entries {
            let state = &dwa.states()[*dwa_state_id as usize];

            for (&label, (target, transition_weight)) in &state.transitions {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.outgoing_transitions_scanned += 1;
                    detail.intersection_calls += 1;
                }
                let next_weight =
                    intersection_cache.intersection(path_weight, transition_weight);
                if next_weight.is_empty() {
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.nonempty_intersections += 1;
                }

                let target_weights = if label >= 0 && (label as usize) < dense_label_limit {
                    let label_idx = label as usize;
                    if !dense_label_touched[label_idx] {
                        dense_label_touched[label_idx] = true;
                        touched_dense_labels.push(label_idx);
                    }
                    &mut dense_raw_targets[label_idx]
                } else {
                    sparse_raw_targets.entry(label).or_default()
                };
                add_target_contribution_profiled(target_weights, *target, next_weight, detail.as_mut());
            }

            let Some((default_target, default_weight)) = state.transitions.get(&DEFAULT_LABEL) else {
                continue;
            };

            if let Some(detail) = detail.as_mut() {
                detail.outgoing_transitions_scanned += 1;
                detail.intersection_calls += 1;
            }
            let fallback_weight = intersection_cache.intersection(path_weight, default_weight);
            if fallback_weight.is_empty() {
                continue;
            }
            if let Some(detail) = detail.as_mut() {
                detail.nonempty_intersections += 1;
            }

            default_touched = true;
            add_target_contribution_profiled(
                &mut default_raw_targets,
                *default_target,
                fallback_weight.clone(),
                detail.as_mut(),
            );

            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.fallback_labels_expanded += 1;
                    detail.fallback_contrib_entries_duplicated += 1;
                }
                if label >= 0 && (label as usize) < dense_label_limit {
                    let label_idx = label as usize;
                    if !dense_label_touched[label_idx] {
                        dense_label_touched[label_idx] = true;
                        touched_dense_labels.push(label_idx);
                    }
                    let target_weights = &mut dense_raw_targets[label_idx];
                    add_target_contribution_profiled(
                        target_weights,
                        *default_target,
                        fallback_weight.clone(),
                        detail.as_mut(),
                    );
                } else {
                    let target_weights = sparse_raw_targets.entry(label).or_default();
                    add_target_contribution_profiled(
                        target_weights,
                        *default_target,
                        fallback_weight.clone(),
                        detail.as_mut(),
                    );
                }
            }

            match possible_by_state.get(*dwa_state_id as usize) {
                Some(PossibleOutgoingIds::All) => {
                    if let Some(detail) = detail.as_mut() {
                        detail.fallback_labels_expanded += dense_label_limit;
                        detail.fallback_contrib_entries_duplicated += 1;
                    }
                    add_target_contribution_profiled(
                        &mut dense_default_all_raw_targets,
                        *default_target,
                        fallback_weight.clone(),
                        detail.as_mut(),
                    );
                }
                Some(PossibleOutgoingIds::Some(ids)) => {
                    for parser_state_id in ids.iter_ones() {
                        if let Some(detail) = detail.as_mut() {
                            detail.fallback_labels_expanded += 1;
                            detail.fallback_contrib_entries_duplicated += 1;
                        }
                        let label_idx = parser_state_id;
                        if !dense_label_touched[label_idx] {
                            dense_label_touched[label_idx] = true;
                            touched_dense_labels.push(label_idx);
                        }
                        let target_weights = &mut dense_raw_targets[label_idx];
                        add_target_contribution_profiled(
                            target_weights,
                            *default_target,
                            fallback_weight.clone(),
                            detail.as_mut(),
                        );
                    }
                }
                Some(PossibleOutgoingIds::Empty) | None => {}
            }
        }
        if let (Some(detail), Some(started_at)) = (detail.as_mut(), scan_started) {
            detail.intersection_scan_ms += elapsed_ms(started_at);
        }

        let label_started = detail.as_ref().map(|_| Instant::now());
        let mut process_label = |label: i32, mut contribs: TargetContribs| {
            if contribs.is_empty() {
                return;
            }

            debug_assert!(contribs.iter().all(|(_, weight)| !weight.is_empty()));
            contribs.sort_unstable_by_key(|(state_id, _)| *state_id);

            let edge_weight = Weight::union_all(contribs.iter().map(|(_, weight)| weight));
            if edge_weight.is_empty() {
                return;
            }

            let to_state = if let [(only_state, only_weight)] = contribs.as_slice() {
                if normalize_singletons {
                    if let Some(existing) = normalized_singleton_subsets.get(only_state).copied() {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_hits += 1;
                        }
                        existing
                    } else {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_misses += 1;
                        }
                        let new_state = result.add_state();
                        subset_map.insert(vec![(*only_state, normalized_singleton_key)], new_state);
                        normalized_singleton_subsets.insert(*only_state, new_state);
                        worklist.push_back((
                            new_state,
                            vec![(*only_state, normalized_singleton_weight.clone())],
                        ));
                        new_state
                    }
                } else {
                    let singleton_key = (*only_state, only_weight.ptr_key());
                    if let Some(existing) = weighted_singleton_subsets.get(&singleton_key).copied() {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_hits += 1;
                        }
                        existing
                    } else {
                        if let Some(detail) = detail.as_mut() {
                            detail.subset_intern_misses += 1;
                        }
                        let new_state = result.add_state();
                        subset_map.insert(vec![singleton_key], new_state);
                        weighted_singleton_subsets.insert(singleton_key, new_state);
                        worklist.push_back((new_state, contribs.into_iter().collect()));
                        new_state
                    }
                }
            } else {
                key_buf.clear();
                key_buf.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
                if let Some(detail) = detail.as_mut() {
                    detail.subset_key_constructions += 1;
                }
                if let Some(existing) = subset_map.get(&key_buf).copied() {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_hits += 1;
                    }
                    existing
                } else {
                    if let Some(detail) = detail.as_mut() {
                        detail.subset_intern_misses += 1;
                    }
                    let new_state = result.add_state();
                    subset_map.insert(key_buf.clone(), new_state);
                    let next_entries: Vec<(u32, Weight)> = contribs.into_iter().collect();
                    worklist.push_back((new_state, next_entries));
                    new_state
                }
            };

            result.add_transition(from_state, label, to_state, edge_weight);
        };

        for label_idx in touched_dense_labels.drain(..) {
            dense_label_touched[label_idx] = false;
            if !dense_default_all_raw_targets.is_empty() {
                extend_target_contribs(
                    &mut dense_raw_targets[label_idx],
                    &dense_default_all_raw_targets,
                );
            }
            process_label(
                label_idx as i32,
                std::mem::take(&mut dense_raw_targets[label_idx]),
            );
        }
        if default_touched {
            default_touched = false;
            process_label(DEFAULT_LABEL, std::mem::take(&mut default_raw_targets));
        }
        for (label, contribs) in sparse_raw_targets.drain() {
            process_label(label, contribs);
        }
        if let (Some(detail), Some(started_at)) = (detail.as_mut(), label_started) {
            detail.label_processing_ms += elapsed_ms(started_at);
        }
    }

    if let Some(detail) = detail {
        detail.emit("fallback");
    }

    result
}

fn determinize_parser_dwa_with_fallbacks(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) -> DWA {
    determinize_parser_dwa_with_fallbacks_impl(
        dwa,
        possible_by_state,
        num_parser_states,
        true,
    )
}

fn optimize_parser_dwa_defaults(
    dwa: &mut DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) {
    loop {
        let mut changed = false;

        for (state_id, possible_ids) in possible_by_state.iter().enumerate() {
            let possible_count = match possible_ids {
                PossibleOutgoingIds::Empty => 0,
                PossibleOutgoingIds::All => num_parser_states as usize,
                PossibleOutgoingIds::Some(ids) => ids.count_ones(),
            };
            if possible_count < 2 {
                continue;
            }

            let state = &dwa.states()[state_id];

            let mut actual_positive = BitSet::new(num_parser_states as usize);
            for &label in state.transitions.keys() {
                if let Some(ps) = parser_state_label(label, num_parser_states) {
                    actual_positive.set(ps as usize);
                }
            }
            match possible_ids {
                PossibleOutgoingIds::Empty => continue,
                PossibleOutgoingIds::All => {
                    if actual_positive.count_ones() != num_parser_states as usize {
                        continue;
                    }
                }
                PossibleOutgoingIds::Some(ids) => {
                    if actual_positive != *ids {
                        continue;
                    }
                }
            }

            let mut shared_target: Option<u32> = None;
            let mut default_weight: Option<Weight> = None;
            let mut valid = true;

            match possible_ids {
                PossibleOutgoingIds::Empty => continue,
                PossibleOutgoingIds::All => {
                    for ps in 0..num_parser_states {
                        let label = ps as i32;
                        let Some((target, weight)) = state.transitions.get(&label) else {
                            valid = false;
                            break;
                        };
                        match shared_target {
                            Some(existing) if existing != *target => {
                                valid = false;
                                break;
                            }
                            None => shared_target = Some(*target),
                            _ => {}
                        }
                        default_weight = Some(match default_weight {
                            Some(existing) => existing.intersection(weight),
                            None => weight.clone(),
                        });
                    }
                }
                PossibleOutgoingIds::Some(ids) => {
                    for ps in ids.iter_ones() {
                        let label = ps as i32;
                        let Some((target, weight)) = state.transitions.get(&label) else {
                            valid = false;
                            break;
                        };
                        match shared_target {
                            Some(existing) if existing != *target => {
                                valid = false;
                                break;
                            }
                            None => shared_target = Some(*target),
                            _ => {}
                        }
                        default_weight = Some(match default_weight {
                            Some(existing) => existing.intersection(weight),
                            None => weight.clone(),
                        });
                    }
                }
            }

            let Some(target) = shared_target else { continue };
            let Some(default_weight) = default_weight else { continue };
            if !valid || default_weight.is_empty() {
                continue;
            }

            let state = &mut dwa.states_mut()[state_id];
            let entry = state.transitions.entry(DEFAULT_LABEL);
            match entry {
                std::collections::btree_map::Entry::Occupied(mut occ) => {
                    let (existing_target, existing_weight) = occ.get_mut();
                    if *existing_target == target {
                        let updated = existing_weight.union(&default_weight);
                        if updated != *existing_weight {
                            *existing_weight = updated;
                            changed = true;
                        }
                    }
                }
                std::collections::btree_map::Entry::Vacant(vac) => {
                    vac.insert((target, default_weight));
                    changed = true;
                }
            }
        }

        for state_id in 0..dwa.states().len() {
            let Some((default_target, default_weight)) =
                dwa.states()[state_id].transitions.get(&DEFAULT_LABEL).cloned()
            else {
                continue;
            };

            let target_final = dwa.states()[default_target as usize].final_weight.clone();
            let Some(target_final) = target_final else { continue };
            let lifted = default_weight.intersection(&target_final);
            if lifted.is_empty() {
                continue;
            }

            if union_final_weight(&mut dwa.states_mut()[state_id].final_weight, lifted.clone()) {
                changed = true;
            }

            let state = &mut dwa.states_mut()[state_id];
            let mut to_remove = Vec::new();
            for (&label, (_, weight)) in state.transitions.iter_mut() {
                let new_weight = weight.difference(&lifted);
                if new_weight != *weight {
                    *weight = new_weight;
                    changed = true;
                }
                if weight.is_empty() {
                    to_remove.push(label);
                }
            }
            for label in to_remove {
                state.transitions.remove(&label);
            }
        }

        for state_id in 0..dwa.states().len() {
            let Some(&(default_target, ref default_weight)) =
                dwa.states()[state_id].transitions.get(&DEFAULT_LABEL)
            else {
                continue;
            };
            let default_target = default_target;
            let default_weight = default_weight.clone();

            let state = &mut dwa.states_mut()[state_id];
            let mut to_remove = Vec::new();
            for (&label, (target, weight)) in state.transitions.iter_mut() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                if *target != default_target {
                    continue;
                }
                let new_weight = weight.difference(&default_weight);
                if new_weight != *weight {
                    *weight = new_weight;
                    changed = true;
                }
                if weight.is_empty() {
                    to_remove.push(label);
                }
            }
            for label in to_remove {
                state.transitions.remove(&label);
            }
        }

        if !changed {
            break;
        }
    }
}

fn subtract_final_weights_from_outgoing_dwa_impl(dwa: &mut DWA, parallel: bool) {
    if parallel {
        use rayon::prelude::*;

        dwa.states_mut().par_iter_mut().for_each_init(
            ScopedWeightOpCache::default,
            |weight_ops, state| {
                let Some(final_weight) = state.final_weight.clone() else {
                    return;
                };
                if final_weight.is_empty() {
                    return;
                }
                state.transitions.retain(|_, (_, weight)| {
                    let new_weight = weight_ops.difference(weight, &final_weight);
                    if new_weight != *weight {
                        *weight = new_weight;
                    }
                    !weight.is_empty()
                });
            },
        );
        return;
    }

    let mut weight_ops = ScopedWeightOpCache::default();
    for state_id in 0..dwa.states().len() {
        let Some(final_weight) = dwa.states()[state_id].final_weight.clone() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        let state = &mut dwa.states_mut()[state_id];
        let mut to_remove = Vec::new();
        for (&label, (_, weight)) in state.transitions.iter_mut() {
            let new_weight = weight_ops.difference(weight, &final_weight);
            if new_weight != *weight {
                *weight = new_weight;
            }
            if weight.is_empty() {
                to_remove.push(label);
            }
        }
        for label in to_remove {
            state.transitions.remove(&label);
        }
    }
}


fn subtract_final_weights_from_outgoing_dwa_global_pairs(dwa: &mut DWA) {
    use rayon::prelude::*;

    let collect_started = Instant::now();
    let mut operands = FxHashMap::<(usize, usize), (Weight, Weight)>::default();
    for state in dwa.states() {
        let Some(final_weight) = state.final_weight.as_ref() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        let final_key = final_weight.ptr_key();
        for (_, edge_weight) in state.transitions.values() {
            let key = (edge_weight.ptr_key(), final_key);
            operands
                .entry(key)
                .or_insert_with(|| (edge_weight.clone(), final_weight.clone()));
        }
    }
    let collect_ms = elapsed_ms(collect_started);

    let compute_started = Instant::now();
    let results = operands
        .into_par_iter()
        .map(|(key, (edge_weight, final_weight))| {
            (key, edge_weight.difference(&final_weight))
        })
        .collect::<FxHashMap<_, _>>();
    let compute_ms = elapsed_ms(compute_started);

    let rewrite_started = Instant::now();
    dwa.states_mut().par_iter_mut().for_each(|state| {
        let Some(final_weight) = state.final_weight.as_ref() else {
            return;
        };
        if final_weight.is_empty() {
            return;
        }
        let final_key = final_weight.ptr_key();
        state.transitions.retain(|_, (_, edge_weight)| {
            let key = (edge_weight.ptr_key(), final_key);
            let new_weight = results
                .get(&key)
                .expect("every outgoing/final weight pair must be precomputed");
            if new_weight != edge_weight {
                *edge_weight = new_weight.clone();
            }
            !edge_weight.is_empty()
        });
    });
    let rewrite_ms = elapsed_ms(rewrite_started);
    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][subtract_final_global_pairs] pairs={} collect_ms={collect_ms:.3} compute_ms={compute_ms:.3} rewrite_ms={rewrite_ms:.3} total_ms={:.3}",
            results.len(),
            collect_ms + compute_ms + rewrite_ms,
        );
    }
}

fn subtract_final_weights_from_outgoing_dwa(dwa: &mut DWA) {
    if std::env::var_os("GLRMASK_PROFILE_SUBTRACT_FINAL_PAIRS").is_some() {
        let started = Instant::now();
        let mut calls = 0usize;
        let mut pairs = FxHashSet::<(usize, usize)>::default();
        let mut edge_weights = FxHashSet::<usize>::default();
        let mut final_weights = FxHashSet::<usize>::default();
        for state in dwa.states() {
            let Some(final_weight) = state.final_weight.as_ref() else { continue; };
            if final_weight.is_empty() { continue; }
            let final_key = final_weight.ptr_key();
            final_weights.insert(final_key);
            for (_, weight) in state.transitions.values() {
                calls += 1;
                let edge_key = weight.ptr_key();
                edge_weights.insert(edge_key);
                pairs.insert((edge_key, final_key));
            }
        }
        eprintln!(
            "[glrmask/profile][subtract_final_pairs] calls={} unique_pairs={} unique_edge_weights={} unique_final_weights={} reuse={:.3} collect_ms={:.3}",
            calls,
            pairs.len(),
            edge_weights.len(),
            final_weights.len(),
            if pairs.is_empty() { 0.0 } else { calls as f64 / pairs.len() as f64 },
            elapsed_ms(started),
        );
    }
    const GLOBAL_PAIR_MIN_TRANSITIONS: usize = 131_072;
    if dwa.num_transitions() >= GLOBAL_PAIR_MIN_TRANSITIONS {
        subtract_final_weights_from_outgoing_dwa_global_pairs(dwa);
    } else {
        subtract_final_weights_from_outgoing_dwa_impl(dwa, rayon::current_num_threads() > 1);
    }
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let mut nwa = NWA::new(0, 0);
    *nwa.states_mut() = vec![crate::automata::weighted::nwa::NWAState::default(); dwa.states().len()];
    nwa.set_start_states(vec![dwa.start_state()]);

    for (state_id, state) in dwa.states().iter().enumerate() {
        if let Some(final_weight) = state.final_weight.clone() {
            nwa.states_mut()[state_id].final_weight = Some(final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            nwa.states_mut()[state_id]
                .transitions
                .entry(label)
                .or_default()
                .push((*target, weight.clone()));
        }
    }

    nwa
}

fn compute_productive_terminal_states(summaries: &StateSummaries) -> Vec<bool> {
    let states = &summaries.states;
    let mut reverse_edges: Vec<Vec<u32>> = vec![Vec::new(); states.len()];
    let mut productive = vec![false; states.len()];
    let mut worklist = VecDeque::new();

    for (state_id, state) in states.iter().enumerate() {
        if state
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            productive[state_id] = true;
            worklist.push_back(state_id as u32);
        }

        for (target, weight) in &state.epsilon_branches {
            if !weight.is_empty() && (*target as usize) < states.len() {
                reverse_edges[*target as usize].push(state_id as u32);
            }
        }

        for branch in &state.branches {
            if (branch.target as usize) < states.len()
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                reverse_edges[branch.target as usize].push(state_id as u32);
            }
        }
    }

    while let Some(target) = worklist.pop_front() {
        for &source in &reverse_edges[target as usize] {
            let source_idx = source as usize;
            if !productive[source_idx] {
                productive[source_idx] = true;
                worklist.push_back(source);
            }
        }
    }

    productive
}

fn append_weighted_template_redirecting_finals(
    arena: &mut NWA,
    template: &NWA,
    weight: &Weight,
    continuation_state: u32,
) -> NwaBody {
    let offset = arena.states().len() as u32;
    let body = arena.append_with_body(template);
    let appended_len = template.states().len();

    for state_id in offset as usize..offset as usize + appended_len {
        let state = &mut arena.states_mut()[state_id];
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets {
                *edge_weight = weight.clone();
            }
        }
        for (_, epsilon_weight) in &mut state.epsilons {
            *epsilon_weight = weight.clone();
        }
    }

    for state_id in offset as usize..offset as usize + appended_len {
        if arena.states_mut()[state_id].final_weight.take().is_some() {
            arena.add_epsilon(state_id as u32, continuation_state, weight.clone());
        }
    }

    body
}

fn append_bundle_redirecting_finals(
    arena: &mut NWA,
    bundle: &NWA,
    continuation_state: u32,
) -> NwaBody {
    let offset = arena.states().len() as u32;
    let body = arena.append_with_body(bundle);
    let appended_len = bundle.states().len();

    for state_id in offset as usize..offset as usize + appended_len {
        let Some(final_weight) = arena.states_mut()[state_id].final_weight.take() else {
            continue;
        };
        if !final_weight.is_empty() {
            arena.add_epsilon(state_id as u32, continuation_state, final_weight);
        }
    }

    body
}

fn append_branch_fragment(
    arena: &mut NWA,
    summaries: &StateSummaries,
    templates: &Templates,
    built_bundle_cache: &mut [Option<Arc<NWA>>],
    bundle_id: usize,
    continuation_state: u32,
    compose_detail: Option<&mut ParserDwaComposeDetailProfile>,
) -> Option<NwaBody> {
    let bundle = summaries.unique_bundles.get(bundle_id)?;
    if !summaries.bundle_accepts.get(bundle_id).copied().unwrap_or(false) {
        return None;
    }

    if bundle.len() == 1 {
        let (&terminal, weight) = bundle.iter().next().expect("len checked");
        if weight.is_empty() {
            return None;
        }
        let template = templates.by_terminal_nwa.get(&terminal)?;
        return Some(append_weighted_template_redirecting_finals(
            arena,
            template,
            weight,
            continuation_state,
        ));
    }

    // STICKY NOTE: keep parser bundles eagerly determinized here.
    //
    // It is tempting to leave multi-terminal bundles nondeterministic or factored so
    // this stage can avoid a large deterministic bundle build. Do not do that. These
    // bundles are the unit on which downstream negative-resolution operates. If a
    // bundle is left nondeterministic, negative-resolution has to distribute one
    // bundle alternatives against the next bundle alternatives, which recreates
    // the same cross-product later and can become a combinatorial explosion between
    // adjacent bundles. Eager determinization pays that cost once, locally, and gives
    // negative-resolution a stable deterministic object to compose.
    //
    // NEVER remove this note without replacing it with an equally explicit invariant
    // explaining why parser-bundle determinization is required. We have repeatedly
    // rediscovered this and incorrectly proposed removing determinization. If the
    // first multi-terminal bundle cannot be determinized, the fix is to reduce the
    // bundle/grammar/compiler state space, not to pass a nondeterministic bundle
    // downstream.
    if built_bundle_cache[bundle_id].is_none() {
        if let Some(detail) = compose_detail {
            let (bundle_nwa, bundle_profile) = templates.build_bundle_profiled(bundle);
            detail.bundle_profile_total_ms += bundle_profile.total_ms;
            detail.bundle_profile_build_group_dfas_ms += bundle_profile.build_group_dfas_ms;
            detail.bundle_profile_union_groups_ms += bundle_profile.union_groups_ms;
            detail.bundle_profile_determinize_ms += bundle_profile.determinize_bundle_ms;
            detail.bundle_profile_minimize_ms += bundle_profile.minimize_ms;
            detail.bundle_profile_dwa_to_nwa_ms += bundle_profile.dwa_to_nwa_ms;
            detail.bundle_profile_result_dwa_states += bundle_profile.result_dwa_states;
            detail.bundle_profile_result_dwa_transitions += bundle_profile.result_dwa_transitions;
            detail.bundle_profile_result_nwa_states += bundle_profile.result_nwa_states;
            detail.bundle_profile_result_nwa_transitions += bundle_profile.result_nwa_transitions;
            eprintln!(
                "[glrmask/profile][parser_bundle] bundle_id={} terminals={} weight_groups={} single_entry_weights={} single_tsid_weights={} total_weight_outer_ranges={} build_group_dfas_ms={:.3} union_groups_ms={:.3} determinize_bundle_ms={:.3} det_pop_ms={:.3} det_alive_ms={:.3} det_final_ms={:.3} det_collect_labels_ms={:.3} det_next_state_ms={:.3} det_edge_weight_ms={:.3} det_lookup_ms={:.3} det_add_transition_ms={:.3} det_states={} det_labels={} det_transitions={} det_edge_subset_total={} det_edge_subset_max={} det_edge_cache_hits={} det_edge_cache_misses={} minimize_ms={:.3} minimize_skipped={} dwa_to_nwa_ms={:.3} total_ms={:.3} result_dwa_states={} result_dwa_transitions={} result_nwa_states={} result_nwa_transitions={}",
                bundle_id,
                bundle_profile.input_terminals,
                bundle_profile.weight_groups,
                bundle_profile.single_entry_weights,
                bundle_profile.single_tsid_weights,
                bundle_profile.total_weight_outer_ranges,
                bundle_profile.build_group_dfas_ms,
                bundle_profile.union_groups_ms,
                bundle_profile.determinize_bundle_ms,
                bundle_profile.determinize_pop_state_ms,
                bundle_profile.determinize_alive_groups_ms,
                bundle_profile.determinize_final_weight_ms,
                bundle_profile.determinize_collect_labels_ms,
                bundle_profile.determinize_next_state_ms,
                bundle_profile.determinize_edge_weight_ms,
                bundle_profile.determinize_state_lookup_ms,
                bundle_profile.determinize_add_transition_ms,
                bundle_profile.determinize_states_visited,
                bundle_profile.determinize_labels_processed,
                bundle_profile.determinize_transitions_added,
                bundle_profile.determinize_edge_subset_total,
                bundle_profile.determinize_edge_subset_max,
                bundle_profile.determinize_edge_cache_hits,
                bundle_profile.determinize_edge_cache_misses,
                bundle_profile.minimize_ms,
                bundle_profile.minimize_skipped,
                bundle_profile.dwa_to_nwa_ms,
                bundle_profile.total_ms,
                bundle_profile.result_dwa_states,
                bundle_profile.result_dwa_transitions,
                bundle_profile.result_nwa_states,
                bundle_profile.result_nwa_transitions,
            );
            built_bundle_cache[bundle_id] = Some(Arc::new(bundle_nwa));
        } else {
            built_bundle_cache[bundle_id] = Some(Arc::new(templates.build_bundle(bundle)));
        }
    }
    let bundle_nwa = built_bundle_cache[bundle_id]
        .as_ref()
        .expect("bundle cache entry just initialized");
    Some(append_bundle_redirecting_finals(
        arena,
        bundle_nwa.as_ref(),
        continuation_state,
    ))
}

fn build_parser_nwa_from_terminal_dwa(
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> Option<(NWA, ParserNwaBuildProfile)> {
    let total_started_at = Instant::now();
    let state_prep_started_at = Instant::now();
    let summaries = build_state_summaries(terminal_dwa, grammar, templates);
    let productive = compute_productive_terminal_states(&summaries);
    let state_prep_ms = elapsed_ms(state_prep_started_at);
    let states = &summaries.states;
    let compose_detail_enabled = parser_dwa_compose_detail_enabled();
    let mut compose_detail = ParserDwaComposeDetailProfile {
        total_states: states.len(),
        productive_states: productive.iter().filter(|&&is_productive| is_productive).count(),
        total_branches: states
            .iter()
            .map(|state| state.epsilon_branches.len() + state.branches.len())
            .sum(),
        productive_branches: 0,
        unique_bundles: summaries.unique_bundles.len(),
        accepting_bundles: summaries.bundle_accepts.iter().filter(|&&accepts| accepts).count(),
        ..ParserDwaComposeDetailProfile::default()
    };

    let productive_start_states: Vec<u32> = summaries
        .start_states
        .iter()
        .copied()
        .filter(|state| productive.get(*state as usize).copied().unwrap_or(false))
        .collect();
    if productive_start_states.is_empty() {
        return None;
    }

    let graph_started_at = Instant::now();
    let mut arena = NWA::new(0, 0);
    let mut continuation_states = vec![u32::MAX; states.len()];

    let state_init_started_at = Instant::now();
    for (state_id, state) in states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        let continuation_state = arena.add_state();
        continuation_states[state_id] = continuation_state;
        if let Some(final_weight) = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
        {
            arena.set_final_weight(continuation_state, final_weight.clone());
        }
    }
    compose_detail.state_init_ms = elapsed_ms(state_init_started_at);

    let mut branch_fragment_memo: FxHashMap<(usize, u32), NwaBody> = FxHashMap::default();
    let mut used_multi_bundle = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            let target_idx = branch.target as usize;
            if productive.get(target_idx).copied().unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
                && summaries.unique_bundles[branch.bundle_id].len() > 1
            {
                used_multi_bundle[branch.bundle_id] = true;
            }
        }
    }

    use rayon::prelude::*;

    let mut built_bundle_cache: Vec<Option<Arc<NWA>>> = vec![None; summaries.unique_bundles.len()];
    if !compose_detail_enabled {
        built_bundle_cache = summaries
            .unique_bundles
            .par_iter()
            .enumerate()
            .map(|(bundle_id, bundle)| {
                used_multi_bundle[bundle_id]
                    .then(|| Arc::new(templates.build_bundle(bundle)))
            })
            .collect();
    }

    let branch_walk_started_at = Instant::now();
    for (state_id, state) in states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        let from = continuation_states[state_id];
        assert_ne!(from, u32::MAX, "missing parser-DWA continuation state");

        for (target, weight) in &state.epsilon_branches {
            let target_idx = *target as usize;
            if weight.is_empty() || !productive.get(target_idx).copied().unwrap_or(false) {
                continue;
            }
            let target_continuation = continuation_states[target_idx];
            assert_ne!(
                target_continuation,
                u32::MAX,
                "missing parser-DWA epsilon target continuation state",
            );
            arena.add_epsilon(from, target_continuation, weight.clone());
            compose_detail.productive_branches += 1;
            compose_detail.epsilon_edges_added += 1;
        }

        for branch in &state.branches {
            let target_idx = branch.target as usize;
            if !productive.get(target_idx).copied().unwrap_or(false)
                || !summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                continue;
            }
            compose_detail.productive_branches += 1;

            let target_continuation = continuation_states[target_idx];
            assert_ne!(
                target_continuation,
                u32::MAX,
                "missing parser-DWA target continuation state",
            );
            let fragment_key = (branch.bundle_id, branch.target);
            let fragment = if let Some(existing) = branch_fragment_memo.get(&fragment_key) {
                if compose_detail_enabled {
                    let memo_hit_started_at = Instant::now();
                    compose_detail.memo_hits += 1;
                    let cloned = existing.clone();
                    compose_detail.memo_hit_clone_ms += elapsed_ms(memo_hit_started_at);
                    cloned
                } else {
                    existing.clone()
                }
            } else {
                if compose_detail_enabled {
                    compose_detail.memo_misses += 1;
                    if built_bundle_cache[branch.bundle_id].is_none()
                        && summaries.unique_bundles[branch.bundle_id].len() > 1
                    {
                        compose_detail.bundle_cache_builds += 1;
                    }
                }
                let fragment_build_started_at = Instant::now();
                let Some(body) = append_branch_fragment(
                    &mut arena,
                    &summaries,
                    &templates,
                    &mut built_bundle_cache,
                    branch.bundle_id,
                    target_continuation,
                    compose_detail_enabled.then_some(&mut compose_detail),
                ) else {
                    continue;
                };
                compose_detail.fragment_build_ms += elapsed_ms(fragment_build_started_at);
                branch_fragment_memo.insert(fragment_key, body.clone());
                body
            };

            let epsilon_link_started_at = Instant::now();
            let fragment_start_states_len = fragment.start_states.len();
            for start in fragment.start_states {
                arena.add_epsilon(from, start, Weight::all());
                compose_detail.epsilon_edges_added += 1;
            }
            compose_detail.fragment_start_states_total += fragment_start_states_len;
            compose_detail.epsilon_link_ms += elapsed_ms(epsilon_link_started_at);
        }
    }
    compose_detail.branch_walk_ms = elapsed_ms(branch_walk_started_at);

    let parser_start_states: Vec<u32> = productive_start_states
        .into_iter()
        .map(|state| continuation_states[state as usize])
        .collect();
    assert!(
        parser_start_states.iter().all(|state| *state != u32::MAX),
        "missing parser-DWA start continuation state",
    );
    arena.set_start_states(parser_start_states);
    let compose_state_ms = elapsed_ms(graph_started_at);

    if compose_detail_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa_compose] total_states={} productive_states={} total_branches={} productive_branches={} unique_bundles={} accepting_bundles={} state_init_ms={:.3} branch_walk_ms={:.3} memo_hit_clone_ms={:.3} fragment_build_ms={:.3} epsilon_link_ms={:.3} memo_hits={} memo_misses={} bundle_cache_builds={} epsilon_edges_added={} fragment_start_states_total={}",
            compose_detail.total_states,
            compose_detail.productive_states,
            compose_detail.total_branches,
            compose_detail.productive_branches,
            compose_detail.unique_bundles,
            compose_detail.accepting_bundles,
            compose_detail.state_init_ms,
            compose_detail.branch_walk_ms,
            compose_detail.memo_hit_clone_ms,
            compose_detail.fragment_build_ms,
            compose_detail.epsilon_link_ms,
            compose_detail.memo_hits,
            compose_detail.memo_misses,
            compose_detail.bundle_cache_builds,
            compose_detail.epsilon_edges_added,
            compose_detail.fragment_start_states_total,
        );
        eprintln!(
            "[glrmask/profile][parser_dwa_compose_bundles] bundle_cache_builds={} bundle_profile_total_ms={:.3} build_group_dfas_ms={:.3} union_groups_ms={:.3} determinize_bundle_ms={:.3} minimize_ms={:.3} dwa_to_nwa_ms={:.3} result_dwa_states_total={} result_dwa_transitions_total={} result_nwa_states_total={} result_nwa_transitions_total={}",
            compose_detail.bundle_cache_builds,
            compose_detail.bundle_profile_total_ms,
            compose_detail.bundle_profile_build_group_dfas_ms,
            compose_detail.bundle_profile_union_groups_ms,
            compose_detail.bundle_profile_determinize_ms,
            compose_detail.bundle_profile_minimize_ms,
            compose_detail.bundle_profile_dwa_to_nwa_ms,
            compose_detail.bundle_profile_result_dwa_states,
            compose_detail.bundle_profile_result_dwa_transitions,
            compose_detail.bundle_profile_result_nwa_states,
            compose_detail.bundle_profile_result_nwa_transitions,
        );
    }

    Some((
        arena,
        ParserNwaBuildProfile {
            state_prep_ms,
            compose_state_ms,
            parser_nwa_build_ms: elapsed_ms(total_started_at),
        },
    ))
}


// Exact compile-time parser-stack domain algebra used by the compiled-subgrammar linker.
fn determinize_boolean_domain_with_supports(domain: &NWA) -> DeterminizedDwaWithSupports {
    fn epsilon_closure(domain: &NWA, seeds: &[u32]) -> Vec<u32> {
        let mut seen = FxHashSet::<u32>::default();
        let mut stack = seeds.to_vec();
        while let Some(state) = stack.pop() {
            if !seen.insert(state) {
                continue;
            }
            let Some(node) = domain.states().get(state as usize) else {
                continue;
            };
            for (target, weight) in &node.epsilons {
                debug_assert!(weight.is_full() || weight.is_empty());
                if !weight.is_empty() {
                    stack.push(*target);
                }
            }
        }
        let mut closure = seen.into_iter().collect::<Vec<_>>();
        closure.sort_unstable();
        closure
    }

    let mut start_seeds = domain.start_states().to_vec();
    start_seeds.sort_unstable();
    start_seeds.dedup();
    let start = epsilon_closure(domain, &start_seeds);
    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![start.clone()];
    if start.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    let mut subset_to_state = FxHashMap::<Vec<u32>, u32>::default();
    let mut subsets = Vec::<Vec<u32>>::new();
    subset_to_state.insert(start.clone(), dwa.start_state());
    subsets.push(start);
    let mut queue = VecDeque::from([dwa.start_state()]);

    while let Some(source_id) = queue.pop_front() {
        let subset = subsets[source_id as usize].clone();
        if subset.iter().any(|&state| {
            domain.states()[state as usize]
                .final_weight
                .as_ref()
                .is_some_and(|weight| !weight.is_empty())
        }) {
            dwa.set_final_weight(source_id, Weight::all());
        }

        // IMPORTANT: DEFAULT_LABEL is still an ordinary symbolic label here.
        // It becomes a parser-state fallback only after this subset's NWA
        // support has been recorded and PossibleOutgoingIds can be derived.
        let mut targets_by_label = BTreeMap::<i32, Vec<u32>>::new();
        for &state in &subset {
            let node = &domain.states()[state as usize];
            for (&label, targets) in &node.transitions {
                if is_negative_label(label) {
                    panic!("boolean parser-domain determinization requires negative-free NWA");
                }
                let seeds = targets_by_label.entry(label).or_default();
                for (target, weight) in targets {
                    debug_assert!(weight.is_full() || weight.is_empty());
                    if !weight.is_empty() {
                        seeds.push(*target);
                    }
                }
            }
        }

        for (label, mut seeds) in targets_by_label {
            seeds.sort_unstable();
            seeds.dedup();
            let closure = epsilon_closure(domain, &seeds);
            if closure.is_empty() {
                continue;
            }
            let target = if let Some(&existing) = subset_to_state.get(&closure) {
                existing
            } else {
                let target = dwa.add_state();
                subset_to_state.insert(closure.clone(), target);
                supports.push(closure.clone());
                subsets.push(closure);
                queue.push_back(target);
                target
            };
            dwa.add_transition(source_id, label, target, Weight::all());
        }
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

pub fn determinize_boolean_parser_stack_domain_nwa(
    table: &GLRTable,
    domain: &NWA,
) -> DWA {
    let determinized = determinize_boolean_domain_with_supports(domain);
    let mut result = determinized.dwa;
    let possible_by_state = build_possible_outgoing_ids_by_state(
        domain,
        &determinized.supports,
        table.num_states,
    );
    if std::env::var_os("GLRMASK_EXPERIMENT_DISABLE_BOOLEAN_DOMAIN_DEFAULT_OPT").is_none() {
        optimize_parser_dwa_defaults(&mut result, &possible_by_state, table.num_states);
    }
    subtract_final_weights_from_outgoing_dwa_impl(&mut result, false);
    determinize_parser_dwa_with_fallbacks(&result, &possible_by_state, table.num_states)
}

pub fn normalize_parser_stack_domain_nwa(table: &GLRTable, domain: &NWA) -> DWA {
    minimize(&determinize_boolean_parser_stack_domain_nwa(table, domain))
}

/// Normalize a boolean parser-stack NWA while preserving its explicit rows.
///
/// This is the exact standalone form needed when several independently
/// supported parser domains will be combined later. Synthesizing DEFAULT rows
/// here would discard the support provenance required by that later union.
pub fn normalize_parser_stack_domain_nwa_preserving_explicit(
    table: &GLRTable,
    domain: &NWA,
) -> DWA {
    let determinized = determinize_boolean_domain_with_supports(domain);
    let mut result = determinized.dwa;
    let possible_by_state = build_possible_outgoing_ids_by_state(
        domain,
        &determinized.supports,
        table.num_states,
    );
    subtract_final_weights_from_outgoing_dwa_impl(&mut result, false);
    let result = determinize_parser_dwa_with_fallbacks(
        &result,
        &possible_by_state,
        table.num_states,
    );
    minimize(&result)
}

/// Normalize an already-positive weighted parser-stack NWA into the ordinary
/// runtime parser DWA representation.
///
/// This is the post-negative-resolution half of
/// `build_parser_dwa_from_terminal_dwa_with_precomputed_templates`: callers
/// that have composed parser-stack effects directly can reuse the exact same
/// support-aware DEFAULT/finality semantics without first reconstructing a
/// terminal automaton.

pub fn normalize_weighted_parser_stack_nwa(table: &GLRTable, parser_nwa: &NWA) -> DWA {
    let num_parser_states = table.num_states;
    let determinized = determinize_with_supports(parser_nwa, Some(num_parser_states));
    let mut parser_dwa = determinized.dwa;
    let possible_by_state = build_possible_outgoing_ids_by_state(
        parser_nwa,
        &determinized.supports,
        num_parser_states,
    );
    if std::env::var_os("GLRMASK_EXPERIMENT_LAZY_DIRECT_DISABLE_DEFAULT_OPT").is_none() {
        optimize_parser_dwa_defaults(&mut parser_dwa, &possible_by_state, num_parser_states);
    }
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa);
    parser_dwa = determinize_parser_dwa_with_fallbacks(
        &parser_dwa,
        &possible_by_state,
        num_parser_states,
    );
    if should_skip_parser_dwa_minimization(parser_dwa.states().len(), parser_dwa.num_transitions()) {
        parser_dwa
    } else {
        minimize(&parser_dwa)
    }
}

pub fn universal_parser_stack_domain_dwa() -> DWA {
    let mut result = DWA::new(0, 1);
    result.set_final_weight(0, Weight::all());
    result
}

pub fn universal_parser_stack_domain_nwa() -> NWA {
    let mut result = NWA::new(0, 0);
    let start = result.add_state();
    result.set_start_states(vec![start]);
    result.set_final_weight(start, Weight::all());
    result
}

/// Exact positive-NWA preimage. Unlike the DWA wrapper below, this deliberately
/// does not determinize or normalize the result. That makes it suitable for a
/// backward dynamic program over a finite terminal suffix DAG: a later
/// terminal template can cancel its pushed states directly against this
/// positive NWA, and determinization can be deferred until the completed root
/// predicates are weighted and unioned.
pub fn build_terminal_bundle_preimage_domain_nwa(
    table: &GLRTable,
    templates: &Templates,
    terminals: &[TerminalID],
    target_domain: &NWA,
) -> Option<NWA> {
    if terminals.is_empty() {
        return None;
    }
    let terminal_weights = terminals
        .iter()
        .copied()
        .map(|terminal| (terminal, Weight::all()))
        .collect::<BTreeMap<_, _>>();
    if terminal_weights
        .keys()
        .any(|terminal| !templates.by_terminal_nwa.contains_key(terminal))
    {
        return None;
    }
    let mut bundle = templates.build_bundle(&terminal_weights);
    for state in bundle.states_mut() {
        for targets in state.transitions.values_mut() {
            for (_, weight) in targets {
                if weight.is_empty() {
                    *weight = Weight::all();
                }
            }
        }
        for (_, weight) in &mut state.epsilons {
            if weight.is_empty() {
                *weight = Weight::all();
            }
        }
        if state.final_weight.as_ref().is_some_and(Weight::is_empty) {
            state.final_weight = Some(Weight::all());
        }
    }

    let mut arena = NWA::new(0, 0);
    let bundle_offset = arena.states().len() as u32;
    let bundle_body = arena.append_with_body(&bundle);
    let bundle_finals = bundle
        .states()
        .iter()
        .enumerate()
        .filter_map(|(local, state)| {
            state
                .final_weight
                .as_ref()
                .is_some_and(|weight| !weight.is_empty())
                .then_some(bundle_offset + local as u32)
        })
        .collect::<Vec<_>>();
    if bundle_finals.is_empty() {
        return None;
    }
    let target_body = arena.append_with_body(target_domain);
    for source in bundle_finals {
        let Some(final_weight) = arena.states_mut()[source as usize].final_weight.take() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        for &target_start in &target_body.start_states {
            arena.add_epsilon(source, target_start, final_weight.clone());
        }
    }
    arena.set_start_states(bundle_body.start_states);
    resolve_negative_codes_in_nwa(
        &mut arena,
        table.construction == GlrTableConstruction::ExperimentalCoreMerged,
    );
    Some(arena)
}


#[derive(Debug, Clone, Copy, Default)]
pub struct ParserStackPreimageProfile {
    pub bundle_ms: f64,
    pub concatenate_ms: f64,
    pub resolve_ms: f64,
    pub normalize_ms: f64,
    pub total_ms: f64,
    pub bundle_states: usize,
    pub concatenated_states: usize,
    pub result_states: usize,
}


fn advance_boolean_parser_domain_state(
    domain: &DWA,
    state: u32,
    parser_state: u32,
) -> Option<u32> {
    let row = domain.states().get(state as usize)?;
    if row
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
    {
        // Parser-stack languages are prefix languages: once the target domain
        // has accepted the visible top-of-stack prefix, any deeper pushed or
        // pre-existing stack suffix is irrelevant. Treat final states as
        // absorbing while consuming the terminal effect's remaining pushes.
        return Some(state);
    }
    let label = parser_state as i32;
    let (target, weight) = row
        .transitions
        .get(&label)
        .or_else(|| row.transitions.get(&DEFAULT_LABEL))?;
    debug_assert!(weight.is_full() || weight.is_empty());
    (!weight.is_empty()).then_some(*target)
}

fn direct_negative_suffix_residuals(
    bundle: &NWA,
    bundle_state: u32,
    domain: &DWA,
    domain_state: u32,
    memo: &mut FxHashMap<(u32, u32), Option<Vec<u32>>>,
) -> Option<Vec<u32>> {
    if let Some(cached) = memo.get(&(bundle_state, domain_state)) {
        return cached.clone();
    }
    let node = bundle.states().get(bundle_state as usize)?;
    let mut residuals = Vec::<u32>::new();
    if node
        .final_weight
        .as_ref()
        .is_some_and(|weight| !weight.is_empty())
    {
        residuals.push(domain_state);
    }
    for (&label, targets) in &node.transitions {
        if !is_negative_label(label) {
            // Stack-effect templates are in read-then-push normal form. Once a
            // push is encountered, seeing another read would require a more
            // general transducer product; decline this direct fast path rather
            // than weakening semantics.
            memo.insert((bundle_state, domain_state), None);
            return None;
        }
        let parser_state = negative_to_positive_label(label) as u32;
        for (target, weight) in targets {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            // Pushes are observed from the final stack top downward, which is
            // the reverse of the negative-edge order in the stack effect. The
            // generic cancellation solver establishes downstream cancellations
            // first and then propagates upstream queries through the derived
            // epsilons. Mirror that algebra directly: resolve the child's later
            // pushes first, then consume this push from each residual domain.
            let Some(child) = direct_negative_suffix_residuals(
                bundle,
                *target,
                domain,
                domain_state,
                memo,
            ) else {
                memo.insert((bundle_state, domain_state), None);
                return None;
            };
            for child_domain in child {
                if let Some(next_domain) = advance_boolean_parser_domain_state(
                    domain,
                    child_domain,
                    parser_state,
                ) {
                    residuals.push(next_domain);
                }
            }
        }
    }
    for (target, weight) in &node.epsilons {
        debug_assert!(weight.is_full() || weight.is_empty());
        if weight.is_empty() {
            continue;
        }
        let Some(child) = direct_negative_suffix_residuals(
            bundle,
            *target,
            domain,
            domain_state,
            memo,
        ) else {
            memo.insert((bundle_state, domain_state), None);
            return None;
        };
        residuals.extend(child);
    }
    residuals.sort_unstable();
    residuals.dedup();
    memo.insert((bundle_state, domain_state), Some(residuals.clone()));
    Some(residuals)
}

/// Algebraic boolean preimage for a read-then-push stack-effect bundle.
///
/// Positive labels are reads from the pre-token parser stack and remain in the
/// resulting automaton. Negative labels are pushes. Because `target_domain` is
/// already deterministic, a pushed parser state is consumed by one ordinary
/// DWA transition (explicit label or DEFAULT fallback); the remaining target
/// state is the exact residual language. This is the relational composition
/// performed by negative-code cancellation, but without materializing the
/// concatenated graph or running a global fixpoint.
pub fn build_prebuilt_terminal_bundle_preimage_domain_dwa_direct_profiled(
    table: &GLRTable,
    bundle: &NWA,
    target_domain: &DWA,
) -> (Option<DWA>, ParserStackPreimageProfile) {
    let total_started_at = Instant::now();
    let mut profile = ParserStackPreimageProfile {
        bundle_states: bundle.states().len(),
        ..ParserStackPreimageProfile::default()
    };
    if bundle.start_states().is_empty() || target_domain.states().is_empty() {
        profile.total_ms = elapsed_ms(total_started_at);
        return (None, profile);
    }

    let build_started_at = Instant::now();
    let target_nwa = target_domain.to_nwa();
    let mut result = NWA::new(0, 0);
    let bundle_offset = 0u32;
    let bundle_body = result.append_with_body(bundle);
    debug_assert_eq!(bundle_offset, 0);
    let target_offset = result.states().len() as u32;
    let target_body = result.append_with_body(&target_nwa);
    debug_assert_eq!(target_body.start_states.len(), 1);

    let target_start = target_domain.start_state();
    let mut memo = FxHashMap::<(u32, u32), Option<Vec<u32>>>::default();
    let bundle_len = bundle.states().len();
    for source in 0..bundle_len {
        let original = &bundle.states()[source];
        let mut positive_transitions = BTreeMap::new();
        let mut epsilons = Vec::<(u32, Weight)>::new();

        if original
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            epsilons.push((target_offset + target_start, Weight::all()));
        }
        for (&label, targets) in &original.transitions {
            if is_negative_label(label) {
                let parser_state = negative_to_positive_label(label) as u32;
                for (target, weight) in targets {
                    debug_assert!(weight.is_full() || weight.is_empty());
                    if weight.is_empty() {
                        continue;
                    }
                    let Some(residuals) = direct_negative_suffix_residuals(
                        bundle,
                        *target,
                        target_domain,
                        target_start,
                        &mut memo,
                    ) else {
                        profile.total_ms = elapsed_ms(total_started_at);
                        return (None, profile);
                    };
                    epsilons.extend(residuals.into_iter().filter_map(|state| {
                        advance_boolean_parser_domain_state(
                            target_domain,
                            state,
                            parser_state,
                        )
                        .map(|residual| (target_offset + residual, Weight::all()))
                    }));
                }
            } else {
                positive_transitions.insert(label, targets.clone());
            }
        }
        // Bundle epsilons belong to the still-reading prefix unless their
        // target immediately enters a negative suffix. Keeping them is exact;
        // any negative transitions at the target are processed when that state
        // is reached by epsilon closure during determinization.
        epsilons.extend(original.epsilons.iter().cloned());
        epsilons.sort_unstable_by_key(|(target, _)| *target);
        epsilons.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);
        let state = &mut result.states_mut()[source];
        state.transitions = positive_transitions;
        state.epsilons = epsilons;
        state.final_weight = None;
    }
    result.set_start_states(bundle_body.start_states);
    // Negative cancellation is now algebraic, but parser stack languages also
    // project finality backward through DEFAULT/epsilon edges. Preserve that
    // exact stack-prefix semantics before ordinary boolean determinization.
    apply_finality_fixpoint(&mut result);
    profile.concatenate_ms = elapsed_ms(build_started_at);
    profile.concatenated_states = result.states().len();

    let normalize_started_at = Instant::now();
    let domain = normalize_parser_stack_domain_nwa(table, &result);
    profile.normalize_ms = elapsed_ms(normalize_started_at);
    profile.result_states = domain.states().len();
    profile.total_ms = elapsed_ms(total_started_at);

    (Some(domain), profile)
}


pub fn build_boolean_terminal_bundle_nwa(
    templates: &Templates,
    terminals: &[TerminalID],
) -> Option<NWA> {
    if terminals.is_empty()
        || terminals
            .iter()
            .any(|terminal| !templates.by_terminal_nwa.contains_key(terminal))
    {
        return None;
    }
    let terminal_weights = terminals
        .iter()
        .copied()
        .map(|terminal| (terminal, Weight::all()))
        .collect::<BTreeMap<_, _>>();
    let mut bundle = templates.build_bundle(&terminal_weights);
    for state in bundle.states_mut() {
        for targets in state.transitions.values_mut() {
            for (_, weight) in targets {
                if weight.is_empty() {
                    *weight = Weight::all();
                }
            }
        }
        for (_, weight) in &mut state.epsilons {
            if weight.is_empty() {
                *weight = Weight::all();
            }
        }
        if state.final_weight.as_ref().is_some_and(Weight::is_empty) {
            state.final_weight = Some(Weight::all());
        }
    }
    Some(bundle)
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LazyBooleanDomainExpr {
    Empty,
    Universal,
    Read { label: i32, child: u32 },
    Union { left: u32, right: u32 },
}

/// Lazy compile-time algebra for boolean parser-stack prefix languages.
///
/// Unlike `SharedBooleanParserDomains`, this representation deliberately does
/// not determinize unions. DEFAULT remains an ordinary symbolic stack matcher
/// until the final NWA normalization, preserving the parser-support semantics
/// used by the generic compiler. The DAG is only an intermediate linking form;
/// runtime artifacts remain ordinary DWAs.
pub struct LazyBooleanParserDomains {
    nodes: Vec<LazyBooleanDomainExpr>,
    prefix_final: Vec<bool>,
    reads: FxHashMap<(i32, u32), u32>,
    unions: FxHashMap<(u32, u32), u32>,
    advance_memo: FxHashMap<(u32, u32), u32>,
    default_advance_memo: FxHashMap<u32, u32>,
    prefix_finalized: FxHashSet<u32>,
}

impl Default for LazyBooleanParserDomains {
    fn default() -> Self {
        Self::new()
    }
}

impl LazyBooleanParserDomains {
    pub const EMPTY: u32 = 0;
    pub const UNIVERSAL: u32 = 1;

    pub fn new() -> Self {
        Self {
            nodes: vec![LazyBooleanDomainExpr::Empty, LazyBooleanDomainExpr::Universal],
            prefix_final: vec![false, true],
            reads: FxHashMap::default(),
            unions: FxHashMap::default(),
            advance_memo: FxHashMap::default(),
            default_advance_memo: FxHashMap::default(),
            prefix_finalized: FxHashSet::from_iter([Self::EMPTY, Self::UNIVERSAL]),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty_root(&self, root: u32) -> bool {
        root == Self::EMPTY
    }

    pub fn is_universal_root(&self, root: u32) -> bool {
        root == Self::UNIVERSAL
    }

    /// Concrete parser-state labels whose derivative can differ from the
    /// wildcard-only derivative. DEFAULT itself is deliberately excluded.
    pub fn explicit_labels(&self, root: u32) -> Vec<i32> {
        let mut labels = BTreeSet::<i32>::new();
        let mut seen = FxHashSet::<u32>::default();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if !seen.insert(node) {
                continue;
            }
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty | LazyBooleanDomainExpr::Universal => {}
                LazyBooleanDomainExpr::Read { label, child } => {
                    if label != DEFAULT_LABEL {
                        debug_assert!(!is_negative_label(label));
                        labels.insert(label);
                    }
                    stack.push(child);
                }
                LazyBooleanDomainExpr::Union { left, right } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        labels.into_iter().collect()
    }

    /// Derivative for any concrete parser state not represented by an explicit
    /// read in the expression. Only symbolic DEFAULT reads survive.
    pub fn advance_default(&mut self, root: u32) -> u32 {
        if root == Self::EMPTY || self.prefix_final[root as usize] {
            return root;
        }
        if let Some(&cached) = self.default_advance_memo.get(&root) {
            return cached;
        }
        let expr = self.nodes[root as usize];
        let result = match expr {
            LazyBooleanDomainExpr::Empty => Self::EMPTY,
            LazyBooleanDomainExpr::Universal => Self::UNIVERSAL,
            LazyBooleanDomainExpr::Read { label, child } => {
                if label == DEFAULT_LABEL { child } else { Self::EMPTY }
            }
            LazyBooleanDomainExpr::Union { left, right } => {
                let left = self.advance_default(left);
                let right = self.advance_default(right);
                self.union(left, right)
            }
        };
        self.default_advance_memo.insert(root, result);
        result
    }

    pub fn read(&mut self, label: i32, child: u32) -> u32 {
        if child == Self::EMPTY {
            return Self::EMPTY;
        }
        let key = (label, child);
        if let Some(&existing) = self.reads.get(&key) {
            return existing;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(LazyBooleanDomainExpr::Read { label, child });
        self.prefix_final.push(false);
        self.reads.insert(key, id);
        id
    }

    pub fn union(&mut self, left: u32, right: u32) -> u32 {
        if left == right || right == Self::EMPTY {
            return left;
        }
        if left == Self::EMPTY {
            return right;
        }
        if left == Self::UNIVERSAL || right == Self::UNIVERSAL {
            return Self::UNIVERSAL;
        }
        let (left, right) = if left < right { (left, right) } else { (right, left) };
        if let Some(&existing) = self.unions.get(&(left, right)) {
            return existing;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(LazyBooleanDomainExpr::Union { left, right });
        self.prefix_final.push(
            self.prefix_final[left as usize] || self.prefix_final[right as usize],
        );
        self.unions.insert((left, right), id);
        id
    }

    pub fn union_all(&mut self, roots: impl IntoIterator<Item = u32>) -> u32 {
        roots.into_iter().fold(Self::EMPTY, |acc, root| self.union(acc, root))
    }

    /// Consume one concrete parser state from the top of the represented stack
    /// language. DEFAULT is a symbolic wildcard here: final parser-DWA fallback
    /// normalization happens only when the expression is exported to an NWA.
    pub fn advance(&mut self, root: u32, parser_state: u32) -> u32 {
        if root == Self::EMPTY || self.prefix_final[root as usize] {
            return root;
        }
        if let Some(&cached) = self.advance_memo.get(&(root, parser_state)) {
            return cached;
        }
        let expr = self.nodes[root as usize];
        let result = match expr {
            LazyBooleanDomainExpr::Empty => Self::EMPTY,
            LazyBooleanDomainExpr::Universal => Self::UNIVERSAL,
            LazyBooleanDomainExpr::Read { label, child } => {
                if label == DEFAULT_LABEL || label == parser_state as i32 {
                    child
                } else {
                    Self::EMPTY
                }
            }
            LazyBooleanDomainExpr::Union { left, right } => {
                let left = self.advance(left, parser_state);
                let right = self.advance(right, parser_state);
                self.union(left, right)
            }
        };
        self.advance_memo.insert((root, parser_state), result);
        result
    }

    /// Apply parser-stack prefix finality after a complete bundle
    /// preimage has been assembled. Epsilon/union finality is represented by
    /// the union node's `prefix_final` bit as it is built. The only remaining
    /// backward-finality edge in this negative-free expression is DEFAULT.
    ///
    /// Finalized roots remain structurally intact: their final bit makes them
    /// absorbing for subsequent pushed parser states, while positive-read
    /// provenance is preserved until export/normalization.
    pub fn finalize_prefix_domain(&mut self, root: u32) -> bool {
        if self.prefix_finalized.contains(&root) {
            return self.prefix_final[root as usize];
        }
        let expr = self.nodes[root as usize];
        let final_value = match expr {
            LazyBooleanDomainExpr::Empty => false,
            LazyBooleanDomainExpr::Universal => true,
            LazyBooleanDomainExpr::Read { label, child } => {
                let child_final = self.finalize_prefix_domain(child);
                label == DEFAULT_LABEL && child_final
            }
            LazyBooleanDomainExpr::Union { left, right } => {
                self.finalize_prefix_domain(left) || self.finalize_prefix_domain(right)
            }
        };
        self.prefix_final[root as usize] = final_value;
        self.prefix_finalized.insert(root);
        final_value
    }

    fn negative_suffix_root(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            if !is_negative_label(label) {
                memo.insert((bundle_state, target_root), None);
                return None;
            }
            let parser_state = negative_to_positive_label(label) as u32;
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let Some(child) = self.negative_suffix_root(
                    bundle,
                    *target,
                    target_root,
                    memo,
                ) else {
                    memo.insert((bundle_state, target_root), None);
                    return None;
                };
                let residual = self.advance(child, parser_state);
                result = self.union(result, residual);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.negative_suffix_root(bundle, *target, target_root, memo) else {
                memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    fn preimage_state(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        positive_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
        negative_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = positive_memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let branch = if is_negative_label(label) {
                    let parser_state = negative_to_positive_label(label) as u32;
                    let Some(child) = self.negative_suffix_root(
                        bundle,
                        *target,
                        target_root,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.advance(child, parser_state)
                } else {
                    let Some(child) = self.preimage_state(
                        bundle,
                        *target,
                        target_root,
                        positive_memo,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.read(label, child)
                };
                result = self.union(result, branch);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.preimage_state(
                bundle,
                *target,
                target_root,
                positive_memo,
                negative_memo,
            ) else {
                positive_memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        positive_memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    pub fn preimage_bundle(&mut self, bundle: &NWA, target_root: u32) -> Option<u32> {
        let mut positive_memo = FxHashMap::default();
        let mut negative_memo = FxHashMap::default();
        let mut result = Self::EMPTY;
        for &start in bundle.start_states() {
            let root = self.preimage_state(
                bundle,
                start,
                target_root,
                &mut positive_memo,
                &mut negative_memo,
            )?;
            result = self.union(result, root);
        }
        if std::env::var_os("GLRMASK_EXPERIMENT_FINALIZE_LAZY_PREIMAGE").is_some() {
            self.finalize_prefix_domain(result);
        }
        Some(result)
    }

    /// Export several support-weighted roots as one shared positive NWA.
    /// Expression nodes reachable from multiple roots are materialized once;
    /// only the fresh global start carries support weights. This keeps the
    /// lazy DAG's structural sharing intact through the final weighted parser
    /// normalization.
    pub fn to_weighted_nwa(&self, roots: &[(u32, Weight)]) -> NWA {
        let mut reachable = FxHashSet::<u32>::default();
        let mut stack = roots
            .iter()
            .filter_map(|(root, weight)| (!weight.is_empty()).then_some(*root))
            .collect::<Vec<_>>();
        while let Some(node) = stack.pop() {
            if !reachable.insert(node) {
                continue;
            }
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty | LazyBooleanDomainExpr::Universal => {}
                LazyBooleanDomainExpr::Read { child, .. } => stack.push(child),
                LazyBooleanDomainExpr::Union { left, right } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        let mut ordered = reachable.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable();
        let mut remap = FxHashMap::<u32, u32>::default();
        let mut nwa = NWA::new(0, 0);
        let global_start = nwa.add_state();
        for node in &ordered {
            remap.insert(*node, nwa.add_state());
        }
        nwa.set_start_states(vec![global_start]);
        for (root, weight) in roots {
            if weight.is_empty() || *root == Self::EMPTY {
                continue;
            }
            if let Some(&target) = remap.get(root) {
                nwa.add_epsilon(global_start, target, weight.clone());
            }
        }
        for node in ordered {
            let from = remap[&node];
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty => {}
                LazyBooleanDomainExpr::Universal => nwa.set_final_weight(from, Weight::all()),
                LazyBooleanDomainExpr::Read { label, child } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_transition(from, label, remap[&child], Weight::all());
                }
                LazyBooleanDomainExpr::Union { left, right } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_epsilon(from, remap[&left], Weight::all());
                    nwa.add_epsilon(from, remap[&right], Weight::all());
                }
            }
        }
        nwa
    }

    pub fn to_nwa(&self, root: u32) -> NWA {
        let mut reachable = FxHashSet::<u32>::default();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            if !reachable.insert(node) {
                continue;
            }
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty | LazyBooleanDomainExpr::Universal => {}
                LazyBooleanDomainExpr::Read { child, .. } => stack.push(child),
                LazyBooleanDomainExpr::Union { left, right } => {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        let mut ordered = reachable.into_iter().collect::<Vec<_>>();
        ordered.sort_unstable();
        let mut remap = FxHashMap::<u32, u32>::default();
        let mut nwa = NWA::new(0, 0);
        for node in &ordered {
            remap.insert(*node, nwa.add_state());
        }
        nwa.set_start_states(vec![remap[&root]]);
        for node in ordered {
            let from = remap[&node];
            match self.nodes[node as usize] {
                LazyBooleanDomainExpr::Empty => {}
                LazyBooleanDomainExpr::Universal => nwa.set_final_weight(from, Weight::all()),
                LazyBooleanDomainExpr::Read { label, child } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_transition(from, label, remap[&child], Weight::all());
                }
                LazyBooleanDomainExpr::Union { left, right } => {
                    if self.prefix_final[node as usize] {
                        nwa.set_final_weight(from, Weight::all());
                    }
                    nwa.add_epsilon(from, remap[&left], Weight::all());
                    nwa.add_epsilon(from, remap[&right], Weight::all());
                }
            }
        }
        nwa
    }
}


#[derive(Clone)]
struct SharedBooleanDomainNode {
    explicit: BTreeMap<i32, u32>,
    default: Option<u32>,
    accepting: bool,
}

/// Canonical shared DAG for boolean parser-stack prefix predicates.
///
/// Node 0 is the empty language and node 1 is the universal/already-accepted
/// prefix language. All other nodes are hash-consed deterministic rows over
/// parser-state labels plus DEFAULT fallback. This is intentionally a compile-
/// time representation: exported parser DWAs remain ordinary runtime DWAs.
pub struct SharedBooleanParserDomains {
    nodes: Vec<SharedBooleanDomainNode>,
    interner: BTreeMap<(bool, Vec<(i32, u32)>, Option<u32>), u32>,
    union_memo: FxHashMap<(u32, u32), u32>,
    derivative_rows: Vec<Option<Arc<Vec<(i32, u32)>>>>,
    prefix_finality_memo: FxHashMap<u32, u32>,
}

impl Default for SharedBooleanParserDomains {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedBooleanParserDomains {
    pub const EMPTY: u32 = 0;
    pub const UNIVERSAL: u32 = 1;

    pub fn new() -> Self {
        let empty = SharedBooleanDomainNode {
            explicit: BTreeMap::new(),
            default: None,
            accepting: false,
        };
        let universal = SharedBooleanDomainNode {
            explicit: BTreeMap::new(),
            default: None,
            accepting: true,
        };
        let mut interner = BTreeMap::new();
        interner.insert((false, Vec::new(), None), Self::EMPTY);
        interner.insert((true, Vec::new(), None), Self::UNIVERSAL);
        Self {
            nodes: vec![empty, universal],
            interner,
            union_memo: FxHashMap::default(),
            derivative_rows: vec![Some(Arc::new(Vec::new())), Some(Arc::new(Vec::new()))],
            prefix_finality_memo: FxHashMap::default(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty_root(&self, root: u32) -> bool {
        root == Self::EMPTY
    }

    pub fn is_universal_root(&self, root: u32) -> bool {
        root == Self::UNIVERSAL
    }

    pub fn explicit_labels(&self, root: u32) -> Vec<i32> {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return Vec::new();
        }
        self.nodes[root as usize].explicit.keys().copied().collect()
    }

    pub fn advance_default(&self, root: u32) -> u32 {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return root;
        }
        self.nodes[root as usize].default.unwrap_or(Self::EMPTY)
    }

    fn make_node(&mut self, mut explicit: BTreeMap<i32, u32>, default: Option<u32>) -> u32 {
        let default = default.filter(|&target| target != Self::EMPTY);
        // DEFAULT remains an additive wildcard branch in this compile-time
        // representation, matching the positive NWA before parser fallback
        // normalization.  Prefix finality still projects backward through a
        // DEFAULT edge.
        if default == Some(Self::UNIVERSAL)
            && std::env::var_os("GLRMASK_EXPERIMENT_DEFER_SHARED_DOMAIN_DEFAULT_FINALITY").is_none()
        {
            return Self::UNIVERSAL;
        }
        explicit.retain(|_, target| {
            if *target == Self::EMPTY {
                return false;
            }
            if Some(*target) == default {
                return false;
            }
            true
        });
        if explicit.is_empty() && default.is_none() {
            return Self::EMPTY;
        }
        let row = explicit.iter().map(|(&label, &target)| (label, target)).collect::<Vec<_>>();
        let key = (false, row, default);
        if let Some(&existing) = self.interner.get(&key) {
            return existing;
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(SharedBooleanDomainNode {
            explicit,
            default,
            accepting: false,
        });
        self.derivative_rows.push(None);
        self.interner.insert(key, id);
        id
    }

    pub fn union(&mut self, left: u32, right: u32) -> u32 {
        if left == right || right == Self::EMPTY {
            return left;
        }
        if left == Self::EMPTY {
            return right;
        }
        if left == Self::UNIVERSAL || right == Self::UNIVERSAL {
            return Self::UNIVERSAL;
        }
        let key = if left < right { (left, right) } else { (right, left) };
        if let Some(&cached) = self.union_memo.get(&key) {
            return cached;
        }
        let left_node = self.nodes[left as usize].clone();
        let right_node = self.nodes[right as usize].clone();
        debug_assert!(!left_node.accepting && !right_node.accepting);

        let default = match (left_node.default, right_node.default) {
            (Some(left), Some(right)) => Some(self.union(left, right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        };
        let mut labels = BTreeSet::new();
        labels.extend(left_node.explicit.keys().copied());
        labels.extend(right_node.explicit.keys().copied());
        let mut explicit = BTreeMap::new();
        for label in labels {
            // Do not fold DEFAULT into explicit labels here.  DEFAULT is a
            // symbolic wildcard NWA branch, so a concrete derivative unions
            // the matching explicit branch with the wildcard branch later.
            let left_child = left_node.explicit.get(&label).copied();
            let right_child = right_node.explicit.get(&label).copied();
            let child = match (left_child, right_child) {
                (Some(left), Some(right)) => self.union(left, right),
                (Some(left), None) => left,
                (None, Some(right)) => right,
                (None, None) => Self::EMPTY,
            };
            explicit.insert(label, child);
        }
        let result = self.make_node(explicit, default);
        self.union_memo.insert(key, result);
        result
    }

    pub fn union_all(&mut self, roots: impl IntoIterator<Item = u32>) -> u32 {
        roots
            .into_iter()
            .fold(Self::EMPTY, |combined, root| self.union(combined, root))
    }

    /// Concrete derivatives for every explicit positive label in a shared
    /// domain row. Each derivative already includes the row's additive DEFAULT
    /// branch, so callers can apply the returned entries as sparse overrides on
    /// top of `advance_default(root)`.
    pub fn explicit_derivatives(&mut self, root: u32) -> Arc<Vec<(i32, u32)>> {
        if let Some(cached) = self
            .derivative_rows
            .get(root as usize)
            .and_then(|slot| slot.as_ref())
        {
            return Arc::clone(cached);
        }
        let node = self.nodes[root as usize].clone();
        let wildcard = node.default.unwrap_or(Self::EMPTY);
        let mut row = Vec::with_capacity(node.explicit.len());
        for (label, explicit) in node.explicit {
            let child = self.union(explicit, wildcard);
            if child != Self::EMPTY {
                row.push((label, child));
            }
        }
        let row = Arc::new(row);
        self.derivative_rows[root as usize] = Some(Arc::clone(&row));
        row
    }

    pub fn advance(&mut self, root: u32, parser_state: u32) -> u32 {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return root;
        }
        let row = self.explicit_derivatives(root);
        match row.binary_search_by_key(&(parser_state as i32), |(label, _)| *label) {
            Ok(index) => row[index].1,
            Err(_) => self.advance_default(root),
        }
    }

    pub fn derivative_row_cache_len(&self) -> usize {
        self.derivative_rows.iter().filter(|row| row.is_some()).count()
    }

    /// Exact preimage through one or more grammar terminals whose only
    /// GLR action is `Skip`: the parser stack is unchanged, but the current
    /// top state must be one where at least one of those terminals is legal.
    pub fn preimage_identity_skip(&mut self, target_root: u32, allowed_states: &[u32]) -> u32 {
        if target_root == Self::EMPTY || allowed_states.is_empty() {
            return Self::EMPTY;
        }
        let mut explicit = BTreeMap::new();
        for &parser_state in allowed_states {
            let child = self.advance(target_root, parser_state);
            if child != Self::EMPTY {
                explicit.insert(parser_state as i32, child);
            }
        }
        self.make_node(explicit, None)
    }

    /// Apply parser-stack prefix finality after a complete relational
    /// preimage has been assembled. Finality propagates through DEFAULT stack
    /// reads (negative/push edges have already been cancelled algebraically),
    /// but never backward through concrete positive reads.
    ///
    /// This must not run while individual template fragments are still being
    /// composed: doing so can erase positive-read provenance before the full
    /// stack effect is known. That was the failure mode of the old eager
    /// `DEFAULT -> UNIVERSAL` simplification in `make_node`.
    pub fn finalize_prefix_domain(&mut self, root: u32) -> u32 {
        if root == Self::EMPTY || root == Self::UNIVERSAL {
            return root;
        }
        if let Some(&cached) = self.prefix_finality_memo.get(&root) {
            return cached;
        }
        let node = self.nodes[root as usize].clone();
        let mut explicit = BTreeMap::new();
        for (label, child) in node.explicit {
            let child = self.finalize_prefix_domain(child);
            if child != Self::EMPTY {
                explicit.insert(label, child);
            }
        }
        let default = node.default.map(|child| self.finalize_prefix_domain(child));
        let result = if default == Some(Self::UNIVERSAL) {
            // A DEFAULT edge can consume an arbitrary deeper parser-stack state.
            // If its target has already accepted the visible prefix, the source
            // itself is prefix-final; from this residual point every deeper
            // suffix is irrelevant.
            Self::UNIVERSAL
        } else {
            self.make_node(explicit, default)
        };
        self.prefix_finality_memo.insert(root, result);
        result
    }

    fn prepend(&mut self, label: i32, child: u32) -> u32 {
        if child == Self::EMPTY {
            return Self::EMPTY;
        }
        if label == DEFAULT_LABEL {
            return self.make_node(BTreeMap::new(), Some(child));
        }
        let mut explicit = BTreeMap::new();
        explicit.insert(label, child);
        self.make_node(explicit, None)
    }

    fn negative_suffix_root(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            if !is_negative_label(label) {
                memo.insert((bundle_state, target_root), None);
                return None;
            }
            let parser_state = negative_to_positive_label(label) as u32;
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let Some(child) = self.negative_suffix_root(
                    bundle,
                    *target,
                    target_root,
                    memo,
                ) else {
                    memo.insert((bundle_state, target_root), None);
                    return None;
                };
                let residual = self.advance(child, parser_state);
                result = self.union(result, residual);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.negative_suffix_root(bundle, *target, target_root, memo) else {
                memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    fn preimage_state(
        &mut self,
        bundle: &NWA,
        bundle_state: u32,
        target_root: u32,
        positive_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
        negative_memo: &mut FxHashMap<(u32, u32), Option<u32>>,
    ) -> Option<u32> {
        if let Some(cached) = positive_memo.get(&(bundle_state, target_root)) {
            return *cached;
        }
        let node = bundle.states().get(bundle_state as usize)?;
        let mut result = if node
            .final_weight
            .as_ref()
            .is_some_and(|weight| !weight.is_empty())
        {
            target_root
        } else {
            Self::EMPTY
        };
        for (&label, targets) in &node.transitions {
            for (target, weight) in targets {
                debug_assert!(weight.is_full() || weight.is_empty());
                if weight.is_empty() {
                    continue;
                }
                let branch = if is_negative_label(label) {
                    let parser_state = negative_to_positive_label(label) as u32;
                    let Some(child) = self.negative_suffix_root(
                        bundle,
                        *target,
                        target_root,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.advance(child, parser_state)
                } else {
                    let Some(child) = self.preimage_state(
                        bundle,
                        *target,
                        target_root,
                        positive_memo,
                        negative_memo,
                    ) else {
                        positive_memo.insert((bundle_state, target_root), None);
                        return None;
                    };
                    self.prepend(label, child)
                };
                result = self.union(result, branch);
            }
        }
        for (target, weight) in &node.epsilons {
            debug_assert!(weight.is_full() || weight.is_empty());
            if weight.is_empty() {
                continue;
            }
            let Some(child) = self.preimage_state(
                bundle,
                *target,
                target_root,
                positive_memo,
                negative_memo,
            ) else {
                positive_memo.insert((bundle_state, target_root), None);
                return None;
            };
            result = self.union(result, child);
        }
        positive_memo.insert((bundle_state, target_root), Some(result));
        Some(result)
    }

    pub fn preimage_bundle(&mut self, bundle: &NWA, target_root: u32) -> Option<u32> {
        let mut positive_memo = FxHashMap::default();
        let mut negative_memo = FxHashMap::default();
        let mut result = Self::EMPTY;
        for &start in bundle.start_states() {
            let root = self.preimage_state(
                bundle,
                start,
                target_root,
                &mut positive_memo,
                &mut negative_memo,
            )?;
            result = self.union(result, root);
        }
        if std::env::var_os("GLRMASK_EXPERIMENT_FINALIZE_SHARED_PREIMAGE").is_some() {
            result = self.finalize_prefix_domain(result);
        }
        Some(result)
    }

    pub fn to_dwa(&self, root: u32) -> DWA {
        let mut output = DWA::new(0, 0);
        let mut remap = FxHashMap::<u32, u32>::default();
        remap.insert(root, output.start_state());
        let mut queue = VecDeque::from([root]);
        while let Some(source) = queue.pop_front() {
            let output_source = remap[&source];
            let node = &self.nodes[source as usize];
            if node.accepting {
                output.set_final_weight(output_source, Weight::all());
            }
            for (&label, &target) in &node.explicit {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(output_source, label, output_target, Weight::all());
            }
            if let Some(target) = node.default {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(
                    output_source,
                    DEFAULT_LABEL,
                    output_target,
                    Weight::all(),
                );
            }
        }
        output
    }

    /// Export the shared compile-time row DAG without interpreting DEFAULT as
    /// deterministic fallback.  In this representation DEFAULT is an additive
    /// wildcard NWA branch; parser-specific support/fallback normalization is
    /// deliberately deferred until after the complete graph is assembled.
    pub fn to_nwa(&self, root: u32) -> NWA {
        let mut output = NWA::new(0, 0);
        let start = output.add_state();
        output.set_start_states(vec![start]);
        let mut remap = FxHashMap::<u32, u32>::default();
        remap.insert(root, start);
        let mut queue = VecDeque::from([root]);
        while let Some(source) = queue.pop_front() {
            let output_source = remap[&source];
            let node = &self.nodes[source as usize];
            if node.accepting {
                output.set_final_weight(output_source, Weight::all());
            }
            for (&label, &target) in &node.explicit {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(output_source, label, output_target, Weight::all());
            }
            if let Some(target) = node.default {
                let output_target = if let Some(&existing) = remap.get(&target) {
                    existing
                } else {
                    let created = output.add_state();
                    remap.insert(target, created);
                    queue.push_back(target);
                    created
                };
                output.add_transition(
                    output_source,
                    DEFAULT_LABEL,
                    output_target,
                    Weight::all(),
                );
            }
        }
        output
    }
}


pub fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    terminal_dwa: &TerminalAutomaton,
    templates: &Templates,
    _vocab: &Vocab,
    _id_map: &InternalIdMap,
    collapse_immediate_acceptance: bool,
) -> DWA {
    let num_parser_states = table.num_states;
    let total_started_at = Instant::now();
    let minimize_skipped = false;
    let profiling_enabled = compile_profile_enabled();
    let (terminal_dwa_transition_count, terminal_dwa_interned_ranges) = if profiling_enabled {
        let stats = terminal_dwa.stats();
        (stats.transitions, stats.interned_ranges)
    } else {
        (0, 0)
    };
    let Some((mut parser_nwa, parser_nwa_profile)) = build_parser_nwa_from_terminal_dwa(terminal_dwa, grammar, templates) else {
        if profiling_enabled {
            eprintln!(
                "[glrmask/profile][parser_dwa_detail] terminal_dwa_states={} terminal_dwa_transitions={} terminal_dwa_interned_ranges={} parser_nwa_built=false pre_minimize_states=0 pre_minimize_transitions=0 post_minimize_states=0 post_minimize_transitions=0 minimize_skipped={} state_prep_ms=0.000 compose_state_ms=0.000 parser_nwa_build_ms=0.000 resolve_negative_ms=0.000 support_determinize_ms=0.000 possible_outgoing_ms=0.000 default_opt_ms=0.000 subtract_final_ms=0.000 fallback_determinize_ms=0.000 minimize_ms=0.000 total_ms={:.3}",
                terminal_dwa.num_states(),
                terminal_dwa_transition_count,
                terminal_dwa_interned_ranges,
                minimize_skipped,
                elapsed_ms(total_started_at),
            );
        }
        return DWA::new(0, 0);
    };

    let resolve_negative_started_at = Instant::now();
    resolve_negative_codes_in_nwa(
        &mut parser_nwa,
        table.construction == GlrTableConstruction::ExperimentalCoreMerged,
    );
    let resolve_negative_ms = elapsed_ms(resolve_negative_started_at);

    let support_determinize_started_at = Instant::now();
    let determinized = determinize_with_supports(&parser_nwa, Some(num_parser_states));
    let support_determinize_ms = elapsed_ms(support_determinize_started_at);
    let mut parser_dwa_pre_minimize = determinized.dwa;

    let guaranteed_read_started_at = Instant::now();
    let immediate_read_rewrites = if collapse_immediate_acceptance {
        collapse_immediate_acceptance_certificates(
            &mut parser_dwa_pre_minimize,
            terminal_dwa,
            grammar,
            table,
        )
    } else {
        0
    };
    let guaranteed_read_rewrites = immediate_read_rewrites;
    let guaranteed_read_ms = elapsed_ms(guaranteed_read_started_at);

    let possible_outgoing_started_at = Instant::now();
    let possible_by_state = build_possible_outgoing_ids_by_state(
        &parser_nwa,
        &determinized.supports,
        num_parser_states,
    );
    let possible_outgoing_ms = elapsed_ms(possible_outgoing_started_at);

    let default_opt_started_at = Instant::now();
    optimize_parser_dwa_defaults(
        &mut parser_dwa_pre_minimize,
        &possible_by_state,
        num_parser_states,
    );
    let default_opt_ms = elapsed_ms(default_opt_started_at);

    let subtract_final_started_at = Instant::now();
    let validate_parallel_subtraction =
        std::env::var_os("GLRMASK_VALIDATE_PARALLEL_FINAL_SUBTRACTION").is_some();
    let serial_reference = validate_parallel_subtraction.then(|| parser_dwa_pre_minimize.clone());
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize);
    if let Some(mut serial_reference) = serial_reference {
        subtract_final_weights_from_outgoing_dwa_impl(&mut serial_reference, false);
        assert_eq!(
            parser_dwa_pre_minimize.start_state(),
            serial_reference.start_state(),
            "parallel final subtraction changed the DWA start state",
        );
        assert_eq!(
            parser_dwa_pre_minimize.states().len(),
            serial_reference.states().len(),
            "parallel final subtraction changed the DWA state count",
        );
        for (parallel_state, serial_state) in parser_dwa_pre_minimize
            .states()
            .iter()
            .zip(serial_reference.states())
        {
            assert_eq!(
                parallel_state.final_weight, serial_state.final_weight,
                "parallel final subtraction changed a final weight",
            );
            assert_eq!(
                parallel_state.transitions, serial_state.transitions,
                "parallel final subtraction changed a transition row",
            );
        }
    }
    let subtract_final_ms = elapsed_ms(subtract_final_started_at);

    let fallback_determinize_started_at = Instant::now();
    let normalized_fallback_started_at = Instant::now();
    let normalized_fallback = determinize_parser_dwa_with_fallbacks(
        &parser_dwa_pre_minimize,
        &possible_by_state,
        num_parser_states,
    );
    let normalized_fallback_ms = elapsed_ms(normalized_fallback_started_at);
    if std::env::var_os("GLRMASK_VALIDATE_FALLBACK_SINGLETON_NORMALIZATION").is_some() {
        let reference_started_at = Instant::now();
        let reference = determinize_parser_dwa_with_fallbacks_impl(
            &parser_dwa_pre_minimize,
            &possible_by_state,
            num_parser_states,
            false,
        );
        let reference_ms = elapsed_ms(reference_started_at);
        let equivalence_started_at = Instant::now();
        let difference = find_difference(&normalized_fallback, &reference)
            .expect("fallback singleton validation requires acyclic parser DWAs");
        let equivalence_ms = elapsed_ms(equivalence_started_at);
        assert!(
            difference.is_none(),
            "normalized parser fallback DWA differs from the legacy weighted-singleton DWA on labels {:?}",
            difference,
        );
        if profiling_enabled {
            eprintln!(
                "[glrmask/profile][fallback_singleton_normalization] normalized_states={} normalized_transitions={} reference_states={} reference_transitions={} normalized_ms={normalized_fallback_ms:.3} reference_ms={reference_ms:.3} equivalence_ms={equivalence_ms:.3} result=equivalent",
                normalized_fallback.num_states(),
                normalized_fallback.num_transitions(),
                reference.num_states(),
                reference.num_transitions(),
            );
        }
    }
    parser_dwa_pre_minimize = normalized_fallback;
    if collapse_immediate_acceptance {
        parser_dwa_pre_minimize = collapse_final_leaf_targets(parser_dwa_pre_minimize);
    }
    let fallback_determinize_ms = elapsed_ms(fallback_determinize_started_at);

    let pre_minimize_state_count = parser_dwa_pre_minimize.states().len();
    let pre_minimize_transition_count = parser_dwa_pre_minimize.num_transitions();
    let minimize_skipped = should_skip_parser_dwa_minimization(
        pre_minimize_state_count,
        pre_minimize_transition_count,
    );
    let (minimized, minimize_ms, post_minimize_state_count, post_minimize_transition_count) =
        if minimize_skipped {
            (
                parser_dwa_pre_minimize,
                0.0,
                pre_minimize_state_count,
                pre_minimize_transition_count,
            )
        } else {
            let minimize_started_at = Instant::now();
            let minimized = minimize(&parser_dwa_pre_minimize);
            let minimize_ms = elapsed_ms(minimize_started_at);
            let post_minimize_state_count = minimized.states().len();
            let post_minimize_transition_count = minimized.num_transitions();
            (
                minimized,
                minimize_ms,
                post_minimize_state_count,
                post_minimize_transition_count,
            )
        };

    if profiling_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa_detail] terminal_dwa_states={} terminal_dwa_transitions={} terminal_dwa_interned_ranges={} parser_nwa_states={} parser_nwa_start_states={} pre_minimize_states={} pre_minimize_transitions={} post_minimize_states={} post_minimize_transitions={} minimize_skipped={} state_prep_ms={:.3} compose_state_ms={:.3} parser_nwa_build_ms={:.3} resolve_negative_ms={:.3} support_determinize_ms={:.3} guaranteed_read_rewrites={} guaranteed_read_ms={:.3} possible_outgoing_ms={:.3} default_opt_ms={:.3} subtract_final_ms={:.3} fallback_determinize_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            terminal_dwa.num_states(),
            terminal_dwa_transition_count,
            terminal_dwa_interned_ranges,
            parser_nwa.states().len(),
            parser_nwa.start_states().len(),
            pre_minimize_state_count,
            pre_minimize_transition_count,
            post_minimize_state_count,
            post_minimize_transition_count,
            minimize_skipped,
            parser_nwa_profile.state_prep_ms,
            parser_nwa_profile.compose_state_ms,
            parser_nwa_profile.parser_nwa_build_ms,
            resolve_negative_ms,
            support_determinize_ms,
            guaranteed_read_rewrites,
            guaranteed_read_ms,
            possible_outgoing_ms,
            default_opt_ms,
            subtract_final_ms,
            fallback_determinize_ms,
            minimize_ms,
            elapsed_ms(total_started_at),
        );
    }

    minimized
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use range_set_blaze::RangeSetBlaze;
    use rustc_hash::FxHashMap;

    use super::{
        PossibleOutgoingIds, build_parser_nwa_from_terminal_dwa,
        collapse_final_leaf_targets, determinize_parser_dwa_with_fallbacks,
        determinize_with_supports, immediate_acceptance_certificates,
        local_epsilon_closure, local_epsilon_closure_canonical,
        try_build_direct_regular_parser_top_accept_parts,
        try_build_direct_regular_parser_top_accept_parts_table_product_reference,
        try_build_immediate_parser_top_accept_parts,
        subtract_final_weights_from_outgoing_dwa_global_pairs,
        subtract_final_weights_from_outgoing_dwa_impl,
    };
    use crate::automata::weighted::dwa::DWA;
    use crate::automata::weighted::nwa::NWA;
    use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::glr::labels::DEFAULT_LABEL;
    use crate::compiler::glr::table::testing::build_test_table;
    use crate::compiler::glr::table::Action;
    use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
    use crate::compiler::stages::templates::Templates;
    use crate::ds::weight::Weight;
    use crate::grammar::flat::{
        DirectRegularAutomaton, GrammarDef, Rule, Symbol, Terminal,
    };

    fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
    }

    fn eval_with_default(dwa: &DWA, word: &[i32]) -> Weight {
        let mut state_id = dwa.start_state();
        let mut accumulated = Weight::all();
        for &label in word {
            let Some((target, edge_weight)) = dwa.states()[state_id as usize]
                .transitions
                .get(&label)
                .or_else(|| dwa.states()[state_id as usize].transitions.get(&DEFAULT_LABEL))
            else {
                return Weight::empty();
            };
            accumulated = accumulated.intersection(edge_weight);
            if accumulated.is_empty() {
                return accumulated;
            }
            state_id = *target;
        }
        dwa.states()[state_id as usize]
            .final_weight
            .as_ref()
            .map_or_else(Weight::empty, |final_weight| {
                accumulated.intersection(final_weight)
            })
    }

    #[test]
    fn flat_canonical_epsilon_closure_matches_map_reference() {
        fn next_u32(state: &mut u64) -> u32 {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*state >> 32) as u32
        }

        const STATE_COUNT: usize = 64;
        let mut random = 0x8f4d_3a2b_1907_65ceu64;
        let mut nwa = NWA::new(0, 0);
        for _ in 0..STATE_COUNT {
            nwa.add_state();
        }
        for source in 0..STATE_COUNT - 1 {
            let remaining = STATE_COUNT - source - 1;
            let edge_count = 1 + next_u32(&mut random) as usize % remaining.min(4);
            for _ in 0..edge_count {
                let target = source + 1 + next_u32(&mut random) as usize % remaining;
                let start = next_u32(&mut random) % 24;
                let end = (start + next_u32(&mut random) % 8).min(31);
                nwa.add_epsilon(source as u32, target as u32, weight(start..=end));
            }
        }

        for case in 0..256 {
            let mut seeds = FxHashMap::<u32, Weight>::default();
            let seed_count = 1 + next_u32(&mut random) as usize % 8;
            for _ in 0..seed_count {
                let state = next_u32(&mut random) as usize % STATE_COUNT;
                let start = next_u32(&mut random) % 24;
                let end = (start + next_u32(&mut random) % 8).min(31);
                let add = weight(start..=end);
                seeds
                    .entry(state as u32)
                    .and_modify(|existing| *existing = existing.union(&add))
                    .or_insert(add);
            }

            let mut reference = seeds.clone();
            let mut reference_weights = vec![None; STATE_COUNT];
            let mut reference_queue = VecDeque::new();
            local_epsilon_closure(
                &nwa,
                &mut reference_weights,
                &mut reference_queue,
                &mut reference,
            );
            let mut reference_canonical = reference.into_iter().collect::<Vec<_>>();
            reference_canonical.sort_unstable_by_key(|(state, _)| *state);

            let mut seed_canonical = seeds.into_iter().collect::<Vec<_>>();
            seed_canonical.sort_unstable_by_key(|(state, _)| *state);
            let mut flat_weights = vec![None; STATE_COUNT];
            let mut flat_queue = VecDeque::new();
            let mut touched = Vec::new();
            let mut flat_canonical = Vec::new();
            local_epsilon_closure_canonical(
                &nwa,
                &mut flat_weights,
                &mut flat_queue,
                &seed_canonical,
                &mut touched,
                &mut flat_canonical,
                &mut super::ScopedWeightOpCache::default(),
                true,
            );

            assert_eq!(
                flat_canonical, reference_canonical,
                "epsilon-closure mismatch in generated case {case}",
            );
            assert!(flat_weights.iter().all(Option::is_none));
            assert!(flat_queue.is_empty());
        }
    }

    #[test]
    fn immediate_acceptance_parts_union_matches_combined_certificates() {
        let mut terminal_dwa = DWA::new(1, 31);
        let accept_a = terminal_dwa.add_state();
        let accept_b = terminal_dwa.add_state();
        let accept_c = terminal_dwa.add_state();
        terminal_dwa.set_final_weight(accept_a, Weight::all());
        terminal_dwa.set_final_weight(accept_b, Weight::all());
        terminal_dwa.set_final_weight(accept_c, Weight::all());
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            0,
            accept_a,
            weight(0..=7),
        );
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            1,
            accept_b,
            weight(4..=15),
        );
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            2,
            accept_c,
            weight(12..=23),
        );
        let terminal_automaton = TerminalAutomaton::Dwa(terminal_dwa);

        let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: (0..3)
                .map(|id| Terminal::Literal {
                    id,
                    bytes: vec![b'a' + id as u8],
                })
                .collect(),
            ..GrammarDef::default()
        });
        let table = build_test_table(
            3,
            3,
            &[
                &[
                    (0, Action::Shift(0, true)),
                    (1, Action::Shift(1, true)),
                ],
                &[
                    (1, Action::Shift(1, true)),
                    (2, Action::Shift(2, true)),
                ],
                &[],
            ],
            &[&[], &[], &[]],
        );

        let parts = try_build_immediate_parser_top_accept_parts(
            &terminal_automaton,
            &grammar,
            &table,
        )
        .expect("test terminal automaton is an immediate-completion family");
        let combined = immediate_acceptance_certificates(&terminal_automaton, &grammar, &table);

        assert_eq!(parts.get(&0).map(Vec::len), Some(2));
        assert_eq!(parts.get(&1).map(Vec::len), Some(2));
        assert!(!parts.contains_key(&2));
        for parser_top in 0..table.num_states {
            let parts_union = parts
                .get(&(parser_top as i32))
                .map(|weights| Weight::union_all(weights.iter()))
                .unwrap_or_else(Weight::empty);
            assert_eq!(parts_union, combined[parser_top as usize]);
        }
    }

    #[test]
    fn direct_regular_parser_product_matches_generic_parser_dwa() {
        let mut terminal_dwa = DWA::new(1, 31);
        let after_zero = terminal_dwa.add_state();
        let accept = terminal_dwa.add_state();
        terminal_dwa.set_final_weight(after_zero, weight(0..=7));
        terminal_dwa.set_final_weight(accept, Weight::all());
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            0,
            after_zero,
            weight(0..=15),
        );
        terminal_dwa.add_transition(after_zero, 0, after_zero, weight(8..=12));
        terminal_dwa.add_transition(after_zero, 1, accept, weight(4..=20));
        terminal_dwa.add_transition(
            terminal_dwa.start_state(),
            2,
            accept,
            weight(16..=23),
        );
        let terminal_automaton = TerminalAutomaton::Dwa(terminal_dwa);

        let grammar = AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            start: 0,
            terminals: (0..3)
                .map(|id| Terminal::Literal {
                    id,
                    bytes: vec![b'a' + id as u8],
                })
                .collect(),
            direct_regular_automaton: Some(DirectRegularAutomaton {
                states: vec![
                    crate::grammar::flat::DirectRegularState {
                        is_accepting: false,
                        transitions: [
                            (0, vec![0]),
                            (1, vec![1]),
                            (2, vec![1]),
                        ]
                        .into_iter()
                        .collect(),
                        epsilons: Vec::new(),
                    },
                    crate::grammar::flat::DirectRegularState {
                        is_accepting: true,
                        transitions: Default::default(),
                        epsilons: Vec::new(),
                    },
                ],
                start_states: vec![0],
            }),
            ..GrammarDef::default()
        });
        let table = build_test_table(
            3,
            3,
            &[
                &[
                    (0, Action::Shift(1, true)),
                    (1, Action::Shift(2, true)),
                    (2, Action::Shift(2, true)),
                ],
                &[
                    (0, Action::Shift(1, true)),
                    (1, Action::Shift(2, true)),
                    (2, Action::Shift(2, true)),
                ],
                &[],
            ],
            &[&[], &[], &[]],
        );
        let templates = Templates::from_direct_regular_table(&table, grammar.num_terminals)
            .expect("test table has direct-regular actions");
        let (mut generic_nwa, _) = build_parser_nwa_from_terminal_dwa(
            &terminal_automaton,
            &grammar,
            &templates,
        )
        .expect("generic parser NWA should build for direct templates");
        resolve_negative_codes_in_nwa(&mut generic_nwa, false);
        let generic = determinize_with_supports(&generic_nwa, Some(table.num_states)).dwa;
        let direct = try_build_direct_regular_parser_top_accept_parts(
            &terminal_automaton,
            &grammar,
            &table,
        )
        .expect("sparse direct product should accept the direct parser metadata");
        let table_reference =
            try_build_direct_regular_parser_top_accept_parts_table_product_reference(
                &terminal_automaton,
                &grammar,
                &table,
            )
            .expect("table-product reference should accept the direct parser table");

        for parser_top in 0..table.num_states {
            let direct_weight = direct
                .get(&(parser_top as i32))
                .map(|weights| Weight::union_all(weights.iter()))
                .unwrap_or_else(Weight::empty);
            let reference_weight = table_reference
                .get(&(parser_top as i32))
                .map(|weights| Weight::union_all(weights.iter()))
                .unwrap_or_else(Weight::empty);
            assert_eq!(
                direct_weight,
                reference_weight,
                "sparse and table products differ at parser top {parser_top}",
            );
            let mut generic_prefix_weight = generic
                .states()
                .get(generic.start_state() as usize)
                .and_then(|state| state.final_weight.clone())
                .unwrap_or_else(Weight::empty);
            generic_prefix_weight = generic_prefix_weight.union(
                &generic.eval_word(&[parser_top as i32]),
            );
            assert_eq!(
                direct_weight,
                generic_prefix_weight,
                "direct product mismatch at parser top {parser_top}",
            );
        }
    }

    #[test]
    fn global_pair_final_subtraction_matches_serial_rows() {
        let mut source = DWA::new(3, 63);
        let states = (0..8).map(|_| source.add_state()).collect::<Vec<_>>();
        let finals = [
            weight(4..=17),
            weight(8..=21),
            weight(12..=27),
            weight(2..=13),
        ];
        let edges = [
            weight(0..=31),
            weight(8..=39),
            weight(16..=47),
            weight(24..=55),
        ];
        for state_id in 0..source.states().len() {
            if state_id % 2 == 0 {
                source.set_final_weight(state_id as u32, finals[state_id % finals.len()].clone());
            }
            for offset in 0..4usize {
                let target = states[(state_id + offset) % states.len()];
                source.add_transition(
                    state_id as u32,
                    (offset + 1) as i32,
                    target,
                    edges[(state_id + offset) % edges.len()].clone(),
                );
            }
        }

        let mut serial = source.clone();
        let mut global = source;
        subtract_final_weights_from_outgoing_dwa_impl(&mut serial, false);
        subtract_final_weights_from_outgoing_dwa_global_pairs(&mut global);
        assert_eq!(serial.start_state(), global.start_state());
        assert_eq!(serial.states(), global.states());
    }

    #[test]
    fn parallel_final_subtraction_matches_serial_rows() {
        let mut source = DWA::new(1, 31);
        let left = source.add_state();
        let right = source.add_state();
        source.set_final_weight(source.start_state(), weight(4..=11));
        source.add_transition(source.start_state(), 1, left, weight(0..=15));
        source.add_transition(source.start_state(), 2, right, weight(8..=20));
        source.set_final_weight(left, weight(0..=3));
        source.add_transition(left, 3, right, weight(0..=9));
        source.set_final_weight(right, weight(16..=23));
        source.add_transition(right, 4, left, weight(12..=27));

        let mut serial = source.clone();
        let mut parallel = source;
        subtract_final_weights_from_outgoing_dwa_impl(&mut serial, false);
        subtract_final_weights_from_outgoing_dwa_impl(&mut parallel, true);

        assert_eq!(serial.start_state(), parallel.start_state());
        assert_eq!(serial.states().len(), parallel.states().len());
        for (serial_state, parallel_state) in serial.states().iter().zip(parallel.states()) {
            assert_eq!(serial_state.final_weight, parallel_state.final_weight);
            assert_eq!(serial_state.transitions, parallel_state.transitions);
        }
    }

    #[test]
    fn fallback_determinization_reuses_singleton_source_states() {
        let mut source = DWA::new(1, 31);
        let middle = source.add_state();
        let leaf = source.add_state();
        source.add_transition(source.start_state(), 10, middle, weight(0..=15));
        source.add_transition(source.start_state(), 11, middle, weight(8..=23));
        source.set_final_weight(middle, weight(4..=19));
        source.add_transition(middle, 12, leaf, weight(2..=27));
        source.set_final_weight(leaf, weight(6..=25));

        let possible = (0..source.states().len())
            .map(|_| PossibleOutgoingIds::Empty)
            .collect::<Vec<_>>();
        let determinized = determinize_parser_dwa_with_fallbacks(&source, &possible, 32);

        for word in [
            vec![],
            vec![10],
            vec![11],
            vec![10, 12],
            vec![11, 12],
            vec![12],
        ] {
            assert_eq!(
                determinized.eval_word(&word),
                source.eval_word(&word),
                "word={word:?}",
            );
        }
        assert_eq!(determinized.states().len(), source.states().len());
    }

    #[test]
    fn fallback_determinization_combines_explicit_and_default_branches() {
        let mut source = DWA::new(1, 31);
        let explicit = source.add_state();
        let fallback = source.add_state();
        source.add_transition(source.start_state(), 7, explicit, weight(0..=15));
        source.add_transition(
            source.start_state(),
            DEFAULT_LABEL,
            fallback,
            weight(8..=23),
        );
        source.set_final_weight(explicit, weight(0..=5));
        source.set_final_weight(fallback, weight(12..=27));

        let possible = vec![
            PossibleOutgoingIds::All,
            PossibleOutgoingIds::Empty,
            PossibleOutgoingIds::Empty,
        ];
        let determinized = determinize_parser_dwa_with_fallbacks(&source, &possible, 32);

        let explicit_result = weight(0..=5);
        let fallback_result = weight(12..=23);
        assert_eq!(
            eval_with_default(&determinized, &[7]),
            explicit_result.union(&fallback_result),
        );
        assert_eq!(eval_with_default(&determinized, &[8]), fallback_result);
        assert_eq!(determinized.eval_word(&[DEFAULT_LABEL]), weight(12..=23));
    }

    #[test]
    fn final_leaf_weights_are_pushed_into_shared_sink_edges() {
        let mut dwa = DWA::new(1, 5);
        let left = dwa.add_state();
        let right = dwa.add_state();
        dwa.add_transition(0, 10, left, weight(0..=5));
        dwa.add_transition(0, 11, right, weight(0..=5));
        dwa.set_final_weight(left, weight(0..=2));
        dwa.set_final_weight(right, weight(3..=4));

        let collapsed = collapse_final_leaf_targets(dwa);

        assert_eq!(collapsed.states().len(), 2);
        assert_eq!(collapsed.eval_word(&[10]), weight(0..=2));
        assert_eq!(collapsed.eval_word(&[11]), weight(3..=4));
        let targets: Vec<u32> = collapsed.states()[collapsed.start_state() as usize]
            .transitions
            .values()
            .map(|(target, _)| *target)
            .collect();
        assert_eq!(targets.len(), 2);
        assert!(targets.iter().all(|target| *target == targets[0]));
        assert!(collapsed.states()[targets[0] as usize]
            .final_weight
            .as_ref()
            .is_some_and(Weight::is_full));
    }

    #[test]
    fn nonleaf_continuations_are_not_shortened() {
        let mut dwa = DWA::new(1, 5);
        let middle = dwa.add_state();
        let leaf = dwa.add_state();
        dwa.add_transition(0, 10, middle, weight(0..=5));
        dwa.add_transition(middle, 11, leaf, weight(0..=5));
        dwa.set_final_weight(leaf, weight(1..=3));

        let collapsed = collapse_final_leaf_targets(dwa);

        assert_eq!(collapsed.states().len(), 3);
        assert!(collapsed.eval_word(&[10]).is_empty());
        assert_eq!(collapsed.eval_word(&[10, 11]), weight(1..=3));
    }
}
