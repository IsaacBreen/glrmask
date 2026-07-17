use std::collections::{hash_map::Entry, BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::{Action, AdmissionPolicy, GLRTable};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::id_map_and_terminal_dwa::types::compile_profile_enabled;
use crate::compiler::stages::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::templates::Templates;
use crate::ds::bitset::BitSet;
use crate::ds::weight::{ScopedWeightOpCache, Weight};

type TerminalBundle = BTreeMap<TerminalID, Weight>;
type BundleSignature = Vec<(TerminalID, Weight)>;
type TargetContribs = SmallVec<[(u32, Weight); 4]>;

const PROFILE_PARSER_DWA_DETERMINIZE_DETAIL_ENV: &str =
    "GLRMASK_PROFILE_PARSER_DWA_DETERMINIZE_DETAIL";

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
struct CachedClosure {
    to_state: u32,
    edge_weight: Weight,
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

fn immediate_acceptance_certificates(
    terminal_automaton: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
) -> Vec<Weight> {
    if table.admission_policy != AdmissionPolicy::RowPresenceExact {
        return vec![Weight::empty(); table.num_states as usize];
    }
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

    let mut cache = FxHashMap::<Vec<TerminalID>, Weight>::default();
    table
        .action
        .iter()
        .map(|row| {
            let terminals: Vec<TerminalID> = row
                .iter()
                .filter_map(|(terminal, action)| {
                    (top_row_action_is_unconditionally_applicable(action)
                        && complete_by_terminal.contains_key(&terminal))
                    .then_some(terminal)
                })
                .collect();
            if terminals.is_empty() {
                return Weight::empty();
            }
            if let Some(weight) = cache.get(&terminals) {
                return weight.clone();
            }
            let weight = Weight::union_all(
                terminals
                    .iter()
                    .filter_map(|terminal| complete_by_terminal.get(terminal)),
            );
            cache.insert(terminals, weight.clone());
            weight
        })
        .collect()
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

pub(crate) fn try_build_immediate_parser_dwa(
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

fn determinize_with_supports(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
) -> DeterminizedDwaWithSupports {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

      #[derive(Default)]
      struct UnionAllCache {
        entries: FxHashMap<Vec<usize>, Weight>,
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

            let mut key: Vec<usize> = meaningful.iter().map(|weight| weight.ptr_key()).collect();
            key.sort_unstable();
            key.dedup();
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
            let weight = Weight::union_all(meaningful.into_iter());
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
    // Memoize local epsilon-closure outputs keyed by pre-closure weighted subsets.
    let mut closure_cache: FxHashMap<Vec<(u32, usize)>, CachedClosure> = FxHashMap::default();
    let mut key_buf: Vec<(u32, usize)> = Vec::new();
    let mut detail =
        ParserDwaDeterminizeDetail::enabled().then(ParserDwaDeterminizeDetail::default);
    let mut union_cache = UnionAllCache {
        profile_enabled: detail.is_some(),
        ..UnionAllCache::default()
    };

    // Deferred final weight computation: store subset entries for each DWA state
    // and compute final weights in parallel after the main loop.
    let mut deferred_final_entries: Vec<(u32, Vec<(u32, Weight)>)> = Vec::new();

    while let Some((from_state, subset_entries)) = worklist.pop_front() {
        if let Some(detail) = detail.as_mut() {
            detail.states_processed += 1;
        }

        // Save subset entries for deferred parallel final weight computation.
        // Only save entries whose NWA states have final weights.
        let has_finals: Vec<(u32, Weight)> = subset_entries.iter()
            .filter(|(nwa_state_id, _)| nwa.states()[*nwa_state_id as usize].final_weight.is_some())
            .map(|(id, w)| (*id, w.clone()))
            .collect();
        if !has_finals.is_empty() {
            deferred_final_entries.push((from_state, has_finals));
        }

        let scan_started = detail.as_ref().map(|_| Instant::now());
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states()[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
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
                dwa.add_transition(from_state, label, cached.to_state, cached.edge_weight);
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
            let edge_weight = union_cache.union_all(contribs.iter().map(|(_, weight)| weight));
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), edge_weight_started) {
                detail.edge_weight_union_ms += elapsed_ms(started_at);
            }
            if edge_weight.is_empty() {
                return;
            }
            let mut target_subset: FxHashMap<u32, Weight> = contribs
                .iter()
                .map(|(state_id, weight)| (*state_id, weight.clone()))
                .collect();
            let closure_started = detail.as_ref().map(|_| Instant::now());
            local_epsilon_closure(
                nwa,
                &mut weight_by_state,
                &mut closure_queue,
                &mut target_subset,
            );
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_started) {
                detail.local_epsilon_closure_miss_ms += elapsed_ms(started_at);
            }
            if target_subset.is_empty() {
                return;
            }
            let mut canon: Vec<(u32, Weight)> = target_subset
                .iter()
                .filter(|(_, w)| !w.is_empty())
                .map(|(id, w)| (*id, w.clone()))
                .collect();
            canon.sort_unstable_by_key(|(state_id, _)| *state_id);
            if canon.is_empty() {
                return;
            }

            let subset_lookup_started = detail.as_ref().map(|_| Instant::now());
            let to_state = if let [(only_state, only_weight)] = canon.as_slice() {
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
                    worklist.push_back((new_state, canon.clone()));
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
                    worklist.push_back((new_state, canon.clone()));
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
            dwa.add_transition(from_state, label, to_state, edge_weight);
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

    let mut final_signature_ids: FxHashMap<Vec<(usize, Vec<usize>)>, usize> = FxHashMap::default();
    let mut final_signature_groups: Vec<Vec<(Weight, SmallVec<[Weight; 4]>)>> = Vec::new();
    let mut final_jobs: Vec<(u32, usize)> = Vec::with_capacity(deferred_final_entries.len());
    let final_grouping_started = detail.as_ref().map(|_| Instant::now());
    for (state_id, entries) in &deferred_final_entries {
        if let Some(detail) = detail.as_mut() {
            detail.final_weight_entries += entries.len();
            detail.final_weight_entries_max = detail.final_weight_entries_max.max(entries.len());
        }

        let mut groups: Vec<(usize, Weight, SmallVec<[Weight; 4]>)> = Vec::new();
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
                let owned_groups = groups
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
        let intern_final_groups =
            std::env::var_os("GLRMASK_EXPERIMENTAL_INTERN_FINAL_GROUPS").is_some();
        let final_weights_by_signature: Vec<Option<Weight>> = if intern_final_groups {
            let intern_started_at = Instant::now();
            let mut component_ids = FxHashMap::<(usize, Vec<usize>), usize>::default();
            let mut components = Vec::<(Weight, SmallVec<[Weight; 4]>)>::new();
            let signature_components = final_signature_groups
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
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let intern_ms = elapsed_ms(intern_started_at);
            let component_results: Vec<(Option<Weight>, f64, f64)> = components
                .par_iter()
                .map_init(ScopedWeightOpCache::default, |weight_ops, (final_w, path_weights)| {
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
                })
                .collect();
            if let Some(detail) = detail.as_mut() {
                detail.final_path_union_ms +=
                    component_results.iter().map(|(_, ms, _)| *ms).sum::<f64>();
                detail.final_intersection_ms +=
                    component_results.iter().map(|(_, _, ms)| *ms).sum::<f64>();
            }
            let output_started_at = Instant::now();
            let results = signature_components
                .par_iter()
                .map(|component_ids| {
                    let weight = Weight::union_all(
                        component_ids
                            .iter()
                            .filter_map(|&component_id| component_results[component_id].0.as_ref()),
                    );
                    (!weight.is_empty()).then_some(weight)
                })
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
                let total_components = signature_components.iter().map(Vec::len).sum::<usize>();
                eprintln!(
                    "[glrmask/profile][parser_final_group_intern] signatures={} total_components={} unique_components={} intern_ms={:.3}",
                    signature_components.len(),
                    total_components,
                    components.len(),
                    intern_ms,
                );
            }
            results
        } else {
            let final_weights_by_signature: Vec<(SmallVec<[Weight; 4]>, f64, f64)> =
                final_signature_groups
                    .par_iter()
                    .map_init(ScopedWeightOpCache::default, |weight_ops, final_groups| {
                        let mut path_union_ms = 0.0;
                        let mut intersection_ms = 0.0;
                        let final_contributions: SmallVec<[Weight; 4]> = final_groups
                            .iter()
                            .filter_map(|(final_w, path_weights)| {
                                let pw_union = if detail_enabled {
                                    let path_union_started = Instant::now();
                                    let pw_union = weight_ops.union_all(path_weights.iter());
                                    path_union_ms += elapsed_ms(path_union_started);
                                    pw_union
                                } else {
                                    weight_ops.union_all(path_weights.iter())
                                };
                                let contribution = if detail_enabled {
                                    let intersection_started = Instant::now();
                                    let contribution = weight_ops.intersection(&pw_union, final_w);
                                    intersection_ms += elapsed_ms(intersection_started);
                                    contribution
                                } else {
                                    weight_ops.intersection(&pw_union, final_w)
                                };
                                (!contribution.is_empty()).then_some(contribution)
                            })
                            .collect();
                        (final_contributions, path_union_ms, intersection_ms)
                    })
                    .collect();
            final_weights_by_signature
                .into_iter()
                .map(|(final_contributions, path_union_ms, intersection_ms)| {
                    if let Some(detail) = detail.as_mut() {
                        detail.final_path_union_ms += path_union_ms;
                        detail.final_intersection_ms += intersection_ms;
                    }
                    let output_union_started = detail.as_ref().map(|_| Instant::now());
                    let final_weight = union_cache.union_all(final_contributions.iter());
                    if let (Some(detail), Some(started_at)) =
                        (detail.as_mut(), output_union_started)
                    {
                        detail.final_output_union_ms += elapsed_ms(started_at);
                    }
                    (!final_weight.is_empty()).then_some(final_weight)
                })
                .collect()
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

    if let Some(detail) = detail {
        detail.emit("support");
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

fn determinize_parser_dwa_with_fallbacks(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
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
    let mut singleton_subsets: FxHashMap<(u32, usize), u32> = FxHashMap::default();
    let start_key = subset_key(&canon_buf);
    subset_map.insert(start_key, result.start_state());
    if let [(state_id, weight)] = canon_buf.as_slice() {
        singleton_subsets.insert((*state_id, weight.ptr_key()), result.start_state());
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
        if let [(dwa_state_id, path_weight)] = subset_entries.as_slice()
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
                    detail.intersection_calls += 1;
                }
                let input_target = *target;
                let next_weight = if path_weight.is_full() {
                    transition_weight.clone()
                } else {
                    intersection_cache.intersection(path_weight, transition_weight)
                };
                if next_weight.is_empty() {
                    remove.push(label);
                    continue;
                }
                if let Some(detail) = detail.as_mut() {
                    detail.nonempty_intersections += 1;
                }
                let singleton_key = (input_target, next_weight.ptr_key());
                let to_state = if let Some(existing) = singleton_subsets.get(&singleton_key).copied() {
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
                    singleton_subsets.insert(singleton_key, new_state);
                    worklist.push_back((new_state, vec![(input_target, next_weight.clone())]));
                    new_state
                };
                *target = to_state;
                *transition_weight = next_weight;
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
                    let new_state = result.add_state();
                    subset_map.insert(vec![singleton_key], new_state);
                    singleton_subsets.insert(singleton_key, new_state);
                    worklist.push_back((new_state, contribs.into_iter().collect()));
                    new_state
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

fn subtract_final_weights_from_outgoing_dwa(dwa: &mut DWA) {
    subtract_final_weights_from_outgoing_dwa_impl(
        dwa,
        std::env::var_os("GLRMASK_EXPERIMENTAL_PARALLEL_FINAL_SUBTRACTION").is_some(),
    );
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

pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates(
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
    resolve_negative_codes_in_nwa(&mut parser_nwa);
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
    parser_dwa_pre_minimize = determinize_parser_dwa_with_fallbacks(
        &parser_dwa_pre_minimize,
        &possible_by_state,
        num_parser_states,
    );
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
    use range_set_blaze::RangeSetBlaze;

    use super::{collapse_final_leaf_targets, subtract_final_weights_from_outgoing_dwa_impl};
    use crate::automata::weighted::dwa::DWA;
    use crate::ds::weight::Weight;

    fn weight(tokens: std::ops::RangeInclusive<u32>) -> Weight {
        Weight::from_token_set_for_tsid(0, RangeSetBlaze::from_iter([tokens]))
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
