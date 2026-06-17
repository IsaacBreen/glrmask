use std::collections::{hash_map::Entry, BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::Vocab;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::GLRTable;
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
    branches: Vec<Branch>,
}

#[derive(Debug, Clone)]
struct StateSummaries {
    states: Vec<StateSummary>,
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
    canon: Vec<(u32, Weight)>,
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
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, TerminalBundle> {
    let Some(state) = terminal_dwa.states().get(state_id as usize) else {
        return BTreeMap::new();
    };

    let mut bundles_by_target = BTreeMap::<u32, TerminalBundle>::new();
    for (&label, (target, weight)) in &state.transitions {
        if label < 0 || label as u32 >= grammar.num_terminals {
            continue;
        }

        bundles_by_target
            .entry(*target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    }

    bundles_by_target
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
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> StateSummaries {
    let mut branches_by_state: Vec<Vec<Branch>> = Vec::with_capacity(terminal_dwa.states().len());
    let mut bundle_ids_by_signature: FxHashMap<BundleSignature, usize> = FxHashMap::default();
    let mut unique_bundles: Vec<TerminalBundle> = Vec::new();

    for (state_id, _state) in terminal_dwa.states().iter().enumerate() {
        let bundles_by_target = group_terminal_edges_by_target(terminal_dwa, grammar, state_id as u32);
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

    let states = terminal_dwa
        .states()
        .iter()
        .enumerate()
        .map(|(state_id, state)| StateSummary {
            final_weight: state.final_weight.clone(),
            branches: std::mem::take(&mut branches_by_state[state_id]),
        })
        .collect();

    StateSummaries {
        states,
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
    let mut worklist: VecDeque<Vec<(u32, Weight)>> = VecDeque::new();
    subset_map.insert(subset_key(&canon_buf), dwa.start_state());
    worklist.push_back(canon_buf.clone());

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

    while let Some(subset_entries) = worklist.pop_front() {
        if let Some(detail) = detail.as_mut() {
            detail.states_processed += 1;
            detail.subset_key_constructions += 1;
        }
        let from_state = subset_map[&subset_key(&subset_entries)];

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
                    add_target_contribution_profiled(
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
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), sort_started) {
                detail.contribution_sort_ms += elapsed_ms(started_at);
            }

            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                if nwa.states()[*only_state as usize].epsilons.is_empty() {
                    let key_started = detail.as_ref().map(|_| Instant::now());
                    key_buf.clear();
                    key_buf.push((*only_state, only_weight.ptr_key()));
                    if let (Some(detail), Some(started_at)) = (detail.as_mut(), key_started) {
                        detail.post_closure_subset_key_ms += elapsed_ms(started_at);
                    }
                    let subset_lookup_started = detail.as_ref().map(|_| Instant::now());
                    let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
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
                        worklist.push_back(vec![(*only_state, only_weight.clone())]);
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
            let closure_entry = closure_cache.entry(pre_closure_key.clone());
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), closure_lookup_started) {
                detail.closure_lookup_ms += elapsed_ms(started_at);
            }
            let cached = match closure_entry {
                Entry::Occupied(entry) => {
                    if let Some(detail) = detail.as_mut() {
                        detail.closure_cache_hits += 1;
                    }
                    entry.into_mut()
                }
                Entry::Vacant(entry) => {
                    if let Some(detail) = detail.as_mut() {
                        detail.closure_cache_misses += 1;
                    }
                    let edge_weight_started = detail.as_ref().map(|_| Instant::now());
                    let edge_weight =
                        union_cache.union_all(contribs.iter().map(|(_, weight)| weight));
                    if let (Some(detail), Some(started_at)) =
                        (detail.as_mut(), edge_weight_started)
                    {
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
                    entry.insert(CachedClosure { canon, edge_weight })
                }
            };

            let subset_key_started = detail.as_ref().map(|_| Instant::now());
            key_buf.clear();
            key_buf.extend(cached.canon.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            if let Some(detail) = detail.as_mut() {
                detail.subset_key_constructions += 1;
            }
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), subset_key_started) {
                detail.post_closure_subset_key_ms += elapsed_ms(started_at);
            }
            let subset_lookup_started = detail.as_ref().map(|_| Instant::now());
            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
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
                worklist.push_back(cached.canon.clone());
                supports.push(cached.canon.iter().map(|(sid, _)| *sid).collect());
                new_state
            };
            if let (Some(detail), Some(started_at)) = (detail.as_mut(), subset_lookup_started) {
                detail.subset_map_lookup_ms += elapsed_ms(started_at);
            }
            let add_transition_started = detail.as_ref().map(|_| Instant::now());
            dwa.add_transition(from_state, label, to_state, cached.edge_weight.clone());
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
        let final_weights_by_signature: Vec<(SmallVec<[Weight; 4]>, f64, f64)> =
            final_signature_groups
                .par_iter()
                .map(|final_groups| {
                    let mut path_union_ms = 0.0;
                    let mut intersection_ms = 0.0;
                    let final_contributions: SmallVec<[Weight; 4]> = final_groups
                        .iter()
                        .filter_map(|(final_w, path_weights)| {
                            let pw_union = if detail_enabled {
                                let path_union_started = Instant::now();
                                let pw_union = Weight::union_all(path_weights.iter());
                                path_union_ms += elapsed_ms(path_union_started);
                                pw_union
                            } else {
                                Weight::union_all(path_weights.iter())
                            };
                            let contribution = if detail_enabled {
                                let intersection_started = Instant::now();
                                let contribution = pw_union.intersection(final_w);
                                intersection_ms += elapsed_ms(intersection_started);
                                contribution
                            } else {
                                pw_union.intersection(final_w)
                            };
                            if contribution.is_empty() {
                                None
                            } else {
                                Some(contribution)
                            }
                        })
                        .collect();
                    (final_contributions, path_union_ms, intersection_ms)
                })
                .collect();
        let mut final_weights_by_signature = final_weights_by_signature;
        let final_weights_by_signature: Vec<Option<Weight>> = final_weights_by_signature
            .drain(..)
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
                if final_weight.is_empty() {
                    None
                } else {
                    Some(final_weight)
                }
            })
            .collect();
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
    let mut worklist: VecDeque<Vec<(u32, Weight)>> = VecDeque::new();
    subset_map.insert(subset_key(&canon_buf), result.start_state());
    worklist.push_back(canon_buf.clone());

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

    while let Some(subset_entries) = worklist.pop_front() {
        dense_default_all_raw_targets.clear();
        if let Some(detail) = detail.as_mut() {
            detail.states_processed += 1;
            detail.subset_key_constructions += 1;
        }
        let from_state = subset_map[&subset_key(&subset_entries)];

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

            key_buf.clear();
            if let Some(detail) = detail.as_mut() {
                detail.subset_key_constructions += 1;
            }
            if contribs.len() == 1 {
                let (only_state, only_weight) = &contribs[0];
                key_buf.push((*only_state, only_weight.ptr_key()));
            } else {
                key_buf.extend(contribs.iter().map(|(sid, w)| (*sid, w.ptr_key())));
            }

            let to_state = if let Some(existing) = subset_map.get(&key_buf).copied() {
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
                worklist.push_back(next_entries);
                new_state
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

fn subtract_final_weights_from_outgoing_dwa(dwa: &mut DWA) {
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
            let new_weight = weight.difference(&final_weight);
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
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: Templates,
) -> Option<(NWA, ParserNwaBuildProfile)> {
    let total_started_at = Instant::now();
    let state_prep_started_at = Instant::now();
    let summaries = build_state_summaries(terminal_dwa, grammar, &templates);
    let productive = compute_productive_terminal_states(&summaries);
    let state_prep_ms = elapsed_ms(state_prep_started_at);
    let states = &summaries.states;
    let compose_detail_enabled = parser_dwa_compose_detail_enabled();
    let mut compose_detail = ParserDwaComposeDetailProfile {
        total_states: states.len(),
        productive_states: productive.iter().filter(|&&is_productive| is_productive).count(),
        total_branches: states.iter().map(|state| state.branches.len()).sum(),
        productive_branches: 0,
        unique_bundles: summaries.unique_bundles.len(),
        accepting_bundles: summaries.bundle_accepts.iter().filter(|&&accepts| accepts).count(),
        ..ParserDwaComposeDetailProfile::default()
    };

    if !productive
        .get(terminal_dwa.start_state() as usize)
        .copied()
        .unwrap_or(false)
    {
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

    let start = continuation_states[terminal_dwa.start_state() as usize];
    assert_ne!(start, u32::MAX, "missing parser-DWA start continuation state");
    arena.set_start_states(vec![start]);
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
    terminal_dwa: &DWA,
    templates: Templates,
    _vocab: &Vocab,
    _id_map: &InternalIdMap,
) -> DWA {
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
                terminal_dwa.states().len(),
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
    let determinized = determinize_with_supports(&parser_nwa, Some(table.num_states));
    let support_determinize_ms = elapsed_ms(support_determinize_started_at);
    let mut parser_dwa_pre_minimize = determinized.dwa;

    let possible_outgoing_started_at = Instant::now();
    let possible_by_state = build_possible_outgoing_ids_by_state(
        &parser_nwa,
        &determinized.supports,
        table.num_states,
    );
    let possible_outgoing_ms = elapsed_ms(possible_outgoing_started_at);

    let default_opt_started_at = Instant::now();
    optimize_parser_dwa_defaults(
        &mut parser_dwa_pre_minimize,
        &possible_by_state,
        table.num_states,
    );
    let default_opt_ms = elapsed_ms(default_opt_started_at);

    let subtract_final_started_at = Instant::now();
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa_pre_minimize);
    let subtract_final_ms = elapsed_ms(subtract_final_started_at);

    let fallback_determinize_started_at = Instant::now();
    parser_dwa_pre_minimize = determinize_parser_dwa_with_fallbacks(
        &parser_dwa_pre_minimize,
        &possible_by_state,
        table.num_states,
    );
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
            "[glrmask/profile][parser_dwa_detail] terminal_dwa_states={} terminal_dwa_transitions={} terminal_dwa_interned_ranges={} parser_nwa_states={} parser_nwa_start_states={} pre_minimize_states={} pre_minimize_transitions={} post_minimize_states={} post_minimize_transitions={} minimize_skipped={} state_prep_ms={:.3} compose_state_ms={:.3} parser_nwa_build_ms={:.3} resolve_negative_ms={:.3} support_determinize_ms={:.3} possible_outgoing_ms={:.3} default_opt_ms={:.3} subtract_final_ms={:.3} fallback_determinize_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            terminal_dwa.states().len(),
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
