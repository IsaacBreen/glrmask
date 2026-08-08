use std::collections::{hash_map::Entry, BTreeMap, VecDeque};
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
use crate::compiler::glr::labels::{
    DEFAULT_LABEL, is_negative_label, negative_to_positive_label,
};
use crate::compiler::glr::table::{
    Action, AdmissionPolicy, GLRTable, GlrTableConstruction,
};
use crate::grammar::flat::TerminalID;
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::compiler::stages::resolve_negatives::{
    apply_finality_fixpoint, remove_redundant_default_transitions,
    resolve_negative_codes_in_nwa,
};
use crate::compiler::stages::templates::Templates;
use crate::templates::compile_bundle::{
    LazyWeightedPrepushBundleSet, WeightedPrepushBundle, WeightedPrepushTarget,
};
use crate::ds::bitset::BitSet;
use crate::ds::weight::{ScopedWeightOpCache, Weight};

fn compile_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
        || std::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some()
}

type TerminalBundle = BTreeMap<TerminalID, Weight>;
type BundleSignature = Vec<(TerminalID, Weight)>;
type BundleTopologySignature = Vec<Vec<TerminalID>>;
type TargetContribs = SmallVec<[(u32, Weight); 4]>;
type DeferredFinalEntries = SmallVec<[(u32, Weight); 4]>;
type FinalPathWeights = SmallVec<[Weight; 4]>;
type FinalGroups = SmallVec<[(Weight, FinalPathWeights); 4]>;

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
    weight_ops: &mut ScopedWeightOpCache,
    mut detail: Option<&mut ParserDwaDeterminizeDetail>,
) {
    if contribs.len() < 2 {
        return;
    }
    static BULK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let bulk = *BULK.get_or_init(|| {
        std::env::var_os("GLRMASK_BULK_MERGE_TARGET_CONTRIBS").is_some()
    });
    if !bulk {
        let mut write = 0usize;
        for read in 1..contribs.len() {
            if contribs[write].0 == contribs[read].0 {
                let merged = weight_ops.union(&contribs[write].1, &contribs[read].1);
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
        return;
    }

    let mut write = 0usize;
    let mut run_start = 0usize;
    while run_start < contribs.len() {
        let target = contribs[run_start].0;
        let mut run_end = run_start + 1;
        while run_end < contribs.len() && contribs[run_end].0 == target {
            run_end += 1;
        }
        let weight = if run_end == run_start + 1 {
            contribs[run_start].1.clone()
        } else {
            if let Some(detail) = detail.as_mut() {
                detail.target_contribution_merges += run_end - run_start - 1;
            }
            Weight::union_all(contribs[run_start..run_end].iter().map(|(_, weight)| weight))
        };
        contribs[write] = (target, weight);
        write += 1;
        run_start = run_end;
    }
    contribs.truncate(write);
}

fn extend_target_contribs(dst: &mut TargetContribs, src: &TargetContribs) {
    for (target, weight) in src {
        add_target_contribution(dst, *target, weight.clone());
    }
}

#[derive(Debug, Clone)]
struct Branch {
    target: u32,
    bundle_id: usize,
    entry_weight: Weight,
    cross_target_group_id: Option<usize>,
}

#[derive(Debug, Clone)]
struct CrossTargetBundleGroup {
    bundle_id: usize,
    target_gates: Vec<(u32, Weight)>,
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
    cross_target_groups: Vec<CrossTargetBundleGroup>,
}

#[derive(Debug, Clone)]
struct DeterminizedDwaWithSupports {
    dwa: DWA,
    supports: Vec<Vec<u32>>,
    weighted_supports: Option<Vec<Vec<(u32, Weight)>>>,
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

fn parser_bundle_gate_factor_profile_enabled() -> bool {
    std::env::var("GLRMASK_PROFILE_PARSER_BUNDLE_GATE_FACTOR")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn parser_bundle_gate_factor_enabled() -> bool {
    match std::env::var("GLRMASK_PARSER_BUNDLE_GATE_FACTOR") {
        Ok(value) => {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
        }
        Err(_) => true,
    }
}

fn parser_bundle_gate_factor_min_estimated_template_states() -> usize {
    std::env::var("GLRMASK_PARSER_BUNDLE_GATE_FACTOR_MIN_ESTIMATED_TEMPLATE_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(10_000)
}

fn parser_bundle_cross_target_gate_factor_enabled() -> bool {
    std::env::var("GLRMASK_PARSER_BUNDLE_CROSS_TARGET_GATE_FACTOR")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn parser_bundle_cross_target_gate_factor_min_estimated_template_states() -> usize {
    std::env::var("GLRMASK_PARSER_BUNDLE_CROSS_TARGET_GATE_FACTOR_MIN_ESTIMATED_TEMPLATE_STATES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(10_000)
}

fn validate_parser_bundle_gate_factor_enabled() -> bool {
    std::env::var("GLRMASK_VALIDATE_PARSER_BUNDLE_GATE_FACTOR")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn intersect_all_nwa_weights(nwa: &mut NWA, gate: &Weight) {
    for state in nwa.states_mut() {
        if let Some(final_weight) = state.final_weight.as_mut() {
            *final_weight = final_weight.intersection(gate);
        }
        for targets in state.transitions.values_mut() {
            for (_, weight) in targets {
                *weight = weight.intersection(gate);
            }
        }
        for (_, weight) in &mut state.epsilons {
            *weight = weight.intersection(gate);
        }
    }
}

/// Replace exact same-target bundle variants by one canonical bundle plus a
/// row gate on the branch entry edge.
///
/// For a fixed continuation target and terminal set, this rewrite is admitted
/// only when every concrete terminal weight satisfies
/// `w[i,t] = G[i] ∩ T[t]`, with `G[i] = union_t w[i,t]` and
/// `T[t] = union_i w[i,t]`. Weighted path composition is intersection and
/// alternatives combine by union, so distributivity makes the row language
/// exactly `G[i] ∩ canonical_bundle_language`. The fixed target is essential:
/// one shared canonical fragment may redirect its finals to only that target.
fn factor_parser_bundle_entry_gates(summaries: &mut StateSummaries, templates: &Templates) {
    if !parser_bundle_gate_factor_enabled() {
        return;
    }

    let started_at = Instant::now();
    let mut ids_by_target_and_terminals =
        FxHashMap::<(u32, Vec<TerminalID>), Vec<usize>>::default();
    for state in &summaries.states {
        for branch in &state.branches {
            let terminals = summaries.unique_bundles[branch.bundle_id]
                .keys()
                .copied()
                .collect::<Vec<_>>();
            let ids = ids_by_target_and_terminals
                .entry((branch.target, terminals))
                .or_default();
            if !ids.contains(&branch.bundle_id) {
                ids.push(branch.bundle_id);
            }
        }
    }


    // Cheap upper bound before touching any Weight algebra. If every candidate
    // group factored perfectly, sharing one variant would avoid at most the sum
    // of its terminal-template states for each additional row. Small parsers
    // cannot amortize even the exact factorization census, so decline them here.
    let max_estimated_removable_template_states = ids_by_target_and_terminals
        .iter()
        .filter(|(_, bundle_ids)| bundle_ids.len() >= 2)
        .map(|((_, terminals), bundle_ids)| {
            let states_per_variant = terminals
                .iter()
                .filter_map(|terminal| templates.by_terminal_nwa.get(terminal))
                .map(|template| template.states().len())
                .sum::<usize>();
            states_per_variant.saturating_mul(bundle_ids.len().saturating_sub(1))
        })
        .sum::<usize>();
    let min_estimated = parser_bundle_gate_factor_min_estimated_template_states();
    if max_estimated_removable_template_states < min_estimated {
        if compile_profile_enabled() || parser_bundle_gate_factor_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_bundle_gate_factor_apply] skipped=small estimated_upper_bound_template_states={} min_estimated_template_states={} total_ms={:.3}",
                max_estimated_removable_template_states,
                min_estimated,
                elapsed_ms(started_at),
            );
        }
        return;
    }

    let mut bundle_ids_by_signature = summaries
        .unique_bundles
        .iter()
        .enumerate()
        .map(|(bundle_id, bundle)| (bundle_signature(bundle), bundle_id))
        .collect::<FxHashMap<_, _>>();
    let mut rewrite = FxHashMap::<(u32, usize), (usize, Weight)>::default();
    let mut factored_groups = 0usize;
    let mut factored_variants = 0usize;
    let mut canonical_bundles_added = 0usize;
    let mut estimated_removable_template_states = 0usize;
    let mut largest_estimated_group_saving = 0usize;

    for ((target, terminals), bundle_ids) in ids_by_target_and_terminals {
        if bundle_ids.len() < 2 || terminals.is_empty() {
            continue;
        }

        let row_gates = bundle_ids
            .iter()
            .map(|&bundle_id| Weight::union_all(summaries.unique_bundles[bundle_id].values()))
            .collect::<Vec<_>>();
        let column_weights = terminals
            .iter()
            .map(|terminal| {
                Weight::union_all(bundle_ids.iter().map(|&bundle_id| {
                    summaries.unique_bundles[bundle_id]
                        .get(terminal)
                        .expect("same-terminal-set bundle group lost a terminal")
                }))
            })
            .collect::<Vec<_>>();

        let exact = bundle_ids.iter().enumerate().all(|(row, &bundle_id)| {
            terminals.iter().enumerate().all(|(column, terminal)| {
                row_gates[row].intersection(&column_weights[column])
                    == *summaries.unique_bundles[bundle_id]
                        .get(terminal)
                        .expect("same-terminal-set bundle group lost a terminal")
            })
        });
        if !exact {
            continue;
        }

        let template_states_per_variant = terminals
            .iter()
            .filter_map(|terminal| templates.by_terminal_nwa.get(terminal))
            .map(|template| template.states().len())
            .sum::<usize>();
        let estimated_group_saving =
            template_states_per_variant.saturating_mul(bundle_ids.len().saturating_sub(1));
        estimated_removable_template_states =
            estimated_removable_template_states.saturating_add(estimated_group_saving);
        largest_estimated_group_saving =
            largest_estimated_group_saving.max(estimated_group_saving);

        let canonical_bundle = terminals
            .iter()
            .copied()
            .zip(column_weights.into_iter())
            .collect::<TerminalBundle>();
        let signature = bundle_signature(&canonical_bundle);
        let canonical_id = if let Some(&bundle_id) = bundle_ids_by_signature.get(&signature) {
            bundle_id
        } else {
            let bundle_id = summaries.unique_bundles.len();
            let accepts = terminal_bundle_has_acceptance(&canonical_bundle, templates);
            summaries.unique_bundles.push(canonical_bundle);
            summaries.bundle_accepts.push(accepts);
            bundle_ids_by_signature.insert(signature, bundle_id);
            canonical_bundles_added += 1;
            bundle_id
        };

        if validate_parser_bundle_gate_factor_enabled() {
            let canonical_nwa = templates.build_bundle(&summaries.unique_bundles[canonical_id]);
            for (&old_bundle_id, gate) in bundle_ids.iter().zip(row_gates.iter()) {
                let original_nwa = templates.build_bundle(&summaries.unique_bundles[old_bundle_id]);
                let original_dwa = determinize_with_supports(&original_nwa, None).dwa;
                let mut gated_canonical_nwa = canonical_nwa.clone();
                intersect_all_nwa_weights(&mut gated_canonical_nwa, gate);
                let gated_canonical_dwa =
                    determinize_with_supports(&gated_canonical_nwa, None).dwa;
                assert!(
                    find_difference(&original_dwa, &gated_canonical_dwa)
                        .expect("parser bundle gate-factor validation requires finite acyclic bundles")
                        .is_none(),
                    "parser bundle gate factorization changed weighted bundle language: target={target} old_bundle={old_bundle_id} canonical_bundle={canonical_id}",
                );
            }
        }

        for (&old_bundle_id, gate) in bundle_ids.iter().zip(row_gates.into_iter()) {
            debug_assert!(!gate.is_empty(), "factored parser bundle row gate must be live");
            rewrite.insert((target, old_bundle_id), (canonical_id, gate));
        }
        factored_groups += 1;
        factored_variants += bundle_ids.len();
    }

    let mut rewritten_branches = 0usize;
    for state in &mut summaries.states {
        for branch in &mut state.branches {
            let Some((canonical_id, gate)) = rewrite.get(&(branch.target, branch.bundle_id)) else {
                continue;
            };
            branch.bundle_id = *canonical_id;
            branch.entry_weight = gate.clone();
            rewritten_branches += 1;
        }
    }

    if compile_profile_enabled() || parser_bundle_gate_factor_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_bundle_gate_factor_apply] factored_groups={} factored_variants={} rewritten_branches={} canonical_bundles_added={} estimated_removable_template_states={} largest_estimated_group_saving={} total_bundles_after={} total_ms={:.3}",
            factored_groups,
            factored_variants,
            rewritten_branches,
            canonical_bundles_added,
            estimated_removable_template_states,
            largest_estimated_group_saving,
            summaries.unique_bundles.len(),
            elapsed_ms(started_at),
        );
    }
}

/// Share one already-factorized canonical bundle across distinct continuation
/// targets when their target domains are pairwise disjoint.
///
/// This pass deliberately runs after same-target factorization. It only accepts
/// terminal sets for which every target now has exactly one bundle row. If
/// `C[j,t] = H[j] ∩ T[t]` for target row `j`, and the `H[j]` are pairwise
/// disjoint, one global canonical bundle `T` can feed every continuation using
/// final epsilon edges weighted by `H[j]`. A source branch retains its narrower
/// same-target entry gate, so it can reach only the continuation whose target
/// gate contains that row domain.
fn factor_parser_bundle_cross_target_gates(
    summaries: &mut StateSummaries,
    templates: &Templates,
) {
    if !parser_bundle_cross_target_gate_factor_enabled() {
        return;
    }

    let started_at = Instant::now();
    let mut rows_by_terminals =
        FxHashMap::<Vec<TerminalID>, BTreeMap<u32, Vec<usize>>>::default();
    for state in &summaries.states {
        for branch in &state.branches {
            let terminals = summaries.unique_bundles[branch.bundle_id]
                .keys()
                .copied()
                .collect::<Vec<_>>();
            let ids = rows_by_terminals
                .entry(terminals)
                .or_default()
                .entry(branch.target)
                .or_default();
            if !ids.contains(&branch.bundle_id) {
                ids.push(branch.bundle_id);
            }
        }
    }

    // Cheap upper bound before any Weight unions/intersections. For each
    // terminal set, even a perfect cross-target factorization can remove at
    // most one template copy per additional continuation target. Small parsers
    // cannot amortize the exact rectangular/disjointness census.
    let max_estimated_incremental_template_states = rows_by_terminals
        .iter()
        .filter_map(|(terminals, by_target)| {
            let target_count = by_target.len();
            (target_count >= 2).then(|| {
                let states_per_copy = terminals
                    .iter()
                    .filter_map(|terminal| templates.by_terminal_nwa.get(terminal))
                    .map(|template| template.states().len())
                    .sum::<usize>();
                states_per_copy.saturating_mul(target_count.saturating_sub(1))
            })
        })
        .sum::<usize>();
    let min_estimated = parser_bundle_cross_target_gate_factor_min_estimated_template_states();
    if max_estimated_incremental_template_states < min_estimated {
        if compile_profile_enabled() || parser_bundle_gate_factor_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_bundle_cross_target_gate_factor_apply] skipped=small estimated_upper_bound_template_states={} min_estimated_template_states={} total_ms={:.3}",
                max_estimated_incremental_template_states,
                min_estimated,
                elapsed_ms(started_at),
            );
        }
        return;
    }

    struct PendingCrossTargetGroup {
        canonical_bundle: TerminalBundle,
        target_rows: Vec<(u32, usize)>,
        target_gates: Vec<Weight>,
        estimated_saving: usize,
    }

    let mut pending_groups = Vec::<PendingCrossTargetGroup>::new();
    let mut estimated_incremental_template_states = 0usize;

    for (terminals, by_target) in rows_by_terminals {
        if terminals.is_empty() || by_target.len() < 2 {
            continue;
        }
        if by_target.values().any(|bundle_ids| bundle_ids.len() != 1) {
            continue;
        }
        let target_rows = by_target
            .iter()
            .map(|(&target, bundle_ids)| (target, bundle_ids[0]))
            .collect::<Vec<_>>();
        let target_gates = target_rows
            .iter()
            .map(|(_, bundle_id)| Weight::union_all(summaries.unique_bundles[*bundle_id].values()))
            .collect::<Vec<_>>();
        let column_weights = terminals
            .iter()
            .map(|terminal| {
                Weight::union_all(target_rows.iter().map(|(_, bundle_id)| {
                    summaries.unique_bundles[*bundle_id]
                        .get(terminal)
                        .expect("cross-target canonical row lost a terminal")
                }))
            })
            .collect::<Vec<_>>();
        let exact = target_rows
            .iter()
            .enumerate()
            .all(|(row, (_, bundle_id))| {
                terminals.iter().enumerate().all(|(column, terminal)| {
                    target_gates[row].intersection(&column_weights[column])
                        == *summaries.unique_bundles[*bundle_id]
                            .get(terminal)
                            .expect("cross-target canonical row lost a terminal")
                })
            });
        if !exact {
            continue;
        }
        let disjoint = (0..target_gates.len()).all(|left| {
            (left + 1..target_gates.len())
                .all(|right| target_gates[left].is_disjoint(&target_gates[right]))
        });
        if !disjoint {
            continue;
        }

        let template_states_per_copy = terminals
            .iter()
            .filter_map(|terminal| templates.by_terminal_nwa.get(terminal))
            .map(|template| template.states().len())
            .sum::<usize>();
        let estimated_saving =
            template_states_per_copy.saturating_mul(target_rows.len().saturating_sub(1));
        estimated_incremental_template_states =
            estimated_incremental_template_states.saturating_add(estimated_saving);
        pending_groups.push(PendingCrossTargetGroup {
            canonical_bundle: terminals
                .iter()
                .copied()
                .zip(column_weights.into_iter())
                .collect(),
            target_rows,
            target_gates,
            estimated_saving,
        });
    }

    if estimated_incremental_template_states < min_estimated {
        if compile_profile_enabled() || parser_bundle_gate_factor_profile_enabled() {
            eprintln!(
                "[glrmask/profile][parser_bundle_cross_target_gate_factor_apply] skipped=exact_small estimated_upper_bound_template_states={} exact_estimated_template_states={} min_estimated_template_states={} exact_groups={} total_ms={:.3}",
                max_estimated_incremental_template_states,
                estimated_incremental_template_states,
                min_estimated,
                pending_groups.len(),
                elapsed_ms(started_at),
            );
        }
        return;
    }

    let mut bundle_ids_by_signature = summaries
        .unique_bundles
        .iter()
        .enumerate()
        .map(|(bundle_id, bundle)| (bundle_signature(bundle), bundle_id))
        .collect::<FxHashMap<_, _>>();
    let mut rewrite = FxHashMap::<(u32, usize), (usize, usize, Weight)>::default();
    let mut factored_groups = 0usize;
    let mut factored_targets = 0usize;
    let mut canonical_bundles_added = 0usize;

    for pending in pending_groups {
        let signature = bundle_signature(&pending.canonical_bundle);
        let canonical_id = if let Some(&bundle_id) = bundle_ids_by_signature.get(&signature) {
            bundle_id
        } else {
            let bundle_id = summaries.unique_bundles.len();
            let accepts = terminal_bundle_has_acceptance(&pending.canonical_bundle, templates);
            summaries.unique_bundles.push(pending.canonical_bundle);
            summaries.bundle_accepts.push(accepts);
            bundle_ids_by_signature.insert(signature, bundle_id);
            canonical_bundles_added += 1;
            bundle_id
        };

        if validate_parser_bundle_gate_factor_enabled() {
            let canonical_nwa = templates.build_bundle(&summaries.unique_bundles[canonical_id]);
            for ((target, old_bundle_id), gate) in
                pending.target_rows.iter().zip(pending.target_gates.iter())
            {
                let original_nwa = templates.build_bundle(&summaries.unique_bundles[*old_bundle_id]);
                let original_dwa = determinize_with_supports(&original_nwa, None).dwa;
                let mut gated_canonical_nwa = canonical_nwa.clone();
                intersect_all_nwa_weights(&mut gated_canonical_nwa, gate);
                let gated_canonical_dwa =
                    determinize_with_supports(&gated_canonical_nwa, None).dwa;
                assert!(
                    find_difference(&original_dwa, &gated_canonical_dwa)
                        .expect("cross-target gate-factor validation requires finite acyclic bundles")
                        .is_none(),
                    "cross-target parser bundle factorization changed weighted bundle language: target={target} old_bundle={old_bundle_id} canonical_bundle={canonical_id}",
                );
            }
        }

        let group_id = summaries.cross_target_groups.len();
        summaries.cross_target_groups.push(CrossTargetBundleGroup {
            bundle_id: canonical_id,
            target_gates: pending
                .target_rows
                .iter()
                .zip(pending.target_gates.iter())
                .map(|((target, _), gate)| (*target, gate.clone()))
                .collect(),
        });
        for ((target, old_bundle_id), gate) in
            pending.target_rows.iter().zip(pending.target_gates.iter())
        {
            rewrite.insert(
                (*target, *old_bundle_id),
                (canonical_id, group_id, gate.clone()),
            );
        }
        factored_groups += 1;
        factored_targets += pending.target_rows.len();
        debug_assert!(pending.estimated_saving > 0);
    }

    let mut rewritten_branches = 0usize;
    for state in &mut summaries.states {
        for branch in &mut state.branches {
            let Some((canonical_id, group_id, target_gate)) =
                rewrite.get(&(branch.target, branch.bundle_id))
            else {
                continue;
            };
            // The old target-specific bundle is independently certified as
            // `target_gate ∩ global_bundle`. Therefore an existing source gate
            // `E` may be moved across the replacement by distributivity:
            // `E ∩ old = E ∩ target_gate ∩ global_bundle`.
            branch.bundle_id = *canonical_id;
            branch.entry_weight = branch.entry_weight.intersection(target_gate);
            branch.cross_target_group_id = Some(*group_id);
            rewritten_branches += 1;
        }
    }

    if compile_profile_enabled() || parser_bundle_gate_factor_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_bundle_cross_target_gate_factor_apply] factored_groups={} factored_targets={} rewritten_branches={} canonical_bundles_added={} estimated_incremental_template_states={} cross_target_groups={} total_ms={:.3}",
            factored_groups,
            factored_targets,
            rewritten_branches,
            canonical_bundles_added,
            estimated_incremental_template_states,
            summaries.cross_target_groups.len(),
            elapsed_ms(started_at),
        );
    }
}

/// Profile an exact rectangular factorization of bundle weights.
///
/// For a fixed continuation target and terminal set, let `w[i,t]` be the
/// token-domain weight on terminal `t` in bundle `i`, let `G[i]` be the union
/// of row `i`, and let `T[t]` be the union of column `t`.  If
///
///     w[i,t] == G[i] ∩ T[t]
///
/// for every row/column, then all bundles in the group are restrictions of one
/// canonical per-terminal bundle by a single row gate.  This function only
/// measures how often that identity holds; it does not change construction.
fn profile_parser_bundle_gate_factorability(summaries: &StateSummaries) {
    if !parser_bundle_gate_factor_profile_enabled() {
        return;
    }

    let started_at = Instant::now();
    let mut ids_by_target_and_terminals =
        FxHashMap::<(u32, Vec<TerminalID>), Vec<usize>>::default();
    for state in &summaries.states {
        for branch in &state.branches {
            let bundle = &summaries.unique_bundles[branch.bundle_id];
            let terminals = bundle.keys().copied().collect::<Vec<_>>();
            let ids = ids_by_target_and_terminals
                .entry((branch.target, terminals))
                .or_default();
            if !ids.contains(&branch.bundle_id) {
                ids.push(branch.bundle_id);
            }
        }
    }

    let mut candidate_groups = 0usize;
    let mut candidate_bundles = 0usize;
    let mut exact_groups = 0usize;
    let mut exact_bundles = 0usize;
    let mut exact_removable_bundles = 0usize;
    let mut exact_group_size_histogram = BTreeMap::<usize, usize>::new();
    let mut largest_exact_group = 0usize;

    for ((_target, terminals), bundle_ids) in ids_by_target_and_terminals {
        if bundle_ids.len() < 2 || terminals.is_empty() {
            continue;
        }
        candidate_groups += 1;
        candidate_bundles += bundle_ids.len();

        let row_gates = bundle_ids
            .iter()
            .map(|&bundle_id| {
                Weight::union_all(summaries.unique_bundles[bundle_id].values())
            })
            .collect::<Vec<_>>();
        let column_weights = terminals
            .iter()
            .map(|terminal| {
                Weight::union_all(bundle_ids.iter().map(|&bundle_id| {
                    summaries.unique_bundles[bundle_id]
                        .get(terminal)
                        .expect("same-terminal-set bundle group lost a terminal")
                }))
            })
            .collect::<Vec<_>>();

        let exact = bundle_ids.iter().enumerate().all(|(row, &bundle_id)| {
            let bundle = &summaries.unique_bundles[bundle_id];
            terminals.iter().enumerate().all(|(column, terminal)| {
                row_gates[row].intersection(&column_weights[column])
                    == *bundle
                        .get(terminal)
                        .expect("same-terminal-set bundle group lost a terminal")
            })
        });
        if !exact {
            continue;
        }

        exact_groups += 1;
        exact_bundles += bundle_ids.len();
        exact_removable_bundles += bundle_ids.len() - 1;
        largest_exact_group = largest_exact_group.max(bundle_ids.len());
        *exact_group_size_histogram.entry(bundle_ids.len()).or_default() += 1;
        eprintln!(
            "[glrmask/profile][parser_bundle_gate_factor_group] target={} terminals={} bundles={:?}",
            _target,
            terminals.len(),
            bundle_ids,
        );
    }

    eprintln!(
        "[glrmask/profile][parser_bundle_gate_factor] unique_bundles={} candidate_groups={} candidate_bundles={} exact_groups={} exact_bundles={} exact_removable_bundles={} largest_exact_group={} exact_group_size_histogram={:?} total_ms={:.3}",
        summaries.unique_bundles.len(),
        candidate_groups,
        candidate_bundles,
        exact_groups,
        exact_bundles,
        exact_removable_bundles,
        largest_exact_group,
        exact_group_size_histogram,
        elapsed_ms(started_at),
    );
}

/// Measure the additional exact sharing available across continuation targets.
///
/// Same-target factorization above needs no separation between row gates. For
/// different targets, a shared canonical interior is still exact when the
/// unioned gate for each target is pairwise disjoint from every other target's
/// gate. Then canonical final states may have one epsilon edge per target,
/// weighted by that target gate: the source-row gate intersects away every
/// foreign continuation.
fn profile_parser_bundle_cross_target_gate_factorability(
    summaries: &StateSummaries,
    templates: &Templates,
) {
    if !parser_bundle_gate_factor_profile_enabled() {
        return;
    }

    let started_at = Instant::now();
    let mut rows_by_terminals = FxHashMap::<Vec<TerminalID>, Vec<(u32, usize)>>::default();
    for state in &summaries.states {
        for branch in &state.branches {
            let terminals = summaries.unique_bundles[branch.bundle_id]
                .keys()
                .copied()
                .collect::<Vec<_>>();
            let rows = rows_by_terminals.entry(terminals).or_default();
            let row = (branch.target, branch.bundle_id);
            if !rows.contains(&row) {
                rows.push(row);
            }
        }
    }

    let mut exact_cross_target_groups = 0usize;
    let mut exact_cross_target_rows = 0usize;
    let mut distinct_targets_total = 0usize;
    let mut incremental_removable_target_copies = 0usize;
    let mut estimated_incremental_template_states = 0usize;
    let mut largest_incremental_group_saving = 0usize;

    for (terminals, rows) in rows_by_terminals {
        if rows.len() < 2 || terminals.is_empty() {
            continue;
        }
        let distinct_targets = rows
            .iter()
            .map(|(target, _)| *target)
            .collect::<rustc_hash::FxHashSet<_>>();
        if distinct_targets.len() < 2 {
            continue;
        }

        let row_gates = rows
            .iter()
            .map(|(_, bundle_id)| Weight::union_all(summaries.unique_bundles[*bundle_id].values()))
            .collect::<Vec<_>>();
        let column_weights = terminals
            .iter()
            .map(|terminal| {
                Weight::union_all(rows.iter().map(|(_, bundle_id)| {
                    summaries.unique_bundles[*bundle_id]
                        .get(terminal)
                        .expect("cross-target same-terminal-set group lost a terminal")
                }))
            })
            .collect::<Vec<_>>();
        let rectangular = rows.iter().enumerate().all(|(row, (_, bundle_id))| {
            terminals.iter().enumerate().all(|(column, terminal)| {
                row_gates[row].intersection(&column_weights[column])
                    == *summaries.unique_bundles[*bundle_id]
                        .get(terminal)
                        .expect("cross-target same-terminal-set group lost a terminal")
            })
        });
        if !rectangular {
            continue;
        }

        let mut gate_by_target = BTreeMap::<u32, Weight>::new();
        for ((target, _), gate) in rows.iter().zip(row_gates.iter()) {
            gate_by_target
                .entry(*target)
                .and_modify(|existing| *existing = existing.union(gate))
                .or_insert_with(|| gate.clone());
        }
        let target_gates = gate_by_target.values().collect::<Vec<_>>();
        let target_disjoint = (0..target_gates.len()).all(|left| {
            (left + 1..target_gates.len())
                .all(|right| target_gates[left].is_disjoint(target_gates[right]))
        });
        if !target_disjoint {
            continue;
        }

        let target_count = gate_by_target.len();
        let template_states_per_copy = terminals
            .iter()
            .filter_map(|terminal| templates.by_terminal_nwa.get(terminal))
            .map(|template| template.states().len())
            .sum::<usize>();
        let estimated_saving =
            template_states_per_copy.saturating_mul(target_count.saturating_sub(1));
        estimated_incremental_template_states =
            estimated_incremental_template_states.saturating_add(estimated_saving);
        largest_incremental_group_saving =
            largest_incremental_group_saving.max(estimated_saving);
        exact_cross_target_groups += 1;
        exact_cross_target_rows += rows.len();
        distinct_targets_total += target_count;
        incremental_removable_target_copies += target_count - 1;
        eprintln!(
            "[glrmask/profile][parser_bundle_cross_target_gate_factor_group] terminals={} rows={} targets={} estimated_incremental_template_states={}",
            terminals.len(),
            rows.len(),
            target_count,
            estimated_saving,
        );
    }

    eprintln!(
        "[glrmask/profile][parser_bundle_cross_target_gate_factor] exact_groups={} rows={} distinct_targets={} incremental_removable_target_copies={} estimated_incremental_template_states={} largest_incremental_group_saving={} total_ms={:.3}",
        exact_cross_target_groups,
        exact_cross_target_rows,
        distinct_targets_total,
        incremental_removable_target_copies,
        estimated_incremental_template_states,
        largest_incremental_group_saving,
        elapsed_ms(started_at),
    );
}

fn parser_bundle_topology_reuse_enabled() -> bool {
    std::env::var("GLRMASK_REUSE_PARSER_BUNDLE_TOPOLOGY")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn profile_template_stack_effect_normal_form(templates: &Templates) {
    if !compile_profile_enabled() {
        return;
    }

    let mut templates_checked = 0usize;
    let mut violating_templates = 0usize;
    let mut total_states = 0usize;
    let mut pre_push_reachable_states = 0usize;
    let mut push_reachable_states = 0usize;
    let mut dual_phase_states = 0usize;
    let mut positive_or_default_edges = 0usize;
    let mut negative_edges = 0usize;

    for template in templates.by_terminal_nwa.values() {
        templates_checked += 1;
        total_states += template.states().len();
        let mut seen_pre = vec![false; template.states().len()];
        let mut seen_push = vec![false; template.states().len()];
        let mut queue = VecDeque::<(u32, bool)>::new();
        for &start in template.start_states() {
            queue.push_back((start, false));
        }
        let mut violates = false;
        while let Some((state_id, push_phase)) = queue.pop_front() {
            let seen = if push_phase {
                &mut seen_push[state_id as usize]
            } else {
                &mut seen_pre[state_id as usize]
            };
            if *seen {
                continue;
            }
            *seen = true;
            let state = &template.states()[state_id as usize];
            for (&label, targets) in &state.transitions {
                let negative = crate::compiler::glr::labels::is_negative_label(label);
                if negative {
                    negative_edges += targets.len();
                } else {
                    positive_or_default_edges += targets.len();
                    if push_phase {
                        violates = true;
                    }
                }
                for (target, _) in targets {
                    queue.push_back((*target, push_phase || negative));
                }
            }
            for (target, _) in &state.epsilons {
                queue.push_back((*target, push_phase));
            }
        }
        violating_templates += usize::from(violates);
        for state_id in 0..template.states().len() {
            pre_push_reachable_states += usize::from(seen_pre[state_id]);
            push_reachable_states += usize::from(seen_push[state_id]);
            dual_phase_states += usize::from(seen_pre[state_id] && seen_push[state_id]);
        }
    }

    eprintln!(
        "[glrmask/profile][parser_template_stack_effect_normal_form] templates={} violating_templates={} total_states={} pre_push_reachable_states={} push_reachable_states={} dual_phase_states={} positive_or_default_edges={} negative_edges={}",
        templates_checked,
        violating_templates,
        total_states,
        pre_push_reachable_states,
        push_reachable_states,
        dual_phase_states,
        positive_or_default_edges,
        negative_edges,
    );
}

fn profile_composed_parser_nwa_label_shape(nwa: &NWA) {
    if !compile_profile_enabled() {
        return;
    }

    let mut negative_only_states = 0usize;
    let mut positive_only_states = 0usize;
    let mut mixed_states = 0usize;
    let mut unlabeled_states = 0usize;
    let mut negative_edges = 0usize;
    let mut positive_or_default_edges = 0usize;
    let mut epsilon_edges = 0usize;
    for state in nwa.states() {
        let mut has_negative = false;
        let mut has_positive = false;
        for (&label, targets) in &state.transitions {
            if crate::compiler::glr::labels::is_negative_label(label) {
                has_negative = true;
                negative_edges += targets.len();
            } else {
                has_positive = true;
                positive_or_default_edges += targets.len();
            }
        }
        epsilon_edges += state.epsilons.len();
        match (has_negative, has_positive) {
            (true, false) => negative_only_states += 1,
            (false, true) => positive_only_states += 1,
            (true, true) => mixed_states += 1,
            (false, false) => unlabeled_states += 1,
        }
    }
    eprintln!(
        "[glrmask/profile][parser_composed_nwa_label_shape] states={} negative_only_states={} positive_only_states={} mixed_states={} unlabeled_states={} negative_edges={} positive_or_default_edges={} epsilon_edges={}",
        nwa.states().len(),
        negative_only_states,
        positive_only_states,
        mixed_states,
        unlabeled_states,
        negative_edges,
        positive_or_default_edges,
        epsilon_edges,
    );

    let n = nwa.states().len();
    let mut eligible = vec![false; n];
    for (state_id, state) in nwa.states().iter().enumerate() {
        let mut has_negative = false;
        let mut has_nonnegative = false;
        for &label in state.transitions.keys() {
            if crate::compiler::glr::labels::is_negative_label(label) {
                has_negative = true;
            } else {
                has_nonnegative = true;
            }
        }
        eligible[state_id] = has_negative && !has_nonnegative;
    }

    let mut internal_outdegree = vec![0usize; n];
    let mut predecessors = vec![Vec::<usize>::new(); n];
    for (source, state) in nwa.states().iter().enumerate() {
        if !eligible[source] {
            continue;
        }
        for targets in state.transitions.values() {
            for (target, _) in targets {
                let target = *target as usize;
                if target < n && eligible[target] {
                    internal_outdegree[source] += 1;
                    predecessors[target].push(source);
                }
            }
        }
        for (target, _) in &state.epsilons {
            let target = *target as usize;
            if target < n && eligible[target] {
                internal_outdegree[source] += 1;
                predecessors[target].push(source);
            }
        }
    }

    let mut queue = VecDeque::new();
    for state_id in 0..n {
        if eligible[state_id] && internal_outdegree[state_id] == 0 {
            queue.push_back(state_id);
        }
    }
    let mut class_by_state = vec![None::<u32>; n];
    let mut class_by_signature =
        FxHashMap::<(usize, Vec<(u8, i32, u64, usize)>), u32>::default();
    let mut processed = 0usize;
    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states()[state_id];
        let final_key = state.final_weight.as_ref().map_or(0, Weight::ptr_key);
        let mut signature = Vec::<(u8, i32, u64, usize)>::new();
        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                let target = *target as usize;
                let target_key = if target < n && eligible[target] {
                    class_by_state[target]
                        .expect("negative-only successor must already be canonicalized")
                        as u64
                } else {
                    (1u64 << 63) | target as u64
                };
                signature.push((0, label, target_key, weight.ptr_key()));
            }
        }
        for (target, weight) in &state.epsilons {
            let target = *target as usize;
            let target_key = if target < n && eligible[target] {
                class_by_state[target]
                    .expect("negative-only epsilon successor must already be canonicalized")
                    as u64
            } else {
                (1u64 << 63) | target as u64
            };
            signature.push((1, 0, target_key, weight.ptr_key()));
        }
        signature.sort_unstable();
        let next_class = class_by_signature.len() as u32;
        let class = *class_by_signature
            .entry((final_key, signature))
            .or_insert(next_class);
        class_by_state[state_id] = Some(class);
        processed += 1;
        for &pred in &predecessors[state_id] {
            internal_outdegree[pred] -= 1;
            if internal_outdegree[pred] == 0 {
                queue.push_back(pred);
            }
        }
    }
    eprintln!(
        "[glrmask/profile][parser_negative_suffix_hashcons_potential] eligible_states={} acyclic_processed={} cyclic_or_blocked={} exact_classes={} removable_states={} compression={:.2}",
        eligible.iter().filter(|&&value| value).count(),
        processed,
        eligible.iter().filter(|&&value| value).count().saturating_sub(processed),
        class_by_signature.len(),
        processed.saturating_sub(class_by_signature.len()),
        processed as f64 / class_by_signature.len().max(1) as f64,
    );
}

fn profile_parser_nwa_reachability(nwa: &NWA, phase: &str) {
    if !compile_profile_enabled() {
        return;
    }
    let mut reachable = vec![false; nwa.states().len()];
    let mut queue = VecDeque::new();
    for &start in nwa.start_states() {
        if (start as usize) < reachable.len() {
            queue.push_back(start);
        }
    }
    while let Some(state_id) = queue.pop_front() {
        let index = state_id as usize;
        if index >= reachable.len() || reachable[index] {
            continue;
        }
        reachable[index] = true;
        let state = &nwa.states()[index];
        for targets in state.transitions.values() {
            for (target, weight) in targets {
                if !weight.is_empty() {
                    queue.push_back(*target);
                }
            }
        }
        for (target, weight) in &state.epsilons {
            if !weight.is_empty() {
                queue.push_back(*target);
            }
        }
    }
    let reachable_count = reachable.iter().filter(|&&value| value).count();
    eprintln!(
        "[glrmask/profile][parser_nwa_reachability] phase={} states={} reachable={} unreachable={} compression={:.2}",
        phase,
        nwa.states().len(),
        reachable_count,
        nwa.states().len().saturating_sub(reachable_count),
        nwa.states().len() as f64 / reachable_count.max(1) as f64,
    );
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

fn bundle_topology_signature(bundle: &TerminalBundle) -> BundleTopologySignature {
    let mut groups = FxHashMap::<Weight, Vec<TerminalID>>::default();
    for (&terminal, weight) in bundle {
        groups.entry(weight.clone()).or_default().push(terminal);
    }
    let mut groups = groups.into_values().collect::<Vec<_>>();
    for group in &mut groups {
        group.sort_unstable();
    }
    groups.sort_unstable();
    groups
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
            branches.push(Branch {
                target,
                bundle_id,
                entry_weight: Weight::all(),
                cross_target_group_id: None,
            });
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
        cross_target_groups: Vec::new(),
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
    (0..table.num_states)
        .map(|parser_top| {
            parts
                .get(&(parser_top as i32))
                .map(|weights| Weight::union_all(weights.iter()))
                .unwrap_or_else(Weight::empty)
        })
        .collect()
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

fn trim_unreachable_nwa(nwa: NWA) -> NWA {
    if nwa.states().is_empty() {
        return nwa;
    }

    let mut reachable = vec![false; nwa.states().len()];
    let mut queue = VecDeque::new();
    for &start in nwa.start_states() {
        if (start as usize) < reachable.len() {
            queue.push_back(start);
        }
    }
    while let Some(state_id) = queue.pop_front() {
        let state_index = state_id as usize;
        if state_index >= reachable.len() || reachable[state_index] {
            continue;
        }
        reachable[state_index] = true;
        let state = &nwa.states()[state_index];
        for targets in state.transitions.values() {
            for (target, weight) in targets {
                if !weight.is_empty() && (*target as usize) < reachable.len() {
                    queue.push_back(*target);
                }
            }
        }
        for (target, weight) in &state.epsilons {
            if !weight.is_empty() && (*target as usize) < reachable.len() {
                queue.push_back(*target);
            }
        }
    }

    if reachable.iter().all(|&value| value) {
        return nwa;
    }

    let (old_states, old_starts) = nwa.into_parts();
    let mut remap = vec![u32::MAX; old_states.len()];
    let mut new_states = Vec::with_capacity(reachable.iter().filter(|&&value| value).count());
    for (old_id, state) in old_states.into_iter().enumerate() {
        if reachable[old_id] {
            remap[old_id] = new_states.len() as u32;
            new_states.push(state);
        }
    }

    for state in &mut new_states {
        for targets in state.transitions.values_mut() {
            targets.retain_mut(|(target, weight)| {
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
        state.transitions.retain(|_, targets| !targets.is_empty());
        state.epsilons.retain_mut(|(target, weight)| {
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

    let starts = old_starts
        .into_iter()
        .filter_map(|start| {
            remap
                .get(start as usize)
                .copied()
                .filter(|&mapped| mapped != u32::MAX)
        })
        .collect();
    NWA::from_parts(new_states, starts)
}

fn trim_resolved_parser_nwa_enabled(state_count: usize) -> bool {
    std::env::var("GLRMASK_TRIM_RESOLVED_PARSER_NWA")
        .ok()
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
        && state_count
            >= std::env::var("GLRMASK_TRIM_RESOLVED_PARSER_NWA_MIN_STATES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(65_536)
}

fn direct_parser_read_projection_enabled() -> bool {
    std::env::var("GLRMASK_DIRECT_PARSER_READ_PROJECTION")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn direct_parser_read_projection_state_limit() -> usize {
    std::env::var("GLRMASK_DIRECT_PARSER_READ_PROJECTION_MAX_STATES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(2_000_000)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ReadProjectionConfig {
    state: u32,
    pending_pushes: SmallVec<[i32; 8]>,
}

fn direct_parser_read_projection(nwa: &NWA) -> Option<NWA> {
    let max_states = direct_parser_read_projection_state_limit();
    let mut projected = NWA::new(0, 0);
    let mut config_to_state = FxHashMap::<ReadProjectionConfig, u32>::default();
    let mut configs = Vec::<ReadProjectionConfig>::new();
    let mut worklist = VecDeque::<u32>::new();
    let mut projected_starts = Vec::new();
    let mut max_pending_depth = 0usize;

    let intern = |config: ReadProjectionConfig,
                  projected: &mut NWA,
                  config_to_state: &mut FxHashMap<ReadProjectionConfig, u32>,
                  configs: &mut Vec<ReadProjectionConfig>,
                  worklist: &mut VecDeque<u32>|
     -> Option<u32> {
        if let Some(&state) = config_to_state.get(&config) {
            return Some(state);
        }
        if configs.len() >= max_states {
            return None;
        }
        let state = projected.add_state();
        config_to_state.insert(config.clone(), state);
        configs.push(config);
        worklist.push_back(state);
        Some(state)
    };

    for &start in nwa.start_states() {
        let config = ReadProjectionConfig {
            state: start,
            pending_pushes: SmallVec::new(),
        };
        let start_state = intern(
            config,
            &mut projected,
            &mut config_to_state,
            &mut configs,
            &mut worklist,
        )?;
        projected_starts.push(start_state);
    }
    projected_starts.sort_unstable();
    projected_starts.dedup();
    projected.set_start_states(projected_starts);

    while let Some(projected_state) = worklist.pop_front() {
        let config = configs[projected_state as usize].clone();
        max_pending_depth = max_pending_depth.max(config.pending_pushes.len());
        let Some(source) = nwa.states().get(config.state as usize) else {
            continue;
        };
        if let Some(final_weight) = source.final_weight.as_ref().filter(|weight| !weight.is_empty()) {
            projected.set_final_weight(projected_state, final_weight.clone());
        }

        for (target, weight) in &source.epsilons {
            if weight.is_empty() {
                continue;
            }
            let target_config = ReadProjectionConfig {
                state: *target,
                pending_pushes: config.pending_pushes.clone(),
            };
            let target_state = intern(
                target_config,
                &mut projected,
                &mut config_to_state,
                &mut configs,
                &mut worklist,
            )?;
            projected.add_epsilon(projected_state, target_state, weight.clone());
        }

        for (&label, targets) in &source.transitions {
            if is_negative_label(label) {
                let pushed = negative_to_positive_label(label);
                for (target, weight) in targets {
                    if weight.is_empty() {
                        continue;
                    }
                    let mut pending_pushes = config.pending_pushes.clone();
                    pending_pushes.push(pushed);
                    let target_config = ReadProjectionConfig {
                        state: *target,
                        pending_pushes,
                    };
                    let target_state = intern(
                        target_config,
                        &mut projected,
                        &mut config_to_state,
                        &mut configs,
                        &mut worklist,
                    )?;
                    projected.add_epsilon(projected_state, target_state, weight.clone());
                }
                continue;
            }

            if let Some(&pending_top) = config.pending_pushes.last() {
                if label != pending_top && label != DEFAULT_LABEL {
                    continue;
                }
                for (target, weight) in targets {
                    if weight.is_empty() {
                        continue;
                    }
                    let mut pending_pushes = config.pending_pushes.clone();
                    pending_pushes.pop();
                    let target_config = ReadProjectionConfig {
                        state: *target,
                        pending_pushes,
                    };
                    let target_state = intern(
                        target_config,
                        &mut projected,
                        &mut config_to_state,
                        &mut configs,
                        &mut worklist,
                    )?;
                    projected.add_epsilon(projected_state, target_state, weight.clone());
                }
                continue;
            }

            for (target, weight) in targets {
                if weight.is_empty() {
                    continue;
                }
                let target_config = ReadProjectionConfig {
                    state: *target,
                    pending_pushes: SmallVec::new(),
                };
                let target_state = intern(
                    target_config,
                    &mut projected,
                    &mut config_to_state,
                    &mut configs,
                    &mut worklist,
                )?;
                projected.add_transition(projected_state, label, target_state, weight.clone());
            }
        }
    }

    if compile_profile_enabled() {
        eprintln!(
            "[glrmask/profile][parser_direct_read_projection] input_states={} output_states={} output_transitions={} max_pending_depth={} state_limit={}",
            nwa.states().len(),
            projected.states().len(),
            projected.num_transitions(),
            max_pending_depth,
            max_states,
        );
    }
    Some(projected)
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

enum RawPossibleOutgoingIds {
    Empty,
    All,
    Some(Vec<u32>),
}

fn snapshot_raw_possible_outgoing_ids(
    parser_nwa: &NWA,
    num_parser_states: u32,
) -> Vec<RawPossibleOutgoingIds> {
    parser_nwa
        .states()
        .iter()
        .map(|state| {
            let mut ids = Vec::new();
            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    return RawPossibleOutgoingIds::All;
                }
                if let Some(parser_state_id) = parser_state_label(label, num_parser_states) {
                    ids.push(parser_state_id);
                }
            }
            if ids.is_empty() {
                RawPossibleOutgoingIds::Empty
            } else {
                RawPossibleOutgoingIds::Some(ids)
            }
        })
        .collect()
}

fn build_possible_outgoing_ids_from_raw(
    state_outgoing_ids: &[RawPossibleOutgoingIds],
    state_supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<PossibleOutgoingIds> {
    let num_parser_states = num_parser_states as usize;
    let all_parser_states = BitSet::all(num_parser_states);
    state_supports
        .iter()
        .map(|support| {
            if support.len() == 1 {
                let state_id = support[0] as usize;
                return match state_outgoing_ids.get(state_id) {
                    Some(RawPossibleOutgoingIds::Empty) => PossibleOutgoingIds::Empty,
                    Some(RawPossibleOutgoingIds::All) => PossibleOutgoingIds::All,
                    Some(RawPossibleOutgoingIds::Some(ids)) => {
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
                    RawPossibleOutgoingIds::Empty => {}
                    RawPossibleOutgoingIds::All => return PossibleOutgoingIds::All,
                    RawPossibleOutgoingIds::Some(state_ids) => {
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

fn drop_dead_leaf_targets_preserving_possible(
    parser_nwa: &mut NWA,
    num_parser_states: u32,
) -> (Vec<RawPossibleOutgoingIds>, usize, usize) {
    let raw_possible = snapshot_raw_possible_outgoing_ids(parser_nwa, num_parser_states);
    let dead_leaf = parser_nwa
        .states()
        .iter()
        .map(|state| {
            state
                .final_weight
                .as_ref()
                .is_none_or(Weight::is_empty)
                && state.transitions.is_empty()
                && state.epsilons.is_empty()
        })
        .collect::<Vec<_>>();
    let dead_count = dead_leaf.iter().filter(|&&is_dead| is_dead).count();
    let mut removed = 0usize;
    for state in parser_nwa.states_mut() {
        state.transitions.retain(|_, targets| {
            let before = targets.len();
            targets.retain(|(target, _)| !dead_leaf[*target as usize]);
            removed += before - targets.len();
            !targets.is_empty()
        });
        let before = state.epsilons.len();
        state.epsilons.retain(|(target, _)| !dead_leaf[*target as usize]);
        removed += before - state.epsilons.len();
    }
    (raw_possible, dead_count, removed)
}

fn build_possible_outgoing_ids_by_state(
    parser_nwa: &NWA,
    state_supports: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<PossibleOutgoingIds> {
    let state_outgoing_ids = snapshot_raw_possible_outgoing_ids(parser_nwa, num_parser_states);
    build_possible_outgoing_ids_from_raw(&state_outgoing_ids, state_supports, num_parser_states)
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
            let contribution = weight_ops.intersection(&current_weight, edge_weight);
            if contribution.is_empty() {
                continue;
            }
            let target_idx = *target as usize;
            if let Some(existing) = &weight_by_state[target_idx] {
                if !contribution.is_subset(existing) {
                    weight_by_state[target_idx] = Some(weight_ops.union(existing, &contribution));
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
    determinize_with_supports_impl(nwa, dense_positive_label_limit, None, false)
}

fn determinize_with_supports_canonical_wide(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
    canonical_min_len: usize,
) -> DeterminizedDwaWithSupports {
    determinize_with_supports_impl(
        nwa,
        dense_positive_label_limit,
        Some(canonical_min_len),
        false,
    )
}

fn determinize_with_weighted_supports(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
) -> DeterminizedDwaWithSupports {
    determinize_with_supports_impl(nwa, dense_positive_label_limit, None, true)
}

fn determinize_with_weighted_supports_canonical_wide(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
    canonical_min_len: usize,
) -> DeterminizedDwaWithSupports {
    determinize_with_supports_impl(
        nwa,
        dense_positive_label_limit,
        Some(canonical_min_len),
        true,
    )
}

fn determinize_with_supports_impl(
    nwa: &NWA,
    dense_positive_label_limit: Option<u32>,
    canonical_min_len_override: Option<usize>,
    retain_weighted_supports: bool,
) -> DeterminizedDwaWithSupports {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

      #[derive(Default)]
      struct UnionAllCache {
        entries: FxHashMap<SmallVec<[usize; 16]>, Weight>,
        canonical_min_len: usize,
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
            // Preserve that cheap ordered key for normal rows. Canonicalize only
            // sufficiently wide rows, where permutation reuse can amortize the
            // extra sort/dedup work.
            if key.len() >= self.canonical_min_len {
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
    let mut weighted_supports = retain_weighted_supports.then(|| vec![Vec::new()]);

    let mut start_subset = FxHashMap::default();
    for &state_id in nwa.start_states() {
        let existing = start_subset.get(&state_id).cloned().unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    epsilon_closure(&mut weight_by_state, &mut closure_queue, &mut start_subset);
    if start_subset.is_empty() {
        return DeterminizedDwaWithSupports {
            dwa,
            supports,
            weighted_supports,
        };
    }

    canonicalize_into(&start_subset, &mut canon_buf);
    supports[0] = canon_buf.iter().map(|(state_id, _)| *state_id).collect();
    if let Some(weighted_supports) = weighted_supports.as_mut() {
        weighted_supports[0] = canon_buf.clone();
    }

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
    let canonical_union_min_len = canonical_min_len_override.unwrap_or_else(|| {
        std::env::var("GLRMASK_UNION_CACHE_CANONICAL_MIN_LEN")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or_else(|| {
                if std::env::var_os("GLRMASK_DISABLE_ORDERED_UNION_CACHE_KEY").is_some() {
                    0
                } else {
                    usize::MAX
                }
            })
    });
    let mut union_cache = UnionAllCache {
        canonical_min_len: canonical_union_min_len,
        profile_enabled: detail.is_some(),
        ..UnionAllCache::default()
    };

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
            merge_sorted_target_contributions(
                &mut contribs,
                &mut intersection_cache,
                detail.as_mut(),
            );
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
                        let weighted_subset = vec![(*only_state, only_weight.clone())];
                        worklist.push_back((new_state, weighted_subset.clone()));
                        supports.push(vec![*only_state]);
                        if let Some(weighted_supports) = weighted_supports.as_mut() {
                            weighted_supports.push(weighted_subset);
                        }
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
                    let weighted_subset = canon.to_vec();
                    worklist.push_back((new_state, weighted_subset.clone()));
                    supports.push(vec![*only_state]);
                    if let Some(weighted_supports) = weighted_supports.as_mut() {
                        weighted_supports.push(weighted_subset);
                    }
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
                    let weighted_subset = canon.to_vec();
                    worklist.push_back((new_state, weighted_subset.clone()));
                    supports.push(canon.iter().map(|(sid, _)| *sid).collect());
                    if let Some(weighted_supports) = weighted_supports.as_mut() {
                        weighted_supports.push(weighted_subset);
                    }
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

    if compile_profile_enabled() {
        let mut support_counts = FxHashMap::<Vec<u32>, usize>::default();
        for support in &supports {
            *support_counts.entry(support.clone()).or_default() += 1;
        }
        let distinct = support_counts.len();
        let repeated_states = supports.len().saturating_sub(distinct);
        let max_multiplicity = support_counts.values().copied().max().unwrap_or(0);
        let repeated_signatures = support_counts.values().filter(|&&count| count > 1).count();
        eprintln!(
            "[glrmask/profile][parser_support_signatures] states={} distinct={} repeated_states={} repeated_signatures={} max_multiplicity={}",
            supports.len(),
            distinct,
            repeated_states,
            repeated_signatures,
            max_multiplicity,
        );
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
    DeterminizedDwaWithSupports {
        dwa,
        supports,
        weighted_supports,
    }
}

fn determinize_parser_dwa_with_fallbacks_impl(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
    normalize_singletons: bool,
    reuse_input_singletons: bool,
) -> DWA {
    fn subset_key(entries: &[(u32, Weight)]) -> Vec<(u32, usize)> {
        entries.iter().map(|(sid, w)| (*sid, w.ptr_key())).collect()
    }

    let dense_label_limit = num_parser_states as usize;
    let reuse_input_singletons = reuse_input_singletons && normalize_singletons;
    let mut result = if reuse_input_singletons {
        dwa.clone()
    } else {
        DWA::new(0, 0)
    };

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
    if reuse_input_singletons {
        normalized_singleton_subsets.extend(
            (0..dwa.states().len()).map(|state_id| (state_id as u32, state_id as u32)),
        );
    } else {
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
    }
    let mut worklist: VecDeque<(u32, Vec<(u32, Weight)>)> = VecDeque::new();
    if reuse_input_singletons {
        for (state_id, state) in dwa.states().iter().enumerate() {
            if state.transitions.contains_key(&DEFAULT_LABEL) {
                worklist.push_back((
                    state_id as u32,
                    vec![(state_id as u32, normalized_singleton_weight.clone())],
                ));
            }
        }
    } else {
        worklist.push_back((result.start_state(), canon_buf.clone()));
    }

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
    let fallback_shape_profile = ParserDwaDeterminizeDetail::enabled();
    let mut fallback_singleton_states_processed = 0usize;
    let mut fallback_pair_states_processed = 0usize;
    let mut fallback_larger_states_processed = 0usize;
    let mut fallback_singleton_result_states_created = 0usize;
    let mut fallback_pair_result_states_created = 0usize;
    let mut fallback_larger_result_states_created = 0usize;

    while let Some((from_state, subset_entries)) = worklist.pop_front() {
        dense_default_all_raw_targets.clear();
        if fallback_shape_profile {
            match subset_entries.len() {
                1 => fallback_singleton_states_processed += 1,
                2 => fallback_pair_states_processed += 1,
                _ => fallback_larger_states_processed += 1,
            }
        }
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

        // In seeded mode singleton states retain their original IDs. Only
        // singleton rows containing DEFAULT need rebuilding; rows without a
        // fallback edge remain valid verbatim. Pair states are newly appended
        // and start empty.
        if reuse_input_singletons && subset_entries.len() == 1 {
            result.states_mut()[from_state as usize].transitions.clear();
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
                if fallback_shape_profile {
                    fallback_singleton_result_states_created += 1;
                }
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
                        if fallback_shape_profile {
                            fallback_singleton_result_states_created += 1;
                        }
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
                        if fallback_shape_profile {
                            fallback_singleton_result_states_created += 1;
                        }
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
                    if fallback_shape_profile {
                        match contribs.len() {
                            2 => fallback_pair_result_states_created += 1,
                            _ => fallback_larger_result_states_created += 1,
                        }
                    }
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
    if fallback_shape_profile {
        eprintln!(
            "[glrmask/profile][parser_fallback_subset_shape] input_states={} output_states={} processed_singletons={} processed_pairs={} processed_larger={} created_singletons={} created_pairs={} created_larger={} normalized_singleton_states={} weighted_singleton_states={} generic_subset_states={}",
            dwa.states().len(),
            result.states().len(),
            fallback_singleton_states_processed,
            fallback_pair_states_processed,
            fallback_larger_states_processed,
            fallback_singleton_result_states_created,
            fallback_pair_result_states_created,
            fallback_larger_result_states_created,
            normalized_singleton_subsets.len(),
            weighted_singleton_subsets.len(),
            subset_map.len(),
        );
    }

    result
}

fn determinize_parser_dwa_with_fallbacks(
    dwa: &DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) -> DWA {
    let reuse_input_singletons = std::env::var("GLRMASK_SEEDED_FALLBACK_NORMALIZATION")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    determinize_parser_dwa_with_fallbacks_impl(
        dwa,
        possible_by_state,
        num_parser_states,
        true,
        reuse_input_singletons,
    )
}

fn optimize_parser_dwa_defaults(
    dwa: &mut DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
) {
    optimize_parser_dwa_defaults_impl(dwa, possible_by_state, num_parser_states, None);
}

fn optimize_parser_dwa_defaults_with_raw_identity(
    dwa: &mut DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
    raw_identity: &mut RawSupportIdentityOracle<'_>,
) {
    optimize_parser_dwa_defaults_impl(
        dwa,
        possible_by_state,
        num_parser_states,
        Some(raw_identity),
    );
}

fn optimize_parser_dwa_defaults_impl(
    dwa: &mut DWA,
    possible_by_state: &[PossibleOutgoingIds],
    num_parser_states: u32,
    mut raw_identity: Option<&mut RawSupportIdentityOracle<'_>>,
) {
    let profile_candidates = std::env::var_os("GLRMASK_PROFILE_DEFAULT_SYNTHESIS_CANDIDATES").is_some();
    let mut profile_multi_possible = 0usize;
    let mut profile_complete_explicit = 0usize;
    let mut profile_shared_target = 0usize;
    let mut profile_insertions = 0usize;
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
            if profile_candidates {
                profile_multi_possible += 1;
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

            if profile_candidates {
                profile_complete_explicit += 1;
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
            if profile_candidates {
                profile_shared_target += 1;
            }
            if let Some(raw_identity) = raw_identity.as_deref_mut()
                && !raw_identity.labels_share_raw_target(
                    state_id as u32,
                    possible_ids,
                    num_parser_states,
                )
            {
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
                            if profile_candidates {
                                profile_insertions += 1;
                            }
                        }
                    }
                }
                std::collections::btree_map::Entry::Vacant(vac) => {
                    vac.insert((target, default_weight));
                    changed = true;
                    if profile_candidates {
                        profile_insertions += 1;
                    }
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
    if profile_candidates {
        eprintln!(
            "[glrmask/profile][parser_default_synthesis_candidates] states={} multi_possible_visits={} complete_explicit_visits={} shared_target_visits={} insertions={}",
            dwa.states().len(),
            profile_multi_possible,
            profile_complete_explicit,
            profile_shared_target,
            profile_insertions,
        );
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
    subtract_final_weights_from_outgoing_dwa_impl(dwa, rayon::current_num_threads() > 1);
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

fn append_bundle_redirecting_finals_to_targets(
    arena: &mut NWA,
    bundle: &NWA,
    target_continuations: &[(u32, Weight)],
) -> NwaBody {
    let offset = arena.states().len() as u32;
    let body = arena.append_with_body(bundle);
    let appended_len = bundle.states().len();

    for state_id in offset as usize..offset as usize + appended_len {
        let Some(final_weight) = arena.states_mut()[state_id].final_weight.take() else {
            continue;
        };
        if final_weight.is_empty() {
            continue;
        }
        for (continuation_state, target_gate) in target_continuations {
            let routed = final_weight.intersection(target_gate);
            if !routed.is_empty() {
                arena.add_epsilon(state_id as u32, *continuation_state, routed);
            }
        }
    }

    body
}


#[derive(Clone, Copy, Debug, Default)]
struct PrepushReconstructDecision {
    candidate: bool,
    admitted: bool,
    max_template: usize,
    sum_template: usize,
    predicted_reconstructed_states: usize,
}

fn prepush_reconstruct_decision(
    enabled: bool,
    bundle: &TerminalBundle,
    templates: &Templates,
) -> PrepushReconstructDecision {
    if !enabled || bundle.len() <= 1 {
        return PrepushReconstructDecision::default();
    }
    let sizes = bundle
        .keys()
        .filter_map(|terminal| templates.by_terminal_nwa.get(terminal))
        .map(|template| template.states().len())
        .collect::<Vec<_>>();
    let max_template = sizes.iter().copied().max().unwrap_or(0);
    let sum_template = sizes.iter().sum::<usize>();
    let threshold = std::env::var("GLRMASK_PREPUSH_RECONSTRUCT_MIN_TEMPLATE_SUM")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let metric = std::env::var("GLRMASK_PREPUSH_RECONSTRUCT_METRIC")
        .ok()
        .unwrap_or_else(|| "sum".to_string());
    let value = if metric.eq_ignore_ascii_case("max") {
        max_template
    } else {
        sum_template
    };
    let candidate = threshold == 0 || value >= threshold;
    if !candidate {
        return PrepushReconstructDecision {
            candidate: false,
            admitted: false,
            max_template,
            sum_template,
            predicted_reconstructed_states: 0,
        };
    }

    let reconstructed_state_cap = std::env::var("GLRMASK_PREPUSH_RECONSTRUCT_MAX_STATES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let unconditional_template_min =
        std::env::var("GLRMASK_PREPUSH_RECONSTRUCT_UNCONDITIONAL_TEMPLATE_MIN")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
    let unconditional = unconditional_template_min != 0
        && max_template >= unconditional_template_min;
    let predicted_reconstructed_states = if reconstructed_state_cap != 0 && !unconditional {
        templates
            .census_prepush_reconstructed_bundle(bundle)
            .reconstructed_states()
    } else {
        0
    };
    let admitted = reconstructed_state_cap == 0
        || unconditional
        || predicted_reconstructed_states <= reconstructed_state_cap;
    PrepushReconstructDecision {
        candidate,
        admitted,
        max_template,
        sum_template,
        predicted_reconstructed_states,
    }
}

fn use_prepush_reconstructed_for_bundle(
    enabled: bool,
    bundle: &TerminalBundle,
    templates: &Templates,
) -> bool {
    prepush_reconstruct_decision(enabled, bundle, templates).admitted
}

fn append_cross_target_branch_fragment(
    arena: &mut NWA,
    summaries: &StateSummaries,
    templates: &Templates,
    built_bundle_cache: &mut [Option<Arc<NWA>>],
    group_id: usize,
    continuation_states: &[u32],
    productive: &[bool],
    use_prepush_reconstructed_bundles: bool,
    compose_detail: Option<&mut ParserDwaComposeDetailProfile>,
) -> Option<NwaBody> {
    let group = summaries.cross_target_groups.get(group_id)?;
    let bundle_id = group.bundle_id;
    let bundle = summaries.unique_bundles.get(bundle_id)?;
    if !summaries.bundle_accepts.get(bundle_id).copied().unwrap_or(false) {
        return None;
    }

    if built_bundle_cache[bundle_id].is_none() {
        if use_prepush_reconstructed_for_bundle(
            use_prepush_reconstructed_bundles,
            bundle,
            templates,
        ) {
            built_bundle_cache[bundle_id] =
                Some(Arc::new(templates.build_prepush_reconstructed_bundle(bundle)));
        } else if let Some(detail) = compose_detail {
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
            built_bundle_cache[bundle_id] = Some(Arc::new(bundle_nwa));
        } else {
            built_bundle_cache[bundle_id] = Some(Arc::new(templates.build_bundle(bundle)));
        }
    }
    let bundle_nwa = built_bundle_cache[bundle_id]
        .as_ref()
        .expect("cross-target canonical bundle cache entry just initialized");
    let target_continuations = group
        .target_gates
        .iter()
        .filter_map(|(target, gate)| {
            productive
                .get(*target as usize)
                .copied()
                .unwrap_or(false)
                .then(|| (continuation_states[*target as usize], gate.clone()))
        })
        .filter(|(continuation, _)| *continuation != u32::MAX)
        .collect::<Vec<_>>();
    if target_continuations.is_empty() {
        return None;
    }
    Some(append_bundle_redirecting_finals_to_targets(
        arena,
        bundle_nwa.as_ref(),
        &target_continuations,
    ))
}

fn append_branch_fragment(
    arena: &mut NWA,
    summaries: &StateSummaries,
    templates: &Templates,
    built_bundle_cache: &mut [Option<Arc<NWA>>],
    bundle_id: usize,
    continuation_state: u32,
    use_prepush_reconstructed_bundles: bool,
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
        if use_prepush_reconstructed_for_bundle(
            use_prepush_reconstructed_bundles,
            bundle,
            templates,
        ) {
            built_bundle_cache[bundle_id] =
                Some(Arc::new(templates.build_prepush_reconstructed_bundle(bundle)));
        } else if let Some(detail) = compose_detail {
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
                "[glrmask/profile][parser_bundle] bundle_id={} terminals={} weight_groups={} overlap_components={} largest_overlap_component={} single_entry_weights={} single_tsid_weights={} total_weight_outer_ranges={} build_group_dfas_ms={:.3} union_groups_ms={:.3} determinize_bundle_ms={:.3} det_pop_ms={:.3} det_alive_ms={:.3} det_final_ms={:.3} det_collect_labels_ms={:.3} det_next_state_ms={:.3} det_edge_weight_ms={:.3} det_lookup_ms={:.3} det_add_transition_ms={:.3} det_states={} det_labels={} det_transitions={} det_edge_subset_total={} det_edge_subset_max={} det_edge_cache_hits={} det_edge_cache_misses={} minimize_ms={:.3} minimize_skipped={} dwa_to_nwa_ms={:.3} total_ms={:.3} result_dwa_states={} result_dwa_transitions={} result_nwa_states={} result_nwa_transitions={} negative_only_states={} positive_only_states={} mixed_label_states={} unlabeled_states={} negative_transitions={} positive_transitions={} truncated_reachable_states={} truncated_push_frontier_states={} truncated_edges_traversed={} prepush_states={} prepush_input_transitions={} prepush_output_edges={} prepush_output_sites={} prepush_output_programs={} prepush_core_states={} prepush_frontier_payloads={} prepush_frontier_final_payloads={} prepush_frontier_push_edges={} prepush_census_ms={:.3} prepush_program_sequences={} prepush_programs_multisequence={} prepush_max_sequences_per_program={} prepush_max_push_depth={}",
                bundle_id,
                bundle_profile.input_terminals,
                bundle_profile.weight_groups,
                bundle_profile.overlap_components,
                bundle_profile.largest_overlap_component,
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
                bundle_profile.result_negative_only_states,
                bundle_profile.result_positive_only_states,
                bundle_profile.result_mixed_label_states,
                bundle_profile.result_unlabeled_states,
                bundle_profile.result_negative_transitions,
                bundle_profile.result_positive_transitions,
                bundle_profile.truncated_reachable_states,
                bundle_profile.truncated_push_frontier_states,
                bundle_profile.truncated_edges_traversed,
                bundle_profile.prepush_states,
                bundle_profile.prepush_input_transitions,
                bundle_profile.prepush_output_edges,
                bundle_profile.prepush_output_sites,
                bundle_profile.prepush_output_programs,
                bundle_profile.prepush_core_states,
                bundle_profile.prepush_frontier_payloads,
                bundle_profile.prepush_frontier_final_payloads,
                bundle_profile.prepush_frontier_push_edges,
                bundle_profile.prepush_census_ms,
                bundle_profile.prepush_program_sequences,
                bundle_profile.prepush_programs_multisequence,
                bundle_profile.prepush_max_sequences_per_program,
                bundle_profile.prepush_max_push_depth,
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


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DirectPrepushNode {
    Continuation(u32),
    BundleCore {
        bundle_id: usize,
        target: u32,
        core_state: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DirectPrepushConfig {
    node: DirectPrepushNode,
    pending: SmallVec<[u32; 8]>,
}

fn append_pending(
    pending: &SmallVec<[u32; 8]>,
    pushes: &[u32],
) -> SmallVec<[u32; 8]> {
    let mut next = pending.clone();
    next.extend_from_slice(pushes);
    next
}

fn consume_pending(
    pending: &SmallVec<[u32; 8]>,
    label: i32,
) -> Option<SmallVec<[u32; 8]>> {
    let Some(&top) = pending.last() else {
        return Some(pending.clone());
    };
    if label != DEFAULT_LABEL && (label < 0 || label as u32 != top) {
        return None;
    }
    let mut next = pending.clone();
    next.pop();
    Some(next)
}

fn weighted_prepush_bundles_fingerprint(
    bundles: &[Option<Arc<WeightedPrepushBundle>>],
) -> u64 {
    #[inline]
    fn mix(hash: u64, value: u64) -> u64 {
        (hash ^ value)
            .wrapping_mul(0x100000001b3)
            .rotate_left(9)
            .wrapping_add(0x9e3779b97f4a7c15)
    }

    let mut hash = 0xcbf29ce484222325u64;
    for (bundle_id, bundle) in bundles.iter().enumerate() {
        let Some(bundle) = bundle else { continue };
        hash = mix(hash, bundle_id as u64);
        hash = mix(hash, bundle.states.len() as u64);
        for state in &bundle.states {
            hash = mix(hash, state.final_weight.structural_hash_cached());
            hash = mix(hash, state.outputs.len() as u64);
            for output in &state.outputs {
                hash = mix(hash, output.pushes.len() as u64);
                for &push in &output.pushes {
                    hash = mix(hash, push as u64);
                }
                hash = mix(hash, output.weight.structural_hash_cached());
            }
            hash = mix(hash, state.transitions.len() as u64);
            for (&label, transition) in &state.transitions {
                hash = mix(hash, label as u32 as u64);
                match transition {
                    WeightedPrepushTarget::Core { target, weight } => {
                        hash = mix(hash, 0);
                        hash = mix(hash, *target as u64);
                        hash = mix(hash, weight.structural_hash_cached());
                    }
                    WeightedPrepushTarget::Outputs(outputs) => {
                        hash = mix(hash, 1);
                        hash = mix(hash, outputs.len() as u64);
                        for output in outputs {
                            hash = mix(hash, output.pushes.len() as u64);
                            for &push in &output.pushes {
                                hash = mix(hash, push as u64);
                            }
                            hash = mix(hash, output.weight.structural_hash_cached());
                        }
                    }
                }
            }
        }
    }
    hash
}


fn direct_prepush_pending_demand(
    node: &DirectPrepushNode,
    summaries: &StateSummaries,
    productive: &[bool],
    bundles: &[Option<Arc<WeightedPrepushBundle>>],
    memo: &mut FxHashMap<DirectPrepushNode, usize>,
    visiting: &mut rustc_hash::FxHashSet<DirectPrepushNode>,
) -> Option<usize> {
    if let Some(&cached) = memo.get(node) {
        return Some(cached);
    }
    if !visiting.insert(node.clone()) {
        // A genuine cycle would permit an unbounded number of future terminal
        // effects in one model token. The current finite-vocabulary terminal
        // automata should be acyclic; decline the quotient if that invariant is
        // ever violated rather than silently truncating pending writes.
        return None;
    }
    let mut best = 0usize;
    match *node {
        DirectPrepushNode::Continuation(state_id) => {
            let state = summaries.states.get(state_id as usize)?;
            // Token acceptance itself observes no pending writes. Other live
            // suffixes may, so take the maximum over all productive branches.
            for (target, weight) in &state.epsilon_branches {
                if weight.is_empty()
                    || !productive.get(*target as usize).copied().unwrap_or(false)
                {
                    continue;
                }
                best = best.max(direct_prepush_pending_demand(
                    &DirectPrepushNode::Continuation(*target),
                    summaries,
                    productive,
                    bundles,
                    memo,
                    visiting,
                )?);
            }
            for branch in &state.branches {
                if !productive
                    .get(branch.target as usize)
                    .copied()
                    .unwrap_or(false)
                    || !summaries
                        .bundle_accepts
                        .get(branch.bundle_id)
                        .copied()
                        .unwrap_or(false)
                {
                    continue;
                }
                let Some(bundle) = bundles.get(branch.bundle_id).and_then(Option::as_ref) else {
                    continue;
                };
                if bundle.states.is_empty() {
                    continue;
                }
                best = best.max(direct_prepush_pending_demand(
                    &DirectPrepushNode::BundleCore {
                        bundle_id: branch.bundle_id,
                        target: branch.target,
                        core_state: 0,
                    },
                    summaries,
                    productive,
                    bundles,
                    memo,
                    visiting,
                )?);
            }
        }
        DirectPrepushNode::BundleCore {
            bundle_id,
            target,
            core_state,
        } => {
            let bundle = bundles.get(bundle_id).and_then(Option::as_ref)?;
            let state = bundle.states.get(core_state as usize)?;
            let continuation = DirectPrepushNode::Continuation(target);
            let continuation_demand = if !state.final_weight.is_empty() || !state.outputs.is_empty() {
                Some(direct_prepush_pending_demand(
                    &continuation,
                    summaries,
                    productive,
                    bundles,
                    memo,
                    visiting,
                )?)
            } else {
                None
            };
            if !state.final_weight.is_empty() {
                best = best.max(continuation_demand.unwrap_or(0));
            }
            for output in &state.outputs {
                if output.weight.is_empty() {
                    continue;
                }
                let suffix = continuation_demand.unwrap_or(0);
                best = best.max(suffix.saturating_sub(output.pushes.len()));
            }
            for (&label, transition) in &state.transitions {
                debug_assert!(!is_negative_label(label));
                match transition {
                    WeightedPrepushTarget::Core {
                        target: core_target,
                        weight,
                    } => {
                        if weight.is_empty() {
                            continue;
                        }
                        let suffix = direct_prepush_pending_demand(
                            &DirectPrepushNode::BundleCore {
                                bundle_id,
                                target,
                                core_state: *core_target,
                            },
                            summaries,
                            productive,
                            bundles,
                            memo,
                            visiting,
                        )?;
                        best = best.max(1usize.saturating_add(suffix));
                    }
                    WeightedPrepushTarget::Outputs(outputs) => {
                        let suffix = direct_prepush_pending_demand(
                            &continuation,
                            summaries,
                            productive,
                            bundles,
                            memo,
                            visiting,
                        )?;
                        for output in outputs {
                            if output.weight.is_empty() {
                                continue;
                            }
                            best = best.max(
                                1usize.saturating_add(suffix.saturating_sub(output.pushes.len())),
                            );
                        }
                    }
                }
            }
        }
    }
    visiting.remove(node);
    memo.insert(node.clone(), best);
    Some(best)
}

fn build_direct_prepush_pending_nwa(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) -> Option<(NWA, usize, usize, usize)> {
    if summaries
        .states
        .iter()
        .flat_map(|state| state.branches.iter())
        .any(|branch| branch.cross_target_group_id.is_some())
    {
        return None;
    }

    let mut used_bundles = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in summaries.states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            if productive
                .get(branch.target as usize)
                .copied()
                .unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                used_bundles[branch.bundle_id] = true;
            }
        }
    }
    let build_started_at = Instant::now();
    let bundles = templates.build_weighted_prepush_bundles_cached(
        &summaries.unique_bundles,
        &used_bundles,
    );
    if std::env::var_os("GLRMASK_DIAG_DIRECT_PREPUSH_COMPONENT_FINGERPRINT").is_some() {
        eprintln!(
            "[glrmask/profile][parser_direct_prepush_bundle_fingerprint] value={:016x}",
            weighted_prepush_bundles_fingerprint(&bundles),
        );
    }
    let bundle_build_ms = elapsed_ms(build_started_at) as usize;

    let mut nwa = NWA::new(0, 0);
    let mut config_to_state = FxHashMap::<DirectPrepushConfig, u32>::default();
    let mut configs = Vec::<DirectPrepushConfig>::new();
    let mut queue = VecDeque::<u32>::new();
    let ensure = |config: DirectPrepushConfig,
                  nwa: &mut NWA,
                  config_to_state: &mut FxHashMap<DirectPrepushConfig, u32>,
                  configs: &mut Vec<DirectPrepushConfig>,
                  queue: &mut VecDeque<u32>| {
        if let Some(&existing) = config_to_state.get(&config) {
            return existing;
        }
        let state = nwa.add_state();
        config_to_state.insert(config.clone(), state);
        configs.push(config);
        queue.push_back(state);
        state
    };

    let mut start_states = Vec::<u32>::new();
    for &start in &summaries.start_states {
        if !productive.get(start as usize).copied().unwrap_or(false) {
            continue;
        }
        let state = ensure(
            DirectPrepushConfig {
                node: DirectPrepushNode::Continuation(start),
                pending: SmallVec::new(),
            },
            &mut nwa,
            &mut config_to_state,
            &mut configs,
            &mut queue,
        );
        start_states.push(state);
    }
    start_states.sort_unstable();
    start_states.dedup();
    nwa.set_start_states(start_states);

    let mut max_pending = 0usize;
    let mut pending_configs = 0usize;
    let mut edge_count = 0usize;
    while let Some(nwa_state) = queue.pop_front() {
        let config = configs[nwa_state as usize].clone();
        max_pending = max_pending.max(config.pending.len());
        pending_configs += usize::from(!config.pending.is_empty());
        match config.node {
            DirectPrepushNode::Continuation(state_id) => {
                let state = &summaries.states[state_id as usize];
                if let Some(final_weight) = state.final_weight.as_ref().filter(|w| !w.is_empty()) {
                    // Leftover writes do not constrain whether this model-token
                    // path is valid; commit applies them after mask selection.
                    nwa.set_final_weight(nwa_state, final_weight.clone());
                }
                for (target, weight) in &state.epsilon_branches {
                    if weight.is_empty()
                        || !productive.get(*target as usize).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    let target_state = ensure(
                        DirectPrepushConfig {
                            node: DirectPrepushNode::Continuation(*target),
                            pending: config.pending.clone(),
                        },
                        &mut nwa,
                        &mut config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, weight.clone());
                    edge_count += 1;
                }
                for branch in &state.branches {
                    if !productive
                        .get(branch.target as usize)
                        .copied()
                        .unwrap_or(false)
                        || !summaries
                            .bundle_accepts
                            .get(branch.bundle_id)
                            .copied()
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    let Some(bundle) = bundles[branch.bundle_id].as_ref() else {
                        continue;
                    };
                    if bundle.states.is_empty() {
                        continue;
                    }
                    let target_state = ensure(
                        DirectPrepushConfig {
                            node: DirectPrepushNode::BundleCore {
                                bundle_id: branch.bundle_id,
                                target: branch.target,
                                core_state: 0,
                            },
                            pending: config.pending.clone(),
                        },
                        &mut nwa,
                        &mut config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, branch.entry_weight.clone());
                    edge_count += 1;
                }
            }
            DirectPrepushNode::BundleCore {
                bundle_id,
                target,
                core_state,
            } => {
                let bundle = bundles[bundle_id]
                    .as_ref()
                    .expect("direct pre-push core requires a built bundle");
                let state = &bundle.states[core_state as usize];
                if !state.final_weight.is_empty() {
                    let target_state = ensure(
                        DirectPrepushConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: config.pending.clone(),
                        },
                        &mut nwa,
                        &mut config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, state.final_weight.clone());
                    edge_count += 1;
                }
                for output in &state.outputs {
                    if output.weight.is_empty() {
                        continue;
                    }
                    let target_state = ensure(
                        DirectPrepushConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: append_pending(&config.pending, &output.pushes),
                        },
                        &mut nwa,
                        &mut config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, output.weight.clone());
                    edge_count += 1;
                }
                for (&label, transition) in &state.transitions {
                    debug_assert!(!is_negative_label(label));
                    match transition {
                        WeightedPrepushTarget::Core {
                            target: core_target,
                            weight,
                        } => {
                            if weight.is_empty() {
                                continue;
                            }
                            if config.pending.is_empty() {
                                let target_state = ensure(
                                    DirectPrepushConfig {
                                        node: DirectPrepushNode::BundleCore {
                                            bundle_id,
                                            target,
                                            core_state: *core_target,
                                        },
                                        pending: SmallVec::new(),
                                    },
                                    &mut nwa,
                                    &mut config_to_state,
                                    &mut configs,
                                    &mut queue,
                                );
                                nwa.add_transition(nwa_state, label, target_state, weight.clone());
                                edge_count += 1;
                            } else if let Some(pending) = consume_pending(&config.pending, label) {
                                let target_state = ensure(
                                    DirectPrepushConfig {
                                        node: DirectPrepushNode::BundleCore {
                                            bundle_id,
                                            target,
                                            core_state: *core_target,
                                        },
                                        pending,
                                    },
                                    &mut nwa,
                                    &mut config_to_state,
                                    &mut configs,
                                    &mut queue,
                                );
                                nwa.add_epsilon(nwa_state, target_state, weight.clone());
                                edge_count += 1;
                            }
                        }
                        WeightedPrepushTarget::Outputs(outputs) => {
                            for output in outputs {
                                if output.weight.is_empty() {
                                    continue;
                                }
                                if config.pending.is_empty() {
                                    let target_state = ensure(
                                        DirectPrepushConfig {
                                            node: DirectPrepushNode::Continuation(target),
                                            pending: append_pending(&SmallVec::new(), &output.pushes),
                                        },
                                        &mut nwa,
                                        &mut config_to_state,
                                        &mut configs,
                                        &mut queue,
                                    );
                                    nwa.add_transition(
                                        nwa_state,
                                        label,
                                        target_state,
                                        output.weight.clone(),
                                    );
                                    edge_count += 1;
                                } else if let Some(pending) = consume_pending(&config.pending, label) {
                                    let target_state = ensure(
                                        DirectPrepushConfig {
                                            node: DirectPrepushNode::Continuation(target),
                                            pending: append_pending(&pending, &output.pushes),
                                        },
                                        &mut nwa,
                                        &mut config_to_state,
                                        &mut configs,
                                        &mut queue,
                                    );
                                    nwa.add_epsilon(
                                        nwa_state,
                                        target_state,
                                        output.weight.clone(),
                                    );
                                    edge_count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let unique_pending = configs
        .iter()
        .map(|config| config.pending.clone())
        .collect::<rustc_hash::FxHashSet<_>>()
        .len();
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_PENDING_DETAIL").is_some() {
        let mut depth_hist = BTreeMap::<usize, usize>::new();
        let mut node_depth_hist = BTreeMap::<(&'static str, usize), usize>::new();
        for config in &configs {
            *depth_hist.entry(config.pending.len()).or_default() += 1;
            let kind = match config.node {
                DirectPrepushNode::Continuation(_) => "continuation",
                DirectPrepushNode::BundleCore { .. } => "core",
            };
            *node_depth_hist.entry((kind, config.pending.len())).or_default() += 1;
        }
        let quotient_counts = (0usize..=max_pending)
            .map(|keep| {
                let count = configs
                    .iter()
                    .map(|config| {
                        let start = config.pending.len().saturating_sub(keep);
                        (config.node.clone(), config.pending[start..].to_vec())
                    })
                    .collect::<rustc_hash::FxHashSet<_>>()
                    .len();
                (keep, count)
            })
            .collect::<Vec<_>>();
        let unique_suffixes = (0usize..=max_pending)
            .map(|keep| {
                let count = configs
                    .iter()
                    .map(|config| {
                        let start = config.pending.len().saturating_sub(keep);
                        config.pending[start..].to_vec()
                    })
                    .collect::<rustc_hash::FxHashSet<_>>()
                    .len();
                (keep, count)
            })
            .collect::<Vec<_>>();
        let mut demand_memo = FxHashMap::<DirectPrepushNode, usize>::default();
        let mut visiting = rustc_hash::FxHashSet::<DirectPrepushNode>::default();
        let mut demand_failed = false;
        for config in &configs {
            if direct_prepush_pending_demand(
                &config.node,
                summaries,
                productive,
                &bundles,
                &mut demand_memo,
                &mut visiting,
            )
            .is_none()
            {
                demand_failed = true;
                break;
            }
        }
        let mut demand_hist = BTreeMap::<usize, usize>::new();
        for &demand in demand_memo.values() {
            *demand_hist.entry(demand).or_default() += 1;
        }
        let demand_quotient_count = if demand_failed {
            0
        } else {
            configs
                .iter()
                .map(|config| {
                    let demand = demand_memo[&config.node];
                    let start = config.pending.len().saturating_sub(demand);
                    (config.node.clone(), config.pending[start..].to_vec())
                })
                .collect::<rustc_hash::FxHashSet<_>>()
                .len()
        };
        eprintln!(
            "[glrmask/profile][parser_direct_prepush_pending_detail] depth_hist={:?} node_depth_hist={:?} quotient_counts={:?} unique_suffixes={:?} demand_failed={} demand_hist={:?} demand_quotient_count={}",
            depth_hist,
            node_depth_hist,
            quotient_counts,
            unique_suffixes,
            demand_failed,
            demand_hist,
            demand_quotient_count,
        );
    }
    Some((nwa, max_pending, pending_configs, unique_pending + bundle_build_ms.saturating_mul(0)))
}



#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DirectCanonicalEdge {
    kind: u8,
    label: i32,
    target: u32,
    weight: Weight,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DirectCanonicalSignature {
    final_weight: Option<Weight>,
    edges: Vec<DirectCanonicalEdge>,
}

struct DirectCanonicalBuilder<'a> {
    summaries: &'a StateSummaries,
    productive: &'a [bool],
    bundles: &'a [Option<Arc<WeightedPrepushBundle>>],
    config_memo: FxHashMap<DirectPrepushConfig, u32>,
    visiting: rustc_hash::FxHashSet<DirectPrepushConfig>,
    class_by_signature: FxHashMap<DirectCanonicalSignature, u32>,
    nwa: NWA,
    raw_configs: usize,
    raw_edges: usize,
    max_pending: usize,
}

impl<'a> DirectCanonicalBuilder<'a> {
    fn new(
        summaries: &'a StateSummaries,
        productive: &'a [bool],
        bundles: &'a [Option<Arc<WeightedPrepushBundle>>],
    ) -> Self {
        Self {
            summaries,
            productive,
            bundles,
            config_memo: FxHashMap::default(),
            visiting: rustc_hash::FxHashSet::default(),
            class_by_signature: FxHashMap::default(),
            nwa: NWA::new(0, 0),
            raw_configs: 0,
            raw_edges: 0,
            max_pending: 0,
        }
    }

    fn class_for(&mut self, config: DirectPrepushConfig) -> Option<u32> {
        if let Some(&cached) = self.config_memo.get(&config) {
            return Some(cached);
        }
        if !self.visiting.insert(config.clone()) {
            return None;
        }
        self.raw_configs += 1;
        self.max_pending = self.max_pending.max(config.pending.len());

        let mut final_weight = None::<Weight>;
        let mut edge_map = BTreeMap::<(u8, i32, u32), Weight>::new();
        let mut add_edge = |this: &mut Self,
                            kind: u8,
                            label: i32,
                            target_config: DirectPrepushConfig,
                            weight: &Weight|
         -> Option<()> {
            if weight.is_empty() {
                return Some(());
            }
            let target = this.class_for(target_config)?;
            this.raw_edges += 1;
            edge_map
                .entry((kind, label, target))
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
            Some(())
        };

        match config.node {
            DirectPrepushNode::Continuation(state_id) => {
                let state = self.summaries.states.get(state_id as usize)?;
                final_weight = state
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                for (target, weight) in &state.epsilon_branches {
                    if weight.is_empty()
                        || !self.productive.get(*target as usize).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    add_edge(
                        self,
                        1,
                        0,
                        DirectPrepushConfig {
                            node: DirectPrepushNode::Continuation(*target),
                            pending: config.pending.clone(),
                        },
                        weight,
                    )?;
                }
                for branch in &state.branches {
                    if !self
                        .productive
                        .get(branch.target as usize)
                        .copied()
                        .unwrap_or(false)
                        || !self
                            .summaries
                            .bundle_accepts
                            .get(branch.bundle_id)
                            .copied()
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    let Some(bundle) = self.bundles.get(branch.bundle_id).and_then(Option::as_ref)
                    else {
                        continue;
                    };
                    if bundle.states.is_empty() {
                        continue;
                    }
                    add_edge(
                        self,
                        1,
                        0,
                        DirectPrepushConfig {
                            node: DirectPrepushNode::BundleCore {
                                bundle_id: branch.bundle_id,
                                target: branch.target,
                                core_state: 0,
                            },
                            pending: config.pending.clone(),
                        },
                        &branch.entry_weight,
                    )?;
                }
            }
            DirectPrepushNode::BundleCore {
                bundle_id,
                target,
                core_state,
            } => {
                let bundle = self.bundles.get(bundle_id).and_then(Option::as_ref)?;
                let state = bundle.states.get(core_state as usize)?;
                if !state.final_weight.is_empty() {
                    add_edge(
                        self,
                        1,
                        0,
                        DirectPrepushConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: config.pending.clone(),
                        },
                        &state.final_weight,
                    )?;
                }
                for output in &state.outputs {
                    if output.weight.is_empty() {
                        continue;
                    }
                    add_edge(
                        self,
                        1,
                        0,
                        DirectPrepushConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: append_pending(&config.pending, &output.pushes),
                        },
                        &output.weight,
                    )?;
                }
                for (&label, transition) in &state.transitions {
                    debug_assert!(!is_negative_label(label));
                    match transition {
                        WeightedPrepushTarget::Core {
                            target: core_target,
                            weight,
                        } => {
                            if weight.is_empty() {
                                continue;
                            }
                            if config.pending.is_empty() {
                                add_edge(
                                    self,
                                    0,
                                    label,
                                    DirectPrepushConfig {
                                        node: DirectPrepushNode::BundleCore {
                                            bundle_id,
                                            target,
                                            core_state: *core_target,
                                        },
                                        pending: SmallVec::new(),
                                    },
                                    weight,
                                )?;
                            } else if let Some(pending) = consume_pending(&config.pending, label) {
                                add_edge(
                                    self,
                                    1,
                                    0,
                                    DirectPrepushConfig {
                                        node: DirectPrepushNode::BundleCore {
                                            bundle_id,
                                            target,
                                            core_state: *core_target,
                                        },
                                        pending,
                                    },
                                    weight,
                                )?;
                            }
                        }
                        WeightedPrepushTarget::Outputs(outputs) => {
                            for output in outputs {
                                if output.weight.is_empty() {
                                    continue;
                                }
                                if config.pending.is_empty() {
                                    add_edge(
                                        self,
                                        0,
                                        label,
                                        DirectPrepushConfig {
                                            node: DirectPrepushNode::Continuation(target),
                                            pending: append_pending(
                                                &SmallVec::new(),
                                                &output.pushes,
                                            ),
                                        },
                                        &output.weight,
                                    )?;
                                } else if let Some(pending) =
                                    consume_pending(&config.pending, label)
                                {
                                    add_edge(
                                        self,
                                        1,
                                        0,
                                        DirectPrepushConfig {
                                            node: DirectPrepushNode::Continuation(target),
                                            pending: append_pending(&pending, &output.pushes),
                                        },
                                        &output.weight,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
        }

        let edges = edge_map
            .into_iter()
            .filter(|(_, weight)| !weight.is_empty())
            .map(|((kind, label, target), weight)| DirectCanonicalEdge {
                kind,
                label,
                target,
                weight,
            })
            .collect::<Vec<_>>();
        let signature = DirectCanonicalSignature {
            final_weight,
            edges,
        };
        let class = if let Some(&existing) = self.class_by_signature.get(&signature) {
            existing
        } else {
            let state_id = self.nwa.add_state();
            if let Some(weight) = signature.final_weight.as_ref() {
                self.nwa.set_final_weight(state_id, weight.clone());
            }
            for edge in &signature.edges {
                if edge.kind == 0 {
                    self.nwa
                        .add_transition(state_id, edge.label, edge.target, edge.weight.clone());
                } else {
                    self.nwa
                        .add_epsilon(state_id, edge.target, edge.weight.clone());
                }
            }
            self.class_by_signature.insert(signature, state_id);
            state_id
        };
        self.visiting.remove(&config);
        self.config_memo.insert(config, class);
        Some(class)
    }
}


#[derive(Clone, Copy, Debug)]
struct PendingStackEntry {
    parent: u32,
    top: u32,
    depth: u16,
}

struct PendingStackInterner {
    entries: Vec<PendingStackEntry>,
    by_parent_top: FxHashMap<(u32, u32), u32>,
}

impl PendingStackInterner {
    fn new() -> Self {
        Self {
            entries: vec![PendingStackEntry {
                parent: 0,
                top: 0,
                depth: 0,
            }],
            by_parent_top: FxHashMap::default(),
        }
    }

    #[inline]
    fn depth(&self, id: u32) -> usize {
        self.entries[id as usize].depth as usize
    }

    #[inline]
    fn push(&mut self, parent: u32, top: u32) -> u32 {
        if let Some(&existing) = self.by_parent_top.get(&(parent, top)) {
            return existing;
        }
        let depth = self.entries[parent as usize]
            .depth
            .checked_add(1)
            .expect("pending write stack depth exceeds u16");
        let id = self.entries.len() as u32;
        self.entries.push(PendingStackEntry { parent, top, depth });
        self.by_parent_top.insert((parent, top), id);
        id
    }

    fn push_many(&mut self, mut pending: u32, pushes: &[u32]) -> u32 {
        for &push in pushes {
            pending = self.push(pending, push);
        }
        pending
    }

    #[inline]
    fn top(&self, pending: u32) -> Option<u32> {
        (pending != 0).then(|| self.entries[pending as usize].top)
    }

    #[inline]
    fn pop_matching(&self, pending: u32, label: i32) -> Option<u32> {
        if pending == 0 {
            return Some(0);
        }
        let entry = self.entries[pending as usize];
        if label != DEFAULT_LABEL && (label < 0 || label as u32 != entry.top) {
            return None;
        }
        Some(entry.parent)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DirectCompactConfig {
    node: DirectPrepushNode,
    pending: u32,
}

#[inline]
fn direct_compact_config_key(config: DirectCompactConfig) -> u128 {
    let (bundle_or_tag, target_or_state, core_state) = match config.node {
        DirectPrepushNode::Continuation(state) => (u32::MAX, state, 0),
        DirectPrepushNode::BundleCore {
            bundle_id,
            target,
            core_state,
        } => (
            u32::try_from(bundle_id).expect("direct pre-push bundle id exceeds u32"),
            target,
            core_state,
        ),
    };
    (config.pending as u128)
        | ((core_state as u128) << 32)
        | ((target_or_state as u128) << 64)
        | ((bundle_or_tag as u128) << 96)
}

#[derive(Clone)]
struct DirectCompactRawEdge {
    label: Option<i32>,
    target: DirectCompactConfig,
    weight: Weight,
}

struct DirectCompactRawBody {
    final_weight: Option<Weight>,
    edges: Vec<DirectCompactRawEdge>,
}

#[derive(Clone)]
struct DirectFlatConfigEdge {
    label: Option<i32>,
    target: u32,
    weight: Weight,
}

#[derive(Clone)]
struct DirectFlatConfigNode {
    final_weight: Option<Weight>,
    edge_start: u32,
    edge_len: u32,
}

#[derive(Clone)]
struct DirectProjectedConfigSummary {
    final_weight: Option<Weight>,
    exits: SmallVec<[(DirectCompactConfig, Weight); 4]>,
}

struct DirectProjectedConfigBuilder<'a> {
    summaries: &'a StateSummaries,
    productive: &'a [bool],
    templates: &'a Templates,
    lazy_bundles: LazyWeightedPrepushBundleSet,
    pending: PendingStackInterner,
    forced_starts: FxHashSet<DirectCompactConfig>,
    summary_memo: FxHashMap<DirectCompactConfig, Option<Arc<DirectProjectedConfigSummary>>>,
    projected_by_config: FxHashMap<DirectCompactConfig, u32>,
    projected_configs: Vec<DirectCompactConfig>,
    queue: VecDeque<u32>,
    weight_ops: ScopedWeightOpCache,
    nwa: NWA,
    final_sink: u32,
    raw_configs_summarized: usize,
    summary_hits: usize,
    raw_edges_examined: usize,
    detail_profile: bool,
    raw_body_calls: usize,
    raw_body_ms: f64,
    continuation_body_calls: usize,
    continuation_body_ms: f64,
    core_body_calls: usize,
    core_body_ms: f64,
}

impl<'a> DirectProjectedConfigBuilder<'a> {
    fn new(
        summaries: &'a StateSummaries,
        productive: &'a [bool],
        templates: &'a Templates,
        lazy_bundles: LazyWeightedPrepushBundleSet,
    ) -> Self {
        let mut nwa = NWA::new(0, 0);
        let final_sink = nwa.add_state();
        nwa.set_final_weight(final_sink, Weight::all());
        Self {
            summaries,
            productive,
            templates,
            lazy_bundles,
            pending: PendingStackInterner::new(),
            forced_starts: FxHashSet::default(),
            summary_memo: FxHashMap::default(),
            projected_by_config: FxHashMap::default(),
            projected_configs: Vec::new(),
            queue: VecDeque::new(),
            weight_ops: ScopedWeightOpCache::default(),
            nwa,
            final_sink,
            raw_configs_summarized: 0,
            summary_hits: 0,
            raw_edges_examined: 0,
            detail_profile: std::env::var_os(
                "GLRMASK_PROFILE_DIRECT_PREPUSH_PROJECTED_CONFIG_DETAIL",
            )
            .is_some(),
            raw_body_calls: 0,
            raw_body_ms: 0.0,
            continuation_body_calls: 0,
            continuation_body_ms: 0.0,
            core_body_calls: 0,
            core_body_ms: 0.0,
        }
    }

    #[inline]
    fn retained(&self, config: DirectCompactConfig) -> bool {
        self.forced_starts.contains(&config)
            || matches!(
                config,
                DirectCompactConfig {
                    node: DirectPrepushNode::BundleCore { .. },
                    pending: 0,
                }
            )
    }

    fn ensure_retained(&mut self, config: DirectCompactConfig) -> u32 {
        debug_assert!(self.retained(config));
        if let Some(&existing) = self.projected_by_config.get(&config) {
            return existing;
        }
        let state_id = self.nwa.add_state();
        self.projected_by_config.insert(config, state_id);
        self.projected_configs.push(config);
        self.queue.push_back(state_id);
        state_id
    }

    fn raw_body(&mut self, config: DirectCompactConfig) -> Option<DirectCompactRawBody> {
        let mut body = DirectCompactRawBody {
            final_weight: None,
            edges: Vec::new(),
        };
        match config.node {
            DirectPrepushNode::Continuation(state_id) => {
                let state = self.summaries.states.get(state_id as usize)?;
                body.final_weight = state
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                for (target, weight) in &state.epsilon_branches {
                    if weight.is_empty()
                        || !self.productive.get(*target as usize).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    body.edges.push(DirectCompactRawEdge {
                        label: None,
                        target: DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(*target),
                            pending: config.pending,
                        },
                        weight: weight.clone(),
                    });
                }
                for branch in &state.branches {
                    if !self
                        .productive
                        .get(branch.target as usize)
                        .copied()
                        .unwrap_or(false)
                        || !self
                            .summaries
                            .bundle_accepts
                            .get(branch.bundle_id)
                            .copied()
                            .unwrap_or(false)
                        || self.templates.lazy_weighted_prepush_bundle_is_empty(
                            &self.lazy_bundles,
                            branch.bundle_id,
                        )
                    {
                        continue;
                    }
                    body.edges.push(DirectCompactRawEdge {
                        label: None,
                        target: DirectCompactConfig {
                            node: DirectPrepushNode::BundleCore {
                                bundle_id: branch.bundle_id,
                                target: branch.target,
                                core_state: 0,
                            },
                            pending: config.pending,
                        },
                        weight: branch.entry_weight.clone(),
                    });
                }
            }
            DirectPrepushNode::BundleCore {
                bundle_id,
                target,
                core_state,
            } => {
                let state = self.templates.lazy_weighted_prepush_state(
                    &mut self.lazy_bundles,
                    bundle_id,
                    core_state,
                )?;
                if !state.final_weight.is_empty() {
                    body.edges.push(DirectCompactRawEdge {
                        label: None,
                        target: DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: config.pending,
                        },
                        weight: state.final_weight.clone(),
                    });
                }
                for output in &state.outputs {
                    if output.weight.is_empty() {
                        continue;
                    }
                    let next_pending = self.pending.push_many(config.pending, &output.pushes);
                    body.edges.push(DirectCompactRawEdge {
                        label: None,
                        target: DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: next_pending,
                        },
                        weight: output.weight.clone(),
                    });
                }

                let mut process_transition =
                    |label: i32, transition: &WeightedPrepushTarget, pending: &mut PendingStackInterner| {
                        debug_assert!(!is_negative_label(label));
                        match transition {
                            WeightedPrepushTarget::Core {
                                target: core_target,
                                weight,
                            } => {
                                if weight.is_empty() {
                                    return;
                                }
                                if config.pending == 0 {
                                    body.edges.push(DirectCompactRawEdge {
                                        label: Some(label),
                                        target: DirectCompactConfig {
                                            node: DirectPrepushNode::BundleCore {
                                                bundle_id,
                                                target,
                                                core_state: *core_target,
                                            },
                                            pending: 0,
                                        },
                                        weight: weight.clone(),
                                    });
                                } else if let Some(next_pending) =
                                    pending.pop_matching(config.pending, label)
                                {
                                    body.edges.push(DirectCompactRawEdge {
                                        label: None,
                                        target: DirectCompactConfig {
                                            node: DirectPrepushNode::BundleCore {
                                                bundle_id,
                                                target,
                                                core_state: *core_target,
                                            },
                                            pending: next_pending,
                                        },
                                        weight: weight.clone(),
                                    });
                                }
                            }
                            WeightedPrepushTarget::Outputs(outputs) => {
                                for output in outputs {
                                    if output.weight.is_empty() {
                                        continue;
                                    }
                                    if config.pending == 0 {
                                        let next_pending = pending.push_many(0, &output.pushes);
                                        body.edges.push(DirectCompactRawEdge {
                                            label: Some(label),
                                            target: DirectCompactConfig {
                                                node: DirectPrepushNode::Continuation(target),
                                                pending: next_pending,
                                            },
                                            weight: output.weight.clone(),
                                        });
                                    } else if let Some(base_pending) =
                                        pending.pop_matching(config.pending, label)
                                    {
                                        let next_pending =
                                            pending.push_many(base_pending, &output.pushes);
                                        body.edges.push(DirectCompactRawEdge {
                                            label: None,
                                            target: DirectCompactConfig {
                                                node: DirectPrepushNode::Continuation(target),
                                                pending: next_pending,
                                            },
                                            weight: output.weight.clone(),
                                        });
                                    }
                                }
                            }
                        }
                    };

                if config.pending == 0 {
                    for (&label, transition) in &state.transitions {
                        process_transition(label, transition, &mut self.pending);
                    }
                } else {
                    let top_label = self
                        .pending
                        .top(config.pending)
                        .expect("nonempty pending stack has a top")
                        as i32;
                    if let Some(transition) = state.transitions.get(&top_label) {
                        process_transition(top_label, transition, &mut self.pending);
                    }
                    if let Some(transition) = state.transitions.get(&DEFAULT_LABEL) {
                        process_transition(DEFAULT_LABEL, transition, &mut self.pending);
                    }
                }
            }
        }
        self.raw_edges_examined += body.edges.len();
        Some(body)
    }

    fn profiled_raw_body(
        &mut self,
        config: DirectCompactConfig,
    ) -> Option<DirectCompactRawBody> {
        let started = self.detail_profile.then(Instant::now);
        let result = self.raw_body(config);
        if let Some(started) = started {
            let ms = elapsed_ms(started);
            self.raw_body_calls += 1;
            self.raw_body_ms += ms;
            match config.node {
                DirectPrepushNode::Continuation(_) => {
                    self.continuation_body_calls += 1;
                    self.continuation_body_ms += ms;
                }
                DirectPrepushNode::BundleCore { .. } => {
                    self.core_body_calls += 1;
                    self.core_body_ms += ms;
                }
            }
        }
        result
    }

    fn merge_summary_exit(
        &mut self,
        exits: &mut SmallVec<[(DirectCompactConfig, Weight); 4]>,
        target: DirectCompactConfig,
        weight: Weight,
    ) {
        if weight.is_empty() {
            return;
        }
        if let Some((_, existing)) = exits
            .iter_mut()
            .find(|(existing_target, _)| *existing_target == target)
        {
            *existing = self.weight_ops.union(existing, &weight);
        } else {
            exits.push((target, weight));
        }
    }

    fn compute_summary(
        &mut self,
        config: DirectCompactConfig,
    ) -> Option<Arc<DirectProjectedConfigSummary>> {
        let body = self.profiled_raw_body(config)?;
        let mut final_weight = body.final_weight;
        let mut exits = SmallVec::<[(DirectCompactConfig, Weight); 4]>::new();
        for edge in body.edges {
            if edge.label.is_some() {
                return None;
            }
            if self.retained(edge.target) {
                self.merge_summary_exit(&mut exits, edge.target, edge.weight);
                continue;
            }
            let child = self.summary_for(edge.target)?;
            if let Some(child_final) = child.final_weight.as_ref() {
                let contribution = self.weight_ops.intersection(&edge.weight, child_final);
                if !contribution.is_empty() {
                    final_weight = Some(match final_weight {
                        Some(existing) => self.weight_ops.union(&existing, &contribution),
                        None => contribution,
                    });
                }
            }
            for (exit, suffix_weight) in &child.exits {
                let contribution = self.weight_ops.intersection(&edge.weight, suffix_weight);
                self.merge_summary_exit(&mut exits, *exit, contribution);
            }
        }
        exits.sort_unstable_by_key(|(config, _)| direct_compact_config_key(*config));
        Some(Arc::new(DirectProjectedConfigSummary {
            final_weight,
            exits,
        }))
    }

    fn summary_for(
        &mut self,
        config: DirectCompactConfig,
    ) -> Option<Arc<DirectProjectedConfigSummary>> {
        debug_assert!(!self.retained(config));
        if let Some(existing) = self.summary_memo.get(&config) {
            let existing = existing.as_ref()?;
            self.summary_hits += 1;
            return Some(Arc::clone(existing));
        }
        self.summary_memo.insert(config, None);
        self.raw_configs_summarized += 1;
        let result = self.compute_summary(config);
        if let Some(summary) = result.as_ref() {
            let slot = self
                .summary_memo
                .get_mut(&config)
                .expect("direct projected summary placeholder disappeared");
            *slot = Some(Arc::clone(summary));
        } else {
            self.summary_memo.remove(&config);
        }
        result
    }

    fn build_flat_projected(
        mut self,
        starts: &[DirectCompactConfig],
    ) -> Option<(NWA, usize, usize, usize, usize)> {
        for &config in starts {
            self.forced_starts.insert(config);
        }

        let mut config_to_id = FxHashMap::<DirectCompactConfig, u32>::default();
        let mut configs = Vec::<DirectCompactConfig>::new();
        let mut nodes = Vec::<DirectFlatConfigNode>::new();
        let mut edges = Vec::<DirectFlatConfigEdge>::new();
        let mut queue = VecDeque::<u32>::new();
        let ensure = |config: DirectCompactConfig,
                      config_to_id: &mut FxHashMap<DirectCompactConfig, u32>,
                      configs: &mut Vec<DirectCompactConfig>,
                      nodes: &mut Vec<DirectFlatConfigNode>,
                      queue: &mut VecDeque<u32>| {
            if let Some(&existing) = config_to_id.get(&config) {
                return existing;
            }
            let id = configs.len() as u32;
            config_to_id.insert(config, id);
            configs.push(config);
            nodes.push(DirectFlatConfigNode {
                final_weight: None,
                edge_start: 0,
                edge_len: 0,
            });
            queue.push_back(id);
            id
        };

        let start_ids = starts
            .iter()
            .copied()
            .map(|config| {
                ensure(
                    config,
                    &mut config_to_id,
                    &mut configs,
                    &mut nodes,
                    &mut queue,
                )
            })
            .collect::<Vec<_>>();

        while let Some(state_id) = queue.pop_front() {
            let config = configs[state_id as usize];
            let body = self.profiled_raw_body(config)?;
            let edge_start = edges.len() as u32;
            for edge in body.edges {
                let target = ensure(
                    edge.target,
                    &mut config_to_id,
                    &mut configs,
                    &mut nodes,
                    &mut queue,
                );
                edges.push(DirectFlatConfigEdge {
                    label: edge.label,
                    target,
                    weight: edge.weight,
                });
            }
            nodes[state_id as usize] = DirectFlatConfigNode {
                final_weight: body.final_weight,
                edge_start,
                edge_len: edges.len() as u32 - edge_start,
            };
        }

        let retained = configs
            .iter()
            .copied()
            .map(|config| self.retained(config))
            .collect::<Vec<_>>();
        let removed_count = retained.iter().filter(|&&keep| !keep).count();

        let mut removed_outdegree = vec![0usize; configs.len()];
        let mut removed_predecessors = vec![Vec::<u32>::new(); configs.len()];
        for (source, node) in nodes.iter().enumerate() {
            if retained[source] {
                continue;
            }
            for edge in &edges[node.edge_start as usize..(node.edge_start + node.edge_len) as usize] {
                debug_assert!(edge.label.is_none());
                let target = edge.target as usize;
                if edge.weight.is_empty() || retained[target] {
                    continue;
                }
                removed_outdegree[source] += 1;
                removed_predecessors[target].push(source as u32);
            }
        }

        let mut summary_final = vec![None::<Weight>; configs.len()];
        let mut summary_exits = vec![Vec::<(u32, Weight)>::new(); configs.len()];
        let mut summary_queue = VecDeque::<u32>::new();
        for (state_id, &outdegree) in removed_outdegree.iter().enumerate() {
            if !retained[state_id] && outdegree == 0 {
                summary_queue.push_back(state_id as u32);
            }
        }
        let mut processed = 0usize;
        while let Some(state_id) = summary_queue.pop_front() {
            let node = &nodes[state_id as usize];
            let mut final_weight = node
                .final_weight
                .as_ref()
                .filter(|weight| !weight.is_empty())
                .cloned();
            let mut exits = SmallVec::<[(u32, Weight); 4]>::new();
            for edge in &edges[node.edge_start as usize..(node.edge_start + node.edge_len) as usize] {
                debug_assert!(edge.label.is_none());
                if edge.weight.is_empty() {
                    continue;
                }
                let target = edge.target as usize;
                if retained[target] {
                    if let Some((_, existing)) =
                        exits.iter_mut().find(|(exit, _)| *exit == edge.target)
                    {
                        *existing = self.weight_ops.union(existing, &edge.weight);
                    } else {
                        exits.push((edge.target, edge.weight.clone()));
                    }
                    continue;
                }
                if let Some(target_final) = summary_final[target].as_ref() {
                    let contribution = self.weight_ops.intersection(&edge.weight, target_final);
                    if !contribution.is_empty() {
                        final_weight = Some(match final_weight {
                            Some(existing) => self.weight_ops.union(&existing, &contribution),
                            None => contribution,
                        });
                    }
                }
                for (exit, suffix_weight) in &summary_exits[target] {
                    let contribution = self.weight_ops.intersection(&edge.weight, suffix_weight);
                    if contribution.is_empty() {
                        continue;
                    }
                    if let Some((_, existing)) =
                        exits.iter_mut().find(|(existing_exit, _)| *existing_exit == *exit)
                    {
                        *existing = self.weight_ops.union(existing, &contribution);
                    } else {
                        exits.push((*exit, contribution));
                    }
                }
            }
            exits.sort_unstable_by_key(|(exit, _)| *exit);
            summary_final[state_id as usize] = final_weight;
            summary_exits[state_id as usize] = exits.into_vec();
            processed += 1;
            for &predecessor in &removed_predecessors[state_id as usize] {
                let degree = &mut removed_outdegree[predecessor as usize];
                *degree -= 1;
                if *degree == 0 {
                    summary_queue.push_back(predecessor);
                }
            }
        }
        if processed != removed_count {
            return None;
        }

        let mut nwa = NWA::new(0, 0);
        let mut new_by_old = vec![u32::MAX; configs.len()];
        for (old_state, &keep) in retained.iter().enumerate() {
            if keep {
                new_by_old[old_state] = nwa.add_state();
            }
        }
        let final_sink = nwa.add_state();
        nwa.set_final_weight(final_sink, Weight::all());

        for (source, node) in nodes.iter().enumerate() {
            if !retained[source] {
                continue;
            }
            let new_source = new_by_old[source];
            let mut source_final = node
                .final_weight
                .as_ref()
                .filter(|weight| !weight.is_empty())
                .cloned();
            for edge in &edges[node.edge_start as usize..(node.edge_start + node.edge_len) as usize] {
                if edge.weight.is_empty() {
                    continue;
                }
                let target = edge.target as usize;
                if retained[target] {
                    let new_target = new_by_old[target];
                    if let Some(label) = edge.label {
                        nwa.add_transition(new_source, label, new_target, edge.weight.clone());
                    } else {
                        nwa.add_epsilon(new_source, new_target, edge.weight.clone());
                    }
                    continue;
                }

                if let Some(target_final) = summary_final[target].as_ref() {
                    let contribution = self.weight_ops.intersection(&edge.weight, target_final);
                    if !contribution.is_empty() {
                        if let Some(label) = edge.label {
                            nwa.add_transition(new_source, label, final_sink, contribution);
                        } else {
                            source_final = Some(match source_final {
                                Some(existing) => self.weight_ops.union(&existing, &contribution),
                                None => contribution,
                            });
                        }
                    }
                }
                for (exit, suffix_weight) in &summary_exits[target] {
                    let contribution = self.weight_ops.intersection(&edge.weight, suffix_weight);
                    if contribution.is_empty() {
                        continue;
                    }
                    let new_target = new_by_old[*exit as usize];
                    if let Some(label) = edge.label {
                        nwa.add_transition(new_source, label, new_target, contribution);
                    } else {
                        nwa.add_epsilon(new_source, new_target, contribution);
                    }
                }
            }
            if let Some(final_weight) = source_final {
                nwa.set_final_weight(new_source, final_weight);
            }
        }
        let projected_starts = start_ids
            .iter()
            .map(|&state_id| new_by_old[state_id as usize])
            .collect::<Vec<_>>();
        nwa.set_start_states(projected_starts);
        let (transition_merges, epsilon_merges) =
            coalesce_parallel_nwa_edges(&mut nwa, &mut self.weight_ops);
        if self.detail_profile {
            eprintln!(
                "[glrmask/profile][parser_direct_prepush_projected_config_flat_detail] raw_configs={} raw_edges={} retained={} removed={} transition_merges={} epsilon_merges={} raw_body_ms={:.3}",
                configs.len(),
                edges.len(),
                retained.iter().filter(|&&keep| keep).count(),
                removed_count,
                transition_merges,
                epsilon_merges,
                self.raw_body_ms,
            );
        }
        self.templates.emit_lazy_weighted_prepush_profile(&self.lazy_bundles);
        Some((nwa, configs.len(), edges.len(), removed_count, self.raw_body_calls))
    }

    fn build(
        mut self,
        starts: &[DirectCompactConfig],
    ) -> Option<(NWA, usize, usize, usize, usize)> {
        for &config in starts {
            self.forced_starts.insert(config);
        }
        let start_states = starts
            .iter()
            .copied()
            .map(|config| self.ensure_retained(config))
            .collect::<Vec<_>>();
        self.nwa.set_start_states(start_states);

        while let Some(state_id) = self.queue.pop_front() {
            let config = self.projected_configs[(state_id - 1) as usize];
            let body = self.profiled_raw_body(config)?;
            let mut final_weight = body.final_weight;
            let mut edge_map = BTreeMap::<(u8, i32, u32), Weight>::new();
            for edge in body.edges {
                let (kind, label) = match edge.label {
                    Some(label) => (0u8, label),
                    None => (1u8, 0),
                };
                if self.retained(edge.target) {
                    let target = self.ensure_retained(edge.target);
                    edge_map
                        .entry((kind, label, target))
                        .and_modify(|existing| {
                            *existing = self.weight_ops.union(existing, &edge.weight)
                        })
                        .or_insert(edge.weight);
                    continue;
                }

                let summary = self.summary_for(edge.target)?;
                if let Some(summary_final) = summary.final_weight.as_ref() {
                    let contribution = self.weight_ops.intersection(&edge.weight, summary_final);
                    if !contribution.is_empty() {
                        if kind == 1 {
                            final_weight = Some(match final_weight {
                                Some(existing) => self.weight_ops.union(&existing, &contribution),
                                None => contribution,
                            });
                        } else {
                            edge_map
                                .entry((0, label, self.final_sink))
                                .and_modify(|existing| {
                                    *existing = self.weight_ops.union(existing, &contribution)
                                })
                                .or_insert(contribution);
                        }
                    }
                }
                for (exit_config, suffix_weight) in &summary.exits {
                    let contribution = self.weight_ops.intersection(&edge.weight, suffix_weight);
                    if contribution.is_empty() {
                        continue;
                    }
                    let target = self.ensure_retained(*exit_config);
                    edge_map
                        .entry((kind, label, target))
                        .and_modify(|existing| {
                            *existing = self.weight_ops.union(existing, &contribution)
                        })
                        .or_insert(contribution);
                }
            }
            if let Some(final_weight) = final_weight.filter(|weight| !weight.is_empty()) {
                self.nwa.set_final_weight(state_id, final_weight);
            }
            for ((kind, label, target), weight) in edge_map {
                if weight.is_empty() {
                    continue;
                }
                if kind == 0 {
                    self.nwa.add_transition(state_id, label, target, weight);
                } else {
                    self.nwa.add_epsilon(state_id, target, weight);
                }
            }
        }
        self.templates.emit_lazy_weighted_prepush_profile(&self.lazy_bundles);
        if self.detail_profile {
            eprintln!(
                "[glrmask/profile][parser_direct_prepush_projected_config_detail] raw_body_calls={} raw_body_ms={:.3} continuation_calls={} continuation_ms={:.3} core_calls={} core_ms={:.3} summarized_configs={} summary_hits={} projected_states={} raw_edges_examined={}",
                self.raw_body_calls,
                self.raw_body_ms,
                self.continuation_body_calls,
                self.continuation_body_ms,
                self.core_body_calls,
                self.core_body_ms,
                self.raw_configs_summarized,
                self.summary_hits,
                self.nwa.states().len(),
                self.raw_edges_examined,
            );
        }
        Some((
            self.nwa,
            self.raw_configs_summarized,
            self.summary_memo.values().filter(|summary| summary.is_some()).count(),
            self.summary_hits,
            self.raw_edges_examined,
        ))
    }
}

struct DirectCompactCanonicalBuilder<'a> {
    summaries: &'a StateSummaries,
    productive: &'a [bool],
    bundles: &'a [Option<Arc<WeightedPrepushBundle>>],
    pending: PendingStackInterner,
    config_memo: FxHashMap<DirectCompactConfig, u32>,
    visiting: rustc_hash::FxHashSet<DirectCompactConfig>,
    class_by_signature: FxHashMap<DirectCanonicalSignature, u32>,
    nwa: NWA,
    raw_configs: usize,
    raw_edges: usize,
    max_pending: usize,
}

impl<'a> DirectCompactCanonicalBuilder<'a> {
    fn new(
        summaries: &'a StateSummaries,
        productive: &'a [bool],
        bundles: &'a [Option<Arc<WeightedPrepushBundle>>],
    ) -> Self {
        Self {
            summaries,
            productive,
            bundles,
            pending: PendingStackInterner::new(),
            config_memo: FxHashMap::default(),
            visiting: rustc_hash::FxHashSet::default(),
            class_by_signature: FxHashMap::default(),
            nwa: NWA::new(0, 0),
            raw_configs: 0,
            raw_edges: 0,
            max_pending: 0,
        }
    }

    fn class_for(&mut self, config: DirectCompactConfig) -> Option<u32> {
        if let Some(&cached) = self.config_memo.get(&config) {
            return Some(cached);
        }
        if !self.visiting.insert(config.clone()) {
            return None;
        }
        self.raw_configs += 1;
        self.max_pending = self.max_pending.max(self.pending.depth(config.pending));
        let mut final_weight = None::<Weight>;
        let mut edge_map = BTreeMap::<(u8, i32, u32), Weight>::new();

        match config.node {
            DirectPrepushNode::Continuation(state_id) => {
                let state = self.summaries.states.get(state_id as usize)?;
                final_weight = state
                    .final_weight
                    .as_ref()
                    .filter(|weight| !weight.is_empty())
                    .cloned();
                let epsilon_branches = state.epsilon_branches.clone();
                let branches = state.branches.clone();
                for (target, weight) in epsilon_branches {
                    if weight.is_empty()
                        || !self.productive.get(target as usize).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    let child = self.class_for(DirectCompactConfig {
                        node: DirectPrepushNode::Continuation(target),
                        pending: config.pending,
                    })?;
                    self.raw_edges += 1;
                    edge_map
                        .entry((1, 0, child))
                        .and_modify(|existing| *existing = existing.union(&weight))
                        .or_insert(weight);
                }
                for branch in branches {
                    if !self
                        .productive
                        .get(branch.target as usize)
                        .copied()
                        .unwrap_or(false)
                        || !self
                            .summaries
                            .bundle_accepts
                            .get(branch.bundle_id)
                            .copied()
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    let Some(bundle) = self.bundles.get(branch.bundle_id).and_then(Option::as_ref)
                    else {
                        continue;
                    };
                    if bundle.states.is_empty() {
                        continue;
                    }
                    let child = self.class_for(DirectCompactConfig {
                        node: DirectPrepushNode::BundleCore {
                            bundle_id: branch.bundle_id,
                            target: branch.target,
                            core_state: 0,
                        },
                        pending: config.pending,
                    })?;
                    self.raw_edges += 1;
                    edge_map
                        .entry((1, 0, child))
                        .and_modify(|existing| *existing = existing.union(&branch.entry_weight))
                        .or_insert(branch.entry_weight);
                }
            }
            DirectPrepushNode::BundleCore {
                bundle_id,
                target,
                core_state,
            } => {
                let bundle = self.bundles.get(bundle_id).and_then(Option::as_ref)?;
                let state = bundle.states.get(core_state as usize)?.clone();
                if !state.final_weight.is_empty() {
                    let child = self.class_for(DirectCompactConfig {
                        node: DirectPrepushNode::Continuation(target),
                        pending: config.pending,
                    })?;
                    self.raw_edges += 1;
                    edge_map
                        .entry((1, 0, child))
                        .and_modify(|existing| *existing = existing.union(&state.final_weight))
                        .or_insert(state.final_weight.clone());
                }
                for output in &state.outputs {
                    if output.weight.is_empty() {
                        continue;
                    }
                    let next_pending = self.pending.push_many(config.pending, &output.pushes);
                    let child = self.class_for(DirectCompactConfig {
                        node: DirectPrepushNode::Continuation(target),
                        pending: next_pending,
                    })?;
                    self.raw_edges += 1;
                    edge_map
                        .entry((1, 0, child))
                        .and_modify(|existing| *existing = existing.union(&output.weight))
                        .or_insert(output.weight.clone());
                }
                for (label, transition) in state.transitions {
                    debug_assert!(!is_negative_label(label));
                    match transition {
                        WeightedPrepushTarget::Core {
                            target: core_target,
                            weight,
                        } => {
                            if weight.is_empty() {
                                continue;
                            }
                            let (kind, edge_label, next_pending) = if config.pending == 0 {
                                (0, label, 0)
                            } else {
                                let Some(next) = self.pending.pop_matching(config.pending, label)
                                else {
                                    continue;
                                };
                                (1, 0, next)
                            };
                            let child = self.class_for(DirectCompactConfig {
                                node: DirectPrepushNode::BundleCore {
                                    bundle_id,
                                    target,
                                    core_state: core_target,
                                },
                                pending: next_pending,
                            })?;
                            self.raw_edges += 1;
                            edge_map
                                .entry((kind, edge_label, child))
                                .and_modify(|existing| *existing = existing.union(&weight))
                                .or_insert(weight);
                        }
                        WeightedPrepushTarget::Outputs(outputs) => {
                            for output in outputs {
                                if output.weight.is_empty() {
                                    continue;
                                }
                                let (kind, edge_label, base_pending) = if config.pending == 0 {
                                    (0, label, 0)
                                } else {
                                    let Some(next) =
                                        self.pending.pop_matching(config.pending, label)
                                    else {
                                        continue;
                                    };
                                    (1, 0, next)
                                };
                                let next_pending =
                                    self.pending.push_many(base_pending, &output.pushes);
                                let child = self.class_for(DirectCompactConfig {
                                    node: DirectPrepushNode::Continuation(target),
                                    pending: next_pending,
                                })?;
                                self.raw_edges += 1;
                                edge_map
                                    .entry((kind, edge_label, child))
                                    .and_modify(|existing| {
                                        *existing = existing.union(&output.weight)
                                    })
                                    .or_insert(output.weight);
                            }
                        }
                    }
                }
            }
        }

        let edges = edge_map
            .into_iter()
            .filter(|(_, weight)| !weight.is_empty())
            .map(|((kind, label, target), weight)| DirectCanonicalEdge {
                kind,
                label,
                target,
                weight,
            })
            .collect::<Vec<_>>();
        let signature = DirectCanonicalSignature {
            final_weight,
            edges,
        };
        let class = if let Some(&existing) = self.class_by_signature.get(&signature) {
            existing
        } else {
            let state_id = self.nwa.add_state();
            if let Some(weight) = signature.final_weight.as_ref() {
                self.nwa.set_final_weight(state_id, weight.clone());
            }
            for edge in &signature.edges {
                if edge.kind == 0 {
                    self.nwa
                        .add_transition(state_id, edge.label, edge.target, edge.weight.clone());
                } else {
                    self.nwa
                        .add_epsilon(state_id, edge.target, edge.weight.clone());
                }
            }
            self.class_by_signature.insert(signature, state_id);
            state_id
        };
        self.visiting.remove(&config);
        self.config_memo.insert(config, class);
        Some(class)
    }
}



fn census_direct_prepush_pending_compact(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) -> Option<(usize, usize, usize, usize, f64)> {
    if summaries
        .states
        .iter()
        .flat_map(|state| state.branches.iter())
        .any(|branch| branch.cross_target_group_id.is_some())
    {
        return None;
    }
    let bundle_started_at = Instant::now();
    let mut used_bundles = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in summaries.states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            if productive
                .get(branch.target as usize)
                .copied()
                .unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                used_bundles[branch.bundle_id] = true;
            }
        }
    }
    let bundles = templates.build_weighted_prepush_bundles_cached(
        &summaries.unique_bundles,
        &used_bundles,
    );
    let bundle_ms = elapsed_ms(bundle_started_at);

    let mut pending = PendingStackInterner::new();
    let mut config_to_id = FxHashMap::<DirectCompactConfig, u32>::default();
    let mut configs = Vec::<DirectCompactConfig>::new();
    let ensure = |config: DirectCompactConfig,
                  config_to_id: &mut FxHashMap<DirectCompactConfig, u32>,
                  configs: &mut Vec<DirectCompactConfig>| {
        if let Some(&existing) = config_to_id.get(&config) {
            return existing;
        }
        let id = configs.len() as u32;
        config_to_id.insert(config.clone(), id);
        configs.push(config);
        id
    };
    for &start in &summaries.start_states {
        if productive.get(start as usize).copied().unwrap_or(false) {
            ensure(
                DirectCompactConfig {
                    node: DirectPrepushNode::Continuation(start),
                    pending: 0,
                },
                &mut config_to_id,
                &mut configs,
            );
        }
    }
    let mut cursor = 0usize;
    let mut edges = 0usize;
    let mut max_pending = 0usize;
    while cursor < configs.len() {
        let config = configs[cursor].clone();
        cursor += 1;
        max_pending = max_pending.max(pending.depth(config.pending));
        match config.node {
            DirectPrepushNode::Continuation(state_id) => {
                let state = &summaries.states[state_id as usize];
                for (target, weight) in &state.epsilon_branches {
                    if weight.is_empty()
                        || !productive.get(*target as usize).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(*target),
                            pending: config.pending,
                        },
                        &mut config_to_id,
                        &mut configs,
                    );
                    edges += 1;
                }
                for branch in &state.branches {
                    if !productive
                        .get(branch.target as usize)
                        .copied()
                        .unwrap_or(false)
                        || !summaries
                            .bundle_accepts
                            .get(branch.bundle_id)
                            .copied()
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    let Some(bundle) = bundles[branch.bundle_id].as_ref() else {
                        continue;
                    };
                    if bundle.states.is_empty() || branch.entry_weight.is_empty() {
                        continue;
                    }
                    ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::BundleCore {
                                bundle_id: branch.bundle_id,
                                target: branch.target,
                                core_state: 0,
                            },
                            pending: config.pending,
                        },
                        &mut config_to_id,
                        &mut configs,
                    );
                    edges += 1;
                }
            }
            DirectPrepushNode::BundleCore {
                bundle_id,
                target,
                core_state,
            } => {
                let state = &bundles[bundle_id]
                    .as_ref()
                    .expect("compact census requires built bundle")
                    .states[core_state as usize];
                if !state.final_weight.is_empty() {
                    ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: config.pending,
                        },
                        &mut config_to_id,
                        &mut configs,
                    );
                    edges += 1;
                }
                for output in &state.outputs {
                    if output.weight.is_empty() {
                        continue;
                    }
                    let next_pending = pending.push_many(config.pending, &output.pushes);
                    ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: next_pending,
                        },
                        &mut config_to_id,
                        &mut configs,
                    );
                    edges += 1;
                }
                let process_transition = |label: i32,
                                              transition: &WeightedPrepushTarget,
                                              pending: &mut PendingStackInterner,
                                              config_to_id: &mut FxHashMap<DirectCompactConfig, u32>,
                                              configs: &mut Vec<DirectCompactConfig>,
                                              edges: &mut usize| {
                    match transition {
                        WeightedPrepushTarget::Core {
                            target: core_target,
                            weight,
                        } => {
                            if weight.is_empty() {
                                return;
                            }
                            let next_pending = if config.pending == 0 {
                                Some(0)
                            } else {
                                pending.pop_matching(config.pending, label)
                            };
                            let Some(next_pending) = next_pending else { return };
                            ensure(
                                DirectCompactConfig {
                                    node: DirectPrepushNode::BundleCore {
                                        bundle_id,
                                        target,
                                        core_state: *core_target,
                                    },
                                    pending: next_pending,
                                },
                                config_to_id,
                                configs,
                            );
                            *edges += 1;
                        }
                        WeightedPrepushTarget::Outputs(outputs) => {
                            for output in outputs {
                                if output.weight.is_empty() {
                                    continue;
                                }
                                let base_pending = if config.pending == 0 {
                                    Some(0)
                                } else {
                                    pending.pop_matching(config.pending, label)
                                };
                                let Some(base_pending) = base_pending else { continue };
                                let next_pending = pending.push_many(base_pending, &output.pushes);
                                ensure(
                                    DirectCompactConfig {
                                        node: DirectPrepushNode::Continuation(target),
                                        pending: next_pending,
                                    },
                                    config_to_id,
                                    configs,
                                );
                                *edges += 1;
                            }
                        }
                    }
                };
                if config.pending == 0 {
                    for (&label, transition) in &state.transitions {
                        process_transition(
                            label,
                            transition,
                            &mut pending,
                            &mut config_to_id,
                            &mut configs,
                            &mut edges,
                        );
                    }
                } else {
                    let top_label = pending
                        .top(config.pending)
                        .expect("nonempty pending stack has a top") as i32;
                    if let Some(transition) = state.transitions.get(&top_label) {
                        process_transition(
                            top_label,
                            transition,
                            &mut pending,
                            &mut config_to_id,
                            &mut configs,
                            &mut edges,
                        );
                    }
                    if let Some(transition) = state.transitions.get(&DEFAULT_LABEL) {
                        process_transition(
                            DEFAULT_LABEL,
                            transition,
                            &mut pending,
                            &mut config_to_id,
                            &mut configs,
                            &mut edges,
                        );
                    }
                }
            }
        }
    }
    Some((configs.len(), edges, pending.entries.len(), max_pending, bundle_ms))
}

fn profile_direct_prepush_pending_census_only(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_PENDING_CENSUS_ONLY").is_none() {
        return;
    }
    let started_at = Instant::now();
    let Some((configs, edges, pending_nodes, max_pending, bundle_ms)) =
        census_direct_prepush_pending_compact(summaries, productive, templates)
    else {
        eprintln!("[glrmask/profile][parser_direct_prepush_pending_census_only] skipped=cross_target");
        return;
    };
    eprintln!(
        "[glrmask/profile][parser_direct_prepush_pending_census_only] configs={} edges={} pending_nodes={} max_pending_depth={} bundle_ms={:.3} total_ms={:.3}",
        configs,
        edges,
        pending_nodes,
        max_pending,
        bundle_ms,
        elapsed_ms(started_at),
    );
}

fn build_direct_prepush_pending_compact_nwa(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) -> Option<(NWA, usize, usize, usize, usize, f64, usize, usize, usize, Vec<u64>)> {
    if summaries
        .states
        .iter()
        .flat_map(|state| state.branches.iter())
        .any(|branch| branch.cross_target_group_id.is_some())
    {
        return None;
    }
    let bundle_started_at = Instant::now();
    let mut used_bundles = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in summaries.states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            if productive
                .get(branch.target as usize)
                .copied()
                .unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                used_bundles[branch.bundle_id] = true;
            }
        }
    }
    let lazy_decoration = std::env::var_os("GLRMASK_PREPUSH_LAZY_DECORATION").is_some();
    let mut lazy_bundles = lazy_decoration.then(|| {
        templates.build_lazy_weighted_prepush_bundles_cached(
            &summaries.unique_bundles,
            &used_bundles,
        )
    });
    let bundles = if lazy_decoration {
        Vec::new()
    } else {
        templates.build_weighted_prepush_bundles_cached(
            &summaries.unique_bundles,
            &used_bundles,
        )
    };
    let bundle_ms = elapsed_ms(bundle_started_at);

    let mut pending = PendingStackInterner::new();
    let mut nwa = NWA::new(0, 0);
    let use_packed_config_key =
        std::env::var_os("GLRMASK_DIRECT_PREPUSH_PACKED_CONFIG_KEY").is_some();
    let config_reserve = std::env::var("GLRMASK_DIRECT_PREPUSH_CONFIG_RESERVE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut packed_config_to_state = FxHashMap::<u128, u32>::default();
    let mut wide_config_to_state = FxHashMap::<DirectCompactConfig, u32>::default();
    if config_reserve > 0 {
        packed_config_to_state.reserve(config_reserve);
        wide_config_to_state.reserve(config_reserve);
    }
    let mut configs = Vec::<DirectCompactConfig>::with_capacity(config_reserve);
    let mut queue = VecDeque::<u32>::with_capacity(config_reserve);
    let ensure = |config: DirectCompactConfig,
                  nwa: &mut NWA,
                  packed_config_to_state: &mut FxHashMap<u128, u32>,
                  wide_config_to_state: &mut FxHashMap<DirectCompactConfig, u32>,
                  configs: &mut Vec<DirectCompactConfig>,
                  queue: &mut VecDeque<u32>| {
        if use_packed_config_key {
            let key = direct_compact_config_key(config);
            if let Some(&existing) = packed_config_to_state.get(&key) {
                return existing;
            }
        } else if let Some(&existing) = wide_config_to_state.get(&config) {
            return existing;
        }
        let state = nwa.add_state();
        if use_packed_config_key {
            packed_config_to_state.insert(direct_compact_config_key(config), state);
        } else {
            wide_config_to_state.insert(config, state);
        }
        configs.push(config);
        queue.push_back(state);
        state
    };
    let mut starts = Vec::<u32>::new();
    for &start in &summaries.start_states {
        if !productive.get(start as usize).copied().unwrap_or(false) {
            continue;
        }
        starts.push(ensure(
            DirectCompactConfig {
                node: DirectPrepushNode::Continuation(start),
                pending: 0,
            },
            &mut nwa,
            &mut packed_config_to_state,
            &mut wide_config_to_state,
            &mut configs,
            &mut queue,
        ));
    }
    starts.sort_unstable();
    starts.dedup();
    nwa.set_start_states(starts);

    let mut max_pending = 0usize;
    let mut edge_count = 0usize;
    let mut unique_core_states = rustc_hash::FxHashSet::<(usize, u32)>::default();
    let mut unique_core_transitions = 0usize;
    let mut unique_core_outputs = 0usize;
    while let Some(nwa_state) = queue.pop_front() {
        let config = configs[nwa_state as usize];
        max_pending = max_pending.max(pending.depth(config.pending));
        match config.node {
            DirectPrepushNode::Continuation(state_id) => {
                let state = &summaries.states[state_id as usize];
                if let Some(final_weight) = state.final_weight.as_ref().filter(|w| !w.is_empty()) {
                    nwa.set_final_weight(nwa_state, final_weight.clone());
                }
                for (target, weight) in &state.epsilon_branches {
                    if weight.is_empty()
                        || !productive.get(*target as usize).copied().unwrap_or(false)
                    {
                        continue;
                    }
                    let target_state = ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(*target),
                            pending: config.pending,
                        },
                        &mut nwa,
                        &mut packed_config_to_state,
            &mut wide_config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, weight.clone());
                    edge_count += 1;
                }
                for branch in &state.branches {
                    if !productive
                        .get(branch.target as usize)
                        .copied()
                        .unwrap_or(false)
                        || !summaries
                            .bundle_accepts
                            .get(branch.bundle_id)
                            .copied()
                            .unwrap_or(false)
                    {
                        continue;
                    }
                    if let Some(set) = lazy_bundles.as_ref() {
                        if templates.lazy_weighted_prepush_bundle_is_empty(set, branch.bundle_id) {
                            continue;
                        }
                    } else {
                        let Some(bundle) = bundles[branch.bundle_id].as_ref() else {
                            continue;
                        };
                        if bundle.states.is_empty() {
                            continue;
                        }
                    }
                    let target_state = ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::BundleCore {
                                bundle_id: branch.bundle_id,
                                target: branch.target,
                                core_state: 0,
                            },
                            pending: config.pending,
                        },
                        &mut nwa,
                        &mut packed_config_to_state,
            &mut wide_config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, branch.entry_weight.clone());
                    edge_count += 1;
                }
            }
            DirectPrepushNode::BundleCore {
                bundle_id,
                target,
                core_state,
            } => {
                let lazy_state = if let Some(set) = lazy_bundles.as_mut() {
                    templates.lazy_weighted_prepush_state(set, bundle_id, core_state)
                } else {
                    None
                };
                let state = if let Some(state) = lazy_state.as_ref() {
                    state.as_ref()
                } else {
                    let bundle = bundles[bundle_id]
                        .as_ref()
                        .expect("compact direct pre-push core requires built bundle");
                    &bundle.states[core_state as usize]
                };
                if unique_core_states.insert((bundle_id, core_state)) {
                    unique_core_transitions += state.transitions.len();
                    unique_core_outputs += state.outputs.len();
                    for transition in state.transitions.values() {
                        if let WeightedPrepushTarget::Outputs(outputs) = transition {
                            unique_core_outputs += outputs.len();
                        }
                    }
                }
                if !state.final_weight.is_empty() {
                    let target_state = ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: config.pending,
                        },
                        &mut nwa,
                        &mut packed_config_to_state,
            &mut wide_config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, state.final_weight.clone());
                    edge_count += 1;
                }
                for output in &state.outputs {
                    if output.weight.is_empty() {
                        continue;
                    }
                    let next_pending = pending.push_many(config.pending, &output.pushes);
                    let target_state = ensure(
                        DirectCompactConfig {
                            node: DirectPrepushNode::Continuation(target),
                            pending: next_pending,
                        },
                        &mut nwa,
                        &mut packed_config_to_state,
            &mut wide_config_to_state,
                        &mut configs,
                        &mut queue,
                    );
                    nwa.add_epsilon(nwa_state, target_state, output.weight.clone());
                    edge_count += 1;
                }
                let process_transition = |label: i32,
                                          transition: &WeightedPrepushTarget,
                                          pending: &mut PendingStackInterner,
                                          nwa: &mut NWA,
                                          packed_config_to_state: &mut FxHashMap<u128, u32>,
                                          wide_config_to_state: &mut FxHashMap<DirectCompactConfig, u32>,
                                          configs: &mut Vec<DirectCompactConfig>,
                                          queue: &mut VecDeque<u32>,
                                          edge_count: &mut usize| {
                    debug_assert!(!is_negative_label(label));
                    match transition {
                        WeightedPrepushTarget::Core {
                            target: core_target,
                            weight,
                        } => {
                            if weight.is_empty() {
                                return;
                            }
                            if config.pending == 0 {
                                let target_state = ensure(
                                    DirectCompactConfig {
                                        node: DirectPrepushNode::BundleCore {
                                            bundle_id,
                                            target,
                                            core_state: *core_target,
                                        },
                                        pending: 0,
                                    },
                                    nwa,
                                    packed_config_to_state,
                                    wide_config_to_state,
                                    configs,
                                    queue,
                                );
                                nwa.add_transition(nwa_state, label, target_state, weight.clone());
                                *edge_count += 1;
                            } else if let Some(next_pending) = pending.pop_matching(config.pending, label) {
                                let target_state = ensure(
                                    DirectCompactConfig {
                                        node: DirectPrepushNode::BundleCore {
                                            bundle_id,
                                            target,
                                            core_state: *core_target,
                                        },
                                        pending: next_pending,
                                    },
                                    nwa,
                                    packed_config_to_state,
                                    wide_config_to_state,
                                    configs,
                                    queue,
                                );
                                nwa.add_epsilon(nwa_state, target_state, weight.clone());
                                *edge_count += 1;
                            }
                        }
                        WeightedPrepushTarget::Outputs(outputs) => {
                            for output in outputs {
                                if output.weight.is_empty() {
                                    continue;
                                }
                                if config.pending == 0 {
                                    let next_pending = pending.push_many(0, &output.pushes);
                                    let target_state = ensure(
                                        DirectCompactConfig {
                                            node: DirectPrepushNode::Continuation(target),
                                            pending: next_pending,
                                        },
                                        nwa,
                                        packed_config_to_state,
                                        wide_config_to_state,
                                        configs,
                                        queue,
                                    );
                                    nwa.add_transition(
                                        nwa_state,
                                        label,
                                        target_state,
                                        output.weight.clone(),
                                    );
                                    *edge_count += 1;
                                } else if let Some(base_pending) =
                                    pending.pop_matching(config.pending, label)
                                {
                                    let next_pending =
                                        pending.push_many(base_pending, &output.pushes);
                                    let target_state = ensure(
                                        DirectCompactConfig {
                                            node: DirectPrepushNode::Continuation(target),
                                            pending: next_pending,
                                        },
                                        nwa,
                                        packed_config_to_state,
                                        wide_config_to_state,
                                        configs,
                                        queue,
                                    );
                                    nwa.add_epsilon(
                                        nwa_state,
                                        target_state,
                                        output.weight.clone(),
                                    );
                                    *edge_count += 1;
                                }
                            }
                        }
                    }
                };
                if config.pending == 0 {
                    for (&label, transition) in &state.transitions {
                        process_transition(
                            label,
                            transition,
                            &mut pending,
                            &mut nwa,
                            &mut packed_config_to_state,
            &mut wide_config_to_state,
                            &mut configs,
                            &mut queue,
                            &mut edge_count,
                        );
                    }
                } else {
                    let top_label = pending
                        .top(config.pending)
                        .expect("nonempty pending stack has a top") as i32;
                    if let Some(transition) = state.transitions.get(&top_label) {
                        process_transition(
                            top_label,
                            transition,
                            &mut pending,
                            &mut nwa,
                            &mut packed_config_to_state,
            &mut wide_config_to_state,
                            &mut configs,
                            &mut queue,
                            &mut edge_count,
                        );
                    }
                    if let Some(transition) = state.transitions.get(&DEFAULT_LABEL) {
                        process_transition(
                            DEFAULT_LABEL,
                            transition,
                            &mut pending,
                            &mut nwa,
                            &mut packed_config_to_state,
            &mut wide_config_to_state,
                            &mut configs,
                            &mut queue,
                            &mut edge_count,
                        );
                    }
                }
            }
        }
    }
    if let Some(set) = lazy_bundles.as_ref() {
        templates.emit_lazy_weighted_prepush_profile(set);
    }
    let color_depth = std::env::var("GLRMASK_PREPUSH_COLOR_PENDING_DEPTH")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1);
    let mut color_by_node =
        FxHashMap::<(DirectPrepushNode, SmallVec<[u32; 8]>), u64>::default();
    let mut context_colors = Vec::with_capacity(configs.len());
    for config in &configs {
        let mut suffix = SmallVec::<[u32; 8]>::new();
        let mut pending_id = config.pending;
        for _ in 0..color_depth {
            if pending_id == 0 {
                break;
            }
            let entry = pending.entries[pending_id as usize];
            suffix.push(entry.top);
            pending_id = entry.parent;
        }
        let key = (config.node.clone(), suffix);
        let next = color_by_node.len() as u64;
        let color = *color_by_node.entry(key).or_insert(next);
        context_colors.push(color);
    }
    Some((
        nwa,
        configs.len(),
        edge_count,
        pending.entries.len(),
        max_pending,
        bundle_ms,
        unique_core_states.len(),
        unique_core_transitions,
        unique_core_outputs,
        context_colors,
    ))
}

fn profile_direct_prepush_pending_compact(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_PENDING_COMPACT").is_none() {
        return;
    }
    let started_at = Instant::now();
    let Some((
        nwa,
        configs,
        edges,
        pending_nodes,
        max_pending,
        bundle_ms,
        unique_core_states,
        unique_core_transitions,
        unique_core_outputs,
        _context_colors,
    )) = build_direct_prepush_pending_compact_nwa(summaries, productive, templates)
    else {
        eprintln!("[glrmask/profile][parser_direct_prepush_pending_compact] skipped=cross_target");
        return;
    };
    eprintln!(
        "[glrmask/profile][parser_direct_prepush_pending_compact] states={} transitions={} configs={} raw_edges={} pending_nodes={} max_pending_depth={} unique_core_states={} unique_core_transitions={} unique_core_outputs={} bundle_ms={:.3} total_ms={:.3}",
        nwa.states().len(),
        NWA::num_transitions(&nwa),
        configs,
        edges,
        pending_nodes,
        max_pending,
        unique_core_states,
        unique_core_transitions,
        unique_core_outputs,
        bundle_ms,
        elapsed_ms(started_at),
    );
}

fn build_direct_prepush_projected_config_nwa(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) -> Option<(NWA, usize, usize, usize, usize, f64)> {
    if summaries
        .states
        .iter()
        .flat_map(|state| state.branches.iter())
        .any(|branch| branch.cross_target_group_id.is_some())
    {
        return None;
    }
    let started = Instant::now();
    let mut used_bundles = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in summaries.states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            if productive
                .get(branch.target as usize)
                .copied()
                .unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                used_bundles[branch.bundle_id] = true;
            }
        }
    }
    let lazy_bundles = templates.build_lazy_weighted_prepush_bundles_cached(
        &summaries.unique_bundles,
        &used_bundles,
    );
    let bundle_ms = elapsed_ms(started);
    let starts = summaries
        .start_states
        .iter()
        .copied()
        .filter(|&start| productive.get(start as usize).copied().unwrap_or(false))
        .map(|start| DirectCompactConfig {
            node: DirectPrepushNode::Continuation(start),
            pending: 0,
        })
        .collect::<Vec<_>>();
    let builder = DirectProjectedConfigBuilder::new(summaries, productive, templates, lazy_bundles);
    if std::env::var_os("GLRMASK_DIRECT_PREPUSH_PROJECTED_CONFIG_FLAT").is_some() {
        let (nwa, raw_configs, raw_edges, removed_configs, _raw_body_calls) =
            builder.build_flat_projected(&starts)?;
        return Some((
            nwa,
            removed_configs,
            removed_configs,
            raw_configs.saturating_sub(removed_configs),
            raw_edges,
            bundle_ms,
        ));
    }
    let (nwa, summarized_configs, summary_entries, summary_hits, raw_edges_examined) =
        builder.build(&starts)?;
    Some((
        nwa,
        summarized_configs,
        summary_entries,
        summary_hits,
        raw_edges_examined,
        bundle_ms,
    ))
}

fn profile_direct_prepush_projected_config(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_PROJECTED_CONFIG").is_none() {
        return;
    }
    let started = Instant::now();
    let Some((
        nwa,
        summarized_configs,
        summary_entries,
        summary_hits,
        raw_edges_examined,
        bundle_ms,
    )) = build_direct_prepush_projected_config_nwa(summaries, productive, templates)
    else {
        eprintln!("[glrmask/profile][parser_direct_prepush_projected_config] skipped=cross_target_or_cycle");
        return;
    };
    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_PROJECTED_CONFIG_SUPPORT").is_some() {
        let (raw, ..) = build_direct_prepush_pending_compact_nwa(summaries, productive, templates)
            .expect("direct projected config validation requires raw compact builder");
        let generic = eliminate_epsilon_only_states(&raw)
            .expect("direct projected config validation requires acyclic epsilon-only projection");
        let direct_dwa = determinize_with_supports(&nwa, None).dwa;
        let generic_dwa = determinize_with_supports(&generic, None).dwa;
        let difference = find_difference(&direct_dwa, &generic_dwa)
            .expect("direct projected config support validation requires acyclic DWAs");
        assert!(
            difference.is_none(),
            "direct config projection changed weighted read language on labels {:?}",
            difference,
        );
        eprintln!(
            "[glrmask/profile][parser_direct_prepush_projected_config_validation] result=equivalent direct_states={} direct_transitions={} generic_states={} generic_transitions={}",
            nwa.states().len(),
            NWA::num_transitions(&nwa),
            generic.states().len(),
            NWA::num_transitions(&generic),
        );
    }
    eprintln!(
        "[glrmask/profile][parser_direct_prepush_projected_config] states={} transitions={} summarized_configs={} summary_entries={} summary_hits={} raw_edges_examined={} bundle_ms={:.3} total_ms={:.3}",
        nwa.states().len(),
        NWA::num_transitions(&nwa),
        summarized_configs,
        summary_entries,
        summary_hits,
        raw_edges_examined,
        bundle_ms,
        elapsed_ms(started),
    );
}

fn build_direct_prepush_compact_hashcons_nwa(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) -> Option<(NWA, usize, usize, usize, usize, f64)> {
    if summaries
        .states
        .iter()
        .flat_map(|state| state.branches.iter())
        .any(|branch| branch.cross_target_group_id.is_some())
    {
        return None;
    }
    let bundle_started_at = Instant::now();
    let mut used_bundles = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in summaries.states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            if productive
                .get(branch.target as usize)
                .copied()
                .unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                used_bundles[branch.bundle_id] = true;
            }
        }
    }
    let bundles = templates.build_weighted_prepush_bundles_cached(
        &summaries.unique_bundles,
        &used_bundles,
    );
    let bundle_ms = elapsed_ms(bundle_started_at);
    let mut builder = DirectCompactCanonicalBuilder::new(summaries, productive, &bundles);
    let mut starts = Vec::<u32>::new();
    for &start in &summaries.start_states {
        if !productive.get(start as usize).copied().unwrap_or(false) {
            continue;
        }
        starts.push(builder.class_for(DirectCompactConfig {
            node: DirectPrepushNode::Continuation(start),
            pending: 0,
        })?);
    }
    starts.sort_unstable();
    starts.dedup();
    builder.nwa.set_start_states(starts);
    Some((
        builder.nwa,
        builder.raw_configs,
        builder.raw_edges,
        builder.max_pending,
        builder.pending.entries.len(),
        bundle_ms,
    ))
}

fn profile_direct_prepush_compact_hashcons(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_COMPACT_HASHCONS").is_none() {
        return;
    }
    let started_at = Instant::now();
    let Some((nwa, raw_configs, raw_edges, max_pending, pending_nodes, bundle_ms)) =
        build_direct_prepush_compact_hashcons_nwa(summaries, productive, templates)
    else {
        eprintln!("[glrmask/profile][parser_direct_prepush_compact_hashcons] skipped=cross_target");
        return;
    };
    eprintln!(
        "[glrmask/profile][parser_direct_prepush_compact_hashcons] states={} transitions={} raw_configs={} raw_edges={} pending_nodes={} max_pending_depth={} bundle_ms={:.3} total_ms={:.3}",
        nwa.states().len(),
        NWA::num_transitions(&nwa),
        raw_configs,
        raw_edges,
        pending_nodes,
        max_pending,
        bundle_ms,
        elapsed_ms(started_at),
    );
}

fn build_direct_prepush_hashcons_nwa(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) -> Option<(NWA, usize, usize, usize, f64)> {
    if summaries
        .states
        .iter()
        .flat_map(|state| state.branches.iter())
        .any(|branch| branch.cross_target_group_id.is_some())
    {
        return None;
    }
    let bundle_started_at = Instant::now();
    let mut used_bundles = vec![false; summaries.unique_bundles.len()];
    for (state_id, state) in summaries.states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        for branch in &state.branches {
            if productive
                .get(branch.target as usize)
                .copied()
                .unwrap_or(false)
                && summaries
                    .bundle_accepts
                    .get(branch.bundle_id)
                    .copied()
                    .unwrap_or(false)
            {
                used_bundles[branch.bundle_id] = true;
            }
        }
    }
    let bundles = templates.build_weighted_prepush_bundles_cached(
        &summaries.unique_bundles,
        &used_bundles,
    );
    let bundle_ms = elapsed_ms(bundle_started_at);
    let mut builder = DirectCanonicalBuilder::new(summaries, productive, &bundles);
    let mut starts = Vec::<u32>::new();
    for &start in &summaries.start_states {
        if !productive.get(start as usize).copied().unwrap_or(false) {
            continue;
        }
        starts.push(builder.class_for(DirectPrepushConfig {
            node: DirectPrepushNode::Continuation(start),
            pending: SmallVec::new(),
        })?);
    }
    starts.sort_unstable();
    starts.dedup();
    builder.nwa.set_start_states(starts);
    Some((
        builder.nwa,
        builder.raw_configs,
        builder.raw_edges,
        builder.max_pending,
        bundle_ms,
    ))
}

fn profile_direct_prepush_hashcons(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_HASHCONS").is_none() {
        return;
    }
    let started_at = Instant::now();
    let Some((nwa, raw_configs, raw_edges, max_pending, bundle_ms)) =
        build_direct_prepush_hashcons_nwa(summaries, productive, templates)
    else {
        eprintln!("[glrmask/profile][parser_direct_prepush_hashcons] skipped=cross_target");
        return;
    };
    let transitions = NWA::num_transitions(&nwa);
    eprintln!(
        "[glrmask/profile][parser_direct_prepush_hashcons] states={} transitions={} raw_configs={} raw_edges={} max_pending_depth={} bundle_ms={:.3} total_ms={:.3}",
        nwa.states().len(),
        transitions,
        raw_configs,
        raw_edges,
        max_pending,
        bundle_ms,
        elapsed_ms(started_at),
    );
}




fn coalesce_parallel_nwa_edges(
    nwa: &mut NWA,
    weight_ops: &mut ScopedWeightOpCache,
) -> (usize, usize) {
    let mut transition_merges = 0usize;
    let mut epsilon_merges = 0usize;
    for state in nwa.states_mut() {
        for targets in state.transitions.values_mut() {
            if targets.len() < 2 {
                continue;
            }
            targets.sort_unstable_by_key(|(target, _)| *target);
            let mut write = 0usize;
            for read in 1..targets.len() {
                if targets[write].0 == targets[read].0 {
                    targets[write].1 = weight_ops.union(&targets[write].1, &targets[read].1);
                    transition_merges += 1;
                } else {
                    write += 1;
                    if write != read {
                        targets[write] = targets[read].clone();
                    }
                }
            }
            targets.truncate(write + 1);
            targets.retain(|(_, weight)| !weight.is_empty());
        }
        if state.epsilons.len() >= 2 {
            state.epsilons.sort_unstable_by_key(|(target, _)| *target);
            let mut write = 0usize;
            for read in 1..state.epsilons.len() {
                if state.epsilons[write].0 == state.epsilons[read].0 {
                    state.epsilons[write].1 =
                        weight_ops.union(&state.epsilons[write].1, &state.epsilons[read].1);
                    epsilon_merges += 1;
                } else {
                    write += 1;
                    if write != read {
                        state.epsilons[write] = state.epsilons[read].clone();
                    }
                }
            }
            state.epsilons.truncate(write + 1);
            state.epsilons.retain(|(_, weight)| !weight.is_empty());
        }
    }
    (transition_merges, epsilon_merges)
}

fn eliminate_epsilon_only_states_with_origins(
    nwa: &NWA,
) -> Option<(NWA, Vec<Option<u32>>)> {
    let n = nwa.states().len();
    if n == 0 {
        return Some((NWA::new(0, 0), Vec::new()));
    }
    let mut is_start = vec![false; n];
    for &start in nwa.start_states() {
        if (start as usize) < n {
            is_start[start as usize] = true;
        }
    }
    let retained = nwa
        .states()
        .iter()
        .enumerate()
        .map(|(state_id, state)| is_start[state_id] || !state.transitions.is_empty())
        .collect::<Vec<_>>();
    let removed_count = retained.iter().filter(|&&keep| !keep).count();
    if removed_count == 0 {
        return Some((
            nwa.clone(),
            (0..n).map(|state_id| Some(state_id as u32)).collect(),
        ));
    }
    let mut removed_outdegree = vec![0usize; n];
    let mut removed_predecessors = vec![Vec::<usize>::new(); n];
    for (source, state) in nwa.states().iter().enumerate() {
        if retained[source] {
            continue;
        }
        debug_assert!(state.transitions.is_empty());
        for (target, weight) in &state.epsilons {
            let target = *target as usize;
            if weight.is_empty() || target >= n || retained[target] {
                continue;
            }
            removed_outdegree[source] += 1;
            removed_predecessors[target].push(source);
        }
    }
    let mut summary_final = vec![None::<Weight>; n];
    let mut summary_exits = vec![Vec::<(u32, Weight)>::new(); n];
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut queue = VecDeque::<usize>::new();
    for state_id in 0..n {
        if !retained[state_id] && removed_outdegree[state_id] == 0 {
            queue.push_back(state_id);
        }
    }
    let mut processed = 0usize;
    let mut total_exit_refs = 0usize;
    let mut max_exits = 0usize;
    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states()[state_id];
        let mut final_weight = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
            .cloned();
        let mut exits = SmallVec::<[(u32, Weight); 4]>::new();
        for (target, edge_weight) in &state.epsilons {
            if edge_weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            let target_idx = *target as usize;
            if retained[target_idx] {
                if let Some((_, existing)) = exits.iter_mut().find(|(exit, _)| *exit == *target) {
                    *existing = weight_ops.union(existing, edge_weight);
                } else {
                    exits.push((*target, edge_weight.clone()));
                }
                continue;
            }
            if let Some(target_final) = summary_final[target_idx].as_ref() {
                let contribution = weight_ops.intersection(edge_weight, target_final);
                if !contribution.is_empty() {
                    final_weight = Some(match final_weight {
                        Some(existing) => weight_ops.union(&existing, &contribution),
                        None => contribution,
                    });
                }
            }
            for (exit, suffix_weight) in &summary_exits[target_idx] {
                let contribution = weight_ops.intersection(edge_weight, suffix_weight);
                if contribution.is_empty() {
                    continue;
                }
                if let Some((_, existing)) = exits
                    .iter_mut()
                    .find(|(existing_exit, _)| *existing_exit == *exit)
                {
                    *existing = weight_ops.union(existing, &contribution);
                } else {
                    exits.push((*exit, contribution));
                }
            }
        }
        exits.sort_unstable_by_key(|(exit, _)| *exit);
        let exits = exits.into_vec();
        total_exit_refs += exits.len();
        max_exits = max_exits.max(exits.len());
        summary_final[state_id] = final_weight;
        summary_exits[state_id] = exits;
        processed += 1;
        for &pred in &removed_predecessors[state_id] {
            removed_outdegree[pred] -= 1;
            if removed_outdegree[pred] == 0 {
                queue.push_back(pred);
            }
        }
    }
    if processed != removed_count {
        return None;
    }
    let mut result = NWA::new(0, 0);
    let mut new_by_old = vec![u32::MAX; n];
    let mut raw_by_new = Vec::<Option<u32>>::new();
    for (state_id, &keep) in retained.iter().enumerate() {
        if keep {
            new_by_old[state_id] = result.add_state();
            raw_by_new.push(Some(state_id as u32));
        }
    }
    let final_sink = result.add_state();
    raw_by_new.push(None);
    result.set_final_weight(final_sink, Weight::all());

    for (source, state) in nwa.states().iter().enumerate() {
        if !retained[source] {
            continue;
        }
        let new_source = new_by_old[source];
        let mut source_final = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
            .cloned();
        for (target, edge_weight) in &state.epsilons {
            if edge_weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            let target_idx = *target as usize;
            if retained[target_idx] {
                result.add_epsilon(new_source, new_by_old[target_idx], edge_weight.clone());
                continue;
            }
            if let Some(target_final) = summary_final[target_idx].as_ref() {
                let contribution = weight_ops.intersection(edge_weight, target_final);
                if !contribution.is_empty() {
                    source_final = Some(match source_final {
                        Some(existing) => weight_ops.union(&existing, &contribution),
                        None => contribution,
                    });
                }
            }
            for (exit, suffix_weight) in &summary_exits[target_idx] {
                let contribution = weight_ops.intersection(edge_weight, suffix_weight);
                if !contribution.is_empty() {
                    result.add_epsilon(
                        new_source,
                        new_by_old[*exit as usize],
                        contribution,
                    );
                }
            }
        }
        if let Some(weight) = source_final {
            result.set_final_weight(new_source, weight);
        }
        for (&label, targets) in &state.transitions {
            for (target, edge_weight) in targets {
                if edge_weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                let target_idx = *target as usize;
                if retained[target_idx] {
                    result.add_transition(
                        new_source,
                        label,
                        new_by_old[target_idx],
                        edge_weight.clone(),
                    );
                    continue;
                }
                if let Some(target_final) = summary_final[target_idx].as_ref() {
                    let contribution = weight_ops.intersection(edge_weight, target_final);
                    if !contribution.is_empty() {
                        result.add_transition(new_source, label, final_sink, contribution);
                    }
                }
                for (exit, suffix_weight) in &summary_exits[target_idx] {
                    let contribution = weight_ops.intersection(edge_weight, suffix_weight);
                    if !contribution.is_empty() {
                        result.add_transition(
                            new_source,
                            label,
                            new_by_old[*exit as usize],
                            contribution,
                        );
                    }
                }
            }
        }
    }
    let starts = nwa
        .start_states()
        .iter()
        .filter_map(|&state| {
            let state = state as usize;
            (state < n && retained[state]).then_some(new_by_old[state])
        })
        .collect::<Vec<_>>();
    result.set_start_states(starts);
    let coalesce_started = Instant::now();
    let (transition_merges, epsilon_merges) =
        coalesce_parallel_nwa_edges(&mut result, &mut weight_ops);
    let coalesce_ms = elapsed_ms(coalesce_started);
    eprintln!(
        "[glrmask/profile][parser_direct_epsilon_only_elimination] input_states={} retained_states={} removed_states={} output_states={} output_transitions={} summary_exit_refs={} summary_max_exits={} transition_merges={} epsilon_merges={} coalesce_ms={:.3}",
        n,
        retained.iter().filter(|&&keep| keep).count(),
        removed_count,
        result.states().len(),
        NWA::num_transitions(&result),
        total_exit_refs,
        max_exits,
        transition_merges,
        epsilon_merges,
        coalesce_ms,
    );
    Some((result, raw_by_new))
}

fn eliminate_epsilon_only_states(nwa: &NWA) -> Option<NWA> {
    eliminate_epsilon_only_states_with_origins(nwa).map(|(projected, _)| projected)
}

struct RawSupportIdentityOracle<'a> {
    raw_nwa: &'a NWA,
    raw_by_projected: &'a [Option<u32>],
    projected_weighted_supports: &'a [Vec<(u32, Weight)>],
    source_cache: FxHashMap<u32, Vec<(u32, Weight)>>,
    target_cache: FxHashMap<(u32, i32), Vec<(u32, Weight)>>,
    weight_by_state: Vec<Option<Weight>>,
    closure_queue: VecDeque<u32>,
    touched: Vec<u32>,
    canon: Vec<(u32, Weight)>,
    weight_ops: ScopedWeightOpCache,
}

impl<'a> RawSupportIdentityOracle<'a> {
    fn new(
        raw_nwa: &'a NWA,
        raw_by_projected: &'a [Option<u32>],
        projected_weighted_supports: &'a [Vec<(u32, Weight)>],
    ) -> Self {
        Self {
            raw_nwa,
            raw_by_projected,
            projected_weighted_supports,
            source_cache: FxHashMap::default(),
            target_cache: FxHashMap::default(),
            weight_by_state: vec![None; raw_nwa.states().len()],
            closure_queue: VecDeque::new(),
            touched: Vec::new(),
            canon: Vec::new(),
            weight_ops: ScopedWeightOpCache::default(),
        }
    }

    fn raw_source_support(&mut self, projected_dwa_state: u32) -> &[(u32, Weight)] {
        if !self.source_cache.contains_key(&projected_dwa_state) {
            let mut seeds = Vec::new();
            for (projected_nwa_state, weight) in
                &self.projected_weighted_supports[projected_dwa_state as usize]
            {
                let Some(raw_state) = self
                    .raw_by_projected
                    .get(*projected_nwa_state as usize)
                    .copied()
                    .flatten()
                else {
                    continue;
                };
                seeds.push((raw_state, weight.clone()));
            }
            seeds.sort_unstable_by_key(|(state_id, _)| *state_id);
            local_epsilon_closure_canonical(
                self.raw_nwa,
                &mut self.weight_by_state,
                &mut self.closure_queue,
                &seeds,
                &mut self.touched,
                &mut self.canon,
                &mut self.weight_ops,
            );
            self.source_cache
                .insert(projected_dwa_state, self.canon.clone());
        }
        &self.source_cache[&projected_dwa_state]
    }

    fn raw_target_support(&mut self, projected_dwa_state: u32, label: i32) -> &[(u32, Weight)] {
        let cache_key = (projected_dwa_state, label);
        if !self.target_cache.contains_key(&cache_key) {
            let source = self.raw_source_support(projected_dwa_state).to_vec();
            let mut contributions = TargetContribs::new();
            for (raw_state, path_weight) in source {
                let Some(targets) = self.raw_nwa.states()[raw_state as usize].transitions.get(&label)
                else {
                    continue;
                };
                for (target, edge_weight) in targets {
                    let contribution = self.weight_ops.intersection(&path_weight, edge_weight);
                    if !contribution.is_empty() {
                        contributions.push((*target, contribution));
                    }
                }
            }
            contributions.sort_unstable_by_key(|(state_id, _)| *state_id);
            merge_sorted_target_contributions(&mut contributions, &mut self.weight_ops, None);
            local_epsilon_closure_canonical(
                self.raw_nwa,
                &mut self.weight_by_state,
                &mut self.closure_queue,
                &contributions,
                &mut self.touched,
                &mut self.canon,
                &mut self.weight_ops,
            );
            self.target_cache.insert(cache_key, self.canon.clone());
        }
        &self.target_cache[&cache_key]
    }

    fn labels_share_raw_target(
        &mut self,
        projected_dwa_state: u32,
        possible_ids: &PossibleOutgoingIds,
        num_parser_states: u32,
    ) -> bool {
        let labels = match possible_ids {
            PossibleOutgoingIds::Empty => return false,
            PossibleOutgoingIds::All => (0..num_parser_states).collect::<Vec<_>>(),
            PossibleOutgoingIds::Some(ids) => ids.iter_ones().map(|id| id as u32).collect(),
        };
        let Some((&first, rest)) = labels.split_first() else {
            return false;
        };
        let first_target = self
            .raw_target_support(projected_dwa_state, first as i32)
            .to_vec();
        rest.iter().all(|label| {
            self.raw_target_support(projected_dwa_state, *label as i32) == first_target.as_slice()
        })
    }
}

fn hashcons_acyclic_weighted_nwa(nwa: &NWA) -> Option<NWA> {
    let n = nwa.states().len();
    let mut outdegree = vec![0usize; n];
    let mut predecessors = vec![Vec::<usize>::new(); n];
    for (source, state) in nwa.states().iter().enumerate() {
        for targets in state.transitions.values() {
            for (target, weight) in targets {
                if weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                outdegree[source] += 1;
                predecessors[*target as usize].push(source);
            }
        }
        for (target, weight) in &state.epsilons {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            outdegree[source] += 1;
            predecessors[*target as usize].push(source);
        }
    }
    let mut queue = VecDeque::<usize>::new();
    for (state, &degree) in outdegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state);
        }
    }
    let mut class_by_state = vec![u32::MAX; n];
    let mut class_by_signature =
        FxHashMap::<(Option<Weight>, Vec<(u8, i32, u32, Weight)>), u32>::default();
    let mut result = NWA::new(0, 0);
    let mut processed = 0usize;
    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states()[state_id];
        let mut edge_map = BTreeMap::<(u8, i32, u32), Weight>::new();
        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                if weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                let target_class = class_by_state[*target as usize];
                if target_class == u32::MAX {
                    return None;
                }
                edge_map
                    .entry((0, label, target_class))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
        for (target, weight) in &state.epsilons {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            let target_class = class_by_state[*target as usize];
            if target_class == u32::MAX {
                return None;
            }
            edge_map
                .entry((1, 0, target_class))
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }
        let edges = edge_map
            .into_iter()
            .filter(|(_, weight)| !weight.is_empty())
            .map(|((kind, label, target), weight)| (kind, label, target, weight))
            .collect::<Vec<_>>();
        let final_weight = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
            .cloned();
        let signature = (final_weight.clone(), edges.clone());
        let class = if let Some(&existing) = class_by_signature.get(&signature) {
            existing
        } else {
            let new_state = result.add_state();
            if let Some(weight) = final_weight {
                result.set_final_weight(new_state, weight);
            }
            for (kind, label, target, weight) in &edges {
                if *kind == 0 {
                    result.add_transition(new_state, *label, *target, weight.clone());
                } else {
                    result.add_epsilon(new_state, *target, weight.clone());
                }
            }
            class_by_signature.insert(signature, new_state);
            new_state
        };
        class_by_state[state_id] = class;
        processed += 1;
        for &pred in &predecessors[state_id] {
            outdegree[pred] -= 1;
            if outdegree[pred] == 0 {
                queue.push_back(pred);
            }
        }
    }
    if processed != n {
        return None;
    }
    let mut starts = nwa
        .start_states()
        .iter()
        .filter_map(|&state| class_by_state.get(state as usize).copied())
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    result.set_start_states(starts);
    Some(result)
}


fn hashcons_acyclic_weighted_nwa_with_origins(nwa: &NWA) -> Option<(NWA, Vec<Vec<u32>>)> {
    let n = nwa.states().len();
    let mut outdegree = vec![0usize; n];
    let mut predecessors = vec![Vec::<usize>::new(); n];
    for (source, state) in nwa.states().iter().enumerate() {
        for targets in state.transitions.values() {
            for (target, weight) in targets {
                if weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                outdegree[source] += 1;
                predecessors[*target as usize].push(source);
            }
        }
        for (target, weight) in &state.epsilons {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            outdegree[source] += 1;
            predecessors[*target as usize].push(source);
        }
    }
    let mut queue = VecDeque::<usize>::new();
    for (state, &degree) in outdegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state);
        }
    }
    let mut class_by_state = vec![u32::MAX; n];
    let mut class_by_signature =
        FxHashMap::<(Option<Weight>, Vec<(u8, i32, u32, Weight)>), u32>::default();
    let mut result = NWA::new(0, 0);
    let mut origins = Vec::<Vec<u32>>::new();
    let mut processed = 0usize;
    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states()[state_id];
        let mut edge_map = BTreeMap::<(u8, i32, u32), Weight>::new();
        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                if weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                let target_class = class_by_state[*target as usize];
                if target_class == u32::MAX {
                    return None;
                }
                edge_map
                    .entry((0, label, target_class))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
        for (target, weight) in &state.epsilons {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            let target_class = class_by_state[*target as usize];
            if target_class == u32::MAX {
                return None;
            }
            edge_map
                .entry((1, 0, target_class))
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }
        let edges = edge_map
            .into_iter()
            .filter(|(_, weight)| !weight.is_empty())
            .map(|((kind, label, target), weight)| (kind, label, target, weight))
            .collect::<Vec<_>>();
        let final_weight = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
            .cloned();
        let signature = (final_weight.clone(), edges.clone());
        let class = if let Some(&existing) = class_by_signature.get(&signature) {
            origins[existing as usize].push(state_id as u32);
            existing
        } else {
            let new_state = result.add_state();
            if let Some(weight) = final_weight {
                result.set_final_weight(new_state, weight);
            }
            for (kind, label, target, weight) in &edges {
                if *kind == 0 {
                    result.add_transition(new_state, *label, *target, weight.clone());
                } else {
                    result.add_epsilon(new_state, *target, weight.clone());
                }
            }
            class_by_signature.insert(signature, new_state);
            origins.push(vec![state_id as u32]);
            new_state
        };
        class_by_state[state_id] = class;
        processed += 1;
        for &pred in &predecessors[state_id] {
            outdegree[pred] -= 1;
            if outdegree[pred] == 0 {
                queue.push_back(pred);
            }
        }
    }
    if processed != n {
        return None;
    }
    let mut starts = nwa
        .start_states()
        .iter()
        .filter_map(|&state| class_by_state.get(state as usize).copied())
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    result.set_start_states(starts);
    Some((result, origins))
}

fn build_possible_outgoing_ids_by_quotient_origins(
    raw_nwa: &NWA,
    quotient_supports: &[Vec<u32>],
    origins_by_quotient_state: &[Vec<u32>],
    num_parser_states: u32,
) -> Vec<PossibleOutgoingIds> {
    let expanded_supports = quotient_supports
        .iter()
        .map(|support| {
            let mut raw = Vec::<u32>::new();
            for &quotient_state in support {
                if let Some(origins) = origins_by_quotient_state.get(quotient_state as usize) {
                    raw.extend_from_slice(origins);
                }
            }
            raw.sort_unstable();
            raw.dedup();
            raw
        })
        .collect::<Vec<_>>();
    build_possible_outgoing_ids_by_state(raw_nwa, &expanded_supports, num_parser_states)
}


fn hashcons_acyclic_weighted_nwa_colored(nwa: &NWA, colors: &[u64]) -> Option<NWA> {
    let n = nwa.states().len();
    if colors.len() != n {
        return None;
    }
    let mut outdegree = vec![0usize; n];
    let mut predecessors = vec![Vec::<usize>::new(); n];
    for (source, state) in nwa.states().iter().enumerate() {
        for targets in state.transitions.values() {
            for (target, weight) in targets {
                if weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                outdegree[source] += 1;
                predecessors[*target as usize].push(source);
            }
        }
        for (target, weight) in &state.epsilons {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            outdegree[source] += 1;
            predecessors[*target as usize].push(source);
        }
    }
    let mut queue = VecDeque::<usize>::new();
    for (state, &degree) in outdegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state);
        }
    }
    let mut class_by_state = vec![u32::MAX; n];
    let mut class_by_signature = FxHashMap::<
        (u64, Option<Weight>, Vec<(u8, i32, u32, Weight)>),
        u32,
    >::default();
    let mut result = NWA::new(0, 0);
    let mut processed = 0usize;
    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states()[state_id];
        let mut edge_map = BTreeMap::<(u8, i32, u32), Weight>::new();
        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                if weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                let target_class = class_by_state[*target as usize];
                if target_class == u32::MAX {
                    return None;
                }
                edge_map
                    .entry((0, label, target_class))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
        for (target, weight) in &state.epsilons {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            let target_class = class_by_state[*target as usize];
            if target_class == u32::MAX {
                return None;
            }
            edge_map
                .entry((1, 0, target_class))
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
        }
        let edges = edge_map
            .into_iter()
            .filter(|(_, weight)| !weight.is_empty())
            .map(|((kind, label, target), weight)| (kind, label, target, weight))
            .collect::<Vec<_>>();
        let final_weight = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
            .cloned();
        let signature = (colors[state_id], final_weight.clone(), edges.clone());
        let class = if let Some(&existing) = class_by_signature.get(&signature) {
            existing
        } else {
            let new_state = result.add_state();
            if let Some(weight) = final_weight {
                result.set_final_weight(new_state, weight);
            }
            for (kind, label, target, weight) in &edges {
                if *kind == 0 {
                    result.add_transition(new_state, *label, *target, weight.clone());
                } else {
                    result.add_epsilon(new_state, *target, weight.clone());
                }
            }
            class_by_signature.insert(signature, new_state);
            new_state
        };
        class_by_state[state_id] = class;
        processed += 1;
        for &pred in &predecessors[state_id] {
            outdegree[pred] -= 1;
            if outdegree[pred] == 0 {
                queue.push_back(pred);
            }
        }
    }
    if processed != n {
        return None;
    }
    let mut starts = nwa
        .start_states()
        .iter()
        .filter_map(|&state| class_by_state.get(state as usize).copied())
        .collect::<Vec<_>>();
    starts.sort_unstable();
    starts.dedup();
    result.set_start_states(starts);
    Some(result)
}

fn exact_acyclic_weighted_structural_classes(nwa: &NWA) -> Option<(usize, usize)> {
    let n = nwa.states().len();
    let mut outdegree = vec![0usize; n];
    let mut predecessors = vec![Vec::<usize>::new(); n];
    for (source, state) in nwa.states().iter().enumerate() {
        for targets in state.transitions.values() {
            for (target, weight) in targets {
                if weight.is_empty() || (*target as usize) >= n {
                    continue;
                }
                outdegree[source] += 1;
                predecessors[*target as usize].push(source);
            }
        }
        for (target, weight) in &state.epsilons {
            if weight.is_empty() || (*target as usize) >= n {
                continue;
            }
            outdegree[source] += 1;
            predecessors[*target as usize].push(source);
        }
    }
    let mut queue = VecDeque::<usize>::new();
    for (state, &degree) in outdegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state);
        }
    }
    let mut class_by_state = vec![u32::MAX; n];
    let mut classes = FxHashMap::<(Option<Weight>, Vec<(u8, i32, u32, Weight)>), u32>::default();
    let mut processed = 0usize;
    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states()[state_id];
        let mut edge_map = BTreeMap::<(u8, i32, u32), Weight>::new();
        let mut add_edge = |kind: u8, label: i32, target: u32, weight: &Weight| -> Option<()> {
            if weight.is_empty() || (target as usize) >= n {
                return Some(());
            }
            let target_class = class_by_state[target as usize];
            if target_class == u32::MAX {
                return None;
            }
            edge_map
                .entry((kind, label, target_class))
                .and_modify(|existing| *existing = existing.union(weight))
                .or_insert_with(|| weight.clone());
            Some(())
        };
        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                add_edge(0, label, *target, weight)?;
            }
        }
        for (target, weight) in &state.epsilons {
            add_edge(1, 0, *target, weight)?;
        }
        let edges = edge_map
            .into_iter()
            .filter(|(_, weight)| !weight.is_empty())
            .map(|((kind, label, target_class), weight)| (kind, label, target_class, weight))
            .collect::<Vec<_>>();
        let final_weight = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
            .cloned();
        let next = classes.len() as u32;
        let class = *classes.entry((final_weight, edges)).or_insert(next);
        class_by_state[state_id] = class;
        processed += 1;
        for &pred in &predecessors[state_id] {
            outdegree[pred] -= 1;
            if outdegree[pred] == 0 {
                queue.push_back(pred);
            }
        }
    }
    (processed == n).then_some((classes.len(), processed))
}

fn profile_direct_prepush_pending(
    summaries: &StateSummaries,
    productive: &[bool],
    templates: &Templates,
) {
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_PENDING").is_none() {
        return;
    }
    let started_at = Instant::now();
    let Some((nwa, max_pending, pending_configs, unique_pending)) =
        build_direct_prepush_pending_nwa(summaries, productive, templates)
    else {
        eprintln!("[glrmask/profile][parser_direct_prepush_pending] skipped=cross_target");
        return;
    };
    let transitions = nwa
        .states()
        .iter()
        .map(|state| {
            state.epsilons.len()
                + state
                    .transitions
                    .values()
                    .map(Vec::len)
                    .sum::<usize>()
        })
        .sum::<usize>();
    let finals = nwa
        .states()
        .iter()
        .filter(|state| state.final_weight.as_ref().is_some_and(|w| !w.is_empty()))
        .count();
    let structural_started_at = Instant::now();
    let structural = exact_acyclic_weighted_structural_classes(&nwa);
    let structural_ms = elapsed_ms(structural_started_at);
    eprintln!(
        "[glrmask/profile][parser_direct_prepush_pending] states={} transitions={} finals={} pending_configs={} unique_pending_stacks={} max_pending_depth={} structural_classes={} structural_acyclic={} structural_ms={:.3} build_ms={:.3}",
        nwa.states().len(),
        transitions,
        finals,
        pending_configs,
        unique_pending,
        max_pending,
        structural.map(|value| value.0).unwrap_or(0),
        structural.is_some(),
        structural_ms,
        elapsed_ms(started_at),
    );
}

fn build_parser_nwa_from_terminal_dwa(
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
    use_prepush_reconstructed_bundles: bool,
) -> Option<(NWA, ParserNwaBuildProfile)> {
    let total_started_at = Instant::now();
    let state_prep_started_at = Instant::now();
    let mut summaries = build_state_summaries(terminal_dwa, grammar, templates);
    profile_parser_bundle_gate_factorability(&summaries);
    profile_parser_bundle_cross_target_gate_factorability(&summaries, templates);
    factor_parser_bundle_entry_gates(&mut summaries, templates);
    factor_parser_bundle_cross_target_gates(&mut summaries, templates);
    let productive = compute_productive_terminal_states(&summaries);
    profile_direct_prepush_pending(&summaries, &productive, templates);
    profile_direct_prepush_hashcons(&summaries, &productive, templates);
    profile_direct_prepush_compact_hashcons(&summaries, &productive, templates);
    profile_direct_prepush_pending_compact(&summaries, &productive, templates);
    profile_direct_prepush_projected_config(&summaries, &productive, templates);
    profile_direct_prepush_pending_census_only(&summaries, &productive, templates);
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
    if compose_detail_enabled {
        let terminal_sets = summaries
            .unique_bundles
            .iter()
            .map(|bundle| bundle.keys().copied().collect::<Vec<_>>())
            .collect::<rustc_hash::FxHashSet<_>>();
        let topology_signatures = summaries
            .unique_bundles
            .iter()
            .map(bundle_topology_signature)
            .collect::<rustc_hash::FxHashSet<_>>();
        eprintln!(
            "[glrmask/profile][parser_bundle_reuse] unique_bundles={} unique_terminal_sets={} unique_weight_partition_topologies={} terminal_set_reuse={:.2} topology_reuse={:.2}",
            summaries.unique_bundles.len(),
            terminal_sets.len(),
            topology_signatures.len(),
            summaries.unique_bundles.len() as f64 / terminal_sets.len().max(1) as f64,
            summaries.unique_bundles.len() as f64 / topology_signatures.len().max(1) as f64,
        );
    }

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
    let mut cross_target_fragment_memo: FxHashMap<usize, NwaBody> = FxHashMap::default();
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

    if std::env::var_os("GLRMASK_PROFILE_PREPUSH_RECONSTRUCTED_BATCH").is_some() {
        let started = Instant::now();
        let mut bundles_built = 0usize;
        let mut states = 0usize;
        let mut transitions = 0usize;
        for (bundle_id, bundle) in summaries.unique_bundles.iter().enumerate() {
            if !used_multi_bundle[bundle_id] {
                continue;
            }
            let reconstructed = templates.build_prepush_reconstructed_bundle(bundle);
            bundles_built += 1;
            states += reconstructed.states().len();
            transitions += NWA::num_transitions(&reconstructed);
        }
        eprintln!(
            "[glrmask/profile][prepush_reconstructed_batch] bundles={} states={} transitions={} total_ms={:.3}",
            bundles_built,
            states,
            transitions,
            elapsed_ms(started),
        );
    }

    if std::env::var_os("GLRMASK_PROFILE_PREPUSH_COMPACT_WRITE_TRIE_BATCH").is_some() {
        let started = Instant::now();
        let mut bundles_built = 0usize;
        let mut states = 0usize;
        let mut transitions = 0usize;
        for (bundle_id, bundle) in summaries.unique_bundles.iter().enumerate() {
            if !used_multi_bundle[bundle_id] {
                continue;
            }
            let compact = templates.build_prepush_compact_write_trie_bundle(bundle);
            bundles_built += 1;
            states += compact.states().len();
            transitions += NWA::num_transitions(&compact);
        }
        eprintln!(
            "[glrmask/profile][prepush_compact_write_trie_batch] bundles={} states={} transitions={} total_ms={:.3}",
            bundles_built,
            states,
            transitions,
            elapsed_ms(started),
        );
    }

    if std::env::var_os("GLRMASK_PROFILE_PREPUSH_FRONTIER_WRITE_TRIE_BATCH").is_some() {
        let started = Instant::now();
        let mut bundles_built = 0usize;
        let mut states = 0usize;
        let mut transitions = 0usize;
        for (bundle_id, bundle) in summaries.unique_bundles.iter().enumerate() {
            if !used_multi_bundle[bundle_id] {
                continue;
            }
            let compact = templates.build_prepush_frontier_write_trie_bundle(bundle);
            bundles_built += 1;
            states += compact.states().len();
            transitions += NWA::num_transitions(&compact);
        }
        eprintln!(
            "[glrmask/profile][prepush_frontier_write_trie_batch] bundles={} states={} transitions={} total_ms={:.3}",
            bundles_built,
            states,
            transitions,
            elapsed_ms(started),
        );
    }

    use rayon::prelude::*;

    let mut built_bundle_cache: Vec<Option<Arc<NWA>>> = vec![None; summaries.unique_bundles.len()];
    if !compose_detail_enabled {
        if use_prepush_reconstructed_bundles {
            let profile_selection =
                std::env::var_os("GLRMASK_PROFILE_PREPUSH_SELECTION").is_some();
            let reconstructed_state_cap = std::env::var("GLRMASK_PREPUSH_RECONSTRUCT_MAX_STATES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0);
            let unconditional_template_min =
                std::env::var("GLRMASK_PREPUSH_RECONSTRUCT_UNCONDITIONAL_TEMPLATE_MIN")
                    .ok()
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(0);
            let candidates = std::sync::atomic::AtomicUsize::new(0);
            let selected = std::sync::atomic::AtomicUsize::new(0);
            built_bundle_cache = summaries
                .unique_bundles
                .par_iter()
                .enumerate()
                .map(|(bundle_id, bundle)| {
                    if !used_multi_bundle[bundle_id] {
                        return None;
                    }
                    let decision = prepush_reconstruct_decision(true, bundle, templates);
                    if decision.candidate {
                        candidates.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let reconstructed = decision.admitted.then(|| {
                            Arc::new(templates.build_prepush_reconstructed_bundle(bundle))
                        });
                        if let Some(reconstructed) = &reconstructed {
                            if decision.predicted_reconstructed_states != 0 {
                            debug_assert_eq!(
                                decision.predicted_reconstructed_states,
                                reconstructed.states().len(),
                                "pre-push census must exactly predict reconstructed state count",
                            );
                            }
                        }
                        if std::env::var_os("GLRMASK_PROFILE_PREPUSH_SELECTION_DETAIL").is_some() {
                            let reconstructed_states = reconstructed
                                .as_ref()
                                .map(|bundle| bundle.states().len())
                                .unwrap_or(0);
                            let reconstructed_transitions = reconstructed
                                .as_ref()
                                .map(|bundle| NWA::num_transitions(bundle))
                                .unwrap_or(0);
                            eprintln!(
                                "[glrmask/profile][prepush_selection_bundle] bundle_id={} terminals={} max_template={} sum_template={} predicted_reconstructed_states={} reconstructed_states={} reconstructed_transitions={} admitted={}",
                                bundle_id,
                                bundle.len(),
                                decision.max_template,
                                decision.sum_template,
                                decision.predicted_reconstructed_states,
                                reconstructed_states,
                                reconstructed_transitions,
                                decision.admitted,
                            );
                        }
                        if let Some(reconstructed) = reconstructed {
                            selected.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            Some(reconstructed)
                        } else {
                            Some(Arc::new(templates.build_bundle(bundle)))
                        }
                    } else {
                        Some(Arc::new(templates.build_bundle(bundle)))
                    }
                })
                .collect();
            if profile_selection {
                eprintln!(
                    "[glrmask/profile][prepush_selection] used_multi={} candidates={} selected={} threshold={} state_cap={} unconditional_template_min={}",
                    used_multi_bundle.iter().filter(|&&used| used).count(),
                    candidates.load(std::sync::atomic::Ordering::Relaxed),
                    selected.load(std::sync::atomic::Ordering::Relaxed),
                    std::env::var("GLRMASK_PREPUSH_RECONSTRUCT_MIN_TEMPLATE_SUM")
                        .ok()
                        .unwrap_or_else(|| "0".to_string()),
                    reconstructed_state_cap,
                    unconditional_template_min,
                );
            }
        } else if parser_bundle_topology_reuse_enabled() {
            let mut bundle_ids_by_topology =
                FxHashMap::<Vec<Vec<TerminalID>>, Vec<usize>>::default();
            for (bundle_id, bundle) in summaries.unique_bundles.iter().enumerate() {
                if used_multi_bundle[bundle_id] {
                    bundle_ids_by_topology
                        .entry(templates.bundle_topology_signature(bundle))
                        .or_default()
                        .push(bundle_id);
                }
            }
            let topology_groups = bundle_ids_by_topology.into_values().collect::<Vec<_>>();
            let built_groups = topology_groups
                .par_iter()
                .map(|bundle_ids| {
                    if bundle_ids.len() == 1 {
                        let bundle_id = bundle_ids[0];
                        return vec![(
                            bundle_id,
                            Arc::new(templates.build_bundle(&summaries.unique_bundles[bundle_id])),
                        )];
                    }

                    let representative = &summaries.unique_bundles[bundle_ids[0]];
                    let Some(skeleton) = templates.build_bundle_skeleton(representative) else {
                        return bundle_ids
                            .iter()
                            .map(|&bundle_id| {
                                (
                                    bundle_id,
                                    Arc::new(templates.build_bundle(
                                        &summaries.unique_bundles[bundle_id],
                                    )),
                                )
                            })
                            .collect();
                    };
                    bundle_ids
                        .iter()
                        .map(|&bundle_id| {
                            (
                                bundle_id,
                                Arc::new(templates.instantiate_bundle_skeleton(
                                    &summaries.unique_bundles[bundle_id],
                                    &skeleton,
                                )),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for group in built_groups {
                for (bundle_id, bundle) in group {
                    built_bundle_cache[bundle_id] = Some(bundle);
                }
            }
        } else {
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
            let fragment = if let Some(group_id) = branch.cross_target_group_id {
                if let Some(existing) = cross_target_fragment_memo.get(&group_id) {
                    if compose_detail_enabled {
                        compose_detail.memo_hits += 1;
                    }
                    existing.clone()
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
                    let Some(body) = append_cross_target_branch_fragment(
                        &mut arena,
                        &summaries,
                        &templates,
                        &mut built_bundle_cache,
                        group_id,
                        &continuation_states,
                        &productive,
                        use_prepush_reconstructed_bundles,
                        compose_detail_enabled.then_some(&mut compose_detail),
                    ) else {
                        continue;
                    };
                    compose_detail.fragment_build_ms += elapsed_ms(fragment_build_started_at);
                    cross_target_fragment_memo.insert(group_id, body.clone());
                    body
                }
            } else {
                let fragment_key = (branch.bundle_id, branch.target);
                if let Some(existing) = branch_fragment_memo.get(&fragment_key) {
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
                        use_prepush_reconstructed_bundles,
                        compose_detail_enabled.then_some(&mut compose_detail),
                    ) else {
                        continue;
                    };
                    compose_detail.fragment_build_ms += elapsed_ms(fragment_build_started_at);
                    branch_fragment_memo.insert(fragment_key, body.clone());
                    body
                }
            };

            let epsilon_link_started_at = Instant::now();
            let fragment_start_states_len = fragment.start_states.len();
            for start in fragment.start_states {
                arena.add_epsilon(from, start, branch.entry_weight.clone());
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

    if std::env::var_os("GLRMASK_PROFILE_PARSER_DWA_ATTRIBUTION").is_some() {
        let mut bundle_occurrences = vec![0usize; summaries.unique_bundles.len()];
        let mut bundle_sources = vec![std::collections::BTreeSet::<usize>::new(); summaries.unique_bundles.len()];
        let mut bundle_targets = vec![std::collections::BTreeSet::<u32>::new(); summaries.unique_bundles.len()];
        for (state_id, state) in states.iter().enumerate() {
            if !productive[state_id] {
                continue;
            }
            for branch in &state.branches {
                if !productive.get(branch.target as usize).copied().unwrap_or(false)
                    || !summaries.bundle_accepts.get(branch.bundle_id).copied().unwrap_or(false)
                {
                    continue;
                }
                bundle_occurrences[branch.bundle_id] += 1;
                bundle_sources[branch.bundle_id].insert(state_id);
                bundle_targets[branch.bundle_id].insert(branch.target);
            }
        }

        let mut ranked = Vec::new();
        for (bundle_id, bundle) in summaries.unique_bundles.iter().enumerate() {
            if bundle_occurrences[bundle_id] == 0 {
                continue;
            }
            let fragment_states = if bundle.len() == 1 {
                bundle.keys().next()
                    .and_then(|terminal| templates.by_terminal_nwa.get(terminal))
                    .map(|template| template.states().len())
                    .unwrap_or(0)
            } else {
                built_bundle_cache[bundle_id]
                    .as_ref()
                    .map(|bundle_nwa| bundle_nwa.states().len())
                    .unwrap_or(0)
            };
            let distinct_targets = bundle_targets[bundle_id].len();
            let appended_states = fragment_states.saturating_mul(distinct_targets);
            ranked.push((appended_states, bundle_id, fragment_states, distinct_targets));
        }
        ranked.sort_unstable_by(|left, right| right.cmp(left));

        eprintln!(
            "[glrmask/profile][parser_attribution_summary] terminal_dwa_states={} productive_terminal_dwa_states={} unique_bundles={} productive_bundle_ids={} unique_bundle_target_fragments={} parser_nwa_states={}",
            states.len(),
            productive.iter().filter(|&&x| x).count(),
            summaries.unique_bundles.len(),
            ranked.len(),
            ranked.iter().map(|(_, _, _, targets)| *targets).sum::<usize>(),
            arena.states().len(),
        );

        for (rank, &(appended_states, bundle_id, fragment_states, distinct_targets)) in ranked.iter().take(30).enumerate() {
            let bundle = &summaries.unique_bundles[bundle_id];
            let mut top_terminals = bundle.keys().copied().map(|terminal| {
                let template_states = templates.by_terminal_nwa.get(&terminal).map(|t| t.states().len()).unwrap_or(0);
                (template_states, terminal, grammar.terminal_display_name(terminal).to_string())
            }).collect::<Vec<_>>();
            top_terminals.sort_unstable_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
            top_terminals.truncate(16);
            let sources = bundle_sources[bundle_id].iter().copied().collect::<Vec<_>>();
            let targets = bundle_targets[bundle_id].iter().copied().collect::<Vec<_>>();
            eprintln!(
                "[glrmask/profile][parser_attribution_bundle] rank={} bundle_id={} terminals={} occurrences={} source_states={:?} targets={:?} distinct_targets={} fragment_states={} appended_states={} top_terminals={:?}",
                rank + 1, bundle_id, bundle.len(), bundle_occurrences[bundle_id], sources, targets, distinct_targets, fragment_states, appended_states, top_terminals
            );
        }

        let mut terminal_stats = Vec::new();
        for (&terminal, template) in &templates.by_terminal_nwa {
            let mut bundles = 0usize;
            let mut occurrences = 0usize;
            let mut heavy_bundle_memberships = 0usize;
            for &(_, bundle_id, _, _) in &ranked {
                if summaries.unique_bundles[bundle_id].contains_key(&terminal) {
                    bundles += 1;
                    occurrences += bundle_occurrences[bundle_id];
                }
            }
            for &(_, bundle_id, _, _) in ranked.iter().take(20) {
                heavy_bundle_memberships += usize::from(summaries.unique_bundles[bundle_id].contains_key(&terminal));
            }
            if bundles > 0 {
                terminal_stats.push((
                    heavy_bundle_memberships,
                    template.states().len(),
                    bundles,
                    occurrences,
                    terminal,
                    grammar.terminal_display_name(terminal).to_string(),
                ));
            }
        }
        terminal_stats.sort_unstable_by(|left, right| {
            right.0.cmp(&left.0)
                .then_with(|| right.1.cmp(&left.1))
                .then_with(|| right.2.cmp(&left.2))
        });
        for (rank, (heavy_memberships, template_states, bundles, occurrences, terminal, name)) in terminal_stats.iter().take(40).enumerate() {
            eprintln!(
                "[glrmask/profile][parser_attribution_terminal] rank={} terminal={} name={:?} template_states={} bundles={} branch_occurrences={} top20_bundle_memberships={}",
                rank + 1, terminal, name, template_states, bundles, occurrences, heavy_memberships
            );
        }
        for (state_id, state) in states.iter().enumerate() {
            if !productive[state_id] {
                continue;
            }
            let mut branch_rows = Vec::new();
            for branch in &state.branches {
                if !productive.get(branch.target as usize).copied().unwrap_or(false)
                    || !summaries.bundle_accepts.get(branch.bundle_id).copied().unwrap_or(false)
                {
                    continue;
                }
                let bundle = &summaries.unique_bundles[branch.bundle_id];
                let fragment_states = if bundle.len() == 1 {
                    bundle.keys().next()
                        .and_then(|terminal| templates.by_terminal_nwa.get(terminal))
                        .map(|template| template.states().len())
                        .unwrap_or(0)
                } else {
                    built_bundle_cache[branch.bundle_id]
                        .as_ref()
                        .map(|bundle_nwa| bundle_nwa.states().len())
                        .unwrap_or(0)
                };
                branch_rows.push((branch.bundle_id, branch.target, bundle.len(), fragment_states));
            }
            branch_rows.sort_unstable();
            eprintln!(
                "[glrmask/profile][parser_attribution_state] state={} final={} epsilon_branches={} branches={:?}",
                state_id,
                state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()),
                state.epsilon_branches.len(),
                branch_rows
            );
        }

        if std::env::var_os("GLRMASK_PROFILE_PARSER_DWA_ABLATION").is_some() {
            let mut global_candidates = templates.by_terminal_nwa.iter()
                .filter(|(terminal, _)| ranked.iter().any(|(_, bundle_id, _, _)| summaries.unique_bundles[*bundle_id].contains_key(terminal)))
                .map(|(&terminal, template)| (template.states().len(), terminal))
                .collect::<Vec<_>>();
            global_candidates.sort_unstable_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
            global_candidates.truncate(8);
            for (template_states, terminal) in global_candidates {
                let mut containing_bundles = 0usize;
                let mut baseline_appended_states = 0usize;
                let mut marginal_appended_states = 0isize;
                for &(appended_states, bundle_id, fragment_states, distinct_targets) in &ranked {
                    let bundle = &summaries.unique_bundles[bundle_id];
                    if !bundle.contains_key(&terminal) {
                        continue;
                    }
                    containing_bundles += 1;
                    baseline_appended_states += appended_states;
                    let mut reduced = bundle.clone();
                    reduced.remove(&terminal);
                    let reduced_states = if reduced.is_empty() {
                        0
                    } else if reduced.len() == 1 {
                        reduced.keys().next()
                            .and_then(|remaining| templates.by_terminal_nwa.get(remaining))
                            .map(|template| template.states().len())
                            .unwrap_or(0)
                    } else {
                        templates.build_bundle(&reduced).states().len()
                    };
                    marginal_appended_states += (fragment_states as isize - reduced_states as isize) * distinct_targets as isize;
                }
                eprintln!(
                    "[glrmask/profile][parser_attribution_terminal_ablation_total] terminal={} name={:?} terminal_template_states={} containing_bundles={} baseline_appended_states={} marginal_appended_states={} parser_nwa_states={}",
                    terminal, grammar.terminal_display_name(terminal), template_states, containing_bundles, baseline_appended_states, marginal_appended_states, arena.states().len()
                );
            }

            for &(appended_states, bundle_id, fragment_states, distinct_targets) in ranked.iter().take(12) {
                let bundle = &summaries.unique_bundles[bundle_id];
                let mut candidates = bundle.keys().copied().map(|terminal| {
                    let template_states = templates.by_terminal_nwa.get(&terminal).map(|t| t.states().len()).unwrap_or(0);
                    (template_states, terminal)
                }).collect::<Vec<_>>();
                candidates.sort_unstable_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
                candidates.truncate(4);
                for (template_states, terminal) in candidates {
                    let mut reduced = bundle.clone();
                    reduced.remove(&terminal);
                    let reduced_states = if reduced.is_empty() {
                        0
                    } else if reduced.len() == 1 {
                        reduced.keys().next()
                            .and_then(|remaining| templates.by_terminal_nwa.get(remaining))
                            .map(|template| template.states().len())
                            .unwrap_or(0)
                    } else {
                        templates.build_bundle(&reduced).states().len()
                    };
                    let delta = fragment_states as isize - reduced_states as isize;
                    eprintln!(
                        "[glrmask/profile][parser_attribution_ablation] bundle_id={} terminals={} distinct_targets={} baseline_fragment_states={} baseline_appended_states={} remove_terminal={} name={:?} terminal_template_states={} reduced_fragment_states={} delta_fragment_states={}",
                        bundle_id, bundle.len(), distinct_targets, fragment_states, appended_states, terminal, grammar.terminal_display_name(terminal), template_states, reduced_states, delta
                    );
                }
            }
        }
    }

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


fn profile_split_template_census(templates: &Templates) {
    if !compile_profile_enabled() {
        return;
    }
    use crate::templates::compile_dfa::{
        specialize_template_dfa_defaults_for_commit_split_input,
        try_split_commit_template_dfas,
    };

    let started_at = Instant::now();
    let mut built = 0usize;
    let mut skipped = 0usize;
    let mut pop_states = 0usize;
    let mut read_states = 0usize;
    let mut push_states = 0usize;
    let mut pop_transitions = 0usize;
    let mut read_transitions = 0usize;
    let mut push_transitions = 0usize;
    let mut phase_links = 0usize;
    let mut max_total_states = 0usize;
    let mut max_push_states = 0usize;

    for dfa in templates.by_terminal.values() {
        let commit = specialize_template_dfa_defaults_for_commit_split_input(dfa);
        let Some(split) = try_split_commit_template_dfas(&commit) else {
            skipped += 1;
            continue;
        };
        built += 1;
        let p = split.pop.states.len();
        let r = split.read.states.len();
        let w = split.push.states.len();
        pop_states += p;
        read_states += r;
        push_states += w;
        max_total_states = max_total_states.max(p + r + w);
        max_push_states = max_push_states.max(w);
        pop_transitions += split
            .pop
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        read_transitions += split
            .read
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        push_transitions += split
            .push
            .states
            .iter()
            .map(|state| state.transitions.len())
            .sum::<usize>();
        phase_links += split.pop_to_read.iter().filter(|x| x.is_some()).count();
        phase_links += split.pop_to_push.iter().filter(|x| x.is_some()).count();
        phase_links += split.read_to_push.iter().filter(|x| x.is_some()).count();
    }

    eprintln!(
        "[glrmask/profile][parser_split_template_census] templates={} built={} skipped={} pop_states={} read_states={} push_states={} total_states={} pop_transitions={} read_transitions={} push_transitions={} total_transitions={} phase_links={} max_total_states={} max_push_states={} total_ms={:.3}",
        templates.by_terminal.len(),
        built,
        skipped,
        pop_states,
        read_states,
        push_states,
        pop_states + read_states + push_states,
        pop_transitions,
        read_transitions,
        push_transitions,
        pop_transitions + read_transitions + push_transitions,
        phase_links,
        max_total_states,
        max_push_states,
        elapsed_ms(started_at),
    );
}


fn build_direct_prepush_hashcons_from_terminal_dwa(
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> Option<NWA> {
    let mut summaries = build_state_summaries(terminal_dwa, grammar, templates);
    factor_parser_bundle_entry_gates(&mut summaries, templates);
    factor_parser_bundle_cross_target_gates(&mut summaries, templates);
    let productive = compute_productive_terminal_states(&summaries);
    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_EPSILON_PROJECTED_PARSER").is_some() {
        let (raw, _, _, _, _, _, _, _, _, _) =
            build_direct_prepush_pending_compact_nwa(&summaries, &productive, templates)?;
        return eliminate_epsilon_only_states(&raw);
    }
    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_COLORED_HASHCONS_PARSER").is_some() {
        let (raw, _, _, _, _, _, _, _, _, colors) =
            build_direct_prepush_pending_compact_nwa(&summaries, &productive, templates)?;
        let quotient = hashcons_acyclic_weighted_nwa_colored(&raw, &colors)?;
        eprintln!(
            "[glrmask/profile][parser_direct_prepush_colored_hashcons] raw_states={} quotient_states={} quotient_transitions={}",
            raw.states().len(),
            quotient.states().len(),
            NWA::num_transitions(&quotient),
        );
        return Some(quotient);
    }
    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_RAW_COMPACT_PARSER").is_some() {
        return build_direct_prepush_pending_compact_nwa(&summaries, &productive, templates)
            .map(|value| value.0);
    }
    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_RAW_PARSER").is_some() {
        let raw = build_direct_prepush_pending_nwa(&summaries, &productive, templates)?.0;
        if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_POSTFINAL_HASHCONS_PARSER").is_some() {
            let mut resolved = raw;
            apply_finality_fixpoint(&mut resolved);
            remove_redundant_default_transitions(&mut resolved);
            let quotient = hashcons_acyclic_weighted_nwa(&resolved)?;
            eprintln!(
                "[glrmask/profile][parser_direct_prepush_postfinal_hashcons] resolved_states={} quotient_states={} quotient_transitions={}",
                resolved.states().len(),
                quotient.states().len(),
                NWA::num_transitions(&quotient),
            );
            return Some(quotient);
        }
        if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_RAW_HASHCONS_PARSER").is_some() {
            let quotient = hashcons_acyclic_weighted_nwa(&raw)?;
            if std::env::var_os("GLRMASK_DIAG_DIRECT_PREPUSH_HASHCONS_LANGUAGE").is_some() {
                let mut raw_read = raw.clone();
                let mut quotient_read = quotient.clone();
                apply_finality_fixpoint(&mut raw_read);
                remove_redundant_default_transitions(&mut raw_read);
                apply_finality_fixpoint(&mut quotient_read);
                remove_redundant_default_transitions(&mut quotient_read);
                let raw_det = determinize_with_supports(&raw_read, None).dwa;
                let quotient_det = determinize_with_supports(&quotient_read, None).dwa;
                let difference = find_difference(&raw_det, &quotient_det)
                    .expect("direct pre-push hashcons read-language diagnostic requires acyclic DWAs");
                eprintln!(
                    "[glrmask/profile][parser_direct_prepush_hashcons_language] raw_states={} quotient_states={} raw_det_states={} quotient_det_states={} difference={:?}",
                    raw.states().len(),
                    quotient.states().len(),
                    raw_det.states().len(),
                    quotient_det.states().len(),
                    difference,
                );
            }
            return Some(quotient);
        }
        return Some(raw);
    }
    build_direct_prepush_hashcons_nwa(&summaries, &productive, templates).map(|value| value.0)
}

fn build_direct_prepush_raw_projection_with_origins(
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> Option<(NWA, NWA, Vec<Option<u32>>)> {
    let mut summaries = build_state_summaries(terminal_dwa, grammar, templates);
    factor_parser_bundle_entry_gates(&mut summaries, templates);
    factor_parser_bundle_cross_target_gates(&mut summaries, templates);
    let productive = compute_productive_terminal_states(&summaries);
    let (raw, _, _, _, _, _, _, _, _, _) =
        build_direct_prepush_pending_compact_nwa(&summaries, &productive, templates)?;
    let (projected, raw_by_projected) = eliminate_epsilon_only_states_with_origins(&raw)?;
    Some((projected, raw, raw_by_projected))
}



fn possible_outgoing_signature(value: &PossibleOutgoingIds) -> Vec<i32> {
    match value {
        PossibleOutgoingIds::Empty => vec![-2],
        PossibleOutgoingIds::All => vec![-1],
        PossibleOutgoingIds::Some(ids) => ids.iter_ones().map(|id| id as i32).collect(),
    }
}

fn diagnose_quotient_possible_correspondence(
    raw_dwa: &DWA,
    raw_possible: &[PossibleOutgoingIds],
    quotient_dwa: &DWA,
    quotient_possible: &[PossibleOutgoingIds],
) {
    let mut queue = VecDeque::<(u32, u32)>::from([(raw_dwa.start_state(), quotient_dwa.start_state())]);
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut raw_signatures_by_quotient = FxHashMap::<u32, FxHashSet<Vec<i32>>>::default();
    while let Some((raw_state, quotient_state)) = queue.pop_front() {
        if !seen.insert((raw_state, quotient_state)) {
            continue;
        }
        if let Some(raw_signature) = raw_possible
            .get(raw_state as usize)
            .map(possible_outgoing_signature)
        {
            raw_signatures_by_quotient
                .entry(quotient_state)
                .or_default()
                .insert(raw_signature);
        }
        let Some(raw_row) = raw_dwa.states().get(raw_state as usize) else {
            continue;
        };
        let Some(quotient_row) = quotient_dwa.states().get(quotient_state as usize) else {
            continue;
        };
        for (&label, (raw_target, raw_weight)) in &raw_row.transitions {
            let Some((quotient_target, quotient_weight)) = quotient_row.transitions.get(&label) else {
                continue;
            };
            if raw_weight.intersection(quotient_weight).is_empty() {
                continue;
            }
            queue.push_back((*raw_target, *quotient_target));
        }
    }
    let ambiguous = raw_signatures_by_quotient
        .values()
        .filter(|signatures| signatures.len() > 1)
        .count();
    let max_signatures = raw_signatures_by_quotient
        .values()
        .map(FxHashSet::len)
        .max()
        .unwrap_or(0);
    let mismatched_single = raw_signatures_by_quotient
        .iter()
        .filter(|(quotient_state, signatures)| {
            if signatures.len() != 1 {
                return false;
            }
            let Some(actual) = quotient_possible.get(**quotient_state as usize) else {
                return false;
            };
            let expected = signatures.iter().next().expect("singleton signature");
            *expected != possible_outgoing_signature(actual)
        })
        .count();
    let raw_start = raw_possible
        .get(raw_dwa.start_state() as usize)
        .map(possible_outgoing_signature)
        .unwrap_or_default();
    let quotient_start = quotient_possible
        .get(quotient_dwa.start_state() as usize)
        .map(possible_outgoing_signature)
        .unwrap_or_default();
    eprintln!(
        "[glrmask/profile][parser_direct_prepush_possible_correspondence] pair_states={} quotient_states_seen={} ambiguous_quotient_states={} max_raw_signatures_per_quotient={} mismatched_single_signature_states={} raw_start_kind={} raw_start_count={} quotient_start_kind={} quotient_start_count={}",
        seen.len(),
        raw_signatures_by_quotient.len(),
        ambiguous,
        max_signatures,
        mismatched_single,
        raw_start.first().copied().unwrap_or(-3),
        raw_start.len(),
        quotient_start.first().copied().unwrap_or(-3),
        quotient_start.len(),
    );
}



fn raw_states_corresponding_to_quotient_state(
    raw_dwa: &DWA,
    quotient_dwa: &DWA,
    wanted_quotient: u32,
) -> Vec<u32> {
    let mut queue = VecDeque::<(u32, u32)>::from([(raw_dwa.start_state(), quotient_dwa.start_state())]);
    let mut seen = FxHashSet::<(u32, u32)>::default();
    let mut raw_states = FxHashSet::<u32>::default();
    while let Some((raw_state, quotient_state)) = queue.pop_front() {
        if !seen.insert((raw_state, quotient_state)) {
            continue;
        }
        if quotient_state == wanted_quotient {
            raw_states.insert(raw_state);
        }
        let Some(raw_row) = raw_dwa.states().get(raw_state as usize) else {
            continue;
        };
        let Some(quotient_row) = quotient_dwa.states().get(quotient_state as usize) else {
            continue;
        };
        for (&label, (raw_target, raw_weight)) in &raw_row.transitions {
            let Some((quotient_target, quotient_weight)) = quotient_row.transitions.get(&label) else {
                continue;
            };
            if raw_weight.intersection(quotient_weight).is_empty() {
                continue;
            }
            queue.push_back((*raw_target, *quotient_target));
        }
    }
    let mut raw_states = raw_states.into_iter().collect::<Vec<_>>();
    raw_states.sort_unstable();
    raw_states
}

fn emit_default_witness_state(tag: &str, dwa: &DWA, state_id: u32, label: i32) {
    let Some(state) = dwa.states().get(state_id as usize) else {
        return;
    };
    let explicit = state.transitions.get(&label);
    let default = state.transitions.get(&DEFAULT_LABEL);
    let explicit_target_final = explicit
        .and_then(|(target, _)| dwa.states().get(*target as usize))
        .and_then(|state| state.final_weight.as_ref());
    let default_target_final = default
        .and_then(|(target, _)| dwa.states().get(*target as usize))
        .and_then(|state| state.final_weight.as_ref());
    eprintln!(
        "[glrmask/profile][parser_default_witness] tag={} state={} transitions={} final={:?} label={} explicit_target={:?} explicit_weight={:?} explicit_target_final={:?} default_target={:?} default_weight={:?} default_target_final={:?}",
        tag,
        state_id,
        state.transitions.len(),
        state.final_weight.as_ref().map(Weight::ptr_key),
        label,
        explicit.map(|(target, _)| *target),
        explicit.map(|(_, weight)| weight.ptr_key()),
        explicit_target_final.map(Weight::ptr_key),
        default.map(|(target, _)| *target),
        default.map(|(_, weight)| weight.ptr_key()),
        default_target_final.map(Weight::ptr_key),
    );
}

fn emit_dwa_difference_trace(tag: &str, left: &DWA, right: &DWA, labels: &[i32]) {
    let mut left_state = left.start_state();
    let mut right_state = right.start_state();
    for (step, &label) in labels.iter().enumerate() {
        let left_row = &left.states()[left_state as usize];
        let right_row = &right.states()[right_state as usize];
        let left_explicit = left_row.transitions.get(&label);
        let right_explicit = right_row.transitions.get(&label);
        let left_effective = left_explicit.or_else(|| left_row.transitions.get(&DEFAULT_LABEL));
        let right_effective = right_explicit.or_else(|| right_row.transitions.get(&DEFAULT_LABEL));
        eprintln!(
            "[glrmask/profile][parser_dwa_difference_trace] tag={} step={} label={} left_state={} right_state={} left_final={} right_final={} final_equal={} left_explicit={} right_explicit={} left_default={} right_default={} left_target={:?} right_target={:?} edge_equal={} left_edge_ranges={:?} right_edge_ranges={:?}",
            tag,
            step,
            label,
            left_state,
            right_state,
            left_row.final_weight.is_some(),
            right_row.final_weight.is_some(),
            left_row.final_weight == right_row.final_weight,
            left_explicit.is_some(),
            right_explicit.is_some(),
            left_row.transitions.contains_key(&DEFAULT_LABEL),
            right_row.transitions.contains_key(&DEFAULT_LABEL),
            left_effective.map(|(target, _)| *target),
            right_effective.map(|(target, _)| *target),
            match (left_effective, right_effective) {
                (Some((_, left_weight)), Some((_, right_weight))) => left_weight == right_weight,
                (None, None) => true,
                _ => false,
            },
            left_effective.map(|(_, weight)| weight.num_ranges()),
            right_effective.map(|(_, weight)| weight.num_ranges()),
        );
        let (Some((next_left, _)), Some((next_right, _))) = (left_effective, right_effective) else {
            return;
        };
        left_state = *next_left;
        right_state = *next_right;
    }
    let left_row = &left.states()[left_state as usize];
    let right_row = &right.states()[right_state as usize];
    eprintln!(
        "[glrmask/profile][parser_dwa_difference_trace] tag={} terminal left_state={} right_state={} left_final={} right_final={} final_equal={} left_final_ranges={:?} right_final_ranges={:?}",
        tag,
        left_state,
        right_state,
        left_row.final_weight.is_some(),
        right_row.final_weight.is_some(),
        left_row.final_weight == right_row.final_weight,
        left_row.final_weight.as_ref().map(Weight::num_ranges),
        right_row.final_weight.as_ref().map(Weight::num_ranges),
    );
}

fn acyclic_dwa_root_fingerprint(dwa: &DWA) -> u64 {
    fn mix(hash: u64, value: u64) -> u64 {
        (hash ^ value)
            .wrapping_mul(0x100000001b3)
            .rotate_left(11)
            .wrapping_add(0x9e3779b97f4a7c15)
    }

    fn visit(dwa: &DWA, state_id: u32, memo: &mut [Option<u64>], visiting: &mut [bool]) -> u64 {
        if let Some(hash) = memo[state_id as usize] {
            return hash;
        }
        assert!(!visiting[state_id as usize], "DWA fingerprint requires an acyclic input");
        visiting[state_id as usize] = true;
        let state = &dwa.states()[state_id as usize];
        let mut hash = 0xcbf29ce484222325u64;
        hash = mix(
            hash,
            state
                .final_weight
                .as_ref()
                .map(Weight::structural_hash_cached)
                .unwrap_or(0),
        );
        for (&label, (target, weight)) in &state.transitions {
            hash = mix(hash, label as u32 as u64);
            hash = mix(hash, weight.structural_hash_cached());
            hash = mix(hash, visit(dwa, *target, memo, visiting));
        }
        visiting[state_id as usize] = false;
        memo[state_id as usize] = Some(hash);
        hash
    }

    let mut memo = vec![None; dwa.states().len()];
    let mut visiting = vec![false; dwa.states().len()];
    visit(dwa, dwa.start_state(), &mut memo, &mut visiting)
}

fn diagnose_direct_prepush_hashcons_tail(
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
    templates: &Templates,
    collapse_immediate_acceptance: bool,
) {
    let mut summaries = build_state_summaries(terminal_dwa, grammar, templates);
    factor_parser_bundle_entry_gates(&mut summaries, templates);
    factor_parser_bundle_cross_target_gates(&mut summaries, templates);
    let productive = compute_productive_terminal_states(&summaries);
    let Some((mut raw, _, _, _)) = build_direct_prepush_pending_nwa(&summaries, &productive, templates) else {
        eprintln!("[glrmask/profile][parser_direct_prepush_tail_diag] skipped=raw_build");
        return;
    };
    apply_finality_fixpoint(&mut raw);
    remove_redundant_default_transitions(&mut raw);
    let Some((mut quotient, quotient_origins)) = hashcons_acyclic_weighted_nwa_with_origins(&raw) else {
        eprintln!("[glrmask/profile][parser_direct_prepush_tail_diag] skipped=hashcons");
        return;
    };
    // Re-running these is idempotent; keep both paths structurally aligned with
    // the ordinary finish routine.
    apply_finality_fixpoint(&mut quotient);
    remove_redundant_default_transitions(&mut quotient);

    let raw_det = determinize_with_supports(&raw, Some(table.num_states));
    let quotient_det = determinize_with_supports(&quotient, Some(table.num_states));
    let diff_support = find_difference(&raw_det.dwa, &quotient_det.dwa)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");

    let mut raw_dwa = raw_det.dwa;
    let mut quotient_dwa = quotient_det.dwa;
    if collapse_immediate_acceptance {
        collapse_immediate_acceptance_certificates(&mut raw_dwa, terminal_dwa, grammar, table);
        collapse_immediate_acceptance_certificates(&mut quotient_dwa, terminal_dwa, grammar, table);
    }
    let diff_collapse = find_difference(&raw_dwa, &quotient_dwa)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");

    let raw_possible = build_possible_outgoing_ids_by_state(&raw, &raw_det.supports, table.num_states);
    let quotient_possible =
        build_possible_outgoing_ids_by_state(&quotient, &quotient_det.supports, table.num_states);
    let quotient_origin_possible = build_possible_outgoing_ids_by_quotient_origins(
        &raw,
        &quotient_det.supports,
        &quotient_origins,
        table.num_states,
    );
    diagnose_quotient_possible_correspondence(
        &raw_dwa,
        &raw_possible,
        &quotient_dwa,
        &quotient_possible,
    );

    let mut raw_noopt = raw_dwa.clone();
    let mut quotient_noopt = quotient_dwa.clone();
    subtract_final_weights_from_outgoing_dwa(&mut raw_noopt);
    subtract_final_weights_from_outgoing_dwa(&mut quotient_noopt);
    let diff_noopt_subtract = find_difference(&raw_noopt, &quotient_noopt)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");
    let raw_noopt_fallback =
        determinize_parser_dwa_with_fallbacks(&raw_noopt, &raw_possible, table.num_states);
    let quotient_noopt_fallback =
        determinize_parser_dwa_with_fallbacks(&quotient_noopt, &quotient_possible, table.num_states);
    let diff_noopt_fallback = find_difference(&raw_noopt_fallback, &quotient_noopt_fallback)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");

    let raw_before_defaults = raw_dwa.clone();
    let quotient_before_defaults = quotient_dwa.clone();
    emit_default_witness_state("raw_before", &raw_dwa, raw_dwa.start_state(), 371);
    emit_default_witness_state(
        "quotient_before",
        &quotient_dwa,
        quotient_dwa.start_state(),
        371,
    );
    optimize_parser_dwa_defaults(&mut raw_dwa, &raw_possible, table.num_states);
    optimize_parser_dwa_defaults(&mut quotient_dwa, &quotient_possible, table.num_states);
    emit_default_witness_state("raw_after", &raw_dwa, raw_dwa.start_state(), 371);
    emit_default_witness_state(
        "quotient_after",
        &quotient_dwa,
        quotient_dwa.start_state(),
        371,
    );
    if let Some((quotient_target, _)) = quotient_before_defaults
        .states()
        .get(quotient_before_defaults.start_state() as usize)
        .and_then(|state| state.transitions.get(&371))
    {
        let raw_states = raw_states_corresponding_to_quotient_state(
            &raw_before_defaults,
            &quotient_before_defaults,
            *quotient_target,
        );
        let mut changed_final = 0usize;
        let mut unchanged_final = 0usize;
        let mut final_pairs = FxHashSet::<(Option<usize>, Option<usize>)>::default();
        for &raw_state in &raw_states {
            let before = raw_before_defaults.states()[raw_state as usize]
                .final_weight
                .as_ref()
                .map(Weight::ptr_key);
            let after = raw_dwa.states()[raw_state as usize]
                .final_weight
                .as_ref()
                .map(Weight::ptr_key);
            changed_final += usize::from(before != after);
            unchanged_final += usize::from(before == after);
            final_pairs.insert((before, after));
        }
        eprintln!(
            "[glrmask/profile][parser_default_witness_correspondence] quotient_target={} raw_states={} changed_final={} unchanged_final={} distinct_final_pairs={}",
            quotient_target,
            raw_states.len(),
            changed_final,
            unchanged_final,
            final_pairs.len(),
        );
    }
    let diff_raw_defaults = find_difference(&raw_before_defaults, &raw_dwa)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");
    let diff_quotient_defaults = find_difference(&quotient_before_defaults, &quotient_dwa)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");
    let diff_defaults = find_difference(&raw_dwa, &quotient_dwa)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");

    let mut quotient_origin_dwa = quotient_before_defaults.clone();
    optimize_parser_dwa_defaults(
        &mut quotient_origin_dwa,
        &quotient_origin_possible,
        table.num_states,
    );
    let diff_origin_defaults = find_difference(&raw_dwa, &quotient_origin_dwa)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");
    subtract_final_weights_from_outgoing_dwa(&mut quotient_origin_dwa);
    let quotient_origin_fallback = determinize_parser_dwa_with_fallbacks(
        &quotient_origin_dwa,
        &quotient_origin_possible,
        table.num_states,
    );

    subtract_final_weights_from_outgoing_dwa(&mut raw_dwa);
    subtract_final_weights_from_outgoing_dwa(&mut quotient_dwa);
    let diff_subtract = find_difference(&raw_dwa, &quotient_dwa)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");

    let mut raw_fallback =
        determinize_parser_dwa_with_fallbacks(&raw_dwa, &raw_possible, table.num_states);
    let mut quotient_fallback =
        determinize_parser_dwa_with_fallbacks(&quotient_dwa, &quotient_possible, table.num_states);
    let diff_fallback = find_difference(&raw_fallback, &quotient_fallback)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");
    let diff_origin_fallback = find_difference(&raw_fallback, &quotient_origin_fallback)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");
    if collapse_immediate_acceptance {
        raw_fallback = collapse_final_leaf_targets(raw_fallback);
        quotient_fallback = collapse_final_leaf_targets(quotient_fallback);
    }
    let diff_leaf = find_difference(&raw_fallback, &quotient_fallback)
        .expect("direct pre-push tail diagnostic requires finite acyclic DWAs");

    eprintln!(
        "[glrmask/profile][parser_direct_prepush_tail_diag] raw_nwa_states={} quotient_nwa_states={} raw_support_dwa_states={} quotient_support_dwa_states={} diff_support={:?} diff_collapse={:?} diff_noopt_subtract={:?} diff_noopt_fallback={:?} diff_raw_defaults={:?} diff_quotient_defaults={:?} diff_defaults={:?} diff_origin_defaults={:?} diff_subtract={:?} diff_fallback={:?} diff_origin_fallback={:?} diff_leaf={:?}",
        raw.states().len(),
        quotient.states().len(),
        raw_dwa.states().len(),
        quotient_dwa.states().len(),
        diff_support,
        diff_collapse,
        diff_noopt_subtract,
        diff_noopt_fallback,
        diff_raw_defaults,
        diff_quotient_defaults,
        diff_defaults,
        diff_origin_defaults,
        diff_subtract,
        diff_fallback,
        diff_origin_fallback,
        diff_leaf,
    );
}


fn finish_full_parser_nwa_for_validation(
    mut parser_nwa: NWA,
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
    collapse_immediate_acceptance: bool,
) -> DWA {
    let num_parser_states = table.num_states;
    resolve_negative_codes_in_nwa(
        &mut parser_nwa,
        table.construction == GlrTableConstruction::ExperimentalCoreMerged,
    );
    if trim_resolved_parser_nwa_enabled(parser_nwa.states().len()) {
        parser_nwa = trim_unreachable_nwa(parser_nwa);
    }
    let determinized = determinize_with_supports(&parser_nwa, Some(num_parser_states));
    let mut parser_dwa = determinized.dwa;
    if collapse_immediate_acceptance {
        collapse_immediate_acceptance_certificates(
            &mut parser_dwa,
            terminal_dwa,
            grammar,
            table,
        );
    }
    let possible_by_state = build_possible_outgoing_ids_by_state(
        &parser_nwa,
        &determinized.supports,
        num_parser_states,
    );
    optimize_parser_dwa_defaults(&mut parser_dwa, &possible_by_state, num_parser_states);
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa);
    let mut parser_dwa =
        determinize_parser_dwa_with_fallbacks(&parser_dwa, &possible_by_state, num_parser_states);
    if collapse_immediate_acceptance {
        parser_dwa = collapse_final_leaf_targets(parser_dwa);
    }
    let skip = should_skip_parser_dwa_minimization(parser_dwa.states().len(), parser_dwa.num_transitions());
    if skip { parser_dwa } else { minimize(&parser_dwa) }
}

fn apply_direct_hashed_finality(nwa: &mut NWA) -> bool {
    let state_count = nwa.states().len();
    if state_count == 0 {
        return true;
    }

    let started = Instant::now();
    let mut predecessors = vec![Vec::<u32>::new(); state_count];
    let mut outdegree = vec![0usize; state_count];
    for (source, state) in nwa.states().iter().enumerate() {
        let mut record = |target: u32, weight: &Weight| {
            if weight.is_empty() || target as usize >= state_count {
                return;
            }
            predecessors[target as usize].push(source as u32);
            outdegree[source] += 1;
        };
        for (target, weight) in &state.epsilons {
            record(*target, weight);
        }
        for (&label, targets) in &state.transitions {
            if label != DEFAULT_LABEL && !is_negative_label(label) {
                continue;
            }
            for (target, weight) in targets {
                record(*target, weight);
            }
        }
    }

    let mut queue = VecDeque::new();
    for (state_id, &degree) in outdegree.iter().enumerate() {
        if degree == 0 {
            queue.push_back(state_id as u32);
        }
    }
    let mut reverse_topo = Vec::with_capacity(state_count);
    while let Some(state_id) = queue.pop_front() {
        reverse_topo.push(state_id);
        for &predecessor in &predecessors[state_id as usize] {
            let degree = &mut outdegree[predecessor as usize];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(predecessor);
            }
        }
    }
    if reverse_topo.len() != state_count {
        return false;
    }

    type FinalitySignature = (Option<usize>, Vec<(u32, usize)>);
    let mut class_by_signature = FxHashMap::<FinalitySignature, u32>::default();
    let mut class_by_state = vec![u32::MAX; state_count];
    let mut final_by_class = Vec::<Option<Weight>>::new();
    let mut weight_ops = ScopedWeightOpCache::default();
    let mut signature_edges = Vec::<(u32, usize)>::new();
    let mut class_hits = 0usize;

    for state_id in reverse_topo {
        let state = &nwa.states()[state_id as usize];
        signature_edges.clear();
        let mut collect_edge = |target: u32, weight: &Weight| {
            if weight.is_empty() || target as usize >= state_count {
                return;
            }
            let target_class = class_by_state[target as usize];
            debug_assert_ne!(target_class, u32::MAX);
            signature_edges.push((target_class, weight.ptr_key()));
        };
        for (target, weight) in &state.epsilons {
            collect_edge(*target, weight);
        }
        for (&label, targets) in &state.transitions {
            if label != DEFAULT_LABEL && !is_negative_label(label) {
                continue;
            }
            for (target, weight) in targets {
                collect_edge(*target, weight);
            }
        }
        signature_edges.sort_unstable();
        let signature = (
            state.final_weight.as_ref().map(Weight::ptr_key),
            signature_edges.clone(),
        );
        if let Some(&class_id) = class_by_signature.get(&signature) {
            class_by_state[state_id as usize] = class_id;
            class_hits += 1;
            continue;
        }

        let mut final_weight = state
            .final_weight
            .as_ref()
            .filter(|weight| !weight.is_empty())
            .cloned();
        let mut merge_edge = |target: u32, edge_weight: &Weight| {
            if edge_weight.is_empty() || target as usize >= state_count {
                return;
            }
            let target_class = class_by_state[target as usize] as usize;
            let Some(target_final) = final_by_class[target_class].as_ref() else {
                return;
            };
            let contribution = weight_ops.intersection(target_final, edge_weight);
            if contribution.is_empty() {
                return;
            }
            final_weight = Some(match final_weight.take() {
                Some(existing) => weight_ops.union(&existing, &contribution),
                None => contribution,
            });
        };
        for (target, weight) in &state.epsilons {
            merge_edge(*target, weight);
        }
        for (&label, targets) in &state.transitions {
            if label != DEFAULT_LABEL && !is_negative_label(label) {
                continue;
            }
            for (target, weight) in targets {
                merge_edge(*target, weight);
            }
        }

        let class_id = final_by_class.len() as u32;
        class_by_signature.insert(signature, class_id);
        class_by_state[state_id as usize] = class_id;
        final_by_class.push(final_weight);
    }

    for (state_id, &class_id) in class_by_state.iter().enumerate() {
        nwa.states_mut()[state_id].final_weight = final_by_class[class_id as usize].clone();
    }
    if std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_FINISH").is_some() {
        eprintln!(
            "[glrmask/profile][parser_direct_hashed_finality] states={} classes={} class_hits={} total_ms={:.3}",
            state_count,
            final_by_class.len(),
            class_hits,
            elapsed_ms(started),
        );
    }
    true
}

fn finish_read_only_parser_nwa_for_validation(
    mut parser_nwa: NWA,
    terminal_dwa: &TerminalAutomaton,
    grammar: &AnalyzedGrammar,
    table: &GLRTable,
    collapse_immediate_acceptance: bool,
    mut raw_identity_source: Option<(NWA, Vec<Option<u32>>)>,
) -> DWA {
    let num_parser_states = table.num_states;
    let profile = std::env::var_os("GLRMASK_PROFILE_DIRECT_PREPUSH_FINISH").is_some();
    let total_started = Instant::now();
    let drop_dead_leaves =
        std::env::var_os("GLRMASK_DIRECT_PREPUSH_DROP_DEAD_LEAVES").is_some();
    let early_drop_dead_leaves = drop_dead_leaves
        && std::env::var_os("GLRMASK_DIRECT_PREPUSH_EARLY_DROP_DEAD_LEAVES").is_some();
    let mut raw_possible_snapshot = None;
    if early_drop_dead_leaves {
        let (raw_possible, dead_count, removed) =
            drop_dead_leaf_targets_preserving_possible(&mut parser_nwa, num_parser_states);
        if profile {
            eprintln!(
                "[glrmask/profile][parser_direct_prepush_early_dead_leaf_drop] dead_leaves={} removed_edges={}",
                dead_count, removed,
            );
        }
        raw_possible_snapshot = Some(raw_possible);
    }
    let finality_started = Instant::now();
    if std::env::var_os("GLRMASK_DIRECT_PREPUSH_HASHED_FINALITY").is_some() {
        let mut reference = std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_HASHED_FINALITY")
            .is_some()
            .then(|| parser_nwa.clone());
        if !apply_direct_hashed_finality(&mut parser_nwa) {
            apply_finality_fixpoint(&mut parser_nwa);
        }
        if let Some(reference) = reference.as_mut() {
            apply_finality_fixpoint(reference);
            assert_eq!(
                parser_nwa
                    .states()
                    .iter()
                    .map(|state| state.final_weight.as_ref())
                    .collect::<Vec<_>>(),
                reference
                    .states()
                    .iter()
                    .map(|state| state.final_weight.as_ref())
                    .collect::<Vec<_>>(),
                "hashed direct finality changed per-state final weights",
            );
        }
    } else {
        apply_finality_fixpoint(&mut parser_nwa);
    }
    let finality_ms = elapsed_ms(finality_started);
    let prune_started = Instant::now();
    remove_redundant_default_transitions(&mut parser_nwa);
    if let Some((raw_nwa, _)) = raw_identity_source.as_mut() {
        apply_finality_fixpoint(raw_nwa);
        remove_redundant_default_transitions(raw_nwa);
    }
    let prune_ms = elapsed_ms(prune_started);
    if drop_dead_leaves && !early_drop_dead_leaves {
        let (raw_possible, dead_count, removed) =
            drop_dead_leaf_targets_preserving_possible(&mut parser_nwa, num_parser_states);
        if profile {
            eprintln!(
                "[glrmask/profile][parser_direct_prepush_dead_leaf_drop] dead_leaves={} removed_edges={}",
                dead_count, removed,
            );
        }
        raw_possible_snapshot = Some(raw_possible);
    }
    let support_started = Instant::now();
    let determinized = if raw_identity_source.is_some() {
        determinize_with_weighted_supports_canonical_wide(
            &parser_nwa,
            Some(num_parser_states),
            16,
        )
    } else {
        determinize_with_supports_canonical_wide(&parser_nwa, Some(num_parser_states), 16)
    };
    let support_ms = elapsed_ms(support_started);
    if profile {
        let support_entries = determinized.supports.iter().map(Vec::len).sum::<usize>();
        let support_max = determinized.supports.iter().map(Vec::len).max().unwrap_or(0);
        let support_nontrivial = determinized.supports.iter().filter(|support| support.len() > 1).count();
        let unique_support_members = determinized
            .supports
            .iter()
            .flat_map(|support| support.iter().copied())
            .collect::<FxHashSet<_>>()
            .len();
        eprintln!(
            "[glrmask/profile][parser_direct_prepush_support_shape] states={} entries={} avg={:.2} max={} nontrivial={} unique_members={}",
            determinized.supports.len(),
            support_entries,
            support_entries as f64 / determinized.supports.len().max(1) as f64,
            support_max,
            support_nontrivial,
            unique_support_members,
        );
    }
    let mut parser_dwa = determinized.dwa;
    let collapse_started = Instant::now();
    if collapse_immediate_acceptance {
        collapse_immediate_acceptance_certificates(
            &mut parser_dwa,
            terminal_dwa,
            grammar,
            table,
        );
    }
    let collapse_ms = elapsed_ms(collapse_started);
    let possible_started = Instant::now();
    let possible_by_state = if let Some(raw_possible_snapshot) = raw_possible_snapshot.as_ref() {
        build_possible_outgoing_ids_from_raw(
            raw_possible_snapshot,
            &determinized.supports,
            num_parser_states,
        )
    } else {
        build_possible_outgoing_ids_by_state(
            &parser_nwa,
            &determinized.supports,
            num_parser_states,
        )
    };
    let possible_ms = elapsed_ms(possible_started);
    let default_started = Instant::now();
    if std::env::var_os("GLRMASK_DIRECT_PREPUSH_SKIP_DEFAULT_OPT").is_none() {
        if let Some((raw_nwa, raw_by_projected)) = raw_identity_source.as_ref() {
            let weighted_supports = determinized
                .weighted_supports
                .as_ref()
                .expect("raw-identity direct finish requires weighted projected supports");
            let mut oracle = RawSupportIdentityOracle::new(
                raw_nwa,
                raw_by_projected,
                weighted_supports,
            );
            optimize_parser_dwa_defaults_with_raw_identity(
                &mut parser_dwa,
                &possible_by_state,
                num_parser_states,
                &mut oracle,
            );
        } else {
            optimize_parser_dwa_defaults(&mut parser_dwa, &possible_by_state, num_parser_states);
        }
    }
    let default_ms = elapsed_ms(default_started);
    let subtract_started = Instant::now();
    subtract_final_weights_from_outgoing_dwa(&mut parser_dwa);
    let subtract_ms = elapsed_ms(subtract_started);
    let fallback_started = Instant::now();
    let mut parser_dwa =
        determinize_parser_dwa_with_fallbacks(&parser_dwa, &possible_by_state, num_parser_states);
    let fallback_ms = elapsed_ms(fallback_started);
    let leaf_started = Instant::now();
    if collapse_immediate_acceptance {
        parser_dwa = collapse_final_leaf_targets(parser_dwa);
    }
    let leaf_ms = elapsed_ms(leaf_started);
    if profile {
        eprintln!(
            "[glrmask/profile][parser_direct_prepush_finish] nwa_states={} nwa_transitions={} support_states={} final_states={} final_transitions={} finality_ms={:.3} prune_ms={:.3} support_ms={:.3} collapse_ms={:.3} possible_ms={:.3} default_ms={:.3} subtract_ms={:.3} fallback_ms={:.3} leaf_ms={:.3} total_ms={:.3}",
            parser_nwa.states().len(),
            NWA::num_transitions(&parser_nwa),
            determinized.supports.len(),
            parser_dwa.states().len(),
            parser_dwa.num_transitions(),
            finality_ms,
            prune_ms,
            support_ms,
            collapse_ms,
            possible_ms,
            default_ms,
            subtract_ms,
            fallback_ms,
            leaf_ms,
            elapsed_ms(total_started),
        );
    }
    parser_dwa
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
    profile_template_stack_effect_normal_form(templates);
    if std::env::var_os("GLRMASK_PROFILE_SPLIT_TEMPLATE_CENSUS").is_some() {
        profile_split_template_census(templates);
    }
    let (terminal_dwa_transition_count, terminal_dwa_interned_ranges) = if profiling_enabled {
        let stats = terminal_dwa.stats();
        (stats.transitions, stats.interned_ranges)
    } else {
        (0, 0)
    };
    let use_prepush_reconstructed_bundles =
        std::env::var_os("GLRMASK_USE_PREPUSH_RECONSTRUCTED_BUNDLES").is_some();
    let Some((mut parser_nwa, parser_nwa_profile)) = build_parser_nwa_from_terminal_dwa(
        terminal_dwa,
        grammar,
        templates,
        use_prepush_reconstructed_bundles,
    ) else {
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
    profile_composed_parser_nwa_label_shape(&parser_nwa);

    let resolve_negative_started_at = Instant::now();
    let mut used_direct_read_projection = false;
    if direct_parser_read_projection_enabled()
        && let Some(mut projected) = direct_parser_read_projection(&parser_nwa)
    {
        let direct_finality_started_at = Instant::now();
        apply_finality_fixpoint(&mut projected);
        let direct_finality_ms = elapsed_ms(direct_finality_started_at);
        let direct_prune_started_at = Instant::now();
        remove_redundant_default_transitions(&mut projected);
        let direct_prune_ms = elapsed_ms(direct_prune_started_at);
        if profiling_enabled {
            eprintln!(
                "[glrmask/profile][parser_direct_read_projection_finish] states={} finality_ms={:.3} prune_defaults_ms={:.3}",
                projected.states().len(),
                direct_finality_ms,
                direct_prune_ms,
            );
        }
        parser_nwa = projected;
        used_direct_read_projection = true;
    }
    if !used_direct_read_projection {
        resolve_negative_codes_in_nwa(
            &mut parser_nwa,
            table.construction == GlrTableConstruction::ExperimentalCoreMerged,
        );
    }
    let resolve_negative_ms = elapsed_ms(resolve_negative_started_at);
    profile_parser_nwa_reachability(&parser_nwa, "post_negative_resolution");
    if std::env::var_os("GLRMASK_PROFILE_POSTNEG_HASHCONS_POTENTIAL").is_some() {
        let started = Instant::now();
        if let Some(quotient) = hashcons_acyclic_weighted_nwa(&parser_nwa) {
            eprintln!(
                "[glrmask/profile][parser_postneg_hashcons_potential] input_states={} input_transitions={} quotient_states={} quotient_transitions={} removed={} total_ms={:.3}",
                parser_nwa.states().len(),
                NWA::num_transitions(&parser_nwa),
                quotient.states().len(),
                NWA::num_transitions(&quotient),
                parser_nwa.states().len().saturating_sub(quotient.states().len()),
                elapsed_ms(started),
            );
        }
    }
    let trim_resolved_nwa_started_at = Instant::now();
    let parser_nwa_states_before_trim = parser_nwa.states().len();
    if trim_resolved_parser_nwa_enabled(parser_nwa_states_before_trim) {
        parser_nwa = trim_unreachable_nwa(parser_nwa);
    }
    let trim_resolved_nwa_ms = elapsed_ms(trim_resolved_nwa_started_at);
    if profiling_enabled && trim_resolved_parser_nwa_enabled(parser_nwa_states_before_trim) {
        eprintln!(
            "[glrmask/profile][parser_resolved_nwa_trim] before_states={} after_states={} removed={} trim_ms={:.3}",
            parser_nwa_states_before_trim,
            parser_nwa.states().len(),
            parser_nwa_states_before_trim.saturating_sub(parser_nwa.states().len()),
            trim_resolved_nwa_ms,
        );
    }
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

    if std::env::var_os("GLRMASK_DIAG_DIRECT_PREPUSH_HASHCONS_TAIL").is_some() {
        diagnose_direct_prepush_hashcons_tail(
            terminal_dwa,
            grammar,
            table,
            templates,
            collapse_immediate_acceptance,
        );
    }

    if std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_PARSER").is_some() {
        let direct_started_at = Instant::now();
        let use_projected_raw_identity =
            std::env::var_os("GLRMASK_VALIDATE_DIRECT_PREPUSH_PROJECTED_RAW_IDENTITY").is_some();
        let (direct_nwa, raw_identity_source) = if use_projected_raw_identity {
            let (projected, raw, raw_by_projected) =
                build_direct_prepush_raw_projection_with_origins(
                    terminal_dwa,
                    grammar,
                    templates,
                )
                .expect(
                    "projected raw-identity parser validation requires acyclic non-cross-target composition",
                );
            (projected, Some((raw, raw_by_projected)))
        } else {
            (
                build_direct_prepush_hashcons_from_terminal_dwa(
                    terminal_dwa,
                    grammar,
                    templates,
                )
                .expect(
                    "direct pre-push parser validation requires acyclic non-cross-target composition",
                ),
                None,
            )
        };
        let direct_nwa_states = direct_nwa.states().len();
        let direct_nwa_transitions = NWA::num_transitions(&direct_nwa);
        let direct_build_ms = elapsed_ms(direct_started_at);
        if std::env::var_os("GLRMASK_DIAG_DIRECT_PREPUSH_VS_RESOLVED_SUPPORT").is_some() {
            let diag_started_at = Instant::now();
            let mut direct_read = direct_nwa.clone();
            apply_finality_fixpoint(&mut direct_read);
            remove_redundant_default_transitions(&mut direct_read);
            let direct_support =
                determinize_with_supports(&direct_read, Some(num_parser_states)).dwa;
            let reference_support =
                determinize_with_supports(&parser_nwa, Some(num_parser_states)).dwa;
            let support_difference = find_difference(&direct_support, &reference_support)
                .expect("direct pre-push support diagnostic requires finite acyclic DWAs");
            if let Some(labels) = support_difference.as_deref() {
                emit_dwa_difference_trace(
                    "direct_vs_resolved_support",
                    &direct_support,
                    &reference_support,
                    labels,
                );
            }
            eprintln!(
                "[glrmask/profile][parser_direct_prepush_vs_resolved_support] direct_states={} direct_transitions={} reference_states={} reference_transitions={} difference={:?} total_ms={:.3}",
                direct_support.states().len(),
                direct_support.num_transitions(),
                reference_support.states().len(),
                reference_support.num_transitions(),
                support_difference,
                elapsed_ms(diag_started_at),
            );
        }
        let finish_started_at = Instant::now();
        let direct = finish_read_only_parser_nwa_for_validation(
            direct_nwa,
            terminal_dwa,
            grammar,
            table,
            collapse_immediate_acceptance,
            raw_identity_source,
        );
        let finish_ms = elapsed_ms(finish_started_at);
        let compare_started_at = Instant::now();
        let difference = find_difference(&direct, &minimized)
            .expect("direct pre-push parser validation requires finite acyclic parser DWAs");
        let compare_ms = elapsed_ms(compare_started_at);
        if std::env::var_os("GLRMASK_DIAG_DIRECT_PREPUSH_FINGERPRINT").is_some() {
            eprintln!(
                "[glrmask/profile][parser_direct_prepush_fingerprint] direct={:016x} reference={:016x} difference={:?}",
                acyclic_dwa_root_fingerprint(&direct),
                acyclic_dwa_root_fingerprint(&minimized),
                difference,
            );
        }
        assert!(
            difference.is_none(),
            "direct pre-push parser changed weighted parser language on labels {:?}",
            difference,
        );
        eprintln!(
            "[glrmask/profile][parser_direct_prepush_validation] result=equivalent direct_nwa_states={} direct_nwa_transitions={} direct_dwa_states={} direct_dwa_transitions={} reference_states={} reference_transitions={} build_ms={:.3} finish_ms={:.3} compare_ms={:.3} total_ms={:.3}",
            direct_nwa_states,
            direct_nwa_transitions,
            direct.states().len(),
            direct.num_transitions(),
            minimized.states().len(),
            minimized.num_transitions(),
            direct_build_ms,
            finish_ms,
            compare_ms,
            elapsed_ms(direct_started_at),
        );
    }


    if std::env::var_os("GLRMASK_VALIDATE_POSTNEG_HASHCONS_PARSER").is_some() {
        let started = Instant::now();
        let quotient = hashcons_acyclic_weighted_nwa(&parser_nwa)
            .expect("post-negative hashcons validation requires an acyclic parser NWA");
        let quotient_states = quotient.states().len();
        let quotient_transitions = NWA::num_transitions(&quotient);
        let hashcons_ms = elapsed_ms(started);
        let finish_started = Instant::now();
        let alt = finish_read_only_parser_nwa_for_validation(
            quotient,
            terminal_dwa,
            grammar,
            table,
            collapse_immediate_acceptance,
            None,
        );
        let finish_ms = elapsed_ms(finish_started);
        let compare_started = Instant::now();
        let difference = find_difference(&alt, &minimized)
            .expect("post-negative hashcons validation requires finite acyclic DWAs");
        let compare_ms = elapsed_ms(compare_started);
        assert!(
            difference.is_none(),
            "post-negative structural hashcons changed weighted parser language on labels {:?}",
            difference,
        );
        eprintln!(
            "[glrmask/profile][parser_postneg_hashcons_validation] result=equivalent input_states={} quotient_states={} quotient_transitions={} alt_states={} alt_transitions={} reference_states={} reference_transitions={} hashcons_ms={:.3} finish_ms={:.3} compare_ms={:.3} total_ms={:.3}",
            parser_nwa.states().len(),
            quotient_states,
            quotient_transitions,
            alt.states().len(),
            alt.num_transitions(),
            minimized.states().len(),
            minimized.num_transitions(),
            hashcons_ms,
            finish_ms,
            compare_ms,
            elapsed_ms(started),
        );
    }

    if std::env::var_os("GLRMASK_VALIDATE_PREPUSH_RECONSTRUCTED_PARSER").is_some() {
        let started = Instant::now();
        let (alt_nwa, alt_profile) = build_parser_nwa_from_terminal_dwa(
            terminal_dwa,
            grammar,
            templates,
            true,
        )
        .expect("reconstructed pre-push parser validation requires a productive parser NWA");
        let build_ms = elapsed_ms(started);
        let alt_nwa_states = alt_nwa.states().len();
        let alt_nwa_transitions = NWA::num_transitions(&alt_nwa);
        let finish_started = Instant::now();
        let alt = finish_full_parser_nwa_for_validation(
            alt_nwa,
            terminal_dwa,
            grammar,
            table,
            collapse_immediate_acceptance,
        );
        let finish_ms = elapsed_ms(finish_started);
        let compare_started = Instant::now();
        let difference = find_difference(&alt, &minimized)
            .expect("reconstructed pre-push parser validation requires finite acyclic DWAs");
        let compare_ms = elapsed_ms(compare_started);
        assert!(
            difference.is_none(),
            "reconstructed pre-push bundles changed weighted parser language on labels {:?}",
            difference,
        );
        eprintln!(
            "[glrmask/profile][prepush_reconstructed_parser_validation] result=equivalent alt_nwa_states={} alt_nwa_transitions={} alt_dwa_states={} alt_dwa_transitions={} reference_states={} reference_transitions={} build_ms={:.3} compose_ms={:.3} finish_ms={:.3} compare_ms={:.3} total_ms={:.3}",
            alt_nwa_states,
            alt_nwa_transitions,
            alt.states().len(),
            alt.num_transitions(),
            minimized.states().len(),
            minimized.num_transitions(),
            build_ms,
            alt_profile.compose_state_ms,
            finish_ms,
            compare_ms,
            elapsed_ms(started),
        );
    }

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
        PossibleOutgoingIds, ScopedWeightOpCache, build_parser_nwa_from_terminal_dwa,
        collapse_final_leaf_targets, determinize_parser_dwa_with_fallbacks,
        determinize_with_supports, immediate_acceptance_certificates,
        local_epsilon_closure, local_epsilon_closure_canonical,
        try_build_direct_regular_parser_top_accept_parts,
        try_build_direct_regular_parser_top_accept_parts_table_product_reference,
        try_build_immediate_parser_top_accept_parts,
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
            let mut weight_ops = ScopedWeightOpCache::default();
            local_epsilon_closure_canonical(
                &nwa,
                &mut flat_weights,
                &mut flat_queue,
                &seed_canonical,
                &mut touched,
                &mut flat_canonical,
                &mut weight_ops,
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
            false,
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
