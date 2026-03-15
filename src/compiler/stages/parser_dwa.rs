#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::automata::unweighted_u32::dfa::DFA as UnweightedDfa;
use crate::automata::unweighted_u32::nfa::NFA as UnweightedNfa;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::labels::DEFAULT_LABEL;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::resolve_negatives::resolve_negative_codes_in_nwa;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::compiler::stages::profile_stats::{
    UnweightedDfaStats,
    WeightedDwaStats,
    WeightedNwaStats,
    collect_unweighted_dfa_stats,
    collect_weighted_dwa_stats,
    collect_weighted_nwa_stats,
};
use crate::compiler::stages::templates::Templates;
use crate::compiler::stages::templates::characterize::{characterize_terminals, TerminalCharacterization};
use crate::compiler::terminal_dwa::{TerminalDwaBuildReport, build_terminal_dwa, build_terminal_dwa_with_report};
use crate::ds::weight::Weight;

type Bundle = BTreeMap<TerminalID, Weight>;

#[derive(Debug, Clone)]
struct Branch {
    target: u32,
    bundle_id: usize,
    bundle: Arc<NWA>,
}

#[derive(Debug, Clone)]
struct StateSummary {
    final_weight: Option<Weight>,
    branches: Vec<Branch>,
}

#[derive(Debug, Clone, Default)]
struct ViableSuffixRecognizer {
    subset_to_state: HashMap<Vec<u32>, u32>,
    possible_outgoing_ids: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
struct DeterminizedDwaWithSupports {
    dwa: DWA,
    supports: Vec<Vec<u32>>,
}

#[derive(Default)]
struct ComposeStateProfile {
    calls: usize,
    memo_hits: usize,
    branches: usize,
    accepting_ms: std::time::Duration,
    continuation_ms: std::time::Duration,
    concat_build_ms: std::time::Duration,
    union_ms: std::time::Duration,
    concat_hits: usize,
    concat_misses: usize,
    concat_left_states: usize,
    concat_right_states: usize,
    max_concat_left_states: usize,
    max_concat_right_states: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CharacterizationStats {
    pub terminals: usize,
    pub shifts: usize,
    pub reduces: usize,
    pub nt_escapes: usize,
    pub nt_rereduces: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TemplateStats {
    pub templates: usize,
    pub total_states: usize,
    pub total_transitions: usize,
    pub max_states: usize,
    pub max_transitions: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct BundleStats {
    pub total_bundles: usize,
    pub unique_bundles: usize,
    pub unique_bundle_targets: usize,
    pub bundle_cache_hits: usize,
    pub group_targets_time: std::time::Duration,
    pub build_bundle_time: std::time::Duration,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ParserDwaBuildReport {
    pub characterize_terminals_time: std::time::Duration,
    pub build_templates_time: std::time::Duration,
    pub build_state_summaries_time: std::time::Duration,
    pub compose_state_time: std::time::Duration,
    pub resolve_negative_codes_time: std::time::Duration,
    pub subtract_final_weights_time: std::time::Duration,
    pub determinize_minimize_time: std::time::Duration,
    pub total_time: std::time::Duration,
    pub characterizations: CharacterizationStats,
    pub templates: TemplateStats,
    pub bundles: BundleStats,
    pub terminal_build: Option<TerminalDwaBuildReport>,
    pub parser_nwa_before_resolve: WeightedNwaStats,
    pub parser_nwa_after_resolve: WeightedNwaStats,
    pub parser_nwa_after_subtract: WeightedNwaStats,
    pub parser_dwa_pre_minimize: WeightedDwaStats,
    pub parser_dwa_minimized: WeightedDwaStats,
    pub negative_edges_before_resolve: usize,
    pub negative_edges_after_resolve: usize,
    pub default_edges_before_resolve: usize,
    pub default_edges_after_resolve: usize,
}

fn collect_characterization_stats(
    characterizations: &std::collections::BTreeMap<TerminalID, crate::compiler::stages::templates::characterize::TerminalCharacterization>,
) -> CharacterizationStats {
    let mut stats = CharacterizationStats {
        terminals: characterizations.len(),
        ..CharacterizationStats::default()
    };
    for characterization in characterizations.values() {
        stats.shifts += characterization.shifts.len();
        stats.reduces += characterization.reduces.len();
        stats.nt_escapes += characterization.nt_escapes.len();
        stats.nt_rereduces += characterization.nt_rereduces.len();
    }
    stats
}

fn collect_template_stats(templates: &Templates) -> TemplateStats {
    let mut stats = TemplateStats {
        templates: templates.by_terminal.len(),
        ..TemplateStats::default()
    };
    for template in templates.by_terminal.values() {
        let shape = collect_unweighted_dfa_stats(template);
        stats.total_states += shape.states;
        stats.total_transitions += shape.transitions;
        stats.max_states = stats.max_states.max(shape.states);
        stats.max_transitions = stats.max_transitions.max(shape.transitions);
    }
    stats
}

fn count_special_edges(nwa: &NWA) -> (usize, usize) {
    let mut negative_edges = 0usize;
    let mut default_edges = 0usize;
    for state in &nwa.states {
        for (&label, targets) in &state.transitions {
            if label == crate::compiler::glr::labels::DEFAULT_LABEL {
                default_edges += targets.len();
            } else if crate::compiler::glr::labels::is_negative_label(label) {
                negative_edges += targets.len();
            }
        }
    }
    (negative_edges, default_edges)
}

fn profile_dump_small_automaton(stage: &str, automaton: &impl std::fmt::Display, num_states: usize) {
    if std::env::var_os("GLRMASK_PROFILE_PARSER_DWA_DUMP_SMALL").is_none() || num_states > 8 {
        return;
    }
    eprintln!("[glrmask/profile][parser_dwa][dump] {stage}\n{automaton}");
}

fn append_dwa(into: &mut DWA, other: &DWA) -> u32 {
    let offset = into.states.len() as u32;
    for _ in &other.states {
        into.add_state();
    }

    for (state_id, state) in other.states.iter().enumerate() {
        let dst_state = offset + state_id as u32;
        if let Some(final_weight) = state.final_weight.clone() {
            into.set_final_weight(dst_state, final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            into.add_transition(dst_state, label, offset + *target, weight.clone());
        }
    }

    offset + other.start_state
}

fn accepting_nwa(final_weight: &Weight) -> Option<NWA> {
    if final_weight.is_empty() {
        return None;
    }

    let mut nwa = NWA::new(0, 0);
    let start = nwa.add_state();
    nwa.start_states.push(start);
    nwa.set_final_weight(start, final_weight.clone());
    Some(nwa)
}

fn group_terminal_edges_by_target(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    state_id: u32,
) -> BTreeMap<u32, Bundle> {
    let Some(state) = terminal_dwa.states.get(state_id as usize) else {
        return BTreeMap::new();
    };

    let mut groups = BTreeMap::<u32, Bundle>::new();
    for (&label, (target, weight)) in &state.transitions {
        if label < 0 || label as u32 >= grammar.num_terminals {
            continue;
        }

        groups
            .entry(*target)
            .or_default()
            .entry(label as TerminalID)
            .and_modify(|existing| *existing = existing.union(weight))
            .or_insert_with(|| weight.clone());
    }

    groups
}

fn build_state_summaries(
    terminal_dwa: &DWA,
    grammar: &AnalyzedGrammar,
    templates: &Templates,
) -> (Vec<StateSummary>, BundleStats) {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some();
    let group_targets_started = std::time::Instant::now();

    // Phase 1: Collect all (state_id, target, bundle_key) triples.
    let mut state_groups: Vec<Vec<(u32, Vec<(TerminalID, Weight)>, Bundle)>> = Vec::with_capacity(terminal_dwa.states.len());
    let mut unique_keys: HashMap<Vec<(TerminalID, Weight)>, usize> = HashMap::new();
    let mut unique_bundles_ordered: Vec<(usize, Vec<(TerminalID, Weight)>, Bundle)> = Vec::new();
    let mut bundle_count = 0usize;
    let mut bundle_cache_hits = 0usize;

    for (state_id, _state) in terminal_dwa.states.iter().enumerate() {
        let groups = group_terminal_edges_by_target(terminal_dwa, grammar, state_id as u32);
        let mut state_entries = Vec::with_capacity(groups.len());
        for (target, bundle) in groups {
            bundle_count += 1;
            let bundle_key: Vec<(TerminalID, Weight)> = bundle
                .iter()
                .map(|(&terminal, weight)| (terminal, weight.clone()))
                .collect();
            if let Some(&existing_id) = unique_keys.get(&bundle_key) {
                bundle_cache_hits += 1;
                state_entries.push((target, bundle_key, bundle));
                let _ = existing_id; // referenced via unique_keys later
            } else {
                let id = unique_bundles_ordered.len();
                unique_keys.insert(bundle_key.clone(), id);
                unique_bundles_ordered.push((id, bundle_key.clone(), bundle.clone()));
                state_entries.push((target, bundle_key, bundle));
            }
        }
        state_groups.push(state_entries);
    }
    let group_targets_time = group_targets_started.elapsed();

    // Phase 2: Build all unique bundles in parallel.
    let build_bundle_started = std::time::Instant::now();
    #[cfg(feature = "rayon")]
    let built_bundles: Vec<Arc<NWA>> = {
        use rayon::prelude::*;
        unique_bundles_ordered
            .par_iter()
            .map(|(_id, _key, bundle)| Arc::new(templates.build_bundle(bundle)))
            .collect()
    };
    #[cfg(not(feature = "rayon"))]
    let built_bundles: Vec<Arc<NWA>> = unique_bundles_ordered
        .iter()
        .map(|(_id, _key, bundle)| Arc::new(templates.build_bundle(bundle)))
        .collect();
    let build_bundle_time = build_bundle_started.elapsed();

    // Phase 3: Assemble StateSummary structs using prebuilt bundles.
    let mut unique_bundle_targets = HashSet::new();
    let summaries: Vec<StateSummary> = terminal_dwa
        .states
        .iter()
        .enumerate()
        .map(|(state_id, state)| {
            let branches = state_groups[state_id]
                .iter()
                .map(|(target, bundle_key, _bundle)| {
                    let bundle_id = unique_keys[bundle_key];
                    let built_bundle = Arc::clone(&built_bundles[bundle_id]);
                    unique_bundle_targets.insert((*target, bundle_id));
                    Branch {
                        target: *target,
                        bundle_id,
                        bundle: built_bundle,
                    }
                })
                .collect();

            StateSummary {
                final_weight: state.final_weight.clone(),
                branches,
            }
        })
        .collect();

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] build_state_summaries_detail group_targets_ms={:.3} build_bundle_ms={:.3} bundles={} unique_bundles={} unique_bundle_targets={} bundle_cache_hits={}",
            group_targets_time.as_secs_f64() * 1000.0,
            build_bundle_time.as_secs_f64() * 1000.0,
            bundle_count,
            unique_bundles_ordered.len(),
            unique_bundle_targets.len(),
            bundle_cache_hits,
        );
    }

    (
        summaries,
        BundleStats {
            total_bundles: bundle_count,
            unique_bundles: unique_bundles_ordered.len(),
            unique_bundle_targets: unique_bundle_targets.len(),
            bundle_cache_hits,
            group_targets_time,
            build_bundle_time,
        },
    )
}

fn concatenate_nwas(left: &NWA, right: &NWA) -> Option<NWA> {
    if left.start_states.is_empty() || right.start_states.is_empty() {
        return None;
    }

    let mut arena = NWA::new(0, 0);
    let right_body = arena.append_with_body(right);
    let left_body = arena.concatenate_in_place(left, &right_body);
    arena.start_states = left_body.start_states.clone();
    Some(arena)
}

fn union_optional_nwa(acc: &mut Option<NWA>, next: &NWA) {
    match acc {
        Some(existing) => {
            let body = existing.append_with_body(next);
            existing.start_states.extend(body.start_states);
        }
        None => *acc = Some(next.clone()),
    }
}

fn compose_state(
    state_id: u32,
    states: &[StateSummary],
    arena: &mut NWA,
    memo: &mut Vec<Option<Option<crate::automata::weighted::nwa::NwaBody>>>,
    concat_memo: &mut HashMap<(usize, u32), Option<crate::automata::weighted::nwa::NwaBody>>,
    mut profile: Option<&mut ComposeStateProfile>,
) -> Option<crate::automata::weighted::nwa::NwaBody> {
    if let Some(profile) = profile.as_deref_mut() {
        profile.calls += 1;
    }
    if let Some(Some(cached)) = memo.get(state_id as usize) {
        if let Some(profile) = profile.as_deref_mut() {
            profile.memo_hits += 1;
        }
        return cached.clone();
    }

    let Some(state) = states.get(state_id as usize) else {
        return None;
    };

    let phase_started_at = std::time::Instant::now();
    let mut composed = state
        .final_weight
        .as_ref()
        .and_then(accepting_nwa)
        .map(|accepting| arena.append_with_body(&accepting));
    if let Some(profile) = profile.as_deref_mut() {
        profile.accepting_ms += phase_started_at.elapsed();
    }

    for branch in &state.branches {
        if let Some(profile) = profile.as_deref_mut() {
            profile.branches += 1;
        }

        let phase_started_at = std::time::Instant::now();
        let Some(continuation) = compose_state(
            branch.target,
            states,
            arena,
            memo,
            concat_memo,
            profile.as_deref_mut(),
        ) else {
            continue;
        };
        if let Some(profile) = profile.as_deref_mut() {
            profile.continuation_ms += phase_started_at.elapsed();
        }

        let concat_key = (branch.bundle_id, branch.target);
        let branch_with_continuation = if let Some(cached) = concat_memo.get(&concat_key) {
            if let Some(profile) = profile.as_deref_mut() {
                profile.concat_hits += 1;
            }
            cached.clone()
        } else {
            if let Some(profile) = profile.as_deref_mut() {
                profile.concat_misses += 1;
                profile.concat_left_states += branch.bundle.states.len();
                profile.concat_right_states += continuation.start_states.len();
                profile.max_concat_left_states = profile.max_concat_left_states.max(branch.bundle.states.len());
                profile.max_concat_right_states =
                    profile.max_concat_right_states.max(continuation.start_states.len());
            }
            let phase_started_at = std::time::Instant::now();
            let built = Some(arena.concatenate_in_place(branch.bundle.as_ref(), &continuation));
            if let Some(profile) = profile.as_deref_mut() {
                profile.concat_build_ms += phase_started_at.elapsed();
            }
            concat_memo.insert(concat_key, built.clone());
            built
        };
        let Some(branch_with_continuation) = branch_with_continuation else {
            continue;
        };

        let phase_started_at = std::time::Instant::now();
        composed = Some(match composed {
            Some(existing) => crate::automata::weighted::nwa::NwaBody::union(&existing, &branch_with_continuation),
            None => branch_with_continuation,
        });
        if let Some(profile) = profile.as_deref_mut() {
            profile.union_ms += phase_started_at.elapsed();
        }
    }

    if let Some(entry) = memo.get_mut(state_id as usize) {
        *entry = Some(composed.clone());
    }
    composed
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

fn add_or_union_transition(
    state: &mut crate::automata::weighted::nwa::NWAState,
    label: i32,
    target: u32,
    add: Weight,
) -> bool {
    if add.is_empty() {
        return false;
    }

    let targets = state.transitions.entry(label).or_default();
    for (existing_target, existing_weight) in targets.iter_mut() {
        if *existing_target == target {
            let updated = existing_weight.union(&add);
            if updated != *existing_weight {
                *existing_weight = updated;
                return true;
            }
            return false;
        }
    }

    targets.push((target, add));
    true
}

fn parser_state_label(label: i32, num_parser_states: u32) -> Option<u32> {
    if label >= 0 && (label as u32) < num_parser_states {
        Some(label as u32)
    } else {
        None
    }
}

fn nwa_to_unweighted(nwa: &NWA) -> UnweightedNfa {
    let mut out = UnweightedNfa::new_empty();
    out.states = vec![crate::automata::unweighted_u32::nfa::NFAState::default(); nwa.states.len()];
    out.start_states = nwa.start_states.clone();

    for (state_id, state) in nwa.states.iter().enumerate() {
        if state.final_weight.as_ref().is_some_and(|weight| !weight.is_empty()) {
            out.states[state_id].is_accepting = true;
        }

        for (&label, targets) in &state.transitions {
            for (target, weight) in targets {
                if !weight.is_empty() {
                    out.states[state_id].transitions.entry(label).or_default().push(*target);
                }
            }
        }

        for (target, weight) in &state.epsilons {
            if !weight.is_empty() {
                out.states[state_id].epsilons.push(*target);
            }
        }
    }

    out
}

fn determinize_unweighted_with_subset_map(nfa: &UnweightedNfa) -> (UnweightedDfa, HashMap<Vec<u32>, u32>) {
    fn epsilon_closure(nfa: &UnweightedNfa, seeds: &[u32]) -> BTreeSet<u32> {
        let mut closed = BTreeSet::new();
        let mut queue: VecDeque<u32> = seeds.iter().copied().collect();
        while let Some(state_id) = queue.pop_front() {
            if closed.insert(state_id) {
                for &target in &nfa.states[state_id as usize].epsilons {
                    if !closed.contains(&target) {
                        queue.push_back(target);
                    }
                }
            }
        }
        closed
    }

    if nfa.states.is_empty() || nfa.start_states.is_empty() {
        return (UnweightedDfa::new(), HashMap::new());
    }

    let mut dfa = UnweightedDfa::new();
    let mut subset_map: HashMap<Vec<u32>, u32> = HashMap::new();
    let mut worklist: VecDeque<Vec<u32>> = VecDeque::new();

    let start_closure = epsilon_closure(nfa, &nfa.start_states);
    let start_key: Vec<u32> = start_closure.iter().copied().collect();
    subset_map.insert(start_key.clone(), dfa.start_state);
    worklist.push_back(start_key);

    while let Some(subset_key) = worklist.pop_front() {
        let dfa_state = subset_map[&subset_key];
        if subset_key.iter().any(|&state_id| nfa.states[state_id as usize].is_accepting) {
            dfa.set_accepting(dfa_state, true);
        }

        let mut label_targets: BTreeMap<i32, BTreeSet<u32>> = BTreeMap::new();
        for &state_id in &subset_key {
            for (&label, targets) in &nfa.states[state_id as usize].transitions {
                let entry = label_targets.entry(label).or_default();
                for &target in targets {
                    entry.insert(target);
                }
            }
        }

        for (label, raw_targets) in label_targets {
            let closure = epsilon_closure(nfa, &raw_targets.iter().copied().collect::<Vec<_>>());
            let next_key: Vec<u32> = closure.iter().copied().collect();
            if next_key.is_empty() {
                continue;
            }

            let next_state = if let Some(&existing) = subset_map.get(&next_key) {
                existing
            } else {
                let new_state = dfa.add_state();
                subset_map.insert(next_key.clone(), new_state);
                worklist.push_back(next_key);
                new_state
            };
            dfa.add_transition(dfa_state, label, next_state);
        }
    }

    (dfa, subset_map)
}

fn build_viable_suffix_recognizer(nwa: &NWA, num_parser_states: u32) -> ViableSuffixRecognizer {
    let unweighted = nwa_to_unweighted(nwa);
    let (dfa, subset_to_state) = determinize_unweighted_with_subset_map(&unweighted);
    let possible_outgoing_ids = dfa
        .states
        .iter()
        .map(|state| {
            let mut ids = BTreeSet::new();
            for &label in state.transitions.keys() {
                if label == DEFAULT_LABEL {
                    ids.extend(0..num_parser_states);
                } else if let Some(parser_state_id) = parser_state_label(label, num_parser_states) {
                    ids.insert(parser_state_id);
                }
            }
            ids.into_iter().collect()
        })
        .collect();

    ViableSuffixRecognizer {
        subset_to_state,
        possible_outgoing_ids,
    }
}

fn determinize_with_supports(nwa: &NWA) -> DeterminizedDwaWithSupports {
    fn canonicalize(subset: &HashMap<u32, Weight>) -> Vec<(u32, Weight)> {
        let mut entries: Vec<_> = subset
            .iter()
            .filter_map(|(&state_id, weight)| (!weight.is_empty()).then_some((state_id, weight.clone())))
            .collect();
        entries.sort_by_key(|(state_id, _)| *state_id);
        entries
    }

    fn epsilon_closure(nwa: &NWA, seed: HashMap<u32, Weight>) -> HashMap<u32, Weight> {
        let mut closure = seed;
        let mut queue: VecDeque<u32> = closure.keys().copied().collect();

        while let Some(state_id) = queue.pop_front() {
            let Some(current_weight) = closure.get(&state_id).cloned() else {
                continue;
            };
            let Some(state) = nwa.states.get(state_id as usize) else {
                continue;
            };
            for (target, edge_weight) in &state.epsilons {
                let contribution = current_weight.intersection(edge_weight);
                if contribution.is_empty() {
                    continue;
                }
                let existing = closure.get(target).cloned().unwrap_or_else(Weight::empty);
                if !contribution.is_subset(&existing) {
                    closure.insert(*target, existing.union(&contribution));
                    queue.push_back(*target);
                }
            }
        }

        closure
    }

    let mut dwa = DWA::new(0, 0);
    let mut supports = vec![Vec::new()];

    let mut start_subset = HashMap::new();
    for &state_id in &nwa.start_states {
        let existing = start_subset.get(&state_id).cloned().unwrap_or_else(Weight::empty);
        start_subset.insert(state_id, existing.union(&Weight::all()));
    }
    let start_subset = epsilon_closure(nwa, start_subset);
    if start_subset.is_empty() {
        return DeterminizedDwaWithSupports { dwa, supports };
    }

    let start_entries = canonicalize(&start_subset);
    supports[0] = start_entries.iter().map(|(state_id, _)| *state_id).collect();

    let mut subset_map: HashMap<Vec<(u32, usize)>, u32> = HashMap::new();
    let mut worklist: VecDeque<(Vec<(u32, usize)>, Vec<(u32, Weight)>)> = VecDeque::new();
    let start_key: Vec<(u32, usize)> = start_entries
        .iter()
        .map(|(state_id, weight)| (*state_id, weight.ptr_key()))
        .collect();
    subset_map.insert(start_key.clone(), dwa.start_state);
    worklist.push_back((start_key, start_entries));

    while let Some((subset_key_ids, subset_entries)) = worklist.pop_front() {
        let from_state = subset_map[&subset_key_ids];

        let mut final_weight = Weight::empty();
        for (nwa_state_id, path_weight) in &subset_entries {
            if let Some(state_final) = nwa.states[*nwa_state_id as usize].final_weight.as_ref() {
                final_weight = final_weight.union(&path_weight.intersection(state_final));
            }
        }
        if !final_weight.is_empty() {
            dwa.set_final_weight(from_state, final_weight);
        }

        let mut raw_targets: HashMap<i32, HashMap<u32, Weight>> = HashMap::new();
        for (nwa_state_id, path_weight) in &subset_entries {
            let state = &nwa.states[*nwa_state_id as usize];
            for (&label, targets) in &state.transitions {
                for (target, transition_weight) in targets {
                    let next_weight = path_weight.intersection(transition_weight);
                    if next_weight.is_empty() {
                        continue;
                    }

                    let target_entry = raw_targets.entry(label).or_default();
                    target_entry
                        .entry(*target)
                        .and_modify(|existing| *existing = existing.union(&next_weight))
                        .or_insert(next_weight);
                }
            }
        }

        for (label, target_subset) in raw_targets {
            if target_subset.is_empty() {
                continue;
            }

            let edge_weight = Weight::union_all(target_subset.values());
            if edge_weight.is_empty() {
                continue;
            }

            let expanded = epsilon_closure(nwa, target_subset);
            if expanded.is_empty() {
                continue;
            }

            let edge_complement = edge_weight.complement();
            let normalized: HashMap<u32, Weight> = if edge_complement.is_empty() {
                expanded
            } else {
                expanded
                    .into_iter()
                    .filter_map(|(state_id, weight)| {
                        let normalized_weight = weight.union(&edge_complement);
                        (!normalized_weight.is_empty()).then_some((state_id, normalized_weight))
                    })
                    .collect()
            };

            let next_entries = canonicalize(&normalized);
            if next_entries.is_empty() {
                continue;
            }

            let next_key_ids: Vec<(u32, usize)> = next_entries
                .iter()
                .map(|(state_id, weight)| (*state_id, weight.ptr_key()))
                .collect();
            let next_support: Vec<u32> = next_entries.iter().map(|(state_id, _)| *state_id).collect();

            let to_state = if let Some(existing) = subset_map.get(&next_key_ids).copied() {
                existing
            } else {
                let new_state = dwa.add_state();
                subset_map.insert(next_key_ids.clone(), new_state);
                worklist.push_back((next_key_ids, next_entries));
                supports.push(next_support);
                new_state
            };

            dwa.add_transition(from_state, label, to_state, edge_weight);
        }
    }

    DeterminizedDwaWithSupports { dwa, supports }
}

fn dwa_to_nwa(dwa: &DWA) -> NWA {
    let mut nwa = NWA::new(0, 0);
    nwa.states = vec![crate::automata::weighted::nwa::NWAState::default(); dwa.states.len()];
    nwa.start_states = vec![dwa.start_state];

    for (state_id, state) in dwa.states.iter().enumerate() {
        if let Some(final_weight) = state.final_weight.clone() {
            nwa.states[state_id].final_weight = Some(final_weight);
        }
        for (&label, (target, weight)) in &state.transitions {
            nwa.states[state_id]
                .transitions
                .entry(label)
                .or_default()
                .push((*target, weight.clone()));
        }
    }

    nwa
}

fn optimize_parser_default_transitions(
    nwa: &mut NWA,
    state_supports: &[Vec<u32>],
    recognizer: &ViableSuffixRecognizer,
    num_parser_states: u32,
) -> bool {
    fn subtract_weight_from_outgoing(
        state: &mut crate::automata::weighted::nwa::NWAState,
        weight_to_subtract: &Weight,
    ) -> bool {
        if weight_to_subtract.is_empty() {
            return false;
        }

        let mut changed = false;
        for (_, edge_weight) in &mut state.epsilons {
            let new_weight = edge_weight.difference(weight_to_subtract);
            if new_weight != *edge_weight {
                *edge_weight = new_weight;
                changed = true;
            }
        }
        for targets in state.transitions.values_mut() {
            for (_, edge_weight) in targets.iter_mut() {
                let new_weight = edge_weight.difference(weight_to_subtract);
                if new_weight != *edge_weight {
                    *edge_weight = new_weight;
                    changed = true;
                }
            }
            targets.retain(|(_, edge_weight)| !edge_weight.is_empty());
        }
        state.epsilons.retain(|(_, edge_weight)| !edge_weight.is_empty());
        state.transitions.retain(|_, targets| !targets.is_empty());
        changed
    }

    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some();
    let possible_by_state: Vec<Vec<u32>> = state_supports
        .iter()
        .map(|support| {
            recognizer
                .subset_to_state
                .get(support)
                .and_then(|state_id| recognizer.possible_outgoing_ids.get(*state_id as usize))
                .cloned()
                .unwrap_or_default()
        })
        .collect();

    let mut any_changed = false;
    let mut iterations = 0usize;

    loop {
        iterations += 1;
        let mut changed = false;

        for (state_id, possible_ids) in possible_by_state.iter().enumerate() {
            if possible_ids.is_empty() {
                continue;
            }
            if possible_ids.len() < 2 {
                continue;
            }

            let possible_set: BTreeSet<u32> = possible_ids.iter().copied().collect();
            let actual_positive: BTreeSet<u32> = nwa.states[state_id]
                .transitions
                .iter()
                .filter_map(|(&label, _)| parser_state_label(label, num_parser_states))
                .collect();
            if actual_positive != possible_set {
                continue;
            }

            let mut shared_target: Option<u32> = None;
            let mut default_weight: Option<Weight> = None;
            let mut valid = true;

            for parser_state_id in possible_ids {
                let Some(targets) = nwa.states[state_id].transitions.get(&(*parser_state_id as i32)) else {
                    valid = false;
                    break;
                };
                if targets.len() != 1 {
                    valid = false;
                    break;
                }

                let (target, weight) = &targets[0];
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

            let Some(target) = shared_target else {
                continue;
            };
            let Some(default_weight) = default_weight else {
                continue;
            };
            if !valid || default_weight.is_empty() {
                continue;
            }

            if add_or_union_transition(&mut nwa.states[state_id], DEFAULT_LABEL, target, default_weight) {
                changed = true;
            }
        }

        for state_id in 0..nwa.states.len() {
            let default_targets = nwa.states[state_id]
                .transitions
                .get(&DEFAULT_LABEL)
                .cloned()
                .unwrap_or_default();
            if default_targets.is_empty() {
                continue;
            }

            let mut lifted_final = Weight::empty();
            for (_index, (target, weight)) in default_targets.iter().enumerate() {
                let Some(target_final) = nwa.states[*target as usize].final_weight.as_ref() else {
                    continue;
                };
                let lifted = weight.intersection(target_final);
                if lifted.is_empty() {
                    continue;
                }
                lifted_final = lifted_final.union(&lifted);
            }

            if union_final_weight(&mut nwa.states[state_id].final_weight, lifted_final.clone()) {
                changed = true;
            }

            if subtract_weight_from_outgoing(&mut nwa.states[state_id], &lifted_final) {
                changed = true;
            }
        }

        for state in &mut nwa.states {
            let mut default_by_target: HashMap<u32, Weight> = HashMap::new();
            if let Some(default_targets) = state.transitions.get(&DEFAULT_LABEL) {
                for (target, weight) in default_targets {
                    default_by_target
                        .entry(*target)
                        .and_modify(|existing| *existing = existing.union(weight))
                        .or_insert_with(|| weight.clone());
                }
            }
            if default_by_target.is_empty() {
                continue;
            }

            for (&label, targets) in state.transitions.iter_mut() {
                if label == DEFAULT_LABEL {
                    continue;
                }
                for (target, weight) in targets.iter_mut() {
                    let Some(default_weight) = default_by_target.get(target) else {
                        continue;
                    };
                    let new_weight = weight.difference(default_weight);
                    if new_weight != *weight {
                        *weight = new_weight;
                        changed = true;
                    }
                }
                targets.retain(|(_, weight)| !weight.is_empty());
            }
            state.transitions.retain(|_, targets| !targets.is_empty());
        }

        if !changed {
            break;
        }
        any_changed = true;
    }

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] default_opt iterations={} changed={}",
            iterations,
            any_changed,
        );
    }

    any_changed
}

pub fn build_parser_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    build_parser_dwa_with_report(table, grammar, tokenizer, vocab, id_map, ignore_terminal).0
}

pub(crate) fn build_parser_dwa_with_report(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> (DWA, ParserDwaBuildReport) {
    // Overlap terminal-DWA construction with template compilation:
    // terminal_dwa depends on (grammar, tokenizer, vocab, id_map)
    // templates depend on (table, grammar) only — no terminal_dwa
    #[cfg(feature = "rayon")]
    let ((terminal_dwa, terminal_build), (characterizations, templates)) = rayon::join(
        || build_terminal_dwa_with_report(grammar, tokenizer, vocab, id_map, ignore_terminal),
        || {
            let characterizations = characterize_terminals(table, grammar);
            let templates = Templates::from_characterizations(&characterizations);
            (characterizations, templates)
        },
    );
    #[cfg(not(feature = "rayon"))]
    let ((terminal_dwa, terminal_build), (characterizations, templates)) = {
        let td = build_terminal_dwa_with_report(grammar, tokenizer, vocab, id_map, ignore_terminal);
        let characterizations = characterize_terminals(table, grammar);
        let templates = Templates::from_characterizations(&characterizations);
        (td, (characterizations, templates))
    };

    let (mut parser_dwa, mut report) =
        build_parser_dwa_from_terminal_dwa_with_precomputed_templates_report(
            table, grammar, tokenizer, &terminal_dwa, characterizations, templates,
        );
    report.terminal_build = Some(terminal_build);
    parser_dwa.clip_weights(id_map.max_internal_token_id());
    (parser_dwa, report)
}

pub(crate) fn build_parser_dwa_from_terminal_dwa(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    terminal_dwa: &DWA,
) -> DWA {
    build_parser_dwa_from_terminal_dwa_with_report(table, grammar, tokenizer, terminal_dwa).0
}

pub(crate) fn build_parser_dwa_from_terminal_dwa_with_report(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    terminal_dwa: &DWA,
) -> (DWA, ParserDwaBuildReport) {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some();
    let phase_started_at = std::time::Instant::now();
    let characterizations = characterize_terminals(table, grammar);
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] characterize_terminals_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    let phase_started_at = std::time::Instant::now();
    let templates = Templates::from_characterizations(&characterizations);
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] from_characterizations_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }
    build_parser_dwa_from_terminal_dwa_with_precomputed_templates_report(
        table, grammar, tokenizer, terminal_dwa, characterizations, templates,
    )
}

pub(crate) fn build_parser_dwa_from_terminal_dwa_with_precomputed_templates_report(
    table: &GLRTable,
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    terminal_dwa: &DWA,
    characterizations: BTreeMap<TerminalID, TerminalCharacterization>,
    templates: Templates,
) -> (DWA, ParserDwaBuildReport) {
    let profile_enabled = std::env::var_os("GLRMASK_PROFILE_PARSER_DWA").is_some();
    let total_started_at = std::time::Instant::now();
    let mut report = ParserDwaBuildReport::default();

    // characterize_terminals and build_templates already done above
    report.characterizations = collect_characterization_stats(&characterizations);
    report.templates = collect_template_stats(&templates);

    let phase_started_at = std::time::Instant::now();
    let (states, bundle_stats) = build_state_summaries(terminal_dwa, grammar, &templates);
    report.build_state_summaries_time = phase_started_at.elapsed();
    report.bundles = bundle_stats;
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] build_state_summaries_ms={:.3} states={}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
            states.len(),
        );
    }

    let phase_started_at = std::time::Instant::now();
    let mut arena = NWA::new(0, 0);
    let mut memo = vec![None; states.len()];
    let mut concat_memo = HashMap::new();
    let compose_profile_enabled = std::env::var_os("GLRMASK_PROFILE_COMPOSE_STATE").is_some();
    let mut compose_profile = compose_profile_enabled.then(ComposeStateProfile::default);
    let Some(parser_body) = compose_state(
        terminal_dwa.start_state,
        &states,
        &mut arena,
        &mut memo,
        &mut concat_memo,
        compose_profile.as_mut(),
    )
    else {
        return (DWA::new(0, 0), report);
    };
    arena.start_states = parser_body.start_states.clone();
    let mut parser_nwa = arena;
    report.compose_state_time = phase_started_at.elapsed();
    report.parser_nwa_before_resolve = collect_weighted_nwa_stats(&parser_nwa);
    let (negative_edges_before_resolve, default_edges_before_resolve) = count_special_edges(&parser_nwa);
    report.negative_edges_before_resolve = negative_edges_before_resolve;
    report.default_edges_before_resolve = default_edges_before_resolve;
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] compose_state_ms={:.3} memo_entries={} {}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
            memo.iter().filter(|entry| entry.is_some()).count(),
            report.parser_nwa_before_resolve,
        );
        if let Some(profile) = compose_profile.as_ref() {
            eprintln!(
                "[glrmask/profile][parser_dwa] compose_state_detail calls={} memo_hits={} branches={} concat_hits={} concat_misses={} accepting_ms={:.3} continuation_ms={:.3} concat_build_ms={:.3} union_ms={:.3}",
                profile.calls,
                profile.memo_hits,
                profile.branches,
                profile.concat_hits,
                profile.concat_misses,
                profile.accepting_ms.as_secs_f64() * 1000.0,
                profile.continuation_ms.as_secs_f64() * 1000.0,
                profile.concat_build_ms.as_secs_f64() * 1000.0,
                profile.union_ms.as_secs_f64() * 1000.0,
            );
            eprintln!(
                "[glrmask/profile][parser_dwa] compose_state_concat_shapes avg_left_states={:.3} avg_right_states={:.3} max_left_states={} max_right_states={}",
                if profile.concat_misses == 0 {
                    0.0
                } else {
                    profile.concat_left_states as f64 / profile.concat_misses as f64
                },
                if profile.concat_misses == 0 {
                    0.0
                } else {
                    profile.concat_right_states as f64 / profile.concat_misses as f64
                },
                profile.max_concat_left_states,
                profile.max_concat_right_states,
            );
        }
    }

    let phase_started_at = std::time::Instant::now();
    resolve_negative_codes_in_nwa(&mut parser_nwa);
    report.resolve_negative_codes_time = phase_started_at.elapsed();
    report.parser_nwa_after_resolve = collect_weighted_nwa_stats(&parser_nwa);
    let (negative_edges_after_resolve, default_edges_after_resolve) = count_special_edges(&parser_nwa);
    report.negative_edges_after_resolve = negative_edges_after_resolve;
    report.default_edges_after_resolve = default_edges_after_resolve;
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] resolve_negative_codes_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let vsr_started_at = std::time::Instant::now();
    let viable_suffix_recognizer = build_viable_suffix_recognizer(&parser_nwa, table.num_states);
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] viable_suffix_ms={:.3} states={}",
            vsr_started_at.elapsed().as_secs_f64() * 1000.0,
            viable_suffix_recognizer.possible_outgoing_ids.len(),
        );
    }

    let phase_started_at = std::time::Instant::now();
    let determinized = determinize_with_supports(&parser_nwa);
    let parser_dwa_pre_minimize = determinized.dwa;
    let det_elapsed = phase_started_at.elapsed();
    report.parser_dwa_pre_minimize = collect_weighted_dwa_stats(&parser_dwa_pre_minimize);
    if profile_enabled {
        let pre_min_default_trans = parser_dwa_pre_minimize.states.iter()
            .filter(|s| s.transitions.contains_key(&DEFAULT_LABEL))
            .count();
        eprintln!(
            "[glrmask/profile][parser_dwa] determinize_with_supports_ms={:.3} states={} default_trans={}",
            det_elapsed.as_secs_f64() * 1000.0,
            report.parser_dwa_pre_minimize.states,
            pre_min_default_trans,
        );
        profile_dump_small_automaton(
            "determinize_with_supports",
            &parser_dwa_pre_minimize,
            parser_dwa_pre_minimize.states.len(),
        );
    }

    let mut optimized_parser_nwa = dwa_to_nwa(&parser_dwa_pre_minimize);
    let optimize_started_at = std::time::Instant::now();
    let default_opt_enabled = std::env::var_os("GLRMASK_DISABLE_PARSER_DEFAULT_OPT").is_none();
    if default_opt_enabled {
        optimize_parser_default_transitions(
            &mut optimized_parser_nwa,
            &determinized.supports,
            &viable_suffix_recognizer,
            table.num_states,
        );
    }
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] default_opt_ms={:.3} enabled={}",
            optimize_started_at.elapsed().as_secs_f64() * 1000.0,
            default_opt_enabled,
        );
        profile_dump_small_automaton(
            "after_default_opt_nwa",
            &optimized_parser_nwa,
            optimized_parser_nwa.states.len(),
        );
    }

    let subtract_started_at = std::time::Instant::now();
    optimized_parser_nwa.subtract_final_weights_from_outgoing();
    report.subtract_final_weights_time = subtract_started_at.elapsed();
    report.parser_nwa_after_subtract = collect_weighted_nwa_stats(&optimized_parser_nwa);
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] subtract_final_weights_ms={:.3}",
            subtract_started_at.elapsed().as_secs_f64() * 1000.0,
        );
        profile_dump_small_automaton(
            "after_subtract_final_nwa",
            &optimized_parser_nwa,
            optimized_parser_nwa.states.len(),
        );
    }

    let min_started_at = std::time::Instant::now();
    let determinized_after_defaults = determinize(&optimized_parser_nwa)
        .expect("parser NWA determinization failed after default-transition optimization");
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] determinize_after_defaults_ms={:.3} states={}",
            min_started_at.elapsed().as_secs_f64() * 1000.0,
            determinized_after_defaults.num_states(),
        );
        profile_dump_small_automaton(
            "determinize_after_defaults",
            &determinized_after_defaults,
            determinized_after_defaults.states.len(),
        );
    }
    let core_dwa = minimize(&determinized_after_defaults);
    let min_elapsed = min_started_at.elapsed();
    report.determinize_minimize_time = phase_started_at.elapsed();
    report.parser_dwa_minimized = collect_weighted_dwa_stats(&core_dwa);
    report.total_time = total_started_at.elapsed();
    if profile_enabled {
        let post_min_default_trans = core_dwa.states.iter()
            .filter(|s| s.transitions.contains_key(&DEFAULT_LABEL))
            .count();
        eprintln!(
            "[glrmask/profile][parser_dwa] determinize_minimize_ms={:.3} det_ms={:.3} min_ms={:.3} pre_min_states={} {} post_min_default_trans={}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
            det_elapsed.as_secs_f64() * 1000.0,
            min_elapsed.as_secs_f64() * 1000.0,
            report.parser_dwa_pre_minimize.states,
            report.parser_dwa_minimized,
            post_min_default_trans,
        );
        eprintln!(
            "[glrmask/profile][parser_dwa] total_ms={:.3} {}",
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            report.parser_dwa_minimized,
        );
        profile_dump_small_automaton("final_minimized", &core_dwa, core_dwa.states.len());

        // Per-state transition count distribution
        let mut trans_dist: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        let mut single_chain = 0usize;
        let mut has_default_count = 0usize;
        for state in &core_dwa.states {
            let n = state.transitions.len();
            *trans_dist.entry(n).or_default() += 1;
            if n == 1 {
                single_chain += 1;
            }
            if state.transitions.contains_key(&DEFAULT_LABEL) {
                has_default_count += 1;
            }
        }
        let dist_str: Vec<String> = trans_dist.iter().map(|(k, v)| format!("{}:{}", k, v)).collect();
        eprintln!(
            "[glrmask/profile][parser_dwa] transition_dist total_states={} single_trans={} has_default={} dist=[{}]",
            core_dwa.states.len(),
            single_chain,
            has_default_count,
            dist_str.join(", "),
        );
    }

    (core_dwa, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::GrammarDef;
    use crate::compiler::grammar::model::tests::*;
    use range_set_blaze::RangeSetBlaze;

    fn token_weight(tokens: &[u32]) -> Weight {
        Weight::from_token_set_for_tsid(
            0,
            RangeSetBlaze::from_iter(tokens.iter().copied().map(|token| token..=token)),
        )
    }

    fn make_vocab_and_preprocessing(gdef: &GrammarDef) -> (Vocab, Tokenizer, InternalIdMap) {
        let tok = crate::compiler::compile::build_tokenizer(gdef);

        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for (i, td) in gdef.terminals.iter().enumerate() {
            entries.push((i as u32, td.name().as_bytes().to_vec()));
        }
        let vocab = Vocab::new(entries, None);
        let id_map = InternalIdMap::build(&tok, &vocab, &std::collections::BTreeMap::new(), None);
        (vocab, tok, id_map)
    }

    #[test]
    fn test_build_parser_dwa_simple() {
        let gdef = simple_ab_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp, None);
        assert!(dwa.num_states() > 0);
    }

    #[test]
    fn test_build_parser_dwa_choice() {
        let gdef = choice_grammar();
        let gg = AnalyzedGrammar::from_grammar_def(&gdef);
        let table = GLRTable::build(&gg);
        let (vocab, tok, vp) = make_vocab_and_preprocessing(&gdef);

        let dwa = build_parser_dwa(&table, &gg, &tok, &vocab, &vp, None);
        assert!(dwa.num_states() > 0);
    }

    #[test]
    fn test_optimize_parser_defaults_synthesizes_and_subtracts_same_destination_weight() {
        let mut dwa = DWA::new(0, 0);
        let accept = dwa.add_state();
        dwa.add_transition(0, 0, accept, token_weight(&[1, 2]));
        dwa.add_transition(0, 1, accept, token_weight(&[2, 3]));

        let mut nwa = dwa_to_nwa(&dwa);
        let supports = vec![vec![0], vec![1]];
        let recognizer = ViableSuffixRecognizer {
            subset_to_state: HashMap::from([(vec![0], 0), (vec![1], 1)]),
            possible_outgoing_ids: vec![vec![0, 1], vec![]],
        };

        assert!(optimize_parser_default_transitions(&mut nwa, &supports, &recognizer, 2));

        let defaults = nwa.states[0].transitions.get(&DEFAULT_LABEL).expect("default edge");
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].0, accept);
        assert_eq!(defaults[0].1, token_weight(&[2]));

        let explicit_zero = &nwa.states[0].transitions[&0][0];
        let explicit_one = &nwa.states[0].transitions[&1][0];
        assert_eq!(explicit_zero.1, token_weight(&[1]));
        assert_eq!(explicit_one.1, token_weight(&[3]));
    }

    #[test]
    fn test_optimize_parser_defaults_lifts_default_weight_into_source_final() {
        let mut nwa = NWA::new(0, 0);
        nwa.states = vec![crate::automata::weighted::nwa::NWAState::default(); 2];
        nwa.start_states = vec![0];
        nwa.states[0]
            .transitions
            .entry(DEFAULT_LABEL)
            .or_default()
            .push((1, token_weight(&[1, 2])));
        nwa.states[1].final_weight = Some(token_weight(&[2, 3]));

        let supports = vec![vec![0], vec![1]];
        let recognizer = ViableSuffixRecognizer {
            subset_to_state: HashMap::from([(vec![0], 0), (vec![1], 1)]),
            possible_outgoing_ids: vec![vec![], vec![]],
        };

        assert!(optimize_parser_default_transitions(&mut nwa, &supports, &recognizer, 4));

        assert_eq!(nwa.states[0].final_weight, Some(token_weight(&[2])));
        let defaults = nwa.states[0].transitions.get(&DEFAULT_LABEL).expect("default edge");
        assert_eq!(defaults.len(), 1);
        assert_eq!(defaults[0].1, token_weight(&[1]));
    }
}
