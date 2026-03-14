#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
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
            "[glrmask/profile][parser_dwa] compose_state_ms={:.3} memo_entries={} nwa_states={}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
            memo.iter().filter(|entry| entry.is_some()).count(),
            parser_nwa.states.len(),
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

    let phase_started_at = std::time::Instant::now();
    parser_nwa.subtract_final_weights_from_outgoing();
    report.subtract_final_weights_time = phase_started_at.elapsed();
    report.parser_nwa_after_subtract = collect_weighted_nwa_stats(&parser_nwa);
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] subtract_final_weights_ms={:.3}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let phase_started_at = std::time::Instant::now();
    let parser_dwa_pre_minimize = determinize(&parser_nwa)
        .expect("parser NWA determinization failed despite acyclic terminal/template composition");
    let det_elapsed = phase_started_at.elapsed();
    report.parser_dwa_pre_minimize = collect_weighted_dwa_stats(&parser_dwa_pre_minimize);
    let min_started_at = std::time::Instant::now();
    let core_dwa = minimize(&parser_dwa_pre_minimize);
    let min_elapsed = min_started_at.elapsed();
    report.determinize_minimize_time = phase_started_at.elapsed();
    report.parser_dwa_minimized = collect_weighted_dwa_stats(&core_dwa);
    report.total_time = total_started_at.elapsed();
    if profile_enabled {
        eprintln!(
            "[glrmask/profile][parser_dwa] determinize_minimize_ms={:.3} det_ms={:.3} min_ms={:.3} pre_min_states={} dwa_states={}",
            phase_started_at.elapsed().as_secs_f64() * 1000.0,
            det_elapsed.as_secs_f64() * 1000.0,
            min_elapsed.as_secs_f64() * 1000.0,
            parser_dwa_pre_minimize.states.len(),
            core_dwa.states.len(),
        );
        eprintln!(
            "[glrmask/profile][parser_dwa] total_ms={:.3} dwa_states={} dwa_transitions={}",
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            core_dwa.num_states(),
            core_dwa.num_transitions(),
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
}
