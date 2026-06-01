//! Composition of Terminal-DWA branches with parser stack-effect templates.
//!
//! This is the mathematical core of Parser-DWA construction.  Each productive
//! Terminal-DWA state becomes a continuation state in a parser NWA.  Each
//! outgoing Terminal-DWA branch is replaced by the template automaton(s) for
//! the terminals in that branch, weighted by the Terminal-DWA edge masks, with
//! template finals redirected to the branch continuation.

use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHashMap;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::{NWA, NwaBody};
use crate::parser::glr::analysis::AnalyzedGrammar;
use crate::compile::template_dfa::Templates;
use crate::sets::weight::Weight;

use super::profiling::{
    elapsed_ms, emit_parser_bundle_profile, emit_parser_dwa_compose_profiles,
    parser_dwa_compose_detail_enabled, ParserDwaComposeDetailProfile, ParserNwaBuildProfile,
};
use super::terminal_projection::{build_state_summaries, compute_productive_terminal_states};
use super::types::StateSummaries;

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
            detail.accumulate_bundle_profile(&bundle_profile);
            emit_parser_bundle_profile(bundle_id, &bundle_profile);
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

pub(crate) fn build_parser_nwa_from_terminal_dwa(
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

    let mut built_bundle_cache: Vec<Option<Arc<NWA>>> = if compose_detail_enabled {
        vec![None; summaries.unique_bundles.len()]
    } else {
        use rayon::prelude::*;

        summaries
            .unique_bundles
            .par_iter()
            .enumerate()
            .map(|(bundle_id, bundle)| {
                used_multi_bundle[bundle_id]
                    .then(|| Arc::new(templates.build_bundle(bundle)))
            })
            .collect()
    };

    let branch_walk_started_at = Instant::now();
    for (state_id, state) in states.iter().enumerate() {
        if !productive[state_id] {
            continue;
        }
        let from = continuation_states[state_id];
        debug_assert_ne!(from, u32::MAX);

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
            debug_assert_ne!(target_continuation, u32::MAX);
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
    debug_assert_ne!(start, u32::MAX);
    arena.set_start_states(vec![start]);
    let compose_state_ms = elapsed_ms(graph_started_at);

    if compose_detail_enabled {
        emit_parser_dwa_compose_profiles(&compose_detail);
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
