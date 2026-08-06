use std::time::Instant;

use range_set_blaze::RangeSetBlaze;

use super::{BuildInput, LocalIdMapTerminalDwa};
use crate::automata::lexer::Lexer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::Weight;
use crate::terminal_dwa::types::TerminalDwaPhaseProfile;

pub(super) struct Finished {
    pub artifact: LocalIdMapTerminalDwa,
    pub compact_ms: f64,
    pub build_ms: f64,
    pub state_classes: usize,
    pub token_classes: usize,
}

fn compact_columns(rows: &[Vec<u32>], tokens: usize, signatures: usize) -> (Vec<u32>, Vec<usize>) {
    let mut classes = vec![0u32; tokens];
    let mut next = vec![0u32; tokens];
    let mut class_count = 1usize;
    let mut seen = Vec::<u32>::new();
    let mut ids = Vec::<u32>::new();
    for (round, row) in rows.iter().enumerate() {
        let needed = class_count * signatures;
        seen.resize(needed, u32::MAX);
        ids.resize(needed, 0);
        let epoch = round as u32;
        let mut next_count = 0u32;
        for token in 0..tokens {
            let key = classes[token] as usize * signatures + row[token] as usize;
            if seen[key] != epoch {
                seen[key] = epoch;
                ids[key] = next_count;
                next_count += 1;
            }
            next[token] = ids[key];
        }
        std::mem::swap(&mut classes, &mut next);
        class_count = next_count as usize;
    }
    let mut reps = vec![usize::MAX; class_count];
    for (token, &class) in classes.iter().enumerate() {
        reps[class as usize] = reps[class as usize].min(token);
    }
    (classes, reps)
}

pub(super) fn finish(
    input: BuildInput<'_>,
    aliases: &[Vec<u32>],
    signatures: &[Vec<u32>],
    state_class: Vec<u32>,
    rows: Vec<Vec<u32>>,
    scan_ms: f64,
    total_ms: impl FnOnce() -> f64,
) -> Option<Finished> {
    let compact_started = Instant::now();
    let (token_class, token_reps) = compact_columns(&rows, aliases.len(), signatures.len());
    let compact_rows = rows
        .iter()
        .map(|row| token_reps.iter().map(|&token| row[token]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let compact_ms = compact_started.elapsed().as_secs_f64() * 1000.0;
    finish_compacted(
        input,
        aliases,
        signatures,
        state_class,
        compact_rows,
        token_class,
        scan_ms,
        compact_ms,
        total_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn finish_compacted(
    input: BuildInput<'_>,
    aliases: &[Vec<u32>],
    signatures: &[Vec<u32>],
    mut state_class: Vec<u32>,
    mut rows: Vec<Vec<u32>>,
    token_class: Vec<u32>,
    scan_ms: f64,
    compact_ms: f64,
    total_ms: impl FnOnce() -> f64,
) -> Option<Finished> {
    let initial = input.tokenizer.initial_state_id() as usize;
    let initial_class = state_class[initial];
    if state_class.iter().filter(|&&class| class == initial_class).count() > 1 {
        rows.push(rows[initial_class as usize].clone());
        state_class[initial] = (rows.len() - 1) as u32;
    }
    let mut state_reps = vec![u32::MAX; rows.len()];
    for (raw, &class) in state_class.iter().enumerate() {
        state_reps[class as usize] = state_reps[class as usize].min(raw as u32);
    }
    let tokenizer_states = ManyToOneIdMap::from_original_to_internal_with_representatives(
        state_class,
        rows.len() as u32,
        state_reps,
    );
    let token_classes = token_class.iter().copied().max().map_or(0, |class| class + 1);
    let mut original_to_internal = vec![u32::MAX; input.vocab.max_token_id() as usize + 1];
    for (unique, originals) in aliases.iter().enumerate() {
        for &original in originals {
            original_to_internal[original as usize] = token_class[unique];
        }
    }
    let vocab_tokens = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
        original_to_internal,
        token_classes,
    );

    let build_started = Instant::now();
    let num_terminals = input.grammar.num_terminals as usize;
    let mut by_terminal = vec![Vec::<(u32, Vec<u32>)>::new(); num_terminals];
    let mut pending = vec![Vec::<u32>::new(); num_terminals];
    let mut touched = Vec::<usize>::new();
    for (state, row) in rows.iter().enumerate() {
        for (token, &signature) in row.iter().enumerate() {
            for &terminal in &signatures[signature as usize] {
                let terminal = terminal as usize;
                if pending[terminal].is_empty() {
                    touched.push(terminal);
                }
                pending[terminal].push(token as u32);
            }
        }
        for terminal in touched.drain(..) {
            by_terminal[terminal].push((state as u32, std::mem::take(&mut pending[terminal])));
        }
    }
    let mut dwa = DWA::new(rows.len() as u32, token_classes.saturating_sub(1));
    let final_state = dwa.add_state();
    dwa.set_final_weight(final_state, Weight::all());
    for (terminal, per_state) in by_terminal.into_iter().enumerate() {
        if per_state.is_empty() {
            continue;
        }
        let weight = Weight::from_per_tsid_token_sets(
            per_state
                .into_iter()
                .map(|(state, tokens)| (state, tokens.into_iter().collect::<RangeSetBlaze<u32>>())),
        );
        if !weight.is_empty() {
            dwa.add_transition(dwa.start_state(), terminal as i32, final_state, weight);
        }
    }
    if dwa.num_transitions() == 0 {
        return None;
    }
    let build_ms = build_started.elapsed().as_secs_f64() * 1000.0;
    let state_classes = rows.len();
    Some(Finished {
        artifact: LocalIdMapTerminalDwa {
            id_map: InternalIdMap {
                tokenizer_states,
                vocab_tokens,
                deferred_vocab_singleton_original_ids: None,
            },
            dwa,
            profile: TerminalDwaPhaseProfile {
                id_map_ms: scan_ms,
                terminal_dwa_ms: build_ms,
                compact_ms,
                split_terminal_dwa_total_ms: total_ms(),
                global_merge_ms: 0.0,
            },
        },
        compact_ms,
        build_ms,
        state_classes,
        token_classes: token_classes as usize,
    })
}
