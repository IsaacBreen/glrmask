use std::collections::{BTreeMap, BTreeSet};
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

fn compact<T: Ord>(values: impl IntoIterator<Item = T>) -> (Vec<u32>, Vec<usize>) {
    let mut ids = BTreeMap::new();
    let mut reps = Vec::new();
    let classes = values.into_iter().enumerate().map(|(i, value)| match ids.get(&value) {
        Some(&id) => id,
        None => { let id = reps.len() as u32; ids.insert(value, id); reps.push(i); id }
    }).collect();
    (classes, reps)
}

pub(super) fn finish(
    input: BuildInput<'_>,
    aliases: &[Vec<u32>],
    signatures: &[Vec<u32>],
    mut state_class: Vec<u32>,
    mut rows: Vec<Vec<u32>>,
    scan_ms: f64,
    total_ms: impl FnOnce() -> f64,
) -> Option<Finished> {
    let compact_started = Instant::now();
    let (token_class, token_reps) = compact((0..aliases.len()).map(|v|
        rows.iter().map(|row| row[v]).collect::<Vec<_>>()
    ));
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
        state_class, rows.len() as u32, state_reps,
    );
    let mut original_to_internal = vec![u32::MAX; input.vocab.max_token_id() as usize + 1];
    for (unique, originals) in aliases.iter().enumerate() {
        for &original in originals { original_to_internal[original as usize] = token_class[unique]; }
    }
    let vocab_tokens = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
        original_to_internal, token_reps.len() as u32,
    );
    let compact_ms = compact_started.elapsed().as_secs_f64() * 1000.0;

    let build_started = Instant::now();
    let mut by_terminal = BTreeMap::<u32, Vec<BTreeSet<u32>>>::new();
    for (state, row) in rows.iter().enumerate() {
        for (token, &signature) in row.iter().enumerate() {
            for &terminal in &signatures[signature as usize] {
                by_terminal.entry(terminal).or_insert_with(|| vec![BTreeSet::new(); rows.len()])
                    [state].insert(token_class[token]);
            }
        }
    }
    let mut dwa = DWA::new(rows.len() as u32, token_reps.len().saturating_sub(1) as u32);
    let final_state = dwa.add_state();
    dwa.set_final_weight(final_state, Weight::all());
    for (terminal, per_state) in by_terminal {
        let weight = Weight::from_per_tsid_token_sets(per_state.into_iter().enumerate().filter_map(
            |(state, tokens)| (!tokens.is_empty()).then(||
                (state as u32, tokens.into_iter().collect::<RangeSetBlaze<u32>>())
            ),
        ));
        if !weight.is_empty() { dwa.add_transition(dwa.start_state(), terminal as i32, final_state, weight); }
    }
    if dwa.num_transitions() == 0 { return None; }
    let build_ms = build_started.elapsed().as_secs_f64() * 1000.0;
    let state_classes = rows.len();
    let token_classes = token_reps.len();
    Some(Finished {
        artifact: LocalIdMapTerminalDwa {
            id_map: InternalIdMap { tokenizer_states, vocab_tokens, deferred_vocab_singleton_original_ids: None },
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
        token_classes,
    })
}
