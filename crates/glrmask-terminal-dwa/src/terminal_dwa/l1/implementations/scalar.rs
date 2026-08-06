//! Deliberately slow scalar L1 definition.
//!
//! Walk every `(raw state, vocabulary token)` pair. After consuming the whole
//! token, the active signature is the union of terminals finalized at every end
//! state and terminals listed in `possible_future_group_ids` there. Intern those
//! signatures, quotient identical state rows and token columns, and directly
//! construct the two-state weighted DWA.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use range_set_blaze::RangeSetBlaze;

use super::{BuildInput, LocalIdMapTerminalDwa};
use crate::automata::lexer::Lexer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::stages::equiv_types::{InternalIdMap, ManyToOneIdMap};
use crate::ds::weight::Weight;
use crate::terminal_dwa::types::TerminalDwaPhaseProfile;

fn compact<T: Ord>(values: impl IntoIterator<Item = T>) -> (Vec<u32>, Vec<usize>) {
    let mut ids = BTreeMap::new();
    let mut representatives = Vec::new();
    let classes = values
        .into_iter()
        .enumerate()
        .map(|(index, value)| match ids.get(&value) {
            Some(&id) => id,
            None => {
                let id = representatives.len() as u32;
                ids.insert(value, id);
                representatives.push(index);
                id
            }
        })
        .collect();
    (classes, representatives)
}

pub(super) fn build(input: BuildInput<'_>) -> Option<LocalIdMapTerminalDwa> {
    if input.vocab.is_empty() {
        return None;
    }
    let total_started = Instant::now();
    let tokens = input.vocab.iter().map(|(id, bytes)| (id, bytes.to_vec())).collect::<Vec<_>>();

    let scan_started = Instant::now();
    let mut signatures = vec![Vec::<u32>::new()];
    let mut signature_ids = BTreeMap::from([(Vec::new(), 0u32)]);
    let mut unique_rows = Vec::<Vec<u32>>::new();
    let mut row_ids = BTreeMap::<Vec<u32>, u32>::new();
    let mut state_class = Vec::with_capacity(input.tokenizer.num_states() as usize);
    for state in 0..input.tokenizer.num_states() {
        let row = tokens
            .iter()
            .map(|(_, bytes)| {
                let mut terminals = BTreeSet::new();
                for end in input.tokenizer.execute_from_state_end_only(bytes, state) {
                    terminals.extend(super::super::collect_active_terminal_signature(
                        input.tokenizer,
                        end,
                        input.active_terminals,
                    ));
                }
                let signature = terminals.into_iter().collect::<Vec<_>>();
                if let Some(&id) = signature_ids.get(&signature) {
                    id
                } else {
                    let id = signatures.len() as u32;
                    signature_ids.insert(signature.clone(), id);
                    signatures.push(signature);
                    id
                }
            })
            .collect::<Vec<_>>();
        let class = if let Some(&class) = row_ids.get(&row) {
            class
        } else {
            let class = unique_rows.len() as u32;
            row_ids.insert(row.clone(), class);
            unique_rows.push(row);
            class
        };
        state_class.push(class);
    }
    let scan_ms = scan_started.elapsed().as_secs_f64() * 1000.0;

    let compact_started = Instant::now();
    let (token_class, token_representatives) = compact((0..tokens.len()).map(|token| {
        unique_rows.iter().map(|row| row[token]).collect::<Vec<_>>()
    }));
    let initial = input.tokenizer.initial_state_id() as usize;
    let initial_class = state_class[initial];
    let mut state_representatives = (0..unique_rows.len())
        .map(|class| state_class.iter().position(|&candidate| candidate == class as u32).unwrap() as u32)
        .collect::<Vec<_>>();
    if state_class.iter().filter(|&&class| class == initial_class).count() > 1 {
        unique_rows.push(unique_rows[initial_class as usize].clone());
        state_class[initial] = (unique_rows.len() - 1) as u32;
        state_representatives.push(initial as u32);
    }

    let tokenizer_states = ManyToOneIdMap::from_original_to_internal_with_representatives(
        state_class,
        unique_rows.len() as u32,
        state_representatives,
    );
    let mut original_token_to_internal = vec![u32::MAX; input.vocab.max_token_id() as usize + 1];
    for ((original, _), &class) in tokens.iter().zip(&token_class) {
        original_token_to_internal[*original as usize] = class;
    }
    let vocab_tokens = ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
        original_token_to_internal,
        token_representatives.len() as u32,
    );
    let compact_ms = compact_started.elapsed().as_secs_f64() * 1000.0;

    let build_started = Instant::now();
    let mut by_terminal = BTreeMap::<u32, Vec<BTreeSet<u32>>>::new();
    for (state, row) in unique_rows.iter().enumerate() {
        for (token, &signature) in row.iter().enumerate() {
            for &terminal in &signatures[signature as usize] {
                by_terminal
                    .entry(terminal)
                    .or_insert_with(|| vec![BTreeSet::new(); unique_rows.len()])[state]
                    .insert(token_class[token]);
            }
        }
    }

    let mut dwa = DWA::new(unique_rows.len() as u32, token_representatives.len().saturating_sub(1) as u32);
    let final_state = dwa.add_state();
    dwa.set_final_weight(final_state, Weight::all());
    for (terminal, rows) in by_terminal {
        let weight = Weight::from_per_tsid_token_sets(rows.into_iter().enumerate().filter_map(
            |(state, tokens)| (!tokens.is_empty()).then(|| {
                (state as u32, tokens.into_iter().collect::<RangeSetBlaze<u32>>())
            }),
        ));
        if !weight.is_empty() {
            dwa.add_transition(dwa.start_state(), terminal as i32, final_state, weight);
        }
    }
    if dwa.num_transitions() == 0 {
        return None;
    }
    let build_ms = build_started.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_started.elapsed().as_secs_f64() * 1000.0;

    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_scalar] partition={} states={} tokens={} cells={} signatures={} state_classes={} token_classes={} scan_ms={:.3} compact_ms={:.3} build_ms={:.3} total_ms={:.3}",
            input.partition_label,
            input.tokenizer.num_states(),
            tokens.len(),
            input.tokenizer.num_states() as usize * tokens.len(),
            signatures.len(),
            unique_rows.len(),
            token_representatives.len(),
            scan_ms,
            compact_ms,
            build_ms,
            total_ms,
        );
    }

    Some(LocalIdMapTerminalDwa {
        id_map: InternalIdMap { tokenizer_states, vocab_tokens, deferred_vocab_singleton_original_ids: None },
        dwa,
        profile: TerminalDwaPhaseProfile {
            id_map_ms: scan_ms,
            terminal_dwa_ms: build_ms,
            compact_ms,
            split_terminal_dwa_total_ms: total_ms,
            global_merge_ms: 0.0,
        },
    })
}
