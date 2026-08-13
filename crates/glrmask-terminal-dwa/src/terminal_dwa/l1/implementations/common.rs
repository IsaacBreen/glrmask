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

fn direct_vocab_id_map(
    max_token_id: u32,
    aliases: &[Vec<u32>],
    token_class: &[u32],
    token_classes: u32,
    preordered_originals: Option<&[(u32, u32)]>,
) -> ManyToOneIdMap {
    let mut original_to_internal = vec![u32::MAX; max_token_id as usize + 1];
    let mut internal_to_originals = vec![Vec::<u32>::new(); token_classes as usize];
    let mut representative_original_ids = vec![u32::MAX; token_classes as usize];

    if let Some(preordered) = preordered_originals {
        debug_assert_eq!(preordered.len(), aliases.iter().map(Vec::len).sum::<usize>());
        for &(original, unique) in preordered {
            let class = token_class[unique as usize] as usize;
            debug_assert!(class < internal_to_originals.len());
            original_to_internal[original as usize] = class as u32;
            if representative_original_ids[class] == u32::MAX {
                representative_original_ids[class] = original;
            }
            // `preordered` is globally sorted by original token ID, so each
            // class is built in the same sorted order the old path produced.
            internal_to_originals[class].push(original);
        }
    } else {
        for (unique, originals) in aliases.iter().enumerate() {
            let class = token_class[unique] as usize;
            debug_assert!(class < internal_to_originals.len());
            let members = &mut internal_to_originals[class];
            for &original in originals {
                original_to_internal[original as usize] = class as u32;
                representative_original_ids[class] =
                    representative_original_ids[class].min(original);
                members.push(original);
            }
        }
        for originals in &mut internal_to_originals {
            originals.sort_unstable();
        }
    }

    ManyToOneIdMap {
        original_to_internal,
        internal_to_originals,
        representative_original_ids,
    }
}

fn direct_vocab_map_enabled(input: &BuildInput<'_>) -> bool {
    std::env::var("GLRMASK_L1_DIRECT_VOCAB_MAP")
        .map(|value| {
            let value = value.trim();
            value.is_empty() || (value != "0" && !value.eq_ignore_ascii_case("false"))
        })
        .unwrap_or(
            matches!(input.partition_label, "p1" | "p2") && input.subset_parent_order.is_none(),
        )
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
        None,
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
    preordered_originals: Option<&[(u32, u32)]>,
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
    let vocab_tokens = if direct_vocab_map_enabled(&input) {
        direct_vocab_id_map(
            input.vocab.max_token_id(),
            aliases,
            &token_class,
            token_classes,
            preordered_originals,
        )
    } else {
        let mut original_to_internal =
            vec![u32::MAX; input.vocab.max_token_id() as usize + 1];
        for (unique, originals) in aliases.iter().enumerate() {
            for &original in originals {
                original_to_internal[original as usize] = token_class[unique];
            }
        }
        ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            original_to_internal,
            token_classes,
        )
    };

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


/// Finish an L1 artifact directly from its exact sparse terminal/token relation.
///
/// Each state row lists `(terminal, token_class)` memberships in terminal-major
/// order. This is already precisely the information encoded by the dense
/// signature matrix used by `finish_compacted`; rebuilding terminal-set
/// signature IDs first would only materialize an unnecessary intermediate.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_sparse_terminal_rows(
    input: BuildInput<'_>,
    aliases: &[Vec<u32>],
    mut state_class: Vec<u32>,
    mut sparse_rows: Vec<Vec<(u32, u32)>>,
    token_class: Vec<u32>,
    preordered_originals: Option<&[(u32, u32)]>,
    scan_ms: f64,
    compact_ms: f64,
    total_ms: impl FnOnce() -> f64,
) -> Option<Finished> {
    let state_map_started = Instant::now();
    let initial = input.tokenizer.initial_state_id() as usize;
    let initial_class = state_class[initial];
    if state_class.iter().filter(|&&class| class == initial_class).count() > 1 {
        sparse_rows.push(sparse_rows[initial_class as usize].clone());
        state_class[initial] = (sparse_rows.len() - 1) as u32;
    }

    let mut state_reps = vec![u32::MAX; sparse_rows.len()];
    for (raw, &class) in state_class.iter().enumerate() {
        state_reps[class as usize] = state_reps[class as usize].min(raw as u32);
    }
    let tokenizer_states = ManyToOneIdMap::from_original_to_internal_with_representatives(
        state_class,
        sparse_rows.len() as u32,
        state_reps,
    );
    let state_map_ms = state_map_started.elapsed().as_secs_f64() * 1000.0;

    let vocab_map_started = Instant::now();
    let token_classes = token_class.iter().copied().max().map_or(0, |class| class + 1);
    let vocab_tokens = if direct_vocab_map_enabled(&input) {
        direct_vocab_id_map(
            input.vocab.max_token_id(),
            aliases,
            &token_class,
            token_classes,
            preordered_originals,
        )
    } else {
        let mut original_to_internal =
            vec![u32::MAX; input.vocab.max_token_id() as usize + 1];
        for (unique, originals) in aliases.iter().enumerate() {
            for &original in originals {
                original_to_internal[original as usize] = token_class[unique];
            }
        }
        ManyToOneIdMap::from_original_to_internal_allowing_unmapped(
            original_to_internal,
            token_classes,
        )
    };
    let vocab_map_ms = vocab_map_started.elapsed().as_secs_f64() * 1000.0;

    let build_started = Instant::now();
    let num_terminals = input.grammar.num_terminals as usize;
    let mut by_terminal = vec![Vec::<(u32, Vec<u32>)>::new(); num_terminals];
    for (state, row) in sparse_rows.iter().enumerate() {
        let mut cursor = 0usize;
        while cursor < row.len() {
            let terminal = row[cursor].0 as usize;
            debug_assert!(terminal < num_terminals);
            let begin = cursor;
            cursor += 1;
            while cursor < row.len() && row[cursor].0 as usize == terminal {
                cursor += 1;
            }
            let tokens = row[begin..cursor]
                .iter()
                .map(|&(_, token)| token)
                .collect::<Vec<_>>();
            by_terminal[terminal].push((state as u32, tokens));
        }
    }

    let mut dwa = DWA::new(sparse_rows.len() as u32, token_classes.saturating_sub(1));
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
    if std::env::var_os("GLRMASK_PROFILE_L1_IMPLEMENTATIONS").is_some() {
        eprintln!(
            "[glrmask/profile][l1_sparse_finish] partition={} state_map_ms={:.3} vocab_map_ms={:.3} build_ms={:.3}",
            input.partition_label, state_map_ms, vocab_map_ms, build_ms,
        );
    }
    let state_classes = sparse_rows.len();
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

#[cfg(test)]
mod direct_vocab_map_tests {
    use super::direct_vocab_id_map;

    #[test]
    fn preordered_originals_match_sorted_alias_path() {
        let aliases = vec![vec![7, 2], vec![9], vec![8, 1, 5], vec![6, 3, 4]];
        let token_class = vec![1, 0, 1, 2];
        let mut preordered = aliases
            .iter()
            .enumerate()
            .flat_map(|(unique, originals)| {
                originals
                    .iter()
                    .copied()
                    .map(move |original| (original, unique as u32))
            })
            .collect::<Vec<_>>();
        preordered.sort_unstable_by_key(|&(original, _)| original);

        let reference = direct_vocab_id_map(9, &aliases, &token_class, 3, None);
        let preordered_map =
            direct_vocab_id_map(9, &aliases, &token_class, 3, Some(&preordered));
        assert_eq!(reference.original_to_internal, preordered_map.original_to_internal);
        assert_eq!(reference.internal_to_originals, preordered_map.internal_to_originals);
        assert_eq!(
            reference.representative_original_ids,
            preordered_map.representative_original_ids
        );
    }
}
