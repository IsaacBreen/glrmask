//! Compatibility surface for the legacy terminal-DWA fallback path.
//!
//! The canonical terminal-DWA implementation lives under
//! `id_map_and_terminal_dwa/`. This module exists only for code paths that
//! already have an `InternalIdMap` and need the old non-split builder.

use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::{NWA, NWAState as NWAStateType};
use crate::compiler::compile::compute_disallowed_follows;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::grammar::flat::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesComputer,
};
use crate::compiler::stages::equiv_types::InternalIdMap;
use crate::ds::bitset::BitSet;
use crate::ds::vocab_prefix_tree::VocabPrefixTree;
use crate::ds::weight::Weight;

use super::id_map_and_terminal_dwa::classify::classify_terminal_path_lengths;
use super::id_map_and_terminal_dwa::grammar_helpers::compute_always_allowed_follows;
use super::id_map_and_terminal_dwa::l2p::nwa_builder::{
    TerminalNwaBuilder, internal_vocab_entries, seed_root_nodes,
};
use super::id_map_and_terminal_dwa::l2p::postprocess::{
    apply_disallowed_follow_constraints, canonicalize_acyclic_nwa,
    collapse_always_allowed, prune_non_coreachable_states,
};
use super::id_map_and_terminal_dwa::types::{
    TerminalColoring, TerminalPathLength,
};
use super::id_map_and_terminal_dwa::classify::classify_vocab_char_type;

fn partition_internal_vocab(
    entries: Vec<(u32, Vec<u8>)>,
) -> [Vec<(usize, Vec<u8>)>; 7] {
    let mut partitions: [Vec<(usize, Vec<u8>)>; 7] = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    for (token_id, bytes) in entries {
        let idx = classify_vocab_char_type(&bytes) as usize;
        partitions[idx].push((token_id as usize, bytes));
    }
    partitions
}

/// Merges partition NWAs that share template states (start, leaf, root nodes).
fn merge_partition_nwas(
    template_state_count: u32,
    partition_nwas: Vec<NWA>,
) -> NWA {
    if partition_nwas.len() == 1 {
        return partition_nwas.into_iter().next().unwrap();
    }

    let mut offsets = Vec::with_capacity(partition_nwas.len());
    let mut cumulative = 0u32;
    for nwa in &partition_nwas {
        offsets.push(cumulative);
        let extra = nwa.num_states() - template_state_count;
        cumulative += extra;
    }

    let total_states = template_state_count + cumulative;
    let renumber = |state: u32, p: usize| -> u32 {
        if state < template_state_count {
            state
        } else {
            state + offsets[p]
        }
    };

    let mut merged = NWA::from_parts(
        Vec::with_capacity(total_states as usize),
        partition_nwas[0].start_states().to_vec(),
    );

    for s in 0..template_state_count as usize {
        let mut state = NWAStateType::default();
        state.final_weight = partition_nwas[0].states()[s].final_weight.clone();

        let mut eps_map: BTreeMap<u32, Weight> = BTreeMap::new();
        let mut trans_map: BTreeMap<i32, BTreeMap<u32, Weight>> = BTreeMap::new();

        for (p, nwa) in partition_nwas.iter().enumerate() {
            let src = &nwa.states()[s];
            for (&label, targets) in &src.transitions {
                let m = trans_map.entry(label).or_default();
                for &(target, ref weight) in targets {
                    let t = renumber(target, p);
                    m.entry(t)
                        .and_modify(|w| *w = w.union(weight))
                        .or_insert_with(|| weight.clone());
                }
            }
            for &(target, ref weight) in &src.epsilons {
                let t = renumber(target, p);
                eps_map
                    .entry(t)
                    .and_modify(|w| *w = w.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }

        state.epsilons = eps_map.into_iter().collect();
        for (label, targets) in trans_map {
            state
                .transitions
                .insert(label, targets.into_iter().collect());
        }

        merged.states_mut().push(state);
    }

    for (p, nwa) in partition_nwas.iter().enumerate() {
        for s in template_state_count as usize..nwa.num_states() as usize {
            let src = &nwa.states()[s];
            let mut state = NWAStateType::default();
            state.final_weight = src.final_weight.clone();

            for (&label, targets) in &src.transitions {
                let v = state.transitions.entry(label).or_default();
                for &(target, ref weight) in targets {
                    v.push((renumber(target, p), weight.clone()));
                }
            }
            for &(target, ref weight) in &src.epsilons {
                state
                    .epsilons
                    .push((renumber(target, p), weight.clone()));
            }

            merged.states_mut().push(state);
        }
    }

    merged
}

fn build_terminal_dwa_for_existing_id_map_inner(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
) -> DWA {
    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states_mut().push(start_state);

    let mut internal_vocab = internal_vocab_entries(vocab, id_map);

    let empty_disallowed = BTreeMap::new();
    let effective_disallowed = disallowed_follows.unwrap_or(&empty_disallowed);
    let terminal_path_lengths = classify_terminal_path_lengths(
        tokenizer,
        vocab,
        effective_disallowed,
        grammar.num_terminals,
        None,
    );
    let all_l1 = terminal_path_lengths.iter().all(|l| matches!(l, TerminalPathLength::Zero | TerminalPathLength::One));

    let partition_trees: Vec<VocabPrefixTree>;
    if all_l1 {
        partition_trees = Vec::new();
    } else {
        let partitions = partition_internal_vocab(std::mem::take(&mut internal_vocab));
        partition_trees = partitions
            .into_iter()
            .map(|entries| VocabPrefixTree::build_owned(entries))
            .collect();
    }

    let roots_by_tokenizer_state = seed_root_nodes(
        &mut nwa,
        start_state,
        id_map,
    );
    let template_state_count = nwa.num_states();

    let num_tsids = id_map.num_tsids();
    let num_tokenizer_states = tokenizer.num_states() as usize;

    if all_l1 {
        let mut pm = PossibleMatchesComputer::new(tokenizer);
        let mut builder = TerminalNwaBuilder::new(
            tokenizer,
            terminal_coloring.clone(),
            &mut pm,
            &mut nwa,
            num_tsids,
            leaf_state,
            ignore_terminal,
            use_terminal_coloring,
            Some(terminal_path_lengths.clone()),
            None,
            num_tokenizer_states,
        );
        builder.build_l1_fast(&internal_vocab, &roots_by_tokenizer_state, id_map);
        builder.flush_transition_buffer();
        drop(builder);
    } else {
        let template_nwa = &nwa;

        let build_one = |tree: &VocabPrefixTree| -> NWA {
            let mut part_nwa = template_nwa.clone();
            let mut pm = PossibleMatchesComputer::new(tokenizer);
            let mut builder = TerminalNwaBuilder::new(
                tokenizer,
                terminal_coloring.clone(),
                &mut pm,
                &mut part_nwa,
                num_tsids,
                leaf_state,
                ignore_terminal,
                use_terminal_coloring,
                Some(terminal_path_lengths.clone()),
                None,
                num_tokenizer_states,
            );
            builder.build_from_trie(&tree.root, &roots_by_tokenizer_state);
            builder.flush_transition_buffer();
            drop(builder);
            part_nwa
        };

        let (nwa_a, (nwa_b, nwa_c)) = rayon::join(
            || build_one(&partition_trees[0]),
            || rayon::join(
                || build_one(&partition_trees[1]),
                || build_one(&partition_trees[2]),
            ),
        );

        nwa = merge_partition_nwas(
            template_state_count,
            vec![nwa_a, nwa_b, nwa_c],
        );
    }

    let always_allowed_by_label = compute_always_allowed_follows(grammar);
    let _ = collapse_always_allowed(&mut nwa, &always_allowed_by_label, grammar.num_terminals as usize);
    let df = compute_disallowed_follows(grammar);
    apply_disallowed_follow_constraints(&mut nwa, &df, grammar.num_terminals as usize);
    prune_non_coreachable_states(&mut nwa);
    canonicalize_acyclic_nwa(&mut nwa);
    let determinized = crate::automata::weighted::determinize::determinize(&nwa)
        .expect("terminal NWA determinization failed despite acyclic token trie construction");
    minimize(&determinized)
}
