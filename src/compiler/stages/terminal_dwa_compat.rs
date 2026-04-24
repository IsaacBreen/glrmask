//! Compatibility surface for the legacy terminal-DWA fallback path.
//!
//! The canonical terminal-DWA implementation lives under
//! `id_map_and_terminal_dwa/`. This module exists only for code paths that
//! already have an `InternalIdMap` and need the old non-split builder.

use std::collections::BTreeMap;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::{minimize, minimize_from_env};
use crate::automata::weighted::nwa::{NWA, NWAState as NWAStateType};
use crate::compiler::compile::compute_disallowed_follows;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::grammar::flat::TerminalID;
use crate::compiler::possible_matches::{
    PossibleMatchesByState, PossibleMatchesComputer, collect_possible_matches_by_internal_tsid,
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
    TerminalColoring, TerminalDwaBuildProfile, TerminalPathLength,
    debug_profile_enabled, terminal_dwa_profile_enabled,
};
use super::id_map_and_terminal_dwa::classify::classify_vocab_char_type;

fn partition_internal_vocab(
    entries: Vec<(u32, Vec<u8>)>,
) -> [Vec<(usize, Vec<u8>)>; 4] {
    let mut partitions: [Vec<(usize, Vec<u8>)>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
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

    let mut merged = NWA {
        states: Vec::with_capacity(total_states as usize),
        start_states: partition_nwas[0].start_states.clone(),
    };

    for s in 0..template_state_count as usize {
        let mut state = NWAStateType::default();
        state.final_weight = partition_nwas[0].states[s].final_weight.clone();

        let mut eps_map: BTreeMap<u32, Weight> = BTreeMap::new();
        let mut trans_map: BTreeMap<i32, BTreeMap<u32, Weight>> = BTreeMap::new();

        for (p, nwa) in partition_nwas.iter().enumerate() {
            let src = &nwa.states[s];
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

        merged.states.push(state);
    }

    for (p, nwa) in partition_nwas.iter().enumerate() {
        for s in template_state_count as usize..nwa.num_states() as usize {
            let src = &nwa.states[s];
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

            merged.states.push(state);
        }
    }

    merged
}

pub(crate) fn build_terminal_dwa_for_existing_id_map(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    let terminal_coloring = TerminalColoring::identity(grammar.num_terminals as usize);
    build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring(
        grammar,
        tokenizer,
        vocab,
        id_map,
        &terminal_coloring,
        false,
        ignore_terminal,
        None,
    )
    .0
}

pub(crate) fn build_terminal_dwa_for_existing_id_map_with_possible_matches_and_coloring(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    terminal_coloring: &TerminalColoring,
    use_terminal_coloring: bool,
    ignore_terminal: Option<TerminalID>,
    disallowed_follows: Option<&BTreeMap<u32, BitSet>>,
) -> (DWA, PossibleMatchesByState) {
    let debug_profile = debug_profile_enabled();
    let total_started_at = std::time::Instant::now();
    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_internal_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let setup_started_at = std::time::Instant::now();
    let mut internal_vocab = internal_vocab_entries(vocab, id_map);
    let internal_vocab_len = internal_vocab.len();

    let full_tree = VocabPrefixTree::build_owned(
        internal_vocab
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect(),
    );

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

    if debug_profile {
        let n0 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::Zero).count();
        let n1 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::One).count();
        let n2 = terminal_path_lengths.iter().filter(|l| **l == TerminalPathLength::TwoPlus).count();
        eprintln!(
            "[glrmask/debug][terminal_dwa] non_partition_build all_l1={} internal_vocab_len={} l0={} l1={} l2p={}",
            all_l1, internal_vocab.len(), n0, n1, n2,
        );
    }

    let partition_sizes: [usize; 3];
    let partition_trees: Vec<VocabPrefixTree>;
    if all_l1 {
        partition_sizes = [0, 0, 0];
        partition_trees = Vec::new();
    } else {
        let partitions = partition_internal_vocab(std::mem::take(&mut internal_vocab));
        partition_sizes = [
            partitions[0].len(),
            partitions[1].len(),
            partitions[2].len(),
        ];
        partition_trees = partitions
            .into_iter()
            .map(|entries| VocabPrefixTree::build_owned(entries))
            .collect();
    }

    let setup_ms = setup_started_at.elapsed().as_secs_f64() * 1000.0;
    let profile_enabled = terminal_dwa_profile_enabled();
    let mut possible_matches = PossibleMatchesComputer::new(tokenizer);

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] start grammar_rules={} grammar_terminals={} grammar_nonterminals={} tokenizer_states={} internal_tokenizer_states={} vocab_entries={} partitions=[{},{},{}] setup_ms={:.3}",
            grammar.rules.len(),
            grammar.num_terminals,
            grammar.num_nonterminals,
            tokenizer.num_states(),
            id_map.num_tsids(),
            internal_vocab_len,
            partition_sizes[0],
            partition_sizes[1],
            partition_sizes[2],
            setup_ms,
        );
    }

    let possible_matches_started_at = std::time::Instant::now();
    let possible_matches_by_state = collect_possible_matches_by_internal_tsid(
        tokenizer,
        &full_tree.root,
        &mut possible_matches,
        &id_map.tokenizer_states,
    );
    let possible_matches_ms = possible_matches_started_at.elapsed().as_secs_f64() * 1000.0;
    let possible_matches_profile = possible_matches.profile();

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=possible_matches states={} cache_entries={} reachable_cache_entries={} ms={:.3}",
            possible_matches_by_state.len(),
            possible_matches_profile.cache_entries,
            possible_matches_profile.reachable_cache_entries,
            possible_matches_ms,
        );
    }

    let seed_started_at = std::time::Instant::now();
    let roots_by_tokenizer_state = seed_root_nodes(
        &mut nwa,
        start_state,
        id_map,
    );
    let seed_ms = seed_started_at.elapsed().as_secs_f64() * 1000.0;
    let template_state_count = nwa.num_states();

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=seed roots={} template_states={} ms={:.3}",
            roots_by_tokenizer_state.entries.len(),
            template_state_count,
            seed_ms,
        );
    }

    let build_trie_started_at = std::time::Instant::now();

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=nwa_build all_l1={} internal_vocab_len={}",
            all_l1,
            if all_l1 { internal_vocab.len() } else { 0 },
        );
    }

    let num_tsids = id_map.num_tsids();
    let num_tokenizer_states = tokenizer.num_states() as usize;

    let profile = if all_l1 {
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
        let prof = builder.profile;
        drop(builder);
        prof
    } else {
        let template_nwa = &nwa;

        let build_one = |tree: &VocabPrefixTree| -> (NWA, TerminalDwaBuildProfile) {
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
            let prof = builder.profile;
            drop(builder);
            (part_nwa, prof)
        };

        let ((nwa_a, prof_a), ((nwa_b, prof_b), (nwa_c, prof_c))) = rayon::join(
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
        TerminalDwaBuildProfile {
            future_terminal_additions: prof_a.future_terminal_additions
                + prof_b.future_terminal_additions
                + prof_c.future_terminal_additions,
            match_transition_additions: prof_a.match_transition_additions
                + prof_b.match_transition_additions
                + prof_c.match_transition_additions,
        }
    };
    let build_trie_ms = build_trie_started_at.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] stage=build_trie nwa_states={} nwa_transitions={} ms={:.3}",
            nwa.num_states(),
            nwa.num_transitions(),
            build_trie_ms,
        );
    }

    let always_allowed_started_at = std::time::Instant::now();
    let always_allowed_by_label = compute_always_allowed_follows(grammar);
    let always_allowed_ms = always_allowed_started_at.elapsed().as_secs_f64() * 1000.0;

    let collapse_started_at = std::time::Instant::now();
    let disable_collapse_always_allowed = std::env::var_os("GLRMASK_DISABLE_TERMINAL_DWA_COLLAPSE_ALWAYS_ALLOWED").is_some();
    let _ = if disable_collapse_always_allowed {
        false
    } else {
        collapse_always_allowed(&mut nwa, &always_allowed_by_label, grammar.num_terminals as usize)
    };
    let collapse_ms = collapse_started_at.elapsed().as_secs_f64() * 1000.0;

    let disallowed_started_at = std::time::Instant::now();
    let disable_disallowed_follows = std::env::var_os("GLRMASK_DISABLE_TERMINAL_DWA_DISALLOWED_FOLLOWS").is_some();
    if !disable_disallowed_follows {
        let df = compute_disallowed_follows(grammar);
        apply_disallowed_follow_constraints(&mut nwa, &df, grammar.num_terminals as usize);
    }
    let disallowed_ms = disallowed_started_at.elapsed().as_secs_f64() * 1000.0;

    let coreachable_prune_started_at = std::time::Instant::now();
    prune_non_coreachable_states(&mut nwa);
    let coreachable_prune_ms = coreachable_prune_started_at.elapsed().as_secs_f64() * 1000.0;

    let canonicalize_started_at = std::time::Instant::now();
    canonicalize_acyclic_nwa(&mut nwa);
    let canonicalize_ms = canonicalize_started_at.elapsed().as_secs_f64() * 1000.0;

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] after_cleanup nwa_states={} nwa_transitions={}",
            nwa.num_states(),
            nwa.num_transitions(),
        );
    }

    let nwa_states = nwa.num_states();
    let nwa_transitions = nwa.num_transitions();

    let determinize_started_at = std::time::Instant::now();
    let determinized = crate::automata::weighted::determinize::determinize(&nwa)
        .expect("terminal NWA determinization failed despite acyclic token trie construction");
    let determinize_ms = determinize_started_at.elapsed().as_secs_f64() * 1000.0;
    let determinized_states = determinized.num_states();
    let determinized_transitions = determinized.num_transitions();

    let minimize_started_at = std::time::Instant::now();
    let dwa = minimize_from_env(&determinized, "GLRMASK_MINIMIZE_MERGE", minimize);
    let minimize_ms = minimize_started_at.elapsed().as_secs_f64() * 1000.0;

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] colors={} future_terminal_additions={} match_transition_additions={}",
            terminal_coloring.num_colors,
            profile.future_terminal_additions,
            profile.match_transition_additions,
        );
    }

    if debug_profile {
        eprintln!(
            "[glrmask/debug][terminal_dwa] tokenizer_states={} internal_tokenizer_states={} vocab_entries={} roots={} possible_matches_states={} possible_matches_cache_entries={} reachable_cache_entries={} nwa_states={} nwa_transitions={} determinized_states={} determinized_transitions={} minimized_states={}",
            tokenizer.num_states(),
            id_map.num_tsids(),
            internal_vocab_len,
            roots_by_tokenizer_state.entries.len(),
            possible_matches_by_state.len(),
            possible_matches_profile.cache_entries,
            possible_matches_profile.reachable_cache_entries,
            nwa_states,
            nwa_transitions,
            determinized_states,
            determinized_transitions,
            dwa.num_states(),
        );
        eprintln!(
            "[glrmask/debug][terminal_dwa] setup_ms={:.3} seed_ms={:.3} build_trie_ms={:.3} possible_matches_ms={:.3} always_allowed_ms={:.3} collapse_ms={:.3} disallowed_ms={:.3} coreachable_prune_ms={:.3} canonicalize_ms={:.3} determinize_ms={:.3} minimize_ms={:.3} total_ms={:.3}",
            setup_ms,
            seed_ms,
            build_trie_ms,
            possible_matches_ms,
            always_allowed_ms,
            collapse_ms,
            disallowed_ms,
            coreachable_prune_ms,
            canonicalize_ms,
            determinize_ms,
            minimize_ms,
            total_started_at.elapsed().as_secs_f64() * 1000.0,
        );
        eprintln!(
            "[glrmask/debug][terminal_dwa] possible_matches cache_hits={} cache_misses={} reachable_hits={} reachable_misses={} child_segments={} byte_steps={} blocked_segments={} recursive_descents={} terminal_insertions={}",
            possible_matches_profile.cache_hits,
            possible_matches_profile.cache_misses,
            possible_matches_profile.reachable_cache_hits,
            possible_matches_profile.reachable_cache_misses,
            possible_matches_profile.child_segments_visited,
            possible_matches_profile.byte_steps,
            possible_matches_profile.blocked_segments,
            possible_matches_profile.recursive_descents,
            possible_matches_profile.terminal_insertions,
        );
    }

    if std::env::var("GLRMASK_DEBUG_DWA_DUMP").map_or(false, |v| v == "1") {
        emit_terminal_dwa_token_map(&dwa, vocab, id_map);
        emit_terminal_dwa_debug_dump(&dwa, id_map);
    }

    (dwa, possible_matches_by_state)
}

fn emit_terminal_dwa_token_map(dwa: &DWA, vocab: &Vocab, id_map: &InternalIdMap) {
    let internal_vocab = internal_vocab_entries(vocab, id_map);
    let internal_bytes: std::collections::BTreeMap<u32, &[u8]> =
        internal_vocab.iter().map(|(id, bytes)| (*id, bytes.as_slice())).collect();
    let mut referenced_tokens = std::collections::BTreeSet::new();
    for state in &dwa.states {
        for (_, (_, weight)) in &state.transitions {
            for tid in weight.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
        if let Some(fw) = &state.final_weight {
            for tid in fw.token_union().iter() {
                referenced_tokens.insert(tid);
            }
        }
    }
    for tid in &referenced_tokens {
        if let Some(bytes) = internal_bytes.get(tid) {
            let originals = id_map.vocab_tokens.internal_to_originals.get(*tid as usize)
                .map(|v| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
                .unwrap_or_else(|| "?".into());
            let representative = id_map
                .vocab_tokens
                .representative_original_id_for_internal(*tid)
                .map_or_else(|| "?".to_string(), |id| id.to_string());
            eprintln!(
                "[glrmask/debug][terminal_dwa][token_map] repr={} internal={} originals=[{}] bytes={:?}",
                representative, tid, originals, String::from_utf8_lossy(bytes)
            );
        }
    }
}

fn format_debug_id_ranges(ids: impl IntoIterator<Item = u32>, wrap_in_braces: bool) -> String {
    let mut ids: Vec<u32> = ids.into_iter().collect();
    ids.sort_unstable();
    ids.dedup();

    let body = if ids.is_empty() {
        String::new()
    } else {
        let mut parts = Vec::new();
        let mut start = ids[0];
        let mut end = ids[0];
        for &id in &ids[1..] {
            if end.checked_add(1) == Some(id) {
                end = id;
                continue;
            }
            if start == end {
                parts.push(start.to_string());
            } else {
                parts.push(format!("{}..={}", start, end));
            }
            start = id;
            end = id;
        }
        if start == end {
            parts.push(start.to_string());
        } else {
            parts.push(format!("{}..={}", start, end));
        }
        parts.join(",")
    };

    if wrap_in_braces {
        format!("{{{body}}}")
    } else {
        body
    }
}

fn format_debug_weight(weight: &crate::ds::weight::Weight, id_map: &InternalIdMap) -> String {
    if weight.is_empty() || weight.is_full() {
        return weight.to_string();
    }

    weight
        .compact_entries()
        .unwrap_or_default()
        .into_iter()
        .map(|(start, end, tokens)| {
            let tsids = (start..=end).filter_map(|internal| {
                id_map
                    .tokenizer_states
                    .representative_original_id_for_internal(internal)
            });
            let token_ids = tokens.iter().filter_map(|internal| {
                id_map
                    .vocab_tokens
                    .representative_original_id_for_internal(internal)
            });
            format!(
                "{}→{}",
                format_debug_id_ranges(tsids, false),
                format_debug_id_ranges(token_ids, true)
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn emit_terminal_dwa_debug_dump(dwa: &DWA, id_map: &InternalIdMap) {
    let num_states = dwa.num_states() as usize;
    let start_state = dwa.start_state as usize;
    let mut incoming_counts = vec![0usize; num_states];
    let mut outgoing_counts = vec![0usize; num_states];
    let mut final_states = 0usize;
    let mut self_loops = 0usize;
    let mut transitions_to_start = 0usize;
    let mut transitions_from_start = 0usize;
    let mut transitions_from_start_to_start = 0usize;

    for (state_id, state) in dwa.states.iter().enumerate() {
        outgoing_counts[state_id] = state.transitions.len();
        for (_, (target, _)) in &state.transitions {
            incoming_counts[*target as usize] += 1;
            if *target as usize == start_state {
                transitions_to_start += 1;
            }
            if state_id == start_state {
                transitions_from_start += 1;
            }
            if state_id == start_state && *target as usize == start_state {
                transitions_from_start_to_start += 1;
            }
            if *target as usize == state_id {
                self_loops += 1;
            }
        }
        if state.final_weight.is_some() {
            final_states += 1;
        }
    }

    eprintln!(
        "[glrmask/debug][terminal_dwa][dump] states={} final_states={} self_loops={} to_start={} from_start={} from_start_to_start={}",
        num_states, final_states, self_loops, transitions_to_start, transitions_from_start, transitions_from_start_to_start,
    );

    for (state_id, state) in dwa.states.iter().enumerate() {
        let incoming = incoming_counts[state_id];
        let outgoing = outgoing_counts[state_id];
        let to_start = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == start_state)
            .count();
        let self_loop_count = state
            .transitions
            .values()
            .filter(|(to, _)| *to as usize == state_id)
            .count();
        let final_weight = state.final_weight.as_ref().map_or_else(
            || "none".to_string(),
            |weight| {
                let debug_weight = format_debug_weight(weight, id_map);
                let internal_weight = weight.to_string();
                if debug_weight == internal_weight {
                    debug_weight
                } else {
                    format!("{debug_weight} [internal {internal_weight}]")
                }
            },
        );
        let start_mark = if state_id == start_state {
            " [START]"
        } else {
            ""
        };

        eprintln!(
            "[glrmask/debug][terminal_dwa][state] id={}{} incoming={} outgoing={} to_start={} self_loops={} final={}",
            state_id,
            start_mark,
            incoming,
            outgoing,
            to_start,
            self_loop_count,
            final_weight,
        );

        for (label, (target, weight)) in &state.transitions {
            eprintln!("    {label} -> State {target}");
            let debug_weight = format_debug_weight(weight, id_map);
            let internal_weight = weight.to_string();
            if debug_weight == internal_weight {
                eprintln!("      weight: {debug_weight}");
            } else {
                eprintln!("      weight: {debug_weight} [internal {internal_weight}]");
            }
        }
    }
}
