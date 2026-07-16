use std::collections::{BTreeMap, HashSet};

use serde_json::json;

use crate::automata::weighted::terminal_automaton::TerminalAutomaton;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::stages::equiv_types::{InternalIdMap, MappedArtifact};
use crate::compiler::stages::id_map_and_terminal_dwa::types::TerminalDwaFamilies;
use crate::ds::weight::Weight;
use crate::Vocab;

#[derive(Clone)]
struct Walk {
    state: u32,
    labels: Vec<i32>,
    states: Vec<u32>,
    witness: Option<Weight>,
}

#[derive(Clone)]
struct Candidate {
    labels: Vec<i32>,
    states: Vec<u32>,
    witness: Weight,
}

fn intersect(existing: &Option<Weight>, next: &Weight) -> Option<Weight> {
    let value = existing
        .as_ref()
        .map_or_else(|| next.clone(), |weight| weight.intersection(next));
    (!value.is_empty()).then_some(value)
}

fn witness_pair_count(weight: &Weight, id_map: &InternalIdMap) -> usize {
    if weight.is_full() {
        return id_map.num_tsids() as usize * id_map.num_internal_tokens() as usize;
    }
    (0..id_map.num_tsids())
        .map(|tsid| weight.tokens_for_tsid(tsid).iter().count())
        .sum()
}

fn readable_token(bytes: &[u8]) -> bool {
    if bytes.is_empty() || bytes.len() > 48 {
        return false;
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    text.chars().all(|ch| !ch.is_control() || matches!(ch, '\n' | '\r' | '\t'))
}

fn display_bytes(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.chars().flat_map(char::escape_default).collect(),
        Err(_) => bytes.iter().map(|byte| format!("\\x{byte:02x}")).collect(),
    }
}

fn examples(weight: &Weight, id_map: &InternalIdMap, vocab: &Vocab, limit: usize) -> Vec<serde_json::Value> {
    let mut result = Vec::new();
    let mut seen_original = HashSet::new();
    let num_internal = id_map.num_internal_tokens();
    for tsid in 0..id_map.num_tsids() {
        let internal_tokens: Vec<u32> = if weight.is_full() {
            (0..num_internal).collect()
        } else {
            weight.tokens_for_tsid(tsid).iter().collect()
        };
        for internal_token in internal_tokens {
            let Some(originals) = id_map
                .vocab_tokens
                .internal_to_originals
                .get(internal_token as usize)
            else {
                continue;
            };
            for &original_token in originals {
                if !seen_original.insert(original_token) {
                    continue;
                }
                let Some(bytes) = vocab.entries.get(&original_token) else {
                    continue;
                };
                if !readable_token(bytes) {
                    continue;
                }
                result.push(json!({
                    "token_id": original_token,
                    "text": display_bytes(bytes),
                    "bytes_hex": bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                }));
                if result.len() >= limit {
                    return result;
                }
            }
        }
    }
    result
}

fn collect_candidates(automaton: &TerminalAutomaton, max_depth: usize, max_expansions: usize) -> (Vec<Candidate>, usize) {
    let mut stack: Vec<Walk> = automaton
        .start_states()
        .into_iter()
        .map(|state| Walk {
            state,
            labels: Vec::new(),
            states: vec![state],
            witness: None,
        })
        .collect();
    let mut candidates = Vec::new();
    let mut sequences = HashSet::new();
    let mut expansions = 0usize;

    while let Some(walk) = stack.pop() {
        if expansions >= max_expansions {
            break;
        }
        expansions += 1;

        match automaton {
            TerminalAutomaton::Dwa(dwa) => {
                let Some(state) = dwa.states().get(walk.state as usize) else { continue };
                if let Some(final_weight) = &state.final_weight {
                    if let Some(witness) = intersect(&walk.witness, final_weight) {
                        if !walk.labels.is_empty() && sequences.insert(walk.labels.clone()) {
                            candidates.push(Candidate {
                                labels: walk.labels.clone(),
                                states: walk.states.clone(),
                                witness,
                            });
                        }
                    }
                }
                if walk.labels.len() >= max_depth {
                    continue;
                }
                for (&label, (target, weight)) in state.transitions.iter().rev() {
                    let Some(witness) = intersect(&walk.witness, weight) else { continue };
                    let mut labels = walk.labels.clone();
                    labels.push(label);
                    let mut states = walk.states.clone();
                    states.push(*target);
                    stack.push(Walk { state: *target, labels, states, witness: Some(witness) });
                }
            }
            TerminalAutomaton::TokenDeterministicNwa(nwa) => {
                let Some(state) = nwa.states().get(walk.state as usize) else { continue };
                if let Some(final_weight) = &state.final_weight {
                    if let Some(witness) = intersect(&walk.witness, final_weight) {
                        if !walk.labels.is_empty() && sequences.insert(walk.labels.clone()) {
                            candidates.push(Candidate {
                                labels: walk.labels.clone(),
                                states: walk.states.clone(),
                                witness,
                            });
                        }
                    }
                }
                if walk.labels.len() >= max_depth {
                    continue;
                }
                for (&label, branches) in state.transitions.iter().rev() {
                    for (target, weight) in branches.iter().rev() {
                        let Some(witness) = intersect(&walk.witness, weight) else { continue };
                        let mut labels = walk.labels.clone();
                        labels.push(label);
                        let mut states = walk.states.clone();
                        states.push(*target);
                        stack.push(Walk { state: *target, labels, states, witness: Some(witness) });
                    }
                }
                for (target, weight) in state.epsilons.iter().rev() {
                    let Some(witness) = intersect(&walk.witness, weight) else { continue };
                    let mut states = walk.states.clone();
                    states.push(*target);
                    stack.push(Walk {
                        state: *target,
                        labels: walk.labels.clone(),
                        states,
                        witness: Some(witness),
                    });
                }
            }
        }
    }

    (candidates, expansions)
}

fn select_candidates(mut candidates: Vec<Candidate>, limit: usize) -> Vec<Candidate> {
    candidates.sort_by(|left, right| {
        left.labels
            .len()
            .cmp(&right.labels.len())
            .then_with(|| left.labels.cmp(&right.labels))
    });
    let mut selected = Vec::new();
    let mut per_length = BTreeMap::<usize, usize>::new();

    // Two examples at each available path length gives a useful shape sample.
    for candidate in &candidates {
        let count = per_length.entry(candidate.labels.len()).or_default();
        if *count < 2 {
            selected.push(candidate.clone());
            *count += 1;
        }
        if selected.len() >= limit.saturating_sub(4) {
            break;
        }
    }

    // Ensure the longest paths discovered are represented too.
    for candidate in candidates.iter().rev() {
        if !selected.iter().any(|item| item.labels == candidate.labels) {
            selected.push(candidate.clone());
        }
        if selected.len() >= limit {
            break;
        }
    }
    selected
}

fn dump_family(
    family_name: &str,
    family: &MappedArtifact<TerminalAutomaton>,
    grammar: &AnalyzedGrammar,
    vocab: &Vocab,
) {
    let stats = family.artifact().stats();
    let (candidates, expansions) = collect_candidates(family.artifact(), 16, 500_000);
    let selected = select_candidates(candidates.clone(), 18);
    eprintln!(
        "[glrmask/debug][terminal_path_summary] {}",
        json!({
            "family": family_name,
            "states": stats.states,
            "transitions": stats.transitions,
            "transition_pairs": stats.transition_pairs,
            "accepting_terminal_sequences_found": candidates.len(),
            "walk_expansions": expansions,
            "max_sample_depth": 16,
        })
    );
    for candidate in selected {
        let names: Vec<&str> = candidate
            .labels
            .iter()
            .map(|&label| {
                if label < 0 {
                    "<negative-label>"
                } else {
                    grammar.terminal_display_name(label as u32)
                }
            })
            .collect();
        eprintln!(
            "[glrmask/debug][terminal_path] {}",
            json!({
                "family": family_name,
                "length": candidate.labels.len(),
                "terminals": names,
                "terminal_ids": candidate.labels,
                "states": candidate.states,
                "witness_pair_count": witness_pair_count(&candidate.witness, family.id_map()),
                "example_tokens": examples(&candidate.witness, family.id_map(), vocab, 6),
            })
        );
    }
}

pub(crate) fn maybe_dump_terminal_paths(
    families: &TerminalDwaFamilies,
    grammar: &AnalyzedGrammar,
    vocab: &Vocab,
) {
    if std::env::var_os("GLRMASK_DEBUG_SAMPLE_TERMINAL_PATHS").is_none() {
        return;
    }
    if let Some(family) = &families.l1 {
        dump_family("l1", family, grammar, vocab);
    }
    if let Some(family) = &families.l2p {
        dump_family("l2p", family, grammar, vocab);
    }
    if let Some(family) = &families.special {
        dump_family("special", family, grammar, vocab);
    }
}

pub(crate) fn maybe_dump_final_terminal_dwa(
    families: &TerminalDwaFamilies,
    grammar: &AnalyzedGrammar,
    vocab: &Vocab,
    num_tokenizer_states: usize,
) {
    if std::env::var_os("GLRMASK_DEBUG_SAMPLE_FINAL_TERMINAL_DWA").is_none() {
        return;
    }
    let merged = crate::compiler::stages::id_map_and_terminal_dwa::merge::debug_merge_terminal_families(
        families,
        num_tokenizer_states,
        vocab.max_token_id(),
    );
    let wrapped = MappedArtifact::new(TerminalAutomaton::Dwa(merged.artifact().clone()), merged.id_map().clone());
    dump_family("final", &wrapped, grammar, vocab);
    if std::env::var_os("GLRMASK_DEBUG_EXIT_AFTER_FINAL_TERMINAL_DWA").is_some() {
        std::process::exit(0);
    }
}
