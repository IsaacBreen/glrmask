#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::determinize::determinize;
use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::minimize::minimize;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::grammar::model::Symbol;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::ds::weight::Weight;

type SuffixKey = (u32, u32, usize);

#[derive(Default)]
struct TokenSuffixMemo {
    built: HashSet<SuffixKey>,
    states: HashMap<SuffixKey, u32>,
}

fn compute_ever_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
    let mut ever_allowed = vec![BTreeSet::new(); grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }
            ever_allowed[*terminal as usize]
                .extend(occurrence_follow_set(grammar, rule.lhs, &rule.rhs, index));
        }
    }

    ever_allowed
        .into_iter()
        .map(|set| set.into_iter().collect())
        .collect()
}

fn occurrence_follow_set(
    grammar: &AnalyzedGrammar,
    lhs: u32,
    rhs: &[Symbol],
    index: usize,
) -> BTreeSet<TerminalID> {
    let mut follows = BTreeSet::new();
    let mut suffix_nullable = true;

    for symbol in &rhs[index + 1..] {
        match symbol {
            Symbol::Terminal(terminal) => {
                if *terminal < grammar.num_terminals {
                    follows.insert(*terminal);
                }
                suffix_nullable = false;
                break;
            }
            Symbol::Nonterminal(nonterminal) => {
                if let Some(first) = grammar.first.get(*nonterminal as usize) {
                    follows.extend(first.iter().copied().filter(|terminal| *terminal != EOF));
                }
                if !grammar.nullable.contains(nonterminal) {
                    suffix_nullable = false;
                    break;
                }
            }
        }
    }

    if suffix_nullable {
        if let Some(follow) = grammar.follow.get(lhs as usize) {
            follows.extend(follow.iter().copied().filter(|terminal| *terminal != EOF));
        }
    }

    follows
}

fn prune_unreachable_states(nwa: &mut NWA) -> bool {
    if nwa.states.is_empty() {
        return false;
    }

    let mut reachable = vec![false; nwa.states.len()];
    let mut queue = VecDeque::new();

    for &start in &nwa.start_states {
        if let Some(flag) = reachable.get_mut(start as usize) {
            if !*flag {
                *flag = true;
                queue.push_back(start);
            }
        }
    }

    while let Some(state_id) = queue.pop_front() {
        let state = &nwa.states[state_id as usize];
        for (target, _) in &state.epsilons {
            if let Some(flag) = reachable.get_mut(*target as usize) {
                if !*flag {
                    *flag = true;
                    queue.push_back(*target);
                }
            }
        }
        for (target, _) in state.transitions.values().flatten() {
            if let Some(flag) = reachable.get_mut(*target as usize) {
                if !*flag {
                    *flag = true;
                    queue.push_back(*target);
                }
            }
        }
    }

    if reachable.iter().all(|flag| *flag) {
        return false;
    }

    let mut remap = vec![u32::MAX; nwa.states.len()];
    let mut new_states = Vec::with_capacity(reachable.iter().filter(|flag| **flag).count());

    for (old_id, state) in nwa.states.iter().enumerate() {
        if reachable[old_id] {
            remap[old_id] = new_states.len() as u32;
            new_states.push(state.clone());
        }
    }

    for state in &mut new_states {
        state.epsilons.retain(|(target, _)| reachable[*target as usize]);
        for (target, _) in &mut state.epsilons {
            *target = remap[*target as usize];
        }

        for targets in state.transitions.values_mut() {
            targets.retain(|(target, _)| reachable[*target as usize]);
            for (target, _) in targets.iter_mut() {
                *target = remap[*target as usize];
            }
        }
        state.transitions.retain(|_, targets| !targets.is_empty());
    }

    nwa.start_states = nwa
        .start_states
        .iter()
        .copied()
        .filter(|state_id| reachable[*state_id as usize])
        .map(|state_id| remap[state_id as usize])
        .collect();
    nwa.states = new_states;
    true
}

fn topological_order(nwa: &NWA) -> Vec<usize> {
    let mut in_degree = vec![0u32; nwa.states.len()];
    for state in &nwa.states {
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] += 1;
            }
        }
    }

    let mut queue = VecDeque::new();
    for (state_id, degree) in in_degree.iter().enumerate() {
        if *degree == 0 {
            queue.push_back(state_id);
        }
    }

    let mut order = Vec::with_capacity(nwa.states.len());
    while let Some(state_id) = queue.pop_front() {
        order.push(state_id);
        let state = &nwa.states[state_id];
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] -= 1;
            if in_degree[*dst as usize] == 0 {
                queue.push_back(*dst as usize);
            }
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] -= 1;
                if in_degree[*dst as usize] == 0 {
                    queue.push_back(*dst as usize);
                }
            }
        }
    }

    order
}

fn intersect_or_insert(entry: &mut Option<HashSet<TerminalID>>, next: &HashSet<TerminalID>) {
    match entry {
        None => *entry = Some(next.clone()),
        Some(existing) => existing.retain(|terminal| next.contains(terminal)),
    }
}

fn prune_disallowed_follows(nwa: &mut NWA, grammar: &AnalyzedGrammar) -> bool {
    let ever_allowed_by_label = compute_ever_allowed_follows(grammar);
    let terminals_count = grammar.num_terminals as usize;

    if ever_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let all_terminals: HashSet<TerminalID> = (0..grammar.num_terminals).collect();
    let disallowed_after: Vec<HashSet<TerminalID>> = (0..terminals_count)
        .map(|label| {
            let mut disallowed = all_terminals.clone();
            for follow in ever_allowed_by_label.get(label).into_iter().flatten() {
                disallowed.remove(follow);
            }
            disallowed
        })
        .collect();

    let topo_order = topological_order(nwa);

    let mut disallowed_intersected: Vec<Option<HashSet<TerminalID>>> = vec![None; nwa.states.len()];
    for &start in &nwa.start_states {
        disallowed_intersected[start as usize] = Some(HashSet::new());
    }

    for &sid in &topo_order {
        let src_disallowed = match &disallowed_intersected[sid] {
            Some(d) => d.clone(),
            None => HashSet::new(),
        };

        let epsilon_targets: Vec<u32> =
            nwa.states[sid].epsilons.iter().map(|(dst, _)| *dst).collect();
        for dst in epsilon_targets {
            intersect_or_insert(&mut disallowed_intersected[dst as usize], &src_disallowed);
        }

        let labeled_targets: Vec<(i32, Vec<u32>)> = nwa.states[sid]
            .transitions
            .iter()
            .map(|(&label, targets)| (label, targets.iter().map(|(dst, _)| *dst).collect()))
            .collect();
        for (label, targets) in labeled_targets {
            let label_dis = if label >= 0 && (label as usize) < disallowed_after.len() {
                &disallowed_after[label as usize]
            } else {
                continue;
            };
            for dst in targets {
                intersect_or_insert(&mut disallowed_intersected[dst as usize], label_dis);
            }
        }
    }

    let mut changed = false;
    for (state_id, state) in nwa.states.iter_mut().enumerate() {
        let disallowed = match &disallowed_intersected[state_id] {
            Some(d) if !d.is_empty() => d,
            _ => continue,
        };

        let before = state.transitions.len();
        state.transitions.retain(|label, _| {
            *label < 0 || !disallowed.contains(&(*label as TerminalID))
        });
        changed |= state.transitions.len() != before;
    }

    if prune_unreachable_states(nwa) {
        changed = true;
    }

    changed
}

fn add_weighted_transition(nwa: &mut NWA, from: u32, label: i32, to: u32, weight: Weight) {
    let Some(state) = nwa.states.get_mut(from as usize) else {
        return;
    };
    let targets = state.transitions.entry(label).or_default();
    if let Some((_, existing_weight)) = targets.iter_mut().find(|(target, _)| *target == to) {
        *existing_weight = existing_weight.union(&weight);
    } else {
        targets.push((to, weight));
    }
}

fn build_token_suffix_paths(
    tokenizer: &Tokenizer,
    nwa: &mut NWA,
    suffix_memo: &mut TokenSuffixMemo,
    source_node: u32,
    current_state: u32,
    origin_state: u32,
    token_id: u32,
    token_bytes: &[u8],
    offset: usize,
    leaf_state: u32,
    grammar: &AnalyzedGrammar,
    internal_tsid: u32,
) {
    let exec = tokenizer.execute_from_state(&token_bytes[offset..], current_state);
    let token_weight = Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([token_id..=token_id]),
    );

    if let Some(end_state) = exec.end_state {
        for terminal_id in tokenizer.tokens_accessible_from_state(end_state) {
            if terminal_id >= grammar.num_terminals {
                continue;
            }
            add_weighted_transition(
                nwa,
                source_node,
                terminal_id as i32,
                leaf_state,
                token_weight.clone(),
            );
        }
    }

    for matched in exec.matches {
        if matched.id >= grammar.num_terminals {
            continue;
        }

        let next_offset = offset + matched.width;
        let destination = if next_offset >= token_bytes.len() {
            leaf_state
        } else {
            let key = (origin_state, token_id, next_offset);
            let continuation_state = *suffix_memo.states.entry(key).or_insert_with(|| nwa.add_state());
            if suffix_memo.built.insert(key) {
                build_token_suffix_paths(
                    tokenizer,
                    nwa,
                    suffix_memo,
                    continuation_state,
                    tokenizer.initial_state(),
                    origin_state,
                    token_id,
                    token_bytes,
                    next_offset,
                    leaf_state,
                    grammar,
                    internal_tsid,
                );
            }
            continuation_state
        };

        add_weighted_transition(
            nwa,
            source_node,
            matched.id as i32,
            destination,
            token_weight.clone(),
        );
    }
}

pub(crate) fn build_terminal_dwa(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
) -> DWA {
    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let mut suffix_memo = TokenSuffixMemo::default();

    for internal_tsid in 0..id_map.num_tsids() {
        let root = nwa.add_state();
        nwa.start_states.push(root);

        for original_state in &id_map.tokenizer_states.internal_to_originals[internal_tsid as usize] {
            for (token_id, token_bytes) in &vocab.entries {
                build_token_suffix_paths(
                    tokenizer,
                    &mut nwa,
                    &mut suffix_memo,
                    root,
                    *original_state,
                    *original_state,
                    *token_id,
                    token_bytes,
                    0,
                    leaf_state,
                    grammar,
                    internal_tsid,
                );
            }
        }
    }

    let _ = prune_disallowed_follows(&mut nwa, grammar);
    let dwa = minimize(
        &determinize(&nwa)
            .expect("terminal NWA determinization failed despite acyclic token trie construction"),
    );
    debug_assert!(
        dwa.states
            .get(dwa.start_state as usize)
            .and_then(|state| state.final_weight.as_ref())
            .is_none(),
        "terminal-DWA start state unexpectedly has a final weight"
    );
    dwa
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::tests::simple_ab_grammar;

    #[test]
    fn test_terminal_dwa_accepts_single_and_multi_terminal_paths() {
        let grammar = simple_ab_grammar();
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"ab".to_vec()), (2, b"b".to_vec())],
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab);

        let terminal_dwa = build_terminal_dwa(&glr_grammar, &tokenizer, &vocab, &id_map);

        assert!(
            !terminal_dwa.eval_word(&[0]).is_empty(),
            "terminal DWA should accept the single-terminal path 'a'"
        );
        assert!(
            !terminal_dwa.eval_word(&[0, 1]).is_empty(),
            "terminal DWA should accept the multi-terminal path 'ab'"
        );
    }
}
