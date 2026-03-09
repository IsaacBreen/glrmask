#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

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
use crate::compiler::possible_matches::PossibleMatchesComputer;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;

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

fn token_weight(internal_tsid: u32, token_id: u32) -> Weight {
    Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([token_id..=token_id]),
    )
}

fn token_set_weight(internal_tsid: u32, token_ids: &RangeSetBlaze<usize>) -> Weight {
    let mut mapped = RangeSetBlaze::new();
    for token_id in token_ids.iter() {
        mapped.insert(token_id as u32);
    }
    Weight::from_token_set_for_tsid(internal_tsid, mapped)
}

fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {
    Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([0..=max_token_id]),
    )
}

#[derive(Clone)]
struct SourceAssoc {
    tsid: u32,
    nodes: Vec<u32>,
}

struct TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    tokenizer: &'tok Tokenizer,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut NWA,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    transition_buffer: BTreeMap<(u32, i32, u32), Weight>,
    epsilon_buffer: BTreeMap<(u32, u32), Weight>,
}

impl<'tok, 'pm, 'nwa> TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    fn add_match_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        target: u32,
        weight: &Weight,
    ) {
        for &source in sources {
            if self.ignore_terminal == Some(label) {
                self.epsilon_buffer
                    .entry((source, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            } else {
                self.transition_buffer
                    .entry((source, label as i32, target))
                    .and_modify(|existing| *existing = existing.union(weight))
                    .or_insert_with(|| weight.clone());
            }
        }
    }

    fn flush_transition_buffer(&mut self) {
        for ((from, target), weight) in std::mem::take(&mut self.epsilon_buffer) {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered epsilon source state must exist");
            state.epsilons.push((target, weight));
        }
        for ((from, label, target), weight) in std::mem::take(&mut self.transition_buffer) {
            let state = self
                .nwa
                .states
                .get_mut(from as usize)
                .expect("buffered transition source state must exist");
            state.transitions.entry(label).or_default().push((target, weight));
        }
    }

    fn build_from_trie(
        &mut self,
        node: &VocabPrefixTreeNode,
        assoc_by_state: &BTreeMap<u32, SourceAssoc>,
    ) {
        for (segment_bytes, child_node) in node.iter_children() {
            let child_token_id = child_node.token_id() as u32;

            let mut next_level_assoc = BTreeMap::<u32, SourceAssoc>::new();
            let mut pending = BTreeMap::<usize, BTreeMap<u32, SourceAssoc>>::new();
            pending.insert(0, assoc_by_state.clone());

            while let Some((pos, states_at_pos)) = pending.pop_first() {
                if pos == segment_bytes.len() {
                    for (tokenizer_state, assoc) in states_at_pos {
                        merge_assoc(&mut next_level_assoc, tokenizer_state, assoc.tsid, &assoc.nodes);
                    }
                    continue;
                }

                for (tokenizer_state, source_assoc) in states_at_pos {
                    let source_nodes = source_assoc.nodes;
                    let tsid = source_assoc.tsid;
                    let child_token_weight = token_weight(tsid, child_token_id);
                    let exec = self
                        .tokenizer
                        .execute_from_state(&segment_bytes[pos..], tokenizer_state);
                    let exec_end_state = exec.end_state;
                    let mut possible_matches_at_end = None;

                    if let Some(end_state) = exec_end_state {
                        if child_node.has_token() {
                            for terminal_id in self.tokenizer.tokens_accessible_from_state(end_state) {
                                self.add_match_from_sources(
                                    &source_nodes,
                                    terminal_id,
                                    self.leaf_state,
                                    &child_token_weight,
                                );
                            }
                        }

                        merge_assoc(&mut next_level_assoc, end_state, tsid, &source_nodes);
                    }

                    for matched in exec.matches {
                        let next_pos = pos + matched.width;

                        if next_pos == segment_bytes.len() && child_node.has_token() {
                            self.add_match_from_sources(
                                &source_nodes,
                                matched.id,
                                self.leaf_state,
                                &child_token_weight,
                            );
                        }

                        let mut continuation_tokens = if next_pos == segment_bytes.len()
                            && child_node.has_token()
                        {
                            let mut remaining = child_node.reachable_token_ids().clone();
                            remaining.remove(child_token_id as usize);
                            if let Some(end_state) = exec_end_state {
                                let matches_at_end = possible_matches_at_end.get_or_insert_with(|| {
                                    self.possible_matches
                                        .possible_matches_for_node(child_node, end_state)
                                });
                                if let Some(pm) = matches_at_end.get(&matched.id) {
                                    subtract_possible_matches(&mut remaining, pm);
                                }
                            }
                            remaining
                        } else {
                            child_node.reachable_token_ids().clone()
                        };

                        if continuation_tokens.is_empty() {
                            continue;
                        }

                        let continuation_weight = token_set_weight(tsid, &continuation_tokens);
                        if continuation_weight.is_empty() {
                            continue;
                        }

                        let continuation_assoc = pending.entry(next_pos).or_default();
                        let destination = continuation_state(
                            continuation_assoc,
                            self.tokenizer.initial_state(),
                            tsid,
                            self.nwa,
                        );

                        self.add_match_from_sources(
                            &source_nodes,
                            matched.id,
                            destination,
                            &continuation_weight,
                        );
                    }
                }
            }

            if !next_level_assoc.is_empty() {
                self.build_from_trie(child_node, &next_level_assoc);
            }
        }
    }
}

fn subtract_possible_matches(
    continuation_tokens: &mut RangeSetBlaze<usize>,
    possible_matches: &RangeSetBlaze<u32>,
) {
    for token_id in possible_matches.iter() {
        continuation_tokens.remove(token_id as usize);
    }
}

fn merge_assoc(into: &mut BTreeMap<u32, SourceAssoc>, state: u32, tsid: u32, nodes: &[u32]) {
    match into.entry(state) {
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            debug_assert_eq!(entry.get().tsid, tsid);
            entry.get_mut().nodes.extend(nodes.iter().copied());
        }
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(SourceAssoc {
                tsid,
                nodes: nodes.to_vec(),
            });
        }
    }
}

fn continuation_state(
    pending: &mut BTreeMap<u32, SourceAssoc>,
    tokenizer_state: u32,
    tsid: u32,
    nwa: &mut NWA,
) -> u32 {
    if let Some(existing) = pending
        .get(&tokenizer_state)
        .and_then(|assoc| {
            debug_assert_eq!(assoc.tsid, tsid);
            assoc.nodes.first()
        })
        .copied()
    {
        return existing;
    }

    let state = nwa.add_state();
    match pending.entry(tokenizer_state) {
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            debug_assert_eq!(entry.get().tsid, tsid);
            entry.get_mut().nodes.push(state);
        }
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(SourceAssoc {
                tsid,
                nodes: vec![state],
            });
        }
    }
    state
}

pub(crate) fn build_terminal_dwa(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);
    let vocab_tree = VocabPrefixTree::build(
        &vocab
            .entries
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect::<Vec<_>>(),
    );
    let mut possible_matches = PossibleMatchesComputer::new(tokenizer);

    for internal_tsid in 0..id_map.num_tsids() {
        let root = nwa.add_state();
        nwa.add_epsilon(
            start_state,
            root,
            all_token_weight(internal_tsid, id_map.max_token_id()),
        );

        let mut assoc_by_state = BTreeMap::<u32, SourceAssoc>::new();
        for original_state in &id_map.tokenizer_states.internal_to_originals[internal_tsid as usize] {
            merge_assoc(&mut assoc_by_state, *original_state, internal_tsid, &[root]);
        }

        let mut builder = TerminalNwaBuilder {
            tokenizer,
            possible_matches: &mut possible_matches,
            nwa: &mut nwa,
            leaf_state,
            ignore_terminal,
            transition_buffer: BTreeMap::new(),
            epsilon_buffer: BTreeMap::new(),
        };
        builder.build_from_trie(&vocab_tree.root, &assoc_by_state);
        builder.flush_transition_buffer();
    }

    let _ = prune_disallowed_follows(&mut nwa, grammar);
    let dwa = minimize(
        &determinize(&nwa)
            .expect("terminal NWA determinization failed despite acyclic token trie construction"),
    );
    dwa
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};
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

        let terminal_dwa = build_terminal_dwa(&glr_grammar, &tokenizer, &vocab, &id_map, None);

        assert!(
            !terminal_dwa.eval_word(&[0]).is_empty(),
            "terminal DWA should accept the single-terminal path 'a'"
        );
        assert!(
            !terminal_dwa.eval_word(&[0, 1]).is_empty(),
            "terminal DWA should accept the multi-terminal path 'ab'"
        );
    }

    #[test]
    fn test_terminal_dwa_treats_ignore_terminal_as_epsilon() {
        let grammar = GrammarDef {
            rules: vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(2)],
            }],
            start: 0,
            terminals: vec![
                Terminal::Literal {
                    id: 0,
                    bytes: b"a".to_vec(),
                },
                Terminal::Pattern {
                    id: 1,
                    pattern: " +".to_string(),
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(1),
        };
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vec![
                (0, b" ".to_vec()),
                (1, b"a".to_vec()),
                (2, b" a".to_vec()),
                (3, b"b".to_vec()),
            ],
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab);

        let terminal_dwa = build_terminal_dwa(
            &glr_grammar,
            &tokenizer,
            &vocab,
            &id_map,
            grammar.ignore_terminal,
        );

        let empty_weight = terminal_dwa.eval_word(&[]);
        assert!(
            empty_weight.token_union().contains(0),
            "ignore-only tokens should appear in the terminal DWA start-state final weight"
        );

        let a_weight = terminal_dwa.eval_word(&[0]);
        assert!(
            a_weight.token_union().contains(1),
            "plain non-ignore terminal tokens should still be accepted"
        );
        assert!(
            a_weight.token_union().contains(2),
            "tokens with ignored prefixes should also be accepted on the same terminal word"
        );
    }
}
