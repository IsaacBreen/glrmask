//! Terminal DWA construction.
//!
//! The paper architecture has a single terminal-side compilation object: the
//! Terminal DWA. This module now reflects that cardinality directly.
//!
//! The current implementation is still reduced relative to `sep1`, but it now
//! walks actual vocabulary tokens through the tokenizer and builds terminal-path
//! structure instead of projecting everything from `possible_matches`.

use std::collections::{BTreeMap, BTreeSet};

use crate::Vocab;
use crate::automata::weighted::nwa::Nwa;
use crate::automata::weighted::weight::Weight;
use crate::compiler::glr::grammar::{EOF, GlrGrammar};
use crate::compiler::grammar_def::TerminalId;
use crate::compiler::grammar_def::Symbol;
use crate::compiler::tokenizer_dfa::TokenizerDfa;
use crate::compiler::vocab_pre::VocabPreprocessing;
use crate::ds::RangeSet;

/// Reduced terminal-side compilation artifact.
#[derive(Debug, Clone)]
pub(crate) struct TerminalDwa {
    pub(crate) nwa: Nwa,
    pub(crate) tsid_roots: Vec<u32>,
    /// Non-greedy terminals at each tokenizer state; reserved for future use by
    /// the suffix-pruning optimisation.
    #[allow(dead_code)]
    pub(crate) non_greedy_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalId>>,
    /// Terminals still reachable on a non-empty continuation at each tokenizer
    /// state; reserved for the suffix-pruning optimisation.
    #[allow(dead_code)]
    pub(crate) possible_future_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalId>>,
}

fn add_or_union_transition(nwa: &mut Nwa, from: u32, label: i32, to: u32, weight: Weight) {
    let targets = nwa.states[from as usize].transitions.entry(label).or_default();
    if let Some((_, existing)) = targets.iter_mut().find(|(dest, _)| *dest == to) {
        *existing = existing.union(&weight);
    } else {
        targets.push((to, weight));
    }
}

#[derive(Debug, Default, Clone)]
struct VocabTrieNode {
    children: BTreeMap<u8, usize>,
    terminal_tokens: RangeSet,
    reachable_tokens: RangeSet,
}

fn build_vocab_trie(vocab: &Vocab) -> Vec<VocabTrieNode> {
    let mut nodes = vec![VocabTrieNode::default()];
    for (token_id, token_bytes) in &vocab.entries {
        let mut node_id = 0usize;
        nodes[node_id].reachable_tokens.insert(*token_id);
        for &byte in token_bytes {
            let next = if let Some(&child) = nodes[node_id].children.get(&byte) {
                child
            } else {
                let child = nodes.len();
                nodes.push(VocabTrieNode::default());
                nodes[node_id].children.insert(byte, child);
                child
            };
            node_id = next;
            nodes[node_id].reachable_tokens.insert(*token_id);
        }
        nodes[node_id].terminal_tokens.insert(*token_id);
    }
    nodes
}

#[derive(Debug)]
struct TerminalPrecomputer<'a> {
    tokenizer: &'a TokenizerDfa,
    vocab_pre: &'a VocabPreprocessing,
    trie: Vec<VocabTrieNode>,
    nwa: Nwa,
    leaf_state: u32,
    tsid_roots: Vec<u32>,
    pending_transitions: BTreeMap<u32, BTreeMap<i32, BTreeMap<u32, Weight>>>,
    pending_token_ids: Vec<Vec<Vec<u32>>>,
    live_tokens: BTreeMap<u32, Weight>,
}

impl<'a> TerminalPrecomputer<'a> {
    fn new(tokenizer: &'a TokenizerDfa, vocab: &Vocab, vocab_pre: &'a VocabPreprocessing) -> Self {
        let mut nwa = Nwa::new(vocab_pre.num_tsids, vocab_pre.max_token);
        let leaf_state = nwa.add_state();
        nwa.set_final_weight(leaf_state, Weight::all(nwa.max_position(), vocab_pre.num_tsids));

        Self {
            tokenizer,
            vocab_pre,
            trie: build_vocab_trie(vocab),
            nwa,
            leaf_state,
            tsid_roots: Vec::with_capacity(vocab_pre.num_tsids as usize),
            pending_transitions: BTreeMap::new(),
            pending_token_ids: Vec::new(),
            live_tokens: BTreeMap::new(),
        }
    }

    fn weight_from_tokens(&self, tsid: u32, token_ids: &RangeSet) -> Weight {
        Weight::from_entries(vec![(tsid, tsid, token_ids.clone())], self.vocab_pre.num_tsids)
    }

    fn update_live_tokens(&mut self, dst: u32, weight: &Weight) {
        if let Some(existing) = self.live_tokens.get_mut(&dst) {
            *existing = existing.union(weight);
        } else {
            self.live_tokens.insert(dst, weight.clone());
        }
    }

    fn add_pending_transition(&mut self, src: u32, label: i32, dst: u32, weight: Weight) {
        self.update_live_tokens(dst, &weight);
        self.pending_transitions
            .entry(src)
            .or_default()
            .entry(label)
            .or_default()
            .entry(dst)
            .and_modify(|existing| *existing = existing.union(&weight))
            .or_insert(weight);
    }

    fn add_pending_token_ids(&mut self, src: u32, label: i32, token_ids: &RangeSet) {
        if label < 0 {
            return;
        }
        let label_idx = label as usize;
        let src_idx = src as usize;
        if src_idx >= self.pending_token_ids.len() {
            self.pending_token_ids.resize_with(src_idx + 1, Vec::new);
        }
        if label_idx >= self.pending_token_ids[src_idx].len() {
            self.pending_token_ids[src_idx].resize_with(label_idx + 1, Vec::new);
        }
        self.pending_token_ids[src_idx][label_idx].extend(token_ids.iter_values());
    }

    fn flush_pending_token_ids(&mut self, tsid: u32) {
        let leaf_state = self.leaf_state;
        let pending = std::mem::take(&mut self.pending_token_ids);
        for (src, labels) in pending.into_iter().enumerate() {
            for (label_idx, mut token_ids) in labels.into_iter().enumerate() {
                if token_ids.is_empty() {
                    continue;
                }
                token_ids.sort_unstable();
                token_ids.dedup();
                let mut tokens = RangeSet::new();
                for token_id in token_ids {
                    tokens.insert(token_id);
                }
                let weight = self.weight_from_tokens(tsid, &tokens);
                self.add_pending_transition(src as u32, label_idx as i32, leaf_state, weight);
            }
        }
    }

    fn flush_pending_transitions(&mut self) {
        for (src, labels) in std::mem::take(&mut self.pending_transitions) {
            for (label, targets) in labels {
                for (dst, weight) in targets {
                    add_or_union_transition(&mut self.nwa, src, label, dst, weight);
                }
            }
        }
    }

    fn get_or_create_assoc_state(
        &mut self,
        assoc_by_state: &mut BTreeMap<u32, Vec<u32>>,
        tokenizer_state: u32,
    ) -> u32 {
        if let Some(existing) = assoc_by_state.get(&tokenizer_state).and_then(|states| states.first()).copied() {
            existing
        } else {
            let state = self.nwa.add_state();
            assoc_by_state.insert(tokenizer_state, vec![state]);
            state
        }
    }

    fn dfs(&mut self, tsid: u32, node_id: usize, assoc_by_state: BTreeMap<u32, Vec<u32>>) {
        let children: Vec<(u8, usize)> = self.trie[node_id]
            .children
            .iter()
            .map(|(&byte, &child_id)| (byte, child_id))
            .collect();

        for (byte, child_id) in children {
            let child = self.trie[child_id].clone();
            let continuation_tokens = child.reachable_tokens.difference(&child.terminal_tokens);
            let mut next_level_assoc: BTreeMap<u32, Vec<u32>> = BTreeMap::new();

            for (tokenizer_state, src_nodes) in &assoc_by_state {
                let next_state = self.tokenizer.dfa.get_transition(*tokenizer_state, byte);
                if next_state == crate::automata::dfa::DEAD {
                    continue;
                }

                next_level_assoc
                    .entry(next_state)
                    .or_default()
                    .extend(src_nodes.iter().copied());

                let matched_terminals = self.tokenizer.matched_terminals(next_state);
                if matched_terminals.is_empty() {
                    continue;
                }

                let continuation_state = if continuation_tokens.is_empty() {
                    None
                } else {
                    Some(self.get_or_create_assoc_state(
                        &mut next_level_assoc,
                        self.tokenizer.initial_state(),
                    ))
                };
                let continuation_weight = continuation_state
                    .map(|_| self.weight_from_tokens(tsid, &continuation_tokens));

                for &src_node in src_nodes {
                    for terminal in &matched_terminals {
                        if !child.terminal_tokens.is_empty() {
                            self.add_pending_token_ids(src_node, *terminal as i32, &child.terminal_tokens);
                        }
                        if let (Some(dst), Some(weight)) = (continuation_state, continuation_weight.as_ref()) {
                            self.add_pending_transition(src_node, *terminal as i32, dst, weight.clone());
                        }
                    }
                }
            }

            if !next_level_assoc.is_empty() {
                self.dfs(tsid, child_id, next_level_assoc);
            }
        }
    }

    fn run_dfs(mut self) -> TerminalDwa {
        for (tsid, &tokenizer_state) in self.vocab_pre.tsid_to_state.iter().enumerate() {
            let root = self.nwa.add_state();
            self.tsid_roots.push(root);
            self.nwa.start_states.push(root);
            self.live_tokens.insert(root, self.weight_from_tokens(tsid as u32, &self.trie[0].reachable_tokens));

            let assoc_by_state = BTreeMap::from([(tokenizer_state, vec![root])]);
            self.dfs(tsid as u32, 0, assoc_by_state);
            self.flush_pending_token_ids(tsid as u32);
        }

        self.flush_pending_transitions();
        TerminalDwa {
            nwa: self.nwa,
            tsid_roots: self.tsid_roots,
            non_greedy_terminals_by_tokenizer_state: (0..self.tokenizer.num_states())
                .map(|state| self.tokenizer.matched_non_greedy_terminals(state))
                .collect(),
            possible_future_terminals_by_tokenizer_state: (0..self.tokenizer.num_states())
                .map(|state| self.tokenizer.possible_future_terminals(state))
                .collect(),
        }
    }
}

fn compute_ever_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    let mut ever_allowed: Vec<BTreeSet<TerminalId>> =
        vec![BTreeSet::new(); grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }

            let suffix = &rule.rhs[index + 1..];
            let mut allowed = grammar.first_of_seq(suffix);
            allowed.remove(&EOF);
            if suffix.is_empty() || grammar.seq_is_nullable(suffix) {
                allowed.extend(
                    grammar.follow[rule.lhs as usize]
                        .iter()
                        .copied()
                        .filter(|follow| *follow != EOF && *follow < grammar.num_terminals),
                );
            }
            ever_allowed[*terminal as usize].extend(
                allowed
                    .into_iter()
                    .filter(|follow| *follow < grammar.num_terminals),
            );
        }
    }

    ever_allowed
        .into_iter()
        .map(|allowed| allowed.into_iter().collect())
        .collect()
}

fn compute_always_allowed_follows(grammar: &GlrGrammar) -> Vec<Vec<TerminalId>> {
    let mut always_allowed: Vec<Option<BTreeSet<TerminalId>>> =
        vec![None; grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }

            let suffix = &rule.rhs[index + 1..];
            let mut allowed = grammar.first_of_seq(suffix);
            allowed.remove(&EOF);
            if suffix.is_empty() || grammar.seq_is_nullable(suffix) {
                allowed.extend(
                    grammar.follow[rule.lhs as usize]
                        .iter()
                        .copied()
                        .filter(|follow| *follow != EOF && *follow < grammar.num_terminals),
                );
            }
            let allowed: BTreeSet<TerminalId> = allowed
                .into_iter()
                .filter(|follow| *follow < grammar.num_terminals)
                .collect();
            match &mut always_allowed[*terminal as usize] {
                None => always_allowed[*terminal as usize] = Some(allowed),
                Some(existing) => existing.retain(|follow| allowed.contains(follow)),
            }
        }
    }

    always_allowed
        .into_iter()
        .map(|allowed| allowed.unwrap_or_default().into_iter().collect())
        .collect()
}

fn collapse_always_allowed(
    nwa: &mut Nwa,
    always_allowed_by_label: &[Vec<TerminalId>],
    terminals_count: usize,
) -> bool {
    if always_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let num_states = nwa.states.len();
    let mut incoming: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); num_states];
    let mut domain: Vec<Weight> = (0..num_states)
        .map(|_| Weight::empty(nwa.num_tsids))
        .collect();
    let all_positions = Weight::all(nwa.max_position(), nwa.num_tsids);

    let mut queue = std::collections::VecDeque::new();
    let mut in_queue = vec![false; num_states];
    for &start in &nwa.start_states {
        domain[start as usize] = all_positions.clone();
        queue.push_back(start);
        in_queue[start as usize] = true;
    }

    while let Some(state_id) = queue.pop_front() {
        in_queue[state_id as usize] = false;
        let state_domain = domain[state_id as usize].clone();
        if state_domain.is_empty() {
            continue;
        }
        let state = &nwa.states[state_id as usize];
        let incoming_labels: Vec<i32> = incoming[state_id as usize].iter().copied().collect();

        for (dest, _) in &state.epsilons {
            let dest_idx = *dest as usize;
            let updated = domain[dest_idx].union(&state_domain);
            if !updated.is_subset(&domain[dest_idx]) {
                domain[dest_idx] = updated;
                if !in_queue[dest_idx] {
                    in_queue[dest_idx] = true;
                    queue.push_back(*dest);
                }
            }
            incoming[dest_idx].extend(incoming_labels.iter().copied());
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            for (dest, weight) in targets {
                let dest_idx = *dest as usize;
                let contrib = state_domain.intersection(weight);
                if !contrib.is_empty() {
                    let updated = domain[dest_idx].union(&contrib);
                    if !updated.is_subset(&domain[dest_idx]) {
                        domain[dest_idx] = updated;
                        if !in_queue[dest_idx] {
                            in_queue[dest_idx] = true;
                            queue.push_back(*dest);
                        }
                    }
                }
                if incoming[dest_idx].insert(label) && !in_queue[dest_idx] {
                    in_queue[dest_idx] = true;
                    queue.push_back(*dest);
                }
            }
        }
    }

    let all_labels: Vec<i32> = (0..terminals_count as i32).collect();
    for &start in &nwa.start_states {
        incoming[start as usize].extend(all_labels.iter().copied());
    }
    for state_id in 0..num_states {
        if !nwa.states[state_id].epsilons.is_empty() {
            incoming[state_id].extend(all_labels.iter().copied());
        }
    }

    let mut allowed_by_state: Vec<BTreeSet<i32>> = vec![BTreeSet::new(); num_states];
    for state_id in 0..num_states {
        let mut labels = incoming[state_id].iter();
        let Some(&first_label) = labels.next() else {
            continue;
        };
        if first_label < 0 || first_label as usize >= always_allowed_by_label.len() {
            continue;
        }
        let mut allowed: BTreeSet<i32> = always_allowed_by_label[first_label as usize]
            .iter()
            .map(|label| *label as i32)
            .collect();
        for &label in labels {
            if label < 0 || label as usize >= always_allowed_by_label.len() {
                continue;
            }
            let follows: BTreeSet<i32> = always_allowed_by_label[label as usize]
                .iter()
                .map(|follow| *follow as i32)
                .collect();
            allowed.retain(|candidate| follows.contains(candidate));
            if allowed.is_empty() {
                break;
            }
        }
        allowed_by_state[state_id] = allowed;
    }

    let final_weights: Vec<Option<Weight>> = nwa.states.iter().map(|state| state.final_weight.clone()).collect();
    let mut changed = false;
    for state_id in 0..num_states {
        let allowed = &allowed_by_state[state_id];
        if allowed.is_empty() || domain[state_id].is_empty() {
            continue;
        }

        let mut labels_to_remove = Vec::new();
        let state = &mut nwa.states[state_id];
        for (&label, targets) in state.transitions.iter_mut() {
            if label < 0 || label as usize >= terminals_count || !allowed.contains(&label) {
                continue;
            }

            let mut retained = Vec::new();
            for (dest, weight) in targets.iter() {
                let Some(final_weight) = final_weights[*dest as usize].as_ref() else {
                    retained.push((*dest, weight.clone()));
                    continue;
                };
                let reach = domain[state_id].intersection(weight);
                if !reach.is_empty() && reach.is_subset(final_weight) {
                    let collapsed = final_weight.intersection(weight);
                    let updated = state
                        .final_weight
                        .clone()
                        .unwrap_or_else(|| Weight::empty(nwa.num_tsids))
                        .union(&collapsed);
                    state.final_weight = if updated.is_empty() { None } else { Some(updated) };
                    changed = true;
                } else {
                    retained.push((*dest, weight.clone()));
                }
            }

            if retained.is_empty() {
                labels_to_remove.push(label);
            } else {
                *targets = retained;
            }
        }
        for label in labels_to_remove {
            state.transitions.remove(&label);
        }
    }

    changed
}

fn prune_disallowed_follows(
    nwa: &mut Nwa,
    ever_allowed_by_label: &[Vec<TerminalId>],
    terminals_count: usize,
) -> bool {
    if ever_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let words_needed = (terminals_count + 63) / 64;
    let new_bitset = || vec![0u64; words_needed];
    let set_bit = |bs: &mut [u64], idx: usize| {
        if idx < terminals_count {
            bs[idx / 64] |= 1u64 << (idx % 64);
        }
    };
    let test_bit = |bs: &[u64], idx: usize| -> bool {
        idx < terminals_count && (bs[idx / 64] & (1u64 << (idx % 64))) != 0
    };
    let is_empty = |bs: &[u64]| -> bool { bs.iter().all(|&word| word == 0) };
    let union_into = |dst: &mut [u64], src: &[u64]| {
        for (lhs, rhs) in dst.iter_mut().zip(src) {
            *lhs |= *rhs;
        }
    };
    let intersect_into = |dst: &mut [u64], src: &[u64]| {
        for (lhs, rhs) in dst.iter_mut().zip(src) {
            *lhs &= *rhs;
        }
    };

    let mut all_terminals = new_bitset();
    for idx in 0..terminals_count {
        set_bit(&mut all_terminals, idx);
    }
    let disallowed_after: Vec<Vec<u64>> = (0..terminals_count)
        .map(|idx| {
            if idx >= ever_allowed_by_label.len() {
                return new_bitset();
            }
            let mut bitset = all_terminals.clone();
            for &allowed in &ever_allowed_by_label[idx] {
                let allowed = allowed as usize;
                if allowed < terminals_count {
                    bitset[allowed / 64] &= !(1u64 << (allowed % 64));
                }
            }
            bitset
        })
        .collect();

    let mut in_degree = vec![0u32; nwa.states.len()];
    for state in &nwa.states {
        for (dest, _) in &state.epsilons {
            in_degree[*dest as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dest, _) in targets {
                in_degree[*dest as usize] += 1;
            }
        }
    }

    let mut topo_queue = std::collections::VecDeque::new();
    for (sid, &degree) in in_degree.iter().enumerate() {
        if degree == 0 {
            topo_queue.push_back(sid as u32);
        }
    }
    let mut topo_order = Vec::with_capacity(nwa.states.len());
    let mut disallowed_union: Vec<Option<Vec<u64>>> = vec![None; nwa.states.len()];
    for &start in &nwa.start_states {
        disallowed_union[start as usize] = Some(new_bitset());
    }

    while let Some(sid) = topo_queue.pop_front() {
        topo_order.push(sid);
        let src_disallowed = disallowed_union[sid as usize]
            .clone()
            .unwrap_or_else(new_bitset);
        let state = &nwa.states[sid as usize];

        for (dest, _) in &state.epsilons {
            let dest_set = disallowed_union[*dest as usize].get_or_insert_with(new_bitset);
            union_into(dest_set, &src_disallowed);
        }
        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            let label_disallowed = &disallowed_after[label as usize];
            for (dest, _) in targets {
                let dest_set = disallowed_union[*dest as usize].get_or_insert_with(new_bitset);
                union_into(dest_set, label_disallowed);
            }
        }

        for (dest, _) in &state.epsilons {
            in_degree[*dest as usize] -= 1;
            if in_degree[*dest as usize] == 0 {
                topo_queue.push_back(*dest);
            }
        }
        for targets in state.transitions.values() {
            for (dest, _) in targets {
                in_degree[*dest as usize] -= 1;
                if in_degree[*dest as usize] == 0 {
                    topo_queue.push_back(*dest);
                }
            }
        }
    }

    let mut disallowed_intersection: Vec<Option<Vec<u64>>> = vec![None; nwa.states.len()];
    for &start in &nwa.start_states {
        disallowed_intersection[start as usize] = Some(new_bitset());
    }
    for &sid in &topo_order {
        let src_disallowed = disallowed_intersection[sid as usize]
            .clone()
            .unwrap_or_else(new_bitset);
        let state = &nwa.states[sid as usize];

        for (dest, _) in &state.epsilons {
            match &mut disallowed_intersection[*dest as usize] {
                None => disallowed_intersection[*dest as usize] = Some(src_disallowed.clone()),
                Some(existing) => intersect_into(existing, &src_disallowed),
            }
        }
        for (&label, targets) in &state.transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            let label_disallowed = &disallowed_after[label as usize];
            for (dest, _) in targets {
                match &mut disallowed_intersection[*dest as usize] {
                    None => disallowed_intersection[*dest as usize] = Some(label_disallowed.clone()),
                    Some(existing) => intersect_into(existing, label_disallowed),
                }
            }
        }
    }

    let mut changed = false;
    for sid in 0..nwa.states.len() {
        let Some(state_disallowed) = &disallowed_intersection[sid] else {
            continue;
        };
        if is_empty(state_disallowed) {
            continue;
        }

        let labels_to_remove: Vec<i32> = nwa.states[sid]
            .transitions
            .keys()
            .copied()
            .filter(|label| *label >= 0 && (*label as usize) < terminals_count && test_bit(state_disallowed, *label as usize))
            .collect();
        if labels_to_remove.is_empty() {
            continue;
        }
        changed = true;
        for label in labels_to_remove {
            nwa.states[sid].transitions.remove(&label);
        }
    }

    changed
}

/// Build the singular terminal-side compilation object from an actual
/// tokenizer/vocabulary walk.
pub(crate) fn build_terminal_dwa(
    tokenizer: &TokenizerDfa,
    vocab: &Vocab,
    vocab_pre: &VocabPreprocessing,
    grammar: &GlrGrammar,
) -> TerminalDwa {
    let mut terminal_dwa = TerminalPrecomputer::new(tokenizer, vocab, vocab_pre).run_dfs();

    let always_allowed_by_label = compute_always_allowed_follows(grammar);
    let _ = collapse_always_allowed(
        &mut terminal_dwa.nwa,
        &always_allowed_by_label,
        grammar.num_terminals as usize,
    );
    let ever_allowed_by_label = compute_ever_allowed_follows(grammar);
    let _ = prune_disallowed_follows(
        &mut terminal_dwa.nwa,
        &ever_allowed_by_label,
        grammar.num_terminals as usize,
    );
    terminal_dwa
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::regex::bytes;
    use crate::compiler::grammar_def::tests::simple_ab_grammar;
    use crate::compiler::glr::grammar::GlrGrammar;
    use crate::compiler::tokenizer_dfa::TokenizerDfa;
    use crate::compiler::vocab_pre::VocabPreprocessing;

    #[test]
    fn test_build_terminal_dwa_collapses_always_allowed_follow_path() {
        let grammar = simple_ab_grammar();
        let glr_grammar = GlrGrammar::from_grammar_def(&grammar);
        let tokenizer = TokenizerDfa::from_grammar_def(&grammar);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())], None);
        let vocab_pre = VocabPreprocessing::compute(&tokenizer, &vocab);

        let terminal_dwa = build_terminal_dwa(&tokenizer, &vocab, &vocab_pre, &glr_grammar);
        let initial_tsid = vocab_pre.state_to_tsid[tokenizer.initial_state() as usize] as usize;
        let root = terminal_dwa.tsid_roots[initial_tsid];
        let a_targets = &terminal_dwa.nwa.states[root as usize].transitions[&0];
        assert!(!a_targets.is_empty());

        let mut combined_a = Weight::empty(vocab_pre.num_tsids);
        for (_, weight) in a_targets {
            combined_a = combined_a.union(weight);
        }
        assert_eq!(combined_a.tokens_for_tsid(initial_tsid as u32), RangeSet::from_range(0, 1));

        for (dest, weight) in a_targets {
            let state = &terminal_dwa.nwa.states[*dest as usize];
            assert!(state.final_weight.is_some());
            assert!(!state.transitions.contains_key(&1));
            if !state.transitions.is_empty() {
                assert_eq!(weight.tokens_for_tsid(initial_tsid as u32), RangeSet::from_range(1, 1));
            }
        }
    }

    #[test]
    fn test_terminal_dwa_carries_tokenizer_greedy_metadata() {
        let grammar = simple_ab_grammar();
        let glr_grammar = GlrGrammar::from_grammar_def(&grammar);
        let tokenizer = TokenizerDfa::from_expr_groups(&[
            crate::automata::regex::ExprGroup {
                expr: bytes(b"a"),
                is_non_greedy: true,
            },
            crate::automata::regex::ExprGroup {
                expr: bytes(b"ab"),
                is_non_greedy: false,
            },
        ]);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())], None);
        let vocab_pre = VocabPreprocessing::compute(&tokenizer, &vocab);

        let terminal_dwa = build_terminal_dwa(&tokenizer, &vocab, &vocab_pre, &glr_grammar);
        let state_after_a = tokenizer.run(b"a") as usize;

        assert!(terminal_dwa.non_greedy_terminals_by_tokenizer_state[state_after_a].contains(&0));
        assert!(terminal_dwa.possible_future_terminals_by_tokenizer_state[state_after_a].contains(&1));
    }
}
