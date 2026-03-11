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
use crate::compiler::stages::profile_stats::{
    WeightedDwaStats,
    WeightedNwaStats,
    collect_weighted_dwa_stats,
    collect_weighted_nwa_stats,
};
use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
use crate::ds::weight::Weight;

#[derive(Debug, Clone, Default)]
pub(crate) struct TerminalDwaBuildReport {
    pub build_vocab_trie_time: std::time::Duration,
    pub build_nwa_from_trie_time: std::time::Duration,
    pub collapse_always_allowed_time: std::time::Duration,
    pub collapse_always_allowed_applied: bool,
    pub prune_disallowed_follows_time: std::time::Duration,
    pub determinize_time: std::time::Duration,
    pub minimize_time: std::time::Duration,
    pub total_time: std::time::Duration,
    pub vocab_entries: usize,
    pub internal_tsids: usize,
    pub terminal_nwa: WeightedNwaStats,
    pub terminal_dwa: WeightedDwaStats,
    pub terminal_minimized_dwa: WeightedDwaStats,
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

fn compute_always_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
    let mut always_allowed = vec![None::<BTreeSet<TerminalID>>; grammar.num_terminals as usize];

    for rule in &grammar.rules {
        for (index, symbol) in rule.rhs.iter().enumerate() {
            let Symbol::Terminal(terminal) = symbol else {
                continue;
            };
            if *terminal >= grammar.num_terminals {
                continue;
            }

            let follows = occurrence_follow_set(grammar, rule.lhs, &rule.rhs, index);
            match &mut always_allowed[*terminal as usize] {
                None => always_allowed[*terminal as usize] = Some(follows),
                Some(existing) => existing.retain(|follow| follows.contains(follow)),
            }
        }
    }

    always_allowed
        .into_iter()
        .map(|set| set.unwrap_or_default().into_iter().collect())
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

fn collapse_always_allowed(
    nwa: &mut NWA,
    always_allowed_by_label: &[Vec<TerminalID>],
    terminals_count: usize,
) -> bool {
    if always_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let topo_order = topological_order(nwa);
    if topo_order.is_empty() {
        return false;
    }

    let mut incoming: Vec<HashSet<TerminalID>> = vec![HashSet::new(); nwa.states.len()];
    let mut domain: Vec<Weight> = vec![Weight::empty(); nwa.states.len()];
    let mut queue = VecDeque::new();
    let mut in_queue = vec![false; nwa.states.len()];

    for &start in &nwa.start_states {
        domain[start as usize] = Weight::all();
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
        let incoming_labels = incoming[state_id as usize].clone();

        for (dst, _) in &state.epsilons {
            let next_domain = domain[*dst as usize].union(&state_domain);
            let domain_changed = !next_domain.is_subset(&domain[*dst as usize]);
            if domain_changed {
                domain[*dst as usize] = next_domain;
            }

            let labels_before = incoming[*dst as usize].len();
            incoming[*dst as usize].extend(incoming_labels.iter().copied());
            let labels_changed = incoming[*dst as usize].len() != labels_before;

            if (domain_changed || labels_changed) && !in_queue[*dst as usize] {
                in_queue[*dst as usize] = true;
                queue.push_back(*dst);
            }
        }

        for (&label, targets) in &state.transitions {
            if label < 0 || (label as usize) >= terminals_count {
                continue;
            }

            for (dst, weight) in targets {
                let contrib = state_domain.intersection(weight);
                let next_domain = domain[*dst as usize].union(&contrib);
                let domain_changed = !next_domain.is_subset(&domain[*dst as usize]);
                if domain_changed {
                    domain[*dst as usize] = next_domain;
                }

                let labels_changed = incoming[*dst as usize].insert(label as TerminalID);
                if (domain_changed || labels_changed) && !in_queue[*dst as usize] {
                    in_queue[*dst as usize] = true;
                    queue.push_back(*dst);
                }
            }
        }
    }

    let mut allowed_by_state: Vec<HashSet<TerminalID>> = vec![HashSet::new(); nwa.states.len()];
    for state_id in 0..nwa.states.len() {
        let Some(&first_label) = incoming[state_id].iter().next() else {
            continue;
        };
        let Some(first_follows) = always_allowed_by_label.get(first_label as usize) else {
            continue;
        };

        let mut allowed: HashSet<TerminalID> = first_follows.iter().copied().collect();
        for &label in incoming[state_id].iter().skip(1) {
            let Some(follows) = always_allowed_by_label.get(label as usize) else {
                continue;
            };
            allowed.retain(|terminal| follows.contains(terminal));
            if allowed.is_empty() {
                break;
            }
        }
        allowed_by_state[state_id] = allowed;
    }

    let mut final_weights: Vec<Option<Weight>> =
        nwa.states.iter().map(|state| state.final_weight.clone()).collect();
    let mut changed = false;

    for &state_id in topo_order.iter().rev() {
        let allowed = &allowed_by_state[state_id];
        if allowed.len() != 1 {
            continue;
        }
        let only_allowed = *allowed.iter().next().expect("singleton set checked above");

        let domain_state = &domain[state_id];
        if domain_state.is_empty() {
            continue;
        }

        let state = &mut nwa.states[state_id];
        let mut state_final_weight = final_weights[state_id].clone();
        let mut labels_to_remove = Vec::new();

        for (&label, targets) in state.transitions.iter_mut() {
            if label < 0 || (label as usize) >= terminals_count {
                continue;
            }
            if label as TerminalID != only_allowed {
                continue;
            }

            let mut new_targets = Vec::new();
            for (dst, weight) in targets.iter() {
                let Some(dst_final_weight) = final_weights[*dst as usize].as_ref() else {
                    new_targets.push((*dst, weight.clone()));
                    continue;
                };

                let reach = domain_state.intersection(weight);
                if !reach.is_empty() && reach.is_subset(dst_final_weight) {
                    let contrib = dst_final_weight.intersection(weight);
                    if !contrib.is_empty() {
                        state_final_weight = Some(match state_final_weight.take() {
                            Some(existing) => existing.union(&contrib),
                            None => contrib,
                        });
                    }
                    changed = true;
                    continue;
                }

                new_targets.push((*dst, weight.clone()));
            }

            if new_targets.is_empty() {
                labels_to_remove.push(label);
            } else {
                *targets = new_targets;
            }
        }

        for label in labels_to_remove {
            state.transitions.remove(&label);
        }

        state.final_weight = state_final_weight.clone();
        final_weights[state_id] = state_final_weight;
    }

    if prune_unreachable_states(nwa) {
        changed = true;
    }

    changed
}

fn should_collapse_always_allowed(vocab: &Vocab) -> bool {
    // This collapse is a compile-time optimization only. On very large LLM
    // vocabularies it can become pathologically expensive because it performs
    // repeated `Weight` set algebra over large token-ID domains. Keep it for
    // small/medium vocabs where it helps, and skip it for huge vocabularies.
    vocab.entries.len() <= 8_192
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

fn token_weight_all_tsids(num_tsids: u32, token_id: u32) -> Weight {
    if num_tsids == 0 {
        return Weight::empty();
    }
    Weight::from_compact_ranges(std::iter::once((
        0..=num_tsids - 1,
        std::iter::once(token_id..=token_id),
    )))
}

fn token_set_weight(
    internal_tsid: u32,
    token_ids: &RangeSetBlaze<usize>,
    original_token_to_internal: &[u32],
) -> Weight {
    let mut mapped = RangeSetBlaze::new();
    for token_id in token_ids.iter() {
        if let Some(&internal_token_id) = original_token_to_internal.get(token_id) {
            debug_assert_ne!(internal_token_id, u32::MAX);
            if internal_token_id != u32::MAX {
                mapped.insert(internal_token_id);
            }
        }
    }
    Weight::from_token_set_for_tsid(internal_tsid, mapped)
}

fn token_set_weight_all_tsids(
    num_tsids: u32,
    token_ids: &RangeSetBlaze<usize>,
    original_token_to_internal: &[u32],
) -> Weight {
    if num_tsids == 0 {
        return Weight::empty();
    }
    let mut mapped = RangeSetBlaze::new();
    for token_id in token_ids.iter() {
        if let Some(&internal_token_id) = original_token_to_internal.get(token_id) {
            debug_assert_ne!(internal_token_id, u32::MAX);
            if internal_token_id != u32::MAX {
                mapped.insert(internal_token_id);
            }
        }
    }
    if mapped.is_empty() {
        return Weight::empty();
    }
    let token_ranges: Vec<_> = mapped.ranges().collect();
    Weight::from_compact_ranges(std::iter::once((0..=num_tsids - 1, token_ranges)))
}

fn all_token_weight(internal_tsid: u32, max_token_id: u32) -> Weight {
    Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([0..=max_token_id]),
    )
}

#[derive(Clone)]
struct SourceAssoc {
    nodes: Vec<u32>,
}

struct TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    tokenizer: &'tok Tokenizer,
    possible_matches: &'pm mut PossibleMatchesComputer<'tok>,
    nwa: &'nwa mut NWA,
    num_tsids: u32,
    leaf_state: u32,
    ignore_terminal: Option<TerminalID>,
    representative_initial_state: u32,
    representative_state_by_original: &'tok [u32],
    original_token_to_internal: &'tok [u32],
    leaf_token_ids_buffer: Vec<Vec<Vec<u32>>>,
    reachable_weight_cache: HashMap<usize, Weight>,
    leaf_weight_cache: HashMap<Vec<u32>, Weight>,
    transition_buffer: BTreeMap<(u32, i32, u32), Weight>,
    epsilon_buffer: BTreeMap<(u32, u32), Weight>,
}

impl<'tok, 'pm, 'nwa> TerminalNwaBuilder<'tok, 'pm, 'nwa> {
    fn buffer_leaf_token_id(&mut self, source: u32, label: TerminalID, internal_token_id: u32) {
        let source_idx = source as usize;
        if source_idx >= self.leaf_token_ids_buffer.len() {
            self.leaf_token_ids_buffer.resize_with(source_idx + 1, Vec::new);
        }
        let labels = &mut self.leaf_token_ids_buffer[source_idx];
        let label_idx = label as usize;
        if label_idx >= labels.len() {
            labels.resize_with(label_idx + 1, Vec::new);
        }
        labels[label_idx].push(internal_token_id);
    }

    fn cached_reachable_weight(&mut self, token_ids: &RangeSetBlaze<usize>) -> Weight {
        let cache_key = token_ids as *const RangeSetBlaze<usize> as usize;
        if let Some(weight) = self.reachable_weight_cache.get(&cache_key) {
            return weight.clone();
        }

        let weight = token_set_weight_all_tsids(
            self.num_tsids,
            token_ids,
            self.original_token_to_internal,
        );
        self.reachable_weight_cache.insert(cache_key, weight.clone());
        weight
    }

    fn cached_leaf_weight(&mut self, token_ids: Vec<u32>) -> Weight {
        if let Some(weight) = self.leaf_weight_cache.get(&token_ids) {
            return weight.clone();
        }

        let mapped = RangeSetBlaze::from_iter(token_ids.iter().copied().map(|token_id| token_id..=token_id));
        let token_ranges: Vec<_> = mapped.ranges().collect();
        let weight = Weight::from_compact_ranges(std::iter::once((
            0..=self.num_tsids - 1,
            token_ranges,
        )));
        self.leaf_weight_cache.insert(token_ids, weight.clone());
        weight
    }

    fn add_leaf_token_from_sources(
        &mut self,
        sources: &[u32],
        label: TerminalID,
        internal_token_id: u32,
    ) {
        if self.ignore_terminal == Some(label) {
            let weight = token_weight_all_tsids(self.num_tsids, internal_token_id);
            self.add_match_from_sources(sources, label, self.leaf_state, &weight);
            return;
        }

        for &source in sources {
            self.buffer_leaf_token_id(source, label, internal_token_id);
        }
    }

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
        for (from, labels_vec) in std::mem::take(&mut self.leaf_token_ids_buffer)
            .into_iter()
            .enumerate()
        {
            for (label_idx, mut token_ids) in labels_vec.into_iter().enumerate() {
                if token_ids.is_empty() {
                    continue;
                }
                token_ids.sort_unstable();
                token_ids.dedup();
                let weight = self.cached_leaf_weight(token_ids);
                self.transition_buffer
                    .entry((from as u32, label_idx as i32, self.leaf_state))
                    .and_modify(|existing| *existing = existing.union(&weight))
                    .or_insert(weight);
            }
        }

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
            let internal_child_token_id = self
                .original_token_to_internal
                .get(child_token_id as usize)
                .copied()
                .filter(|internal_token_id| *internal_token_id != u32::MAX)
                .expect("representative vocab token must map to an internal token id");

            let mut next_level_assoc = BTreeMap::<u32, SourceAssoc>::new();
            let mut pending = BTreeMap::<usize, BTreeMap<u32, SourceAssoc>>::new();
            pending.insert(0, assoc_by_state.clone());

            while let Some((pos, states_at_pos)) = pending.pop_first() {
                if pos == segment_bytes.len() {
                    for (tokenizer_state, assoc) in states_at_pos {
                        merge_assoc(&mut next_level_assoc, tokenizer_state, &assoc.nodes);
                    }
                    continue;
                }

                for (tokenizer_state, source_assoc) in states_at_pos {
                    let source_nodes = source_assoc.nodes;
                    let exec = self
                        .tokenizer
                        .execute_from_state(&segment_bytes[pos..], tokenizer_state);
                    let exec_end_state = exec.end_state.map(|end_state| {
                        self.representative_state_by_original
                            .get(end_state as usize)
                            .copied()
                            .unwrap_or(end_state)
                    });
                    let mut possible_matches_at_end = None;

                    if let Some(end_state) = exec_end_state {
                        if child_node.has_token() {
                            for terminal_id in self.tokenizer.tokens_accessible_from_state(end_state) {
                                self.add_leaf_token_from_sources(
                                    &source_nodes,
                                    terminal_id,
                                    internal_child_token_id,
                                );
                            }
                        }

                        merge_assoc(&mut next_level_assoc, end_state, &source_nodes);
                    }

                    for matched in exec.matches {
                        let next_pos = pos + matched.width;

                        if next_pos == segment_bytes.len() && child_node.has_token() {
                            self.add_leaf_token_from_sources(
                                &source_nodes,
                                matched.id,
                                internal_child_token_id,
                            );
                        }

                        let continuation_weight = if next_pos == segment_bytes.len()
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
                            if remaining.is_empty() {
                                continue;
                            }
                            token_set_weight_all_tsids(
                                self.num_tsids,
                                &remaining,
                                self.original_token_to_internal,
                            )
                        } else {
                            self.cached_reachable_weight(child_node.reachable_token_ids())
                        };
                        if continuation_weight.is_empty() {
                            continue;
                        }

                        let continuation_assoc = pending.entry(next_pos).or_default();
                        let destination = continuation_state(
                            continuation_assoc,
                            self.representative_initial_state,
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

fn merge_assoc(into: &mut BTreeMap<u32, SourceAssoc>, state: u32, nodes: &[u32]) {
    match into.entry(state) {
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry.get_mut().nodes.extend(nodes.iter().copied());
        }
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(SourceAssoc { nodes: nodes.to_vec() });
        }
    }
}

fn continuation_state(
    pending: &mut BTreeMap<u32, SourceAssoc>,
    tokenizer_state: u32,
    nwa: &mut NWA,
) -> u32 {
    if let Some(existing) = pending
        .get(&tokenizer_state)
        .and_then(|assoc| assoc.nodes.first())
        .copied()
    {
        return existing;
    }

    let state = nwa.add_state();
    match pending.entry(tokenizer_state) {
        std::collections::btree_map::Entry::Occupied(mut entry) => {
            entry.get_mut().nodes.push(state);
        }
        std::collections::btree_map::Entry::Vacant(entry) => {
            entry.insert(SourceAssoc { nodes: vec![state] });
        }
    }
    state
}

fn terminal_profile_enabled() -> bool {
    std::env::var_os("GLRMASK_PROFILE_COMPILE").is_some()
}

fn log_terminal_profile(enabled: bool, phase: &str, started_at: std::time::Instant) {
    if enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] {phase}_ms={:.3}",
            started_at.elapsed().as_secs_f64() * 1000.0
        );
    }
}

fn representative_original_ids(map: &crate::compiler::stages::equivalence_analysis::ManyToOneIdMap) -> Vec<u32> {
    map.original_to_internal
        .iter()
        .enumerate()
        .map(|(original_id, _)| {
            map.representative_original_id_for_original(original_id as u32)
                .unwrap_or(original_id as u32)
        })
        .collect()
}

fn representative_vocab_entries(vocab: &Vocab, id_map: &InternalIdMap) -> Vec<(u32, Vec<u8>)> {
    id_map
        .vocab_tokens
        .internal_to_originals
        .iter()
        .filter_map(|original_ids| {
            let representative = *original_ids.first()?;
            let bytes = vocab.entries.get(&representative)?.clone();
            Some((representative, bytes))
        })
        .collect()
}

pub(crate) fn build_terminal_dwa(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> DWA {
    build_terminal_dwa_with_report(grammar, tokenizer, vocab, id_map, ignore_terminal).0
}

pub(crate) fn build_terminal_dwa_with_report(
    grammar: &AnalyzedGrammar,
    tokenizer: &Tokenizer,
    vocab: &Vocab,
    id_map: &InternalIdMap,
    ignore_terminal: Option<TerminalID>,
) -> (DWA, TerminalDwaBuildReport) {
    let profile_enabled = terminal_profile_enabled();
    let total_started_at = std::time::Instant::now();
    let mut report = TerminalDwaBuildReport {
        vocab_entries: vocab.entries.len(),
        internal_tsids: id_map.num_tsids() as usize,
        ..TerminalDwaBuildReport::default()
    };

    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let start_state = nwa.add_state();
    nwa.start_states.push(start_state);

    let phase_started_at = std::time::Instant::now();
    let representative_vocab = representative_vocab_entries(vocab, id_map);
    let vocab_tree = VocabPrefixTree::build(
        &representative_vocab
            .iter()
            .map(|(token_id, bytes)| (*token_id as usize, bytes.clone()))
            .collect::<Vec<_>>(),
    );
    let representative_state_by_original = representative_original_ids(&id_map.tokenizer_states);
    let representative_initial_state = representative_state_by_original
        .get(tokenizer.initial_state() as usize)
        .copied()
        .unwrap_or_else(|| tokenizer.initial_state());
    let mut possible_matches = PossibleMatchesComputer::new(tokenizer);
    report.build_vocab_trie_time = phase_started_at.elapsed();
    log_terminal_profile(profile_enabled, "build_vocab_trie", phase_started_at);

    let phase_started_at = std::time::Instant::now();
    let mut assoc_by_state = BTreeMap::<u32, SourceAssoc>::new();
    for internal_tsid in 0..id_map.num_tsids() {
        let root = nwa.add_state();
        nwa.add_epsilon(
            start_state,
            root,
            all_token_weight(internal_tsid, id_map.max_token_id()),
        );

        let representative_state = id_map
            .tokenizer_states
            .representative_original_id_for_internal(internal_tsid)
            .expect("internal tokenizer state class must have a representative original state");
        merge_assoc(&mut assoc_by_state, representative_state, &[root]);
    }

    let mut builder = TerminalNwaBuilder {
        tokenizer,
        possible_matches: &mut possible_matches,
        nwa: &mut nwa,
        num_tsids: id_map.num_tsids(),
        leaf_state,
        ignore_terminal,
        representative_initial_state,
        representative_state_by_original: &representative_state_by_original,
        original_token_to_internal: &id_map.vocab_tokens.original_to_internal,
        leaf_token_ids_buffer: Vec::new(),
        reachable_weight_cache: HashMap::new(),
        leaf_weight_cache: HashMap::new(),
        transition_buffer: BTreeMap::new(),
        epsilon_buffer: BTreeMap::new(),
    };
    builder.build_from_trie(&vocab_tree.root, &assoc_by_state);
    builder.flush_transition_buffer();
    report.build_nwa_from_trie_time = phase_started_at.elapsed();
    log_terminal_profile(profile_enabled, "build_nwa_from_trie", phase_started_at);

    if should_collapse_always_allowed(vocab) {
        let phase_started_at = std::time::Instant::now();
        let always_allowed_by_label = compute_always_allowed_follows(grammar);
        let _ = collapse_always_allowed(&mut nwa, &always_allowed_by_label, grammar.num_terminals as usize);
        report.collapse_always_allowed_applied = true;
        report.collapse_always_allowed_time = phase_started_at.elapsed();
        log_terminal_profile(profile_enabled, "collapse_always_allowed", phase_started_at);
    } else if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] collapse_always_allowed_skipped=1 vocab_entries={} threshold=8192",
            vocab.entries.len(),
        );
    }

    let phase_started_at = std::time::Instant::now();
    let _ = prune_disallowed_follows(&mut nwa, grammar);
    report.prune_disallowed_follows_time = phase_started_at.elapsed();
    log_terminal_profile(profile_enabled, "prune_disallowed_follows", phase_started_at);

    report.terminal_nwa = collect_weighted_nwa_stats(&nwa);

    let phase_started_at = std::time::Instant::now();
    let determinized = determinize(&nwa)
        .expect("terminal NWA determinization failed despite acyclic token trie construction");
    report.determinize_time = phase_started_at.elapsed();
    report.terminal_dwa = collect_weighted_dwa_stats(&determinized);
    log_terminal_profile(profile_enabled, "determinize", phase_started_at);

    let phase_started_at = std::time::Instant::now();
    let dwa = minimize(&determinized);
    report.minimize_time = phase_started_at.elapsed();
    report.terminal_minimized_dwa = collect_weighted_dwa_stats(&dwa);
    log_terminal_profile(profile_enabled, "minimize", phase_started_at);
    report.total_time = total_started_at.elapsed();

    if profile_enabled {
        eprintln!(
            "[glrmask/profile][terminal_dwa] total_ms={:.3} vocab_entries={} internal_tsids={} nwa_states={} dwa_states={}",
            total_started_at.elapsed().as_secs_f64() * 1000.0,
            vocab.entries.len(),
            id_map.num_tsids(),
            report.terminal_nwa.states,
            dwa.num_states(),
        );
    }

    (dwa, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};
    use crate::compiler::grammar::model::tests::simple_ab_grammar;
    use std::collections::BTreeSet;

    fn expand_original_tokens(weight: &Weight, id_map: &InternalIdMap) -> BTreeSet<u32> {
        let mut original_tokens = BTreeSet::new();
        for internal_token_id in weight.token_union().iter() {
            if let Some(original_ids) = id_map.vocab_tokens.internal_to_originals.get(internal_token_id as usize) {
                original_tokens.extend(original_ids.iter().copied());
            } else {
                original_tokens.insert(internal_token_id);
            }
        }
        original_tokens
    }

    fn build_literal_terminal_dwa(
        rules: Vec<Rule>,
        literals: &[&[u8]],
        vocab_entries: Vec<(u32, &[u8])>,
    ) -> (DWA, InternalIdMap) {
        let grammar = GrammarDef {
            rules,
            start: 0,
            terminals: literals
                .iter()
                .enumerate()
                .map(|(id, bytes)| Terminal::Literal {
                    id: id as u32,
                    bytes: bytes.to_vec(),
                })
                .collect(),
            ..Default::default()
        };
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = crate::compiler::compile::build_tokenizer(&grammar);
        let vocab = Vocab::new(
            vocab_entries
                .into_iter()
                .map(|(id, bytes)| (id, bytes.to_vec()))
                .collect(),
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab);
        (build_terminal_dwa(&glr_grammar, &tokenizer, &vocab, &id_map, None), id_map)
    }

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

        let a_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&a_weight, &id_map);
        assert!(
            original_tokens.contains(&0),
            "terminal DWA should still accept the explicit single-terminal token 'a'"
        );
        assert!(
            original_tokens.contains(&1),
            "always-allowed collapse should make the multi-terminal token 'ab' available on the 'a' terminal word"
        );
        assert!(
            terminal_dwa.eval_word(&[0, 1]).is_empty(),
            "after collapse, the explicit multi-terminal word 'ab' should no longer be required"
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
                    utf8: true,
                },
                Terminal::Literal {
                    id: 2,
                    bytes: b"b".to_vec(),
                },
            ],
            ignore_terminal: Some(1),
            ..Default::default()
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
        let empty_original_tokens = expand_original_tokens(&empty_weight, &id_map);
        assert!(
            empty_original_tokens.contains(&0),
            "ignore-only tokens should appear in the terminal DWA start-state final weight"
        );

        let a_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&a_weight, &id_map);
        assert!(
            original_tokens.contains(&1),
            "plain non-ignore terminal tokens should still be accepted"
        );
        assert!(
            original_tokens.contains(&2),
            "tokens with ignored prefixes should also be accepted on the same terminal word"
        );
    }

    #[test]
    fn test_terminal_dwa_collapses_always_allowed_chain_to_first_terminal() {
        let (terminal_dwa, id_map) = build_literal_terminal_dwa(
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2)],
            }],
            &[b"a", b"b", b"c"],
            vec![(0, b"a"), (1, b"ab"), (2, b"abc")],
        );

        let first_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&first_weight, &id_map);
        assert!(original_tokens.contains(&0), "single-terminal token should still be accepted");
        assert!(original_tokens.contains(&1), "always-allowed suffix 'b' should collapse into the 'a' state");
        assert!(original_tokens.contains(&2), "always-allowed chain 'b' then 'c' should collapse all the way into the 'a' state");
    }

    #[test]
    fn test_terminal_dwa_does_not_collapse_non_always_follow() {
        let (terminal_dwa, id_map) = build_literal_terminal_dwa(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Terminal(1), Symbol::Terminal(2)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Terminal(3)],
                },
            ],
            &[b"a", b"b", b"c", b"d"],
            vec![(0, b"a"), (1, b"ab"), (2, b"abc"), (3, b"ad")],
        );

        let first_weight = terminal_dwa.eval_word(&[0]);
        let original_tokens = expand_original_tokens(&first_weight, &id_map);
        assert!(original_tokens.contains(&0), "the explicit 'a' token should still be accepted");
        assert!(!original_tokens.contains(&1), "'b' is only ever allowed after 'a', not always allowed, so 'ab' must not collapse");
        assert!(!original_tokens.contains(&2), "the 'abc' chain must not collapse when the first follow is not always allowed");
    }
}
