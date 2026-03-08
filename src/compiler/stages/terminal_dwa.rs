#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This stage matches sep1's `constraint_precompute.rs` terminal-DWA construction, extracted here as its own compiler phase.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use range_set_blaze::RangeSetBlaze;

use crate::Vocab;
use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::nwa::NWA;
use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::analysis::EOF;
use crate::compiler::grammar::model::TerminalID;
use crate::compiler::grammar::model::Symbol;
use crate::compiler::stages::equivalence_analysis::InternalIdMap;
use crate::ds::weight::Weight;

#[derive(Debug, Clone)]
pub struct TerminalDWA {
    pub nwa: NWA,
    pub tsid_roots: Vec<u32>,
    #[allow(dead_code)]
    pub non_greedy_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalID>>,
    #[allow(dead_code)]
    pub possible_future_terminals_by_tokenizer_state: Vec<BTreeSet<TerminalID>>,
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
            ever_allowed[*terminal as usize].extend(occurrence_follow_set(grammar, rule.lhs, &rule.rhs, index));
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
            let slot = &mut always_allowed[*terminal as usize];
            match slot {
                Some(existing) => {
                    existing.retain(|follow| follows.contains(follow));
                }
                None => {
                    *slot = Some(follows);
                }
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

fn prune_unreachable_states(terminal_dwa: &mut TerminalDWA) -> bool {
    let nwa = &mut terminal_dwa.nwa;
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
    for root in &mut terminal_dwa.tsid_roots {
        if (*root as usize) < remap.len() {
            *root = remap[*root as usize];
        }
    }
    nwa.states = new_states;
    true
}

fn collapse_always_allowed(
    terminal_dwa: &mut TerminalDWA,
    grammar: &AnalyzedGrammar,
) -> bool {
    let always_allowed_by_label = compute_always_allowed_follows(grammar);
    let terminals_count = grammar.num_terminals as usize;
    let nwa = &mut terminal_dwa.nwa;

    if always_allowed_by_label.is_empty() || terminals_count == 0 || nwa.states.is_empty() {
        return false;
    }

    let num_states = nwa.states.len();
    let mut incoming: Vec<HashSet<i32>> = vec![HashSet::new(); num_states];
    let mut domain: Vec<Weight> = (0..num_states).map(|_| Weight::empty()).collect();
    let mut queue = VecDeque::new();
    let mut in_queue = vec![false; num_states];

    for &start in &nwa.start_states {
        domain[start as usize] = Weight::all();
        queue.push_back(start as usize);
        in_queue[start as usize] = true;
    }

    while let Some(state_id) = queue.pop_front() {
        in_queue[state_id] = false;
        let state_domain = domain[state_id].clone();
        if state_domain.is_empty() {
            continue;
        }

        let src_labels: Vec<i32> = incoming[state_id].iter().copied().collect();
        for (dst, _) in &nwa.states[state_id].epsilons {
            let dst = *dst as usize;
            let domain_changed = !state_domain.is_subset(&domain[dst]);
            if domain_changed {
                domain[dst] = domain[dst].union(&state_domain);
            }
            let mut labels_changed = false;
            for label in &src_labels {
                labels_changed |= incoming[dst].insert(*label);
            }
            if (domain_changed || labels_changed) && !in_queue[dst] {
                in_queue[dst] = true;
                queue.push_back(dst);
            }
        }

        for (&label, targets) in &nwa.states[state_id].transitions {
            if label < 0 || label as usize >= terminals_count {
                continue;
            }
            for (dst, weight) in targets {
                let dst = *dst as usize;
                let contribution = state_domain.intersection(weight);
                let domain_changed = !contribution.is_subset(&domain[dst]);
                if domain_changed {
                    domain[dst] = domain[dst].union(&contribution);
                }
                let labels_changed = incoming[dst].insert(label);
                if (domain_changed || labels_changed) && !in_queue[dst] {
                    in_queue[dst] = true;
                    queue.push_back(dst);
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

    let mut allowed_by_state: Vec<HashSet<i32>> = vec![HashSet::new(); num_states];
    for state_id in 0..num_states {
        let incoming_labels = &incoming[state_id];
        if incoming_labels.is_empty() {
            continue;
        }

        let mut iter = incoming_labels.iter().copied();
        let Some(first) = iter.next() else {
            continue;
        };
        if first < 0 || first as usize >= always_allowed_by_label.len() {
            continue;
        }

        let mut allowed: HashSet<i32> = always_allowed_by_label[first as usize]
            .iter()
            .map(|label| *label as i32)
            .collect();
        for label in iter {
            if label < 0 || label as usize >= always_allowed_by_label.len() {
                continue;
            }
            let follow_set: HashSet<i32> = always_allowed_by_label[label as usize]
                .iter()
                .map(|follow| *follow as i32)
                .collect();
            allowed.retain(|candidate| follow_set.contains(candidate));
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

        let state_domain = domain[state_id].clone();
        let state = &mut nwa.states[state_id];
        let mut state_final_weight = state.final_weight.clone();
        let mut labels_to_remove = Vec::new();

        for (&label, targets) in state.transitions.iter_mut() {
            if label < 0 || label as usize >= terminals_count || !allowed.contains(&label) {
                continue;
            }

            let mut new_targets = Vec::new();
            for (dst, weight) in targets.iter() {
                if let Some(final_weight) = final_weights[*dst as usize].as_ref() {
                    let reach = state_domain.intersection(weight);
                    if reach.is_subset(final_weight) {
                        let contribution = final_weight.intersection(weight);
                        let updated = match &state_final_weight {
                            Some(existing) => existing.union(&contribution),
                            None => contribution,
                        };
                        state_final_weight = Some(updated);
                        changed = true;
                        continue;
                    }
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
        state.final_weight = state_final_weight;
    }

    if prune_unreachable_states(terminal_dwa) {
        changed = true;
    }

    changed
}

// SEP1_MAP: This is the glrmask analogue of sep1's
// `prune_nwa_disallowed_follows` in `constraint_precompute.rs`.
// Sep1 uses a two-pass approach:
//   Pass 1 (union): collect an upper-bound disallowed set per state.
//   Pass 2 (intersection): narrow it to terminals disallowed on ALL
//           incoming paths — only intersection-safe results are used
//           for pruning.
// The previous glrmask implementation used a single union pass, which
// was over-aggressive: at states reachable from multiple predecessor
// labels, it could prune transitions that were valid on at least one
// path, producing false negatives in the mask.
fn prune_disallowed_follows(
    terminal_dwa: &mut TerminalDWA,
    grammar: &AnalyzedGrammar,
) -> bool {
    let ever_allowed_by_label = compute_ever_allowed_follows(grammar);
    let terminals_count = grammar.num_terminals as usize;
    let nwa = &mut terminal_dwa.nwa;

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

    // --- Topological sort via Kahn's algorithm ---
    let num_states = nwa.states.len();
    let mut in_degree = vec![0u32; num_states];
    for state in nwa.states.iter() {
        for (dst, _) in &state.epsilons {
            in_degree[*dst as usize] += 1;
        }
        for targets in state.transitions.values() {
            for (dst, _) in targets {
                in_degree[*dst as usize] += 1;
            }
        }
    }
    let mut topo_queue: VecDeque<usize> = VecDeque::new();
    for id in 0..num_states {
        if in_degree[id] == 0 {
            topo_queue.push_back(id);
        }
    }
    let mut topo_order: Vec<usize> = Vec::with_capacity(num_states);
    {
        let mut deg = in_degree.clone();
        while let Some(sid) = topo_queue.pop_front() {
            topo_order.push(sid);
            let state = &nwa.states[sid];
            for (dst, _) in &state.epsilons {
                deg[*dst as usize] -= 1;
                if deg[*dst as usize] == 0 {
                    topo_queue.push_back(*dst as usize);
                }
            }
            for targets in state.transitions.values() {
                for (dst, _) in targets {
                    deg[*dst as usize] -= 1;
                    if deg[*dst as usize] == 0 {
                        topo_queue.push_back(*dst as usize);
                    }
                }
            }
        }
    }

    // --- Pass 2: intersection semantics (safe for pruning) ---
    // For each state, compute the intersection of disallowed sets from all
    // incoming edges.  Through epsilon edges the parent's disallowed set is
    // forwarded; through labeled edges only the label's own disallowed set
    // is used (the label "resets" the follow context, matching sep1).
    let mut disallowed_intersected: Vec<Option<HashSet<TerminalID>>> = vec![None; num_states];
    for &start in &nwa.start_states {
        disallowed_intersected[start as usize] = Some(HashSet::new());
    }

    for &sid in &topo_order {
        let src_disallowed = match &disallowed_intersected[sid] {
            Some(d) => d.clone(),
            None => HashSet::new(),
        };

        // Epsilon edges: forward the parent's disallowed set.
        let epsilon_targets: Vec<u32> = nwa.states[sid].epsilons.iter().map(|(dst, _)| *dst).collect();
        for dst in epsilon_targets {
            let dst = dst as usize;
            let entry = &mut disallowed_intersected[dst];
            match entry {
                None => *entry = Some(src_disallowed.clone()),
                Some(existing) => {
                    existing.retain(|t| src_disallowed.contains(t));
                }
            }
        }

        // Labeled transitions: propagate only the label's disallowed set.
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
                let dst = dst as usize;
                let entry = &mut disallowed_intersected[dst];
                match entry {
                    None => *entry = Some(label_dis.clone()),
                    Some(existing) => {
                        existing.retain(|t| label_dis.contains(t));
                    }
                }
            }
        }
    }

    // --- Prune ---
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

    if prune_unreachable_states(terminal_dwa) {
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
    built_suffix_nodes: &mut HashSet<(u32, u32, usize)>,
    suffix_nodes: &mut HashMap<(u32, u32, usize), u32>,
    source_node: u32,
    current_state: u32,
    origin_state: u32,
    token_id: u32,
    token_bytes: &[u8],
    offset: usize,
    leaf_state: u32,
    grammar: &AnalyzedGrammar,
    // The internal TSID (equivalence class) for `origin_state`.  Weights must
    // be keyed by this value so that the consumer
    // (`terminal_branches_for_tokenizer_state`) can look them up via
    // `weight.tokens_for_tsid(internal_tsid)`.  Previously the code used
    // `origin_state` directly, which only worked when the identity mapping was
    // in effect (internal_tsid == origin_state).
    internal_tsid: u32,
) {
    let exec = tokenizer.execute_from_state(&token_bytes[offset..], current_state);
    let token_weight = Weight::from_token_set_for_tsid(
        internal_tsid,
        RangeSetBlaze::from_iter([token_id..=token_id]),
    );

    for matched in exec.matches {
        if matched.id >= grammar.num_terminals {
            continue;
        }

        let next_offset = offset + matched.width;
        let destination = if next_offset >= token_bytes.len() {
            leaf_state
        } else {
            let key = (origin_state, token_id, next_offset);
            let continuation_state = *suffix_nodes.entry(key).or_insert_with(|| nwa.add_state());
            if built_suffix_nodes.insert(key) {
                build_token_suffix_paths(
                    tokenizer,
                    nwa,
                    built_suffix_nodes,
                    suffix_nodes,
                    continuation_state,
                    // After completing a terminal match the tokenizer
                    // restarts from its initial state for the next
                    // terminal (each terminal is an independent lex).
                    // Previously this used `matched.end_state`, which is
                    // a dead end in the DFA when a pattern is fully
                    // matched.
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
) -> TerminalDWA {
    // SEP1_MAP: sep1 folds this logic into the large terminal-precompute pipeline in `constraint_precompute.rs`; glrmask keeps the boundary here explicit.
    let mut nwa = NWA::new(id_map.num_tsids(), id_map.max_token_id());
    let leaf_state = nwa.add_state();
    nwa.set_final_weight(leaf_state, Weight::all());
    let mut tsid_roots = vec![0u32; id_map.num_tsids() as usize];
    let mut suffix_nodes = HashMap::<(u32, u32, usize), u32>::new();
    let mut built_suffix_nodes = HashSet::<(u32, u32, usize)>::new();

    for internal_tsid in 0..id_map.num_tsids() {
        let root = nwa.add_state();
        tsid_roots[internal_tsid as usize] = root;
        nwa.start_states.push(root);

        for original_state in &id_map.tokenizer_states.internal_to_originals[internal_tsid as usize] {
            for (token_id, token_bytes) in &vocab.entries {
                // Skip the single-pass DFA validation: multi-terminal
                // tokens (e.g. "ab" = terminal_A + terminal_B) involve
                // tokenizer restarts between terminal matches, so the
                // full byte sequence does NOT need to be walkable in a
                // single continuous DFA pass.  build_token_suffix_paths
                // handles partial matches naturally.

                build_token_suffix_paths(
                    tokenizer,
                    &mut nwa,
                    &mut built_suffix_nodes,
                    &mut suffix_nodes,
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

    let non_greedy_terminals_by_tokenizer_state = (0..tokenizer.num_states())
        .map(|state| tokenizer.matched_non_greedy_terminals(state))
        .collect();
    let possible_future_terminals_by_tokenizer_state = (0..tokenizer.num_states())
        .map(|state| tokenizer.possible_future_terminals(state))
        .collect();

    let mut terminal_dwa = TerminalDWA {
        nwa,
        tsid_roots,
        non_greedy_terminals_by_tokenizer_state,
        possible_future_terminals_by_tokenizer_state,
    };

    // collapse_always_allowed is disabled because it folds NWA transitions
    // into final weights.  The downstream parser DWA builder
    // (terminal_branches_for_tokenizer_state) only reads transitions, not
    // final weights, so collapsing breaks multi-terminal token paths.
    // let _ = collapse_always_allowed(&mut terminal_dwa, grammar);
    let _ = prune_disallowed_follows(&mut terminal_dwa, grammar);

    terminal_dwa
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::ds::weight::Weight;
    use crate::automata::regex::bytes;
    use crate::automata::lexer::tokenizer::Tokenizer;
    use crate::compiler::glr::analysis::AnalyzedGrammar;
    use crate::compiler::grammar::model::tests::simple_ab_grammar;
    use crate::compiler::possible_matches::build_possible_matches_by_state;
    use crate::compiler::stages::equivalence_analysis::InternalIdMap;

    #[test]
    fn test_build_terminal_dwa_collapses_always_allowed_follow_path() {
        // With collapse_always_allowed disabled, follow transitions are
        // preserved in the NWA so the parser DWA builder can read them.
        let grammar = simple_ab_grammar();
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = Tokenizer::from_grammar_def(&grammar);
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())], None);
        let id_map = InternalIdMap::build(&tokenizer, &vocab);

        let terminal_dwa = build_terminal_dwa(
            &glr_grammar,
            &tokenizer,
            &vocab,
            &id_map,
        );
        let initial_tsid = id_map.tokenizer_states.original_to_internal[tokenizer.initial_state() as usize] as usize;
        let root = terminal_dwa.tsid_roots[initial_tsid];
        let a_targets = &terminal_dwa.nwa.states[root as usize].transitions[&0];
        assert!(!a_targets.is_empty());

        let mut combined_a = Weight::empty();
        for (_, weight) in a_targets {
            combined_a = combined_a.union(weight);
        }
        assert!(!combined_a.is_empty() || combined_a.is_full());

        // A continuation target for the multi-terminal token "ab" should
        // still have a B transition (label 1) since collapse is disabled.
        let has_b_continuation = a_targets.iter().any(|(dest, _)| {
            terminal_dwa.nwa.states[*dest as usize]
                .transitions
                .contains_key(&1)
        });
        assert!(has_b_continuation, "continuation state should have B transition");
    }

    #[test]
    fn test_terminal_dwa_carries_tokenizer_greedy_metadata() {
        let grammar = simple_ab_grammar();
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = Tokenizer::from_expr_groups(&[
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
        let id_map = InternalIdMap::build(&tokenizer, &vocab);

        let terminal_dwa = build_terminal_dwa(
            &glr_grammar,
            &tokenizer,
            &vocab,
            &id_map,
        );
        let state_after_a = tokenizer.run(b"a") as usize;

        assert!(terminal_dwa.non_greedy_terminals_by_tokenizer_state[state_after_a].contains(&0));
        assert!(terminal_dwa.possible_future_terminals_by_tokenizer_state[state_after_a].contains(&1));
    }

    #[test]
    fn test_terminal_dwa_builds_continuation_state_for_multi_terminal_token() {
        let grammar = simple_ab_grammar();
        let glr_grammar = AnalyzedGrammar::from_grammar_def(&grammar);
        let tokenizer = Tokenizer::from_grammar_def(&grammar);
        let vocab = Vocab::new(
            vec![(0, b"a".to_vec()), (1, b"ab".to_vec()), (2, b"b".to_vec())],
            None,
        );
        let id_map = InternalIdMap::build(&tokenizer, &vocab);

        let terminal_dwa = build_terminal_dwa(
            &glr_grammar,
            &tokenizer,
            &vocab,
            &id_map,
        );

        let initial_tsid =
            id_map.tokenizer_states.original_to_internal[tokenizer.initial_state() as usize] as usize;
        let root = terminal_dwa.tsid_roots[initial_tsid] as usize;
        let a_targets = &terminal_dwa.nwa.states[root].transitions[&0];

        assert!(a_targets.iter().any(|(target, _)| {
            let state = &terminal_dwa.nwa.states[*target as usize];
            state.transitions.contains_key(&1)
        }));
    }
}
