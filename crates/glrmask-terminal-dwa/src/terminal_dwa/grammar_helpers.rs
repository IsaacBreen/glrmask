//! Grammar analysis helpers: terminal coloring, follow-set computations.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::compiler::glr::analysis::AnalyzedGrammar;
use crate::compiler::glr::table::GLRTable;
use crate::ds::bitset::BitSet;
use crate::grammar::flat::{Symbol, TerminalID};

use super::types::{ColorId, TerminalColoring};

/// Convert parser-visible disallowed follows into a token-path relation where
/// the ignore terminal is transparent: ignore may follow anything, anything may
/// follow ignore, and ignore may follow itself.
///
/// Keep parser-visible follow tables raw for grammar/table semantics. Use this
/// only for byte/tokenizer path analysis.
pub fn ignore_transparent_disallowed_follows(
    disallowed_follows: &BTreeMap<u32, BitSet>,
    ignore_terminal: Option<TerminalID>,
) -> BTreeMap<u32, BitSet> {
    let Some(ignore_terminal) = ignore_terminal else {
        return disallowed_follows.clone();
    };

    let mut adjusted = disallowed_follows.clone();
    adjusted.remove(&ignore_terminal);
    for bits in adjusted.values_mut() {
        if (ignore_terminal as usize) < bits.len() {
            bits.clear(ignore_terminal as usize);
        }
    }
    adjusted.retain(|_, bits| !bits.is_zero());
    adjusted
}

/// Compute terminal colors so terminals in the same action row get different
/// colors.
pub fn compute_terminal_coloring(table: &GLRTable) -> TerminalColoring {
    let num_terminals = table.num_terminals as usize;
    if num_terminals <= 1 {
        return TerminalColoring::identity(num_terminals);
    }

    let mut row_terminals = Vec::new();
    let mut rows_by_terminal = vec![Vec::new(); num_terminals];
    for row in &table.action {
        let mut terminals: Vec<usize> = row
            .keys()
            .filter(|&terminal| (terminal as usize) < num_terminals)
            .map(|terminal| terminal as usize)
            .collect();
        if terminals.is_empty() {
            continue;
        }
        terminals.sort_unstable();
        terminals.dedup();

        let row_index = row_terminals.len();
        for &terminal in &terminals {
            rows_by_terminal[terminal].push(row_index);
        }
        row_terminals.push(terminals);
    }

    if row_terminals.is_empty() {
        return TerminalColoring {
            terminal_to_color: vec![0; num_terminals],
            num_colors: 1,
        };
    }

    let mut terminal_order: Vec<usize> = (0..num_terminals).collect();
    terminal_order.sort_unstable_by(|&left, &right| {
        rows_by_terminal[right]
            .len()
            .cmp(&rows_by_terminal[left].len())
            .then_with(|| left.cmp(&right))
    });

    let mut row_used_colors = vec![Vec::<ColorId>::new(); row_terminals.len()];
    let mut terminal_to_color = vec![0; num_terminals];
    let mut color_seen_at_epoch = Vec::<usize>::new();
    let mut epoch = 0usize;
    let mut num_colors = 0usize;

    for terminal in terminal_order {
        if rows_by_terminal[terminal].is_empty() {
            terminal_to_color[terminal] = 0;
            continue;
        }

        epoch = epoch.wrapping_add(1);
        if epoch == 0 {
            color_seen_at_epoch.fill(0);
            epoch = 1;
        }

        for &row_index in &rows_by_terminal[terminal] {
            for &color in &row_used_colors[row_index] {
                let color = color as usize;
                if color >= color_seen_at_epoch.len() {
                    color_seen_at_epoch.resize(color + 1, 0);
                }
                color_seen_at_epoch[color] = epoch;
            }
        }

        let mut color = 0usize;
        while color < color_seen_at_epoch.len() && color_seen_at_epoch[color] == epoch {
            color += 1;
        }
        if color == color_seen_at_epoch.len() {
            color_seen_at_epoch.push(0);
        }

        let color_id = color as ColorId;
        terminal_to_color[terminal] = color_id;
        num_colors = num_colors.max(color + 1);

        for &row_index in &rows_by_terminal[terminal] {
            row_used_colors[row_index].push(color_id);
        }
    }

    TerminalColoring {
        terminal_to_color,
        num_colors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::glr::table::testing::build_test_table;
    use crate::compiler::glr::table::Action;

    fn assert_row_colors_are_unique(table: &GLRTable, coloring: &TerminalColoring) {
        for row in &table.action {
            let mut colors = BTreeSet::new();
            for terminal in row.keys() {
                if terminal < table.num_terminals {
                    let color = coloring.color_for(terminal);
                    assert!(
                        colors.insert(color),
                        "terminal {terminal} reused color {color} in one action row"
                    );
                }
            }
        }
    }

    #[test]
    fn terminal_coloring_keeps_action_row_terminals_distinct() {
        let table = build_test_table(
            3,
            6,
            &[
                &[(0, Action::Accept), (2, Action::Accept), (4, Action::Accept)],
                &[(1, Action::Accept), (2, Action::Accept)],
                &[(1, Action::Accept), (3, Action::Accept), (5, Action::Accept)],
            ],
            &[&[], &[], &[]],
        );

        let coloring = compute_terminal_coloring(&table);

        assert_row_colors_are_unique(&table, &coloring);
        assert!(coloring.num_colors <= 3);
    }

    #[test]
    fn terminal_coloring_handles_sparse_high_terminal_count() {
        let table = build_test_table(
            2,
            50_000,
            &[
                &[(10, Action::Accept), (20_000, Action::Accept)],
                &[(20_000, Action::Accept), (49_999, Action::Accept)],
            ],
            &[&[], &[]],
        );

        let coloring = compute_terminal_coloring(&table);

        assert_eq!(coloring.terminal_to_color.len(), 50_000);
        assert_row_colors_are_unique(&table, &coloring);
        assert!(coloring.num_colors <= 2);
    }

    #[test]
    fn direct_regular_follow_sets_match_generic_analysis() {
        use crate::grammar::flat::{
            DirectRegularAutomaton, DirectRegularState, GrammarDef, Rule, Symbol, Terminal,
        };

        let mut grammar = GrammarDef {
            rules: vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(2), Symbol::Nonterminal(1)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(3)],
                },
            ],
            start: 0,
            terminals: (0..4)
                .map(|id| Terminal::Literal {
                    id,
                    bytes: vec![b'a' + id as u8],
                })
                .collect(),
            direct_regular_automaton: Some(DirectRegularAutomaton {
                states: vec![
                    DirectRegularState {
                        transitions: BTreeMap::from([(0, vec![1]), (1, vec![2])]),
                        ..DirectRegularState::default()
                    },
                    DirectRegularState {
                        transitions: BTreeMap::from([(2, vec![1]), (3, vec![2])]),
                        ..DirectRegularState::default()
                    },
                    DirectRegularState {
                        is_accepting: true,
                        ..DirectRegularState::default()
                    },
                ],
                start_states: vec![0],
            }),
            ..GrammarDef::default()
        };

        let direct = AnalyzedGrammar::from_grammar_def(&grammar);
        let direct_ever = compute_ever_allowed_follows(&direct);
        let direct_always = compute_always_allowed_follows(&direct);

        grammar.direct_regular_automaton = None;
        let generic = AnalyzedGrammar::from_grammar_def(&grammar);
        assert_eq!(direct_ever, compute_ever_allowed_follows(&generic));
        assert_eq!(direct_always, compute_always_allowed_follows(&generic));
    }
}

fn direct_regular_follow_sets(
    grammar: &AnalyzedGrammar,
) -> Option<(Vec<Vec<TerminalID>>, Vec<Vec<TerminalID>>)> {
    let automaton = grammar.direct_regular_automaton.as_ref()?;
    let state_count = automaton.states.len();
    let terminal_count = grammar.num_terminals as usize;

    // Compute epsilon-closure output labels once over the SCC condensation
    // instead of running one whole-graph DFS per source state. The condensation
    // is acyclic, so a reverse topological union gives every SCC the exact set
    // of terminal labels reachable through zero or more epsilon edges.
    let mut reverse_epsilons = vec![Vec::<u32>::new(); state_count];
    for (source, state) in automaton.states.iter().enumerate() {
        for &target in &state.epsilons {
            if let Some(reverse) = reverse_epsilons.get_mut(target as usize) {
                reverse.push(source as u32);
            }
        }
    }

    let mut visited = vec![false; state_count];
    let mut postorder = Vec::with_capacity(state_count);
    let mut dfs = Vec::<(u32, usize)>::new();
    for root in 0..state_count as u32 {
        if visited[root as usize] {
            continue;
        }
        visited[root as usize] = true;
        dfs.push((root, 0));
        while let Some((state_id, next_edge)) = dfs.last_mut() {
            let epsilons = &automaton.states[*state_id as usize].epsilons;
            let mut descended = false;
            while *next_edge < epsilons.len() {
                let target = epsilons[*next_edge];
                *next_edge += 1;
                let target_index = target as usize;
                if target_index < state_count && !visited[target_index] {
                    visited[target_index] = true;
                    dfs.push((target, 0));
                    descended = true;
                    break;
                }
            }
            if !descended {
                let (finished, _) = dfs.pop().expect("DFS frame must exist");
                postorder.push(finished);
            }
        }
    }

    let mut scc_by_state = vec![u32::MAX; state_count];
    let mut scc_count = 0u32;
    let mut stack = Vec::<u32>::new();
    for &root in postorder.iter().rev() {
        if scc_by_state[root as usize] != u32::MAX {
            continue;
        }
        scc_by_state[root as usize] = scc_count;
        stack.push(root);
        while let Some(state_id) = stack.pop() {
            for &predecessor in &reverse_epsilons[state_id as usize] {
                if scc_by_state[predecessor as usize] == u32::MAX {
                    scc_by_state[predecessor as usize] = scc_count;
                    stack.push(predecessor);
                }
            }
        }
        scc_count += 1;
    }
    let scc_count = scc_count as usize;

    let mut scc_edges = vec![Vec::<u32>::new(); scc_count];
    let mut closure_labels = vec![BitSet::new(terminal_count); scc_count];
    for (source, state) in automaton.states.iter().enumerate() {
        let source_scc = scc_by_state[source] as usize;
        for &terminal in state.transitions.keys() {
            if (terminal as usize) < terminal_count {
                closure_labels[source_scc].set(terminal as usize);
            }
        }
        for &target in &state.epsilons {
            let Some(&target_scc) = scc_by_state.get(target as usize) else {
                continue;
            };
            if target_scc as usize != source_scc {
                scc_edges[source_scc].push(target_scc);
            }
        }
    }
    for edges in &mut scc_edges {
        edges.sort_unstable();
        edges.dedup();
    }

    let mut indegree = vec![0u32; scc_count];
    for edges in &scc_edges {
        for &target in edges {
            indegree[target as usize] += 1;
        }
    }
    let mut ready = VecDeque::new();
    for (scc, &degree) in indegree.iter().enumerate() {
        if degree == 0 {
            ready.push_back(scc as u32);
        }
    }
    let mut topological = Vec::with_capacity(scc_count);
    while let Some(scc) = ready.pop_front() {
        topological.push(scc);
        for &target in &scc_edges[scc as usize] {
            let degree = &mut indegree[target as usize];
            *degree -= 1;
            if *degree == 0 {
                ready.push_back(target);
            }
        }
    }
    debug_assert_eq!(topological.len(), scc_count);

    for &source in topological.iter().rev() {
        for &target in &scc_edges[source as usize] {
            let source = source as usize;
            let target = target as usize;
            if source < target {
                let (left, right) = closure_labels.split_at_mut(target);
                left[source].union_with(&right[0]);
            } else {
                let (left, right) = closure_labels.split_at_mut(source);
                right[0].union_with(&left[target]);
            }
        }
    }

    let mut ever = vec![BitSet::new(terminal_count); terminal_count];
    let mut always = vec![None::<BitSet>; terminal_count];
    for state in &automaton.states {
        for (&terminal, targets) in &state.transitions {
            if (terminal as usize) >= terminal_count {
                continue;
            }
            for &target in targets {
                let Some(&target_scc) = scc_by_state.get(target as usize) else {
                    continue;
                };
                let follows = &closure_labels[target_scc as usize];
                ever[terminal as usize].union_with(follows);
                match &mut always[terminal as usize] {
                    None => always[terminal as usize] = Some(follows.clone()),
                    Some(existing) => existing.intersect_with(follows),
                }
            }
        }
    }

    Some((
        ever.into_iter()
            .map(|set| set.iter_ones().map(|terminal| terminal as TerminalID).collect())
            .collect(),
        always
            .into_iter()
            .map(|set| {
                set.into_iter()
                    .flat_map(|set| set.iter_ones().collect::<Vec<_>>())
                    .map(|terminal| terminal as TerminalID)
                    .collect()
            })
            .collect(),
    ))
}

/// Compute both existential and universal terminal-follow relations together.
/// Direct-regular grammars share one epsilon-closure traversal between the two
/// results instead of repeating the same whole-automaton analysis.
pub fn compute_allowed_follow_sets(
    grammar: &AnalyzedGrammar,
) -> (Vec<Vec<TerminalID>>, Vec<Vec<TerminalID>>) {
    direct_regular_follow_sets(grammar).unwrap_or_else(|| {
        (
            compute_ever_allowed_follows(grammar),
            compute_always_allowed_follows(grammar),
        )
    })
}

/// For each terminal, collect the set of terminals that can ever follow it
/// in any rule derivation.
pub fn compute_ever_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
    if let Some((ever, _)) = direct_regular_follow_sets(grammar) {
        return ever;
    }
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

/// For each terminal, the terminals that ALWAYS follow it in every occurrence.
pub fn compute_always_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
    if let Some((_, always)) = direct_regular_follow_sets(grammar) {
        return always;
    }

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
                    follows.extend(
                        first
                            .iter_ones()
                            .filter(|bit| *bit < grammar.num_terminals as usize)
                            .map(|bit| bit as TerminalID),
                    );
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
            follows.extend(
                follow
                    .iter_ones()
                    .filter(|bit| *bit < grammar.num_terminals as usize)
                    .map(|bit| bit as TerminalID),
            );
        }
    }

    follows
}
