//! Grammar analysis helpers: terminal coloring, follow-set computations.

use std::collections::BTreeSet;

use crate::compiler::glr::analysis::{AnalyzedGrammar, EOF};
use crate::compiler::glr::table::GLRTable;
use crate::grammar::flat::{Symbol, TerminalID};
use crate::ds::bitset::BitSet;

use super::types::{ColorId, TerminalColoring};

/// Compute a graph coloring of terminals based on GLR table row adjacency.
/// Terminals in the same action row get different colors.
pub(crate) fn compute_terminal_coloring(table: &GLRTable) -> TerminalColoring {
    let num_terminals = table.num_terminals as usize;
    if num_terminals <= 1 {
        return TerminalColoring::identity(num_terminals);
    }

    let mut adjacency = vec![BitSet::new(num_terminals); num_terminals];
    for row in &table.action {
        let terminals: Vec<usize> = row
            .keys()
            .copied()
            .filter(|&terminal| (terminal as usize) < num_terminals)
            .map(|terminal| terminal as usize)
            .collect();
        for left_idx in 0..terminals.len() {
            let left = terminals[left_idx];
            for &right in &terminals[left_idx + 1..] {
                adjacency[left].set(right);
                adjacency[right].set(left);
            }
        }
    }

    let degrees: Vec<usize> = adjacency.iter().map(BitSet::count_ones).collect();
    let mut terminal_to_color = vec![ColorId::MAX; num_terminals];
    let mut neighbor_colors = vec![BitSet::new(num_terminals); num_terminals];
    let mut num_colors = 0usize;

    for _ in 0..num_terminals {
        let next_terminal = (0..num_terminals)
            .filter(|&terminal| terminal_to_color[terminal] == ColorId::MAX)
            .max_by(|&left, &right| {
                neighbor_colors[left]
                    .count_ones()
                    .cmp(&neighbor_colors[right].count_ones())
                    .then_with(|| degrees[left].cmp(&degrees[right]))
                    .then_with(|| right.cmp(&left))
            })
            .expect("there should always be an uncolored terminal");

        let mut color = 0usize;
        while neighbor_colors[next_terminal].contains(color) {
            color += 1;
        }
        terminal_to_color[next_terminal] = color as ColorId;
        num_colors = num_colors.max(color + 1);

        for neighbor in adjacency[next_terminal].iter_ones() {
            if terminal_to_color[neighbor] == ColorId::MAX {
                neighbor_colors[neighbor].set(color);
            }
        }
    }

    TerminalColoring {
        terminal_to_color,
        num_colors,
    }
}

/// For each terminal, collect the set of terminals that can ever follow it
/// in any rule derivation.
pub(crate) fn compute_ever_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
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
pub(crate) fn compute_always_allowed_follows(grammar: &AnalyzedGrammar) -> Vec<Vec<TerminalID>> {
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
