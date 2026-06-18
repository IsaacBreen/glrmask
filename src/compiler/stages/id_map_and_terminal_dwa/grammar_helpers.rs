//! Grammar analysis helpers: terminal coloring, follow-set computations.

use std::collections::{BTreeMap, BTreeSet};

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
pub(crate) fn ignore_transparent_disallowed_follows(
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
pub(crate) fn compute_terminal_coloring(table: &GLRTable) -> TerminalColoring {
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
