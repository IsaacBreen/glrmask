// src/glr/synthetic.rs
use crate::glr::grammar::{Production, Symbol, Terminal};
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, Stage7ShiftsAndReducesLookaheadValue, TerminalID};
use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SyntheticTerminalAnalysis {
    /// The set of terminals that can be represented by a synthetic terminal.
    pub co_occurring_terminals: BTreeSet<TerminalID>,
    /// A score indicating the benefit of this transformation.
    /// Higher is better. It's calculated as (num_terminals_in_set - 1) * num_occurrences.
    pub score: usize,
}

/// Analyzes the parser table to find the best set of co-occurring terminals for reduction lookaheads.
pub fn analyze_for_synthetic_terminals(parser: &GLRParser) -> Option<SyntheticTerminalAnalysis> {
    // Map from a reduction action (NT, len) to a map of lookahead sets to their frequency.
    let mut reduce_lookaheads: BTreeMap<(NonTerminalID, usize), BTreeMap<BTreeSet<TerminalID>, usize>> =
        BTreeMap::new();

    for row in parser.table.values() {
        // Map from a reduction action to the set of lookahead terminals that trigger it in this state.
        let mut lookaheads_for_reduce: BTreeMap<(NonTerminalID, usize), BTreeSet<TerminalID>> =
            BTreeMap::new();

        for (&terminal_id, action) in &row.shifts_and_reduces_full {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce {
                    nonterminal_id,
                    len,
                    ..
                } => {
                    lookaheads_for_reduce
                        .entry((*nonterminal_id, *len))
                        .or_default()
                        .insert(terminal_id);
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                    for (&len, nts) in reduces {
                        for (&nonterminal_id, _) in nts {
                            lookaheads_for_reduce
                                .entry((nonterminal_id, len))
                                .or_default()
                                .insert(terminal_id);
                        }
                    }
                }
                _ => {}
            }
        }

        for (reduce_key, terminals) in lookaheads_for_reduce {
            if terminals.len() > 1 {
                *reduce_lookaheads
                    .entry(reduce_key)
                    .or_default()
                    .entry(terminals)
                    .or_default() += 1;
            }
        }
    }

    // Now, find the most promising set of terminals.
    // A good candidate is a set that appears frequently for one or more reductions.
    let mut set_scores: BTreeMap<BTreeSet<TerminalID>, usize> = BTreeMap::new();
    for lookahead_groups in reduce_lookaheads.values() {
        for (terminal_set, count) in lookahead_groups {
            // Score is how many terminals we can replace, times how many times this exact situation occurs.
            let score_increase = (terminal_set.len().saturating_sub(1)) * *count;
            *set_scores.entry(terminal_set.clone()).or_default() += score_increase;
        }
    }

    if let Some((best_set, best_score)) = set_scores.into_iter().max_by_key(|(_, score)| *score) {
        if best_score > 0 {
            return Some(SyntheticTerminalAnalysis {
                co_occurring_terminals: best_set,
                score: best_score,
            });
        }
    }

    None
}

/// Modifies a set of productions and terminal maps to include a new synthetic terminal.
/// The synthetic terminal is inserted before any terminal from the `co_occurring_terminals` set.
pub fn apply_synthetic_terminal_to_grammar(
    productions: &mut Vec<Production>,
    terminal_map: &mut BiBTreeMap<Terminal, TerminalID>,
    co_occurring_terminals: &BTreeSet<TerminalID>,
    synthetic_terminal_name: String,
) -> TerminalID {
    // 1. Create and add the new synthetic terminal.
    let synthetic_terminal = Terminal::RegexName(synthetic_terminal_name);
    let next_terminal_id = terminal_map.len();
    let synthetic_terminal_id = TerminalID(next_terminal_id);
    terminal_map.insert(synthetic_terminal.clone(), synthetic_terminal_id);
    let synthetic_symbol = Symbol::Terminal(synthetic_terminal);

    // 2. Create a set of the actual symbols to look for in the RHS of productions.
    let co_occurring_symbols: BTreeSet<Symbol> = co_occurring_terminals
        .iter()
        .map(|tid| {
            let terminal = terminal_map
                .get_by_right(tid)
                .expect("TerminalID from analysis not found in terminal map");
            Symbol::Terminal(terminal.clone())
        })
        .collect();

    // 3. Transform productions by prepending the synthetic symbol where needed.
    let mut new_productions = Vec::new();
    for prod in productions.iter() {
        let mut new_rhs = Vec::new();
        for symbol in &prod.rhs {
            if co_occurring_symbols.contains(symbol) {
                new_rhs.push(synthetic_symbol.clone());
            }
            new_rhs.push(symbol.clone());
        }
        new_productions.push(Production {
            lhs: prod.lhs.clone(),
            rhs: new_rhs,
        });
    }

    *productions = new_productions;
    synthetic_terminal_id
}