use crate::glr::grammar::{Production, Symbol, Terminal};
use crate::glr::table::{Stage7ShiftsAndReducesLookaheadValue, Table, TerminalID, NonTerminalID, ProductionID};
use bimap::BiBTreeMap;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SyntheticTerminalAnalysis {
    pub co_occurring_terminals: BTreeSet<TerminalID>,
    pub synthetic_terminal_name: String,
    pub estimated_gain: f64, // Some metric of how good this is
}

pub fn analyze_for_synthetic_terminal(
    table: &Table,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
) -> Option<SyntheticTerminalAnalysis> {
    // 1. For each distinct reduction action, find its set of lookahead terminals.
    let mut reduce_lookaheads: BTreeMap<(usize, NonTerminalID), BTreeSet<TerminalID>> = BTreeMap::new();

    for row in table.values() {
        for (&tid, action) in &row.shifts_and_reduces_full {
            let mut process_reduces = |reduces: &BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>| {
                for (&len, nts) in reduces {
                    for (&nt_id, _) in nts {
                        reduce_lookaheads.entry((len, nt_id)).or_default().insert(tid);
                    }
                }
            };
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                    reduce_lookaheads.entry((*len, *nonterminal_id)).or_default().insert(tid);
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => process_reduces(reduces),
                _ => {}
            }
        }
    }

    // 2. Find co-occurrence counts for pairs of terminals across all lookahead sets.
    let mut pair_counts: BTreeMap<(TerminalID, TerminalID), usize> = BTreeMap::new();
    for lookaheads in reduce_lookaheads.values() {
        if lookaheads.len() < 2 {
            continue;
        }
        let terminals: Vec<_> = lookaheads.iter().copied().collect();
        for i in 0..terminals.len() {
            for j in (i + 1)..terminals.len() {
                let t1 = terminals[i];
                let t2 = terminals[j];
                let pair = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                *pair_counts.entry(pair).or_default() += 1;
            }
        }
    }

    // 3. Find the best pair to synthesize.
    if let Some((&(t1, t2), &count)) = pair_counts.iter().max_by_key(|(_, &count)| count) {
        if count < 2 {
            return None;
        } // Heuristic: don't bother for pairs that only co-occur once.

        let co_occurring_terminals = BTreeSet::from([t1, t2]);
        let t1_name = terminal_map.get_by_right(&t1).unwrap().to_string().replace(|c: char| !c.is_alphanumeric(), "");
        let t2_name = terminal_map.get_by_right(&t2).unwrap().to_string().replace(|c: char| !c.is_alphanumeric(), "");
        let synthetic_terminal_name = format!("__synth_{}_{}", t1_name, t2_name);

        // Gain is just the count for now.
        let estimated_gain = count as f64;

        return Some(SyntheticTerminalAnalysis {
            co_occurring_terminals,
            synthetic_terminal_name,
            estimated_gain,
        });
    }

    None
}

pub fn apply_synthetic_terminal(
    productions: &mut Vec<Production>,
    terminal_map: &mut BiBTreeMap<Terminal, TerminalID>,
    analysis: &SyntheticTerminalAnalysis,
) -> (TerminalID, BTreeSet<TerminalID>) {
    let synth_terminal = Terminal::RegexName(analysis.synthetic_terminal_name.clone());
    let synth_symbol = Symbol::Terminal(synth_terminal.clone());

    // Add to terminal_map
    let new_tid = TerminalID(terminal_map.len());
    terminal_map.insert(synth_terminal, new_tid);

    for prod in productions.iter_mut() {
        let mut new_rhs = Vec::new();
        for symbol in &prod.rhs {
            if let Symbol::Terminal(t) = symbol {
                if let Some(tid) = terminal_map.get_by_left(t) {
                    if analysis.co_occurring_terminals.contains(tid) {
                        new_rhs.push(synth_symbol.clone());
                    }
                }
            }
            new_rhs.push(symbol.clone());
        }
        prod.rhs = new_rhs;
    }

    (new_tid, analysis.co_occurring_terminals.clone())
}
