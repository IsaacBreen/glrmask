use crate::glr::grammar::{Production, Symbol, Terminal};
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, ProductionID, Stage7ShiftsAndReducesLookaheadValue, StateID, TerminalID};
use crate::json_serialization::{JSONConvertible, JSONNode};
use std::collections::{BTreeMap, BTreeSet};
use std::collections::BTreeMap as StdMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntheticTerminalInfo {
    pub synthetic_terminal: Terminal,
    pub synthetic_terminal_id: TerminalID,
    pub represented_terminals: BTreeSet<TerminalID>,
}

impl JSONConvertible for SyntheticTerminalInfo {
    fn to_json(&self) -> JSONNode {
        let mut obj = StdMap::new();
        obj.insert("synthetic_terminal".to_string(), self.synthetic_terminal.to_json());
        obj.insert("synthetic_terminal_id".to_string(), self.synthetic_terminal_id.to_json());
        obj.insert("represented_terminals".to_string(), self.represented_terminals.to_json());
        JSONNode::Object(obj)
    }

    fn from_json(node: JSONNode) -> Result<Self, String> {
        match node {
            JSONNode::Object(mut obj) => Ok(SyntheticTerminalInfo {
                synthetic_terminal: obj.remove("synthetic_terminal").ok_or("Missing synthetic_terminal")
                    .and_then(Terminal::from_json)?,
                synthetic_terminal_id: obj.remove("synthetic_terminal_id").ok_or("Missing synthetic_terminal_id")
                    .and_then(TerminalID::from_json)?,
                represented_terminals: obj.remove("represented_terminals").ok_or("Missing represented_terminals")
                    .and_then(|n| BTreeSet::<TerminalID>::from_json(n))?,
            }),
            _ => Err("Expected JSONNode::Object for SyntheticTerminalInfo".to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SyntheticTerminalAnalysis {
    pub candidate_terminals: BTreeSet<TerminalID>,
    pub score: f64,
    pub num_contexts: usize,
}

/// Analyzes the parser's table to find a candidate set of terminals for creating a synthetic terminal.
/// The goal is to find a set of terminals that frequently co-occur as lookaheads for the same reduction actions.
pub fn analyze_for_synthetic_terminals(parser: &GLRParser) -> Option<SyntheticTerminalAnalysis> {
    // 1. For each distinct reduction, find its set of lookahead terminals.
    let mut reduce_lookaheads: BTreeMap<(StateID, NonTerminalID, usize), BTreeSet<TerminalID>> = BTreeMap::new();

    for (state_id, row) in &parser.table {
        for (terminal_id, action) in &row.shifts_and_reduces_full {
            let mut process_reduces = |reduces: &BTreeMap<usize, BTreeMap<NonTerminalID, BTreeSet<ProductionID>>>| {
                for (len, nts) in reduces {
                    for (nonterminal_id, _) in nts {
                        reduce_lookaheads.entry((*state_id, *nonterminal_id, *len)).or_default().insert(*terminal_id);
                    }
                }
            };

            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, .. } => {
                    reduce_lookaheads.entry((*state_id, *nonterminal_id, *len)).or_default().insert(*terminal_id);
                },
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => process_reduces(reduces),
                _ => {}
            }
        }
    }

    // 2. Count co-occurrences of all pairs of terminals in these lookahead sets.
    let mut pair_counts: BTreeMap<(TerminalID, TerminalID), usize> = BTreeMap::new();
    for lookahead_set in reduce_lookaheads.values() {
        let terminals: Vec<_> = lookahead_set.iter().copied().collect();
        for i in 0..terminals.len() {
            for j in (i + 1)..terminals.len() {
                let t1 = terminals[i];
                let t2 = terminals[j];
                let pair = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                *pair_counts.entry(pair).or_default() += 1;
            }
        }
    }

    if pair_counts.is_empty() {
        return None;
    }

    // 3. Find the most frequent pair to seed our candidate set.
    let (&(t1, t2), _) = pair_counts.iter().max_by_key(|(_, &count)| count).unwrap();
    let mut candidate_set = BTreeSet::from([t1, t2]);

    // 4. Greedily expand the set.
    loop {
        let mut best_candidate: Option<TerminalID> = None;
        let mut best_affinity = 0;

        let all_terminals: BTreeSet<_> = parser.terminal_map.right_values().copied().collect();
        for &terminal_id in all_terminals.difference(&candidate_set) {
            let mut current_affinity = 0;
            for &member_id in &candidate_set {
                let pair = if terminal_id < member_id { (terminal_id, member_id) } else { (member_id, terminal_id) };
                current_affinity += pair_counts.get(&pair).copied().unwrap_or(0);
            }

            if current_affinity > best_affinity {
                best_affinity = current_affinity;
                best_candidate = Some(terminal_id);
            }
        }

        if let Some(best) = best_candidate {
            candidate_set.insert(best);
        } else {
            break;
        }
    }

    // 5. Score the final candidate set.
    let mut num_contexts = 0;
    for lookaheads in reduce_lookaheads.values() {
        if candidate_set.is_subset(lookaheads) {
            num_contexts += 1;
        }
    }

    if num_contexts == 0 || candidate_set.len() < 2 {
        return None;
    }

    // Score = (size - 1) * number of contexts where this saves lookaheads.
    let score = (candidate_set.len() as f64 - 1.0) * num_contexts as f64;

    Some(SyntheticTerminalAnalysis {
        candidate_terminals: candidate_set,
        score,
        num_contexts,
    })
}
