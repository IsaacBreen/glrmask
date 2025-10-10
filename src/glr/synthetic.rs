// src/glr/synthetic.rs
use crate::glr::parser::GLRParser;
use crate::glr::table::{Reduce, Stage7ShiftsAndReducesLookaheadValue, TerminalID};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone)]
pub struct SyntheticTerminalAnalysis {
    pub members: BTreeSet<TerminalID>,
    pub score: f64,
}

/// Analyzes the parser table to find sets of terminals that frequently co-occur as lookaheads for the same reduction.
/// Such sets are good candidates for being represented by a single synthetic terminal to reduce lookahead branching.
pub fn analyze_synthetic_terminals(parser: &GLRParser) -> Vec<SyntheticTerminalAnalysis> {
    let mut reduce_lookaheads: BTreeMap<Reduce, BTreeSet<TerminalID>> = BTreeMap::new();
    let mut reduce_counts: BTreeMap<Reduce, usize> = BTreeMap::new();

    for row in parser.table.values() {
        for (&tid, action) in &row.shifts_and_reduces_full {
            let mut process_reduces = |reduces: &BTreeMap<usize, BTreeMap<_, _>>| {
                for (&len, nts) in reduces {
                    for (&nt_id, pids) in nts {
                        let reduce = Reduce { nonterminal_id: nt_id, len, production_ids: pids.clone() };
                        reduce_lookaheads.entry(reduce.clone()).or_default().insert(tid);
                        *reduce_counts.entry(reduce).or_default() += 1;
                    }
                }
            };

            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                    let reduce = Reduce { nonterminal_id: *nonterminal_id, len: *len, production_ids: production_ids.clone() };
                    reduce_lookaheads.entry(reduce.clone()).or_default().insert(tid);
                    *reduce_counts.entry(reduce).or_default() += 1;
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                    process_reduces(reduces);
                }
                _ => {}
            }
        }
    }

    let mut analyses: Vec<SyntheticTerminalAnalysis> = Vec::new();
    for (reduce, members) in reduce_lookaheads {
        if members.len() > 1 {
            let count = reduce_counts.get(&reduce).copied().unwrap_or(0);
            // Score is how many lookaheads we save, times how many places this reduction occurs.
            let score = ((members.len() - 1) * count) as f64;
            if score > 0.0 {
                analyses.push(SyntheticTerminalAnalysis { members, score });
            }
        }
    }

    // Sort by score, descending, to prioritize the most impactful sets.
    analyses.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    analyses
}
