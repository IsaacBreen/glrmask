use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

use bimap::BiBTreeMap;

use crate::glr::grammar::{Production, Symbol, Terminal};
use crate::glr::table::{
    Table, Stage7ShiftsAndReducesLookaheadValue, TerminalID, NonTerminalID, ProductionID,
};

/// A unique identity for a specific kind of reduce action, across states.
/// It groups (NT, len, PIDs) so that if a state has a reduce with these exact values,
/// we consider it the "same" reduction for co-occurrence analysis.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ReduceSignature {
    nt: NonTerminalID,
    len: usize,
    pids: BTreeSet<ProductionID>,
}

/// Represents a proposed co-occurring set (we pick pairs for simplicity)
/// together with stats about how often it appears and its estimated benefit.
#[derive(Debug, Clone)]
pub struct SyntheticCandidate {
    /// The chosen members (terminals); we return pairs in this implementation.
    pub members: BTreeSet<Terminal>,
    /// Number of signature-groups that contain all `members`.
    pub support: usize,
    /// Total number of signature-groups considered in the analysis.
    pub total_signature_groups: usize,
    /// Estimated total token-lookahead savings if this set is introduced once:
    /// counted as the number of signature-groups covered by the set (one unit per group).
    pub estimated_gain_tokens: f64,
}

/// A concrete plan to introduce a synthetic terminal for a chosen set.
#[derive(Debug, Clone)]
pub struct SyntheticTerminalPlan {
    /// The synthetic terminal (RegexName) we will inject in the grammar before each member.
    pub synthetic: Terminal,
    /// The members (real terminals) that the synthetic terminal stands for.
    pub members: BTreeSet<Terminal>,
    /// Same stats as the candidate for transparency.
    pub support: usize,
    pub total_signature_groups: usize,
    pub estimated_gain_tokens: f64,
}

/// A mapping to track which synthetic terminal expands to which concrete terminals.
#[derive(Debug, Clone)]
pub struct SyntheticTerminalMapping {
    pub synthetic: Terminal,
    pub members: BTreeSet<Terminal>,
}

/// Compute reduce lookahead sets per (state, signature). Returns all terminal sets (as Terminals).
fn collect_signature_terminal_sets(
    table: &Table,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
) -> Vec<BTreeSet<Terminal>> {
    // Build a per-row map: signature -> set of TerminalIDs that cause that signature in this row.
    let mut all_sets: Vec<BTreeSet<Terminal>> = Vec::new();
    for (_state_id, row) in table {
        let mut sig_map: BTreeMap<ReduceSignature, BTreeSet<TerminalID>> = BTreeMap::new();
        for (tid, action) in &row.shifts_and_reduces_full {
            match action {
                Stage7ShiftsAndReducesLookaheadValue::Reduce { nonterminal_id, len, production_ids } => {
                    let sig = ReduceSignature {
                        nt: *nonterminal_id,
                        len: *len,
                        pids: production_ids.clone(),
                    };
                    sig_map.entry(sig).or_default().insert(*tid);
                }
                Stage7ShiftsAndReducesLookaheadValue::Split { reduces, .. } => {
                    for (len, nts) in reduces {
                        for (nt, pids) in nts {
                            let sig = ReduceSignature {
                                nt: *nt,
                                len: *len,
                                pids: pids.clone(),
                            };
                            sig_map.entry(sig).or_default().insert(*tid);
                        }
                    }
                }
                Stage7ShiftsAndReducesLookaheadValue::Shift(_) => {
                    // ignore
                }
            }
        }
        // Convert per-row signature groups to Terminal sets and store them
        for (_sig, tids) in sig_map {
            if tids.is_empty() {
                continue;
            }
            let mut ts: BTreeSet<Terminal> = BTreeSet::new();
            for tid in tids {
                if let Some(term) = terminal_map.get_by_right(&tid) {
                    ts.insert(term.clone());
                }
            }
            if !ts.is_empty() {
                all_sets.push(ts);
            }
        }
    }
    all_sets
}

/// Finds the best co-occurring pair of terminals across all reduce-signature terminal sets.
/// Heuristic: choose the pair with the largest support (number of signature-groups containing both).
pub fn analyze_best_cooccurring_pair(
    table: &Table,
    terminal_map: &BiBTreeMap<Terminal, TerminalID>,
) -> Option<SyntheticCandidate> {
    let all_sets = collect_signature_terminal_sets(table, terminal_map);
    if all_sets.is_empty() {
        return None;
    }

    // Universe of terminals appearing in any set
    let mut universe: BTreeSet<Terminal> = BTreeSet::new();
    for s in &all_sets {
        universe.extend(s.iter().cloned());
    }
    let universe_vec: Vec<Terminal> = universe.into_iter().collect();

    // Count pair co-occurrence
    let mut pair_counts: BTreeMap<(Terminal, Terminal), usize> = BTreeMap::new();
    for set in &all_sets {
        if set.len() < 2 { continue; }
        // enumerate pairs in this set
        let v: Vec<&Terminal> = set.iter().collect();
        for i in 0..v.len() {
            for j in (i + 1)..v.len() {
                let a = v[i].clone();
                let b = v[j].clone();
                let key = if a <= b { (a, b) } else { (b, a) };
                *pair_counts.entry(key).or_default() += 1;
            }
        }
    }

    // Find best pair by support
    let mut best: Option<((Terminal, Terminal), usize)> = None;
    for (pair, &cnt) in &pair_counts {
        match best {
            None => best = Some((pair.clone(), cnt)),
            Some((_, best_cnt)) => {
                if cnt > best_cnt {
                    best = Some((pair.clone(), cnt));
                }
            }
        }
    }

    let ((t1, t2), support) = best?;
    let mut members = BTreeSet::new();
    members.insert(t1);
    members.insert(t2);

    let total_signature_groups = all_sets.len();
    let estimated_gain_tokens = support as f64;

    Some(SyntheticCandidate {
        members,
        support,
        total_signature_groups,
        estimated_gain_tokens,
    })
}

/// Generates a stable-looking synthetic terminal name from the set members.
/// The resulting name is a valid identifier and should not require quoting in formatter.
fn make_synthetic_name(members: &BTreeSet<Terminal>) -> String {
    // Build a deterministic string and hash it.
    let mut hasher = DefaultHasher::new();
    let mut parts: Vec<String> = Vec::new();
    for t in members {
        match t {
            Terminal::RegexName(s) => {
                parts.push(format!("R:{}", s));
            }
            Terminal::Literal(bytes) => {
                // simple hex prefix, truncated for name brevity
                let head = bytes.iter().take(4).map(|b| format!("{:02X}", b)).collect::<String>();
                parts.push(format!("L:{}", head));
            }
        }
    }
    parts.sort();
    parts.join("|").hash(&mut hasher);
    let h = hasher.finish();
    format!("__SYN_PAIR_{:016X}", h)
}

/// Turns a candidate into a concrete plan by choosing a synthetic name (or using a provided one).
pub fn propose_plan_for_pair(
    candidate: &SyntheticCandidate,
    custom_name: Option<&str>,
) -> SyntheticTerminalPlan {
    let synthetic_name = custom_name.map(|s| s.to_string()).unwrap_or_else(|| make_synthetic_name(&candidate.members));
    let synthetic = Terminal::RegexName(synthetic_name);
    SyntheticTerminalPlan {
        synthetic,
        members: candidate.members.clone(),
        support: candidate.support,
        total_signature_groups: candidate.total_signature_groups,
        estimated_gain_tokens: candidate.estimated_gain_tokens,
    }
}

/// Applies the synthetic terminal plan to a grammar, inserting the synthetic terminal
/// before each occurrence of any member terminal. Returns the updated productions and
/// a mapping so the caller can remember how to generate the synthetic token at runtime.
///
/// IMPORTANT:
/// - You must ensure your tokenizer/input stream injects the synthetic token
///   immediately before any real terminal in `plan.members`. This is outside parser code.
pub fn apply_synthetic_to_grammar(
    productions: &[Production],
    plan: &SyntheticTerminalPlan,
) -> (Vec<Production>, SyntheticTerminalMapping) {
    let members: BTreeSet<Terminal> = plan.members.iter().cloned().collect();
    let synthetic_symbol = Symbol::Terminal(plan.synthetic.clone());

    let mut new_productions = Vec::with_capacity(productions.len());
    for p in productions {
        let mut new_rhs = Vec::with_capacity(p.rhs.len() * 2);
        for sym in &p.rhs {
            match sym {
                Symbol::Terminal(t) if members.contains(t) => {
                    // Insert synthetic token immediately before the member terminal
                    new_rhs.push(synthetic_symbol.clone());
                    new_rhs.push(sym.clone());
                }
                _ => {
                    new_rhs.push(sym.clone());
                }
            }
        }
        new_productions.push(Production {
            lhs: p.lhs.clone(),
            rhs: new_rhs,
        });
    }

    (
        new_productions,
        SyntheticTerminalMapping {
            synthetic: plan.synthetic.clone(),
            members,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glr::grammar::{prod, t, nt};
    use crate::glr::table::{assign_terminal_ids, assign_non_terminal_ids, generate_glr_parser_with_maps};
    use std::collections::BTreeMap as StdMap;
    use crate::glr::parser::ActionFn;
    use bimap::BiBTreeMap;

    #[test]
    fn synthetic_insertion_basic() {
        // Minimal grammar: S -> A
        // A -> 'x' | 'y' | 'z'
        // This test focuses on insertion logic, not on analysis.
        let productions = vec![
            prod("S", vec![nt("A")]),
            prod("A", vec![t("x")]),
            prod("A", vec![t("y")]),
            prod("A", vec![t("z")]),
        ];
        // Plan: synthetic for {x, y}
        let mut members = BTreeSet::new();
        members.insert(Terminal::RegexName("x".into()));
        members.insert(Terminal::RegexName("y".into()));
        let candidate = SyntheticCandidate {
            members: members.clone(),
            support: 1,
            total_signature_groups: 1,
            estimated_gain_tokens: 1.0,
        };
        let plan = propose_plan_for_pair(&candidate, Some("__SYN_TEST"));
        let (new_prods, mapping) = apply_synthetic_to_grammar(&productions, &plan);

        assert_eq!(mapping.members.len(), 2);
        assert!(mapping.members.contains(&Terminal::RegexName("x".into())));
        assert!(mapping.members.contains(&Terminal::RegexName("y".into())));
        assert_eq!(mapping.synthetic, Terminal::RegexName("__SYN_TEST".into()));

        // Verify that S -> A remains unchanged
        assert_eq!(new_prods[0].rhs.len(), 1);
        // Verify that terminals x and y have synthetic before them; z is unchanged.
        // A -> __SYN_TEST 'x'
        assert_eq!(format!("{}", new_prods[1]), "A -> __SYN_TEST x");
        // A -> __SYN_TEST 'y'
        assert_eq!(format!("{}", new_prods[2]), "A -> __SYN_TEST y");
        // A -> 'z'
        assert_eq!(format!("{}", new_prods[3]), "A -> z");
    }
}
