//! GLR grammar with numeric IDs, FIRST and FOLLOW sets.
//!
//! Converts a [`GrammarDef`] into a canonical form with an augmented start
//! rule (S' → S), computes nullable nonterminals, FIRST and FOLLOW sets
//! for SLR(1) table construction.

use std::collections::BTreeSet;

use super::super::grammar_def::{GrammarDef, NonterminalId, Rule, Symbol, TerminalId};

/// EOF pseudo-terminal. Must not collide with any real terminal.
pub const EOF: TerminalId = u32::MAX;

/// An augmented GLR grammar ready for table generation.
#[derive(Debug, Clone)]
pub struct GlrGrammar {
    /// All production rules (augmented start is at index 0).
    pub rules: Vec<Rule>,
    /// Start nonterminal of the augmented grammar.
    pub start: NonterminalId,
    /// Number of real terminals (not counting EOF).
    pub num_terminals: u32,
    /// Number of nonterminals (including augmented start).
    pub num_nonterminals: u32,
    /// Nullable nonterminals.
    pub nullable: BTreeSet<NonterminalId>,
    /// FIRST(A) for each nonterminal A (indexed by NT id).
    pub first: Vec<BTreeSet<TerminalId>>,
    /// FOLLOW(A) for each nonterminal A (indexed by NT id).
    pub follow: Vec<BTreeSet<TerminalId>>,
}

impl GlrGrammar {
    /// Build from a [`GrammarDef`].
    ///
    /// Augments the grammar with a new start rule `S' → S`.
    pub fn from_grammar_def(g: &GrammarDef) -> Self {
        let orig_nt_count = g.num_nonterminals();
        let num_terminals = g.num_terminals();

        // Augmented start NT = one past the highest original NT id.
        let aug_start = orig_nt_count;
        let num_nonterminals = orig_nt_count + 1;

        // Augmented rule: S' → S   (index 0)
        let mut rules = vec![Rule {
            lhs: aug_start,
            rhs: vec![Symbol::Nonterminal(g.start)],
        }];
        rules.extend_from_slice(&g.rules);

        let nullable = compute_nullable(&rules, num_nonterminals);
        let first = compute_first(&rules, num_nonterminals, &nullable);
        let follow = compute_follow(&rules, num_nonterminals, aug_start, &first, &nullable);

        GlrGrammar {
            rules,
            start: aug_start,
            num_terminals,
            num_nonterminals,
            nullable,
            first,
            follow,
        }
    }

    /// FIRST set for a sequence of symbols.
    pub fn first_of_seq(&self, seq: &[Symbol]) -> BTreeSet<TerminalId> {
        let mut result = BTreeSet::new();
        let n = self.num_nonterminals as usize;
        for sym in seq {
            match sym {
                Symbol::Terminal(t) => {
                    result.insert(*t);
                    return result;
                }
                Symbol::Nonterminal(nt) => {
                    if (*nt as usize) < n {
                        result.extend(&self.first[*nt as usize]);
                    }
                    if !self.nullable.contains(nt) {
                        return result;
                    }
                }
            }
        }
        result
    }

    /// Whether a symbol sequence can derive ε.
    pub fn seq_is_nullable(&self, seq: &[Symbol]) -> bool {
        seq.iter().all(|s| match s {
            Symbol::Terminal(_) => false,
            Symbol::Nonterminal(nt) => self.nullable.contains(nt),
        })
    }
}

// ---------------------------------------------------------------------------
// Nullable (fixed-point)
// ---------------------------------------------------------------------------

fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalId> {
    let mut nullable = BTreeSet::new();
    loop {
        let mut changed = false;
        for r in rules {
            if r.lhs < num_nt
                && !nullable.contains(&r.lhs)
                && r.rhs.iter().all(|s| match s {
                    Symbol::Terminal(_) => false,
                    Symbol::Nonterminal(nt) => nullable.contains(nt),
                })
            {
                nullable.insert(r.lhs);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    nullable
}

// ---------------------------------------------------------------------------
// FIRST sets (fixed-point)
// ---------------------------------------------------------------------------

fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    nullable: &BTreeSet<NonterminalId>,
) -> Vec<BTreeSet<TerminalId>> {
    let n = num_nt as usize;
    let mut first: Vec<BTreeSet<TerminalId>> = vec![BTreeSet::new(); n];
    loop {
        let mut changed = false;
        for r in rules {
            let lhs = r.lhs as usize;
            if lhs >= n {
                continue;
            }
            for sym in &r.rhs {
                match sym {
                    Symbol::Terminal(t) => {
                        if first[lhs].insert(*t) {
                            changed = true;
                        }
                        break;
                    }
                    Symbol::Nonterminal(nt) => {
                        let nt_idx = *nt as usize;
                        if nt_idx < n {
                            let ext: Vec<TerminalId> = first[nt_idx].iter().copied().collect();
                            for t in ext {
                                if first[lhs].insert(t) {
                                    changed = true;
                                }
                            }
                        }
                        if !nullable.contains(nt) {
                            break;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    first
}

// ---------------------------------------------------------------------------
// FOLLOW sets (fixed-point)
// ---------------------------------------------------------------------------

fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    start: NonterminalId,
    first: &[BTreeSet<TerminalId>],
    nullable: &BTreeSet<NonterminalId>,
) -> Vec<BTreeSet<TerminalId>> {
    let n = num_nt as usize;
    let mut follow: Vec<BTreeSet<TerminalId>> = vec![BTreeSet::new(); n];

    // $ ∈ FOLLOW(S').
    if (start as usize) < n {
        follow[start as usize].insert(EOF);
    }

    loop {
        let mut changed = false;
        for r in rules {
            let lhs = r.lhs as usize;
            for (i, sym) in r.rhs.iter().enumerate() {
                let Symbol::Nonterminal(nt) = sym else {
                    continue;
                };
                let nt_idx = *nt as usize;
                if nt_idx >= n {
                    continue;
                }

                // β = rhs[i+1 ..]
                let beta = &r.rhs[i + 1..];
                let mut beta_nullable = true;
                for s in beta {
                    match s {
                        Symbol::Terminal(t) => {
                            if follow[nt_idx].insert(*t) {
                                changed = true;
                            }
                            beta_nullable = false;
                            break;
                        }
                        Symbol::Nonterminal(b) => {
                            let b_idx = *b as usize;
                            if b_idx < n {
                                let ext: Vec<TerminalId> = first[b_idx].iter().copied().collect();
                                for t in ext {
                                    if follow[nt_idx].insert(t) {
                                        changed = true;
                                    }
                                }
                            }
                            if !nullable.contains(b) {
                                beta_nullable = false;
                                break;
                            }
                        }
                    }
                }

                // If β is empty or entirely nullable → FOLLOW(nt) ∪= FOLLOW(lhs)
                if beta_nullable && lhs < n {
                    let ext: Vec<TerminalId> = follow[lhs].iter().copied().collect();
                    for t in ext {
                        if follow[nt_idx].insert(t) {
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    follow
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar_def::tests::*;

    #[test]
    fn test_glr_grammar_simple() {
        let g = GlrGrammar::from_grammar_def(&simple_ab_grammar());
        // Augmented: S' → S;  S → a b
        assert_eq!(g.rules.len(), 2);
        assert_eq!(g.num_nonterminals, 2); // S, S'
        assert_eq!(g.num_terminals, 2);
        assert!(g.nullable.is_empty());
        // FIRST(S) = {a}
        assert!(g.first[0].contains(&0));
        assert!(!g.first[0].contains(&1));
        // FOLLOW(S) = {$}
        assert!(g.follow[0].contains(&EOF));
    }

    #[test]
    fn test_glr_grammar_choice() {
        let g = GlrGrammar::from_grammar_def(&choice_grammar());
        // S → a | b  →  FIRST(S) = {a, b}
        assert!(g.first[0].contains(&0));
        assert!(g.first[0].contains(&1));
    }

    #[test]
    fn test_glr_grammar_two_nt() {
        let g = GlrGrammar::from_grammar_def(&two_nt_grammar());
        // S → A b, A → a.  FIRST(A) = {a}, FIRST(S) = FIRST(A) = {a}
        assert!(g.first[0].contains(&0)); // FIRST(S) has 'a'
        assert!(g.first[1].contains(&0)); // FIRST(A) has 'a'
        // FOLLOW(A) = FIRST(b...) = {b}
        assert!(g.follow[1].contains(&1)); // 'b' follows A
    }
}
