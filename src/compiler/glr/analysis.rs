#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]


use std::collections::{BTreeSet, BTreeMap};

use crate::compiler::grammar::model::{GrammarDef, NonterminalID, Rule, Symbol, TerminalID};


pub const EOF: TerminalID = u32::MAX;


#[derive(Debug, Clone)]
pub struct AnalyzedGrammar {
    
    pub rules: Vec<Rule>,
    #[allow(dead_code)]
    
    pub start: NonterminalID,
    
    pub num_terminals: u32,
    
    pub num_nonterminals: u32,
    
    pub nullable: BTreeSet<NonterminalID>,
    
    pub first: Vec<BTreeSet<TerminalID>>,
    
    pub follow: Vec<BTreeSet<TerminalID>>,
}

impl AnalyzedGrammar {
    
    
    
    
    
    
    
    
    
    
    
    pub fn from_grammar_def(g: &GrammarDef) -> Self {
        let mut rules = Vec::with_capacity(g.rules.len() + 1);
        let augmented_start = g.num_nonterminals();
        rules.push(Rule {
            lhs: augmented_start,
            rhs: vec![Symbol::Nonterminal(g.start)],
        });
        rules.extend(g.rules.iter().cloned());

        let num_nonterminals = augmented_start + 1;
        let nullable = compute_nullable(&rules, num_nonterminals);
        let first = compute_first(&rules, num_nonterminals, &nullable);
        let follow = compute_follow(&rules, num_nonterminals, augmented_start, &first, &nullable);

        Self {
            rules,
            start: augmented_start,
            num_terminals: g.num_terminals(),
            num_nonterminals,
            nullable,
            first,
            follow,
        }
    }

    /// Debug check: asserts the grammar has no right recursion, no indirect left
    /// recursion, and no nullable nonterminals.  These are preconditions required
    /// by the terminal-characterization stage (`characterize.rs`).
    ///
    /// Violations are reported as an `Err` with a structured diagnostic message
    /// listing every violated condition and the nonterminals involved.
    ///
    /// **Note:** This check is intentionally separate from the panic inside
    /// `characterize_terminal`.  By running it *before* characterisation we
    /// produce an actionable description of the grammar shape that caused
    /// the failure, making it easier to locate the rule that needs
    /// transformation.
    pub fn debug_check_grammar_preconditions(&self) -> Result<(), String> {
        let mut violations: Vec<String> = Vec::new();

        // 1. Nullable nonterminals (skip the augmented start symbol)
        if !self.nullable.is_empty() {
            let ids: Vec<u32> = self.nullable.iter()
                .filter(|&&nt| nt < self.num_nonterminals - 1) // skip augmented start
                .copied()
                .collect();
            if !ids.is_empty() {
                violations.push(format!(
                    "nullable nonterminals detected: {:?}. \
                     Rules with ε-productions or all-nullable RHS create \
                     reduce chains that the characterisation stage cannot \
                     handle when combined with recursion.",
                    ids,
                ));
            }
        }

        // 2. Right recursion (direct or indirect)
        let rr_graph = build_right_reachability_graph(&self.rules, &self.nullable);
        if let Some(cycle) = find_indirect_rr_cycle(&rr_graph) {
            violations.push(format!(
                "right-recursive cycle detected: {:?}. \
                 Right recursion causes unbounded reduce chains in \
                 terminal characterisation.  Convert to left recursion \
                 or inline the cycle.",
                cycle,
            ));
        }

        // 3. Indirect left recursion
        let lr_graph = build_left_reachability_graph(&self.rules, &self.nullable);
        if let Some(cycle) = find_indirect_lr_cycle(&lr_graph) {
            // Direct left recursion (cycle of length 1, A→A...) is fine for
            // GLR; only indirect cycles (length ≥ 2) are flagged.
            if cycle.len() >= 2 {
                violations.push(format!(
                    "indirect left-recursive cycle detected: {:?}. \
                     Indirect left recursion may create unbounded GSS \
                     growth.  Inline or rewrite the cycle.",
                    cycle,
                ));
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "grammar precondition violations ({} found):\n{}",
                violations.len(),
                violations.iter()
                    .enumerate()
                    .map(|(i, v)| format!("  {}. {}", i + 1, v))
                    .collect::<Vec<_>>()
                    .join("\n"),
            ))
        }
    }

    
    pub fn first_of_seq(&self, seq: &[Symbol]) -> BTreeSet<TerminalID> {
        let mut out = BTreeSet::new();
        for symbol in seq {
            match symbol {
                Symbol::Terminal(terminal) => {
                    out.insert(*terminal);
                    return out;
                }
                Symbol::Nonterminal(nonterminal) => {
                    if let Some(first) = self.first.get(*nonterminal as usize) {
                        out.extend(first.iter().copied());
                    }
                    if !self.nullable.contains(nonterminal) {
                        return out;
                    }
                }
            }
        }
        out
    }

    
    pub fn seq_is_nullable(&self, seq: &[Symbol]) -> bool {
        seq.iter().all(|symbol| match symbol {
            Symbol::Terminal(_) => false,
            Symbol::Nonterminal(nonterminal) => self.nullable.contains(nonterminal),
        })
    }
}


pub fn normalize_for_mask(g: &GrammarDef) -> GrammarDef {
    g.clone()
}


#[allow(dead_code)]
pub(crate) fn eliminate_direct_left_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) {
    let _ = rules;
    let _ = fresh_nt;
}


pub(crate) fn eliminate_right_recursion(
    rules: &mut Vec<Rule>,
    fresh_nt: &mut impl FnMut() -> NonterminalID,
) {
    let _ = rules;
    let _ = fresh_nt;
}


fn max_nt_id(rules: &[Rule]) -> u32 {
    rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .max()
        .unwrap_or(0)
}


fn build_right_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    let mut graph = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
    for rule in rules {
        let suffix = rule
            .rhs
            .iter()
            .rev()
            .take_while(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                Symbol::Terminal(_) => false,
            })
            .collect::<Vec<_>>();
        for symbol in suffix.into_iter().rev() {
            if let Symbol::Nonterminal(nonterminal) = symbol {
                graph.entry(rule.lhs).or_default().insert(*nonterminal);
            }
        }
        if let Some(Symbol::Nonterminal(nonterminal)) = rule.rhs.last() {
            graph.entry(rule.lhs).or_default().insert(*nonterminal);
        }
    }
    graph
}


fn find_indirect_rr_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    fn dfs(
        node: NonterminalID,
        graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
        colors: &mut BTreeMap<NonterminalID, u8>,
        stack: &mut Vec<NonterminalID>,
    ) -> Option<Vec<NonterminalID>> {
        colors.insert(node, 1);
        stack.push(node);
        for &next in graph.get(&node).into_iter().flatten() {
            match colors.get(&next).copied().unwrap_or(0) {
                0 => {
                    if let Some(cycle) = dfs(next, graph, colors, stack) {
                        return Some(cycle);
                    }
                }
                1 => {
                    if let Some(start) = stack.iter().position(|&entry| entry == next) {
                        return Some(stack[start..].to_vec());
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        colors.insert(node, 2);
        None
    }

    let mut colors = BTreeMap::new();
    let mut stack = Vec::new();
    for &node in graph.keys() {
        if colors.get(&node).copied().unwrap_or(0) == 0 {
            if let Some(cycle) = dfs(node, graph, &mut colors, &mut stack) {
                return Some(cycle);
            }
        }
    }
    None
}


/// Build a graph where an edge A → B means B appears at the left edge of
/// a production for A (possibly after nullable symbols).
fn build_left_reachability_graph(
    rules: &[Rule],
    nullable: &BTreeSet<NonterminalID>,
) -> BTreeMap<NonterminalID, BTreeSet<NonterminalID>> {
    let mut graph = BTreeMap::<NonterminalID, BTreeSet<NonterminalID>>::new();
    for rule in rules {
        for symbol in &rule.rhs {
            match symbol {
                Symbol::Nonterminal(nonterminal) => {
                    graph.entry(rule.lhs).or_default().insert(*nonterminal);
                    if !nullable.contains(nonterminal) {
                        break;
                    }
                }
                Symbol::Terminal(_) => break,
            }
        }
    }
    graph
}

/// Find an indirect left-recursive cycle (length ≥ 2) in the left-reachability
/// graph.  Direct self-loops (A → A …) are excluded — they are fine for GLR.
fn find_indirect_lr_cycle(
    graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
) -> Option<Vec<NonterminalID>> {
    // Reuse the same DFS as find_indirect_rr_cycle but filter to length ≥ 2.
    fn dfs(
        node: NonterminalID,
        graph: &BTreeMap<NonterminalID, BTreeSet<NonterminalID>>,
        colors: &mut BTreeMap<NonterminalID, u8>,
        stack: &mut Vec<NonterminalID>,
    ) -> Option<Vec<NonterminalID>> {
        colors.insert(node, 1);
        stack.push(node);
        for &next in graph.get(&node).into_iter().flatten() {
            match colors.get(&next).copied().unwrap_or(0) {
                0 => {
                    if let Some(cycle) = dfs(next, graph, colors, stack) {
                        return Some(cycle);
                    }
                }
                1 => {
                    if let Some(start) = stack.iter().position(|&entry| entry == next) {
                        let cycle = stack[start..].to_vec();
                        if cycle.len() >= 2 {
                            return Some(cycle);
                        }
                    }
                }
                _ => {}
            }
        }
        stack.pop();
        colors.insert(node, 2);
        None
    }

    let mut colors = BTreeMap::new();
    let mut stack = Vec::new();
    for &node in graph.keys() {
        if colors.get(&node).copied().unwrap_or(0) == 0 {
            if let Some(cycle) = dfs(node, graph, &mut colors, &mut stack) {
                return Some(cycle);
            }
        }
    }
    None
}


fn inline_right_end(
    rules: &mut Vec<Rule>,
    from_nt: NonterminalID,
    to_nt: NonterminalID,
    nullable: &BTreeSet<NonterminalID>,
) {
    let _ = rules;
    let _ = from_nt;
    let _ = to_nt;
    let _ = nullable;
}


fn is_direct_right_recursive(rule: &Rule) -> bool {
    matches!(rule.rhs.last(), Some(Symbol::Nonterminal(nonterminal)) if *nonterminal == rule.lhs)
}


fn resolve_direct_rr_single_nt(
    rules: &mut Vec<Rule>,
    nt: NonterminalID,
    new_nt: NonterminalID,
) {
    let _ = rules;
    let _ = nt;
    let _ = new_nt;
}


pub(crate) fn inline_epsilon_rules(rules: &[Rule]) -> Vec<Rule> {
    rules.to_vec()
}


fn compute_nullable(rules: &[Rule], num_nt: u32) -> BTreeSet<NonterminalID> {
    let mut nullable = BTreeSet::new();
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            if rule.lhs >= num_nt {
                continue;
            }
            let rhs_nullable = rule.rhs.is_empty()
                || rule.rhs.iter().all(|symbol| match symbol {
                    Symbol::Terminal(_) => false,
                    Symbol::Nonterminal(nonterminal) => nullable.contains(nonterminal),
                });
            if rhs_nullable && nullable.insert(rule.lhs) {
                changed = true;
            }
        }
    }
    nullable
}


fn compute_first(
    rules: &[Rule],
    num_nt: u32,
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BTreeSet<TerminalID>> {
    let mut first = vec![BTreeSet::new(); num_nt as usize];
    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            let lhs = rule.lhs as usize;
            for symbol in &rule.rhs {
                match symbol {
                    Symbol::Terminal(terminal) => {
                        changed |= first[lhs].insert(*terminal);
                        break;
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        let additions = first[*nonterminal as usize].clone();
                        let old_len = first[lhs].len();
                        first[lhs].extend(additions);
                        changed |= first[lhs].len() != old_len;
                        if !nullable.contains(nonterminal) {
                            break;
                        }
                    }
                }
            }
        }
    }
    first
}


fn compute_follow(
    rules: &[Rule],
    num_nt: u32,
    start: NonterminalID,
    first: &[BTreeSet<TerminalID>],
    nullable: &BTreeSet<NonterminalID>,
) -> Vec<BTreeSet<TerminalID>> {
    let mut follow = vec![BTreeSet::new(); num_nt as usize];
    if let Some(start_follow) = follow.get_mut(start as usize) {
        start_follow.insert(EOF);
    }

    let mut changed = true;
    while changed {
        changed = false;
        for rule in rules {
            let lhs_follow = follow[rule.lhs as usize].clone();
            for (index, symbol) in rule.rhs.iter().enumerate() {
                let Symbol::Nonterminal(nonterminal) = symbol else {
                    continue;
                };

                let suffix = &rule.rhs[index + 1..];
                let mut additions = BTreeSet::new();
                let mut suffix_nullable = true;
                for suffix_symbol in suffix {
                    match suffix_symbol {
                        Symbol::Terminal(terminal) => {
                            additions.insert(*terminal);
                            suffix_nullable = false;
                            break;
                        }
                        Symbol::Nonterminal(next_nonterminal) => {
                            additions.extend(first[*next_nonterminal as usize].iter().copied());
                            if !nullable.contains(next_nonterminal) {
                                suffix_nullable = false;
                                break;
                            }
                        }
                    }
                }
                if suffix_nullable {
                    additions.extend(lhs_follow.iter().copied());
                }

                let target = &mut follow[*nonterminal as usize];
                let old_len = target.len();
                target.extend(additions);
                changed |= target.len() != old_len;
            }
        }
    }

    follow
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::grammar::model::tests::*;
    use crate::compiler::grammar::model::{GrammarDef, Rule, Symbol, Terminal};

    fn term(id: u32, name: &str) -> Terminal {
        Terminal::Literal { id, bytes: name.as_bytes().to_vec() }
    }

    #[test]
    fn test_glr_grammar_simple() {
        let g = AnalyzedGrammar::from_grammar_def(&simple_ab_grammar());
        
        assert_eq!(g.rules.len(), 2);
        assert_eq!(g.num_nonterminals, 2); 
        assert_eq!(g.num_terminals, 2);
        assert!(g.nullable.is_empty());
        
        assert!(g.first[0].contains(&0));
        assert!(!g.first[0].contains(&1));
        
        assert!(g.follow[0].contains(&EOF));
    }

    #[test]
    fn test_glr_grammar_choice() {
        let g = AnalyzedGrammar::from_grammar_def(&choice_grammar());
        
        assert!(g.first[0].contains(&0));
        assert!(g.first[0].contains(&1));
    }

    #[test]
    fn test_glr_grammar_two_nt() {
        let g = AnalyzedGrammar::from_grammar_def(&two_nt_grammar());
        
        assert!(g.first[0].contains(&0)); 
        assert!(g.first[1].contains(&0)); 
        
        assert!(g.follow[1].contains(&1)); 
    }

    /// Simple non-recursive grammar passes all checks.
    #[test]
    fn test_preconditions_simple_grammar_passes() {
        let g = AnalyzedGrammar::from_grammar_def(&simple_ab_grammar());
        assert!(g.debug_check_grammar_preconditions().is_ok());
    }

    /// Grammar with nullable nonterminal is flagged.
    #[test]
    fn test_preconditions_nullable_detected() {
        // S -> A | ε  ;  A -> 'a'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1)] },
                Rule { lhs: 0, rhs: vec![] }, // S -> ε
                Rule { lhs: 1, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nullable"));
    }

    /// Grammar with right recursion is flagged.
    #[test]
    fn test_preconditions_right_recursion_detected() {
        // S -> 'a' S | 'a'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("right-recursive"));
    }

    /// Grammar with indirect left recursion is flagged.
    #[test]
    fn test_preconditions_indirect_left_recursion_detected() {
        // A -> B 'a'  ;  B -> A 'b'
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)] },
                Rule { lhs: 1, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(1)] },
            ],
            start: 0,
            terminals: vec![term(0, "a"), term(1, "b")],
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("indirect left-recursive"));
    }

    /// Direct left recursion (S -> S 'a' | 'a') should NOT be flagged as
    /// "indirect left recursion" — it is safe for GLR.
    #[test]
    fn test_preconditions_direct_left_recursion_ok() {
        // S -> S 'a' | 'a'  (direct LR, should only flag nullable if empty alt is absent)
        let gdef = GrammarDef {
            rules: vec![
                Rule { lhs: 0, rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)] },
                Rule { lhs: 0, rhs: vec![Symbol::Terminal(0)] },
            ],
            start: 0,
            terminals: vec![term(0, "a")],
        };
        let g = AnalyzedGrammar::from_grammar_def(&gdef);
        let result = g.debug_check_grammar_preconditions();
        // Should NOT contain "indirect left-recursive"
        match &result {
            Ok(()) => {} // fine — no violations
            Err(msg) => assert!(!msg.contains("indirect left-recursive"),
                "direct left recursion should not be flagged as indirect: {}", msg),
        }
    }}
