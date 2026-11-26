//! Tests of the retained preprocessing and resource guards.

use super::*;
use crate::grammar::flat::Terminal;

fn nt(id: u32) -> Symbol {
    Symbol::Nonterminal(id)
}

fn t(id: u32) -> Symbol {
    Symbol::Terminal(id)
}

fn rule(lhs: u32, rhs: Vec<Symbol>) -> Rule {
    Rule { lhs, rhs }
}

/// Independent finite-word fixed point (same universe semantics as the
/// normalization suite's helper: `max_len` bounds the tested word universe,
/// not derivation depth).
fn bounded_language(rules: &[Rule], start: u32, max_len: usize) -> BTreeSet<Vec<u32>> {
    let n = rules
        .iter()
        .flat_map(|rule| {
            std::iter::once(rule.lhs).chain(rule.rhs.iter().filter_map(|symbol| match symbol {
                Symbol::Nonterminal(nonterminal) => Some(*nonterminal),
                Symbol::Terminal(_) => None,
            }))
        })
        .chain(std::iter::once(start))
        .max()
        .unwrap_or(0) as usize
        + 1;
    let mut languages = vec![BTreeSet::<Vec<u32>>::new(); n];

    loop {
        let mut changed = false;
        for rule in rules {
            let mut words = BTreeSet::from([Vec::<u32>::new()]);
            for symbol in &rule.rhs {
                let suffixes = match symbol {
                    Symbol::Terminal(terminal) => BTreeSet::from([vec![*terminal]]),
                    Symbol::Nonterminal(nonterminal) => languages[*nonterminal as usize].clone(),
                };
                let mut joined = BTreeSet::new();
                for prefix in &words {
                    for suffix in &suffixes {
                        if prefix.len() + suffix.len() <= max_len {
                            let mut word = prefix.clone();
                            word.extend_from_slice(suffix);
                            joined.insert(word);
                        }
                    }
                }
                words = joined;
            }
            let target = &mut languages[rule.lhs as usize];
            let old_len = target.len();
            target.extend(words);
            changed |= target.len() != old_len;
        }
        if !changed {
            return languages[start as usize].clone();
        }
    }
}

fn grammar(rules: Vec<Rule>, start: u32, num_terminals: u32) -> GrammarDef {
    GrammarDef {
        rules,
        start,
        terminals: (0..num_terminals)
            .map(|id| Terminal::Literal {
                id,
                bytes: vec![b'x'],
            })
            .collect(),
        ..Default::default()
    }
}

#[test]
fn nullable_repetition_preprocessing_stays_linear() {
    let mut rhs = vec![nt(1); 40];
    rhs.push(t(1));
    let rules = vec![rule(0, rhs), rule(1, vec![]), rule(1, vec![t(0)])];
    let source = grammar(rules.clone(), 0, 2);
    let prepared = preprocess_to_grammar_def(
        &source,
        &CompileLimits::default(),
        &mut CompileBudget::new(),
    ).unwrap();
    assert!(prepared.rules.len() < 500);
    assert_eq!(
        bounded_language(&rules, 0, 6),
        bounded_language(&prepared.rules, prepared.start, 6),
    );
}
#[test]
fn unit_scc_linear_preprocessing_collapses_long_cycle_and_charges_budget() {
    let n = 20_000;
    let mut rules = (0..n).map(|i| rule(i, vec![nt((i + 1) % n)])).collect::<Vec<_>>();
    rules.push(rule(n - 1, vec![t(0)]));
    let mut budget = CompileBudget::new();
    let (collapsed, start) = collapse_unit_sccs(rules.clone(), n - 1, &CompileLimits::default(), &mut budget).unwrap();
    assert_eq!(start, 0);
    assert_eq!(dedup_rules(collapsed), dedup_rules(vec![rule(0, vec![nt(0)]), rule(0, vec![t(0)])]));
    assert!(budget.candidate_work >= n as u64);
    let mut budget = CompileBudget::new();
    let limits = CompileLimits { max_candidate_work: 1, ..Default::default() };
    assert!(matches!(collapse_unit_sccs(rules, 0, &limits, &mut budget),
        Err(CompileDecline::ResourceLimit { stage: Stage::Preprocess, .. })));
}
