mod tests {
    use super::*;
    use crate::grammar::flat::{GrammarDef, Terminal};

    fn analyzed_grammar(rules: Vec<Rule>, start: NonterminalID) -> AnalyzedGrammar {
        AnalyzedGrammar::from_grammar_def(&GrammarDef {
            rules,
            start,
            terminals: vec![Terminal::Literal {
                id: 0,
                bytes: b"a".to_vec(),
            }],
            ..GrammarDef::default()
        })
    }

    fn bounded_language(
        rules: &[Rule],
        start: NonterminalID,
        num_nonterminals: u32,
        max_len: usize,
    ) -> BTreeSet<Vec<TerminalID>> {
        fn rhs_language(
            rhs: &[Symbol],
            languages: &[BTreeSet<Vec<TerminalID>>],
            max_len: usize,
        ) -> BTreeSet<Vec<TerminalID>> {
            let mut acc = BTreeSet::from([Vec::new()]);
            for symbol in rhs {
                let part = match symbol {
                    Symbol::Terminal(terminal) => {
                        BTreeSet::from([vec![*terminal]])
                    }
                    Symbol::Nonterminal(nonterminal) => {
                        languages[*nonterminal as usize].clone()
                    }
                };
                let mut next = BTreeSet::new();
                for prefix in &acc {
                    for suffix in &part {
                        if prefix.len() + suffix.len() <= max_len {
                            let mut combined = prefix.clone();
                            combined.extend(suffix);
                            next.insert(combined);
                        }
                    }
                }
                acc = next;
                if acc.is_empty() {
                    break;
                }
            }
            acc
        }

        let mut languages = vec![BTreeSet::new(); num_nonterminals as usize];
        loop {
            let mut changed = false;
            for rule in rules {
                let derived = rhs_language(&rule.rhs, &languages, max_len);
                let target = &mut languages[rule.lhs as usize];
                let old_len = target.len();
                target.extend(derived);
                changed |= target.len() != old_len;
            }
            if !changed {
                break;
            }
        }
        languages[start as usize].clone()
    }

    #[test]
    fn table_build_normal_form_rejects_nullable_zero_length_rules() {
        let grammar = analyzed_grammar(vec![Rule { lhs: 0, rhs: Vec::new() }], 0);

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("nullable nonterminals reachable"));
        assert!(error.contains("zero-length productions reachable"));
    }

    #[test]
    fn table_build_normal_form_rejects_direct_right_recursion() {
        let grammar = analyzed_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0), Symbol::Nonterminal(0)],
                },
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
        );

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("right-recursive cycle detected"));
    }

    #[test]
    fn table_build_normal_form_rejects_indirect_left_recursion() {
        let grammar = analyzed_grammar(
            vec![
                Rule {
                    lhs: 0,
                    rhs: vec![Symbol::Nonterminal(1), Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Nonterminal(0), Symbol::Terminal(0)],
                },
                Rule {
                    lhs: 1,
                    rhs: vec![Symbol::Terminal(0)],
                },
            ],
            0,
        );

        let error = grammar.check_table_build_normal_form().unwrap_err();
        assert!(error.contains("indirect left-recursive cycle detected"));
    }

    #[test]
    fn table_build_normal_form_accepts_simple_nonnullable_grammar() {
        let grammar = analyzed_grammar(
            vec![Rule {
                lhs: 0,
                rhs: vec![Symbol::Terminal(0)],
            }],
            0,
        );

        assert!(grammar.check_table_build_normal_form().is_ok());
    }

    #[test]
    fn nontrivial_sccs_include_multiple_disjoint_cycles_and_skip_self_loops() {
        let graph = BTreeMap::from([
            (0, BTreeSet::from([1])),
            (1, BTreeSet::from([0])),
            (2, BTreeSet::from([3])),
            (3, BTreeSet::from([4])),
            (4, BTreeSet::from([2])),
            (5, BTreeSet::from([5])),
            (6, BTreeSet::from([7])),
        ]);

        let mut sccs = find_nontrivial_sccs(&graph);
        sccs.sort_by_key(|component| component.iter().next().copied().unwrap_or(u32::MAX));

        assert_eq!(sccs, vec![BTreeSet::from([0, 1]), BTreeSet::from([2, 3, 4])]);
    }

    #[test]
    fn nullable_run_compression_preserves_nullable_only_nonempty_derivations() {
        let rules = vec![
            Rule {
                lhs: 3,
                rhs: vec![Symbol::Terminal(1)],
            },
            Rule { lhs: 2, rhs: vec![] },
            Rule {
                lhs: 2,
                rhs: vec![Symbol::Nonterminal(2), Symbol::Nonterminal(3)],
            },
            Rule {
                lhs: 0,
                rhs: vec![
                    Symbol::Terminal(0),
                    Symbol::Nonterminal(1),
                    Symbol::Nonterminal(2),
                    Symbol::Terminal(0),
                ],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Terminal(2)],
            },
            Rule {
                lhs: 5,
                rhs: vec![Symbol::Terminal(3)],
            },
            Rule { lhs: 4, rhs: vec![] },
            Rule {
                lhs: 4,
                rhs: vec![Symbol::Nonterminal(4), Symbol::Nonterminal(5)],
            },
            Rule {
                lhs: 1,
                rhs: vec![Symbol::Nonterminal(4)],
            },
        ];

        let source = bounded_language(&rules, 0, 6, 3);
        let transformed_rules = inline_null_productions(&rules, 6);
        let transformed =
            bounded_language(&transformed_rules, 0, max_nt_id(&transformed_rules) + 1, 3);
        let mut normalized_rules = rules.clone();
        normalize_grammar(&mut normalized_rules, 0);
        let normalized =
            bounded_language(&normalized_rules, 0, max_nt_id(&normalized_rules) + 1, 3);

        assert!(source.contains(&vec![0, 3, 0]));
        assert_eq!(transformed, source);
        assert_eq!(normalized, source);
    }
}
