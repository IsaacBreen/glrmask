#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use crate::finite_automata::*;
    use crate::{choice, seq};

    #[test]
    fn test_literal() {
        let expr: Expr = eat_u8(b'a');
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(!regex.could_match(b"b"));

        assert!(!regex.definitely_matches(b""));
        assert!(regex.could_match(b""));
        assert!(regex.definitely_matches(b"ab"));
        assert!(regex.definitely_matches(b"aa"));
    }

    #[test]
    fn test_quantifier() {
        let expr = rep(eat_u8(b'a'));
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b""));
        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"aaaa"));
        assert!(regex.could_match(b"b"));

        let mut state = regex.init();
        state.execute(b"aa");
        assert_eq!(state.matches, BTreeMap::from([(0, 2)]));
        assert!(!state.done());
    }

    #[test]
    fn test_choice() {
        let expr = choice![eat_u8(b'a'), eat_u8(b'b')];
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"b"));
        assert!(!regex.could_match(b"c"));
    }

    #[test]
    fn test_seq() {
        let expr = seq![eat_u8(b'a'), eat_u8(b'b')];
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.could_match(b"a"));
        assert!(!regex.definitely_matches(b"a"));
        assert!(!regex.could_match(b"b"));
        assert!(regex.definitely_matches(b"ab"));
        assert!(regex.definitely_matches(b"abab"));
        assert!(!regex.could_match(b"c"));
    }

    #[test]
    fn test_opt() {
        let expr = opt(eat_u8(b'a'));
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b""));
        assert!(regex.definitely_fully_matches(b"a"));
        assert!(!regex.could_fully_match(b"aa"));
        assert!(regex.could_match(b"b"));
    }

    #[test]
    fn test_0() {
        let expr = eat_u8(0);
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"\0"));
        assert!(!regex.could_match(b"1"));
    }

    #[test]
    fn test_epsilon() {
        let expr = eps();
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b""));
        assert!(regex.definitely_matches(b"a"));
        assert!(!regex.definitely_fully_matches(b"a"));
    }

    #[test]
    fn test_u8seq() {
        let expr = Expr::U8Seq(vec![b'a', b'b']);
        dbg!(&expr);
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"ab"));
        assert!(regex.could_match(b"a"));
        assert!(!regex.could_match(b"b"));
        assert!(!regex.could_match(b"ba"));
    }
}

#[cfg(test)]
mod complex_tests {
    use std::collections::BTreeMap;
    use crate::finite_automata::*;
    use crate::{choice, groups, seq};

    #[test]
    fn test_nested_quantifiers() {
        let expr = rep1(rep(eat_u8(b'a')));
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"aa"));
        assert!(regex.definitely_fully_matches(b"aaa"));
        assert!(regex.definitely_fully_matches(b""));
    }

    #[test]
    fn test_complex_choice() {
        let expr = choice![
            seq![eat_u8(b'a'), rep1(eat_u8(b'b'))],
            eat_u8(b'c'),
        ];
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"ab"));
        assert!(regex.definitely_fully_matches(b"abb"));
        assert!(regex.definitely_fully_matches(b"c"));
        assert!(regex.could_match(b"a"));
        assert!(!regex.definitely_matches(b"a"));
        assert!(!regex.could_match(b"b"));
        assert!(regex.definitely_matches(b"cc"));
        assert_eq!(regex.fully_matches(b"cc"), Some(false));
    }

    #[test]
    fn test_complex_seq_with_quantifiers() {
        let expr = seq![
            rep(eat_u8(b'a')),
            eat_u8(b'b'),
            rep1(eat_u8(b'c')),
        ];
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"bc"));
        assert!(regex.definitely_fully_matches(b"bcc"));
        assert!(regex.definitely_fully_matches(b"abcc"));
        assert!(regex.definitely_fully_matches(b"aaabccc"));
        assert!(regex.could_match(b"a"));
        assert!(regex.could_match(b"b"));
        assert!(!regex.could_match(b"c"));
    }

    #[test]
    fn test_complex_pattern() {
        let expr = seq![
            rep(choice![eat_u8(b'a'), eat_u8(b'b')]),
            eat_u8(b'c'),
            rep1(choice![eat_u8(b'd'), eat_u8(b'e')]),
        ];
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"cd"));
        assert!(regex.definitely_fully_matches(b"ce"));
        assert!(regex.definitely_fully_matches(b"cde"));
        assert!(regex.definitely_fully_matches(b"aced"));
        assert!(regex.definitely_fully_matches(b"bacde"));
        assert!(regex.could_match(b"a"));
        assert!(!regex.definitely_matches(b"a"));
        assert!(!regex.definitely_matches(b"b"));
        assert!(regex.could_match(b"c"));
        assert!(!regex.definitely_matches(b"c"));
        assert!(!regex.could_match(b"d"));
    }

    #[test]
    fn test_complex_epsilon() {
        let expr = groups![
            eps(),
            rep1(eat_u8(b'a')),
        ];
        let regex = expr.build();
        let mut state = regex.init();
        dbg!(&regex);
        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 0), (1, 1)]));
    }
}

#[cfg(test)]
mod even_more_complex_tests {
    use std::collections::BTreeMap;
    use crate::finite_automata::*;
    use crate::{choice, groups, seq};
    use crate::datastructures::u8set::U8Set;

    #[test]
    fn test_overlapping_u8_classes() {
        let expr = seq![
            choice![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'c')],
            choice![eat_u8(b'b'), eat_u8(b'c'), eat_u8(b'd')],
        ];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"bc"));
        assert!(regex.definitely_fully_matches(b"cb"));
        assert!(regex.definitely_fully_matches(b"ab"));
        assert!(regex.definitely_fully_matches(b"cd"));
    }

    #[test]
    fn test_nested_seqs_with_quantifiers() {
        let expr = seq![
            rep(seq![eat_u8(b'a'), rep1(eat_u8(b'b'))]),
            eat_u8(b'c'),
        ];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"c"));
        assert!(regex.definitely_fully_matches(b"abc"));
        assert!(regex.definitely_fully_matches(b"abbc"));
        assert!(regex.definitely_fully_matches(b"ababbabc"));
        assert!(!regex.could_match(b"ac"));
    }

    #[test]
    fn test_choice_with_empty_option() {
        let expr = choice![eat_u8(b'a'), seq![]];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b""));
    }

    #[test]
    fn test_complex_pattern_with_overlapping_quantifiers() {
        let expr = seq![
            rep(eat_u8(b'a')),
            rep1(eat_u8(b'a')),
        ];
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(regex.definitely_fully_matches(b"aa"));
        assert!(regex.could_match(b""));
        assert!(regex.could_fully_match(b""));
        assert!(!regex.could_match(b"b"));
    }

    #[test]
    fn test_matching_at_different_positions() {
        let expr: Expr = eat_u8(b'a');
        let regex = expr.build();

        assert!(regex.definitely_fully_matches(b"a"));
        assert!(!regex.could_match(b"ba"));
        assert!(regex.definitely_matches(b"ab"));
        assert!(!regex.definitely_fully_matches(b"ab"));
        assert!(!regex.could_match(b"bab"));
        assert!(!regex.could_match(b"b"));
    }

    #[test]
    fn test_lots_of_words() {
        let words = [
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "async",
            "await",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
        ];

        let expr = Expr::Choice(
            words
                .iter()
                .map(|word| {
                    Expr::Seq(word.bytes().map(|c| Expr::U8Seq(vec![c])).collect())
                })
                .collect(),
        );
        let regex = expr.build();
        dbg!(&regex);

        assert!(regex.definitely_fully_matches(b"False"));
        assert!(regex.definitely_fully_matches(b"None"));
        assert!(regex.definitely_fully_matches(b"True"));
        assert!(regex.definitely_fully_matches(b"and"));
        assert!(regex.definitely_fully_matches(b"as"));
        assert!(regex.definitely_fully_matches(b"assert"));
    }

    #[test]
    fn test_multiple_finalizers() {
        let expr = groups![
            eat_u8(b'a'),
            seq![eat_u8(b'a'), eat_u8(b'a')],
        ];

        let regex = expr.build();
        dbg!(&regex);

        let mut state = regex.init();

        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 1)]));

        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 1), (1, 2)]));
    }

    #[test]
    fn test_multiple_finalizers_greedy() {
        let expr = groups![
            rep(eat_u8(b'a')),
            eat_u8(b'a'),
        ];

        let regex = expr.build();
        dbg!(&regex);

        let mut state = regex.init();

        state.execute(b"aa");
        assert_eq!(state.matches, BTreeMap::from([(0, 2), (1, 1)]));
    }

    #[test]
    fn test_non_greedy_matching() {
        let expr = groups![
            non_greedy_group(rep(eat_u8(b'a'))),
            eat_u8(b'a'),
        ];

        let regex = expr.build();

        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        assert_eq!(regex_state.matches.get(&0), Some(&0));
        assert_eq!(regex_state.matches.get(&1), Some(&1));
    }

    #[test]
    fn test_greedy_matching() {
        let expr = groups![
            rep(eat_u8(b'a')),
            eat_u8(b'a'),
        ];

        let regex = expr.build();

        let mut regex_state = regex.init();
        regex_state.execute(b"aaa");

        assert_eq!(regex_state.matches.get(&0), Some(&3));
        assert_eq!(regex_state.matches.get(&1), Some(&1));
    }

    #[test]
    fn test_triple_quoted_string() {
        let non_greedy_expr = groups![
            non_greedy_group(seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())),
                Expr::U8Seq(b"\"\"\"".to_vec())
            ])
        ];
        let non_greedy_regex = non_greedy_expr.build();

        let greedy_expr = groups![
            seq![
                Expr::U8Seq(b"\"\"\"".to_vec()),
                rep(Expr::U8Class(U8Set::all())),
                Expr::U8Seq(b"\"\"\"".to_vec())
            ]
        ];
        let greedy_regex = greedy_expr.build();

        let input = b"\"\"\"hello\"\"\"world\"\"\"";

        let mut non_greedy_state = non_greedy_regex.init();
        non_greedy_state.execute(input);
        assert_eq!(
            non_greedy_state.matches.get(&0),
            Some(&b"\"\"\"hello\"\"\"".len())
        );

        let mut greedy_state = greedy_regex.init();
        greedy_state.execute(input);
        assert_eq!(greedy_state.matches.get(&0), Some(&input.len()));
    }
}

#[cfg(test)]
mod possible_future_group_ids_tests {
    use std::collections::BTreeSet;
    use crate::finite_automata::*;
    use crate::{choice, groups, seq};

    fn run_test(expr: impl Into<ExprGroups>, expected_possible_future_group_ids: BTreeSet<GroupID>) {
        let regex = expr.into().build();
        let state = regex.init();
        assert_eq!(
            state.possible_future_group_ids(),
            expected_possible_future_group_ids
        );
    }

    #[test]
    fn test_possible_future_group_ids() {
        run_test(seq![], BTreeSet::new());
        run_test(eat_u8(b'a'), BTreeSet::from([0]));
        run_test(
            groups![eat_u8(b'a'), eat_u8(b'b')],
            BTreeSet::from([0, 1]),
        );
        run_test(
            seq![eat_u8(b'a'), eat_u8(b'b')],
            BTreeSet::from([0]),
        );
        run_test(rep(eat_u8(b'a')), BTreeSet::from([0]));
        run_test(
            groups![
                choice![opt(eat_u8(b'a')), rep(eat_u8(b'b')), eat_u8(b'c')],
                eat_u8(b'a'),
            ],
            BTreeSet::from([0, 1]),
        );
        run_test(
            groups![
                eat_u8(b'a'),
                seq![eat_u8(b'a'), eat_u8(b'a')],
            ],
            BTreeSet::from([0, 1]),
        );
    }

    #[test]
    fn test_possible_future_group_ids_excludes_current_state() {
        let expr = groups![
            eps(),
            eat_u8(b'a'),
        ];
        let regex = expr.build();
        let start_state_index = regex.dfa.start_state;
        let start_state_data = &regex.dfa.states[start_state_index];

        assert_eq!(
            start_state_data.possible_future_group_ids,
            BTreeSet::from([1])
        );
    }
}

#[cfg(test)]
mod group_id_to_u8set_tests {
    use std::collections::BTreeSet;
    use crate::finite_automata::*;
    use crate::{choice, groups, seq};

    fn build_dfa_with_groups(exprs: Vec<Expr>) -> Regex {
        let expr_groups = ExprGroups {
            groups: exprs.into_iter().map(ExprGroup::from).collect(),
        };
        expr_groups.build()
    }

    #[test]
    fn test_compute_group_id_to_u8set_single_group() {
        let expr = groups![
            eat_u8(b'a')
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 1);
        assert!(group_id_to_u8set.contains_key(&0));
        let u8set = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set.contains(b'a'));
        assert_eq!(u8set.iter().collect::<Vec<u8>>(), vec![b'a']);
    }

    #[test]
    fn test_compute_group_id_to_u8set_multiple_groups() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'b'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 2);
        assert!(group_id_to_u8set.contains_key(&0));
        assert!(group_id_to_u8set.contains_key(&1));

        let u8set_a = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_a.contains(b'a'));
        assert_eq!(u8set_a.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_b = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_b.contains(b'b'));
        assert_eq!(u8set_b.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_compute_group_id_to_u8set_overlapping_groups() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'a'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 2);
        assert!(group_id_to_u8set.contains_key(&0));
        assert!(group_id_to_u8set.contains_key(&1));

        let u8set_a0 = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_a0.contains(b'a'));
        assert_eq!(u8set_a0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_a1 = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_a1.contains(b'a'));
        assert_eq!(u8set_a1.iter().collect::<Vec<u8>>(), vec![b'a']);
    }

    #[test]
    fn test_get_u8set_for_group_existing_group() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'b'),
        ];
        let regex = expr.build();

        let regex_state = regex.init();

        let u8set_group0 = regex_state.get_u8set_for_group(0);
        assert!(u8set_group0.contains(b'a'));
        assert_eq!(u8set_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert!(u8set_group1.contains(b'b'));
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_get_u8set_for_group_nonexistent_group() {
        let expr = groups![
            eat_u8(b'a')
        ];
        let regex = expr.build();

        let regex_state = regex.init();

        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), Vec::<u8>::new());
    }

    #[test]
    fn test_group_id_to_u8set_nested_groups() {
        let expr = groups![
            rep(choice![eat_u8(b'a'), eat_u8(b'b')]),
            eat_u8(b'c'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        dbg!(&regex);
        dbg!(&regex.dfa.states[0].possible_future_group_ids);
        dbg!(group_id_to_u8set);
        assert_eq!(group_id_to_u8set.len(), 2);

        let u8set_group0 = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_group0.contains(b'a'));
        assert!(u8set_group0.contains(b'b'));
        assert_eq!(u8set_group0.iter().collect::<Vec<u8>>(), vec![b'a', b'b']);

        let u8set_group1 = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_group1.contains(b'c'));
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), vec![b'c']);
    }

    #[test]
    fn test_group_id_to_u8set_nonexistent_group() {
        let expr = groups![
            eat_u8(b'a')
        ];
        let regex = expr.build();

        let regex_state = regex.init();
        let u8set_group1 = regex_state.get_u8set_for_group(1);
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), Vec::<u8>::new());
    }

    #[test]
    fn test_group_id_to_u8set_overlapping_groups() {
        let expr = groups![
            eat_u8(b'a'),
            eat_u8(b'a'),
        ];
        let regex = expr.build();

        let group_id_to_u8set = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set.len(), 2);

        let u8set_group0 = group_id_to_u8set.get(&0).unwrap();
        assert!(u8set_group0.contains(b'a'));
        assert_eq!(u8set_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let u8set_group1 = group_id_to_u8set.get(&1).unwrap();
        assert!(u8set_group1.contains(b'a'));
        assert_eq!(u8set_group1.iter().collect::<Vec<u8>>(), vec![b'a']);
    }

    #[test]
    fn test_get_u8set_for_group_after_transition() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')],
            seq![eat_u8(b'a'), eat_u8(b'c')],
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 2);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));
        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();
        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));

        let mut regex_state = regex.init();
        regex_state.execute(b"a");

        assert_eq!(
            regex.dfa.states[regex_state.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1])
        );

        let group_id_to_u8set_new =
            &regex.dfa.states[regex_state.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_new.len(), 2);
        assert!(group_id_to_u8set_new.contains_key(&0));
        assert!(group_id_to_u8set_new.contains_key(&1));

        let u8set_new_group0 = group_id_to_u8set_new.get(&0).unwrap();
        let u8set_new_group1 = group_id_to_u8set_new.get(&1).unwrap();

        assert!(u8set_new_group0.contains(b'b'));
        assert!(u8set_new_group1.contains(b'c'));
    }

    #[test]
    fn test_group_id_to_u8set_after_multiple_transitions() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'c')],
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'd')],
            seq![eat_u8(b'a'), eat_u8(b'b'), eat_u8(b'e')],
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 3);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));
        assert!(group_id_to_u8set_0.contains_key(&2));

        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();
        let u8set_0_group2 = group_id_to_u8set_0.get(&2).unwrap();

        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));
        assert!(u8set_0_group2.contains(b'a'));

        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");

        assert_eq!(
            regex.dfa.states[regex_state_a.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1, 2])
        );

        let group_id_to_u8set_a =
            &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_a.len(), 3);
        assert!(group_id_to_u8set_a.contains_key(&0));
        assert!(group_id_to_u8set_a.contains_key(&1));
        assert!(group_id_to_u8set_a.contains_key(&2));

        let u8set_a_group0 = group_id_to_u8set_a.get(&0).unwrap();
        let u8set_a_group1 = group_id_to_u8set_a.get(&1).unwrap();
        let u8set_a_group2 = group_id_to_u8set_a.get(&2).unwrap();

        assert!(u8set_a_group0.contains(b'b'));
        assert!(u8set_a_group1.contains(b'b'));
        assert!(u8set_a_group2.contains(b'b'));

        let mut regex_state_ab = regex.init();
        regex_state_ab.execute(b"ab");

        assert_eq!(
            regex.dfa.states[regex_state_ab.current_state].possible_future_group_ids,
            BTreeSet::from([0, 1, 2])
        );

        let group_id_to_u8set_ab =
            &regex.dfa.states[regex_state_ab.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_ab.len(), 3);
        assert!(group_id_to_u8set_ab.contains_key(&0));
        assert!(group_id_to_u8set_ab.contains_key(&1));
        assert!(group_id_to_u8set_ab.contains_key(&2));

        let u8set_ab_group0 = group_id_to_u8set_ab.get(&0).unwrap();
        let u8set_ab_group1 = group_id_to_u8set_ab.get(&1).unwrap();
        let u8set_ab_group2 = group_id_to_u8set_ab.get(&2).unwrap();

        assert!(u8set_ab_group0.contains(b'c'));
        assert!(u8set_ab_group1.contains(b'd'));
        assert!(u8set_ab_group2.contains(b'e'));
    }

    #[test]
    fn test_group_id_to_u8set_after_consuming_all() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')]
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 1);
        assert!(group_id_to_u8set_0.contains_key(&0));

        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        assert!(u8set_0_group0.contains(b'a'));
        assert_eq!(u8set_0_group0.iter().collect::<Vec<u8>>(), vec![b'a']);

        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");
        assert_eq!(
            regex.dfa.states[regex_state_a.current_state].possible_future_group_ids,
            BTreeSet::from([0])
        );

        let group_id_to_u8set_a =
            &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_a.len(), 1);
        assert!(group_id_to_u8set_a.contains_key(&0));

        let u8set_a_group0 = group_id_to_u8set_a.get(&0).unwrap();
        assert!(u8set_a_group0.contains(b'b'));
        assert_eq!(u8set_a_group0.iter().collect::<Vec<u8>>(), vec![b'b']);
    }

    #[test]
    fn test_get_u8set_for_group_multiple_transitions() {
        let expr = groups![
            seq![eat_u8(b'a'), eat_u8(b'b')],
            seq![eat_u8(b'a'), eat_u8(b'c')],
        ];
        let regex = expr.build();

        let group_id_to_u8set_0 = &regex.dfa.states[0].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_0.len(), 2);
        assert!(group_id_to_u8set_0.contains_key(&0));
        assert!(group_id_to_u8set_0.contains_key(&1));

        let u8set_0_group0 = group_id_to_u8set_0.get(&0).unwrap();
        let u8set_0_group1 = group_id_to_u8set_0.get(&1).unwrap();

        assert!(u8set_0_group0.contains(b'a'));
        assert!(u8set_0_group1.contains(b'a'));
        assert_eq!(u8set_0_group0.iter().collect::<Vec<u8>>(), vec![b'a']);
        assert_eq!(u8set_0_group1.iter().collect::<Vec<u8>>(), vec![b'a']);

        let mut regex_state_a = regex.init();
        regex_state_a.execute(b"a");

        let group_id_to_u8set_a =
            &regex.dfa.states[regex_state_a.current_state].group_id_to_u8set;
        assert_eq!(group_id_to_u8set_a.len(), 2);
        assert!(group_id_to_u8set_a.contains_key(&0));
        assert!(group_id_to_u8set_a.contains_key(&1));

        let u8set_a_group0 = group_id_to_u8set_a.get(&0).unwrap();
        let u8set_a_group1 = group_id_to_u8set_a.get(&1).unwrap();

        assert!(u8set_a_group0.contains(b'b'));
        assert!(u8set_a_group1.contains(b'c'));
        assert_eq!(u8set_a_group0.iter().collect::<Vec<u8>>(), vec![b'b']);
        assert_eq!(u8set_a_group1.iter().collect::<Vec<u8>>(), vec![b'c']);
    }
}

#[cfg(test)]
mod group_u8set_tests {
    use std::collections::{BTreeMap, BTreeSet};
    use crate::datastructures::char_transitions::CharTransitions;
    use crate::datastructures::compressed_state_set::DenseStateSet;
    use crate::finite_automata::*;

    #[test]
    fn test_get_u8set_for_group() {
        let mut dfa = DFA {
            states: Vec::new(),
            start_state: 0,
            non_greedy_finalizers: BTreeSet::new(),
        };

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: DenseStateSet::new(2),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: DenseStateSet::new(2),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: DenseStateSet::new_from_slice(2, &[0]),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states.push(DFAState {
            transitions: CharTransitions::new(),
            finalizers: DenseStateSet::new_from_slice(2, &[1]),
            possible_future_group_ids: BTreeSet::new(),
            group_id_to_u8set: BTreeMap::new(),
        });

        dfa.states[0].transitions.insert(b'a', 1);
        dfa.states[1].transitions.insert(b'b', 2);
        dfa.states[1].transitions.insert(b'c', 3);

        dfa.compute_possible_future_group_ids();
        dfa.compute_group_id_to_u8set();

        let regex = Regex { dfa };

        let state0 = regex.init_to_state(0);
        let u8set_group0_state0 = state0.get_u8set_for_group(0);
        let u8set_group1_state0 = state0.get_u8set_for_group(1);
        assert!(u8set_group0_state0.contains(b'a'));
        assert!(u8set_group1_state0.contains(b'a'));

        let state1 = regex.init_to_state(1);
        let u8set_group0_state1 = state1.get_u8set_for_group(0);
        let u8set_group1_state1 = state1.get_u8set_for_group(1);
        assert!(u8set_group0_state1.contains(b'b'));
        assert!(!u8set_group0_state1.contains(b'c'));
        assert!(u8set_group1_state1.contains(b'c'));
        assert!(!u8set_group1_state1.contains(b'b'));

        let state2 = regex.init_to_state(2);
        let u8set_group0_state2 = state2.get_u8set_for_group(0);
        let u8set_group1_state2 = state2.get_u8set_for_group(1);
        assert!(u8set_group0_state2.iter().next().is_none());
        assert!(u8set_group1_state2.iter().next().is_none());

        let state3 = regex.init_to_state(3);
        let u8set_group0_state3 = state3.get_u8set_for_group(0);
        let u8set_group1_state3 = state3.get_u8set_for_group(1);
        assert!(u8set_group0_state3.iter().next().is_none());
        assert!(u8set_group1_state3.iter().next().is_none());
    }
}

#[cfg(test)]
mod tests_nov_24 {
    use std::collections::BTreeMap;
    use crate::{choice, groups, seq};
    use crate::finite_automata::*;

    #[test]
    fn test_eat_u8() {
        let expr = groups![
            eat_u8(b'a'),
            seq![eat_u8(b'a'), eat_u8(b'b')],
        ];

        let regex = expr.build();
        dbg!(&regex);
        let mut state = regex.init();
        state.execute(b"a");
        assert_eq!(state.matches, BTreeMap::from([(0, 1)]));
        state.clear_matches();

        state.execute(b"b");
        assert_eq!(state.matches, BTreeMap::from([(1, 2)]));
    }

    #[test]
    fn test_reasonable_number_of_states() {
        let expr = choice![eat_u8(b'a'), eat_u8(b'b'),];
        let regex = expr.build();
        dbg!(&regex);
        assert_eq!(regex.dfa.states.len(), 2);
    }
}

#[cfg(test)]
mod test_python {
    use std::collections::BTreeMap;
    use crate::finite_automata::*;
    use crate::datastructures::u8set::U8Set;
    use crate::{choice, seq};

    #[ignore]
    #[test]
    fn test_full_python_tokenizer_recognizes_name() {
        let digit = Expr::U8Class(U8Set::from_range(b'0', b'9'));
        let alph_lower = Expr::U8Class(U8Set::from_range(b'a', b'z'));
        let alph_upper = Expr::U8Class(U8Set::from_range(b'A', b'Z'));
        let name_start = choice![alph_lower.clone(), alph_upper.clone(), eat_u8(b'_')];
        let name_middle = choice![name_start.clone(), digit.clone()];

        let ignore = rep(choice![
             eat_u8(b' '),
             seq![eat_u8(b'#'), rep(Expr::U8Class(U8Set::all().without(b'\n'))), opt(eat_u8(b'\n'))],
         ]);

        let tokens_core: BTreeMap<&str, Expr> = BTreeMap::from([
            ("NAME", seq![name_start, rep(name_middle)]),
            ("NUMBER", choice![
                rep1(digit.clone()),
                seq![rep1(digit.clone()), eat_u8(b'.'), rep(digit.clone())],
                seq![eat_u8(b'.'), rep1(digit.clone())],
            ]),
            ("STRING", choice![
                seq![eat_u8(b'"'), rep(Expr::U8Class(U8Set::all().without(b'"'))), eat_u8(b'"')],
                seq![eat_u8(b'\''), rep(Expr::U8Class(U8Set::all().without(b'\''))), eat_u8(b'\'')],
            ]),
            ("FSTRING_START", Expr::U8Seq(b"f'".to_vec())),
            ("FSTRING_END", Expr::U8Seq(b"'".to_vec())),
            ("FSTRING_MIDDLE", rep1(Expr::U8Class(U8Set::all().difference(&U8Set::from_slice(&[b'{', b'}']))))),
            ("NEWLINE", eps()),
            ("INDENT", eps()),
            ("DEDENT", eps()),
            ("TYPE_COMMENT", eps()),
            ("ENDMARKER", eps()),
            ("LPAREN", eat_u8(b'(')),
            ("RPAREN", eat_u8(b')')),
            ("LSQB", eat_u8(b'[')),
            ("RSQB", eat_u8(b']')),
            ("LBRACE", eat_u8(b'{')),
            ("RBRACE", eat_u8(b'}')),
            ("COMMA", eat_u8(b',')),
            ("COLON", eat_u8(b':')),
            ("DOT", eat_u8(b'.')),
            ("SEMI", eat_u8(b';')),
            ("PLUS", eat_u8(b'+')),
            ("MINUS", eat_u8(b'-')),
            ("STAR", eat_u8(b'*')),
            ("SLASH", eat_u8(b'/')),
            ("VBAR", eat_u8(b'|')),
            ("AMPER", eat_u8(b'&')),
            ("LESS", eat_u8(b'<')),
            ("GREATER", eat_u8(b'>')),
            ("EQUAL", eat_u8(b'=')),
            ("PERCENT", eat_u8(b'%')),
            ("CIRCUMFLEX", eat_u8(b'^')),
            ("TILDE", eat_u8(b'~')),
            ("AT", eat_u8(b'@')),
            ("EXCLAMATION", eat_u8(b'!')),
            ("DOUBLESTAR", Expr::U8Seq(b"**".to_vec())),
            ("DOUBLESLASH", Expr::U8Seq(b"//".to_vec())),
            ("LEFTSHIFT", Expr::U8Seq(b"<<".to_vec())),
            ("RIGHTSHIFT", Expr::U8Seq(b">>".to_vec())),
            ("EQEQUAL", Expr::U8Seq(b"==".to_vec())),
            ("NOTEQUAL", Expr::U8Seq(b"!=".to_vec())),
            ("LESSEQUAL", Expr::U8Seq(b"<=".to_vec())),
            ("GREATEREQUAL", Expr::U8Seq(b">=".to_vec())),
            ("ATEQUAL", Expr::U8Seq(b"@=".to_vec())),
            ("PLUSEQUAL", Expr::U8Seq(b"+=".to_vec())),
            ("MINEQUAL", Expr::U8Seq(b"-=".to_vec())),
            ("STAREQUAL", Expr::U8Seq(b"*=".to_vec())),
            ("SLASHEQUAL", Expr::U8Seq(b"/=".to_vec())),
            ("PERCENTEQUAL", Expr::U8Seq(b"%=".to_vec())),
            ("AMPEREQUAL", Expr::U8Seq(b"&=".to_vec())),
            ("VBAREQUAL", Expr::U8Seq(b"|=".to_vec())),
            ("CIRCUMFLEXEQUAL", Expr::U8Seq(b"^=".to_vec())),
            ("LEFTSHIFTEQUAL", Expr::U8Seq(b"<<=".to_vec())),
            ("RIGHTSHIFTEQUAL", Expr::U8Seq(b">>=".to_vec())),
            ("DOUBLESTAREQUAL", Expr::U8Seq(b"**=".to_vec())),
            ("DOUBLESLASHEQUAL", Expr::U8Seq(b"//=".to_vec())),
            ("RARROW", Expr::U8Seq(b"->".to_vec())),
            ("ELLIPSIS", Expr::U8Seq(b"...".to_vec())),
            ("COLONEQUAL", Expr::U8Seq(b":=".to_vec())),
        ]);

        let mut token_groups: Vec<ExprGroup> = Vec::new();
        let mut token_name_to_id: BTreeMap<&str, GroupID> = BTreeMap::new();
        for (name, core_expr) in tokens_core {
            let group_id = token_groups.len();
            token_name_to_id.insert(name, group_id);
            token_groups.push(greedy_group(seq![ignore.clone(), core_expr]));
        }

        let expr_groups = groups(token_groups);
        let regex = expr_groups.build();

        let mut state = regex.init();
        state.execute(b"hello");

        assert!(state.definitely_matches(), "Tokenizer should match 'hello'");
        assert_eq!(
            state.matches.get(&token_name_to_id["NAME"]),
            Some(&5),
            "NAME token should be matched at position 5"
        );
    }
}

#[cfg(test)]
mod reproduction_tests {
    use std::sync::Arc;
    use crate::finite_automata::*;

    /// Test for the expression: (i*)* { (i*)* } (i*)*
    /// 
    /// This test repros a bug where double-repeated quantifier (i*)*
    /// causes the DFA to collapse to a single state that incorrectly
    /// accepts just "{" or "}" alone.
    #[test]
    fn test_double_star_repro() {
        // Build (i*)* - a double quantifier with Shared wrapper
        // The actual structure from GrammarDefinition is:
        // Quantifier(Shared(Quantifier(U8Seq([105]), ZeroOrMore)), ZeroOrMore)
        let i_star = Expr::Quantifier(
            Box::new(Expr::U8Seq(vec![b'i'])),
            QuantifierType::ZeroOrMore,
        );
        let i_star_star = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(i_star))),
            QuantifierType::ZeroOrMore,
        );
        
        // Build (i*)* { (i*)* } (i*)*
        let expr = Expr::Seq(vec![
            i_star_star.clone(),
            Expr::U8Seq(vec![b'{']),
            i_star_star.clone(),
            Expr::U8Seq(vec![b'}']),
            i_star_star,
        ]);
        
        println!("Expr: {}", expr);
        
        // Build regex directly from the expr
        let regex = expr.build();
        println!("Regex:\n{}", regex);
        println!("Regex states: {}", regex.dfa.states.len());
        
        // CRITICAL: The regex MUST have 3 states, not 1
        assert_eq!(regex.dfa.states.len(), 3, 
            "Regex should have 3 states. A 1-state DFA is incorrect!");
        
        // Verify start state structure
        let start_state = regex.dfa.start_state;
        let dfa_start = &regex.dfa.states[start_state];
        
        // Start state should NOT be a finalizer (can't accept empty string)
        assert!(dfa_start.finalizers.is_empty(),
            "Start state should NOT be a finalizer - empty string is not valid!");
        
        // Start state: 'i' should loop to 0, '{' should go to state 1 (not 0!)
        let i_target = dfa_start.transitions.get(b'i');
        let brace_target = dfa_start.transitions.get(b'{');
        
        assert_eq!(i_target, Some(&0), "'i' from start should go to state 0");
        assert_eq!(brace_target, Some(&1), "'{{' from start should go to state 1, not 0!");
    }

    #[test]
    fn test_optimize_nested_quantifiers() {
        // Test that the optimizer properly simplifies nested quantifiers
        use std::sync::Arc;
        
        // Build (i*)* with Shared wrapper
        let i_star = Expr::Quantifier(
            Box::new(Expr::U8Seq(vec![b'i'])),
            QuantifierType::ZeroOrMore,
        );
        let i_star_shared = Expr::Shared(Arc::new(i_star.clone()));
        let i_star_star = Expr::Quantifier(
            Box::new(i_star_shared.clone()),
            QuantifierType::ZeroOrMore,
        );
        
        println!("Before optimize: {}", i_star_star);
        let optimized = i_star_star.clone().optimize();
        println!("After optimize: {}", optimized);
        
        // After optimization, (i*)* should become i*
        // So the optimized result should be equivalent to i_star
        // (but wrapped in Shared)
        
        // Build DFA for both
        let dfa_original = i_star.clone().build();
        let dfa_optimized = optimized.build();
        
        println!("Original i* DFA states: {}", dfa_original.dfa.states.len());
        println!("Optimized (i*)* DFA states: {}", dfa_optimized.dfa.states.len());
        
        // Both should have same number of states (2 states for i*)
        assert_eq!(dfa_original.dfa.states.len(), dfa_optimized.dfa.states.len(),
            "Optimized (i*)* should have same states as i*");
    }

    #[test]
    fn test_js_whitespace_pattern() {
        // Reproduce the JS whitespace pattern that causes DFA explosion
        // WS = [\t\n\r ]+ | "//" [^\n]* | "/*" ("*" [^/] | [^*])* "*/"
        // The pattern WS WS* should simplify to WS+
        
        use std::sync::Arc;
        use crate::datastructures::u8set::U8Set;
        
        // Simplified version: space = [ ]+
        let space_char = Expr::U8Class(U8Set::from_chars(" \t\n\r"));
        let space_plus = Expr::Quantifier(Box::new(space_char.clone()), QuantifierType::OneOrMore);
        
        // WS WS* pattern (what we see in the JS grammar)
        let ws_shared = Expr::Shared(Arc::new(space_plus.clone()));
        let ws_star = Expr::Quantifier(Box::new(ws_shared.clone()), QuantifierType::ZeroOrMore);
        let ws_ws_star = Expr::Seq(vec![ws_shared.clone(), ws_star.clone()]);
        
        println!("WS WS* before optimize: {}", ws_ws_star);
        let optimized = ws_ws_star.clone().optimize();
        println!("WS WS* after optimize: {}", optimized);
        
        // This should ideally become WS+ (or at least WS WS* should not explode DFA)
        let dfa = optimized.build();
        println!("DFA states: {}", dfa.dfa.states.len());
        
        // WS WS* = WS+ should be a small DFA (2-3 states max)
        assert!(dfa.dfa.states.len() <= 3, 
            "WS WS* should have at most 3 DFA states, got {}", dfa.dfa.states.len());
    }

    #[test]
    fn test_js_complex_whitespace_pattern() {
        // Test the full JS whitespace pattern with comments
        // WS = space+ | "//" [^\n]* | "/*" (...) "*/"
        // The problematic pattern is: WS WS*
        
        use std::sync::Arc;
        use crate::datastructures::u8set::U8Set;
        
        // Build: space+ | "//" [^\n]* 
        // (simplified - skip block comments for now)
        
        let space_class = Expr::U8Class(U8Set::from_chars(" \t\n\r"));
        let space_plus = Expr::Quantifier(Box::new(space_class.clone()), QuantifierType::OneOrMore);
        
        let non_newline = {
            let mut set = U8Set::all();
            set.remove(b'\n');
            Expr::U8Class(set)
        };
        let non_newline_star = Expr::Quantifier(Box::new(non_newline), QuantifierType::ZeroOrMore);
        let line_comment = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            non_newline_star,
        ]);
        
        // WS = space+ | line_comment
        let ws = Expr::Choice(vec![space_plus.clone(), line_comment]);
        let ws_shared = Expr::Shared(Arc::new(ws));
        
        // WS WS* pattern (what appears throughout JS grammar)
        let ws_star = Expr::Quantifier(Box::new(ws_shared.clone()), QuantifierType::ZeroOrMore);
        let ws_ws_star = Expr::Seq(vec![ws_shared.clone(), ws_star.clone()]);
        
        println!("Complex WS WS* before optimize: {}", ws_ws_star);
        let optimized = ws_ws_star.clone().optimize();
        println!("Complex WS WS* after optimize: {}", optimized);
        
        let dfa = optimized.build();
        println!("Complex WS WS* DFA states: {}", dfa.dfa.states.len());
        
        // This should be manageable
        assert!(dfa.dfa.states.len() <= 10, 
            "Complex WS WS* should have <= 10 DFA states, got {}", dfa.dfa.states.len());
    }

    #[test]
    fn test_js_nested_blocks_pattern() {
        // Test the pattern that actually causes the explosion
        // Something like: { (WS WS*)* statement_list (WS WS*)* }
        // Nested repetition of the whitespace pattern inside braces
        
        use std::sync::Arc;
        use crate::datastructures::u8set::U8Set;
        
        let space_class = Expr::U8Class(U8Set::from_chars(" \t\n\r"));
        let space_plus = Expr::Quantifier(Box::new(space_class.clone()), QuantifierType::OneOrMore);
        let ws_shared = Expr::Shared(Arc::new(space_plus.clone()));
        
        // WS*
        let ws_star = Expr::Quantifier(Box::new(ws_shared.clone()), QuantifierType::ZeroOrMore);
        
        // statement = "x"
        let statement = Expr::U8Seq(b"x".to_vec());
        
        // block = "{" WS* statement WS* "}"
        let block = Expr::Seq(vec![
            Expr::U8Seq(b"{".to_vec()),
            ws_star.clone(),
            statement,
            ws_star.clone(),
            Expr::U8Seq(b"}".to_vec()),
        ]);
        
        // nested_blocks = block+
        let block_shared = Expr::Shared(Arc::new(block));
        let nested_blocks = Expr::Quantifier(Box::new(block_shared), QuantifierType::OneOrMore);
        
        println!("Nested blocks pattern: {}", nested_blocks);
        let optimized = nested_blocks.clone().optimize();
        println!("After optimize: {}", optimized);
        
        let dfa = optimized.build();
        println!("Nested blocks DFA states: {}", dfa.dfa.states.len());
        
        // This is the key test - nested repetition should not explode
        assert!(dfa.dfa.states.len() <= 50, 
            "Nested blocks should have <= 50 DFA states, got {}", dfa.dfa.states.len());
    }

    #[test]
    fn test_multi_group_dfa_explosion() {
        // Test overlapping prefix hypothesis:
        // The issue is that "//" and "/*" share prefix "/", so the DFA must track
        // which comment type we're in at each WS position.
        
        use std::sync::Arc;
        use crate::datastructures::u8set::U8Set;
        
        // Test 1: Non-overlapping prefixes "aa" and "bb"
        let aa = Expr::U8Seq(b"aa".to_vec());
        let bb = Expr::U8Seq(b"bb".to_vec());
        let non_overlap_ws = Expr::Choice(vec![aa, bb]);
        let non_overlap_ws_shared = Expr::Shared(Arc::new(non_overlap_ws));
        let non_overlap_ws_star = Expr::Quantifier(Box::new(non_overlap_ws_shared.clone()), QuantifierType::ZeroOrMore);
        
        let pattern1 = Expr::Seq(vec![
            Expr::U8Seq(b"x".to_vec()),
            non_overlap_ws_star.clone(),
            Expr::U8Seq(b"y".to_vec()),
            non_overlap_ws_star.clone(),
            Expr::U8Seq(b"z".to_vec()),
        ]);
        
        println!("\n=== Test 1: Non-overlapping prefixes (aa | bb)* ===");
        let regex1 = pattern1.build();
        println!("DFA states: {}", regex1.dfa.states.len());
        
        // Test 2: Overlapping prefixes "ab" and "ac"  
        let ab = Expr::U8Seq(b"ab".to_vec());
        let ac = Expr::U8Seq(b"ac".to_vec());
        let overlap_ws = Expr::Choice(vec![ab, ac]);
        let overlap_ws_shared = Expr::Shared(Arc::new(overlap_ws));
        let overlap_ws_star = Expr::Quantifier(Box::new(overlap_ws_shared.clone()), QuantifierType::ZeroOrMore);
        
        let pattern2 = Expr::Seq(vec![
            Expr::U8Seq(b"x".to_vec()),
            overlap_ws_star.clone(),
            Expr::U8Seq(b"y".to_vec()),
            overlap_ws_star.clone(),
            Expr::U8Seq(b"z".to_vec()),
        ]);
        
        println!("\n=== Test 2: Overlapping prefixes (ab | ac)* ===");
        let regex2 = pattern2.build();
        println!("DFA states: {}", regex2.dfa.states.len());
        
        // Test 3: More repetitions with overlap
        let pattern3 = Expr::Seq(vec![
            Expr::U8Seq(b"p".to_vec()),
            overlap_ws_star.clone(),
            Expr::U8Seq(b"q".to_vec()),
            overlap_ws_star.clone(),
            Expr::U8Seq(b"r".to_vec()),
            overlap_ws_star.clone(),
            Expr::U8Seq(b"s".to_vec()),
            overlap_ws_star.clone(),
            Expr::U8Seq(b"t".to_vec()),
        ]);
        
        println!("\n=== Test 3: More repetitions (4 WS* slots) ===");
        let regex3 = pattern3.build();
        println!("DFA states: {}", regex3.dfa.states.len());
        
        // Test 4: Simulate line comment vs block comment (different endings)
        // Line: "//" followed by anything except newline, then newline
        // Block: "/*" followed by anything except */
        // Both start with "/"
        let non_newline = {
            let mut set = U8Set::all();
            set.remove(b'\n');
            Expr::U8Class(set)
        };
        
        let line_comment = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            Expr::Quantifier(Box::new(non_newline), QuantifierType::ZeroOrMore),
        ]);
        
        // Simplified block comment: "/*" [^*]* "*/"
        let not_star = {
            let mut set = U8Set::all();
            set.remove(b'*');
            Expr::U8Class(set)
        };
        let block_comment = Expr::Seq(vec![
            Expr::U8Seq(b"/*".to_vec()),
            Expr::Quantifier(Box::new(not_star), QuantifierType::ZeroOrMore),
            Expr::U8Seq(b"*/".to_vec()),
        ]);
        
        let comment_ws = Expr::Choice(vec![line_comment, block_comment.clone()]);
        let comment_ws_shared = Expr::Shared(Arc::new(comment_ws));
        let comment_ws_star = Expr::Quantifier(Box::new(comment_ws_shared.clone()), QuantifierType::ZeroOrMore);
        
        let pattern4 = Expr::Seq(vec![
            Expr::U8Seq(b"x".to_vec()),
            comment_ws_star.clone(),
            Expr::U8Seq(b"y".to_vec()),
            comment_ws_star.clone(),
            Expr::U8Seq(b"z".to_vec()),
        ]);
        
        println!("\n=== Test 4: Line + Block comments (overlapping /) ===");
        let regex4 = pattern4.build();
        println!("DFA states: {}", regex4.dfa.states.len());
        
        // Test 5: Same but without the overlapping prefix
        // Line: "#" [^\n]*
        // Block: "/*" [^*]* "*/"
        let line_comment2 = Expr::Seq(vec![
            Expr::U8Seq(b"#".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'\n');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
        ]);
        
        let comment_ws2 = Expr::Choice(vec![line_comment2, block_comment.clone()]);
        let comment_ws_shared2 = Expr::Shared(Arc::new(comment_ws2));
        let comment_ws_star2 = Expr::Quantifier(Box::new(comment_ws_shared2.clone()), QuantifierType::ZeroOrMore);
        
        let pattern5 = Expr::Seq(vec![
            Expr::U8Seq(b"x".to_vec()),
            comment_ws_star2.clone(),
            Expr::U8Seq(b"y".to_vec()),
            comment_ws_star2.clone(),
            Expr::U8Seq(b"z".to_vec()),
        ]);
        
        println!("\n=== Test 5: # line + /* block (non-overlapping) ===");
        let regex5 = pattern5.build();
        println!("DFA states: {}", regex5.dfa.states.len());
        
        // Test 6: Add space to the mix - the real WS pattern
        let space = Expr::U8Class(U8Set::from_chars(" \t\n\r"));
        let space_plus = Expr::Quantifier(Box::new(space.clone()), QuantifierType::OneOrMore);
        
        // Line: "//" [^\n]*
        let line_comment3 = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'\n');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
        ]);
        
        // Block: "/*" [^*]* "*/"
        let not_star2 = {
            let mut set = U8Set::all();
            set.remove(b'*');
            Expr::U8Class(set)
        };
        let block_comment2 = Expr::Seq(vec![
            Expr::U8Seq(b"/*".to_vec()),
            Expr::Quantifier(Box::new(not_star2), QuantifierType::ZeroOrMore),
            Expr::U8Seq(b"*/".to_vec()),
        ]);
        
        // WS = space+ | line_comment | block_comment
        let full_ws = Expr::Choice(vec![space_plus.clone(), line_comment3, block_comment2]);
        let full_ws_shared = Expr::Shared(Arc::new(full_ws));
        let full_ws_star = Expr::Quantifier(Box::new(full_ws_shared.clone()), QuantifierType::ZeroOrMore);
        
        let pattern6 = Expr::Seq(vec![
            Expr::U8Seq(b"x".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"y".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"z".to_vec()),
        ]);
        
        println!("\n=== Test 6: Full WS (space+ | // | /**/), 2 WS* slots ===");
        let regex6 = pattern6.build();
        println!("DFA states: {}", regex6.dfa.states.len());
        
        // Test 7: More WS* slots
        let pattern7 = Expr::Seq(vec![
            Expr::U8Seq(b"p".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"q".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"r".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"s".to_vec()),
        ]);
        
        println!("\n=== Test 7: Full WS, 3 WS* slots ===");
        let regex7 = pattern7.build();
        println!("DFA states: {}", regex7.dfa.states.len());
        
        // Test 8: With repetition
        let pattern8 = Expr::Seq(vec![
            Expr::U8Seq(b"x".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"y".to_vec()),
            Expr::Quantifier(
                Box::new(Expr::Seq(vec![
                    full_ws_star.clone(),
                    Expr::U8Seq(b",".to_vec()),
                    full_ws_star.clone(),
                    Expr::U8Seq(b"z".to_vec()),
                ])),
                QuantifierType::ZeroOrMore,
            ),
        ]);
        
        println!("\n=== Test 8: Full WS with repetition ===");
        let regex8 = pattern8.build();
        println!("DFA states: {}", regex8.dfa.states.len());
        
        // Test 9: 4 WS* slots
        let pattern9 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
        ]);
        
        println!("\n=== Test 9: Full WS, 4 WS* slots ===");
        let regex9 = pattern9.build();
        println!("DFA states: {}", regex9.dfa.states.len());
        
        // Test 10: 5 WS* slots
        let pattern10 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            full_ws_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 10: Full WS, 5 WS* slots ===");
        let regex10 = pattern10.build();
        println!("DFA states: {}", regex10.dfa.states.len());
        
        // Test 11: space only, 5 WS* slots
        let space_only_star = Expr::Quantifier(Box::new(space_plus.clone()), QuantifierType::ZeroOrMore);
        let pattern11 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            space_only_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            space_only_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            space_only_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            space_only_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            space_only_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 11: Space only, 5 WS* slots ===");
        let regex11 = pattern11.build();
        println!("DFA states: {}", regex11.dfa.states.len());
        
        // Test 12: Line comments only (no block), 5 WS* slots
        // Line comments terminate at newline, so "y" clearly separates regions
        let line_only = Expr::Quantifier(Box::new({
            let mut set = U8Set::all();
            set.remove(b'\n');
            Expr::U8Class(set)
        }), QuantifierType::ZeroOrMore);
        let line_comment_only = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            line_only,
        ]);
        let line_ws = Expr::Choice(vec![space_plus.clone(), line_comment_only.clone()]);
        let line_ws_star = Expr::Quantifier(Box::new(Expr::Shared(Arc::new(line_ws))), QuantifierType::ZeroOrMore);
        
        let pattern12 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            line_ws_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            line_ws_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            line_ws_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            line_ws_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            line_ws_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 12: Space + line comments only, 5 WS* slots ===");
        let regex12 = pattern12.build();
        println!("DFA states: {}", regex12.dfa.states.len());
        
        // Test 13: Block comments only (no line), 5 WS* slots
        // This should show explosion because block comments can "contain" any character
        let not_star3 = {
            let mut set = U8Set::all();
            set.remove(b'*');
            Expr::U8Class(set)
        };
        let block_comment_only = Expr::Seq(vec![
            Expr::U8Seq(b"/*".to_vec()),
            Expr::Quantifier(Box::new(not_star3), QuantifierType::ZeroOrMore),
            Expr::U8Seq(b"*/".to_vec()),
        ]);
        let block_ws = Expr::Choice(vec![space_plus.clone(), block_comment_only]);
        let block_ws_star = Expr::Quantifier(Box::new(Expr::Shared(Arc::new(block_ws))), QuantifierType::ZeroOrMore);
        
        let pattern13 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            block_ws_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            block_ws_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            block_ws_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            block_ws_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            block_ws_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 13: Space + block comments only, 5 WS* slots ===");
        let regex13 = pattern13.build();
        println!("DFA states: {}", regex13.dfa.states.len());
        
        // Test 14: Line comments only - space is NOT in whitespace
        let line_only2 = Expr::Quantifier(Box::new({
            let mut set = U8Set::all();
            set.remove(b'\n');
            Expr::U8Class(set)
        }), QuantifierType::ZeroOrMore);
        let line_comment_only2 = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            line_only2,
        ]);
        let line_ws_only_star = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(line_comment_only2))), 
            QuantifierType::ZeroOrMore
        );
        
        let pattern14 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            line_ws_only_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            line_ws_only_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            line_ws_only_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            line_ws_only_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            line_ws_only_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 14: Line comments ONLY (no space), 5 WS* slots ===");
        let regex14 = pattern14.build();
        println!("DFA states: {}", regex14.dfa.states.len());
        
        // Test 15: Space that does NOT include newline + line comments
        // This removes the ambiguity at newline
        let space_no_newline = Expr::U8Class(U8Set::from_chars(" \t\r"));
        let space_no_newline_plus = Expr::Quantifier(Box::new(space_no_newline), QuantifierType::OneOrMore);
        
        let line_ws_no_newline = Expr::Choice(vec![space_no_newline_plus.clone(), line_comment_only.clone()]);
        let line_ws_no_newline_star = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(line_ws_no_newline))), 
            QuantifierType::ZeroOrMore
        );
        
        let pattern15 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            line_ws_no_newline_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            line_ws_no_newline_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            line_ws_no_newline_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            line_ws_no_newline_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            line_ws_no_newline_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 15: Space (no newline) + line comments, 5 WS* slots ===");
        let regex15 = pattern15.build();
        println!("DFA states: {}", regex15.dfa.states.len());
        
        // Test 16: All three, but space doesn't include newline
        // Line: "//" [^\n]*
        // Block: "/*" [^*]* "*/"
        // Space: [ \t\r]+ (no newline)
        let line_for_16 = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'\n');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
        ]);
        
        let block_for_16 = Expr::Seq(vec![
            Expr::U8Seq(b"/*".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'*');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
            Expr::U8Seq(b"*/".to_vec()),
        ]);
        
        let full_ws_no_newline = Expr::Choice(vec![
            space_no_newline_plus.clone(),
            line_for_16,
            block_for_16,
        ]);
        let full_ws_no_newline_star = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(full_ws_no_newline))),
            QuantifierType::ZeroOrMore
        );
        
        let pattern16 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            full_ws_no_newline_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            full_ws_no_newline_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            full_ws_no_newline_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            full_ws_no_newline_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            full_ws_no_newline_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 16: Space (no newline) + line + block, 5 WS* slots ===");
        let regex16 = pattern16.build();
        println!("DFA states: {}", regex16.dfa.states.len());
        
        // Test 17: Line + block only (no space at all)
        let line_for_17 = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'\n');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
        ]);
        
        let block_for_17 = Expr::Seq(vec![
            Expr::U8Seq(b"/*".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'*');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
            Expr::U8Seq(b"*/".to_vec()),
        ]);
        
        let line_block_only = Expr::Choice(vec![line_for_17, block_for_17]);
        let line_block_only_star = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(line_block_only))),
            QuantifierType::ZeroOrMore
        );
        
        let pattern17 = Expr::Seq(vec![
            Expr::U8Seq(b"a".to_vec()),
            line_block_only_star.clone(),
            Expr::U8Seq(b"b".to_vec()),
            line_block_only_star.clone(),
            Expr::U8Seq(b"c".to_vec()),
            line_block_only_star.clone(),
            Expr::U8Seq(b"d".to_vec()),
            line_block_only_star.clone(),
            Expr::U8Seq(b"e".to_vec()),
            line_block_only_star.clone(),
            Expr::U8Seq(b"f".to_vec()),
        ]);
        
        println!("\n=== Test 17: Line + block only (no space), 5 WS* slots ===");
        let regex17 = pattern17.build();
        println!("DFA states: {}", regex17.dfa.states.len());
        
        // Test 18-21: Line + block with varying WS* slots
        for n_slots in 1..=4 {
            let mut pattern = vec![Expr::U8Seq(b"a".to_vec())];
            let letters = ['b', 'c', 'd', 'e', 'f'];
            for i in 0..n_slots {
                pattern.push(line_block_only_star.clone());
                pattern.push(Expr::U8Seq(vec![letters[i] as u8]));
            }
            let expr = Expr::Seq(pattern);
            let regex = expr.build();
            println!("Test 18.{}: Line + block, {} WS* slots: {} states", n_slots, n_slots, regex.dfa.states.len());
        }
        
        // Test 19: Line + block with separators that can't appear in comments
        // Use \x00 as separator (not allowed in line comments which exclude \n, and not in block which exclude *)
        // Actually line comments allow everything except \n. Block comments allow everything except *.
        // So \n can't appear in line, and * can't appear in block.
        // Let's use \n as separator (ends line comments, and forces ambiguity resolution)
        
        // Actually, the issue is that line comment body [^\n]* does NOT include \n, so \n forces end.
        // And block comment body [^*]* does NOT include *, but that doesn't help since the separator is different.
        
        // Let me try using newline as separator for line-comment-only pattern:
        let line_only_for_19 = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'\n');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
        ]);
        let line_only_star_19 = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(line_only_for_19))),
            QuantifierType::ZeroOrMore
        );
        
        // Pattern: "\n" WS* "\n" WS* "\n" ...
        // Using \n as separator - this can't appear inside line comment body
        for n_slots in 1..=5 {
            let mut pattern = vec![Expr::U8Seq(b"\n".to_vec())];
            for _ in 0..n_slots {
                pattern.push(line_only_star_19.clone());
                pattern.push(Expr::U8Seq(b"\n".to_vec()));
            }
            let expr = Expr::Seq(pattern);
            let regex = expr.build();
            println!("Test 19.{}: Line only, \\n separator, {} WS* slots: {} states", n_slots, n_slots, regex.dfa.states.len());
        }
        
        // Test 20: Block only with */ as separator
        let block_only_for_20 = Expr::Seq(vec![
            Expr::U8Seq(b"/*".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'*');
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
            Expr::U8Seq(b"*/".to_vec()),
        ]);
        let block_only_star_20 = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(block_only_for_20))),
            QuantifierType::ZeroOrMore
        );
        
        // Use "*/" as separator - can't appear inside block comment (would end it)
        for n_slots in 1..=5 {
            let mut pattern = vec![Expr::U8Seq(b"*/".to_vec())];
            for _ in 0..n_slots {
                pattern.push(block_only_star_20.clone());
                pattern.push(Expr::U8Seq(b"*/".to_vec()));
            }
            let expr = Expr::Seq(pattern);
            let regex = expr.build();
            println!("Test 20.{}: Block only, */ separator, {} WS* slots: {} states", n_slots, n_slots, regex.dfa.states.len());
        }
        
        // Test 21: What about line + block with a separator that can't appear in EITHER?
        // Line excludes \n, block excludes *
        // So we need a multi-character separator that includes both \n and *
        // Let's use "\n*" - can't be in line (has \n) and can't be in block (has *)
        // But actually we need to be careful - the DFA might still have states for partial matches
        
        // Let's try: use a unique separator like "\x00" which we ensure is not in either char class
        // Our line comment uses [^\n], our block uses [^*]
        // So \x00 CAN appear in both! That's not helpful.
        
        // Let's redefine comments to exclude more characters:
        // Line: "//" [^\n\x00]*
        // Block: "/*" [^*\x00]* "*/"
        // Separator: "\x00"
        
        let line_restricted = Expr::Seq(vec![
            Expr::U8Seq(b"//".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'\n');
                set.remove(0);  // \x00
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
        ]);
        
        let block_restricted = Expr::Seq(vec![
            Expr::U8Seq(b"/*".to_vec()),
            Expr::Quantifier(Box::new({
                let mut set = U8Set::all();
                set.remove(b'*');
                set.remove(0);  // \x00
                Expr::U8Class(set)
            }), QuantifierType::ZeroOrMore),
            Expr::U8Seq(b"*/".to_vec()),
        ]);
        
        let line_block_restricted = Expr::Choice(vec![line_restricted, block_restricted]);
        let line_block_restricted_star = Expr::Quantifier(
            Box::new(Expr::Shared(Arc::new(line_block_restricted))),
            QuantifierType::ZeroOrMore
        );
        
        for n_slots in 1..=5 {
            let mut pattern = vec![Expr::U8Seq(vec![0u8])];  // \x00
            for _ in 0..n_slots {
                pattern.push(line_block_restricted_star.clone());
                pattern.push(Expr::U8Seq(vec![0u8]));
            }
            let expr = Expr::Seq(pattern);
            let regex = expr.build();
            println!("Test 21.{}: Line+block (\\x00 excluded), \\x00 separator, {} WS* slots: {} states", n_slots, n_slots, regex.dfa.states.len());
        }
    }
}
