#[cfg(test)]
mod tests {
    use super::{
        ParserGSS,
        advance_stacks,
        apply_guarded_stack_shifts_to_vstack,
        stack_can_advance_on,
        stack_can_advance_on_any,
        try_advance_pop1_reduce_plus_stackshift_wave,
    };
    use crate::parser::glr::accumulator::TerminalsDisallowed;
    use crate::parser::glr::table::testing::build_test_table;
    use crate::parser::glr::table::{Action, GuardedStackShift, StackShift, StackShiftGuard};
    use crate::ds::bitset::BitSet;

    #[test]
    fn advance_stacks_matches_reduce_fanout_collapse_fast_path() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            5,
            1,
            &[
                &[],
                &[],
                &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
                &[(token, Action::StackShifts(vec![StackShift { pop: 2, pushes: vec![7] }]))],
                &[(token, Action::Reduce(nt, 1))],
            ],
            &[
                &[(nt, (2, false))],
                &[(nt, (3, false))],
                &[],
                &[],
                &[],
            ],
        );

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 4], acc.clone()),
            (vec![1, 4], acc),
        ]);
        let expected = ParserGSS::from_single_stack(vec![7], TerminalsDisallowed::new());

        assert_eq!(advance_stacks(&table, &before, token), expected);
    }

    #[test]
    fn advance_stacks_selective_pure_frontier_shift_keeps_only_actionable_top() {
        let token = 0;
        let mut action_rows = vec![Vec::new(); 134];
        action_rows[131] = vec![(
            token,
            Action::StackShifts(vec![StackShift {
                pop: 0,
                pushes: vec![96],
            }]),
        )];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();
        let goto_rows = vec![Vec::new(); 134];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();
        let table = build_test_table(134, 1, &action_refs, &goto_refs);

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 1, 17, 47, 74, 131], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 132], acc.clone()),
            (vec![0_u32, 1, 17, 47, 74, 133], acc),
        ]);
        let expected = ParserGSS::from_single_stack(
            vec![0_u32, 1, 17, 47, 74, 131, 96],
            TerminalsDisallowed::new(),
        );

        assert_eq!(advance_stacks(&table, &before, token), expected);
    }

    #[test]
    fn pop1_reduce_plus_stackshift_wave_fast_path_matches_snowplow_shape() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 989];
        action_rows[655] = vec![(
            token,
            Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
        )];
        action_rows[659] = vec![(
            token,
            Action::StackShifts(vec![
                StackShift { pop: 1, pushes: vec![654] },
                StackShift { pop: 1, pushes: vec![988] },
            ]),
        )];
        action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();

        let mut goto_rows = vec![Vec::new(); 989];
        goto_rows[87] = vec![(nt, (659, true))];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();

        let table = build_test_table(989, 1, &action_refs, &goto_refs);
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 987], acc.clone()),
            (vec![0_u32, 87, 655], acc),
        ]);
        let expected = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 975], TerminalsDisallowed::new()),
            (vec![0_u32, 654], TerminalsDisallowed::new()),
            (vec![0_u32, 988], TerminalsDisallowed::new()),
        ]);

        let mut fast_stacks = try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token)
            .expect("fast path should match this wave")
            .to_stacks();
        let mut expected_stacks = expected.to_stacks();
        fast_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(fast_stacks, expected_stacks);

        let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks();
        actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        assert_eq!(actual_stacks, expected_stacks);
    }

    #[test]
    fn pop1_reduce_plus_stackshift_wave_rejects_cross_product_base() {
        let token = 0;
        let nt = 0;
        let mut action_rows = vec![Vec::new(); 989];
        action_rows[655] = vec![(
            token,
            Action::StackShifts(vec![StackShift { pop: 1, pushes: vec![975] }]),
        )];
        action_rows[659] = vec![(
            token,
            Action::StackShifts(vec![
                StackShift { pop: 1, pushes: vec![654] },
                StackShift { pop: 1, pushes: vec![988] },
            ]),
        )];
        action_rows[987] = vec![(token, Action::Reduce(nt, 1))];
        let action_refs: Vec<&[(u32, Action)]> =
            action_rows.iter().map(|row| row.as_slice()).collect();

        let mut goto_rows = vec![Vec::new(); 989];
        goto_rows[87] = vec![(nt, (659, true))];
        let goto_refs: Vec<&[(u32, (u32, bool))]> =
            goto_rows.iter().map(|row| row.as_slice()).collect();

        let table = build_test_table(989, 1, &action_refs, &goto_refs);
        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0_u32, 87, 987], acc.clone()),
            (vec![1_u32, 87, 655], acc),
        ]);

        assert_eq!(
            try_advance_pop1_reduce_plus_stackshift_wave(&table, &before, token),
            None
        );
    }

    #[test]
    fn can_advance_consults_admission_rows_not_execution_actions() {
        let token = 0;
        let mut table = build_test_table(
            2,
            1,
            &[&[], &[(token, Action::Shift(1, false))]],
            &[&[], &[]],
        );
        table.advance[1].clear(token as usize);

        let stack = ParserGSS::from_single_stack(vec![1], TerminalsDisallowed::new());
        assert!(table.action(1, token).is_some());
        assert!(!stack_can_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(!stack_can_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn can_advance_rechecks_guarded_stack_shifts_against_concrete_stack() {
        let token = 0;
        let table = build_test_table(
            3,
            1,
            &[
                &[],
                &[],
                &[(
                    token,
                    Action::GuardedStackShifts(vec![GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![0],
                        }],
                        pop: 2,
                        pushes: vec![7],
                    }]),
                )],
            ],
            &[&[], &[], &[]],
        );

        let stack = ParserGSS::from_single_stack(vec![1, 2], TerminalsDisallowed::new());

        assert!(table.advance_row_allows(2, token));
        assert!(advance_stacks(&table, &stack, token).is_empty());
        assert!(!stack_can_advance_on(&table, &stack, token));

        let mut terminals = BitSet::new(1);
        terminals.set(token as usize);
        assert!(!stack_can_advance_on_any(&table, &stack, &terminals));
    }

    #[test]
    fn advance_stacks_materializes_single_concrete_path_for_split() {
        let token = 0;
        let nt = 0;
        let table = build_test_table(
            6,
            1,
            &[
                &[],
                &[],
                &[(token, Action::Split {
                    shift: Some((3, false)),
                    reduces: vec![(nt, 1)],
                    accept: false,
                })],
                &[],
                &[(token, Action::Shift(5, false))],
                &[],
            ],
            &[
                &[(nt, (4, false))],
                &[],
                &[],
                &[],
                &[],
                &[],
            ],
        );

        let acc = TerminalsDisallowed::new();
        let before = ParserGSS::from_stacks(&[
            (vec![0, 1], acc.clone()),
            (vec![0, 2], acc.clone()),
        ])
        .popn(1)
        .push(2);
        let expected = ParserGSS::from_stacks(&[
            (vec![0, 2, 3], acc.clone()),
            (vec![0, 4, 5], acc),
        ]);

        let mut actual_stacks = advance_stacks(&table, &before, token).to_stacks();
        let mut expected_stacks = expected.to_stacks();
        actual_stacks.sort_by(|left, right| left.0.cmp(&right.0));
        expected_stacks.sort_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(actual_stacks, expected_stacks);
    }

    #[test]
    fn indexed_guarded_vstack_matches_linear_guarded_vstack() {
        let token = 0;
        let mut table = build_test_table(
            1,
            1,
            &[&[(
                token,
                Action::GuardedStackShifts(vec![
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![10, 20],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![1],
                            },
                        ],
                        pop: 3,
                        pushes: vec![50],
                    },
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![10],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![2],
                            },
                        ],
                        pop: 3,
                        pushes: vec![51],
                    },
                    GuardedStackShift {
                        guards: vec![StackShiftGuard {
                            pop: 1,
                            states: vec![10, 20],
                        }],
                        pop: 2,
                        pushes: vec![52],
                    },
                    GuardedStackShift {
                        guards: vec![
                            StackShiftGuard {
                                pop: 1,
                                states: vec![30],
                            },
                            StackShiftGuard {
                                pop: 2,
                                states: vec![1],
                            },
                        ],
                        pop: 3,
                        pushes: vec![53],
                    },
                ]),
            )]],
            &[&[]],
        );
        table.rebuild_guarded_shift_index();

        let shifts = match table.action(0, token) {
            Some(Action::GuardedStackShifts(shifts)) => shifts,
            other => panic!("expected guarded stack shifts, got {other:?}"),
        };
        let index = table
            .guarded_shift_index(0, token)
            .expect("expected guarded shift index");

        let stack_a = ParserGSS::from_single_stack(vec![0, 1, 10, 99], TerminalsDisallowed::new());
        let stack_b = ParserGSS::from_single_stack(vec![0, 2, 10, 99], TerminalsDisallowed::new());

        for stack in [&stack_a, &stack_b] {
            let vstack = stack.try_virtual_stack().expect("expected virtual stack");
            let mut indexed = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, Some(index)).to_stacks();
            let mut linear = apply_guarded_stack_shifts_to_vstack(&vstack, shifts, None).to_stacks();
            indexed.sort_by(|left, right| left.0.cmp(&right.0));
            linear.sort_by(|left, right| left.0.cmp(&right.0));
            assert_eq!(indexed, linear);
        }
    }
}

/// Precise predicate for whether this parser stack can advance on any terminal in
/// `terminals`.
///
/// Returns `true` if and only if at least one current parser path can definitely
/// advance on one of the given terminals. Returns `false` if no current parser
/// path can advance on any of them.
///
/// Ordinary actions are applicable from the top state/action row. In particular,
/// LR(1) reduce lookaheads are precise: if a row has a reduce action for one of
/// these terminals, that reduce is a valid parser transition for that lookahead
/// under the table invariants; it does not require an additional lower-stack
/// guard check here. `GuardedStackShifts` also have lower-stack predicates, so
/// they must evaluate their guards against the current GSS before this predicate
/// can return `true`.
///
