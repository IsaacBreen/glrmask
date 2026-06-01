#[cfg(test)]
mod tests {
    use super::*;

    fn table_with_stack_shifts(
        shifts: Vec<StackShift>,
        goto_rows: &[(u32, &[(NonterminalID, (u32, bool))])],
    ) -> GLRTable {
        let num_states = 8;
        let mut action = vec![ActionRow::default(); num_states];
        action[0].insert(0, Action::StackShifts(shifts));

        let mut goto = vec![GotoRow::default(); num_states];
        for &(state, row) in goto_rows {
            for &(nt, target) in row {
                goto[state as usize].insert(nt, target);
            }
        }

        GLRTable {
            action,
            goto,
            num_states: num_states as u32,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        }
    }

    fn stack_shifts_at_start(table: &GLRTable) -> Vec<StackShift> {
        match table.action(0, 0).expect("expected action at state 0 terminal 0") {
            Action::StackShifts(shifts) => shifts.clone(),
            action => panic!("expected stack shifts, got {action:?}"),
        }
    }

    #[test]
    fn canonicalizes_stack_shift_predecessor_to_goto_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![StackShift {
                pop: 1,
                pushes: vec![1, 3, 4],
            }]
        );
    }

    #[test]
    fn leaves_stack_shift_predecessors_unchanged_when_canonicalization_is_disabled() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors_with_enabled(false);

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn does_not_canonicalize_stack_shift_predecessors_when_shared_goto_target_differs() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (22, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn does_not_canonicalize_empty_goto_row_to_nonempty_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ],
            &[(1, &[(10, (20, true))])],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![2, 3, 4],
                },
            ]
        );
    }

    #[test]
    fn canonicalizes_buried_middle_stack_shift_predecessor_to_goto_superset() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 1, 3, 4],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 2, 3, 4],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![StackShift {
                pop: 1,
                pushes: vec![9, 1, 3, 4],
            }]
        );
    }

    #[test]
    fn does_not_canonicalize_top_pushed_state_even_when_goto_rows_are_compatible() {
        let mut table = table_with_stack_shifts(
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 1],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 2],
                },
            ],
            &[
                (1, &[(10, (20, true)), (11, (21, false))]),
                (2, &[(10, (20, true))]),
            ],
        );

        table.canonicalize_stack_shift_predecessors();

        assert_eq!(
            stack_shifts_at_start(&table),
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 1],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![9, 3, 2],
                },
            ]
        );
    }

    #[test]
    fn reduce_frame_allows_origin_dependent_multiple_goto_targets() {
        let mut table = table_with_stack_shifts(Vec::new(), &[
            (1, &[(10, (3, false))]),
            (2, &[(10, (4, false))]),
        ]);
        table.num_states = 6;
        table.action.resize(6, ActionRow::default());
        table.goto.resize(6, GotoRow::default());

        let mut predecessors = vec![BTreeSet::new(); 6];
        predecessors[5] = BTreeSet::from([1, 2]);

        let result = apply_reduce_to_frame(
            &table,
            &predecessors,
            5,
            StackEffectFrame {
                pop: 0,
                pushes: Vec::new(),
                guards: Vec::new(),
            },
            10,
            1,
            &mut FxHashMap::default(),
        );

        let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
            panic!("expected frames");
        };
        assert!(origin_dependent);
        assert_eq!(
            frames,
            vec![
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![3],
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![1],
                    }],
                },
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![4],
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![2],
                    }],
                },
            ]
        );
    }

    #[test]
    fn reduce_frame_allows_origin_dependent_single_goto_target() {
        let mut table = table_with_stack_shifts(Vec::new(), &[
            (1, &[(10, (3, false))]),
            (2, &[(10, (3, false))]),
        ]);
        table.num_states = 6;
        table.action.resize(6, ActionRow::default());
        table.goto.resize(6, GotoRow::default());

        let mut predecessors = vec![BTreeSet::new(); 6];
        predecessors[5] = BTreeSet::from([1, 2]);

        let result = apply_reduce_to_frame(
            &table,
            &predecessors,
            5,
            StackEffectFrame {
                pop: 0,
                pushes: Vec::new(),
                guards: Vec::new(),
            },
            10,
            1,
            &mut FxHashMap::default(),
        );

        let Some(ReduceFrameResult::Frames { frames, origin_dependent }) = result else {
            panic!("expected frames");
        };
        assert!(origin_dependent);
        assert_eq!(
            frames,
            vec![
                StackEffectFrame {
                    pop: 1,
                    pushes: vec![3],
                    guards: Vec::new(),
                }
            ]
        );
    }

    #[test]
    fn inline_action_to_stack_shifts_keeps_multishift_replacement_reduce_chain() {
        let mut action = vec![ActionRow::default(); 5];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, false)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(4, false));

        let mut goto = vec![GotoRow::default(); 5];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 5,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 5];
        predecessors[2].insert(1);

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut FxHashMap::default(),
        );

        let Some(Action::StackShifts(shifts)) = result else {
            panic!("expected multi-stack-shift action, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                StackShift {
                    pop: 0,
                    pushes: vec![4],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![3, 4],
                },
            ]
        );
    }

    #[test]
    fn inline_action_to_stack_shifts_handles_replace_shift_and_replace_goto() {
        let mut action = vec![ActionRow::default(); 6];
        action[2].insert(
            0,
            Action::Split {
                shift: Some((4, true)),
                reduces: vec![(10, 1)],
                accept: false,
            },
        );
        action[3].insert(0, Action::Shift(5, true));

        let mut goto = vec![GotoRow::default(); 6];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 6,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 6];
        predecessors[2].insert(1);

        let action = table.action(2, 0).expect("expected split action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut FxHashMap::default(),
        );

        let Some(Action::StackShifts(shifts)) = result else {
            panic!("expected replacement stack shifts, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                StackShift {
                    pop: 1,
                    pushes: vec![4],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![5],
                },
            ]
        );
    }

    #[test]
    fn inline_action_to_stack_shifts_guards_divergent_replace_gotos_by_predecessor() {
        let mut action = vec![ActionRow::default(); 9];
        action[2].insert(0, Action::Reduce(10, 1));
        action[3].insert(0, Action::Shift(7, false));
        action[4].insert(0, Action::Shift(8, false));

        let mut goto = vec![GotoRow::default(); 9];
        goto[1].insert(10, (3, true));
        goto[6].insert(10, (4, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 9,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 9];
        predecessors[2].extend([1, 6]);

        let action = table.action(2, 0).expect("expected reduce action");
        let result = try_inline_action_to_stack_shifts(
            &table,
            &predecessors,
            2,
            0,
            action,
            &mut FxHashMap::default(),
            &mut FxHashMap::default(),
        );

        let Some(Action::GuardedStackShifts(shifts)) = result else {
            panic!("expected guarded replacement stack shifts, got {result:?}");
        };
        assert_eq!(
            shifts,
            vec![
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![1],
                    }],
                    pop: 2,
                    pushes: vec![3, 7],
                },
                GuardedStackShift {
                    guards: vec![StackShiftGuard {
                        pop: 1,
                        states: vec![6],
                    }],
                    pop: 2,
                    pushes: vec![4, 8],
                },
            ]
        );
    }

    #[test]
    fn compatible_goto_unit_destination_still_refuses_replace_goto() {
        let action = vec![ActionRow::default(); 4];
        let mut goto = vec![GotoRow::default(); 4];
        goto[1].insert(10, (3, true));

        let table = GLRTable {
            action,
            goto,
            num_states: 4,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        let mut predecessors = vec![BTreeSet::new(); 4];
        predecessors[2].insert(1);

        assert_eq!(unit_reduce_destination(&table, &predecessors, 2, 10), None);
    }

    #[test]
    fn suffix_quotient_collapses_same_pop_stack_shift_fanout() {
        let token0 = 0;
        let token1 = 1;
        let mut action = vec![ActionRow::default(); 8];
        action[0].insert(
            token0,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![1, 2],
                },
                StackShift {
                    pop: 1,
                    pushes: vec![3, 4],
                },
            ]),
        );
        action[2].insert(
            token1,
            Action::StackShifts(vec![
                StackShift {
                    pop: 1,
                    pushes: vec![5],
                },
                StackShift {
                    pop: 2,
                    pushes: vec![6],
                },
            ]),
        );
        action[4].insert(
            token1,
            Action::StackShifts(vec![StackShift {
                pop: 2,
                pushes: vec![7],
            }]),
        );

        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 8],
            num_states: 8,
            num_terminals: 2,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        assert!(matches!(table.action(0, token0), Some(Action::Shift(_, true))));
        assert!(
            table.ambiguous_actions().is_empty(),
            "{:#?}",
            table.ambiguous_actions()
        );
    }

    #[test]
    fn suffix_quotient_preserves_guarded_stack_shift_guards() {
        let token = 0;
        let guard = StackShiftGuard {
            pop: 1,
            states: vec![9],
        };
        let mut action = vec![ActionRow::default(); 12];
        action[0].insert(
            token,
            Action::GuardedStackShifts(vec![
                GuardedStackShift {
                    guards: vec![guard.clone()],
                    pop: 1,
                    pushes: vec![1, 2],
                },
                GuardedStackShift {
                    guards: vec![guard.clone()],
                    pop: 1,
                    pushes: vec![3, 4],
                },
            ]),
        );

        let mut table = GLRTable {
            action,
            goto: vec![GotoRow::default(); 12],
            num_states: 12,
            num_terminals: 1,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        table.rebuild_advance_rows_from_actions();

        table.quotient_recognizer_stack_suffixes();

        let Some(Action::GuardedStackShifts(shifts)) = table.action(0, token) else {
            panic!("expected one guarded stack-shift action");
        };
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0].guards.len(), 1);
        assert_eq!(shifts[0].guards[0].pop, guard.pop);
        assert!(!shifts[0].guards[0].states.is_empty());
        assert_eq!(shifts[0].pop, 1);
        assert_eq!(shifts[0].pushes.len(), 1);
        assert!(
            table.ambiguous_actions().is_empty(),
            "{:#?}",
            table.ambiguous_actions()
        );
    }

    #[test]
    fn suffix_quotient_rolls_back_nested_created_states_on_outer_failure() {
        let outer_suffixes = vec![vec![10, 1], vec![10, 2]];

        let mut table = GLRTable {
            action: vec![ActionRow::default(); 11],
            goto: vec![GotoRow::default(); 11],
            num_states: 11,
            num_terminals: 0,
            num_rules: 0,
            rules: Vec::new(),
            nonterminal_display_names: Vec::new(),
            advance: Vec::new(),
            forwarded_shifts: FxHashSet::default(),
            guarded_shift_index: Vec::new(),
        };
        table.goto[1].insert(0, (3, false));
        table.goto[2].insert(0, (4, false));
        table.goto[1].insert(1, (5, false));
        table.goto[2].insert(1, (6, false));
        table.rebuild_advance_rows_from_actions();

        let original_num_states = table.num_states;
        let original_action_len = table.action.len();
        let original_goto_len = table.goto.len();
        let original_advance_len = table.advance.len();

        let mut quotient = SuffixQuotient {
            suffix_to_state: FxHashMap::default(),
            failed_suffixes: FxHashSet::default(),
            max_states: 2,
            max_alts: 8,
            max_width: 8,
            created_states: 0,
        };

        assert_eq!(
            quotient.ensure_suffix_state(&mut table, outer_suffixes.clone()),
            Err(())
        );
        assert_eq!(table.num_states, original_num_states);
        assert_eq!(table.action.len(), original_action_len);
        assert_eq!(table.goto.len(), original_goto_len);
        assert_eq!(table.advance.len(), original_advance_len);
        assert_eq!(quotient.created_states, 0);
        assert!(quotient.failed_suffixes.contains(&outer_suffixes));
        assert!(
            quotient
                .suffix_to_state
                .values()
                .all(|&state| state < original_num_states)
        );
    }
}
