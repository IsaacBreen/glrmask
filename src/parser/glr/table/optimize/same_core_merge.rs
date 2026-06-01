enum ActionSig {
    Shift(u32, bool),
    StackShifts(Vec<(u32, Vec<u32>)>),
    GuardedStackShifts(Vec<(Vec<(u32, Vec<u32>)>, u32, Vec<u32>)>),
    Reduce(NonterminalID, u32),
    Split {
        shift: Option<(u32, bool)>,
        reduces: Vec<(NonterminalID, u32)>,
        accept: bool,
    },
    Accept,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RowSignature {
    core_class: u32,
    action: Vec<(TerminalID, ActionSig)>,
    goto: Vec<(NonterminalID, (u32, bool))>,
    advance: Option<BitSet>,
}

fn remap_action_to_partition(action: &Action, partition: &[u32]) -> ActionSig {
    match action {
        Action::Shift(target, replace) => ActionSig::Shift(partition[*target as usize], *replace),
        Action::StackShifts(shifts) => ActionSig::StackShifts(
            shifts
                .iter()
                .map(|shift| {
                    (
                        shift.pop,
                        shift.pushes.iter().map(|&state| partition[state as usize]).collect(),
                    )
                })
                .collect(),
        ),
        Action::GuardedStackShifts(shifts) => ActionSig::GuardedStackShifts(
            shifts
                .iter()
                .map(|shift| {
                    let guards = shift
                        .guards
                        .iter()
                        .map(|guard| {
                            let mut states: Vec<u32> = guard
                                .states
                                .iter()
                                .map(|&state| partition[state as usize])
                                .collect();
                            states.sort_unstable();
                            states.dedup();
                            (guard.pop, states)
                        })
                        .collect();
                    let pushes = shift
                        .pushes
                        .iter()
                        .map(|&state| partition[state as usize])
                        .collect();
                    (guards, shift.pop, pushes)
                })
                .collect(),
        ),
        Action::Reduce(nt, len) => ActionSig::Reduce(*nt, *len),
        Action::Split {
            shift,
            reduces,
            accept,
        } => ActionSig::Split {
            shift: shift.map(|(target, replace)| (partition[target as usize], replace)),
            reduces: reduces.clone(),
            accept: *accept,
        },
        Action::Accept => ActionSig::Accept,
    }
}

fn core_classes(core_keys: &[Vec<Item>]) -> Vec<u32> {
    let mut class_of = vec![0; core_keys.len()];
    let mut key_to_class: FxHashMap<Vec<Item>, u32> = FxHashMap::default();
    let mut next = 0u32;

    for (state, key) in core_keys.iter().enumerate() {
        let class = *key_to_class.entry(key.clone()).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        class_of[state] = class;
    }

    class_of
}

fn refine_same_core_partition(table: &GLRTable, core_keys: &[Vec<Item>]) -> Vec<u32> {
    let nstates = table.num_states as usize;
    let has_advance_rows = table.advance.len() == nstates;
    let core_class_of = core_classes(core_keys);
    let mut partition = core_class_of.clone();

    loop {
        let mut sig_to_part: FxHashMap<RowSignature, u32> = FxHashMap::default();
        let mut next_partition = vec![0u32; nstates];
        let mut next_id = 0u32;

        for state in 0..nstates {
            let action = table.action[state]
                .iter()
                .map(|(terminal, action)| {
                    (terminal, remap_action_to_partition(action, &partition))
                })
                .collect();
            let goto = table.goto[state]
                .iter()
                .map(|(&nt, &(target, replace))| (nt, (partition[target as usize], replace)))
                .collect();
            let signature = RowSignature {
                core_class: core_class_of[state],
                action,
                goto,
                advance: has_advance_rows.then(|| table.advance[state].clone()),
            };

            let class = *sig_to_part.entry(signature).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            next_partition[state] = class;
        }

        if next_partition == partition {
            return partition;
        }
        partition = next_partition;
    }
}

pub(super) fn merge_same_core_lr1_states(table: GLRTable, core_keys: &[Vec<Item>]) -> GLRTable {
    let partition = refine_same_core_partition(&table, core_keys);
    let nstates = table.num_states as usize;
    let ngroups = partition.iter().copied().max().map(|x| x + 1).unwrap_or(0) as usize;

    let mut representatives = vec![u32::MAX; ngroups];
    for state in 0..nstates {
        let group = partition[state] as usize;
        if representatives[group] == u32::MAX {
            representatives[group] = state as u32;
        }
    }

    let action = representatives
        .iter()
        .map(|&rep| {
            table.action[rep as usize]
                .iter()
                .map(|(terminal, action)| (terminal, remap_action_targets(action, &partition)))
                .collect()
        })
        .collect();
    let goto = representatives
        .iter()
        .map(|&rep| {
            table.goto[rep as usize]
                .iter()
                .map(|(&nt, &(target, replace))| (nt, (partition[target as usize], replace)))
                .collect()
        })
        .collect();
    let advance = if table.advance.len() == nstates {
        representatives
            .iter()
            .map(|&rep| table.advance[rep as usize].clone())
            .collect()
    } else {
        Vec::new()
    };

    // Remap forwarded_shifts to use merged state IDs
    let forwarded_shifts: FxHashSet<(u32, TerminalID)> = table.forwarded_shifts
        .iter()
        .map(|&(state, terminal)| (partition[state as usize], terminal))
        .collect();

    GLRTable {
        action,
        goto,
        num_states: ngroups as u32,
        num_terminals: table.num_terminals,
        num_rules: table.num_rules,
        rules: table.rules,
        nonterminal_display_names: table.nonterminal_display_names,
        advance,
        forwarded_shifts,
        guarded_shift_index: Vec::new(),
    }
}
