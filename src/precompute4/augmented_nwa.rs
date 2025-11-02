use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use crate::glr::grammar::regex_name;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{
    compute_all_characterizations, compute_below_bottom_characterization, BelowBottomCharacterization,
};
use crate::weighted_automata::{
    NWA as WaNWA,
    NWAStates as WaNWAStates,
    NWARest as WaNWARest,
    StateID as WaStateID,
    Weight as WaWeight,
};

/// Error while building an AugmentedNwa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentedNwaBuildError {
    ParserStateIdOutOfRange { state_id: ParserStateID },
}

/// Per-terminal augmented NWA metadata:
/// - `nwa` is the NWA meta (start state).
/// - `nt_nodes` maps each nonterminal to its dedicated node in the NWA.
/// - `end_map` accumulates example stacks that reach designated end states.
///
/// Alphabet: parser StateID values encoded as u16 (fails if any StateID > u16::MAX).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AugmentedNwaRest {
    pub nwa: WaNWARest,
    pub nt_nodes: BTreeMap<NonTerminalID, WaStateID>,
    /// Map from NWA state id to sets of parser stacks.
    /// Each parser stack is represented as a Vec<ParserStateID>.
    pub end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>>,
}

/// Per-terminal augmented NWA bundle:
/// - `states` are the underlying NWA states
/// - `rest` holds meta and auxiliary maps (nt_nodes, end_map)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AugmentedNwa {
    pub states: WaNWAStates,
    pub rest: AugmentedNwaRest,
}

fn encode_symbol(id: ParserStateID) -> Result<u16, AugmentedNwaBuildError> {
    u16::try_from(id.0).map_err(|_| AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id: id })
}

/// Build an augmented NWA for a single terminal by first computing its
/// BelowBottomCharacterization, then materializing the NWA.
pub fn build_augmented_nwa_for_terminal(
    parser: &GLRParser,
    terminal_id: TerminalID,
) -> Result<AugmentedNwa, AugmentedNwaBuildError> {
    let bb = compute_below_bottom_characterization(parser, terminal_id);
    build_augmented_nwa_from_characterization(parser, &bb)
}

/// Identity NWA for ignore terminals: passes through any stack unchanged.
pub fn build_augmented_nwa_for_ignore_terminal() -> AugmentedNwa {
    let mut states = WaNWAStates::default();
    let start_state = states.add_state();

    let mut end_map = BTreeMap::new();
    end_map.insert(start_state, BTreeSet::from([vec![]]));

    AugmentedNwa {
        states,
        rest: AugmentedNwaRest {
            nwa: WaNWARest { start_state },
            nt_nodes: BTreeMap::new(),
            end_map,
        },
    }
}

/// Build augmented NWAs for all terminals.
pub fn build_augmented_nwas(
    parser: &GLRParser,
) -> Result<BTreeMap<TerminalID, AugmentedNwa>, AugmentedNwaBuildError> {
    let all = compute_all_characterizations(parser);

    crate::debug!(5, "\n--- Terminal Characterizations ---");
    for (terminal_id, bb) in &all {
        let terminal = parser.terminal_map.get_by_right(terminal_id).cloned().unwrap_or(regex_name("UNKNOWN"));
        crate::debug!(5, "Terminal {} ({}) Characterization:\n{}", terminal_id.0, terminal, bb);
    }
    crate::debug!(5, "--- End Terminal Characterizations ---\n");

    let mut out = BTreeMap::new();
    for (term, bb) in all {
        let aug = build_augmented_nwa_from_characterization(parser, &bb)?;
        out.insert(term, aug);
    }
    Ok(out)
}

/// Core builder: turns a BelowBottomCharacterization into an AugmentedNwa.
///
/// Construction rules:
/// - Create one initial start node, one unique `end_state`, and one node per nonterminal.
/// - Initial shifts: start --(initial_state)--> end_state; record [initial_state, shift_state].
/// - Initial reduces (initial_state, len, nt): first step on initial_state, then len default steps to nt.
/// - Per-nonterminal reduces:
///   - Reveal-and-rereduces (revealed_state, len, reduce_nt): first on revealed_state, then len default steps to reduce_nt.
///   - Reveal-goto-shift escapes: from nt node, on revealed_state go to end_state; record [revealed_state, goto_state, shift_state].
pub fn build_augmented_nwa_from_characterization(
    parser: &GLRParser,
    bb: &BelowBottomCharacterization,
) -> Result<AugmentedNwa, AugmentedNwaBuildError> {
    let mut states = WaNWAStates::default();
    let start = states.add_state();
    let end_state = states.add_state();

    let mut nt_nodes: BTreeMap<NonTerminalID, WaStateID> = BTreeMap::new();
    for &nt in parser.non_terminal_map.right_values() {
        let id = states.add_state();
        nt_nodes.insert(nt, id);
    }

    let mut end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();
    let w_all = WaWeight::all();

    for &(initial_state, shift_state) in &bb.initial_shifts {
        let ch = encode_symbol(initial_state)?;
        states.add_transition(start, ch, end_state, w_all.clone());
        end_map.entry(end_state).or_default().insert(vec![initial_state, shift_state]);
    }

    for &(initial_state, len, nt) in &bb.initial_reduces {
        let ch = encode_symbol(initial_state)?;
        let target_nt = *nt_nodes.get(&nt).expect("nonterminal node must exist");
        let mut from = start;
        let next_state = if len == 0 { target_nt } else { states.add_state() };
        states.add_transition(from, ch, next_state, w_all.clone());
        from = next_state;
        for i in 0..len {
            let to = if i == len - 1 { target_nt } else { states.add_state() };
            states.add_default_transition(from, to, w_all.clone());
            from = to;
        }
    }

    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt = *nt_nodes.get(nt).expect("reduce_characterizations nt must exist");

        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let ch = encode_symbol(revealed_state)?;
            let dst_nt = *nt_nodes.get(&reduce_nt).expect("reduce target nonterminal must exist");
            let mut from = src_nt;
            let next_state = if len == 0 { dst_nt } else { states.add_state() };
            states.add_transition(from, ch, next_state, w_all.clone());
            from = next_state;
            for i in 0..len {
                let to = if i == len - 1 { dst_nt } else { states.add_state() };
                states.add_default_transition(from, to, w_all.clone());
                from = to;
            }
        }

        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let ch = encode_symbol(revealed_state)?;
            states.add_transition(src_nt, ch, end_state, w_all.clone());
            end_map.entry(end_state).or_default().insert(vec![revealed_state, goto_state, shift_state]);
        }
    }

    Ok(AugmentedNwa {
        states,
        rest: AugmentedNwaRest {
            nwa: WaNWARest { start_state: start },
            nt_nodes,
            end_map,
        }
    })
}

impl AugmentedNwa {
    /// Process a parser-state stack through this augmented NWA.
    pub fn process_stack(
        &self,
        stack: &[ParserStateID],
    ) -> Result<Vec<(WaStateID, WaStateID, WaWeight)>, AugmentedNwaBuildError> {
        let mut encoded: Vec<u16> = Vec::with_capacity(stack.len());
        for &s in stack { encoded.push(encode_symbol(s)?); }
        Ok(self.states.process_stack_u16(&self.rest.nwa, &encoded))
    }

    /// Combine helper when both left and right reside in the SAME shared states buffer.
    /// This does not copy states; it only wires epsilons and updates end_map in left.
    pub fn combine_right_into_on_shared_states(
        shared_states: &mut WaNWAStates,
        left: &mut AugmentedNwaRest,
        right: &AugmentedNwaRest,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        let left_end_snapshot: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = left.end_map.clone();
        let mut new_end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();

        for (left_end_state, stacks) in left_end_snapshot {
            for left_stack in stacks {
                // Consume from top-of-stack first
                let mut encoded: Vec<u16> = Vec::with_capacity(left_stack.len());
                for &s in left_stack.iter().rev() { encoded.push(encode_symbol(s)?); }
                let stops = shared_states.process_stack_u16(&right.nwa, &encoded);

                for (pos, right_stop_state, path_weight) in stops {
                    let combined_weight = &path_weight & weight;
                    shared_states.add_epsilon_transition(left_end_state, right_stop_state, combined_weight);

                    let reachable = shared_states.reachable_states_ignoring_labels(right_stop_state);
                    for r_state in reachable {
                        if let Some(r_stacks) = right.end_map.get(&r_state) {
                            for r_stack in r_stacks {
                                let keep_len = left_stack.len().saturating_sub(pos);
                                let mut combined: Vec<ParserStateID> = left_stack[..keep_len].to_vec();
                                combined.extend(r_stack.iter().cloned());
                                new_end_map.entry(r_state).or_default().insert(combined);
                            }
                        }
                    }
                }
            }
        }
        left.end_map = new_end_map;
        Ok(())
    }

    /// Rebase a disjoint AugmentedNwa onto an existing shared states storage.
    pub fn rebase_onto_shared(
        shared_states: &mut WaNWAStates,
        src_states: &WaNWAStates,
        rest: &mut AugmentedNwaRest,
    ) -> Vec<WaStateID> {
        let mapping = shared_states.append_copy_from(src_states);

        rest.nwa.start_state = mapping[rest.nwa.start_state];
        let mut new_nt_nodes = BTreeMap::new();
        for (nt, st) in rest.nt_nodes.iter() { new_nt_nodes.insert(*nt, mapping[*st]); }
        rest.nt_nodes = new_nt_nodes;

        let mut new_end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();
        for (k, v) in rest.end_map.iter() { new_end_map.insert(mapping[*k], v.clone()); }
        rest.end_map = new_end_map;

        mapping
    }

    /// Non-commutative combination: fold `right` into `self` (left).
    ///
    /// Implementation note: rebase the right's states/rest into `self` first, then combine on shared states.
    pub fn combine_right_into(
        &mut self,
        right: &AugmentedNwa,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        let mut remapped_right = right.rest.clone();
        Self::rebase_onto_shared(&mut self.states, &right.states, &mut remapped_right);
        Self::combine_right_into_on_shared_states(&mut self.states, &mut self.rest, &remapped_right, weight)
    }

    /// Union of two augmented NWAs into self.
    pub fn union_with(&mut self, other: &AugmentedNwa) {
        let mapping = self.states.append_copy_from(&other.states);

        for (other_end_state, stacks) in &other.rest.end_map {
            let mapped_end_state = mapping[*other_end_state];
            self.rest.end_map.entry(mapped_end_state).or_default().extend(stacks.clone());
        }

        let new_start = self.states.add_state();
        let old_self_start = self.rest.nwa.start_state;
        let old_other_start_mapped = mapping[other.rest.nwa.start_state];

        self.states.add_epsilon_transition(new_start, old_self_start, WaWeight::all());
        self.states.add_epsilon_transition(new_start, old_other_start_mapped, WaWeight::all());
        self.rest.nwa.start_state = new_start;
    }
}

impl Display for AugmentedNwa {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Augmented NWA")?;
        writeln!(f, "  Nonterminal Nodes:")?;
        for (nt, state) in &self.rest.nt_nodes {
            writeln!(f, "    - NT {}: State {}", nt.0, state)?;
        }
        if !self.rest.end_map.is_empty() {
            writeln!(f, "  End Map:")?;
            for (state, stacks) in &self.rest.end_map {
                writeln!(f, "    - State {}:", state)?;
                for stack in stacks {
                    let stack_str: Vec<String> = stack.iter().map(|s| s.0.to_string()).collect();
                    writeln!(f, "      - [{}]", stack_str.join(", "))?;
                }
            }
        }
        writeln!(f, "  Underlying NWA:")?;
        let nwa = WaNWA { states: self.states.clone(), rest: self.rest.nwa.clone() };
        for line in nwa.to_string().lines() {
            writeln!(f, "    {}", line)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_symbol_fails_on_large_state_ids() {
        #[cfg(target_pointer_width = "64")]
        {
            let big: ParserStateID = ParserStateID(u32::MAX as usize + 10usize);
            let err = super::encode_symbol(big).unwrap_err();
            match err {
                AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id } => assert_eq!(state_id, big),
            }
        }
    }

    fn build_simple_aug_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let end = states.add_state();
        states.add_transition(start, 100, end, WaWeight::all());

        let mut end_map = BTreeMap::new();
        end_map.insert(
            end,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        );

        AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes: BTreeMap::new(),
                end_map,
            },
        }
    }

    #[test]
    fn test_combine_with_ignore_on_left() {
        let mut lhs = build_augmented_nwa_for_ignore_terminal();
        let mut rhs = build_simple_aug_nwa();
        rhs.states.set_final_weight(1, WaWeight::all());
        let weight = WaWeight::all();

        crate::debug!(5, "Left NWA (ignore):\n{}", lhs);
        crate::debug!(5, "Right NWA (simple):\n{}", rhs);

        lhs.combine_right_into(&rhs, &weight).unwrap();

        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        states.add_epsilon_transition(start, s1, WaWeight::all());
        states.add_transition(s1, 100, s2, WaWeight::all());
        states.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            s2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes: BTreeMap::new(),
                end_map: expected_end_map,
            },
        };

        crate::debug!(5, "Expected NWA:\n{}", expected_aug_nwa);
        crate::debug!(5, "Resulting NWA:\n{}", lhs);

        assert_eq!(lhs, expected_aug_nwa);
    }

    #[test]
    fn test_combine_with_ignore_on_right() {
        let mut lhs = build_simple_aug_nwa();
        let mut rhs = build_augmented_nwa_for_ignore_terminal();
        rhs.states.set_final_weight(rhs.rest.nwa.start_state, WaWeight::all());
        let weight = WaWeight::all();

        lhs.combine_right_into(&rhs, &weight).unwrap();

        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        states.add_transition(start, 100, s1, WaWeight::all());
        states.add_epsilon_transition(s1, s2, WaWeight::all());
        states.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            s2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes: BTreeMap::new(),
                end_map: expected_end_map,
            },
        };

        assert_eq!(lhs, expected_aug_nwa);
    }

    fn build_terminal1_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(start, 0, end_state, WaWeight::all());
        states.add_transition(start, 1, nt1_state, WaWeight::all());
        states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]),
        )]);

        AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes,
                end_map,
            },
        }
    }

    fn build_terminal2_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(start, 1, nt1_state, WaWeight::all());
        states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes,
                end_map: BTreeMap::new(),
            },
        }
    }

    fn build_terminal0_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(start, 1, nt1_state, WaWeight::all());
        states.add_transition(start, 2, end_state, WaWeight::all());
        states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(2), ParserStateID(0)]]),
        )]);

        AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes,
                end_map,
            },
        }
    }

    #[test]
    fn test_right_to_left_combination() {
        let mut states = WaNWAStates::default();
        let initial_state = states.add_state();
        states.set_final_weight(initial_state, WaWeight::all());
        let mut current_rest = AugmentedNwaRest {
            nwa: WaNWARest { start_state: initial_state },
            nt_nodes: BTreeMap::new(),
            end_map: BTreeMap::from([(initial_state, BTreeSet::from([vec![]]))]),
        };

        let terminal_nwas_with_id = vec![
            (0, build_terminal0_nwa()),
            (2, build_terminal2_nwa()),
            (1, build_terminal1_nwa()),
        ];
        let weight = WaWeight::all();

        for (i, (term_id, term_nwa)) in terminal_nwas_with_id.iter().rev().enumerate() {
            crate::debug!(5, "\n--- Combination Step {} (Term {} on LEFT) ---", i, term_id);
            crate::debug!(5, "LEFT NWA (Term {}):\n{}", term_id, term_nwa);
            crate::debug!(5, "RIGHT NWA (Current): {:?}", current_rest.nwa.start_state);

            let mut left_rest = term_nwa.rest.clone();
            AugmentedNwa::rebase_onto_shared(&mut states, &term_nwa.states, &mut left_rest);

            AugmentedNwa::combine_right_into_on_shared_states(&mut states, &mut left_rest, &current_rest, &weight).unwrap();
            current_rest = left_rest;

            let left_dbg = AugmentedNwa { states: states.clone(), rest: current_rest.clone() };
            crate::debug!(5, "Resulting Combined NWA:\n{}", left_dbg);
        }

        assert!(current_rest.nwa.start_state < states.len());
    }

    fn build_nwa_from_prompt_left() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let end_state = states.add_state();
        let nt0_state = states.add_state();
        let nt1_state = states.add_state();

        states.add_transition(start, 1, nt1_state, WaWeight::all());
        states.add_transition(start, 2, end_state, WaWeight::all());
        states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(2), ParserStateID(0)]]),
        )]);

        AugmentedNwa { states, rest: AugmentedNwaRest { nwa: WaNWARest { start_state: start }, nt_nodes, end_map } }
    }

    fn build_nwa_from_prompt_right() -> AugmentedNwa {
        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        let s3 = states.add_state();
        let s4 = states.add_state();
        let end_state = states.add_state();

        let w3 = WaWeight::from_item(3);
        states.add_epsilon_transition(start, s1, w3.clone());
        states.add_transition(s1, 0, s2, WaWeight::all());
        states.add_transition(s1, 1, s4, WaWeight::all());
        states.add_transition(s1, 3, s3, WaWeight::all());
        states.add_epsilon_transition(s2, end_state, w3.clone());
        states.set_final_weight(end_state, WaWeight::all());

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]),
        )]);

        AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes: BTreeMap::new(),
                end_map,
            },
        }
    }

    #[test]
    fn test_combination_from_prompt_example() {
        let mut self_nwa = build_nwa_from_prompt_left();
        let right_nwa = build_nwa_from_prompt_right();
        let weight = WaWeight::all();
        crate::debug!(5, "RIGHT NWA:\n{}", right_nwa);
        crate::debug!(5, "LEFT NWA:\n{}", self_nwa);

        self_nwa.combine_right_into(&right_nwa, &weight).unwrap();

        let mut states = WaNWAStates::default();
        let start = states.add_state();
        let s1 = states.add_state();
        let s2 = states.add_state();
        let s3 = states.add_state();
        let s4 = states.add_state();
        let s5 = states.add_state();
        let s6 = states.add_state();
        let s7 = states.add_state();
        let s8 = states.add_state();
        let s9 = states.add_state();

        states.add_transition(start, 1, s3, WaWeight::all());
        states.add_transition(start, 2, s1, WaWeight::all());
        states.add_transition(start, 3, s2, WaWeight::all());

        let w3 = WaWeight::from_item(3);
        states.add_epsilon_transition(s1, s9, w3.clone());

        states.add_epsilon_transition(s4, s5, w3.clone());
        states.add_transition(s5, 0, s6, WaWeight::all());
        states.add_transition(s5, 1, s8, WaWeight::all());
        states.add_transition(s5, 3, s7, WaWeight::all());
        states.add_epsilon_transition(s6, s9, w3.clone());
        states.set_final_weight(s9, WaWeight::all());

        let expected_nt_nodes = BTreeMap::from([
            (NonTerminalID(0), s2),
            (NonTerminalID(1), s3),
        ]);

        let expected_end_map = BTreeMap::from([(
            s9,
            BTreeSet::from([vec![ParserStateID(2), ParserStateID(0), ParserStateID(1)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            rest: AugmentedNwaRest {
                nwa: WaNWARest { start_state: start },
                nt_nodes: expected_nt_nodes,
                end_map: expected_end_map,
            },
        };

        crate::debug!(5, "Expected NWA:\n{}", expected_aug_nwa);

        assert_eq!(self_nwa, expected_aug_nwa);
    }
}
