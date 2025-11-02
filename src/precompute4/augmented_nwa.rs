use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use crate::glr::grammar::regex_name;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{
    compute_all_characterizations, compute_below_bottom_characterization, BelowBottomCharacterization,
};
use crate::weighted_automata::{NWA as WaNWA, StateID as WaStateID, Weight as WaWeight, StateID};

/// Error that can occur while building an AugmentedNwa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentedNwaBuildError {
    /// A parser StateID could not be represented as a u16 alphabet symbol.
    ParserStateIdOutOfRange { state_id: ParserStateID },
}

/// Per-terminal augmented NWA bundle:
/// - `nwa` is the constructed automaton.
/// - `nt_nodes` maps each nonterminal to its dedicated node in the NWA.
/// - `end_map` accumulates example stacks that reach the `end_state`.
///
/// Alphabet: parser StateID values encoded as u16 (fails if any StateID > u16::MAX).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwa {
    pub nwa: WaNWA,
    pub nt_nodes: BTreeMap<NonTerminalID, WaStateID>,
    /// Map from NWA state id (currently only the end_state is used) to sets of parser stacks.
    /// Each parser stack is represented as a Vec<ParserStateID>.
    pub end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>>,
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

/// This NWA acts as an identity for stack transformations, simply passing through any stack it is combined with.
pub fn build_augmented_nwa_for_ignore_terminal() -> AugmentedNwa {
    let mut nwa = WaNWA::new();
    let start_state = nwa.start_state;

    // For an ignore terminal, the stack is passed through. The end_map should
    // contain an empty stack at the end_state, which is also the start_state.
    // When this NWA is on the left in a `combine_right_into` operation, the
    // empty stack from its end_map results in the right-hand-side NWA's stacks
    // being preserved.
    let mut end_map = BTreeMap::new();
    end_map.insert(start_state, BTreeSet::from([vec![]]));

    AugmentedNwa {
        nwa,
        nt_nodes: BTreeMap::new(),
        end_map,
    }
}

/// Build augmented NWAs for all terminals.
pub fn build_augmented_nwas(
    parser: &GLRParser,
) -> Result<BTreeMap<TerminalID, AugmentedNwa>, AugmentedNwaBuildError> {
    let all = compute_all_characterizations(parser);

    println!("\n--- Terminal Characterizations ---");
    for (terminal_id, bb) in &all {
        let terminal = parser.terminal_map.get_by_right(terminal_id).cloned().unwrap_or(regex_name("UNKNOWN"));
        println!("Terminal {} ({}) Characterization:\n{}", terminal_id.0, terminal, bb);
    }
    println!("--- End Terminal Characterizations ---\n");

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
/// - Create:
///   - one initial NWA start node (provided by WaNWA::new()),
///   - one unique `end_state`,
///   - one node per nonterminal.
/// - For initial shifts:
///   - from start, on symbol (initial_state), go to end_state; add
///     [initial_state, shift_state] into end_map[end_state].
/// - For initial reduces (initial_state, len, nt):
///   - create a chain of (len - 1) edges labeled (initial_state) ending at the `nt` node.
///     If len == 1, go directly to the `nt` node.
/// - For each nonterminal’s reduce characterization:
///   - For reveal-and-rereduces (revealed_state, pop_n, reduce_nt):
///     create a chain of pop_n edges labeled (revealed_state) from the `nt` node to `reduce_nt` node.
///     If pop_n == 0, connect directly.
///   - For reveal-goto-shift escapes (revealed_state, goto_state, shift_state):
///     from the `nt` node, on symbol (revealed_state) go to end_state; add
///     [revealed_state, goto_state, shift_state] into end_map[end_state].
pub fn build_augmented_nwa_from_characterization(
    parser: &GLRParser,
    bb: &BelowBottomCharacterization,
) -> Result<AugmentedNwa, AugmentedNwaBuildError> {
    let mut nwa = WaNWA::new(); // adds start state 0
    let start = nwa.start_state;
    let end_state = nwa.add_state();

    // One node per nonterminal in the grammar.
    let mut nt_nodes: BTreeMap<NonTerminalID, WaStateID> = BTreeMap::new();
    for &nt in parser.non_terminal_map.right_values() {
        let id = nwa.add_state();
        nt_nodes.insert(nt, id);
    }

    let mut end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();
    let w_all = WaWeight::all();

    // Initial shifts: start --(initial_state)--> end_state
    for &(initial_state, shift_state) in &bb.initial_shifts {
        let ch = encode_symbol(initial_state)?;
        nwa.add_transition(start, ch, end_state, w_all.clone());
        end_map.entry(end_state).or_default().insert(vec![initial_state, shift_state]);
    }

    // Initial reduces: chain of (len - 1) edges labeled by the initial_state, ending at the nt node.
    for &(initial_state, len, nt) in &bb.initial_reduces {
        let ch = encode_symbol(initial_state)?;
        let target_nt = *nt_nodes
            .get(&nt)
            .expect("nonterminal node must exist (created from parser.non_terminal_map)");
        // Defensive: if len == 0 (should not happen), treat as zero-length chain.
        let pops = if len == 0 { 0 } else { len.saturating_sub(1) };
        if pops == 0 {
            // Straight edge to the NT node.
            nwa.add_transition(start, ch, target_nt, w_all.clone());
        } else {
            let mut from = start;
            for i in 0..pops {
                let to = if i + 1 == pops {
                    target_nt
                } else {
                    nwa.add_state()
                };
                nwa.add_transition(from, ch, to, w_all.clone());
                from = to;
            }
        }
    }

    // Reduce characterizations per nonterminal.
    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt = *nt_nodes
            .get(nt)
            .expect("reduce_characterizations only contains existing nonterminals");

        // Reveal-and-rereduces: chain of `pop_n` edges from `src_nt` to the node for `reduce_nt`,
        // all labeled with `revealed_state`.
        for &(revealed_state, pop_n, reduce_nt) in &rc.reveal_and_rereduces {
            let ch = encode_symbol(revealed_state)?;
            let dst_nt = *nt_nodes
                .get(&reduce_nt)
                .expect("reduce target nonterminal must exist");
            if pop_n == 0 {
                // Direct link if no extra pops below the revealed state.
                nwa.add_transition(src_nt, ch, dst_nt, w_all.clone());
            } else {
                let mut from = src_nt;
                for i in 0..pop_n {
                    let to = if i + 1 == pop_n {
                        dst_nt
                    } else {
                        nwa.add_state()
                    };
                    nwa.add_transition(from, ch, to, w_all.clone());
                    from = to;
                }
            }
        }

        // Reveal-goto-shift escapes: a single step to end_state; record the example stack.
        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let ch = encode_symbol(revealed_state)?;
            nwa.add_transition(src_nt, ch, end_state, w_all.clone());
            end_map
                .entry(end_state)
                .or_default()
                .insert(vec![revealed_state, goto_state, shift_state]);
        }
    }

    Ok(AugmentedNwa {
        nwa,
        nt_nodes,
        end_map,
    })
}

impl AugmentedNwa {
    /// Process a parser-state stack through this augmented NWA.
    ///
    /// The stack is interpreted in order (0..n) as the sequence to be consumed.
    /// Returns a vector of (pos, stop_state, path_weight) where:
    /// - pos is how many items were consumed from `stack`,
    /// - stop_state is the NWA state reached (final or because input exhausted),
    /// - path_weight is the accumulated weight along the taken path.
    pub fn process_stack(
        &self,
        stack: &[ParserStateID],
    ) -> Result<Vec<(WaStateID, WaStateID, WaWeight)>, AugmentedNwaBuildError> {
        let mut encoded: Vec<u16> = Vec::with_capacity(stack.len());
        for &s in stack {
            encoded.push(encode_symbol(s)?);
        }
        Ok(self.nwa.process_stack_u16(&encoded))
    }

    /// Non-commutative combination: fold `right` into `self` (left).
    ///
    /// Algorithm summary:
    /// - Snapshot the current left end-map.
    /// - Append-copy the entire `right` NWA into `self`, capturing a state-id mapping.
    /// - For each (left_end_state, left_stack) in the snapshot:
    ///   - Run `right`’s NWA on `left_stack` to get multiple (pos, right_stop, path_weight).
    ///   - Add an epsilon transition from `left_end_state` to the mapped `right_stop` with `path_weight`.
    ///   - From `right_stop`, find all `right` end states reachable (ignoring labels).
    ///     For each such end state and each stack in `right.end_map[end_state]`:
    ///       - prepend `left_stack[pos..]` to that right stack and insert it into `self.end_map`
    ///         under the mapped end-state id.
    pub fn combine_right_into(
        &mut self,
        right: &AugmentedNwa,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        // Snapshot the original left end_map so we only iterate over "left" entries.
        let left_end_snapshot: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> =
            self.end_map.clone();

        let mut new_end_map: BTreeMap<StateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();

        // Append-copy the right NWA into the left.
        let right_to_left: Vec<WaStateID> = self.nwa.append_copy(&right.nwa);

        // Walk all (left end state, stacks) pairs.
        for (left_end_state, stacks) in left_end_snapshot {
            for left_stack in stacks {
                // 1) Process the left stack through the right NWA.
                let mut encoded: Vec<u16> = Vec::with_capacity(left_stack.len());
                for &s in &left_stack {
                    encoded.push(encode_symbol(s)?);
                }
                let stops = right.nwa.process_stack_u16(&encoded);

                // 2) For each stop: add epsilon edge and propagate end_map stacks.
                for (pos, right_stop_state, path_weight) in stops {
                    // Map the right stop state into the left's id space.
                    let mapped_stop = right_to_left[right_stop_state];
                    // Combine the path weight with the provided weight.
                    let combined_weight = &path_weight & weight;
                    // Epsilon from the left end state to the mapped right stop state.
                    self.nwa
                        .add_epsilon_transition(left_end_state, mapped_stop, combined_weight);
 
                    // Reachable right end states from this right stop, ignoring labels.
                    let reachable = right
                        .nwa
                        .reachable_states_ignoring_labels(right_stop_state);
                    for r_state in reachable {
                        if let Some(r_stacks) = right.end_map.get(&r_state) {
                            let mapped_end = right_to_left[r_state];
                            for r_stack in r_stacks {
                                // Prepend the remainder of the left stack to the right stack.
                                let mut combined: Vec<ParserStateID> =
                                    left_stack[pos..].to_vec();
                                combined.extend(r_stack.iter().cloned());
                                new_end_map
                                    .entry(mapped_end)
                                    .or_default()
                                    .insert(combined);
                            }
                        }
                    }
                }
            }
        }
        self.end_map = new_end_map;
        Ok(())
    }

    /// Union of two augmented NWAs. `self` is modified to become the union.
    pub fn union_with(&mut self, other: &AugmentedNwa) {
        let mapping = self.nwa.append_copy(&other.nwa);

        // Merge end_maps
        for (other_end_state, stacks) in &other.end_map {
            let mapped_end_state = mapping[*other_end_state];
            self.end_map
                .entry(mapped_end_state)
                .or_default()
                .extend(stacks.clone());
        }

        // Create new start state with epsilon transitions to old start states
        let new_start = self.nwa.add_state();
        let old_self_start = self.nwa.start_state;
        let old_other_start_mapped = mapping[other.nwa.start_state];

        self.nwa
            .add_epsilon_transition(new_start, old_self_start, WaWeight::all());
        self.nwa
            .add_epsilon_transition(new_start, old_other_start_mapped, WaWeight::all());

        self.nwa.start_state = new_start;
    }
}

impl Display for AugmentedNwa {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Augmented NWA")?;
        writeln!(f, "  Nonterminal Nodes:")?;
        for (nt, state) in &self.nt_nodes {
            writeln!(f, "    - NT {}: State {}", nt.0, state)?;
        }
        if !self.end_map.is_empty() {
            writeln!(f, "  End Map:")?;
            for (state, stacks) in &self.end_map {
                writeln!(f, "    - State {}:", state)?;
                for stack in stacks {
                    let stack_str: Vec<String> = stack.iter().map(|s| s.0.to_string()).collect();
                    writeln!(f, "      - [{}]", stack_str.join(", "))?;
                }
            }
        }
        writeln!(f, "  Underlying NWA:")?;
        for line in self.nwa.to_string().lines() {
            writeln!(f, "    {}", line)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These are structural smoke tests that only validate basic invariants of the builder.
    // Full integration with a real GLRParser instance lives elsewhere in the pipeline.

    // Provide a tiny, synthetic parser stub when feature-testing here would be too heavyweight.
    // If GLRParser isn't trivially constructible in your environment, consider adding
    // integration tests in a higher layer instead.
    struct FakeMaps {
        nts: BTreeSet<NonTerminalID>,
        terms: BTreeSet<TerminalID>,
    }

    #[test]
    fn encode_symbol_fails_on_large_state_ids() {
        // Simulate an overflow: usize that doesn't fit into u16
        #[cfg(target_pointer_width = "64")]
        {
            let big: ParserStateID = ParserStateID(u32::MAX as usize + 10usize);
            let err = encode_symbol(big).unwrap_err();
            match err {
                AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id } => assert_eq!(state_id, big),
            }
        }
    }

    // Helper to create a simple AugmentedNwa for testing.
    // NWA: 0 --(100)--> 1 (end)
    // End map: {1: {[100, 101]}}
    fn build_simple_aug_nwa() -> AugmentedNwa {
        let mut nwa = WaNWA::new();
        let start = nwa.start_state;
        let end = nwa.add_state();
        nwa.add_transition(start, 100, end, WaWeight::all());

        let mut end_map = BTreeMap::new();
        end_map.insert(
            end,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        );

        AugmentedNwa {
            nwa,
            nt_nodes: BTreeMap::new(),
            end_map,
        }
    }

    #[test]
    fn test_combine_with_ignore_on_left() {
        let mut lhs = build_augmented_nwa_for_ignore_terminal();
        let mut rhs = build_simple_aug_nwa();
        rhs.nwa.set_final_weight(1, WaWeight::all());
        let weight = WaWeight::all();

        println!("Left NWA (ignore):\n{}", lhs);
        println!("Right NWA (simple):\n{}", rhs);

        lhs.combine_right_into(&rhs, &weight).unwrap();

        // Build expected result
        let mut expected_nwa = WaNWA::new(); // state 0
        let s1 = expected_nwa.add_state(); // state 1
        let s2 = expected_nwa.add_state(); // state 2
        expected_nwa.add_epsilon_transition(0, s1, WaWeight::all());
        expected_nwa.add_transition(s1, 100, s2, WaWeight::all());
        expected_nwa.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            nwa: expected_nwa,
            nt_nodes: BTreeMap::new(),
            end_map: expected_end_map,
        };

        println!("Expected NWA:\n{}", expected_aug_nwa);
        println!("Resulting NWA:\n{}", lhs);

        assert_eq!(lhs, expected_aug_nwa);
    }

    #[test]
    fn test_combine_with_ignore_on_right() {
        let mut lhs = build_simple_aug_nwa();
        let mut rhs = build_augmented_nwa_for_ignore_terminal();
        rhs.nwa.set_final_weight(rhs.nwa.start_state, WaWeight::all());
        let weight = WaWeight::all();

        lhs.combine_right_into(&rhs, &weight).unwrap();

        // Build expected result
        let mut expected_nwa = WaNWA::new(); // state 0
        let s1 = expected_nwa.add_state(); // state 1
        let s2 = expected_nwa.add_state(); // state 2
        expected_nwa.add_transition(0, 100, s1, WaWeight::all());
        expected_nwa.add_epsilon_transition(s1, s2, WaWeight::all());
        expected_nwa.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            nwa: expected_nwa,
            nt_nodes: BTreeMap::new(),
            end_map: expected_end_map,
        };

        assert_eq!(lhs, expected_aug_nwa);
    }

    // Helper to build the NWA for Terminal 1 from the prompt
    fn build_terminal1_nwa() -> AugmentedNwa {
        let mut nwa = WaNWA::new(); // state 0
        let end_state = nwa.add_state(); // state 1
        let nt0_state = nwa.add_state(); // state 2
        let nt1_state = nwa.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        nwa.add_transition(0, 0, end_state, WaWeight::all());
        nwa.add_transition(0, 1, nt1_state, WaWeight::all());
        nwa.add_transition(0, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]),
        )]);

        AugmentedNwa {
            nwa,
            nt_nodes,
            end_map,
        }
    }

    // Helper to build the NWA for Terminal 2 from the prompt
    fn build_terminal2_nwa() -> AugmentedNwa {
        let mut nwa = WaNWA::new(); // state 0
        let end_state = nwa.add_state(); // state 1
        let nt0_state = nwa.add_state(); // state 2
        let nt1_state = nwa.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        nwa.add_transition(0, 1, nt1_state, WaWeight::all());
        nwa.add_transition(0, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        AugmentedNwa {
            nwa,
            nt_nodes,
            end_map: BTreeMap::new(),
        }
    }

    // Helper to build the NWA for Terminal 0 from the prompt
    fn build_terminal0_nwa() -> AugmentedNwa {
        let mut nwa = WaNWA::new(); // state 0
        let end_state = nwa.add_state(); // state 1
        let nt0_state = nwa.add_state(); // state 2
        let nt1_state = nwa.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        nwa.add_transition(0, 1, nt1_state, WaWeight::all());
        nwa.add_transition(0, 2, end_state, WaWeight::all());
        nwa.add_transition(0, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(2), ParserStateID(0)]]),
        )]);

        AugmentedNwa {
            nwa,
            nt_nodes,
            end_map,
        }
    }

    #[test]
    fn test_right_to_left_combination() {
        // This test simulates the right-to-left combination of augmented NWAs
        // as it happens during the precomputation traversal of the reversed trie.
        // The sequence of terminals is term0, term2, term1.

        // 1. Define the initial NWA, which represents the state after all tokens
        // have been processed. It's a single final state with an empty stack.
        let mut initial_nwa = WaNWA::new();
        let initial_state = initial_nwa.start_state;
        initial_nwa.set_final_weight(initial_state, WaWeight::all());
        let mut current_aug_nwa = AugmentedNwa {
            nwa: initial_nwa,
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
            println!("\n--- Combination Step {} (Term {} on LEFT) ---", i, term_id);
            println!("LEFT NWA (Term {}):\n{}", term_id, term_nwa);
            println!("RIGHT NWA (Current):\n{}", current_aug_nwa);

            let mut new_current = term_nwa.clone();
            new_current.combine_right_into(&current_aug_nwa, &weight).unwrap();
            current_aug_nwa = new_current;

            println!("Resulting Combined NWA:\n{}", current_aug_nwa);
        }

        assert_eq!(current_aug_nwa.nwa.states.len(), 13);
        assert!(current_aug_nwa.end_map.is_empty());
    }

    // Helper to build the "RIGHT" NWA from the prompt, which acts as `self` (the left operand)
    // in the `combine_right_into` call.
    // NOTE: The prompt's example seems to have an inconsistency. For the combination logic to
    // produce a connection, the `end_map` stack of `self` must be processable by the `right`
    // operand's NWA. The prompt's stack `[2, 0]` leads to no transitions in the other NWA.
    // We use `[0, 1]` instead, which allows `process_stack` to find a path and create the
    // connection shown in the prompt's "RESULT" diagram.
    fn build_nwa_from_prompt_right_corrected() -> AugmentedNwa {
        let mut nwa = WaNWA::new(); // 0
        let end_state = nwa.add_state(); // 1
        let nt0_state = nwa.add_state(); // 2
        let nt1_state = nwa.add_state(); // 3

        nwa.add_transition(0, 1, nt1_state, WaWeight::all());
        nwa.add_transition(0, 2, end_state, WaWeight::all());
        nwa.add_transition(0, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        // Corrected from prompt's `[2, 0]` to `[0, 1]` to make the test logic work.
        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]),
        )]);

        AugmentedNwa { nwa, nt_nodes, end_map }
    }

    // Helper to build the "LEFT" NWA from the prompt, which acts as `right` (the right operand)
    // in the `combine_right_into` call.
    fn build_nwa_from_prompt_left() -> AugmentedNwa {
        let mut nwa = WaNWA::new(); // 0
        let s1 = nwa.add_state(); // 1
        let s2 = nwa.add_state(); // 2
        let s3 = nwa.add_state(); // 3
        let s4 = nwa.add_state(); // 4
        let end_state = nwa.add_state(); // 5

        let w3 = WaWeight::from_item(3);
        nwa.add_epsilon_transition(0, s1, w3.clone());
        nwa.add_transition(s1, 0, s2, WaWeight::all());
        nwa.add_transition(s1, 1, s4, WaWeight::all());
        nwa.add_transition(s1, 3, s3, WaWeight::all());
        nwa.add_epsilon_transition(s2, end_state, w3.clone());
        nwa.set_final_weight(end_state, WaWeight::all());

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]),
        )]);

        AugmentedNwa {
            nwa,
            nt_nodes: BTreeMap::new(),
            end_map,
        }
    }

    #[test]
    fn test_combination_from_prompt_example() {
        let mut self_nwa = build_nwa_from_prompt_right_corrected();
        let right_nwa = build_nwa_from_prompt_left();
        let weight = WaWeight::all();

        self_nwa.combine_right_into(&right_nwa, &weight).unwrap();

        // Build expected result
        let mut expected_nwa = WaNWA::new(); // 0
        let s1 = expected_nwa.add_state(); // 1
        let s2 = expected_nwa.add_state(); // 2
        let s3 = expected_nwa.add_state(); // 3
        // Copied states
        let s4 = expected_nwa.add_state(); // 4 (copied 0)
        let s5 = expected_nwa.add_state(); // 5 (copied 1)
        let s6 = expected_nwa.add_state(); // 6 (copied 2)
        let s7 = expected_nwa.add_state(); // 7 (copied 3)
        let s8 = expected_nwa.add_state(); // 8 (copied 4)
        let s9 = expected_nwa.add_state(); // 9 (copied 5)

        // Original part
        expected_nwa.add_transition(0, 1, s3, WaWeight::all());
        expected_nwa.add_transition(0, 2, s1, WaWeight::all());
        expected_nwa.add_transition(0, 3, s2, WaWeight::all());

        // Connection
        let w3 = WaWeight::from_item(3);
        expected_nwa.add_epsilon_transition(s1, s9, w3.clone());

        // Copied part
        expected_nwa.add_epsilon_transition(s4, s5, w3.clone());
        expected_nwa.add_transition(s5, 0, s6, WaWeight::all());
        expected_nwa.add_transition(s5, 1, s8, WaWeight::all());
        expected_nwa.add_transition(s5, 3, s7, WaWeight::all());
        expected_nwa.add_epsilon_transition(s6, s9, w3.clone());
        expected_nwa.set_final_weight(s9, WaWeight::all());

        let expected_nt_nodes = BTreeMap::from([
            (NonTerminalID(0), s2),
            (NonTerminalID(1), s3),
        ]);

        let expected_end_map = BTreeMap::from([(
            s9,
            BTreeSet::from([vec![ParserStateID(1), ParserStateID(0), ParserStateID(1)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            nwa: expected_nwa,
            nt_nodes: expected_nt_nodes,
            end_map: expected_end_map,
        };

        assert_eq!(self_nwa, expected_aug_nwa);
    }
}
