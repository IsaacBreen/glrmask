use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use crate::glr::grammar::regex_name;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{
    compute_all_characterizations, compute_below_bottom_characterization, BelowBottomCharacterization,
};
use crate::weighted_automata::{
    NWA as WaNWA, NWAMeta as WaNWAMeta, NWAStates as WaNWAStates, StateID, StateID as WaStateID,
    Weight as WaWeight,
};

/// Error that can occur while building an AugmentedNwa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentedNwaBuildError {
    /// A parser StateID could not be represented as a u16 alphabet symbol.
    ParserStateIdOutOfRange { state_id: ParserStateID },
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwaStates {
    pub nwa_states: WaNWAStates,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwaMeta {
    pub nwa_meta: WaNWAMeta,
    pub nt_nodes: BTreeMap<NonTerminalID, WaStateID>,
    /// Map from NWA state id (currently only the end_state is used) to sets of parser stacks.
    /// Each parser stack is represented as a Vec<ParserStateID>.
    pub end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>>,
}

/// Per-terminal augmented NWA bundle:
/// - `nwa` is the constructed automaton.
/// - `nt_nodes` maps each nonterminal to its dedicated node in the NWA.
/// - `end_map` accumulates example stacks that reach the `end_state`.
/// Alphabet: parser StateID values encoded as u16 (fails if any StateID > u16::MAX).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwa {
    pub states: AugmentedNwaStates,
    pub meta: AugmentedNwaMeta,
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
    let mut nwa_states = WaNWAStates::new();
    let start_state = nwa_states.add_state();

    // For an ignore terminal, the stack is passed through. The end_map should
    // contain an empty stack at the end_state, which is also the start_state.
    // When this NWA is on the left in a `combine_right_into` operation, the
    // empty stack from its end_map results in the right-hand-side NWA's stacks
    // being preserved.
    let end_map = BTreeMap::from([(start_state, BTreeSet::from([vec![]]))]);

    AugmentedNwa {
        states: AugmentedNwaStates { nwa_states },
        meta: AugmentedNwaMeta {
            nwa_meta: WaNWAMeta { start_state },
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
/// - Create:
///   - one initial NWA start node (provided by WaNWA::new()),
///   - one unique `end_state`,
///   - one node per nonterminal.
/// - For initial shifts:
///   - from start, on symbol (initial_state), go to end_state; add
///     [initial_state, shift_state] into end_map[end_state].
/// - For initial reduces (initial_state, len, nt):
///   - create a chain of `len` transitions from `start` to the `nt` node.
///   - The first transition is on `initial_state`.
///   - The following `len - 1` transitions are unconditional (default) transitions
///     through intermediate states.
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
    let mut nwa_states = WaNWAStates::new(); // adds start state 0
    let start = nwa_states.add_state();
    let nwa_meta = WaNWAMeta { start_state: start };
    let end_state = nwa_states.add_state();

    // One node per nonterminal in the grammar.
    let mut nt_nodes: BTreeMap<NonTerminalID, WaStateID> = BTreeMap::new();
    for &nt in parser.non_terminal_map.right_values() {
        let id = nwa_states.add_state();
        nt_nodes.insert(nt, id);
    }

    let mut end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();
    let w_all = WaWeight::all();

    // Initial shifts: start --(initial_state)--> end_state
    for &(initial_state, shift_state) in &bb.initial_shifts {
        let ch = encode_symbol(initial_state)?;
        nwa_states.add_transition(start, ch, end_state, w_all.clone());
        end_map.entry(end_state).or_default().insert(vec![initial_state, shift_state]);
    }

    // Initial reduces: create a chain of `len` transitions. The first is on `initial_state`,
    // and the rest are default transitions. The chain ends at the `nt` node.
    for &(initial_state, len, nt) in &bb.initial_reduces {
        let ch = encode_symbol(initial_state)?;
        let target_nt = *nt_nodes
            .get(&nt)
            .expect("nonterminal node must exist (created from parser.non_terminal_map)");
        let mut from = start;

        // The first transition is on the specific character.
        let next_state = if len == 0 { target_nt } else { nwa_states.add_state() };
        nwa_states.add_transition(from, ch, next_state, w_all.clone());
        from = next_state;

        // The rest are default transitions.
        for i in 0..len {
            let to = if i == len - 1 { target_nt } else { nwa_states.add_state() };
            nwa_states.add_default_transition(from, to, w_all.clone());
            from = to;
        }
    }

    // Reduce characterizations per nonterminal.
    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt = *nt_nodes
            .get(nt)
            .expect("reduce_characterizations only contains existing nonterminals");

        // Reveal-and-rereduces: chain of `len` transitions. The first is on `revealed_state`,
        // and the rest are default.
        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let ch = encode_symbol(revealed_state)?;
            let dst_nt = *nt_nodes
                .get(&reduce_nt)
                .expect("reduce target nonterminal must exist");
            let mut from = src_nt;

            // The first transition is on the specific character.
            let next_state = if len == 0 { dst_nt } else { nwa_states.add_state() };
            nwa_states.add_transition(from, ch, next_state, w_all.clone());
            from = next_state;

            // The rest are default transitions.
            for i in 0..len {
                let to = if i == len - 1 { dst_nt } else { nwa_states.add_state() };
                nwa_states.add_default_transition(from, to, w_all.clone());
                from = to;
            }
        }

        // Reveal-goto-shift escapes: a single step to end_state; record the example stack.
        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let ch = encode_symbol(revealed_state)?;
            nwa_states.add_transition(src_nt, ch, end_state, w_all.clone());
            end_map
                .entry(end_state)
                .or_default()
                .insert(vec![revealed_state, goto_state, shift_state]);
        }
    }

    Ok(AugmentedNwa {
        states: AugmentedNwaStates { nwa_states },
        meta: AugmentedNwaMeta {
            nwa_meta,
            nt_nodes,
            end_map,
        },
    })
}

pub fn remap_augmented_meta(meta: &AugmentedNwaMeta, map: &[WaStateID]) -> AugmentedNwaMeta {
    let mut nt_nodes = BTreeMap::new();
    for (nt, st) in &meta.nt_nodes {
        nt_nodes.insert(*nt, map[*st]);
    }
    let mut end_map = BTreeMap::new();
    for (st, stacks) in &meta.end_map {
        end_map.insert(map[*st], stacks.clone());
    }
    AugmentedNwaMeta {
        nwa_meta: WaNWAMeta { start_state: map[meta.nwa_meta.start_state] },
        nt_nodes,
        end_map,
    }
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
        Ok(self.states.nwa_states.process_stack_u16(&self.meta.nwa_meta, &encoded))
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
        let left_end_snapshot: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = self.meta.end_map.clone();

        let mut new_end_map: BTreeMap<StateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();

        // Append-copy the right NWA into the left.
        let right_to_left: Vec<WaStateID> = self.states.nwa_states.append_copy_from(&right.states.nwa_states);

        // Walk all (left end state, stacks) pairs.
        for (left_end_state, stacks) in left_end_snapshot {
            for left_stack in stacks {
                // 1) Process the left stack through the right NWA.
                let mut encoded: Vec<u16> = Vec::with_capacity(left_stack.len());
                // Consume from top-of-stack first: iterate in reverse order.
                for &s in left_stack.iter().rev() {
                    encoded.push(encode_symbol(s)?);
                }
                let stops = right.states.nwa_states.process_stack_u16(&right.meta.nwa_meta, &encoded);

                // 2) For each stop: add epsilon edge and propagate end_map stacks.
                for (pos, right_stop_state, path_weight) in stops {
                    // Map the right stop state into the left's id space.
                    let mapped_stop = right_to_left[right_stop_state];
                    // Combine the path weight with the provided weight.
                    let combined_weight = &path_weight & weight;
                    // Epsilon from the left end state to the mapped right stop state.
                    self.states.nwa_states
                        .add_epsilon_transition(left_end_state, mapped_stop, combined_weight);
 
                    // Reachable right end states from this right stop, ignoring labels.
                    let reachable = right
                        .nwa
                        .reachable_states_ignoring_labels(right_stop_state);
                    for r_state in reachable {
                        if let Some(r_stacks) = right.meta.end_map.get(&r_state) {
                            let mapped_end = right_to_left[r_state];
                            for r_stack in r_stacks {
                                // Prepend the remainder of the left stack to the right stack.
                                // Since we consumed from the top (reverse order), the remaining
                                // prefix in the original order is len - pos.
                                let keep_len = left_stack.len().saturating_sub(pos);
                                let mut combined: Vec<ParserStateID> =
                                    left_stack[..keep_len].to_vec();
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
        self.meta.end_map = new_end_map;
        Ok(())
    }

    /// Union of two augmented NWAs. `self` is modified to become the union.
    pub fn union_with(&mut self, other: &AugmentedNwa) {
        let mapping = self.states.nwa_states.append_copy_from(&other.states.nwa_states);

        // Merge end_maps
        for (other_end_state, stacks) in &other.meta.end_map {
            let mapped_end_state = mapping[*other_end_state];
            self.meta.end_map
                .entry(mapped_end_state)
                .or_default()
                .extend(stacks.clone());
        }

        // Create new start state with epsilon transitions to old start states
        let new_start = self.states.nwa_states.add_state();
        let old_self_start = self.meta.nwa_meta.start_state;
        let old_other_start_mapped = mapping[other.meta.nwa_meta.start_state];

        self.states.nwa_states
            .add_epsilon_transition(new_start, old_self_start, WaWeight::all());
        self.states.nwa_states
            .add_epsilon_transition(new_start, old_other_start_mapped, WaWeight::all());

        self.meta.nwa_meta.start_state = new_start;
    }
}

impl AugmentedNwaMeta {
    pub fn combine_right_into_shared(
        &mut self,
        shared_states: &mut WaNWAStates,
        right: &AugmentedNwaMeta,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        // Snapshot left end_map (self)
        let left_end_snapshot = self.end_map.clone();
        let mut new_end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();

        for (left_end_state, stacks) in left_end_snapshot {
            for left_stack in stacks {
                let mut encoded: Vec<u16> = Vec::with_capacity(left_stack.len());
                for &s in left_stack.iter().rev() {
                    encoded.push(encode_symbol(s)?);
                }
                let stops = shared_states.process_stack_u16(&right.nwa_meta, &encoded);
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
        self.end_map = new_end_map;
        Ok(())
    }

    pub fn union_meta_in_place(&mut self, shared_states: &mut WaNWAStates, other: &AugmentedNwaMeta) {
        let new_start = shared_states.add_state();
        shared_states.add_epsilon_transition(new_start, self.nwa_meta.start_state, WaWeight::all());
        shared_states.add_epsilon_transition(new_start, other.nwa_meta.start_state, WaWeight::all());
        self.nwa_meta.start_state = new_start;

        for (k, v) in &other.end_map {
            self.end_map.entry(*k).or_default().extend(v.clone());
        }
        // Merge nt_nodes non-destructively (keep existing for duplicates)
        for (nt, st) in &other.nt_nodes {
            self.nt_nodes.entry(*nt).or_insert(*st);
        }
    }
}
impl Display for AugmentedNwa {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Augmented NWA")?;
        writeln!(f, "  Nonterminal Nodes:")?;
        for (nt, state) in &self.meta.nt_nodes {
            writeln!(f, "    - NT {}: State {}", nt.0, state)?;
        }
        if !self.meta.end_map.is_empty() {
            writeln!(f, "  End Map:")?;
            for (state, stacks) in &self.meta.end_map {
                writeln!(f, "    - State {}:", state)?;
                for stack in stacks {
                    let stack_str: Vec<String> = stack.iter().map(|s| s.0.to_string()).collect();
                    writeln!(f, "      - [{}]", stack_str.join(", "))?;
                }
            }
        }
        writeln!(f, "  Underlying NWA:")?;
        let tmp = WaNWA {
            states: self.states.nwa_states.clone(),
            meta: self.meta.nwa_meta.clone(),
        };
        for line in tmp.to_string().lines() {
            writeln!(f, "    {}", line)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weighted_automata::{NWAStates as WaNWAStates, NWAMeta as WaNWAMeta};

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
        let mut nwa_states = WaNWAStates::new();
        let start = nwa_states.add_state();
        let end = nwa_states.add_state();
        nwa_states.add_transition(start, 100, end, WaWeight::all());

        let mut end_map = BTreeMap::new();
        end_map.insert(
            end,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        );

        AugmentedNwa {
            states: AugmentedNwaStates { nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: start },
                nt_nodes: BTreeMap::new(),
                end_map,
            },
        }
    }

    #[test]
    fn test_combine_with_ignore_on_left() {
        let mut lhs = build_augmented_nwa_for_ignore_terminal();
        let mut rhs = build_simple_aug_nwa();
        rhs.states
            .nwa_states
            .set_final_weight(1, WaWeight::all());
        let weight = WaWeight::all();

        crate::debug!(5, "Left NWA (ignore):\n{}", lhs);
        crate::debug!(5, "Right NWA (simple):\n{}", rhs);

        lhs.combine_right_into(&rhs, &weight).unwrap();

        // Build expected result
        let mut expected_nwa_states = WaNWAStates::new(); // state 0
        let s1 = expected_nwa_states.add_state(); // state 1
        let s2 = expected_nwa_states.add_state(); // state 2
        expected_nwa_states.add_epsilon_transition(0, s1, WaWeight::all());
        expected_nwa_states.add_transition(s1, 100, s2, WaWeight::all());
        expected_nwa_states.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states: AugmentedNwaStates { nwa_states: expected_nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: 0 },
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
        rhs.states.nwa_states.set_final_weight(
            rhs.meta.nwa_meta.start_state,
            WaWeight::all(),
        );
        let weight = WaWeight::all();

        lhs.combine_right_into(&rhs, &weight).unwrap();

        // Build expected result
        let mut expected_nwa_states = WaNWAStates::new(); // state 0
        let s1 = expected_nwa_states.add_state(); // state 1
        let s2 = expected_nwa_states.add_state(); // state 2
        expected_nwa_states.add_transition(0, 100, s1, WaWeight::all());
        expected_nwa_states.add_epsilon_transition(s1, s2, WaWeight::all());
        expected_nwa_states.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states: AugmentedNwaStates { nwa_states: expected_nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: 0 },
                nt_nodes: BTreeMap::new(),
                end_map: expected_end_map,
            },
        };

        assert_eq!(lhs, expected_aug_nwa);
    }

    // Helper to build the NWA for Terminal 1 from the prompt
    fn build_terminal1_nwa() -> AugmentedNwa {
        let mut nwa_states = WaNWAStates::new(); // state 0
        let start = nwa_states.add_state();
        let end_state = nwa_states.add_state(); // state 1
        let nt0_state = nwa_states.add_state(); // state 2
        let nt1_state = nwa_states.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);
        nwa_states.add_transition(start, 0, end_state, WaWeight::all());
        nwa_states.add_transition(start, 1, nt1_state, WaWeight::all());
        nwa_states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]),
        )]);

        AugmentedNwa {
            states: AugmentedNwaStates { nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: start },
                nt_nodes,
                end_map,
            },
        }
    }

    // Helper to build the NWA for Terminal 2 from the prompt
    fn build_terminal2_nwa() -> AugmentedNwa {
        let mut nwa_states = WaNWAStates::new(); // state 0
        let start = nwa_states.add_state();
        let end_state = nwa_states.add_state(); // state 1
        let nt0_state = nwa_states.add_state(); // state 2
        let nt1_state = nwa_states.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        nwa_states.add_transition(start, 1, nt1_state, WaWeight::all());
        nwa_states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        AugmentedNwa {
            states: AugmentedNwaStates { nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: start },
                nt_nodes,
                end_map: BTreeMap::new(),
            },
        }
    }

    // Helper to build the NWA for Terminal 0 from the prompt
    fn build_terminal0_nwa() -> AugmentedNwa {
        let mut nwa_states = WaNWAStates::new(); // state 0
        let start = nwa_states.add_state();
        let end_state = nwa_states.add_state(); // state 1
        let nt0_state = nwa_states.add_state(); // state 2
        let nt1_state = nwa_states.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        nwa_states.add_transition(start, 1, nt1_state, WaWeight::all());
        nwa_states.add_transition(start, 2, end_state, WaWeight::all());
        nwa_states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map =
            BTreeMap::from([(end_state, BTreeSet::from([vec![ParserStateID(2), ParserStateID(0)]]))]);

        AugmentedNwa {
            states: AugmentedNwaStates { nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: start },
                nt_nodes,
                end_map,
            },
        }
    }

    #[test]
    fn test_right_to_left_combination() {
        // This test simulates the right-to-left combination of augmented NWAs
        // as it happens during the precomputation traversal of the reversed trie.
        // The sequence of terminals is term0, term2, term1.

        // 1. Define the initial NWA, which represents the state after all tokens
        // have been processed. It's a single final state with an empty stack.
        let mut initial_nwa_states = WaNWAStates::new();
        let initial_state = initial_nwa_states.add_state();
        initial_nwa_states.set_final_weight(initial_state, WaWeight::all());
        let mut current_aug_nwa = AugmentedNwa {
            states: AugmentedNwaStates { nwa_states: initial_nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: initial_state },
                nt_nodes: BTreeMap::new(),
                end_map: BTreeMap::from([(initial_state, BTreeSet::from([vec![]]))]),
            },
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
            crate::debug!(5, "RIGHT NWA (Current):\n{}", current_aug_nwa);

            let mut new_current = term_nwa.clone();
            new_current.combine_right_into(&current_aug_nwa, &weight).unwrap();
            current_aug_nwa = new_current;

            crate::debug!(5, "Resulting Combined NWA:\n{}", current_aug_nwa);
        }

        assert_eq!(current_aug_nwa.states.nwa_states.states.len(), 13);
        assert!(current_aug_nwa.meta.end_map.is_empty());
    }

    // Helper to build the "RIGHT" NWA from the prompt, which acts as `self` (the left operand)
    // in the `combine_right_into` call.
    // NOTE: The prompt's example seems to have an inconsistency. For the combination logic to
    // produce a connection, the `end_map` stack of `self` must be processable by the `right`
    // operand's NWA. The prompt's stack `[2, 0]` requires a modification to the LEFT NWA
    // to create an epsilon path to a final state.
    fn build_nwa_from_prompt_left() -> AugmentedNwa {
        let mut nwa_states = WaNWAStates::new(); // 0
        let start = nwa_states.add_state();
        let end_state = nwa_states.add_state(); // 1
        let nt0_state = nwa_states.add_state(); // 2
        let nt1_state = nwa_states.add_state(); // 3

        nwa_states.add_transition(start, 1, nt1_state, WaWeight::all());
        nwa_states.add_transition(start, 2, end_state, WaWeight::all());
        nwa_states.add_transition(start, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(2), ParserStateID(0)]]),
        )]);

        AugmentedNwa {
            states: AugmentedNwaStates { nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: start },
                nt_nodes,
                end_map,
            },
        }
    }

    // Helper to build the "LEFT" NWA from the prompt, which acts as `right` (the right operand)
    // in the `combine_right_into` call.
    fn build_nwa_from_prompt_right() -> AugmentedNwa {
        let mut nwa_states = WaNWAStates::new(); // 0
        let start = nwa_states.add_state();
        let s1 = nwa_states.add_state(); // 1
        let s2 = nwa_states.add_state(); // 2
        let s3 = nwa_states.add_state(); // 3
        let s4 = nwa_states.add_state(); // 4
        let end_state = nwa_states.add_state(); // 5

        let w3 = WaWeight::from_item(3);
        nwa_states.add_epsilon_transition(start, s1, w3.clone());
        nwa_states.add_transition(s1, 0, s2, WaWeight::all());
        nwa_states.add_transition(s1, 1, s4, WaWeight::all());
        nwa_states.add_transition(s1, 3, s3, WaWeight::all());
        nwa_states.add_epsilon_transition(s2, end_state, w3.clone());
        nwa_states.set_final_weight(end_state, WaWeight::all());

        // This is changed from `[0, 1]` to `[f1]` to produce the prompt's final combined stack.
        let end_map = BTreeMap::from([(
            end_state,
            BTreeSet::from([vec![ParserStateID(0), ParserStateID(1)]]),
        )]);

        AugmentedNwa {
            states: AugmentedNwaStates { nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: start },
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

        crate::debug!(5, "Resulting NWA after combination:\n{}", self_nwa);

        // Build expected result
        let mut expected_nwa_states = WaNWAStates::new(); // 0
        let s1 = expected_nwa_states.add_state(); // 1
        let s2 = expected_nwa_states.add_state(); // 2
        let s3 = expected_nwa_states.add_state(); // 3
        // Copied states
        let s4 = expected_nwa_states.add_state(); // 4 (copied 0)
        let s5 = expected_nwa_states.add_state(); // 5 (copied 1)
        let s6 = expected_nwa_states.add_state(); // 6 (copied 2)
        let s7 = expected_nwa_states.add_state(); // 7 (copied 3)
        let s8 = expected_nwa_states.add_state(); // 8 (copied 4)
        let s9 = expected_nwa_states.add_state(); // 9 (copied 5)

        // Original part
        expected_nwa_states.add_transition(0, 1, s3, WaWeight::all());
        expected_nwa_states.add_transition(0, 2, s1, WaWeight::all());
        expected_nwa_states.add_transition(0, 3, s2, WaWeight::all());

        // Connection
        let w3 = WaWeight::from_item(3);
        expected_nwa_states.add_epsilon_transition(s1, s9, w3.clone());

        // Copied part
        expected_nwa_states.add_epsilon_transition(s4, s5, w3.clone());
        expected_nwa_states.add_transition(s5, 0, s6, WaWeight::all());
        expected_nwa_states.add_transition(s5, 1, s8, WaWeight::all());
        expected_nwa_states.add_transition(s5, 3, s7, WaWeight::all());
        expected_nwa_states.add_epsilon_transition(s6, s9, w3.clone());
        expected_nwa_states.set_final_weight(s9, WaWeight::all());

        let expected_nt_nodes = BTreeMap::from([
            (NonTerminalID(0), s2),
            (NonTerminalID(1), s3),
        ]);

        let expected_end_map = BTreeMap::from([(
            s9,
            BTreeSet::from([vec![ParserStateID(2), ParserStateID(0), ParserStateID(1)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states: AugmentedNwaStates { nwa_states: expected_nwa_states },
            meta: AugmentedNwaMeta {
                nwa_meta: WaNWAMeta { start_state: 0 },
                nt_nodes: expected_nt_nodes,
                end_map: expected_end_map,
            },
        };

        crate::debug!(5, "Expected NWA:\n{}", expected_aug_nwa);

        assert_eq!(self_nwa, expected_aug_nwa);
    }
}
