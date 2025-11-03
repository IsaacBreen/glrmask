use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};
use crate::glr::grammar::regex_name;
use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{
    compute_all_characterizations, compute_below_bottom_characterization, BelowBottomCharacterization,
};
use crate::weighted_automata::{
    NWA as WaNWA, NWAStates as WaNWAStates, NWABody as WaNWABody, StateID as WaStateID,
    Weight as WaWeight, StateID, DWA,
};

/// Error that can occur while building an AugmentedNwa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentedNwaBuildError {
    /// A parser StateID could not be represented as a u16 alphabet symbol.
    ParserStateIdOutOfRange { state_id: ParserStateID },
}

/// Per-terminal augmented NWA bundle with separated "states" and "body":
/// - `body.nwa` has the start state (and other metadata) for the underlying NWA.
/// - `states` holds the underlying NWA's states vector (sharable).
/// - `nt_nodes` maps each nonterminal to its dedicated node in the NWA (inside body).
/// - `end_map` accumulates example stacks that reach the end states.
///   Keys are NWA state ids; stacks are Vec<ParserStateID>.
///
/// Alphabet: parser StateID values encoded as u16 (fails if any StateID > u16::MAX).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwaBody {
    pub nwa: WaNWABody,
    pub nt_nodes: BTreeMap<NonTerminalID, WaStateID>,
    /// Map from NWA state id (currently only end_state is used) to sets of parser stacks.
    /// Each parser stack is represented as a Vec<ParserStateID>.
    pub end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentedNwa {
    pub states: WaNWAStates,
    pub body: AugmentedNwaBody,
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
    let mut states = WaNWAStates::default();
    let start_state = states.add_state();

    // For an ignore terminal, the stack is passed through. The end_map should
    // contain an empty stack at the end_state, which is also the start_state.
    // When this NWA is on the left in a `combine_right_into` operation, the
    // empty stack from its end_map results in the right-hand-side NWA's stacks
    // being preserved.
    let mut end_map = BTreeMap::new();
    end_map.insert(start_state, BTreeSet::from([vec![]]));

    AugmentedNwa {
        states,
        body: AugmentedNwaBody {
            nwa: WaNWABody { start_state },
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
///   - one initial NWA start node,
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
    let mut states = WaNWAStates::default();
    let start = states.add_state();
    let end_state = states.add_state();
    let mut body = AugmentedNwaBody {
        nwa: WaNWABody { start_state: start },
        nt_nodes: BTreeMap::new(),
        end_map: BTreeMap::new(),
    };

    // One node per nonterminal in the grammar.
    for &nt in parser.non_terminal_map.right_values() {
        let id = states.add_state();
        body.nt_nodes.insert(nt, id);
    }

    let w_all = WaWeight::all();

    // Initial shifts: start --(initial_state)--> end_state
    for &(initial_state, shift_state) in &bb.initial_shifts {
        let ch = encode_symbol(initial_state)?;
        states.add_transition(start, ch, end_state, w_all.clone());
        body.end_map.entry(end_state).or_default().insert(vec![initial_state, shift_state]);
    }

    // Initial reduces: create a chain of `len` transitions. The first is on `initial_state`,
    // and the rest are default transitions. The chain ends at the `nt` node.
    for &(initial_state, len, nt) in &bb.initial_reduces {
        let ch = encode_symbol(initial_state)?;
        let target_nt = *body
            .nt_nodes
            .get(&nt)
            .expect("nonterminal node must exist (created from parser.non_terminal_map)");
        let mut from = start;

        // The first transition is on the specific character.
        let next_state = if len == 0 { target_nt } else { states.add_state() };
        states.add_transition(from, ch, next_state, w_all.clone());
        from = next_state;

        // The rest are default transitions.
        for i in 0..len {
            let to = if i == len - 1 { target_nt } else { states.add_state() };
            states.add_default_transition(from, to, w_all.clone());
            from = to;
        }
    }

    // Reduce characterizations per nonterminal.
    for (nt, rc) in &bb.reduce_characterizations {
        let src_nt = *body
            .nt_nodes
            .get(nt)
            .expect("reduce_characterizations only contains existing nonterminals");

        // Reveal-and-rereduces: chain of `len` transitions. The first is on `revealed_state`,
        // and the rest are default.
        for &(revealed_state, len, reduce_nt) in &rc.reveal_and_rereduces {
            let ch = encode_symbol(revealed_state)?;
            let dst_nt = *body
                .nt_nodes
                .get(&reduce_nt)
                .expect("reduce target nonterminal must exist");
            let mut from = src_nt;

            // The first transition is on the specific character.
            let next_state = if len == 0 { dst_nt } else { states.add_state() };
            states.add_transition(from, ch, next_state, w_all.clone());
            from = next_state;

            // The rest are default transitions.
            for i in 0..len {
                let to = if i == len - 1 { dst_nt } else { states.add_state() };
                states.add_default_transition(from, to, w_all.clone());
                from = to;
            }
        }

        // Reveal-goto-shift escapes: a single step to end_state; record the example stack.
        for &(revealed_state, goto_state, shift_state) in &rc.reveal_goto_shift_escapes {
            let ch = encode_symbol(revealed_state)?;
            states.add_transition(src_nt, ch, end_state, w_all.clone());
            body
                .end_map
                .entry(end_state)
                .or_default()
                .insert(vec![revealed_state, goto_state, shift_state]);
        }
    }

    Ok(AugmentedNwa { states, body })
}

impl AugmentedNwaBody {
    /// Remap all state-ids in this body using a mapping slice (old_id -> new_id).
    pub fn remap_states(&mut self, mapping: &[WaStateID]) {
        self.nwa.start_state = mapping[self.nwa.start_state];
        // nt_nodes remap
        for v in self.nt_nodes.values_mut() {
            *v = mapping[*v];
        }
        // end_map keys remap
        let mut new_end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();
        for (k, v) in std::mem::take(&mut self.end_map) {
            new_end_map.insert(mapping[k], v);
        }
        self.end_map = new_end_map;
    }

    /// Process a parser-state stack through the underlying NWA referenced by this body and the provided states.
    pub fn process_stack(
        &self,
        states: &WaNWAStates,
        stack: &[ParserStateID],
    ) -> Result<Vec<(WaStateID, WaStateID, WaWeight)>, AugmentedNwaBuildError> {
        let mut encoded: Vec<u16> = Vec::with_capacity(stack.len());
        for &s in stack {
            encoded.push(encode_symbol(s)?);
        }
        Ok(states.process_stack_u16_from_start(self.nwa.start_state, &encoded))
    }

    /// Non-commutative combination on a shared states vec:
    /// - Does NOT append-copy any states (assumes both left and right bodies already refer to the same states vec).
    /// - Adds epsilon transitions in `states` to connect left end states to right stop states.
    /// - Rebuilds left end_map based on right's reachable end states and stacks, prepending the unconsumed remainder
    ///   of the left stacks.
    pub fn combine_right_into_on_shared(
        states: &mut WaNWAStates,
        left: &mut AugmentedNwaBody,
        right: &AugmentedNwaBody,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        // Snapshot the original left end_map so we only iterate over "left" entries.
        let left_end_snapshot: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> =
            left.end_map.clone();

        let mut new_end_map: BTreeMap<StateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();

        // Walk all (left end state, stacks) pairs.
        for (left_end_state, stacks) in left_end_snapshot {
            for left_stack in stacks {
                // 1) Process the left stack through the right NWA starting at right.nwa.start_state.
                let mut encoded: Vec<u16> = Vec::with_capacity(left_stack.len());
                // Consume from top-of-stack first: iterate in reverse order.
                for &s in left_stack.iter().rev() {
                    encoded.push(encode_symbol(s)?);
                }
                let stops = states.process_stack_u16_from_start(right.nwa.start_state, &encoded);

                // 2) For each stop: add epsilon edge and propagate end_map stacks.
                for (pos, right_stop_state, path_weight) in stops {
                    // Combine the path weight with the provided weight.
                    let combined_weight = &path_weight & weight;
                    // Epsilon from the left end state to the right stop state.
                    states
                        .add_epsilon_transition(left_end_state, right_stop_state, combined_weight);
 
                    // Reachable right end states from this right stop, ignoring labels.
                    let reachable = states.reachable_states_ignoring_labels(right_stop_state);
                    for r_state in reachable {
                        if let Some(r_stacks) = right.end_map.get(&r_state) {
                            let mapped_end = r_state; // identity in shared states
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
        left.end_map = new_end_map;
        Ok(())
    }

    /// In-place union on shared states vec:
    /// - Does NOT append-copy any states (assumes both bodies refer to the same states vec).
    /// - Creates a new start state with epsilons to old starts.
    /// - Merges end_map sets.
    pub fn union_with_on_shared(
        states: &mut WaNWAStates,
        left: &mut AugmentedNwaBody,
        right: &AugmentedNwaBody,
    ) {
        // Merge end_maps
        for (other_end_state, stacks) in &right.end_map {
            left.end_map
                .entry(*other_end_state)
                .or_default()
                .extend(stacks.clone());
        }

        // Create new start state with epsilon transitions to old start states
        let new_start = states.add_state();
        let old_self_start = left.nwa.start_state;
        let old_other_start = right.nwa.start_state;

        states
            .add_epsilon_transition(new_start, old_self_start, WaWeight::all());
        states
            .add_epsilon_transition(new_start, old_other_start, WaWeight::all());

        left.nwa.start_state = new_start;
    }
}

impl AugmentedNwa {
    /// Combine 'right' into 'self' (convenience version). This version handles states copying:
    /// - Append-copy the right.states into self.states.
    /// - Remap right.body state-ids using the mapping.
    /// - Apply shared combine logic to connect left and right inside self.states.
    pub fn combine_right_into(
        &mut self,
        right: &AugmentedNwa,
        weight: &WaWeight,
    ) -> Result<(), AugmentedNwaBuildError> {
        // Append-copy the right NWA states into the left (self).
        let right_to_left: Vec<WaStateID> = self.states.append_copy_from(&right.states);
        let mut mapped_right = right.body.clone();
        mapped_right.remap_states(&right_to_left);
        AugmentedNwaBody::combine_right_into_on_shared(&mut self.states, &mut self.body, &mapped_right, weight)
    }

    /// Union of two augmented NWAs. `self` is modified to become the union.
    /// This convenience version copies `other` states, then connects starts.
    pub fn union_with(&mut self, other: &AugmentedNwa) {
        let mapping = self.states.append_copy_from(&other.states);

        // Remap other's body into self
        let mut other_body = other.body.clone();
        other_body.remap_states(&mapping);

        // Merge maps
        for (mapped_end_state, stacks) in &other_body.end_map {
            self.body
                .end_map
                .entry(*mapped_end_state)
                .or_default()
                .extend(stacks.clone());
        }

        // Create new start state with epsilon transitions to old start states
        let new_start = self.states.add_state();
        let old_self_start = self.body.nwa.start_state;
        let old_other_start_mapped = other_body.nwa.start_state;

        self.states
            .add_epsilon_transition(new_start, old_self_start, WaWeight::all());
        self.states
            .add_epsilon_transition(new_start, old_other_start_mapped, WaWeight::all());

        self.body.nwa.start_state = new_start;
    }

    /// Determinize to DWA using combined NWA separation.
    pub fn determinize(&self) -> DWA {
        self.states.determinize(&self.body.nwa)
    }
}

impl Display for AugmentedNwa {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Augmented NWA")?;
        writeln!(f, "  Nonterminal Nodes:")?;
        for (nt, state) in &self.body.nt_nodes {
            writeln!(f, "    - NT {}: State {}", nt.0, state)?;
        }
        if !self.body.end_map.is_empty() {
            writeln!(f, "  End Map:")?;
            for (state, stacks) in &self.body.end_map {
                writeln!(f, "    - State {}:", state)?;
                for stack in stacks {
                    let stack_str: Vec<String> = stack.iter().map(|s| s.0.to_string()).collect();
                    writeln!(f, "      - [{}]", stack_str.join(", "))?;
                }
            }
        }
        writeln!(f, "  Underlying NWA (start: {}):", self.body.nwa.start_state)?;
        for (id, state) in self.states.states.iter().enumerate() {
            writeln!(f, "    State {}:", id)?;
            if let Some(w) = &state.final_weight {
                writeln!(f, "      final_weight: {}", w)?;
            }
            for (to, weight) in &state.epsilon_transitions {
                writeln!(f, "      ε -> {} (weight: {})", to, weight)?;
            }
            if let Some(default) = &state.transitions.default {
                for (to, weight) in default {
                    writeln!(f, "      * -> {} (weight: {})", to, weight)?;
                }
            }
            for (on, transitions) in &state.transitions.exceptions {
                for (to, weight) in transitions {
                    let char_repr = if let Some(c) = char::from_u32(*on as u32) {
                        if c.is_ascii_graphic() || c == ' ' {
                            format!("'{}'", c)
                        } else {
                            format!("{}", *on)
                        }
                    } else {
                        format!("{}", *on)
                    };
                    writeln!(f, "      {} -> {} (weight: {})", char_repr, to, weight)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_symbol_fails_on_large_state_ids() {
        // Simulate an overflow: usize that doesn't fit into u16
        #[cfg(target_pointer_width = "64")]
        {
            let big: ParserStateID = ParserStateID(u32::MAX as usize + 10usize);
            let err = super::encode_symbol(big).unwrap_err();
            match err {
                AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id } => assert_eq!(state_id, big),
            }
        }
    }

    // Helper to create a simple AugmentedNwa for testing.
    // NWA: 0 --(100)--> 1 (end)
    // End map: {1: {[100, 101]}}
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
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: start },
                nt_nodes: BTreeMap::new(),
                end_map,
            },
        }
    }

    #[test]
    fn test_combine_with_ignore_on_left() {
        let mut lhs = build_augmented_nwa_for_ignore_terminal();
        let mut rhs = build_simple_aug_nwa();
        // mark rhs's end as final
        let end_state = rhs.body.end_map.keys().cloned().next().unwrap();
        rhs.states.set_final_weight(end_state, WaWeight::all());
        let weight = WaWeight::all();

        crate::debug!(5, "Left NWA (ignore):\n{}", lhs);
        crate::debug!(5, "Right NWA (simple):\n{}", rhs);

        lhs.combine_right_into(&rhs, &weight).unwrap();

        // Build expected result
        let mut states = WaNWAStates::default();
        let s0 = states.add_state(); // 0
        let s1 = states.add_state(); // 1
        let s2 = states.add_state(); // 2
        states.add_epsilon_transition(s0, s1, WaWeight::all());
        states.add_transition(s1, 100, s2, WaWeight::all());
        states.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            s2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: s0 },
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
        // mark rhs start as final
        rhs.states.set_final_weight(rhs.body.nwa.start_state, WaWeight::all());
        let weight = WaWeight::all();

        lhs.combine_right_into(&rhs, &weight).unwrap();

        // Build expected result
        let mut states = WaNWAStates::default();
        let s0 = states.add_state(); // 0
        let s1 = states.add_state(); // 1
        let s2 = states.add_state(); // 2
        states.add_transition(s0, 100, s1, WaWeight::all());
        states.add_epsilon_transition(s1, s2, WaWeight::all());
        states.set_final_weight(s2, WaWeight::all());

        let expected_end_map = BTreeMap::from([(
            s2,
            BTreeSet::from([vec![ParserStateID(100), ParserStateID(101)]]),
        )]);

        let expected_aug_nwa = AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: s0 },
                nt_nodes: BTreeMap::new(),
                end_map: expected_end_map,
            },
        };

        assert_eq!(lhs, expected_aug_nwa);
    }

    // Helper to build the NWA for Terminal 1 from the prompt
    fn build_terminal1_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default(); // state 0
        let end_state = states.add_state(); // state 1
        let nt0_state = states.add_state(); // state 2
        let nt1_state = states.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(0, 0, end_state, WaWeight::all());
        states.add_transition(0, 1, nt1_state, WaWeight::all());
        states.add_transition(0, 3, nt0_state, WaWeight::all());

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
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: 0 },
                nt_nodes,
                end_map,
            },
        }
    }

    // Helper to build the NWA for Terminal 2 from the prompt
    fn build_terminal2_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default(); // state 0
        let end_state = states.add_state(); // state 1
        let nt0_state = states.add_state(); // state 2
        let nt1_state = states.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(0, 1, nt1_state, WaWeight::all());
        states.add_transition(0, 3, nt0_state, WaWeight::all());

        let nt_nodes = BTreeMap::from([
            (NonTerminalID(0), nt0_state),
            (NonTerminalID(1), nt1_state),
        ]);

        AugmentedNwa {
            states,
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: 0 },
                nt_nodes,
                end_map: BTreeMap::new(),
            },
        }
    }

    // Helper to build the NWA for Terminal 0 from the prompt
    fn build_terminal0_nwa() -> AugmentedNwa {
        let mut states = WaNWAStates::default(); // state 0
        let end_state = states.add_state(); // state 1
        let nt0_state = states.add_state(); // state 2
        let nt1_state = states.add_state(); // state 3
        assert_eq!(end_state, 1);
        assert_eq!(nt0_state, 2);
        assert_eq!(nt1_state, 3);

        states.add_transition(0, 1, nt1_state, WaWeight::all());
        states.add_transition(0, 2, end_state, WaWeight::all());
        states.add_transition(0, 3, nt0_state, WaWeight::all());

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
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: 0 },
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
        let mut states = WaNWAStates::default();
        let initial_state = states.add_state();
        states.set_final_weight(initial_state, WaWeight::all());
        let mut current_aug_nwa = AugmentedNwa {
            states: states.clone(),
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: initial_state },
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

        assert_eq!(current_aug_nwa.states.states.len(), 13);
        assert!(current_aug_nwa.body.end_map.is_empty());
    }

    // Helper to build the "RIGHT" NWA from the prompt, which acts as `self` (the left operand)
    // in the `combine_right_into` call.
    // NOTE: The prompt's example seems to have an inconsistency. For the combination logic to
    // produce a connection, the `end_map` stack of `self` must be processable by the `right`
    // operand's NWA. The prompt's stack `[2, 0]` requires a modification to the LEFT NWA
    // to create an epsilon path to a final state.
    fn build_nwa_from_prompt_left() -> AugmentedNwa {
        let mut states = WaNWAStates::default(); // 0
        let end_state = states.add_state(); // 1
        let nt0_state = states.add_state(); // 2
        let nt1_state = states.add_state(); // 3

        states.add_transition(0, 1, nt1_state, WaWeight::all());
        states.add_transition(0, 2, end_state, WaWeight::all());
        states.add_transition(0, 3, nt0_state, WaWeight::all());

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
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: 0 },
                nt_nodes,
                end_map,
            },
        }
    }

    // Helper to build the "LEFT" NWA from the prompt, which acts as `right` (the right operand)
    // in the `combine_right_into` call.
    fn build_nwa_from_prompt_right() -> AugmentedNwa {
        let mut states = WaNWAStates::default(); // 0
        let s1 = states.add_state(); // 1
        let s2 = states.add_state(); // 2
        let s3 = states.add_state(); // 3
        let s4 = states.add_state(); // 4
        let end_state = states.add_state(); // 5

        let w3 = WaWeight::from_item(3);
        states.add_epsilon_transition(0, s1, w3.clone());
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
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: 0 },
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
        let mut states = WaNWAStates::default(); // 0
        let s1 = states.add_state(); // 1
        let s2 = states.add_state(); // 2
        let s3 = states.add_state(); // 3
        // Copied states
        let s4 = states.add_state(); // 4 (copied 0)
        let s5 = states.add_state(); // 5 (copied 1)
        let s6 = states.add_state(); // 6 (copied 2)
        let s7 = states.add_state(); // 7 (copied 3)
        let s8 = states.add_state(); // 8 (copied 4)
        let s9 = states.add_state(); // 9 (copied 5)

        // Original part
        states.add_transition(0, 1, s3, WaWeight::all());
        states.add_transition(0, 2, s1, WaWeight::all());
        states.add_transition(0, 3, s2, WaWeight::all());

        // Connection
        let w3 = WaWeight::from_item(3);
        states.add_epsilon_transition(s1, s9, w3.clone());

        // Copied part
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
            body: AugmentedNwaBody {
                nwa: WaNWABody { start_state: 0 },
                nt_nodes: expected_nt_nodes,
                end_map: expected_end_map,
            },
        };

        crate::debug!(5, "Expected NWA:\n{}", expected_aug_nwa);

        assert_eq!(self_nwa, expected_aug_nwa);
    }
}
