use std::collections::{BTreeMap, BTreeSet};

use crate::glr::parser::GLRParser;
use crate::glr::table::{NonTerminalID, StateID as ParserStateID, TerminalID};
use crate::precompute4::characterize::{
    compute_all_characterizations, compute_below_bottom_characterization, BelowBottomCharacterization,
};
use crate::weighted_automata::{NWA as WaNWA, StateID as WaStateID, Weight as WaWeight};

/// Error that can occur while building an AugmentedNwa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentedNwaBuildError {
    /// A parser StateID could not be represented as a u16 alphabet symbol.
    ParserStateIdOutOfRange { state_id: ParserStateID },
}

/// Per-terminal augmented NWA bundle:
/// - `nwa` is the constructed automaton.
/// - `end_state` is the unique end node where we accumulate “escape” stacks.
/// - `nt_nodes` maps each nonterminal to its dedicated node in the NWA.
/// - `end_map` accumulates example stacks that reach the `end_state`.
///
/// Alphabet: parser StateID values encoded as u16 (fails if any StateID > u16::MAX).
#[derive(Debug, Clone)]
pub struct AugmentedNwa {
    pub terminal: TerminalID,
    pub nwa: WaNWA,
    pub end_state: WaStateID,
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

/// Build augmented NWAs for all terminals.
pub fn build_augmented_nwas(
    parser: &GLRParser,
) -> Result<BTreeMap<TerminalID, AugmentedNwa>, AugmentedNwaBuildError> {
    let all = compute_all_characterizations(parser);
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
        terminal: bb.terminal,
        nwa,
        end_state,
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
                                self.end_map
                                    .entry(mapped_end)
                                    .or_default()
                                    .insert(combined);
                            }
                        }
                    }
                }
            }
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
}
