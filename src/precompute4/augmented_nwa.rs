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
    u16::try_from(id).map_err(|_| AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id: id })
}

// Helper: encode an entire stack to u16 symbols.
fn encode_stack_symbols(stack: &[ParserStateID]) -> Result<Vec<u16>, AugmentedNwaBuildError> {
    let mut out = Vec::with_capacity(stack.len());
    for &s in stack {
        out.push(encode_symbol(s)?);
    }
    Ok(out)
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

/// Non-commutatively compose two AugmentedNwa automata: left then right.
///
/// Steps:
/// - Clone the entire `right.nwa` into `left.nwa` once, capturing the base offset.
/// - For each (left_end_state, stack) in `left.end_map`, feed `stack` through the
///   `right.nwa` from its start, yielding a set of (pos, state, weight).
/// - For each such result, add an epsilon transition from `left_end_state` to the
///   cloned `right` state with the corresponding `weight`.
/// - Update `left.end_state` to the cloned `right.end_state` and replace `left.end_map`
///   with a remapped clone of `right.end_map` (keys offset by the clone base), so further
///   compositions continue from the new end.
pub fn combine_augmented_nwas_noncommutative(
    left: &mut AugmentedNwa,
    right: &AugmentedNwa,
    ) -> Result<(), AugmentedNwaBuildError> {
    // 1) Clone the entire right NWA into the left NWA and obtain the base offset.
    let base = left.nwa.append_clone_of(&right.nwa);

    // 2) For every escape stack recorded on the left, thread it through the right NWA.
    //    We work over a snapshot of the map to avoid borrow conflicts as we mutate left.nwa.
    let left_end_entries: Vec<(WaStateID, BTreeSet<Vec<ParserStateID>>)> =
        left.end_map.iter().map(|(k, v)| (*k, v.clone())).collect();
    for (left_end_state, stacks) in left_end_entries {
        for stack in stacks {
            let encoded = encode_stack_symbols(&stack)?;
            let results = right.nwa.process_symbol_stack(&encoded);
            for (_pos, right_state, w) in results {
                left.nwa.add_epsilon_transition(left_end_state, base + right_state, w.clone());
            }
        }
    }

    // 3) Update left's end_state and end_map to those of the appended right portion.
    left.end_state = base + right.end_state;
    let mut new_end_map: BTreeMap<WaStateID, BTreeSet<Vec<ParserStateID>>> = BTreeMap::new();
    for (st, stacks) in &right.end_map {
        new_end_map.insert(base + *st, stacks.clone());
    }
    left.end_map = new_end_map;
    Ok(())
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
            let big: ParserStateID = u32::MAX as usize + 10usize;
            let err = encode_symbol(big).unwrap_err();
            match err {
                AugmentedNwaBuildError::ParserStateIdOutOfRange { state_id } => assert_eq!(state_id, big),
            }
        }
    }
}
