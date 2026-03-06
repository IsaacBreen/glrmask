//! Hot-path mask computation.
//!
//! Computes the set of allowed LLM tokens for the current constraint state.
//!
//! # Algorithm
//!
//! For each (tokenizer_state, parser_stacks) pair in the current state:
//! 1. Look up the TSID for the tokenizer state
//! 2. For each parser stack, walk the DWA bottom-to-top reading parser
//!    state IDs as labels
//! 3. Project the DWA transition weights to the TSID column to get
//!    token-space RangeSets
//! 4. Union all results into the final mask

use std::collections::BTreeMap;

use crate::automata::weighted::dwa::CompDwa;
use crate::automata::weighted::nwa::Label;
use crate::automata::weighted::weight::Weight;
use crate::ds::bitset::BitSet;
use crate::ds::rangeset::RangeSet;

/// Compute the allowed-token mask for the current constraint state.
///
/// `state`: tokenizer DFA state → list of parser stacks (each stack is bottom-to-top)
/// `dwa`: the compiled parser DWA
/// `state_to_tsid`: tokenizer DFA state → TSID mapping
/// `max_token`: maximum token ID
/// `num_tsids`: number of token-set IDs
///
/// Returns a BitSet where bit `i` is set iff token `i` is allowed.
pub fn compute_mask(
    state: &BTreeMap<u32, Vec<Vec<u32>>>,
    dwa: &CompDwa,
    state_to_tsid: &[u32],
    max_token: u32,
    num_tsids: u32,
) -> BitSet {
    let vocab_size = max_token as usize + 1;
    let mut mask = BitSet::new(vocab_size);

    for (&tok_state, stacks) in state {
        // Look up the TSID for this tokenizer state.
        let tsid = if (tok_state as usize) < state_to_tsid.len() {
            state_to_tsid[tok_state as usize]
        } else {
            continue; // unreachable state
        };
        if tsid == u32::MAX {
            continue; // unreachable state
        }

        for stack in stacks {
            let tokens = walk_dwa_weighted(dwa, stack, tsid, num_tsids);
            for pos in tokens.iter_values() {
                if pos <= max_token {
                    mask.set(pos as usize);
                }
            }
        }
    }

    mask
}

/// Walk the DWA with weight intersection along the path.
///
/// Reads parser state IDs from the stack bottom-to-top. At each step,
/// intersects the running accumulator with the transition weight.
/// The final result is the projection of (accumulated ∩ final_weight) to TSID.
fn walk_dwa_weighted(dwa: &CompDwa, stack: &[u32], tsid: u32, num_tsids: u32) -> RangeSet {
    if stack.is_empty() || dwa.states.is_empty() {
        return RangeSet::new();
    }

    let max_pos = dwa.max_token * num_tsids + num_tsids.saturating_sub(1);
    let mut acc = Weight::all(max_pos, num_tsids);
    let mut dwa_state = dwa.start_state;

    // Read stack bottom-to-top.
    for (_i, &parser_state) in stack.iter().enumerate() {
        let label: Label = parser_state as i32;
        if let Some(&(next_state, ref weight)) =
            dwa.states[dwa_state as usize].transitions.get(&label)
        {
            if next_state as usize >= dwa.states.len() {
                return RangeSet::new();
            }
            acc = acc.intersection(weight);
            if acc.is_empty() {
                return RangeSet::new();
            }
            dwa_state = next_state;
        } else {
            // No transition for this parser state → dead end.
            return RangeSet::new();
        }
    }

    // Check if the DWA state after reading the stack is accepting.
    if dwa_state as usize >= dwa.states.len() {
        return RangeSet::new();
    }
    if let Some(ref final_weight) = dwa.states[dwa_state as usize].final_weight {
        let result = acc.intersection(final_weight);
        result.tokens_for_tsid(tsid)
    } else {
        RangeSet::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_dwa_empty_stack() {
        let dwa = CompDwa::new(1, 5);
        let tokens = walk_dwa_weighted(&dwa, &[], 0, 1);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_walk_dwa_no_transition() {
        let dwa = CompDwa::new(1, 5);
        let tokens = walk_dwa_weighted(&dwa, &[0], 0, 1);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_walk_dwa_simple_accepting() {
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = CompDwa::new(nt, max_tok);
        let s1 = dwa.add_state();

        let w_all = Weight::all(max_tok * nt + (nt - 1), nt);
        let w_tokens = Weight::from_positions(&RangeSet::from_range(1, 3), nt);
        dwa.add_transition(0, 42, s1, w_all);
        dwa.set_final_weight(s1, w_tokens);

        let tokens = walk_dwa_weighted(&dwa, &[42], 0, nt);
        assert!(!tokens.is_empty());
        assert!(tokens.contains(1));
        assert!(tokens.contains(2));
        assert!(tokens.contains(3));
    }
}
