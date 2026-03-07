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
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::automata::weighted::dwa::DWA;
use crate::automata::weighted::nwa::Label;
use crate::ds::bitset::BitSet;

/// Flat low-level tokenizer-state → parser-stack map used by the runtime mask walk.
pub(crate) type FlatStateStacks = BTreeMap<u32, Vec<Vec<u32>>>;

/// Low-level mask helper over an explicit tokenizer-state → stack-map shape.
///
/// `state`: tokenizer DFA state → list of parser stacks (each stack is bottom-to-top)
/// `dwa`: the compiled parser DWA
/// `state_to_tsid`: tokenizer DFA state → TSID mapping
/// `max_token`: maximum token ID
/// `num_tsids`: number of token-set IDs
///
/// Returns a BitSet where bit `i` is set iff token `i` is allowed.
pub(crate) fn compute_mask_from_state_map(
    state: &FlatStateStacks,
    dwa: &DWA,
    state_to_tsid: &[u32],
    max_token: u32,
    num_tsids: u32,
) -> BitSet {
    unimplemented!()
}

/// Walk the DWA with weight intersection along the path.
///
/// Projects DWA weights to the target TSID column *first*, then intersects
/// in token-space only. This avoids carrying N×M-space accumulators when
/// only a single TSID column is needed.
fn walk_dwa_weighted(dwa: &DWA, stack: &[u32], tsid: u32, _num_tsids: u32) -> RangeSetBlaze<u32> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_set_blaze::RangeSetBlaze;
    use crate::ds::weight::Weight;

    #[test]
    fn test_walk_dwa_empty_stack() {
        let dwa = DWA::new(1, 5);
        let tokens = walk_dwa_weighted(&dwa, &[], 0, 1);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_walk_dwa_no_transition() {
        let dwa = DWA::new(1, 5);
        let tokens = walk_dwa_weighted(&dwa, &[0], 0, 1);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_walk_dwa_simple_accepting() {
        let nt = 1u32;
        let max_tok = 5u32;
        let mut dwa = DWA::new(nt, max_tok);
        let s1 = dwa.add_state();

        let w_all = Weight::all();
        let w_tokens = Weight::all();
        dwa.add_transition(0, 42, s1, w_all);
        dwa.set_final_weight(s1, w_tokens);

        let tokens = walk_dwa_weighted(&dwa, &[42], 0, nt);
        assert!(!tokens.is_empty());
        assert!(tokens.contains(1));
        assert!(tokens.contains(2));
        assert!(tokens.contains(3));
    }
}
