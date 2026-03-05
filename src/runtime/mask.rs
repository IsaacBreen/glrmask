//! Hot-path mask computation.

use crate::automata::weighted::dwa::Dwa;
use crate::ds::bitset::BitSet;

/// Compute the allowed-token mask for a given DWA state.
///
/// For each token `t`, looks up `tsid = vocab_mapping[t]`, then checks
/// whether `dwa.step(state, tsid)` leads to a non-dead state with non-negative weight.
///
/// This is the innermost hot loop at inference time.
pub fn compute_mask(dwa: &Dwa, state: u32, vocab_mapping: &[u32], vocab_size: usize) -> BitSet {
    let mut mask = BitSet::new(vocab_size);
    for token_id in 0..vocab_size {
        let tsid = vocab_mapping[token_id];
        let (target, weight) = dwa.step(state, tsid);
        if target != u32::MAX && weight >= 0 {
            mask.set(token_id);
        }
    }
    mask
}
