
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This is the nearest analogue to sep1's `state_equivalence_analysis_fast.rs`, but currently reduced to the identity mapping instead of sep1's real DFA-state partitioning.

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::ManyToOneIdMap;


pub(crate) fn analyze_state_equivalences(tokenizer: &Tokenizer) -> ManyToOneIdMap {
    ManyToOneIdMap {
        original_to_internal: (0..tokenizer.dfa.num_states() as u32).collect(),
        internal_to_originals: (0..tokenizer.dfa.num_states() as u32)
            .map(|state| vec![state])
            .collect(),
    }
}
