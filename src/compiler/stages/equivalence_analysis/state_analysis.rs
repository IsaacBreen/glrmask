//! Tokenizer-state-side equivalence analysis.
#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::stages::equivalence_analysis::ManyToOneIdMap;

/// Analyze tokenizer-state equivalence classes for compiler use.
pub(crate) fn analyze_state_equivalences(tokenizer: &Tokenizer) -> ManyToOneIdMap {
    ManyToOneIdMap {
        original_to_internal: (0..tokenizer.dfa.num_states() as u32).collect(),
        internal_to_originals: (0..tokenizer.dfa.num_states() as u32)
            .map(|state| vec![state])
            .collect(),
    }
}
