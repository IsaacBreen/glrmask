use std::collections::BTreeMap;

use crate::constraint::{Trie3GodWrapper, PrecomputeNode3Index};

/// Simplifying LLM token bitsets was found to be unsound in previous iterations.
/// Keep as a no-op to avoid semantic changes.
pub fn simplify_llm_token_bvs_trie3(
    _roots: &BTreeMap<crate::tokenizer::TokenizerStateID, PrecomputeNode3Index>,
    _trie3_god: &Trie3GodWrapper,
    _max_llm_token_id: usize,
) {
    // Intentionally no-op; see notes in the original monolithic implementation.
}
