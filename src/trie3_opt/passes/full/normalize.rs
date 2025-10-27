use std::collections::BTreeMap;

use crate::constraint::{LLMTokenBV, PrecomputeNode3Index, Trie3GodWrapper};
use crate::datastructures::trie::Trie;
use crate::tokenizer::TokenizerStateID;

/// Recompute the derived 'live_tokens' field for all nodes from scratch as the union of outgoing
/// LLM masks. This normalizes away incidental differences introduced by earlier passes and ensures
/// 'live_tokens' remains a pure derivative that does not affect optimization semantics.
pub fn normalize_live_tokens_trie3(
    roots: &BTreeMap<TokenizerStateID, PrecomputeNode3Index>,
    trie3_god: &Trie3GodWrapper,
) {
    let roots_vec: Vec<_> = roots.values().cloned().collect();
    let all_nodes = Trie::all_nodes(trie3_god, &roots_vec);
    if all_nodes.is_empty() {
        return;
    }
    for n in &all_nodes {
        if let Some(mut w) = n.write(trie3_god) {
            let mut new_live = LLMTokenBV::zeros();
            for ((_, llm_bv), _) in w.children() {
                new_live |= llm_bv;
            }
            w.value.live_tokens = new_live;
        }
    }
}
