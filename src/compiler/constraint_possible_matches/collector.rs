//! Constraint-specific possible-match collector entry points.
//!
//! Thin forwarding layer to `crate::compiler::possible_matches` so callers in
//! `pipeline.rs` and `compile.rs` do not depend on `possible_matches.rs`
//! directly.  Future constraint-specific collector optimizations should be
//! implemented or routed through this module.

use std::collections::BTreeMap;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::compiler::possible_matches::{
    self as pm,
};
use crate::grammar::flat::TerminalID;
use crate::ds::vocab_prefix_tree::VocabPrefixTreeNode;

pub(crate) use crate::compiler::possible_matches::DenseTrieClassBuildResult;
pub(crate) use crate::compiler::possible_matches::PossibleMatchesProfile;
pub(crate) use pm::emit_possible_matches_profile_summary;

/// STICKY NOTE: DO NOT REMOVE THIS COMMENT.
/// possible_matches MUST be computed for each ORIGINAL tokenizer state.
/// Do NOT collapse this to an internal TSID, representative state, or
/// tokenizer-state equivalence class, even if that looks like an easy
/// optimization. This exact mistake has recurred and it silently changes
/// semantics by merging distinct tokenizer futures.

pub(crate) fn collect_possible_matches_by_original_tsid_dense(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    pm::collect_possible_matches_by_original_tsid_dense(tokenizer, root, num_internal_tokens)
}

pub(crate) fn collect_possible_matches_by_selected_original_tsid_dense(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    pm::collect_possible_matches_by_selected_original_tsid_dense(tokenizer, root, num_internal_tokens, entries)
}

pub(crate) fn collect_possible_matches_dense_trie_class_build_with_classes(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (DenseTrieClassBuildResult, PossibleMatchesProfile) {
    pm::collect_possible_matches_dense_trie_class_build_with_classes(tokenizer, root, num_internal_tokens, entries)
}

pub(crate) fn collect_possible_matches_dense_trie_class_build(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    num_internal_tokens: u32,
    entries: &[u32],
) -> (BTreeMap<u32, BTreeMap<TerminalID, Box<[u64]>>>, PossibleMatchesProfile) {
    pm::collect_possible_matches_dense_trie_class_build(tokenizer, root, num_internal_tokens, entries)
}

pub(crate) fn count_root_child_internal_tsid_signatures(
    tokenizer: &Tokenizer,
    root: &VocabPrefixTreeNode,
    representative_states: &[u32],
    state_to_internal_tsid: &[u32],
) -> usize {
    pm::count_root_child_internal_tsid_signatures(tokenizer, root, representative_states, state_to_internal_tsid)
}
