//! Scan relation and CanMatch construction.
//!
//! This module implements the paper's bridge between lexer scanning and parser
//! admissibility.  For a lexer state `q` and token-byte fragment `b`, scanning
//! records both completed terminals and the lexer state left at the fragment
//! boundary.  If that boundary state is not a terminal boundary, masking must
//! additionally require that the parser can accept at least one terminal in
//! `CanMatch(q')`.
//!
//! The construction is split by mathematical responsibility:
//!
//! - [`terminal_sequences`] computes sparse terminal/token maps over a vocab
//!   trie and is also used by Terminal-DWA pair partitioning.
//! - [`collector`] builds grouped interval maps from lexer states to terminal
//!   completions.
//! - [`ordered_vocab`] owns byte-sorted vocabulary tries and cache policy.
//! - [`vocab_equivalence`] computes the CanMatch-specific vocabulary quotient.
//! - [`vocab_materialize`] turns interval maps into runtime weights.
//! - [`root_collect`] contains the small-root sparse collection path.
//! - [`compute`] is the only compile-pipeline entry point.
//!
//! A crucial non-equivalence is preserved by this layout: Terminal-DWA
//! equivalence is about completed terminal sequences; scan-relation equivalence
//! is about partial-terminal completions from lexer boundary states.  Reusing one
//! quotient for the other is unsound.

#[allow(unused_imports)]
mod prelude {
    pub(super) use std::collections::BTreeMap;
    pub(super) use std::hash::Hasher;
    pub(super) use std::sync::{Arc, Mutex, OnceLock};
    pub(super) use std::time::Instant;

    pub(super) use range_set_blaze::RangeSetBlaze;
    pub(super) use rustc_hash::FxHashMap;

    pub(super) use crate::automata::lexer::tokenizer::Tokenizer;
    pub(super) use crate::compile::scan_relation::profile::elapsed_ms;
    pub(super) use crate::compile::terminal_dwa::pair_partition::equivalence_analysis::compat::{
        FlatDfa, FlatDfaState, TokenizerView,
    };
    pub(super) use crate::compile::terminal_dwa::pair_partition::equivalence_analysis::vocab::fast as vocab_equivalence_analysis;
    pub(super) use crate::compile::id_space::{InternalIdMap, ManyToOneIdMap, MappedArtifact};
    pub(super) use crate::sets::bitset::BitSet;
    pub(super) use crate::ds::vocab_prefix_tree::{VocabPrefixTree, VocabPrefixTreeNode};
    pub(super) use crate::sets::weight::{shared_rangeset, Weight};
    pub(super) use crate::grammar::flat::TerminalID;
    pub(super) use crate::vocab::VocabDerivedArtifact;
    pub(super) use crate::Vocab;
}

pub(crate) mod collector;
mod compute;
mod legacy_materialize;
mod ordered_vocab;
pub(crate) mod profile;
mod root_collect;
pub(crate) mod terminal_sequences;
mod types;
mod vocab_equivalence;
mod vocab_materialize;

pub(crate) use compute::{
    compute_scan_relation,
    compute_scan_relation_for_vocab,
    prepare_vocab_for_scan_relation,
};
pub(crate) use ordered_vocab::build_internal_token_bytes_from_groups;
pub(crate) use types::{
    RuntimeCanMatchByTerminal,
    ScanRelationComputation,
    ScanRelationConfig,
    ScanRelationProfile,
    ScanRelationVocabMap,
    SignatureClassId,
};
