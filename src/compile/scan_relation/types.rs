//! Mathematical interface types for compile-time scan-relation construction.
//!
//! The paper-level relation is intentionally represented in two layers:
//!
//! - `Scan(q, b)` is the lexer-level relation: scanning byte fragment `b` from
//!   lexer state `q` emits a sequence of completed grammar terminals and leaves
//!   a lexer state at the fragment boundary.
//! - `CanMatch(q')` is the continuation relation used when that boundary state
//!   is not a terminal boundary.  It records which grammar terminals can still
//!   be completed from `q'` by appending more bytes.
//!
//! Runtime masking consumes the second layer, not the construction machinery.
//! The weights produced here are already expressed over the final shared
//! internal token space used by the runtime artifact.

use super::prelude::*;

/// Runtime CanMatch table indexed by grammar terminal.
///
/// For each terminal `t`, the weight maps tokenizer-state classes to internal
/// token ids whose bytes can complete `t` from that class.  This is a runtime
/// artifact; it should not expose the compile-time trie or sweep-line details.
pub(crate) type RuntimeCanMatchByTerminal = BTreeMap<TerminalID, Weight>;

/// Identifier of a vocabulary equivalence class induced by CanMatch signatures.
pub(crate) type SignatureClassId = u32;

/// Label used while materializing weights: `(tokenizer_state_class, terminal)`.
pub(super) type StateTerminalLabel = (u32, TerminalID);

/// Vocabulary quotient induced by scan-relation signatures.
#[derive(Debug, Clone)]
pub(crate) struct ScanRelationVocabMap {
    pub(crate) original_to_internal: Vec<u32>,
    pub(crate) internal_to_originals: Vec<Vec<u32>>,
}

/// Construction policy for the scan relation.
///
/// The type is deliberately empty while the refactor is underway.  It marks the
/// boundary where historical environment variables should eventually become
/// explicit `CompileOptions` fields.
///
/// Warning: Terminal-DWA equivalence maps must never be reused for this
/// relation.  Terminal-DWA equivalence is a quotient by completed terminal
/// sequences; CanMatch equivalence is a quotient by possible completions from
/// partial lexer states.  The former does not imply the latter.
#[derive(Debug, Clone)]
pub(crate) struct ScanRelationConfig;

/// Timing summary reported to the compile pipeline.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ScanRelationProfile {
    pub(crate) scan_relation_collect_ms: f64,
    pub(crate) scan_relation_vocab_ms: f64,
}

/// Complete output of scan-relation construction.
#[derive(Debug)]
pub(crate) struct ScanRelationComputation {
    pub(crate) mapped_can_match: MappedArtifact<RuntimeCanMatchByTerminal>,
    pub(crate) profile: ScanRelationProfile,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SweepEvent {
    pub(super) add: bool,
    pub(super) group_id: u32,
}

#[derive(Debug, Clone)]
pub(super) struct SweepGroup {
    pub(super) label_ids: Box<[u32]>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct SweepBuildStats {
    pub(super) used_state_classes: usize,
    pub(super) terminal_groups: usize,
    pub(super) terminal_labels: usize,
    pub(super) group_label_refs: usize,
    pub(super) total_intervals: usize,
    pub(super) total_events: usize,
}
