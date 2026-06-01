//! Extracted Commit submodule.
//!
//! This file is part of the publication cleanup split of the Commit transition relation.

use super::*;

pub(super) type ParserStatesByTokenizer = FxHashMap<u32, ParserGSS>;

pub(super) const SMALL_NORMALIZED_MATCH_LINEAR_SCAN_MAX: usize = 8;

pub(super) struct NormalizedMatch {
    pub(super) terminal_id: u32,
    pub(super) width: usize,
    pub(super) ignored: bool,
}

pub(super) const SINGLE_CONCRETE_STACK_EFFECT_MAX_DEPTH: usize = 256;

pub(super) type AdvanceResultCache = FxHashMap<(usize, u32), (ParserGSS, ParserGSS)>;

pub(super) enum LinearFastPathResult {
    Complete(Result<ParserGSS, String>),
    Continue { gss: ParserGSS, offset: usize },
    Restart,
}

pub(super) struct DirectLinearStep {
    pub(super) width: usize,
    pub(super) terminal: u32,
    pub(super) ignored: bool,
    pub(super) end_state: Option<u32>,
}

