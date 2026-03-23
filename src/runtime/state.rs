#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub use super::ambiguous::AmbiguousConstraintState;
pub use super::unambiguous::UnambiguousConstraintState;
use super::ambiguous::{CommitDebugMetrics, CommitDebugTrace, MaskDebugMetrics};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConstraintStateSummary {
    pub tokenizer_state_count: usize,
    pub nonempty_tokenizer_state_count: usize,
    pub parser_top_values_total: usize,
    pub parser_top_values_max: usize,
    pub parser_upperbranch_nodes_total: usize,
    pub parser_upperbranch_nodes_max: usize,
    pub parser_interface_nodes_total: usize,
    pub parser_interface_nodes_max: usize,
    pub parser_lower_nodes_total: usize,
    pub parser_lower_nodes_max: usize,
    pub parser_unique_nodes_total: usize,
    pub parser_unique_nodes_max: usize,
    pub parser_total_edges_total: usize,
    pub parser_accumulator_instances_total: usize,
    pub parser_max_depth: isize,
}

pub trait ConstraintStateTrait {
    fn summary(&self) -> ConstraintStateSummary;

    fn is_complete(&self) -> bool;

    fn is_finished(&self) -> bool {
        self.is_complete()
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), String>;

    fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String>;

    fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String>;

    fn mask(&self) -> Vec<u32>;

    fn fill_mask(&self, buf: &mut [u32]);

    fn force(&self) -> Vec<u32>;

    fn debug_commit_bytes_metrics(&self, bytes: &[u8]) -> CommitDebugMetrics;

    fn debug_commit_bytes_trace(&self, bytes: &[u8]) -> CommitDebugTrace;

    fn debug_commit_token_metrics(&self, token_id: u32) -> Result<CommitDebugMetrics, String>;

    fn debug_commit_token_trace(&self, token_id: u32) -> Result<CommitDebugTrace, String>;

    fn debug_mask_metrics(&self) -> MaskDebugMetrics;
}

impl ConstraintStateTrait for AmbiguousConstraintState<'_> {
    fn summary(&self) -> ConstraintStateSummary {
        AmbiguousConstraintState::summary(self)
    }

    fn is_complete(&self) -> bool {
        AmbiguousConstraintState::is_complete(self)
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), String> {
        AmbiguousConstraintState::commit_token(self, token_id)
    }

    fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        AmbiguousConstraintState::commit_bytes(self, bytes)
    }

    fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        AmbiguousConstraintState::commit_tokens(self, tokens)
    }

    fn mask(&self) -> Vec<u32> {
        AmbiguousConstraintState::mask(self)
    }

    fn fill_mask(&self, buf: &mut [u32]) {
        AmbiguousConstraintState::fill_mask(self, buf)
    }

    fn force(&self) -> Vec<u32> {
        AmbiguousConstraintState::force(self)
    }

    fn debug_commit_bytes_metrics(&self, bytes: &[u8]) -> CommitDebugMetrics {
        AmbiguousConstraintState::debug_commit_bytes_metrics(self, bytes)
    }

    fn debug_commit_bytes_trace(&self, bytes: &[u8]) -> CommitDebugTrace {
        AmbiguousConstraintState::debug_commit_bytes_trace(self, bytes)
    }

    fn debug_commit_token_metrics(&self, token_id: u32) -> Result<CommitDebugMetrics, String> {
        AmbiguousConstraintState::debug_commit_token_metrics(self, token_id)
    }

    fn debug_commit_token_trace(&self, token_id: u32) -> Result<CommitDebugTrace, String> {
        AmbiguousConstraintState::debug_commit_token_trace(self, token_id)
    }

    fn debug_mask_metrics(&self) -> MaskDebugMetrics {
        AmbiguousConstraintState::debug_mask_metrics(self)
    }
}

impl ConstraintStateTrait for UnambiguousConstraintState<'_> {
    fn summary(&self) -> ConstraintStateSummary {
        UnambiguousConstraintState::summary(self)
    }

    fn is_complete(&self) -> bool {
        UnambiguousConstraintState::is_complete(self)
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), String> {
        UnambiguousConstraintState::commit_token(self, token_id)
    }

    fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        UnambiguousConstraintState::commit_bytes(self, bytes)
    }

    fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        UnambiguousConstraintState::commit_tokens(self, tokens)
    }

    fn mask(&self) -> Vec<u32> {
        UnambiguousConstraintState::mask(self)
    }

    fn fill_mask(&self, buf: &mut [u32]) {
        UnambiguousConstraintState::fill_mask(self, buf)
    }

    fn force(&self) -> Vec<u32> {
        UnambiguousConstraintState::force(self)
    }

    fn debug_commit_bytes_metrics(&self, bytes: &[u8]) -> CommitDebugMetrics {
        UnambiguousConstraintState::debug_commit_bytes_metrics(self, bytes)
    }

    fn debug_commit_bytes_trace(&self, bytes: &[u8]) -> CommitDebugTrace {
        UnambiguousConstraintState::debug_commit_bytes_trace(self, bytes)
    }

    fn debug_commit_token_metrics(&self, token_id: u32) -> Result<CommitDebugMetrics, String> {
        UnambiguousConstraintState::debug_commit_token_metrics(self, token_id)
    }

    fn debug_commit_token_trace(&self, token_id: u32) -> Result<CommitDebugTrace, String> {
        UnambiguousConstraintState::debug_commit_token_trace(self, token_id)
    }

    fn debug_mask_metrics(&self) -> MaskDebugMetrics {
        UnambiguousConstraintState::debug_mask_metrics(self)
    }
}

#[derive(Debug, Clone)]
pub enum ConstraintState<'a> {
    Ambiguous(AmbiguousConstraintState<'a>),
    Unambiguous(UnambiguousConstraintState<'a>),
}

impl<'a> ConstraintState<'a> {
    pub fn summary(&self) -> ConstraintStateSummary {
        match self {
            Self::Ambiguous(state) => state.summary(),
            Self::Unambiguous(state) => state.summary(),
        }
    }

    pub fn is_complete(&self) -> bool {
        match self {
            Self::Ambiguous(state) => state.is_complete(),
            Self::Unambiguous(state) => state.is_complete(),
        }
    }

    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }

    pub fn commit_token(&mut self, token_id: u32) -> Result<(), String> {
        match self {
            Self::Ambiguous(state) => state.commit_token(token_id),
            Self::Unambiguous(state) => state.commit_token(token_id),
        }
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        match self {
            Self::Ambiguous(state) => state.commit_bytes(bytes),
            Self::Unambiguous(state) => state.commit_bytes(bytes),
        }
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        match self {
            Self::Ambiguous(state) => state.commit_tokens(tokens),
            Self::Unambiguous(state) => state.commit_tokens(tokens),
        }
    }

    pub fn mask(&self) -> Vec<u32> {
        match self {
            Self::Ambiguous(state) => state.mask(),
            Self::Unambiguous(state) => state.mask(),
        }
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        match self {
            Self::Ambiguous(state) => state.fill_mask(buf),
            Self::Unambiguous(state) => state.fill_mask(buf),
        }
    }

    pub fn force(&self) -> Vec<u32> {
        match self {
            Self::Ambiguous(state) => state.force(),
            Self::Unambiguous(state) => state.force(),
        }
    }

    pub fn debug_commit_bytes_metrics(&self, bytes: &[u8]) -> CommitDebugMetrics {
        match self {
            Self::Ambiguous(state) => state.debug_commit_bytes_metrics(bytes),
            Self::Unambiguous(state) => state.debug_commit_bytes_metrics(bytes),
        }
    }

    pub fn debug_commit_bytes_trace(&self, bytes: &[u8]) -> CommitDebugTrace {
        match self {
            Self::Ambiguous(state) => state.debug_commit_bytes_trace(bytes),
            Self::Unambiguous(state) => state.debug_commit_bytes_trace(bytes),
        }
    }

    pub fn debug_commit_token_metrics(&self, token_id: u32) -> Result<CommitDebugMetrics, String> {
        match self {
            Self::Ambiguous(state) => state.debug_commit_token_metrics(token_id),
            Self::Unambiguous(state) => state.debug_commit_token_metrics(token_id),
        }
    }

    pub fn debug_commit_token_trace(&self, token_id: u32) -> Result<CommitDebugTrace, String> {
        match self {
            Self::Ambiguous(state) => state.debug_commit_token_trace(token_id),
            Self::Unambiguous(state) => state.debug_commit_token_trace(token_id),
        }
    }

    pub fn debug_mask_metrics(&self) -> MaskDebugMetrics {
        match self {
            Self::Ambiguous(state) => state.debug_mask_metrics(),
            Self::Unambiguous(state) => state.debug_mask_metrics(),
        }
    }
}

impl ConstraintStateTrait for ConstraintState<'_> {
    fn summary(&self) -> ConstraintStateSummary {
        ConstraintState::summary(self)
    }

    fn is_complete(&self) -> bool {
        ConstraintState::is_complete(self)
    }

    fn commit_token(&mut self, token_id: u32) -> Result<(), String> {
        ConstraintState::commit_token(self, token_id)
    }

    fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        ConstraintState::commit_bytes(self, bytes)
    }

    fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        ConstraintState::commit_tokens(self, tokens)
    }

    fn mask(&self) -> Vec<u32> {
        ConstraintState::mask(self)
    }

    fn fill_mask(&self, buf: &mut [u32]) {
        ConstraintState::fill_mask(self, buf)
    }

    fn force(&self) -> Vec<u32> {
        ConstraintState::force(self)
    }

    fn debug_commit_bytes_metrics(&self, bytes: &[u8]) -> CommitDebugMetrics {
        ConstraintState::debug_commit_bytes_metrics(self, bytes)
    }

    fn debug_commit_bytes_trace(&self, bytes: &[u8]) -> CommitDebugTrace {
        ConstraintState::debug_commit_bytes_trace(self, bytes)
    }

    fn debug_commit_token_metrics(&self, token_id: u32) -> Result<CommitDebugMetrics, String> {
        ConstraintState::debug_commit_token_metrics(self, token_id)
    }

    fn debug_commit_token_trace(&self, token_id: u32) -> Result<CommitDebugTrace, String> {
        ConstraintState::debug_commit_token_trace(self, token_id)
    }

    fn debug_mask_metrics(&self) -> MaskDebugMetrics {
        ConstraintState::debug_mask_metrics(self)
    }
}
