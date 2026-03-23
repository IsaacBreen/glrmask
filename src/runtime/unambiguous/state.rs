#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::ds::leveled_gss::LeveledGSS;
use crate::runtime::ambiguous::AmbiguousConstraintState;
use crate::runtime::state::ConstraintStateSummary;
use crate::runtime::{CommitDebugMetrics, CommitDebugTrace, Constraint, MaskDebugMetrics};

#[derive(Debug, Clone)]
pub struct UnambiguousConstraintState<'a> {
    pub(crate) stack: Vec<u32>,
    pub(crate) tsid: u32,
    pub(crate) constraint: &'a Constraint,
}

impl<'a> UnambiguousConstraintState<'a> {
    pub(crate) fn new(constraint: &'a Constraint) -> Self {
        Self {
            stack: vec![0],
            tsid: constraint.tokenizer.initial_state(),
            constraint,
        }
    }

    fn failed(constraint: &'a Constraint) -> Self {
        Self {
            stack: Vec::new(),
            tsid: constraint.tokenizer.initial_state(),
            constraint,
        }
    }

    fn as_ambiguous(&self) -> AmbiguousConstraintState<'a> {
        let mut state = BTreeMap::new();
        if !self.stack.is_empty() {
            let gss = LeveledGSS::from_stacks(&[(self.stack.clone(), BTreeMap::new())]);
            state.insert(self.tsid, gss);
        }
        AmbiguousConstraintState {
            constraint: self.constraint,
            state,
        }
    }

    fn from_ambiguous(state: AmbiguousConstraintState<'a>) -> Result<Self, String> {
        let AmbiguousConstraintState { constraint, state } = state;

        if state.is_empty() {
            return Ok(Self::failed(constraint));
        }

        if state.len() != 1 {
            return Err(format!(
                "unambiguous state invariant violated: expected 1 tokenizer state, found {}",
                state.len()
            ));
        }

        let (tsid, gss) = state.into_iter().next().expect("state is nonempty");
        if gss.is_empty() {
            return Ok(Self::failed(constraint));
        }

        let stacks = gss.to_stacks();
        if stacks.len() != 1 {
            return Err(format!(
                "unambiguous state invariant violated: expected 1 parser stack, found {}",
                stacks.len()
            ));
        }

        let (stack, disallowed) = stacks.into_iter().next().expect("stacks is nonempty");
        if !disallowed.is_empty() {
            return Err(
                "unambiguous state invariant violated: found tokenizer ambiguity bookkeeping"
                    .to_string(),
            );
        }

        Ok(Self {
            stack,
            tsid,
            constraint,
        })
    }

    fn apply_projected_result(
        &mut self,
        projected: AmbiguousConstraintState<'a>,
        result: Result<(), String>,
    ) -> Result<(), String> {
        match Self::from_ambiguous(projected) {
            Ok(next) => {
                *self = next;
                result
            }
            Err(err) => {
                *self = Self::failed(self.constraint);
                Err(err)
            }
        }
    }

    pub fn summary(&self) -> ConstraintStateSummary {
        self.as_ambiguous().summary()
    }

    pub fn is_complete(&self) -> bool {
        self.as_ambiguous().is_complete()
    }

    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }

    pub fn commit_token(&mut self, token_id: u32) -> Result<(), String> {
        let mut projected = self.as_ambiguous();
        let result = projected.commit_token(token_id);
        self.apply_projected_result(projected, result)
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        let mut projected = self.as_ambiguous();
        let result = projected.commit_bytes(bytes);
        self.apply_projected_result(projected, result)
    }

    pub fn commit_tokens(&mut self, tokens: &[u32]) -> Result<(), String> {
        for &token in tokens {
            self.commit_token(token)?;
        }
        Ok(())
    }

    pub fn mask(&self) -> Vec<u32> {
        self.as_ambiguous().mask()
    }

    pub fn fill_mask(&self, buf: &mut [u32]) {
        self.as_ambiguous().fill_mask(buf);
    }

    pub fn force(&self) -> Vec<u32> {
        self.as_ambiguous().force()
    }

    pub fn debug_commit_bytes_metrics(&self, bytes: &[u8]) -> CommitDebugMetrics {
        self.as_ambiguous().debug_commit_bytes_metrics(bytes)
    }

    pub fn debug_commit_bytes_trace(&self, bytes: &[u8]) -> CommitDebugTrace {
        self.as_ambiguous().debug_commit_bytes_trace(bytes)
    }

    pub fn debug_commit_token_metrics(&self, token_id: u32) -> Result<CommitDebugMetrics, String> {
        self.as_ambiguous().debug_commit_token_metrics(token_id)
    }

    pub fn debug_commit_token_trace(&self, token_id: u32) -> Result<CommitDebugTrace, String> {
        self.as_ambiguous().debug_commit_token_trace(token_id)
    }

    pub fn debug_mask_metrics(&self) -> MaskDebugMetrics {
        self.as_ambiguous().debug_mask_metrics()
    }
}
