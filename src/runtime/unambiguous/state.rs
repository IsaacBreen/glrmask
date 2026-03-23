#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use crate::compiler::glr::parser::{advance_stack_vectors, stacks_accept};
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

    fn fail(&mut self) {
        self.stack.clear();
        self.tsid = self.constraint.tokenizer.initial_state();
    }

    fn token_bytes_for_id(&self, token_id: u32) -> Option<&[u8]> {
        self.constraint
            .token_bytes_dense
            .get(token_id as usize)
            .and_then(|bytes| bytes.as_deref())
            .or_else(|| self.constraint.token_bytes.get(&token_id).map(Vec::as_slice))
    }

    fn matched_terminal(&self, tokenizer_state: u32) -> Result<Option<u32>, String> {
        let mut matched = self.constraint.tokenizer.matched_terminals_iter(tokenizer_state);
        let first = matched.next();
        if matched.next().is_some() {
            return Err(
                "unambiguous state invariant violated: tokenizer state matched multiple terminals"
                    .to_string(),
            );
        }
        Ok(first)
    }

    fn advance_parser(&mut self, terminal_id: u32) -> Result<(), String> {
        let mut advanced = advance_stack_vectors(
            &self.constraint.table,
            std::slice::from_ref(&self.stack),
            terminal_id,
        );

        match advanced.len() {
            0 => {
                self.fail();
                Err("commit rejected: no valid parser states remain".to_string())
            }
            1 => {
                self.stack = advanced.pop().expect("single advanced stack exists");
                Ok(())
            }
            count => {
                self.fail();
                Err(format!(
                    "unambiguous state invariant violated: parser advanced to {count} stacks"
                ))
            }
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
        if self.stack.is_empty() {
            return ConstraintStateSummary::default();
        }

        let depth = self.stack.len().saturating_sub(1) as isize;
        ConstraintStateSummary {
            tokenizer_state_count: 1,
            nonempty_tokenizer_state_count: 1,
            parser_top_values_total: 1,
            parser_top_values_max: 1,
            parser_upperbranch_nodes_total: 0,
            parser_upperbranch_nodes_max: 0,
            parser_interface_nodes_total: 0,
            parser_interface_nodes_max: 0,
            parser_lower_nodes_total: self.stack.len(),
            parser_lower_nodes_max: self.stack.len(),
            parser_unique_nodes_total: self.stack.len(),
            parser_unique_nodes_max: self.stack.len(),
            parser_total_edges_total: self.stack.len().saturating_sub(1),
            parser_accumulator_instances_total: 1,
            parser_max_depth: depth,
        }
    }

    pub fn is_complete(&self) -> bool {
        !self.stack.is_empty()
            && self.tsid == self.constraint.tokenizer.initial_state()
            && stacks_accept(&self.constraint.table, std::slice::from_ref(&self.stack))
    }

    pub fn is_finished(&self) -> bool {
        self.is_complete()
    }

    pub fn commit_token(&mut self, token_id: u32) -> Result<(), String> {
        let bytes = self
            .token_bytes_for_id(token_id)
            .ok_or_else(|| format!("commit_token: token_id {token_id} not in vocabulary"))?
            .to_vec();
        self.commit_bytes(&bytes)
    }

    pub fn commit_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.is_empty() {
            return Ok(());
        }
        if self.stack.is_empty() {
            return Err("commit rejected: no valid parser states remain".to_string());
        }

        let initial_tsid = self.constraint.tokenizer.initial_state();

        for &byte in bytes {
            let Some(next_tsid) = self.constraint.tokenizer.step(self.tsid, byte) else {
                self.fail();
                return Err("commit rejected: no valid parser states remain".to_string());
            };
            self.tsid = next_tsid;

            let Some(terminal_id) = self.matched_terminal(self.tsid)? else {
                continue;
            };

            if Some(terminal_id) == self.constraint.ignore_terminal {
                self.tsid = initial_tsid;
                continue;
            }

            self.advance_parser(terminal_id)?;
            self.tsid = initial_tsid;
        }

        Ok(())
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
