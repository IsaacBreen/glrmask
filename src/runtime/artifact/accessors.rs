//! Public and crate-internal accessors over the runtime artifact.
//!
//! These methods expose the compiled artifact without explaining how masks or
//! commits are evaluated.  They are intentionally separate from cache building
//! and mask materialization.

use std::collections::BTreeMap;
use std::sync::Mutex;

use range_set_blaze::RangeSetBlaze;

use crate::automata::weighted::dwa::DWA;
use crate::grammar::flat::TerminalID;
use crate::runtime::state::ConstraintState;

use super::Constraint;

impl Constraint {
    /// # Example
    ///
    /// ```ignore
    /// let constraint = glrmask::Constraint::from_ebnf("start = "a";", &vocab)?;
    /// let mut state = constraint.start();
    /// let mut mask = vec![0; constraint.mask_len()];
    /// state.fill_mask(&mut mask);
    /// state.commit_token(0)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn start(&self) -> ConstraintState<'_> {
        let state = self.initial_state_map();
        let state = ConstraintState {
            constraint: self,
            state,
            buffers: Default::default(),
            generation: 0,
            mask_cache: Mutex::new(None),
            mask_scratch: Mutex::new(Default::default()),
        };
        state.prefill_mask_cache();
        state
    }

    /// Return the number of `u32` words needed to store masks over original token ids.
    pub fn mask_len(&self) -> usize {
        self.token_bytes
            .keys()
            .max()
            .map(|token_id| (*token_id as usize / 32) + 1)
            .unwrap_or(0)
    }

    /// Map compact runtime-internal token ids to the original vocabulary token ids.
    ///
    /// Mask evaluation may merge equivalent original tokens into one internal id
    /// after Terminal-DWA/Parser-DWA reconciliation.  This accessor exposes that
    /// final token-space quotient for diagnostics and benchmark compatibility.
    pub fn internal_to_original_token_ids(&self) -> &[Vec<u32>] {
        &self.internal_token_to_tokens
    }

    /// Map original vocabulary token ids to compact runtime-internal token ids.
    ///
    /// Entries are indexed by original token id.  Values identify the internal
    /// token class used during mask traversal/materialization.
    pub fn original_to_internal_token_ids(&self) -> &[u32] {
        &self.original_token_to_internal
    }



    /// Return the number of states in the compiled GLR parser table.
    pub fn num_parser_states(&self) -> u32 {
        self.table.num_states
    }

    pub(crate) fn parser_dwa(&self) -> &DWA {
        &self.parser_dwa
    }


    pub(crate) fn can_match_for_state_internal(
        &self,
        tokenizer_state: u32,
    ) -> Option<BTreeMap<TerminalID, RangeSetBlaze<u32>>> {
        // Return can_match in the final shared constraint-internal vocab
        // space. These ids match parser-DWA weight token ids after reconciliation.
        let internal_tsid = self.internal_tsid_for_state(tokenizer_state);
        let mut result = BTreeMap::new();
        for (&terminal, weight) in &self.can_match {
            let tokens = weight.tokens_for_tsid(internal_tsid);
            if !tokens.is_empty() {
                result.insert(terminal, tokens);
            }
        }
        if result.is_empty() {
            None
        } else {
            Some(result)
        }
    }


    pub(crate) fn max_original_token_id(&self) -> Option<u32> {
        self.token_bytes.keys().next_back().copied()
    }


}
