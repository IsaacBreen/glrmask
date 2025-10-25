use std::collections::HashSet;

use crate::constraint::{
    GrammarConstraint, GrammarConstraintState, LLMTokenBV, PrecomputeNode1Index,
};
use crate::glr::table::{NonTerminalID, StateID};
use crate::types::TerminalID;

// Types for special precomputation
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub enum SpecialPrecomputeDest {
        Reduce { pop: usize, dest_nt: NonTerminalID },
        Escape { push_states: Vec<StateID> },
    }

    // (Option<NonTerminalID>, StateID, TerminalID, SpecialPrecomputeDest)
    pub type SpecialPrecomputeNormalEdge =
        (Option<NonTerminalID>, StateID, TerminalID, SpecialPrecomputeDest);

    // (Option<NonTerminalID>, TerminalID, (usize, NonTerminalID), LLMTokenBV, PrecomputeNode1Index, PrecomputeNode1Index)
    pub type SpecialPrecomputeSuperEdge = (
        Option<NonTerminalID>,
        TerminalID,
        (usize, NonTerminalID),
        LLMTokenBV,
        PrecomputeNode1Index,
        PrecomputeNode1Index,
    );

    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub struct SpecialPrecomputation {
        pub normal_edges: HashSet<SpecialPrecomputeNormalEdge>,
        pub super_edges: HashSet<SpecialPrecomputeSuperEdge>,
    }

    pub fn precompute_special(_gc: &GrammarConstraint) -> SpecialPrecomputation {
        todo!()
    }

    pub fn get_mask4(gcs: &GrammarConstraintState) -> LLMTokenBV {
        todo!()
    }