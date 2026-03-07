




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::automata::lexer::tokenizer::Tokenizer;
use crate::automata::weighted::dwa::DWA;
use crate::compiler::glr::table::GLRTable;
use crate::compiler::grammar_def::TerminalID;
use crate::ds::leveled_gss::LeveledGSS;

use super::state::ConstraintState;

// SEP1_MAP: `TerminalTokensByState` is closest to sep1
// `GrammarConstraint.possible_matches` in `grammars2024/src/constraint.rs`.
// glrmask stores the tokenizer-state/TSID/terminal lookup directly instead of
// sep1's internal-token-bitset representation.
pub(crate) type TokenizerStateID = u32;
pub(crate) type TSID = u32;
pub(crate) type TerminalTokensByState =
    BTreeMap<TokenizerStateID, BTreeMap<TSID, BTreeMap<TerminalID, RangeSetBlaze<u32>>>>;





#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
// SEP1_MAP: `Constraint` is the direct runtime-artifact analogue of sep1
// `GrammarConstraint` in `grammars2024/src/constraint.rs`, but with sep1's
// runtime responsibilities split away from build/config code.
pub struct Constraint {
    
    
    pub(crate) parser_dwa: DWA,

    
    pub(crate) table: GLRTable,

    
    pub(crate) tokenizer: Tokenizer,

    
    
    
    #[serde(with = "crate::runtime::serde::serde_nested_btmap_rsb")]
    pub(crate) terminal_tokens_by_state: TerminalTokensByState,

    
    pub(crate) eos_token_id: Option<u32>,

    
    pub(crate) token_bytes: BTreeMap<u32, Vec<u8>>,
}

impl Constraint {
    // SEP1_MAP: `start()` corresponds to sep1 `GrammarConstraint::init()` in
    // `grammars2024/src/constraint.rs`; glrmask seeds the initial tokenizer
    // state and parser GSS directly instead of delegating through sep1's GLR
    // parser wrapper.
    
    pub fn start(&self) -> ConstraintState<'_> {
        
        
        let initial_parser_state = 0u32;
        let initial_tok_state = self.tokenizer.initial_state();

        let mut state = BTreeMap::new();
        let gss = LeveledGSS::from_stacks(&[(vec![initial_parser_state], BTreeMap::new())]);
        state.insert(initial_tok_state, gss);

        ConstraintState { constraint: self, state }
    }

    
    
    
    
    pub fn mask_len(&self) -> usize {
        unimplemented!()
    }

    // SEP1_MAP: nearest sep1 analogue is reading `GrammarConstraint.parser_dwa`
    // directly from `grammars2024/src/constraint.rs`; no separate accessor there.
    
    pub(crate) fn parser_dwa(&self) -> &DWA {
        unimplemented!()
    }
}