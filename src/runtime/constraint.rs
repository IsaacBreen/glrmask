




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

pub(crate) type TokenizerStateID = u32;
pub(crate) type TSID = u32;
pub(crate) type TerminalTokensByState =
    BTreeMap<TokenizerStateID, BTreeMap<TSID, BTreeMap<TerminalID, RangeSetBlaze<u32>>>>;





#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(dead_code)]
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

    
    pub(crate) fn parser_dwa(&self) -> &DWA {
        unimplemented!()
    }
}