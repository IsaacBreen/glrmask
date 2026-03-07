




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

// SEP1_MAP: This module is the trimmed glrmask counterpart to sep1's `equivalence_analysis/` tree, keeping only the ID-remapping surface the compiler still needs.

pub mod combined;
pub mod state_analysis;
pub mod vocab_analysis;


#[derive(Debug, Clone)]
pub struct ManyToOneIdMap {
    
    
    pub original_to_internal: Vec<u32>,
    
    
    pub internal_to_originals: Vec<Vec<u32>>,
}

impl ManyToOneIdMap {
    
    pub fn num_internal_ids(&self) -> u32 {
        self.internal_to_originals.len() as u32
    }

    
    pub fn max_original_id(&self) -> u32 {
        self.original_to_internal
            .len()
            .checked_sub(1)
            .map(|i| i as u32)
            .unwrap_or(0)
    }
}


#[derive(Debug, Clone)]
pub struct InternalIdMap {
    
    pub tokenizer_states: ManyToOneIdMap,
    
    pub vocab_tokens: ManyToOneIdMap,
}

impl InternalIdMap {
    
    pub fn build(tokenizer: &crate::automata::lexer::tokenizer::Tokenizer, vocab: &crate::Vocab) -> Self {
        combined::analyze_equivalences(tokenizer, vocab)
    }

    
    pub fn num_tsids(&self) -> u32 {
        self.tokenizer_states.num_internal_ids()
    }

    
    pub fn max_token_id(&self) -> u32 {
        self.vocab_tokens.max_original_id()
    }
}

pub(crate) use combined::analyze_equivalences;
