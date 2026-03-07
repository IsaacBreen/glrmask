








































#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

#![deny(warnings)]

pub(crate) mod automata;
pub(crate) mod compiler;
pub(crate) mod ds;
pub(crate) mod import;
pub(crate) mod runtime;


pub use runtime::{Constraint, ConstraintState};
pub use compiler::debug::{AutomataDebug, CompileDebug, TerminalDebug};

use thiserror::Error;


#[derive(Error, Debug)]
pub enum GlrMaskError {
    #[error("Grammar parse error: {0}")]
    GrammarParse(String),

    #[error("Compilation error: {0}")]
    Compilation(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

pub type Result<T> = std::result::Result<T, GlrMaskError>;





#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Vocab {
    
    pub entries: Vec<(u32, Vec<u8>)>,
    
    pub eos_token_id: Option<u32>,
}

impl Vocab {
    
    const EOS_BYTES: &[u8] = b"<|endoftext|>";

    
    
    
    
    pub fn new(entries: Vec<(u32, Vec<u8>)>, eos_token_id: Option<u32>) -> Self {
        unimplemented!()
    }

    
    pub fn len(&self) -> usize {
        unimplemented!()
    }

    
    pub fn is_empty(&self) -> bool {
        unimplemented!()
    }

    
    pub fn max_token_id(&self) -> u32 {
        unimplemented!()
    }
}
