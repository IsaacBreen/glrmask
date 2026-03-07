




#![allow(dead_code)]
#![allow(unused_mut)]
#![allow(unused_variables)]
#![allow(unused_imports)]

pub mod ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;

pub use ast as grammar_expr;

use crate::compiler::debug::CompileDebug;
use crate::runtime::Constraint;


fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (ebnf, vocab);
    unimplemented!()
}


fn from_ebnf_with_debug(
    ebnf: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (ebnf, vocab);
    unimplemented!()
}


fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (lark, vocab);
    unimplemented!()
}


fn from_lark_with_debug(
    lark: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (lark, vocab);
    unimplemented!()
}


fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Constraint> {
    let _ = (schema, vocab);
    unimplemented!()
}


fn from_json_schema_with_debug(
    schema: &str,
    vocab: &crate::Vocab,
) -> crate::Result<(Constraint, CompileDebug)> {
    let _ = (schema, vocab);
    unimplemented!()
}

impl Constraint {
    
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_ebnf(ebnf, vocab)
    }

    
    
    
    pub(crate) fn from_ebnf_with_debug(
        ebnf: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_ebnf_with_debug(ebnf, vocab)
    }

    
    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_lark(lark, vocab)
    }

    
    pub(crate) fn from_lark_with_debug(
        lark: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_lark_with_debug(lark, vocab)
    }

    
    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        from_json_schema(schema, vocab)
    }

    
    pub(crate) fn from_json_schema_with_debug(
        schema: &str,
        vocab: &crate::Vocab,
    ) -> crate::Result<(Self, CompileDebug)> {
        from_json_schema_with_debug(schema, vocab)
    }
}
