pub use crate::grammar::ast as ast;
pub mod ebnf;
pub mod json_schema;
pub mod lark;
pub mod numeric_range;

#[cfg(test)]
mod test_grammar_import;

#[cfg(test)]
mod test_json_schema;

use crate::compiler::compile_owned;
use crate::grammar::flat::GrammarDef;
use crate::runtime::Constraint;

type GrammarParser = fn(&str) -> crate::Result<GrammarDef>;

fn compile_from_source(
    source: &str,
    vocab: &crate::Vocab,
    parse: GrammarParser,
) -> crate::Result<Constraint> {
    let grammar = parse(source)?;
    Ok(compile_owned(grammar, vocab))
}

impl Constraint {
    pub fn from_ebnf(ebnf: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(ebnf, vocab, ebnf::parse_ebnf)
    }

    pub fn from_lark(lark: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(lark, vocab, lark::parse_lark)
    }

    pub fn from_json_schema(schema: &str, vocab: &crate::Vocab) -> crate::Result<Self> {
        compile_from_source(schema, vocab, json_schema::json_schema_to_grammar)
    }
}
