use crate::{Constraint, DynamicConstraint, Error, Result, Vocab};

/// Grammar source to compile.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grammar<'a> {
    Ebnf(&'a str),
    Lark(&'a str),
    JsonSchema(&'a str),
    Glrm(&'a str),
}

impl<'a> Grammar<'a> {
    pub const fn ebnf(source: &'a str) -> Self { Self::Ebnf(source) }
    pub const fn lark(source: &'a str) -> Self { Self::Lark(source) }
    pub const fn json_schema(source: &'a str) -> Self { Self::JsonSchema(source) }
    pub const fn glrm(source: &'a str) -> Self { Self::Glrm(source) }
}

/// Options shared by static and dynamic constraint compilation.
#[derive(Debug, Clone, Copy)]
pub struct CompileOptions<'a> {
    end_tokens: &'a [u32],
    subgrammars: &'a [(&'a str, &'a Constraint)],
}

impl<'a> Default for CompileOptions<'a> {
    fn default() -> Self {
        Self { end_tokens: &[], subgrammars: &[] }
    }
}

impl<'a> CompileOptions<'a> {
    /// Declare model token IDs that may terminate generation once the grammar accepts.
    pub const fn end_tokens(mut self, end_tokens: &'a [u32]) -> Self {
        self.end_tokens = end_tokens;
        self
    }

    /// Bind compiled child constraints to `extern g name;` declarations in GLRM.
    pub const fn subgrammars(mut self, subgrammars: &'a [(&'a str, &'a Constraint)]) -> Self {
        self.subgrammars = subgrammars;
        self
    }
}

impl Constraint {
    /// Compile a reusable static constraint.
    pub fn compile(
        grammar: Grammar<'_>,
        vocab: &Vocab,
        options: &CompileOptions<'_>,
    ) -> Result<Self> {
        let end_tokens = options.end_tokens;
        let mut constraint = match grammar {
            Grammar::Ebnf(source) => {
                reject_non_glrm_subgrammars(options)?;
                Self::from_ebnf_with_end_tokens(source, vocab, end_tokens)
            }
            Grammar::Lark(source) => {
                reject_non_glrm_subgrammars(options)?;
                Self::from_lark_with_end_tokens(source, vocab, end_tokens)
            }
            Grammar::JsonSchema(source) => {
                reject_non_glrm_subgrammars(options)?;
                Self::from_json_schema_with_end_tokens(source, vocab, end_tokens)
            }
            Grammar::Glrm(source) if options.subgrammars.is_empty() => {
                Self::from_glrm_grammar_with_end_tokens(source, vocab, end_tokens)
            }
            Grammar::Glrm(source) => Self::from_glrm_grammar_with_subgrammars_and_end_tokens(
                source,
                options.subgrammars,
                vocab,
                end_tokens,
            ),
        }?;
        constraint.cache_serialized_artifact_for_save();
        Ok(constraint)
    }
}

impl DynamicConstraint {
    /// Compile a lower-build-latency dynamic constraint.
    pub fn compile(
        grammar: Grammar<'_>,
        vocab: &Vocab,
        options: &CompileOptions<'_>,
    ) -> Result<Self> {
        if !options.subgrammars.is_empty() {
            return Err(Error::Compilation(
                "compiled subgrammar bindings are not supported by DynamicConstraint".to_owned(),
            ));
        }
        let end_tokens = options.end_tokens;
        match grammar {
            Grammar::Ebnf(source) => Self::from_ebnf_with_end_tokens(source, vocab, end_tokens),
            Grammar::Lark(source) => Self::from_lark_with_end_tokens(source, vocab, end_tokens),
            Grammar::JsonSchema(source) => {
                Self::from_json_schema_with_end_tokens(source, vocab, end_tokens)
            }
            Grammar::Glrm(source) => {
                Self::from_glrm_grammar_with_end_tokens(source, vocab, end_tokens)
            }
        }
    }
}

fn reject_non_glrm_subgrammars(options: &CompileOptions<'_>) -> Result<()> {
    if options.subgrammars.is_empty() {
        Ok(())
    } else {
        Err(Error::Compilation(
            "compiled subgrammar bindings require a GLRM grammar".to_owned(),
        ))
    }
}
