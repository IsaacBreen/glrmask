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
    end_token_ids: &'a [u32],
    subgrammars: &'a [(&'a str, &'a Constraint)],
    external_terminal_bindings: &'a [ExternalTerminalBinding<'a>],
}

/// Binds a GLRM v1 `extern t NAME;` declaration to exact model token IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExternalTerminalBinding<'a> {
    name: &'a str,
    token_ids: &'a [u32],
}

impl<'a> ExternalTerminalBinding<'a> {
    pub const fn new(name: &'a str, token_ids: &'a [u32]) -> Self {
        Self { name, token_ids }
    }

    pub(crate) const fn name(&self) -> &'a str {
        self.name
    }

    pub(crate) const fn token_ids(&self) -> &'a [u32] {
        self.token_ids
    }
}

impl<'a> Default for CompileOptions<'a> {
    fn default() -> Self {
        Self { end_token_ids: &[], subgrammars: &[], external_terminal_bindings: &[] }
    }
}

impl<'a> CompileOptions<'a> {
    /// Declare model token IDs that may terminate generation once the grammar accepts.
    pub const fn end_token_ids(mut self, end_token_ids: &'a [u32]) -> Self {
        self.end_token_ids = end_token_ids;
        self
    }

    /// Bind compiled child constraints to `extern g name;` declarations in GLRM.
    pub const fn subgrammars(mut self, subgrammars: &'a [(&'a str, &'a Constraint)]) -> Self {
        self.subgrammars = subgrammars;
        self
    }

    /// Bind named GLRM v1 external terminals to one or more exact token IDs.
    pub const fn external_terminal_bindings(
        mut self,
        bindings: &'a [ExternalTerminalBinding<'a>],
    ) -> Self {
        self.external_terminal_bindings = bindings;
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
        let end_token_ids = options.end_token_ids;
        match grammar {
            Grammar::Ebnf(source) => {
                reject_non_glrm_subgrammars(options)?;
                Self::from_ebnf_with_end_tokens(source, vocab, end_token_ids)
            }
            Grammar::Lark(source) => {
                reject_non_glrm_subgrammars(options)?;
                Self::from_lark_with_end_tokens(source, vocab, end_token_ids)
            }
            Grammar::JsonSchema(source) => {
                reject_non_glrm_subgrammars(options)?;
                Self::from_json_schema_with_end_tokens(source, vocab, end_token_ids)
            }
            Grammar::Glrm(source) if options.subgrammars.is_empty() => {
                Self::from_glrm_grammar_with_bindings_and_end_tokens(
                    source,
                    vocab,
                    options.external_terminal_bindings,
                    end_token_ids,
                )
            }
            Grammar::Glrm(source) => Self::from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
                source,
                options.subgrammars,
                vocab,
                options.external_terminal_bindings,
                end_token_ids,
            ),
        }
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
        let end_token_ids = options.end_token_ids;
        match grammar {
            Grammar::Ebnf(source) => Self::from_ebnf_with_end_tokens(source, vocab, end_token_ids),
            Grammar::Lark(source) => Self::from_lark_with_end_tokens(source, vocab, end_token_ids),
            Grammar::JsonSchema(source) => {
                Self::from_json_schema_with_end_tokens(source, vocab, end_token_ids)
            }
            Grammar::Glrm(source) => {
                Self::from_glrm_grammar_with_bindings_and_end_tokens(
                    source,
                    vocab,
                    options.external_terminal_bindings,
                    end_token_ids,
                )
            }
        }
    }
}

fn reject_non_glrm_subgrammars(options: &CompileOptions<'_>) -> Result<()> {
    if options.subgrammars.is_empty() && options.external_terminal_bindings.is_empty() {
        Ok(())
    } else {
        Err(Error::Compilation(
            "subgrammar and external-terminal bindings require a GLRM grammar".to_owned(),
        ))
    }
}
