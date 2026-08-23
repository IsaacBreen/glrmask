use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::runtime::{Constraint as RuntimeConstraint, ConstraintState as RuntimeConstraintState};
use crate::{DynamicConstraint, DynamicConstraintState, Error, Result, Vocab};

/// Target-neutral grammar source, optionally with source-level subgrammar bindings.
///
/// `Grammar::bind_grammar` resolves only `extern grammar` declarations, so it
/// remains independent of a decoder vocabulary. Exact-token externs are bound
/// later through [`ConstraintSpecBuilder::bind_token`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Grammar<'a> {
    source: GrammarSource<'a>,
    grammar_bindings: BTreeMap<String, Grammar<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GrammarSource<'a> {
    Ebnf(&'a str),
    Lark(&'a str),
    JsonSchema(&'a str),
    Glrm(&'a str),
}

impl<'a> Grammar<'a> {
    pub fn ebnf(source: &'a str) -> Self { Self::new(GrammarSource::Ebnf(source)) }
    pub fn lark(source: &'a str) -> Self { Self::new(GrammarSource::Lark(source)) }
    pub fn json_schema(source: &'a str) -> Self { Self::new(GrammarSource::JsonSchema(source)) }
    pub fn glrm(source: &'a str) -> Self { Self::new(GrammarSource::Glrm(source)) }

    fn new(source: GrammarSource<'a>) -> Self {
        Self { source, grammar_bindings: BTreeMap::new() }
    }

    /// Bind an `extern grammar NAME;` directly at the target-neutral grammar layer.
    ///
    /// This is useful when both parent and child are still grammar sources.
    /// Target-bound realizations such as compiled constraints belong on
    /// [`ConstraintSpecBuilder::bind_grammar`] instead.
    pub fn bind_grammar(mut self, name: impl AsRef<str>, grammar: Grammar<'a>) -> Result<Self> {
        let name = name.as_ref();
        let GrammarSource::Glrm(source) = self.source else {
            return Err(Error::Compilation(
                "source-level subgrammar bindings require a GLRM parent grammar".to_owned(),
            ));
        };
        let declarations = crate::grammar::glrm::external_declarations(source)?;
        if declarations.token_names.iter().any(|declared| declared == name) {
            return Err(Error::Compilation(format!(
                "external {name:?} has kind token, not grammar",
            )));
        }
        if !declarations.grammar_names.iter().any(|declared| declared == name) {
            return Err(Error::Compilation(format!(
                "no external grammar named {name:?} is declared",
            )));
        }
        if self.grammar_bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "external grammar {name:?} was bound more than once",
            )));
        }
        self.grammar_bindings.insert(name.to_owned(), grammar);
        Ok(self)
    }

    fn glrm_source(&self) -> Option<&'a str> {
        match self.source {
            GrammarSource::Glrm(source) => Some(source),
            _ => None,
        }
    }

    fn into_source_only_and_bindings(self) -> (Self, BTreeMap<String, Grammar<'a>>) {
        let Self { source, grammar_bindings } = self;
        (Self::new(source), grammar_bindings)
    }
}

/// Lowering and compilation policy shared by static and dynamic compilation.
///
/// Version 1 defines no public policy switches yet. The non-exhaustive shape
/// permits future lowering choices without mixing target bindings into this type.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub struct CompileOptions {}

/// A compiled constraint that can start an independent mutable sequence state.
pub trait Constraint {
    type State<'a>
    where
        Self: 'a;

    fn start(&self) -> Self::State<'_>;
    fn mask_len(&self) -> usize;
}

impl Constraint for RuntimeConstraint {
    type State<'a> = RuntimeConstraintState<'a>;

    fn start(&self) -> Self::State<'_> { RuntimeConstraint::start(self) }
    fn mask_len(&self) -> usize { RuntimeConstraint::mask_len(self) }
}

impl Constraint for DynamicConstraint {
    type State<'a> = DynamicConstraintState<'a>;

    fn start(&self) -> Self::State<'_> { DynamicConstraint::start(self) }
    fn mask_len(&self) -> usize { DynamicConstraint::mask_len(self) }
}

/// A target-bound, immutable grammar specification with complete extern bindings.
#[derive(Debug, Clone)]
pub struct ConstraintSpec<'a> {
    grammar: Grammar<'a>,
    vocab: &'a Vocab,
    token_bindings: BTreeMap<String, Vec<u32>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
}

/// Builder for a target-bound [`ConstraintSpec`].
#[derive(Debug)]
pub struct ConstraintSpecBuilder<'a> {
    grammar: Grammar<'a>,
    vocab: &'a Vocab,
    declared_tokens: BTreeSet<String>,
    declared_grammars: BTreeSet<String>,
    token_bindings: BTreeMap<String, Vec<u32>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
}

/// Accepted input to [`ConstraintSpecBuilder::bind_grammar`].
///
/// This is public only to support the conversion trait; its variants are not a
/// standalone binding API.
#[doc(hidden)]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum GrammarBinding<'a> {
    Source(Grammar<'a>),
    Spec(Box<ConstraintSpec<'a>>),
    #[doc(hidden)]
    StaticBorrowed(&'a RuntimeConstraint),
    #[doc(hidden)]
    StaticOwned(Arc<RuntimeConstraint>),
    #[doc(hidden)]
    DynamicBorrowed(&'a DynamicConstraint),
    #[doc(hidden)]
    DynamicOwned(Arc<DynamicConstraint>),
}

/// Converts a source, spec, or compiled artifact into one grammar realization.
pub trait IntoGrammarBinding<'a> {
    #[doc(hidden)]
    fn into_grammar_binding(self) -> GrammarBinding<'a>;
}

impl<'a> IntoGrammarBinding<'a> for Grammar<'a> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> { GrammarBinding::Source(self) }
}

impl<'a> IntoGrammarBinding<'a> for ConstraintSpec<'a> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::Spec(Box::new(self))
    }
}

impl<'a> IntoGrammarBinding<'a> for &ConstraintSpec<'a> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::Spec(Box::new(self.clone()))
    }
}

impl<'a> IntoGrammarBinding<'a> for RuntimeConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::StaticOwned(Arc::new(self))
    }
}

impl<'a> IntoGrammarBinding<'a> for &'a RuntimeConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::StaticBorrowed(self)
    }
}

impl<'a> IntoGrammarBinding<'a> for Arc<RuntimeConstraint> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::StaticOwned(self)
    }
}

impl<'a> IntoGrammarBinding<'a> for DynamicConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::DynamicOwned(Arc::new(self))
    }
}

impl<'a> IntoGrammarBinding<'a> for &'a DynamicConstraint {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::DynamicBorrowed(self)
    }
}

impl<'a> IntoGrammarBinding<'a> for Arc<DynamicConstraint> {
    fn into_grammar_binding(self) -> GrammarBinding<'a> {
        GrammarBinding::DynamicOwned(self)
    }
}

impl<'a> ConstraintSpec<'a> {
    /// Start building a specification for `grammar` and this exact decoder target.
    pub fn builder(
        grammar: Grammar<'a>,
        vocab: &'a Vocab,
    ) -> Result<ConstraintSpecBuilder<'a>> {
        ConstraintSpecBuilder::new(grammar, vocab)
    }

    /// Compile this specification into a reusable static artifact.
    pub fn compile_static(&self, options: &CompileOptions) -> Result<RuntimeConstraint> {
        let _ = options;
        let token_bindings = self.token_binding_refs();
        if self.grammar_bindings.is_empty() {
            return compile_static_source(&self.grammar, self.vocab, &token_bindings);
        }

        let children = self.compile_children(options)?;
        let child_refs = children
            .iter()
            .map(|(name, child)| (name.as_str(), child.as_ref()))
            .collect::<Vec<_>>();
        let source = self.grammar.glrm_source().ok_or_else(|| {
            Error::Compilation("external grammar bindings require a GLRM grammar".to_owned())
        })?;
        RuntimeConstraint::from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
            source,
            &child_refs,
            self.vocab,
            &token_bindings,
            &[],
        )
    }

    /// Compile this specification into a lower-build-latency dynamic artifact.
    pub fn compile_dynamic(&self, options: &CompileOptions) -> Result<DynamicConstraint> {
        let token_bindings = self.token_binding_refs();
        if self.grammar_bindings.is_empty() {
            return compile_dynamic_source(&self.grammar, self.vocab, &token_bindings);
        }

        let children = self.compile_children(options)?;
        let child_refs = children
            .iter()
            .map(|(name, child)| (name.as_str(), child.as_ref()))
            .collect::<Vec<_>>();
        let source = self.grammar.glrm_source().ok_or_else(|| {
            Error::Compilation("external grammar bindings require a GLRM grammar".to_owned())
        })?;
        DynamicConstraint::from_glrm_grammar_with_subgrammars_and_bindings(
            source,
            &child_refs,
            self.vocab,
            &token_bindings,
        )
    }

    fn token_binding_refs(&self) -> Vec<(&str, &[u32])> {
        self.token_bindings
            .iter()
            .map(|(name, ids)| (name.as_str(), ids.as_slice()))
            .collect()
    }

    fn compile_children(
        &self,
        options: &CompileOptions,
    ) -> Result<Vec<(String, CompiledChild<'_>)>> {
        self.grammar_bindings
            .iter()
            .map(|(name, binding)| Ok((name.clone(), binding.compile(self.vocab, options)?)))
            .collect()
    }

    fn targets(&self, vocab: &Vocab) -> bool {
        self.vocab.entries_map() == vocab.entries_map()
    }
}

impl<'a> ConstraintSpecBuilder<'a> {
    fn new(grammar: Grammar<'a>, vocab: &'a Vocab) -> Result<Self> {
        let (grammar, source_bindings) = grammar.into_source_only_and_bindings();
        let declarations = match grammar.source {
            GrammarSource::Glrm(source) => crate::grammar::glrm::external_declarations(source)?,
            _ => crate::grammar::glrm::GlrmExternalDeclarations {
                token_names: Vec::new(),
                grammar_names: Vec::new(),
            },
        };
        let declared_grammars = declarations
            .grammar_names
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        for name in source_bindings.keys() {
            if !declared_grammars.contains(name) {
                return Err(Error::Compilation(format!(
                    "source-level binding was supplied for unknown external grammar {name:?}",
                )));
            }
        }
        Ok(Self {
            grammar,
            vocab,
            declared_tokens: declarations.token_names.into_iter().collect(),
            declared_grammars,
            token_bindings: BTreeMap::new(),
            grammar_bindings: source_bindings
                .into_iter()
                .map(|(name, grammar)| (name, GrammarBinding::Source(grammar)))
                .collect(),
        })
    }

    /// Bind an `extern token NAME;` declaration to exact decoder token IDs.
    pub fn bind_token(
        mut self,
        name: impl AsRef<str>,
        token_ids: impl IntoIterator<Item = u32>,
    ) -> Result<Self> {
        let name = name.as_ref();
        self.require_kind(name, ExternKind::Token)?;
        if self.token_bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "external token {name:?} was bound more than once",
            )));
        }
        let token_ids = token_ids.into_iter().collect::<Vec<_>>();
        if token_ids.is_empty() {
            return Err(Error::Compilation(format!(
                "external token {name:?} must bind at least one exact token ID",
            )));
        }
        let unique = token_ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != token_ids.len() {
            return Err(Error::Compilation(format!(
                "external token {name:?} contains a duplicate token ID",
            )));
        }
        self.token_bindings
            .insert(name.to_owned(), unique.into_iter().collect());
        Ok(self)
    }

    /// Bind an `extern grammar NAME;` declaration to a source, spec, or artifact.
    pub fn bind_grammar<T>(mut self, name: impl AsRef<str>, realization: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        let name = name.as_ref();
        self.require_kind(name, ExternKind::Grammar)?;
        if self.grammar_bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "external grammar {name:?} was bound more than once",
            )));
        }
        let mut realization = realization.into_grammar_binding();
        realization.bind_target(self.vocab, name)?;
        self.grammar_bindings.insert(name.to_owned(), realization);
        Ok(self)
    }

    /// Finish the immutable specification after verifying binding completeness.
    pub fn build(self) -> Result<ConstraintSpec<'a>> {
        if let Some(name) = self
            .declared_tokens
            .iter()
            .find(|name| !self.token_bindings.contains_key(*name))
        {
            return Err(Error::Compilation(format!(
                "GLRM declares external token {name:?}, but no exact-token binding was supplied",
            )));
        }
        if let Some(name) = self
            .declared_grammars
            .iter()
            .find(|name| !self.grammar_bindings.contains_key(*name))
        {
            return Err(Error::Compilation(format!(
                "GLRM declares external grammar {name:?}, but no realization was supplied",
            )));
        }
        Ok(ConstraintSpec {
            grammar: self.grammar,
            vocab: self.vocab,
            token_bindings: self.token_bindings,
            grammar_bindings: self.grammar_bindings,
        })
    }

    fn require_kind(&self, name: &str, expected: ExternKind) -> Result<()> {
        let correct = match expected {
            ExternKind::Token => self.declared_tokens.contains(name),
            ExternKind::Grammar => self.declared_grammars.contains(name),
        };
        if correct {
            return Ok(());
        }
        let wrong_kind = match expected {
            ExternKind::Token => self.declared_grammars.contains(name),
            ExternKind::Grammar => self.declared_tokens.contains(name),
        };
        if wrong_kind {
            return Err(Error::Compilation(format!(
                "external {name:?} has kind {}, not {}",
                expected.opposite_name(),
                expected.name(),
            )));
        }
        Err(Error::Compilation(format!(
            "no external {} named {name:?} is declared",
            expected.name(),
        )))
    }
}

#[derive(Clone, Copy)]
enum ExternKind {
    Token,
    Grammar,
}

impl ExternKind {
    fn name(self) -> &'static str {
        match self {
            Self::Token => "token",
            Self::Grammar => "grammar",
        }
    }

    fn opposite_name(self) -> &'static str {
        match self {
            Self::Token => "grammar",
            Self::Grammar => "token",
        }
    }
}

enum CompiledChild<'a> {
    Borrowed(&'a RuntimeConstraint),
    Owned(RuntimeConstraint),
}

impl CompiledChild<'_> {
    fn as_ref(&self) -> &RuntimeConstraint {
        match self {
            Self::Borrowed(constraint) => constraint,
            Self::Owned(constraint) => constraint,
        }
    }
}

fn static_constraint_targets(constraint: &RuntimeConstraint, vocab: &Vocab) -> bool {
    constraint.token_bytes.as_ref() == vocab.entries_map()
}

impl GrammarBinding<'_> {
    fn bind_target(&mut self, vocab: &Vocab, name: &str) -> Result<()> {
        let dynamic_materialization = match self {
            Self::DynamicBorrowed(constraint) => Some(constraint.composition_constraints(vocab)),
            Self::DynamicOwned(constraint) => Some(constraint.composition_constraints(vocab)),
            _ => None,
        };
        if let Some(materialized) = dynamic_materialization {
            let alternatives = materialized.map_err(|error| {
                Error::Compilation(format!(
                    "external grammar {name:?} cannot be used as a compiled child: {error}",
                ))
            })?;
            let constraint = collapse_alternatives(alternatives, vocab)?;
            *self = Self::StaticOwned(Arc::new(constraint));
            return Ok(());
        }

        let compatible = match self {
            Self::Source(_) => return Ok(()),
            Self::Spec(spec) => spec.targets(vocab),
            Self::StaticBorrowed(constraint) => static_constraint_targets(constraint, vocab),
            Self::StaticOwned(constraint) => static_constraint_targets(constraint, vocab),
            Self::DynamicBorrowed(_) | Self::DynamicOwned(_) => {
                unreachable!("dynamic binding handled above")
            }
        };
        if compatible {
            Ok(())
        } else {
            Err(Error::Compilation(format!(
                "external grammar {name:?} was built for an incompatible vocabulary",
            )))
        }
    }

    fn compile<'a>(
        &'a self,
        vocab: &Vocab,
        options: &CompileOptions,
    ) -> Result<CompiledChild<'a>> {
        match self {
            Self::Source(grammar) => {
                let spec = ConstraintSpec::builder(grammar.clone(), vocab)?.build()?;
                Ok(CompiledChild::Owned(spec.compile_static(options)?))
            }
            Self::Spec(spec) => Ok(CompiledChild::Owned(spec.compile_static(options)?)),
            Self::StaticBorrowed(constraint) => Ok(CompiledChild::Borrowed(constraint)),
            Self::StaticOwned(constraint) => Ok(CompiledChild::Borrowed(constraint)),
            Self::DynamicBorrowed(_) | Self::DynamicOwned(_) => {
                unreachable!("dynamic bindings are materialized at bind time")
            }
        }
    }
}

fn collapse_alternatives(
    mut alternatives: Vec<RuntimeConstraint>,
    vocab: &Vocab,
) -> Result<RuntimeConstraint> {
    if alternatives.len() == 1 {
        return Ok(alternatives.pop().expect("length checked"));
    }
    if alternatives.is_empty() {
        return Err(Error::Compilation(
            "external grammar realization has no alternatives".to_owned(),
        ));
    }

    let mut source = String::from("glrm 1;\nstart start;\n");
    for index in 0..alternatives.len() {
        source.push_str(&format!("extern grammar alternative_{index};\n"));
    }
    source.push_str("nt start = ");
    for index in 0..alternatives.len() {
        if index != 0 {
            source.push_str(" | ");
        }
        source.push_str(&format!("alternative_{index}"));
    }
    source.push_str(";\n");

    let names = (0..alternatives.len())
        .map(|index| format!("alternative_{index}"))
        .collect::<Vec<_>>();
    let children = names
        .iter()
        .zip(&alternatives)
        .map(|(name, child)| (name.as_str(), child))
        .collect::<Vec<_>>();
    RuntimeConstraint::from_glrm_grammar_with_subgrammars(&source, &children, vocab)
}

fn compile_static_source(
    grammar: &Grammar<'_>,
    vocab: &Vocab,
    token_bindings: &[(&str, &[u32])],
) -> Result<RuntimeConstraint> {
    match grammar.source {
        GrammarSource::Ebnf(source) => RuntimeConstraint::from_ebnf(source, vocab),
        GrammarSource::Lark(source) => RuntimeConstraint::from_lark(source, vocab),
        GrammarSource::JsonSchema(source) => RuntimeConstraint::from_json_schema(source, vocab),
        GrammarSource::Glrm(source) => RuntimeConstraint::from_glrm_grammar_with_bindings_and_end_tokens(
            source,
            vocab,
            token_bindings,
            &[],
        ),
    }
}

fn compile_dynamic_source(
    grammar: &Grammar<'_>,
    vocab: &Vocab,
    token_bindings: &[(&str, &[u32])],
) -> Result<DynamicConstraint> {
    match grammar.source {
        GrammarSource::Ebnf(source) => DynamicConstraint::from_ebnf(source, vocab),
        GrammarSource::Lark(source) => DynamicConstraint::from_lark(source, vocab),
        GrammarSource::JsonSchema(source) => DynamicConstraint::from_json_schema(source, vocab),
        GrammarSource::Glrm(source) => DynamicConstraint::from_glrm_grammar_with_bindings_and_end_tokens(
            source,
            vocab,
            token_bindings,
            &[],
        ),
    }
}

impl RuntimeConstraint {
    /// Compile a reusable static constraint without external declarations.
    pub fn compile(
        grammar: Grammar<'_>,
        vocab: &Vocab,
        options: &CompileOptions,
    ) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile_static(options)
    }
}

impl DynamicConstraint {
    /// Compile a dynamic constraint without external declarations.
    pub fn compile(
        grammar: Grammar<'_>,
        vocab: &Vocab,
        options: &CompileOptions,
    ) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile_dynamic(options)
    }
}
