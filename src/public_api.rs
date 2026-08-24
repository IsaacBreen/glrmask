use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::runtime::Constraint as RuntimeConstraint;
use crate::{DynamicConstraint, Error, Result, Vocab};

/// Grammar source with optional source-level subgrammar bindings.
///
/// [`Grammar::bind_grammar`] binds source children. Use [`ConstraintSpecBuilder`]
/// for exact tokens or compiled child constraints.
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

    /// Bind an `extern grammar NAME;` to another source grammar.
    ///
    /// Use [`ConstraintSpecBuilder::bind_grammar`] for compiled children.
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

/// A grammar, vocabulary, and complete set of extern bindings.
#[derive(Debug, Clone)]
pub struct ConstraintSpec<'a> {
    grammar: Grammar<'a>,
    vocab: &'a Vocab,
    token_bindings: BTreeMap<String, Vec<u32>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
    unbound_grammar_names: Vec<String>,
}

/// Builder for [`ConstraintSpec`].
#[derive(Debug)]
pub struct ConstraintSpecBuilder<'a> {
    grammar: Grammar<'a>,
    vocab: &'a Vocab,
    declared_tokens: BTreeSet<String>,
    declared_grammars: BTreeSet<String>,
    token_bindings: BTreeMap<String, Vec<u32>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
}

/// Internal input accepted by [`ConstraintSpecBuilder::bind_grammar`].
#[doc(hidden)]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum GrammarBinding<'a> {
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

/// Converts a supported child grammar into an internal binding.
#[doc(hidden)]
pub(crate) trait IntoGrammarBinding<'a> {
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
    /// Start a specification for `grammar` and `vocab`.
    pub fn builder(
        grammar: Grammar<'a>,
        vocab: &'a Vocab,
    ) -> Result<ConstraintSpecBuilder<'a>> {
        ConstraintSpecBuilder::new(grammar, vocab)
    }

    /// Compile this specification into a [`Constraint`](crate::Constraint).
    pub fn compile(&self) -> Result<RuntimeConstraint> {
        let mut constraint = self.compile_static_uncached()?;
        constraint.cache_serialized_artifact_for_save();
        Ok(constraint)
    }

    fn compile_static_uncached(&self) -> Result<RuntimeConstraint> {
        let token_bindings = self.token_binding_refs();
        if self.grammar_bindings.is_empty() {
            return compile_static_source(&self.grammar, self.vocab, &token_bindings);
        }

        let children = self.compile_children()?;
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

    /// Compile this specification into a [`DynamicConstraint`].
    pub fn compile_dynamic(&self) -> Result<DynamicConstraint> {
        if !self.unbound_grammar_names.is_empty() {
            return Err(Error::Compilation(format!(
                "dynamic compilation requires bindings for external grammars: {}",
                self.unbound_grammar_names.join(", "),
            )));
        }
        let token_bindings = self.token_binding_refs();
        if self.grammar_bindings.is_empty() {
            return compile_dynamic_source(&self.grammar, self.vocab, &token_bindings);
        }

        let children = self.compile_children()?;
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

    fn compile_children(&self) -> Result<Vec<(String, CompiledChild<'_>)>> {
        self.grammar_bindings
            .iter()
            .map(|(name, binding)| Ok((name.clone(), binding.compile(self.vocab)?)))
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

    /// Bind an `extern token NAME;` declaration to exact token IDs.
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

    /// Bind an `extern grammar NAME;` to a source, spec, or compiled constraint.
    #[allow(private_bounds)]
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

    /// Check that every extern is bound and finish the specification.
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
        let unbound_grammar_names = self
            .declared_grammars
            .iter()
            .filter(|name| !self.grammar_bindings.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        // A completely unbound grammar manifest is a reusable compiled parent:
        // each unresolved call remains unreachable until `Constraint::bind_grammar`
        // supplies its child. Preserve the existing all-at-once builder rule when
        // the caller has started binding grammars, so accidental partial specs are
        // still rejected.
        if !unbound_grammar_names.is_empty() && !self.grammar_bindings.is_empty() {
            return Err(Error::Compilation(format!(
                "GLRM declares external grammar {:?}, but no realization was supplied",
                unbound_grammar_names[0],
            )));
        }
        Ok(ConstraintSpec {
            grammar: self.grammar,
            vocab: self.vocab,
            token_bindings: self.token_bindings,
            grammar_bindings: self.grammar_bindings,
            unbound_grammar_names,
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

    fn compile<'a>(&'a self, vocab: &Vocab) -> Result<CompiledChild<'a>> {
        match self {
            Self::Source(grammar) => {
                let spec = ConstraintSpec::builder(grammar.clone(), vocab)?.build()?;
                Ok(CompiledChild::Owned(spec.compile_static_uncached()?))
            }
            Self::Spec(spec) => Ok(CompiledChild::Owned(spec.compile_static_uncached()?)),
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
        GrammarSource::Glrm(source) => RuntimeConstraint::from_glrm_grammar_with_unbound_subgrammars_bindings_and_end_tokens(
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
    /// Compile `grammar` into a [`Constraint`](crate::Constraint) for `vocab`.
    pub fn compile(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile()
    }

    /// Bind one unresolved `extern grammar NAME;` in this compiled constraint.
    ///
    /// The parent remains reusable. Loaded constraints keep their large parser
    /// DWA and pooled weights in packed form; the first bind lazily decodes only
    /// small composition metadata before linking. The supplied child must
    /// already be fully bound.
    pub fn bind_grammar(
        &mut self,
        name: impl AsRef<str>,
        child: &RuntimeConstraint,
        vocab: &Vocab,
    ) -> Result<Self> {
        use crate::compiler::constraint_compose::{
            CompiledSubgrammarInput, compose_constraints_owned_parent,
        };

        self.prepare_for_composition_internal(vocab)?;
        let name = name.as_ref();
        let placeholder_terminal = self
            .unbound_grammar_placeholders
            .get(name)
            .copied()
            .ok_or_else(|| {
                let available = self
                    .unbound_grammar_placeholders
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                Error::Compilation(if available.is_empty() {
                    format!("constraint has no unresolved external grammar named {name:?}")
                } else {
                    format!(
                        "constraint has no unresolved external grammar named {name:?}; available: {available}"
                    )
                })
            })?;

        let child_has_unbound = if child.deferred_composition_metadata_blob.is_some() {
            let mut metadata = child.clone();
            metadata
                .materialize_composition_metadata_for_compilation()
                .map_err(Error::Compilation)?;
            !metadata.unbound_grammar_placeholders.is_empty()
        } else {
            !child.unbound_grammar_placeholders.is_empty()
        };
        if child_has_unbound {
            return Err(Error::Compilation(
                "a compiled child with unresolved external grammars cannot yet be bound; bind the child first"
                    .to_owned(),
            ));
        }

        // Preparation materializes the compiler-side views once on the cached
        // parent. Cloning here preserves that reusable parent while the fast
        // linker consumes its fork. Loaded automata retain shared packed row
        // storage, so this fork is cheap in the cached-parent case.
        let mut parent = self.clone();
        parent.unbound_grammar_placeholders.remove(name);
        let remaining_parent_slots = parent.unbound_grammar_placeholders.clone();
        let input = CompiledSubgrammarInput {
            placeholder_terminal,
            additional_placeholder_terminals: &[],
            constraint: child,
        };
        let composition = compose_constraints_owned_parent(parent, &[input], vocab)
            .map_err(Error::Compilation)?;
        let parent_offset = composition.terminal_offsets[0];
        let mut result = composition.constraint;
        result.unbound_grammar_placeholders = remaining_parent_slots
            .into_iter()
            .map(|(slot, terminal)| (slot, parent_offset + terminal))
            .collect();
        Ok(result)
    }

    pub(crate) fn prepare_for_composition_internal(&mut self, vocab: &Vocab) -> Result<()> {
        self.bind_vocab_exact(vocab).map_err(Error::Compilation)?;
        // Composition metadata is small and compiler-facing. Keep the large
        // parser DWA and pooled non-DWA weights in their packed runtime forms;
        // the segmented/two-DWA linker consumes them through runtime views.
        // Legacy flattened linker paths materialize them on demand.
        self.materialize_composition_metadata_for_compilation()
            .map_err(Error::Compilation)?;
        Ok(())
    }

}

impl DynamicConstraint {
    /// Compile `grammar` into a [`DynamicConstraint`] for `vocab`.
    pub fn compile(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile_dynamic()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b">".to_vec()),
            (2, b"[".to_vec()),
            (3, b"]".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b"x".to_vec()),
            (7, b"<a>".to_vec()),
            (8, b"<b>".to_vec()),
            (9, b"<[a]>".to_vec()),
        ])
    }

    fn accepts(constraint: &RuntimeConstraint, bytes: &[u8]) -> bool {
        let mut state = constraint.start();
        state.commit_bytes(bytes).is_ok() && state.is_accepting()
    }

    #[test]
    fn compiled_parent_can_bind_external_grammar_after_compile_and_load() {
        let vocab = vocab();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    extern grammar payload;
                    start document;
                    nt document = "x" | "<" payload ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        assert_eq!(
            parent.unbound_grammar_placeholders.keys().collect::<Vec<_>>(),
            vec![&"payload".to_owned()],
        );
        assert!(accepts(&parent, b"x"));
        assert!(!accepts(&parent, b"<a>"));

        let child_a = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"a\";"),
            &vocab,
        )
        .unwrap();
        let child_b = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"b\";"),
            &vocab,
        )
        .unwrap();

        let with_a = parent.bind_grammar("payload", &child_a, &vocab).unwrap();
        let with_b = parent.bind_grammar("payload", &child_b, &vocab).unwrap();
        assert!(accepts(&with_a, b"x"));
        assert!(accepts(&with_a, b"<a>"));
        assert!(!accepts(&with_a, b"<b>"));
        assert!(accepts(&with_b, b"<b>"));
        assert!(!accepts(&with_b, b"<a>"));
        assert!(parent.unbound_grammar_placeholders.contains_key("payload"));

        let saved = parent.save();
        let mut loaded = RuntimeConstraint::load(&saved).unwrap();
        let loaded_with_a = loaded.bind_grammar("payload", &child_a, &vocab).unwrap();
        assert!(accepts(&loaded_with_a, b"<a>"));
        assert!(!accepts(&loaded_with_a, b"<b>"));
    }

    #[test]
    fn compiled_parent_can_fill_multiple_slots_incrementally() {
        let vocab = vocab();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    extern grammar left;
                    extern grammar right;
                    start document;
                    nt document = "<" left right ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let a = RuntimeConstraint::compile(
            Grammar::glrm(r#"glrm 1; start value; nt value = "a";"#),
            &vocab,
        )
        .unwrap();
        let b = RuntimeConstraint::compile(
            Grammar::glrm(r#"glrm 1; start value; nt value = "b";"#),
            &vocab,
        )
        .unwrap();
        let mut half = parent.bind_grammar("left", &a, &vocab).unwrap();
        assert!(half.unbound_grammar_placeholders.contains_key("right"));
        let full = half.bind_grammar("right", &b, &vocab).unwrap();
        assert!(full.unbound_grammar_placeholders.is_empty());
        assert!(accepts(&full, b"<ab>"));

        let unresolved_child = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar leaf; start value; nt value = leaf;",
            ),
            &vocab,
        )
        .unwrap();
        let err = parent
            .bind_grammar("left", &unresolved_child, &vocab)
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("a compiled child with unresolved external grammars cannot yet be bound"));
    }

    #[test]
    fn dynamic_compile_rejects_unbound_external_grammar() {
        let vocab = vocab();
        let err = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar payload; start document; nt document = payload;",
            ),
            &vocab,
        )
        .unwrap_err();
        assert!(err.to_string().contains("dynamic compilation requires bindings"));
    }
}
