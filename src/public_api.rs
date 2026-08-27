use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend,
    compose_constraints_owned_parent_segmented_shared,
};
use crate::runtime::Constraint as RuntimeConstraint;
use crate::{BoundaryTriggerDetail, DynamicConstraint, Error, Result, Vocab};

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
    boundary_trigger_detail: BoundaryTriggerDetail,
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
    boundary_trigger_detail: BoundaryTriggerDetail,
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
        let mut constraint = self.compile_static_with_trigger_uncached()?;
        // A constraint with unresolved external grammars is explicitly a
        // reusable late-bind parent.  Cache its exact tokenizer-reset
        // terminal -> model-token relation now, while compilation already owns
        // the vocabulary, instead of rescanning the vocabulary on every later
        // bind.  The cache is part of composition metadata and therefore
        // survives ordinary save/load.
        if !constraint.late_grammar_slots.is_empty() {
            constraint.ensure_composition_reset_tokens_by_terminal();
        }
        // Encoding a segmented dynamic leaf into the ordinary static artifact
        // format must not turn compilation itself into an eager dynamic -> DWA
        // materialization. Preserve the live hybrid and defer that work until
        // an explicit `save()` call. Purely static results keep the cheap
        // cached-save behavior.
        if !retains_dynamic_component(&constraint) {
            constraint.cache_serialized_artifact_for_save();
        }
        Ok(constraint)
    }

    fn compile_static_with_trigger_uncached(&self) -> Result<RuntimeConstraint> {
        let mut constraint = self.compile_static_uncached()?;
        constraint
            .build_boundary_trigger(self.boundary_trigger_detail)
            .map_err(Error::Compilation)?;
        Ok(constraint)
    }

    fn compile_static_uncached(&self) -> Result<RuntimeConstraint> {
        let token_bindings = self.token_binding_refs();
        if self.grammar_bindings.is_empty() {
            if let Some(source) = self.grammar.glrm_source() {
                return RuntimeConstraint::from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
                    source,
                    &[],
                    self.vocab,
                    &token_bindings,
                    &[],
                );
            }
            return compile_static_source(&self.grammar, self.vocab, &token_bindings);
        }

        let source = self.grammar.glrm_source().ok_or_else(|| {
            Error::Compilation("external grammar bindings require a GLRM grammar".to_owned())
        })?;
        let parent = RuntimeConstraint::from_glrm_grammar_with_subgrammars_bindings_and_end_tokens(
            source,
            &[],
            self.vocab,
            &token_bindings,
            &[],
        )?;
        let children = self.compile_children(ChildCompileMode::Static)?;
        let children = prepare_compiled_children(
            children,
            self.vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )?;
        compose_named_children(
            parent,
            &children,
            self.vocab,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
    }

    /// Compile this specification into a [`DynamicConstraint`].
    pub fn compile_dynamic(&self) -> Result<DynamicConstraint> {
        let mut constraint = self.compile_dynamic_uncached()?;
        for component in constraint.constraints_mut() {
            component
                .build_boundary_trigger(self.boundary_trigger_detail)
                .map_err(Error::Compilation)?;
        }
        Ok(constraint)
    }

    fn compile_dynamic_uncached(&self) -> Result<DynamicConstraint> {
        let token_bindings = self.token_binding_refs();
        if self.grammar_bindings.is_empty() {
            if let Some(source) = self.grammar.glrm_source() {
                return DynamicConstraint::from_glrm_grammar_with_subgrammars_and_bindings(
                    source,
                    &[],
                    self.vocab,
                    &token_bindings,
                );
            }
            return compile_dynamic_source(&self.grammar, self.vocab, &token_bindings);
        }

        let source = self.grammar.glrm_source().ok_or_else(|| {
            Error::Compilation("external grammar bindings require a GLRM grammar".to_owned())
        })?;
        let parents = DynamicConstraint::from_glrm_grammar_with_subgrammars_and_bindings(
            source,
            &[],
            self.vocab,
            &token_bindings,
        )?;
        let children = self.compile_children(ChildCompileMode::Dynamic)?;
        let children = prepare_compiled_children(
            children,
            self.vocab,
            SegmentedBoundaryBackend::Dynamic,
        )?;
        let alternatives = parents
            .clone_constraints()
            .into_iter()
            .map(|parent| {
                compose_named_children(
                    parent,
                    &children,
                    self.vocab,
                    SegmentedBoundaryBackend::Dynamic,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DynamicConstraint::from_constraints(alternatives))
    }

    fn token_binding_refs(&self) -> Vec<(&str, &[u32])> {
        self.token_bindings
            .iter()
            .map(|(name, ids)| (name.as_str(), ids.as_slice()))
            .collect()
    }

    fn compile_children(
        &self,
        mode: ChildCompileMode,
    ) -> Result<Vec<(String, CompiledChild<'_>)>> {
        self.grammar_bindings
            .iter()
            .map(|(name, binding)| Ok((name.clone(), binding.compile(self.vocab, mode)?)))
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
            boundary_trigger_detail: BoundaryTriggerDetail::None,
        })
    }

    /// Request reusable dynamic-boundary trigger metadata for this component.
    ///
    /// The default is [`BoundaryTriggerDetail::None`], which adds no trigger
    /// construction cost. This setting is independent of static vs dynamic
    /// ordinary masking.
    pub fn boundary_trigger_detail(mut self, detail: BoundaryTriggerDetail) -> Self {
        self.boundary_trigger_detail = detail;
        self
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

    /// Check that every exact-token extern is bound and finish the specification.
    ///
    /// External grammars may remain unresolved. Their names are retained in
    /// the compiled constraint and can be supplied later with
    /// [`RuntimeConstraint::bind_grammar`] or [`DynamicConstraint::bind_grammar`].
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
        Ok(ConstraintSpec {
            grammar: self.grammar,
            vocab: self.vocab,
            token_bindings: self.token_bindings,
            grammar_bindings: self.grammar_bindings,
            unbound_grammar_names,
            boundary_trigger_detail: self.boundary_trigger_detail,
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

#[derive(Clone, Copy)]
enum ChildCompileMode {
    Static,
    Dynamic,
}

enum CompiledChild<'a> {
    StaticBorrowed(&'a RuntimeConstraint),
    StaticOwned(RuntimeConstraint),
    DynamicBorrowed(&'a DynamicConstraint),
    DynamicOwned(DynamicConstraint),
}

impl CompiledChild<'_> {
    fn into_constraints(self) -> Vec<RuntimeConstraint> {
        match self {
            Self::StaticBorrowed(constraint) => vec![constraint.clone()],
            Self::StaticOwned(constraint) => vec![constraint],
            Self::DynamicBorrowed(constraint) => constraint.clone_constraints(),
            Self::DynamicOwned(constraint) => constraint.clone_constraints(),
        }
    }
}

fn static_constraint_targets(constraint: &RuntimeConstraint, vocab: &Vocab) -> bool {
    constraint.token_bytes_match_vocab(vocab)
}

impl GrammarBinding<'_> {
    fn bind_target(&mut self, vocab: &Vocab, name: &str) -> Result<()> {
        let compatible = match self {
            Self::Source(_) => return Ok(()),
            Self::Spec(spec) => spec.targets(vocab),
            Self::StaticBorrowed(constraint) => static_constraint_targets(constraint, vocab),
            Self::StaticOwned(constraint) => static_constraint_targets(constraint, vocab),
            Self::DynamicBorrowed(constraint) => constraint.targets_vocab(vocab),
            Self::DynamicOwned(constraint) => constraint.targets_vocab(vocab),
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
        mode: ChildCompileMode,
    ) -> Result<CompiledChild<'a>> {
        match self {
            Self::Source(grammar) => {
                let spec = ConstraintSpec::builder(grammar.clone(), vocab)?.build()?;
                match mode {
                    ChildCompileMode::Static => {
                        Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_uncached()?))
                    }
                    ChildCompileMode::Dynamic => {
                        Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                    }
                }
            }
            Self::Spec(spec) => match mode {
                ChildCompileMode::Static => {
                    Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_uncached()?))
                }
                ChildCompileMode::Dynamic => {
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                }
            },
            Self::StaticBorrowed(constraint) => Ok(CompiledChild::StaticBorrowed(constraint)),
            Self::StaticOwned(constraint) => Ok(CompiledChild::StaticBorrowed(constraint)),
            Self::DynamicBorrowed(constraint) => Ok(CompiledChild::DynamicBorrowed(constraint)),
            Self::DynamicOwned(constraint) => Ok(CompiledChild::DynamicBorrowed(constraint)),
        }
    }

    /// One-shot late binding can consume the user's binding. Preserve that
    /// ownership instead of immediately downgrading an owned/Arc constraint to
    /// a borrowed child and deep-cloning it again during preparation.
    fn into_compiled<'a>(
        self,
        vocab: &Vocab,
        mode: ChildCompileMode,
    ) -> Result<CompiledChild<'a>>
    where
        Self: 'a,
    {
        match self {
            Self::Source(grammar) => {
                let spec = ConstraintSpec::builder(grammar, vocab)?.build()?;
                match mode {
                    ChildCompileMode::Static => {
                        Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_uncached()?))
                    }
                    ChildCompileMode::Dynamic => {
                        Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                    }
                }
            }
            Self::Spec(spec) => match mode {
                ChildCompileMode::Static => {
                    Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_uncached()?))
                }
                ChildCompileMode::Dynamic => {
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                }
            },
            Self::StaticBorrowed(constraint) => Ok(CompiledChild::StaticBorrowed(constraint)),
            Self::StaticOwned(constraint) => Ok(CompiledChild::StaticOwned(
                Arc::try_unwrap(constraint).unwrap_or_else(|shared| (*shared).clone()),
            )),
            Self::DynamicBorrowed(constraint) => Ok(CompiledChild::DynamicBorrowed(constraint)),
            Self::DynamicOwned(constraint) => Ok(CompiledChild::DynamicOwned(
                Arc::try_unwrap(constraint).unwrap_or_else(|shared| (*shared).clone()),
            )),
        }
    }
}

fn prepare_compiled_children(
    children: Vec<(String, CompiledChild<'_>)>,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Vec<(String, Arc<RuntimeConstraint>)>> {
    children
        .into_iter()
        .map(|(name, child)| {
            let mut constraint = collapse_dynamic_alternatives(
                child.into_constraints(),
                vocab,
                boundary_backend,
            )?;
            // `compose_named_children` always uses the segmented linker. Decode
            // composition-only metadata on this first owned child copy so the
            // linker does not clone the whole compiled child a second time
            // merely to materialize the same cold metadata.
            match boundary_backend {
                SegmentedBoundaryBackend::StaticParserDwa => constraint
                    .materialize_composition_metadata_for_compilation()
                    .map_err(Error::Compilation)?,
                SegmentedBoundaryBackend::Dynamic => constraint
                    .materialize_composition_link_metadata_for_compilation()
                    .map_err(Error::Compilation)?,
            }
            Ok((name, Arc::new(constraint)))
        })
        .collect()
}

fn collapse_dynamic_alternatives(
    mut alternatives: Vec<RuntimeConstraint>,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
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
        .cloned()
        .zip(alternatives.into_iter().map(Arc::new))
        .collect::<Vec<_>>();
    let parent = RuntimeConstraint::from_glrm_grammar_with_subgrammars(&source, &[], vocab)?;
    let mut union = compose_named_children(parent, &children, vocab, boundary_backend)?;
    for slot in &mut union.late_grammar_slots {
        if let Some((alternative, nested)) = slot.name.split_once('.')
            && alternative.starts_with("alternative_")
        {
            slot.name = nested.to_owned();
        }
    }
    Ok(union)
}

fn compose_named_children(
    parent: RuntimeConstraint,
    children: &[(String, Arc<RuntimeConstraint>)],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<RuntimeConstraint> {
    let bound_names = children
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<BTreeSet<_>>();
    let mut present = Vec::new();
    let mut matching_terminals = Vec::new();
    for (child_index, (name, _)) in children.iter().enumerate() {
        let terminals = parent
            .late_grammar_slots
            .iter()
            .filter(|slot| slot.name == *name)
            .map(|slot| slot.terminal_id)
            .collect::<Vec<_>>();
        if !terminals.is_empty() {
            present.push(child_index);
            matching_terminals.push(terminals);
        }
    }
    if present.is_empty() {
        return Ok(parent);
    }

    let remaining_parent_slots = parent
        .late_grammar_slots
        .iter()
        .filter(|slot| !bound_names.contains(slot.name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let inputs = present
        .iter()
        .zip(&matching_terminals)
        .map(|(&child_index, terminals)| CompiledSubgrammarInput {
            placeholder_terminal: terminals[0],
            additional_placeholder_terminals: &terminals[1..],
            constraint: children[child_index].1.as_ref(),
        })
        .collect::<Vec<_>>();
    let shared_children = present
        .iter()
        .map(|&child_index| Arc::clone(&children[child_index].1))
        .collect::<Vec<_>>();
    let mut composition = compose_constraints_owned_parent_segmented_shared(
        parent,
        &inputs,
        &shared_children,
        vocab,
        boundary_backend,
    )
    .map_err(Error::Compilation)?;
    composition.constraint.late_grammar_slots = remaining_parent_slots;
    for (component_index, &child_index) in present.iter().enumerate() {
        let terminal_offset = composition.terminal_offsets[component_index + 1];
        let (binding_name, child) = &children[child_index];
        composition.constraint.late_grammar_slots.extend(
            child.late_grammar_slots.iter().map(|slot| crate::runtime::LateGrammarSlot {
                name: format!("{binding_name}.{}", slot.name),
                terminal_id: terminal_offset + slot.terminal_id,
            }),
        );
    }
    if composition
        .constraint
        .sanitize_late_grammar_placeholder_token_domain()
    {
        composition.constraint.rebuild_runtime_caches();
    }
    Ok(composition.constraint)
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

    /// Bind one retained `extern grammar` slot using a fully compiled boundary.
    ///
    /// Other named slots remain unresolved and may be bound by later calls.
    /// The compiled parent and the supplied child's masking backends are reused;
    /// only their cross-boundary behavior is compiled here.
    #[allow(private_bounds)]
    pub fn bind_grammar<'a, T>(&self, name: impl AsRef<str>, child: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_static_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
    }

    /// Bind one retained `extern grammar` slot using the dynamic boundary walker.
    ///
    /// Component-local masking remains independently static or dynamic according
    /// to the backend with which each component was supplied.
    #[allow(private_bounds)]
    pub fn bind_grammar_dynamic_boundary<'a, T>(
        &self,
        name: impl AsRef<str>,
        child: T,
    ) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_static_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::Dynamic,
        )
    }

    /// Private compatibility hook for internal benchmark/composition callers.
    /// Public late binding no longer requires the caller to resupply `Vocab`,
    /// but old internal cached-parent probes still use this to eagerly prepare
    /// only the small compiler-facing composition metadata while leaving the
    /// packed runtime automata untouched.
    pub(crate) fn prepare_for_composition_internal(&mut self, vocab: &Vocab) -> Result<()> {
        self.bind_vocab_exact(vocab).map_err(Error::Compilation)?;
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

    /// Bind one retained `extern grammar` slot using a fully compiled boundary.
    ///
    /// Dynamic parent alternatives and dynamic component-local masking remain
    /// dynamic; the boundary choice is independent of component backends.
    #[allow(private_bounds)]
    pub fn bind_grammar<'a, T>(&self, name: impl AsRef<str>, child: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_dynamic_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::StaticParserDwa,
        )
    }

    /// Bind one retained `extern grammar` slot using the dynamic boundary walker.
    #[allow(private_bounds)]
    pub fn bind_grammar_dynamic_boundary<'a, T>(
        &self,
        name: impl AsRef<str>,
        child: T,
    ) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_dynamic_parent_grammar(
            self,
            name.as_ref(),
            child,
            SegmentedBoundaryBackend::Dynamic,
        )
    }
}

fn constraint_vocab(constraint: &RuntimeConstraint) -> Vocab {
    constraint
        .late_bind_vocab
        .get_or_init(|| {
            Vocab::new(
                constraint
                    .token_bytes_iter()
                    .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
                    .collect(),
            )
        })
        .clone()
}

fn retains_dynamic_component(constraint: &RuntimeConstraint) -> bool {
    if constraint.uses_dynamic_runtime() {
        return true;
    }
    constraint
        .static_dynamic_overlay
        .as_ref()
        .is_some_and(|overlay| {
            overlay
                .segmented_parser_components
                .iter()
                .any(|component| retains_dynamic_component(component.constraint.as_ref()))
        })
}

fn require_late_grammar_slot(
    parents: &[RuntimeConstraint],
    name: &str,
) -> Result<()> {
    if parents.iter().any(|parent| {
        parent
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == name)
    }) {
        return Ok(());
    }
    Err(Error::Compilation(format!(
        "compiled constraint has no unresolved external grammar named {name:?}",
    )))
}

fn prepare_late_child<'a, T>(
    name: &str,
    child: T,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Vec<(String, Arc<RuntimeConstraint>)>>
where
    T: IntoGrammarBinding<'a>,
{
    let mut binding = child.into_grammar_binding();
    binding.bind_target(vocab, name)?;
    let mode = match boundary_backend {
        SegmentedBoundaryBackend::StaticParserDwa => ChildCompileMode::Static,
        SegmentedBoundaryBackend::Dynamic => ChildCompileMode::Dynamic,
    };
    let compiled = binding.into_compiled(vocab, mode)?;
    prepare_compiled_children(
        vec![(name.to_owned(), compiled)],
        vocab,
        boundary_backend,
    )
}

fn bind_static_parent_grammar<'a, T>(
    parent: &RuntimeConstraint,
    name: &str,
    child: T,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<RuntimeConstraint>
where
    T: IntoGrammarBinding<'a>,
{
    let profile = std::env::var_os("GLRMASK_PROFILE_PUBLIC_BIND").is_some();
    let total_started = Instant::now();
    let phase = Instant::now();
    require_late_grammar_slot(std::slice::from_ref(parent), name)?;
    let require_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let vocab = constraint_vocab(parent);
    let vocab_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let children = prepare_late_child(name, child, &vocab, boundary_backend)?;
    let child_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let parent = parent.clone();
    let parent_clone_ms = phase.elapsed().as_secs_f64() * 1000.0;

    let phase = Instant::now();
    let result = compose_named_children(parent, &children, &vocab, boundary_backend)?;
    let compose_ms = phase.elapsed().as_secs_f64() * 1000.0;

    // Late binding is a construction operation, not an implicit persistence
    // request. Serializing an all-static result here made bind latency include
    // the full first-save cost even when the caller never saves the result.
    // `Constraint::save()` still installs/reuses the serialized artifact cache
    // when persistence is actually requested; ordinary `Constraint::compile()`
    // keeps its existing eager first-save priming policy.
    let cache_save_ms = 0.0;
    if profile {
        eprintln!(
            "[glrmask/profile][public_bind_static_parent] require_ms={require_ms:.3} vocab_ms={vocab_ms:.3} child_ms={child_ms:.3} parent_clone_ms={parent_clone_ms:.3} compose_ms={compose_ms:.3} cache_save_ms={cache_save_ms:.3} total_ms={:.3}",
            total_started.elapsed().as_secs_f64() * 1000.0,
        );
    }
    Ok(result)
}

fn bind_dynamic_parent_grammar<'a, T>(
    parent: &DynamicConstraint,
    name: &str,
    child: T,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<DynamicConstraint>
where
    T: IntoGrammarBinding<'a>,
{
    let parents = parent.clone_constraints();
    require_late_grammar_slot(&parents, name)?;
    let first = parents.first().ok_or_else(|| {
        Error::Compilation("dynamic parent has no alternatives".to_owned())
    })?;
    let vocab = constraint_vocab(first);
    if !parents
        .iter()
        .all(|alternative| static_constraint_targets(alternative, &vocab))
    {
        return Err(Error::Compilation(
            "dynamic parent alternatives target incompatible vocabularies".to_owned(),
        ));
    }
    let children = prepare_late_child(name, child, &vocab, boundary_backend)?;
    let alternatives = parents
        .into_iter()
        .map(|alternative| {
            compose_named_children(alternative, &children, &vocab, boundary_backend)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DynamicConstraint::from_constraints(alternatives))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::Lexer;

    fn component_backend_flags(constraint: &RuntimeConstraint) -> Vec<bool> {
        constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime")
            .segmented_parser_components
            .iter()
            .map(|component| component.constraint.uses_dynamic_runtime())
            .collect()
    }

    fn leaf_backend_flags(constraint: &RuntimeConstraint) -> Vec<bool> {
        match constraint.static_dynamic_overlay.as_ref() {
            Some(overlay) if !overlay.segmented_parser_components.is_empty() => overlay
                .segmented_parser_components
                .iter()
                .flat_map(|component| leaf_backend_flags(&component.constraint))
                .collect(),
            _ => vec![constraint.uses_dynamic_runtime()],
        }
    }

    fn assert_recursive_compiler_views_detached(constraint: &RuntimeConstraint) {
        if !constraint.uses_compact_segmented_parser_runtime() {
            return;
        }
        assert_eq!(
            constraint.table.num_states, 0,
            "recursive coordinator retained a flattened LR state machine",
        );
        assert!(constraint.table.action.is_empty() && constraint.table.goto.is_empty());
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(
            overlay.recursive_compiler_table.get().is_some(),
            "recursive coordinator lost its lazy compiler table",
        );
        for component in &overlay.segmented_parser_components {
            assert_recursive_compiler_views_detached(&component.constraint);
        }
    }

    fn poison_materialized_outer_table(constraint: &mut RuntimeConstraint) {
        constraint.recursive_parser_layout().unwrap().unwrap();
        constraint.table.action.clear();
        constraint.table.goto.clear();
        constraint.table.advance.clear();
        constraint.table.unconditional_advance.clear();
        constraint.table.rules.clear();
        constraint.table.forwarded_shifts.clear();
        constraint.table.control_terminals.clear();
        constraint.table.skip_terminals.clear();
        constraint.table.guarded_shift_index.clear();
        constraint.table.direct_regular_wide_frontiers.clear();
        constraint.table.num_states = 0;
        constraint.table.num_terminals = 0;
        constraint.table.num_rules = 0;
    }

    fn assert_static_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime");
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert!(
            !overlay.segmented_boundary_shards.is_empty(),
            "static boundary policy must publish component-owned shards",
        );
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
        )));
    }

    fn assert_dynamic_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("bound constraint must use the explicit segmented runtime");
        assert!(overlay.segmented_boundary_parser.is_none());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
        )));
    }

    #[test]
    fn segmented_composition_preserves_supplied_component_backends() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            // Forces B to carry a strictly internal parent -> child crossing.
            (2, b"xy".to_vec()),
        ]);
        let parent = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        );
        let static_child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let dynamic_child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        let static_with_dynamic = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &dynamic_child)
            .unwrap()
            .build()
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(component_backend_flags(&static_with_dynamic), vec![false, true]);
        assert!(
            static_with_dynamic.serialized_artifact_cache.is_none(),
            "compiling a hybrid must not eagerly staticify its dynamic leaf for serialization",
        );
        assert_static_boundary(&static_with_dynamic);

        let dynamic_with_static = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &static_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        for alternative in dynamic_with_static.clone_constraints() {
            assert_eq!(component_backend_flags(&alternative), vec![true, false]);
            let overlay = alternative.static_dynamic_overlay.as_ref().unwrap();
            assert!(overlay.segmented_boundary_parser.is_none());
            assert!(overlay.segmented_boundary_terminal_trie.is_none());
            assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
                shard.backend,
                crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
            )));
        }

        let dynamic_with_dynamic = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", &dynamic_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        for alternative in dynamic_with_dynamic.clone_constraints() {
            assert_eq!(component_backend_flags(&alternative), vec![true, true]);
        }
    }

    #[test]
    fn compiled_parent_late_binding_preserves_leaf_backends_and_boundary_choice() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"z".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let one_slot = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        );
        let static_parent = RuntimeConstraint::compile(one_slot.clone(), &vocab).unwrap();
        let dynamic_parent = DynamicConstraint::compile(one_slot, &vocab).unwrap();
        let static_child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let dynamic_child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        let bound = static_parent.bind_grammar("child", &static_child).unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, false]);
        assert_static_boundary(&bound);
        assert!(
            bound.serialized_artifact_cache.is_none(),
            "late binding must not pay first-save serialization eagerly",
        );

        let bound = static_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, false]);
        assert_dynamic_boundary(&bound);

        let bound = static_parent.bind_grammar("child", &dynamic_child).unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, true]);
        assert_static_boundary(&bound);
        assert!(
            bound.serialized_artifact_cache.is_none(),
            "late binding a dynamic child must not trigger serialization-time staticification",
        );

        let bound = static_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap();
        assert_eq!(leaf_backend_flags(&bound), vec![false, true]);
        assert_dynamic_boundary(&bound);

        let bound = dynamic_parent.bind_grammar("child", &static_child).unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false]);
            assert_static_boundary(&alternative);
        }

        let bound = dynamic_parent
            .bind_grammar_dynamic_boundary("child", &static_child)
            .unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false]);
            assert_dynamic_boundary(&alternative);
        }

        let bound = dynamic_parent.bind_grammar("child", &dynamic_child).unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, true]);
            assert_static_boundary(&alternative);
        }

        let bound = dynamic_parent
            .bind_grammar_dynamic_boundary("child", &dynamic_child)
            .unwrap();
        for alternative in bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, true]);
            assert_dynamic_boundary(&alternative);
        }

        let two_slots = Grammar::glrm(
            "glrm 1; start start; extern grammar left; extern grammar right; \
             nt start = \"x\" left right;",
        );
        let open = RuntimeConstraint::compile(two_slots, &vocab).unwrap();
        let partially_bound = open.bind_grammar("left", &static_child).unwrap();
        assert!(
            partially_bound.late_bind_vocab.get().is_some(),
            "partially bound constraints should carry the shared vocabulary cache into the next bind",
        );
        assert!(partially_bound
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "right"));
        let fully_bound = partially_bound
            .bind_grammar(
                "right",
                DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_static_boundary(&fully_bound);

        let fully_bound = partially_bound
            .bind_grammar_dynamic_boundary(
                "right",
                DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_dynamic_boundary(&fully_bound);

        let open = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar left; extern grammar right; \
                 nt start = \"x\" left right;",
            ),
            &vocab,
        )
        .unwrap();
        let partially_bound = open.bind_grammar("left", &static_child).unwrap();
        let dynamic_right =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap();
        let fully_bound = partially_bound.bind_grammar("right", &dynamic_right).unwrap();
        for alternative in fully_bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false, true]);
            assert_static_boundary(&alternative);
        }
        let fully_bound = partially_bound
            .bind_grammar_dynamic_boundary("right", &dynamic_right)
            .unwrap();
        for alternative in fully_bound.clone_constraints() {
            assert_eq!(leaf_backend_flags(&alternative), vec![true, false, true]);
            assert_dynamic_boundary(&alternative);
        }
    }

    #[test]
    fn all_static_late_bound_segmented_roundtrip_preserves_fused_boundary_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        for dynamic_boundary in [false, true] {
            let bound = if dynamic_boundary {
                parent
                    .bind_grammar_dynamic_boundary("child", child.clone())
                    .unwrap()
            } else {
                parent.bind_grammar("child", child.clone()).unwrap()
            };
            let live_mask = bound.start().mask();
            assert_ne!(
                live_mask[0] & (1 << 2),
                0,
                "fused parent/child token must be live before serialization",
            );

            let loaded = RuntimeConstraint::load(bound.save()).unwrap();
            let overlay = loaded
                .static_dynamic_overlay
                .as_ref()
                .expect("round-tripped segmented runtime must retain A/B metadata");
            assert!(
                !overlay.segmented_static_baseline,
                "new segmented saves must not synthesize a flattened static parser baseline",
            );
            assert!(
                overlay.segmented_parser_components.len() == 2,
                "new segmented saves must retain the parent and child A components exactly",
            );
            assert!(overlay
                .segmented_parser_components
                .iter()
                .all(|component| component.root_disallowed_terminal.is_none()));
            assert_eq!(
                loaded.start().mask(),
                live_mask,
                "all-static segmented A+B changed across save/load (dynamic_boundary={dynamic_boundary})",
            );

            let mut state = loaded.start();
            state.commit_token(2).unwrap();
            assert!(state.is_accepting());

            let mut live_state = bound.start();
            let mut loaded_state = loaded.start();
            for token in [0, 1] {
                assert_eq!(
                    loaded_state.mask(),
                    live_state.mask(),
                    "all-static segmented state changed after round-trip before token {token} (dynamic_boundary={dynamic_boundary})",
                );
                live_state.commit_token(token).unwrap();
                loaded_state.commit_token(token).unwrap();
            }
            assert_eq!(loaded_state.is_accepting(), live_state.is_accepting());
            assert!(loaded_state.is_accepting());
        }
    }

    #[test]
    fn dynamic_boundary_token_triggers_survive_component_roundtrip() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"z".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y""#),
            &vocab,
        )
        .unwrap();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let mut child = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "y""#),
            &vocab,
        )
        .unwrap();

        parent.build_boundary_token_trigger().unwrap();
        child.build_boundary_token_trigger().unwrap();
        assert!(parent
            .boundary_trigger
            .token_summary()
            .expect("parent token trigger must be built")
            .contains(&2));

        let loaded_parent = RuntimeConstraint::load(parent.save()).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save()).unwrap();
        let bound = loaded_parent
            .bind_grammar_dynamic_boundary("child", loaded_child)
            .unwrap();
        assert_eq!(bound.start().mask(), reference.start().mask());

        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("dynamic composition must retain component runtime metadata");
        let parent_tokens = overlay.segmented_parser_components[0]
            .constraint
            .boundary_trigger
            .token_summary()
            .expect("parent Tokens trigger must survive save/load");
        let child_tokens = overlay.segmented_parser_components[1]
            .constraint
            .boundary_trigger
            .token_summary()
            .expect("child Tokens trigger must survive save/load");
        assert!(parent_tokens.contains(&2));
        assert!(child_tokens.iter().all(|&token| token < 4));
    }

    #[test]
    fn exact_boundary_triggers_roundtrip_and_gate_entry_and_finish() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y" "z""#),
            &vocab,
        )
        .unwrap();
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child \"z\";",
            ),
            &vocab,
        )
        .unwrap();
        let mut child = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "y""#),
            &vocab,
        )
        .unwrap();

        parent.build_exact_boundary_trigger().unwrap();
        child.build_exact_boundary_trigger().unwrap();
        assert!(matches!(
            parent.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert!(matches!(
            child.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let loaded_parent = RuntimeConstraint::load(parent.save()).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save()).unwrap();
        let bound = loaded_parent
            .bind_grammar_dynamic_boundary("child", loaded_child)
            .unwrap();
        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("dynamic composition must retain component runtime metadata");
        assert!(overlay.segmented_parser_components.iter().all(|component| {
            matches!(
                component.constraint.boundary_trigger,
                crate::runtime::BoundaryTrigger::Exact(_)
            )
        }));

        for tokens in [&[5][..], &[0, 4][..], &[3, 2][..], &[0, 1, 2][..]] {
            let mut actual = bound.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "Exact-trigger dynamic boundary mismatch before {tokens:?} token {token}",
                );
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn exact_trigger_supports_control_bearing_composed_component() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b">".to_vec()),
            (3, b"!".to_vec()),
            (4, b"<a>!".to_vec()),
            (5, b"a>!".to_vec()),
            (6, b">!".to_vec()),
        ]);

        let inner_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"glrm 1; start inner; extern grammar leaf; nt inner = "<" leaf ">";"#,
            ),
            &vocab,
        )
        .unwrap();
        let leaf = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "a""#), &vocab)
            .unwrap();
        let mut inner = inner_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        assert_recursive_compiler_views_detached(&inner);
        let mut compiler_view = inner.clone();
        compiler_view
            .prepare_recursive_compiler_table_for_composition()
            .unwrap();
        assert!(
            !compiler_view.table.control_terminals.is_empty(),
            "fixture must exercise Exact construction over compiler-materialized linker controls",
        );
        inner.build_exact_boundary_trigger().unwrap();
        assert_recursive_compiler_views_detached(&inner);
        assert!(matches!(
            inner.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let outer = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"glrm 1; start document; extern grammar inner; nt document = inner "!";"#,
            ),
            &vocab,
        )
        .unwrap();
        let composed = outer
            .bind_grammar_dynamic_boundary("inner", RuntimeConstraint::load(inner.save()).unwrap())
            .unwrap();
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "<" "a" ">" "!""#),
            &vocab,
        )
        .unwrap();

        let overlay = composed.static_dynamic_overlay.as_ref().unwrap();
        assert!(matches!(
            overlay.segmented_parser_components[1]
                .constraint
                .boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        for tokens in [&[4][..], &[0, 5][..], &[0, 1, 6][..], &[0, 1, 2, 3][..]] {
            let mut actual = composed.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "control-bearing Exact trigger mismatch before {tokens:?} token {token}",
                );
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn loaded_constraint_trigger_upgrade_resaves_updated_link_metadata() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        let mut loaded = RuntimeConstraint::load(parent.save()).unwrap();
        assert!(loaded.deferred_composition_metadata_blob.is_some());
        loaded.build_exact_boundary_trigger().unwrap();
        assert!(matches!(
            loaded.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert!(
            loaded.deferred_composition_metadata_blob.is_some(),
            "trigger upgrade should not force materialization of the heavy parser-cache section",
        );

        let reloaded = RuntimeConstraint::load(loaded.save()).unwrap();
        let bound = reloaded
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        let overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        assert!(matches!(
            overlay.segmented_parser_components[0]
                .constraint
                .boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));
        assert_ne!(bound.start().mask()[0] & (1 << 2), 0);
    }

    #[test]
    fn exact_triggers_survive_outer_runtime_lexer_product_multi_source_states() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"c".to_vec()),
            (3, b"!".to_vec()),
            (4, b"?".to_vec()),
            (5, b"c!".to_vec()),
            (6, b"c?".to_vec()),
        ]);
        let child_spec = ConstraintSpec::builder(
            Grammar::ebnf(r#"start ::= "a" "b" "c""#),
            &vocab,
        )
        .unwrap()
        .boundary_trigger_detail(crate::BoundaryTriggerDetail::Exact)
        .build()
        .unwrap();
        // Keep one child component so its local-LR -> composed-LR relation is
        // functional, while the parent also has an equivalent local lexical
        // lane. The outer lexer product can then coalesce parent + child lanes
        // after `a`/`b` without relying on duplicate child call sites.
        let parent = Grammar::glrm(
            "glrm 1; start document; extern grammar child; \
             nt document = child \"!\" | \"a\" \"b\" \"c\" \"?\";",
        );
        let composed = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", child_spec)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        let alternatives = composed.clone_constraints();
        assert_eq!(alternatives.len(), 1);
        let composed = &alternatives[0];
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "a" "b" "c" "!" | "a" "b" "c" "?""#),
            &vocab,
        )
        .unwrap();

        let explicitly_disabled = std::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "" | "0" | "false" | "no" | "off"
                )
            });

        let mut actual = composed.start();
        let mut expected = reference.start();
        for token in [0, 1] {
            assert_eq!(actual.mask(), expected.mask());
            actual.commit_token(token).unwrap();
            expected.commit_token(token).unwrap();
        }

        if !explicitly_disabled
            && let Some(source_offset) = composed.runtime_source_state_offset()
        {
            let product_state = *actual.state.keys().next().unwrap();
            assert!(product_state < source_offset);
            assert!(
                composed
                    .runtime_product_source_states(product_state)
                    .is_some_and(|sources| sources.len() >= 2),
                "a selected outer product state must preserve every represented lexer lane",
            );
        }

        let mask = actual.mask();
        let reference_mask = expected.mask();
        assert_eq!(mask, reference_mask);
        assert_ne!(mask[0] & (1 << 5), 0, "c! must cross child -> parent internally");
        assert_ne!(mask[0] & (1 << 6), 0, "c? must remain valid on the parent-local branch");
    }

    #[test]
    fn exact_trigger_detail_is_independent_of_dynamic_component_compilation() {
        let vocab = Vocab::new(vec![
            (0, b"y".to_vec()),
            (1, b"z".to_vec()),
            (2, b"yz".to_vec()),
        ]);
        let spec = ConstraintSpec::builder(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap()
            .boundary_trigger_detail(crate::BoundaryTriggerDetail::Exact)
            .build()
            .unwrap();

        let static_constraint = spec.compile().unwrap();
        assert!(matches!(
            static_constraint.boundary_trigger,
            crate::runtime::BoundaryTrigger::Exact(_)
        ));

        let dynamic_constraint = spec.compile_dynamic().unwrap();
        assert!(dynamic_constraint.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let loaded_dynamic = DynamicConstraint::load(&dynamic_constraint.save()).unwrap();
        assert!(loaded_dynamic.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let transfer = dynamic_constraint.clone().into_saved();
        let loaded_transfer = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(loaded_transfer.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));
    }

    #[test]
    fn boundary_token_trigger_follows_lexeme_resets_inside_model_token() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"q".to_vec()),
            (2, b"y".to_vec()),
            (3, b"xqy".to_vec()),
        ]);
        let mut parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" \"q\" child;",
            ),
            &vocab,
        )
        .unwrap();
        parent.build_boundary_token_trigger().unwrap();
        assert!(
            parent
                .boundary_trigger
                .token_summary()
                .expect("Tokens trigger must be built")
                .contains(&3),
            "a token that reaches the child only after multiple local lexemes must remain boundary-relevant",
        );

        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        assert_ne!(bound.start().mask()[0] & (1 << 3), 0);
    }

    #[test]
    fn nested_dynamic_boundaries_preserve_token_crossing_leaf_middle_and_outer() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert_recursive_compiler_views_detached(&bound);
        assert_recursive_compiler_views_detached(&loaded);
        let mut no_trigger = bound.clone();
        {
            let overlay = no_trigger.static_dynamic_overlay.as_mut().unwrap();
            for component in &mut overlay.segmented_parser_components {
                let Some(shard) = component.boundary.as_mut() else {
                    continue;
                };
                if !matches!(
                    shard.backend,
                    crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
                ) {
                    continue;
                }
                shard.candidate_tokens = None;
                std::sync::Arc::make_mut(&mut component.constraint).boundary_trigger =
                    crate::runtime::BoundaryTrigger::None;
            }
        }
        let mut no_outer_tokenizer = loaded.clone();
        {
            let root = no_outer_tokenizer
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_components[0]
                .constraint
                .clone();
            no_outer_tokenizer.tokenizer = root.tokenizer.clone();
            no_outer_tokenizer.tokenizer_fast_transitions =
                root.tokenizer_fast_transitions.clone();
            no_outer_tokenizer.tokenizer_has_epsilon_transitions =
                root.tokenizer_has_epsilon_transitions;
        }
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        for constraint in [
            &bound,
            &loaded,
            &no_trigger,
            &no_outer_tokenizer,
            &no_outer_table,
        ] {
            let mut pending = vec![Vec::<u32>::new()];
            while let Some(path) = pending.pop() {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in &path {
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                let actual_mask = actual.mask();
                let expected_mask = expected.mask();
                assert_eq!(actual_mask, expected_mask, "mask mismatch at {path:?}");
                assert_eq!(
                    actual.is_accepting(),
                    expected.is_accepting(),
                    "acceptance mismatch at {path:?}",
                );
                if path.len() == 5 {
                    continue;
                }
                for token in 0..8u32 {
                    if actual_mask
                        .get(token as usize / 32)
                        .is_none_or(|word| *word & (1u32 << (token % 32)) == 0)
                    {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push(token);
                    pending.push(next);
                }
            }
        }

        for tokens in [
            &[7][..],
            &[0, 6][..],
            &[0, 1, 5][..],
            &[0, 1, 2, 3, 4][..],
        ] {
            let mut actual = bound.start();
            let mut expected = monolithic.start();
            for &token in tokens {
                assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(actual.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn recursive_live_parser_expands_static_nested_component_with_native_boundary() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        // Static B is transported exactly onto the recursive parser coordinate,
        // so the middle wrapper can expose its intact parent/child leaves.
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.uses_compact_segmented_parser_runtime());
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 3);
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            for tokens in [
                &[7][..],
                &[0, 6][..],
                &[0, 1, 5][..],
                &[0, 1, 2, 3, 4][..],
            ] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn recursive_special_token_routes_through_active_child_leaf_and_returns_to_parent() {
        let vocab = Vocab::new(vec![(0, b"X".to_vec()), (1, b"!".to_vec())]);
        let child = RuntimeConstraint::compile(
            Grammar::glrm("start child; nt child ::= @token(100);"),
            &vocab,
        )
        .unwrap();
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar child; nt document = \"X\" child \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = parent.bind_grammar("child", child).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        for constraint in [&bound, &loaded, &no_outer_table] {
            let mut state = constraint.start();
            state.commit_token(0).unwrap();
            let mask = state.mask();
            assert_ne!(mask[100 / 32] & (1u32 << (100 % 32)), 0);

            state.commit_token(100).unwrap();
            let mask = state.mask();
            assert_ne!(mask[1 / 32] & (1u32 << (1 % 32)), 0);

            state.commit_token(1).unwrap();
            assert!(state.is_accepting());
        }
    }

    #[test]
    fn nested_static_boundaries_use_recursive_parser_coordinate_live_and_loaded() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
            (5, b"a]!".to_vec()),
            (6, b"[a]!".to_vec()),
            (7, b"X[a]!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.uses_compact_segmented_parser_runtime());

        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent.bind_grammar("middle", middle).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        assert!(
            bound
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_component_union_root_dispatch
                .is_empty(),
            "recursive runtime must not retain materialized-state component dispatch",
        );
        assert!(
            bound
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_state_offsets
                .is_empty(),
            "recursive runtime must derive state intervals from its component tree",
        );
        let bound_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        assert!(bound_overlay.segmented_boundary_shards.iter().all(|shard| {
            shard.start_parser_states.len() == 0
        }));
        assert!(bound_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.len() == 0)
        }));
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 3);

        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert!(loaded.uses_compact_segmented_parser_runtime());
        assert!(
            loaded
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_component_union_root_dispatch
                .is_empty(),
        );
        assert!(
            loaded
                .static_dynamic_overlay
                .as_ref()
                .unwrap()
                .segmented_parser_state_offsets
                .is_empty(),
        );
        let loaded_overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert!(loaded_overlay.segmented_boundary_shards.iter().all(|shard| {
            shard.start_parser_states.len() == 0
        }));
        assert!(loaded_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.len() == 0)
        }));
        assert_eq!(loaded.recursive_parser_layout().unwrap().unwrap().leaves.len(), 3);

        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);
        fn compare_reachable_prefix_tree(
            actual_constraint: &RuntimeConstraint,
            expected_constraint: &RuntimeConstraint,
            max_depth: usize,
        ) {
            let mut pending = vec![Vec::<u32>::new()];
            while let Some(path) = pending.pop() {
                let mut actual = actual_constraint.start();
                let mut expected = expected_constraint.start();
                for &token in &path {
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                let actual_mask = actual.mask();
                let expected_mask = expected.mask();
                assert_eq!(actual_mask, expected_mask, "mask mismatch at {path:?}");
                assert_eq!(
                    actual.is_accepting(),
                    expected.is_accepting(),
                    "acceptance mismatch at {path:?}",
                );
                if path.len() == max_depth {
                    continue;
                }
                for token in 0..8u32 {
                    let word = token as usize / 32;
                    let bit = token % 32;
                    if actual_mask
                        .get(word)
                        .is_none_or(|word| *word & (1u32 << bit) == 0)
                    {
                        continue;
                    }
                    let mut next = path.clone();
                    next.push(token);
                    pending.push(next);
                }
            }
        }
        for constraint in [&bound, &loaded, &no_outer_table] {
            compare_reachable_prefix_tree(constraint, &monolithic, 5);
            for tokens in [
                &[7][..],
                &[0, 6][..],
                &[0, 1, 5][..],
                &[0, 1, 2, 3, 4][..],
            ] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn nested_static_nullable_wrapper_returns_in_recursive_live_runtime() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"X!".to_vec()),
            (4, b"Xa!".to_vec()),
            (5, b"a!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start middle; extern grammar leaf; nt middle = leaf?;"),
            &vocab,
        )
        .unwrap();
        assert!(middle_parent.table.embedded_start_nullable());
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.table.embedded_start_nullable());
        assert!(middle.uses_compact_segmented_parser_runtime());

        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        let loaded = RuntimeConstraint::load(&bound.save()).unwrap();
        assert!(loaded.uses_compact_segmented_parser_runtime());
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"a\"? \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        poison_materialized_outer_table(&mut no_outer_table);

        for constraint in [&bound, &loaded, &no_outer_table] {
            for tokens in [&[3][..], &[4][..], &[0, 2][..], &[0, 5][..], &[0, 1, 2][..]] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    actual.commit_token(token).unwrap();
                    expected.commit_token(token).unwrap();
                }
                assert_eq!(actual.is_accepting(), expected.is_accepting(), "{tokens:?}");
                assert!(actual.is_accepting(), "{tokens:?}");
            }
        }
    }

    #[test]
    fn recursive_parser_layout_flattens_state_coordinate_but_preserves_outer_wrapper_owner() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; nt leaf = \"a\";"),
            &vocab,
        )
        .unwrap();
        let leaf_states = leaf.table.num_states;
        let leaf_tokenizer_states = leaf.tokenizer.num_states();
        let leaf_tokenizer_reset = leaf.runtime_commit_initial_state();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle_parent_states = middle_parent.table.num_states;
        let middle_parent_tokenizer_states = middle_parent.tokenizer.num_states();
        let middle_parent_tokenizer_reset = middle_parent.runtime_commit_initial_state();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; nt document = \"X\" middle \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let outer_parent_states = outer_parent.table.num_states;
        let outer_parent_tokenizer_states = outer_parent.tokenizer.num_states();
        let outer_parent_tokenizer_reset = outer_parent.runtime_commit_initial_state();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let layout = constraint
                .recursive_parser_layout()
                .unwrap()
                .expect("nested composition must expose a recursive parser layout");
            assert_eq!(layout.component_offsets, vec![0, outer_parent_states]);
            assert_eq!(layout.leaves.len(), 3);
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| leaf.component_path.clone())
                    .collect::<Vec<_>>(),
                vec![vec![0], vec![1, 0], vec![1, 1]],
            );
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| (leaf.state_offset, leaf.state_count, leaf.top_component))
                    .collect::<Vec<_>>(),
                vec![
                    (0, outer_parent_states, 0),
                    (outer_parent_states, middle_parent_states, 1),
                    (
                        outer_parent_states + middle_parent_states,
                        leaf_states,
                        1,
                    ),
                ],
            );
            assert_eq!(
                layout.total_states,
                outer_parent_states + middle_parent_states + leaf_states,
            );
            assert_eq!(
                layout.leaf_tokenizer_state_offsets,
                vec![
                    0,
                    outer_parent_tokenizer_states,
                    outer_parent_tokenizer_states + middle_parent_tokenizer_states,
                ],
            );
            assert_eq!(
                layout.total_tokenizer_states,
                outer_parent_tokenizer_states
                    + middle_parent_tokenizer_states
                    + leaf_tokenizer_states,
            );
            for (global_terminal, targets) in layout.terminal_targets.iter().enumerate() {
                for &(leaf_index, local_terminal) in targets {
                    let scoped = constraint
                        .recursive_terminal_scoped_id(leaf_index as usize, local_terminal)
                        .unwrap();
                    assert_eq!(
                        constraint.recursive_terminal_leaf_local(scoped),
                        Some((leaf_index as usize, local_terminal)),
                        "terminal routing lost global={global_terminal} leaf={leaf_index} local={local_terminal}",
                    );
                }
            }
            let tokenizer_counts = [
                outer_parent_tokenizer_states,
                middle_parent_tokenizer_states,
                leaf_tokenizer_states,
            ];
            let tokenizer_resets = [
                outer_parent_tokenizer_reset,
                middle_parent_tokenizer_reset,
                leaf_tokenizer_reset,
            ];
            for leaf_index in 0..layout.leaves.len() {
                for local_state in 0..tokenizer_counts[leaf_index] {
                    let scoped = constraint
                        .recursive_tokenizer_scoped_state(leaf_index, local_state)
                        .unwrap();
                    assert_eq!(
                        constraint.recursive_tokenizer_leaf_state(scoped),
                        Some((leaf_index, local_state)),
                    );
                }
                assert_eq!(
                    constraint.recursive_tokenizer_reset_state(leaf_index),
                    constraint.recursive_tokenizer_scoped_state(
                        leaf_index,
                        tokenizer_resets[leaf_index],
                    ),
                );
            }
            assert!(layout.leaves[1].state_count >= 2);
            let root_top = layout.leaves[0].state_offset;
            let middle_top_a = layout.leaves[1].state_offset;
            let middle_top_b = middle_top_a + 1;
            let leaf_top = layout.leaves[2].state_offset;
            let acc0 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(10, 20);
            let acc1 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(11, 21);
            let acc2 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(12, 22);
            let acc3 = crate::compiler::glr::accumulator::TerminalsDisallowed::new()
                .with_insert(13, 23);
            let mixed = crate::compiler::glr::parser::ParserGSS::from_stacks(&[
                (vec![root_top], acc0.clone()),
                (vec![root_top, middle_top_a], acc1.clone()),
                (vec![root_top, middle_top_b], acc2.clone()),
                (vec![root_top, leaf_top], acc3.clone()),
            ]);
            let partitions = constraint
                .partition_recursive_parser_gss_by_active_leaf(&mixed)
                .expect("recursive GSS must partition by active tokenizer leaf");
            assert_eq!(
                partitions.iter().map(|(leaf, _)| *leaf).collect::<Vec<_>>(),
                vec![0, 1, 2],
            );
            let mut middle_stacks = partitions[1].1.to_stacks(16).unwrap();
            middle_stacks.sort_by_key(|(stack, _)| *stack.last().unwrap());
            assert_eq!(
                middle_stacks,
                vec![
                    (vec![root_top, middle_top_a], acc1),
                    (vec![root_top, middle_top_b], acc2),
                ],
            );
            assert_eq!(
                partitions[0].1.to_stacks(16).unwrap(),
                vec![(vec![root_top], acc0)],
            );
            assert_eq!(
                partitions[2].1.to_stacks(16).unwrap(),
                vec![(vec![root_top, leaf_top], acc3)],
            );
            assert_eq!(constraint.recursive_parser_state_span().unwrap(), layout.total_states);
            assert_eq!(layout.links.len(), 2);
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 1 && link.child_component == 2));
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 1));
        }
    }

    #[test]
    fn recursive_parser_reference_executes_nested_calls_without_materialized_child_table() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"[".to_vec()),
            (2, b"a".to_vec()),
            (3, b"]".to_vec()),
            (4, b"!".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; t A = \"a\"; nt leaf = A;"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; t L = \"[\"; t R = \"]\"; nt middle = L leaf R;",
            ),
            &vocab,
        )
        .unwrap();
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", &leaf)
            .unwrap();
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; t X = \"X\"; t BANG = \"!\"; nt document = X middle BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", &middle)
            .unwrap();

        let outer_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        let middle_outer_offset = outer_overlay.segmented_parser_components[1].terminal_offset;
        let middle_overlay = middle.static_dynamic_overlay.as_ref().unwrap();
        let middle_parent_offset = middle_overlay.segmented_parser_components[0].terminal_offset;
        let leaf_offset = middle_overlay.segmented_parser_components[1].terminal_offset;
        let terminals = [
            outer_overlay.segmented_parser_components[0].terminal_offset
                + terminal(&outer_parent, "X"),
            middle_outer_offset + middle_parent_offset + terminal(&middle_parent, "L"),
            middle_outer_offset + leaf_offset + terminal(&leaf, "A"),
            middle_outer_offset + middle_parent_offset + terminal(&middle_parent, "R"),
            outer_overlay.segmented_parser_components[0].terminal_offset
                + terminal(&outer_parent, "BANG"),
        ];

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                vec![0],
                crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
            );
            let mut parser = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &terminals {
                parser = constraint
                    .advance_recursive_segmented_parser_reference(&parser, terminal)
                    .unwrap()
                    .unwrap();
                assert!(
                    !parser.is_empty(),
                    "recursive parser rejected terminal {terminal}",
                );
            }
            assert_eq!(
                constraint
                    .recursive_segmented_parser_is_finished_reference(&parser)
                    .unwrap(),
                Some(true),
            );

            let mut invalid = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &[terminals[0], terminals[1], terminals[3]] {
                invalid = constraint
                    .advance_recursive_segmented_parser_reference(&invalid, terminal)
                    .unwrap()
                    .unwrap();
            }
            assert!(invalid.is_empty(), "recursive parser accepted a missing leaf token");
        }
    }

    #[test]
    fn recursive_parser_reference_resolves_new_link_through_precomposed_parent() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b">".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar left; extern grammar right; t LT = \"<\"; t GT = \">\"; nt document = LT left right GT;",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start left; t A = \"a\"; nt left = A;"),
            &vocab,
        )
        .unwrap();
        let right = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start right; t B = \"b\"; nt right = B;"),
            &vocab,
        )
        .unwrap();
        let half = parent
            .bind_grammar_dynamic_boundary("left", &left)
            .unwrap();
        let full = half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();

        let full_overlay = full.static_dynamic_overlay.as_ref().unwrap();
        let half_offset = full_overlay.segmented_parser_components[0].terminal_offset;
        let right_offset = full_overlay.segmented_parser_components[1].terminal_offset;
        let half_overlay = half.static_dynamic_overlay.as_ref().unwrap();
        let original_parent_offset = half_overlay.segmented_parser_components[0].terminal_offset;
        let left_offset = half_overlay.segmented_parser_components[1].terminal_offset;
        let terminals = [
            half_offset + original_parent_offset + terminal(&parent, "LT"),
            half_offset + left_offset + terminal(&left, "A"),
            right_offset + terminal(&right, "B"),
            half_offset + original_parent_offset + terminal(&parent, "GT"),
        ];

        for constraint in [&full, &RuntimeConstraint::load(&full.save()).unwrap()] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            assert_eq!(
                layout
                    .leaves
                    .iter()
                    .map(|leaf| leaf.component_path.clone())
                    .collect::<Vec<_>>(),
                vec![vec![0, 0], vec![0, 1], vec![1]],
            );
            assert_eq!(layout.links.len(), 2);
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 1));
            assert!(layout
                .links
                .iter()
                .any(|link| link.parent_component == 0 && link.child_component == 2));

            let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                vec![0],
                crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
            );
            let mut parser = constraint
                .close_recursive_segmented_parser_reference(&start)
                .unwrap()
                .unwrap();
            for &terminal in &terminals {
                parser = constraint
                    .advance_recursive_segmented_parser_reference(&parser, terminal)
                    .unwrap()
                    .unwrap();
                assert!(!parser.is_empty(), "recursive parser rejected terminal {terminal}");
            }
            assert_eq!(
                constraint
                    .recursive_segmented_parser_is_finished_reference(&parser)
                    .unwrap(),
                Some(true),
            );
        }
    }

    #[test]
    fn loaded_recursive_composition_rebinds_from_lazy_compiler_views() {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b">".to_vec()),
            (4, b"<ab>".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar left; extern grammar right; nt document = \"<\" left right \">\";",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start left; nt left = \"a\";"),
            &vocab,
        )
        .unwrap();
        let right = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start right; nt right = \"b\";"),
            &vocab,
        )
        .unwrap();

        let half = parent
            .bind_grammar_dynamic_boundary("left", &left)
            .unwrap();
        let half_overlay = half.static_dynamic_overlay.as_ref().unwrap();
        let half_layout = half.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(
            half.tokenizer.num_states(),
            half_overlay.segmented_parser_components[0]
                .constraint
                .tokenizer
                .num_states(),
            "recursive coordinator must retain only the root-leaf tokenizer",
        );
        assert!(
            half.tokenizer.num_states() < half_layout.total_tokenizer_states,
            "test fixture must actually contain more than one tokenizer leaf",
        );
        assert_eq!(
            half.table.num_states, 0,
            "recursive coordinator must not retain a flattened LR state machine",
        );
        assert!(half.table.action.is_empty() && half.table.goto.is_empty());
        assert!(half_overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.global_to_local_parser_state.is_empty()));

        let loaded_half = RuntimeConstraint::load(&half.save()).unwrap();
        let loaded_half_overlay = loaded_half.static_dynamic_overlay.as_ref().unwrap();
        let loaded_half_layout = loaded_half.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(
            loaded_half.tokenizer.num_states(),
            loaded_half_overlay.segmented_parser_components[0]
                .constraint
                .tokenizer
                .num_states(),
            "loaded recursive coordinator must not reconstruct the outer union tokenizer eagerly",
        );
        assert!(loaded_half.tokenizer.num_states() < loaded_half_layout.total_tokenizer_states);
        assert_eq!(
            loaded_half.table.num_states, 0,
            "loaded recursive coordinator must keep the flattened parser table lazy",
        );
        assert!(loaded_half.table.action.is_empty() && loaded_half.table.goto.is_empty());
        assert!(loaded_half_overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.global_to_local_parser_state.is_empty()));

        let fresh_full = half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();
        let loaded_full = loaded_half
            .bind_grammar_dynamic_boundary("right", &right)
            .unwrap();
        for constraint in [&fresh_full, &loaded_full] {
            assert_recursive_compiler_views_detached(constraint);
        }
        let fresh_start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
            vec![0],
            crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
        );
        let loaded_start = fresh_start.clone();
        let fresh_closed = fresh_full
            .close_compact_segmented_parser(&fresh_start)
            .unwrap();
        let loaded_closed = loaded_full
            .close_compact_segmented_parser(&loaded_start)
            .unwrap();
        assert!(fresh_closed.semantically_eq(&loaded_closed, 4096).unwrap());
        for terminal in 0..fresh_full.table.num_terminals {
            let fresh_advanced = fresh_full
                .advance_compact_segmented_parser(&fresh_closed, terminal)
                .unwrap();
            let loaded_advanced = loaded_full
                .advance_compact_segmented_parser(&loaded_closed, terminal)
                .unwrap();
            assert!(
                fresh_advanced
                    .semantically_eq(&loaded_advanced, 4096)
                    .unwrap(),
                "recursive parser advance differs for terminal {terminal}",
            );
        }
        for constraint in [&fresh_full, &loaded_full] {
            let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
            assert!(overlay
                .segmented_parser_components
                .iter()
                .all(|component| component.global_to_local_parser_state.is_empty()));
            let mut split = constraint.start();
            for token in [0, 1, 2, 3] {
                assert_ne!(split.mask()[0] & (1 << token), 0);
                split.commit_token(token).unwrap();
            }
            assert!(split.is_accepting());

            let mut fused = constraint.start();
            assert_ne!(fused.mask()[0] & (1 << 4), 0);
            fused.commit_token(4).unwrap();
            assert!(fused.is_accepting());
        }
    }

    #[test]
    fn recursive_parser_reference_preserves_nested_nullable_wrapper_return() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
        ]);
        let terminal = |constraint: &RuntimeConstraint, name: &str| {
            constraint
                .terminal_display_names
                .iter()
                .position(|candidate| candidate == name)
                .unwrap() as u32
        };
        let leaf = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start leaf; t A = \"a\"; nt leaf = A?;"),
            &vocab,
        )
        .unwrap();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start middle; extern grammar leaf; nt middle = leaf;"),
            &vocab,
        )
        .unwrap();
        assert!(!middle_parent.table.embedded_start_nullable());
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", &leaf)
            .unwrap();
        assert!(middle.table.embedded_start_nullable());
        let outer_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar middle; t X = \"X\"; t BANG = \"!\"; nt document = X middle BANG;",
            ),
            &vocab,
        )
        .unwrap();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", &middle)
            .unwrap();

        let outer_overlay = bound.static_dynamic_overlay.as_ref().unwrap();
        let middle_outer_offset = outer_overlay.segmented_parser_components[1].terminal_offset;
        let middle_overlay = middle.static_dynamic_overlay.as_ref().unwrap();
        let leaf_offset = middle_overlay.segmented_parser_components[1].terminal_offset;
        let x = outer_overlay.segmented_parser_components[0].terminal_offset
            + terminal(&outer_parent, "X");
        let a = middle_outer_offset + leaf_offset + terminal(&leaf, "A");
        let bang = outer_overlay.segmented_parser_components[0].terminal_offset
            + terminal(&outer_parent, "BANG");

        for constraint in [&bound, &RuntimeConstraint::load(&bound.save()).unwrap()] {
            let layout = constraint.recursive_parser_layout().unwrap().unwrap();
            let outer_link = layout
                .links
                .iter()
                .find(|link| link.parent_component == 0 && link.child_component == 1)
                .expect("outer link must target the middle wrapper root leaf");
            assert!(outer_link.child_start_nullable);

            for terminals in [&[x, bang][..], &[x, a, bang][..]] {
                let start = crate::compiler::glr::parser::ParserGSS::from_single_stack(
                    vec![0],
                    crate::compiler::glr::accumulator::TerminalsDisallowed::new(),
                );
                let mut parser = constraint
                    .close_recursive_segmented_parser_reference(&start)
                    .unwrap()
                    .unwrap();
                for &terminal in terminals {
                    parser = constraint
                        .advance_recursive_segmented_parser_reference(&parser, terminal)
                        .unwrap()
                        .unwrap();
                    assert!(!parser.is_empty(), "recursive nullable parser rejected {terminals:?}");
                }
                assert_eq!(
                    constraint
                        .recursive_segmented_parser_is_finished_reference(&parser)
                        .unwrap(),
                    Some(true),
                    "{terminals:?}",
                );
            }
        }
    }

    #[test]
    fn static_boundary_shards_are_authoritative_across_multiple_internal_crossings() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar left; extern grammar right; \
                 nt start = \"x\" left right;",
            ),
            &vocab,
        )
        .unwrap();
        let left = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let right = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab)
            .unwrap();
        let bound = parent
            .bind_grammar("left", left)
            .unwrap()
            .bind_grammar("right", right)
            .unwrap();

        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("static composition must retain segmented A+B metadata");
        assert!(
            !overlay.segmented_boundary_shards.is_empty(),
            "static composition must publish component-scoped B shards",
        );
        assert!(
            overlay.segmented_boundary_parser.is_none(),
            "partitioned static B must not retain a redundant global boundary parser",
        );
        assert!(overlay.segmented_boundary_terminal_trie.is_none());

        let loaded = RuntimeConstraint::load(bound.save()).unwrap();
        let loaded_overlay = loaded.static_dynamic_overlay.as_ref().unwrap();
        assert!(loaded_overlay.segmented_boundary_parser.is_none());
        assert!(loaded_overlay.segmented_boundary_terminal_trie.is_none());
        assert!(!loaded_overlay.segmented_boundary_shards.is_empty());
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y" "z""#),
            &vocab,
        )
        .unwrap();

        for tokens in [&[5][..], &[0, 4][..], &[0, 1, 2][..]] {
            let mut sharded = bound.start();
            let mut restored = loaded.start();
            let mut expected = reference.start();
            for &token in tokens {
                assert_eq!(
                    sharded.mask(),
                    expected.mask(),
                    "partitioned static B mismatch before {tokens:?} token {token}",
                );
                assert_eq!(
                    restored.mask(),
                    expected.mask(),
                    "restored static shards differ before {tokens:?} token {token}",
                );
                sharded.commit_token(token).unwrap();
                restored.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            assert_eq!(sharded.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert_eq!(restored.is_accepting(), expected.is_accepting(), "{tokens:?}");
            assert!(sharded.is_accepting(), "{tokens:?}");
        }
    }

    #[test]
    fn authoritative_ab_keeps_component_parser_dwas_unchanged_across_scoped_ignores() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b" ".to_vec()),
            (2, b"\t".to_vec()),
            (3, b"a".to_vec()),
            (4, b"!".to_vec()),
            (5, b"X \ta!".to_vec()),
            (6, b"X\t a!".to_vec()),
            (7, b"Xa\t !".to_vec()),
            (8, b"Xa \t!".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS = " "+;
                    extern grammar child;
                    nt document = "X" child "!";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start child;
                    ignore CHILD_WS;
                    t CHILD_WS = "\t"+;
                    nt child = "a";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        assert!(
            !parent.composition_reset_tokens_by_terminal.is_empty(),
            "late-bind parent compilation should precompute reset-token composition metadata",
        );
        let mut cached_parent = RuntimeConstraint::load(parent.save()).unwrap();
        cached_parent
            .materialize_composition_metadata_for_compilation()
            .unwrap();
        assert_eq!(
            cached_parent.composition_reset_tokens_by_terminal,
            parent.composition_reset_tokens_by_terminal,
            "late-bind reset-token cache must survive save/load",
        );
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS = " "+;
                    g child = {
                        start child;
                        ignore CHILD_WS;
                        t CHILD_WS = "\t"+;
                        nt child = "a";
                    };
                    nt document = "X" child "!";
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let parent_dwa = parent.parser_dwa.clone();
        let child_dwa = child.parser_dwa.clone();
        let bound = parent
            .bind_grammar_dynamic_boundary("child", child.clone())
            .unwrap();
        let overlay = bound
            .static_dynamic_overlay
            .as_ref()
            .expect("authoritative A+B metadata");
        assert!(overlay.segmented_mask_authoritative);
        assert!(!overlay.segmented_static_baseline);
        assert_eq!(overlay.segmented_parser_components.len(), 2);
        assert_eq!(overlay.segmented_parser_components[0].constraint.parser_dwa, parent_dwa);
        assert_eq!(overlay.segmented_parser_components[1].constraint.parser_dwa, child_dwa);
        assert!(overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.root_disallowed_terminal.is_none()));

        let loaded = RuntimeConstraint::load(bound.save()).unwrap();
        assert!(bound.uses_compact_segmented_parser_runtime());
        assert!(loaded.uses_compact_segmented_parser_runtime());
        let loaded_overlay = loaded
            .static_dynamic_overlay
            .as_ref()
            .expect("round-tripped authoritative A+B metadata");
        assert_eq!(loaded_overlay.segmented_parser_components.len(), 2);
        let mut loaded_parent = loaded_overlay.segmented_parser_components[0]
            .constraint
            .as_ref()
            .clone();
        let mut loaded_child = loaded_overlay.segmented_parser_components[1]
            .constraint
            .as_ref()
            .clone();
        loaded_parent.materialize_parser_dwa_for_compilation().unwrap();
        loaded_child.materialize_parser_dwa_for_compilation().unwrap();
        assert_eq!(loaded_parent.parser_dwa, parent_dwa);
        assert_eq!(loaded_child.parser_dwa, child_dwa);
        for (kind, constraint) in [("source", &bound), ("loaded", &loaded)] {
            let mut actual = constraint.start();
            let mut expected = monolithic.start();
            assert_eq!(actual.mask(), expected.mask(), "{kind} initial mask");
            for token in [1, 0, 2, 3, 1, 4] {
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
                assert_eq!(actual.mask(), expected.mask(), "after token {token}");
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());

            for token in [5, 6, 7, 8] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                assert_eq!(
                    actual.commit_token(token).is_ok(),
                    expected.commit_token(token).is_ok(),
                    "fused token {token}",
                );
                assert_eq!(actual.is_accepting(), expected.is_accepting());
            }
        }
    }

    #[test]
    fn authoritative_ab_handles_multi_terminal_parent_token_after_zero_width_return() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"a".to_vec()),
            (2, b"!".to_vec()),
            (3, b"?".to_vec()),
            (4, b"!?".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    extern grammar child;
                    nt document = "X" child "!" "?";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start child; nt child = \"a\";"),
            &vocab,
        )
        .unwrap();
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm(
                r#"
                    glrm 1;
                    start document;
                    g child = { start child; nt child = "a"; };
                    nt document = "X" child "!" "?";
                "#,
            ),
            &vocab,
        )
        .unwrap();

        let bound = parent
            .bind_grammar_dynamic_boundary("child", child)
            .unwrap();
        for constraint in [&bound, &RuntimeConstraint::load(bound.save()).unwrap()] {
            let mut actual = constraint.start();
            let mut expected = monolithic.start();
            for token in [0, 1] {
                assert_eq!(actual.mask(), expected.mask(), "before token {token}");
                actual.commit_token(token).unwrap();
                expected.commit_token(token).unwrap();
            }
            let actual_mask = actual.mask();
            let expected_mask = expected.mask();
            assert_eq!(actual_mask, expected_mask, "after child completion");
            assert_ne!(actual_mask[0] & (1 << 4), 0, "fused parent token !? must be allowed");
            actual.commit_token(4).unwrap();
            expected.commit_token(4).unwrap();
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn late_bind_vocab_cache_is_reused_and_not_serialized() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        assert!(
            parent.late_bind_vocab.get().is_some(),
            "fresh compilation should retain the supplied vocabulary for later binds",
        );

        let loaded_parent = RuntimeConstraint::load(parent.save()).unwrap();
        assert!(
            loaded_parent.late_bind_vocab.get().is_none(),
            "late-bind vocabulary memoization is runtime-only and must not enter the wire format",
        );
        let first = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert!(loaded_parent.late_bind_vocab.get().is_some());
        let second = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), second.start().mask());

        let loaded = RuntimeConstraint::load(loaded_parent.save()).unwrap();
        assert!(
            loaded.late_bind_vocab.get().is_none(),
            "late-bind vocabulary memoization is runtime-only and must not enter the wire format",
        );
        let rebound = loaded
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), rebound.start().mask());
    }
}

#[cfg(test)]
mod cached_parent_main_tests {
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
        let parent = RuntimeConstraint::compile(
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
            parent
                .late_grammar_slots
                .iter()
                .map(|slot| slot.name.as_str())
                .collect::<Vec<_>>(),
            vec!["payload"],
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

        let with_a = parent.bind_grammar("payload", &child_a).unwrap();
        let with_b = parent.bind_grammar("payload", &child_b).unwrap();
        assert!(accepts(&with_a, b"x"));
        assert!(accepts(&with_a, b"<a>"));
        assert!(!accepts(&with_a, b"<b>"));
        assert!(accepts(&with_b, b"<b>"));
        assert!(!accepts(&with_b, b"<a>"));
        assert!(parent
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "payload"));

        let saved = parent.save();
        let loaded = RuntimeConstraint::load(&saved).unwrap();
        let loaded_with_a = loaded.bind_grammar("payload", &child_a).unwrap();
        assert!(accepts(&loaded_with_a, b"<a>"));
        assert!(!accepts(&loaded_with_a, b"<b>"));
    }

    #[test]
    fn compiled_parent_can_fill_multiple_slots_incrementally() {
        let vocab = vocab();
        let parent = RuntimeConstraint::compile(
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
        let half = parent.bind_grammar("left", &a).unwrap();
        assert!(half
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "right"));
        let full = half.bind_grammar("right", &b).unwrap();
        assert!(full.late_grammar_slots.is_empty());
        assert!(accepts(&full, b"<ab>"));

        let unresolved_child = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar leaf; start value; nt value = leaf;",
            ),
            &vocab,
        )
        .unwrap();
        // Main used to reject this because independently compiled unresolved
        // children could collide in the private placeholder-token namespace.
        // The successor architecture transports those slots by qualified name
        // and sanitizes the private linker token from the public token domain.
        let open_left = parent.bind_grammar("left", &unresolved_child).unwrap();
        assert!(open_left
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "left.leaf"));
        let left = open_left.bind_grammar("left.leaf", &a).unwrap();
        let nested_full = left.bind_grammar("right", &b).unwrap();
        assert!(nested_full.late_grammar_slots.is_empty());
        assert!(accepts(&nested_full, b"<ab>"));
    }

    #[test]
    fn dynamic_compile_preserves_unbound_external_grammar() {
        let vocab = vocab();
        let open = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; extern grammar payload; start document; nt document = payload;",
            ),
            &vocab,
        )
        .unwrap();
        assert!(open
            .clone_constraints()
            .iter()
            .all(|alternative| alternative
                .late_grammar_slots
                .iter()
                .any(|slot| slot.name == "payload")));
        let child = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start value; nt value = \"a\";"),
            &vocab,
        )
        .unwrap();
        let bound = open.bind_grammar("payload", &child).unwrap();
        let mut state = bound.start();
        state.commit_bytes(b"a").unwrap();
        assert!(state.is_accepting());
    }
}
