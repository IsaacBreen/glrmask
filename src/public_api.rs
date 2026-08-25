use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend,
    compose_constraints_owned_parent_segmented,
};
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
        // Serialization is intentionally lazy. Compiling a reusable constraint
        // must not silently pay the first-save cost or shift serialization work
        // into build latency.
        self.compile_static_uncached()
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
        // Dynamic compilation preserves dynamic component-local masking and
        // evaluates cross-component behavior through the exact terminal-NWA
        // boundary walker rather than compiling B into a parser DWA.
        let boundary_backend = SegmentedBoundaryBackend::DynamicTerminalNwa;
        let children = prepare_compiled_children(children, self.vocab, boundary_backend)?;
        let alternatives = parents
            .clone_constraints()
            .into_iter()
            .map(|parent| compose_named_children(parent, &children, self.vocab, boundary_backend))
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

    /// Check that every target-specific exact-token extern is bound and finish
    /// the specification. External grammars may remain unresolved in the
    /// compiled artifact and be linked later.
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
    fn clone_constraints(&self) -> Vec<RuntimeConstraint> {
        match self {
            Self::StaticBorrowed(constraint) => vec![(*constraint).clone()],
            Self::StaticOwned(constraint) => vec![constraint.clone()],
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
                        Ok(CompiledChild::StaticOwned(spec.compile_static_uncached()?))
                    }
                    ChildCompileMode::Dynamic => {
                        Ok(CompiledChild::DynamicOwned(spec.compile_dynamic()?))
                    }
                }
            }
            Self::Spec(spec) => match mode {
                ChildCompileMode::Static => {
                    Ok(CompiledChild::StaticOwned(spec.compile_static_uncached()?))
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
}

fn prepare_compiled_children(
    children: Vec<(String, CompiledChild<'_>)>,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Vec<(String, RuntimeConstraint)>> {
    children
        .into_iter()
        .map(|(name, child)| {
            let constraint = collapse_dynamic_alternatives(
                child.clone_constraints(),
                vocab,
                boundary_backend,
            )?;
            Ok((name, constraint))
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
        .zip(alternatives)
        .collect::<Vec<_>>();
    let parent = RuntimeConstraint::from_glrm_grammar_with_subgrammars(&source, &[], vocab)?;
    let mut union = compose_named_children(parent, &children, vocab, boundary_backend)?;
    for slot in &mut union.late_grammar_slots {
        if let Some((alternative, nested)) = slot.name.split_once("::")
            && alternative.starts_with("alternative_")
        {
            slot.name = nested.to_owned();
        }
    }
    Ok(union)
}

fn compose_named_children(
    parent: RuntimeConstraint,
    children: &[(String, RuntimeConstraint)],
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
            constraint: &children[child_index].1,
        })
        .collect::<Vec<_>>();
    let mut composition = compose_constraints_owned_parent_segmented(
        parent,
        &inputs,
        vocab,
        boundary_backend,
    )
    .map_err(Error::Compilation)?;
    composition.constraint.late_grammar_slots = remaining_parent_slots;
    for (component_index, &child_index) in present.iter().enumerate() {
        let terminal_offset = composition.terminal_offsets[component_index + 1];
        let (binding_name, child) = &children[child_index];
        composition.constraint.late_grammar_slots.extend(
            child
                .late_grammar_slots
                .iter()
                .map(|slot| crate::runtime::LateGrammarSlot {
                    name: format!("{binding_name}::{}", slot.name),
                    terminal_id: terminal_offset + slot.terminal_id,
                }),
        );
    }
    Ok(composition.constraint)
}

fn compose_named_shared_child(
    parent: RuntimeConstraint,
    name: &str,
    child: Arc<RuntimeConstraint>,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<RuntimeConstraint> {
    let matching_terminals = parent
        .late_grammar_slots
        .iter()
        .filter(|slot| slot.name == name)
        .map(|slot| slot.terminal_id)
        .collect::<Vec<_>>();
    if matching_terminals.is_empty() {
        return Ok(parent);
    }

    let remaining_parent_slots = parent
        .late_grammar_slots
        .iter()
        .filter(|slot| slot.name != name)
        .cloned()
        .collect::<Vec<_>>();
    let input = CompiledSubgrammarInput {
        placeholder_terminal: matching_terminals[0],
        additional_placeholder_terminals: &matching_terminals[1..],
        constraint: child.as_ref(),
    };
    let mut composition = crate::compiler::constraint_compose::compose_constraints_owned_parent_segmented_shared(
        parent,
        std::slice::from_ref(&input),
        std::slice::from_ref(&child),
        vocab,
        boundary_backend,
    )
    .map_err(Error::Compilation)?;
    composition.constraint.late_grammar_slots = remaining_parent_slots;
    let terminal_offset = composition.terminal_offsets[1];
    composition.constraint.late_grammar_slots.extend(
        child
            .late_grammar_slots
            .iter()
            .map(|slot| crate::runtime::LateGrammarSlot {
                name: format!("{name}::{}", slot.name),
                terminal_id: terminal_offset + slot.terminal_id,
            }),
    );
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

fn constraint_vocab(constraint: &RuntimeConstraint) -> Vocab {
    if let Some(vocab) = constraint.late_bind_vocab.get() {
        return vocab.clone();
    }
    // Compiler-created constraints retain the canonical token-byte map and can
    // expose it as an O(1) Vocab view. Loaded artifacts deliberately keep token
    // bytes packed; only the uncommon path with no compiled child to borrow a
    // vocabulary from has to materialize that packed map here.
    let vocab = if !constraint.token_bytes.is_empty() || constraint.packed_token_bytes.is_none() {
        Vocab::from_entries_arc(Arc::clone(&constraint.token_bytes))
    } else {
        Vocab::new(
        constraint
            .token_bytes_iter()
            .map(|(token_id, bytes)| (token_id, bytes.to_vec()))
            .collect(),
        )
    };
    let _ = constraint.late_bind_vocab.set(vocab.clone());
    vocab
}

fn binding_vocab(binding: &GrammarBinding<'_>, parent: &RuntimeConstraint) -> Vocab {
    match binding {
        GrammarBinding::StaticBorrowed(child) => constraint_vocab(child),
        GrammarBinding::StaticOwned(child) => constraint_vocab(child),
        // Source/spec children need a vocabulary before they can be compiled.
        // Dynamic bindings may contain alternatives with packed storage, so
        // keep the generic parent-derived fallback for those less common paths.
        GrammarBinding::Source(_)
        | GrammarBinding::Spec(_)
        | GrammarBinding::DynamicBorrowed(_)
        | GrammarBinding::DynamicOwned(_) => constraint_vocab(parent),
    }
}

fn require_late_grammar_slot(parents: &[RuntimeConstraint], name: &str) -> Result<()> {
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
    mode: ChildCompileMode,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Vec<(String, RuntimeConstraint)>>
where
    T: IntoGrammarBinding<'a>,
{
    let mut binding = child.into_grammar_binding();
    binding.bind_target(vocab, name)?;
    let compiled = binding.compile(vocab, mode)?;
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
) -> Result<RuntimeConstraint>
where
    T: IntoGrammarBinding<'a>,
{
    require_late_grammar_slot(std::slice::from_ref(parent), name)?;
    let mut binding = child.into_grammar_binding();
    // Prefer a supplied compiled child's canonical vocabulary Arc. A loaded
    // parent stores token bytes packed; reconstructing a 100k+ entry Vocab from
    // it on every bind can dwarf the link itself.
    let vocab = binding_vocab(&binding, parent);
    binding.bind_target(&vocab, name)?;
    if !parent.token_bytes_match_vocab(&vocab) {
        return Err(Error::Compilation(format!(
            "external grammar {name:?} was built for an incompatible vocabulary",
        )));
    }
    let mut owned_parent = parent.clone();
    owned_parent
        .bind_vocab_exact(&vocab)
        .map_err(Error::Compilation)?;

    // Static late linking keeps retained component parser artifacts and
    // publishes an exact deterministic boundary parser B. DynamicConstraint
    // uses the symbolic terminal-NWA backend instead; keeping static B
    // deterministic preserves the low per-token runtime of Constraint.
    let boundary_backend = SegmentedBoundaryBackend::StaticParserDwa;

    // Preserve ownership when the caller gives us an owned/static compiled
    // child. The segmented runtime stores Arc<Constraint>, so moving an owned
    // child (or cloning an existing Arc) should never deep-copy its artifact.
    match binding {
        GrammarBinding::StaticOwned(shared) => compose_named_shared_child(
            owned_parent,
            name,
            shared,
            &vocab,
            boundary_backend,
        ),
        GrammarBinding::Source(grammar) => {
            let spec = ConstraintSpec::builder(grammar, &vocab)?.build()?;
            let child = Arc::new(spec.compile_static_uncached()?);
            compose_named_shared_child(owned_parent, name, child, &vocab, boundary_backend)
        }
        GrammarBinding::Spec(spec) => {
            let child = Arc::new(spec.compile_static_uncached()?);
            compose_named_shared_child(owned_parent, name, child, &vocab, boundary_backend)
        }
        GrammarBinding::StaticBorrowed(constraint) => {
            // A borrowed child cannot outlive this call, while the returned
            // composed constraint must own it. Pay exactly one ownership copy,
            // then retain that copy by Arc instead of cloning it again inside
            // segmented publication.
            let child = Arc::new(constraint.clone());
            compose_named_shared_child(owned_parent, name, child, &vocab, boundary_backend)
        }
        binding @ (GrammarBinding::DynamicBorrowed(_) | GrammarBinding::DynamicOwned(_)) => {
            let compiled = binding.compile(&vocab, ChildCompileMode::Static)?;
            let children = prepare_compiled_children(
                vec![(name.to_owned(), compiled)],
                &vocab,
                boundary_backend,
            )?;
            compose_named_children(owned_parent, &children, &vocab, boundary_backend)
        }
    }
}

fn bind_dynamic_parent_grammar<'a, T>(
    parent: &DynamicConstraint,
    name: &str,
    child: T,
) -> Result<DynamicConstraint>
where
    T: IntoGrammarBinding<'a>,
{
    let parents = parent.clone_constraints();
    require_late_grammar_slot(&parents, name)?;
    let first = parents
        .first()
        .ok_or_else(|| Error::Compilation("dynamic parent has no alternatives".to_owned()))?;
    let vocab = constraint_vocab(first);
    if !parents
        .iter()
        .all(|alternative| static_constraint_targets(alternative, &vocab))
    {
        return Err(Error::Compilation(
            "dynamic parent alternatives target incompatible vocabularies".to_owned(),
        ));
    }
    let boundary_backend = SegmentedBoundaryBackend::DynamicTerminalNwa;
    let children = prepare_late_child(
        name,
        child,
        &vocab,
        ChildCompileMode::Dynamic,
        boundary_backend,
    )?;
    let alternatives = parents
        .into_iter()
        .map(|alternative| {
            compose_named_children(alternative, &children, &vocab, boundary_backend)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(DynamicConstraint::from_constraints(alternatives))
}

impl RuntimeConstraint {
    /// Compile `grammar` into a [`Constraint`](crate::Constraint) for `vocab`.
    pub fn compile(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile()
    }

    /// Bind one unresolved `extern grammar` slot in an already-compiled parent.
    ///
    /// The parent and child remain intact runtime components. Linking compiles
    /// only the composed coordinates and cross-component boundary behavior.
    #[allow(private_bounds)]
    pub fn bind_grammar<'a, T>(&self, name: impl AsRef<str>, child: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_static_parent_grammar(self, name.as_ref(), child)
    }
}

impl DynamicConstraint {
    /// Compile `grammar` into a [`DynamicConstraint`] for `vocab`.
    pub fn compile(grammar: Grammar<'_>, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile_dynamic()
    }

    /// Bind one unresolved `extern grammar` slot in an already-compiled dynamic
    /// parent while preserving dynamic component-local masking.
    #[allow(private_bounds)]
    pub fn bind_grammar<'a, T>(&self, name: impl AsRef<str>, child: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        bind_dynamic_parent_grammar(self, name.as_ref(), child)
    }
}

#[cfg(test)]
mod hybrid_tests {
    use super::*;

    fn component_backend_flags(constraint: &RuntimeConstraint) -> Vec<bool> {
        constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("hybrid composition must publish segmented runtime metadata")
            .segmented_parser_components
            .iter()
            .map(|component| component.constraint().uses_dynamic_runtime())
            .collect()
    }

    fn assert_static_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("hybrid composition must publish segmented runtime metadata");
        assert!(overlay.segmented_boundary_parser.is_some());
        assert!(overlay.segmented_boundary_terminal_trie.is_none());
    }

    fn assert_dynamic_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint
            .static_dynamic_overlay
            .as_ref()
            .expect("hybrid composition must publish segmented runtime metadata");
        assert!(overlay.segmented_boundary_parser.is_none());
        let boundary = overlay
            .segmented_boundary_terminal_trie
            .as_ref()
            .expect("dynamic composition must publish a terminal-NWA boundary");
        assert!(boundary.symbolic_nwa.is_some());
    }

    #[test]
    fn unresolved_grammar_sentinel_is_not_a_model_token_coordinate() {
        // 32 model-token IDs make the first private sentinel ID 32, exactly one
        // word beyond a one-word public mask. If the sentinel enters the
        // runtime token quotient/cache this shape panics while building masks.
        let vocab = Vocab::new(
            (0u32..32)
                .map(|token| (token, vec![token as u8]))
                .collect(),
        );
        let parent = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start start; extern grammar child; nt start = child;"),
            &vocab,
        )
        .unwrap();
        assert!(parent.late_grammar_slots.iter().any(|slot| slot.name == "child"));
        assert!(
            parent.serialized_artifact_cache.is_none(),
            "compile must not eagerly serialize/cache the artifact",
        );
        assert!(
            parent
                .internal_token_to_tokens
                .iter()
                .flatten()
                .all(|&token| token < 32),
            "compiler-private linker sentinel leaked into the model-token coordinate",
        );
        assert_eq!(parent.mask_len(), 1);
        assert_eq!(parent.internal_token_count(), 1);
        assert_eq!(parent.internal_token_groups().unwrap().len(), 1);
        assert!(parent.internal_token_groups().unwrap()[0].is_empty());

        // Current artifacts omit the explicit inverse token map, but the
        // private empty linker class must survive through the serialized mask
        // cache dimensions so later composition sees the same compiler
        // coordinate without exposing a model token.
        let loaded = RuntimeConstraint::load(&parent.save()).unwrap();
        assert_eq!(loaded.internal_token_count(), 1);
        assert_eq!(loaded.internal_token_groups().unwrap().len(), 1);
        assert!(loaded.internal_token_groups().unwrap()[0].is_empty());
        assert_eq!(loaded.mask_len(), 1);
        assert_eq!(loaded.start().mask(), vec![0]);

        let mask = parent.start().mask();
        assert_eq!(mask.len(), 1);
        assert_eq!(mask[0], 0, "unresolved grammar slot must not be traversable");
    }

    #[test]
    fn segmented_composition_preserves_supplied_component_backends() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
        );
        let static_child =
            RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
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
        assert_eq!(
            component_backend_flags(&static_with_dynamic),
            vec![false, true]
        );
        assert_static_boundary(&static_with_dynamic);
        assert!(
            static_with_dynamic.serialized_artifact_cache.is_none(),
            "compiling a live hybrid must not eagerly staticify its dynamic leaf for save()",
        );
        let mut state = static_with_dynamic.start();
        state.commit_token(2).unwrap();
        assert!(state.is_accepting());

        let dynamic_with_static = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", &static_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_dynamic()
            .unwrap();
        for alternative in dynamic_with_static.clone_constraints() {
            assert_eq!(component_backend_flags(&alternative), vec![true, false]);
            assert_dynamic_boundary(&alternative);
        }
        let mut state = dynamic_with_static.start();
        state.commit_token(2).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn static_late_binding_reuses_owned_child_arc_and_static_boundary() {
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
        let child = Arc::new(
            RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap(),
        );
        let composed = parent
            .bind_grammar("child", Arc::clone(&child))
            .unwrap();
        assert_static_boundary(&composed);
        let overlay = composed
            .static_dynamic_overlay
            .as_ref()
            .expect("late binding must publish segmented runtime metadata");
        assert_eq!(overlay.segmented_parser_components.len(), 2);
        assert!(
            Arc::ptr_eq(
                overlay.segmented_parser_components[1].constraint_arc(),
                &child,
            ),
            "owned/Arc child must be retained without a deep artifact clone",
        );
        let mut state = composed.start();
        state.commit_token(2).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn loaded_static_late_bind_keeps_parent_parser_dwa_packed() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let compiled_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let loaded_parent = RuntimeConstraint::load(&compiled_parent.save()).unwrap();
        assert!(loaded_parent.packed_parser_dwa.is_some());
        let loaded_shell_states = loaded_parent.parser_dwa.num_states();

        let child = Arc::new(
            RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap(),
        );
        let composed = loaded_parent.bind_grammar("child", child).unwrap();
        let overlay = composed
            .static_dynamic_overlay
            .as_ref()
            .expect("late binding must publish segmented runtime metadata");
        let retained_parent = overlay.segmented_parser_components[0].constraint();
        assert!(
            retained_parent.packed_parser_dwa.is_some(),
            "segmented late binding must retain the loaded parent's packed parser DWA",
        );
        assert_eq!(
            retained_parent.parser_dwa.num_states(),
            loaded_shell_states,
            "segmented late binding must not replace the loaded parser-DWA shell with the expanded compiler DWA",
        );

        let mut state = composed.start();
        state.commit_token(2).unwrap();
        assert!(state.is_accepting());
    }

    #[test]
    fn dynamic_late_binding_can_recurse_through_rebased_slots() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
            (4, b"yz".to_vec()),
            (5, b"xyz".to_vec()),
        ]);
        let parent = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar leaf; nt start = \"y\" leaf;",
            ),
            &vocab,
        )
        .unwrap();
        let once = parent.bind_grammar("child", &child).unwrap();
        assert!(once.clone_constraints().iter().any(|alternative| {
            alternative
                .late_grammar_slots
                .iter()
                .any(|slot| slot.name == "child::leaf")
        }));

        let once = DynamicConstraint::load(&once.save()).unwrap();
        assert!(once.clone_constraints().iter().any(|alternative| {
            alternative
                .late_grammar_slots
                .iter()
                .any(|slot| slot.name == "child::leaf")
        }));

        let leaf = DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap();
        let twice = once.bind_grammar("child::leaf", &leaf).unwrap();
        assert!(twice
            .clone_constraints()
            .iter()
            .all(|alternative| alternative.late_grammar_slots.is_empty()));

        let mut fused = twice.start();
        fused.commit_token(5).unwrap();
        assert!(fused.is_accepting());

        let mut split = twice.start();
        split.commit_token(0).unwrap();
        split.commit_token(1).unwrap();
        split.commit_token(2).unwrap();
        assert!(split.is_accepting());
    }

    #[test]
    fn dynamic_boundary_preserves_scoped_component_ignores() {
        let vocab = Vocab::new(vec![
            (0, b" ".to_vec()),
            (1, b"\t".to_vec()),
            (2, b"<".to_vec()),
            (3, b">".to_vec()),
            (4, b"a".to_vec()),
            (5, b"b".to_vec()),
            (6, b"\ta".to_vec()),
            (7, b"b\t".to_vec()),
            (8, b"a b".to_vec()),
        ]);
        let parent = DynamicConstraint::compile(
            Grammar::glrm(
                r#"
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS ::= " "+;
                    extern grammar child;
                    nt document ::= "<" child ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let child = DynamicConstraint::compile(
            Grammar::glrm(
                r#"
                    start value;
                    ignore CHILD_WS;
                    t CHILD_WS ::= "\t"+;
                    nt value ::= "a" "b";
                "#,
            ),
            &vocab,
        )
        .unwrap();
        let composed = parent.bind_grammar("child", &child).unwrap();
        for alternative in composed.clone_constraints() {
            assert_dynamic_boundary(&alternative);
        }

        let inline = DynamicConstraint::compile(
            Grammar::glrm(
                r#"
                    start document;
                    ignore PARENT_WS;
                    t PARENT_WS ::= " "+;
                    g child ::= {
                        start value;
                        ignore CHILD_WS;
                        t CHILD_WS ::= "\t"+;
                        nt value ::= "a" "b";
                    };
                    nt document ::= "<" child ">";
                "#,
            ),
            &vocab,
        )
        .unwrap();

        for sequence in [
            &[2u32, 4, 5, 3][..],
            &[0, 2, 1, 4, 1, 5, 1, 3, 0][..],
            &[2, 6, 7, 3][..],
            &[2, 4, 0, 5, 3][..],
            &[1, 2, 4, 5, 3][..],
        ] {
            let mut actual = composed.start();
            let mut expected = inline.start();
            for &token in sequence {
                assert_eq!(
                    actual.mask(),
                    expected.mask(),
                    "mask differs before token {token} in {sequence:?}",
                );
                let actual_result = actual.commit_token(token).is_ok();
                let expected_result = expected.commit_token(token).is_ok();
                assert_eq!(
                    actual_result, expected_result,
                    "commit differs for token {token} in {sequence:?}",
                );
                if !expected_result {
                    break;
                }
            }
            assert_eq!(
                actual.is_accepting(),
                expected.is_accepting(),
                "acceptance differs for {sequence:?}",
            );
        }

        for bytes in [
            &b"<ab>"[..],
            &b" <\ta\t\tb\t> "[..],
            &b"<a\tb\t>"[..],
            &b"<a b>"[..],
            &b"\t<ab>"[..],
            &b"<\ta b>"[..],
        ] {
            let mut actual = composed.start();
            let mut expected = inline.start();
            assert_eq!(
                actual.commit_bytes(bytes).is_ok(),
                expected.commit_bytes(bytes).is_ok(),
                "commit result differs for {:?}",
                String::from_utf8_lossy(bytes),
            );
            assert_eq!(
                actual.is_accepting(),
                expected.is_accepting(),
                "acceptance differs for {:?}",
                String::from_utf8_lossy(bytes),
            );
        }
    }

    #[test]
    fn static_late_bind_does_not_eagerly_serialize() {
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
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
        let bound = parent.bind_grammar("child", &child).unwrap();
        assert!(
            bound.serialized_artifact_cache.is_none(),
            "late bind must not eagerly serialize/cache the composed artifact",
        );
    }

    #[test]
    fn static_hybrid_save_load_materializes_equivalent_snapshot() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"xyz".to_vec()),
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
        let live = parent.bind_grammar("child", &child).unwrap();
        assert!(crate::compiler::constraint_compose::segmented_constraint_retains_dynamic(&live));

        let loaded = RuntimeConstraint::load(&live.save()).unwrap();
        assert!(!crate::compiler::constraint_compose::segmented_constraint_retains_dynamic(&loaded));

        let mut live_state = live.start();
        let mut loaded_state = loaded.start();
        assert_eq!(live_state.mask(), loaded_state.mask());
        assert!(allowed_token(&live_state.mask(), 2));
        assert!(!allowed_token(&live_state.mask(), 3));
        live_state.commit_token(2).unwrap();
        loaded_state.commit_token(2).unwrap();
        assert_eq!(live_state.is_accepting(), loaded_state.is_accepting());

        let mut live_state = live.start();
        let mut loaded_state = loaded.start();
        live_state.commit_token(0).unwrap();
        loaded_state.commit_token(0).unwrap();
        assert_eq!(live_state.mask(), loaded_state.mask());
        live_state.commit_token(1).unwrap();
        loaded_state.commit_token(1).unwrap();
        assert_eq!(live_state.is_accepting(), loaded_state.is_accepting());
    }

    #[test]
    fn dynamic_hybrid_save_load_keeps_equivalent_dynamic_runtime() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
        let live = parent.bind_grammar("child", &child).unwrap();
        let loaded = DynamicConstraint::load(&live.save()).unwrap();

        let mut live_state = live.start();
        let mut loaded_state = loaded.start();
        assert_eq!(live_state.mask(), loaded_state.mask());
        live_state.commit_token(2).unwrap();
        loaded_state.commit_token(2).unwrap();
        assert!(live_state.is_accepting());
        assert!(loaded_state.is_accepting());
        assert!(loaded
            .clone_constraints()
            .iter()
            .all(RuntimeConstraint::uses_dynamic_runtime));
    }

    fn allowed_token(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word)
            .is_some_and(|value| value & (1u32 << bit) != 0)
    }

    #[test]
    fn open_dynamic_parent_slots_survive_artifact_roundtrip() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let parent = DynamicConstraint::compile(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        assert!(
            parent
                .clone_constraints()
                .iter()
                .any(|alternative| alternative.late_grammar_slots.iter().any(|slot| slot.name == "child"))
        );

        let loaded = DynamicConstraint::load(&parent.save()).unwrap();
        let child =
            DynamicConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
        let bound = loaded.bind_grammar("child", &child).unwrap();
        let mut state = bound.start();
        assert!(state.commit_token(2).is_ok());
        assert!(state.is_accepting());
    }
}
