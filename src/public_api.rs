#[cfg(test)]
use crate::test_modes::MaskMode;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock};
use snapshot_map::SnapshotMap as OrdMap;
use crate::grammar::glrm::{ExternKind, GlrmExternalDeclarations, ParsedComponent};
use glrmask_grammar::{LarkDocument,LarkExternKind,LarkTokenSet};
use std::time::Instant;

use rayon::prelude::*;

use crate::compiler::constraint_compose::{
    CompiledSubgrammarInput, SegmentedBoundaryBackend,
    compose_constraints_owned_parent_segmented_shared,
};
use crate::runtime::Constraint as RuntimeConstraint;
use crate::dynamic_constraint::DynamicConstraint;
use crate::runtime::BoundaryTriggerDetail;
use crate::{Error, TokenSet, Result, Vocab};

mod graph;
mod snapshot_map;
mod event_cache;
mod prepared_artifact;
mod artifact_limits;
pub(crate) mod module;
mod expr;
mod stack;
pub use module::{Module, CompileOptions, Binding};
pub use expr::Expr;
pub(crate) use stack::with_deep_stack;
#[allow(unused_imports)]
pub(crate) use stack::{DEEP_STACK_RED_ZONE, DEEP_STACK_SIZE};

/// How masks are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum MaskStrategy {
    /// Reuses any precompiled work; otherwise [`ParserDwa`](Self::ParserDwa), or
    /// [`TokenTrie`](Self::TokenTrie) when that would be too large. The default.
    #[default]
    Auto,
    /// Builds a parser DWA for the grammar. Slowest to compile, with the fastest
    /// masks.
    ParserDwa,
    /// Computes each mask by walking the vocabulary. Fastest to compile, with
    /// slower masks. See [`TokenTrie`](crate::TokenTrie) for its settings.
    TokenTrie(TokenTrie),
}

impl MaskStrategy {
    /// Same as [`TokenTrie`](Self::TokenTrie) with the default settings.
    pub const FAST_COMPILE: Self = Self::TokenTrie(TokenTrie::new());
    /// Same as [`ParserDwa`](Self::ParserDwa).
    pub const FAST_MASKS: Self = Self::ParserDwa;
}

impl From<TokenTrie> for MaskStrategy {
    fn from(settings: TokenTrie) -> Self {
        Self::TokenTrie(settings)
    }
}

/// Settings for [`MaskStrategy::TokenTrie`].
///
/// ```
/// # use glrmask::{CompileOptions, ParserBackend, TokenTrie};
/// let options = CompileOptions::default().mask_strategy(
///     TokenTrie::new()
///         .parser_backend(ParserBackend::TemplateDfa)
///         .merge_equivalent_tokens(true),
/// );
/// # let _ = options;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TokenTrie {
    parser_backend: ParserBackend,
    merge_equivalent_tokens: bool,
}

impl TokenTrie {
    /// The default settings: an LR table, with tokens not merged.
    pub const fn new() -> Self {
        Self {
            parser_backend: ParserBackend::LrTable,
            merge_equivalent_tokens: false,
        }
    }

    /// Sets how the parser is represented. Defaults to [`ParserBackend::LrTable`].
    pub const fn parser_backend(mut self, backend: ParserBackend) -> Self {
        self.parser_backend = backend;
        self
    }

    /// Sets whether tokens that the grammar treats the same are merged, so masks
    /// walk fewer tokens. Off by default.
    ///
    /// Tokens are not merged in a composition that includes precompiled parts.
    pub const fn merge_equivalent_tokens(mut self, enable: bool) -> Self {
        self.merge_equivalent_tokens = enable;
        self
    }

    pub(crate) const fn backend(&self) -> ParserBackend {
        self.parser_backend
    }

    pub(crate) const fn merges(&self) -> bool {
        self.merge_equivalent_tokens
    }
}

/// How masks are computed at grammar boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum BoundaryMaskStrategy {
    /// [`ParserDwa`](Self::ParserDwa), or [`TokenTrie`](Self::TokenTrie) when that
    /// would be too large. The default.
    #[default]
    Auto,
    /// Builds a parser DWA across each boundary. Slower to compile, with faster
    /// masks. Fails with [`ErrorKind::ResourceLimit`](crate::ErrorKind::ResourceLimit)
    /// if it would be too large.
    ParserDwa,
    /// Walks the vocabulary at each boundary as needed. Faster to compile, with
    /// slower masks.
    TokenTrie,
}

impl BoundaryMaskStrategy {
    /// Same as [`TokenTrie`](Self::TokenTrie).
    pub const FAST_COMPILE: Self = Self::TokenTrie;
    /// Same as [`ParserDwa`](Self::ParserDwa).
    pub const FAST_MASKS: Self = Self::ParserDwa;
}

/// Internal mask engine. The wire tags are unchanged so `GLRASM03` module
/// artifacts keep their meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub(crate) enum Engine {
    /// Reuses retained work; otherwise [`ParserDwa`](Self::ParserDwa).
    #[default]
    Auto,
    ParserDwa,
    /// A retained [`TokenTrie`](Self::TokenTrie) template-DFA shell ("B shell").
    /// An LR-table TokenTrie build retains nothing, so it has no key.
    TokenTrie,
    /// [`TokenTrie`](Self::TokenTrie) over the merged vocabulary. Key 3 is
    /// retained only for a template-DFA body (combination C); an LR-table
    /// merged body (combination D) retains nothing, exactly like an unmerged
    /// LR TokenTrie build.
    MergedTokenTrie,
}

impl Engine {
    pub(crate) fn key(self) -> u8 {
        match self {
            Self::Auto => 0,
            Self::ParserDwa => 1,
            Self::TokenTrie => 2,
            Self::MergedTokenTrie => 3,
        }
    }

    pub(crate) fn is_token_trie(self) -> bool {
        matches!(self, Self::TokenTrie | Self::MergedTokenTrie)
    }
}

/// A named placeholder in a grammar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extern {
    pub(crate) name: String,
    pub(crate) kind: ExternKind,
}

impl Extern {
    /// The extern's name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// What the extern can be bound to.
    pub fn kind(&self) -> ExternKind {
        self.kind
    }
}

/// A grammar's or module's externs, in declaration order.
#[derive(Debug, Clone)]
pub struct Externs<'a> {
    pub(crate) source: ExternsSource<'a>,
    pub(crate) only_unbound: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum ExternsSource<'a> {
    Module(&'a Module),
    Grammar(&'a Grammar),
}

impl ExternsSource<'_> {
    fn declarations(&self) -> Option<&GlrmExternalDeclarations> {
        match self {
            Self::Module(module) => module.node.source.declarations().ok(),
            Self::Grammar(grammar) => grammar.declarations().ok(),
        }
    }
    fn bound(&self, name: &str) -> bool {
        match self {
            Self::Module(module) => module.node.bindings.contains_key(name),
            Self::Grammar(grammar) => grammar.bindings.contains_key(name),
        }
    }
}

impl<'a> Externs<'a> {
    /// The extern named `name`, if any.
    pub fn get(&self, name: &str) -> Option<Extern> {
        let declared = self.source.declarations()?;
        if self.only_unbound && self.source.bound(name) {
            return None;
        }
        let kind = *declared.kinds.get(name)?;
        Some(Extern {
            name: name.to_string(),
            kind,
        })
    }

    /// The number of externs.
    pub fn len(&self) -> usize {
        let Some(declared) = self.source.declarations() else { return 0; };
        if self.only_unbound {
            declared.names.iter().filter(|n| !self.source.bound(n)).count()
        } else {
            declared.names.len()
        }
    }

    /// Whether there are no externs.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The externs not yet bound.
    pub fn unbound(&self) -> Externs<'a> {
        Externs {
            source: self.source.clone(),
            only_unbound: true,
        }
    }

    /// The externs, in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = Extern> + '_ {
        let items: Vec<Extern> = match self.source.declarations() {
            Some(declared) => declared
                .names
                .iter()
                .filter(|name| !self.only_unbound || !self.source.bound(name))
                .filter_map(|name| {
                    let kind = *declared.kinds.get(name)?;
                    Some(Extern {
                        name: name.clone(),
                        kind,
                    })
                })
                .collect(),
            None => Vec::new(),
        };
        items.into_iter()
    }
}

impl<'a> IntoIterator for Externs<'a> {
    type Item = Extern;
    type IntoIter = std::vec::IntoIter<Extern>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}

impl<'a> IntoIterator for &'a Externs<'a> {
    type Item = Extern;
    type IntoIter = std::vec::IntoIter<Extern>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter().collect::<Vec<_>>().into_iter()
    }
}

/// How the parser is represented and run when masks are computed with
/// [`MaskStrategy::TokenTrie`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ParserBackend {
    /// An LR table. Faster to compile, with slower masks. The default.
    #[default]
    LrTable,
    /// Template DFAs. Slower to compile, with faster masks.
    TemplateDfa,
}

/// Final compile/link options. `prepare_with` retains only the boundary metadata request; choose optimization and end tokens at final link.
#[derive(Debug, Clone)]
pub struct BuildOptions {
    end_tokens: Vec<u32>,
    optimization: MaskStrategy,
    engine_backend: ParserBackend,
    merge_equivalent_tokens: bool,
    parser_backend: ParserBackend,
    boundary_trigger: Option<BoundaryTriggerDetail>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            end_tokens: Vec::new(),
            optimization: MaskStrategy::default(),
            engine_backend: ParserBackend::LrTable,
            merge_equivalent_tokens: false,
            parser_backend: ParserBackend::TemplateDfa,
            boundary_trigger: None,
        }
    }
}

impl BuildOptions {
    /// Root stopping controls, allowed only at acceptance. Commit the selected end token before stopping; the policy is not inherited by embedded children.
    pub fn end_tokens(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        self.end_tokens = ids.into_iter().collect();
        self
    }

    /// Defaults to [`MaskStrategy::Auto`].
    pub fn optimization(mut self, optimization: MaskStrategy) -> Self {
        self.optimization = optimization;
        self
    }

    /// Parses with the requested parser backend.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn engine_backend(mut self, backend: ParserBackend) -> Self {
        self.engine_backend = backend;
        self
    }

    /// Merges tokens that the grammar treats the same, so masks walk fewer
    /// tokens. Off by default.
    pub fn merge_equivalent_tokens(mut self, enable: bool) -> Self {
        self.merge_equivalent_tokens = enable;
        self
    }

    /// Optional composition metadata. At final link an explicit request, including `None`, overrides the prepared request.
    pub fn boundary_trigger(mut self, detail: BoundaryTriggerDetail) -> Self {
        self.boundary_trigger = Some(detail);
        self
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn parser_backend(mut self, backend: ParserBackend) -> Self {
        assert!(backend != ParserBackend::LrTable,
            "LR-BACKED CONSTRAINT REQUEST IS FORBIDDEN: only native template Constraints may be materialized");
        self.parser_backend = backend; self
    }

    pub(crate) fn end_token_ids(&self) -> &[u32] {
        &self.end_tokens
    }

    /// The internal engine for these options.
    pub(crate) fn engine(&self) -> Result<Engine> {
        Ok(engine_for(self.optimization))
    }

    /// The TokenTrie parser backend for these options.
    pub(crate) fn token_trie_backend(&self) -> crate::runtime::parser_backend::DynamicParserBackend {
        token_trie_backend(self.engine_backend)
    }
}

/// Derive the internal engine from the public strategy.
pub(crate) fn engine_for(strategy: MaskStrategy) -> Engine {
    match strategy {
        MaskStrategy::Auto => Engine::Auto,
        MaskStrategy::ParserDwa => Engine::ParserDwa,
        MaskStrategy::TokenTrie(settings) if settings.merges() => Engine::MergedTokenTrie,
        MaskStrategy::TokenTrie(_) => Engine::TokenTrie,
    }
}

pub(crate) fn token_trie_backend(
    parser_backend: ParserBackend,
) -> crate::runtime::parser_backend::DynamicParserBackend {
    use crate::runtime::parser_backend::DynamicParserBackend;
    if parser_backend == ParserBackend::TemplateDfa {
        DynamicParserBackend::Native
    } else {
        DynamicParserBackend::LrTable
    }
}

impl Engine {
    /// Link options that reproduce this engine for an already prepared body.
    pub(crate) fn build_options(self) -> BuildOptions {
        let (optimization, engine_backend, merge_equivalent_tokens) = match self {
            Self::Auto => (MaskStrategy::Auto, ParserBackend::LrTable, false),
            Self::ParserDwa => (MaskStrategy::ParserDwa, ParserBackend::LrTable, false),
            Self::TokenTrie => (MaskStrategy::FAST_COMPILE, ParserBackend::LrTable, false),
            Self::MergedTokenTrie => (
                MaskStrategy::TokenTrie(
                    TokenTrie::new()
                        .parser_backend(ParserBackend::TemplateDfa)
                        .merge_equivalent_tokens(true),
                ),
                ParserBackend::TemplateDfa,
                true,
            ),
        };
        BuildOptions {
            optimization,
            engine_backend,
            merge_equivalent_tokens,
            parser_backend: ParserBackend::TemplateDfa,
            ..BuildOptions::default()
        }
    }
}

/// Vocabulary-specific compiled machinery, possibly with unresolved slots.
///
/// Binding changes only this handle; clones and attached children retain independent snapshots.
#[derive(Debug, Clone)]
pub struct PreparedGrammar {
    inner: Arc<RuntimeConstraint>,
    /// Compiled-only immutable attachments. Delaying composition is what lets
    /// the terminal link call choose the boundary/runtime trade-off.
    bindings: OrdMap<String, ModuleBinding>,
    /// Immutable-instance certificate; clones share it until a binding changes.
    analysis: graph::SummaryCache,
    event_adapters: Arc<event_cache::ExactAdapterCache>,
    boundary_trigger: BoundaryTriggerDetail,
    /// Authored interface, independent of optimization and runtime reachability.
    slot_kinds: Arc<BTreeMap<String, ExternKind>>,
    slot_spans: Arc<BTreeMap<String,glrmask_grammar::SourceSpan>>,
    /// Vocabulary-free artifact bytes retained by a deferred Module load.
    /// `compile`/`precompile` rehydrates from these with the supplied vocabulary.
    pub(crate) retained_artifact: Option<Arc<[u8]>>,
    /// Fingerprint recorded by `retained_artifact`.
    pub(crate) retained_vocab_fingerprint: Option<[u8; 32]>,
}

/// Unstable spelling retained only for repository tooling. The public type is
/// PreparedGrammar; normal dependency builds expose no compatibility alias.
#[cfg(any(test,feature="internal-api"))]
#[doc(hidden)]
pub type UnlinkedConstraint = PreparedGrammar;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Requirement {
    pub path: Vec<String>,
    pub kind: ExternKind,
    pub source_span: Option<glrmask_grammar::SourceSpan>,
}
impl Requirement {
    pub fn name(&self) -> String { self.path.join("::") }
}

#[derive(Debug, Clone)]
enum ModuleBinding {
    Module(Arc<PreparedGrammar>),
    Constraint(Arc<RuntimeConstraint>),
    TokenSet(Arc<event_cache::ExactEvent>),
}

#[derive(Debug, Clone)]
pub(crate) enum GrammarValue {
    Source(Arc<Grammar>),
    TokenSet(TokenSet),
}


#[doc(hidden)]
pub(crate) trait IntoGrammarValue {
    fn into_grammar_value(self) -> Result<GrammarValue>;
}

#[derive(Debug, Clone)]
pub(crate) enum UnlinkedValue<'a> {
    Prepared(Arc<PreparedGrammar>),
    StaticBorrowed(&'a RuntimeConstraint),
    StaticOwned(Arc<RuntimeConstraint>),
    TokenSet(TokenSet),
}

#[doc(hidden)]
pub(crate) trait IntoUnlinkedValue<'a> {
    fn into_unlinked_value(self) -> UnlinkedValue<'a>;
}

fn lark_declarations(document:&LarkDocument)->impl Iterator<Item=(String,ExternKind)>+'_ {
    document.externs.iter().map(|e|(e.name.clone(),match e.kind {
        LarkExternKind::Token=>ExternKind::Token,
        LarkExternKind::Grammar=>ExternKind::Grammar,
    }))
}

fn attached_path<'n,T:Clone>(name:&'n str, bindings:&OrdMap<String,T>) -> Option<(&'n str,&'n str)> {
    // Look up only the qualified prefixes, never scan every unrelated binding.
    name.rmatch_indices("::").find_map(|(index,_)| {
        let head=&name[..index]; let tail=&name[index+2..];
        (!tail.is_empty() && bindings.contains_key(head)).then_some((head,tail))
    })
}
fn check_binding_kind(name:&str,kind:ExternKind,grammar:bool)->Result<()> {
    if matches!((kind,grammar),(ExternKind::Token,true)) {
        return Err(Error::Compilation(format!("extern {name:?} is a token extern; bind it to a token ID or token set")));
    }
    Ok(())
}

/// A grammar, written as source text or built from an [`Expr`].
#[derive(Debug, Clone)]
pub struct Grammar {
    source: GrammarSource,
    module_snapshot: Arc<OnceLock<Module>>,
    bindings: OrdMap<String, GrammarValue>,
    declarations_cache: Arc<OnceLock<Result<GlrmExternalDeclarations>>>,
    analysis: graph::SummaryCache,
    parsed_component: Arc<OnceLock<std::result::Result<ParsedComponent, Error>>>,
    exact_value: Option<TokenSet>,
    lark_document:Arc<OnceLock<std::result::Result<LarkDocument,Error>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum GrammarSource {
    Ebnf(Arc<str>),
    Lark(Arc<str>),
    JsonSchema(Arc<str>),
    Glrm(Arc<str>),
    Constructed,
}

#[derive(Default)]
struct SourcePreparation {
    prepared: std::collections::HashMap<usize,Arc<PreparedGrammar>>,
    retained: Vec<Arc<Grammar>>,
    visited: usize,
}

impl Grammar {

    /// Resolve the parser-facing expression graph without selecting an LR
    /// normal form. Source bindings are not part of this constructor boundary;
    /// use a closed grammar or the ordinary compiled-component linker.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn prepare_parser_grammar(&self) -> Result<crate::template_parser::PreparedParserGrammar> {
        if !self.bindings.is_empty() {
            return Err(Error::Compilation("custom parser construction requires a source without external bindings".into()));
        }
        let (source, parse): (&str, fn(&str) -> Result<glrmask_grammar::NamedGrammar>) = match &self.source {
            GrammarSource::Ebnf(source) => (source, crate::import::parse_ebnf_to_named),
            GrammarSource::Lark(source) => (source, crate::import::parse_lark_to_named),
            GrammarSource::JsonSchema(source) => (source, crate::import::parse_json_schema_to_named),
            GrammarSource::Glrm(source) => (source, crate::import::parse_glrm_to_named),
            GrammarSource::Constructed => return crate::template_parser::PreparedParserGrammar::from_named(&self.component()?.lower_closed()?),
        };
        crate::import::with_large_import_stack(source.len(), || {
            crate::template_parser::PreparedParserGrammar::from_named(&parse(source)?)
        })
    }

    /// Resolve this grammar, invoke a compile-time parser constructor once, and
    /// assemble its template relations using the ordinary masking machinery.
    /// For repeated experiments, retain `prepare_parser_grammar()` instead.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn compile_with_parser(&self, vocab: &Vocab,
        compiler: &(impl crate::template_parser::ParserCompiler + ?Sized),
        options: crate::template_parser::TemplateBuildOptions) -> Result<crate::Constraint> {
        self.prepare_parser_grammar()?.compile_parser(compiler)?.compile_with(vocab, options)
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn ebnf(source: &str) -> Self { Self::new(GrammarSource::Ebnf(Arc::from(source))) }
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn lark(source: &str) -> Self { Self::new(GrammarSource::Lark(Arc::from(source))) }
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn json_schema(source: &str) -> Self { Self::new(GrammarSource::JsonSchema(Arc::from(source))) }
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn glrm(source: &str) -> Self {
        let grammar=Self::new(GrammarSource::Glrm(Arc::from(source)));
        let _=grammar.parsed_component.set(ParsedComponent::parse_glrm_for_tools(source).map_err(Into::into));
        grammar
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn from_ebnf(source: impl AsRef<str>) -> Result<Self> {
        let grammar = Self::new(GrammarSource::Ebnf(Arc::from(source.as_ref())));
        grammar.component()?;
        Ok(grammar)
    }

    /// Creates a grammar from Lark source.
    pub fn from_lark(source: impl AsRef<str>) -> Result<Self> {
        let grammar = Self::new(GrammarSource::Lark(Arc::from(source.as_ref())));
        grammar.lark_document()?;
        grammar.declarations()?;
        Ok(grammar)
    }


    /// Creates a grammar from JSON Schema source.
    pub fn from_json_schema(source: impl AsRef<str>) -> Result<Self> {
        let source = source.as_ref();
        crate::import::parse_json_schema_to_named(source)?;
        Ok(Self::new(GrammarSource::JsonSchema(Arc::from(source))))
    }

    /// Creates a grammar from GLRM source.
    pub fn from_glrm(source: impl AsRef<str>) -> Result<Self> {
        let grammar = Self::new(GrammarSource::Glrm(Arc::from(source.as_ref())));
        grammar.component()?;
        Ok(grammar)
    }

    fn new(source: GrammarSource) -> Self {
        Self { source, module_snapshot: Arc::new(OnceLock::new()), bindings: OrdMap::new(), declarations_cache: Arc::new(OnceLock::new()), analysis: graph::empty_cache(), parsed_component: Arc::new(OnceLock::new()), exact_value: None, lark_document:Arc::new(OnceLock::new()) }
    }

    pub(crate) fn single_rule_body(&self) -> Option<&crate::grammar::ast::GrammarExpr> {
        self.component().ok()?.single_rule_body()
    }

    fn component(&self) -> Result<&ParsedComponent> {
        self.parsed_component.get_or_init(|| match &self.source {
            GrammarSource::Glrm(source) => ParsedComponent::parse_glrm(source).map_err(Into::into),
            GrammarSource::Ebnf(source) => crate::import::parse_ebnf_to_named(source)
                .and_then(|n|ParsedComponent::from_named(n,&[]).map_err(Into::into)),
            GrammarSource::Lark(source) => crate::import::parse_lark_to_named(source)
                .and_then(|n|ParsedComponent::from_named(n,&[]).map_err(Into::into)),
            // JSON Schema lowers to an ordinary named grammar, so it can join a
            // mixed compile as a dynamic component shell. The schema pipeline
            // prepares it exactly as the static schema route does, including
            // lexer partitioning for the generated terminal classes.
            GrammarSource::JsonSchema(source) => crate::import::prepare_json_schema_named_for_component(source)
                .and_then(|n|ParsedComponent::from_named(n,&[]).map_err(Into::into)),
            GrammarSource::Constructed => Err(Error::Compilation("constructed grammar has no semantic description".into())),
        }).as_ref().map_err(Clone::clone)
    }

    fn lark_document(&self)->Result<&LarkDocument> {
        let GrammarSource::Lark(source)=&self.source else{return Err(Error::Compilation("not a Lark description".into()));};
        self.lark_document.get_or_init(||glrmask_grammar::parse_lark_document(source).map_err(Into::into))
            .as_ref().map_err(Clone::clone)
    }

    fn resolve_lark(&self,vocab:&Vocab)->Result<Self> {
        use range_set_blaze::RangeSetBlaze;
        let GrammarSource::Lark(source)=&self.source else{return Err(Error::Compilation("not a Lark description".into()));};
        let document=self.lark_document()?;
        let domain=RangeSetBlaze::from_iter(vocab.byte_token_ranges());
        let mut values=std::collections::HashMap::new();
        for (name,set) in &document.events {
            // A named atom is a token extern; its binding supplies the IDs.
            if let LarkTokenSet::Named(external)=set {
                values.insert(name.clone(),glrmask_grammar::GrammarExpr::Ref(external.clone()));
                continue;
            }
            let ranges=(|| -> Result<RangeSetBlaze<u32>> { Ok(match set {
                LarkTokenSet::Any=>domain.clone(),
                LarkTokenSet::Ranges{ranges,negate}=>{
                    for &(start,end) in ranges {
                        if !vocab.contains_token_range(start,end){return Err(Error::Compilation(format!("token range {start}..={end} includes IDs that are not in the vocabulary")));}
                    }
                    let selected=RangeSetBlaze::from_iter(ranges.iter().map(|&(a,b)|a..=b));
                    if *negate {
                        &domain-&selected
                    } else {
                        crate::import::reject_special_token_ids(&selected, vocab)?;
                        selected
                    }
                }
                LarkTokenSet::Named(_)=>unreachable!("handled above"),
            }) })().map_err(|e|e.with_source_span(document.source_spans.get(name).copied()))?;
            let mut atoms=ranges.ranges().map(|range|if range.start()==range.end(){
                glrmask_grammar::GrammarExpr::SpecialToken(*range.start())
            }else{glrmask_grammar::GrammarExpr::SpecialTokenRange(*range.start(),*range.end())}).collect::<Vec<_>>();
            let expression=if atoms.len()==1{atoms.pop().unwrap()}else{glrmask_grammar::GrammarExpr::Choice(atoms)};
            values.insert(name.clone(),expression);
        }
        let named=document.resolve_events(&values);
        let externs=lark_declarations(document)
            .chain(document.schemas.iter().map(|(name,_)|(name.clone(),ExternKind::Grammar))).collect::<Vec<_>>();
        let mut resolved=Self::new(GrammarSource::Constructed);
        let spans=externs.iter().filter_map(|(name,_)|document.source_spans.get(name).map(|span|(name.clone(),*span))).collect();
        let _=resolved.parsed_component.set(ParsedComponent::from_named_with_ignores(named,&externs,&document.resolved_additional_ignores())
            .and_then(|component|component.with_source_spans(&spans)).map_err(Into::into));
        let _=resolved.component()?;
        resolved.bindings=self.bindings.clone();
        for (name,range) in &document.schemas {
            // Preserve a real child boundary and the JSON frontend's own whitespace
            // rules. The source span borrows the original Lark document safely.
            let child=Self::new(GrammarSource::JsonSchema(Arc::from(&source[range.clone()])));
            resolved.bindings.insert(name.clone(),GrammarValue::Source(Arc::new(child)));
        }
        Ok(resolved)
    }

    fn declarations(&self) -> Result<&GlrmExternalDeclarations> {
        self.declarations_cache.get_or_init(|| match &self.source {
            GrammarSource::Glrm(_) | GrammarSource::Constructed => Ok(self.component()?.declarations().clone()),
            GrammarSource::Lark(_)=>{
                let document=self.lark_document()?;
                let declared=lark_declarations(document).collect::<Vec<_>>();
                let kinds=declared.iter().cloned().collect::<BTreeMap<_,_>>();
                let spans=declared.iter().filter_map(|(name,_)|document.source_spans.get(name).map(|span|(name.clone(),*span))).collect();
                Ok(GlrmExternalDeclarations{names:declared.into_iter().map(|(name,_)|name).collect(),kinds,spans})
            }
            _ => Ok(GlrmExternalDeclarations::default()),
        }).as_ref().map_err(Clone::clone)
    }

    fn validate_exact_target(&self, vocab:&Vocab) -> Result<()> {
        if let Some(set) = &self.exact_value
            && let Some(id) = vocab.first_unknown_token_id(set)
        {
            return Err(Error::Compilation(format!(
                "token ID {id} is not in the vocabulary",
            )));
        }
        Ok(())
    }

    fn validate_ebnf(&self, vocab: &Vocab) -> Result<()> {
        if let GrammarSource::Ebnf(source) = &self.source {
            crate::import::validate_ebnf_source(source, vocab)?;
        }
        Ok(())
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn exact(tokens: TokenSet) -> Self { Self::from_token_set(tokens) }

    pub(crate) fn from_token_set(tokens: TokenSet) -> Self {
        let exprs=tokens.range_pairs().iter().map(|&(a,b)|if a==b {
            glrmask_grammar::GrammarExpr::SpecialToken(a)
        } else {glrmask_grammar::GrammarExpr::SpecialTokenRange(a,b)}).collect::<Vec<_>>();
        let expr=if exprs.len()==1 {exprs.into_iter().next().unwrap()}
            else {glrmask_grammar::GrammarExpr::Choice(exprs)};
        let named=glrmask_grammar::NamedGrammar {
            start:"__exact_event".into(),
            rules:vec![glrmask_grammar::NamedRule {name:"__exact_event".into(),expr,is_terminal:false,is_internal:false}],
            ignore:None,lexer_partitions:Default::default(),lexer_literal_partitions:Default::default(),default_lexer_partition:None,
        };
        let mut grammar=Self::new(GrammarSource::Constructed);
        let _=grammar.parsed_component.set(ParsedComponent::from_named(named,&[]).map_err(Into::into));
        grammar.exact_value=Some(tokens);grammar
    }

    /// Includes unresolved slots in attached open children. Diagnostic expansion is bounded and may return an error for very large shared graphs.
    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn requirements(&self) -> Result<Vec<Requirement>> {
        graph::source_requirements(self)
    }

    /// Binds the extern `name` to `value`. Fails if `name` is undeclared or already
    /// bound, or `value` is the wrong kind.
    #[allow(private_bounds)]
    pub fn bind<T>(&mut self, name: impl AsRef<str>, value: T) -> Result<()>
    where
        T: IntoGrammarValue,
    {
        let next = self.bind_value(name.as_ref(), value.into_grammar_value()?)?;
        *self = next;
        Ok(())
    }

    // Keep the large historical differential fixture corpus readable while it
    // exercises the same mutable operation. Not part of the normal public API.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    #[allow(private_bounds)]
    pub fn bind_snapshot<T>(&self, name: impl AsRef<str>, value: T) -> Result<Self>
    where T: IntoGrammarValue {
        let mut next = self.clone();
        next.bind(name, value)?;
        Ok(next)
    }

    fn bind_value(&self, name:&str, mut value:GrammarValue) -> Result<Self> {
        let declared = self.declarations()?;
        let previous = graph::source_summary(self)?;
        if !declared.kinds.contains_key(name) {
            if let Some((head,tail))=attached_path(name,&self.bindings) {
                let Some(GrammarValue::Source(child))=self.bindings.get(head) else {
                    return Err(Error::Compilation(format!("{head:?} is not a source component")));
                };
                let child=child.bind_value(tail,value)?;
                let summary=graph::source_summary(&child)?;
                let mut next=self.clone();
                next.bindings.insert(head.to_owned(),GrammarValue::Source(Arc::new(child)));
                next.analysis=graph::filled_cache(previous.changed(head,Some(summary)));
                next.module_snapshot=Arc::new(OnceLock::new());
                graph::check_source_limits(&next)?;
                return Ok(next);
            }
            return Err(Error::Compilation(format!("there is no extern named {name:?}")));
        }
        if self.bindings.contains_key(name) {
            return Err(Error::Compilation(format!("extern {name:?} is already bound")));
        }
        check_binding_kind(name,declared.kinds[name],matches!(value,GrammarValue::Source(_)))
            .map_err(|e|e.with_source_span(declared.spans.get(name).copied()))?;
        // A one-event value is a lossless grammar-fragment widening. Preserve
        // the independently scoped child boundary, not a decoded-byte alias.
        if declared.kinds[name]==ExternKind::Grammar {
            value=match value {
                GrammarValue::TokenSet(tokens)=>GrammarValue::Source(Arc::new(Self::from_token_set(tokens))),
                other=>other,
            };
        }
        let child_summary=match &value {
            GrammarValue::Source(child)=>Some(graph::source_summary(child)?),
            _=>None,
        };
        let mut next=self.clone();next.bindings.insert(name.to_owned(),value);
        next.analysis=graph::filled_cache(previous.changed(name,child_summary));
        next.module_snapshot=Arc::new(OnceLock::new());
        graph::check_source_limits(&next)?;
        Ok(next)
    }

    /// Compatibility spelling for source-only grammar binding.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn bind_grammar(self, name: impl AsRef<str>, grammar: Grammar) -> Result<Self> {
        self.bind_value(name.as_ref(), GrammarValue::Source(Arc::new(grammar)))
    }

    /// The grammar's externs.
    pub fn externs(&self) -> Externs<'_> {
        Externs {
            source: ExternsSource::Grammar(self),
            only_unbound: false,
        }
    }

    /// Compiles the grammar for `vocab`.
    pub fn compile(&self, vocab: &Vocab) -> Result<RuntimeConstraint> {
        Module::new(self).compile(vocab)
    }

    /// Compiles as much as possible for `vocab` now, even with externs unbound, and
    /// returns a [`Module`] that keeps the work.
    pub fn precompile(&self, vocab: &Vocab) -> Result<Module> {
        self.precompile_with(vocab, CompileOptions::default())
    }

    /// Like [`precompile`](Self::precompile), with `options`.
    pub fn precompile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<Module> {
        let mut module = Module::new(self);
        module.precompile_with(vocab, options)?;
        Ok(module)
    }

    /// Like [`compile`](Self::compile), with `options`.
    pub fn compile_with(
        &self,
        vocab: &Vocab,
        options: CompileOptions,
    ) -> Result<RuntimeConstraint> {
        Module::new(self).compile_with(vocab, options)
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn compile_with_build_options(
        &self,
        vocab: &Vocab,
        options: BuildOptions,
    ) -> Result<RuntimeConstraint> {
        graph::source_preflight(self)?;
        let engine = options.engine()?;
        if matches!(self.source,GrammarSource::Lark(_)){return self.resolve_lark(vocab)?.compile_with_build_options(vocab,options);}
        self.validate_exact_target(vocab)?;
        self.validate_ebnf(vocab)?;
        if let Some(name) = self.unresolved_external_names()?.first() {
            return Err(Error::Compilation(format!(
                "extern {name:?} is not bound; bind it before compiling",
            )));
        }
        if engine == Engine::TokenTrie {
            let constraint = self.compile_token_trie(
                vocab,
                options.boundary_trigger.unwrap_or_default(),
                options.token_trie_backend(),
            )?;
            return constraint.with_end_tokens(options.end_token_ids());
        }
        if engine == Engine::MergedTokenTrie && options.engine_backend == ParserBackend::LrTable {
            let constraint = self.compile_merged_token_trie(
                vocab,
                options.boundary_trigger.unwrap_or_default(),
                ParserBackend::LrTable,
            )?;
            return constraint.with_end_tokens(options.end_token_ids());
        }
        if options.parser_backend == ParserBackend::TemplateDfa
            && engine == Engine::MergedTokenTrie && self.bindings.is_empty() {
                return self.finish_template_attempt(vocab, &options, || {
                    let budget = crate::compiler::build_budget::BuildBudgetScope::enter();
                    // A partial template build must not leave prepared caches on
                    // the source reused by the rescue attempt.
                    let attempt_grammar = self.clone();
                    let attempt_options = options.clone();
                    budget.install(move || {
                        crate::error::catch_template_attempt(move || -> Result<RuntimeConstraint> {
                            // Use the existing O2 runtime with the bounded grammar normal
                            // form required by finite templates. Ordinary dynamic grammar
                            // preparation intentionally allows recursive action closures.
                            let dynamic = match &attempt_grammar.source {
                                GrammarSource::Glrm(_) | GrammarSource::Constructed => crate::import::compile_bounded_template_from_named(attempt_grammar.resolve_dynamic_source(vocab,false)?,vocab,crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,None),
                                GrammarSource::Ebnf(source) => DynamicConstraint::from_ebnf_with_bounded_template_parser(source, vocab),
                                GrammarSource::Lark(source) => DynamicConstraint::from_lark_with_bounded_template_parser(source, vocab),
                                GrammarSource::JsonSchema(source) => DynamicConstraint::from_json_schema_with_bounded_template_parser(source, vocab),
                            }?;
                            let mut components = dynamic.into_constraints();
                            if components.len() != 1 {
                                return Err(Error::Compilation("the TemplateDfa parser backend doesn't support a choice of several grammars yet; use the LrTable backend".into()));
                            }
                            let mut constraint = components.pop().unwrap();
                            ensure_runnable_constraint(&constraint)?;
                            apply_final_boundary_trigger(&mut constraint, attempt_options.boundary_trigger.unwrap_or_default())?;
                            constraint.with_end_tokens(attempt_options.end_token_ids())
                        })
                    })
                });
            }
        if !self.bindings.is_empty() {
            return self.finish_template_attempt(vocab, &options, || {
                let budget = crate::compiler::build_budget::BuildBudgetScope::enter();
                let attempt_grammar = self.clone();
                let attempt_options = options.clone();
                budget.install(move || {
                    crate::error::catch_template_attempt(move || {
                        attempt_grammar.prepare(vocab)?.link_with(attempt_options)
                    })
                })
            });
        }
        self.finish_template_attempt(vocab, &options, || {
            let budget = crate::compiler::build_budget::BuildBudgetScope::enter();
            let attempt_grammar = self.clone();
            let attempt_options = options.clone();
            budget.install(move || {
                crate::error::catch_template_attempt(move || -> Result<RuntimeConstraint> {
                    let mut spec = ConstraintSpec::builder(attempt_grammar.clone(), vocab)?.build()?;
                    spec.automatic_boundary_selection = true;
                    if let Some(name) = spec.unbound_grammar_names.first() {
                        return Err(Error::Compilation(format!(
                            "extern {name:?} is not bound; bind it before compiling",
                        )));
                    }
                    let mut constraint = spec.compile_final(engine)?;
                    ensure_runnable_constraint(&constraint)?;
                    let detail = attempt_options.boundary_trigger.unwrap_or_default();
                    if attempt_options.parser_backend == ParserBackend::TemplateDfa {
                        // Preserve the component-only proper-prefix certificate while the
                        // LR/rule metadata still exists. A table-free constraint can be
                        // bound later as a compiled child and must not widen that boundary
                        // query back to the full model vocabulary merely because its table
                        // was intentionally removed.
                        crate::compiler::boundary_candidates::persist_boundary_candidate_summary(
                            &mut constraint,
                            vocab,
                        );
                        constraint.install_template_parser_from_compile()?;
                    }
                    let _ = constraint.late_bind_vocab.set(vocab.clone());
                    apply_final_boundary_trigger(&mut constraint, detail)?;
                    constraint.with_end_tokens(attempt_options.end_token_ids())
                })
            })
        })
    }

    /// Give a template-strategy decline one terminal-ended rescue attempt with
    /// the same strategy. If that also fails, the original typed decline is
    /// returned; the requested strategy is never silently replaced.
    ///
    /// The terminal-ended rescue is opt-in:
    /// `GLRMASK_EXPERIMENT_TERMINAL_ENDED_RESCUE=1` enables it after a typed
    /// template decline. `GLRMASK_EXPERIMENT_TERMINAL_ENDED_FIRST=1`
    /// additionally tries it before the ordinary template attempt.
    /// `GLRMASK_REPORT_ROUTE=1` emits one route line for this top-level compile.
    fn finish_template_attempt<F>(
        &self,
        vocab: &Vocab,
        options: &BuildOptions,
        attempt: F,
    ) -> Result<RuntimeConstraint>
    where
        F: FnOnce() -> Result<RuntimeConstraint>,
    {
        use crate::compiler::terminal_ended_rescue::{
            report_route, RescueExperiment, RescueOutcome, Route,
        };
        let experiment = RescueExperiment::read();
        let strategy = options.engine()?;
        let template = options.parser_backend == ParserBackend::TemplateDfa;
        let te_first = experiment.terminal_ended_rescue
            && experiment.terminal_ended_first
            && template
            && self.bindings.is_empty()
            && matches!(
                strategy,
                Engine::ParserDwa | Engine::MergedTokenTrie | Engine::Auto
            );
        let mut route = Route::Normal;
        let mut outcome = RescueOutcome::NotTried;
        let result = (|| -> Result<RuntimeConstraint> {
            // EXPERIMENT: terminal-ended first.
            if te_first {
                match self.try_terminal_ended_rescue(vocab, options) {
                    Ok(constraint) => {
                        route = Route::TerminalEnded;
                        outcome = RescueOutcome::Ok;
                        return Ok(constraint);
                    }
                    Err(rescue_error) => {
                        outcome = RescueOutcome::Declined(rescue_error.to_string());
                    }
                }
            }
            match attempt() {
                Ok(constraint) => Ok(constraint),
                Err(error) if error.is_strategy_decline() => {
                    // Opt-in experiment: give the original source one
                    // terminal-ended rescue attempt with the same strategy.
                    // Without the switch, the typed decline above is returned
                    // unchanged.
                    if template && experiment.terminal_ended_rescue && !te_first {
                        match self.try_terminal_ended_rescue(vocab, options) {
                            Ok(constraint) => {
                                if crate::compiler::compile::compile_profile_enabled() {
                                    eprintln!(
                                        "[glrmask/profile][terminal_ended_rescue] strategy={:?} outcome=ok",
                                        strategy,
                                    );
                                }
                                route = Route::TerminalEnded;
                                outcome = RescueOutcome::Ok;
                                return Ok(constraint);
                            }
                            Err(rescue_error) => {
                                if crate::compiler::compile::compile_profile_enabled() {
                                    eprintln!(
                                        "[glrmask/profile][terminal_ended_rescue] strategy={:?} outcome=declined reason={rescue_error}",
                                        strategy,
                                    );
                                }
                                outcome = RescueOutcome::Declined(rescue_error.to_string());
                            }
                        }
                    }
                    if crate::compiler::compile::compile_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][template_decline] strategy={:?} stage={} reason={error}",
                            strategy,
                            error.decline_stage(),
                        );
                    }
                    // Only Auto may pick a different exact strategy; an explicit
                    // strategy request is never silently replaced.
                    if strategy == Engine::Auto {
                        // The fallback is exactly an explicit TokenTrie request
                        // with the same two switches.
                        let constraint = self
                            .compile_requested_token_trie(
                                vocab,
                                options.boundary_trigger.unwrap_or_default(),
                                options.engine_backend,
                                options.merge_equivalent_tokens,
                            )?
                            .with_end_tokens(options.end_token_ids())?;
                        route = Route::TokenTrieFallback;
                        return Ok(constraint);
                    }
                    Err(error)
                }
                Err(error) => Err(error),
            }
        })();
        report_route(experiment.report_route, route, strategy, experiment, &outcome);
        result
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn prepare(&self, vocab: &Vocab) -> Result<PreparedGrammar> {
        self.prepare_with(vocab, BuildOptions::default())
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn prepare_with(&self, vocab: &Vocab, options: BuildOptions) -> Result<PreparedGrammar> {
        self.prepare_unlinked(vocab, options)
    }

    // The public preparation entry points are internal-api only, but composition
    // and artifact loading prepare children in every build.
    pub(crate) fn prepare_unlinked(&self, vocab: &Vocab, options: BuildOptions) -> Result<PreparedGrammar> {
        with_deep_stack(|| {
            let _ = graph::source_first(self)?;
            let module = self.prepare_cached(vocab, options, &mut SourcePreparation::default(), 0)?;
            module.validate_slot_manifest()?;
            Ok(module.as_ref().clone())
        })
    }

    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn compile_unlinked(&self,vocab:&Vocab)->Result<PreparedGrammar> {self.prepare(vocab)}
    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn compile_unlinked_with(&self,vocab:&Vocab,options:BuildOptions)->Result<PreparedGrammar> {self.prepare_with(vocab,options)}

    fn prepare_cached(&self,vocab:&Vocab,options:BuildOptions,cache:&mut SourcePreparation,depth:usize)
        -> Result<Arc<PreparedGrammar>> {
        if depth>graph::MAX_DEPTH {return Err(Error::Compilation("grammars are nested more than 64 levels deep".into()));}
        let identity=graph::source_identity(self);
        if let Some(module)=cache.prepared.get(&identity){return Ok(module.clone());}
        cache.visited+=1;
        if cache.visited>graph::MAX_NODES {return Err(Error::Compilation("the composition has too many parts".into()));}

        if matches!(self.source,GrammarSource::Lark(_)) {
            let resolved=Arc::new(self.resolve_lark(vocab)?);
            cache.retained.push(resolved.clone());
            let module=resolved.prepare_cached(vocab,options,cache,depth)?;
            cache.prepared.insert(identity,module.clone());return Ok(module);
        }
        self.validate_exact_target(vocab)?;
        self.validate_ebnf(vocab)?;
        let declarations=self.declarations()?;
        let slot_kinds=Arc::new(declarations.kinds.clone());
        let slot_spans=Arc::new(declarations.spans.clone());
        // Compile this component locally. Grammar-child attachments remain
        // compiled values in the unlinked graph and are linked only by link_with,
        // where the caller's MaskStrategy choice is finally known.
        let (local_grammar, source_bindings) = self.clone().into_source_only_and_bindings();
        let mut builder = ConstraintSpec::builder(local_grammar, vocab)?;
        let event_adapters=Arc::new(event_cache::ExactAdapterCache::default());
        let mut bindings = OrdMap::<String, ModuleBinding>::new();
        let mut source_children = Vec::<(String, Arc<Grammar>)>::new();
        for (name, value) in source_bindings {
            match value {
                GrammarValue::TokenSet(tokens) => {
                    if let Some(id) = vocab.first_unknown_token_id(&tokens) {
                        return Err(Error::Compilation(format!(
                            "extern {name:?} is bound to token ID {id}, which is not in the vocabulary",
                        )));
                    }
                    bindings.insert(name.clone(),ModuleBinding::TokenSet(event_adapters.leaf(tokens.range_pairs())?));
                    builder.token_bindings.insert(name, tokens.range_pairs().to_vec());
                }
                GrammarValue::Source(child) => {
                    source_children.push((name, child));
                }
            }
        }
        let mut spec = builder.build()?;
        spec.allow_open_source_tokens = true;
        spec.automatic_boundary_selection = true;
        // A leaf has no independent child work. Entering the global Rayon pool
        // here lets a worker waiting on a compiler-local pool re-enter another
        // unrelated compilation while retaining its heavy compiler stack.
        let (constraint, source_modules) = if source_children.is_empty() {
            (spec.compile_boxed(), Ok(Vec::new()))
        } else {
            rayon::join(
                || spec.compile_boxed(),
                || {
                    source_children
                        .into_iter()
                        .map(|(name, child)| {
                            cache.retained.push(child.clone());
                            Ok((name, ModuleBinding::Module(child.prepare_cached(vocab,BuildOptions::default(),cache,depth+1)?)))
                        })
                        .collect::<Result<Vec<_>>>()
                },
            )
        };
        let constraint = constraint?;
        for (name, binding) in source_modules? {
            bindings.insert(name, binding);
        }
        let module = PreparedGrammar {
            inner: Arc::from(constraint),
            bindings,
            analysis: graph::empty_cache(),
            event_adapters,
            boundary_trigger: options.boundary_trigger.unwrap_or_default(),
            slot_kinds,
            slot_spans,
            retained_artifact: None,
            retained_vocab_fingerprint: None,
        };
        module.validate_local_slot_manifest(vocab)?;
        let module=Arc::new(module);
        cache.prepared.insert(identity,module.clone());
        Ok(module)
    }

    // Resolve source bindings as typed AST scopes before ordinary Dynamic
    // compilation. No compiled component is converted into a parser table.
    fn resolve_dynamic_source(&self, vocab: &Vocab, embedded: bool) -> Result<glrmask_grammar::NamedGrammar> {
        match &self.source {
            GrammarSource::Ebnf(source) => {
                crate::import::validate_ebnf_source(source, vocab)?;
                crate::import::parse_ebnf_to_named(source)
            }
            GrammarSource::Lark(_) => self.resolve_lark(vocab)?.resolve_dynamic_source(vocab,embedded),
            GrammarSource::JsonSchema(source) => {
                let mut named = crate::import::parse_json_schema_to_named_dynamic(source)?;
                if embedded {
                    named = crate::grammar::factoring::factor_named_grammar(named);
                    crate::import::prepare_dynamic_json_schema_named(&mut named)?;
                }
                Ok(named)
            }
            GrammarSource::Glrm(_) | GrammarSource::Constructed => {
                self.validate_exact_target(vocab)?;
                let mut children=Vec::new();let mut tokens=Vec::new();
                for (name,value) in &self.bindings {
                    match value {
                        GrammarValue::Source(child)=>children.push((name.as_str(),ParsedComponent::from_named(child.resolve_dynamic_source(vocab,true)?,&[])?)),
                        GrammarValue::TokenSet(ids)=>{
                            if let Some(id) = vocab.first_unknown_token_id(ids) {
                                return Err(Error::Compilation(format!("token ID {id} is not in the vocabulary")));
                            }
                            tokens.push((name.as_str(),ids.range_pairs().to_vec()));
                        },
                    }
                }
                let token_refs=tokens.iter().map(|(n,ids)|(*n,ids.as_slice())).collect::<Vec<_>>();
                Ok(self.component()?.with_tokens(&token_refs)?.with_children(&children)?.lower_closed()?)
            }
        }
    }

    /// Lower this source exactly as the ordinary static template route does,
    /// stopping at the pre-normalization `GrammarDef`. The normal route and the
    /// terminal-ended rescue share this function so their inputs cannot drift.
    pub(crate) fn lower_static_template_input(
        &self,
        vocab: &Vocab,
    ) -> Result<crate::grammar::flat::GrammarDef> {
        match &self.source {
            GrammarSource::Ebnf(source) => crate::import::lower_static_ebnf(source),
            GrammarSource::Lark(_) => {
                // The normal route resolves Lark into a Constructed component
                // first; raw parse_lark_to_named cannot express events,
                // imports or schemas.
                let resolved = self.resolve_lark(vocab)?;
                resolved.lower_static_template_input(vocab)
            }
            GrammarSource::JsonSchema(source) => crate::import::lower_static_json_schema(source),
            GrammarSource::Glrm(_) | GrammarSource::Constructed => {
                self.validate_exact_target(vocab)?;
                let named = crate::import::lower_static_component(self.component()?, vocab)?;
                crate::import::lower_static_named_grammar(named, &[])
            }
        }
    }

    /// Lower the original source to the flat grammar consumed by the
    /// terminal-ended construction. Stage 1 remains restricted to plain single
    /// grammars: bindings and extern declarations decline the rescue. Lark is
    /// resolved first so events (and, until composed-unit support lands, the
    /// declarations produced by imports/schemas) take the normal route.
    fn rescue_source_grammar(
        &self,
        vocab: &Vocab,
    ) -> Result<crate::grammar::flat::GrammarDef> {
        let resolved;
        let source = if matches!(self.source, GrammarSource::Lark(_)) {
            resolved = self.resolve_lark(vocab)?;
            &resolved
        } else {
            self
        };
        if !source.bindings.is_empty() {
            return Err(Error::StrategyUnsupported(
                "terminal-ended rescue is limited to plain single grammars".into(),
            ));
        }
        if !source.declarations()?.names.is_empty() {
            return Err(Error::StrategyUnsupported(
                "terminal-ended rescue does not accept extern declarations".into(),
            ));
        }
        let grammar = source.lower_static_template_input(vocab)?;
        crate::compiler::terminal_ended_rescue::check_rescue_scope(&grammar)?;
        Ok(grammar)
    }

    fn try_terminal_ended_rescue(
        &self,
        vocab: &Vocab,
        options: &BuildOptions,
    ) -> Result<RuntimeConstraint> {
        let source = self.rescue_source_grammar(vocab)?;
        let mut constraint = crate::compiler::terminal_ended_rescue::rescue(&source, vocab)?;
        apply_final_boundary_trigger(&mut constraint, options.boundary_trigger.unwrap_or_default())?;
        constraint.with_end_tokens(options.end_token_ids())
    }

    // Use the ordinary Dynamic compiler with one complete, unsplit source
    // body. Resolve bindings as typed nodes before compilation, without
    // rebuilding the resulting Constraint through the native/O2 path.
    pub(crate) fn compile_token_trie(
        &self,
        vocab: &Vocab,
        boundary: BoundaryTriggerDetail,
        backend: crate::runtime::parser_backend::DynamicParserBackend,
    ) -> Result<RuntimeConstraint> {
        graph::source_preflight(self)?;
        if matches!(self.source, GrammarSource::Lark(_)) {
            return self.resolve_lark(vocab)?.compile_token_trie(vocab, boundary, backend);
        }
        self.validate_exact_target(vocab)?;
        self.validate_ebnf(vocab)?;
        let named = self.resolve_dynamic_source(vocab, false)?;
        let construction = match &self.source {
            GrammarSource::JsonSchema(_) => crate::compiler::glr::table::GlrTableConstruction::Lalr,
            _ => crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        };
        let mut constraint = crate::import::compile_dynamic_named_single_with_backend(
            named,
            vocab,
            construction,
            matches!(self.source, GrammarSource::JsonSchema(_)),
            backend,
        )?;
        let _ = constraint.late_bind_vocab.set(vocab.clone());
        ensure_runnable_constraint(&constraint)?;
        apply_final_boundary_trigger(&mut constraint, boundary)?;
        Ok(constraint)
    }

    /// Compile the whole grammar as TokenTrie over the merged vocabulary
    /// (combinations C and D). `TemplateDfa` installs a template parser; `LrTable`
    /// keeps an LR table and never installs one.
    pub(crate) fn compile_merged_token_trie(
        &self,
        vocab: &Vocab,
        boundary: BoundaryTriggerDetail,
        backend: ParserBackend,
    ) -> Result<RuntimeConstraint> {
        graph::source_preflight(self)?;
        if matches!(self.source, GrammarSource::Lark(_)) {
            return self
                .resolve_lark(vocab)?
                .compile_merged_token_trie(vocab, boundary, backend);
        }
        self.validate_exact_target(vocab)?;
        self.validate_ebnf(vocab)?;
        let merge_error = |e: Error| {
            if e.is_strategy_decline() {
                e
            } else {
                Error::Compilation(format!(
                    "cannot merge equivalent tokens for this grammar: {e}"
                ))
            }
        };
        let backend = token_trie_backend(backend);
        let has_bindings = !self.bindings.is_empty();
        let mut inner = match self.source {
            GrammarSource::Glrm(_) | GrammarSource::Constructed | GrammarSource::Ebnf(_) => {
                if has_bindings {
                    self.compile_merged_from_resolved_source(vocab, backend)?
                } else if backend == crate::runtime::parser_backend::DynamicParserBackend::LrTable {
                    crate::import::compile_component_shell_balanced_lr(
                        self.component()?,
                        vocab,
                        &[],
                    )
                    .map_err(merge_error)?
                } else {
                    crate::import::compile_component_shell_balanced(
                        self.component()?,
                        vocab,
                        &[],
                    )
                    .map_err(merge_error)?
                }
            }
            _ => {
                if has_bindings || backend == crate::runtime::parser_backend::DynamicParserBackend::LrTable {
                    self.compile_merged_from_resolved_source(vocab, backend)?
                } else {
                    let dynamic = match &self.source {
                        GrammarSource::Lark(source) => {
                            DynamicConstraint::from_lark_with_bounded_template_parser(source, vocab)
                        }
                        GrammarSource::JsonSchema(source) => {
                            DynamicConstraint::from_json_schema_with_bounded_template_parser(
                                source, vocab,
                            )
                        }
                        _ => unreachable!(),
                    }
                    .map_err(merge_error)?;
                    let mut components = dynamic.into_constraints();
                    if components.len() != 1 {
                        return Err(Error::Compilation(
                            "cannot merge equivalent tokens in a grammar made of several parts".into(),
                        ));
                    }
                    components.pop().unwrap()
                }
            }
        };
        let _ = inner.late_bind_vocab.set(vocab.clone());
        if backend != crate::runtime::parser_backend::DynamicParserBackend::LrTable {
            inner.install_template_parser_from_compile()?;
        }
        ensure_runnable_constraint(&inner)?;
        apply_final_boundary_trigger(&mut inner, boundary)?;
        Ok(inner)
    }

    /// Resolve bindings exactly as the unmerged flatten path does, then compile
    /// the whole resolved grammar through the merged-vocabulary lane.
    fn compile_merged_from_resolved_source(
        &self,
        vocab: &Vocab,
        backend: crate::runtime::parser_backend::DynamicParserBackend,
    ) -> Result<RuntimeConstraint> {
        let named = self.resolve_dynamic_source(vocab, false)?;
        let construction = match &self.source {
            GrammarSource::JsonSchema(_) => crate::compiler::glr::table::GlrTableConstruction::Lalr,
            _ => crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        };
        let merge_error = |e: Error| {
            if e.is_strategy_decline() {
                e
            } else {
                Error::Compilation(format!(
                    "cannot merge equivalent tokens for this grammar: {e}"
                ))
            }
        };
        crate::import::compile_dynamic_named_single_merged(
            named,
            vocab,
            construction,
            matches!(self.source, GrammarSource::JsonSchema(_)),
            backend,
        )
        .map_err(merge_error)
    }

    /// Compile a TokenTrie request with the supplied switches, whether they
    /// were explicit or reached through `Auto`'s decline fallback.
    fn compile_requested_token_trie(
        &self,
        vocab: &Vocab,
        boundary: BoundaryTriggerDetail,
        parser_backend: ParserBackend,
        merge_equivalent_tokens: bool,
    ) -> Result<RuntimeConstraint> {
        if merge_equivalent_tokens {
            self.compile_merged_token_trie(vocab, boundary, parser_backend)
        } else {
            self.compile_token_trie(vocab, boundary, token_trie_backend(parser_backend))
        }
    }

    /// Creates a grammar from an expression.
    pub fn from_expr(expr: impl Into<Expr>) -> Result<Grammar> {
        let expr = expr.into();
        Self::from_expr_ref(&expr)
    }

    pub(crate) fn from_expr_ref(expr: &Expr) -> Result<Grammar> {
        let module = Module::from_expr(expr)?;
        module.to_source()
    }

    fn unresolved_external_names(&self) -> Result<BTreeSet<String>> {
        Ok(graph::source_first(self)?.map(|r|r.name()).into_iter().collect())
    }

    fn glrm_source(&self) -> Option<&str> {
        match &self.source {
            GrammarSource::Glrm(source) => Some(source),
            _ => None,
        }
    }

    fn into_source_only_and_bindings(self) -> (Self, OrdMap<String, GrammarValue>) {
        let mut source=self;
        let bindings=std::mem::take(&mut source.bindings);
        source.analysis=graph::empty_cache();
        source.module_snapshot=Arc::new(OnceLock::new());
        (source,bindings)
    }
}

impl IntoGrammarValue for Grammar {
    fn into_grammar_value(self) -> Result<GrammarValue> {
        Ok(GrammarValue::Source(Arc::new(self)))
    }
}

impl IntoGrammarValue for &Grammar {
    fn into_grammar_value(self) -> Result<GrammarValue> {
        Ok(GrammarValue::Source(Arc::new(self.clone())))
    }
}

impl IntoGrammarValue for TokenSet {
    fn into_grammar_value(self) -> Result<GrammarValue> {
        Ok(GrammarValue::TokenSet(self))
    }
}

impl IntoGrammarValue for u32 {
    fn into_grammar_value(self) -> Result<GrammarValue> {
        Ok(GrammarValue::TokenSet(TokenSet::from(self)))
    }
}

impl IntoGrammarValue for Expr {
    fn into_grammar_value(self)->Result<GrammarValue> {
        Ok(GrammarValue::Source(Arc::new(Grammar::from_expr_ref(&self)?)))
    }
}
impl IntoGrammarValue for &Expr {
    fn into_grammar_value(self)->Result<GrammarValue> {
        Ok(GrammarValue::Source(Arc::new(Grammar::from_expr_ref(self)?)))
    }
}

#[cfg(any(test,feature="internal-api"))]
impl<'a> IntoUnlinkedValue<'a> for &'a RuntimeConstraint {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::StaticBorrowed(self)
    }
}

#[cfg(any(test,feature="internal-api"))]
impl<'a> IntoUnlinkedValue<'a> for RuntimeConstraint {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::StaticOwned(Arc::new(self))
    }
}

#[cfg(any(test,feature="internal-api"))]
impl<'a> IntoUnlinkedValue<'a> for Arc<RuntimeConstraint> {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::StaticOwned(self)
    }
}

impl<'a> IntoUnlinkedValue<'a> for &'a PreparedGrammar {
    fn into_unlinked_value(self)->UnlinkedValue<'a>{UnlinkedValue::Prepared(Arc::new(self.clone()))}
}
impl<'a> IntoUnlinkedValue<'a> for PreparedGrammar {
    fn into_unlinked_value(self)->UnlinkedValue<'a>{UnlinkedValue::Prepared(Arc::new(self))}
}

impl<'a> IntoUnlinkedValue<'a> for TokenSet {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::TokenSet(self)
    }
}

impl<'a> IntoUnlinkedValue<'a> for u32 {
    fn into_unlinked_value(self) -> UnlinkedValue<'a> {
        UnlinkedValue::TokenSet(TokenSet::from(self))
    }
}


/// Strategy used to build a [`VocabPartition`].
///
/// All strategies are conservative: they may keep tokens separate when a more
/// expensive proof could merge them, but they never intentionally merge tokens
/// whose grammar-visible behavior differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VocabPartitionStrategy {
    /// Choose automatically from the observable lexer topology.
    #[default]
    Automatic,
    /// Prefer the Static-derived quotient, which is usually more compact but may
    /// spend more time proving multi-token equivalences on small lexers.
    Compact,
    /// Use the dedicated vocabulary-only relation. This often compiles faster on
    /// small lexers but may return a finer partition with more classes.
    Dedicated,
}

/// A conservative grammar-specific equivalence partition of model vocabulary tokens.
///
/// Tokens in the same class have been proved interchangeable by the fast
/// vocabulary analysis. The partition may be finer than the token partition of
/// a fully compiled [`Constraint`](crate::Constraint): expensive proof steps may
/// deliberately be skipped, in which case affected tokens remain separated.
#[derive(Debug, Clone)]
pub struct VocabPartition {
    original_to_class: Vec<u32>,
    classes: Vec<Vec<u32>>,
    representatives: Vec<u32>,
    class_output_masks: Vec<Vec<(u32, u32)>>,
}

impl VocabPartition {
    /// Analyze `grammar` for `vocab` without constructing terminal/parser DWAs.
    pub fn compile(grammar: Grammar, vocab: &Vocab) -> Result<Self> {
        Self::compile_with_strategy(grammar, vocab, VocabPartitionStrategy::Automatic)
    }

    /// Analyze `grammar` for `vocab` using an explicit partition strategy.
    pub fn compile_with_strategy(
        grammar: Grammar,
        vocab: &Vocab,
        strategy: VocabPartitionStrategy,
    ) -> Result<Self> {
        let profile = crate::compiler::pipeline::compile_top_profile_enabled();
        let total_started = profile.then(Instant::now);
        if !grammar.bindings.is_empty() {
            return Err(Error::Compilation(
                "vocabulary partition analysis does not yet support bound grammar values".to_owned(),
            ));
        }
        let source_kind = match &grammar.source {
            GrammarSource::Ebnf(_) => "ebnf",
            GrammarSource::Lark(_) => "lark",
            GrammarSource::JsonSchema(_) => "json_schema",
            GrammarSource::Glrm(_) => "glrm",
            GrammarSource::Constructed => return Err(Error::Compilation("use normal preparation for constructed event grammars".into())),
        };
        let source = match &grammar.source {
            GrammarSource::Ebnf(source)
            | GrammarSource::Lark(source)
            | GrammarSource::JsonSchema(source)
            | GrammarSource::Glrm(source) => source,
            GrammarSource::Constructed => unreachable!("handled before source lookup"),
        };
        let lower_started = profile.then(Instant::now);
        let grammar_def = crate::import::lower_source_for_vocab_partition(source_kind, source)?;
        let lower_ms = lower_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0);
        let compile_started = profile.then(Instant::now);
        let map = crate::error::catch_internal_invariant(|| {
            crate::compiler::vocab_partition::compile_vocab_partition_owned(grammar_def, vocab, strategy)
        })?;
        if profile {
            eprintln!(
                "[glrmask/profile][vocab_partition_compile] source={} lower_ms={:.3} compile_ms={:.3} total_ms={:.3}",
                source_kind,
                lower_ms,
                compile_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
                total_started.map_or(0.0, |started| started.elapsed().as_secs_f64() * 1000.0),
            );
        }
        let classes = map.internal_to_originals;
        // Internal O2/vocab-only compilation is allowed to carry only grouped
        // class membership and deliberately omit the model-vocab-sized dense
        // original-token -> class vector. `VocabPartition` is a public API,
        // however, and explicitly exposes that dense coordinate through
        // `class_of()` / `original_to_class()`. Materialize it once at this
        // boundary rather than forcing every internal fast path to carry it.
        let original_to_class = if map.original_to_internal.is_empty() && !classes.is_empty() {
            let mut dense = vec![u32::MAX; vocab.max_token_id() as usize + 1];
            for (class_id, class) in classes.iter().enumerate() {
                for &token_id in class {
                    dense[token_id as usize] = class_id as u32;
                }
            }
            dense
        } else {
            map.original_to_internal
        };
        let class_output_masks = classes
            .iter()
            .map(|class| {
                let mut words = Vec::<(u32, u32)>::new();
                for &token_id in class {
                    let word = token_id / 32;
                    let bit = 1u32 << (token_id % 32);
                    if let Some((last_word, last_bits)) = words.last_mut()
                        && *last_word == word
                    {
                        *last_bits |= bit;
                    } else {
                        words.push((word, bit));
                    }
                }
                words
            })
            .collect();
        Ok(Self {
            original_to_class,
            classes,
            representatives: map.representative_original_ids,
            class_output_masks,
        })
    }

    /// Number of equivalence classes.
    pub fn num_classes(&self) -> usize { self.classes.len() }

    /// Class containing `token_id`, or `None` if that token ID was not in the vocabulary.
    pub fn class_of(&self, token_id: u32) -> Option<u32> {
        self.original_to_class
            .get(token_id as usize)
            .copied()
            .filter(|&class| class != u32::MAX)
    }

    /// Original model token IDs grouped by equivalence class.
    pub fn classes(&self) -> &[Vec<u32>] { &self.classes }

    /// A stable original-token representative for `class_id`.
    pub fn representative(&self, class_id: u32) -> Option<u32> {
        self.representatives
            .get(class_id as usize)
            .copied()
            .filter(|&token| token != u32::MAX)
    }

    /// Dense original-token-ID to class-ID map. Missing token IDs contain `u32::MAX`.
    pub fn original_to_class(&self) -> &[u32] { &self.original_to_class }

    /// Number of packed `u64` words needed for a class-space mask.
    pub fn internal_mask_len(&self) -> usize { self.num_classes().div_ceil(64) }

    /// Number of packed `u32` words needed for an original-token-space mask.
    pub fn original_mask_len(&self) -> usize { self.original_to_class.len().div_ceil(32) }

    /// Expand a packed class-space mask into the original model-token ID space.
    ///
    /// Bit `i` in `internal_mask` selects equivalence class `i`. Every original
    /// token in each selected class is set in the returned packed `u32` mask.
    /// Bits beyond [`Self::num_classes`] are ignored.
    pub fn expand_mask(&self, internal_mask: &[u64]) -> Vec<u32> {
        let mut out = vec![0u32; self.original_mask_len()];
        self.fill_expanded_mask(internal_mask, &mut out);
        out
    }

    /// Expand a packed class-space mask into an existing original-token mask buffer.
    ///
    /// `out` must contain at least [`Self::original_mask_len`] `u32` words. The
    /// entire supplied buffer is cleared before expansion, matching the overwrite
    /// semantics of [`ConstraintState::fill_mask`](crate::ConstraintState::fill_mask).
    pub fn fill_expanded_mask(&self, internal_mask: &[u64], out: &mut [u32]) {
        let required = self.original_mask_len();
        assert!(
            out.len() >= required,
            "expanded mask buffer is smaller than original vocabulary mask"
        );
        out.fill(0);

        for (word_index, &word) in internal_mask.iter().enumerate() {
            let class_base = word_index * 64;
            if class_base >= self.num_classes() {
                break;
            }
            let mut selected = word;
            while selected != 0 {
                let bit = selected.trailing_zeros() as usize;
                let class_id = class_base + bit;
                if class_id >= self.num_classes() {
                    break;
                }
                for &(output_word, output_bits) in &self.class_output_masks[class_id] {
                    out[output_word as usize] |= output_bits;
                }
                selected &= selected - 1;
            }
        }
    }
}

/// A grammar, vocabulary, and complete set of extern bindings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BoundarySummaryPolicy {
    /// Do not prepare or persist the static-composition candidate summary.
    Disabled,
    /// Prepare the summary only when a later static composition actually needs
    /// it. This is the default so ordinary standalone compilation pays no tax.
    #[default]
    LazyOnCompose,
    /// Prepare the bounded conservative summary during component compilation so
    /// later composition can consume it without first-touch analysis.
    BoundedPrepare,
}

/// A grammar, vocabulary, and complete set of extern bindings.
#[derive(Debug, Clone)]
pub struct ConstraintSpec<'a> {
    grammar: Grammar,
    vocab: &'a Vocab,
    token_bindings: BTreeMap<String, Vec<(u32, u32)>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
    unbound_grammar_names: Vec<String>,
    // Only explicit module compilation may preserve unbound source token
    // slots. Legacy specification compilation keeps its validation contract.
    allow_open_source_tokens: bool,
    automatic_boundary_selection: bool,
    // These belong to this source component. Register their terminal identity
    // before linking: another component may independently use the same hidden
    // token ID, which is not a globally unique linker coordinate.
    open_token_placeholders: BTreeMap<String, u32>,
    boundary_trigger_detail: BoundaryTriggerDetail,
    boundary_summary_policy: BoundarySummaryPolicy,
}

/// Builder for [`ConstraintSpec`].
#[derive(Debug)]
pub struct ConstraintSpecBuilder<'a> {
    grammar: Grammar,
    vocab: &'a Vocab,
    declared_externals: BTreeSet<String>,
    token_bindings: BTreeMap<String, Vec<(u32, u32)>>,
    grammar_bindings: BTreeMap<String, GrammarBinding<'a>>,
    boundary_trigger_detail: BoundaryTriggerDetail,
    boundary_summary_policy: BoundarySummaryPolicy,
}

/// Internal input accepted by [`ConstraintSpecBuilder::bind_grammar`].
#[doc(hidden)]
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum GrammarBinding<'a> {
    Source(Grammar),
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

impl<'a> IntoGrammarBinding<'a> for Grammar {
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
        grammar: Grammar,
        vocab: &'a Vocab,
    ) -> Result<ConstraintSpecBuilder<'a>> {
        ConstraintSpecBuilder::new(grammar, vocab)
    }

    /// Compile this specification into a [`Constraint`](crate::Constraint).
    pub fn compile(&self) -> Result<RuntimeConstraint> {
        Ok(*self.compile_boxed()?)
    }

    // Keep the large native value off compiler/join frames that can remain
    // live while Rayon executes another source compilation on this worker.
    fn compile_boxed(&self) -> Result<Box<RuntimeConstraint>> {
        with_deep_stack(|| {
            let mut constraint = self.compile_static_with_trigger_boxed()?;
            // A constraint with unresolved external grammars is explicitly a
            // reusable late-bind parent.  Cache its exact tokenizer-reset
            // terminal -> model-token relation now, while compilation already owns
            // the vocabulary, instead of rescanning the vocabulary on every later
            // bind.  The cache is part of composition metadata and therefore
            // survives ordinary save/load.
            if !constraint.late_grammar_slots.is_empty() {
                constraint.ensure_composition_reset_tokens_by_terminal();
            }
            match self.boundary_summary_policy {
                BoundarySummaryPolicy::Disabled => {
                    let _ = constraint.boundary_candidate_summary.set(
                        crate::runtime::BoundaryCandidateSummary::Unknown {
                            reason: crate::runtime::SummaryUnavailable::Disabled,
                        },
                    );
                }
                BoundarySummaryPolicy::LazyOnCompose => {}
                BoundarySummaryPolicy::BoundedPrepare => {
                    crate::compiler::boundary_candidates::prepare_boundary_candidate_summary(
                        &constraint,
                        self.vocab,
                    );
                }
            }
            // Keep first-save latency bounded without charging serialization to
            // every ordinary compile. Static virtual-residual constraints can carry
            // very large runtime sections, while exceptionally large parser-template
            // caches are expensive to encode on demand. Prime only those structural
            // tails; ordinary constraints serialize when save() is actually called.
            if should_prime_first_save_artifact(&constraint) {
                constraint.cache_serialized_artifact_for_save();
            }
            Ok(constraint)
        })
    }

    fn compile_final(&self, optimization: Engine) -> Result<RuntimeConstraint> {
        match optimization {
            Engine::TokenTrie => Ok(self.compile_dynamic()?.into_constraint()),
            Engine::MergedTokenTrie => {
                let dynamic = self.compile_dynamic()?;
                collapse_dynamic_alternatives(
                    dynamic.into_constraints(),
                    self.vocab,
                    SegmentedBoundaryBackend::Dynamic,
                )
            }
            Engine::Auto | Engine::ParserDwa => self.compile(),
        }
    }

    fn compile_static_with_trigger_boxed(&self) -> Result<Box<RuntimeConstraint>> {
        let mut constraint = self.compile_static_uncached()?;
        constraint
            .build_boundary_trigger(self.boundary_trigger_detail)
            .map_err(Error::Compilation)?;
        Ok(constraint)
    }

    fn register_open_token_placeholders(&self, parent: &mut RuntimeConstraint) -> Result<()> {
        for (name, &placeholder_id) in &self.open_token_placeholders {
            let terminals = parent.special_token_terminals.iter()
                .filter(|special| special.token_id == placeholder_id)
                .map(|special| special.terminal_id).collect::<BTreeSet<_>>();
            if terminals.len() != 1 {
                return Err(Error::Compilation(format!(
                    "compiled external token {name:?} did not resolve to exactly one component-local terminal",
                )));
            }
            parent.late_grammar_slots.push(crate::runtime::LateGrammarSlot {
                name: name.clone(),
                terminal_id: *terminals.iter().next().expect("length checked"),
            });
        }
        if !self.open_token_placeholders.is_empty() {
            parent.serialized_artifact_cache = None;
            if parent.sanitize_late_grammar_placeholder_token_domain() {
                parent.rebuild_runtime_caches();
            }
        }
        Ok(())
    }

    #[inline(never)]
    fn compile_static_parent_boxed(&self) -> Result<Box<RuntimeConstraint>> {
        let token_bindings = self.token_binding_refs();
        let parent=compile_static_source(&self.grammar,self.vocab,&token_bindings)?;
        Ok(Box::new(parent))
    }

    fn compile_static_uncached(&self) -> Result<Box<RuntimeConstraint>> {
        let compile_parent = || self.compile_static_parent_boxed();
        if self.grammar_bindings.is_empty() {
            let mut parent = compile_parent()?;
            self.register_open_token_placeholders(&mut parent)?;
            return Ok(parent);
        }

        // Parent-local compilation and child-local compilation have no
        // semantic dependency. Run them in the same Rayon pool and join only
        // before boundary construction, which genuinely needs both artifacts.
        let (parent, children) = rayon::join(
            compile_parent,
            || self.compile_children(ChildCompileMode::Static),
        );
        let mut parent = parent?;
        self.register_open_token_placeholders(&mut parent)?;
        let children = children?;
        let children = prepare_compiled_children(
            children, self.vocab, SegmentedBoundaryBackend::StaticParserDwa,
        )?;
        let boundary_backend = if self.automatic_boundary_selection {
            select_supported_boundary(&parent, &children)?
        } else {
            SegmentedBoundaryBackend::StaticParserDwa
        };
        compose_boxed_parent(parent, &children, self.vocab, boundary_backend)
    }

    /// Compile this specification into a [`DynamicConstraint`].
    pub fn compile_dynamic(&self) -> Result<DynamicConstraint> {
        Ok(*self.compile_dynamic_boxed()?)
    }

    fn compile_dynamic_boxed(&self) -> Result<Box<DynamicConstraint>> {
        let mut constraint = self.compile_dynamic_uncached()?;
        for component in constraint.constraints_mut() {
            // Retain the supplied shared vocabulary, just as static compilation
            // does. Reconstructing it from every token's bytes on the first bind
            // costs more than compiling many small dynamic components.
            let _ = component.late_bind_vocab.set(self.vocab.clone());
            component
                .build_boundary_trigger(self.boundary_trigger_detail)
                .map_err(Error::Compilation)?;
            match self.boundary_summary_policy {
                BoundarySummaryPolicy::Disabled => {
                    let _ = component.boundary_candidate_summary.set(
                        crate::runtime::BoundaryCandidateSummary::Unknown {
                            reason: crate::runtime::SummaryUnavailable::Disabled,
                        },
                    );
                }
                BoundarySummaryPolicy::LazyOnCompose => {}
                BoundarySummaryPolicy::BoundedPrepare => {
                    crate::compiler::boundary_candidates::prepare_boundary_candidate_summary(
                        component,
                        self.vocab,
                    );
                }
            }
        }
        Ok(constraint)
    }

    #[inline(never)]
    fn compile_dynamic_parent_boxed(&self) -> Result<Box<DynamicConstraint>> {
        let token_bindings = self.token_binding_refs();
        let parent=compile_dynamic_source(&self.grammar,self.vocab,&token_bindings)?;
        Ok(Box::new(parent))
    }

    fn compile_dynamic_uncached(&self) -> Result<Box<DynamicConstraint>> {
        if self.grammar_bindings.is_empty() {
            return self.compile_dynamic_parent_boxed();
        }
        if !matches!(self.grammar.source,GrammarSource::Glrm(_)|GrammarSource::Constructed) {
            return Err(Error::Compilation(
                "external grammar bindings require a GLRM grammar".to_owned(),
            ));
        }
        let (parents, children) = rayon::join(
            || self.compile_dynamic_parent_boxed(),
            || self.compile_children(ChildCompileMode::Dynamic),
        );
        let parents = parents?;
        let children = prepare_compiled_children(
            children?, self.vocab, SegmentedBoundaryBackend::Dynamic,
        )?;
        compose_boxed_dynamic_parents(parents, &children, self.vocab)
    }

    fn token_binding_refs(&self) -> Vec<(&str, &[(u32, u32)])> {
        self.token_bindings
            .iter()
            .map(|(name, ids)| (name.as_str(), ids.as_slice()))
            .collect()
    }

    fn compile_children(
        &self,
        mode: ChildCompileMode,
    ) -> Result<Vec<(String, CompiledChild<'_>)>> {
        let bindings = self.grammar_bindings.iter().collect::<Vec<_>>();
        bindings
            .par_iter()
            .map(|(name, binding)| {
                Ok((
                    (*name).clone(),
                    binding.compile(self.vocab, mode, self.allow_open_source_tokens)?,
                ))
            })
            .collect()
    }

    fn targets(&self, vocab: &Vocab) -> bool {
        self.vocab.same_model_vocab(vocab)
    }
}

impl<'a> ConstraintSpecBuilder<'a> {
    fn new(grammar: Grammar, vocab: &'a Vocab) -> Result<Self> {
        let (grammar, source_bindings) = grammar.into_source_only_and_bindings();
        grammar.validate_ebnf(vocab)?;
        let declarations = grammar.declarations()?;
        let declared_externals = declarations.names.iter().cloned().collect::<BTreeSet<_>>();
        let mut token_bindings = BTreeMap::new();
        let mut grammar_bindings = BTreeMap::new();
        for (name, value) in source_bindings {
            match value {
                GrammarValue::TokenSet(tokens) => {
                    if !declared_externals.contains(&name) {
                        return Err(Error::Compilation(format!(
                            "there is no token extern named {name:?}",
                        )));
                    }
                    if let Some(id) = vocab.first_unknown_token_id(&tokens) {
                        return Err(Error::Compilation(format!(
                            "extern {name:?} is bound to token ID {id}, which is not in the vocabulary",
                        )));
                    }
                    token_bindings.insert(name, tokens.range_pairs().to_vec());
                }
                GrammarValue::Source(child) => {
                    if !declared_externals.contains(&name) {
                        return Err(Error::Compilation(format!(
                            "there is no grammar extern named {name:?}",
                        )));
                    }
                    grammar_bindings.insert(name, GrammarBinding::Source(child.as_ref().clone()));
                }
            }
        }
        for (name, binding) in &mut grammar_bindings {
            binding.bind_target(vocab, name)?;
        }
        Ok(Self {
            grammar,
            vocab,
            declared_externals,
            token_bindings,
            grammar_bindings,
            boundary_trigger_detail: BoundaryTriggerDetail::None,
            boundary_summary_policy: BoundarySummaryPolicy::LazyOnCompose,
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

    /// Configure preparation of the grammar-aware static composition boundary
    /// candidate summary. This is independent of [`BoundaryTriggerDetail`],
    /// which controls dynamic-runtime trigger metadata.
    pub fn boundary_summary_policy(mut self, policy: BoundarySummaryPolicy) -> Self {
        self.boundary_summary_policy = policy;
        self
    }

    /// Bind an `extern NAME;` declaration to token IDs.
    pub fn bind_token(
        mut self,
        name: impl AsRef<str>,
        token_ids: impl IntoIterator<Item = u32>,
    ) -> Result<Self> {
        let name = name.as_ref();
        self.require_declared(name)?;
        check_binding_kind(name,self.grammar.declarations()?.kinds[name],false)?;
        if self.token_bindings.contains_key(name) || self.grammar_bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "extern {name:?} is already bound",
            )));
        }
        let token_ids = token_ids.into_iter().collect::<Vec<_>>();
        let unique = token_ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != token_ids.len() {
            return Err(Error::Compilation(format!(
                "the token IDs bound to extern {name:?} contain a duplicate",
            )));
        }
        self.token_bindings
            .insert(name.to_owned(), range_set_blaze::RangeSetBlaze::from_iter(unique)
                .ranges().map(|range| (*range.start(), *range.end())).collect());
        Ok(self)
    }

    /// Bind an `extern NAME;` to a source, spec, or compiled constraint.
    #[allow(private_bounds)]
    pub fn bind_grammar<T>(mut self, name: impl AsRef<str>, realization: T) -> Result<Self>
    where
        T: IntoGrammarBinding<'a>,
    {
        let name = name.as_ref();
        self.require_declared(name)?;
        check_binding_kind(name,self.grammar.declarations()?.kinds[name],true)?;
        if self.grammar_bindings.contains_key(name) || self.token_bindings.contains_key(name) {
            return Err(Error::Compilation(format!(
                "extern {name:?} is already bound",
            )));
        }
        let mut realization = realization.into_grammar_binding();
        realization.bind_target(self.vocab, name)?;
        self.grammar_bindings.insert(name.to_owned(), realization);
        Ok(self)
    }

    /// Finish the specification, retaining any unresolved external slots.
    ///
    /// External slots may remain unresolved. Their names are retained in
    /// the compiled constraint and can be supplied later with
    /// [`RuntimeConstraint::bind_grammar`] or [`DynamicConstraint::bind_grammar`].
    pub fn build(self) -> Result<ConstraintSpec<'a>> {
        let unbound_grammar_names = self
            .declared_externals
            .iter()
            .filter(|name| !self.grammar_bindings.contains_key(*name) && !self.token_bindings.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        Ok(ConstraintSpec {
            grammar: self.grammar,
            vocab: self.vocab,
            token_bindings: self.token_bindings,
            grammar_bindings: self.grammar_bindings,
            unbound_grammar_names,
            allow_open_source_tokens: false,
            automatic_boundary_selection: false,
            open_token_placeholders: BTreeMap::new(),
            boundary_trigger_detail: self.boundary_trigger_detail,
            boundary_summary_policy: self.boundary_summary_policy,
        })
    }

    fn require_declared(&self, name: &str) -> Result<()> {
        if self.declared_externals.contains(name) {
            return Ok(());
        }
        Err(Error::Compilation(format!("there is no extern named {name:?}")))
    }
}

#[derive(Clone, Copy)]
enum ChildCompileMode {
    Static,
    Dynamic,
}

enum CompiledChild<'a> {
    StaticBorrowed(&'a RuntimeConstraint),
    StaticOwned(Box<RuntimeConstraint>),
    DynamicBorrowed(&'a DynamicConstraint),
    DynamicOwned(Box<DynamicConstraint>),
}

impl CompiledChild<'_> {
    fn into_constraints(self) -> Vec<RuntimeConstraint> {
        match self {
            Self::StaticBorrowed(constraint) => vec![constraint.clone()],
            Self::StaticOwned(constraint) => vec![*constraint],
            Self::DynamicBorrowed(constraint) => constraint.clone_constraints(),
            Self::DynamicOwned(constraint) => constraint.clone_constraints(),
        }
    }
}

fn static_constraint_targets(constraint: &RuntimeConstraint, vocab: &Vocab) -> bool {
    constraint
        .late_bind_vocab
        .get()
        .map_or_else(
            || constraint.token_bytes_match_vocab(vocab),
            |compiled_vocab| compiled_vocab.same_model_vocab(vocab),
        )
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
                "the module bound to extern {name:?} was precompiled for a different vocabulary",
            )))
        }
    }

    fn compile<'a>(
        &'a self,
        vocab: &Vocab,
        mode: ChildCompileMode,
        allow_open_source_tokens: bool,
    ) -> Result<CompiledChild<'a>> {
        match self {
            Self::Source(grammar) => match mode {
                ChildCompileMode::Static => {
                    if !allow_open_source_tokens
                        && let Some(name) = grammar.unresolved_external_names()?.first() {
                            return Err(Error::Compilation(format!(
                                "extern {name:?} is unbound; bind it before compiling, or precompile to leave it open",
                            )));
                        }
                    let module = grammar.prepare_unlinked(vocab, BuildOptions::default())?;
                    Ok(CompiledChild::StaticOwned(Box::new(module.materialize_template_components(Engine::Auto)?)))
                }
                ChildCompileMode::Dynamic => {
                    let spec = ConstraintSpec::builder(grammar.clone(), vocab)?.build()?;
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic_boxed()?))
                }
            },
            Self::Spec(spec) => match mode {
                ChildCompileMode::Static => {
                    Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_boxed()?))
                }
                ChildCompileMode::Dynamic => {
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic_boxed()?))
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
            Self::Source(grammar) => match mode {
                ChildCompileMode::Static => {
                    if let Some(name) = grammar.unresolved_external_names()?.first() {
                        return Err(Error::Compilation(format!(
                            "extern {name:?} is unbound; bind it before compiling, or precompile to leave it open",
                        )));
                    }
                    let module = grammar.prepare_unlinked(vocab, BuildOptions::default())?;
                    Ok(CompiledChild::StaticOwned(Box::new(module.materialize_template_components(Engine::Auto)?)))
                }
                ChildCompileMode::Dynamic => {
                    let spec = ConstraintSpec::builder(grammar, vocab)?.build()?;
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic_boxed()?))
                }
            },
            Self::Spec(spec) => match mode {
                ChildCompileMode::Static => {
                    Ok(CompiledChild::StaticOwned(spec.compile_static_with_trigger_boxed()?))
                }
                ChildCompileMode::Dynamic => {
                    Ok(CompiledChild::DynamicOwned(spec.compile_dynamic_boxed()?))
                }
            },
            Self::StaticBorrowed(constraint) => Ok(CompiledChild::StaticBorrowed(constraint)),
            Self::StaticOwned(constraint) => Ok(CompiledChild::StaticOwned(Box::new(
                Arc::try_unwrap(constraint).unwrap_or_else(|shared| (*shared).clone()),
            ))),
            Self::DynamicBorrowed(constraint) => Ok(CompiledChild::DynamicBorrowed(constraint)),
            Self::DynamicOwned(constraint) => Ok(CompiledChild::DynamicOwned(Box::new(
                Arc::try_unwrap(constraint).unwrap_or_else(|shared| (*shared).clone()),
            ))),
        }
    }
}

/// Every compiled component must already own its native parser; linking
/// cannot materialize an LR-backed Constraint as an intermediate.
fn require_composable_parser(constraint: &RuntimeConstraint) -> Result<()> {
    if !constraint.has_template_parser() || constraint.table.is_present() {
        return Err(Error::Compilation(
            "a composition can't mix parts precompiled with the LrTable and TemplateDfa parser backends".into(),
        ));
    }
    Ok(())
}

fn prepare_compiled_children(
    children: Vec<(String, CompiledChild<'_>)>,
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Vec<(String, Arc<RuntimeConstraint>)>> {
    children
        .into_iter()
        .map(|(name, child)| {
            let alternatives = child.into_constraints();
            for alternative in &alternatives { require_composable_parser(alternative)?; }
            let mut constraint = collapse_dynamic_alternatives(
                alternatives, vocab, boundary_backend,
            )?;
            // Root termination is policy, not an embedded grammar terminal.
            // The native artifact cache already contains body-only bytes.
            constraint.end_tokens = Arc::from([]);
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
        source.push_str(&format!("extern alternative_{index};\n"));
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

#[inline(never)]
fn compose_boxed_dynamic_parents(
    parents: Box<DynamicConstraint>,
    children: &[(String, Arc<RuntimeConstraint>)],
    vocab: &Vocab,
) -> Result<Box<DynamicConstraint>> {
    let alternatives = parents.clone_constraints().into_iter()
        .map(|parent| compose_named_children(
            parent, children, vocab, SegmentedBoundaryBackend::Dynamic,
        ))
        .collect::<Result<Vec<_>>>()?;
    Ok(Box::new(DynamicConstraint::from_constraints(alternatives)))
}

// Unbox only after the independent compiler jobs have completed. Keeping this
// in a separate frame prevents the by-value linker slots from occupying every
// suspended source-compiler frame in unoptimized builds.
#[inline(never)]
fn compose_boxed_parent(
    parent: Box<RuntimeConstraint>,
    children: &[(String, Arc<RuntimeConstraint>)],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<Box<RuntimeConstraint>> {
    compose_named_children(*parent, children, vocab, boundary_backend).map(Box::new)
}

fn compose_named_children(
    parent: RuntimeConstraint,
    children: &[(String, Arc<RuntimeConstraint>)],
    vocab: &Vocab,
    boundary_backend: SegmentedBoundaryBackend,
) -> Result<RuntimeConstraint> {
    if parent.has_template_parser() {
        let mut linked = crate::runtime::parser_backend::link::compose(parent, children, vocab)?;
        if matches!(boundary_backend, SegmentedBoundaryBackend::StaticParserDwa) {
            match crate::compiler::template_boundary::install(&mut linked, vocab) {
                Ok(()) => {}
                // The backend was selected heuristically. When the exact static
                // constructor declines a shard for a resource budget, the
                // segmented dynamic boundary that `compose` already installed
                // is the exact fallback; it supports the same nested shape.
                Err(error) if crate::compiler::template_boundary::is_template_budget_decline(&error) => {
                    if crate::env::var_os("GLRMASK_PROFILE_COMPILE_SUMMARY").is_some() {
                        eprintln!("[glrmask/profile][static_template_boundary] fallback=dynamic reason=budget_decline");
                    }
                }
                Err(error) => return Err(error),
            }
        }
        return Ok(linked);
    }
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
    // This branch is only for ordinary retained-LR Dynamic components.
    // Native components were dispatched above and may never enter LR machinery.
    if !parent.table.is_present() || present.iter().any(|&i|
        children[i].1.has_template_parser() || !children[i].1.table.is_present()) {
        return Err(Error::Compilation("a composition can't mix parts precompiled with the LrTable and TemplateDfa parser backends".into()));
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
    grammar: &Grammar,
    vocab: &Vocab,
    token_bindings: &[(&str, &[(u32, u32)])],
) -> Result<RuntimeConstraint> {
    match &grammar.source {
        GrammarSource::Ebnf(source) => RuntimeConstraint::from_ebnf(source, vocab),
        GrammarSource::Lark(source) => RuntimeConstraint::from_lark(source, vocab),
        GrammarSource::JsonSchema(source) => RuntimeConstraint::from_json_schema(source, vocab),
        GrammarSource::Glrm(_) | GrammarSource::Constructed => {
            grammar.validate_exact_target(vocab)?;
            crate::import::compile_component_shell(grammar.component()?,vocab,token_bindings,false)
        },
    }
}

fn compile_dynamic_source(
    grammar: &Grammar,
    vocab: &Vocab,
    token_bindings: &[(&str, &[(u32, u32)])],
) -> Result<DynamicConstraint> {
    match &grammar.source {
        GrammarSource::Ebnf(source) => DynamicConstraint::from_ebnf(source, vocab),
        GrammarSource::Lark(source) => DynamicConstraint::from_lark(source, vocab),
        GrammarSource::JsonSchema(source) => DynamicConstraint::from_json_schema(source, vocab),
        GrammarSource::Glrm(_) | GrammarSource::Constructed => {
            grammar.validate_exact_target(vocab)?;
            Ok(DynamicConstraint::from_constraints(vec![crate::import::compile_component_shell(grammar.component()?,vocab,token_bindings,true)?]))
        },
    }
}

#[cfg(any(test, feature = "internal-api"))]
impl RuntimeConstraint {
    /// Compile `grammar` into a [`Constraint`](crate::Constraint) for `vocab`.
    pub fn compile(grammar: Grammar, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile()
    }

    /// Bind one retained `extern NAME;` slot using a fully compiled boundary.
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

    /// Bind one retained `extern NAME;` slot using the dynamic boundary walker.
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
        crate::compiler::boundary_candidates::persist_boundary_candidate_summary(self, vocab);
        if crate::env::var_os("GLRMASK_PREPARE_BOUNDARY_COMPLETION").is_some() {
            crate::compiler::boundary_precomputed_completion::prepare_component(self, vocab)
                .map_err(crate::GlrMaskError::Compilation)?;
        }
        Ok(())
    }
}

#[cfg(any(test, feature = "internal-api"))]
impl DynamicConstraint {
    /// Compile `grammar` into a [`DynamicConstraint`] for `vocab`.
    pub fn compile(grammar: Grammar, vocab: &Vocab) -> Result<Self> {
        ConstraintSpec::builder(grammar, vocab)?.build()?.compile_dynamic()
    }


    /// Compile `grammar` into a dynamic constraint whose local mask walk uses
    /// one representative per proven vocabulary-equivalence class.
    ///
    /// The complete original vocabulary is still retained for commits,
    /// composition compatibility, and boundary triggers. Consequently a
    /// partition-optimized constraint can be composed with ordinary static or
    /// dynamic constraints without changing the public token coordinate.
    pub fn compile_with_vocab_partition(grammar: Grammar, vocab: &Vocab) -> Result<Self> {
        if !grammar.bindings.is_empty() {
            return Err(Error::Compilation(
                "compile_with_vocab_partition does not yet compile source-level bound subgrammars; compile the components first and bind the compiled constraints"
                    .to_owned(),
            ));
        }
        match &grammar.source {
            GrammarSource::Ebnf(source) => Self::from_ebnf_with_vocab_partition(source, vocab),
            GrammarSource::Lark(source) => Self::from_lark_with_vocab_partition(source, vocab),
            GrammarSource::JsonSchema(source) => {
                Self::from_json_schema_with_vocab_partition(source, vocab)
            }
            GrammarSource::Glrm(source) => {
                Self::from_glrm_grammar_with_vocab_partition(source, vocab)
            }
            GrammarSource::Constructed=>crate::import::compile_bounded_template_from_named(grammar.component()?.lower_closed()?,vocab,crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,None),
        }
    }

    /// Bind one retained `extern NAME;` slot using a fully compiled boundary.
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

    /// Bind one retained `extern NAME;` slot using the dynamic boundary walker.
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

pub(crate) fn constraint_vocab(constraint: &RuntimeConstraint) -> Vocab {
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

// Native precompiled components are immutable across this materialization.
// Revalidate the persisted certificate in the same coordinate as the linker;
// recompute unavailable/stale certificates before installing any parser view.
fn prepare_immutable_template_component_boundary_summary(
    constraint: &mut RuntimeConstraint,
    vocab: &Vocab,
) {
    if matches!(
        crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(constraint, vocab),
        Ok(Some(_)),
    ) {
        constraint.serialized_artifact_cache = None;
        return;
    }
    crate::compiler::boundary_candidates::persist_boundary_candidate_summary(constraint, vocab);
}

fn apply_final_boundary_trigger(constraint: &mut RuntimeConstraint, detail: BoundaryTriggerDetail) -> Result<()> {
    if detail == BoundaryTriggerDetail::None {
        constraint.boundary_trigger = crate::runtime::BoundaryTrigger::None;
        constraint.serialized_artifact_cache = None;
        return Ok(());
    }
    if detail == BoundaryTriggerDetail::Exact && !constraint.has_template_parser() {
        return Err(Error::Compilation("Exact boundary triggers require native template components; retained-LR Dynamic is unsupported".into()));
    }
    constraint.build_boundary_trigger(detail).map_err(Error::Compilation)
}

fn ensure_runnable_constraint(constraint: &RuntimeConstraint) -> Result<()> {
    if let Some(slot) = constraint.late_grammar_slots.first() {
        return Err(Error::Compilation(format!(
            "extern {:?} is unbound; bind it before compiling, or precompile to leave it open; bind it before compiling, or precompile to leave it open",
            slot.name,
        )));
    }
    Ok(())
}

/// The current bounded static linker requires nonnullable child links. Decide
/// from retained semantic metadata before construction, never by catching a
/// compiler error or silently falling back during mask generation.
fn select_supported_boundary(
    parent: &RuntimeConstraint,
    children: &[(String, Arc<RuntimeConstraint>)],
) -> Result<SegmentedBoundaryBackend> {
    fn requires_dynamic_boundary(constraint: &RuntimeConstraint) -> bool {
        // A FastBuild child can carry exact virtual lexer residuals. Static
        // boundary shards cannot execute those coordinates, even when the
        // child is non-nullable. Select the supported backend from metadata
        // before linking rather than asking callers to know internal engines.
        constraint.tokenizer.has_virtual_residual_runtime()
            || constraint.static_dynamic_overlay.as_ref().is_some_and(|overlay| {
                overlay.segmented_parser_links.iter().any(|link| link.child_start_nullable)
                    || overlay.segmented_parser_components.iter()
                        .any(|component| requires_dynamic_boundary(&component.constraint))
            })
    }
    require_composable_parser(parent)?;
    for (_, child) in children { require_composable_parser(child)?; }
    if requires_dynamic_boundary(parent) {
        return Ok(SegmentedBoundaryBackend::Dynamic);
    }
    for (_, child) in children {
        if child.composition_start_nullable().map_err(Error::Compilation)?
            || requires_dynamic_boundary(child)
        {
            return Ok(SegmentedBoundaryBackend::Dynamic);
        }
    }
    Ok(SegmentedBoundaryBackend::StaticParserDwa)
}

fn compile_exact_token_adapter(vocab: &Vocab, token_ids: &[(u32, u32)]) -> Result<RuntimeConstraint> {
    use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};
    let named = NamedGrammar {
        start: "__exact_tokens".to_owned(),
        rules: vec![NamedRule {
            name: "__exact_tokens".to_owned(),
            expr: crate::import::choice_or_single(
                token_ids.iter().map(|&(start, end)| {
                    if start == end { GrammarExpr::SpecialToken(start) }
                    else { GrammarExpr::SpecialTokenRange(start, end) }
                }).collect(),
            ),
            is_terminal: false,
            is_internal: false,
        }],
        ignore: None,
        lexer_partitions: BTreeMap::new(),
        lexer_literal_partitions: BTreeMap::new(),
        default_lexer_partition: None,
    };
    crate::import::compile_from_named_grammar(
        named,
        vocab,
        "exact_token_binding",
        crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
        &[],
    )
}

fn boundary_for_optimization(
    parent: &RuntimeConstraint,
    children: &[(String, Arc<RuntimeConstraint>)],
    optimization: Engine,
) -> Result<SegmentedBoundaryBackend> {
    match optimization {
        Engine::TokenTrie | Engine::MergedTokenTrie => Ok(SegmentedBoundaryBackend::Dynamic),
        Engine::Auto | Engine::ParserDwa => {
            select_supported_boundary(parent, children)
        }
    }
}

impl PreparedGrammar {
    #[cfg(test)]
    pub(crate) fn boundary_test_inner(&self) -> &RuntimeConstraint {
        &self.inner
    }

    fn direct_slot_names(&self) -> BTreeSet<&str> {
        self.inner
            .late_grammar_slots
            .iter()
            .map(|slot| slot.name.as_str())
            .collect()
    }

    fn first_open_slot(&self) -> Result<Option<String>> {
        Ok(graph::prepared_first(self)?.map(|r|r.name()))
    }

    fn targets_vocab(&self, vocab: &Vocab) -> bool {
        graph::prepared_targets(self,vocab)
    }

    fn validate_slot_manifest(&self) -> Result<()> {
        graph::prepared_summary(self).map(|_|())
    }

    fn validate_local_slot_manifest(&self, vocab: &Vocab) -> Result<()> {
        if self.slot_spans.iter().any(|(name,span)|!self.slot_kinds.contains_key(name)
            || span.start>span.end || span.line==0 || span.column==0) {
            return Err(Error::Serialization("invalid prepared requirement source coordinates".into()));
        }
        if !self.inner.end_tokens.is_empty() {
            return Err(Error::Serialization(
                "unlinked-constraint artifact contains final-root policy".to_owned(),
            ));
        }
        let direct_slots = self.direct_slot_names();
        if direct_slots.iter().any(|name|!self.slot_kinds.contains_key(*name)) {
            return Err(Error::Serialization("compiled slot missing from authored interface".into()));
        }
        if !static_constraint_targets(self.inner.as_ref(),vocab) {
            return Err(Error::Serialization("prepared component targets an incompatible vocabulary".into()));
        }
        for (name, binding) in &self.bindings {
            if name.is_empty() || !self.slot_kinds.contains_key(name) {
                return Err(Error::Serialization(format!(
                    "unlinked-constraint binding {name:?} has no compiled linker slot",
                )));
            }
            check_binding_kind(name,self.slot_kinds[name],!matches!(binding,ModuleBinding::TokenSet(_)))?;
            match binding {
                ModuleBinding::TokenSet(ids) => {
                    let mut previous = None;
                    for &(start, end) in ids.iter() {
                        if !vocab.contains_token_range(start, end)
                            || previous.is_some_and(|value| u64::from(value) + 1 >= u64::from(start))
                        {
                            return Err(Error::Serialization(format!(
                                "unlinked-constraint token-set binding {name:?} is invalid for its vocabulary",
                            )));
                        }
                        previous = Some(end);
                    }
                }
                ModuleBinding::Module(_) => {} // Validated once by the graph traversal.
                ModuleBinding::Constraint(child) => {
                    if !static_constraint_targets(child.as_ref(), vocab) {
                        return Err(Error::Serialization(format!(
                            "unlinked-constraint grammar binding {name:?} has incompatible vocabulary",
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn require_direct_slot(&self,name:&str)->Result<()> {
        if !self.slot_kinds.contains_key(name){return Err(Error::Compilation(format!("there is no unbound extern named {name:?}")));}
        if self.bindings.contains_key(name){return Err(Error::Compilation(format!("extern {name:?} is already bound")));}
        Ok(())
    }

    /// Unresolved paths, including slots inside attached children. Diagnostic expansion is bounded.
    #[cfg(any(test,feature="internal-api"))]
    #[doc(hidden)]
    pub fn requirements(&self)->Result<Vec<Requirement>> {
        graph::prepared_requirements(self)
    }

    pub fn unbound(&self)->Result<Vec<Requirement>> {graph::prepared_requirements(self)}

    /// Authored slots, including bound slots and distinct attached occurrences.
    pub fn externs(&self)->Result<Vec<Requirement>> {
        self.validate_slot_manifest()?;
        let mut pending=vec![(self,Vec::<String>::new())];let mut out=Vec::new();let mut visits=0;
        while let Some((module,prefix))=pending.pop() {
            visits+=1;if visits>131_072 {return Err(Error::Compilation("the composition expands to too many parts".into()));}
            for (name,&kind) in module.slot_kinds.iter() {
                let mut path=prefix.clone();path.extend(name.split("::").map(str::to_owned));
                out.push(Requirement {path,kind,source_span:module.slot_spans.get(name).copied()});
                if out.len()>65_536 {return Err(Error::Compilation("the composition expands to too many parts".into()));}
            }
            for (name,binding) in &module.bindings {if let ModuleBinding::Module(child)=binding {
                let mut path=prefix.clone();path.extend(name.split("::").map(str::to_owned));pending.push((child,path));
            }}
        }
        out.sort_by(|a,b|a.path.cmp(&b.path));Ok(out)
    }


    fn bind_value(&self, name: &str, value: UnlinkedValue<'_>) -> Result<Self> {
        let previous=graph::prepared_summary(self)?;
        if !self.slot_kinds.contains_key(name)
            && let Some((head,tail))=attached_path(name,&self.bindings) {
                let Some(ModuleBinding::Module(child))=self.bindings.get(head) else {
                    return Err(Error::Compilation(format!("external grammar {head:?} is not a prepared component")));
                };
                let child=child.bind_value(tail,value)?;
                let summary=graph::prepared_summary(&child)?;
                let mut next=self.clone();next.bindings.insert(head.to_owned(),ModuleBinding::Module(Arc::new(child)));
                next.analysis=graph::filled_cache(previous.changed(head,Some(summary)));
                graph::check_prepared_limits(&next)?;return Ok(next);
            }
        self.require_direct_slot(name)?;
        check_binding_kind(name,self.slot_kinds[name],matches!(value,UnlinkedValue::Prepared(_)|UnlinkedValue::StaticBorrowed(_)|UnlinkedValue::StaticOwned(_)))
            .map_err(|e|e.with_source_span(self.slot_spans.get(name).copied()))?;
        let vocab = constraint_vocab(self.inner.as_ref());
        let binding = match value {
            UnlinkedValue::Prepared(child) => {
                if !child.targets_vocab(&vocab){return Err(Error::Compilation("the bound modules were precompiled for different vocabularies".into()));}
                let _=graph::prepared_first(&child)?;
                ModuleBinding::Module(child)
            }
            UnlinkedValue::TokenSet(tokens) => {
                self.require_direct_slot(name)?;
                if let Some(id) = vocab.first_unknown_token_id(&tokens) {
                    return Err(Error::Compilation(format!(
                        "extern {name:?} is bound to token ID {id}, which is not in the vocabulary",
                    )));
                }
                ModuleBinding::TokenSet(self.event_adapters.leaf(tokens.range_pairs())?)
            }
            UnlinkedValue::StaticBorrowed(child) => {
                self.require_direct_slot(name)?;
                if !static_constraint_targets(child, &vocab) {
                    return Err(Error::Compilation(format!(
                        "the module bound to extern {name:?} was precompiled for a different vocabulary",
                    )));
                }
                ModuleBinding::Constraint(Arc::new(child.clone()))
            }
            UnlinkedValue::StaticOwned(child) => {
                self.require_direct_slot(name)?;
                if !static_constraint_targets(child.as_ref(), &vocab) {
                    return Err(Error::Compilation(format!(
                        "the module bound to extern {name:?} was precompiled for a different vocabulary",
                    )));
                }
                ModuleBinding::Constraint(child)
            }
        };
        let child_summary=match &binding { ModuleBinding::Module(child)=>Some(graph::prepared_summary(child)?),_=>None };
        let mut next = self.clone();
        next.bindings.insert(name.to_owned(), binding);
        next.analysis=graph::filled_cache(previous.changed(name,child_summary));
        graph::check_prepared_limits(&next)?;
        Ok(next)
    }

    /// Fill a declared slot once with `PreparedGrammar` or `TokenSet`. Attach a parent before binding descendants. Prepared children must use the same byte mapping and special-token IDs. Attached children are snapshots; failed binds leave this handle unchanged.
    #[allow(private_bounds)]
    pub fn bind<'a, T>(&mut self, name: impl AsRef<str>, value: T) -> Result<()>
    where T: IntoUnlinkedValue<'a> {
        let next=self.bind_value(name.as_ref(),value.into_unlinked_value())?;
        *self=next;
        Ok(())
    }

    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    #[allow(private_bounds)]
    pub fn bind_snapshot<'a, T>(&self,name:impl AsRef<str>,value:T)->Result<Self>
    where T:IntoUnlinkedValue<'a> {
        let mut next=self.clone();next.bind(name,value)?;Ok(next)
    }

    fn materialize(&self, optimization: Engine) -> Result<RuntimeConstraint> {
        self.validate_slot_manifest()?;
        if self.bindings.is_empty() {
            return Ok(self.inner.as_ref().clone());
        }
        let vocab = constraint_vocab(self.inner.as_ref());
        let active=self.direct_slot_names();
        let binding_entries = self.bindings.iter().filter(|(name,_)|active.contains(name.as_str())).collect::<Vec<_>>();
        if binding_entries.is_empty(){return Ok(self.inner.as_ref().clone());}
        let raw_children = binding_entries
            .par_iter()
            .map(|(name, binding)| {
                let mut child = match binding {
                    ModuleBinding::Module(module) => {
                        let mut child = module.materialize(optimization)?;
                        child.build_boundary_trigger(module.boundary_trigger).map_err(Error::Compilation)?;
                        child
                    },
                    ModuleBinding::Constraint(constraint) => constraint.as_ref().clone(),
                    ModuleBinding::TokenSet(ids) => {
                        ids.compiled(&vocab)?.as_ref().clone()
                    }
                };
                // Standalone termination is root policy, never an embedded body.
                child.end_tokens = Arc::from([]);
                Ok(((*name).clone(), child))
            })
            .collect::<Result<Vec<_>>>()?;
        let probe_children = raw_children
            .iter()
            .map(|(name, child)| (name.clone(), Arc::new(child.clone())))
            .collect::<Vec<_>>();
        let backend = boundary_for_optimization(
            self.inner.as_ref(),
            &probe_children,
            optimization,
        )?;
        let prepared = prepare_compiled_children(
            raw_children
                .into_iter()
                .map(|(name, child)| (name, CompiledChild::StaticOwned(Box::new(child))))
                .collect(),
            &vocab,
            backend,
        )?;
        compose_named_children(
            self.inner.as_ref().clone(),
            &prepared,
            &vocab,
            backend,
        )
    }

    /// Requires every slot to be bound. Uses default build options and inherits the prepared boundary request.
    pub fn link(&self) -> Result<RuntimeConstraint> {
        self.link_with(BuildOptions::default())
    }

    fn requires_template_linker(&self) -> bool {
        // Sparse regular frontends also have no LR action/rule inventory.
        // An explicit template request converts their finite depth-one
        // programs directly instead of sending them through the LR splicer.
        let needs_native = |constraint: &RuntimeConstraint| {
            constraint.has_template_parser() || constraint.uses_sparse_direct_regular_runtime()
        };
        needs_native(self.inner.as_ref()) || self.bindings.values().any(|binding| match binding {
            ModuleBinding::Module(module) => module.requires_template_linker(),
            ModuleBinding::Constraint(constraint) => needs_native(constraint.as_ref()),
            ModuleBinding::TokenSet(_) => false,
        })
    }

    fn materialize_template_components(&self, optimization: Engine) -> Result<RuntimeConstraint> {
        self.validate_slot_manifest()?;
        let vocab = constraint_vocab(self.inner.as_ref());
        let mut parent = self.inner.as_ref().clone();
        if parent.has_template_parser() {
            prepare_immutable_template_component_boundary_summary(
                &mut parent,
                &vocab,
            );
        }
        parent.install_template_parser()?;
        if self.bindings.is_empty() { return Ok(parent); }
        let active=self.direct_slot_names();
        let children = self.bindings.iter().filter(|(name,_)|active.contains(name.as_str())).map(|(name, binding)| {
            let mut child = match binding {
                ModuleBinding::Module(module) => module.materialize_template_components(optimization)?,
                ModuleBinding::Constraint(constraint) => constraint.as_ref().clone(),
                ModuleBinding::TokenSet(ids) => ids.compiled(&vocab)?.as_ref().clone(),
            };
            child.end_tokens = Arc::from([]);
            if child.has_template_parser() {
                prepare_immutable_template_component_boundary_summary(
                    &mut child,
                    &vocab,
                );
            }
            child.install_template_parser()?;
            if let ModuleBinding::Module(module) = binding {
                child.build_boundary_trigger(module.boundary_trigger).map_err(Error::Compilation)?;
            }
            Ok((name.clone(), Arc::new(child)))
        }).collect::<Result<Vec<_>>>()?;
        if children.is_empty(){return Ok(parent);}
        crate::error::catch_internal_invariant(|| {
            let mut linked = crate::runtime::parser_backend::link::compose_owned(parent, children, &vocab)?;
            if optimization == Engine::ParserDwa {
                crate::compiler::template_boundary::install(&mut linked, &vocab)?;
            }
            Ok(linked)
        })?
    }

    /// Requires all slots to be bound. Selects final optimization and root end tokens; an explicit boundary request overrides the prepared request. End tokens become allowed only at acceptance and must be committed before stopping.
    pub fn link_with(&self, options: BuildOptions) -> Result<RuntimeConstraint> {
        graph::prepared_preflight(self)?;
        if let Some(name) = self.first_open_slot()? {
            return Err(Error::Compilation(format!(
                "extern {name:?} is unbound; bind it before compiling, or precompile to leave it open",
            )));
        }
        let engine = options.engine()?;
        let mut constraint = if options.parser_backend == ParserBackend::TemplateDfa && self.requires_template_linker() {
            self.materialize_template_components(engine)?
        } else { self.materialize(engine)? };
        ensure_runnable_constraint(&constraint)?;
        if options.parser_backend == ParserBackend::TemplateDfa {
            constraint.install_template_parser()?;
        }
        apply_final_boundary_trigger(&mut constraint, options.boundary_trigger.unwrap_or(self.boundary_trigger))?;
        constraint.with_end_tokens(options.end_token_ids())
    }

    /// Preserves compiled machinery, vocabulary and unresolved slots. Shared definitions are serialized once. Artifact compatibility across GLRMask releases is not guaranteed.
    pub fn save(&self) -> Vec<u8> {
        with_deep_stack(|| prepared_artifact::save(self))
    }

    /// Rejects malformed or incompatible prepared artifacts. The supplied
    /// vocabulary must match the saved token-ID/byte fingerprint; loaded
    /// definitions share its caches. Open slots and shared definitions are
    /// preserved.
    pub fn load<'b>(bytes: impl Into<std::borrow::Cow<'b, [u8]>>, vocab: &Vocab) -> Result<Self> {
        with_deep_stack(|| prepared_artifact::load(bytes.into().as_ref(), Some(vocab)))
    }

}

const FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD: usize = 100_000;

fn should_prime_first_save_artifact(constraint: &RuntimeConstraint) -> bool {
    if retains_dynamic_component(constraint) {
        return false;
    }
    if constraint.tokenizer.has_virtual_residual_runtime() {
        return true;
    }
    let mut template_states = 0usize;
    for template in constraint
        .composition_parser_templates_by_terminal
        .iter()
        .flatten()
    {
        template_states = template_states.saturating_add(template.states.len());
        if template_states >= FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD {
            return true;
        }
    }
    false
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
        "there is no unbound grammar extern named {name:?}",
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
    let profile = crate::env::var_os("GLRMASK_PROFILE_PUBLIC_BIND").is_some();
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
    parents.first().ok_or_else(|| {
        Error::Compilation("dynamic parent has no alternatives".to_owned())
    })?;
    // Cache on the retained parent, not a temporary clone discarded after this
    // bind. Loaded constraints must reconstruct the vocabulary at most once.
    let vocab = constraint_vocab(&parent.inner);
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
fn native_dynamic_fixture(grammar: Grammar, vocab: &Vocab) -> Result<DynamicConstraint> {
    assert!(grammar.bindings.is_empty(), "fixture supplies compiled bindings separately");
    let constraint = match &grammar.source {
        // Keep the ordinary full-vocabulary lane in EBNF comparison fixtures,
        // while selecting its parser representation explicitly.
        GrammarSource::Ebnf(source) => {
            let named = crate::import::parse_ebnf_to_named(source)?;
            let factored = crate::grammar::factoring::factor_named_grammar(named);
            crate::compiler::pipeline::compile_dynamic_owned_with_backend(
                crate::grammar::ast::lower(&factored)?, vocab,
                crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
                crate::runtime::parser_backend::DynamicParserBackend::Native,
            )?
        }
        GrammarSource::Glrm(source) => crate::import::compile_dynamic_glrm_fixture(
            source, vocab, crate::runtime::parser_backend::DynamicParserBackend::Native,
        )?,
        GrammarSource::Lark(source) => DynamicConstraint::from_lark_with_bounded_template_parser(source, vocab)?,
        GrammarSource::JsonSchema(source) => DynamicConstraint::from_json_schema_with_bounded_template_parser(source, vocab)?,
        GrammarSource::Constructed=>crate::import::compile_bounded_template_from_named(grammar.component()?.lower_closed()?,vocab,crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,None)?,
    };
    assert!(constraint.clone_constraints().iter().all(|body|
        body.has_template_parser() && !body.table.is_present() && body.uses_dynamic_runtime()));
    Ok(constraint)
}

#[cfg(test)]
impl ConstraintSpec<'_> {
    /// These fixtures exercise native composition, independent of the ordinary
    /// Dynamic development override. Children must already be compiled native
    /// fixtures; source-child compilation is covered by the public API gates.
    fn compile_native_dynamic_fixture(&self) -> Result<DynamicConstraint> {
        assert!(self.token_bindings.is_empty());
        let parents = native_dynamic_fixture(self.grammar.clone(), self.vocab)?;
        let mut constraint = if self.grammar_bindings.is_empty() {
            parents
        } else {
            let children = prepare_compiled_children(
                self.compile_children(ChildCompileMode::Dynamic)?, self.vocab,
                SegmentedBoundaryBackend::Dynamic,
            )?;
            *compose_boxed_dynamic_parents(Box::new(parents), &children, self.vocab)?
        };
        for body in constraint.constraints_mut() {
            let _ = body.late_bind_vocab.set(self.vocab.clone());
            body.build_boundary_trigger(self.boundary_trigger_detail).map_err(Error::Compilation)?;
        }
        Ok(constraint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automata::lexer::tokenizer::Lexer;

    fn compose_owned_test_fixture() -> (Vocab, PreparedGrammar, RuntimeConstraint) {
        let vocab = Vocab::new(vec![
            (0, b"<".to_vec()), (1, b"a".to_vec()), (2, b">".to_vec()),
            (3, b"<a>".to_vec()), (4, b"a>".to_vec()), (5, b"<a".to_vec()),
            (6, b"b".to_vec()), (7, b"ab".to_vec()), (8, b"<ab>".to_vec()),
        ]).with_special_tokens([99]);
        let parent = Grammar::glrm("glrm 1; start root; extern grammar child; nt root = \"<\" child \">\";")
            .prepare(&vocab).unwrap();
        let child = Grammar::glrm("glrm 1; start value; nt value = \"a\" | \"ab\";")
            .compile(&vocab).unwrap().with_end_tokens(&[99]).unwrap();
        let parent = PreparedGrammar::load(parent.save(), &vocab).unwrap();
        let child = RuntimeConstraint::load(child.save(), &vocab).unwrap();
        assert!(child.deferred_composition_metadata_blob.is_some());
        assert!(!child.composition_link_metadata_materialized);
        (vocab, parent, child)
    }

    fn assert_compose_owned_behavior_equal(left: &RuntimeConstraint, right: &RuntimeConstraint) {
        let mut prefixes = vec![Vec::<u32>::new()];
        let mut saw_eof = false;
        for depth in 0..=4 {
            let mut next = Vec::new();
            for prefix in prefixes {
                let mut a = left.start(); let mut b = right.start();
                for &token in &prefix { assert!(a.commit_token(token)); assert!(b.commit_token(token)); }
                let before_a = a.mask(); let before_b = b.mask();
                assert_eq!(before_a, before_b, "mask at {prefix:?}");
                assert_eq!(a.is_accepting(), b.is_accepting());
                // The end token is admitted exactly when the grammar is
                // complete, and committing it makes the state accepting.
                let mut eof = a.clone();
                let allowed_eof = token_allowed(&before_a, 99);
                assert_eq!(eof.commit_token(99), allowed_eof, "root EOF policy");
                if allowed_eof { assert!(eof.is_accepting()); }
                for token in (0..=8).chain([99]) {
                    let mut trial_a = a.clone(); let mut trial_b = b.clone();
                    let ok_a = trial_a.commit_token(token);
                    let ok_b = trial_b.commit_token(token);
                    assert_eq!(ok_a, ok_b, "commit {token} at {prefix:?}");
                    assert_eq!(trial_a.mask(), trial_b.mask());
                    assert_eq!(trial_a.is_accepting(), trial_b.is_accepting());
                    if token == 99 && token_allowed(&before_a, token) { assert!(ok_a); saw_eof = true; }
                    if depth < 4 && token != 99 && token_allowed(&before_a, token) && ok_a {
                        let mut path = prefix.clone(); path.push(token); next.push(path);
                    }
                }
                assert_eq!(a.mask(), before_a, "cloned trial mutated original matcher");
                assert_eq!(b.mask(), before_b);
            }
            prefixes = next;
        }
        assert!(saw_eof, "fixture must exercise accepting EOF commit");
    }

    #[test]
    fn compose_owned_loaded_private_and_shared_children_preserve_inputs() {
        let (vocab, parent, child) = compose_owned_test_fixture();
        let parent_bytes = parent.save(); let parent_mask = parent.inner.start().mask();
        let child_bytes = child.save(); let child_mask = child.start().mask();
        let prepare = || {
            let mut p = parent.inner.as_ref().clone();
            prepare_immutable_template_component_boundary_summary(&mut p, &vocab);
            p.install_template_parser().unwrap();
            let mut c = child.clone(); c.end_tokens = Arc::from([]);
            prepare_immutable_template_component_boundary_summary(&mut c, &vocab);
            c.install_template_parser().unwrap();
            (p, Arc::new(c))
        };
        let (p, c) = prepare(); let private_pointer = Arc::as_ptr(&c);
        let owned = crate::runtime::parser_backend::link::compose_owned(p, vec![("child".into(), c)], &vocab)
            .unwrap().with_end_tokens(&[99]).unwrap();
        let retained = &owned.static_dynamic_overlay.as_ref().unwrap().segmented_parser_components[1].constraint;
        assert_eq!(Arc::as_ptr(retained), private_pointer, "private child was copied unnecessarily");
        let (p, c) = prepare(); let borrowed_pointer = Arc::as_ptr(&c);
        let borrowed = crate::runtime::parser_backend::link::compose(p, &[("child".into(), c.clone())], &vocab)
            .unwrap().with_end_tokens(&[99]).unwrap();
        assert_ne!(Arc::as_ptr(&borrowed.static_dynamic_overlay.as_ref().unwrap().segmented_parser_components[1].constraint), borrowed_pointer);
        let (p, c) = prepare(); let shared_bytes = c.save(); let shared_mask = c.start().mask();
        let shared = crate::runtime::parser_backend::link::compose_owned(p, vec![("child".into(), c.clone())], &vocab)
            .unwrap().with_end_tokens(&[99]).unwrap();
        assert_ne!(Arc::as_ptr(&shared.static_dynamic_overlay.as_ref().unwrap().segmented_parser_components[1].constraint), Arc::as_ptr(&c));
        assert!(!c.composition_link_metadata_materialized, "shared child mutated in place");
        assert_eq!(c.save(), shared_bytes); assert_eq!(c.start().mask(), shared_mask);
        assert_eq!(owned.save(), borrowed.save()); assert_eq!(owned.save(), shared.save());
        assert_compose_owned_behavior_equal(&owned, &borrowed);
        assert_compose_owned_behavior_equal(&owned, &shared);
        let reloaded = RuntimeConstraint::load(owned.save(), &vocab).unwrap();
        assert_compose_owned_behavior_equal(&owned, &reloaded);
        assert_eq!(parent.save(), parent_bytes); assert_eq!(parent.inner.start().mask(), parent_mask);
        assert_eq!(child.save(), child_bytes); assert_eq!(child.start().mask(), child_mask);
    }

    #[test]
    fn compose_owned_repeated_public_links_preserve_retained_inputs() {
        let (vocab, parent, child) = compose_owned_test_fixture();
        let parent_bytes = parent.save(); let parent_mask = parent.inner.start().mask();
        let child = Arc::new(child); let child_bytes = child.save(); let child_mask = child.start().mask();
        let options = MaskMode::C.build_options().parser_backend(ParserBackend::TemplateDfa).end_tokens([99]);
        let borrowed_bound = parent.bind_snapshot("child", child.as_ref()).unwrap();
        let owned_bound = parent.bind_snapshot("child", Arc::clone(&child)).unwrap();
        let borrowed_bytes = borrowed_bound.save(); let owned_bytes = owned_bound.save();
        let mut first = None;
        for _ in 0..3 {
            let a = borrowed_bound.link_with(options.clone()).unwrap();
            let b = owned_bound.link_with(options.clone()).unwrap();
            assert_compose_owned_behavior_equal(&a, &b);
            let bytes = a.save(); assert_eq!(bytes, b.save());
            if let Some(previous) = first.as_ref() { assert_eq!(&bytes, previous); } else { first = Some(bytes); }
            assert_eq!(borrowed_bound.save(), borrowed_bytes); assert_eq!(owned_bound.save(), owned_bytes);
            assert_eq!(parent.save(), parent_bytes); assert_eq!(parent.inner.start().mask(), parent_mask);
            assert_eq!(child.save(), child_bytes); assert_eq!(child.start().mask(), child_mask);
        }
        assert_eq!(RuntimeConstraint::load(first.unwrap(), &vocab).unwrap().parser_backend(), ParserBackend::TemplateDfa);
    }

    fn token_allowed(mask: &[u32], token_id: u32) -> bool {
        let word = token_id as usize / 32;
        let bit = token_id % 32;
        mask.get(word)
            .is_some_and(|bits| bits & (1u32 << bit) != 0)
    }

    #[test]
    fn immutable_template_certificate_reuse_recomputes_stale_versions_and_preserves_disabled() {
        use crate::runtime::{BoundaryCandidateSummary, OriginalTokenSet, SummaryUnavailable};
        let vocab = Vocab::new(vec![
            (0, b"[a]tail".to_vec()), (1, b"[]tail".to_vec()),
            (2, b"[".to_vec()), (3, b"a".to_vec()), (4, b"]".to_vec()),
        ]);
        let mut component = Grammar::glrm(
            r#"glrm 1; start value; nt value = "[" "a"? "]";"#,
        ).compile(&vocab).unwrap();
        crate::compiler::boundary_candidates::persist_boundary_candidate_summary(&mut component, &vocab);
        let expected = crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(&component, &vocab)
            .unwrap().expect("fresh component must carry a checked certificate");
        assert!(!expected.is_empty());
        let mut stale = component.boundary_candidate_summary.take().unwrap();
        let BoundaryCandidateSummary::Known { fingerprint, tokens, .. } = &mut stale else {
            panic!("expected known certificate");
        };
        fingerprint.algorithm_version -= 1;
        *tokens = OriginalTokenSet::from_sorted_unique(Vec::new(), vocab.max_token_id());
        component.boundary_candidate_summary.set(stale).unwrap();
        assert!(crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(&component, &vocab)
            .unwrap().is_none());
        prepare_immutable_template_component_boundary_summary(&mut component, &vocab);
        assert_eq!(crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(&component, &vocab)
            .unwrap(), Some(expected));
        component.boundary_candidate_summary.take();
        component.boundary_candidate_summary.set(BoundaryCandidateSummary::Unknown {
            reason: SummaryUnavailable::Disabled,
        }).unwrap();
        prepare_immutable_template_component_boundary_summary(&mut component, &vocab);
        assert!(matches!(component.boundary_candidate_summary.get(),
            Some(BoundaryCandidateSummary::Unknown { reason: SummaryUnavailable::Disabled })));
    }

    #[test]
    fn immutable_template_certificate_reuse_invalidates_unchanged_resave_cache() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec()), (1, b"ab".to_vec())]);
        let component = Grammar::glrm(r#"glrm 1; start value; nt value = "a";"#)
            .compile(&vocab).unwrap();
        let mut loaded = RuntimeConstraint::load(component.save(), &vocab).unwrap();
        let expected = crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(&loaded, &vocab)
            .unwrap().expect("loaded native component must carry a checked certificate");
        assert!(loaded.serialized_artifact_cache.is_some());
        prepare_immutable_template_component_boundary_summary(&mut loaded, &vocab);
        assert!(loaded.serialized_artifact_cache.is_none());
        assert_eq!(crate::compiler::boundary_candidates::persisted_boundary_candidate_ids(&loaded, &vocab)
            .unwrap(), Some(expected));
    }

    #[test]
    fn vocab_partition_covers_sparse_vocab_and_merges_identical_tokens() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (3, b"a".to_vec()),
            (7, b"b".to_vec()),
            (11, b"ab".to_vec()),
            (67, b"a".to_vec()),
        ]);
        let partition = VocabPartition::compile(Grammar::ebnf(r#"start ::= "a"+"#), &vocab)
            .unwrap();

        assert_eq!(partition.class_of(1), None);
        for token in [0, 3, 7, 11, 67] {
            assert!(partition.class_of(token).is_some(), "token {token} is unmapped");
        }
        assert_eq!(partition.class_of(0), partition.class_of(3));
        assert_eq!(partition.class_of(0), partition.class_of(67));
        let mut covered = partition.classes().iter().flatten().copied().collect::<Vec<_>>();
        covered.sort_unstable();
        assert_eq!(covered, vec![0, 3, 7, 11, 67]);
        for class in 0..partition.num_classes() as u32 {
            let representative = partition.representative(class).unwrap();
            assert_eq!(partition.class_of(representative), Some(class));
        }

        assert_eq!(partition.internal_mask_len(), partition.num_classes().div_ceil(64));
        assert_eq!(partition.original_mask_len(), 3);

        let class = partition.class_of(0).unwrap() as usize;
        let mut internal = vec![0u64; partition.internal_mask_len()];
        internal[class / 64] |= 1u64 << (class % 64);
        let expanded = partition.expand_mask(&internal);
        assert_ne!(expanded[0] & (1u32 << 0), 0);
        assert_ne!(expanded[0] & (1u32 << 3), 0);
        assert_eq!(expanded[0] & (1u32 << 1), 0);
        assert_ne!(expanded[2] & (1u32 << 3), 0);

        let mut reused = vec![u32::MAX; partition.original_mask_len() + 2];
        partition.fill_expanded_mask(&internal, &mut reused);
        assert_eq!(&reused[..partition.original_mask_len()], expanded.as_slice());
        assert_eq!(&reused[partition.original_mask_len()..], &[0, 0]);
    }

    #[test]
    fn vocab_partition_expands_multiple_classes_and_ignores_high_internal_bits() {
        let partition = VocabPartition {
            original_to_class: vec![0, u32::MAX, 1, 0, u32::MAX, 2, 2, u32::MAX, 1],
            classes: vec![vec![0, 3], vec![2, 8], vec![5, 6]],
            representatives: vec![0, 2, 5],
            class_output_masks: vec![
                vec![(0, (1u32 << 0) | (1u32 << 3))],
                vec![(0, (1u32 << 2) | (1u32 << 8))],
                vec![(0, (1u32 << 5) | (1u32 << 6))],
            ],
        };

        let expanded = partition.expand_mask(&[(1u64 << 0) | (1u64 << 2) | (1u64 << 63)]);
        assert_eq!(expanded, vec![(1u32 << 0) | (1u32 << 3) | (1u32 << 5) | (1u32 << 6)]);
    }

    #[test]
    fn vocab_partition_supports_json_schema_frontend() {
        let vocab = Vocab::new(vec![
            (0, b"null".to_vec()),
            (1, b"true".to_vec()),
            (2, b"false".to_vec()),
            (3, b"0".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let partition = VocabPartition::compile(
            Grammar::json_schema(r#"{"type":["null","boolean"]}"#),
            &vocab,
        )
        .unwrap();
        assert!(partition.num_classes() > 0);
        assert!(partition.num_classes() <= vocab.len());
    }

    #[test]
    fn explicit_vocab_partition_strategies_are_conservative() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"ba".to_vec()),
            (4, b"aa".to_vec()),
            (5, b"bb".to_vec()),
        ]);
        let source = r#"start ::= ("a" | "b")+"#;
        let compact = VocabPartition::compile_with_strategy(
            Grammar::ebnf(source),
            &vocab,
            VocabPartitionStrategy::Compact,
        )
        .unwrap();
        let dedicated = VocabPartition::compile_with_strategy(
            Grammar::ebnf(source),
            &vocab,
            VocabPartitionStrategy::Dedicated,
        )
        .unwrap();

        for class in dedicated.classes() {
            let mut compact_class = None;
            for &token_id in class {
                let current = compact.class_of(token_id).unwrap();
                assert!(compact_class.is_none_or(|expected| expected == current));
                compact_class = Some(current);
            }
        }
    }
    #[test]
    fn vocab_partition_rejects_bound_subgrammar_instead_of_ignoring_it() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let grammar = Grammar::glrm(
            "glrm 1; extern grammar child; start root; nt root = child;",
        )
        .bind_grammar("child", Grammar::ebnf(r#"start ::= "a""#))
        .unwrap();
        let error = VocabPartition::compile(grammar, &vocab).unwrap_err();
        assert!(error.to_string().contains("bound grammar values"));
    }

    #[test]
    fn partition_optimized_dynamic_mask_matches_ordinary_dynamic() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let grammar_source = "start ::= [ab]+".to_owned();
        let grammar = Grammar::ebnf(&grammar_source);
        let partition = VocabPartition::compile(grammar.clone(), &vocab).unwrap();
        assert_eq!(partition.class_of(0), partition.class_of(1));
        assert!(partition.num_classes() < vocab.len());

        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        assert_eq!(optimized.inner.token_bytes_count(), vocab.len());
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "O2 must not silently fall back to ordinary dynamic for tiny grammars",
        );
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .canonical_token_count()
                < ordinary
                    .inner
                    .dynamic_mask_vocab_for_runtime()
                    .canonical_token_count(),
            "optimized dynamic runtime did not reduce the mask vocabulary",
        );

        let ordinary_start = ordinary.start();
        let optimized_start = optimized.start();
        assert_eq!(ordinary_start.mask(), optimized_start.mask());

        for token in [0u32, 1, 2, 3, 4, 5] {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            assert!(ordinary_state.commit_token(token));
            assert!(optimized_state.commit_token(token));
            assert_eq!(ordinary_state.is_accepting(), optimized_state.is_accepting());
            assert_eq!(ordinary_state.mask(), optimized_state.mask(), "token {token}");
        }
    }

    #[test]
    fn singleton_literal_partition_matches_dynamic_masks_exactly() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"{".to_vec()),
            (2, b"}".to_vec()),
            (3, b"{}".to_vec()),
            (4, b"{}x".to_vec()),
            (5, b"{".to_vec()),
            (6, b"x{}".to_vec()),
        ]);
        let grammar_source = r#"start ::= "{" "}""#.to_owned();
        let grammar = Grammar::ebnf(&grammar_source);
        let partition = VocabPartition::compile(grammar.clone(), &vocab).unwrap();

        assert_eq!(partition.num_classes(), 4);
        assert_eq!(partition.class_of(0), partition.class_of(4));
        assert_eq!(partition.class_of(0), partition.class_of(6));
        assert_eq!(partition.class_of(1), partition.class_of(5));
        assert_ne!(partition.class_of(1), partition.class_of(2));
        assert_ne!(partition.class_of(1), partition.class_of(3));
        assert_ne!(partition.class_of(2), partition.class_of(3));

        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        let ordinary_start = ordinary.start();
        let optimized_start = optimized.start();
        assert_eq!(ordinary_start.mask(), optimized_start.mask());
        for token in [0u32, 1, 2, 3, 4, 5, 6] {
            assert_eq!(
                token_allowed(&ordinary_start.mask(), token),
                token_allowed(&optimized_start.mask(), token),
                "start mask differs for token {token}",
            );
        }

        for open in [1u32, 5] {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            assert!(ordinary_state.commit_token(open));
            assert!(optimized_state.commit_token(open));
            assert_eq!(ordinary_state.is_accepting(), optimized_state.is_accepting());
            assert_eq!(ordinary_state.mask(), optimized_state.mask());
        }

        let mut ordinary_done = ordinary.start();
        let mut optimized_done = optimized.start();
        assert!(ordinary_done.commit_token(3));
        assert!(optimized_done.commit_token(3));
        assert!(ordinary_done.is_accepting());
        assert!(optimized_done.is_accepting());
        assert_eq!(ordinary_done.mask(), optimized_done.mask());

        // The production singleton path uses the flat alias representation and
        // deliberately defers its first transfer-artifact serialization. Verify
        // that both survive the real external-vocab save/load boundary.
        let saved = optimized.save_without_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&saved, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), optimized.start().mask());
        let mut loaded_done = loaded.start();
        assert!(loaded_done.commit_token(3));
        assert!(loaded_done.is_accepting());
    }

    #[test]
    fn singleton_literal_partition_matches_all_reachable_cross_terminal_prefixes() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (2, b"b".to_vec()),
            (5, b"c".to_vec()),
            (9, b"d".to_vec()),
            (12, b"ab".to_vec()),
            (18, b"bc".to_vec()),
            (24, b"cd".to_vec()),
            (31, b"abc".to_vec()),
            (37, b"bcd".to_vec()),
            (44, b"abcd".to_vec()),
            (51, b"x".to_vec()),
            (58, b"bd".to_vec()),
            (63, b"abcx".to_vec()),
            (71, b"bc".to_vec()),
        ]);
        let token_ids = vocab.iter().map(|(token_id, _)| token_id).collect::<Vec<_>>();
        let grammar_source = r#"start ::= "ab" "cd""#.to_owned();
        let grammar = Grammar::ebnf(&grammar_source);

        // Compile ordinary first so the optimized compile also exercises the
        // cached byte-order lookup used by prepared production vocabularies.
        let ordinary = DynamicConstraint::compile(grammar.clone(), &vocab).unwrap();
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(grammar, &vocab).unwrap();

        let mut pending = vec![Vec::<u32>::new()];
        let mut seen = std::collections::BTreeSet::<Vec<u32>>::new();
        seen.insert(Vec::new());
        while let Some(prefix) = pending.pop() {
            let mut ordinary_state = ordinary.start();
            let mut optimized_state = optimized.start();
            for &token in &prefix {
                assert!(ordinary_state.commit_token(token));
                assert!(optimized_state.commit_token(token));
            }

            assert_eq!(
                ordinary_state.is_accepting(),
                optimized_state.is_accepting(),
                "acceptance differs after token prefix {prefix:?}",
            );
            let ordinary_mask = ordinary_state.mask();
            let optimized_mask = optimized_state.mask();
            assert_eq!(
                ordinary_mask, optimized_mask,
                "mask differs after token prefix {prefix:?}",
            );

            for &token in &token_ids {
                let ordinary_allowed = token_allowed(&ordinary_mask, token);
                let optimized_allowed = token_allowed(&optimized_mask, token);
                assert_eq!(
                    ordinary_allowed, optimized_allowed,
                    "token {token} differs after token prefix {prefix:?}",
                );
                if ordinary_allowed {
                    let mut next = prefix.clone();
                    next.push(token);
                    // Four one-byte tokens can consume the complete language;
                    // no valid path can require a deeper token prefix.
                    if next.len() <= 4 && seen.insert(next.clone()) {
                        pending.push(next);
                    }
                }
            }
        }
    }

    #[test]
    fn partition_optimized_dynamic_save_load_preserves_quotient_mask() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"x".to_vec()),
        ]);
        let grammar_source = format!(
            "start ::= [ab]+{}",
            (0..17)
                .map(|index| format!(r#" | "unused_{index}""#))
                .collect::<String>()
        );
        let optimized =
            DynamicConstraint::compile_with_vocab_partition(Grammar::ebnf(&grammar_source), &vocab)
                .unwrap();
        let original_canonical = optimized
            .inner
            .dynamic_mask_vocab_for_runtime()
            .canonical_token_count();
        assert!(original_canonical < vocab.len());
        let bytes = optimized.save_without_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert_eq!(loaded.start().mask(), optimized.start().mask());
        assert_eq!(
            loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .canonical_token_count(),
            original_canonical,
            "save/load must retain the representative vocabulary rather than rebuilding the full vocabulary",
        );
    }

    #[test]
    fn partition_optimized_bounded_string_schema_does_not_fall_back_to_ordinary_dynamic() {
        let vocab = Vocab::new(vec![
            (0, b"\"abcdefgh\"".to_vec()),
            (1, b"\"ijklmnop\"".to_vec()),
            (2, b"true".to_vec()),
            (3, b"false".to_vec()),
            (4, b"{".to_vec()),
            (5, b"}".to_vec()),
            (6, b":".to_vec()),
            (7, b",".to_vec()),
            (8, b" ".to_vec()),
        ]);
        let mut properties = serde_json::Map::new();
        properties.insert(
            "bounded".to_owned(),
            serde_json::json!({"type":"string", "minLength":8, "maxLength":8}),
        );
        for index in 0..20 {
            properties.insert(format!("flag_{index}"), serde_json::json!({"type":"boolean"}));
        }
        let schema = serde_json::json!({
            "type":"object",
            "properties": properties,
            "additionalProperties": false
        })
        .to_string();

        let optimized = DynamicConstraint::compile_with_vocab_partition(
            Grammar::json_schema(&schema),
            &vocab,
        )
        .unwrap();
        assert!(
            optimized
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "O2 must retain its grammar quotient for bounded-string residual schemas",
        );

        let bytes = optimized.save_without_vocab();
        let loaded = DynamicConstraint::load_with_vocab(&bytes, &vocab).unwrap();
        assert!(
            loaded
                .inner
                .dynamic_mask_vocab_for_runtime()
                .is_grammar_quotiented(),
            "external-vocab transfer must preserve the O2 grammar quotient",
        );
        assert_eq!(loaded.start().mask(), optimized.start().mask());
    }

    #[test]
    fn partition_optimized_plain_bounded_string_builds_a_real_quotient() {
        let vocab = Vocab::new(vec![
            (0, b"\"".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"aa".to_vec()),
            (4, b"bb".to_vec()),
            (5, b"ab".to_vec()),
            (6, b"ba".to_vec()),
            (7, b"\"a".to_vec()),
            (8, b"\"b".to_vec()),
            (9, b"x".to_vec()),
        ]);
        let schema = r#"{"type":"string","maxLength":100}"#;

        let ordinary =
            DynamicConstraint::compile(Grammar::json_schema(schema), &vocab).unwrap();
        let optimized = DynamicConstraint::compile_with_vocab_partition(
            Grammar::json_schema(schema),
            &vocab,
        )
        .unwrap();

        let optimized_vocab = optimized.inner.dynamic_mask_vocab_for_runtime();
        assert!(optimized_vocab.is_grammar_quotiented());
        assert!(
            optimized_vocab.canonical_token_count() < vocab.len(),
            "bounded-string O2 must not degrade to the identity vocabulary quotient",
        );
        assert_eq!(ordinary.start().mask(), optimized.start().mask());

        let mut ordinary_state = ordinary.start();
        let mut optimized_state = optimized.start();
        assert!(ordinary_state.commit_token(0));
        assert!(optimized_state.commit_token(0));
        assert_eq!(ordinary_state.mask(), optimized_state.mask());
    }

    #[test]
    fn partition_optimized_multi_alternative_roundtrips_without_sharing_one_quotient() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"aa".to_vec()),
            (3, b"bb".to_vec()),
            (4, b"ab".to_vec()),
            (5, b"ba".to_vec()),
            (6, b"x".to_vec()),
        ]);
        let first = DynamicConstraint::compile_with_vocab_partition(
            Grammar::ebnf(r#"start ::= [ab]+"#),
            &vocab,
        )
        .unwrap();
        let second = DynamicConstraint::compile_with_vocab_partition(
            Grammar::ebnf(r#"start ::= "a"+"#),
            &vocab,
        )
        .unwrap();
        let optimized = DynamicConstraint::from_alternatives(vec![first, second]);
        assert_eq!(optimized.clone_constraints().len(), 2);
        let expected_mask = optimized.start().mask();

        // The self-contained wire has one shared vocab accelerator, so this
        // case intentionally drops the per-alternative quotients and rebuilds
        // the full vocab lazily after load rather than applying one quotient to
        // every alternative.
        let loaded = DynamicConstraint::load(&optimized.save(), &vocab).unwrap();
        assert_eq!(loaded.start().mask(), expected_mask);

        // The external-vocab transfer has per-alternative metadata and can keep
        // each quotient independently.
        let transferred = DynamicConstraint::load_with_vocab(
            &optimized.save_without_vocab(),
            &vocab,
        )
        .unwrap();
        assert_eq!(transferred.start().mask(), expected_mask);
    }

    #[test]
    fn partition_optimized_component_keeps_boundary_token_distinctions() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ax".to_vec()),
            (3, b"ay".to_vec()),
            (4, b"bx".to_vec()),
            (5, b"by".to_vec()),
            (6, b"x".to_vec()),
            (7, b"y".to_vec()),
        ]);
        let child_grammar = Grammar::ebnf(r#"start ::= [ab]"#);
        let child_partition = VocabPartition::compile(child_grammar.clone(), &vocab).unwrap();
        assert_eq!(child_partition.class_of(2), child_partition.class_of(3));
        assert_eq!(child_partition.class_of(4), child_partition.class_of(5));

        let ordinary_child = native_dynamic_fixture(child_grammar.clone(), &vocab).unwrap();
        let optimized_child =
            DynamicConstraint::from_ebnf_with_bounded_template_parser(
                r#"start ::= [ab]"#, &vocab,
            ).unwrap();
        assert_eq!(optimized_child.inner.token_bytes_count(), vocab.len());

        let parent = Grammar::glrm(
            "glrm 1; start start; extern grammar child; nt start = child \"x\";",
        );
        let ordinary = ConstraintSpec::builder(parent.clone(), &vocab)
            .unwrap()
            .bind_grammar("child", &ordinary_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_native_dynamic_fixture()
            .unwrap();
        let optimized = ConstraintSpec::builder(parent, &vocab)
            .unwrap()
            .bind_grammar("child", &optimized_child)
            .unwrap()
            .build()
            .unwrap()
            .compile_native_dynamic_fixture()
            .unwrap();

        let ordinary_mask = ordinary.start().mask();
        let optimized_mask = optimized.start().mask();
        assert_eq!(ordinary_mask, optimized_mask);
        assert!(token_allowed(&optimized_mask, 2), "ax must cross child -> parent boundary");
        assert!(!token_allowed(&optimized_mask, 3), "ay must remain rejected by the parent boundary");
        assert!(token_allowed(&optimized_mask, 4), "bx must cross child -> parent boundary");
        assert!(!token_allowed(&optimized_mask, 5), "by must remain rejected by the parent boundary");
    }

    #[test]
    fn first_save_priming_threshold_tracks_large_composition_cache() {
        let vocab = Vocab::new(vec![(0, b"a".to_vec())]);
        let mut constraint = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1;\nstart start;\nt A = \"a\";\nnt start = A;\n"),
            &vocab,
        )
        .unwrap();
        assert!(!should_prime_first_save_artifact(&constraint));

        let mut large = crate::automata::unweighted_u32::dfa::DFA::new();
        large
            .states
            .resize_with(FIRST_SAVE_TEMPLATE_STATE_PRIME_THRESHOLD, Default::default);
        constraint.composition_parser_templates_by_terminal = vec![Some(large)];
        assert!(should_prime_first_save_artifact(&constraint));
    }

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
        assert!(constraint.has_template_parser());
        assert!(!constraint.table.is_present(), "native parser retained an LR table");
        if !constraint.uses_compact_segmented_parser_runtime() {
            return;
        }
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(
            constraint.template_parser.as_ref().unwrap().link_grammar.is_some(),
            "recursive coordinator lost its pure composition grammar metadata",
        );
        for component in &overlay.segmented_parser_components {
            assert_recursive_compiler_views_detached(&component.constraint);
        }
    }

    fn assert_no_materialized_outer_table(constraint: &mut RuntimeConstraint) {
        constraint.recursive_parser_layout().unwrap().unwrap();
        // Native constraints have no table to poison, including at leaves.
        // Keep exercising the cloned/loaded instance after proving absence.
        assert_recursive_compiler_views_detached(constraint);
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
    fn mixed_native_lr_component_composition_returns_compilation_error() {
        let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec()), (2, b"xy".to_vec())]);
        let lr = |source: &str| crate::import::compile_dynamic_glrm_fixture(
            source, &vocab, crate::runtime::parser_backend::DynamicParserBackend::LrTable,
        ).unwrap();
        let source = "start start; extern grammar child; nt start ::= 'x' child;";
        let parent = RuntimeConstraint::compile(Grammar::glrm(source), &vocab).unwrap();
        let child = lr("start start; nt start ::= 'y';");
        assert!(child.inner.table.is_present());
        assert!(!child.inner.has_template_parser());
        let spec = ConstraintSpec::builder(Grammar::glrm(source), &vocab).unwrap()
            .bind_grammar("child", &child).unwrap().build().unwrap();
        for error in [spec.compile().unwrap_err(), spec.compile_dynamic().unwrap_err(),
            parent.bind_grammar("child", &child).unwrap_err(),
            parent.bind_grammar_dynamic_boundary("child", &child).unwrap_err()] {
            assert!(matches!(error.kind(), crate::ErrorKind::Compilation(_)));
            let message=error.to_string();
            assert!(message.contains("a composition can't mix parts precompiled with the LrTable and TemplateDfa parser backends"), "{message}");
        }
        let lr_parent = lr(source);
        let native_child = RuntimeConstraint::compile(Grammar::ebnf("start ::= 'y'"), &vocab).unwrap();
        for error in [lr_parent.bind_grammar("child", &native_child).unwrap_err(),
            lr_parent.bind_grammar_dynamic_boundary("child", &native_child).unwrap_err()] {
            assert!(matches!(error.kind(), crate::ErrorKind::Compilation(_)));
            let message=error.to_string();
            assert!(message.contains("a composition can't mix parts precompiled with the LrTable and TemplateDfa parser backends"), "{message}");
        }
        // Failed linking must leave the retained LR object usable.
        assert_eq!(child.start().mask()[0] & 2, 2);
        assert!(child.inner.table.is_present());
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
            native_dynamic_fixture(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

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
            .compile_native_dynamic_fixture()
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
            .compile_native_dynamic_fixture()
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
        let dynamic_parent = native_dynamic_fixture(one_slot, &vocab).unwrap();
        let static_child = RuntimeConstraint::compile(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();
        let dynamic_child =
            native_dynamic_fixture(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

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
                native_dynamic_fixture(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_static_boundary(&fully_bound);

        let fully_bound = partially_bound
            .bind_grammar_dynamic_boundary(
                "right",
                native_dynamic_fixture(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap(),
            )
            .unwrap();
        assert!(fully_bound.late_grammar_slots.is_empty());
        assert_eq!(leaf_backend_flags(&fully_bound), vec![false, false, true]);
        assert_dynamic_boundary(&fully_bound);

        let open = native_dynamic_fixture(
            Grammar::glrm(
                "glrm 1; start start; extern grammar left; extern grammar right; \
                 nt start = \"x\" left right;",
            ),
            &vocab,
        )
        .unwrap();
        let partially_bound = open.bind_grammar("left", &static_child).unwrap();
        let dynamic_right =
            native_dynamic_fixture(Grammar::ebnf(r#"start ::= "z""#), &vocab).unwrap();
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

            let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
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
            assert!(state.commit_token(2));
            assert!(state.is_accepting());

            let mut live_state = bound.start();
            let mut loaded_state = loaded.start();
            for token in [0, 1] {
                assert_eq!(
                    loaded_state.mask(),
                    live_state.mask(),
                    "all-static segmented state changed after round-trip before token {token} (dynamic_boundary={dynamic_boundary})",
                );
                assert!(live_state.commit_token(token));
                assert!(loaded_state.commit_token(token));
            }
            assert_eq!(loaded_state.is_accepting(), live_state.is_accepting());
            assert!(loaded_state.is_accepting());
        }
    }

    #[test]
    fn composed_dynamic_constraint_transfer_roundtrip_preserves_recursive_runtime() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
            (3, b"xyz".to_vec()),
        ]);
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "x" "y""#),
            &vocab,
        )
        .unwrap();
        let parent = native_dynamic_fixture(
            Grammar::glrm(
                "glrm 1; start start; extern grammar child; nt start = \"x\" child;",
            ),
            &vocab,
        )
        .unwrap();
        let child = native_dynamic_fixture(Grammar::ebnf(r#"start ::= "y""#), &vocab)
            .unwrap();

        for dynamic_boundary in [false, true] {
            let bound = if dynamic_boundary {
                parent
                    .bind_grammar_dynamic_boundary("child", &child)
                    .unwrap()
            } else {
                parent.bind_grammar("child", &child).unwrap()
            };
            assert_eq!(bound.start().mask(), reference.start().mask());

            let transfer = bound.clone().into_saved();
            let loaded = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
            assert_eq!(
                loaded.start().mask(),
                bound.start().mask(),
                "composed dynamic transfer changed initial mask (dynamic_boundary={dynamic_boundary})",
            );
            let mut state = loaded.start();
            assert!(state.commit_token(2));
            assert!(state.is_accepting());
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

        let loaded_parent = RuntimeConstraint::load_body_external_artifact(parent.save(), &vocab).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save(), &vocab).unwrap();
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

        let loaded_parent = RuntimeConstraint::load_body_external_artifact(parent.save(), &vocab).unwrap();
        let loaded_child = RuntimeConstraint::load(child.save(), &vocab).unwrap();
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
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
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
        assert!(
            inner.parser_has_controls(),
            "fixture must exercise Exact construction over explicit native linker controls",
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
            .bind_grammar_dynamic_boundary("inner", RuntimeConstraint::load(inner.save(), &vocab).unwrap())
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
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
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

        let mut loaded = RuntimeConstraint::load_body_external_artifact(parent.save(), &vocab).unwrap();
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

        let reloaded = RuntimeConstraint::load_body_external_artifact(loaded.save(), &vocab).unwrap();
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
        let child = child_spec.compile_native_dynamic_fixture().unwrap();
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
            .bind_grammar("child", &child)
            .unwrap()
            .build()
            .unwrap()
            .compile_native_dynamic_fixture()
            .unwrap();
        let alternatives = composed.clone_constraints();
        assert_eq!(alternatives.len(), 1);
        let composed = &alternatives[0];
        let reference = RuntimeConstraint::compile(
            Grammar::ebnf(r#"start ::= "a" "b" "c" "!" | "a" "b" "c" "?""#),
            &vocab,
        )
        .unwrap();

        let explicitly_disabled = crate::env::var("GLRMASK_COMPOSE_RUNTIME_LEXER_PRODUCT")
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
            assert!(actual.commit_token(token));
            assert!(expected.commit_token(token));
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

        let dynamic_constraint = spec.compile_native_dynamic_fixture().unwrap();
        assert!(dynamic_constraint.clone_constraints().iter().all(|constraint| {
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let loaded_dynamic = DynamicConstraint::load(&dynamic_constraint.save(), &vocab).unwrap();
        assert!(loaded_dynamic.clone_constraints().iter_mut().all(|constraint| {
            constraint.materialize_composition_link_metadata_for_compilation().unwrap();
            matches!(constraint.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_))
        }));

        let transfer = dynamic_constraint.clone().into_saved();
        let loaded_transfer = DynamicConstraint::load_with_vocab(&transfer, &vocab).unwrap();
        assert!(loaded_transfer.clone_constraints().iter_mut().all(|constraint| {
            constraint.materialize_composition_link_metadata_for_compilation().unwrap();
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

        let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
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
        assert_no_materialized_outer_table(&mut no_outer_table);
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
                    assert!(actual.commit_token(token));
                    assert!(expected.commit_token(token));
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
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
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

        for constraint in [&bound, &RuntimeConstraint::load(bound.save(), &vocab).unwrap()] {
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
                    assert!(actual.commit_token(token));
                    assert!(expected.commit_token(token));
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

        let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
        let mut no_outer_table = loaded.clone();
        assert_no_materialized_outer_table(&mut no_outer_table);
        for constraint in [&bound, &loaded, &no_outer_table] {
            let mut state = constraint.start();
            assert!(state.commit_token(0));
            let mask = state.mask();
            assert_ne!(mask[100 / 32] & (1u32 << (100 % 32)), 0);

            assert!(state.commit_token(100));
            let mask = state.mask();
            assert_ne!(mask[1 / 32] & (1u32 << 1), 0);

            assert!(state.commit_token(1));
            assert!(state.is_accepting());
        }
    }

    #[test]
    fn recursive_static_virtual_residual_child_handles_fused_boundaries_and_roundtrip() {
        let vocab = Vocab::new(vec![
            (0, b"X".to_vec()),
            (1, b"!".to_vec()),
            (2, b"\"".to_vec()),
            (3, b"x:".to_vec()),
            (4, b"a".to_vec()),
            (5, b"X\"".to_vec()),
            (6, b"\"!".to_vec()),
            (7, b"xa".to_vec()),
            (8, b":".to_vec()),
            (9, b"X\"x:".to_vec()),
            (10, b"a\"!".to_vec()),
            (11, b"aa".to_vec()),
            (12, b"a\"".to_vec()),
        ]);
        let schema = r#"{"type":"string","format":"uri","pattern":"^[a-z][a-z0-9+.-]*:[a-z]*$","minLength":1,"maxLength":5000}"#;
        let child = RuntimeConstraint::from_json_schema(
            schema,
            &vocab,
        )
        .unwrap();
        assert!(child.tokenizer.has_virtual_residual_runtime());

        let parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start document; extern grammar payload; nt document = \"X\" payload \"!\";",
            ),
            &vocab,
        )
        .unwrap();
        let static_bound = parent.bind_grammar("payload", &child).unwrap();
        assert!(static_bound.static_dynamic_overlay.as_ref().unwrap()
            .segmented_boundary_shards.iter().all(|shard| !matches!(
                shard.backend, crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect)));
        let static_loaded = RuntimeConstraint::load(static_bound.save(), &vocab).unwrap();
        let bound = parent
            .bind_grammar_dynamic_boundary("payload", child)
            .unwrap();
        let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
        // Keep the schema's exact internal regular nodes; source serialization
        // is not the composition protocol and may intentionally have a narrower
        // syntax than the JSON frontend's already constructed representation.
        let child_named = crate::import::parse_json_schema_to_named(schema).unwrap();
        let named = crate::grammar::glrm::from_glrm_with_named_subgrammars_and_external_terminals(
            r#"start document extern grammar payload nt document = "X" payload "!""#,
            &[("payload", child_named)], &[],
        ).unwrap();
        let inline = crate::import::compile_from_named_grammar(named, &vocab,
            "virtual_uri_inline_reference", crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,
            &[]).unwrap();

        let token_allowed = |mask: &[u32], token: u32| {
            mask[token as usize / 32] & (1u32 << (token % 32)) != 0
        };
        for constraint in [&static_bound, &static_loaded, &bound, &loaded] {
            let mut state = constraint.start();
            let mut reference = inline.start();
            assert_eq!(state.mask(), reference.mask());
            assert!(token_allowed(&state.mask(), 9));
            assert!(state.commit_token(9));
            assert!(reference.commit_token(9));
            assert_eq!(state.mask(), reference.mask());
            assert!(token_allowed(&state.mask(), 10));
            assert!(state.commit_token(10));
            assert!(reference.commit_token(10));
            assert_eq!(state.mask(), reference.mask());
            assert_eq!(state.is_accepting(), reference.is_accepting());
            assert!(state.is_accepting());

            let mut exact = constraint.start();
            let mut exact_bytes = b"X\"x:".to_vec();
            exact_bytes.extend(std::iter::repeat_n(b'a', 4998));
            exact_bytes.extend_from_slice(b"\"!");
            assert!(exact.commit_bytes(&exact_bytes));
            assert!(exact.is_accepting());

            let mut over = constraint.start();
            let mut over_bytes = b"X\"x:".to_vec();
            over_bytes.extend(std::iter::repeat_n(b'a', 4999));
            over_bytes.extend_from_slice(b"\"!");
            assert!(!over.commit_bytes(&over_bytes));
        }
    }

    #[test]
    fn nested_boundaries_use_recursive_parser_coordinate_live_and_loaded() {
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
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();
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
            shard.start_parser_states.is_empty()
        }));
        assert!(bound_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.is_empty())
        }));
        let layout = bound.recursive_parser_layout().unwrap().unwrap();
        assert_eq!(layout.leaves.len(), 3);

        let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
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
            shard.start_parser_states.is_empty()
        }));
        assert!(loaded_overlay.segmented_parser_components.iter().all(|component| {
            component
                .boundary
                .as_ref()
                .is_none_or(|shard| shard.start_parser_states.is_empty())
        }));
        assert_eq!(loaded.recursive_parser_layout().unwrap().unwrap().leaves.len(), 3);

        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"[\" \"a\" \"]\" \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        assert_no_materialized_outer_table(&mut no_outer_table);
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
                    assert!(actual.commit_token(token));
                    assert!(expected.commit_token(token));
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
                    assert!(actual.commit_token(token));
                    assert!(expected.commit_token(token));
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
        assert!(middle_parent.composition_start_nullable().unwrap());
        let middle = middle_parent.bind_grammar("leaf", leaf).unwrap();
        assert!(middle.composition_start_nullable().unwrap());
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
        let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
        assert!(loaded.uses_compact_segmented_parser_runtime());
        let monolithic = RuntimeConstraint::compile(
            Grammar::glrm("glrm 1; start document; nt document = \"X\" \"a\"? \"!\";"),
            &vocab,
        )
        .unwrap();
        let mut no_outer_table = loaded.clone();
        assert_no_materialized_outer_table(&mut no_outer_table);

        for constraint in [&bound, &loaded, &no_outer_table] {
            for tokens in [&[3][..], &[4][..], &[0, 2][..], &[0, 5][..], &[0, 1, 2][..]] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                for &token in tokens {
                    assert_eq!(actual.mask(), expected.mask(), "before {tokens:?} token {token}");
                    assert!(actual.commit_token(token));
                    assert!(expected.commit_token(token));
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
        let leaf_states = leaf.parser_symbol_count();
        let leaf_tokenizer_states = leaf.tokenizer.num_states();
        let leaf_tokenizer_reset = leaf.runtime_commit_initial_state();
        let middle_parent = RuntimeConstraint::compile(
            Grammar::glrm(
                "glrm 1; start middle; extern grammar leaf; nt middle = \"[\" leaf \"]\";",
            ),
            &vocab,
        )
        .unwrap();
        let middle_parent_states = middle_parent.parser_symbol_count();
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
        let outer_parent_states = outer_parent.parser_symbol_count();
        let outer_parent_tokenizer_states = outer_parent.tokenizer.num_states();
        let outer_parent_tokenizer_reset = outer_parent.runtime_commit_initial_state();
        let bound = outer_parent
            .bind_grammar_dynamic_boundary("middle", middle)
            .unwrap();

        for constraint in [&bound, &RuntimeConstraint::load(bound.save(), &vocab).unwrap()] {
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

        for constraint in [&bound, &RuntimeConstraint::load(bound.save(), &vocab).unwrap()] {
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

        for constraint in [&full, &RuntimeConstraint::load(full.save(), &vocab).unwrap()] {
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
        assert_recursive_compiler_views_detached(&half);
        assert!(half_overlay
            .segmented_parser_components
            .iter()
            .all(|component| component.global_to_local_parser_state.is_empty()));

        let loaded_half = RuntimeConstraint::load_body_external_artifact(half.save(), &vocab).unwrap();
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
        assert_recursive_compiler_views_detached(&loaded_half);
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
        for terminal in 0..fresh_full.parser_terminal_count() {
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
                assert!(split.commit_token(token));
            }
            assert!(split.is_accepting());

            let mut fused = constraint.start();
            assert_ne!(fused.mask()[0] & (1 << 4), 0);
            assert!(fused.commit_token(4));
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
        assert!(!middle_parent.composition_start_nullable().unwrap());
        let middle = middle_parent
            .bind_grammar_dynamic_boundary("leaf", &leaf)
            .unwrap();
        assert!(middle.composition_start_nullable().unwrap());
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

        for constraint in [&bound, &RuntimeConstraint::load(bound.save(), &vocab).unwrap()] {
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

        let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
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
                assert!(sharded.commit_token(token));
                assert!(restored.commit_token(token));
                assert!(expected.commit_token(token));
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
        let mut cached_parent = RuntimeConstraint::load_body_external_artifact(parent.save(), &vocab).unwrap();
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

        let loaded = RuntimeConstraint::load(bound.save(), &vocab).unwrap();
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
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
                assert_eq!(actual.mask(), expected.mask(), "after token {token}");
            }
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());

            for token in [5, 6, 7, 8] {
                let mut actual = constraint.start();
                let mut expected = monolithic.start();
                assert_eq!(
                    actual.commit_token(token),
                    expected.commit_token(token),
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
        for constraint in [&bound, &RuntimeConstraint::load(bound.save(), &vocab).unwrap()] {
            let mut actual = constraint.start();
            let mut expected = monolithic.start();
            for token in [0, 1] {
                assert_eq!(actual.mask(), expected.mask(), "before token {token}");
                assert!(actual.commit_token(token));
                assert!(expected.commit_token(token));
            }
            let actual_mask = actual.mask();
            let expected_mask = expected.mask();
            assert_eq!(actual_mask, expected_mask, "after child completion");
            assert_ne!(actual_mask[0] & (1 << 4), 0, "fused parent token !? must be allowed");
            assert!(actual.commit_token(4));
            assert!(expected.commit_token(4));
            assert!(actual.is_accepting());
            assert!(expected.is_accepting());
        }
    }

    #[test]
    fn dynamic_late_bind_vocab_cache_is_shared_reused_and_not_serialized() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()), (1, b"y".to_vec()), (2, b"xy".to_vec()),
        ]);
        let parent = native_dynamic_fixture(
            Grammar::glrm("glrm 1; start start; extern grammar child; nt start = \"x\" child;"),
            &vocab,
        ).unwrap();
        let child = native_dynamic_fixture(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();
        let retained = parent.inner.late_bind_vocab.get().expect("retain supplied vocabulary");
        assert!(Arc::ptr_eq(&retained.entries_arc(), &vocab.entries_arc()));

        let loaded = DynamicConstraint::from_constraints(vec![
            RuntimeConstraint::load_body_external_artifact(parent.inner.save(), &vocab).unwrap(),
        ]);
        assert!(loaded.inner.late_bind_vocab.get().is_some());
        let first = loaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        let first_backing = loaded.inner.late_bind_vocab.get()
            .expect("prime retained loaded parent, not a throwaway clone").entries_arc();
        let second = loaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        assert!(Arc::ptr_eq(
            &first_backing, &loaded.inner.late_bind_vocab.get().unwrap().entries_arc(),
        ));
        assert_eq!(first.start().mask(), second.start().mask());
        let reloaded = DynamicConstraint::from_constraints(vec![
            RuntimeConstraint::load_body_external_artifact(loaded.inner.save(), &vocab).unwrap(),
        ]);
        assert!(reloaded.inner.late_bind_vocab.get().is_some(), "loaded vocab is runtime-only, never serialized");
        let rebound = reloaded.bind_grammar_dynamic_boundary("child", &child).unwrap();
        assert_eq!(first.start().mask(), rebound.start().mask());
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
            native_dynamic_fixture(Grammar::ebnf(r#"start ::= "y""#), &vocab).unwrap();

        assert!(
            parent.late_bind_vocab.get().is_some(),
            "fresh compilation should retain the supplied vocabulary for later binds",
        );

        let loaded_parent = RuntimeConstraint::load_body_external_artifact(parent.save(), &vocab).unwrap();
        assert!(
            loaded_parent.late_bind_vocab.get().is_some(),
            "the loader supplies the vocabulary; it is runtime-only and never serialized",
        );
        let first = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert!(loaded_parent.late_bind_vocab.get().is_some());
        let second = loaded_parent
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), second.start().mask());

        let loaded = RuntimeConstraint::load_body_external_artifact(loaded_parent.save(), &vocab).unwrap();
        assert!(
            loaded.late_bind_vocab.get().is_some(),
            "the loader supplies the vocabulary; it is runtime-only and never serialized",
        );
        let rebound = loaded
            .bind_grammar_dynamic_boundary("child", &child)
            .unwrap();
        assert_eq!(first.start().mask(), rebound.start().mask());
    }
}


#[cfg(test)]
mod final_unlinked_optimization_tests {
    use super::*;

    fn assert_static_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(!overlay.segmented_boundary_shards.is_empty());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::StaticParser(_)
        )));
    }

    fn assert_dynamic_boundary(constraint: &RuntimeConstraint) {
        let overlay = constraint.static_dynamic_overlay.as_ref().unwrap();
        assert!(!overlay.segmented_boundary_shards.is_empty());
        assert!(overlay.segmented_boundary_shards.iter().all(|shard| matches!(
            shard.backend,
            crate::runtime::SegmentedBoundaryShardBackend::DynamicDirect
        )));
    }

    #[test]
    fn unlinked_bind_defers_composition_until_link_and_link_optimization_selects_boundary() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        let host = Grammar::glrm(
            r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
        )
        .prepare(&vocab)
        .unwrap();
        let child = Grammar::ebnf(r#"start ::= "y""#)
            .compile(&vocab)
            .unwrap();

        let bound = host.bind_snapshot("child", &child).unwrap();
        assert!(bound.inner.late_grammar_slots.iter().any(|slot| slot.name == "child"));
        assert!(bound.inner.static_dynamic_overlay.is_none(), "bind must not compile a boundary eagerly");

        let fast_build = bound.link_with(
            MaskMode::C.build_options(),
        ).unwrap();
        let fast_runtime = bound.link_with(
            BuildOptions::default().optimization(MaskStrategy::ParserDwa),
        ).unwrap();
        assert_dynamic_boundary(&fast_build);
        assert_static_boundary(&fast_runtime);
        assert_eq!(fast_build.start().mask(), fast_runtime.start().mask());

        let loaded = PreparedGrammar::load(bound.save(), &vocab).unwrap();
        assert!(loaded.inner.late_grammar_slots.iter().any(|slot| slot.name == "child"));
        assert_dynamic_boundary(&loaded.link_with(
            MaskMode::C.build_options(),
        ).unwrap());
        assert_static_boundary(&loaded.link_with(
            BuildOptions::default().optimization(MaskStrategy::ParserDwa),
        ).unwrap());
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
        state.commit_bytes(bytes) && state.is_accepting()
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
        let loaded = RuntimeConstraint::load_body_external_artifact(&saved, &vocab).unwrap();
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
        let half = parent.bind_grammar_dynamic_boundary("left", &a).unwrap();
        assert!(half
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "right"));
        let full = half.bind_grammar_dynamic_boundary("right", &b).unwrap();
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
        let open_left = parent
            .bind_grammar_dynamic_boundary("left", &unresolved_child)
            .unwrap();
        assert!(open_left
            .late_grammar_slots
            .iter()
            .any(|slot| slot.name == "left.leaf"));
        let left = open_left.bind_grammar_dynamic_boundary("left.leaf", &a).unwrap();
        let nested_full = left.bind_grammar_dynamic_boundary("right", &b).unwrap();
        assert!(nested_full.late_grammar_slots.is_empty());
        assert!(accepts(&nested_full, b"<ab>"));
    }

    #[test]
    fn dynamic_compile_preserves_unbound_external_grammar() {
        let vocab = vocab();
        let open = native_dynamic_fixture(
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
        assert!(state.commit_bytes(b"a"));
        assert!(state.is_accepting());
    }

    #[test]
    fn final_optimization_dispatch_selects_dynamic_build_and_static_runtime_paths() {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
        ]);
        let grammar = Grammar::ebnf(r#"start ::= "a" "b""#);
        let fast_build = grammar
            .compile_with_build_options(
                &vocab,
                MaskMode::C.build_options(),
            )
            .unwrap();
        let fast_runtime = grammar
            .compile_with_build_options(
                &vocab,
                BuildOptions::default().optimization(MaskStrategy::ParserDwa),
            )
            .unwrap();
        assert!(fast_build.uses_dynamic_runtime());
        assert!(!fast_runtime.uses_dynamic_runtime());

        let mut build_state = fast_build.start();
        let mut runtime_state = fast_runtime.start();
        assert_eq!(build_state.mask(), runtime_state.mask());
        assert!(build_state.commit_token(2));
        assert!(runtime_state.commit_token(2));
        assert_eq!(build_state.is_accepting(), runtime_state.is_accepting());

        let loaded = RuntimeConstraint::load(fast_build.save(), &vocab).unwrap();
        assert!(loaded.uses_dynamic_runtime());
        assert_eq!(loaded.start().mask(), fast_runtime.start().mask());
    }
}

impl RuntimeConstraint {
    /// Inspect the actual stored parser backend, not a process-wide switch.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn parser_backend(&self) -> ParserBackend {
        if self.has_template_parser() {
            assert!(!self.table.is_present(), "template parser retained LR storage");
            ParserBackend::TemplateDfa
        } else { ParserBackend::LrTable }
    }

    /// Diagnostic: whether this body retains an LR table.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn runtime_has_lr_table(&self) -> bool {
        self.table.is_present()
    }

    /// Diagnostic: whether this body retains a template parser.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn runtime_has_template_parser(&self) -> bool {
        self.has_template_parser()
    }

    /// Diagnostic: whether the mask vocabulary is grammar-quotiented.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn runtime_is_grammar_quotiented(&self) -> bool {
        self.dynamic_mask_vocab.is_grammar_quotiented()
    }
}


#[cfg(test)]
mod final_boundary_option_tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec()), (2, b"xy".to_vec())])
    }

    #[test]
    fn trigger_metadata_exists_before_first_use_in_every_public_mode() {
        let v = vocab();
        let grammar = Grammar::from_ebnf(r#"start ::= "x" "y""#).unwrap();
        for mode in MaskMode::ALL {
            for detail in [BoundaryTriggerDetail::None, BoundaryTriggerDetail::TokenSet, BoundaryTriggerDetail::Exact] {
                let result = grammar.compile_with_build_options(&v,
                    mode.build_options().boundary_trigger(detail));
                if (mode == MaskMode::A || mode == MaskMode::D) && detail == BoundaryTriggerDetail::Exact {
                    assert!(result.unwrap_err().to_string().contains("retained-LR Dynamic"));
                    continue;
                }
                let constraint = result.unwrap();
                assert!(matches!((&constraint.boundary_trigger, detail),
                    (crate::runtime::BoundaryTrigger::None, BoundaryTriggerDetail::None) |
                    (crate::runtime::BoundaryTrigger::TokenSet(_), BoundaryTriggerDetail::TokenSet) |
                    (crate::runtime::BoundaryTrigger::Exact(_), BoundaryTriggerDetail::Exact)));
            }
        }
    }

    #[test]
    fn deferred_trigger_metadata_is_built_at_link_and_explicit_none_overrides() {
        let v = vocab();
        let parent = Grammar::from_glrm(r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#).unwrap()
            .prepare_with(&v, BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::Exact)).unwrap();
        assert!(matches!(parent.inner.boundary_trigger, crate::runtime::BoundaryTrigger::None));
        let parent = PreparedGrammar::load(parent.save(), &v).unwrap();
        let child = Grammar::from_ebnf(r#"start ::= "y""#).unwrap().prepare(&v).unwrap();
        let bound = parent.bind_snapshot("child", &child).unwrap();
        assert!(matches!(bound.inner.boundary_trigger, crate::runtime::BoundaryTrigger::None));
        let exact = bound.link().unwrap();
        assert!(matches!(exact.boundary_trigger, crate::runtime::BoundaryTrigger::Exact(_)));
        let none = bound.link_with(BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::None)).unwrap();
        assert!(matches!(none.boundary_trigger, crate::runtime::BoundaryTrigger::None));
        assert_eq!(exact.start().mask(), none.start().mask());
    }

    #[test]
    fn persisted_child_request_is_scoped_to_the_child() {
        let v = vocab();
        let mut parent = Grammar::from_glrm(r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#).unwrap()
            .prepare(&v).unwrap();
        let child = Grammar::from_ebnf(r#"start ::= "y""#).unwrap().prepare_with(&v,
            BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::TokenSet)).unwrap();
        parent.bindings.insert("child".to_owned(), ModuleBinding::Module(Arc::new(child)));
        let parent = PreparedGrammar::load(parent.save(), &v).unwrap();
        let linked = parent.clone().link_with(
            BuildOptions::default().boundary_trigger(BoundaryTriggerDetail::None),
        ).unwrap();
        assert!(matches!(linked.boundary_trigger, crate::runtime::BoundaryTrigger::None));
        let gated = parent.link_with(BuildOptions::default()).unwrap();
        assert!(matches!(gated.boundary_trigger, crate::runtime::BoundaryTrigger::TokenSet(_)));
    }
}


#[cfg(test)]
mod final_ordinary_dynamic_selection_tests {
    use super::*;
    #[test]
    fn fast_build_uses_the_actual_dynamic_compiler_and_preserves_loaded_representation() {
        let v=Vocab::new(vec![(0,b"(".to_vec()),(1,b")".to_vec()),(2,b"()".to_vec()),(3,b"x".to_vec())]);
        let g=Grammar::from_ebnf(r#"start ::= "(" start ")" start | """#).unwrap();
        let actual=g.compile_with_build_options(&v,BuildOptions::default().optimization(MaskStrategy::FAST_COMPILE)).unwrap();
        let oracle=DynamicConstraint::compile(g.clone(),&v).unwrap().into_constraint();
        assert!(actual.uses_dynamic_runtime());
        assert!(!actual.has_template_parser()); assert!(actual.table.is_present());
        assert_eq!(actual.dynamic_mask_vocab.is_grammar_quotiented(),oracle.dynamic_mask_vocab.is_grammar_quotiented());
        for bytes in [actual.save(),actual.save()] {
            let restored=RuntimeConstraint::load(bytes,&v).unwrap();
            assert!(restored.uses_dynamic_runtime());
            assert!(!restored.has_template_parser()); assert!(restored.table.is_present());
            for prefix in [b"".as_slice(),b"(",b"()",b"()("] {
                let mut a=restored.start();let mut b=oracle.start_dynamic();
                assert!(a.commit_bytes(prefix));assert!(b.commit_bytes(prefix));
                assert_eq!(a.mask(),b.mask());assert_eq!(a.is_accepting(),b.is_accepting());
            }
        }
        let balanced=g.compile_with_build_options(&v,MaskMode::C.build_options()).unwrap();
        assert!(balanced.uses_dynamic_runtime()); assert!(balanced.has_template_parser());
        assert!(!balanced.table.is_present());
        for mode in [MaskMode::Auto,MaskMode::ParserDwa] {
            assert!(!g.compile_with_build_options(&v,mode.build_options()).unwrap().uses_dynamic_runtime());
        }
    }
}

#[cfg(test)]
mod final_unsplit_dynamic_tests {
    use super::*;
    use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};
    use crate::grammar::expr_nfa::ExprNFA;
    use crate::automata::unweighted_u32::nfa::NFA;

    #[test]
    fn unsplit_embedded_regions_match_existing_multiple_root_union() {
        fn region(byte: u8) -> GrammarExpr {
            let mut nfa=NFA::new_empty();let start=nfa.add_state();let end=nfa.add_state();
            nfa.start_states.push(start);nfa.set_accepting(end);
            nfa.add_transition(start,0,end);nfa.add_transition(end,0,end);
            let mut expr=ExprNFA::new(nfa,vec![GrammarExpr::Literal(vec![byte])]);
            expr.prefer_direct_nfa_emission=true;
            GrammarExpr::ExprNFA(Box::new(expr))
        }
        let named=NamedGrammar {
            start:"root".into(),ignore:None,lexer_partitions:BTreeMap::new(),
            lexer_literal_partitions:BTreeMap::new(),default_lexer_partition:None,
            rules:vec![
                NamedRule {name:"root".into(),expr:GrammarExpr::Choice(vec![GrammarExpr::Sequence(vec![GrammarExpr::Ref("a".into()),GrammarExpr::Literal(b"x".to_vec())]),GrammarExpr::Sequence(vec![GrammarExpr::Ref("b".into()),GrammarExpr::Literal(b"x".to_vec())])]),is_terminal:false,is_internal:false},
                NamedRule {name:"a".into(),expr:region(b'a'),is_terminal:false,is_internal:false},
                NamedRule {name:"b".into(),expr:region(b'b'),is_terminal:false,is_internal:false},
            ],
        };
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"aa".to_vec()),(3,b"bb".to_vec()),(4,b"ab".to_vec()),(5,b"x".to_vec()),(6,b"ax".to_vec()),(7,b"a".to_vec()),(8,b"bx".to_vec())]);
        let oracle=crate::import::compile_dynamic_named_union_for_test(named.clone(),&v).unwrap();
        assert!(oracle.clone_constraints().len()>1,"oracle must exercise the existing splitter");
        let c=crate::import::compile_dynamic_named_single(named,&v,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,false).unwrap();
        assert!(c.uses_dynamic_runtime());assert!(!c.has_template_parser());
        for actual in [c.clone(),RuntimeConstraint::load(c.save(), &v).unwrap(),
            RuntimeConstraint::load(c.save(),&v).unwrap()] {
            for prefix in [b"".as_slice(),b"a",b"aa",b"b",b"bb",b"ax",b"bx",b"ab",b"ba"] {
                let mut a=actual.start();let mut o=oracle.start();
                let _=a.commit_bytes(prefix);let _=o.commit_bytes(prefix);
                assert_eq!(a.mask(),o.mask(),"after {prefix:?}");
                assert_eq!(a.is_accepting(),o.is_accepting());
                assert_eq!((a.mask().iter().all(|word| *word == 0)),(o.mask().iter().all(|word| *word == 0)));
                assert_eq!(a.forced_canonical_reference(),o.forced_canonical_reference());
            }
        }
    }
}

#[cfg(test)]
mod bare_direct_region_regression_tests {
    use super::*;
    use crate::grammar::ast::{GrammarExpr, NamedGrammar, NamedRule};
    use crate::grammar::expr_nfa::ExprNFA;
    use crate::automata::unweighted_u32::nfa::NFA;

    #[test]
    fn bare_region_union_and_unsplit_lr_preserve_the_complete_language() {
        fn region(byte:u8)->GrammarExpr {
            let mut nfa=NFA::new_empty();let start=nfa.add_state();let end=nfa.add_state();
            nfa.start_states.push(start);nfa.set_accepting(end);
            nfa.add_transition(start,0,end);nfa.add_transition(end,0,end);
            GrammarExpr::ExprNFA(Box::new(ExprNFA::new(nfa,vec![GrammarExpr::Literal(vec![byte])])
                .with_embedded_direct_nfa_emission()))
        }
        let named=NamedGrammar { start:"root".into(),ignore:None,lexer_partitions:BTreeMap::new(),
            lexer_literal_partitions:BTreeMap::new(),default_lexer_partition:None,
            rules:vec![
                NamedRule{name:"root".into(),expr:GrammarExpr::Choice(vec![GrammarExpr::Ref("a".into()),GrammarExpr::Ref("b".into())]),is_terminal:false,is_internal:false},
                NamedRule{name:"a".into(),expr:region(b'a'),is_terminal:false,is_internal:false},
                NamedRule{name:"b".into(),expr:region(b'b'),is_terminal:false,is_internal:false}] };
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"aa".to_vec()),
            (3,b"bb".to_vec()),(4,b"ab".to_vec()),(7,b"a".to_vec())]);
        let union=crate::import::compile_dynamic_named_union_for_test(named.clone(),&v).unwrap();
        let single=crate::import::compile_dynamic_named_single(named,&v,
            crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged,false).unwrap();
        assert_eq!(union.clone_constraints().len(),2);
        for c in union.clone_constraints() {
            assert!(c.table.is_present()); assert!(!c.table.action.is_empty());
            assert!(c.direct_regular_automaton.is_none()); assert!(!c.has_template_parser());
        }
        assert_eq!(union.start().mask(),vec![143]);
        for actual in [single.clone(),RuntimeConstraint::load(single.save(), &v).unwrap(),
            RuntimeConstraint::load(single.save(),&v).unwrap()] {
            for (prefix,mask,accept,reject) in [(b"".as_slice(),143,false,false),(b"a",133,true,false),
                (b"aa",133,true,false),(b"b",10,true,false),(b"bb",10,true,false),
                (b"ab",0,false,true),(b"ba",0,false,true)] {
                let mut a=actual.start();let mut o=union.start();
                let ae= a.commit_bytes(prefix);let oe= o.commit_bytes(prefix);
                assert_eq!(!ae,!oe,"after {prefix:?}");
                assert_eq!(a.mask(),vec![mask],"unsplit after {prefix:?}");
                assert_eq!(o.mask(),vec![mask],"union after {prefix:?}");
                assert_eq!((a.is_accepting(),(a.mask().iter().all(|word| *word == 0))),(accept,reject));
                assert_eq!((o.is_accepting(),(o.mask().iter().all(|word| *word == 0))),(accept,reject));
                assert_eq!(a.forced_canonical_reference(),o.forced_canonical_reference());
            }
        }
    }
}

#[cfg(test)]
mod static_whole_token_regression_tests {
    use super::*;
    #[test]
    fn compressed_whole_word_is_allowed_before_and_after_reload() {
        let mut source=String::from("start: r0\n");for i in 0..30 {source.push_str(&format!("r{i}: \"a\" r{}\n",i+1));}source.push_str("r30: \"b\"\n");
        let mut word=vec![b'a';30];word.push(b'b');
        for entries in [vec![(3,word.clone())],vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"aa".to_vec()),(3,word.clone())]] {
            let singleton=entries.len()==1;let v=Vocab::new(entries);
            for mode in [MaskMode::Auto,MaskMode::ParserDwa] {
                let c=Grammar::from_lark(&source).unwrap().compile_with_build_options(&v,mode.build_options()).unwrap();
                assert!(c.direct_regular_automaton.is_some());
                for c in [c.clone(),RuntimeConstraint::load(c.save(), &v).unwrap(),RuntimeConstraint::load(c.save(),&v).unwrap()] {
                    let mut state=c.start();assert_eq!(state.mask(),vec![if singleton {8} else {13}]);
                    assert!(state.commit_token(3));assert!(state.is_accepting());assert!(state.mask().iter().all(|word| *word == 0));
                }
            }
        }
    }
}

#[cfg(test)]
mod tiny_cfg_compile_blowup_tests {
    use super::*;

    // Random tiny CFGs reduced from the compile-time blow-up repro. Symbols
    // >= 0 are nonterminals; symbol < 0 is byte b'a' + (-symbol - 1).
    struct Case {
        start: usize,
        bound: usize,
        rules: &'static [(usize, &'static [isize])],
    }

    // Minimised from case 296 by deleting rules and symbols while the
    // compile stayed above the timeout. It keeps the nullable left-recursive
    // core that produced the template blow-up.
    const COMPACT: Case = Case {
        start: 1,
        bound: 3,
        rules: &[
            (1, &[-3]),
            (0, &[1, -2]),
            (1, &[1, 0, 1]),
            (1, &[]),
        ],
    };

    fn lark(case: &Case) -> String {
        let mut by_lhs: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        let mut defined = BTreeSet::new();
        for (lhs, rhs) in case.rules {
            defined.insert(*lhs);
            let mut alternatives = Vec::new();
            let mut text = String::new();
            for symbol in *rhs {
                if !text.is_empty() {
                    text.push(' ');
                }
                if *symbol >= 0 {
                    text.push_str(&format!("n{symbol}"));
                } else {
                    let byte = (b'a' + ((-*symbol - 1) as u8)) as char;
                    text.push('"');
                    text.push(byte);
                    text.push('"');
                }
            }
            if text.is_empty() {
                text.push_str("\"\"");
            }
            alternatives.push(text);
            by_lhs.entry(*lhs).or_default().extend(alternatives);
        }
        for (lhs, rhs) in case.rules {
            for symbol in *rhs {
                assert!(
                    *symbol < 0 || defined.contains(&(*symbol as usize)),
                    "referenced nonterminal must have a production",
                );
            }
        }
        let mut source = String::new();
        source.push_str(&format!("start: n{}\n", case.start));
        for (lhs, alternatives) in &by_lhs {
            source.push_str(&format!("n{lhs}: {}\n", alternatives.join(" | ")));
        }
        source
    }

    fn vocabulary() -> Vocab {
        Vocab::new(vec![(0, vec![b'a']), (1, vec![b'b']), (2, vec![b'c'])])
    }

    fn bounded_language(case: &Case) -> BTreeSet<Vec<u8>> {
        // Prefix validity is a property of the unbounded language: e.g. the
        // prefix "bcc" is witnessed only by the four-byte word "bccb". Derive
        // a generous margin beyond the checked word length so every prefix of
        // a checked word has a witness; membership of the short words
        // themselves is unaffected.
        let limit = case.bound + 2;
        let mut sets: BTreeMap<usize, BTreeSet<Vec<u8>>> = BTreeMap::new();
        for (lhs, _) in case.rules {
            sets.entry(*lhs).or_default();
        }
        loop {
            let mut changed = false;
            for (lhs, rhs) in case.rules {
                let mut words: BTreeSet<Vec<u8>> = std::iter::once(Vec::new()).collect();
                for symbol in *rhs {
                    let next = if *symbol >= 0 {
                        sets.get(&(*symbol as usize)).cloned().unwrap_or_default()
                    } else {
                        std::iter::once(vec![b'a' + ((-*symbol - 1) as u8)]).collect()
                    };
                    let mut product = BTreeSet::new();
                    for prefix in &words {
                        for suffix in &next {
                            if prefix.len() + suffix.len() <= limit {
                                let mut word = prefix.clone();
                                word.extend_from_slice(suffix);
                                product.insert(word);
                            }
                        }
                    }
                    words = product;
                }
                let target = sets.get_mut(lhs).expect("defined above");
                for word in words {
                    changed |= target.insert(word);
                }
            }
            if !changed {
                break;
            }
        }
        sets.remove(&case.start).unwrap_or_default()
    }

    fn all_words(case: &Case) -> Vec<Vec<u8>> {
        let mut words = vec![Vec::new()];
        let mut level = vec![Vec::new()];
        for _ in 0..case.bound {
            let mut next = Vec::new();
            for word in &level {
                for byte in *b"abc" {
                    let mut extended = word.clone();
                    extended.push(byte);
                    next.push(extended);
                }
            }
            words.extend(next.iter().cloned());
            level = next;
        }
        words
    }

    fn assert_exact(constraint: &RuntimeConstraint, case: &Case) {
        let language = bounded_language(case);
        let words = all_words(case);
        for word in &words {
            let mut state = constraint.start();
            let mut alive = true;
            for (position, byte) in word.iter().enumerate() {
                let mask = state.mask();
                for candidate in 0..3u8 {
                    let mut prefix = word[..position].to_vec();
                    prefix.push(b'a' + candidate);
                    let expected = language
                        .iter()
                        .any(|word| word.starts_with(&prefix));
                    let id = candidate as usize;
                    let allowed = mask[id / 32] & (1 << (id % 32)) != 0;
                    assert_eq!(
                        allowed, expected,
                        "mask mismatch at {:?}[{position}] for token {candidate}",
                        String::from_utf8_lossy(word),
                    );
                }
                for id in 3..(mask.len() * 32) {
                    assert_eq!(
                        mask[id / 32] & (1 << (id % 32)),
                        0,
                        "unexpected token id {id} at {:?}[{position}]",
                        String::from_utf8_lossy(word),
                    );
                }
                let id = (byte - b'a') as usize;
                if mask[id / 32] & (1 << (id % 32)) == 0 {
                    // The prefix is already outside the language; the exact
                    // mask assertion above is the check for this word.
                    alive = false;
                    break;
                }
                assert!(state
                    .commit_token((byte - b'a') as u32));
            }
            if alive {
                assert_eq!(
                    state.is_accepting(),
                    language.contains(word),
                    "acceptance mismatch for {:?}",
                    String::from_utf8_lossy(word),
                );
            } else {
                assert!(
                    !language.contains(word),
                    "mask rejected an in-language word {:?}",
                    String::from_utf8_lossy(word),
                );
            }
        }
    }

    fn compile(case: &Case, strategy: MaskMode) -> RuntimeConstraint {
        let source = lark(case);
        let grammar = Grammar::from_lark(&source).unwrap();
        let module = Module::new(&grammar);
        module
            .compile_with(&vocabulary(), strategy.compile_options())
            .unwrap()
    }

    #[test]
    fn compact_nullable_left_recursive_grammar_builds_minimal_templates() {
        for strategy in [MaskMode::C, MaskMode::ParserDwa] {
            let constraint = compile(&COMPACT, strategy);
            assert!(
                !constraint.table.is_present(),
                "the compact grammar must keep its finite templates",
            );
            assert!(constraint.template_parser.is_some());
            assert_exact(&constraint, &COMPACT);
        }
    }
}

#[cfg(test)]
mod module_build_overhead_tests {
    use super::*;
    use crate::__private::VocabExt as _;
    use std::time::Instant;

    fn jsb_like_schema(properties: usize) -> String {
        let props = (0..properties)
            .map(|i| (format!("p{i}"), serde_json::json!({"type": "integer"})))
            .collect::<serde_json::Map<_, _>>();
        serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": props,
        })
        .to_string()
    }

    fn jsb_like_vocab(tokens: u32) -> Vocab {
        let vocab = Vocab::new((0..tokens).map(|id| (id, format!("t{id}").into_bytes())).collect());
        vocab.prepare_for_compile();
        vocab
    }

    fn minimum(values: &[f64]) -> f64 {
        values.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// A standalone Module builds the same default TokenSet trigger as the
    /// direct Grammar path, and must stay close to it in build time.
    #[test]
    fn module_standalone_build_stays_close_to_grammar_build() {
        let vocab = jsb_like_vocab(16_384);
        let source = jsb_like_schema(80);

        let grammar = Grammar::from_json_schema(&source).unwrap();
        let direct = grammar
            .compile_with_build_options(
                &vocab,
                BuildOptions::default()
                    .optimization(MaskStrategy::ParserDwa)
                    .end_tokens([0u32]),
            )
            .expect("direct build");
        let standalone = Grammar::from_expr(Expr::sequence([
            Expr::from(grammar.clone()),
            Expr::from(TokenSet::new([0u32])),
        ]))
        .expect("root expression")
        .compile(&vocab)
        .expect("module build");
        assert!(
            matches!(standalone.boundary_trigger, crate::runtime::BoundaryTrigger::TokenSet(_)),
            "standalone Module must carry the default TokenSet trigger"
        );
        assert_eq!(
            standalone.boundary_trigger.token_summary(),
            direct.boundary_trigger.token_summary(),
            "Module and Grammar must agree on the standalone token set"
        );

        // One untimed warm-up pair: the first compile of each shape pays lazy
        // pool and cache initialization that neither steady-state path should
        // be charged for.
        {
            let grammar = Grammar::from_json_schema(&source).unwrap();
            grammar
                .compile_with_build_options(
                    &vocab,
                    BuildOptions::default()
                        .optimization(MaskStrategy::ParserDwa)
                        .end_tokens([0u32]),
                )
                .expect("warmup direct build");
            let grammar = Grammar::from_json_schema(&source).unwrap();
            Grammar::from_expr(Expr::sequence([
                Expr::from(grammar.clone()),
                Expr::from(TokenSet::new([0u32])),
            ]))
            .expect("root expression")
            .compile(&vocab)
            .expect("warmup module build");
        }

        let mut grammar_ms = Vec::new();
        let mut module_ms = Vec::new();
        for _ in 0..7 {
            let grammar = Grammar::from_json_schema(&source).unwrap();
            let started = Instant::now();
            grammar
                .compile_with_build_options(
                    &vocab,
                    BuildOptions::default()
                        .optimization(MaskStrategy::ParserDwa)
                        .end_tokens([0u32]),
                )
                .expect("direct build");
            grammar_ms.push(started.elapsed().as_secs_f64() * 1000.0);

            let grammar = Grammar::from_json_schema(&source).unwrap();
            let started = Instant::now();
            Grammar::from_expr(Expr::sequence([
                Expr::from(grammar.clone()),
                Expr::from(TokenSet::new([0u32])),
            ]))
            .expect("root expression")
            .compile(&vocab)
            .expect("module build");
            module_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        // Uncontended work time: noise only ever adds time, so the minimum is
        // the robust estimator for these single-shot compiles.
        let grammar_min = minimum(&grammar_ms);
        let module_min = minimum(&module_ms);
        eprintln!(
            "[module-build-overhead] grammar_min={grammar_min:.3}ms module_min={module_min:.3}ms ratio={:.2} grammar_ms={grammar_ms:?} module_ms={module_ms:?}",
            module_min / grammar_min
        );
        assert!(
            module_min <= grammar_min * 1.5,
            "Module build minimum {module_min:.3}ms is more than 1.5x the direct Grammar build minimum {grammar_min:.3}ms; grammar samples {grammar_ms:?}, module samples {module_ms:?}",
        );
    }
}

#[cfg(test)]
mod mask_strategy_configuration_tests {
    use super::*;
    use crate::__private::VocabExt as _;
    use crate::test_modes::MaskMode;

    fn merged_vocab() -> Vocab {
        let vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (1, b"b".to_vec()),
            (2, b"ab".to_vec()),
            (3, b"x".to_vec()),
            (4, b"y".to_vec()),
        ]);
        vocab.prepare_for_compile();
        vocab
    }

    fn composed_module(child: &Grammar) -> Module {
        let mut parent = Module::new(
            &Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "x" child "y";"#,
            )
            .unwrap(),
        );
        parent.bind("child", child).unwrap();
        parent
    }

    fn assert_same_language(
        reference: &crate::ConstraintState<'_>,
        candidate: &crate::ConstraintState<'_>,
        depth: usize,
    ) -> usize {
        let mask = reference.mask();
        assert_eq!(mask, candidate.mask(), "mask mismatch at depth {depth}");
        assert_eq!(
            reference.is_accepting(),
            candidate.is_accepting(),
            "acceptance mismatch at depth {depth}",
        );
        if depth == 0 {
            return 1;
        }
        let mut visited = 1;
        for (word, bits) in mask.iter().enumerate() {
            for bit in 0..32 {
                if bits & (1 << bit) == 0 {
                    continue;
                }
                let token = (word * 32 + bit) as u32;
                let mut next_reference = reference.clone();
                assert!(next_reference.commit_token(token));
                let mut next_candidate = candidate.clone();
                assert!(next_candidate.commit_token(token));
                visited += assert_same_language(&next_reference, &next_candidate, depth - 1);
            }
        }
        visited
    }

    fn assert_inputs_match(
        merged: &RuntimeConstraint,
        unmerged: &RuntimeConstraint,
        inputs: &[&[u32]],
    ) {
        for input in inputs {
            let mut merged_state = merged.start();
            let mut unmerged_state = unmerged.start();
            for &token in *input {
                assert_eq!(
                    merged_state.mask(),
                    unmerged_state.mask(),
                    "mask mismatch before token {token} in {input:?}",
                );
                assert_eq!(
                    merged_state.is_accepting(),
                    unmerged_state.is_accepting(),
                    "acceptance mismatch before token {token} in {input:?}",
                );
                let merged_commit = merged_state.commit_token(token);
                let unmerged_commit = unmerged_state.commit_token(token);
                assert_eq!(merged_commit, unmerged_commit, "commit mismatch for {input:?}");
                if !merged_commit {
                    break;
                }
            }
            assert_eq!(merged_state.mask(), unmerged_state.mask(), "final mask {input:?}");
            assert_eq!(
                merged_state.is_accepting(),
                unmerged_state.is_accepting(),
                "final acceptance {input:?}",
            );
        }
    }

    #[test]
    fn merged_compositions_match_unmerged_masks() {
        // Regression for byte-spanning child boundaries: "xay" can be one
        // token spanning x + child(a) + y, while "ay" cannot.
        let tokens: [(u32, Vec<u8>); 9] = [
            (0, b"x".to_vec()),
            (1, b"a".to_vec()),
            (2, b"y".to_vec()),
            (3, b"xay".to_vec()),
            (4, b"xy".to_vec()),
            (5, b"ay".to_vec()),
            (6, b"aa".to_vec()),
            (7, b"yx".to_vec()),
            (8, b"".to_vec()),
        ];
        let vocab = Vocab::new(tokens.to_vec());
        let child = Grammar::from_glrm("start root t A = /a?/ nt root = A").unwrap();
        let mut parent = Module::new(
            &Grammar::from_glrm(
                r#"glrm 1; start root; extern grammar C; nt root = "x" C "y";"#,
            )
            .unwrap(),
        );
        parent.bind("C", child).unwrap();
        let unmerged = parent
            .compile_with(&vocab, MaskMode::A.compile_options())
            .expect("unmerged TokenTrie compile");
        let merged = parent
            .compile_with(&vocab, MaskMode::C.compile_options())
            .expect("merged compile");
        let merged_lr = parent
            .compile_with(&vocab, MaskMode::D.compile_options())
            .expect("merged LR compile");
        assert!(merged.runtime_is_grammar_quotiented(), "C flattened composition must be quotiented");
        assert!(merged_lr.runtime_is_grammar_quotiented(), "D flattened composition must be quotiented");
        assert_eq!(merged_lr.parser_backend(), ParserBackend::LrTable);
        assert!(merged_lr.runtime_has_lr_table());
        assert!(!merged_lr.runtime_has_template_parser());
        assert_inputs_match(
            &merged,
            &unmerged,
            &[
                &[],
                &[3],
                &[4],
                &[5],
                &[0],
                &[0, 1],
                &[0, 1, 2],
                &[3, 3],
                &[4, 3],
            ],
        );
        assert_inputs_match(
            &merged_lr,
            &unmerged,
            &[
                &[],
                &[3],
                &[4],
                &[5],
                &[0],
                &[0, 1],
                &[0, 1, 2],
                &[3, 3],
                &[4, 3],
            ],
        );

        // A second composition, with an ordinary child grammar.
        let vocab = merged_vocab();
        let child = Grammar::from_ebnf(r#"start ::= "a" | "b""#).unwrap();
        let parent = composed_module(&child);
        let unmerged = parent
            .compile_with(&vocab, MaskMode::A.compile_options())
            .expect("unmerged TokenTrie compile");
        let merged = parent
            .compile_with(&vocab, MaskMode::C.compile_options())
            .expect("merged compile");
        let merged_lr = parent
            .compile_with(&vocab, MaskMode::D.compile_options())
            .expect("merged LR compile");
        assert!(merged.runtime_is_grammar_quotiented());
        assert!(merged_lr.runtime_is_grammar_quotiented());
        assert_inputs_match(&merged, &unmerged, &[&[], &[0], &[0, 1], &[0, 3], &[3, 0], &[4, 0]]);
        assert_inputs_match(&merged_lr, &unmerged, &[&[], &[0], &[0, 1], &[0, 3], &[3, 0], &[4, 0]]);

        // Two levels of composition flatten into one partition.
        let nested_vocab = Vocab::new(vec![
            (0, b"a".to_vec()),
            (3, b"x".to_vec()),
            (4, b"y".to_vec()),
            (5, b"(".to_vec()),
            (6, b")".to_vec()),
            (7, b"(a)".to_vec()),
            (8, Vec::new()),
        ]);
        let leaf = Grammar::from_ebnf(r#"start ::= "a"?"#).unwrap();
        let mut middle = Module::new(
            &Grammar::from_glrm(r#"glrm 1; start root; extern grammar leaf; nt root = "(" leaf ")";"#)
                .unwrap(),
        );
        middle.bind("leaf", leaf).unwrap();
        let mut top = Module::new(
            &Grammar::from_glrm(r#"glrm 1; start root; extern grammar middle; nt root = "x" middle "y";"#)
                .unwrap(),
        );
        top.bind("middle", middle).unwrap();
        let unmerged = top
            .compile_with(&nested_vocab, MaskMode::A.compile_options())
            .unwrap();
        let merged_lr = top
            .compile_with(&nested_vocab, MaskMode::D.compile_options())
            .unwrap();
        assert!(merged_lr.runtime_is_grammar_quotiented());
        assert_inputs_match(&merged_lr, &unmerged, &[&[], &[0], &[0, 5], &[4, 0], &[0, 7, 3]]);

        // A local C compile still uses its own quotient.
        let local = Grammar::from_ebnf(r#"start ::= "a" "b" | "ab""#)
            .unwrap()
            .compile_with(&merged_vocab(), MaskMode::C.compile_options())
            .unwrap();
        assert!(local.dynamic_mask_vocab.is_grammar_quotiented());
    }

    #[test]
    fn merged_request_over_retained_parts_keeps_them_and_matches_unmerged() {
        let vocab = merged_vocab();
        let child = Grammar::from_ebnf(r#"start ::= "a" | "b""#).unwrap();
        let mut parent = composed_module(&child);
        parent
            .precompile_with(&vocab, MaskMode::ParserDwa.compile_options())
            .unwrap();
        let unmerged = parent
            .compile_with(&vocab, MaskMode::A.compile_options())
            .unwrap();
        for mode in [MaskMode::C, MaskMode::D] {
            let merged = parent
                .compile_with(&vocab, mode.compile_options())
                .unwrap();
            assert_inputs_match(&merged, &unmerged, &[&[], &[0], &[0, 1], &[0, 3], &[3, 0], &[4, 0]]);
        }
    }

    #[test]
    fn merged_flatten_resolves_lark_and_json_schema_children() {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"a".to_vec()),
            (3, b"\"a\"".to_vec()),
            (4, Vec::new()),
        ]);
        let lark = Grammar::from_lark(r#"start: "a""#).unwrap();
        let json = Grammar::from_json_schema(r#"{"type":"string","maxLength":1}"#).unwrap();
        let build = || {
            let mut parent = Module::new(
                &Grammar::from_glrm(
                    r#"glrm 1; start root; extern grammar L; extern grammar J; nt root = "x" L "y" | J;"#,
                )
                .unwrap(),
            );
            parent.bind("L", lark.clone()).unwrap();
            parent.bind("J", json.clone()).unwrap();
            parent
        };
        let unmerged = build()
            .compile_with(&vocab, MaskMode::A.compile_options())
            .unwrap();
        for mode in [MaskMode::C, MaskMode::D] {
            let merged = build()
                .compile_with(&vocab, mode.compile_options())
                .unwrap();
            assert!(
                merged.runtime_is_grammar_quotiented(),
                "{mode:?}: flattened composition must be quotiented",
            );
            assert_inputs_match(&merged, &unmerged, &[&[], &[0], &[0, 2], &[0, 2, 1], &[3], &[0, 3]]);
        }
    }
}
