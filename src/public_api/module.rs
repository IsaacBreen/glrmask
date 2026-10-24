//! Owned assembly snapshots. Local compilation is independent of occurrence bindings.
#[cfg(test)]
use crate::test_modes::MaskMode;
use super::*;
use super::expr::check_expr_unbound;
use std::collections::{HashMap, BTreeMap, BTreeSet};
mod artifact;
#[cfg(test)]
mod mixed_validation;
#[cfg(test)]
thread_local! { static BIND_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }


/// Options for compiling a grammar or module.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompileOptions {
    pub(crate) mask_strategy: MaskStrategy,
    pub(crate) boundary_mask_strategy: BoundaryMaskStrategy,
    pub(crate) parser_backend: ParserBackend,
    pub(crate) merge_equivalent_tokens: bool,
}

impl CompileOptions {
    /// Sets how masks are computed. Defaults to [`MaskStrategy::Auto`].
    pub fn mask_strategy(mut self, strategy: MaskStrategy) -> Self {
        self.mask_strategy = strategy;
        self
    }

    /// Sets how masks are computed at grammar boundaries. Defaults to
    /// [`BoundaryMaskStrategy::Auto`].
    pub fn boundary_mask_strategy(mut self, strategy: BoundaryMaskStrategy) -> Self {
        self.boundary_mask_strategy = strategy;
        self
    }

    /// Sets how the parser is represented for parts compiled with
    /// [`TokenTrie`](MaskStrategy::TokenTrie). Defaults to [`ParserBackend::LrTable`].
    pub fn parser_backend(mut self, backend: ParserBackend) -> Self {
        self.parser_backend = backend;
        self
    }

    /// Sets whether [`TokenTrie`](MaskStrategy::TokenTrie) merges tokens that the
    /// grammar treats the same, so masks walk fewer tokens. Requires
    /// [`ParserBackend::TemplateDfa`]. Off by default.
    pub fn merge_equivalent_tokens(mut self, enable: bool) -> Self {
        self.merge_equivalent_tokens = enable;
        self
    }

    /// The internal engine for these options, rejecting combination D.
    pub(crate) fn engine(&self) -> Result<Engine> {
        engine_for(
            self.mask_strategy,
            self.parser_backend,
            self.merge_equivalent_tokens,
        )
    }

    /// The TokenTrie parser backend for these options.
    fn token_trie_backend(&self) -> crate::runtime::parser_backend::DynamicParserBackend {
        token_trie_backend(self.parser_backend)
    }
}

/// A value that can be bound to an extern.
///
/// Converted automatically from a [`Module`], [`Grammar`], [`Expr`], [`TokenSet`]
/// or token ID.
pub struct Binding(pub(crate) BindingKind);

#[derive(Debug, Clone)]
pub(crate) enum BindingKind {
    Module(Module),
    Expr(Expr),
    TokenSet(TokenSet),
}

impl From<Module> for Binding { fn from(m: Module) -> Self { Self(BindingKind::Module(m)) } }
impl From<&Module> for Binding { fn from(m: &Module) -> Self { Self(BindingKind::Module(m.clone())) } }
impl From<Grammar> for Binding { fn from(g: Grammar) -> Self { Self(BindingKind::Module(Module::new(&g))) } }
impl From<&Grammar> for Binding { fn from(g: &Grammar) -> Self { Self(BindingKind::Module(Module::new(g))) } }
impl From<Expr> for Binding { fn from(e: Expr) -> Self { Self(BindingKind::Expr(e)) } }
impl From<&Expr> for Binding { fn from(e: &Expr) -> Self { Self(BindingKind::Expr(e.clone())) } }
impl From<TokenSet> for Binding { fn from(t: TokenSet) -> Self { Self(BindingKind::TokenSet(t)) } }
impl From<&TokenSet> for Binding { fn from(t: &TokenSet) -> Self { Self(BindingKind::TokenSet(t.clone())) } }
impl From<u32> for Binding { fn from(id: u32) -> Self { Self(BindingKind::TokenSet(TokenSet::from(id))) } }

#[derive(Debug, Clone)]
pub(crate) enum BoundValue {
    Module(Module),
    TokenSet(TokenSet),
}

/// A grammar you can compose with others and partly compile before its externs are bound.
#[derive(Debug, Clone)]
pub struct Module {
    pub(crate) node: Arc<Node>,
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) source: Arc<Grammar>,
    pub(crate) prepared: BTreeMap<u8, Arc<dyn RetainedPreparation>>,
    pub(crate) bindings: OrdMap<String, BoundValue>,
    pub(crate) certificate: Arc<OnceLock<Result<Arc<Certificate>>>>,
    #[cfg(test)]
    pub(crate) force_boundary_dwa_decline: std::sync::Arc<std::sync::atomic::AtomicBool>,
    #[cfg(test)]
    pub(crate) force_parser_dwa_decline: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub(crate) trait RetainedPreparation: Send + Sync + std::fmt::Debug {
    fn prepared(&self) -> &PreparedGrammar;
    /// Rehydrates a vocabulary-free retained body, or returns an equivalent
    /// already-bound preparation.
    fn rehydrate_vocab(&self, vocab: &Vocab) -> Result<Arc<dyn RetainedPreparation>>;
}
impl RetainedPreparation for PreparedGrammar {
    fn prepared(&self) -> &PreparedGrammar { self }
    fn rehydrate_vocab(&self, vocab: &Vocab) -> Result<Arc<dyn RetainedPreparation>> {
        let Some(wire) = self.retained_artifact.as_ref() else {
            return Ok(Arc::new(self.clone()));
        };
        if self.retained_vocab_fingerprint != Some(crate::compiler::compile::vocab_fingerprint(vocab)) {
            return Err(error("module retained work was precompiled for a different vocabulary"));
        }
        prepared_artifact::load(wire.as_ref(), Some(vocab))
            .map(|prepared| Arc::new(prepared) as Arc<dyn RetainedPreparation>)
    }
}
impl std::ops::Deref for dyn RetainedPreparation {
    type Target = PreparedGrammar;
    fn deref(&self) -> &Self::Target { self.prepared() }
}
pub(crate) fn retain_local(prepared: PreparedGrammar) -> Arc<dyn RetainedPreparation> {
    Arc::new(prepared)
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Certificate {
    height: usize,
    instances: u128,
    ports: u128,
    target: Option<Vocab>,
    pending_sets: Vec<TokenSet>,
}
impl Certificate {
    fn target(&mut self, vocab: &Vocab) -> Result<()> {
        if self.target.as_ref().is_some_and(|v| !v.same_model_vocab(vocab)) {
            return Err(error("module graph targets incompatible vocabularies"));
        }
        if self.target.is_none() {
            // The first supplied vocabulary must validate every pending token
            // set. Set the target first so `tokens` checks rather than requeues.
            self.target = Some(vocab.clone());
            for set in std::mem::take(&mut self.pending_sets) {
                self.tokens(&set)?;
            }
        }
        Ok(())
    }
    fn tokens(&mut self, set: &TokenSet) -> Result<()> {
        if let Some(vocab) = &self.target {
            if let Some(id) = vocab.first_unknown_token_id(set) {
                return Err(error(format!("token ID {id} is not present in the target vocabulary")));
            }
        } else {
            self.pending_sets.push(set.clone());
        }
        Ok(())
    }
    fn value(&mut self, value: &BoundValue) -> Result<()> {
        match value {
            BoundValue::TokenSet(set) => self.tokens(set)?,
            BoundValue::Module(m) => {
                let child = m.certificate()?;
                self.height = self.height.max(child.height + 1);
                if let Some(v) = &child.target { self.target(v)?; }
                for set in &child.pending_sets { self.tokens(set)?; }
            }
        }
        Ok(())
    }
}

fn error(message: impl Into<String>) -> Error { Error::Compilation(message.into()) }

fn own_local(grammar: &Grammar) -> Grammar {
    Grammar {
        source: grammar.source.clone(),
        module_snapshot: Arc::new(OnceLock::new()),
        bindings: OrdMap::new(),
        declarations_cache: grammar.declarations_cache.clone(),
        analysis: graph::empty_cache(),
        parsed_component: grammar.parsed_component.clone(),
        exact_value: grammar.exact_value.clone(),
        lark_document: grammar.lark_document.clone(),
    }
}

/// The best retained preparation, in mask-speed order: `ParserDwa` (key 1),
/// merged `TokenTrie` (key 3), then a retained template-DFA `TokenTrie` shell
/// (key 2).
fn best_retained(prepared: &BTreeMap<u8, Arc<dyn RetainedPreparation>>) -> Option<Engine> {
    if prepared.contains_key(&Engine::ParserDwa.key()) {
        Some(Engine::ParserDwa)
    } else if prepared.contains_key(&Engine::MergedTokenTrie.key()) {
        Some(Engine::MergedTokenTrie)
    } else if prepared.contains_key(&Engine::TokenTrie.key()) {
        Some(Engine::TokenTrie)
    } else {
        None
    }
}

/// How one node is compiled by the current call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NodePlan {
    /// The engine preparation this node uses or builds.
    engine: Engine,
    /// The parser backend for a `TokenTrie` build.
    backend: ParserBackend,
}

/// Resolve one node's plan. `any_retained` is true when any node in the graph
/// already owns compiled machinery; a `TokenTrie` built in that graph must be
/// template-backed because it will be linked with template parts.
fn plan_node(
    requested: Engine,
    prepared: &BTreeMap<u8, Arc<dyn RetainedPreparation>>,
    any_retained: bool,
    requested_backend: ParserBackend,
) -> NodePlan {
    let retained = best_retained(prepared);
    match requested {
        Engine::Auto => NodePlan {
            engine: retained.unwrap_or(Engine::ParserDwa),
            backend: ParserBackend::TemplateDfa,
        },
        Engine::ParserDwa => NodePlan {
            engine: Engine::ParserDwa,
            backend: ParserBackend::TemplateDfa,
        },
        Engine::MergedTokenTrie => NodePlan {
            engine: match retained {
                Some(Engine::ParserDwa) | Some(Engine::MergedTokenTrie) => retained.unwrap(),
                _ => Engine::MergedTokenTrie,
            },
            backend: ParserBackend::TemplateDfa,
        },
        Engine::TokenTrie => match retained {
            Some(engine) => NodePlan {
                engine,
                backend: ParserBackend::TemplateDfa,
            },
            None => NodePlan {
                engine: Engine::TokenTrie,
                backend: if any_retained {
                    ParserBackend::TemplateDfa
                } else {
                    requested_backend
                },
            },
        },
    }
}

pub(crate) fn target_boundary() -> BoundaryTriggerDetail {
    // The public boundary trigger is always the LLM token set. BoundaryMaskStrategy
    // selects the boundary-parser wiring only.
    BoundaryTriggerDetail::TokenSet
}

impl Module {
    #[cfg(test)]
    pub(crate) fn force_boundary_dwa_decline(&mut self) {
        Arc::make_mut(&mut self.node)
            .force_boundary_dwa_decline
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn force_parser_dwa_decline(&mut self) {
        Arc::make_mut(&mut self.node)
            .force_parser_dwa_decline
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Grammar {
    pub(crate) fn module(&self) -> Module { Module::new(self) }
}

impl Module {
    /// Creates a module from `grammar`.
    pub fn new(grammar: &Grammar) -> Self {
        fn convert(grammar: &Grammar, cache: &mut HashMap<usize, Module>) -> Module {
            if let Some(module) = grammar.module_snapshot.get() { return module.clone(); }
            let id = graph::source_identity(grammar);
            if let Some(module) = cache.get(&id) { return module.clone(); }
            let mut bindings = OrdMap::new();
            for (name, value) in &grammar.bindings {
                let value = match value {
                    GrammarValue::Source(child) => BoundValue::Module(convert(child, cache)),
                    GrammarValue::TokenSet(t) => BoundValue::TokenSet(t.clone()),
                };
                bindings.insert(name.clone(), value);
            }
            let module = Module {
                node: Arc::new(Node {
                    source: Arc::new(own_local(grammar)),
                    prepared: BTreeMap::new(),
                    bindings,
                    certificate: Arc::new(OnceLock::new()),
                    #[cfg(test)]
                    force_boundary_dwa_decline: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                    #[cfg(test)]
                    force_parser_dwa_decline: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                }),
            };
            let _ = grammar.module_snapshot.set(module.clone());
            let module = grammar.module_snapshot.get().cloned().unwrap_or(module);
            cache.insert(id, module.clone());
            module
        }
        convert(grammar, &mut HashMap::new())
    }

    /// The module's externs, in declaration order.
    pub fn externs(&self) -> Externs<'_> {
        Externs {
            source: ExternsSource::Module(self),
            only_unbound: false,
        }
    }

    // Lets bindings surface a parse error where the Rust view must stay infallible.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn check_source(&self) -> Result<()> {
        self.node.source.declarations().map(|_| ())
    }

    /// Binds the extern `name` to `value`: a module, grammar, expression, token
    /// set or token ID.
    ///
    /// Fails if `name` is undeclared or already bound, or if `value` is the wrong
    /// kind for it. A token extern takes only a token set or token ID.
    pub fn bind(&mut self, name: &str, value: impl Into<Binding>) -> Result<()> {
        let next = self.bound(name, value.into())?;
        *self = next;
        Ok(())
    }

    fn bound(&self, name: &str, binding: Binding) -> Result<Self> {
        if name.contains("::") {
            return Err(error("qualified paths are not supported"));
        }
        let declared = self.node.source.declarations()?;
        let Some(&kind) = declared.kinds.get(name) else {
            return Err(error(format!("no external named {name:?}")));
        };
        if self.node.bindings.contains_key(name) {
            return Err(error(format!("extern \"{name}\" is already bound")));
        }
        let value = match binding.0 {
            BindingKind::Module(child) => {
                check_binding_kind(name, kind, true)
                    .map_err(|e| e.with_source_span(declared.spans.get(name).copied()))?;
                let unbound = child.externs().unbound();
                if !unbound.is_empty() {
                    let names: Vec<String> = unbound.iter().map(|e| e.name().to_string()).collect();
                    return Err(error(format!("cannot bind \"{name}\": the value has unbound externs [{}]", names.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", "))));
                }
                BoundValue::Module(child)
            }
            BindingKind::Expr(expr) => {
                check_binding_kind(name, kind, true)
                    .map_err(|e| e.with_source_span(declared.spans.get(name).copied()))?;
                let unbound = check_expr_unbound(&expr);
                if !unbound.is_empty() {
                    return Err(error(format!("cannot bind \"{name}\": the value has unbound externs [{}]", unbound.iter().map(|n| format!("\"{n}\"")).collect::<Vec<_>>().join(", "))));
                }
                let child = Module::from_expr(&expr)?;
                BoundValue::Module(child)
            }
            BindingKind::TokenSet(tokens) => {
                check_binding_kind(name, kind, false)
                    .map_err(|e| e.with_source_span(declared.spans.get(name).copied()))?;
                BoundValue::TokenSet(tokens)
            }
        };

        let previous = self.certificate()?;
        let mut certificate = previous.as_ref().clone();
        certificate.value(&value)?;
        let counts = |v: Option<&BoundValue>| -> Result<(u128, u128)> {
            if let Some(BoundValue::Module(m)) = v {
                let c = m.certificate()?;
                Ok((c.instances, c.ports))
            } else {
                Ok((0, 0))
            }
        };
        let old = self.node.bindings.get(name);
        let (old_nodes, old_ports) = counts(old)?;
        let (new_nodes, new_ports) = counts(Some(&value))?;
        certificate.instances = if certificate.instances == u128::MAX { u128::MAX } else { certificate.instances.saturating_sub(old_nodes).saturating_add(new_nodes) };
        certificate.ports = if certificate.ports == u128::MAX { u128::MAX } else { certificate.ports.saturating_sub(old_ports).saturating_add(new_ports).saturating_add(u128::from(old.is_none())) };
        let mut next = self.clone();
        let node = Arc::make_mut(&mut next.node);
        node.bindings.insert(name.to_owned(), value);
        node.certificate = Arc::new(OnceLock::from(Ok(Arc::new(certificate))));
        next.check_limits()?;
        Ok(next)
    }

    pub(crate) fn from_expr(expr: &Expr) -> Result<Self> {
        let (grammar, leaves) = expr.skeleton()?;
        let mut module = Module::new(&grammar);
        for (name, leaf) in leaves {
            module.bind(&name, leaf)?;
        }
        Ok(module)
    }

    pub(crate) fn certificate(&self) -> Result<Arc<Certificate>> {
        self.node.certificate.get_or_init(|| {
            #[cfg(test)]
            BIND_WORK.with(|c| c.set(c.get() + 1));
            let mut c = Certificate { instances: 1, ports: self.node.bindings.len() as u128, ..Certificate::default() };
            for p in self.node.prepared.values() {
                // A vocabulary-free retained body has no token bytes yet; its
                // vocabulary is checked in `validate_retained_vocab` at
                // compile/precompile time.
                if p.inner.pending_vocab_fingerprint.is_none() {
                    c.target(&constraint_vocab(&p.inner))?;
                }
            }
            if let Some(t) = &self.node.source.exact_value { c.tokens(t)?; }
            for value in self.node.bindings.values() {
                c.value(value)?;
                if let BoundValue::Module(m) = value {
                    let child = m.certificate()?;
                    c.instances = c.instances.saturating_add(child.instances);
                    c.ports = c.ports.saturating_add(child.ports);
                }
            }
            if c.height > graph::MAX_DEPTH { return Err(error("Module depth exceeds 64")); }
            Ok(Arc::new(c))
        }).clone()
    }

    fn check_limits(&self) -> Result<()> {
        let c = self.certificate()?;
        if c.height > graph::MAX_DEPTH { return Err(error("Module depth exceeds 64")); }
        if c.instances > graph::MAX_NODES as u128 || c.ports > 1_048_576 { self.index()?; }
        Ok(())
    }

    pub(crate) fn index(&self) -> Result<Vec<Module>> {
        fn visit(module: &Module, active: &mut BTreeSet<usize>, done: &mut BTreeSet<usize>, out: &mut Vec<Module>, depth: usize) -> Result<()> {
            if depth > graph::MAX_DEPTH { return Err(error("module attachment depth exceeds 64")); }
            #[cfg(test)]
            BIND_WORK.with(|c| c.set(c.get() + 1));
            let id = Arc::as_ptr(&module.node) as usize;
            if active.contains(&id) { return Err(error("cyclic module graph")); }
            if done.contains(&id) { return Ok(()); }
            active.insert(id);
            for value in module.node.bindings.values() {
                if let BoundValue::Module(child) = value { visit(child, active, done, out, depth + 1)?; }
            }
            active.remove(&id);
            done.insert(id);
            out.push(module.clone());
            if out.len() > graph::MAX_NODES { return Err(error("too many module instances")); }
            Ok(())
        }
        let mut out = Vec::new();
        visit(self, &mut BTreeSet::new(), &mut BTreeSet::new(), &mut out, 0)?;
        let mut summaries = HashMap::<usize, (usize, usize)>::new();
        let mut ports = 0usize;
        for module in &out {
            let mut height = 0;
            let mut expanded = 1usize;
            ports = ports.saturating_add(module.node.bindings.len());
            if ports > 1_048_576 { return Err(error("too many Module ports")); }
            for value in module.node.bindings.values() {
                if let BoundValue::Module(child) = value {
                    let (h, e) = summaries[&(Arc::as_ptr(&child.node) as usize)];
                    height = height.max(h + 1);
                    expanded = expanded.saturating_add(e);
                }
            }
            if height > graph::MAX_DEPTH { return Err(error("Module depth exceeds 64")); }
            summaries.insert(Arc::as_ptr(&module.node) as usize, (height, expanded));
        }
        Ok(out)
    }

    pub(crate) fn validate_targets(&self, supplied: Option<&Vocab>) -> Result<()> {
        let mut c = self.certificate()?.as_ref().clone();
        if let Some(v) = supplied { c.target(v)?; }
        Ok(())
    }

    /// Retained precompiled work is vocabulary-specific. Verify it against the
    /// vocabulary of the compile or precompile call that will use it.
    fn validate_retained_vocab(&self, vocab: &Vocab) -> Result<()> {
        let expected = crate::compiler::compile::vocab_fingerprint(vocab);
        for module in &self.index()? {
            for prepared in module.node.prepared.values() {
                let prepared = prepared.prepared();
                let recorded = prepared
                    .retained_vocab_fingerprint
                    .unwrap_or_else(|| prepared.inner.vocab_fingerprint());
                if recorded != expected {
                    return Err(error(
                        "module retained work was precompiled for a different vocabulary",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Compiles as much as possible for `vocab` now, so later compiles are fast.
    ///
    /// Externs may still be unbound. The work is kept in the module and included
    /// by [`save`](Self::save).
    pub fn precompile(&mut self, vocab: &Vocab) -> Result<()> {
        self.precompile_with(vocab, CompileOptions::default())
    }

    /// Like [`precompile`](Self::precompile), with `options`.
    pub fn precompile_with(&mut self, vocab: &Vocab, options: CompileOptions) -> Result<()> {
        with_deep_stack(|| self.precompile_on_stack(vocab, &options, true))
    }

    /// Rehydrates vocabulary-free retained work loaded from a Module artifact.
    ///
    /// Deferred bodies are decoded once here, against the vocabulary that
    /// `validate_retained_vocab` has already checked.
    fn rehydrate_retained(&self, vocab: &Vocab) -> Result<Module> {
        let modules = self.index()?;
        let mut instances = HashMap::<usize, Module>::new();
        for m in modules {
            let mut next = m.clone();
            let mut c = m.certificate()?.as_ref().clone();
            c.target(vocab)?;
            let node = Arc::make_mut(&mut next.node);
            for (&engine, prepared) in &m.node.prepared {
                node.prepared.insert(engine, prepared.rehydrate_vocab(vocab)?);
            }
            node.certificate = Arc::new(OnceLock::from(Ok(Arc::new(c))));
            for (name, value) in &m.node.bindings {
                if let BoundValue::Module(child) = value {
                    node.bindings.insert(
                        name.clone(),
                        BoundValue::Module(instances[&(Arc::as_ptr(&child.node) as usize)].clone()),
                    );
                }
            }
            instances.insert(Arc::as_ptr(&m.node) as usize, next);
        }
        instances
            .remove(&(Arc::as_ptr(&self.node) as usize))
            .ok_or_else(|| error("missing rehydrated Module root"))
    }

    /// Internal compile preparation: retain compilation artifacts and the
    /// default boundary token set.
    fn precompile_for_compile(&mut self, vocab: &Vocab, options: &CompileOptions) -> Result<()> {
        with_deep_stack(|| self.precompile_on_stack(vocab, options, true))
    }

    fn precompile_on_stack(&mut self, vocab: &Vocab, options: &CompileOptions, build_boundary: bool) -> Result<()> {
        self.validate_targets(Some(vocab))?;
        self.validate_retained_vocab(vocab)?;
        if self.index()?.iter().any(|module| module.node.prepared.values().any(|prepared| {
            prepared.prepared().retained_artifact.is_some()
        })) {
            *self = self.rehydrate_retained(vocab)?;
        }
        let modules = self.index()?;
        let requested = options.engine()?;
        let any_retained = modules.iter().any(|m| best_retained(&m.node.prepared).is_some());
        let target_b = target_boundary();
        let plan_of = |m: &Module| plan_node(requested, &m.node.prepared, any_retained, options.parser_backend);
        let backend_of = |plan: NodePlan| token_trie_backend(plan.backend);

        // Keyed by source identity: every node has exactly one plan.
        let mut locals: HashMap<usize, (Engine, Arc<dyn RetainedPreparation>)> = HashMap::new();

        for m in &modules {
            let plan = plan_of(m);
            // A pure LR TokenTrie precompile retains nothing.
            if plan.engine == Engine::TokenTrie && plan.backend == ParserBackend::LrTable {
                continue;
            }
            let source = Arc::as_ptr(&m.node.source.parsed_component) as usize;
            if let Some(p) = m.node.prepared.get(&plan.engine.key()) {
                // `boundary_trigger` records the requested detail; the trigger
                // itself is built lazily in `materialize`.
                if p.prepared().boundary_trigger >= target_b {
                    locals.insert(source, (plan.engine, p.clone()));
                }
            }
        }

        let mut seen = std::collections::HashSet::new();
        let jobs: Vec<&Module> = modules.iter().filter(|m| {
            let plan = plan_of(m);
            if plan.engine == Engine::TokenTrie && plan.backend == ParserBackend::LrTable {
                return false;
            }
            let source = Arc::as_ptr(&m.node.source.parsed_component) as usize;
            !locals.contains_key(&source) && seen.insert(source)
        }).collect();

        let results = jobs.par_iter().map(|m| {
            #[cfg(test)]
            let _probe = preparation_probe::enter(m);
            let plan = plan_of(m);
            let source = Arc::as_ptr(&m.node.source.parsed_component) as usize;
            let built = match m.prepare_local_for(vocab, plan.engine, target_b, build_boundary, backend_of(plan)) {
                Ok(p) => (plan.engine, p),
                Err(error) if requested == Engine::Auto
                    && plan.engine == Engine::ParserDwa
                    && any_retained
                    && error.is_strategy_decline() =>
                {
                    // Auto may fall back per part, but only when the graph keeps
                    // retained work that a whole-composition retry would discard.
                    let fallback = NodePlan { engine: Engine::TokenTrie, backend: ParserBackend::TemplateDfa };
                    let p = m.prepare_local_for(vocab, fallback.engine, target_b, build_boundary, backend_of(fallback))?;
                    (fallback.engine, p)
                }
                Err(error) => return Err(error),
            };
            Ok((source, built.0, retain_local(built.1)))
        }).collect::<Result<Vec<_>>>()?;

        for (source, engine, prepared) in results {
            locals.insert(source, (engine, prepared));
        }

        let mut instances = HashMap::<usize, Module>::new();
        for m in modules {
            let mut next = m.clone();
            let mut c = m.certificate()?.as_ref().clone();
            c.target(vocab)?;
            let source = Arc::as_ptr(&m.node.source.parsed_component) as usize;
            let node = Arc::make_mut(&mut next.node);
            if let Some((engine, prepared)) = locals.get(&source) {
                node.prepared.insert(engine.key(), prepared.clone());
            }
            node.certificate = Arc::new(OnceLock::from(Ok(Arc::new(c))));
            for (name, value) in &m.node.bindings {
                if let BoundValue::Module(child) = value {
                    node.bindings.insert(name.clone(), BoundValue::Module(instances[&(Arc::as_ptr(&child.node) as usize)].clone()));
                }
            }
            instances.insert(Arc::as_ptr(&m.node) as usize, next);
        }
        *self = instances.remove(&(Arc::as_ptr(&self.node) as usize)).unwrap();
        Ok(())
    }

    fn prepare_local_for(
        &self,
        vocab: &Vocab,
        engine: Engine,
        target_b: BoundaryTriggerDetail,
        build_boundary: bool,
        backend: crate::runtime::parser_backend::DynamicParserBackend,
    ) -> Result<PreparedGrammar> {
        if let Some(existing) = self.node.prepared.get(&engine.key()) {
            let mut upgraded = existing.prepared().clone();
            if build_boundary && upgraded.inner.boundary_trigger.detail() < target_b {
                Arc::make_mut(&mut upgraded.inner)
                    .build_boundary_trigger(target_b)
                    .map_err(Error::Compilation)?;
            }
            upgraded.boundary_trigger = target_b;
            return Ok(upgraded);
        }
        let source = &self.node.source;
        if matches!(source.source, GrammarSource::Lark(_)) {
            let mut resolved = source.resolve_lark(vocab)?;
            // Token externs (named Lark atoms) must be bound before the local
            // component is linked; grammar externs stay open for `materialize`.
            for (name, value) in &self.node.bindings {
                if resolved.bindings.contains_key(name) { continue; }
                match value {
                    BoundValue::TokenSet(t) => resolved = resolved.bind_value(name, GrammarValue::TokenSet(t.clone()))?,
                    BoundValue::Module(_) => {}
                }
            }
            let mut module = Module::new(&resolved);
            let mut instances = HashMap::<usize, Module>::new();
            for m in module.index()? {
                let mut next = m.clone();
                let p = m.prepare_local_for(vocab, engine, target_b, build_boundary, backend)?;
                let node = Arc::make_mut(&mut next.node);
                node.prepared.insert(engine.key(), retain_local(p));
                node.certificate = Arc::new(OnceLock::new());
                for (name, value) in &m.node.bindings {
                    if let BoundValue::Module(child) = value {
                        node.bindings.insert(name.clone(), BoundValue::Module(instances[&(Arc::as_ptr(&child.node) as usize)].clone()));
                    }
                }
                instances.insert(Arc::as_ptr(&m.node) as usize, next);
            }
            module = instances.remove(&(Arc::as_ptr(&module.node) as usize)).unwrap();
            let mut local = module.node.prepared.get(&engine.key()).ok_or_else(|| error("module is not prepared"))?.prepared().clone();
            for (name, value) in &module.node.bindings {
                match value {
                    BoundValue::Module(child) => {
                        let child_prep = child.node.prepared.get(&engine.key()).ok_or_else(|| error("child is not prepared"))?.prepared().clone();
                        local.bind(name, child_prep)?;
                    }
                    BoundValue::TokenSet(t) => local.bind(name, t.clone())?,
                }
            }
            #[cfg(test)]
            static_build_probe::record(self, engine);
            return Ok(local);
        }

        let declared = source.declarations()?;
        // A typed template decline must survive so the public boundary can
        // attempt the rescue; only real failures are rewrapped.
        let token_class_error = |e: Error| {
            if e.is_strategy_decline() {
                e
            } else {
                Error::Compilation(format!("cannot merge equivalent tokens for this grammar: {e}"))
            }
        };
        let mut inner = match engine {
            // Only a mixed compile prepares a TOKEN_TRIE module: a dynamic shell
            // on the table-free parser keeps its extern slots, so it composes
            // with retained static components.
            Engine::TokenTrie => crate::import::compile_component_shell_dynamic_with_backend(
                source.component()?,
                vocab,
                backend,
            )?,
            Engine::MergedTokenTrie => match source.source {
                GrammarSource::Glrm(_) | GrammarSource::Constructed | GrammarSource::Ebnf(_) => {
                    crate::import::compile_component_shell_balanced(source.component()?, vocab, &[])
                        .map_err(token_class_error)?
                }
                _ => {
                    let dynamic = match &source.source {
                        GrammarSource::Lark(s) => DynamicConstraint::from_lark_with_bounded_template_parser(s, vocab),
                        GrammarSource::JsonSchema(s) => DynamicConstraint::from_json_schema_with_bounded_template_parser(s, vocab),
                        _ => unreachable!(),
                    }.map_err(token_class_error)?;
                    let mut components = dynamic.into_constraints();
                    if components.len() != 1 {
                        return Err(Error::Compilation("cannot merge equivalent tokens for this grammar: requires one compiled component".into()));
                    }
                    components.pop().unwrap()
                }
            },
            Engine::Auto | Engine::ParserDwa => {
                #[cfg(test)]
                if self
                    .node
                    .force_parser_dwa_decline
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    return Err(Error::ResourceLimit(
                        "grammar_normalization build budget exhausted".into(),
                    ));
                }
                let mut p = source.prepare_unlinked(vocab, BuildOptions::default())?;
                let inner = Arc::make_mut(&mut p.inner);
                crate::compiler::boundary_candidates::persist_boundary_candidate_summary(inner, vocab);
                let _ = inner.late_bind_vocab.set(vocab.clone());
                inner.install_template_parser_from_compile()?;
                if build_boundary {
                    inner.build_boundary_trigger(target_b).map_err(Error::Compilation)?;
                }
                p.boundary_trigger = target_b;
                #[cfg(test)]
                static_build_probe::record(self, engine);
                return Ok(p);
            }
        };

        let _ = inner.late_bind_vocab.set(vocab.clone());
        if engine == Engine::MergedTokenTrie {
            inner.install_template_parser_from_compile()?;
        }
        if build_boundary {
            inner.build_boundary_trigger(target_b).map_err(Error::Compilation)?;
        }
        let prepared = PreparedGrammar {
            inner: Arc::new(inner),
            bindings: OrdMap::new(),
            analysis: graph::empty_cache(),
            event_adapters: Arc::new(event_cache::ExactAdapterCache::default()),
            boundary_trigger: target_b,
            slot_kinds: Arc::new(declared.kinds.clone()),
            slot_spans: Arc::new(declared.spans.clone()),
            retained_artifact: None,
            retained_vocab_fingerprint: None,
        };
        prepared.validate_local_slot_manifest(vocab)?;
        prepared.validate_slot_manifest()?;
        #[cfg(test)]
        static_build_probe::record(self, engine);
        Ok(prepared)
    }

    /// Recognizes the root shorthand `Sequence([body, end-token set])` and
    /// returns the body module with the sorted end-token IDs.
    fn end_token_root(&self) -> Result<Option<(Module, Vec<u32>)>> {
        self.end_token_root_at(0)
    }

    fn end_token_root_at(&self, depth: usize) -> Result<Option<(Module, Vec<u32>)>> {
        if depth > 8 {
            return Ok(None);
        }
        let Some(body) = self.node.source.single_rule_body() else {
            return Ok(None);
        };
        match body {
            crate::grammar::ast::GrammarExpr::Sequence(parts) if parts.len() == 2 => {
                let (crate::grammar::ast::GrammarExpr::Ref(a), crate::grammar::ast::GrammarExpr::Ref(b)) =
                    (&parts[0], &parts[1])
                else {
                    return Ok(None);
                };
                let (Some(BoundValue::Module(body)), Some(BoundValue::TokenSet(tokens))) =
                    (self.node.bindings.get(a), self.node.bindings.get(b))
                else {
                    return Ok(None);
                };
                let mut ids: Vec<u32> = tokens
                    .range_pairs()
                    .iter()
                    .flat_map(|&(start, end)| start..=end)
                    .collect();
                ids.sort_unstable();
                if ids.is_empty() {
                    return Ok(None);
                }
                Ok(Some((body.clone(), ids)))
            }
            crate::grammar::ast::GrammarExpr::Ref(name) => {
                match self.node.bindings.get(name) {
                    Some(BoundValue::Module(body)) => body.end_token_root_at(depth + 1),
                    _ => Ok(None),
                }
            }
            _ => Ok(None),
        }
    }

    /// Compiles the module for `vocab`, reusing any precompiled work.
    ///
    /// Every extern must be bound.
    pub fn compile(&self, vocab: &Vocab) -> Result<RuntimeConstraint> {
        self.compile_with(vocab, CompileOptions::default())
    }

    /// Like [`compile`](Self::compile), with `options`.
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<RuntimeConstraint> {
        with_deep_stack(|| self.compile_with_on_stack(vocab, options, &[]))
    }

    fn compile_with_on_stack(&self, vocab: &Vocab, options: CompileOptions, end_tokens: &[u32]) -> Result<RuntimeConstraint> {
        if let Some(slot) = self.externs().unbound().iter().next() {
            return Err(error(format!("extern {:?} is unbound", slot.name())));
        }
        if end_tokens.is_empty()
            && let Some((body, ids)) = self.end_token_root()?
        {
            return body.compile_with_on_stack(vocab, options, &ids);
        }
        self.validate_targets(Some(vocab))?;
        self.validate_retained_vocab(vocab)?;
        let modules = self.index()?;
        let mut expanded = HashMap::<usize, usize>::new();
        for module in &modules {
            let mut count = 1usize;
            for value in module.node.bindings.values() {
                if let BoundValue::Module(child) = value {
                    count = count.saturating_add(expanded[&(Arc::as_ptr(&child.node) as usize)]);
                }
            }
            expanded.insert(Arc::as_ptr(&module.node) as usize, count);
        }
        if expanded[&(Arc::as_ptr(&self.node) as usize)] > graph::MAX_NODES {
            return Err(error("expanded Module runtime construction exceeds its limit"));
        }
        // TOKEN_TRIE without retained static work compiles the whole
        // composition as one ordinary Dynamic grammar, inlined at source level.
        // Over retained static work it is mixed: retained parts stay static and
        // every other module is compiled as its own dynamic component.
        let requested = options.engine()?;
        let any_retained = modules.iter().any(|m| best_retained(&m.node.prepared).is_some());
        // The whole-composition flatten path is only valid when the request is
        // TokenTrie and no part owns precompiled machinery.
        let whole = requested == Engine::TokenTrie && !any_retained;
        // EXPERIMENT: one route report per top-level Module compile.
        let experiment = crate::compiler::terminal_ended_rescue::RescueExperiment::read();
        if whole {
            let result = self
                .flatten_source()?
                .compile_token_trie(vocab, BoundaryTriggerDetail::TokenSet, options.token_trie_backend())?;
            let result = result.with_end_tokens(end_tokens)?;
            crate::compiler::terminal_ended_rescue::report_route(
                experiment.report_route,
                crate::compiler::terminal_ended_rescue::Route::Normal,
                requested,
                experiment,
                &crate::compiler::terminal_ended_rescue::RescueOutcome::NotTried,
            );
            return Ok(result);
        }
        let te_first = experiment.terminal_ended_rescue
            && experiment.terminal_ended_first
            && self.node.bindings.is_empty()
            && matches!(
                requested,
                Engine::ParserDwa | Engine::MergedTokenTrie | Engine::Auto
            );
        let mut route = crate::compiler::terminal_ended_rescue::Route::Normal;
        let mut outcome = crate::compiler::terminal_ended_rescue::RescueOutcome::NotTried;
        let result = (|| -> Result<RuntimeConstraint> {
            let try_rescue = || -> Result<RuntimeConstraint> {
                let grammar = self.flatten_source()?;
                let source = grammar.rescue_source_grammar(vocab)?;
                let mut constraint =
                    crate::compiler::terminal_ended_rescue::rescue(&source, vocab)?;
                apply_final_boundary_trigger(&mut constraint, target_boundary())?;
                constraint.with_end_tokens(end_tokens)
            };
            if te_first {
                match try_rescue() {
                    Ok(constraint) => {
                        route = crate::compiler::terminal_ended_rescue::Route::TerminalEnded;
                        outcome = crate::compiler::terminal_ended_rescue::RescueOutcome::Ok;
                        return Ok(constraint);
                    }
                    Err(rescue_error) => {
                        outcome = crate::compiler::terminal_ended_rescue::RescueOutcome::Declined(
                            rescue_error.to_string(),
                        );
                    }
                }
            }
            let build = || {
                let mut prepared = self.clone();
                let budget = crate::compiler::build_budget::BuildBudgetScope::enter();
                let end_tokens = end_tokens.to_vec();
                let compile_options = options.clone();
                budget.install(move || {
                    crate::error::catch_template_attempt(move || -> Result<RuntimeConstraint> {
                        prepared.precompile_for_compile(vocab, &compile_options)?;
                        let constraint = prepared.materialize(
                            vocab,
                            &mut HashMap::new(),
                            compile_options.boundary_mask_strategy,
                        )?;
                        ensure_runnable_constraint(&constraint)?;
                        constraint.with_end_tokens(&end_tokens)
                    })
                })
            };
            let attempt = build();
            match attempt {
                Ok(constraint) => Ok(constraint),
                Err(error) if error.is_strategy_decline() => {
                    // Opt-in experiment: one terminal-ended rescue attempt from
                    // the original flattened source, restricted to plain single
                    // modules. Without the switch, the typed decline is returned
                    // unchanged.
                    if experiment.terminal_ended_rescue
                        && self.node.bindings.is_empty()
                        && !te_first
                    {
                        match try_rescue() {
                            Ok(constraint) => {
                                if crate::compiler::compile::compile_profile_enabled() {
                                    eprintln!(
                                        "[glrmask/profile][terminal_ended_rescue] strategy={requested:?} outcome=ok",
                                    );
                                }
                                route =
                                    crate::compiler::terminal_ended_rescue::Route::TerminalEnded;
                                outcome =
                                    crate::compiler::terminal_ended_rescue::RescueOutcome::Ok;
                                return Ok(constraint);
                            }
                            Err(rescue_error) => {
                                if crate::compiler::compile::compile_profile_enabled() {
                                    eprintln!(
                                        "[glrmask/profile][terminal_ended_rescue] strategy={requested:?} outcome=declined reason={rescue_error}",
                                    );
                                }
                                outcome =
                                    crate::compiler::terminal_ended_rescue::RescueOutcome::Declined(
                                        rescue_error.to_string(),
                                    );
                            }
                        }
                    }
                    if crate::compiler::compile::compile_profile_enabled() {
                        eprintln!(
                            "[glrmask/profile][template_decline] strategy={requested:?} stage={} reason={error}",
                            error.decline_stage(),
                        );
                    }
                    // Only Auto may pick a different exact strategy; an explicit
                    // strategy request is never silently replaced. With retained
                    // work the per-part fallback above already kept it; the
                    // whole-composition retry is only allowed when nothing is
                    // retained.
                    if requested == Engine::Auto && !any_retained {
                        // The fallback is exactly an explicit TokenTrie request
                        // with the same switches.
                        if let Ok(grammar) = self.flatten_source() {
                            let constraint = grammar
                                .compile_requested_token_trie(
                                    vocab,
                                    BoundaryTriggerDetail::TokenSet,
                                    options.parser_backend,
                                    options.merge_equivalent_tokens,
                                )?
                                .with_end_tokens(end_tokens)?;
                            route =
                                crate::compiler::terminal_ended_rescue::Route::TokenTrieFallback;
                            return Ok(constraint);
                        }
                    }
                    Err(error)
                }
                Err(error) => Err(error),
            }
        })();
        crate::compiler::terminal_ended_rescue::report_route(
            experiment.report_route,
            route,
            requested,
            experiment,
            &outcome,
        );
        result
    }

    fn materialize(
        &self,
        vocab: &Vocab,
        cache: &mut HashMap<usize, RuntimeConstraint>,
        boundary_strategy: BoundaryMaskStrategy,
    ) -> Result<RuntimeConstraint> {
        let id = Arc::as_ptr(&self.node) as usize;
        if let Some(c) = cache.get(&id) { return Ok(c.clone()); }
        let engine = best_retained(&self.node.prepared)
            .ok_or_else(|| error("missing local compilation"))?;
        let local = self.node.prepared.get(&engine.key()).ok_or_else(|| error("missing local compilation"))?;
        // Rebuilding the pre-composition parent is only needed when the rare
        // `Auto` boundary decline has to redo composition, so keep it in a
        // closure instead of snapshotting the composed constraint.
        let build_base = || -> Result<RuntimeConstraint> {
            let mut base = if local.bindings.is_empty() {
                local.inner.as_ref().clone()
            } else {
                local.link_with(engine.build_options())?
            };
            base.materialize_composition_link_metadata_for_compilation().map_err(Error::Compilation)?;
            Ok(base)
        };
        let mut parent = build_base()?;
        let active = parent.late_grammar_slots.iter().map(|s| s.name.as_str()).collect::<BTreeSet<_>>();
        let children = self.node.bindings.iter().filter(|(name, _)| active.contains(name.as_str())).map(|(name, value)| {
            let child = match value {
                BoundValue::Module(child) => child.materialize(vocab, cache, boundary_strategy)?,
                BoundValue::TokenSet(t) => self.event_adapter(t.range_pairs(), vocab, parent.has_template_parser(), engine)?,
            };
            Ok((name.clone(), Arc::new(child)))
        }).collect::<Result<Vec<_>>>()?;
        let linked = !children.is_empty();
        let all_static = parent.has_template_parser()
            && children.iter().all(|(_, child)| child.has_template_parser());
        let mut child_token_union = Vec::new();
        for (_, child) in &children {
            if let crate::runtime::BoundaryTrigger::TokenSet(child_tokens) = &child.boundary_trigger {
                child_token_union.extend_from_slice(child_tokens);
            }
        }
        if linked {
            // Composition consumes the pre-composition parent. Factor it into
            // a closure so the rare `Auto` decline can redo composition from
            // the same inputs without a full `Constraint` snapshot per build.
            let compose = |parent: RuntimeConstraint| -> Result<RuntimeConstraint> {
                let composed = if all_static {
                    crate::error::catch_internal_invariant(|| {
                        crate::runtime::parser_backend::link::compose_owned(parent, children.clone(), vocab)
                    })??
                } else {
                    crate::error::catch_internal_invariant(|| {
                        compose_named_children(parent, &children, vocab, SegmentedBoundaryBackend::Dynamic)
                    })??
                };
                Ok(composed)
            };
            parent = compose(parent)?;
            // All-template compositions may still install the static boundary
            // parser; retained parts keep their own machinery.
            if boundary_strategy != BoundaryMaskStrategy::TokenTrie
                && all_static
                && matches!(engine, Engine::ParserDwa | Engine::Auto)
            {
                // `Auto` mirrors `MaskStrategy::Auto`: prefer the static
                // boundary parser, and fall back to the token-trie boundary
                // when it declines on budget instead of failing the build.
                let declined = {
                    #[cfg(test)]
                    let forced_decline = self
                        .node
                        .force_boundary_dwa_decline
                        .load(std::sync::atomic::Ordering::Relaxed);
                    #[cfg(not(test))]
                    let forced_decline = false;
                    if forced_decline {
                        true
                    } else {
                        match crate::compiler::template_boundary::install(&mut parent, vocab) {
                            Ok(()) => false,
                            Err(error) if crate::compiler::template_boundary::is_template_budget_decline(&error) => true,
                            Err(error) => return Err(error),
                        }
                    }
                };
                if declined && boundary_strategy == BoundaryMaskStrategy::ParserDwa {
                    return Err(Error::ResourceLimit(
                        crate::error::BOUNDARY_DWA_DECLINE.into(),
                    ));
                }
                if declined {
                    parent = compose(build_base()?)?;
                }
            }
            let target_b = target_boundary();
            apply_final_boundary_trigger(&mut parent, target_b)?;
            // A composed trigger is the union of the parent's own set and the
            // precomputed sets of its children. Sorted merge keeps this cheap.
            let mut tokens = parent
                .boundary_trigger
                .token_summary()
                .map(|tokens| tokens.to_vec())
                .unwrap_or_else(|| child_token_union.clone());
            tokens.extend_from_slice(&child_token_union);
            tokens.sort_unstable();
            tokens.dedup();
            parent.boundary_trigger = crate::runtime::BoundaryTrigger::TokenSet(Arc::from(
                tokens.into_boxed_slice(),
            ));
        }
        cache.insert(id, parent.clone());
        Ok(parent)
    }

    fn event_adapter(&self, ranges: &[(u32, u32)], vocab: &Vocab, native: bool, engine: Engine) -> Result<RuntimeConstraint> {
        let local = self.node.prepared.get(&engine.key()).ok_or_else(|| error("missing prepared event owner"))?;
        let leaf = local.event_adapters.leaf(ranges)?;
        Ok(leaf.compiled_for_representation(vocab, native)?.as_ref().clone())
    }

    pub(super) fn to_source(&self) -> Result<Grammar> {
        if !self.node.prepared.is_empty() {
            return Err(error("expressions with precompiled operands cannot be converted to source Grammar; bind directly to Module instead"));
        }
        self.flatten_source()
    }

    // The whole composition as one source Grammar, with every bound child
    // inlined as a scoped source body. Retained work and settings are ignored.
    fn flatten_source(&self) -> Result<Grammar> {
        let mut grammar = self.node.source.as_ref().clone();
        for (name, value) in &self.node.bindings {
            let value = match value {
                BoundValue::Module(child) => GrammarValue::Source(Arc::new(child.flatten_source()?)),
                BoundValue::TokenSet(t) => GrammarValue::TokenSet(t.clone()),
            };
            grammar = grammar.bind_value(name, value)?;
        }
        Ok(grammar)
    }

    /// Saves the module, including precompiled work, as bytes.
    ///
    /// The vocabulary is not included. Bytes saved by one GLRMask version may not
    /// load in another.
    pub fn save(&self) -> Vec<u8> { with_deep_stack(|| artifact::save(self)) }
    /// Loads a module saved with [`save`](Self::save).
    ///
    /// Precompiled work applies only to the vocabulary it was built for;
    /// compiling with a different one is an error.
    pub fn load<'b>(bytes: impl Into<std::borrow::Cow<'b, [u8]>>) -> Result<Self> {
        with_deep_stack(|| artifact::load(bytes.into().as_ref(), None))
    }
}

#[cfg(test)]
pub(crate) mod static_build_probe {
    use super::*;
    use std::sync::Mutex;
    // Each count keeps a liveness check for its grammar, because a dropped
    // grammar's address can be reused by a grammar in a parallel test.
    type Key = (usize, u8);
    type Entry = (Box<dyn Fn() -> bool + Send>, usize);
    fn counts() -> &'static Mutex<HashMap<Key, Entry>> {
        static COUNTS: OnceLock<Mutex<HashMap<Key, Entry>>> = OnceLock::new();
        COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
    }
    fn key(m: &Module) -> usize { Arc::as_ptr(&m.node.source.parsed_component) as usize }
    pub(crate) fn record(m: &Module, engine: Engine) {
        let mut counts = counts().lock().unwrap();
        let entry = counts.entry((key(m), engine.key())).or_insert_with(|| (Box::new(|| false), 0));
        if !(entry.0)() {
            let weak = Arc::downgrade(&m.node.source.parsed_component);
            *entry = (Box::new(move || weak.strong_count() > 0), 0);
        }
        entry.1 += 1;
    }
    /// Static preparations (ParserDwa and merged) built so far for this
    /// module's grammar.
    pub(crate) fn builds(m: &Module) -> usize {
        let source = key(m);
        counts().lock().unwrap().iter().filter(|((candidate, engine), entry)| {
            *candidate == source && (*engine == 1 || *engine == 3) && (entry.0)()
        }).map(|(_, entry)| entry.1).sum()
    }
    /// Preparations built for one exact engine.
    pub(crate) fn builds_for(m: &Module, engine: Engine) -> usize {
        counts().lock().unwrap().get(&(key(m), engine.key())).filter(|e| (e.0)()).map_or(0, |e| e.1)
    }
}

/// Records mixed-compile fallbacks on the calling test thread.
///
/// `build(mixed)` is matched on the caller, so a thread-local log attributes a
/// fallback to the compile that triggered it even when test threads run in
/// parallel.
#[cfg(test)]
pub(crate) mod mixed_compile_probe {
    use super::*;
    thread_local! {
        static FALLBACKS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    pub(crate) fn record(error: &Error) {
        FALLBACKS.with(|log| log.borrow_mut().push(error.to_string()));
    }
    /// Drains the fallback reasons recorded since the last call.
    pub(crate) fn take() -> Vec<String> {
        FALLBACKS.with(|log| std::mem::take(&mut *log.borrow_mut()))
    }
}

#[cfg(test)]
mod preparation_probe {
    use super::*;
    use std::sync::{Mutex, Condvar};
    use std::sync::atomic::{AtomicUsize, Ordering};
    pub(super) struct Probe {
        pub active: AtomicUsize,
        pub max: AtomicUsize,
        pub tasks: AtomicUsize,
        pub workers: usize,
        pub arrivals: Mutex<usize>,
        pub ready: Condvar,
        pub thread_ids: Mutex<BTreeSet<usize>>,
        pub pool_sizes: Mutex<BTreeSet<usize>>,
    }
    pub(super) fn registry() -> &'static Mutex<HashMap<usize, Arc<Probe>>> {
        static REGISTRY: OnceLock<Mutex<HashMap<usize, Arc<Probe>>>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
    }
    pub(super) struct Guard(Option<Arc<Probe>>);
    pub(super) fn enter(m: &Module) -> Guard {
        let probe = registry().lock().unwrap().get(&(Arc::as_ptr(&m.node.source.parsed_component) as usize)).cloned();
        if let Some(p) = &probe {
            p.tasks.fetch_add(1, Ordering::SeqCst);
            let active = p.active.fetch_add(1, Ordering::SeqCst) + 1;
            p.max.fetch_max(active, Ordering::SeqCst);
            p.thread_ids.lock().unwrap().insert(rayon::current_thread_index().unwrap());
            p.pool_sizes.lock().unwrap().insert(rayon::current_num_threads());
            let mut arrivals = p.arrivals.lock().unwrap();
            *arrivals += 1;
            p.ready.notify_all();
            let (_arrivals, timeout) = p.ready.wait_timeout_while(arrivals, std::time::Duration::from_secs(10), |n| *n < p.workers).unwrap();
            assert!(!timeout.timed_out(), "independent definition tasks failed to enter concurrently");
        }
        Guard(probe)
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            if let Some(p) = &self.0 {
                p.active.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }
}

#[cfg(test)]
mod precompile_boundary_tests {
    use super::*;
    use crate::__private::VocabExt as _;

    fn vocab() -> Vocab {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"xy".to_vec()),
        ]);
        vocab.prepare_for_compile();
        vocab
    }

    fn tokens(constraint: &RuntimeConstraint) -> Option<Arc<[u32]>> {
        match &constraint.boundary_trigger {
            crate::runtime::BoundaryTrigger::TokenSet(tokens) => Some(Arc::clone(tokens)),
            _ => None,
        }
    }

    #[test]
    fn precompile_builds_tokens_and_compile_reuses_the_same_arc() {
        let vocab = vocab();
        let grammar = Grammar::from_ebnf(r#"start ::= "x" "y" | "xy""#).unwrap();

        let mut module = Module::new(&grammar);
        module.precompile(&vocab).expect("precompile");
        let prepared_trigger = module
            .node
            .prepared
            .values()
            .next()
            .and_then(|prepared| tokens(&prepared.prepared().inner))
            .expect("precompiled Module must carry a Tokens trigger");
        let compiled = module
            .compile_with(&vocab, CompileOptions::default())
            .expect("compile after precompile");
        let compiled_trigger = tokens(&compiled).expect("compiled Module must carry Tokens");
        assert!(
            Arc::ptr_eq(&prepared_trigger, &compiled_trigger),
            "compile_with must reuse the precompiled Tokens Arc"
        );
        assert_eq!(prepared_trigger.as_ref(), compiled_trigger.as_ref());

        let plain = Module::new(&grammar)
            .compile_with(&vocab, CompileOptions::default())
            .expect("plain compile");
        assert!(tokens(&plain).is_some(), "plain compile must build the default trigger");

        let bytes = module.save();
        let loaded = Module::load(&bytes).expect("Module artifact round-trip");
        let loaded_trigger = loaded
            .node
            .prepared
            .values()
            .next()
            .and_then(|prepared| tokens(&prepared.prepared().inner))
            .expect("loaded Module must keep its TokenSet");
        assert_eq!(loaded_trigger.as_ref(), prepared_trigger.as_ref());

        let constraint_bytes = compiled.save();
        let mut loaded_constraint = RuntimeConstraint::load(&constraint_bytes, &vocab).expect("Constraint round-trip");
        loaded_constraint
            .materialize_composition_metadata_for_compilation()
            .expect("materialize loaded metadata");
        let loaded_constraint_trigger = tokens(&loaded_constraint).expect("loaded Constraint must keep its TokenSet");
        assert_eq!(loaded_constraint_trigger.as_ref(), prepared_trigger.as_ref());

        let mut old = b"GLRASM01".to_vec();
        old.extend_from_slice(&bytes[8..]);
        let error = Module::load(&old).expect_err("old Module artifact must be rejected");
        assert!(error.to_string().contains("unsupported Module artifact version"), "{error}");
    }

    #[test]
    fn compile_after_precompile_never_downgrades_static_parts() {
        let vocab = vocab();
        let grammar = Grammar::from_ebnf(r#"start ::= "x" "y" | "xy""#).unwrap();
        let mut module = Module::new(&grammar);
        module
            .precompile_with(
                &vocab,
                CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa),
            )
            .expect("precompile with ParserDwa");
        let prepared_trigger = module
            .node
            .prepared
            .get(&Engine::ParserDwa.key())
            .and_then(|prepared| tokens(&prepared.prepared().inner))
            .expect("ParserDwa preparation must carry a Tokens trigger");

        let compiled = module
            .compile_with(
                &vocab,
                CompileOptions::default().mask_strategy(MaskStrategy::TokenTrie),
            )
            .expect("TokenTrie compile after ParserDwa precompile");
        assert!(
            !compiled.uses_dynamic_runtime(),
            "a TokenTrie request must not replace retained static parts with dynamic compilation"
        );
        let compiled_trigger = tokens(&compiled).expect("compiled Module must carry Tokens");
        assert!(
            Arc::ptr_eq(&prepared_trigger, &compiled_trigger),
            "compile must reuse the precompiled Tokens Arc"
        );
        assert!(
            module.node.prepared.contains_key(&Engine::ParserDwa.key()),
            "compile must not discard the retained static part"
        );

        let expected = Module::new(&grammar)
            .compile_with(&vocab, CompileOptions::default())
            .expect("plain ParserDwa compile");
        fn compare(a: &crate::ConstraintState<'_>, b: &crate::ConstraintState<'_>, depth: usize) {
            assert_eq!(a.mask(), b.mask(), "mask mismatch at depth {depth}");
            assert_eq!(a.is_accepting(), b.is_accepting(), "acceptance mismatch at depth {depth}");
            if depth == 0 { return; }
            for token in a.mask() {
                let mut next_a = a.clone();
                if !next_a.commit_token(token) { continue; }
                let mut next_b = b.clone();
                assert!(next_b.commit_token(token), "same masks accept the same token");
                compare(&next_a, &next_b, depth - 1);
            }
        }
        compare(&compiled.start(), &expected.start(), 3);
    }

    fn same_language(a: &crate::ConstraintState<'_>, b: &crate::ConstraintState<'_>, depth: usize) {
        let mask = a.mask();
        assert_eq!(mask, b.mask(), "mask mismatch at depth {depth}");
        assert_eq!(a.is_accepting(), b.is_accepting(), "acceptance mismatch at depth {depth}");
        if depth == 0 { return; }
        let tokens = mask.iter().enumerate().flat_map(|(word, bits)| {
            (0..32).filter(move |bit| bits & (1 << bit) != 0).map(move |bit| (word * 32 + bit) as u32)
        });
        for token in tokens {
            let mut next_a = a.clone();
            assert!(next_a.commit_token(token), "masked token commits");
            let mut next_b = b.clone();
            assert!(next_b.commit_token(token), "same masks accept the same token");
            same_language(&next_a, &next_b, depth - 1);
        }
    }

    const PARENT: &str = r#"glrm 1; start start; extern grammar child; nt start = "x" child | "y" child "x";"#;
    const CHILD: &str = r#"start ::= "y" | "x" "y" | "xy""#;

    fn assembly(precompile_parent: bool, precompile_child: bool) -> (Module, Module, Module) {
        let vocab = vocab();
        let static_options = CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa);
        let mut child = Module::new(&Grammar::from_ebnf(CHILD).unwrap());
        if precompile_child {
            child.precompile_with(&vocab, static_options.clone()).expect("precompile child");
        }
        let mut parent = Module::new(&Grammar::from_glrm(PARENT).unwrap());
        if precompile_parent {
            parent.precompile_with(&vocab, static_options).expect("precompile unbound parent");
        }
        let mut assembly = parent.clone();
        assembly.bind("child", child.clone()).expect("bind child");
        if precompile_parent { (assembly, child, parent) } else { (assembly, parent, child) }
    }

    fn reference(strategy: MaskMode) -> RuntimeConstraint {
        assembly(false, false)
            .0
            .compile_with(&vocab(), strategy.compile_options())
            .expect("reference compile")
    }

    #[test]
    fn token_trie_compile_keeps_a_precompiled_child_static_and_the_parent_dynamic() {
        let vocab = vocab();
        let (module, dynamic_part, static_part) = assembly(false, true);
        let before = static_build_probe::builds(&dynamic_part);
        let static_before = static_build_probe::builds(&static_part);
        let mixed = module
            .compile_with(&vocab, CompileOptions::default().mask_strategy(MaskStrategy::TokenTrie))
            .expect("mixed compile");
        assert_eq!(static_build_probe::builds(&dynamic_part), before, "the non-precompiled part must not be built statically");
        assert_eq!(static_build_probe::builds(&static_part), static_before, "the precompiled part must be reused, not rebuilt");
        same_language(&mixed.start(), &reference(MaskMode::ParserDwa).start(), 4);
        same_language(&mixed.start(), &reference(MaskMode::A).start(), 4);
    }

    #[test]
    fn token_trie_compile_keeps_a_precompiled_parent_static_and_the_child_dynamic() {
        let vocab = vocab();
        let (module, dynamic_part, static_part) = assembly(true, false);
        let before = static_build_probe::builds(&dynamic_part);
        let static_before = static_build_probe::builds(&static_part);
        let mixed = module
            .compile_with(&vocab, CompileOptions::default().mask_strategy(MaskStrategy::TokenTrie))
            .expect("mixed compile");
        assert_eq!(static_build_probe::builds(&dynamic_part), before, "the non-precompiled part must not be built statically");
        assert_eq!(static_build_probe::builds(&static_part), static_before, "the precompiled part must be reused, not rebuilt");
        same_language(&mixed.start(), &reference(MaskMode::ParserDwa).start(), 4);
        same_language(&mixed.start(), &reference(MaskMode::A).start(), 4);
    }
}

#[cfg(test)]
mod b_shell_tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"a".to_vec()),
            (2, b"b".to_vec()),
            (3, b"y".to_vec()),
            (4, b"xay".to_vec()),
            (5, b"ab".to_vec()),
        ])
    }

    #[test]
    fn b_shell_survives_save_and_is_reused_by_an_lr_request() {
        let vocab = vocab();
        let grammar = Grammar::from_ebnf(r#"start ::= "x" ("a" | "b") "y" | "xay""#).unwrap();
        let mut module = Module::new(&grammar);
        module
            .precompile_with(&vocab, MaskMode::B.compile_options())
            .expect("precompile the B shell");
        assert!(module.node.prepared.contains_key(&Engine::TokenTrie.key()));

        let loaded = Module::load(module.save()).expect("load the B shell");
        assert!(
            loaded.node.prepared.contains_key(&Engine::TokenTrie.key()),
            "the B shell must survive Module save/load",
        );
        let before = static_build_probe::builds_for(&loaded, Engine::TokenTrie);
        let compiled = loaded
            .compile_with(&vocab, MaskMode::A.compile_options())
            .expect("compile an LR request over a B shell");
        assert!(compiled.uses_dynamic_runtime());
        assert_eq!(
            static_build_probe::builds_for(&loaded, Engine::TokenTrie),
            before,
            "the LR request rebuilt the retained B shell",
        );
        let reference = loaded
            .compile_with(&vocab, MaskMode::ParserDwa.compile_options())
            .expect("ParserDwa reference");
        let mut left = reference.start();
        let mut right = compiled.start();
        assert_eq!(left.mask(), right.mask());
        for token in [0u32, 1, 3, 0] {
            assert_eq!(left.commit_token(token), right.commit_token(token));
            assert_eq!(left.mask(), right.mask());
            assert_eq!(left.is_accepting(), right.is_accepting());
        }
    }
}

#[cfg(test)]
mod grammar_precompile_tests {
    use super::*;

    fn vocab() -> Vocab {
        Vocab::new(vec![
            (0, b"x".to_vec()), (1, b"a".to_vec()), (2, b"b".to_vec()),
            (3, b"y".to_vec()), (4, b"xay".to_vec()), (5, b"ab".to_vec()),
        ])
    }

    const HOST: &str = "glrm 1; start start; extern grammar child; nt start = \"x\" child \"y\";";

    #[test]
    fn grammar_precompile_returns_a_module_that_retains_its_work() {
        let vocab = vocab();
        let options = CompileOptions::default().mask_strategy(MaskStrategy::ParserDwa);
        let child = Grammar::from_ebnf(r#"start ::= "a""#).unwrap();

        let module = child.precompile_with(&vocab, options).expect("grammar precompile");
        let builds = static_build_probe::builds(&module);
        assert_eq!(builds, 1, "precompiling the grammar must build and retain one static preparation");

        let compiled = module
            .compile_with(&vocab, CompileOptions::default().mask_strategy(MaskStrategy::TokenTrie))
            .expect("TokenTrie compile after a ParserDwa grammar precompile");
        assert!(!compiled.uses_dynamic_runtime(), "the returned Module must keep its retained static work");
        assert_eq!(static_build_probe::builds(&module), builds, "the static preparation must be reused, not rebuilt");
        assert!(module.node.prepared.contains_key(&Engine::ParserDwa.key()));

        // Composing through the returned Module still matches the explicit
        // Module::new + bind + precompile path.
        let mut parent = Grammar::from_glrm(HOST).unwrap().precompile(&vocab).expect("open parent precompile");
        parent.bind("child", module).expect("bind child");
        let composed = parent.compile(&vocab).expect("compile composed modules");
        let mut state = composed.start();
        assert!(state.commit_token(4));
        assert!(state.is_accepting());

        let mut reference_parent = Module::new(&Grammar::from_glrm(HOST).unwrap());
        reference_parent.bind("child", Module::new(&child)).expect("bind reference child");
        reference_parent.precompile(&vocab).expect("precompile reference parent");
        let reference = reference_parent.compile(&vocab).expect("compile reference");
        assert_eq!(composed.start().mask(), reference.start().mask());
        assert_eq!(composed.start().is_accepting(), reference.start().is_accepting());
    }
}

#[cfg(test)]
mod boundary_gate_tests {
    use super::*;
    use crate::__private::VocabExt as _;

    fn vocab() -> Vocab {
        let vocab = Vocab::new(
            [b"x".as_slice(), b"y", b"z", b"x!", b"ab!", b"a", b"b", b" "]
                .into_iter()
                .enumerate()
                .map(|(id, bytes)| (id as u32, bytes.to_vec()))
                .collect(),
        );
        vocab.prepare_for_compile();
        vocab
    }

    fn compile_linked(parent: &str, child: Module, vocab: &Vocab) -> RuntimeConstraint {
        let mut module = Module::new(&Grammar::from_glrm(parent).unwrap());
        module.bind("child", child).expect("bind child");
        module
            .compile_with(vocab, CompileOptions::default())
            .expect("compile linked")
    }

    fn walk(
        gated: &mut crate::ConstraintState<'_>,
        ungated: &mut crate::ConstraintState<'_>,
        depth: usize,
    ) {
        assert_eq!(gated.mask(), ungated.mask(), "mask mismatch at depth {depth}");
        if depth == 0 {
            return;
        }
        let allowed = gated.mask();
        for token in allowed {
            if ungated.mask().binary_search(&token).is_err() {
                continue;
            }
            let mut next_gated = gated.clone();
            let mut next_ungated = ungated.clone();
            if !next_gated.commit_token(token) {
                continue;
            }
            assert!(next_ungated
                .commit_token(token), "ungated commit must accept a gated token");
            walk(&mut next_gated, &mut next_ungated, depth - 1);
        }
    }

    fn assert_gate_agrees(composed: &RuntimeConstraint, label: &str) {
        assert!(
            matches!(composed.boundary_trigger, crate::runtime::BoundaryTrigger::TokenSet(_)),
            "{label}: expected a TokenSet gate"
        );
        let mut ungated = composed.clone();
        ungated.boundary_trigger = crate::runtime::BoundaryTrigger::None;
        let mut gated_state = composed.start();
        let mut ungated_state = ungated.start();
        walk(&mut gated_state, &mut ungated_state, 4);
    }

    /// There is currently no runtime mask path that reads
    /// `BoundaryTrigger::TokenSet`; it is retained composition metadata today.
    /// This guard pins that future consumers never change observable masks.
    #[test]
    fn token_set_gate_never_changes_masks() {
        let vocab = vocab();
        let child = Module::new(&Grammar::from_lark("start: \"y\"?").unwrap());
        let simple = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            child.clone(),
            &vocab,
        );
        assert_gate_agrees(&simple, "nullable child");

        let postamble = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = "x" child "y";"#,
            child.clone(),
            &vocab,
        );
        assert_gate_agrees(&postamble, "parent postamble");

        let mid_token = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = child;"#,
            Module::new(&Grammar::from_ebnf(r#"start ::= "x""#).unwrap()),
            &vocab,
        );
        assert_gate_agrees(&mid_token, "child ends mid-token");

        let after_prefix = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = "a" child;"#,
            Module::new(&Grammar::from_ebnf(r#"start ::= "b""#).unwrap()),
            &vocab,
        );
        assert_gate_agrees(&after_prefix, "bound slot after prefix");

        let recursive = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = list; nt list = "x" child list | "z";"#,
            child.clone(),
            &vocab,
        );
        assert_gate_agrees(&recursive, "recursive parent");

        let nested_child = {
            let mut middle = Module::new(&Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            ).unwrap());
            middle.bind("child", child).unwrap();
            middle
        };
        let nested = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            nested_child,
            &vocab,
        );
        assert_gate_agrees(&nested, "two-level nested");

        let ignored = Grammar::from_lark("start: \"x\" \"y\"\n%ignore WS\nWS: /[ \\t]+/").unwrap()
            .compile_with_build_options(
                &vocab,
                BuildOptions::default().optimization(MaskStrategy::ParserDwa),
            )
            .expect("ignore grammar");
        assert_gate_agrees(&ignored, "ignore/whitespace");
    }
}

#[cfg(test)]
mod boundary_strategy_tests {
    use super::*;
    use crate::__private::VocabExt as _;

    fn vocab() -> Vocab {
        let vocab = Vocab::new(vec![
            (0, b"x".to_vec()),
            (1, b"y".to_vec()),
            (2, b"z".to_vec()),
            (3, b"xy".to_vec()),
        ]);
        vocab.prepare_for_compile();
        vocab
    }

    #[test]
    fn token_trie_and_parser_dwa_boundaries_produce_identical_masks() {
        let vocab = vocab();
        let child = Module::new(&Grammar::from_ebnf(r#"start ::= "y" "z""#).unwrap());
        let compile = |strategy| {
            let mut parent = Module::new(&Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            ).unwrap());
            parent.bind("child", child.clone()).unwrap();
            parent
                .compile_with(
                    &vocab,
                    CompileOptions::default().boundary_mask_strategy(strategy),
                )
                .unwrap()
        };
        let static_boundary = compile(BoundaryMaskStrategy::ParserDwa);
        let token_trie = compile(BoundaryMaskStrategy::TokenTrie);
        for prefix in [&[0u32][..0], &[0][..], &[0, 1][..]] {
            let mut a = static_boundary.start();
            let mut b = token_trie.start();
            for &token in prefix {
                assert!(a.commit_token(token));
                assert!(b.commit_token(token));
            }
            assert_eq!(a.mask(), b.mask(), "mask mismatch after {prefix:?}");
        }
    }
}

#[cfg(test)]
mod boundary_error_tests {
    use super::*;
    use crate::__private::VocabExt as _;

    #[test]
    fn parser_dwa_budget_decline_is_typed_and_token_trie_succeeds() {
        let vocab = {
            let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
            vocab.prepare_for_compile();
            vocab
        };
        let compile = |strategy| {
            let mut parent = Module::new(&Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            ).unwrap());
            parent
                .bind("child", Module::new(&Grammar::from_ebnf(r#"start ::= "y""#).unwrap()))
                .unwrap();
            parent.force_boundary_dwa_decline();
            parent.compile_with(
                &vocab,
                CompileOptions::default().boundary_mask_strategy(strategy),
            )
        };
        let error = compile(BoundaryMaskStrategy::ParserDwa).unwrap_err();
        match error.kind() {
            crate::ErrorKind::ResourceLimit(message) => {
                assert_eq!(message, crate::error::BOUNDARY_DWA_DECLINE);
                assert!(message.contains("TokenTrie"));
            }
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        compile(BoundaryMaskStrategy::TokenTrie).expect("TokenTrie must bypass the static boundary parser");
    }

    #[test]
    fn auto_falls_back_to_token_trie_where_parser_dwa_errors() {
        let vocab = {
            let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
            vocab.prepare_for_compile();
            vocab
        };
        let compile = |strategy| {
            let mut parent = Module::new(&Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            ).unwrap());
            parent
                .bind("child", Module::new(&Grammar::from_ebnf(r#"start ::= "y""#).unwrap()))
                .unwrap();
            parent.force_boundary_dwa_decline();
            parent.compile_with(
                &vocab,
                CompileOptions::default().boundary_mask_strategy(strategy),
            )
        };
        compile(BoundaryMaskStrategy::ParserDwa).expect_err("explicit ParserDwa must report the budget decline");
        let auto = compile(BoundaryMaskStrategy::Auto).expect("Auto must fall back to the token-trie boundary");
        let token_trie = compile(BoundaryMaskStrategy::TokenTrie).expect("TokenTrie must build without the static boundary");
        let mut auto_state = auto.start();
        let mut trie_state = token_trie.start();
        assert_eq!(auto_state.mask(), trie_state.mask());
        assert!(auto_state.commit_token(0));
        assert!(trie_state.commit_token(0));
        assert_eq!(auto_state.mask(), trie_state.mask());
    }
}
