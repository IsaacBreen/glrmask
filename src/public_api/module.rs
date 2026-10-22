//! Owned assembly snapshots. Local compilation is independent of occurrence bindings.
use super::*;
use super::expr::check_expr_unbound;
use std::collections::{HashMap, BTreeMap, BTreeSet};
mod artifact;
#[cfg(test)]
thread_local! { static BIND_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }


/// Root generation policy, applied only to the top-level sequence.
///
/// Never inherited by embedded modules.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompileOptions {
    pub(crate) end_tokens: Vec<u32>,
}

impl CompileOptions {
    /// Sets the accepted end-of-sequence token IDs.
    ///
    /// They are allowed only at acceptance and must be committed before stopping.
    pub fn end_tokens(mut self, ids: impl IntoIterator<Item = u32>) -> Self {
        self.end_tokens = ids.into_iter().collect();
        self
    }
}

/// A typed value that can be bound to an unresolved module extern.
pub struct Binding(pub(crate) BindingKind);

#[derive(Debug, Clone)]
pub(crate) enum BindingKind {
    Module(Module),
    Expr(Expr),
    Token(Token),
    ExactTokens(ExactTokens),
}

impl From<Module> for Binding { fn from(m: Module) -> Self { Self(BindingKind::Module(m)) } }
impl From<&Module> for Binding { fn from(m: &Module) -> Self { Self(BindingKind::Module(m.clone())) } }
impl From<Grammar<'_>> for Binding { fn from(g: Grammar<'_>) -> Self { Self(BindingKind::Module(Module::new(&g))) } }
impl From<&Grammar<'_>> for Binding { fn from(g: &Grammar<'_>) -> Self { Self(BindingKind::Module(Module::new(g))) } }
impl From<Expr> for Binding { fn from(e: Expr) -> Self { Self(BindingKind::Expr(e)) } }
impl From<&Expr> for Binding { fn from(e: &Expr) -> Self { Self(BindingKind::Expr(e.clone())) } }
impl From<Token> for Binding { fn from(t: Token) -> Self { Self(BindingKind::Token(t)) } }
impl From<&Token> for Binding { fn from(t: &Token) -> Self { Self(BindingKind::Token(*t)) } }
impl From<ExactTokens> for Binding { fn from(t: ExactTokens) -> Self { Self(BindingKind::ExactTokens(t)) } }
impl From<&ExactTokens> for Binding { fn from(t: &ExactTokens) -> Self { Self(BindingKind::ExactTokens(t.clone())) } }

#[derive(Debug, Clone)]
pub(crate) enum BoundValue {
    Module(Module),
    Token(Token),
    ExactTokens(ExactTokens),
}

/// A reusable, independently configured grammar with bindings and any retained
/// compilation work.
///
/// May still have unresolved externs.
#[derive(Debug, Clone)]
pub struct Module {
    pub(crate) node: Arc<Node>,
}

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) source: Arc<Grammar<'static>>,
    pub(crate) mask_strategy: MaskStrategy,
    pub(crate) boundary_mask_strategy: BoundaryMaskStrategy,
    pub(crate) prepared: BTreeMap<MaskStrategy, Arc<dyn RetainedPreparation>>,
    pub(crate) bindings: OrdMap<String, BoundValue>,
    pub(crate) certificate: Arc<OnceLock<Result<Arc<Certificate>>>>,
    #[cfg(test)]
    pub(crate) force_boundary_dwa_decline: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

pub(crate) trait RetainedPreparation: Send + Sync + std::fmt::Debug {
    fn prepared(&self) -> &PreparedGrammar;
}
impl RetainedPreparation for PreparedGrammar {
    fn prepared(&self) -> &PreparedGrammar { self }
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
    pending_tokens: im::OrdSet<u32>,
}
impl Certificate {
    fn target(&mut self, vocab: &Vocab) -> Result<()> {
        if self.target.as_ref().is_some_and(|v| !v.same_model_vocab(vocab)) {
            return Err(error("module graph targets incompatible vocabularies"));
        }
        if self.target.is_none() {
            for &id in &self.pending_tokens {
                if !vocab.contains_exact_token_id(id) { return Err(error(format!("unknown exact token ID {id}"))); }
            }
            self.pending_tokens.clear();
            self.target = Some(vocab.clone());
        }
        Ok(())
    }
    fn token(&mut self, id: u32) -> Result<()> {
        if let Some(v) = &self.target {
            if !v.contains_exact_token_id(id) { return Err(error(format!("unknown exact token ID {id}"))); }
        } else { self.pending_tokens.insert(id); }
        Ok(())
    }
    fn value(&mut self, value: &BoundValue) -> Result<()> {
        match value {
            BoundValue::Token(t) => self.token(t.id())?,
            BoundValue::ExactTokens(t) => self.target(t.target_vocab())?,
            BoundValue::Module(m) => {
                let child = m.certificate()?;
                self.height = self.height.max(child.height + 1);
                if let Some(v) = &child.target { self.target(v)?; }
                for &id in &child.pending_tokens { self.token(id)?; }
            }
        }
        Ok(())
    }
}

fn error(message: impl Into<String>) -> Error { Error::Compilation(message.into()) }

fn own_local(grammar: &Grammar<'_>) -> Grammar<'static> {
    let text = |source: &SourceText<'_>| match source {
        SourceText::Owned(text) => SourceText::Owned(text.clone()),
        SourceText::Borrowed(text) => SourceText::Owned(grammar.owned_text.get_or_init(|| Arc::from(*text)).clone()),
    };
    Grammar {
        source: match &grammar.source {
            GrammarSource::Ebnf(s) => GrammarSource::Ebnf(text(s)),
            GrammarSource::Lark(s) => GrammarSource::Lark(text(s)),
            GrammarSource::JsonSchema(s) => GrammarSource::JsonSchema(text(s)),
            GrammarSource::Glrm(s) => GrammarSource::Glrm(text(s)),
            GrammarSource::Constructed => GrammarSource::Constructed,
        },
        owned_text: grammar.owned_text.clone(),
        module_snapshot: Arc::new(OnceLock::new()),
        bindings: OrdMap::new(),
        declarations_cache: grammar.declarations_cache.clone(),
        analysis: graph::empty_cache(),
        parsed_component: grammar.parsed_component.clone(),
        exact_value: grammar.exact_value.clone(),
        exact_token_id: grammar.exact_token_id,
        lark_options: grammar.lark_options.clone(),
        lark_document: grammar.lark_document.clone(),
    }
}

pub(crate) fn resolve_engine(strategy: MaskStrategy, prepared: &BTreeMap<MaskStrategy, Arc<dyn RetainedPreparation>>) -> MaskStrategy {
    match strategy {
        MaskStrategy::Auto => {
            if prepared.contains_key(&MaskStrategy::ParserDwa) {
                MaskStrategy::ParserDwa
            } else if prepared.contains_key(&MaskStrategy::TokenClassTrie) {
                MaskStrategy::TokenClassTrie
            } else {
                MaskStrategy::ParserDwa
            }
        }
        explicit => explicit,
    }
}

pub(crate) fn target_boundary(_strategy: MaskStrategy, _boundary: BoundaryMaskStrategy) -> BoundaryTriggerDetail {
    // The public boundary trigger is always the LLM token set. BoundaryMaskStrategy
    // is retained for the separate boundary-parser wiring and no longer selects
    // the trigger detail here.
    BoundaryTriggerDetail::TokenSet
}

/// Compilation settings are per-module: `mask_strategy` applies at compile time
/// and `boundary_mask_strategy` selects the boundary parser.
impl Module {
    /// The mask computation strategy.
    pub fn mask_strategy(&self) -> MaskStrategy {
        self.node.mask_strategy
    }

    /// Sets the mask computation strategy, applied at compile time.
    ///
    /// The returned `Result` reserves room for future strategy validation.
    pub fn set_mask_strategy(&mut self, strategy: MaskStrategy) -> Result<()> {
        Arc::make_mut(&mut self.node).mask_strategy = strategy;
        Ok(())
    }

    /// The boundary parser strategy.
    pub fn boundary_mask_strategy(&self) -> BoundaryMaskStrategy {
        self.node.boundary_mask_strategy
    }

    /// Sets the boundary parser strategy for this module.
    ///
    /// A [`BoundaryMaskStrategy::ParserDwa`] build that exceeds its budget
    /// returns [`Error::ResourceLimit`] with a `boundary_parser_dwa:` message;
    /// choose [`BoundaryMaskStrategy::Dynamic`] (`FAST_BUILD`) to keep the
    /// dynamic boundary. The returned `Result` reserves room for future
    /// strategy validation.
    pub fn set_boundary_mask_strategy(&mut self, strategy: BoundaryMaskStrategy) -> Result<()> {
        Arc::make_mut(&mut self.node).boundary_mask_strategy = strategy;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn force_boundary_dwa_decline(&mut self) {
        Arc::make_mut(&mut self.node)
            .force_boundary_dwa_decline
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Grammar<'_> {
    pub(crate) fn module(&self) -> Module { Module::new(self) }
}

impl Module {
    /// Wraps a source grammar with default settings.
    pub fn new(grammar: &Grammar<'_>) -> Self {
        fn convert(grammar: &Grammar<'_>, cache: &mut HashMap<usize, Module>) -> Module {
            if let Some(module) = grammar.module_snapshot.get() { return module.clone(); }
            let id = graph::source_identity(grammar);
            if let Some(module) = cache.get(&id) { return module.clone(); }
            let mut bindings = OrdMap::new();
            for (name, value) in &grammar.bindings {
                let value = match value {
                    GrammarValue::Source(child) => BoundValue::Module(convert(child, cache)),
                    GrammarValue::Token(t) => BoundValue::Token(*t),
                    GrammarValue::ExactTokens(t) => BoundValue::ExactTokens(t.clone()),
                };
                bindings.insert(name.clone(), value);
            }
            let module = Module {
                node: Arc::new(Node {
                    source: Arc::new(own_local(grammar)),
                    mask_strategy: MaskStrategy::Auto,
                    boundary_mask_strategy: BoundaryMaskStrategy::ParserDwa,
                    prepared: BTreeMap::new(),
                    bindings,
                    certificate: Arc::new(OnceLock::new()),
                    #[cfg(test)]
                    force_boundary_dwa_decline: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
                }),
            };
            let _ = grammar.module_snapshot.set(module.clone());
            let module = grammar.module_snapshot.get().cloned().unwrap_or(module);
            cache.insert(id, module.clone());
            module
        }
        convert(grammar, &mut HashMap::new())
    }

    /// A view over the module's declared externs, in declaration order.
    ///
    /// A source that fails to parse yields an empty view; [`bind`](Self::bind),
    /// [`precompile`](Self::precompile) and [`compile`](Self::compile) report
    /// the parse error.
    pub fn externs(&self) -> Externs<'_> {
        Externs {
            module: self,
            only_unbound: false,
        }
    }

    // Lets bindings surface a parse error where the Rust view must stay infallible.
    #[cfg(any(test, feature = "internal-api"))]
    #[doc(hidden)]
    pub fn check_source(&self) -> Result<()> {
        self.node.source.declarations().map(|_| ())
    }

    /// Binds the extern `name` to a module, expression, token, or exact token
    /// sequence.
    ///
    /// Fails if the extern is undeclared or already bound.
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
            BindingKind::Token(token) => {
                check_binding_kind(name, kind, false)
                    .map_err(|e| e.with_source_span(declared.spans.get(name).copied()))?;
                BoundValue::Token(token)
            }
            BindingKind::ExactTokens(exact) => {
                check_binding_kind(name, kind, false)
                    .map_err(|e| e.with_source_span(declared.spans.get(name).copied()))?;
                BoundValue::ExactTokens(exact)
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
                c.target(&constraint_vocab(&p.inner))?;
            }
            if let Some(t) = &self.node.source.exact_value { c.target(t.target_vocab())?; }
            if let Some(id) = self.node.source.exact_token_id { c.token(id)?; }
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

    /// Retains vocabulary-specific compilation artifacts for every module in
    /// this assembly.
    ///
    /// Modules using [`MaskStrategy::TokenTrie`] retain no work and are compiled
    /// from source by [`Self::compile`].
    pub fn precompile(&mut self, vocab: &Vocab) -> Result<()> {
        with_deep_stack(|| self.precompile_on_stack(vocab, true))
    }

    /// Internal compile_with preparation: retain compilation artifacts and the
    /// default boundary token set.
    fn precompile_for_compile(&mut self, vocab: &Vocab) -> Result<()> {
        with_deep_stack(|| self.precompile_on_stack(vocab, true))
    }

    fn precompile_on_stack(&mut self, vocab: &Vocab, build_boundary: bool) -> Result<()> {
        self.validate_targets(Some(vocab))?;
        let modules = self.index()?;

        type JobKey = (usize, MaskStrategy, u8);
        let mut locals: HashMap<JobKey, Arc<dyn RetainedPreparation>> = HashMap::new();

        for m in &modules {
            let engine = resolve_engine(m.node.mask_strategy, &m.node.prepared);
            if engine == MaskStrategy::TokenTrie {
                continue;
            }
            let target_b = target_boundary(engine, m.node.boundary_mask_strategy);
            let key = (Arc::as_ptr(&m.node.source.parsed_component) as usize, engine, target_b as u8);
            if let Some(p) = m.node.prepared.get(&engine) {
                // `boundary_trigger` records the requested detail; the trigger
                // itself is built lazily in `materialize`.
                if p.prepared().boundary_trigger >= target_b {
                    locals.insert(key, p.clone());
                }
            }
        }

        let mut seen = std::collections::HashSet::new();
        let jobs: Vec<&Module> = modules.iter().filter(|m| {
            let engine = resolve_engine(m.node.mask_strategy, &m.node.prepared);
            if engine == MaskStrategy::TokenTrie {
                return false;
            }
            let target_b = target_boundary(engine, m.node.boundary_mask_strategy);
            let key = (Arc::as_ptr(&m.node.source.parsed_component) as usize, engine, target_b as u8);
            !locals.contains_key(&key) && seen.insert(key)
        }).collect();

        let results = jobs.par_iter().map(|m| {
            #[cfg(test)]
            let _probe = preparation_probe::enter(m);
            let engine = resolve_engine(m.node.mask_strategy, &m.node.prepared);
            let target_b = target_boundary(engine, m.node.boundary_mask_strategy);
            let key = (Arc::as_ptr(&m.node.source.parsed_component) as usize, engine, target_b as u8);
            m.prepare_local_for(vocab, engine, target_b, build_boundary)
                .map(|p| (key, retain_local(p)))
        }).collect::<Result<Vec<_>>>()?;

        locals.extend(results);

        let mut instances = HashMap::<usize, Module>::new();
        for m in modules {
            let mut next = m.clone();
            let mut c = m.certificate()?.as_ref().clone();
            c.target(vocab)?;
            let engine = resolve_engine(m.node.mask_strategy, &m.node.prepared);
            let node = Arc::make_mut(&mut next.node);
            if engine != MaskStrategy::TokenTrie {
                let target_b = target_boundary(engine, m.node.boundary_mask_strategy);
                let key = (Arc::as_ptr(&m.node.source.parsed_component) as usize, engine, target_b as u8);
                node.prepared.insert(engine, locals[&key].clone());
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
        engine: MaskStrategy,
        target_b: BoundaryTriggerDetail,
        build_boundary: bool,
    ) -> Result<PreparedGrammar> {
        if let Some(existing) = self.node.prepared.get(&engine) {
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
            let resolved = source.resolve_lark(vocab)?;
            let mut module = Module::new(&resolved);
            module.set_mask_strategy(engine)?;
            let mut instances = HashMap::<usize, Module>::new();
            for m in module.index()? {
                let mut next = m.clone();
                let p = m.prepare_local_for(vocab, engine, target_b, build_boundary)?;
                let node = Arc::make_mut(&mut next.node);
                node.prepared.insert(engine, retain_local(p));
                node.certificate = Arc::new(OnceLock::new());
                for (name, value) in &m.node.bindings {
                    if let BoundValue::Module(child) = value {
                        node.bindings.insert(name.clone(), BoundValue::Module(instances[&(Arc::as_ptr(&child.node) as usize)].clone()));
                    }
                }
                instances.insert(Arc::as_ptr(&m.node) as usize, next);
            }
            module = instances.remove(&(Arc::as_ptr(&module.node) as usize)).unwrap();
            let mut local = module.node.prepared.get(&engine).ok_or_else(|| error("module is not prepared"))?.prepared().clone();
            for (name, value) in &module.node.bindings {
                match value {
                    BoundValue::Module(child) => {
                        let child_prep = child.node.prepared.get(&engine).ok_or_else(|| error("child is not prepared"))?.prepared().clone();
                        local.bind(name, child_prep)?;
                    }
                    BoundValue::Token(t) => local.bind(name, *t)?,
                    BoundValue::ExactTokens(t) => local.bind(name, t.clone())?,
                }
            }
            return Ok(local);
        }

        let declared = source.declarations()?;
        // A typed template decline must survive so the public boundary can
        // attempt the rescue; only real failures are rewrapped.
        let token_class_error = |e: Error| {
            if e.is_strategy_decline() {
                e
            } else {
                Error::Compilation(format!("mask strategy TOKEN_CLASS_TRIE cannot compile this grammar: {e}"))
            }
        };
        let mut inner = match engine {
            MaskStrategy::TokenTrie => unreachable!("TOKEN_TRIE modules are never precompiled"),
            MaskStrategy::TokenClassTrie => match source.source {
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
                        return Err(Error::Compilation("mask strategy TOKEN_CLASS_TRIE cannot compile this grammar: requires one compiled component".into()));
                    }
                    components.pop().unwrap()
                }
            },
            MaskStrategy::Auto | MaskStrategy::ParserDwa => {
                let mut p = source.prepare_unlinked(vocab, BuildOptions::default())?;
                let inner = Arc::make_mut(&mut p.inner);
                crate::compiler::boundary_candidates::persist_boundary_candidate_summary(inner, vocab);
                let _ = inner.late_bind_vocab.set(vocab.clone());
                inner.install_template_parser_from_compile()?;
                if build_boundary {
                    inner.build_boundary_trigger(target_b).map_err(Error::Compilation)?;
                }
                p.boundary_trigger = target_b;
                return Ok(p);
            }
        };

        let _ = inner.late_bind_vocab.set(vocab.clone());
        if engine == MaskStrategy::TokenClassTrie {
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
        };
        prepared.validate_local_slot_manifest(vocab)?;
        prepared.validate_slot_manifest()?;
        Ok(prepared)
    }

    /// Compiles the module assembly against `vocab` with default options.
    pub fn compile(&self, vocab: &Vocab) -> Result<RuntimeConstraint> {
        self.compile_with(vocab, CompileOptions::default())
    }

    /// Compiles the module assembly against `vocab` with root compile options.
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<RuntimeConstraint> {
        with_deep_stack(|| self.compile_with_on_stack(vocab, options))
    }

    fn compile_with_on_stack(&self, vocab: &Vocab, options: CompileOptions) -> Result<RuntimeConstraint> {
        if let Some(slot) = self.externs().unbound().iter().next() {
            return Err(error(format!("extern {:?} is unbound", slot.name())));
        }
        self.validate_targets(Some(vocab))?;
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
        // TOKEN_TRIE compiles the whole composition as one ordinary Dynamic
        // grammar, inlined at source level. Auto parts join either kind of
        // composition; explicit strategies of the other kind are an error.
        let token_trie = resolve_engine(self.node.mask_strategy, &self.node.prepared) == MaskStrategy::TokenTrie;
        let conflicting = |m: &Module| match m.node.mask_strategy {
            MaskStrategy::TokenTrie => !token_trie,
            MaskStrategy::ParserDwa | MaskStrategy::TokenClassTrie => token_trie,
            _ => false,
        };
        if modules.iter().any(conflicting) {
            return Err(error("cannot combine TOKEN_TRIE parts with PARSER_DWA or TOKEN_CLASS_TRIE parts in one composition"));
        }
        // EXPERIMENT: one route report per top-level Module compile.
        let experiment = crate::compiler::terminal_ended_rescue::RescueExperiment::read();
        let engine = resolve_engine(self.node.mask_strategy, &self.node.prepared);
        if token_trie {
            let result = self.flatten_source()?.compile_token_trie(vocab, BoundaryTriggerDetail::TokenSet)?;
            let result = result.with_end_tokens(&options.end_tokens)?;
            crate::compiler::terminal_ended_rescue::report_route(
                experiment.report_route,
                crate::compiler::terminal_ended_rescue::Route::Normal,
                engine,
                experiment,
                &crate::compiler::terminal_ended_rescue::RescueOutcome::NotTried,
            );
            return Ok(result);
        }
        let te_first = experiment.terminal_ended_rescue
            && experiment.terminal_ended_first
            && self.node.bindings.is_empty()
            && matches!(
                engine,
                MaskStrategy::ParserDwa | MaskStrategy::TokenClassTrie | MaskStrategy::Auto
            );
        let mut route = crate::compiler::terminal_ended_rescue::Route::Normal;
        let mut outcome = crate::compiler::terminal_ended_rescue::RescueOutcome::NotTried;
        let result = (|| -> Result<RuntimeConstraint> {
            let try_rescue = || -> Result<RuntimeConstraint> {
                let grammar = self.flatten_source()?;
                let source = grammar.rescue_source_grammar(vocab)?;
                let mut constraint =
                    crate::compiler::terminal_ended_rescue::rescue(&source, vocab)?;
                apply_final_boundary_trigger(
                    &mut constraint,
                    target_boundary(engine, self.node.boundary_mask_strategy),
                )?;
                constraint.with_end_tokens(&options.end_tokens)
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
            let mut prepared = self.clone();
            let attempt = {
                let budget = crate::compiler::build_budget::BuildBudgetScope::enter();
                let end_tokens = options.end_tokens.clone();
                budget.install(move || {
                    crate::error::catch_template_attempt(move || -> Result<RuntimeConstraint> {
                        prepared.precompile_for_compile(vocab)?;
                        let constraint = prepared.materialize(vocab, &mut HashMap::new())?;
                        ensure_runnable_constraint(&constraint)?;
                        constraint.with_end_tokens(&end_tokens)
                    })
                })
            };
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
                                        "[glrmask/profile][terminal_ended_rescue] strategy={engine:?} outcome=ok",
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
                                        "[glrmask/profile][terminal_ended_rescue] strategy={engine:?} outcome=declined reason={rescue_error}",
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
                            "[glrmask/profile][template_decline] strategy={engine:?} stage={} reason={error}",
                            error.decline_stage(),
                        );
                    }
                    // Only Auto may pick a different exact strategy; an explicit
                    // strategy request is never silently replaced.
                    if self.node.mask_strategy == MaskStrategy::Auto {
                        if let Ok(grammar) = self.flatten_source() {
                            let constraint = grammar
                                .compile_token_trie(vocab, BoundaryTriggerDetail::TokenSet)?
                                .with_end_tokens(&options.end_tokens)?;
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
            engine,
            experiment,
            &outcome,
        );
        result
    }

    fn materialize(&self, vocab: &Vocab, cache: &mut HashMap<usize, RuntimeConstraint>) -> Result<RuntimeConstraint> {
        let id = Arc::as_ptr(&self.node) as usize;
        if let Some(c) = cache.get(&id) { return Ok(c.clone()); }
        let engine = resolve_engine(self.node.mask_strategy, &self.node.prepared);
        let local = self.node.prepared.get(&engine).ok_or_else(|| error("missing local compilation"))?;
        let mut parent = if local.bindings.is_empty() {
            local.inner.as_ref().clone()
        } else {
            local.link_with(BuildOptions::default().optimization(engine))?
        };
        parent.materialize_composition_link_metadata_for_compilation().map_err(Error::Compilation)?;
        let active = parent.late_grammar_slots.iter().map(|s| s.name.as_str()).collect::<BTreeSet<_>>();
        let children = self.node.bindings.iter().filter(|(name, _)| active.contains(name.as_str())).map(|(name, value)| {
            let child = match value {
                BoundValue::Module(child) => child.materialize(vocab, cache)?,
                BoundValue::Token(t) => self.event_adapter(&[(t.id(), t.id())], vocab, parent.has_template_parser(), engine)?,
                BoundValue::ExactTokens(t) => self.event_adapter(t.range_pairs(), vocab, parent.has_template_parser(), engine)?,
            };
            Ok((name.clone(), Arc::new(child)))
        }).collect::<Result<Vec<_>>>()?;
        let linked = !children.is_empty();
        let mut child_token_union = Vec::new();
        for (_, child) in &children {
            if let crate::runtime::BoundaryTrigger::TokenSet(child_tokens) = &child.boundary_trigger {
                child_token_union.extend_from_slice(child_tokens);
            }
        }
        if linked {
            parent = if parent.has_template_parser() {
                crate::error::catch_internal_invariant(|| crate::runtime::parser_backend::link::compose_owned(parent, children, vocab))??
            } else {
                crate::error::catch_internal_invariant(|| compose_named_children(parent, &children, vocab, SegmentedBoundaryBackend::Dynamic))??
            };
            if self.node.boundary_mask_strategy == BoundaryMaskStrategy::ParserDwa
                && parent.has_template_parser()
                && matches!(engine, MaskStrategy::ParserDwa | MaskStrategy::Auto)
            {
                #[cfg(test)]
                if self
                    .node
                    .force_boundary_dwa_decline
                    .load(std::sync::atomic::Ordering::Relaxed)
                {
                    return Err(Error::ResourceLimit(
                        "boundary_parser_dwa: forced test decline; use BoundaryMaskStrategy::FAST_BUILD (Dynamic)".into(),
                    ));
                }
                match crate::compiler::template_boundary::install(&mut parent, vocab) {
                    Ok(()) => {}
                    Err(error) if crate::compiler::template_boundary::is_template_budget_decline(&error) => {
                        return Err(Error::ResourceLimit(
                            "boundary_parser_dwa: static boundary parser DWA declined within the build budget; use BoundaryMaskStrategy::FAST_BUILD (Dynamic) to keep the segmented dynamic boundary".into(),
                        ));
                    }
                    Err(error) => return Err(error),
                }
            }
            let target_b = target_boundary(engine, self.node.boundary_mask_strategy);
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

    fn event_adapter(&self, ranges: &[(u32, u32)], vocab: &Vocab, native: bool, engine: MaskStrategy) -> Result<RuntimeConstraint> {
        let local = self.node.prepared.get(&engine).ok_or_else(|| error("missing prepared event owner"))?;
        let leaf = local.event_adapters.leaf(ranges)?;
        Ok(leaf.compiled_for_representation(vocab, native)?.as_ref().clone())
    }

    pub(super) fn to_source(&self) -> Result<Grammar<'static>> {
        if !self.node.prepared.is_empty() || self.node.mask_strategy != MaskStrategy::default() || self.node.boundary_mask_strategy != BoundaryMaskStrategy::default() {
            return Err(error("expressions with configured or precompiled operands cannot be converted to source Grammar; bind directly to Module instead"));
        }
        self.flatten_source()
    }

    // The whole composition as one source Grammar, with every bound child
    // inlined as a scoped source body. Retained work and settings are ignored.
    fn flatten_source(&self) -> Result<Grammar<'static>> {
        let mut grammar = self.node.source.as_ref().clone();
        for (name, value) in &self.node.bindings {
            let value = match value {
                BoundValue::Module(child) => GrammarValue::Source(Arc::new(child.flatten_source()?)),
                BoundValue::Token(t) => GrammarValue::Token(*t),
                BoundValue::ExactTokens(t) => GrammarValue::ExactTokens(t.clone()),
            };
            grammar = grammar.bind_value(name, value)?;
        }
        Ok(grammar)
    }

    /// Serializes the module assembly and any retained precompiled artifacts.
    pub fn save(&self) -> Vec<u8> { with_deep_stack(|| artifact::save(self)) }
    /// Deserializes a module assembly, verifying it against `vocab` when supplied.
    pub fn load(bytes: &[u8], vocab: Option<&Vocab>) -> Result<Self> { with_deep_stack(|| artifact::load(bytes, vocab)) }
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
        let grammar = Grammar::from_ebnf(r#"start ::= "x" "y" | "xy""#);

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
        let loaded = Module::load(&bytes, Some(&vocab)).expect("Module artifact round-trip");
        let loaded_trigger = loaded
            .node
            .prepared
            .values()
            .next()
            .and_then(|prepared| tokens(&prepared.prepared().inner))
            .expect("loaded Module must keep its TokenSet");
        assert_eq!(loaded_trigger.as_ref(), prepared_trigger.as_ref());

        let constraint_bytes = compiled.save();
        let mut loaded_constraint = RuntimeConstraint::load(&constraint_bytes).expect("Constraint round-trip");
        loaded_constraint
            .materialize_composition_metadata_for_compilation()
            .expect("materialize loaded metadata");
        let loaded_constraint_trigger = tokens(&loaded_constraint).expect("loaded Constraint must keep its TokenSet");
        assert_eq!(loaded_constraint_trigger.as_ref(), prepared_trigger.as_ref());

        let mut old = b"GLRASM01".to_vec();
        old.extend_from_slice(&bytes[8..]);
        let error = Module::load(&old, Some(&vocab)).expect_err("old Module artifact must be rejected");
        assert!(error.to_string().contains("unsupported Module artifact version"), "{error}");
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
        let mut module = Module::new(&Grammar::from_glrm(parent));
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
            if next_gated.commit_token(token).is_err() {
                continue;
            }
            next_ungated
                .commit_token(token)
                .expect("ungated commit must accept a gated token");
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
        let child = Module::new(&Grammar::from_lark("start: \"y\"?"));
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
            Module::new(&Grammar::from_ebnf(r#"start ::= "x""#)),
            &vocab,
        );
        assert_gate_agrees(&mid_token, "child ends mid-token");

        let after_prefix = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = "a" child;"#,
            Module::new(&Grammar::from_ebnf(r#"start ::= "b""#)),
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
            ));
            middle.bind("child", child).unwrap();
            middle
        };
        let nested = compile_linked(
            r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            nested_child,
            &vocab,
        );
        assert_gate_agrees(&nested, "two-level nested");

        let ignored = Grammar::from_lark("start: \"x\" \"y\"\n%ignore WS\nWS: /[ \\t]+/")
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
    fn dynamic_and_parser_dwa_boundaries_produce_identical_masks() {
        let vocab = vocab();
        let child = Module::new(&Grammar::from_ebnf(r#"start ::= "y" "z""#));
        let compile = |strategy| {
            let mut parent = Module::new(&Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            ));
            parent.bind("child", child.clone()).unwrap();
            parent.set_boundary_mask_strategy(strategy).unwrap();
            parent
                .compile_with(&vocab, CompileOptions::default())
                .unwrap()
        };
        let static_boundary = compile(BoundaryMaskStrategy::ParserDwa);
        let dynamic = compile(BoundaryMaskStrategy::Dynamic);
        for prefix in [&[0u32][..0], &[0][..], &[0, 1][..]] {
            let mut a = static_boundary.start();
            let mut b = dynamic.start();
            for &token in prefix {
                a.commit_token(token).unwrap();
                b.commit_token(token).unwrap();
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
    fn parser_dwa_budget_decline_is_typed_and_dynamic_succeeds() {
        let vocab = {
            let vocab = Vocab::new(vec![(0, b"x".to_vec()), (1, b"y".to_vec())]);
            vocab.prepare_for_compile();
            vocab
        };
        let compile = |strategy| {
            let mut parent = Module::new(&Grammar::from_glrm(
                r#"glrm 1; start start; extern grammar child; nt start = "x" child;"#,
            ));
            parent
                .bind("child", Module::new(&Grammar::from_ebnf(r#"start ::= "y""#)))
                .unwrap();
            parent.set_boundary_mask_strategy(strategy).unwrap();
            parent.force_boundary_dwa_decline();
            parent.compile_with(&vocab, CompileOptions::default())
        };
        let error = compile(BoundaryMaskStrategy::ParserDwa).unwrap_err();
        match error {
            Error::ResourceLimit(message) => {
                assert!(message.starts_with("boundary_parser_dwa:"));
                assert!(message.contains("FAST_BUILD"));
            }
            other => panic!("expected ResourceLimit, got {other:?}"),
        }
        compile(BoundaryMaskStrategy::Dynamic).expect("Dynamic must bypass the static boundary parser");
    }
}
