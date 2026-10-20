//! Owned assembly snapshots. Local compilation is independent of occurrence bindings.
use super::*;
use super::expr::check_expr_unbound;
use std::collections::{HashMap, BTreeMap, BTreeSet};
mod artifact;
#[cfg(test)]
thread_local! { static BIND_WORK: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

/// Root generation policy; never inherited by embedded modules.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CompileOptions {
    pub(crate) end_tokens: Vec<u32>,
}

impl CompileOptions {
    /// Configures accepted end-of-sequence token IDs.
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

/// An owned grammar assembly with local settings, occurrence bindings, and optional retained precompilation.
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

pub(crate) fn target_boundary(strategy: MaskStrategy, boundary: BoundaryMaskStrategy) -> BoundaryTriggerDetail {
    if strategy == MaskStrategy::TokenTrie {
        BoundaryTriggerDetail::Tokens
    } else {
        boundary.to_detail()
    }
}

/// Settings are per-Module; `mask_strategy` is applied at compile time and `boundary_mask_strategy` defines a minimum detail level.
impl Module {
    /// Returns the module's mask generation strategy.
    pub fn mask_strategy(&self) -> MaskStrategy {
        self.node.mask_strategy
    }

    /// Sets the module's mask generation strategy, applied at compile time.
    ///
    /// The returned `Result` reserves room for future strategy validation.
    pub fn set_mask_strategy(&mut self, strategy: MaskStrategy) -> Result<()> {
        Arc::make_mut(&mut self.node).mask_strategy = strategy;
        Ok(())
    }

    /// Returns the minimum boundary mask detail level.
    pub fn boundary_mask_strategy(&self) -> BoundaryMaskStrategy {
        self.node.boundary_mask_strategy
    }

    /// Sets the minimum boundary mask detail level for this module.
    ///
    /// The returned `Result` reserves room for future strategy validation.
    pub fn set_boundary_mask_strategy(&mut self, strategy: BoundaryMaskStrategy) -> Result<()> {
        Arc::make_mut(&mut self.node).boundary_mask_strategy = strategy;
        Ok(())
    }
}

impl Grammar<'_> {
    pub(crate) fn module(&self) -> Module { Module::new(self) }
}

impl Module {
    /// Creates a new module wrapping a source grammar with default settings.
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
                    boundary_mask_strategy: BoundaryMaskStrategy::FastBuild,
                    prepared: BTreeMap::new(),
                    bindings,
                    certificate: Arc::new(OnceLock::new()),
                }),
            };
            let _ = grammar.module_snapshot.set(module.clone());
            let module = grammar.module_snapshot.get().cloned().unwrap_or(module);
            cache.insert(id, module.clone());
            module
        }
        convert(grammar, &mut HashMap::new())
    }

    /// Returns a view over the module's declared externs, in declaration order.
    ///
    /// Sources are parsed lazily, so a source that fails to parse yields an
    /// empty view; [`bind`](Self::bind), [`precompile`](Self::precompile) and
    /// [`compile`](Self::compile) report the parse error.
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

    /// Binds an unresolved extern by name to a module, expression, token, or exact token sequence.
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

    /// Retains vocabulary-specific compilation artifacts across all modules in this assembly.
    ///
    /// Modules using `TOKEN_TRIE` retain no work and are compiled from source by `compile`.
    pub fn precompile(&mut self, vocab: &Vocab) -> Result<()> {
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
                let current_b = p.prepared().inner.boundary_trigger.detail();
                if current_b >= target_b {
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
            m.prepare_local_for(vocab, engine, target_b).map(|p| (key, retain_local(p)))
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

    fn prepare_local_for(&self, vocab: &Vocab, engine: MaskStrategy, target_b: BoundaryTriggerDetail) -> Result<PreparedGrammar> {
        if let Some(existing) = self.node.prepared.get(&engine) {
            let mut upgraded = existing.prepared().clone();
            Arc::make_mut(&mut upgraded.inner).build_boundary_trigger(target_b).map_err(Error::Compilation)?;
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
                let p = m.prepare_local_for(vocab, engine, target_b)?;
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
        let mut inner = match engine {
            MaskStrategy::TokenTrie => unreachable!("TOKEN_TRIE modules are never precompiled"),
            MaskStrategy::TokenClassTrie => match source.source {
                GrammarSource::Glrm(_) | GrammarSource::Constructed | GrammarSource::Ebnf(_) => {
                    crate::import::compile_component_shell_balanced(source.component()?, vocab, &[])
                        .map_err(|e| Error::Compilation(format!("mask strategy TOKEN_CLASS_TRIE cannot compile this grammar: {e}")))?
                }
                _ => {
                    let dynamic = match &source.source {
                        GrammarSource::Lark(s) => DynamicConstraint::from_lark_with_bounded_template_parser(s, vocab),
                        GrammarSource::JsonSchema(s) => DynamicConstraint::from_json_schema_with_bounded_template_parser(s, vocab),
                        _ => unreachable!(),
                    }.map_err(|e| Error::Compilation(format!("mask strategy TOKEN_CLASS_TRIE cannot compile this grammar: {e}")))?;
                    let mut components = dynamic.into_constraints();
                    if components.len() != 1 {
                        return Err(Error::Compilation("mask strategy TOKEN_CLASS_TRIE cannot compile this grammar: requires one compiled component".into()));
                    }
                    components.pop().unwrap()
                }
            },
            MaskStrategy::Auto | MaskStrategy::ParserDwa => {
                let mut p = source.prepare(vocab)?;
                let inner = Arc::make_mut(&mut p.inner);
                crate::compiler::boundary_candidates::persist_boundary_candidate_summary(inner, vocab);
                inner.install_template_parser_from_compile()?;
                inner.build_boundary_trigger(target_b).map_err(Error::Compilation)?;
                p.boundary_trigger = target_b;
                return Ok(p);
            }
        };

        let _ = inner.late_bind_vocab.set(vocab.clone());
        if engine == MaskStrategy::TokenClassTrie {
            inner.install_template_parser_from_compile()?;
        }
        inner.build_boundary_trigger(target_b).map_err(Error::Compilation)?;
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

    /// Compiles the assembly against a target vocabulary using default options.
    pub fn compile(&self, vocab: &Vocab) -> Result<RuntimeConstraint> {
        self.compile_with(vocab, CompileOptions::default())
    }

    /// Compiles the assembly against a target vocabulary with root compile options.
    pub fn compile_with(&self, vocab: &Vocab, options: CompileOptions) -> Result<RuntimeConstraint> {
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
        if token_trie {
            let constraint = self.flatten_source()?.compile_token_trie(vocab, BoundaryTriggerDetail::Tokens)?;
            return constraint.with_end_tokens(&options.end_tokens);
        }
        let mut prepared = self.clone();
        prepared.precompile(vocab)?;
        let constraint = prepared.materialize(vocab, &mut HashMap::new())?;
        ensure_runnable_constraint(&constraint)?;
        constraint.with_end_tokens(&options.end_tokens)
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
        if linked {
            parent = if parent.has_template_parser() {
                crate::error::catch_internal_invariant(|| crate::runtime::parser_backend::link::compose_owned(parent, children, vocab))??
            } else {
                crate::error::catch_internal_invariant(|| compose_named_children(parent, &children, vocab, SegmentedBoundaryBackend::Dynamic))??
            };
            if parent.has_template_parser() && matches!(engine, MaskStrategy::ParserDwa | MaskStrategy::Auto) {
                crate::compiler::template_boundary::install(&mut parent, vocab)?;
            }
            let target_b = target_boundary(engine, self.node.boundary_mask_strategy);
            apply_final_boundary_trigger(&mut parent, target_b)?;
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

    /// Serializes the module assembly and any retained precompiled artifacts to bytes.
    pub fn save(&self) -> Vec<u8> { artifact::save(self) }
    /// Deserializes a module assembly from bytes, verifying against an optional vocabulary.
    pub fn load(bytes: &[u8], vocab: Option<&Vocab>) -> Result<Self> { artifact::load(bytes, vocab) }
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
