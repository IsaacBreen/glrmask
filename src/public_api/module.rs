//! Owned assembly snapshots. Local compilation is independent of occurrence bindings.
use super::*;
use std::collections::HashMap;
mod artifact;

/// Existing component-local compiler choices. Children retain their own options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModuleOptions {
    optimization: Optimization,
    boundary_trigger: BoundaryTriggerDetail,
}
impl ModuleOptions {
    pub fn optimization(mut self, value: Optimization) -> Self { self.optimization=value; self }
    pub fn optimization_value(&self)->Optimization {self.optimization}
    pub fn boundary_trigger(mut self, value: BoundaryTriggerDetail) -> Self { self.boundary_trigger=value; self }
    pub fn boundary_trigger_value(&self)->BoundaryTriggerDetail {self.boundary_trigger}
}

/// Root generation policy; never inherited by embedded modules.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompileOptions { end_tokens: Vec<u32> }
impl CompileOptions {
    pub fn end_tokens(mut self, ids: impl IntoIterator<Item=u32>) -> Self { self.end_tokens=ids.into_iter().collect(); self }
}

/// Owned source, local options and reusable preparation, with independent occurrence bindings.
#[derive(Debug, Clone)]
pub struct Module { node: Arc<Node> }
#[derive(Debug, Clone)]
struct Node {
    source: Arc<Grammar<'static>>,
    options: ModuleOptions,
    prepared: Option<Arc<PreparedGrammar>>,
    bindings: OrdMap<String,ModuleValue>,
}

/// Snapshot value accepted by [`Module::bind`].
#[derive(Debug, Clone)]
pub enum ModuleValue { Module(Module), Token(Token), ExactTokens(ExactTokens) }
/// Conversion into an owned assembly value. Expression validation can fail.
pub trait IntoModuleValue { fn into_module_value(self) -> Result<ModuleValue>; }
impl IntoModuleValue for ModuleValue { fn into_module_value(self)->Result<ModuleValue>{Ok(self)} }
impl IntoModuleValue for Module { fn into_module_value(self)->Result<ModuleValue>{Ok(ModuleValue::Module(self))} }
impl IntoModuleValue for &Module { fn into_module_value(self)->Result<ModuleValue>{self.clone().into_module_value()} }
impl IntoModuleValue for Grammar<'_> { fn into_module_value(self)->Result<ModuleValue>{Module::new(&self).into_module_value()} }
impl IntoModuleValue for &Grammar<'_> { fn into_module_value(self)->Result<ModuleValue>{Module::new(self).into_module_value()} }
impl IntoModuleValue for Token { fn into_module_value(self)->Result<ModuleValue>{Ok(ModuleValue::Token(self))} }
impl IntoModuleValue for &Token { fn into_module_value(self)->Result<ModuleValue>{(*self).into_module_value()} }
impl IntoModuleValue for ExactTokens { fn into_module_value(self)->Result<ModuleValue>{Ok(ModuleValue::ExactTokens(self))} }
impl IntoModuleValue for &ExactTokens { fn into_module_value(self)->Result<ModuleValue>{self.clone().into_module_value()} }
impl IntoModuleValue for Expr { fn into_module_value(self)->Result<ModuleValue>{Module::from_expr(&self)?.into_module_value()} }
impl IntoModuleValue for &Expr { fn into_module_value(self)->Result<ModuleValue>{Module::from_expr(self)?.into_module_value()} }

fn error(message:impl Into<String>)->Error { Error::Compilation(message.into()) }
fn own_local(grammar:&Grammar<'_>)->Grammar<'static> {
    let text=|source:&SourceText<'_>|match source {
        SourceText::Owned(text)=>SourceText::Owned(text.clone()),
        SourceText::Borrowed(text)=>SourceText::Owned(grammar.owned_text.get_or_init(||Arc::from(*text)).clone()),
    };
    Grammar {
        source:match &grammar.source {
            GrammarSource::Ebnf(s)=>GrammarSource::Ebnf(text(s)), GrammarSource::Lark(s)=>GrammarSource::Lark(text(s)),
            GrammarSource::JsonSchema(s)=>GrammarSource::JsonSchema(text(s)), GrammarSource::Glrm(s)=>GrammarSource::Glrm(text(s)),
            GrammarSource::Constructed=>GrammarSource::Constructed,
        },
        owned_text:grammar.owned_text.clone(), module_snapshot:Arc::new(OnceLock::new()), bindings:OrdMap::new(),
        declarations_cache:grammar.declarations_cache.clone(), analysis:graph::empty_cache(),
        parsed_component:grammar.parsed_component.clone(), exact_value:grammar.exact_value.clone(),
        exact_token_id:grammar.exact_token_id, lark_options:grammar.lark_options.clone(), lark_document:grammar.lark_document.clone(),
    }
}

impl Grammar<'_> {
    /// Own the source once and snapshot its current source bindings.
    pub fn module(&self)->Module { Module::new(self) }
    pub fn unbound(&self)->Result<Vec<UnboundExtern>> { graph::source_requirements(self) }
    /// All declared slots, including slots that already have bindings.
    pub fn externs(&self)->Result<Vec<UnboundExtern>> { self.module().externs() }
}

impl Module {
    pub fn new(grammar:&Grammar<'_>)->Self {
        fn convert(grammar:&Grammar<'_>, cache:&mut HashMap<usize,Module>)->Module {
            if let Some(module)=grammar.module_snapshot.get(){return module.clone();}
            let id=graph::source_identity(grammar);
            if let Some(module)=cache.get(&id) {return module.clone();}
            let mut bindings=OrdMap::new();
            for (name,value) in &grammar.bindings {
                let value=match value {
                    GrammarValue::Source(child)=>ModuleValue::Module(convert(child,cache)),
                    GrammarValue::Token(t)=>ModuleValue::Token(*t), GrammarValue::ExactTokens(t)=>ModuleValue::ExactTokens(t.clone()),
                };
                bindings.insert(name.clone(),value);
            }
            let module=Module {node:Arc::new(Node {source:Arc::new(own_local(grammar)), options:ModuleOptions::default(),prepared:None,bindings})};
            let _=grammar.module_snapshot.set(module.clone());
            let module=grammar.module_snapshot.get().cloned().unwrap_or(module);
            cache.insert(id,module.clone());module
        }
        convert(grammar,&mut HashMap::new())
    }
    pub fn options(&self)->&ModuleOptions { &self.node.options }
    /// Preserve source, attachments and prepared children. Incompatible retained local work is an error.
    pub fn with_options(&self, options:ModuleOptions)->Result<Self> {
        if options.optimization==Optimization::FastBuild && options.boundary_trigger==BoundaryTriggerDetail::Exact {
            return Err(error("FastBuild Module with an Exact boundary trigger is unsupported; exact triggers require a native component"));
        }
        if options==self.node.options {return Ok(self.clone());}
        if self.node.prepared.is_some() {return Err(error("cannot change options of a locally precompiled module"));}
        let mut next=self.clone(); Arc::make_mut(&mut next.node).options=options; Ok(next)
    }
    pub fn set_options(&mut self,options:ModuleOptions)->Result<()> { *self=self.with_options(options)?;Ok(()) }

    /// Fill one declared slot atomically; attached values are snapshots.
    pub fn bind(&mut self,path:impl AsRef<str>,value:impl IntoModuleValue)->Result<()> {
        let value=value.into_module_value()?;
        let next=self.bound(path.as_ref(),value,0)?;
        next.index()?;
        next.validate_targets(None)?;
        *self=next;Ok(())
    }
    fn bound(&self,path:&str,value:ModuleValue,depth:usize)->Result<Self> {
        if depth>graph::MAX_DEPTH {return Err(error("module attachment depth exceeds 64"));}
        let declared=self.node.source.declarations()?;
        let mut next=self.clone();
        if let Some(&kind)=declared.kinds.get(path) {
            if self.node.bindings.contains_key(path) {return Err(error(format!("external {path:?} was bound more than once")));}
            check_binding_kind(path,kind,matches!(value,ModuleValue::Module(_)))
                .map_err(|e|e.with_source_span(declared.spans.get(path).copied()))?;
            Arc::make_mut(&mut next.node).bindings.insert(path.to_owned(),value);
        } else if let Some((head,tail))=attached_path(path,&self.node.bindings) {
            let Some(ModuleValue::Module(child))=self.node.bindings.get(head) else {return Err(error(format!("{head:?} has no descendant slots")));};
            let child=child.bound(tail,value,depth+1)?;
            Arc::make_mut(&mut next.node).bindings.insert(head.to_owned(),ModuleValue::Module(child));
        } else {return Err(error(format!("no external named {path:?}; attach its parent before binding descendants")));}
        Ok(next)
    }

    // Distinct instances are visited once. Occurrence expansion is separate and bounded.
    fn index(&self)->Result<Vec<Module>> {
        fn visit(module:&Module,active:&mut BTreeSet<usize>,done:&mut BTreeSet<usize>,out:&mut Vec<Module>,depth:usize)->Result<()> {
            if depth>graph::MAX_DEPTH {return Err(error("module attachment depth exceeds 64"));}
            let id=Arc::as_ptr(&module.node) as usize;
            if active.contains(&id) {return Err(error("cyclic module graph"));}
            if done.contains(&id) {return Ok(());}
            active.insert(id);
            for value in module.node.bindings.values() {if let ModuleValue::Module(child)=value {visit(child,active,done,out,depth+1)?;}}
            active.remove(&id);done.insert(id);out.push(module.clone());
            if out.len()>graph::MAX_NODES {return Err(error("too many module instances"));} Ok(())
        }
        let mut out=Vec::new();visit(self,&mut BTreeSet::new(),&mut BTreeSet::new(),&mut out,0)?;
        let mut summaries=HashMap::<usize,(usize,usize)>::new();let mut ports=0usize;
        for module in &out {
            let mut height=0;let mut expanded=1usize;
            ports=ports.saturating_add(module.node.bindings.len());
            if ports>1_048_576 {return Err(error("too many Module ports"));}
            for value in module.node.bindings.values(){if let ModuleValue::Module(child)=value {
                let (h,e)=summaries[&(Arc::as_ptr(&child.node) as usize)];height=height.max(h+1);
                expanded=expanded.saturating_add(e);
            }}
            if height>graph::MAX_DEPTH {return Err(error("Module depth exceeds 64"));}
            summaries.insert(Arc::as_ptr(&module.node) as usize,(height,expanded));
        }
        Ok(out)
    }
    fn slots(&self,all:bool)->Result<Vec<UnboundExtern>> {
        let modules=self.index()?;let mut counts=HashMap::<usize,usize>::new();
        for module in modules {
            let declared=module.node.source.declarations()?;
            let mut count=declared.kinds.keys().filter(|name|all || !module.node.bindings.contains_key(name.as_str())).count();
            for value in module.node.bindings.values(){if let ModuleValue::Module(child)=value {
                count=count.saturating_add(counts[&(Arc::as_ptr(&child.node) as usize)]);
            }}
            counts.insert(Arc::as_ptr(&module.node) as usize,count);
        }
        let count=counts[&(Arc::as_ptr(&self.node) as usize)];
        if count==0 {return Ok(Vec::new());}
        if count>65_536 {return Err(error("too many expanded extern slots"));}
        let mut pending=vec![(self,Vec::<String>::new())];let mut out=Vec::new();let mut visits=0;
        while let Some((module,prefix))=pending.pop() {
            visits+=1;if visits>graph::MAX_NODES {return Err(error("module diagnostic expansion limit exceeded"));}
            let declared=module.node.source.declarations()?;
            for (name,&kind) in &declared.kinds {
                let mut path=prefix.clone();path.extend(name.split("::").map(str::to_owned));
                if all || !module.node.bindings.contains_key(name) {
                    out.push(UnboundExtern {path,kind,source_span:declared.spans.get(name).copied()});
                    if out.len()>65_536 {return Err(error("too many expanded extern slots"));}
                }
            }
            for (name,value) in &module.node.bindings {if let ModuleValue::Module(child)=value {
                if counts[&(Arc::as_ptr(&child.node) as usize)]!=0 {
                    let mut path=prefix.clone();path.extend(name.split("::").map(str::to_owned));pending.push((child,path));
                }
            }}
        }
        out.sort_by(|a,b|a.path.cmp(&b.path));Ok(out)
    }
    pub fn unbound(&self)->Result<Vec<UnboundExtern>> { self.slots(false) }
    pub fn externs(&self)->Result<Vec<UnboundExtern>> { self.slots(true) }

    fn validate_targets(&self,supplied:Option<&Vocab>)->Result<()> {
        let modules=self.index()?;let mut target=supplied.cloned();
        for module in &modules {
            let mut check=|vocab:&Vocab|->Result<()> {
                if target.as_ref().is_some_and(|v|!v.same_model_vocab(vocab)) {return Err(error("module graph targets incompatible vocabularies"));}
                if target.is_none(){target=Some(vocab.clone());}Ok(())
            };
            if let Some(p)=&module.node.prepared {check(&constraint_vocab(&p.inner))?;}
            if let Some(t)=&module.node.source.exact_value {check(t.target_vocab())?;}
            for value in module.node.bindings.values(){if let ModuleValue::ExactTokens(t)=value {check(t.target_vocab())?;}}
        }
        if let Some(v)=&target {for module in &modules {
            module.node.source.validate_exact_target(v)?;
            for value in module.node.bindings.values(){if let ModuleValue::Token(t)=value {
                if !v.contains_exact_token_id(t.id()){return Err(error(format!("unknown exact token ID {}",t.id())));}
            }}
        }}
        Ok(())
    }

    /// Retain local reusable work even while descendants remain open. Failure is atomic.
    pub fn precompile(&mut self,vocab:&Vocab)->Result<()> {
        self.index()?;self.validate_targets(Some(vocab))?;
        let mut locals=HashMap::new();
        for m in self.index()? {if let Some(p)=&m.node.prepared {
            locals.insert((Arc::as_ptr(&m.node.source.parsed_component) as usize,m.node.options.optimization as u8,m.node.options.boundary_trigger as u8),p.clone());
        }}
        *self=self.prepared(vocab,&mut HashMap::new(),&mut locals)?;Ok(())
    }
    fn prepared(&self,vocab:&Vocab,instances:&mut HashMap<usize,Module>,locals:&mut HashMap<(usize,u8,u8),Arc<PreparedGrammar>>)->Result<Self> {
        let id=Arc::as_ptr(&self.node) as usize;
        if let Some(m)=instances.get(&id){return Ok(m.clone());}
        let mut next=self.clone();
        let key=(Arc::as_ptr(&self.node.source.parsed_component) as usize,self.node.options.optimization as u8,self.node.options.boundary_trigger as u8);
        let local=if let Some(p)=&self.node.prepared {p.clone()}
            else if let Some(p)=locals.get(&key){p.clone()}
            else {let p=Arc::new(self.prepare_local(vocab)?);locals.insert(key,p.clone());p};
        let node=Arc::make_mut(&mut next.node);node.prepared=Some(local);
        for (name,value) in &self.node.bindings {
            if let ModuleValue::Module(child)=value {
                node.bindings.insert(name.clone(),ModuleValue::Module(child.prepared(vocab,instances,locals)?));
            }
        }
        instances.insert(id,next.clone());Ok(next)
    }
    fn prepare_local(&self,vocab:&Vocab)->Result<PreparedGrammar> {
        let source=&self.node.source;
        let options=&self.node.options;
        if matches!(source.source,GrammarSource::Lark(_)) {
            let resolved=source.resolve_lark(vocab)?;
            if options.optimization==Optimization::FastBuild && !resolved.bindings.is_empty() {
                if !source.declarations()?.kinds.is_empty() {return Err(error("FastBuild Module with both inline schemas and open imports is unsupported"));}
                let inner=source.compile_with(vocab,BuildOptions::default().optimization(Optimization::FastBuild).boundary_trigger(options.boundary_trigger))?;
                return Ok(PreparedGrammar {inner:Arc::new(inner),bindings:OrdMap::new(),analysis:graph::empty_cache(),
                    event_adapters:Arc::new(event_cache::ExactAdapterCache::default()),boundary_trigger:options.boundary_trigger,
                    slot_kinds:Arc::new(BTreeMap::new()),slot_spans:Arc::new(BTreeMap::new())});
            }
            let mut module=Module::new(&resolved).with_options(options.clone())?;
            module.precompile(vocab)?;
            return module.as_prepared();
        }
        let declared=source.declarations()?;
        let mut inner=match options.optimization {
            Optimization::FastBuild=>match source.source {
                GrammarSource::Glrm(_)|GrammarSource::Constructed=>crate::import::compile_component_shell(source.component()?,vocab,&[],true)?,
                _=>crate::import::compile_dynamic_named_single(source.resolve_dynamic_source(vocab,false)?,vocab,
                    if matches!(source.source,GrammarSource::JsonSchema(_)){crate::compiler::glr::table::GlrTableConstruction::Lalr}
                    else{crate::compiler::glr::table::GlrTableConstruction::ExperimentalCoreMerged},matches!(source.source,GrammarSource::JsonSchema(_)))?,
            },
            Optimization::Balanced=>match source.source {
                GrammarSource::Glrm(_)|GrammarSource::Constructed|GrammarSource::Ebnf(_)=>crate::import::compile_component_shell_balanced(source.component()?,vocab,&[])?,
                _=>source.compile_with(vocab,BuildOptions::default().optimization(Optimization::Balanced))?,
            },
            Optimization::Auto|Optimization::FastRuntime=>{
                let mut p=source.prepare(vocab)?;
                let inner=Arc::make_mut(&mut p.inner);
                crate::compiler::boundary_candidates::persist_boundary_candidate_summary(inner,vocab);
                inner.install_template_parser_from_compile()?;
                inner.build_boundary_trigger(options.boundary_trigger).map_err(Error::Compilation)?;
                p.boundary_trigger=options.boundary_trigger;
                return Ok(p);
            }
        };
        let _=inner.late_bind_vocab.set(vocab.clone());
        if options.optimization==Optimization::Balanced {inner.install_template_parser_from_compile()?;}
        inner.build_boundary_trigger(options.boundary_trigger).map_err(Error::Compilation)?;
        let prepared=PreparedGrammar {inner:Arc::new(inner),bindings:OrdMap::new(),analysis:graph::empty_cache(),
            event_adapters:Arc::new(event_cache::ExactAdapterCache::default()),boundary_trigger:options.boundary_trigger,
            slot_kinds:Arc::new(declared.kinds.clone()),slot_spans:Arc::new(declared.spans.clone())};
        prepared.validate_local_slot_manifest(vocab)?;prepared.validate_slot_manifest()?;Ok(prepared)
    }
    fn as_prepared(&self)->Result<PreparedGrammar> {
        let mut local=self.node.prepared.as_ref().ok_or_else(||error("module is not prepared"))?.as_ref().clone();
        for (name,value) in &self.node.bindings {
            match value {
                ModuleValue::Module(child)=>local.bind(name,child.as_prepared()?)?,
                ModuleValue::Token(t)=>local.bind(name,*t)?, ModuleValue::ExactTokens(t)=>local.bind(name,t.clone())?,
            }
        }
        Ok(local)
    }
    pub fn compile(&self,vocab:&Vocab)->Result<RuntimeConstraint> { self.compile_with(vocab,CompileOptions::default()) }
    /// Complete pending local compilation and linking; root options do not change local optimization.
    pub fn compile_with(&self,vocab:&Vocab,options:CompileOptions)->Result<RuntimeConstraint> {
        if let Some(slot)=self.unbound()?.first(){return Err(error(format!("external {:?} is unbound",slot.name())));}
        self.validate_targets(Some(vocab))?;
        let modules=self.index()?;
        let mut expanded=HashMap::<usize,usize>::new();
        for module in &modules {
            let mut count=1usize;
            for value in module.node.bindings.values(){if let ModuleValue::Module(child)=value {
                count=count.saturating_add(expanded[&(Arc::as_ptr(&child.node) as usize)]);
            }}
            expanded.insert(Arc::as_ptr(&module.node) as usize,count);
        }
        if expanded[&(Arc::as_ptr(&self.node) as usize)]>graph::MAX_NODES {return Err(error("expanded Module runtime construction exceeds its limit"));}
        for module in modules {
            let has_lr=module.node.options.optimization==Optimization::FastBuild;
            for value in module.node.bindings.values() {
                if has_lr || matches!(value,ModuleValue::Module(child) if child.node.options.optimization==Optimization::FastBuild) {
                    return Err(error("compiled Module composition requires native components; retained FastBuild/LR components cannot be mixed or converted"));
                }
            }
        }
        let mut prepared=self.clone();prepared.precompile(vocab)?;
        let constraint=prepared.materialize(vocab,&mut HashMap::new())?;
        ensure_runnable_constraint(&constraint)?;constraint.with_end_tokens(&options.end_tokens)
    }
    fn materialize(&self,vocab:&Vocab,cache:&mut HashMap<usize,RuntimeConstraint>)->Result<RuntimeConstraint> {
        let id=Arc::as_ptr(&self.node) as usize;
        if let Some(c)=cache.get(&id){return Ok(c.clone());}
        let local=self.node.prepared.as_ref().ok_or_else(||error("missing local compilation"))?;
        // Includes compiler-generated Lark schema attachments without discarding their work.
        let mut parent=if local.bindings.is_empty(){local.inner.as_ref().clone()}
            else {local.link_with(BuildOptions::default().optimization(self.node.options.optimization))?};
        if self.node.options.boundary_trigger!=BoundaryTriggerDetail::None {
            // Decode retained query metadata; loading may keep it in the serialized section.
            // This restores existing work and never invokes a compiler.
            parent.materialize_composition_link_metadata_for_compilation().map_err(Error::Compilation)?;
        }
        let active=parent.late_grammar_slots.iter().map(|s|s.name.as_str()).collect::<BTreeSet<_>>();
        let children=self.node.bindings.iter().filter(|(name,_)|active.contains(name.as_str())).map(|(name,value)|{
            let child=match value {
                ModuleValue::Module(child)=>child.materialize(vocab,cache)?,
                ModuleValue::Token(t)=>Grammar::exact_token(*t).compile(vocab)?,
                ModuleValue::ExactTokens(t)=>Grammar::exact(t.clone()).compile(vocab)?,
            };
            if !parent.has_template_parser() || !child.has_template_parser() {
                return Err(error("compiled Module composition requires native components; retained FastBuild/LR components cannot be mixed or converted"));
            }
            Ok((name.clone(),Arc::new(child)))
        }).collect::<Result<Vec<_>>>()?;
        let linked=!children.is_empty();
        if linked {
            parent=crate::error::catch_internal_invariant(||crate::runtime::parser_backend::link::compose_owned(parent,children,vocab))??;
            if matches!(self.node.options.optimization,Optimization::Auto|Optimization::FastRuntime) {
                crate::compiler::template_boundary::install(&mut parent,vocab)?;
            }
        }
        if linked {apply_final_boundary_trigger(&mut parent,self.node.options.boundary_trigger)?;}
        cache.insert(id,parent.clone());Ok(parent)
    }
    pub(super) fn to_source(&self)->Result<Grammar<'static>> {
        if self.node.prepared.is_some() || self.node.options!=ModuleOptions::default() {
            return Err(error("Grammar::from_expr requires source-only operands; use Module::from_expr for configured/precompiled operands"));
        }
        let mut grammar=self.node.source.as_ref().clone();
        for (name,value) in &self.node.bindings {
            match value {
                ModuleValue::Module(child)=>grammar.bind(name,child.to_source()?)?,
                ModuleValue::Token(t)=>grammar.bind(name,*t)?,ModuleValue::ExactTokens(t)=>grammar.bind(name,t.clone())?,
            }
        }
        Ok(grammar)
    }
    pub fn save(&self)->Vec<u8> { artifact::save(self) }
    pub fn load(bytes:impl AsRef<[u8]>)->Result<Self> { artifact::load(bytes.as_ref(),None) }
    pub fn load_with_vocab(bytes:impl AsRef<[u8]>,vocab:&Vocab)->Result<Self> {artifact::load(bytes.as_ref(),Some(vocab))}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn child(module:&Module,name:&str)->Module {match module.node.bindings.get(name).unwrap(){ModuleValue::Module(m)=>m.clone(),_=>panic!("not a module")}}
    #[test]
    fn local_work_is_shared_while_occurrence_environments_are_independent() {
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"b".to_vec()),(2,b"ab".to_vec())]);
        let g=Grammar::from_glrm("glrm 1; start s; extern grammar item; nt s = item;");
        assert!(Arc::ptr_eq(&g.module().node,&g.module().node),"Grammar::module snapshots are cheap");
        let mut m=Module::from_expr(&Expr::sequence([&g,&g])).unwrap();
        m.precompile(&v).unwrap();
        let a=child(&m,"leaf0");let b=child(&m,"leaf1");
        assert!(Arc::ptr_eq(a.node.prepared.as_ref().unwrap(),b.node.prepared.as_ref().unwrap()));
        let retained=a.node.prepared.as_ref().unwrap().clone();
        m.bind("leaf0::item",Grammar::from_ebnf("start ::= \"a\"")).unwrap();
        m.bind("leaf1::item",Grammar::from_ebnf("start ::= \"b\"")).unwrap();
        m.precompile(&v).unwrap();
        assert!(Arc::ptr_eq(&retained,child(&m,"leaf0").node.prepared.as_ref().unwrap()));
        assert!(Arc::ptr_eq(&retained,child(&m,"leaf1").node.prepared.as_ref().unwrap()));
        let loaded=Module::load(m.save()).unwrap();
        let a=child(&loaded,"leaf0");let b=child(&loaded,"leaf1");
        assert!(Arc::ptr_eq(&a.node.source,&b.node.source));
        assert!(Arc::ptr_eq(a.node.prepared.as_ref().unwrap(),b.node.prepared.as_ref().unwrap()));
        assert_ne!(child(&a,"item").node.source.source,child(&b,"item").node.source.source);
    }
    #[test]
    fn source_snapshot_cache_is_invalidated_by_binding_and_never_stores_configuration() {
        let mut g=Grammar::from_glrm("glrm 1; start s; extern grammar item; nt s = item;");
        let before=g.module();g.bind("item",Grammar::from_ebnf("start ::= \"a\"")).unwrap();
        let after=g.module();assert!(!Arc::ptr_eq(&before.node,&after.node));assert_eq!(before.unbound().unwrap().len(),1);
        assert!(after.unbound().unwrap().is_empty());
        let changed=after.with_options(ModuleOptions::default().optimization(Optimization::Balanced)).unwrap();
        assert_eq!(g.module().options(),&ModuleOptions::default());assert_ne!(changed.options(),g.module().options());
    }
    #[test]
    fn boundary_options_build_and_roundtrip_local_work_without_leaking_settings() {
        let v=Vocab::new(vec![(0,b"a".to_vec()),(1,b"ab".to_vec())]);
        let g=Grammar::from_ebnf("start ::= \"a\"");
        assert!(g.module().with_options(ModuleOptions::default().optimization(Optimization::FastBuild).boundary_trigger(BoundaryTriggerDetail::Exact)).is_err());
        for mode in [Optimization::Auto,Optimization::Balanced,Optimization::FastRuntime,Optimization::FastBuild] {
            let mut m=g.module().with_options(ModuleOptions::default().optimization(mode).boundary_trigger(BoundaryTriggerDetail::Tokens)).unwrap();
            m.precompile(&v).unwrap();let p=m.node.prepared.as_ref().unwrap();
            assert!(p.inner.boundary_trigger.token_summary().is_some());
            let loaded=Module::load(m.save()).unwrap();
            let mut retained=loaded.node.prepared.as_ref().unwrap().inner.as_ref().clone();
            retained.materialize_composition_link_metadata_for_compilation().unwrap();
            assert!(retained.boundary_trigger.token_summary().is_some());
            assert!(loaded.materialize(&v,&mut HashMap::new()).unwrap().boundary_trigger.token_summary().is_some());
        }
        for mode in [Optimization::Auto,Optimization::Balanced,Optimization::FastRuntime] {
            let mut m=g.module().with_options(ModuleOptions::default().optimization(mode).boundary_trigger(BoundaryTriggerDetail::Exact)).unwrap();
            m.precompile(&v).unwrap();
            assert!(matches!(m.node.prepared.as_ref().unwrap().inner.boundary_trigger,crate::runtime::BoundaryTrigger::Exact(_)));
            let loaded=Module::load(m.save()).unwrap();
            assert!(matches!(loaded.materialize(&v,&mut HashMap::new()).unwrap().boundary_trigger,crate::runtime::BoundaryTrigger::Exact(_)));
        }
        assert_eq!(g.module().options().boundary_trigger_value(),BoundaryTriggerDetail::None);
    }

}
