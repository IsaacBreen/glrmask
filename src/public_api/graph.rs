//! Bounded graph analysis separates shared instances from expanded public paths.
//! A closed DAG is visited once per actual node, even if it has exponentially
//! many root-to-leaf paths. Ordinary recursive grammar rules are not graph edges.
use super::*;
use std::collections::{HashMap, HashSet};

pub(super) const MAX_DEPTH: usize = 64;
pub(super) const MAX_NODES: usize = 131_072;
const MAX_PORTS: usize = 1_048_576;
const MAX_REQUIREMENTS: usize = 65_536;
const MAX_EXPANDED_INSTANCES: usize = 131_072;

pub(super) struct View<'a, T> {
    pub(super) node: &'a T,
    local: Vec<Requirement>,
    pub(super) children: Vec<(&'a str, &'a T)>,
    ports: usize,
}
pub(super) struct Entry<'a, T> {
    pub(super) view: View<'a, T>,
    height: usize,
    first: Option<Requirement>,
    expanded: usize,
}
pub(super) struct Index<'a, T> {
    pub(super) nodes: HashMap<usize, Entry<'a, T>>,
    pub(super) order: Vec<usize>,
    root: usize,
}
pub(super) trait InstanceNode { fn identity(&self) -> usize; }
impl InstanceNode for Grammar { fn identity(&self)->usize { source_identity(self) } }
impl InstanceNode for PreparedGrammar { fn identity(&self)->usize { prepared_identity(self) } }
fn key<T:InstanceNode>(node:&T)->usize { node.identity() }
pub(super) fn source_identity(node:&Grammar)->usize { Arc::as_ptr(&node.analysis) as usize }
pub(super) fn prepared_identity(node:&PreparedGrammar)->usize { Arc::as_ptr(&node.analysis) as usize }
fn limit(message: &str) -> Error {
    Error::Compilation(message.to_string())
}
fn prepend(prefix: &str, child: &Requirement) -> Requirement {
    let mut path = prefix.split("::").map(str::to_owned).collect::<Vec<_>>();
    path.extend(child.path.iter().cloned());
    Requirement { path, kind: child.kind, source_span: child.source_span }
}

fn analyze<'a, T:InstanceNode>(root: &'a T, describe: impl Fn(&'a T) -> Result<View<'a, T>>) -> Result<Index<'a, T>> {
    fn visit<'a, T:InstanceNode>(node: &'a T, depth: usize, describe: &impl Fn(&'a T) -> Result<View<'a, T>>,
        index: &mut Index<'a, T>, active: &mut HashSet<usize>, ports: &mut usize) -> Result<()> {
        if depth > MAX_DEPTH { return Err(limit("grammars are nested more than 64 levels deep")); }
        let id = key(node);
        if let Some(entry) = index.nodes.get(&id) {
            if depth + entry.height > MAX_DEPTH { return Err(limit("grammars are nested more than 64 levels deep")); }
            return Ok(());
        }
        if !active.insert(id) { return Err(Error::Compilation("a module is bound inside itself".into())); }
        if index.nodes.len() + active.len() > MAX_NODES { return Err(limit("the composition has too many parts")); }
        let view = describe(node)?;
        *ports = ports.checked_add(view.ports).ok_or_else(||limit("port count overflow"))?;
        if *ports > MAX_PORTS { return Err(limit("the composition has too many externs")); }
        let mut height = 0;
        let mut first = view.local.first().cloned();
        let mut expanded = 1usize;
        for &(name, child) in &view.children {
            visit(child, depth + 1, describe, index, active, ports)?;
            let summary = &index.nodes[&key(child)];
            height = height.max(summary.height + 1);
            expanded = expanded.saturating_add(summary.expanded).min(MAX_EXPANDED_INSTANCES + 1);
            if first.is_none() { first = summary.first.as_ref().map(|r|prepend(name,r)); }
        }
        active.remove(&id);
        index.nodes.insert(id, Entry { view, height, first, expanded });
        index.order.push(id);
        Ok(())
    }
    let mut index = Index { nodes: HashMap::new(), order: Vec::new(), root: key(root) };
    visit(root, 0, &describe, &mut index, &mut HashSet::new(), &mut 0)?;
    Ok(index)
}
impl<T:InstanceNode> Index<'_, T> {
    fn first(&self) -> Option<Requirement> { self.nodes[&self.root].first.clone() }
    fn requirements(&self) -> Result<Vec<Requirement>> {
        if self.nodes[&self.root].first.is_none() { return Ok(Vec::new()); }
        let mut result = Vec::new();
        let mut pending = vec![(self.root, Vec::<String>::new())];
        let mut visited = 0usize;
        while let Some((id, prefix)) = pending.pop() {
            visited += 1;
            if visited > MAX_NODES { return Err(limit("the composition expands to too many parts")); }
            let entry = &self.nodes[&id];
            for local in &entry.view.local {
                if result.len() == MAX_REQUIREMENTS { return Err(limit("the composition expands to too many parts")); }
                let mut requirement = local.clone();
                requirement.path.splice(0..0, prefix.iter().cloned());
                result.push(requirement);
            }
            for &(name, child) in &entry.view.children {
                let child_id = key(child);
                if self.nodes[&child_id].first.is_none() { continue; }
                let mut path = prefix.clone(); path.extend(name.split("::").map(str::to_owned));
                pending.push((child_id,path));
            }
        }
        result.sort_by(|a,b|a.path.cmp(&b.path));
        Ok(result)
    }
    fn expansion_preflight(&self) -> Result<()> {
        if self.nodes[&self.root].expanded > MAX_EXPANDED_INSTANCES {
            return Err(limit("compiling would expand the composition to more than 131072 parts"));
        }
        Ok(())
    }
}

fn source_index<'a>(root: &'a Grammar) -> Result<Index<'a, Grammar>> {
    analyze(root, |grammar| {
        // One descriptor lookup per actual instance, never one per output path.
        let declared = grammar.declarations()?;
        let local = declared.kinds.iter().filter(|(name,_)|!grammar.bindings.contains_key(*name))
            .map(|(name,kind)|Requirement { path:name.split("::").map(str::to_owned).collect(), kind:*kind,
                source_span:declared.spans.get(name).copied() }).collect();
        let children = grammar.bindings.iter().filter_map(|(name,binding)|match binding {
            GrammarValue::Source(child) => Some((name.as_str(),child.as_ref())), _ => None,
        }).collect();
        Ok(View { node:grammar,local,children,ports:declared.kinds.len() })
    })
}
pub(super) fn source_requirements(root: &Grammar) -> Result<Vec<Requirement>> { source_summary(root)?.requirements() }
pub(super) fn source_first(root: &Grammar) -> Result<Option<Requirement>> { Ok(source_summary(root)?.first()) }
pub(super) fn source_preflight(root: &Grammar) -> Result<()> { source_summary(root)?.expansion_preflight() }

fn prepared_for_vocab<'a>(root: &'a PreparedGrammar, vocab: &Vocab) -> Result<Index<'a, PreparedGrammar>> {
    analyze(root, |module| {
        module.validate_local_slot_manifest(vocab)?;
        let local = module.slot_kinds.iter().filter(|(name,_)|!module.bindings.contains_key(*name))
            .map(|(name,kind)|Requirement { path:name.split("::").map(str::to_owned).collect(),kind:*kind,
                source_span:module.slot_spans.get(name).copied() }).collect();
        let children = module.bindings.iter().filter_map(|(name,binding)|match binding {
            ModuleBinding::Module(child) => Some((name.as_str(),child.as_ref())), _=>None,
        }).collect();
        Ok(View { node:module,local,children,ports:module.slot_kinds.len() })
    })
}
pub(super) fn prepared_index(root:&PreparedGrammar)->Result<Index<'_,PreparedGrammar>> {
    prepared_for_vocab(root,&constraint_vocab(root.inner.as_ref()))
}
pub(super) fn prepared_requirements(root:&PreparedGrammar)->Result<Vec<Requirement>> { prepared_summary(root)?.requirements() }
pub(super) fn prepared_first(root:&PreparedGrammar)->Result<Option<Requirement>> { Ok(prepared_summary(root)?.first()) }
pub(super) fn prepared_targets(root:&PreparedGrammar,vocab:&Vocab)->bool { static_constraint_targets(root.inner.as_ref(),vocab) && prepared_summary(root).is_ok() }
pub(super) fn prepared_preflight(root:&PreparedGrammar)->Result<()> { prepared_summary(root)?.expansion_preflight() }


// Published snapshots have immutable topology and immutable definition data.
// Their certificates are shared by cheap public clones. Binding constructs a
// new certificate by changing only one local slot or one ancestor spine. The
// conservative whole-graph validator above remains the serialization/trust
// boundary and the fallback for exact unique-node resource accounting.
pub(super) type SummaryCache = Arc<OnceLock<Result<Arc<Summary>>>>;
pub(super) fn empty_cache()->SummaryCache { Arc::new(OnceLock::new()) }
pub(super) fn filled_cache(summary:Summary)->SummaryCache {
    let cache=OnceLock::new(); let _=cache.set(Ok(Arc::new(summary))); Arc::new(cache)
}

#[derive(Clone,Debug)]
pub(super) struct Summary {
    local: im::OrdMap<String,Arc<Requirement>>,
    children: im::OrdMap<String,Arc<Summary>>,
    open_children: im::OrdMap<String,Arc<Summary>>,
    heights: im::OrdMap<usize,usize>,
    direct_ports: usize,
    // These are occurrence counts and therefore safe upper bounds for unique
    // DAG nodes/ports. Never reject sharing on an upper bound alone.
    instances: u128,
    ports: u128,
    requirements_cache: Arc<OnceLock<Result<Arc<[Requirement]>>>>,
}
impl Summary {
    fn new(kinds:&BTreeMap<String,ExternKind>,spans:&BTreeMap<String,glrmask_grammar::SourceSpan>)->Self {
        Self {
            local:kinds.iter().map(|(name,&kind)| (name.clone(),Arc::new(Requirement {
                path:name.split("::").map(str::to_owned).collect(),kind,source_span:spans.get(name).copied(),
            }))).collect(),
            children:im::OrdMap::new(),open_children:im::OrdMap::new(),heights:im::OrdMap::new(),
            direct_ports:kinds.len(),instances:1,ports:kinds.len() as u128,
            requirements_cache:Arc::new(OnceLock::new()),
        }
    }
    fn closed(&self)->bool { self.local.is_empty() && self.open_children.is_empty() }
    fn height(&self)->usize { self.heights.iter().next_back().map_or(0,|(&height,_)|height+1) }
    fn adjust_height(&mut self,height:usize,add:bool) {
        let count=self.heights.get(&height).copied().unwrap_or(0);
        if add {self.heights.insert(height,count+1);}
        else if count<=1 {self.heights.remove(&height);}
        else {self.heights.insert(height,count-1);}
    }
    pub(super) fn changed(&self,name:&str,child:Option<Arc<Summary>>)->Self {
        let mut next=self.clone();
        next.requirements_cache=Arc::new(OnceLock::new());
        next.local.remove(name);
        let old=next.children.remove(name);
        next.open_children.remove(name);
        if let Some(old)=&old {next.adjust_height(old.height(),false);}
        if let Some(child)=&child {
            next.adjust_height(child.height(),true);
            next.children.insert(name.to_owned(),child.clone());
            if !child.closed() {next.open_children.insert(name.to_owned(),child.clone());}
        }
        // Updating an unsaturated aggregate is O(1). Very large shared DAGs
        // may saturate; recompute from relative children rather than subtract
        // an inexact count. Closing requirements does not alter topology.
        let old_nodes=old.as_ref().map_or(0,|c|c.instances);
        let old_ports=old.as_ref().map_or(0,|c|c.ports);
        let new_nodes=child.as_ref().map_or(0,|c|c.instances);
        let new_ports=child.as_ref().map_or(0,|c|c.ports);
        next.instances=if self.instances==u128::MAX || old_nodes==u128::MAX {
            next.children.values().fold(1u128,|sum,c|sum.saturating_add(c.instances))
        } else {self.instances.saturating_sub(old_nodes).saturating_add(new_nodes)};
        next.ports=if self.ports==u128::MAX || old_ports==u128::MAX {
            next.children.values().fold(next.direct_ports as u128,|sum,c|sum.saturating_add(c.ports))
        } else {self.ports.saturating_sub(old_ports).saturating_add(new_ports)};
        next
    }
    fn first(&self)->Option<Requirement> {
        self.local.iter().next().map(|(_,r)|r.as_ref().clone()).or_else(|| {
            self.open_children.iter().next().and_then(|(name,child)|child.first().map(|r|prepend(name,&r)))
        })
    }
    fn check_height(&self)->Result<()> {
        if self.height()>MAX_DEPTH {Err(limit("grammars are nested more than 64 levels deep"))} else {Ok(())}
    }
    fn expansion_preflight(&self)->Result<()> {
        if self.instances>MAX_EXPANDED_INSTANCES as u128 {
            Err(limit("compiling would expand the composition to more than 131072 parts"))
        } else {Ok(())}
    }
    fn requirements(&self)->Result<Vec<Requirement>> {
        if self.closed(){return Ok(Vec::new());}
        let result=self.requirements_cache.get_or_init(|| {
            let mut result=Vec::new();let mut pending=vec![(self,Vec::<String>::new())];let mut visited=0usize;
            while let Some((node,prefix))=pending.pop() {
                visited+=1;
                if visited>MAX_NODES {return Err(limit("the composition expands to too many parts"));}
                for local in node.local.values() {
                    if result.len()==MAX_REQUIREMENTS {return Err(limit("the composition expands to too many parts"));}
                    let mut requirement=local.as_ref().clone();
                    requirement.path.splice(0..0,prefix.iter().cloned());result.push(requirement);
                }
                for (name,child) in &node.open_children {
                    let mut path=prefix.clone();path.extend(name.split("::").map(str::to_owned));
                    pending.push((child.as_ref(),path));
                }
            }
            result.sort_by(|a,b|a.path.cmp(&b.path));Ok(Arc::from(result))
        });
        result.as_ref().map(|r|r.to_vec()).map_err(Clone::clone)
    }
}

pub(super) fn source_summary(root:&Grammar)->Result<Arc<Summary>> {
    root.analysis.get_or_init(|| {
        let declared=root.declarations()?;
        let mut summary=Summary::new(&declared.kinds,&declared.spans);
        for (name,value) in &root.bindings {
            let kind=*declared.kinds.get(name).ok_or_else(||Error::Compilation(format!("unknown source binding {name:?}")))?;
            check_binding_kind(name,kind,matches!(value,GrammarValue::Source(_)))?;
            let child=match value {GrammarValue::Source(c)=>Some(source_summary(c)?),_=>None};
            summary=summary.changed(name,child);
        }
        summary.check_height()?;
        if summary.instances>MAX_NODES as u128 || summary.ports>MAX_PORTS as u128 {source_index(root)?;}
        Ok(Arc::new(summary))
    }).clone()
}
pub(super) fn prepared_summary(root:&PreparedGrammar)->Result<Arc<Summary>> {
    root.analysis.get_or_init(|| {
        let vocab=constraint_vocab(root.inner.as_ref());
        root.validate_local_slot_manifest(&vocab)?;
        let mut summary=Summary::new(&root.slot_kinds,&root.slot_spans);
        for (name,value) in &root.bindings {
            let child=match value {
                ModuleBinding::Module(c)=> {
                    if !static_constraint_targets(c.inner.as_ref(),&vocab) {
                        return Err(Error::Compilation("the bound modules were precompiled for different vocabularies".into()));
                    }
                    Some(prepared_summary(c)?)
                },_=>None,
            };
            summary=summary.changed(name,child);
        }
        summary.check_height()?;
        if summary.instances>MAX_NODES as u128 || summary.ports>MAX_PORTS as u128 {prepared_index(root)?;}
        Ok(Arc::new(summary))
    }).clone()
}
pub(super) fn check_source_limits(root:&Grammar)->Result<()> {
    let summary=source_summary(root)?;summary.check_height()?;
    if summary.instances>MAX_NODES as u128 || summary.ports>MAX_PORTS as u128 {source_index(root)?;}
    Ok(())
}
pub(super) fn check_prepared_limits(root:&PreparedGrammar)->Result<()> {
    let summary=prepared_summary(root)?;summary.check_height()?;
    if summary.instances>MAX_NODES as u128 || summary.ports>MAX_PORTS as u128 {prepared_index(root)?;}
    Ok(())
}


#[cfg(test)]
mod snapshot_tests {
    use super::*;
    #[test]
    fn incremental_requirements_match_full_graph_after_every_update() {
        let text=(0..256).map(|i|format!("extern token E{i}\n")).collect::<String>()+"start root nt root=E0";
        let mut g=Grammar::from_glrm(&text).unwrap();
        let original=g.clone();
        for i in 0..256 {
            assert_eq!(g.requirements().unwrap(),source_index(&g).unwrap().requirements().unwrap());
            g.bind(&format!("E{i}"),TokenSet::from(7)).unwrap();
            assert_eq!(source_summary(&g).unwrap().closed(),i==255);
        }
        assert!(g.requirements().unwrap().is_empty());
        assert_eq!(original.requirements().unwrap().len(),256);
    }
    #[test]
    fn snapshot_child_spines_preserve_sibling_requirements() {
        let child=Grammar::from_glrm("extern token MARK start root nt root=MARK").unwrap();
        let mut p=Grammar::from_glrm("extern grammar LEFT extern grammar RIGHT start root nt root=LEFT RIGHT").unwrap();
        p.bind("LEFT",&child).unwrap();p.bind("RIGHT",&child).unwrap();
        let old=p.clone();p.bind("LEFT::MARK",TokenSet::from(17)).unwrap();
        assert_eq!(p.requirements().unwrap().iter().map(Requirement::name).collect::<Vec<_>>(),["RIGHT::MARK"]);
        assert_eq!(p.requirements().unwrap(),source_index(&p).unwrap().requirements().unwrap());
        assert_eq!(old.requirements().unwrap().len(),2);assert_eq!(child.requirements().unwrap().len(),1);
    }
    #[test]
    fn shared_exponential_graph_validates_unique_nodes_without_expanding_closed_paths() {
        let mut g=Grammar::from_glrm("start root nt root=eps").unwrap();
        for _ in 0..22 {
            let mut p=Grammar::from_glrm("extern grammar L extern grammar R start root nt root=L R").unwrap();
            p.bind("L",&g).unwrap();p.bind("R",&g).unwrap();g=p;
        }
        assert!(g.requirements().unwrap().is_empty());
        assert!(source_summary(&g).is_ok());
        assert_eq!(source_index(&g).unwrap().nodes.len(),23);
        assert!(source_preflight(&g).is_err(),"expanded runtime construction still bounded");
    }
    #[test]
    fn depth_error_is_atomic_and_does_not_invalidate_old_snapshot() {
        let mut g=Grammar::from_glrm("start root nt root=eps").unwrap();
        for _ in 0..MAX_DEPTH {
            let mut p=Grammar::from_glrm("extern grammar C start root nt root=C").unwrap();p.bind("C",&g).unwrap();g=p;
        }
        let mut p=Grammar::from_glrm("extern grammar C start root nt root=C").unwrap();
        assert!(p.bind("C",&g).is_err());
        assert_eq!(p.requirements().unwrap()[0].name(),"C");
        assert!(g.requirements().unwrap().is_empty());
    }
}
