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
fn key<T>(node: &T) -> usize { node as *const T as usize }
fn limit(message: &str) -> Error {
    Error::Compilation(format!("component graph resource limit: {message}"))
}
fn prepend(prefix: &str, child: &Requirement) -> Requirement {
    let mut path = prefix.split("::").map(str::to_owned).collect::<Vec<_>>();
    path.extend(child.path.iter().cloned());
    Requirement { path, kind: child.kind, source_span: child.source_span }
}

fn analyze<'a, T>(root: &'a T, describe: impl Fn(&'a T) -> Result<View<'a, T>>) -> Result<Index<'a, T>> {
    fn visit<'a, T>(node: &'a T, depth: usize, describe: &impl Fn(&'a T) -> Result<View<'a, T>>,
        index: &mut Index<'a, T>, active: &mut HashSet<usize>, ports: &mut usize) -> Result<()> {
        if depth > MAX_DEPTH { return Err(limit("attachment depth exceeds 64")); }
        let id = key(node);
        if let Some(entry) = index.nodes.get(&id) {
            if depth + entry.height > MAX_DEPTH { return Err(limit("attachment depth exceeds 64")); }
            return Ok(());
        }
        if !active.insert(id) { return Err(Error::Compilation("cyclic component attachment graph".into())); }
        if index.nodes.len() + active.len() > MAX_NODES { return Err(limit("too many distinct instances")); }
        let view = describe(node)?;
        *ports = ports.checked_add(view.ports).ok_or_else(||limit("port count overflow"))?;
        if *ports > MAX_PORTS { return Err(limit("too many declared ports")); }
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
impl<T> Index<'_, T> {
    fn first(&self) -> Option<Requirement> { self.nodes[&self.root].first.clone() }
    fn requirements(&self) -> Result<Vec<Requirement>> {
        if self.nodes[&self.root].first.is_none() { return Ok(Vec::new()); }
        let mut result = Vec::new();
        let mut pending = vec![(self.root, Vec::<String>::new())];
        let mut visited = 0usize;
        while let Some((id, prefix)) = pending.pop() {
            visited += 1;
            if visited > MAX_NODES { return Err(limit("expanded requirement traversal exceeds its limit")); }
            let entry = &self.nodes[&id];
            for local in &entry.view.local {
                if result.len() == MAX_REQUIREMENTS { return Err(limit("too many expanded requirement paths")); }
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
            return Err(limit("linking would expand more than 131072 component instances; the prepared DAG can still be saved or specialized"));
        }
        Ok(())
    }
}

fn source_index<'a, 's>(root: &'a Grammar<'s>) -> Result<Index<'a, Grammar<'s>>> {
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
pub(super) fn source_requirements(root: &Grammar<'_>) -> Result<Vec<Requirement>> { source_index(root)?.requirements() }
pub(super) fn source_first(root: &Grammar<'_>) -> Result<Option<Requirement>> { Ok(source_index(root)?.first()) }
pub(super) fn source_preflight(root: &Grammar<'_>) -> Result<()> { source_index(root)?.expansion_preflight() }

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
pub(super) fn prepared_requirements(root:&PreparedGrammar)->Result<Vec<Requirement>> { prepared_index(root)?.requirements() }
pub(super) fn prepared_first(root:&PreparedGrammar)->Result<Option<Requirement>> { Ok(prepared_index(root)?.first()) }
pub(super) fn prepared_targets(root:&PreparedGrammar,vocab:&Vocab)->bool { prepared_for_vocab(root,vocab).is_ok() }
pub(super) fn prepared_preflight(root:&PreparedGrammar)->Result<()> { prepared_index(root)?.expansion_preflight() }
