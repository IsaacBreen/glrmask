//! Prepared artifacts are flat DAGs, not recursively embedded byte blobs.
//! Definitions are stored once, instance environments separately, and references
//! point backwards in postorder. Loading cannot recurse through attacker-chosen
//! serialized component graphs or duplicate a shared definition per path.
use super::*;
use std::collections::HashMap;

const MAGIC: &[u8; 8] = b"GLRMOD15";
const FINGERPRINT_LEN: usize = 32;
const HEADER: usize = 16 + FINGERPRINT_LEN;

#[derive(serde::Serialize, serde::Deserialize)]
struct Definition {
    bytes: u64,
    runnable: bool,
    #[serde(deserialize_with="artifact_limits::items")]
    exact_only_token_ids: Vec<u32>,
}
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy)]
enum SlotKind { Token, Grammar }
impl From<ExternKind> for SlotKind {
    fn from(kind: ExternKind) -> Self {
        match kind { ExternKind::Token => Self::Token, ExternKind::Grammar => Self::Grammar, _ => unreachable!("every extern kind has a binding name") }
    }
}
impl From<SlotKind> for ExternKind {
    fn from(kind: SlotKind) -> Self {
        match kind { SlotKind::Token => Self::Token, SlotKind::Grammar => Self::Grammar }
    }
}
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy)]
struct Span { start: usize, end: usize, line: usize, column: usize }
impl From<glrmask_grammar::SourceSpan> for Span {
    fn from(span: glrmask_grammar::SourceSpan) -> Self {
        Self::new(span.start, span.end, span.line, span.column)
    }
}
impl From<Span> for glrmask_grammar::SourceSpan {
    fn from(span: Span) -> Self {
        Self { start: span.start, end: span.end, line: span.line, column: span.column }
    }
}
#[derive(serde::Serialize, serde::Deserialize)]
enum Binding {
    Prepared(u32),
    Runnable(u32),
    Events(#[serde(deserialize_with="artifact_limits::items")] Vec<(u32, u32)>),
}
#[derive(serde::Serialize, serde::Deserialize)]
struct Instance {
    definition: u32,
    slot_kinds: BTreeMap<String, SlotKind>,
    slot_spans: BTreeMap<String, Span>,
    boundary_trigger: u8,
    #[serde(deserialize_with="artifact_limits::items")]
    bindings: Vec<(String, Binding)>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct Manifest {
    #[serde(deserialize_with="artifact_limits::definitions")]
    definitions: Vec<Definition>,
    #[serde(deserialize_with="artifact_limits::nodes")]
    instances: Vec<Instance>,
}
fn error(message: impl Into<String>) -> Error { Error::Serialization(message.into()) }

/// Reject malformed topology before allocating or decoding any compiled body.
fn preflight(manifest: &Manifest) -> Result<()> {
    let mut used = vec![false; manifest.definitions.len()];
    let mut heights = Vec::<usize>::with_capacity(manifest.instances.len());
    let mut edges = Vec::<Vec<usize>>::with_capacity(manifest.instances.len());
    let mut ports = 0usize;
    for (index, instance) in manifest.instances.iter().enumerate() {
        ports = ports.checked_add(instance.slot_kinds.len()).ok_or_else(||error("prepared port count overflow"))?;
        if ports > 1_048_576 { return Err(error("too many prepared ports")); }
        let definition = instance.definition as usize;
        if definition >= used.len() || manifest.definitions[definition].runnable {
            return Err(error("invalid prepared definition reference"));
        }
        used[definition] = true;
        if instance.boundary_trigger > 2 { return Err(error("invalid prepared boundary-trigger policy")); }
        if instance.slot_spans.iter().any(|(name,span)|!instance.slot_kinds.contains_key(name)
            || span.start > span.end || span.line == 0 || span.column == 0) {
            return Err(error("invalid prepared requirement source coordinates"));
        }
        let mut names = std::collections::BTreeSet::new();
        let mut children = Vec::new(); let mut height = 0usize;
        for (name, binding) in &instance.bindings {
            let kind = instance.slot_kinds.get(name).ok_or_else(||error("unknown prepared binding"))?;
            if !names.insert(name) { return Err(error("duplicate prepared binding")); }
            check_binding_kind(name,(*kind).into(),!matches!(binding,Binding::Events(_))).map_err(|e|error(e.to_string()))?;
            match binding {
                Binding::Prepared(child) => {
                    let child = *child as usize;
                    if child >= index { return Err(error("cyclic or forward prepared instance reference")); }
                    height = height.max(heights[child] + 1); children.push(child);
                }
                Binding::Runnable(definition) => {
                    let definition = *definition as usize;
                    if definition >= used.len() || !manifest.definitions[definition].runnable {
                        return Err(error("invalid runnable definition reference"));
                    }
                    used[definition] = true;
                }
                Binding::Events(_) => {}
            }
        }
        if height > graph::MAX_DEPTH { return Err(error("grammars are nested more than 64 levels deep")); }
        heights.push(height); edges.push(children);
    }
    if used.iter().any(|used|!used) { return Err(error("unreferenced compiled definitions in prepared artifact")); }
    let mut reached = vec![false; edges.len()]; let mut pending = vec![edges.len() - 1];
    while let Some(id) = pending.pop() {
        if reached[id] { continue; }
        reached[id] = true; pending.extend(edges[id].iter().copied());
    }
    if reached.iter().any(|seen|!seen) { return Err(error("unreachable instances in prepared artifact")); }
    Ok(())
}

pub(super) fn save(root: &PreparedGrammar) -> Vec<u8> {
    let graph = graph::prepared_index(root).expect("prepared instances are validated when constructed");
    let mut definitions = Vec::new();
    let mut bodies = Vec::<Vec<u8>>::new();
    let mut ids = HashMap::<(usize, bool), u32>::new();
    let mut fingerprint = None;
    let mut definition = |value: &Arc<RuntimeConstraint>, runnable: bool| -> u32 {
        let key = (Arc::as_ptr(value) as usize, runnable);
        if let Some(&id) = ids.get(&key) { return id; }
        let id = definitions.len() as u32;
        let body = value.save_body_external_artifact();
        let recorded = value.vocab_fingerprint();
        match fingerprint {
            Some(expected) => assert_eq!(
                expected, recorded,
                "prepared definitions must target one vocabulary"
            ),
            None => fingerprint = Some(recorded),
        }
        definitions.push(Definition {
            bytes: body.len() as u64,
            runnable,
            exact_only_token_ids: if runnable { Vec::new() }
                else { constraint_vocab(value).special_token_ids().collect() },
        });
        bodies.push(body); ids.insert(key, id); id
    };
    let mut instances = Vec::with_capacity(graph.order.len());
    let mut instance_ids = HashMap::<usize, u32>::new();
    for id in &graph.order {
        let module = graph.nodes[id].view.node;
        let body = definition(&module.inner, false);
        let bindings = module.bindings.iter().map(|(name, value)| {
            let value = match value {
                ModuleBinding::Module(child) => Binding::Prepared(instance_ids[&(graph::prepared_identity(child))]),
                ModuleBinding::Constraint(child) => Binding::Runnable(definition(child, true)),
                ModuleBinding::TokenSet(ids) => Binding::Events(ids.to_vec()),
            };
            (name.clone(), value)
        }).collect();
        instance_ids.insert(*id, instances.len() as u32);
        instances.push(Instance {
            definition: body,
            slot_kinds: module.slot_kinds.iter().map(|(name, &kind)| (name.clone(), kind.into())).collect(),
            slot_spans: module.slot_spans.iter().map(|(name, &span)| (name.clone(), span.into())).collect(),
            boundary_trigger: match module.boundary_trigger {
                BoundaryTriggerDetail::None => 0, BoundaryTriggerDetail::TokenSet => 1, BoundaryTriggerDetail::Exact => 2,
            },
            bindings,
        });
    }
    let manifest = bincode::serialize(&Manifest { definitions, instances }).expect("prepared manifest is serializable");
    let fingerprint = fingerprint.expect("prepared artifact has at least one definition");
    let mut result = Vec::with_capacity(HEADER + manifest.len() + bodies.iter().map(Vec::len).sum::<usize>());
    result.extend_from_slice(MAGIC); result.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
    result.extend_from_slice(&fingerprint);
    result.extend_from_slice(&manifest);
    for body in bodies { result.extend_from_slice(&body); }
    result
}

pub(super) fn load(data: &[u8], supplied: Option<&Vocab>) -> Result<PreparedGrammar> {
    use bincode::Options;
    if data.starts_with(b"GLRMOD") && !data.starts_with(MAGIC) {
        return Err(error("unsupported prepared-grammar artifact version"));
    }
    if data.len() < HEADER || !data.starts_with(MAGIC) { return Err(error("invalid prepared-grammar artifact header")); }
    let mut recorded_fingerprint = [0u8; FINGERPRINT_LEN];
    recorded_fingerprint.copy_from_slice(&data[16..HEADER]);
    let manifest_len = u64::from_le_bytes(data[8..16].try_into().unwrap());
    if data.len()>1<<30 || manifest_len>1<<30 {return Err(error("excessive prepared artifact length"));}
    let start = usize::try_from(manifest_len).ok().and_then(|n|n.checked_add(HEADER))
        .filter(|&n|n<=data.len()).ok_or_else(||error("invalid prepared manifest length"))?;
    let manifest: Manifest = bincode::DefaultOptions::new().with_fixint_encoding().with_limit(manifest_len)
        .reject_trailing_bytes().deserialize(&data[HEADER..start])
        .map_err(|e|error(format!("invalid prepared manifest: {e}")))?;
    if manifest.instances.is_empty() || manifest.instances.len()>graph::MAX_NODES
        || manifest.definitions.is_empty() || manifest.definitions.len()>graph::MAX_NODES * 2 {
        return Err(error("invalid or excessive prepared graph size"));
    }
    preflight(&manifest)?;
    let mut offset = start;
    let mut slices = Vec::with_capacity(manifest.definitions.len());
    for definition in &manifest.definitions {
        let end=usize::try_from(definition.bytes).ok().and_then(|n|offset.checked_add(n))
            .filter(|&n|n<=data.len()).ok_or_else(||error("invalid prepared definition length"))?;
        slices.push(offset..end); offset=end;
    }
    if offset!=data.len() { return Err(error("trailing data in prepared-grammar artifact")); }
    let mut definitions=Vec::<Arc<RuntimeConstraint>>::with_capacity(slices.len());
    for (definition,range) in manifest.definitions.iter().zip(slices) {
        let bytes=&data[range];
        let inner=match supplied {
            Some(vocab) => RuntimeConstraint::load_body_external_artifact(bytes, vocab)?,
            None => {
                let (inner, fingerprint) =
                    RuntimeConstraint::load_deferred_body(bytes, definition.runnable)?;
                if fingerprint != recorded_fingerprint {
                    return Err(error("prepared definition targets a different vocabulary"));
                }
                inner
            }
        };
        if definition.runnable {
            if !definition.exact_only_token_ids.is_empty() { return Err(error("runnable definition has duplicate event-domain metadata")); }
            if !inner.late_grammar_slots.is_empty() {
                return Err(error("runnable definition has unresolved slots"));
            }
        } else {
            let mut previous=None;
            for &id in &definition.exact_only_token_ids {
                if previous.is_some_and(|p|p>=id) {
                    return Err(error("invalid prepared exact-only token domain"));
                }
                if inner.token_bytes_for_id(id).is_some() {
                    return Err(error("invalid prepared exact-only token domain"));
                }
                previous=Some(id);
            }
        }
        definitions.push(Arc::new(inner));
    }
    let event_caches=(0..definitions.len()).map(|_|Arc::new(event_cache::ExactAdapterCache::default())).collect::<Vec<_>>();
    let mut instances=Vec::<Arc<PreparedGrammar>>::with_capacity(manifest.instances.len());
    let mut heights=Vec::<usize>::with_capacity(manifest.instances.len());
    let mut edges=Vec::<Vec<usize>>::with_capacity(manifest.instances.len());
    let mut ports=0usize;
    for (index, instance) in manifest.instances.into_iter().enumerate() {
        ports=ports.checked_add(instance.slot_kinds.len()).ok_or_else(||error("prepared port count overflow"))?;
        if ports>1_048_576 { return Err(error("too many prepared ports")); }
        let def=instance.definition as usize;
        if def>=definitions.len() || manifest.definitions[def].runnable { return Err(error("invalid prepared definition reference")); }
        let mut bindings=OrdMap::new();let mut children=Vec::new();let mut height=0;
        for (name,binding) in instance.bindings {
            if !instance.slot_kinds.contains_key(&name) || bindings.contains_key(&name) { return Err(error("unknown or duplicate prepared binding")); }
            let value=match binding {
                Binding::Prepared(child) => {
                    let child=child as usize;
                    if child>=index { return Err(error("cyclic or forward prepared instance reference")); }
                    height=height.max(heights[child]+1);children.push(child);
                    ModuleBinding::Module(instances[child].clone())
                }
                Binding::Runnable(definition) => {
                    let definition=definition as usize;
                    if definition>=definitions.len() || !manifest.definitions[definition].runnable { return Err(error("invalid runnable definition reference")); }
                    ModuleBinding::Constraint(definitions[definition].clone())
                }
                Binding::Events(ranges) => ModuleBinding::TokenSet(event_caches[def].leaf(&ranges)?),
            };
            bindings.insert(name,value);
        }
        if height>graph::MAX_DEPTH { return Err(error("grammars are nested more than 64 levels deep")); }
        let boundary_trigger=match instance.boundary_trigger {0=>BoundaryTriggerDetail::None,1=>BoundaryTriggerDetail::TokenSet,
            2=>BoundaryTriggerDetail::Exact,_=>return Err(error("invalid prepared boundary-trigger policy"))};
        let slot_kinds = instance.slot_kinds.into_iter().map(|(name, kind)| (name, kind.into())).collect();
        let slot_spans = instance.slot_spans.into_iter().map(|(name, span)| (name, span.into())).collect();
        instances.push(Arc::new(PreparedGrammar {inner:definitions[def].clone(),bindings,boundary_trigger,
            slot_kinds:Arc::new(slot_kinds),slot_spans:Arc::new(slot_spans),analysis:graph::empty_cache(),event_adapters:event_caches[def].clone(),
            retained_artifact: None, retained_vocab_fingerprint: None}));
        heights.push(height);edges.push(children);
    }
    let mut reached=vec![false;instances.len()];let mut pending=vec![instances.len()-1];
    while let Some(id)=pending.pop() {if reached[id]{continue;}reached[id]=true;pending.extend(edges[id].iter().copied());}
    if reached.iter().any(|seen|!seen) { return Err(error("unreachable instances in prepared artifact")); }
    let root=instances.pop().unwrap();
    root.validate_slot_manifest()?;
    let mut root=Arc::try_unwrap(root).unwrap_or_else(|root|root.as_ref().clone());
    if supplied.is_none() {
        root.retained_artifact = Some(Arc::from(data));
        root.retained_vocab_fingerprint = Some(recorded_fingerprint);
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corrupt(bytes: &[u8], mutate: impl FnOnce(&mut Manifest)) -> Vec<u8> {
        let size = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let mut manifest: Manifest = bincode::deserialize(&bytes[HEADER..HEADER+size]).unwrap();
        mutate(&mut manifest);
        let encoded = bincode::serialize(&manifest).unwrap();
        let mut result = MAGIC.to_vec();
        result.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        result.extend_from_slice(&bytes[16..HEADER]);
        result.extend_from_slice(&encoded); result.extend_from_slice(&bytes[HEADER+size..]); result
    }

    #[test]
    fn malformed_graphs_fail_before_compiled_bodies_are_loaded() {
        let v = Vocab::new(vec![(0,b"x".to_vec())]);
        let child = Grammar::from_glrm("start root nt root=\"x\"").unwrap().prepare(&v).unwrap();
        let parent = Grammar::from_glrm("extern grammar CHILD start root nt root=CHILD").unwrap()
            .prepare(&v).unwrap().bind_snapshot("CHILD",child).unwrap();
        let bytes = parent.save();
        let mutations: Vec<fn(&mut Manifest)> = vec![
            |m|m.instances.last_mut().unwrap().bindings[0].1=Binding::Prepared(u32::MAX),
            |m|{let last=m.instances.len() as u32-1;m.instances.last_mut().unwrap().bindings[0].1=Binding::Prepared(last);},
            |m|m.instances.last_mut().unwrap().bindings.push(("CHILD".into(),Binding::Prepared(0))),
            |m|m.instances.last_mut().unwrap().bindings[0].0="UNKNOWN".into(),
            |m|{m.instances.last_mut().unwrap().slot_kinds.insert("CHILD".into(),SlotKind::Token);},
            |m|m.instances.last_mut().unwrap().boundary_trigger=255,
            |m|m.instances.last_mut().unwrap().definition=u32::MAX,
            |m|m.instances.last_mut().unwrap().bindings.clear(),
            |m|m.definitions.push(Definition{bytes:0,runnable:false,exact_only_token_ids:vec![]}),
            |m|m.instances.last_mut().unwrap().slot_spans.get_mut("CHILD").unwrap().line=0,
        ];
        for (index,mutation) in mutations.into_iter().enumerate() {
            assert!(load(&corrupt(&bytes,mutation),None).is_err(),"corruption {index}");
        }
        assert!(load(&bytes,None).is_ok(), "{:?}", load(&bytes,None).err());
    }
}
